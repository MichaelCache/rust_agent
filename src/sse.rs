//! OpenAI 兼容接口的 SSE 流解析。
//!
//! `async-openai` 的流式类型 `ChatCompletionStreamResponseDelta` 只认标准字段，
//! Ollama（以及 vLLM / LM Studio 等）把思考过程放在非标准的 `reasoning` /
//! `reasoning_content` 字段里，官方类型会**直接丢掉**这些内容。所以这里自己
//! 解析 `data:` 行：既保留思考内容，也能拿到工具调用分段。
//!
//! 解析始终是增量的：分片边界可能落在任何位置，未收齐的行留在缓冲区里等下一批
//! 字节，绝不能按“一次 read = 一整行”来假设。

use serde::{Deserialize, Serialize};

use crate::error::{AgentError, Result};

/// `delta` 里的工具调用片段：名称与参数是**逐段**下发的，需要按 `index` 拼接。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCallChunk {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionChunk>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FunctionChunk {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// 一条 `data:` 行对应的增量。
///
/// 只声明我们真正用得到的字段：OpenAI 响应里还有 `logprobs`、`obfuscation`
/// 等大量字段，全部建模既没必要也容易随版本变动而失效。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    /// vLLM / DeepSeek / 部分 OpenAI 兼容网关
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Ollama 的 OpenAI 兼容层用的字段名
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
}

/// 思考内容所在的字段名（按顺序取第一个非空的）。
impl StreamDelta {
    /// 取出本段思考内容：`reasoning_content` 优先，其次 `reasoning`。
    pub fn thinking(&self) -> Option<&str> {
        for candidate in [&self.reasoning_content, &self.reasoning] {
            if let Some(t) = candidate.as_deref()
                && !t.is_empty()
            {
                return Some(t);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StreamChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: StreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// 流式响应的一整块（`data: {...}` 里 JSON 的结构）。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StreamChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<StreamChoice>,
    /// 只有带 `stream_options.include_usage` 时，最后一个块的 `usage` 才非空
    #[serde(default)]
    pub usage: Option<serde_json::Value>,
}

/// SSE 行缓冲：把任意切分的字节流还原成完整事件。
#[derive(Default)]
pub struct SseBuffer {
    buf: Vec<u8>,
}

impl SseBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// 喂入一批字节，返回其中已经收齐的块。
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<StreamChunk>> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();

        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            // 行尾的 \n 与可能的 \r 都不属于内容
            let line = &line[..line.len() - 1];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if let Some(chunk) = parse_line(line)? {
                out.push(chunk);
            }
        }
        Ok(out)
    }

    /// 流结束：有些服务端最后一行不带换行，这里补一次解析。
    pub fn finish(&mut self) -> Result<Vec<StreamChunk>> {
        let rest = std::mem::take(&mut self.buf);
        match parse_line(&rest)? {
            Some(chunk) => Ok(vec![chunk]),
            None => Ok(Vec::new()),
        }
    }
}

/// 解析一行 SSE。返回 `Ok(None)` 表示这行不需要处理（空行、注释、`[DONE]`）。
fn parse_line(line: &[u8]) -> Result<Option<StreamChunk>> {
    let line = std::str::from_utf8(line)
        .map_err(|e| AgentError::Stream(format!("流式响应不是合法 UTF-8: {e}")))?
        .trim();
    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None); // event: / id: 等字段对我们没有意义
    };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }

    serde_json::from_str(data)
        .map(Some)
        .map_err(|e| AgentError::Stream(format!("解析流式响应失败: {e}；原始数据: {data}")))
}

/// 把逐段下发的工具调用按 `index` 拼成完整调用。
#[derive(Debug, Default)]
pub struct ToolCallAssembler {
    slots: Vec<PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAssembler {
    /// 追加一段，返回这次是否产生了新的工具调用槽位。
    pub fn push(&mut self, chunk: &ToolCallChunk) -> bool {
        let idx = chunk.index as usize;
        let is_new = idx >= self.slots.len();
        while self.slots.len() <= idx {
            self.slots.push(PartialToolCall::default());
        }
        let slot = &mut self.slots[idx];
        if let Some(id) = &chunk.id
            && slot.id.is_empty()
        {
            slot.id = id.clone();
        }
        if let Some(f) = &chunk.function {
            if let Some(name) = &f.name {
                slot.name.push_str(name);
            }
            if let Some(args) = &f.arguments {
                slot.arguments.push_str(args);
            }
        }
        is_new
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.name.is_empty())
    }

    /// 拼接完成的 `(id, 名称, 参数 JSON)`。
    pub fn finish(&self) -> Vec<(String, String, String)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.name.is_empty())
            .map(|(i, s)| {
                let id = if s.id.is_empty() {
                    format!("call_{i}")
                } else {
                    s.id.clone()
                };
                (id, s.name.clone(), s.arguments.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(input: &[&str]) -> Vec<StreamChunk> {
        let mut buf = SseBuffer::new();
        let mut out = Vec::new();
        for piece in input {
            out.extend(buf.push(piece.as_bytes()).unwrap());
        }
        out.extend(buf.finish().unwrap());
        out
    }

    #[test]
    fn parses_reasoning_and_content_fields() {
        let out = chunks(&[
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"先想\"}}]}\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"再想\"}}]}\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"答案\"}}]}\n",
            "data: [DONE]\n",
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].choices[0].delta.thinking(), Some("先想"));
        assert_eq!(out[1].choices[0].delta.thinking(), Some("再想"));
        assert_eq!(out[2].choices[0].delta.content.as_deref(), Some("答案"));
    }

    #[test]
    fn survives_split_lines_and_missing_trailing_newline() {
        // 一整条事件被切成 3 段，且最后一段没有换行符
        let out = chunks(&[
            "data: {\"choi",
            "ces\":[{\"index\":0,\"delta\":{\"content\":\"你好\"}}]}",
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("你好"));
    }

    #[test]
    fn ignores_comments_blank_lines_and_done() {
        let out = chunks(&[": keepalive\n", "\n", "event: ping\n", "data: [DONE]\n"]);
        assert!(out.is_empty());
    }

    #[test]
    fn reports_invalid_json() {
        let mut buf = SseBuffer::new();
        let err = buf.push(b"data: {not json}\n").unwrap_err();
        assert!(err.to_string().contains("解析流式响应失败"), "{err}");
    }

    #[test]
    fn assembles_tool_call_across_chunks() {
        let mut asm = ToolCallAssembler::default();
        let mk = |json: &str| -> ToolCallChunk { serde_json::from_str(json).unwrap() };
        assert!(asm.push(&mk(
            r#"{"index":0,"id":"call_1","function":{"name":"read_","arguments":"{\"pa"}}"#
        )));
        assert!(!asm.push(&mk(
            r#"{"index":0,"function":{"name":"file","arguments":"th\":\"a.rs\"}"}}"#
        )));
        assert!(!asm.is_empty());
        assert_eq!(
            asm.finish(),
            vec![(
                "call_1".to_string(),
                "read_file".to_string(),
                r#"{"path":"a.rs"}"#.to_string()
            )]
        );
    }

    #[test]
    fn invents_id_when_server_omits_it() {
        let mut asm = ToolCallAssembler::default();
        let chunk: ToolCallChunk =
            serde_json::from_str(r#"{"index":0,"function":{"name":"list_directory"}}"#).unwrap();
        asm.push(&chunk);
        assert_eq!(asm.finish()[0].0, "call_0");
    }
}
