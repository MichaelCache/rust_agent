//! 流式对话：把 SSE 分片喂成结构化事件，并实时回调给调用方。
//!
//! 设计要点：
//! - 只读一遍网络流，思考内容、正文、工具调用三种增量同时产出，谁都不丢；
//! - 思考内容（`reasoning` / `reasoning_content` 字段，或正文里的思维链标签）
//!   通过 [`ChatEvent::Reasoning`] 单独上报，**不写回历史**，避免污染下一轮请求；
//! - 回调是同步的（`&mut` 闭包或实现 [`ChatStreamHandler`] 的类型即可），
//!   往终端打字不需要 async；确实需要异步副作用时在闭包里起个任务即可；
//! - 收包与解析在独立任务里跑，回调再慢也只影响消费端，不会卡住收包。

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, FunctionCall,
};
use futures_util::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::{
    error::{AgentError, Result},
    sse::{SseBuffer, StreamChunk, ToolCallAssembler},
};

// ---------------------------------------------------------------------------
// 事件
// ---------------------------------------------------------------------------

/// 流式对话过程中产生的事件。
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// 思考内容增量（模型自己的推理过程）
    Reasoning(String),
    /// 最终回答的正文增量（思维链已剥离）
    Text(String),
    /// 一个新的工具调用开始（只报一次；参数随后用 [`ChatEvent::ToolArguments`] 增量上报）
    ToolCall { id: String, name: String },
    /// 工具参数 JSON 的增量
    ToolArguments { id: String, delta: String },
    /// 模型结束本轮生成
    Finished {
        finish_reason: Option<String>,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
    },
}

/// 事件回调。返回 `Err` 会中断整个流（例如 stdout 写失败）。
pub trait ChatStreamHandler {
    fn on_event(&mut self, event: &ChatEvent) -> Result<()>;
}

/// `()` 表示不关心任何事件。
impl ChatStreamHandler for () {
    fn on_event(&mut self, _event: &ChatEvent) -> Result<()> {
        Ok(())
    }
}

/// 把 `&mut H` 包一层，好在多轮循环里复用同一个 handler。
///
/// 需要这一层是因为 `&mut F` 自己就实现了 `FnMut`，直接给 `&mut H` 写 blanket impl
/// 会和下面的闭包 impl 冲突。
pub struct HandlerRef<'a, H>(pub &'a mut H);

impl<H> ChatStreamHandler for HandlerRef<'_, H>
where
    H: ChatStreamHandler,
{
    fn on_event(&mut self, event: &ChatEvent) -> Result<()> {
        self.0.on_event(event)
    }
}

impl<F> ChatStreamHandler for F
where
    F: FnMut(&ChatEvent) -> Result<()>,
{
    fn on_event(&mut self, event: &ChatEvent) -> Result<()> {
        self(event)
    }
}

// ---------------------------------------------------------------------------
// 一轮流式对话的结果
// ---------------------------------------------------------------------------

/// 一轮流式对话汇总出来的结果。
#[derive(Debug, Default, Clone)]
pub struct ChatTurn {
    /// 正文全文（思维链已剥离）
    pub text: String,
    /// 思考内容全文
    pub reasoning: String,
    /// 工具调用：`(id, 名称, 参数 JSON)`
    pub tool_calls: Vec<(String, String, String)>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    /// 服务端是否给出过任何增量（用于区分“模型真没说话”和“服务端异常”）
    pub saw_any_delta: bool,
}

impl ChatTurn {
    /// 转成可以写回历史的 OpenAI 工具调用结构。
    pub fn to_tool_calls(&self) -> Vec<ChatCompletionMessageToolCalls> {
        self.tool_calls
            .iter()
            .map(|(id, name, arguments)| {
                ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                    id: id.clone(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                })
            })
            .collect()
    }

    /// 工具调用名列表，用于日志。
    pub fn tool_names(&self) -> Vec<&str> {
        self.tool_calls.iter().map(|(_, n, _)| n.as_str()).collect()
    }

    /// 既没有正文也没有工具调用。
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.tool_calls.is_empty()
    }
}

// ---------------------------------------------------------------------------
// 思维链剥离（流式版）
// ---------------------------------------------------------------------------

/// 所有会被识别的思维链标记。`\x3C` / `\x3E` 是尖括号的转义写法，
/// 避免源码里再出现模型特殊 token 的字形。
const THINK_TAGS: [&str; 4] = [
    "\x3Cthink\x3E",
    "\x3C/think\x3E",
    "\x3Cthinking\x3E",
    "\x3C/thinking\x3E",
];

/// 逐段剥离思维链。
///
/// 和 [`crate::text::strip_thinking`] 的区别是这个版本**保留原文**：标记只是
/// 不出现在产出的“干净文本”里，原始字节一个都不动——模型特殊 token 被切碎
/// （`<thi` + `nk>`）会让服务端拒绝历史里的 `content`，而我们只决定给用户看什么。
///
/// 增量流的难点在于标记本身可能被切开，所以遇到“可能是标记前缀”的尾巴要先压住，
/// 等下一批字节再判断；流结束时再放出来。
#[derive(Debug, Default)]
pub struct ThinkingStripper {
    buf: String,
    in_think: bool,
}

impl ThinkingStripper {
    pub fn new() -> Self {
        Self::default()
    }

    /// 喂入一段正文增量，返回其中可以直接展示的部分。
    pub fn feed(&mut self, chunk: &str) -> String {
        self.buf.push_str(chunk);
        let mut out = String::new();

        loop {
            match self.next_marker() {
                // 碰到真正的标记：放掉它前面的正文，翻转状态，继续往后看
                Marker::Tag { start, tag } => {
                    let head: String = self.buf[..start].to_string();
                    self.buf.drain(..start + tag.len());
                    self.push_visible(&head, &mut out);
                    self.in_think = tag == THINK_TAGS[0] || tag == THINK_TAGS[2];
                }
                // 没有标记，但尾巴可能是被切开的标记：压住，等下一批字节
                Marker::MaybePrefix { start } => {
                    let tail = self.buf.split_off(start);
                    let head = std::mem::replace(&mut self.buf, tail);
                    self.push_visible(&head, &mut out);
                    break;
                }
                // 干净文本：整段交给用户（在思维链里的话就丢掉）
                Marker::None => {
                    let head = std::mem::take(&mut self.buf);
                    self.push_visible(&head, &mut out);
                    break;
                }
            }
        }

        out
    }

    /// 流结束：把压住的尾巴放出来（还停在思维链里就直接丢掉）。
    pub fn finish(&mut self) -> String {
        if self.in_think {
            self.buf.clear();
            return String::new();
        }
        std::mem::take(&mut self.buf)
    }

    /// 只有不在思维链里才把文本交给用户。
    fn push_visible(&self, text: &str, out: &mut String) {
        if !self.in_think {
            out.push_str(text);
        }
    }

    /// 找下一个要处理的标记：开标记任何位置都算；闭标记只在思维链里才算。
    fn next_marker(&self) -> Marker {
        // THINK_TAGS 是两两成对的：偶数下标是开标记，奇数下标是闭标记
        let tags = THINK_TAGS
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0 || self.in_think)
            .map(|(_, tag)| *tag);

        match tags
            .filter_map(|tag| self.buf.find(tag).map(|pos| (pos, tag)))
            .min_by_key(|(pos, tag)| (*pos, std::cmp::Reverse(tag.len())))
        {
            Some((start, tag)) => Marker::Tag { start, tag },
            None => match longest_tag_prefix_at_end(&self.buf) {
                Some(start) => Marker::MaybePrefix { start },
                None => Marker::None,
            },
        }
    }
}

/// [`ThinkingStripper::next_marker`] 的判定结果。
enum Marker {
    /// 找到一个完整标记：起始字节位置 + 标记本身
    Tag { start: usize, tag: &'static str },
    /// 尾巴可能是被切开的标记：从该位置起压住
    MaybePrefix { start: usize },
    /// 没有任何标记
    None,
}

/// 返回 `text` 末尾那个“可能是某个标记前缀”的起始字节位置。
fn longest_tag_prefix_at_end(text: &str) -> Option<usize> {
    for (start, _) in text.char_indices().rev() {
        let suffix = &text[start..];
        if suffix.len() > 10 {
            break; // 最长的标记 11 字节，再长就不可能还是它的前缀
        }
        if suffix.starts_with('\x3C') && THINK_TAGS.iter().any(|t| t.starts_with(suffix)) {
            return Some(start);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 流式响应 → 事件
// ---------------------------------------------------------------------------

/// SSE 数据块 → 事件，并把整轮结果累积下来。
struct ChunkRouter {
    stripper: ThinkingStripper,
    tools: ToolCallAssembler,
    turn: ChatTurn,
}

impl ChunkRouter {
    fn new() -> Self {
        Self {
            stripper: ThinkingStripper::new(),
            tools: ToolCallAssembler::default(),
            turn: ChatTurn::default(),
        }
    }

    /// 把一块 SSE 数据摊平成事件（思考 → 正文 → 工具 → 结束）。
    fn route(&mut self, chunk: &StreamChunk) -> Vec<ChatEvent> {
        let mut events = Vec::new();

        if let Some(usage) = &chunk.usage {
            if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.turn.prompt_tokens = Some(v as u32);
            }
            if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.turn.completion_tokens = Some(v as u32);
            }
        }

        for choice in &chunk.choices {
            if let Some(thinking) = choice.delta.thinking() {
                self.turn.saw_any_delta = true;
                self.turn.reasoning.push_str(thinking);
                events.push(ChatEvent::Reasoning(thinking.to_string()));
            }

            let raw = choice.delta.content.clone().unwrap_or_default();
            if !raw.is_empty() {
                self.turn.saw_any_delta = true;
                let visible = self.stripper.feed(&raw);
                if !visible.is_empty() {
                    self.turn.text.push_str(&visible);
                    events.push(ChatEvent::Text(visible));
                }
            }

            for call in choice.delta.tool_calls.iter().flatten() {
                self.turn.saw_any_delta = true;
                let is_new = self.tools.push(call);
                let id = call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_{}", call.index));
                if is_new && let Some(name) = call.function.as_ref().and_then(|f| f.name.clone()) {
                    events.push(ChatEvent::ToolCall {
                        id: id.clone(),
                        name,
                    });
                }
                if let Some(args) = call.function.as_ref().and_then(|f| f.arguments.clone())
                    && !args.is_empty()
                {
                    events.push(ChatEvent::ToolArguments { id, delta: args });
                }
            }

            if let Some(reason) = &choice.finish_reason {
                let tail = self.stripper.finish();
                if !tail.is_empty() {
                    self.turn.text.push_str(&tail);
                    events.push(ChatEvent::Text(tail));
                }
                self.turn.finish_reason = Some(reason.clone());
                events.push(ChatEvent::Finished {
                    finish_reason: Some(reason.clone()),
                    prompt_tokens: self.turn.prompt_tokens,
                    completion_tokens: self.turn.completion_tokens,
                });
            }
        }

        events
    }

    /// 收尾：有些服务端最后一块只带 usage，不带 choices。
    fn finish_turn(mut self) -> ChatTurn {
        let tail = self.stripper.finish();
        if !tail.is_empty() {
            self.turn.text.push_str(&tail);
        }
        self.turn.tool_calls = self.tools.finish();
        self.turn
    }
}

// ---------------------------------------------------------------------------
// ChatStream
// ---------------------------------------------------------------------------

/// 一次流式对话。
///
/// 同时是 [`Stream`]（`Item = Result<ChatEvent>`）和 [`ChatStream::collect`] 的来源：
/// 每个事件先经过回调、再交给流的消费者，两条路互不干扰。
pub struct ChatStream<H = ()> {
    rx: mpsc::UnboundedReceiver<Result<ChatEvent>>,
    handler: H,
    shared: std::sync::Arc<std::sync::Mutex<Shared>>,
    ended: bool,
}

#[derive(Debug, Default)]
struct Shared {
    /// 网络任务收尾后的整轮结果
    turn: Option<ChatTurn>,
    /// 网络任务失败的原因
    error: Option<String>,
}

impl<H> ChatStream<H> {
    /// 取下一个事件。
    pub async fn next_event(&mut self) -> Option<Result<ChatEvent>>
    where
        H: ChatStreamHandler + Unpin,
    {
        <Self as StreamExt>::next(self).await
    }
}

impl<H> ChatStream<H>
where
    H: ChatStreamHandler + Unpin,
{
    /// 驱动到结束，返回本轮完整结果。
    pub async fn collect(mut self) -> Result<ChatTurn> {
        while let Some(event) = self.next_event().await {
            event?;
        }
        let guard = self.shared.lock().expect("流式结果锁被投毒");
        if let Some(err) = &guard.error {
            return Err(AgentError::Stream(err.clone()));
        }
        Ok(guard.turn.clone().unwrap_or_default())
    }
}

impl<H> Stream for ChatStream<H>
where
    H: ChatStreamHandler + Unpin,
{
    type Item = Result<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        match this.rx.poll_recv(cx) {
            Poll::Ready(Some(Ok(event))) => match this.handler.on_event(&event) {
                Ok(()) => Poll::Ready(Some(Ok(event))),
                Err(e) => {
                    this.ended = true;
                    Poll::Ready(Some(Err(e)))
                }
            },
            Poll::Ready(Some(Err(e))) => {
                this.ended = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                this.ended = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// ---------------------------------------------------------------------------
// 网络任务
// ---------------------------------------------------------------------------

/// 发起一次流式请求，返回事件流。
///
/// `handler` 在**消费端**被同步调用：只有你 poll 这个流的时候才回调，
/// 因此里面做阻塞操作只会拖慢自己，不会拖住收包任务。
pub(crate) fn open<H>(
    http: reqwest::Client,
    url: String,
    body: serde_json::Value,
    base_url: String,
    model: String,
    idle_timeout: Duration,
    handler: H,
) -> ChatStream<H>
where
    H: ChatStreamHandler,
{
    let (tx, rx) = mpsc::unbounded_channel();
    let shared = std::sync::Arc::new(std::sync::Mutex::new(Shared::default()));
    let worker_shared = std::sync::Arc::clone(&shared);

    tokio::spawn(async move {
        let outcome = read_stream(http, url, body, base_url, model, idle_timeout, &tx).await;
        let mut guard = worker_shared.lock().expect("流式结果锁被投毒");
        match outcome {
            Ok(turn) => guard.turn = Some(turn),
            Err(e) => {
                guard.error = Some(e.to_string());
                let _ = tx.send(Err(e));
            }
        }
    });

    ChatStream {
        rx,
        handler,
        shared,
        ended: false,
    }
}

async fn read_stream(
    http: reqwest::Client,
    url: String,
    body: serde_json::Value,
    base_url: String,
    model: String,
    idle_timeout: Duration,
    tx: &mpsc::UnboundedSender<Result<ChatEvent>>,
) -> Result<ChatTurn> {
    let response = http
        .post(&url)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .json(&body)
        .send()
        .await
        .map_err(|e| transport_error(&base_url, &model, e))?;

    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        return Err(AgentError::Stream(format!(
            "LLM 接口返回 HTTP {status}：{}",
            detail.trim()
        )));
    }

    let mut router = ChunkRouter::new();
    let mut decoder = SseBuffer::new();
    let mut byte_stream = response.bytes_stream();
    // 首包与包间共用一个滚动超时：既能兜住“连上了但一个字不吐”，
    // 也能兜住“吐到一半卡死”。
    let mut deadline = tokio::time::Instant::now() + idle_timeout;

    loop {
        let next = tokio::select! {
            biased;
            _ = tokio::time::sleep_until(deadline) => {
                return Err(AgentError::Stream(format!(
                    "流式响应已 {} 秒没有新数据（可用 --timeout 调大）",
                    idle_timeout.as_secs()
                )));
            }
            next = byte_stream.next() => next,
        };

        let Some(bytes) = next else { break };
        let bytes = bytes.map_err(|e| transport_error(&base_url, &model, e))?;
        deadline = tokio::time::Instant::now() + idle_timeout;

        for chunk in decoder.push(&bytes)? {
            for event in router.route(&chunk) {
                if tx.send(Ok(event)).is_err() {
                    // 消费者提前退出：不再读网络
                    return Ok(router.finish_turn());
                }
            }
        }
    }

    for chunk in decoder.finish()? {
        for event in router.route(&chunk) {
            if tx.send(Ok(event)).is_err() {
                return Ok(router.finish_turn());
            }
        }
    }

    Ok(router.finish_turn())
}

/// 把 reqwest 的传输错误包成带提示的 LLM 错误。
fn transport_error(base_url: &str, model: &str, source: reqwest::Error) -> AgentError {
    AgentError::Llm {
        base_url: base_url.to_string(),
        model: model.to_string(),
        source: Box::new(async_openai::error::OpenAIError::Reqwest(source)),
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const OPEN: &str = THINK_TAGS[0];
    const CLOSE: &str = THINK_TAGS[1];

    /// 把整段文本按“逐字符”喂进去，模拟最坏的分片边界。
    fn feed_by_char(text: &str) -> String {
        let mut s = ThinkingStripper::new();
        let mut out = String::new();
        for ch in text.chars() {
            out.push_str(&s.feed(&ch.to_string()));
        }
        out.push_str(&s.finish());
        out
    }

    #[test]
    fn stripper_removes_closed_block_regardless_of_chunking() {
        assert_eq!(feed_by_char(&format!("前{OPEN}中间{CLOSE}后")), "前后");
    }

    #[test]
    fn stripper_keeps_literal_angle_brackets() {
        assert_eq!(feed_by_char("a < b 3 < 4"), "a < b 3 < 4");
        assert_eq!(feed_by_char("结束<"), "结束<");
        assert_eq!(feed_by_char("<th 不是标签"), "<th 不是标签");
    }

    #[test]
    fn stripper_drops_to_end_when_block_never_closes() {
        assert_eq!(feed_by_char(&format!("答案{OPEN}还在想很久很久")), "答案");
    }

    #[test]
    fn stripper_handles_thinking_variant() {
        assert_eq!(
            feed_by_char("\x3Cthinking\x3E想法\x3C/thinking\x3E结果"),
            "结果"
        );
    }

    #[test]
    fn stripper_handles_two_blocks() {
        assert_eq!(
            feed_by_char(&format!("{OPEN}思考{CLOSE}答案{OPEN}再想{CLOSE}尾巴")),
            "答案尾巴"
        );
    }

    #[test]
    fn stripper_handles_whole_block_in_one_chunk() {
        // 一整段推理（带闭标记）在同一次 feed 里到达：闭标记在中间，必须能识别
        let mut s = ThinkingStripper::new();
        assert_eq!(
            s.feed(&format!("{OPEN}内部推理{CLOSE}最终答案")),
            "最终答案"
        );
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn stripper_keeps_answer_that_arrives_before_close_tag() {
        // 闭标记不带斜杠的变体、以及正文早于闭标记到达的情况
        let mut s = ThinkingStripper::new();
        assert_eq!(s.feed(OPEN), "");
        assert_eq!(s.feed("想法"), "");
        assert_eq!(s.feed(CLOSE), "");
        assert_eq!(s.feed("答案"), "答案");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn stripper_treats_stray_close_tag_as_plain_text() {
        // 没进思维链却出现闭标记：按普通文本放出去，不能被吞掉
        let mut s = ThinkingStripper::new();
        assert_eq!(s.feed(CLOSE), "");
        assert_eq!(s.feed("正文"), CLOSE.to_string() + "正文");
        let mut s = ThinkingStripper::new();
        assert_eq!(s.feed(CLOSE), "");
        assert_eq!(s.finish(), CLOSE);
    }

    #[test]
    fn stripper_is_incremental_across_blocks() {
        let mut s = ThinkingStripper::new();
        assert_eq!(s.feed("答"), "答");
        assert_eq!(s.feed(OPEN), "");
        assert_eq!(s.feed("想"), "");
        assert_eq!(s.feed(CLOSE), "");
        assert_eq!(s.feed("案"), "案");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn detects_tag_prefix_at_end() {
        assert_eq!(longest_tag_prefix_at_end("abc<thi"), Some(3));
        assert_eq!(longest_tag_prefix_at_end("abc"), None);
        assert_eq!(longest_tag_prefix_at_end("abc<"), Some(3));
        assert_eq!(longest_tag_prefix_at_end("x\x3C/think\x3E"), Some(1));
        // 普通的长文本不该被误判
        assert_eq!(longest_tag_prefix_at_end("0123456789abcdef"), None);
    }

    #[test]
    fn router_emits_reasoning_then_text_then_tools() {
        let mut router = ChunkRouter::new();
        let chunk: StreamChunk = serde_json::from_str(
            r#"{"choices":[{"index":0,"delta":{"reasoning":"想","content":"答",
                "tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{}"}}]},
                "finish_reason":"tool_calls"}]}"#,
        )
        .unwrap();
        let kinds: Vec<&str> = router
            .route(&chunk)
            .iter()
            .map(|e| match e {
                ChatEvent::Reasoning(_) => "reasoning",
                ChatEvent::Text(_) => "text",
                ChatEvent::ToolCall { .. } => "tool_call",
                ChatEvent::ToolArguments { .. } => "tool_args",
                ChatEvent::Finished { .. } => "finished",
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["reasoning", "text", "tool_call", "tool_args", "finished"]
        );

        let turn = router.finish_turn();
        assert_eq!(turn.text, "答");
        assert_eq!(turn.reasoning, "想");
        assert_eq!(turn.tool_names(), vec!["read_file"]);
        assert_eq!(turn.finish_reason.as_deref(), Some("tool_calls"));
        assert!(turn.saw_any_delta);
    }

    #[test]
    fn router_strips_inline_thinking_from_text() {
        let mut router = ChunkRouter::new();
        let chunk: StreamChunk = serde_json::from_str(&format!(
            r#"{{"choices":[{{"index":0,"delta":{{"content":"{OPEN}内部推理{CLOSE}最终答案"}}}}]}}"#
        ))
        .unwrap();
        // 事件里给用户的正文不含思考内容（开头的标记还在等后续字节，先压住）
        let streamed: String = router
            .route(&chunk)
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(!streamed.contains("内部推理"), "{streamed}");
        // 但整轮汇总后的正文必须是完整的、干净的
        let turn = router.finish_turn();
        assert_eq!(turn.text, "最终答案");
        assert!(turn.reasoning.is_empty(), "正文里的思考不算 reasoning 字段");
    }

    #[test]
    fn router_records_usage_from_trailing_chunk() {
        let mut router = ChunkRouter::new();
        let chunk: StreamChunk = serde_json::from_str(
            r#"{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":11,"total_tokens":18}}"#,
        )
        .unwrap();
        assert!(router.route(&chunk).is_empty());
        let turn = router.finish_turn();
        assert_eq!(turn.prompt_tokens, Some(7));
        assert_eq!(turn.completion_tokens, Some(11));
    }

    #[test]
    fn turn_converts_tool_calls_for_history() {
        let turn = ChatTurn {
            tool_calls: vec![("call_1".into(), "read_file".into(), "{}".into())],
            ..Default::default()
        };
        let calls = turn.to_tool_calls();
        assert_eq!(calls.len(), 1);
        match &calls[0] {
            ChatCompletionMessageToolCalls::Function(f) => {
                assert_eq!(f.id, "call_1");
                assert_eq!(f.function.name, "read_file");
            }
            other => panic!("期望 function 工具调用，得到 {other:?}"),
        }
    }
}
