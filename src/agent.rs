//! Agent 主循环：LLM 请求 → 执行 MCP 工具 → 回填结果 → 直到给出最终回答。

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionTools, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};
use tracing::{debug, info, warn};

use crate::{
    error::{AgentError, Result},
    llm,
    mcp::McpClient,
    stream::{self, ChatStreamHandler, ChatTurn, HandlerRef},
    text::truncate_with_note,
    workspace::Workspace,
};

pub const DEFAULT_MODEL: &str = "qwen3.8:latest";
pub const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";
pub const DEFAULT_API_KEY: &str = "ollama";
pub const DEFAULT_MAX_STEPS: usize = 25;
pub const DEFAULT_MAX_TOOL_RESULT_CHARS: usize = 40_000;
pub const DEFAULT_MAX_HISTORY_MESSAGES: usize = 200;
pub const DEFAULT_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    /// 允许 Agent 读写的工作目录
    pub workspace: PathBuf,
    /// MCP 工具服务器二进制；None 时自动查找
    pub server_bin: Option<PathBuf>,
    /// 最多几轮“LLM + 工具调用”
    pub max_steps: usize,
    /// 采样温度；None 表示用服务端默认
    pub temperature: Option<f32>,
    /// 单次 LLM 请求超时
    pub request_timeout: Duration,
    /// 单条工具结果最多回填多少字符，防止把上下文撑爆
    pub max_tool_result_chars: usize,
    /// 历史消息条数上限，超过就丢弃最早的完整轮次（交互模式下防止超出模型上下文）
    pub max_history_messages: usize,
    /// 是否允许 `run_command` 使用 shell（管道/重定向）
    pub allow_shell: bool,
    /// 追加到 `run_command` 白名单的程序名
    pub extra_allowed_commands: Vec<String>,
    pub system_prompt: Option<String>,
}

impl AgentConfig {
    /// 转发给 `code-tools-server` 的命令行参数
    fn server_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if self.allow_shell {
            args.push("--allow-shell".to_string());
        }
        for program in &self.extra_allowed_commands {
            args.push("--allow-command".to_string());
            args.push(program.clone());
        }
        args
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: DEFAULT_API_KEY.to_string(),
            workspace: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            server_bin: None,
            max_steps: DEFAULT_MAX_STEPS,
            temperature: None,
            request_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_tool_result_chars: DEFAULT_MAX_TOOL_RESULT_CHARS,
            max_history_messages: DEFAULT_MAX_HISTORY_MESSAGES,
            allow_shell: false,
            extra_allowed_commands: Vec::new(),
            system_prompt: None,
        }
    }
}

pub struct Agent {
    cfg: AgentConfig,
    /// 流式对话直连用的 HTTP 客户端
    http: reqwest::Client,
    mcp: McpClient,
    tools: Vec<ChatCompletionTools>,
    tool_names: Vec<String>,
    /// 完整对话历史（REPL 多轮时持续累积）
    history: Vec<ChatCompletionRequestMessage>,
}

impl Agent {
    /// 连接 LLM 与 MCP 工具服务器。
    pub async fn connect(mut cfg: AgentConfig) -> Result<Self> {
        let ws = Workspace::new(&cfg.workspace).map_err(AgentError::Workspace)?;
        cfg.workspace = ws.root().to_path_buf();

        let server_bin = resolve_server_bin(cfg.server_bin.as_deref())?;
        let server_args = cfg.server_args();
        info!(
            server = %server_bin.display(),
            workspace = %cfg.workspace.display(),
            allow_shell = cfg.allow_shell,
            extra_commands = ?cfg.extra_allowed_commands,
            "启动 MCP 工具服务器"
        );

        let mcp = McpClient::spawn_with_args(&server_bin, ws.root(), &server_args).await?;
        let mcp_tools = mcp.list_tools().await?;
        if mcp_tools.is_empty() {
            warn!("MCP 服务器未提供任何工具，Agent 将只能靠模型自身知识回答");
        }
        let tools = llm::mcp_tools_to_openai(&mcp_tools);
        let tool_names = llm::tool_names(&tools);
        info!(tools = ?tool_names, "工具已就绪");

        // 对话本身走下面的流式直连（async-openai 的流式类型会丢掉 reasoning 字段），
        // 这里只建一个 HTTP 客户端，超时策略由我们按“分片间隔”控制，
        // 因此明确只设连接超时，避免长回答被 reqwest 整体超时掐断。
        let http = reqwest::Client::builder()
            .connect_timeout(cfg.request_timeout)
            .build()
            .map_err(|e| AgentError::Stream(format!("构造 HTTP 客户端失败: {e}")))?;
        let system_prompt = cfg
            .system_prompt
            .clone()
            .unwrap_or_else(|| default_system_prompt(&cfg.workspace, &tool_names));
        let history = vec![ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_prompt)
                .build()
                .map_err(|e| AgentError::Request(e.to_string()))?,
        )];

        Ok(Self {
            cfg,
            http,
            mcp,
            tools,
            tool_names,
            history,
        })
    }

    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    pub fn workspace(&self) -> &Path {
        &self.cfg.workspace
    }

    pub fn tool_names(&self) -> &[String] {
        &self.tool_names
    }

    /// 清空对话历史（保留 system prompt）。
    pub fn reset(&mut self) {
        self.history.truncate(1);
    }

    /// 历史过长时丢弃最早的**完整轮次**。
    ///
    /// OpenAI 协议要求带 `tool_calls` 的 assistant 消息后面必须紧跟对应的 tool 消息，
    /// 所以只能从某个 user 消息处整体切掉前缀，绝不能按条数硬截断。
    fn compact_history(&mut self) {
        while self.history.len() > self.cfg.max_history_messages {
            let Some(cut) = safe_cut_index(&self.history) else {
                break; // 只剩一轮，切不动了
            };
            let dropped = self.history.drain(1..cut).count();
            warn!(
                dropped,
                remaining = self.history.len(),
                "对话历史超过 {} 条，已丢弃最早的 {dropped} 条消息",
                self.cfg.max_history_messages
            );
        }
    }

    /// 执行一轮“用户任务 → 最终回答”。
    ///
    /// 内部走流式（每来一段就回调），这里只是把事件丢掉、把最终文本收集起来。
    /// 想在打字过程中实时展示（含思考内容）请用 [`Agent::ask_stream_with`]。
    pub async fn ask(&mut self, task: &str) -> Result<String> {
        self.ask_stream_with(task, ()).await
    }

    /// 执行一轮任务，并实时回调每一个流式事件。
    ///
    /// `handler` 可以是 `&mut` 闭包，也可以是实现了 [`ChatStreamHandler`] 的类型。
    /// 回调里收到 [`ChatEvent::Reasoning`] 就是模型的思考过程（Ollama 的
    /// `reasoning` 字段、其他网关的 `reasoning_content`、以及正文里的思维链标签
    /// 都会被归到这里），[`ChatEvent::Text`] 是最终回答正文。
    ///
    /// 返回值仍是这一轮的最终回答（思考内容已剥离，且不会写回历史）。
    pub async fn ask_stream_with<H>(&mut self, task: &str, mut handler: H) -> Result<String>
    where
        H: ChatStreamHandler + Unpin,
    {
        let task = task.trim();
        if task.is_empty() {
            return Err(AgentError::Config("任务内容为空".to_string()));
        }
        // 先清理历史（只丢完整轮次），再追加本轮，避免中途打断 tool_calls/tool 配对
        self.compact_history();
        self.history.push(user_message(task)?);

        for step in 1..=self.cfg.max_steps {
            if let Some(answer) = self.step(step, &mut handler).await? {
                return Ok(answer);
            }
        }

        warn!(
            max_steps = self.cfg.max_steps,
            "达到工具调用上限，要求模型直接总结"
        );
        self.force_final_answer(&mut handler).await
    }

    /// 一轮 LLM 调用。返回 `Some(最终回答)` 表示结束，`None` 表示本轮只调用了工具。
    async fn step<H>(&mut self, step: usize, handler: &mut H) -> Result<Option<String>>
    where
        H: ChatStreamHandler + Unpin,
    {
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&self.cfg.model)
            .messages(self.history.clone())
            .tools(self.tools.clone());
        if let Some(t) = self.cfg.temperature {
            builder.temperature(t);
        }
        let request = builder
            .build()
            .map_err(|e| AgentError::Request(e.to_string()))?;

        let turn = self.chat_stream(request, handler).await?;

        if turn.tool_calls.is_empty() {
            return Ok(Some(self.finish_answer(turn)?));
        }

        let tool_calls = turn.to_tool_calls();
        info!(
            step,
            finish = ?turn.finish_reason,
            tools = ?turn.tool_names(),
            "执行工具调用"
        );

        // 先把 assistant(tool_calls) 写进历史，顺序不能反
        self.history
            .push(assistant_message(&turn.text, tool_calls.clone())?);

        for call in &tool_calls {
            let (call_id, tool_name, arguments) = match call {
                ChatCompletionMessageToolCalls::Function(f) => (
                    f.id.clone(),
                    f.function.name.clone(),
                    f.function.arguments.clone(),
                ),
                ChatCompletionMessageToolCalls::Custom(c) => {
                    self.history.push(tool_message(
                        &c.id,
                        "本 Agent 不支持 custom tool（仅支持 function 工具）。",
                    )?);
                    continue;
                }
            };

            let content = match self.execute_tool(&tool_name, &arguments).await {
                Ok(text) => text,
                Err(e) => format!("[工具执行失败] {e}"),
            };
            debug!(tool = %tool_name, len = content.len(), "工具返回");
            self.history.push(tool_message(&call_id, &content)?);
        }

        Ok(None)
    }

    /// 把这一轮的流式结果收口成最终回答（同时写进历史）。
    fn finish_answer(&mut self, turn: ChatTurn) -> Result<String> {
        if let (Some(prompt), Some(completion)) = (turn.prompt_tokens, turn.completion_tokens) {
            debug!(
                prompt_tokens = prompt,
                completion_tokens = completion,
                "usage"
            );
        }
        let answer = if !turn.text.trim().is_empty() {
            turn.text.trim().to_string()
        } else if turn.saw_any_delta {
            // 只想了没说话：把思考过程当作回答，总比一片空白强
            let reasoning = turn.reasoning.trim();
            if reasoning.is_empty() {
                "（模型没有返回任何内容，可能是上下文过长或服务端异常）".to_string()
            } else {
                format!("（模型只输出了思考过程，没有给出最终回答）\n{reasoning}")
            }
        } else {
            "（模型没有返回任何内容，可能是上下文过长或服务端异常）".to_string()
        };
        self.history.push(assistant_message(&answer, Vec::new())?);
        Ok(answer)
    }

    /// 调用一次流式 LLM 请求，把事件转交给 `handler`，返回整轮结果。
    ///
    /// `handler` 不能中途换，所以要 `&mut` 引用；超时由 [`crate::stream`]
    /// 按“分片间隔”计算，避免长回答被整体超时误杀。
    async fn chat_stream<H>(
        &self,
        request: CreateChatCompletionRequest,
        handler: &mut H,
    ) -> Result<ChatTurn>
    where
        H: ChatStreamHandler + Unpin,
    {
        let body = serde_json::to_value(&request)?;
        // 直连 SSE 时必须自己声明 stream / stream_options：
        // 只有服务端支持 include_usage 时，末尾才会多出一个带 usage 的块。
        let body = match body {
            serde_json::Value::Object(mut map) => {
                map.insert("stream".to_string(), serde_json::Value::Bool(true));
                map.insert(
                    "stream_options".to_string(),
                    serde_json::json!({ "include_usage": true }),
                );
                serde_json::Value::Object(map)
            }
            other => other,
        };

        // 流式请求直连 SSE，但沿用同一个 base_url 规则与连接池
        let stream = stream::open(
            self.http.clone(),
            llm::chat_completions_url(&self.cfg.base_url),
            body,
            self.cfg.base_url.clone(),
            self.cfg.model.clone(),
            self.cfg.request_timeout,
            HandlerRef(handler),
        );
        stream.collect().await
    }

    /// 执行一次 MCP 工具调用，把结果整理成回填给模型的文本。
    async fn execute_tool(
        &self,
        name: &str,
        raw_arguments: &str,
    ) -> std::result::Result<String, String> {
        if !self.tool_names.iter().any(|n| n == name) {
            return Err(format!(
                "未知工具 `{name}`。可用工具：{}",
                self.tool_names.join(", ")
            ));
        }
        let arguments = parse_tool_arguments(raw_arguments)?;
        let result = self
            .mcp
            .call_tool(name, arguments)
            .await
            .map_err(|e| format!("MCP 调用失败: {e}"))?;

        let mut text = result
            .content
            .iter()
            .map(|c| match c {
                rmcp::model::ContentBlock::Text(t) => t.text.clone(),
                other => format!("[非文本内容: {other:?}]"),
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error == Some(true) {
            text = format!("[工具返回错误] {text}");
        }
        if text.trim().is_empty() {
            text = "(工具没有返回内容)".to_string();
        }
        Ok(truncate_with_note(
            &text,
            self.cfg.max_tool_result_chars,
            "工具结果过长已截断",
        ))
    }

    /// 达到步数上限后，禁用工具再问一次，逼模型给出结论。
    async fn force_final_answer<H>(&mut self, handler: &mut H) -> Result<String>
    where
        H: ChatStreamHandler + Unpin,
    {
        self.history.push(user_message(&format!(
            "已达到工具调用步数上限（{} 步）。请不要再调用工具，直接根据目前掌握的信息总结：\
             已经完成了什么、修改了哪些文件、还有什么没做。",
            self.cfg.max_steps
        ))?);

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(&self.cfg.model)
            .messages(self.history.clone());
        if let Some(t) = self.cfg.temperature {
            builder.temperature(t);
        }
        let request = builder
            .build()
            .map_err(|e| AgentError::Request(e.to_string()))?;

        let turn = self.chat_stream(request, handler).await?;
        let answer = if !turn.text.trim().is_empty() {
            turn.text.trim().to_string()
        } else {
            format!(
                "已达到最大工具调用步数（{} 步）而未能得出最终结论。",
                self.cfg.max_steps
            )
        };
        self.history.push(assistant_message(&answer, Vec::new())?);
        Ok(answer)
    }

    /// 关闭 MCP 子进程。
    pub async fn shutdown(self) {
        self.mcp.shutdown().await;
    }
}

// ---------------------------------------------------------------------------
// 消息构造
// ---------------------------------------------------------------------------

/// 返回可以安全丢弃的历史前缀上界：`drain(1..cut)` 之后，第一条非 system 消息
/// 仍然是 user 消息，从而不会把 assistant(tool_calls) 和它的 tool 结果拆散。
/// 找不到（只剩一轮）时返回 `None`。
fn safe_cut_index(messages: &[ChatCompletionRequestMessage]) -> Option<usize> {
    messages
        .iter()
        .enumerate()
        .skip(2) // 0 是 system，1 若是 user 则无可再切
        .find_map(|(i, m)| match m {
            ChatCompletionRequestMessage::User(_) => Some(i),
            _ => None,
        })
}

fn user_message(content: &str) -> Result<ChatCompletionRequestMessage> {
    Ok(ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessageArgs::default()
            .content(content)
            .build()
            .map_err(|e| AgentError::Request(e.to_string()))?,
    ))
}

/// 构造 assistant 消息：只有非空内容才带 `content`，
/// 避免给 Ollama 发 `content: ""` 这种容易被拒的组合。
fn assistant_message(
    text: &str,
    tool_calls: Vec<ChatCompletionMessageToolCalls>,
) -> Result<ChatCompletionRequestMessage> {
    let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
    if !text.is_empty() {
        builder.content(text);
    }
    if !tool_calls.is_empty() {
        builder.tool_calls(tool_calls);
    }
    Ok(ChatCompletionRequestMessage::Assistant(
        builder
            .build()
            .map_err(|e| AgentError::Request(e.to_string()))?,
    ))
}

fn tool_message(tool_call_id: &str, content: &str) -> Result<ChatCompletionRequestMessage> {
    Ok(ChatCompletionRequestMessage::Tool(
        ChatCompletionRequestToolMessageArgs::default()
            .content(content)
            .tool_call_id(tool_call_id)
            .build()
            .map_err(|e| AgentError::Request(e.to_string()))?,
    ))
}

/// 解析模型给出的工具参数。
///
/// 兼容三种情况：空字符串（无参工具）、正常 JSON 对象、被双层编码成字符串的 JSON
/// （部分本地模型的常见毛病）。
fn parse_tool_arguments(
    raw: &str,
) -> std::result::Result<Option<serde_json::Map<String, serde_json::Value>>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return Ok(if trimmed == "{}" {
            Some(serde_json::Map::new())
        } else {
            None
        });
    }
    let parsed: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| format!("工具参数不是合法 JSON: {e}；收到的原始参数: {trimmed}"))?;

    let value = match parsed {
        serde_json::Value::Object(_) => parsed,
        serde_json::Value::String(inner) => serde_json::from_str(&inner).map_err(|e| {
            format!("工具参数疑似被双重编码但无法解析: {e}；收到的原始参数: {trimmed}")
        })?,
        other => {
            return Err(format!(
                "工具参数必须是 JSON 对象，收到: {other}（原始参数: {trimmed}）"
            ));
        }
    };

    match value {
        serde_json::Value::Object(map) => Ok(Some(map)),
        other => Err(format!("工具参数必须是 JSON 对象，收到: {other}")),
    }
}

// ---------------------------------------------------------------------------
// 工具服务器定位
// ---------------------------------------------------------------------------

/// 按优先级查找 `code-tools-server`：显式参数 → 环境变量 → 与当前可执行文件同级
/// → `target/{debug,release}`。
pub fn resolve_server_bin(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return if p.is_file() {
            Ok(p.to_path_buf())
        } else {
            Err(AgentError::ServerNotFound(p.display().to_string()))
        };
    }
    if let Ok(env_path) = std::env::var("CODE_TOOLS_SERVER") {
        let p = PathBuf::from(env_path);
        if p.is_file() {
            return Ok(p);
        }
        return Err(AgentError::ServerNotFound(p.display().to_string()));
    }

    let file_name = format!("code-tools-server{}", std::env::consts::EXE_SUFFIX);

    // 1) 与当前可执行文件同级（cargo run / cargo build 都适用）
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join(&file_name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    // 2) 相对当前目录的常见位置
    for rel in ["target/debug", "target/release"] {
        let candidate = PathBuf::from(rel).join(&file_name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(AgentError::ServerNotFound(format!(
        "{file_name}（已尝试：可执行文件同级目录、target/debug、target/release）"
    )))
}

// ---------------------------------------------------------------------------
// 默认 system prompt
// ---------------------------------------------------------------------------

pub fn default_system_prompt(workspace: &Path, tools: &[String]) -> String {
    format!(
        "你是一个运行在用户本机的代码助手，通过工具读写代码。\n\
         工作目录：{workspace}（所有文件操作都被限制在该目录内）\n\
         可用工具：{tools}\n\n\
         工作规则：\n\
         1. 修改任何文件之前，必须先用 read_file 读取它的当前内容，禁止凭空猜测或凭记忆编辑；\n\
         2. 修改用 edit_file，old_text 必须与文件内容逐字符一致（包含缩进与换行）；一次只改一处，必要时分多次调用；\n\
         3. 不确定文件路径时，先用 list_directory 或 search_code 定位；\n\
         4. 工具返回错误时，先读错误信息再调整参数重试，不要重复同样的失败调用；\n\
         5. 不要把整个文件内容复述给用户，只说明改了什么、为什么；\n\
         6. 改完代码后尽量用 run_command 验证（例如 cargo check、cargo test、git diff）；\n\
         7. 调用 run_command 时用 command + args 的 argv 形式，例如 command=\"cargo\", args=[\"check\"]；\
         不支持管道、重定向与 && 串联，一次只跑一条命令；\n\
         8. 完成任务后用中文简洁总结：改了哪些文件、关键改动、验证结果、需要用户注意的地方。",
        workspace = workspace.display(),
        tools = tools.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_object() {
        let args = parse_tool_arguments(r#"{"path":"src/main.rs"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(args["path"], "src/main.rs");
    }

    #[test]
    fn parses_empty_and_blank() {
        assert!(parse_tool_arguments("").unwrap().is_none());
        assert!(parse_tool_arguments("{}").unwrap().unwrap().is_empty());
    }

    #[test]
    fn parses_double_encoded_object() {
        let args = parse_tool_arguments(r#""{\"path\":\"a.rs\"}""#)
            .unwrap()
            .unwrap();
        assert_eq!(args["path"], "a.rs");
    }

    #[test]
    fn reports_invalid_json_instead_of_panicking() {
        let err = parse_tool_arguments("{not json").unwrap_err();
        assert!(err.contains("不是合法 JSON"), "{err}");
        assert!(parse_tool_arguments("[1,2,3]").is_err());
    }

    /// 流式模式下 `finish_answer` 收到的 `ChatTurn::text` 已经是剥离思维链后的正文；
    /// 这里验证“只有工具调用、正文为空”时不写 `content: ""`（Ollama 会拒绝这种组合）。
    #[test]
    fn assistant_message_with_tool_calls_and_no_text_omits_content() {
        let calls = ChatTurn {
            tool_calls: vec![("call_1".into(), "read_file".into(), "{}".into())],
            ..Default::default()
        }
        .to_tool_calls();
        let message = assistant_message("", calls).unwrap();
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["role"], "assistant");
        assert!(json.get("content").is_none(), "{json}");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(json["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn assistant_message_with_text_keeps_content() {
        let message = assistant_message("最终回答", Vec::new()).unwrap();
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["content"], "最终回答");
        assert!(json.get("tool_calls").is_none(), "{json}");
    }

    #[test]
    fn cut_index_keeps_tool_pairs_intact() {
        // system, user1, assistant1(tool_calls), tool1, user2, assistant2
        let history = vec![
            user_message("system 占位").unwrap(), // 位置 0 当作 system
            user_message("任务一").unwrap(),
            assistant_message("", Vec::new()).unwrap(),
            tool_message("call_1", "结果").unwrap(),
            user_message("任务二").unwrap(),
            assistant_message("回答", Vec::new()).unwrap(),
        ];
        // 只能切到第 4 条（任务二），这样第 4 条之后的配对不受影响
        assert_eq!(safe_cut_index(&history), Some(4));

        let mut trimmed = history.clone();
        trimmed.drain(1..4);
        assert_eq!(trimmed.len(), 3);
        assert!(matches!(trimmed[1], ChatCompletionRequestMessage::User(_)));
    }

    #[test]
    fn cut_index_is_none_for_single_turn() {
        let history = vec![
            user_message("system 占位").unwrap(),
            user_message("任务一").unwrap(),
            tool_message("call_1", "结果").unwrap(),
        ];
        assert_eq!(safe_cut_index(&history), None);
    }

    #[test]
    fn forwards_command_policy_flags_to_server() {
        let mut cfg = AgentConfig::default();
        assert!(cfg.server_args().is_empty(), "默认不该额外传参数");

        cfg.allow_shell = true;
        cfg.extra_allowed_commands = vec!["just".into(), "make".into()];
        assert_eq!(
            cfg.server_args(),
            vec![
                "--allow-shell",
                "--allow-command",
                "just",
                "--allow-command",
                "make"
            ]
        );
    }
}
