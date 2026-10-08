//! 统一错误类型。

use async_openai::error::OpenAIError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error(
        "调用 LLM 失败 (base_url={base_url}): {source}\n提示: 确认 `ollama serve` 已启动，且模型名 `{model}` 存在 (`ollama list`)"
    )]
    Llm {
        base_url: String,
        model: String,
        // 装箱：OpenAIError 变体很大，直接内联会让 AgentError 膨胀到几百字节
        #[source]
        source: Box<OpenAIError>,
    },

    #[error("调用 LLM 超时（超过 {0} 秒）。本地大模型首次加载较慢，可用 --timeout 调大")]
    LlmTimeout(u64),

    #[error("LLM 返回了空的 choices（服务端异常）")]
    EmptyChoices,

    #[error("MCP 通信错误: {0}")]
    Mcp(#[from] rmcp::service::ServiceError),

    #[error("MCP 初始化失败（子进程未按 MCP 协议应答）: {0}")]
    McpInit(#[from] Box<rmcp::service::ClientInitializeError>),

    #[error("JSON 序列化/解析错误: {0}")]
    Json(#[from] serde_json::Error),

    #[error("工作目录无效: {0}")]
    Workspace(String),

    #[error(
        "找不到 MCP 工具服务器二进制 {0}。请先 `cargo build`（release 模式用 `cargo build --release`），或用 --server <路径> / 环境变量 CODE_TOOLS_SERVER 指定"
    )]
    ServerNotFound(String),

    #[error("构造请求失败: {0}")]
    Request(String),

    #[error("{0}")]
    Config(String),
}
