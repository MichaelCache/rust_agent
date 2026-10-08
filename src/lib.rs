//! 一个跑在本机的极简代码 Agent。
//!
//! 组成：
//! - [`agent`]：主循环（LLM ↔ 工具），本文档的核心；
//! - [`llm`]：Ollama 的 OpenAI 兼容客户端 + 工具 schema 转换；
//! - [`mcp`]：MCP 客户端（拉起子进程）与 [`mcp::CodeTools`]（工具服务器本体）；
//! - [`workspace`]：把文件操作限制在工作目录内的沙箱；
//! - [`text`]：思维链剥离、UTF-8 安全截断等小工具。
//!
//! 两个二进制：
//! - `rust_agent`：命令行 Agent（`src/main.rs`）；
//! - `code-tools-server`：stdio MCP 服务器（`src/bin/code-tools-server.rs`）。

pub mod agent;
pub mod error;
pub mod llm;
pub mod mcp;
pub mod text;
pub mod workspace;

pub use agent::{Agent, AgentConfig};
pub use error::{AgentError, Result};

/// 初始化 tracing。
///
/// **必须写 stderr**：MCP 服务器的 stdout 是 JSON-RPC 通道，任何日志写进 stdout
/// 都会破坏协议（原实现 `tracing_subscriber::fmt::init()` 正是写到 stdout 的）。
pub fn init_tracing(default_directives: &str) {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_directives));
    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .compact()
        .try_init();
}
