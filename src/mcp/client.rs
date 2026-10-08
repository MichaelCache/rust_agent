//! MCP 客户端：以子进程方式拉起 `code-tools-server`，通过 stdio 走 JSON-RPC。
//!
//! 关键点：
//! - 子进程显式带上 `--workspace`，避免它把“启动它的目录”当成工作目录；
//! - `kill_on_drop(true)`，父进程退出时不会留下孤儿进程；
//! - 工具列表用 `list_all_tools()`，自动处理分页。

use std::path::Path;

use rmcp::{
    RoleClient,
    model::{CallToolRequestParams, CallToolResult, Tool},
    service::{RunningService, ServiceExt},
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use tokio::process::Command;
use tracing::debug;

use crate::error::Result;

pub struct McpClient {
    service: RunningService<RoleClient, ()>,
}

impl McpClient {
    pub async fn spawn(server_bin: &Path, workspace: &Path) -> Result<Self> {
        Self::spawn_with_args(server_bin, workspace, &[]).await
    }

    /// `extra_args` 会附加到服务器命令行，例如 `--allow-shell` / `--allow-command <程序>`。
    pub async fn spawn_with_args(
        server_bin: &Path,
        workspace: &Path,
        extra_args: &[String],
    ) -> Result<Self> {
        let cmd = Command::new(server_bin).configure(|c| {
            c.arg("--workspace").arg(workspace);
            c.args(extra_args);
            c.current_dir(workspace);
            c.kill_on_drop(true);
        });
        debug!(bin = %server_bin.display(), args = ?extra_args, "spawn MCP server");
        let service = ()
            .serve(TokioChildProcess::new(cmd)?)
            .await
            .map_err(|e| crate::error::AgentError::McpInit(Box::new(e)))?;
        Ok(Self { service })
    }

    pub async fn list_tools(&self) -> Result<Vec<Tool>> {
        Ok(self.service.peer().list_all_tools().await?)
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult> {
        let params = match arguments {
            Some(args) => CallToolRequestParams::new(name.to_string()).with_arguments(args),
            None => CallToolRequestParams::new(name.to_string()),
        };
        Ok(self.service.call_tool(params).await?)
    }

    /// 优雅关闭子进程。
    pub async fn shutdown(self) {
        let _ = self.service.cancel().await;
    }
}
