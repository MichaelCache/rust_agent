//! `code-tools-server`：以 stdio 方式提供代码读写工具的 MCP 服务器。

use std::{io::ErrorKind, path::PathBuf, process::ExitCode};

use rmcp::{ServiceExt, transport::stdio};
use rust_agent::{
    init_tracing,
    mcp::{CodeTools, CommandPolicy},
};

const USAGE: &str = "\
code-tools-server — MCP stdio 服务器（读/写/搜索/编辑代码 + 执行命令）

用法:
  code-tools-server [--workspace <目录>] [命令策略选项]

参数:
  --workspace <目录>        允许读写的根目录，默认当前目录
  --allow-shell             允许 run_command 用 sh -c 执行任意命令行（管道/重定向），默认关闭
  --allow-command <程序名>  把程序加入 run_command 白名单，可重复使用
  -h, --help                显示帮助

环境变量:
  CODE_TOOLS_ALLOW_SHELL=1              等同 --allow-shell
  CODE_TOOLS_ALLOW_COMMANDS=a,b,c       等同重复 --allow-command

说明:
  - 默认只允许白名单程序，并以 argv 方式执行（不经 shell，无法注入）；
  - 黑名单护栏（rm -rf /、sudo、git reset --hard…）只是防手滑，不是安全边界；
  - 命令以当前用户权限运行，能访问工作目录之外的文件。
  - 该进程只通过 stdin/stdout 说 JSON-RPC，日志一律走 stderr。
  - 通常不用手动启动，由 rust_agent 自动拉起。
";

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing("rust_agent=info,warn");

    let args = match parse_args() {
        Ok(Some(args)) => args,
        Ok(None) => return ExitCode::SUCCESS, // --help
        Err(msg) => {
            eprintln!("参数错误: {msg}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    match serve(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("code-tools-server 退出: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn serve(args: ServerArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut policy = CommandPolicy::with_extra_programs(&args.extra_commands);
    policy.allow_shell = args.allow_shell;

    let tools = CodeTools::with_policy(args.workspace, policy)
        .map_err(|e| std::io::Error::new(ErrorKind::InvalidInput, e))?;
    tracing::info!(
        workspace = %tools.workspace().display(),
        allow_shell = args.allow_shell,
        extra_commands = ?args.extra_commands,
        "code-tools-server 就绪"
    );

    let service = tools.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

struct ServerArgs {
    workspace: PathBuf,
    allow_shell: bool,
    extra_commands: Vec<String>,
}

/// 返回 `Ok(None)` 表示用户要的是帮助信息。
fn parse_args() -> Result<Option<ServerArgs>, String> {
    let mut args = std::env::args().skip(1);
    let mut workspace: Option<PathBuf> = None;
    let mut allow_shell = env_flag("CODE_TOOLS_ALLOW_SHELL");
    let mut extra_commands = env_list("CODE_TOOLS_ALLOW_COMMANDS");

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workspace" | "-w" => {
                let value = args.next().ok_or("--workspace 需要一个目录参数")?;
                workspace = Some(PathBuf::from(value));
            }
            "--allow-shell" => allow_shell = true,
            "--allow-command" => {
                let value = args.next().ok_or("--allow-command 需要一个程序名")?;
                extra_commands.push(value);
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            other => return Err(format!("未知参数: {other}")),
        }
    }

    let workspace = match workspace {
        Some(ws) => ws,
        None => std::env::current_dir().map_err(|e| format!("无法获取当前目录: {e}"))?,
    };
    Ok(Some(ServerArgs {
        workspace,
        allow_shell,
        extra_commands,
    }))
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "no"
        })
        .unwrap_or(false)
}

fn env_list(key: &str) -> Vec<String> {
    std::env::var(key)
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}
