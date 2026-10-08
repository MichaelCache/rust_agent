//! `rust_agent`：命令行入口。
//!
//! 用法：
//! ```text
//! cargo run -- "查看 src/main.rs 里的 TODO 并列出来"
//! cargo run -- --workspace /path/to/project --max-steps 40 "把 foo() 改名为 bar()"
//! cargo run                # 无参数进入交互模式
//! ```

use std::{io::Write, process::ExitCode, time::Duration};

use rust_agent::{
    Agent, AgentConfig,
    agent::{
        DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MAX_STEPS, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECS,
    },
    init_tracing,
    stream::{ChatEvent, ChatStreamHandler, HandlerRef},
};
use tokio::io::{AsyncBufReadExt, BufReader};

const USAGE: &str = "\
rust_agent — 基于 Ollama(OpenAI 兼容接口) + MCP 工具的代码 Agent

用法:
  rust_agent [选项] [任务...]     给出任务则执行一次后退出
  rust_agent [选项]               不带任务则进入交互模式（REPL）

选项:
  -m, --model <名称>        模型名，默认 qwen3.8:latest（环境变量 AGENT_MODEL）
      --base-url <URL>      OpenAI 兼容接口地址，默认 http://localhost:11434/v1（环境变量 OPENAI_BASE_URL）
      --api-key <KEY>       API key，Ollama 不校验，默认 ollama（环境变量 OPENAI_API_KEY）
  -C, --workspace <目录>    工作目录（读写沙箱根），默认当前目录（环境变量 AGENT_WORKSPACE）
      --server <路径>       code-tools-server 路径，默认自动查找（环境变量 CODE_TOOLS_SERVER）
      --max-steps <N>       最多工具调用轮数，默认 25
      --timeout <秒>        单次 LLM 请求超时，默认 600
      --temperature <F>     采样温度，默认用服务端默认值
      --system <提示词>     覆盖默认 system prompt
      --allow-shell         允许 run_command 执行任意 shell 命令行（管道/重定向）；默认关闭
      --allow-command <名>  把程序加入 run_command 白名单，可重复使用
  -q, --quiet               只输出最终回答（关闭 stderr 进度日志与思考内容）
      --no-thinking         不显示模型的思考过程（只流式输出最终回答）
  -h, --help                显示本帮助

环境变量:
  RUST_LOG                  例如 RUST_LOG=rust_agent=debug 看每次工具调用细节

交互模式内置命令:
  /reset  清空对话上下文    /help  帮助    /exit  退出
";

const REPL_HELP: &str = "\
可用命令:
  /reset   清空对话历史（保留 system prompt）
  /help    显示本帮助
  /exit    退出（也可 Ctrl-D）
其它任意输入都会被当作任务交给 Agent；Agent 会在同一会话里记住上下文。
回答是流式输出的：思考内容以暗色竖线标出（--no-thinking 可关），最终回答实时打印。
";

#[tokio::main]
async fn main() -> ExitCode {
    let cli = match Cli::parse(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(msg) => {
            eprintln!("参数错误: {msg}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    if cli.help {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    init_tracing(if cli.quiet {
        "rust_agent=error,warn"
    } else {
        "rust_agent=info,warn"
    });

    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\n❌ {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> rust_agent::Result<()> {
    let mut agent = Agent::connect(cli.config.clone()).await?;

    eprintln!(
        "模型: {}  接口: {}\n工作目录: {}\n工具: {}\n命令执行: {}\n",
        agent.model(),
        cli.config.base_url,
        agent.workspace().display(),
        agent.tool_names().join(", "),
        if cli.config.allow_shell {
            "shell 模式（允许管道/重定向，风险自负）".to_string()
        } else {
            format!(
                "白名单 argv 模式（追加程序: {}）",
                if cli.config.extra_allowed_commands.is_empty() {
                    "无".to_string()
                } else {
                    cli.config.extra_allowed_commands.join(", ")
                }
            )
        }
    );

    let mut printer = Printer::new(cli.show_thinking, cli.quiet);

    match cli.task {
        Some(task) => {
            let answer = agent
                .ask_stream_with(&task, HandlerRef(&mut printer))
                .await?;
            // --quiet 是给管道用的，正文已经逐字流出来了，不能再重复一遍
            if answer.trim().is_empty() && !cli.quiet {
                println!("（模型没有返回任何内容，可能是上下文过长或服务端异常）");
            }
        }
        None => repl(&mut agent, &mut printer).await?,
    }

    agent.shutdown().await;
    Ok(())
}

async fn repl(agent: &mut Agent, printer: &mut Printer) -> rust_agent::Result<()> {
    eprintln!("交互模式（/help 看命令，Ctrl-D 退出）");

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!("\n> ");
        let _ = std::io::stdout().flush();
        // 流式输出会把提示符顶掉，先记下来，等真正有内容时再抹掉
        printer.mark_prompt();

        let Some(line) = lines.next_line().await? else {
            eprintln!("\n再见。");
            return Ok(());
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => {
                eprintln!("再见。");
                return Ok(());
            }
            "/reset" => {
                agent.reset();
                eprintln!("已清空对话历史。");
                continue;
            }
            "/help" => {
                eprint!("{REPL_HELP}");
                continue;
            }
            _ => {}
        }

        // 流式输出会把提示符顶掉，先自己抹掉它
        match agent.ask_stream_with(input, HandlerRef(printer)).await {
            Ok(_) => {}
            // 单次任务失败不退出会话，方便直接改参数重试
            Err(e) => eprintln!("\n❌ {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 终端流式输出
// ---------------------------------------------------------------------------

/// 把流式对话事件实时打到终端。
///
/// - 思考内容（[`ChatEvent::Reasoning`]）默认用暗色 + 竖线标出来，`--no-thinking` 可关；
/// - 正文（[`ChatEvent::Text`]）逐字写出并立即 flush，所以是真正的“边生成边显示”；
/// - 正文全部走 stdout，进度/思考标记走 stderr，`--quiet` 时只看得到正文。
pub struct Printer {
    show_thinking: bool,
    quiet: bool,
    /// 思考块是否已经开始（用来决定画不画分隔线）
    thinking_open: bool,
    /// 是否已经输出过正文
    printed_text: bool,
    /// 正文最后一个字符（避免模型自带换行时又多打一个空行）
    last_char: Option<char>,
    /// 提示符 `> ` 是否还挂在屏幕上（需要回车擦掉）
    prompt_pending: bool,
}

impl Printer {
    pub fn new(show_thinking: bool, quiet: bool) -> Self {
        Self {
            show_thinking: show_thinking && !quiet,
            quiet,
            thinking_open: false,
            printed_text: false,
            last_char: None,
            prompt_pending: false,
        }
    }

    /// 抹掉 REPL 刚打出来的提示符，免得它粘在流式输出的开头。
    pub fn clear_prompt(&mut self) {
        if self.prompt_pending {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
            self.prompt_pending = false;
        }
    }

    fn mark_prompt(&mut self) {
        self.prompt_pending = true;
    }

    fn open_thinking(&mut self) -> rust_agent::Result<()> {
        if self.thinking_open {
            return Ok(());
        }
        self.clear_prompt();
        self.thinking_open = true;
        let mut err = std::io::stderr();
        write!(err, "\n\x1b[2m\x1b[36m 思考中…\x1b[0m\n")?;
        Ok(())
    }

    fn close_thinking(&mut self) -> rust_agent::Result<()> {
        if !self.thinking_open {
            return Ok(());
        }
        self.thinking_open = false;
        let mut err = std::io::stderr();
        writeln!(err, "\x1b[2m\x1b[36m{line}\x1b[0m", line = "─".repeat(48))?;
        err.flush()?;
        Ok(())
    }
}

impl ChatStreamHandler for Printer {
    fn on_event(&mut self, event: &ChatEvent) -> rust_agent::Result<()> {
        match event {
            ChatEvent::Reasoning(delta) => {
                if self.show_thinking {
                    self.open_thinking()?;
                    let mut err = std::io::stderr();
                    write!(err, "\x1b[2m\x1b[36m│ {delta}\x1b[0m")?;
                    err.flush()?;
                }
            }
            ChatEvent::Text(delta) => {
                self.clear_prompt();
                self.close_thinking()?;
                self.printed_text = true;
                self.last_char = delta.chars().next_back();
                let mut out = std::io::stdout();
                write!(out, "{delta}")?;
                out.flush()?;
            }
            ChatEvent::ToolCall { name, .. } => {
                self.clear_prompt();
                self.close_thinking()?;
                if !self.quiet {
                    eprintln!("\n⚙ 调用工具 {name}");
                }
            }
            ChatEvent::Finished { .. } => {
                self.clear_prompt();
                self.close_thinking()?;
                // 只在需要时补一个换行：模型自己带换行时不再多打空行
                if !self.quiet && self.printed_text && self.last_char != Some('\n') {
                    let mut out = std::io::stdout();
                    writeln!(out)?;
                    out.flush()?;
                }
                self.printed_text = false;
                self.last_char = None;
                self.mark_prompt();
            }
            ChatEvent::ToolArguments { .. } => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 参数解析（手写，避免为一个 CLI 引入 clap）
// ---------------------------------------------------------------------------

struct Cli {
    config: AgentConfig,
    task: Option<String>,
    quiet: bool,
    show_thinking: bool,
    help: bool,
}

impl Cli {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut cfg = AgentConfig {
            model: env_or("AGENT_MODEL", DEFAULT_MODEL),
            base_url: env_or("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            api_key: env_or("OPENAI_API_KEY", DEFAULT_API_KEY),
            workspace: std::env::var("AGENT_WORKSPACE")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| ".".into())),
            server_bin: std::env::var("CODE_TOOLS_SERVER")
                .ok()
                .map(std::path::PathBuf::from),
            max_steps: DEFAULT_MAX_STEPS,
            allow_shell: env_flag("CODE_TOOLS_ALLOW_SHELL"),
            extra_allowed_commands: env_list("CODE_TOOLS_ALLOW_COMMANDS"),
            temperature: None,
            request_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            ..Default::default()
        };

        let mut task_parts: Vec<String> = Vec::new();
        let mut quiet = false;
        let mut show_thinking = true;
        let mut help = false;
        let mut args = args.peekable();

        while let Some(arg) = args.next() {
            let mut take_value = |name: &str| -> Result<String, String> {
                args.next().ok_or_else(|| format!("{name} 需要一个参数值"))
            };
            match arg.as_str() {
                "-h" | "--help" => help = true,
                "-q" | "--quiet" => quiet = true,
                "--no-thinking" => show_thinking = false,
                "-m" | "--model" => cfg.model = take_value("--model")?,
                "--base-url" => cfg.base_url = take_value("--base-url")?,
                "--api-key" => cfg.api_key = take_value("--api-key")?,
                "-C" | "--workspace" => {
                    cfg.workspace = std::path::PathBuf::from(take_value("--workspace")?)
                }
                "--server" => cfg.server_bin = Some(take_value("--server")?.into()),
                "--system" => cfg.system_prompt = Some(take_value("--system")?),
                "--allow-shell" => cfg.allow_shell = true,
                "--allow-command" => cfg
                    .extra_allowed_commands
                    .push(take_value("--allow-command")?),
                "--max-steps" => {
                    cfg.max_steps = take_value("--max-steps")?
                        .parse()
                        .map_err(|e| format!("--max-steps 需要正整数: {e}"))?
                }
                "--timeout" => {
                    let secs: u64 = take_value("--timeout")?
                        .parse()
                        .map_err(|e| format!("--timeout 需要秒数: {e}"))?;
                    cfg.request_timeout = Duration::from_secs(secs.max(1));
                }
                "--temperature" => {
                    cfg.temperature = Some(
                        take_value("--temperature")?
                            .parse()
                            .map_err(|e| format!("--temperature 需要数字: {e}"))?,
                    )
                }
                "--" => {
                    task_parts.extend(args.by_ref());
                    break;
                }
                other if other.starts_with('-') && other.len() > 1 => {
                    return Err(format!("未知参数 {other}（用 --help 查看用法）"));
                }
                other => task_parts.push(other.to_string()),
            }
        }

        if cfg.max_steps == 0 {
            return Err("--max-steps 必须大于 0".to_string());
        }

        let task = if task_parts.is_empty() {
            None
        } else {
            Some(task_parts.join(" "))
        };

        Ok(Self {
            config: cfg,
            task,
            quiet,
            show_thinking,
            help,
        })
    }
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

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}
