//! 代码读写工具集（MCP 服务器）。
//!
//! 六个工具：read_file / list_directory / search_code / write_file / edit_file / run_command。
//! 文件类工具通过 [`Workspace`] 把操作限制在工作目录内；
//! `run_command` 是真正的进程执行，见 [`crate::mcp::command`] 里的安全模型说明。
//!
//! 与原实现的区别：
//! - 工具方法返回 `Result<String, String>`，出错时 MCP 会带上 `isError: true`，
//!   模型能明确看到失败原因，而不是把错误当成正常结果；
//! - `read_file` 的越界行号不再 panic；
//! - `list_directory` 真正支持递归，并跳过 `.git`/`target` 之类噪声目录；
//! - `search_code` 默认按普通文本匹配（原实现强制当正则，遇到 `(`、`[` 就报错），
//!   可用 `regex: true` 切换；并限制结果条数与单文件大小；
//! - `edit_file` 在“多处匹配且未指定 replace_all”时报错，避免改错地方；
//! - `write_file` 会自动创建父目录，并保留可选的 `.bak` 备份。

use std::path::{Path, PathBuf};
use std::time::Duration;

use rmcp::{ServerHandler, handler::server::wrapper::Parameters, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::mcp::command::{self, CommandPolicy, build_invocation};
use crate::text::truncate_chars;
use crate::workspace::Workspace;

const DEFAULT_READ_LINES: usize = 400;
const MAX_READ_LINES: usize = 4000;
const MAX_LINE_CHARS: usize = 2000;
const MAX_LIST_ENTRIES: usize = 800;
const DEFAULT_SEARCH_RESULTS: usize = 100;
const MAX_SEARCH_RESULTS: usize = 500;
const MAX_SEARCH_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_SEARCH_FILES: usize = 20_000;
const MAX_MATCHED_LINE_CHARS: usize = 300;

/// 递归时默认跳过的目录（噪声大、体积大）。
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
];

// ---------------------------------------------------------------------------
// 工具入参
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadFileRequest {
    /// 文件路径，相对于工作目录，也可以是工作目录内的绝对路径
    #[schemars(description = "文件路径（相对工作目录，或工作目录内的绝对路径）")]
    pub path: String,
    /// 从第几行开始读（1 起，含）
    #[schemars(description = "起始行号，从 1 开始（含）。默认 1")]
    pub start_line: Option<usize>,
    /// 最多读多少行
    #[schemars(description = "最多读取的行数。默认 400，上限 4000")]
    pub line_count: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListDirectoryRequest {
    #[schemars(description = "目录路径（相对工作目录）。省略则为工作目录根")]
    pub path: Option<String>,
    #[schemars(description = "是否递归列出子目录。默认 false")]
    pub recursive: Option<bool>,
    #[schemars(description = "递归深度上限（仅在 recursive=true 时有效）。默认 3，上限 8")]
    pub max_depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchCodeRequest {
    #[schemars(description = "要搜索的内容：默认按普通文本子串匹配；regex=true 时按正则表达式")]
    pub pattern: String,
    #[schemars(description = "搜索起始路径（相对工作目录）。省略则为工作目录根")]
    pub path: Option<String>,
    #[schemars(description = "是否把 pattern 当作正则表达式。默认 false（普通文本，无需转义）")]
    pub regex: Option<bool>,
    #[schemars(description = "只搜索该扩展名，例如 \"rs\" 或 \".rs\"。省略则搜索所有文本文件")]
    pub extension: Option<String>,
    #[schemars(description = "是否区分大小写。默认 false")]
    pub case_sensitive: Option<bool>,
    #[schemars(description = "最多返回多少条匹配。默认 100，上限 500")]
    pub max_results: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteFileRequest {
    #[schemars(description = "要写入的文件路径（相对工作目录）")]
    pub path: String,
    #[schemars(description = "完整的新文件内容（覆盖写入）")]
    pub content: String,
    #[schemars(description = "父目录不存在时是否自动创建。默认 true")]
    pub create_dirs: Option<bool>,
    #[schemars(description = "覆盖已有文件前是否备份为 <文件名>.bak。默认 false")]
    pub backup: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EditFileRequest {
    #[schemars(description = "要编辑的文件路径（相对工作目录）")]
    pub path: String,
    #[schemars(description = "要被替换的原文，必须与文件内容逐字符精确匹配（包含缩进与换行）")]
    pub old_text: String,
    #[schemars(description = "替换后的新文本")]
    pub new_text: String,
    #[schemars(
        description = "是否替换全部匹配。默认 false：此时若匹配到多处会报错，需要你提供更多上下文或显式设为 true"
    )]
    pub replace_all: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunCommandRequest {
    #[schemars(
        description = "要执行的程序名，例如 \"cargo\"、\"git\"、\"rg\"。必须是白名单内的程序；只有开启 shell 模式后才可以传完整命令行"
    )]
    pub command: String,
    #[schemars(
        description = "程序参数列表，例如 [\"test\", \"--quiet\"]。默认模式下必须提供（命令按 argv 直接执行，不经过 shell）"
    )]
    pub args: Option<Vec<String>>,
    #[schemars(description = "工作目录（相对工作目录根）。省略则为工作目录根")]
    pub cwd: Option<String>,
    #[schemars(description = "超时秒数。默认 60，上限 600")]
    pub timeout_secs: Option<u64>,
}

// ---------------------------------------------------------------------------
// 服务器实现
// ---------------------------------------------------------------------------

/// rmcp 3.x 的 `#[tool_handler]` 默认调用 `Self::tool_router()`（由 `#[tool_router]` 生成），
/// 因此结构体里不需要再存一份 router。
#[derive(Debug, Clone)]
pub struct CodeTools {
    ws: Workspace,
    commands: CommandPolicy,
}

#[tool_router]
impl CodeTools {
    pub fn new(workspace: PathBuf) -> Result<Self, String> {
        Self::with_policy(workspace, CommandPolicy::default())
    }

    pub fn with_policy(workspace: PathBuf, commands: CommandPolicy) -> Result<Self, String> {
        Ok(Self {
            ws: Workspace::new(workspace)?,
            commands,
        })
    }

    /// 只读暴露工作目录，供启动日志使用。
    pub fn workspace(&self) -> &Path {
        self.ws.root()
    }

    #[tool(
        description = "读取文本文件内容，返回带行号的输出。大文件请用 start_line/line_count 分段读取。"
    )]
    async fn read_file(
        &self,
        Parameters(req): Parameters<ReadFileRequest>,
    ) -> Result<String, String> {
        let path = self.ws.resolve(&req.path)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|e| format!("无法访问 {}: {e}", self.ws.display(&path)))?;
        if meta.is_dir() {
            return Err(format!(
                "{} 是目录，请改用 list_directory",
                self.ws.display(&path)
            ));
        }
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| format!("读取失败 {}: {e}", self.ws.display(&path)))?;
        if bytes.iter().take(8192).any(|b| *b == 0) {
            return Err(format!(
                "{} 看起来是二进制文件（{} 字节），已拒绝读取",
                self.ws.display(&path),
                bytes.len()
            ));
        }
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();

        let start_idx = req
            .start_line
            .unwrap_or(1)
            .max(1)
            .saturating_sub(1)
            .min(total);
        let want = req
            .line_count
            .unwrap_or(DEFAULT_READ_LINES)
            .clamp(1, MAX_READ_LINES);
        let end_idx = (start_idx + want).min(total);

        let mut out = format!(
            "{} （共 {} 行，{} 字节）\n",
            self.ws.display(&path),
            total,
            bytes.len()
        );
        if total == 0 {
            out.push_str("[文件为空]\n");
            return Ok(out);
        }
        if start_idx >= total {
            out.push_str(&format!(
                "[start_line={} 超出文件末尾，文件只有 {} 行]\n",
                req.start_line.unwrap_or(1),
                total
            ));
            return Ok(out);
        }
        for (i, line) in lines[start_idx..end_idx].iter().enumerate() {
            out.push_str(&format!(
                "{:>5} | {}\n",
                start_idx + i + 1,
                truncate_chars(line, MAX_LINE_CHARS)
            ));
        }
        if end_idx < total {
            out.push_str(&format!(
                "\n[已显示第 {}-{} 行，共 {} 行；继续读取请设置 start_line={}]\n",
                start_idx + 1,
                end_idx,
                total,
                end_idx + 1
            ));
        }
        Ok(out)
    }

    #[tool(description = "列出目录内容（可递归），目录以 [DIR] 标记，文件带字节数。")]
    async fn list_directory(
        &self,
        Parameters(req): Parameters<ListDirectoryRequest>,
    ) -> Result<String, String> {
        let dir = self.ws.resolve(req.path.as_deref().unwrap_or("."))?;
        let meta = tokio::fs::metadata(&dir)
            .await
            .map_err(|e| format!("无法访问 {}: {e}", self.ws.display(&dir)))?;
        if !meta.is_dir() {
            return Err(format!(
                "{} 不是目录，读取内容请用 read_file",
                self.ws.display(&dir)
            ));
        }
        let recursive = req.recursive.unwrap_or(false);
        let max_depth = req.max_depth.unwrap_or(3).clamp(1, 8);

        let mut out = String::new();
        let mut count = 0usize;
        let mut truncated = false;
        let mut skipped: Vec<String> = Vec::new();

        // (目录, 相对深度)
        let mut stack: Vec<(PathBuf, usize)> = vec![(dir.clone(), 0)];
        while let Some((current, depth)) = stack.pop() {
            let mut rd = tokio::fs::read_dir(&current)
                .await
                .map_err(|e| format!("目录读取失败 {}: {e}", self.ws.display(&current)))?;
            let mut entries: Vec<(PathBuf, bool, u64)> = Vec::new();
            while let Some(entry) = rd
                .next_entry()
                .await
                .map_err(|e| format!("目录读取失败 {}: {e}", self.ws.display(&current)))?
            {
                let p = entry.path();
                let Ok(ft) = entry.file_type().await else {
                    continue;
                };
                let is_dir = ft.is_dir();
                let size = if is_dir {
                    0
                } else {
                    entry.metadata().await.map(|m| m.len()).unwrap_or(0)
                };
                entries.push((p, is_dir, size));
            }
            entries.sort_by(|a, b| a.0.cmp(&b.0));

            for (p, is_dir, size) in entries {
                if count >= MAX_LIST_ENTRIES {
                    truncated = true;
                    break;
                }
                let name = p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                if is_dir {
                    // 噪声目录照常列出（让用户看得见它存在），只是递归时不进去
                    out.push_str(&format!("[DIR]  {}/\n", self.ws.display(&p)));
                    count += 1;
                    if recursive {
                        if IGNORED_DIRS.contains(&name.as_str()) {
                            skipped.push(self.ws.display(&p));
                        } else if depth < max_depth {
                            stack.push((p, depth + 1));
                        }
                    }
                } else {
                    out.push_str(&format!("[FILE] {} ({} 字节)\n", self.ws.display(&p), size));
                    count += 1;
                }
            }
            if truncated {
                break;
            }
        }

        if out.is_empty() {
            out.push_str("[目录为空]\n");
        }
        if truncated {
            out.push_str(&format!("\n[结果超过 {MAX_LIST_ENTRIES} 条已截断]\n"));
        }
        if !skipped.is_empty() {
            out.push_str(&format!(
                "\n[递归时未展开的噪声目录（需要时请直接指定 path）: {}]\n",
                skipped.join(", ")
            ));
        }
        Ok(out)
    }

    #[tool(
        description = "在目录中递归搜索代码内容，返回 路径:行号: 内容。默认按普通文本匹配（无需转义），regex=true 时按正则匹配。"
    )]
    async fn search_code(
        &self,
        Parameters(req): Parameters<SearchCodeRequest>,
    ) -> Result<String, String> {
        if req.pattern.is_empty() {
            return Err("pattern 不能为空".to_string());
        }
        let root = self.ws.resolve(req.path.as_deref().unwrap_or("."))?;
        let matcher = Matcher::new(
            &req.pattern,
            req.regex.unwrap_or(false),
            req.case_sensitive.unwrap_or(false),
        )?;
        let ext_filter = req.extension.as_ref().map(|e| {
            if e.starts_with('.') {
                e.clone()
            } else {
                format!(".{e}")
            }
        });
        let max_results = req
            .max_results
            .unwrap_or(DEFAULT_SEARCH_RESULTS)
            .clamp(1, MAX_SEARCH_RESULTS);

        let mut results: Vec<String> = Vec::new();
        let mut scanned_files = 0usize;
        let mut truncated = false;
        let mut hit_file_cap = false;

        // 目标可能是一个文件（模型经常直接给出文件名），也可能是一个目录
        let root_meta = tokio::fs::metadata(&root)
            .await
            .map_err(|e| format!("无法访问 {}: {e}", self.ws.display(&root)))?;
        let single_file = root_meta.is_file();

        let candidates = if single_file {
            vec![root.clone()]
        } else {
            let mut files: Vec<PathBuf> = Vec::new();
            let mut stack = vec![root.clone()];
            let mut is_root_dir = true;
            while let Some(current) = stack.pop() {
                let mut rd = match tokio::fs::read_dir(&current).await {
                    Ok(rd) => rd,
                    // 指定目录本身读不了 → 直接报错；子目录读不了 → 跳过
                    Err(e) if is_root_dir => {
                        return Err(format!("目录读取失败 {}: {e}", self.ws.display(&current)));
                    }
                    Err(_) => continue,
                };
                is_root_dir = false;
                while let Some(entry) = rd.next_entry().await.map_err(|e| e.to_string())? {
                    let p = entry.path();
                    let Ok(ft) = entry.file_type().await else {
                        continue;
                    };
                    let name = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    if ft.is_dir() {
                        if IGNORED_DIRS.contains(&name.as_str()) {
                            continue;
                        }
                        stack.push(p);
                        continue;
                    }
                    if let Some(ext) = &ext_filter
                        && !name.ends_with(ext.as_str())
                    {
                        continue;
                    }
                    if files.len() >= MAX_SEARCH_FILES {
                        hit_file_cap = true;
                        break;
                    }
                    files.push(p);
                }
            }
            files
        };

        'outer: for p in candidates {
            let Ok(meta) = tokio::fs::metadata(&p).await else {
                continue;
            };
            if meta.len() > MAX_SEARCH_FILE_BYTES {
                continue;
            }
            let Ok(bytes) = tokio::fs::read(&p).await else {
                continue;
            };
            if bytes.iter().take(4096).any(|b| *b == 0) {
                continue; // 二进制
            }
            scanned_files += 1;
            let text = String::from_utf8_lossy(&bytes);
            for (lineno, line) in text.lines().enumerate() {
                if matcher.is_match(line) {
                    results.push(format!(
                        "{}:{}: {}",
                        self.ws.display(&p),
                        lineno + 1,
                        truncate_chars(line.trim(), MAX_MATCHED_LINE_CHARS)
                    ));
                    if results.len() >= max_results {
                        truncated = true;
                        break 'outer;
                    }
                }
            }
        }

        if results.is_empty() {
            let scope = if single_file {
                format!("文件 {}", self.ws.display(&root))
            } else {
                format!("已扫描 {scanned_files} 个文件")
            };
            let extra = if hit_file_cap {
                format!("（文件数已达上限 {MAX_SEARCH_FILES}，建议缩小 path 或加 extension）")
            } else {
                String::new()
            };
            return Ok(format!(
                "未找到匹配（{scope}）{extra}。可以放宽 pattern；若 pattern 含正则元字符，默认按普通文本搜索。"
            ));
        }
        let mut out = results.join("\n");
        out.push_str(&format!("\n\n[共 {} 条匹配", results.len()));
        if truncated {
            out.push_str(&format!("，已达上限 {max_results} 条，可能还有更多"));
        }
        out.push_str(&format!("；已扫描 {scanned_files} 个文件]\n"));
        Ok(out)
    }

    #[tool(description = "把完整内容写入文件（覆盖写入）。父目录不存在时会自动创建。")]
    async fn write_file(
        &self,
        Parameters(req): Parameters<WriteFileRequest>,
    ) -> Result<String, String> {
        let path = self.ws.resolve(&req.path)?;
        if path.is_dir() {
            return Err(format!("{} 是目录，不能写入", self.ws.display(&path)));
        }
        let previous_len = tokio::fs::metadata(&path).await.map(|m| m.len()).ok();

        if req.create_dirs.unwrap_or(true)
            && let Some(parent) = path.parent()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("创建目录失败 {}: {e}", self.ws.display(parent)))?;
        }
        if req.backup.unwrap_or(false) && previous_len.is_some() {
            let mut bak = path.clone().into_os_string();
            bak.push(".bak");
            let bak = PathBuf::from(bak);
            tokio::fs::copy(&path, &bak)
                .await
                .map_err(|e| format!("备份失败 {}: {e}", self.ws.display(&bak)))?;
        }

        tokio::fs::write(&path, req.content.as_bytes())
            .await
            .map_err(|e| format!("写入失败 {}: {e}", self.ws.display(&path)))?;

        let action = match previous_len {
            Some(old) => format!("已覆盖 {}（原 {old} 字节）", self.ws.display(&path)),
            None => format!("已创建 {}", self.ws.display(&path)),
        };
        Ok(format!(
            "{action}，写入 {} 字节 / {} 行",
            req.content.len(),
            req.content.lines().count()
        ))
    }

    #[tool(
        description = "在文件中精确替换文本。old_text 必须逐字符匹配（含缩进）；默认要求唯一匹配，否则报错。"
    )]
    async fn edit_file(
        &self,
        Parameters(req): Parameters<EditFileRequest>,
    ) -> Result<String, String> {
        if req.old_text.is_empty() {
            return Err("old_text 不能为空；要整体重写文件请用 write_file".to_string());
        }
        if req.old_text == req.new_text {
            return Err("old_text 与 new_text 相同，无需修改".to_string());
        }
        let path = self.ws.resolve(&req.path)?;
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| format!("读取失败 {}: {e}", self.ws.display(&path)))?;

        // 允许模型用 \n 写 old_text 而文件是 CRLF
        let (old_text, new_text) = if content.contains(&req.old_text) {
            (req.old_text.clone(), req.new_text.clone())
        } else if content.contains("\r\n") && content.contains(&to_crlf(&req.old_text)) {
            (to_crlf(&req.old_text), to_crlf(&req.new_text))
        } else {
            return Err(not_found_hint(&content, &req.old_text));
        };

        let matches = find_matches(&content, &old_text);
        let replace_all = req.replace_all.unwrap_or(false);
        if matches.len() > 1 && !replace_all {
            let lines: Vec<String> = matches
                .iter()
                .take(10)
                .map(|(l, _)| l.to_string())
                .collect();
            return Err(format!(
                "old_text 在 {} 中匹配到 {} 处（行 {}），为避免改错位置已中止。请提供更长、更唯一的上下文，或显式设置 replace_all=true。",
                self.ws.display(&path),
                matches.len(),
                lines.join(", ")
            ));
        }

        let new_content = if replace_all {
            content.replace(&old_text, &new_text)
        } else {
            content.replacen(&old_text, &new_text, 1)
        };
        tokio::fs::write(&path, new_content.as_bytes())
            .await
            .map_err(|e| format!("写入失败 {}: {e}", self.ws.display(&path)))?;

        let lines: Vec<String> = matches.iter().map(|(l, _)| l.to_string()).collect();
        let old_lines = old_text.lines().count();
        let new_lines = new_text.lines().count();
        Ok(format!(
            "已将 {} 的第 {} 行处 {} 处匹配替换完成（该片段 {} 行 → {} 行）。",
            self.ws.display(&path),
            lines.join(", "),
            matches.len(),
            old_lines,
            new_lines
        ))
    }

    #[tool(
        description = "在工作目录内执行命令并返回结果（退出码、stdout、stderr）。默认只允许白名单程序，且必须用 command + args 的 argv 形式，不支持管道与重定向；主要用于 cargo check/test、git status/diff、rg 等验证与查看。命令能读写工作目录之外的文件，请只在必要时使用。"
    )]
    async fn run_command(
        &self,
        Parameters(req): Parameters<RunCommandRequest>,
    ) -> Result<String, String> {
        let invocation = build_invocation(&req.command, req.args.as_deref(), &self.commands)?;

        let cwd = match req.cwd.as_deref() {
            Some(dir) => {
                let path = self.ws.resolve(dir)?;
                let meta = tokio::fs::metadata(&path)
                    .await
                    .map_err(|e| format!("无法访问 cwd {}: {e}", self.ws.display(&path)))?;
                if !meta.is_dir() {
                    return Err(format!("cwd {} 不是目录", self.ws.display(&path)));
                }
                path
            }
            None => self.ws.root().to_path_buf(),
        };

        let timeout = Duration::from_secs(
            req.timeout_secs
                .unwrap_or(self.commands.default_timeout.as_secs())
                .clamp(1, self.commands.max_timeout.as_secs()),
        );

        command::execute(invocation, &cwd, timeout, self.commands.max_output_chars).await
    }
}

#[tool_handler(
    name = "code-tools-server",
    version = "0.1.0",
    instructions = "代码读写工具集：读取文件、列目录、搜索代码、写入文件、精确编辑、执行命令（默认白名单 argv 模式）。文件类工具的路径都被限制在工作目录内。"
)]
impl ServerHandler for CodeTools {}

// ---------------------------------------------------------------------------
// 辅助
// ---------------------------------------------------------------------------

enum Matcher {
    Literal {
        needle: String,
        case_sensitive: bool,
    },
    Regex(regex::Regex),
}

impl Matcher {
    fn new(pattern: &str, is_regex: bool, case_sensitive: bool) -> Result<Self, String> {
        if is_regex {
            let expr = if case_sensitive {
                pattern.to_string()
            } else {
                format!("(?i){pattern}")
            };
            regex::Regex::new(&expr).map(Self::Regex).map_err(|e| {
                format!("正则表达式无效: {e}（若想按普通文本搜索，请设置 regex=false）")
            })
        } else {
            Ok(Self::Literal {
                needle: if case_sensitive {
                    pattern.to_string()
                } else {
                    pattern.to_lowercase()
                },
                case_sensitive,
            })
        }
    }

    fn is_match(&self, line: &str) -> bool {
        match self {
            Self::Literal {
                needle,
                case_sensitive,
            } => {
                if *case_sensitive {
                    line.contains(needle.as_str())
                } else {
                    line.to_lowercase().contains(needle.as_str())
                }
            }
            Self::Regex(re) => re.is_match(line),
        }
    }
}

fn to_crlf(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\n', "\r\n")
}

/// 返回 (行号, 字节偏移) 列表
fn find_matches(content: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(pos) = content[from..].find(needle) {
        let abs = from + pos;
        let line = content[..abs].matches('\n').count() + 1;
        out.push((line, abs));
        from = abs + needle.len();
        if out.len() >= 1000 {
            break;
        }
    }
    out
}

fn not_found_hint(content: &str, old_text: &str) -> String {
    let mut hint = String::from("old_text 未在文件中找到。");
    if content.contains(old_text.trim()) {
        hint.push_str(" 注意：去掉首尾空白后可以匹配，可能是缩进或行尾空白不一致。");
    } else if content.split_whitespace().collect::<String>()
        == old_text.split_whitespace().collect::<String>()
    {
        hint.push_str(" 注意：忽略空白后一致，请检查空格/缩进/换行符（文件可能是 CRLF）。");
    } else {
        hint.push_str(" 请先用 read_file 确认当前内容，再复制其中的原文作为 old_text。");
    }
    hint
}
