# rust_agent

一个跑在本机的极简代码 Agent：**Ollama（OpenAI 兼容接口）+ MCP 工具服务器**。

- LLM：`qwen3.8:latest`（经 `http://localhost:11434/v1` 调用，走 OpenAI 兼容协议）
- 工具：独立的 MCP stdio 子进程 `code-tools-server`，提供读 / 写 / 搜索 / 编辑代码，以及（受策略约束的）命令执行能力
- 主循环：LLM → 工具调用 → 结果回填 → 直到给出最终回答

```
┌──────────────┐  OpenAI /chat/completions   ┌─────────────────────┐
│  rust_agent  │ ──────────────────────────► │ Ollama (qwen3.8)    │
│  (Agent 循环) │ ◄──── SSE 增量（流式）────── │ 11434/v1            │
└──────┬───────┘        tool_calls           └─────────────────────┘
       │ MCP over stdio (JSON-RPC, 子进程)
       ▼
┌──────────────────────┐
│ code-tools-server    │  read_file / list_directory / search_code
│ (文件操作沙箱)        │  write_file / edit_file / run_command
└──────────────────────┘
```

`run_command` 是唯一会“跳出”文件沙箱的工具（它执行的是真实进程），默认受限运行，见下文。

## 快速开始

```bash
# 1. 确保 Ollama 在跑，并且模型存在
ollama serve &
ollama list          # 应能看到 qwen3.8:latest

# 2. 编译（一次生成两个二进制）
cargo build

# 3. 跑一个任务
cargo run -- --workspace /path/to/your/project "查看 src/main.rs，列出所有 TODO 并给出行号"

# 4. 或者进入交互模式（多轮，Agent 记得上下文）
cargo run -- --workspace /path/to/your/project
```

`cargo run` 会自动找到同目录下的 `code-tools-server`，不需要手动启动它。

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model <名称>` | 模型名 | `qwen3.8:latest`（`AGENT_MODEL`） |
| `--base-url <URL>` | OpenAI 兼容接口地址 | `http://localhost:11434/v1`（`OPENAI_BASE_URL`） |
| `--api-key <KEY>` | API key（Ollama 不校验） | `ollama`（`OPENAI_API_KEY`） |
| `-C, --workspace <目录>` | 允许读写的根目录 | 当前目录（`AGENT_WORKSPACE`） |
| `--server <路径>` | `code-tools-server` 路径 | 自动查找（`CODE_TOOLS_SERVER`） |
| `--max-steps <N>` | 最多工具调用轮数 | `25` |
| `--timeout <秒>` | 单次 LLM 请求超时 | `600` |
| `--temperature <F>` | 采样温度 | 服务端默认 |
| `--system <提示词>` | 覆盖默认 system prompt | 内置 |
| `--allow-shell` | 允许 `run_command` 执行任意 shell 命令行（管道/重定向） | 关闭（`CODE_TOOLS_ALLOW_SHELL=1`） |
| `--allow-command <名>` | 把程序加入 `run_command` 白名单，可重复 | 无（`CODE_TOOLS_ALLOW_COMMANDS=a,b`） |
| `-q, --quiet` | 只输出最终回答（关闭进度日志与思考内容） | 关闭 |
| `--no-thinking` | 不显示模型思考过程，只流式输出最终回答 | 关闭（默认显示思考） |

交互模式内置命令：`/reset` 清空上下文、`/help`、`/exit`。
看每次工具调用的细节：`RUST_LOG=rust_agent=debug cargo run -- "..."`。

### 流式输出

对话全程走 SSE 增量输出，**思考内容和正文都是边生成边显示的**：

- 思考过程（Ollama 的 `reasoning` 字段、其他网关的 `reasoning_content`、以及正文里的思维链标记）用暗色竖线标在 stderr 上，`--no-thinking` 或 `-q` 可关；
- 最终回答按增量实时写到 stdout 并立即 flush，可以直接 `| less`、`| tee` 或接别的程序；
- 工具调用参数是跨分片拼装的（本地模型常把 `{"path":"a.rs"}` 拆成几段下发），拼完整才执行，工具名会实时提示；
- 思考内容只用于展示：**不会写回历史**，也不会出现在最终回答里；
- 超时按“分片间隔”计算：长回答不会被整体超时误杀，服务端卡住一样会被 `--timeout` 拦下。

```bash
# 默认：思考过程 + 实时回答
cargo run -- -C ./myproj "解释 src/agent.rs 的主循环"

# 只想拿干净的回答（适合管道）
cargo run -- -q -C ./myproj "总结一下这次改动" > summary.md
```

## 工具

| 工具 | 说明 |
| --- | --- |
| `read_file` | 带行号读取，支持 `start_line` / `line_count` 分段；识别二进制文件 |
| `list_directory` | 列目录，`recursive` + `max_depth`；噪声目录（`.git`/`target`/`node_modules`…）不展开 |
| `search_code` | 递归搜索，**默认按普通文本**匹配，`regex: true` 切换正则；`path` 可以是目录也可以是单个文件；输出 `路径:行号: 内容` |
| `write_file` | 覆盖写入，自动创建父目录，可选 `.bak` 备份 |
| `edit_file` | 精确替换；默认要求唯一匹配，否则报错并要求补充上下文 |
| `run_command` | 执行命令并返回退出码 / stdout / stderr；默认白名单 argv 模式，用于 `cargo check`、`cargo test`、`git diff`、`rg` 等验证 |

前五个工具的所有路径都被限制在 `--workspace` 之内，越权访问会被明确拒绝。`run_command` 不受路径沙箱约束，见下一节。

## run_command 的安全模型

**这不是沙箱，请先读完再决定怎么用。**

| 层面 | 做法 |
| --- | --- |
| 默认执行方式 | **白名单 + argv**：只能用预设程序（`cargo`/`rustc`/`git`/`rg`/`ls`/`cat`…），以 `Command::new(prog).args(args)` 直接执行，不经过 shell，所以 `;`、`|`、`&&`、`$( )`、重定向都只是普通参数，无法注入 |
| 放开方式 | `--allow-command <程序>` 追加白名单程序（按 basename 匹配，`./scripts/build.sh` 与 `build.sh` 等价）；`--allow-shell` 才允许 `sh -c` 执行整行命令（支持管道/重定向） |
| 危险操作护栏 | 内置黑名单（`rm -rf /`、`sudo`、`mkfs`、`dd of=/dev/…`、`shutdown`、`curl \| sh`、`git reset --hard`、`git clean -f`、`git push --force`、`git config --global`、全局安装包…）命中即拒绝 |
| cwd | 只能在 `--workspace` 之内（默认工作目录根） |
| 超时 | 默认 60s、上限 600s；超时**杀掉整个进程组**（只杀 shell 的话，`sh -c cargo test` 的子进程会活下来） |
| 输出 | 头部与尾部分别保留，中间省略，单条结果字符数有上限（默认 3 万字符） |
| 卡死防护 | stdin 接 `/dev/null`（等待输入的命令直接拿到 EOF），stdout/stderr 由独立任务持续读取（否则管道写满会死锁），`GIT_TERMINAL_PROMPT=0`/`GIT_PAGER=cat` 避免交互式挂起 |

必须清楚的三件事：

1. **黑名单只是防手滑的护栏，不是安全边界**，绕过方式很多（`git -c alias`、`find -exec`、构建脚本……）。真正的边界只有白名单和操作系统权限。
2. **命令以你自己的用户权限运行，能读写工作目录之外的文件**。文件类工具的沙箱管不到它。
3. **白名单里有构建/包管理工具，等价于允许执行项目里定义的构建脚本与测试代码**（`build.rs`、npm scripts、Makefile）。这是这类工具的固有性质，不是本实现的疏漏。

典型用法：

```bash
# 默认：只能说 cargo check / cargo test 这类 argv 调用
cargo run -- -C ./myproj "修复编译错误，并用 cargo check 验证"

# 需要跑项目自定义脚本
cargo run -- -C ./myproj --allow-command just --allow-command ./scripts/build.sh "..."

# 需要管道/重定向（风险自负）
cargo run -- -C ./myproj --allow-shell "用 cargo test 2>&1 | tail -30 看失败用例"
```

## 测试

```bash
cargo test          # 86 个测试，全部不依赖真实 LLM
```

- `tests/workspace_sandbox.rs`（10）：路径沙箱（`..` 逃逸、绝对路径越权、符号链接穿透、失效链接、新建文件、UTF-8 路径）
- `tests/mcp_tools.rs`（26）：**真的把 `code-tools-server` 拉起来**走 MCP 协议，覆盖分段读取、越界行号、递归列目录、字面量/正则搜索、CRLF 编辑、多处匹配拒绝、越权读写；`run_command` 的 argv 无注入、非 0 退出码算正常信息、白名单拒绝、`--allow-command` 追加、shell 开关、黑名单拦截、cwd 越权拒绝、超时终止、**进程组清理**、超大输出截断保尾部
- `tests/streaming_chat.rs`（6）：**起来一个假 LLM（真 SSE 服务端）** 跑流式全流程，覆盖思考/正文事件顺序、思维链不进正文也不进历史、工具调用跨分片拼装与结果回填、**增量确实是边收边到（不是等服务端收尾）**、HTTP 错误、服务端沉默触发超时
- 单元测试（53）：SSE 解析（分片切断行、`[DONE]`、非法 JSON、按 index 拼工具调用）、流式思维链剥离（逐字符喂也不漏）、思维链剥离、UTF-8 安全截断、MCP→OpenAI schema 转换、工具参数解析、命令策略（白名单/黑名单/shell 开关/参数校验）、历史裁剪不拆散 tool 配对

## 已知限制

- `run_command` 默认不能跑管道/重定向（argv 模式），模型需要把一条命令拆开写；要灵活就开 `--allow-shell`，但那时护栏就只是护栏了。
- 没有文件变更的 diff 预览 / 撤销栈，`edit_file` 的 `backup: true` 是唯一的后悔药。
- 流式输出下正文会边生成边打印：如果中途报错，屏幕上已经出现的文字不会回滚（错误信息会另起一行打到 stderr）。
- 依赖服务端支持 `stream_options.include_usage` 才会回传 token 用量；不支持时只是少一行 debug 日志，不影响回答。
- 单会话；没有持久化对话历史。
- DeepSeek/其他 OpenAI 兼容服务也能用（改 `--base-url`/`--model`/`--api-key`），但工具调用兼容性只在 Ollama + qwen3.8 上实测过。
