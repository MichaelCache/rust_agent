//! 端到端测试：真的把 `code-tools-server` 作为子进程拉起来，通过 MCP 协议调用工具。
//!
//! 这一层不依赖 LLM，覆盖的是“工具本体 + 沙箱 + 协议转换”，
//! 也就是原实现里 panic / 报错最集中的部分。

use std::fs;
use std::path::{Path, PathBuf};

use rmcp::model::ContentBlock;
use rust_agent::mcp::McpClient;

fn fresh_workspace(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("mcp")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

async fn setup(name: &str) -> (McpClient, PathBuf) {
    setup_with_args(name, &[]).await
}

async fn setup_with_args(name: &str, extra: &[String]) -> (McpClient, PathBuf) {
    let ws = fresh_workspace(name);
    let bin = Path::new(env!("CARGO_BIN_EXE_code-tools-server"));
    let client = McpClient::spawn_with_args(bin, &ws, extra)
        .await
        .expect("无法启动 code-tools-server");
    (client, ws)
}

fn arg_list(items: &[&str]) -> serde_json::Value {
    serde_json::Value::Array(
        items
            .iter()
            .map(|s| serde_json::Value::String((*s).to_string()))
            .collect(),
    )
}

/// 调用工具，返回 (文本结果, 是否 isError)。
async fn call(client: &McpClient, name: &str, args: serde_json::Value) -> (String, bool) {
    assert!(args.is_object(), "工具参数必须是对象");
    let result = client
        .call_tool(name, Some(args.as_object().unwrap().clone()))
        .await
        .unwrap_or_else(|e| panic!("调用 {name} 失败: {e}"));
    let text = result
        .content
        .iter()
        .filter_map(|c| match c {
            ContentBlock::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    (text, result.is_error == Some(true))
}

#[tokio::test]
async fn exposes_exactly_the_expected_code_tools() {
    let (client, _ws) = setup("list-tools").await;
    let tools = client.list_tools().await.unwrap();
    let mut names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "edit_file",
            "list_directory",
            "read_file",
            "run_command",
            "search_code",
            "write_file"
        ]
    );
    // 每个工具都必须带可用的 JSON Schema，否则模型没法正确调用
    for t in &tools {
        let schema = serde_json::Value::Object((*t.input_schema).clone());
        assert_eq!(schema["type"], "object", "{} 的 schema 不是 object", t.name);
        assert!(
            schema["properties"].is_object(),
            "{} 没有 properties",
            t.name
        );
        assert!(
            t.description.as_ref().is_some_and(|d| !d.is_empty()),
            "{} 缺少描述",
            t.name
        );
    }
    client.shutdown().await;
}

#[tokio::test]
async fn read_file_numbers_lines_and_reports_total() {
    let (client, ws) = setup("read").await;
    fs::write(ws.join("sample.rs"), "fn a() {}\nfn b() {}\n// TODO 整理\n").unwrap();

    let (text, is_err) = call(
        &client,
        "read_file",
        serde_json::json!({"path": "sample.rs"}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("共 3 行"), "{text}");
    assert!(text.contains("1 | fn a() {}"), "{text}");
    assert!(text.contains("3 | // TODO 整理"), "{text}");

    // 分段读取
    let (text, is_err) = call(
        &client,
        "read_file",
        serde_json::json!({"path": "sample.rs", "start_line": 2, "line_count": 1}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("2 | fn b() {}"), "{text}");
    assert!(!text.contains("fn a()"), "{text}");

    client.shutdown().await;
}

#[tokio::test]
async fn read_file_beyond_eof_does_not_panic() {
    // 原实现 lines[start..end] 在 start > len 时直接 panic
    let (client, ws) = setup("read-eof").await;
    fs::write(ws.join("small.txt"), "只有一行\n").unwrap();

    let (text, is_err) = call(
        &client,
        "read_file",
        serde_json::json!({"path": "small.txt", "start_line": 999}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("超出文件末尾"), "{text}");

    let (text, is_err) = call(
        &client,
        "read_file",
        serde_json::json!({"path": "empty.txt"}),
    )
    .await;
    assert!(is_err, "读不存在的文件应该是错误");
    assert!(text.contains("无法访问"), "{text}");

    client.shutdown().await;
}

#[tokio::test]
async fn read_file_rejects_directory() {
    let (client, ws) = setup("read-dir").await;
    fs::create_dir_all(ws.join("sub")).unwrap();
    let (text, is_err) = call(&client, "read_file", serde_json::json!({"path": "sub"})).await;
    assert!(is_err);
    assert!(text.contains("list_directory"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn search_code_treats_pattern_as_literal_by_default() {
    // 原实现强制按正则编译，"fn main() {" 这种含元字符的查询会直接报错
    let (client, ws) = setup("search").await;
    fs::write(
        ws.join("a.rs"),
        "fn main() {\n    let x = 1; // TODO: 改这里\n}\n",
    )
    .unwrap();
    fs::create_dir_all(ws.join("target/debug")).unwrap();
    fs::write(ws.join("target/debug/junk.rs"), "TODO 不该被搜到\n").unwrap();

    let (text, is_err) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "fn main() {"}),
    )
    .await;
    assert!(!is_err, "字面量搜索不该报错: {text}");
    assert!(text.contains("a.rs:1"), "{text}");

    // 默认忽略大小写
    let (text, _) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "todo: 改这里"}),
    )
    .await;
    assert!(text.contains("a.rs:2"), "{text}");

    // target/ 属于噪声目录，应被跳过
    assert!(!text.contains("target/debug/junk.rs"), "{text}");

    // 正则模式
    let (text, is_err) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "let\\s+x\\s*=\\s*\\d+", "regex": true}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("a.rs:2"), "{text}");

    // 非法正则要给出可读错误，而不是 panic
    let (text, is_err) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "([", "regex": true}),
    )
    .await;
    assert!(is_err);
    assert!(text.contains("正则表达式无效"), "{text}");

    client.shutdown().await;
}

#[tokio::test]
async fn list_directory_is_recursive_and_skips_noise() {
    let (client, ws) = setup("list").await;
    fs::create_dir_all(ws.join("src/inner")).unwrap();
    fs::create_dir_all(ws.join("target")).unwrap();
    fs::write(ws.join("src/lib.rs"), "pub fn x() {}").unwrap();
    fs::write(ws.join("src/inner/deep.rs"), "// deep").unwrap();
    fs::write(ws.join("target/ignored.rs"), "// ignored").unwrap();

    let (flat, is_err) = call(&client, "list_directory", serde_json::json!({})).await;
    assert!(!is_err, "{flat}");
    assert!(flat.contains("[DIR]  src/"), "{flat}");
    // 噪声目录也列出来，让用户知道它存在
    assert!(flat.contains("[DIR]  target/"), "{flat}");
    assert!(!flat.contains("lib.rs"), "非递归时不该展开子目录: {flat}");

    // 指定子目录时能看到里面的文件
    let (one, _) = call(
        &client,
        "list_directory",
        serde_json::json!({"path": "src"}),
    )
    .await;
    assert!(one.contains("[FILE] src/lib.rs"), "{one}");

    let (deep, _) = call(
        &client,
        "list_directory",
        serde_json::json!({"recursive": true, "max_depth": 5}),
    )
    .await;
    assert!(deep.contains("src/inner/deep.rs"), "{deep}");
    assert!(deep.contains("未展开的噪声目录"), "{deep}");
    assert!(!deep.contains("ignored.rs"), "{deep}");

    client.shutdown().await;
}

#[tokio::test]
async fn write_file_creates_nested_directories_and_backup() {
    let (client, ws) = setup("write").await;

    let (text, is_err) = call(
        &client,
        "write_file",
        serde_json::json!({"path": "src/deep/new/file.rs", "content": "// 新建\nfn a() {}\n"}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("已创建"), "{text}");
    assert_eq!(
        fs::read_to_string(ws.join("src/deep/new/file.rs")).unwrap(),
        "// 新建\nfn a() {}\n"
    );

    // 覆盖 + 备份
    let (text, is_err) = call(
        &client,
        "write_file",
        serde_json::json!({"path": "src/deep/new/file.rs", "content": "// 改过\n", "backup": true}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("已覆盖"), "{text}");
    assert_eq!(
        fs::read_to_string(ws.join("src/deep/new/file.rs.bak")).unwrap(),
        "// 新建\nfn a() {}\n"
    );

    client.shutdown().await;
}

#[tokio::test]
async fn edit_file_replaces_unique_match_and_reports_lines() {
    let (client, ws) = setup("edit").await;
    fs::write(
        ws.join("code.rs"),
        "fn a() {\n    let v = 1;\n}\n\nfn b() {\n    let v = 2;\n}\n",
    )
    .unwrap();

    let (text, is_err) = call(
        &client,
        "edit_file",
        serde_json::json!({
            "path": "code.rs",
            "old_text": "fn a() {\n    let v = 1;\n}",
            "new_text": "fn a() {\n    let v = 100;\n}"
        }),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("第 1 行"), "{text}");

    let updated = fs::read_to_string(ws.join("code.rs")).unwrap();
    assert!(updated.contains("let v = 100;"), "{updated}");
    // 另一处没被误改
    assert!(updated.contains("let v = 2;"), "{updated}");

    client.shutdown().await;
}

#[tokio::test]
async fn edit_file_refuses_ambiguous_match() {
    let (client, ws) = setup("edit-ambiguous").await;
    fs::write(ws.join("dup.rs"), "let x = 1;\nlet y = 2;\nlet x = 1;\n").unwrap();

    let (text, is_err) = call(
        &client,
        "edit_file",
        serde_json::json!({"path": "dup.rs", "old_text": "let x = 1;", "new_text": "let x = 9;"}),
    )
    .await;
    assert!(is_err, "多处匹配应当拒绝: {text}");
    assert!(text.contains("匹配到 2 处"), "{text}");
    // 文件必须保持原样
    assert_eq!(
        fs::read_to_string(ws.join("dup.rs")).unwrap(),
        "let x = 1;\nlet y = 2;\nlet x = 1;\n"
    );

    // 显式 replace_all 才允许全改
    let (text, is_err) = call(
        &client,
        "edit_file",
        serde_json::json!({
            "path": "dup.rs", "old_text": "let x = 1;",
            "new_text": "let x = 9;", "replace_all": true
        }),
    )
    .await;
    assert!(!is_err, "{text}");
    assert_eq!(
        fs::read_to_string(ws.join("dup.rs")).unwrap(),
        "let x = 9;\nlet y = 2;\nlet x = 9;\n"
    );

    client.shutdown().await;
}

#[tokio::test]
async fn edit_file_handles_crlf_and_missing_text() {
    let (client, ws) = setup("edit-crlf").await;
    fs::write(ws.join("win.txt"), "line1\r\nline2\r\nline3\r\n").unwrap();

    // 模型按 \n 给出 old_text，文件其实是 CRLF
    let (text, is_err) = call(
        &client,
        "edit_file",
        serde_json::json!({
            "path": "win.txt", "old_text": "line2\nline3", "new_text": "line2\nLINE3"
        }),
    )
    .await;
    assert!(!is_err, "{text}");
    assert_eq!(
        fs::read_to_string(ws.join("win.txt")).unwrap(),
        "line1\r\nline2\r\nLINE3\r\n"
    );

    let (text, is_err) = call(
        &client,
        "edit_file",
        serde_json::json!({"path": "win.txt", "old_text": "根本不存在的文本", "new_text": "x"}),
    )
    .await;
    assert!(is_err);
    assert!(text.contains("未在文件中找到"), "{text}");

    client.shutdown().await;
}

#[tokio::test]
async fn sandbox_blocks_escapes() {
    let (client, ws) = setup("escape").await;
    fs::write(ws.join("ok.txt"), "in workspace").unwrap();

    // 1) 相对路径向上穿越
    let (text, is_err) = call(
        &client,
        "read_file",
        serde_json::json!({"path": "../../etc/passwd"}),
    )
    .await;
    assert!(is_err, "必须拒绝越权读取: {text}");
    assert!(text.contains("拒绝访问"), "{text}");

    // 2) 绝对路径越权
    let (text, is_err) = call(
        &client,
        "read_file",
        serde_json::json!({"path": "/etc/passwd"}),
    )
    .await;
    assert!(is_err, "{text}");
    assert!(text.contains("拒绝访问"), "{text}");

    // 3) 越权写入
    let (text, is_err) = call(
        &client,
        "write_file",
        serde_json::json!({"path": "/tmp/evil_should_not_exist.txt", "content": "x"}),
    )
    .await;
    assert!(is_err, "{text}");
    assert!(text.contains("拒绝访问"), "{text}");
    assert!(!Path::new("/tmp/evil_should_not_exist.txt").exists());

    // 4) 工作目录内的正常读写仍然可用
    let (text, is_err) = call(&client, "read_file", serde_json::json!({"path": "ok.txt"})).await;
    assert!(!is_err, "{text}");
    assert!(text.contains("in workspace"), "{text}");

    client.shutdown().await;
}

#[tokio::test]
async fn search_code_accepts_a_file_path() {
    // 模型经常把文件名当 path 传进来；原实现会对文件调用 read_dir 然后静默返回“未找到”
    let (client, ws) = setup("search-file").await;
    fs::write(ws.join("one.rs"), "fn target_fn() {}\n").unwrap();
    fs::write(ws.join("two.rs"), "fn other_fn() {}\n").unwrap();

    let (text, is_err) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "target_fn", "path": "one.rs"}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("one.rs:1"), "应该搜到单文件内容: {text}");
    assert!(!text.contains("other_fn"), "不该搜到别的文件: {text}");

    let (text, is_err) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "根本不存在的符号", "path": "one.rs"}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(
        text.contains("文件 one.rs"),
        "未命中时要说清搜索范围: {text}"
    );

    client.shutdown().await;
}

#[tokio::test]
async fn search_code_reports_unreadable_root() {
    let (client, _ws) = setup("search-bad-root").await;
    let (text, is_err) = call(
        &client,
        "search_code",
        serde_json::json!({"pattern": "x", "path": "nope-dir"}),
    )
    .await;
    assert!(is_err, "不存在的搜索根应该报错而不是静默返回空结果: {text}");
    assert!(text.contains("无法访问"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn unknown_tool_name_returns_error_not_panic() {
    let (client, _ws) = setup("unknown").await;
    let result = client.call_tool("no_such_tool", None).await;
    assert!(result.is_err(), "未知工具应当是协议错误");
    client.shutdown().await;
}

// ---------------------------------------------------------------------------
// run_command：策略与执行行为
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_command_executes_allowlisted_program_as_argv() {
    let (client, _ws) = setup("cmd-basic").await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "echo", "args": arg_list(&["hello", "world"])}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("hello world"), "{text}");
    assert!(text.contains("退出码 0"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_does_not_interpret_shell_syntax_in_argv_mode() {
    // argv 模式下 `;` `|` `&&` 都只是普通参数，不会被 shell 解释
    let (client, ws) = setup("cmd-no-injection").await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({
            "command": "echo",
            "args": arg_list(&["safe; touch pwned", "| cat", "&& echo x"])
        }),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("safe; touch pwned"), "{text}");
    assert!(!ws.join("pwned").exists(), "argv 模式不允许发生命令注入");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_reports_nonzero_exit_as_information() {
    // grep 没匹配到会返回 1：这是信息，不是工具错误
    let (client, ws) = setup("cmd-exit-code").await;
    fs::write(ws.join("a.txt"), "hello\n").unwrap();
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({
            "command": "grep", "args": arg_list(&["-c", "找不到的内容", "a.txt"])
        }),
    )
    .await;
    assert!(!is_err, "非 0 退出码不该标记为工具错误: {text}");
    assert!(text.contains("退出码 1"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_rejects_program_outside_allowlist() {
    let (client, ws) = setup("cmd-not-allowed").await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "bash", "args": arg_list(&["-c", "echo hi"])}),
    )
    .await;
    assert!(is_err, "{text}");
    assert!(text.contains("不在允许列表内"), "{text}");
    assert!(
        text.contains("--allow-command"),
        "应告诉用户怎么放开: {text}"
    );
    let _ = ws;
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_accepts_extra_allowlisted_program() {
    // 通过 --allow-command 追加（这里用 env 程序的替代者：true 之外选一个稳定的）
    let (client, _ws) = setup_with_args(
        "cmd-extra",
        &["--allow-command".to_string(), "echo".to_string()],
    )
    .await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "echo", "args": arg_list(&["extra-ok"])}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("extra-ok"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_shell_line_requires_flag() {
    let (client, _ws) = setup("cmd-shell-off").await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "echo a && echo b"}),
    )
    .await;
    assert!(is_err, "{text}");
    assert!(text.contains("未开启 shell 模式"), "{text}");
    assert!(text.contains("argv 形式"), "应给出可操作建议: {text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_shell_mode_runs_pipelines_when_enabled() {
    let (client, _ws) = setup_with_args("cmd-shell-on", &["--allow-shell".to_string()]).await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "echo first && echo second"}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("first"), "{text}");
    assert!(text.contains("second"), "{text}");

    // 管道可用
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "printf 'b\\na\\n' | sort"}),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("a\nb") || text.contains("a\r\nb"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_denylist_blocks_destructive_git() {
    let (client, _ws) = setup("cmd-denylist").await;
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "git", "args": arg_list(&["reset", "--hard"])}),
    )
    .await;
    assert!(is_err, "危险命令必须被拦下: {text}");
    assert!(text.contains("拒绝执行"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_cwd_is_confined_to_workspace() {
    let (client, ws) = setup("cmd-cwd").await;
    fs::create_dir_all(ws.join("sub")).unwrap();

    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({
            "command": "pwd", "args": arg_list(&[]), "cwd": "sub"
        }),
    )
    .await;
    assert!(!is_err, "{text}");
    assert!(text.contains("sub"), "{text}");

    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({
            "command": "pwd", "args": arg_list(&[]), "cwd": "../../.."
        }),
    )
    .await;
    assert!(is_err, "cwd 逃出工作目录必须被拒绝: {text}");
    assert!(text.contains("拒绝访问"), "{text}");
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_timeout_kills_process() {
    let (client, _ws) = setup("cmd-timeout").await;
    let started = std::time::Instant::now();
    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({
            "command": "sleep", "args": arg_list(&["30"]), "timeout_secs": 1
        }),
    )
    .await;
    let elapsed = started.elapsed();
    assert!(is_err, "超时应当是工具错误: {text}");
    assert!(text.contains("超时"), "{text}");
    assert!(
        elapsed < std::time::Duration::from_secs(15),
        "超时后应尽快返回，实际用了 {elapsed:?}"
    );
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_truncates_huge_output_keeping_tail() {
    let (client, ws) = setup("cmd-huge-output").await;
    let mut content = String::from("HEAD-MARKER\n");
    for i in 0..40_000 {
        content.push_str(&format!("padding line {i}\n"));
    }
    content.push_str("TAIL-MARKER\n");
    fs::write(ws.join("big.txt"), content).unwrap();

    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({"command": "cat", "args": arg_list(&["big.txt"])}),
    )
    .await;
    assert!(!is_err, "{}", &text[..text.len().min(200)]);
    assert!(text.contains("HEAD-MARKER"), "头部要保留");
    assert!(text.contains("TAIL-MARKER"), "尾部要保留");
    assert!(text.contains("已省略中间部分"), "要说明发生过截断");
    assert!(
        text.chars().count() < 40_000,
        "输出必须被限制，实际 {} 字符",
        text.chars().count()
    );
    client.shutdown().await;
}

#[tokio::test]
async fn run_command_kills_whole_process_group_on_timeout() {
    // shell 模式下的“孙进程”也要被清掉：只杀 sh 的话，
    // `sleep 2 && touch marker` 这样的后台子任务会活下来继续执行。
    // 这里用「文件是否被创建」来判定，而不是 pgrep（pgrep -f 会误匹配到
    // 外层父 shell 的命令行文本）。
    let (client, ws) = setup_with_args("cmd-killpg", &["--allow-shell".to_string()]).await;
    let marker = ws.join("grandchild-survived.marker");

    let (text, is_err) = call(
        &client,
        "run_command",
        serde_json::json!({
            "command": format!("sleep 2 && touch {} & echo started; wait", marker.display()),
            "timeout_secs": 1
        }),
    )
    .await;
    assert!(is_err, "{text}");
    assert!(text.contains("超时"), "{text}");

    // 后台子任务本该在 t≈2s 触发；若进程组没被杀干净，此刻文件就存在了
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    assert!(
        !marker.exists(),
        "超时后进程组里的子进程仍在运行（标记文件被创建）"
    );
    client.shutdown().await;
}
