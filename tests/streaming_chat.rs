//! 端到端测试：真的拉起 `code-tools-server`，并对着一个本地假 LLM（SSE 流）跑完整流程。
//!
//! 覆盖的重点是流式改造后的行为：
//! - 思考内容（`reasoning` 字段 / 正文里的思维链标记）作为独立事件流出来，不进历史；
//! - 正文增量按到达顺序实时回调，最终答案与增量拼接结果一致；
//! - 工具调用参数被跨分片正确拼接，工具执行结果会回填给下一轮请求。

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use rust_agent::{
    Agent, AgentConfig,
    stream::{ChatEvent, ChatStreamHandler, HandlerRef},
};
use tokio::sync::Mutex;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

// ---------------------------------------------------------------------------
// 假 LLM 服务端
// ---------------------------------------------------------------------------

/// 脚本化的假 LLM：第 N 次请求返回第 N 段脚本。
#[derive(Clone)]
struct FakeLlm {
    base_url: String,
    /// 收到的请求体（按顺序），用来断言“工具结果有没有回填给模型”
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
}

impl FakeLlm {
    async fn start(scripts: Vec<Vec<String>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let served = Arc::new(AtomicUsize::new(0));
        let scripts = Arc::new(scripts);
        let captured = Arc::clone(&requests);

        tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    break;
                };
                let index = served.fetch_add(1, Ordering::SeqCst);
                let script = scripts.get(index).cloned().unwrap_or_default();
                let captured = Arc::clone(&captured);
                tokio::spawn(async move {
                    serve(socket, script, captured).await;
                });
            }
        });

        Self {
            base_url: format!("http://{addr}/v1"),
            requests,
        }
    }

    async fn request_bodies(&self) -> Vec<serde_json::Value> {
        self.requests.lock().await.clone()
    }
}

async fn serve(
    mut socket: TcpStream,
    script: Vec<String>,
    captured: Arc<Mutex<Vec<serde_json::Value>>>,
) {
    let Some(body) = read_request_body(&mut socket).await else {
        return;
    };
    if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&body) {
        captured.lock().await.push(json);
    }

    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
    if socket.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    for piece in &script {
        // 约定：以 `@pause:<毫秒>` 开头的“块”表示发送前先停一会儿，
        // 用来验证客户端确实是边收边显示，而不是等全部内容到齐再一次性吐出
        if let Some(ms) = piece.strip_prefix("@pause:") {
            tokio::time::sleep(Duration::from_millis(ms.parse().unwrap_or(0))).await;
            continue;
        }
        if socket.write_all(piece.as_bytes()).await.is_err() {
            return;
        }
        // 真流式：给客户端机会逐块收到，而不是一次读全
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let _ = socket.shutdown().await;
}

/// 读出一个完整请求的 body。
///
/// reqwest 发 JSON 请求体时用的是 `Transfer-Encoding: chunked`，所以只看
/// `Content-Length` 会拿到被截断的 body；两种编码都得支持。
async fn read_request_body(socket: &mut TcpStream) -> Option<Vec<u8>> {
    const SEP: &[u8] = b"\r\n\r\n";
    let mut raw: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut header_end = None;

    loop {
        if header_end.is_none() {
            header_end = raw
                .windows(SEP.len())
                .position(|w| w == SEP)
                .map(|i| i + SEP.len());
        }
        if let Some(end) = header_end {
            let head = String::from_utf8_lossy(&raw[..end]).to_ascii_lowercase();
            if let Some(len) = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
            {
                if raw.len() >= end + len {
                    return Some(raw[end..end + len].to_vec());
                }
            } else if head.contains("transfer-encoding: chunked") {
                if let Some(body) = decode_chunked(&raw[end..]) {
                    return Some(body);
                }
            } else {
                return Some(raw[end..].to_vec());
            }
        }
        match socket.read(&mut tmp).await {
            Ok(0) | Err(_) => return None,
            Ok(n) => raw.extend_from_slice(&tmp[..n]),
        }
    }
}

/// 解 chunked 编码；还没收齐时返回 `None`。
fn decode_chunked(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        let line_end = data[pos..].windows(2).position(|w| w == b"\r\n")? + pos;
        let size = usize::from_str_radix(
            std::str::from_utf8(&data[pos..line_end])
                .ok()?
                .split(';')
                .next()?
                .trim(),
            16,
        )
        .ok()?;
        let body_start = line_end + 2;
        if size == 0 {
            return Some(out);
        }
        if data.len() < body_start + size + 2 {
            return None;
        }
        out.extend_from_slice(&data[body_start..body_start + size]);
        pos = body_start + size + 2;
    }
}

/// 造一个 SSE data 行。
fn sse(delta: serde_json::Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({"id":"chatcmpl-1","object":"chat.completion.chunk","model":"fake",
            "choices":[{"index":0,"delta":delta}]})
    )
}

/// 结束块（带 finish_reason）+ usage 块 + `[DONE]`。
fn finish(reason: &str) -> Vec<String> {
    vec![
        format!(
            "data: {}\n\n",
            serde_json::json!({"id":"chatcmpl-1","object":"chat.completion.chunk","model":"fake",
                "choices":[{"index":0,"delta":{},"finish_reason":reason}]})
        ),
        format!(
            "data: {}\n\n",
            serde_json::json!({"id":"chatcmpl-1","object":"chat.completion.chunk","model":"fake",
                "choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}})
        ),
        "data: [DONE]\n\n".to_string(),
    ]
}

/// 把正文按字符切成多块，模拟逐 token 下发。
fn content_chunks(text: &str) -> Vec<String> {
    text.chars()
        .map(|c| sse(serde_json::json!({"content": c.to_string()})))
        .collect()
}

// ---------------------------------------------------------------------------
// 测试脚手架
// ---------------------------------------------------------------------------

fn fresh_workspace(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("streaming")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn test_config(base_url: &str, workspace: PathBuf) -> AgentConfig {
    AgentConfig {
        base_url: base_url.to_string(),
        workspace,
        server_bin: Some(PathBuf::from(env!("CARGO_BIN_EXE_code-tools-server"))),
        max_steps: 6,
        request_timeout: Duration::from_secs(10),
        ..Default::default()
    }
}

/// 记录事件顺序的 handler。
#[derive(Default)]
struct Recorder {
    events: Vec<(String, String)>,
}

impl ChatStreamHandler for Recorder {
    fn on_event(&mut self, event: &ChatEvent) -> rust_agent::Result<()> {
        let entry = match event {
            ChatEvent::Reasoning(d) => Some(("reasoning", d.clone())),
            ChatEvent::Text(d) => Some(("text", d.clone())),
            ChatEvent::ToolCall { name, .. } => Some(("tool_call", name.clone())),
            ChatEvent::Finished { finish_reason, .. } => {
                Some(("finished", finish_reason.clone().unwrap_or_default()))
            }
            ChatEvent::ToolArguments { .. } => None,
        };
        if let Some(entry) = entry {
            self.events.push((entry.0.to_string(), entry.1));
        }
        Ok(())
    }
}

impl Recorder {
    fn texts(&self) -> String {
        self.events
            .iter()
            .filter(|(kind, _)| kind == "text")
            .map(|(_, d)| d.as_str())
            .collect()
    }

    fn reasoning(&self) -> String {
        self.events
            .iter()
            .filter(|(kind, _)| kind == "reasoning")
            .map(|(_, d)| d.as_str())
            .collect()
    }

    /// 事件顺序（去掉具体内容）。
    fn order(&self) -> Vec<String> {
        self.events.iter().map(|(kind, _)| kind.clone()).collect()
    }
}

// ---------------------------------------------------------------------------
// 用例
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streams_reasoning_and_answer_in_arrival_order() {
    let workspace = fresh_workspace("reasoning");
    let mut script = vec![
        sse(serde_json::json!({"reasoning": "先看看"})),
        sse(serde_json::json!({"reasoning_content": "文件在哪"})),
    ];
    script.extend(content_chunks("答案是 42"));
    script.extend(finish("stop"));
    // 思路：第三块不是正文而是“思考”字段，正文只有一段
    let llm = FakeLlm::start(vec![script]).await;

    let mut agent = Agent::connect(test_config(&llm.base_url, workspace))
        .await
        .expect("无法连接 Agent");
    let mut recorder = Recorder::default();
    let answer = agent
        .ask_stream_with("随便问一句", HandlerRef(&mut recorder))
        .await
        .expect("流式对话失败");
    agent.shutdown().await;

    assert_eq!(answer, "答案是 42");
    assert_eq!(recorder.texts(), "答案是 42");
    assert_eq!(recorder.reasoning(), "先看看文件在哪");
    // 事件顺序：两段思考 → 六个正文分片 → 结束（正文是逐字符下发的）
    let mut expected = vec!["reasoning", "reasoning"];
    expected.extend(std::iter::repeat_n("text", "答案是 42".chars().count()));
    expected.push("finished");
    assert_eq!(recorder.order(), expected);
    assert_eq!(recorder.events.last().unwrap().1, "stop");
}

#[tokio::test]
async fn strips_inline_thinking_from_streamed_text() {
    let workspace = fresh_workspace("inline-thinking");
    let mut script = content_chunks("<think>我要先想一下</think>答案是 42");
    script.extend(finish("stop"));
    let llm = FakeLlm::start(vec![script]).await;

    let mut agent = Agent::connect(test_config(&llm.base_url, workspace))
        .await
        .unwrap();
    let mut recorder = Recorder::default();
    let answer = agent
        .ask_stream_with("问", HandlerRef(&mut recorder))
        .await
        .unwrap();
    agent.shutdown().await;

    assert_eq!(answer, "答案是 42", "思维链必须从最终回答里剥离");
    assert_eq!(recorder.texts(), "答案是 42");
    assert!(
        !recorder.texts().contains("我要先想一下"),
        "流式正文里不能出现思考内容：{:?}",
        recorder.events
    );
}

#[tokio::test]
async fn streams_tool_call_then_final_answer_and_feeds_result_back() {
    let workspace = fresh_workspace("tool-call");
    std::fs::write(workspace.join("hello.txt"), "你好，世界\n").unwrap();

    // 第一轮：思考 + 分片下发的工具调用；第二轮：读到文件内容后的最终回答
    let mut first = vec![sse(serde_json::json!({"reasoning": "得先读文件"}))];
    first.push(sse(serde_json::json!({
        "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": "{\"pa"}}]
    })));
    first.push(sse(serde_json::json!({
        "tool_calls": [{"index": 0, "function": {"arguments": "th\":\"hello.txt\"}"}}]
    })));
    first.extend(finish("tool_calls"));

    let mut second = content_chunks("文件里写的是“你好，世界”。");
    second.extend(finish("stop"));

    let llm = FakeLlm::start(vec![first, second]).await;
    let mut agent = Agent::connect(test_config(&llm.base_url, workspace))
        .await
        .unwrap();
    let mut recorder = Recorder::default();
    let answer = agent
        .ask_stream_with("hello.txt 里写了什么？", HandlerRef(&mut recorder))
        .await
        .unwrap();
    agent.shutdown().await;

    assert_eq!(answer, "文件里写的是“你好，世界”。");
    assert!(
        recorder.order().contains(&"tool_call".to_string()),
        "必须上报工具调用事件：{:?}",
        recorder.events
    );
    assert_eq!(
        recorder
            .events
            .iter()
            .find(|(kind, _)| kind == "tool_call")
            .unwrap()
            .1,
        "read_file"
    );

    // 第二轮请求里必须带上工具结果，而且不能带思考内容
    let bodies = llm.request_bodies().await;
    assert_eq!(bodies.len(), 2, "应该恰好两轮请求");
    let second_body = serde_json::to_string(&bodies[1]).unwrap();
    assert!(
        second_body.contains("你好，世界"),
        "工具结果没有回填给模型：{second_body}"
    );
    assert!(
        second_body.contains("\"role\":\"tool\""),
        "缺少 tool 角色消息：{second_body}"
    );
    assert!(
        !second_body.contains("得先读文件"),
        "思考内容不该写回历史：{second_body}"
    );
    assert!(
        bodies[1]["stream"] == serde_json::Value::Bool(true),
        "必须请求流式：{second_body}"
    );
}

#[tokio::test]
async fn reports_http_error_from_server() {
    let workspace = fresh_workspace("http-error");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut tmp = [0u8; 2048];
            let _ = socket.read(&mut tmp).await;
            let body = r#"{"error":"model not found"}"#;
            let response = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    });

    let mut agent = Agent::connect(test_config(&format!("http://{addr}/v1"), workspace))
        .await
        .unwrap();
    let err = agent.ask("你好").await.unwrap_err();
    agent.shutdown().await;
    let text = err.to_string();
    assert!(text.contains("404"), "{text}");
    assert!(text.contains("model not found"), "{text}");
}

#[tokio::test]
async fn times_out_when_server_goes_silent() {
    let workspace = fresh_workspace("idle-timeout");
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut tmp = [0u8; 2048];
            let _ = socket.read(&mut tmp).await;
            // 声明成流式，然后一个字节都不发
            let head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
            let _ = socket.write_all(head.as_bytes()).await;
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });

    let mut config = test_config(&format!("http://{addr}/v1"), workspace);
    config.request_timeout = Duration::from_secs(1);
    let mut agent = Agent::connect(config).await.unwrap();
    let err = agent.ask("你好").await.unwrap_err();
    agent.shutdown().await;
    assert!(err.to_string().contains("没有新数据"), "{err}");
}

/// 关键行为：内容必须“边生成边到达”，而不是等服务端收尾后一次性吐出。
#[tokio::test]
async fn deltas_arrive_before_the_stream_ends() {
    let workspace = fresh_workspace("incremental");
    let script = vec![
        sse(serde_json::json!({"reasoning": "先想"})),
        sse(serde_json::json!({"content": "第一段"})),
        "@pause:700".to_string(),
        sse(serde_json::json!({"content": "第二段"})),
        "@pause:200".to_string(),
        sse(serde_json::json!({"content": "第三段"})),
        "@pause:300".to_string(),
        "@pause:300".to_string(),
        sse(serde_json::json!({"content": "尾巴"})),
        "@pause:1200".to_string(),
    ]
    .into_iter()
    .chain(finish("stop"))
    .collect();
    let llm = FakeLlm::start(vec![script]).await;

    let mut agent = Agent::connect(test_config(&llm.base_url, workspace))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::oneshot::channel();
    let (seen_tx, mut seen_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut handler = MoveHandler {
        tx: Some(tx),
        seen: seen_tx,
    };
    let task = tokio::spawn(async move {
        let answer = agent
            .ask_stream_with("讲三段话", HandlerRef(&mut handler))
            .await;
        (agent, answer)
    });

    // 第二批内容到达前（约 700ms 的暂停期内）就应该已经看到第一段
    let first = tokio::time::timeout(Duration::from_millis(500), rx)
        .await
        .expect("等了 500ms 还没收到任何正文增量，说明输出被缓冲了")
        .expect("回调通道被关闭");
    assert!(first.contains("第一段"), "{first:?}");

    let (agent, answer) = task.await.unwrap();
    let answer = answer.unwrap();
    agent.shutdown().await;

    assert_eq!(answer, "第一段第二段第三段尾巴");
    let mut seen = Vec::new();
    while let Ok(item) = seen_rx.try_recv() {
        seen.push(item);
    }
    // 正文被拆成多段收到（不是一次性的一大块）
    assert!(seen.len() >= 3, "正文应该分段到达：{seen:?}");
    assert_eq!(seen.concat(), "第一段第二段第三段尾巴");
}

/// 收到第一段正文后立刻通知测试的 handler。
struct MoveHandler {
    tx: Option<tokio::sync::oneshot::Sender<String>>,
    seen: tokio::sync::mpsc::UnboundedSender<String>,
}

impl ChatStreamHandler for MoveHandler {
    fn on_event(&mut self, event: &ChatEvent) -> rust_agent::Result<()> {
        if let ChatEvent::Text(delta) = event {
            let _ = self.seen.send(delta.clone());
            if let Some(tx) = self.tx.take() {
                let _ = tx.send(delta.clone());
            }
        }
        Ok(())
    }
}
