//! LLM 客户端构造 + MCP 工具 → OpenAI function-calling schema 的转换。

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObject},
};
use rmcp::model::Tool as McpTool;
use tracing::warn;

/// 构造指向 Ollama 的 OpenAI 兼容客户端。
///
/// 注意 base_url 必须带 `/v1` 后缀，Ollama 的兼容层挂在 `http://localhost:11434/v1`。
/// 对话本身走 [`crate::stream`] 的直连 SSE（async-openai 的流式类型会丢掉
/// `reasoning` 字段），这个客户端负责非流式接口并持有同一份配置。
pub fn build_client(base_url: &str, api_key: &str) -> Client<OpenAIConfig> {
    client_with(base_url, api_key, reqwest::Client::new())
}

/// 用外部传入的 HTTP 客户端构造 LLM 客户端（便于统一超时/连接池配置）。
pub fn client_with(base_url: &str, api_key: &str, http: reqwest::Client) -> Client<OpenAIConfig> {
    Client::build(http, client_config(base_url, api_key))
}

/// async-openai 的配置规则：`{base_url}/chat/completions`。
///
/// 流式直连那条路要用 [`chat_completions_url`] 拼出同样的地址。
pub fn client_config(base_url: &str, api_key: &str) -> OpenAIConfig {
    OpenAIConfig::new()
        .with_api_base(base_url)
        // Ollama 不校验 key，但字段不能为空
        .with_api_key(api_key)
}

/// 拼接 chat completions 的完整地址。
///
/// 必须复刻 async-openai `get_api_url` 那一步：用户给的 base_url 带不带结尾斜杠
/// 都得拼对，否则 `.../v1` + `/chat/completions` 会被服务端 307 到带斜杠的地址
/// （有些服务端还只把裸路径塞进 Location）。
pub fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim().trim_end_matches('/'))
}

/// 把 MCP 的 `Tool` 列表转换成 OpenAI 的 `tools` 参数。
///
/// MCP 的 `input_schema` 本身就是 JSON Schema，可以直接透传。
pub fn mcp_tools_to_openai(tools: &[McpTool]) -> Vec<ChatCompletionTools> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        if !is_valid_function_name(&t.name) {
            warn!(name = %t.name, "跳过名称不符合 OpenAI 规范的 MCP 工具");
            continue;
        }
        out.push(ChatCompletionTools::Function(ChatCompletionTool {
            function: FunctionObject {
                name: t.name.to_string(),
                description: t.description.as_ref().map(|d| d.to_string()),
                parameters: Some(serde_json::Value::Object((*t.input_schema).clone())),
                strict: None,
            },
        }));
    }
    out
}

/// 取出转换后的工具名，用于校验模型是否在瞎编工具名。
pub fn tool_names(tools: &[ChatCompletionTools]) -> Vec<String> {
    tools
        .iter()
        .map(|t| match t {
            ChatCompletionTools::Function(f) => f.function.name.clone(),
            ChatCompletionTools::Custom(c) => c.custom.name.clone(),
        })
        .collect()
}

/// OpenAI 要求函数名匹配 `^[a-zA-Z0-9_-]{1,64}$`
fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rmcp::model::Tool` 是 `#[non_exhaustive]`，测试里用反序列化构造。
    fn mcp_tool(name: &str) -> McpTool {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "description": "测试工具",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}
        }))
        .unwrap()
    }

    #[test]
    fn converts_mcp_tool_to_openai_function() {
        let converted = mcp_tools_to_openai(&[mcp_tool("read_file")]);
        assert_eq!(converted.len(), 1);
        let json = serde_json::to_value(&converted[0]).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "read_file");
        assert_eq!(json["function"]["parameters"]["type"], "object");
        assert_eq!(tool_names(&converted), vec!["read_file"]);
    }

    #[test]
    fn rejects_invalid_names() {
        assert!(is_valid_function_name("read_file"));
        assert!(!is_valid_function_name("读写"));
        assert!(!is_valid_function_name(""));
        assert!(!is_valid_function_name(&"a".repeat(65)));
        assert!(mcp_tools_to_openai(&[mcp_tool("bad name")]).is_empty());
    }
}
