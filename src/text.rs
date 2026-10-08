//! 纯文本小工具：去除思维链、按字符截断。

/// Qwen3 等“思考型”模型的思维链标记（ASCII 写法）。
///
/// 这里用 `\x3C` / `\x3E` 转义写出尖括号，避免源码里再次出现模型特殊 token 的字形。
const OPEN_TAG: &str = "\x3Cthink\x3E";
const CLOSE_TAG: &str = "\x3C/think\x3E";

/// 去掉思维链片段。
///
/// 两个原因必须去掉：
/// 1. 回传给 Ollama 的历史里不能包含思维链标记，否则会被拒绝；
/// 2. 直接展示给用户时思维链只是噪声。
///
/// 顺带兼容 `<thinking>` 写法；未闭合的开标签意味着其后全部是思考内容，直接丢弃。
pub fn strip_thinking(text: &str) -> String {
    let open_tags = [OPEN_TAG, "\x3Cthinking\x3E"];
    let close_tags = [CLOSE_TAG, "\x3C/thinking\x3E"];

    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    loop {
        let Some(idx) = open_tags.iter().filter_map(|t| rest.find(t)).min() else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..idx]);

        let after_open = &rest[idx..];
        let Some(close_pos) = close_tags.iter().filter_map(|t| after_open.find(t)).min() else {
            break; // 未闭合：后面全是思考内容，丢弃
        };
        let close_tag = close_tags
            .iter()
            .find(|t| after_open[close_pos..].starts_with(**t))
            .expect("close tag 必然匹配其一");
        rest = &after_open[close_pos + close_tag.len()..];
    }

    out.trim().to_string()
}

/// 按字符（而非字节）截断，避免切坏 UTF-8。
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str(" …[已截断]");
    out
}

/// 截断到最多 `max_chars` 个字符，并附带说明（用于工具结果）。
pub fn truncate_with_note(s: &str, max_chars: usize, note: &str) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str(&format!("\n…[{note}，原始长度 {} 字符]", s.chars().count()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn think(body: &str) -> String {
        format!("{OPEN_TAG}{body}{CLOSE_TAG}")
    }

    #[test]
    fn strips_closed_thinking_block() {
        assert_eq!(strip_thinking(&think("思考过程")), "");
        assert_eq!(strip_thinking(&format!("a{}c", think("b"))), "ac");
        assert_eq!(
            strip_thinking(&format!("{}答案", think("思考过程"))),
            "答案"
        );
    }

    #[test]
    fn strips_unclosed_thinking_block() {
        assert_eq!(strip_thinking(&format!("答案前{OPEN_TAG}还在想")), "答案前");
    }

    #[test]
    fn strips_thinking_variant_tag() {
        assert_eq!(
            strip_thinking("\x3Cthinking\x3E想了很久\x3C/thinking\x3E结果"),
            "结果"
        );
    }

    #[test]
    fn keeps_plain_text() {
        assert_eq!(strip_thinking("  普通回答  "), "普通回答");
        assert_eq!(
            strip_thinking("提到 think 这个词不算标记"),
            "提到 think 这个词不算标记"
        );
    }

    #[test]
    fn strips_multiple_blocks() {
        assert_eq!(
            strip_thinking(&format!("{}中{}后", think("思考1"), think("思考2"))),
            "中后"
        );
    }

    #[test]
    fn truncation_is_utf8_safe() {
        assert_eq!(truncate_chars("中文abc", 2), "中文 …[已截断]");
        assert_eq!(truncate_chars("abc", 10), "abc");
        assert!(truncate_with_note("中文abc", 2, "太长").contains("原始长度 5 字符"));
    }
}
