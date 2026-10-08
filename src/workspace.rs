//! 工作目录沙箱：把所有文件操作限制在一个根目录之内。
//!
//! 原实现的两个致命问题：
//! 1. `canonicalize()` 对“还不存在的文件”必然失败，所以 `write_file` 新建文件时总是报错；
//! 2. 只在最后做 `starts_with` 检查，无法处理中间路径不存在 / 符号链接穿透。
//!
//! 这里改为：先做词法归一化（处理 `.` 和 `..`），再把“最深的已存在祖先”canonicalize
//! （从而解析符号链接），拼回剩余部分后做前缀校验。

use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Workspace {
    /// 已 canonicalize 的根目录
    root: PathBuf,
}

impl Workspace {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref();
        let root = root
            .canonicalize()
            .map_err(|e| format!("工作目录不存在或不可访问 {}: {e}", root.display()))?;
        if !root.is_dir() {
            return Err(format!("工作目录不是目录: {}", root.display()));
        }
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 校验并解析路径。允许最后若干级不存在（用于新建文件）。
    pub fn resolve(&self, raw: &str) -> Result<PathBuf, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("path 不能为空".to_string());
        }
        if raw.contains('\0') {
            return Err("path 含有非法字符".to_string());
        }

        let p = Path::new(raw);
        let candidate = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        };
        let normalized = normalize(&candidate);

        let ancestor = deepest_existing(&normalized);
        let canon = ancestor.canonicalize().map_err(|e| {
            format!(
                "路径无效（可能是失效的符号链接）{}: {e}",
                ancestor.display()
            )
        })?;
        let rest = normalized.strip_prefix(&ancestor).unwrap_or(Path::new(""));
        let resolved = if rest.as_os_str().is_empty() {
            canon
        } else {
            canon.join(rest)
        };

        if !resolved.starts_with(&self.root) {
            return Err(format!(
                "拒绝访问工作目录之外的路径: {raw}（工作目录: {}）",
                self.root.display()
            ));
        }
        Ok(resolved)
    }

    /// 相对工作目录展示路径，仅用于给模型/用户的提示信息。
    pub fn display(&self, abs: &Path) -> String {
        match abs.strip_prefix(&self.root) {
            Ok(rel) if !rel.as_os_str().is_empty() => rel.display().to_string(),
            _ => abs.display().to_string(),
        }
    }
}

/// 纯词法归一化：解析 `.` 与 `..`，不做任何磁盘访问。
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                // pop() 在根目录返回 false：绝对路径下的 `..` 停在根，相对路径才需要保留
                if !out.pop() && !path.is_absolute() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// 找到最深的“已存在”祖先（`symlink_metadata` 对失效符号链接也返回 Ok，
/// 这样后续 canonicalize 会失败并报错，而不是悄悄写到链接目标）。
fn deepest_existing(path: &Path) -> PathBuf {
    let mut cur = path.to_path_buf();
    loop {
        if cur.symlink_metadata().is_ok() {
            return cur;
        }
        match cur.parent() {
            Some(parent) if parent != cur => cur = parent.to_path_buf(),
            _ => return cur,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_handles_dot_and_dotdot() {
        assert_eq!(
            normalize(Path::new("/a/b/../c/./d")),
            PathBuf::from("/a/c/d")
        );
        assert_eq!(normalize(Path::new("/../etc")), PathBuf::from("/etc"));
    }
}
