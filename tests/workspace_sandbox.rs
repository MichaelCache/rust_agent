//! 工作目录沙箱的行为测试（原实现最危险的几个 bug 都在这里）。

use std::fs;
use std::path::{Path, PathBuf};

use rust_agent::workspace::Workspace;

/// 用一个干净的临时目录做测试根，位置在 `target/tmp` 下。
fn fresh_dir(name: &str) -> PathBuf {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sandbox")
        .join(name);
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    base
}

#[test]
fn resolves_relative_path_inside_root() {
    let root = fresh_dir("relative");
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();

    let ws = Workspace::new(&root).unwrap();
    let resolved = ws.resolve("src/main.rs").unwrap();
    assert_eq!(resolved, root.canonicalize().unwrap().join("src/main.rs"));
    // `.` 与重复分隔符都要能归一化
    assert_eq!(ws.resolve("./src//main.rs").unwrap(), resolved);
}

#[test]
fn allows_path_that_does_not_exist_yet() {
    // 这是原实现的关键 bug：canonicalize() 对不存在的文件必然失败，
    // 导致 write_file 永远无法新建文件。
    let root = fresh_dir("new-file");
    let ws = Workspace::new(&root).unwrap();

    let resolved = ws.resolve("src/deep/new_file.rs").unwrap();
    assert_eq!(
        resolved,
        root.canonicalize().unwrap().join("src/deep/new_file.rs")
    );
}

#[test]
fn rejects_parent_dir_escape() {
    let root = fresh_dir("escape-dotdot");
    let ws = Workspace::new(&root).unwrap();

    let err = ws.resolve("../../etc/passwd").unwrap_err();
    assert!(err.contains("拒绝访问"), "{err}");
    let err = ws.resolve("src/../../outside").unwrap_err();
    assert!(err.contains("拒绝访问"), "{err}");
}

#[test]
fn rejects_absolute_path_outside_root() {
    let root = fresh_dir("escape-abs");
    let ws = Workspace::new(&root).unwrap();

    let err = ws.resolve("/etc/passwd").unwrap_err();
    assert!(err.contains("拒绝访问"), "{err}");
}

#[test]
fn accepts_absolute_path_inside_root() {
    let root = fresh_dir("abs-inside");
    fs::write(root.join("a.txt"), "hi").unwrap();
    let ws = Workspace::new(&root).unwrap();

    let abs = root.canonicalize().unwrap().join("a.txt");
    assert_eq!(ws.resolve(abs.to_str().unwrap()).unwrap(), abs);
}

#[test]
fn rejects_empty_and_nul_paths() {
    let root = fresh_dir("empty");
    let ws = Workspace::new(&root).unwrap();
    assert!(ws.resolve("   ").is_err());
    assert!(ws.resolve("a\0b").is_err());
}

#[cfg(unix)]
#[test]
fn rejects_symlink_escape() {
    let root = fresh_dir("symlink");
    let outside = fresh_dir("symlink-outside");
    fs::write(outside.join("secret.txt"), "top secret").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

    let ws = Workspace::new(&root).unwrap();
    // 通过符号链接读到根目录之外 —— 必须被拒绝
    let err = ws.resolve("link/secret.txt").unwrap_err();
    assert!(err.contains("拒绝访问"), "{err}");
}

#[cfg(unix)]
#[test]
fn rejects_dangling_symlink() {
    let root = fresh_dir("dangling");
    std::os::unix::fs::symlink("/tmp/does-not-exist-xyz", root.join("dangling")).unwrap();

    let ws = Workspace::new(&root).unwrap();
    // 失效链接既不能读也不能作为写入目标（否则会写到链接目标处）
    assert!(ws.resolve("dangling").is_err());
}

#[test]
fn root_must_exist() {
    let missing = Path::new(env!("CARGO_TARGET_TMPDIR")).join("definitely/missing/dir");
    assert!(Workspace::new(&missing).is_err());
}

#[test]
fn display_is_relative_to_root() {
    let root = fresh_dir("display");
    fs::create_dir_all(root.join("src")).unwrap();
    let ws = Workspace::new(&root).unwrap();
    let abs = ws.resolve("src/lib.rs").unwrap();
    assert_eq!(ws.display(&abs), "src/lib.rs");
}
