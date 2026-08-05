//! Vocabulary oracle over the user-facing surface.
//!
//! A green test suite is weak evidence of rename completeness: the old
//! names still compile fine. The oracle is a vocabulary scan — the README,
//! docs, examples, and rendered API surface must never carry the pre-rename
//! product names.

use std::fs;
use std::path::{Path, PathBuf};

/// Terms that must not appear anywhere on the user-facing surface, with the
/// reason each is forbidden.
const FORBIDDEN: &[(&str, &str)] = &[
    ("g_hyper", "pre-rename crate name"),
    ("gHyper", "pre-rename product name"),
    ("g_file", "pre-rename storage crate name"),
    ("gFile", "pre-rename storage product name"),
    ("GeoFile", "pre-rename storage handle type name"),
];

/// Files preserved wholesale: explicitly historical documents whose entire
/// content is a record of the past.
const PRESERVED_FILES: &[&str] = &["docs/GEOH_FORMAT.md"];

/// Historical statements that stay true under their original names — dated
/// measurement records are facts about the past, and rewriting them to
/// satisfy this scan would falsify history. (file suffix, line substring).
const PRESERVED_HISTORY: &[(&str, &str)] = &[
    ("BENCHMARKS.md", "# gHyper Benchmarks — measured 2026-07-09"),
    ("BENCHMARKS.md", "Quantized semantic storage (gFile v0.7.0, format v4)"),
    ("BENCHMARKS.md", "`gFile/tests/quantized.rs`"),
    ("BENCHMARKS.md", "## gFile end-to-end operations"),
    ("BENCHMARKS.md", "cargo bench --bench throughput` in gFile"),
    ("BENCHMARKS.md", "First recorded gFile-side numbers"),
];

/// The user-facing surface: everything a user reads before (or instead of)
/// the source code — plus `src`, which carries the rendered API docs and
/// the doc-tests.
const SURFACE: &[&str] = &[
    "README.md",
    "PROOF.md",
    "BENCHMARKS.md",
    "docs",
    "examples",
    "src",
];

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        out.push(root.to_path_buf());
        return;
    }
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        collect_files(&path, out);
    }
}

#[test]
fn user_facing_surface_never_carries_pre_rename_vocabulary() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();

    for surface in SURFACE {
        let mut files = Vec::new();
        collect_files(&manifest_dir.join(surface), &mut files);
        for file in files {
            if PRESERVED_FILES.iter().any(|p| file.ends_with(p)) {
                continue;
            }
            let bytes = fs::read(&file).unwrap_or_default();
            let text = String::from_utf8_lossy(&bytes);
            // Implementation code may reference whatever it needs; what must
            // stay clean is the text a user sees — rendered doc comments.
            let doc_comments_only =
                *surface == "src" && file.extension().is_some_and(|e| e == "rs");
            for (lineno, line) in text.lines().enumerate() {
                if doc_comments_only {
                    let t = line.trim_start();
                    if !t.starts_with("///") && !t.starts_with("//!") {
                        continue;
                    }
                }
                let preserved = PRESERVED_HISTORY.iter().any(|(suffix, substring)| {
                    file.ends_with(suffix) && line.contains(substring)
                });
                if preserved {
                    continue;
                }
                for (term, why) in FORBIDDEN {
                    if line.contains(term) {
                        violations.push(format!(
                            "{}:{}: contains `{}` ({})\n    {}",
                            file.strip_prefix(&manifest_dir).unwrap_or(&file).display(),
                            lineno + 1,
                            term,
                            why,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "forbidden vocabulary on the user-facing surface:\n{}",
        violations.join("\n")
    );
}
