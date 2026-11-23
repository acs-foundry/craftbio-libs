#![forbid(unsafe_code)]

//! `craftbio-report` — Shared XSS-Safe HTML/XML Escaping, Offline Mol* Asset Resolution,
//! and Portable Reporting Utilities for all CraftBio Applications.

pub mod dossier;

use std::path::PathBuf;

pub use dossier::{
    ArtCraftBundleSummary, HtmlDossierSummary, export_artcraft_bundle, resolve_artcraft_cli,
    write_biomol_html_dossier,
};

/// Resolve the shared CraftBio local data directory.
///
/// `CRAFT_DATA_DIR` if set, otherwise `$XDG_DATA_HOME/CraftBio`, otherwise
/// `$HOME/.local/share/CraftBio`. Set `CRAFT_DATA_DIR` to keep large assets off the root
/// partition. No path is hardcoded for any particular machine.
#[must_use]
pub fn craft_data_dir() -> PathBuf {
    let non_blank = |k: &str| {
        std::env::var(k)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };

    if let Some(dir) = non_blank("CRAFT_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = non_blank("XDG_DATA_HOME") {
        return PathBuf::from(xdg).join("CraftBio");
    }
    if let Some(home) = non_blank("HOME") {
        return PathBuf::from(home).join(".local/share/CraftBio");
    }
    PathBuf::from("CraftBio")
}

/// Escape text safely for HTML/XML/SVG inclusion (prevents XSS and control-char corruption).
#[must_use]
pub fn escape_markup(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
            continue;
        }
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Safely serialize a JSON value for embedding inside an HTML `<script>` block.
pub fn safe_json_for_script(val: &serde_json::Value) -> Result<String, serde_json::Error> {
    let raw = serde_json::to_string(val)?;
    Ok(raw
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029"))
}

/// Load locally cached offline Mol* static bundle (`molstar.css`, `molstar.js`) if installed.
#[must_use]
pub fn load_offline_molstar_assets() -> Option<(String, String)> {
    let dir = craft_data_dir().join("assets").join("molstar");
    let css = std::fs::read_to_string(dir.join("molstar.css")).ok()?;
    let js = std::fs::read_to_string(dir.join("molstar.js")).ok()?;
    if css.is_empty() || js.is_empty() {
        None
    } else {
        Some((css, js))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_xss_payloads_cleanly() {
        let escaped = escape_markup("<script>alert('x') & \"y\"</script>");
        assert!(!escaped.contains('<'));
        assert!(escaped.contains("&lt;script&gt;"));
    }

    #[test]
    fn writes_dossier_html_and_artcraft_bundle() -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("craftbio_rep_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let html_path = base.join("dossier.html");
        let sum = write_biomol_html_dossier(
            "Test <Complex>",
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n",
            "# Report",
            &[("SVG Chart", "<svg></svg>")],
            false,
            &html_path,
        )?;
        assert!(sum.bytes > 500);
        let raw = std::fs::read_to_string(&html_path)?;
        assert!(raw.contains("&lt;Complex&gt;"));

        let bundle = export_artcraft_bundle(
            "Test ArtCraft",
            b"\x89PNG\r\n\x1a\n",
            &[("RMSD", "0.88 A".to_string())],
            "# Summary",
            &base.join("artcraft"),
        )?;
        assert!(std::path::Path::new(&bundle.gridcraft_tsv).is_file());
        assert!(std::path::Path::new(&bundle.wordcraft_md).is_file());
        assert!(std::path::Path::new(&bundle.deckcraft_slides_md).is_file());
        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}
