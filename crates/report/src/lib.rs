#![forbid(unsafe_code)]

//! `craftbio-report` — Shared XSS-Safe HTML/XML Escaping, Offline Mol* Asset Resolution,
//! and Portable Reporting Utilities for all CraftBio Applications.

use std::path::PathBuf;

/// Resolve the shared CraftBio local data directory (`CRAFT_DATA_DIR` or `/home/acs/Datos/CraftBio`).
#[must_use]
pub fn craft_data_dir() -> PathBuf {
    if let Ok(env_dir) = std::env::var("CRAFT_DATA_DIR")
        && !env_dir.trim().is_empty()
    {
        return PathBuf::from(env_dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/acs".to_string());
    let datos = PathBuf::from(&home).join("Datos").join("CraftBio");
    if datos.exists() {
        datos
    } else {
        PathBuf::from(home).join(".local/share/craftbio")
    }
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
}
