#![forbid(unsafe_code)]

//! Pure-Rust Bounded WebAssembly Binary Inspector (aligned with `storytold/craft-wasm`):
//! validates `.wasm` magic, version, LEB128 section boundaries, export names, and custom sections
//! without ever executing untrusted code.

use serde::{Deserialize, Serialize};

use crate::SandboxError;

/// Maximum WASM module size inspected (32 MiB).
pub const MAX_WASM_BYTES: usize = 32 * 1024 * 1024;

/// Summary of a validated WebAssembly plugin module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPluginReport {
    pub version: u32,
    pub total_bytes: usize,
    pub section_count: usize,
    pub sections: Vec<(String, usize)>,
    pub exports: Vec<String>,
    pub custom_sections: Vec<String>,
}

fn read_leb_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, SandboxError> {
    let mut result = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        let b = *bytes
            .get(*pos)
            .ok_or_else(|| SandboxError::RejectedPath("truncated WASM LEB128".to_string()))?;
        *pos += 1;
        result |= u32::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }
    Err(SandboxError::RejectedPath(
        "overlong WASM LEB128".to_string(),
    ))
}

fn read_name(bytes: &[u8], pos: &mut usize) -> Result<String, SandboxError> {
    let len = read_leb_u32(bytes, pos)? as usize;
    let slice = bytes
        .get(*pos..*pos + len)
        .ok_or_else(|| SandboxError::RejectedPath("truncated WASM UTF-8 name".to_string()))?;
    *pos += len;
    String::from_utf8(slice.to_vec())
        .map_err(|_| SandboxError::RejectedPath("invalid UTF-8 in WASM name".to_string()))
}

/// Safely inspect and validate a `.wasm` binary plugin header, sections, and exports (never executes code).
pub fn inspect_wasm_plugin(bytes: &[u8]) -> Result<WasmPluginReport, SandboxError> {
    if bytes.len() < 8 {
        return Err(SandboxError::RejectedPath(
            "WASM binary < 8 bytes".to_string(),
        ));
    }
    if bytes.len() > MAX_WASM_BYTES {
        return Err(SandboxError::SizeLimitExceeded {
            size: bytes.len() as u64,
            cap: MAX_WASM_BYTES as u64,
        });
    }
    if &bytes[0..4] != b"\0asm" {
        return Err(SandboxError::RejectedPath(
            "missing '\\0asm' WASM magic".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 1 {
        return Err(SandboxError::RejectedPath(format!(
            "unsupported WASM version {version} (expected 1)"
        )));
    }

    let mut pos = 8usize;
    let mut sections = Vec::new();
    let mut exports = Vec::new();
    let mut custom_sections = Vec::new();

    while pos < bytes.len() {
        let sec_id = bytes[pos];
        pos += 1;
        let sec_len = read_leb_u32(bytes, &mut pos)? as usize;
        let sec_end = pos.checked_add(sec_len).ok_or_else(|| {
            SandboxError::RejectedPath("WASM section length overflow".to_string())
        })?;
        if sec_end > bytes.len() {
            return Err(SandboxError::RejectedPath(
                "WASM section extends past EOF".to_string(),
            ));
        }
        let sec_name = match sec_id {
            0 => "custom",
            1 => "type",
            2 => "import",
            3 => "function",
            4 => "table",
            5 => "memory",
            6 => "global",
            7 => "export",
            8 => "start",
            9 => "element",
            10 => "code",
            11 => "data",
            12 => "data_count",
            _ => "unknown",
        };
        sections.push((sec_name.to_string(), sec_len));

        if sec_id == 0 {
            let mut cpos = pos;
            if let Ok(cname) = read_name(bytes, &mut cpos) {
                custom_sections.push(cname);
            }
        } else if sec_id == 7 {
            let mut epos = pos;
            if let Ok(count) = read_leb_u32(bytes, &mut epos) {
                for _ in 0..(count as usize).min(256) {
                    let Ok(ename) = read_name(bytes, &mut epos) else {
                        break;
                    };
                    if epos >= sec_end {
                        break;
                    }
                    epos += 1; // export kind byte
                    let _ = read_leb_u32(bytes, &mut epos); // index
                    exports.push(ename);
                }
            }
        }
        pos = sec_end;
    }

    Ok(WasmPluginReport {
        version,
        total_bytes: bytes.len(),
        section_count: sections.len(),
        sections,
        exports,
        custom_sections,
    })
}
