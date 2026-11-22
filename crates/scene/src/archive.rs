#![forbid(unsafe_code)]

//! Security-hardened, deterministic pure-Rust PKZIP (`.biomol.zip`) archive writer & reader.
//!
//! Enforces strict decompression-bomb bounds (`MAX_ARCHIVE_ENTRIES = 256`,
//! `MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 MiB`), path-traversal rejection (`..`, absolute paths,
//! backslashes, NUL bytes, Unix symlinks), and exact IEEE CRC-32 verification on every entry.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;

use crate::SceneError;

/// Maximum number of entries allowed inside a single `.biomol.zip` portable session.
pub const MAX_ARCHIVE_ENTRIES: usize = 256;

/// Maximum total uncompressed payload size across all entries (`128 MiB` decompression-bomb cap).
pub const MAX_ARCHIVE_UNCOMPRESSED_BYTES: usize = 128 * 1024 * 1024;

const SIG_LOCAL_FILE_HEADER: u32 = 0x0403_4b50;
const SIG_CENTRAL_DIR_HEADER: u32 = 0x0201_4b50;
const SIG_END_OF_CENTRAL_DIR: u32 = 0x0605_4b50;

const ZIP_METHOD_STORE: u16 = 0;
const ZIP_METHOD_DEFLATE: u16 = 8;
const ZIP_FLAG_UTF8: u16 = 0x0800;
const ZIP_FIXED_DOS_DATE: u16 = 0x5421; // Deterministic 2022-01-01
const ZIP_UNIX_REGULAR_FILE_ATTR: u32 = 0o100_644 << 16;

/// Validate that a ZIP entry name is strictly relative, UTF-8, normalized, and free of traversal.
pub fn validate_entry_name(name: &str) -> Result<(), SceneError> {
    if name.is_empty() || name.len() > 240 {
        return Err(SceneError::Archive(format!(
            "invalid archive entry name length: '{name}'"
        )));
    }
    if name.starts_with('/')
        || name.starts_with('\\')
        || name.ends_with('/')
        || name.contains('\\')
        || name.contains('\0')
        || name.contains(':')
    {
        return Err(SceneError::Archive(format!(
            "unsafe archive entry path rejected: '{name}'"
        )));
    }
    for segment in name.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(SceneError::Archive(format!(
                "path-traversal segment rejected in archive entry: '{name}'"
            )));
        }
    }
    Ok(())
}

struct CentralEntryRecord {
    name: Vec<u8>,
    method: u16,
    crc32: u32,
    compressed_size: u32,
    uncompressed_size: u32,
    local_header_offset: u32,
}

/// Encode `(relative_path, raw_bytes)` entries into a verified standard PKZIP buffer.
pub fn write_verified_zip_bytes(entries: &[(&str, &[u8])]) -> Result<Vec<u8>, SceneError> {
    if entries.len() > MAX_ARCHIVE_ENTRIES {
        return Err(SceneError::Archive(format!(
            "archive entry count {} exceeds cap {}",
            entries.len(),
            MAX_ARCHIVE_ENTRIES
        )));
    }

    let mut total_uncompressed = 0usize;
    let mut seen_names = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let mut central_records = Vec::with_capacity(entries.len());

    for &(name, payload) in entries {
        validate_entry_name(name)?;
        if !seen_names.insert(name) {
            return Err(SceneError::Archive(format!(
                "duplicate archive entry name: '{name}'"
            )));
        }
        total_uncompressed = total_uncompressed.saturating_add(payload.len());
        if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES {
            return Err(SceneError::Archive(format!(
                "total uncompressed session size {total_uncompressed} exceeds {} bytes cap",
                MAX_ARCHIVE_UNCOMPRESSED_BYTES
            )));
        }

        let crc32 = crc32fast::hash(payload);
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(payload)?;
        let deflated = encoder.finish()?;

        let (method, body): (u16, &[u8]) = if deflated.len() < payload.len() {
            (ZIP_METHOD_DEFLATE, deflated.as_slice())
        } else {
            (ZIP_METHOD_STORE, payload)
        };

        let local_header_offset = u32::try_from(out.len())
            .map_err(|_| SceneError::Archive("archive offset exceeds 32-bit PKZIP limit".into()))?;
        let compressed_size = u32::try_from(body.len())
            .map_err(|_| SceneError::Archive("compressed entry exceeds 32-bit limit".into()))?;
        let uncompressed_size = u32::try_from(payload.len())
            .map_err(|_| SceneError::Archive("uncompressed entry exceeds 32-bit limit".into()))?;
        let name_bytes = name.as_bytes();
        let name_len = u16::try_from(name_bytes.len())
            .map_err(|_| SceneError::Archive("entry name exceeds 16-bit limit".into()))?;

        // Local File Header (30 bytes + name)
        out.extend_from_slice(&SIG_LOCAL_FILE_HEADER.to_le_bytes());
        out.extend_from_slice(&20_u16.to_le_bytes()); // version needed 2.0
        out.extend_from_slice(&ZIP_FLAG_UTF8.to_le_bytes());
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // mod time
        out.extend_from_slice(&ZIP_FIXED_DOS_DATE.to_le_bytes());
        out.extend_from_slice(&crc32.to_le_bytes());
        out.extend_from_slice(&compressed_size.to_le_bytes());
        out.extend_from_slice(&uncompressed_size.to_le_bytes());
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // extra len
        out.extend_from_slice(name_bytes);
        out.extend_from_slice(body);

        central_records.push(CentralEntryRecord {
            name: name_bytes.to_vec(),
            method,
            crc32,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        });
    }

    let cd_offset = u32::try_from(out.len()).map_err(|_| {
        SceneError::Archive("central directory offset exceeds 32-bit PKZIP limit".into())
    })?;

    for rec in &central_records {
        let name_len = u16::try_from(rec.name.len())
            .map_err(|_| SceneError::Archive("entry name exceeds 16-bit limit".into()))?;
        out.extend_from_slice(&SIG_CENTRAL_DIR_HEADER.to_le_bytes());
        out.extend_from_slice(&((3_u16 << 8) | 20_u16).to_le_bytes()); // version made by (Unix, 2.0)
        out.extend_from_slice(&20_u16.to_le_bytes()); // version needed
        out.extend_from_slice(&ZIP_FLAG_UTF8.to_le_bytes());
        out.extend_from_slice(&rec.method.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // mod time
        out.extend_from_slice(&ZIP_FIXED_DOS_DATE.to_le_bytes());
        out.extend_from_slice(&rec.crc32.to_le_bytes());
        out.extend_from_slice(&rec.compressed_size.to_le_bytes());
        out.extend_from_slice(&rec.uncompressed_size.to_le_bytes());
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // extra len
        out.extend_from_slice(&0_u16.to_le_bytes()); // comment len
        out.extend_from_slice(&0_u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0_u16.to_le_bytes()); // internal attrs
        out.extend_from_slice(&ZIP_UNIX_REGULAR_FILE_ATTR.to_le_bytes());
        out.extend_from_slice(&rec.local_header_offset.to_le_bytes());
        out.extend_from_slice(&rec.name);
    }

    let cd_size = u32::try_from(out.len().saturating_sub(cd_offset as usize))
        .map_err(|_| SceneError::Archive("central directory size exceeds 32-bit limit".into()))?;
    let total_entries = u16::try_from(central_records.len())
        .map_err(|_| SceneError::Archive("entry count exceeds 16-bit limit".into()))?;

    // End of Central Directory Record (22 bytes)
    out.extend_from_slice(&SIG_END_OF_CENTRAL_DIR.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&total_entries.to_le_bytes());
    out.extend_from_slice(&total_entries.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());

    Ok(out)
}

/// Write a verified standard PKZIP (`.biomol.zip`) file to disk, returning the byte count written.
pub fn write_verified_zip(path: &Path, entries: &[(&str, &[u8])]) -> Result<usize, SceneError> {
    let bytes = write_verified_zip_bytes(entries)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &bytes)?;
    Ok(bytes.len())
}

fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, SceneError> {
    let slice = data
        .get(offset..offset.saturating_add(2))
        .ok_or_else(|| SceneError::Archive("unexpected EOF reading u16".into()))?;
    let arr: [u8; 2] = slice
        .try_into()
        .map_err(|_| SceneError::Archive("short u16 slice".into()))?;
    Ok(u16::from_le_bytes(arr))
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, SceneError> {
    let slice = data
        .get(offset..offset.saturating_add(4))
        .ok_or_else(|| SceneError::Archive("unexpected EOF reading u32".into()))?;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| SceneError::Archive("short u32 slice".into()))?;
    Ok(u32::from_le_bytes(arr))
}

/// Read and strictly verify every entry in a PKZIP buffer (bounds, path traversal, symlinks, CRC-32).
pub fn read_verified_zip_bytes(data: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, SceneError> {
    read_verified_zip_bytes_with_limit(data, MAX_ARCHIVE_UNCOMPRESSED_BYTES)
}

/// Same as [`read_verified_zip_bytes`] with a caller-specified uncompressed byte ceiling (capped at 128 MiB).
pub fn read_verified_zip_bytes_with_limit(
    data: &[u8],
    max_uncompressed_bytes: usize,
) -> Result<BTreeMap<String, Vec<u8>>, SceneError> {
    let effective_limit = max_uncompressed_bytes.min(MAX_ARCHIVE_UNCOMPRESSED_BYTES);
    if data.len() < 22 {
        return Err(SceneError::Archive(
            "file is smaller than minimum 22-byte ZIP EOCD record".into(),
        ));
    }

    // Locate End of Central Directory record scanning backwards up to 65557 bytes
    let search_start = data.len().saturating_sub(65_557);
    let mut eocd_pos = None;
    let mut i = data.len() - 22;
    loop {
        if read_u32_le(data, i)? == SIG_END_OF_CENTRAL_DIR {
            let comment_len = read_u16_le(data, i + 20)? as usize;
            if i + 22 + comment_len == data.len() {
                eocd_pos = Some(i);
                break;
            }
        }
        if i <= search_start {
            break;
        }
        i -= 1;
    }
    let eocd = eocd_pos.ok_or_else(|| {
        SceneError::Archive("missing PKZIP End-of-Central-Directory record".into())
    })?;

    let total_entries = read_u16_le(data, eocd + 10)? as usize;
    if total_entries > MAX_ARCHIVE_ENTRIES {
        return Err(SceneError::Archive(format!(
            "archive contains {total_entries} entries, exceeding cap {MAX_ARCHIVE_ENTRIES}"
        )));
    }
    let cd_size = read_u32_le(data, eocd + 12)? as usize;
    let cd_offset = read_u32_le(data, eocd + 16)? as usize;
    let cd_end = cd_offset
        .checked_add(cd_size)
        .ok_or_else(|| SceneError::Archive("central directory bounds overflow".into()))?;
    if cd_end > eocd {
        return Err(SceneError::Archive(
            "central directory extends past EOCD record".into(),
        ));
    }

    let mut entries = BTreeMap::new();
    let mut cursor = cd_offset;
    let mut total_extracted = 0usize;

    for _ in 0..total_entries {
        if cursor.saturating_add(46) > cd_end {
            return Err(SceneError::Archive(
                "truncated central directory header".into(),
            ));
        }
        if read_u32_le(data, cursor)? != SIG_CENTRAL_DIR_HEADER {
            return Err(SceneError::Archive(
                "invalid central directory entry signature".into(),
            ));
        }

        let version_made_by = read_u16_le(data, cursor + 4)?;
        let method = read_u16_le(data, cursor + 10)?;
        let expected_crc = read_u32_le(data, cursor + 16)?;
        let comp_size = read_u32_le(data, cursor + 20)? as usize;
        let uncomp_size = read_u32_le(data, cursor + 24)? as usize;
        let name_len = read_u16_le(data, cursor + 28)? as usize;
        let extra_len = read_u16_le(data, cursor + 30)? as usize;
        let comment_len = read_u16_le(data, cursor + 32)? as usize;
        let ext_attrs = read_u32_le(data, cursor + 38)?;
        let local_offset = read_u32_le(data, cursor + 42)? as usize;

        // Reject Unix symlinks (0o120000 in upper 16 bits of external attributes)
        let host_os = version_made_by >> 8;
        if host_os == 3 {
            let unix_mode = (ext_attrs >> 16) & 0o170_000;
            if unix_mode == 0o120_000 {
                return Err(SceneError::Archive(
                    "symbolic link entry rejected inside verified archive".into(),
                ));
            }
        }

        let name_start = cursor + 46;
        let name_end = name_start.saturating_add(name_len);
        let name_bytes = data
            .get(name_start..name_end)
            .ok_or_else(|| SceneError::Archive("truncated entry filename".into()))?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| SceneError::Archive("non-UTF8 archive entry name rejected".into()))?;
        validate_entry_name(name)?;

        if entries.contains_key(name) {
            return Err(SceneError::Archive(format!(
                "duplicate archive entry rejected: '{name}'"
            )));
        }

        let remaining_budget = effective_limit.saturating_sub(total_extracted);
        if uncomp_size > remaining_budget {
            return Err(SceneError::Archive(format!(
                "decompression bomb guard triggered on '{name}' (declared {uncomp_size} B > remaining {remaining_budget} B)"
            )));
        }

        // Validate Local File Header and slice compressed payload
        if local_offset.saturating_add(30) > cd_offset {
            return Err(SceneError::Archive("invalid local header offset".into()));
        }
        if read_u32_le(data, local_offset)? != SIG_LOCAL_FILE_HEADER {
            return Err(SceneError::Archive(
                "invalid local file header signature".into(),
            ));
        }
        let local_name_len = read_u16_le(data, local_offset + 26)? as usize;
        let local_extra_len = read_u16_le(data, local_offset + 28)? as usize;
        let payload_start = local_offset
            .checked_add(30)
            .and_then(|v| v.checked_add(local_name_len))
            .and_then(|v| v.checked_add(local_extra_len))
            .ok_or_else(|| SceneError::Archive("local payload offset overflow".into()))?;
        let payload_end = payload_start
            .checked_add(comp_size)
            .ok_or_else(|| SceneError::Archive("local payload size overflow".into()))?;
        if payload_end > cd_offset {
            return Err(SceneError::Archive(format!(
                "compressed payload for '{name}' overlaps central directory"
            )));
        }
        let comp_slice = &data[payload_start..payload_end];

        // Decompress with hard streaming byte ceiling (defeats spoofed header sizes!)
        let decoded = match method {
            ZIP_METHOD_STORE => {
                if comp_slice.len() > remaining_budget {
                    return Err(SceneError::Archive(format!(
                        "stored entry '{name}' exceeds remaining decompression budget"
                    )));
                }
                comp_slice.to_vec()
            }
            ZIP_METHOD_DEFLATE => {
                let mut decoder = DeflateDecoder::new(comp_slice)
                    .take((remaining_budget as u64).saturating_add(1));
                let mut buf = Vec::with_capacity(uncomp_size.min(1024 * 1024));
                decoder.read_to_end(&mut buf)?;
                if buf.len() > remaining_budget {
                    return Err(SceneError::Archive(format!(
                        "decompression bomb guard triggered while inflating '{name}' (> {effective_limit} B cap)"
                    )));
                }
                buf
            }
            other => {
                return Err(SceneError::Archive(format!(
                    "unsupported ZIP compression method {other} for '{name}'"
                )));
            }
        };

        if decoded.len() != uncomp_size {
            return Err(SceneError::Archive(format!(
                "size mismatch for '{name}': header={uncomp_size}, actual={}",
                decoded.len()
            )));
        }
        let actual_crc = crc32fast::hash(&decoded);
        if actual_crc != expected_crc {
            return Err(SceneError::Archive(format!(
                "CRC-32 integrity failure for '{name}': expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}"
            )));
        }

        total_extracted = total_extracted.saturating_add(decoded.len());
        entries.insert(name.to_string(), decoded);

        cursor = name_end
            .checked_add(extra_len)
            .and_then(|v| v.checked_add(comment_len))
            .ok_or_else(|| SceneError::Archive("central directory cursor overflow".into()))?;
    }

    Ok(entries)
}

/// Read and strictly verify a `.biomol.zip` file from disk.
pub fn read_verified_zip(path: &Path) -> Result<BTreeMap<String, Vec<u8>>, SceneError> {
    let raw = std::fs::read(path)?;
    read_verified_zip_bytes(&raw)
}
