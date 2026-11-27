#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::expect_used))]

//! `craftbio-fetch` — shared, allow-listed HTTPS text fetch for CraftBio apps.
//!
//! The API is deliberately narrow: the caller declares the hosts it trusts, a
//! cache namespace and a byte ceiling. Downloads happen only when the caller
//! invokes [`fetch_text`]; nothing is sent on startup and no design data is
//! uploaded. Responses are cached under the shared `craftbio-report` data
//! directory and every request is HTTPS-only, including redirects.

use std::{fs, path::PathBuf, process::Command};

use craftbio_report::craft_data_dir;

/// Caller-declared limits for one fetch operation.
#[derive(Debug, Clone, Copy)]
pub struct FetchPolicy<'a> {
    /// Exact HTTPS hosts allowed for this call. Ports are accepted; userinfo is not.
    pub allowed_hosts: &'a [&'a str],
    /// Maximum downloaded and cached text size, in bytes.
    pub max_bytes: u64,
    /// Short product token sent as `User-Agent`.
    pub user_agent: &'a str,
}

/// A validated text response plus its local cache location.
#[derive(Debug, Clone)]
pub struct FetchedText {
    pub text: String,
    pub cached: bool,
    pub cache_path: PathBuf,
}

fn authority_host(authority: &str) -> Option<&str> {
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let host = match authority.rsplit_once(':') {
        Some((host, port))
            if !port.is_empty() && port.chars().all(|ascii| ascii.is_ascii_digit()) =>
        {
            host
        }
        _ => authority,
    };
    if host.is_empty() { None } else { Some(host) }
}

fn validate_url(url: &str, policy: &FetchPolicy<'_>) -> Result<(), String> {
    if url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("fetch URL contains whitespace or control characters".to_string());
    }
    let Some(rest) = url.strip_prefix("https://") else {
        return Err("fetch URL must use HTTPS".to_string());
    };
    let authority = match rest.find(['/', '?', '#']) {
        Some(index) => &rest[..index],
        None => rest,
    };
    let Some(host) = authority_host(authority) else {
        return Err("fetch URL has no valid host".to_string());
    };
    if !policy
        .allowed_hosts
        .iter()
        .any(|allowed| host.eq_ignore_ascii_case(allowed))
    {
        return Err(format!(
            "fetch host '{host}' is not in the caller allow-list"
        ));
    }
    Ok(())
}

fn validate_component(raw: &str, label: &str) -> Result<String, String> {
    let clean = raw.trim();
    if clean.is_empty() || clean.len() > 128 {
        return Err(format!("{label} must be 1–128 characters"));
    }
    if clean.contains("..")
        || clean.starts_with('.')
        || clean.starts_with('-')
        || clean.contains(['/', '\\'])
        || !clean
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return Err(format!(
            "{label} may contain only letters, digits, '_', '-' and '.'"
        ));
    }
    Ok(clean.to_string())
}

fn validate_policy(policy: &FetchPolicy<'_>) -> Result<(), String> {
    if policy.allowed_hosts.is_empty() {
        return Err("fetch policy has no allowed hosts".to_string());
    }
    if policy.max_bytes == 0 {
        return Err("fetch policy has a zero byte limit".to_string());
    }
    if policy.user_agent.trim().is_empty() || policy.user_agent.len() > 128 {
        return Err("fetch policy user-agent must be 1–128 characters".to_string());
    }
    // The user-agent is passed as one argv element, so internal spaces are safe;
    // only control characters could corrupt the request or terminal output.
    if policy.user_agent.chars().any(char::is_control) {
        return Err("fetch policy user-agent contains control characters".to_string());
    }
    Ok(())
}

fn short_stderr(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim()
        .chars()
        .take(240)
        .collect()
}

/// Resolve the shared cache directory for one namespace.
pub fn cache_dir(namespace: &str) -> Result<PathBuf, String> {
    let namespace = validate_component(namespace, "cache namespace")?;
    Ok(craft_data_dir().join("cache").join(namespace))
}

/// Download a text record under the caller's policy, or reuse a valid cache entry.
///
/// # Errors
///
/// Returns an error for a disallowed URL, an unsafe cache key, a failed HTTPS
/// request, an oversized or empty response, or an unwritable cache directory.
pub fn fetch_text(
    url: &str,
    namespace: &str,
    key: &str,
    policy: &FetchPolicy<'_>,
) -> Result<FetchedText, String> {
    validate_policy(policy)?;
    validate_url(url, policy)?;
    let namespace = validate_component(namespace, "cache namespace")?;
    let key = validate_component(key, "cache key")?;

    let dir = craft_data_dir().join("cache").join(namespace);
    fs::create_dir_all(&dir).map_err(|error| format!("cache mkdir failed: {error}"))?;
    let path = dir.join(format!("{key}.txt"));
    let tmp = dir.join(format!("{key}.tmp"));

    if path.is_file() {
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cached record is not readable text: {error}"))?;
        if !text.trim().is_empty() {
            return Ok(FetchedText {
                text,
                cached: true,
                cache_path: path,
            });
        }
    }

    let max_filesize = policy.max_bytes.to_string();
    let output = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--max-time",
            "30",
            "--max-filesize",
            max_filesize.as_str(),
            "--user-agent",
            policy.user_agent,
            "-o",
        ])
        .arg(&tmp)
        .arg("--")
        .arg(url)
        .output()
        .map_err(|error| format!("curl is required for public record import: {error}"))?;
    if !output.status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(format!(
            "HTTPS fetch failed for allowed host: {}",
            short_stderr(&output.stderr)
        ));
    }

    let text = fs::read_to_string(&tmp)
        .map_err(|error| format!("downloaded record is not readable text: {error}"))?;
    if text.trim().is_empty() {
        let _ = fs::remove_file(&tmp);
        return Err("provider returned an empty record".to_string());
    }
    fs::rename(&tmp, &path).map_err(|error| format!("cache finalize failed: {error}"))?;
    Ok(FetchedText {
        text,
        cached: false,
        cache_path: path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy<'a>(hosts: &'a [&'a str]) -> FetchPolicy<'a> {
        FetchPolicy {
            allowed_hosts: hosts,
            max_bytes: 1024,
            user_agent: "CraftBioTest/1.0",
        }
    }

    #[test]
    fn allows_only_declared_https_hosts() {
        let hosts = ["rest.uniprot.org", "eutils.ncbi.nlm.nih.gov"];
        assert!(
            validate_url(
                "https://rest.uniprot.org/uniprotkb/P04637.fasta",
                &policy(&hosts)
            )
            .is_ok()
        );
        assert!(
            validate_url(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_000546",
                &policy(&hosts)
            )
            .is_ok()
        );
        assert!(validate_url("http://rest.uniprot.org/x", &policy(&hosts)).is_err());
        assert!(validate_url("https://example.org/x", &policy(&hosts)).is_err());
        assert!(validate_url("https://user@rest.uniprot.org/x", &policy(&hosts)).is_err());
        assert!(validate_url("https://rest.uniprot.org/a b", &policy(&hosts)).is_err());
    }

    #[test]
    fn rejects_unsafe_cache_components() {
        assert!(validate_component("genecraft", "namespace").is_ok());
        assert!(validate_component("../escape", "cache key").is_err());
        assert!(validate_component("a/b", "cache key").is_err());
        assert!(validate_component("", "cache key").is_err());
    }

    #[test]
    fn policy_rejects_missing_hosts_and_agents() {
        let mut empty = policy(&[]);
        assert!(validate_policy(&empty).is_err());
        empty = policy(&["rest.uniprot.org"]);
        empty.user_agent = "";
        assert!(validate_policy(&empty).is_err());
        empty.user_agent = "GeneCraft/0.2 (CraftBio Suite)";
        assert!(validate_policy(&empty).is_ok());
        empty.user_agent = "bad\nagent";
        assert!(validate_policy(&empty).is_err());
    }

    #[test]
    fn cache_dir_uses_the_shared_data_directory() {
        let dir = cache_dir("genecraft").expect("valid namespace");
        assert!(dir.ends_with("cache/genecraft"));
    }
}
