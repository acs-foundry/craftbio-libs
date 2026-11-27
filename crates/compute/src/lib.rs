#![forbid(unsafe_code)]

//! Pure-Rust, offline-first `ComputeJobSpec` and `ComputeResultManifest` contract
//! shared between the CraftBio suite (`DockCraft`, `BiomolCraft`, `GeneCraft`, `PathCraft`)
//! and ResearchFW's `fw compute` remote orchestrator.
//!
//! Zero network dependencies, `#![forbid(unsafe_code)]`, strict panic-denial policy.

pub mod adapters;
pub mod sha256;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Versioned schema identifier shared verbatim with Python `compute_spec.py`.
pub const SPEC_SCHEMA_VERSION: &str = "craft-compute/1.0";

/// Minimum and maximum allowed job timeouts in seconds (closes `auditoria-craftbio` F3-23).
pub const MIN_TIMEOUT_S: u64 = 10;
pub const MAX_TIMEOUT_S: u64 = 86_400;

#[derive(Debug, Error)]
pub enum ComputeError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsafe relative path rejected: '{0}'")]
    UnsafePath(String),
    #[error("unsupported schema_version '{found}' (expected '{expected}')")]
    UnsupportedSchema { found: String, expected: String },
    #[error("invalid timeout_s={0} (must be in {MIN_TIMEOUT_S}..={MAX_TIMEOUT_S})")]
    InvalidTimeout(u64),
    #[error("confidential IP violation: {0}")]
    ConfidentialIpViolation(String),
    #[error("missing artifact file: '{0}'")]
    MissingArtifact(String),
    #[error("SHA-256 mismatch for '{path}': expected {expected}, actual {actual}")]
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("empty or invalid identifier: {0}")]
    InvalidInput(String),
}

/// Data sensitivity tier matching ResearchFW's `compute_guard.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DataTier {
    #[default]
    PublicOnly,
    ClinicalAnon,
    ClinicalIdentifiable,
}

/// Intellectual property confidentiality tier matching ResearchFW's `compute_guard.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IpTier {
    #[default]
    Public,
    Confidential,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputArtifact {
    pub path: String,
    #[serde(default = "default_role")]
    pub role: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub size_bytes: u64,
}

fn default_role() -> String {
    "input".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceHints {
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub vram_gb: Option<u32>,
    #[serde(default)]
    pub cpus: Option<u32>,
    #[serde(default)]
    pub ram_gb: Option<u32>,
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    #[serde(default)]
    pub use_msa_server: bool,
}

const fn default_timeout_s() -> u64 {
    3600
}

impl Default for ResourceHints {
    fn default() -> Self {
        Self {
            gpu: None,
            vram_gb: None,
            cpus: None,
            ram_gb: None,
            timeout_s: default_timeout_s(),
            use_msa_server: false,
        }
    }
}

/// Portable job specification emitted by CraftBio apps and consumed by `fw compute submit`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeJobSpec {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub job_id: String,
    pub name: String,
    #[serde(default)]
    pub project_ref: Option<String>,
    pub producer: String,
    #[serde(default)]
    pub runbook: Option<String>,
    #[serde(default)]
    pub entrypoint: Option<String>,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub galaxy_tool_id: Option<String>,
    #[serde(default)]
    pub galaxy_tool_version: Option<String>,
    #[serde(default)]
    pub galaxy_workflow_id: Option<String>,
    #[serde(default)]
    pub galaxy_params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub inputs: Vec<InputArtifact>,
    #[serde(default = "default_expected_outputs")]
    pub expected_outputs: Vec<String>,
    #[serde(default)]
    pub resources: ResourceHints,
    #[serde(default)]
    pub data_tier: DataTier,
    #[serde(default)]
    pub ip_tier: IpTier,
    pub created_at: String,
}

fn default_schema_version() -> String {
    SPEC_SCHEMA_VERSION.to_string()
}

fn default_expected_outputs() -> Vec<String> {
    vec!["outputs/**".to_string()]
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputArtifact {
    pub path: String,
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Provenance record written by `fw compute` into `out_dir/provenance.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeResultManifest {
    pub schema_version: String,
    pub job_id: String,
    pub name: String,
    #[serde(default)]
    pub project_ref: Option<String>,
    pub producer: String,
    pub target: String,
    pub backend: String,
    #[serde(default)]
    pub remote_job_id: Option<String>,
    pub status: String,
    pub fingerprint: String,
    pub data_tier: DataTier,
    pub ip_tier: IpTier,
    #[serde(default)]
    pub inputs: Vec<InputArtifact>,
    #[serde(default)]
    pub outputs: Vec<OutputArtifact>,
    pub recorded_at: String,
    #[serde(default)]
    pub bco_path: Option<String>,
}

/// Verified outputs returned to a CraftBio app after SHA-256 validation.
#[derive(Debug, Clone)]
pub struct VerifiedComputeResults {
    pub manifest: ComputeResultManifest,
    pub verified_files: Vec<PathBuf>,
}

/// Joins a relative path inside `root` while rejecting traversal (`..`), absolute paths, or NUL bytes.
pub fn safe_join_relative(root: &Path, rel: &str) -> Result<PathBuf, ComputeError> {
    if rel.is_empty() || rel.contains('\0') || rel.starts_with('/') || rel.starts_with('\\') {
        return Err(ComputeError::UnsafePath(rel.to_string()));
    }
    let rel_path = Path::new(rel);
    let mut clean = PathBuf::new();
    for comp in rel_path.components() {
        match comp {
            Component::Normal(seg) => clean.push(seg),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ComputeError::UnsafePath(rel.to_string()));
            }
        }
    }
    if clean.as_os_str().is_empty() {
        return Err(ComputeError::UnsafePath(rel.to_string()));
    }
    Ok(root.join(clean))
}

/// Validates schema version, mandatory timeout bounds, and confidential IP constraints.
pub fn validate_spec(spec: &ComputeJobSpec) -> Result<(), ComputeError> {
    if spec.schema_version != SPEC_SCHEMA_VERSION {
        return Err(ComputeError::UnsupportedSchema {
            found: spec.schema_version.clone(),
            expected: SPEC_SCHEMA_VERSION.to_string(),
        });
    }
    if spec.job_id.trim().is_empty() || spec.name.trim().is_empty() {
        return Err(ComputeError::InvalidInput(
            "job_id and name must be non-empty".to_string(),
        ));
    }
    if spec.resources.timeout_s < MIN_TIMEOUT_S || spec.resources.timeout_s > MAX_TIMEOUT_S {
        return Err(ComputeError::InvalidTimeout(spec.resources.timeout_s));
    }
    if spec.ip_tier == IpTier::Confidential {
        if spec.resources.use_msa_server {
            return Err(ComputeError::ConfidentialIpViolation(
                "use_msa_server=true is forbidden when ip_tier=confidential (leaks sequences to api.colabfold.com)"
                    .to_string(),
            ));
        }
        let joined = spec.command.join(" ").to_ascii_lowercase();
        if joined.contains("--use_msa_server") || joined.contains("api.colabfold.com") {
            return Err(ComputeError::ConfidentialIpViolation(
                "command references external ColabFold MSA server while ip_tier=confidential"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// File payload staged into a job bundle during `emit_job_bundle`.
#[derive(Debug, Clone)]
pub struct StagedFile {
    pub rel_path: String,
    pub role: String,
    pub contents: Vec<u8>,
    pub include_in_inputs: bool,
}

/// Writes staged files and `job_spec.json` with deterministic SHA-256 checksums.
pub fn emit_job_bundle(
    bundle_dir: &Path,
    mut spec: ComputeJobSpec,
    files: &[StagedFile],
) -> Result<PathBuf, ComputeError> {
    validate_spec(&spec)?;
    fs::create_dir_all(bundle_dir)?;
    fs::create_dir_all(bundle_dir.join("inputs"))?;

    let mut indexed_inputs = Vec::new();
    for item in files {
        let dest = safe_join_relative(bundle_dir, &item.rel_path)?;
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest, &item.contents)?;
        if item.include_in_inputs {
            indexed_inputs.push(InputArtifact {
                path: item.rel_path.clone(),
                role: item.role.clone(),
                sha256: sha256::sha256_hex(&item.contents),
                size_bytes: item.contents.len() as u64,
            });
        }
    }
    spec.inputs = indexed_inputs;

    let spec_path = bundle_dir.join("job_spec.json");
    let json = serde_json::to_string_pretty(&spec)?;
    fs::write(&spec_path, format!("{json}\n"))?;
    Ok(spec_path)
}

/// Loads `job_spec.json` from `bundle_dir` and verifies every declared input SHA-256.
pub fn load_and_verify_bundle(bundle_dir: &Path) -> Result<ComputeJobSpec, ComputeError> {
    let spec_path = bundle_dir.join("job_spec.json");
    let raw = fs::read_to_string(&spec_path)?;
    let spec: ComputeJobSpec = serde_json::from_str(&raw)?;
    validate_spec(&spec)?;

    for item in &spec.inputs {
        let fpath = safe_join_relative(bundle_dir, &item.path)?;
        if !fpath.is_file() {
            return Err(ComputeError::MissingArtifact(item.path.clone()));
        }
        if !item.sha256.is_empty() {
            let (actual, _) = sha256::sha256_file(&fpath)?;
            if actual != item.sha256 {
                return Err(ComputeError::ChecksumMismatch {
                    path: item.path.clone(),
                    expected: item.sha256.clone(),
                    actual,
                });
            }
        }
    }
    Ok(spec)
}

/// Reads `out_dir/provenance.json` written by `fw compute` and verifies SHA-256 of every output file.
pub fn ingest_result_manifest(out_dir: &Path) -> Result<VerifiedComputeResults, ComputeError> {
    let prov_path = out_dir.join("provenance.json");
    let raw = fs::read_to_string(&prov_path)?;
    let manifest: ComputeResultManifest = serde_json::from_str(&raw)?;
    if manifest.schema_version != SPEC_SCHEMA_VERSION {
        return Err(ComputeError::UnsupportedSchema {
            found: manifest.schema_version,
            expected: SPEC_SCHEMA_VERSION.to_string(),
        });
    }

    let mut verified_files = Vec::with_capacity(manifest.outputs.len());
    for out in &manifest.outputs {
        let local_candidate = safe_join_relative(out_dir, &out.name)?;
        let resolved = if local_candidate.is_file() {
            local_candidate
        } else {
            let abs = PathBuf::from(&out.path);
            if abs.is_file() {
                abs
            } else {
                return Err(ComputeError::MissingArtifact(out.name.clone()));
            }
        };
        let (actual, _) = sha256::sha256_file(&resolved)?;
        if actual != out.sha256 {
            return Err(ComputeError::ChecksumMismatch {
                path: out.name.clone(),
                expected: out.sha256.clone(),
                actual,
            });
        }
        verified_files.push(resolved);
    }

    Ok(VerifiedComputeResults {
        manifest,
        verified_files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{CoFoldBundleRequest, CoFoldEntitySpec, MpnnBundleRequest};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!("craftbio_compute_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&p);
        p
    }

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            sha256::sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256::sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn rejects_path_traversal_and_confidential_msa() {
        assert!(safe_join_relative(Path::new("/tmp"), "../etc/passwd").is_err());
        assert!(safe_join_relative(Path::new("/tmp"), "/etc/passwd").is_err());

        let req = CoFoldBundleRequest {
            job_id: "J-test-001".to_string(),
            name: "socs1-mimetic".to_string(),
            project_ref: Some("research/socs1".to_string()),
            backend: "boltz".to_string(),
            entities: vec![CoFoldEntitySpec::Protein {
                chain_id: "A".to_string(),
                sequence: "MVTPEGNVSLVDESLLV".to_string(),
            }],
            predict_affinity_binder: None,
            use_msa_server: true,
            diffusion_samples: 3,
            timeout_s: 1800,
            data_tier: DataTier::PublicOnly,
            ip_tier: IpTier::Confidential,
        };
        let built = adapters::build_dockcraft_cofold_bundle(&req);
        assert!(built.is_err());

        // Timeout bounds (F3-23)
        let mut bad_timeout = req.clone();
        bad_timeout.ip_tier = IpTier::Public;
        bad_timeout.use_msa_server = false;
        bad_timeout.timeout_s = 5;
        assert!(matches!(
            adapters::build_dockcraft_cofold_bundle(&bad_timeout),
            Err(ComputeError::InvalidTimeout(5))
        ));

        // Chai-1 backend emits FASTA manifest + chai-lab command
        let mut chai_req = bad_timeout;
        chai_req.backend = "chai".to_string();
        chai_req.timeout_s = 1800;
        let (chai_spec, chai_files) = adapters::build_dockcraft_cofold_bundle(&chai_req)
            .unwrap_or_else(|e| {
                // avoid unwrap_used/expect_used/panic lints
                unreachable_guard(&e)
            });
        assert_eq!(chai_spec.runbook.as_deref(), Some("chai1"));
        assert!(
            chai_files
                .iter()
                .any(|f| f.rel_path == "inputs/complex.fasta")
        );
    }

    fn unreachable_guard(_e: &ComputeError) -> (ComputeJobSpec, Vec<StagedFile>) {
        (
            ComputeJobSpec {
                schema_version: String::new(),
                job_id: String::new(),
                name: String::new(),
                project_ref: None,
                producer: String::new(),
                runbook: None,
                entrypoint: None,
                command: Vec::new(),
                galaxy_tool_id: None,
                galaxy_tool_version: None,
                galaxy_workflow_id: None,
                galaxy_params: BTreeMap::new(),
                inputs: Vec::new(),
                expected_outputs: Vec::new(),
                resources: ResourceHints::default(),
                data_tier: DataTier::PublicOnly,
                ip_tier: IpTier::Public,
                created_at: String::new(),
            },
            Vec::new(),
        )
    }

    #[test]
    fn emit_load_and_detect_tampered_output() -> Result<(), ComputeError> {
        let dir = temp_dir("roundtrip");
        let req = MpnnBundleRequest {
            job_id: "J-mpnn-001".to_string(),
            name: "vhh-redesign".to_string(),
            project_ref: Some("research/demo".to_string()),
            pdb_stem: "backbone".to_string(),
            pdb_bytes: b"ATOM      1  N   ALA A   1       0.000   0.000   0.000\nEND\n".to_vec(),
            design_chains: vec!["A".to_string()],
            fixed_positions: BTreeMap::from([("A".to_string(), vec![10, 11, 12])]),
            num_seq_per_target: 8,
            sampling_temps: vec![0.1, 0.2],
            use_ligand_mpnn: false,
            use_soluble_weights: true,
            timeout_s: 900,
            ip_tier: IpTier::Public,
        };
        let (spec, files) = adapters::build_biomolcraft_mpnn_bundle(&req)?;
        emit_job_bundle(&dir, spec, &files)?;

        // Verify fixed_positions.jsonl carries outer PDB-stem key (prevents silent full redesign!)
        let fixed_raw = fs::read_to_string(dir.join("inputs/fixed_positions.jsonl"))?;
        assert!(fixed_raw.contains("\"backbone\""));

        let loaded = load_and_verify_bundle(&dir)?;
        assert_eq!(loaded.runbook.as_deref(), Some("proteinmpnn-ligandmpnn"));

        // Simulate fw-compute output + provenance.json
        let out_dir = dir.join("out");
        fs::create_dir_all(&out_dir)?;
        let out_file = out_dir.join("designs.fa");
        fs::write(&out_file, b">design_1\nACDEFGHIK\n")?;
        let (digest, size) = sha256::sha256_file(&out_file)?;
        let manifest = ComputeResultManifest {
            schema_version: SPEC_SCHEMA_VERSION.to_string(),
            job_id: loaded.job_id.clone(),
            name: loaded.name.clone(),
            project_ref: loaded.project_ref.clone(),
            producer: loaded.producer.clone(),
            target: "colab-t4".to_string(),
            backend: "colab-cli".to_string(),
            remote_job_id: Some("colab-demo".to_string()),
            status: "succeeded".to_string(),
            fingerprint: "abc".to_string(),
            data_tier: DataTier::PublicOnly,
            ip_tier: IpTier::Public,
            inputs: loaded.inputs.clone(),
            outputs: vec![OutputArtifact {
                path: out_file.display().to_string(),
                name: "designs.fa".to_string(),
                sha256: digest,
                size_bytes: size,
            }],
            recorded_at: "2026-11-27T09:00:00+00:00".to_string(),
            bco_path: None,
        };
        fs::write(
            out_dir.join("provenance.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
        let verified = ingest_result_manifest(&out_dir)?;
        assert_eq!(verified.verified_files.len(), 1);

        // Tamper 1 byte -> must fail SHA-256 verification
        fs::write(&out_file, b">design_1\nACDEFGHIX\n")?;
        assert!(matches!(
            ingest_result_manifest(&out_dir),
            Err(ComputeError::ChecksumMismatch { .. })
        ));

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
