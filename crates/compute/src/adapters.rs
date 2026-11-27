//! High-level bundle builders translating CraftBio domain configs (`DockCraft`, `BiomolCraft`, `GeneCraft`)
//! into validated `ComputeJobSpec` packages ready for `fw compute submit`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::{
    ComputeError, ComputeJobSpec, DataTier, IpTier, ResourceHints, SPEC_SCHEMA_VERSION, StagedFile,
    validate_spec,
};

/// Single molecular entity for Boltz-2 / Chai-1 co-folding (mirrors `DockCraft::CoFoldEntity`).
#[derive(Debug, Clone)]
pub enum CoFoldEntitySpec {
    Protein { chain_id: String, sequence: String },
    LigandSmiles { chain_id: String, smiles: String },
    IonOrCcd { chain_id: String, ccd_code: String },
}

#[derive(Debug, Clone)]
pub struct CoFoldBundleRequest {
    pub job_id: String,
    pub name: String,
    pub project_ref: Option<String>,
    pub backend: String, // "boltz" | "chai"
    pub entities: Vec<CoFoldEntitySpec>,
    pub predict_affinity_binder: Option<String>,
    pub use_msa_server: bool,
    pub diffusion_samples: usize,
    pub timeout_s: u64,
    pub data_tier: DataTier,
    pub ip_tier: IpTier,
}

fn sanitize_single_line(raw: &str) -> Result<String, ComputeError> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.contains('\n')
        || trimmed.contains('\r')
        || trimmed.contains('\0')
    {
        return Err(ComputeError::InvalidInput(
            "entity sequence/chain/smiles must be non-empty single-line text".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

/// Builds a validated Boltz-2 / Chai-1 bundle for `DockCraft` (`--emit-job-bundle`).
pub fn build_dockcraft_cofold_bundle(
    req: &CoFoldBundleRequest,
) -> Result<(ComputeJobSpec, Vec<StagedFile>), ComputeError> {
    if req.entities.is_empty() {
        return Err(ComputeError::InvalidInput(
            "co-fold bundle requires at least one entity".to_string(),
        ));
    }
    let mut yaml = String::from("version: 1\nsequences:\n");
    for ent in &req.entities {
        match ent {
            CoFoldEntitySpec::Protein { chain_id, sequence } => {
                let cid = sanitize_single_line(chain_id)?;
                let seq = sanitize_single_line(sequence)?;
                let _ = writeln!(yaml, "  - protein:\n      id: {cid}\n      sequence: {seq}");
            }
            CoFoldEntitySpec::LigandSmiles { chain_id, smiles } => {
                let cid = sanitize_single_line(chain_id)?;
                let smi = sanitize_single_line(smiles)?;
                let escaped = smi.replace('\'', "''");
                let _ = writeln!(
                    yaml,
                    "  - ligand:\n      id: {cid}\n      smiles: '{escaped}'"
                );
            }
            CoFoldEntitySpec::IonOrCcd { chain_id, ccd_code } => {
                let cid = sanitize_single_line(chain_id)?;
                let ccd = sanitize_single_line(ccd_code)?;
                let _ = writeln!(yaml, "  - ligand:\n      id: {cid}\n      ccd: {ccd}");
            }
        }
    }
    if let Some(binder) = &req.predict_affinity_binder {
        let cid = sanitize_single_line(binder)?;
        let _ = writeln!(yaml, "properties:\n  - affinity:\n      binder: {cid}");
    }

    let is_chai = req.backend.eq_ignore_ascii_case("chai");
    let runbook = if is_chai { "chai1" } else { "boltz2" };
    let samples = req.diffusion_samples.clamp(1, 25);

    let (manifest_rel, manifest_bytes, runner_script, command) = if is_chai {
        let mut fasta = String::new();
        for ent in &req.entities {
            match ent {
                CoFoldEntitySpec::Protein { chain_id, sequence } => {
                    let cid = sanitize_single_line(chain_id)?;
                    let seq = sanitize_single_line(sequence)?;
                    let _ = writeln!(fasta, ">protein|name=chain_{cid}\n{seq}");
                }
                CoFoldEntitySpec::LigandSmiles { chain_id, smiles } => {
                    let cid = sanitize_single_line(chain_id)?;
                    let smi = sanitize_single_line(smiles)?;
                    let _ = writeln!(fasta, ">ligand|name=lig_{cid}\n{smi}");
                }
                CoFoldEntitySpec::IonOrCcd { chain_id, ccd_code } => {
                    let cid = sanitize_single_line(chain_id)?;
                    let ccd = sanitize_single_line(ccd_code)?;
                    let _ = writeln!(fasta, ">ligand|name=ion_{cid}\n{ccd}");
                }
            }
        }
        let script = "import importlib.util, shutil, subprocess, sys\n\
from pathlib import Path\n\
out = Path('/content/outputs')\n\
out.mkdir(parents=True, exist_ok=True)\n\
if shutil.which('chai-lab') is None:\n\
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'chai_lab'])\n\
cmd = ['chai-lab', 'fold', '/content/inputs/complex.fasta', str(out)]\n\
sys.exit(subprocess.call(cmd))\n"
            .to_string();
        let cmd = vec![
            "chai-lab".to_string(),
            "fold".to_string(),
            "inputs/complex.fasta".to_string(),
        ];
        (
            "inputs/complex.fasta".to_string(),
            fasta.into_bytes(),
            script,
            cmd,
        )
    } else {
        let msa_flag = if req.use_msa_server {
            "--use_msa_server"
        } else {
            ""
        };
        let script = format!(
            "import importlib.util, json, shutil, subprocess, sys\n\
from pathlib import Path\n\
work = Path('/content/boltz_work')\n\
out = Path('/content/outputs')\n\
work.mkdir(parents=True, exist_ok=True)\n\
out.mkdir(parents=True, exist_ok=True)\n\
if shutil.which('boltz') is None:\n\
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'boltz'])\n\
cmd = ['boltz', 'predict', '/content/inputs/complex.yaml', '--out_dir', str(work), '--diffusion_samples', '{samples}']\n\
if '{msa_flag}':\n\
    cmd.append('{msa_flag}')\n\
if importlib.util.find_spec('cuequivariance_ops_torch') is None:\n\
    cmd.append('--no_kernels')\n\
rc = subprocess.call(cmd)\n\
copied = []\n\
for p in sorted(work.rglob('*')):\n\
    if not p.is_file():\n\
        continue\n\
    low = p.name.lower()\n\
    if low.endswith(('.cif', '.pdb')) or ((low.startswith('confidence') or low.startswith('affinity')) and low.endswith('.json')):\n\
        dst = out / p.name\n\
        shutil.copy2(p, dst)\n\
        copied.append(p.name)\n\
(out / 'cofold_run_summary.json').write_text(json.dumps({{'rc': rc, 'cmd': cmd, 'artifacts': copied}}, indent=2) + '\\n', encoding='utf-8')\n\
sys.exit(rc)\n"
        );
        let mut cmd = vec![
            "boltz".to_string(),
            "predict".to_string(),
            "inputs/complex.yaml".to_string(),
            "--diffusion_samples".to_string(),
            samples.to_string(),
        ];
        if req.use_msa_server {
            cmd.push("--use_msa_server".to_string());
        }
        (
            "inputs/complex.yaml".to_string(),
            yaml.into_bytes(),
            script,
            cmd,
        )
    };

    let spec = ComputeJobSpec {
        schema_version: SPEC_SCHEMA_VERSION.to_string(),
        job_id: req.job_id.clone(),
        name: req.name.clone(),
        project_ref: req.project_ref.clone(),
        producer: "dockcraft".to_string(),
        runbook: Some(runbook.to_string()),
        entrypoint: Some("run_cofold.py".to_string()),
        command,
        galaxy_tool_id: None,
        galaxy_tool_version: None,
        galaxy_workflow_id: None,
        galaxy_params: BTreeMap::new(),
        inputs: Vec::new(),
        expected_outputs: vec![
            "outputs/**/*.cif".to_string(),
            "outputs/**/*.json".to_string(),
        ],
        resources: ResourceHints {
            gpu: Some("L4".to_string()),
            vram_gb: Some(24),
            cpus: Some(4),
            ram_gb: Some(16),
            timeout_s: req.timeout_s,
            use_msa_server: req.use_msa_server,
        },
        data_tier: req.data_tier,
        ip_tier: req.ip_tier,
        created_at: "2026-11-27T00:00:00+00:00".to_string(),
    };
    validate_spec(&spec)?;

    let files = vec![
        StagedFile {
            rel_path: manifest_rel,
            role: "cofold_manifest".to_string(),
            contents: manifest_bytes,
            include_in_inputs: true,
        },
        StagedFile {
            rel_path: "run_cofold.py".to_string(),
            role: "script".to_string(),
            contents: runner_script.into_bytes(),
            include_in_inputs: false,
        },
    ];
    Ok((spec, files))
}

#[derive(Debug, Clone)]
pub struct MpnnBundleRequest {
    pub job_id: String,
    pub name: String,
    pub project_ref: Option<String>,
    pub pdb_stem: String,
    pub pdb_bytes: Vec<u8>,
    pub design_chains: Vec<String>,
    pub fixed_positions: BTreeMap<String, Vec<u32>>,
    pub num_seq_per_target: usize,
    pub sampling_temps: Vec<f64>,
    pub use_ligand_mpnn: bool,
    pub use_soluble_weights: bool,
    pub timeout_s: u64,
    pub ip_tier: IpTier,
}

/// Builds a ProteinMPNN / LigandMPNN / SolubleMPNN bundle encoding the exact silent gotchas
/// (outer PDB-stem JSONL key for fixed positions + space-separated sampling_temp & chain list).
pub fn build_biomolcraft_mpnn_bundle(
    req: &MpnnBundleRequest,
) -> Result<(ComputeJobSpec, Vec<StagedFile>), ComputeError> {
    let stem = sanitize_single_line(&req.pdb_stem)?;
    if req.pdb_bytes.is_empty() || req.design_chains.is_empty() {
        return Err(ComputeError::InvalidInput(
            "MPNN bundle requires non-empty PDB bytes and at least one design chain".to_string(),
        ));
    }
    let chains_arg = req.design_chains.join(" ");
    let temps_arg = if req.sampling_temps.is_empty() {
        "0.1".to_string()
    } else {
        req.sampling_temps
            .iter()
            .map(|t| format!("{t:.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    };

    // CRITICAL GOTCHA ENCODED: outer key MUST be the PDB stem (`{stem: {chain: [res...]}}`),
    // otherwise ProteinMPNN silently ignores fixed positions and redesigns every residue!
    let mut outer = BTreeMap::new();
    outer.insert(stem.clone(), req.fixed_positions.clone());
    let fixed_jsonl = format!("{}\n", serde_json::to_string(&outer)?);

    let mpnn_entry = if req.use_ligand_mpnn {
        "run.py"
    } else {
        "protein_mpnn_run.py"
    };
    let soluble_flag = if req.use_soluble_weights {
        " --use_soluble_model"
    } else {
        ""
    };
    let runner_script = format!(
        "import subprocess, sys\n\
         from pathlib import Path\n\
         out = Path('/content/outputs')\n\
         out.mkdir(parents=True, exist_ok=True)\n\
         cmd = ['python3', '{mpnn_entry}', '--pdb_path', '/content/inputs/{stem}.pdb', \
                '--pdb_path_chains', '{chains_arg}', '--fixed_positions_jsonl', '/content/inputs/fixed_positions.jsonl', \
                '--out_folder', str(out), '--num_seq_per_target', '{nseq}', '--sampling_temp', '{temps_arg}']\n\
         if '{soluble_flag}'.strip():\n\
             cmd.append('--use_soluble_model')\n\
         sys.exit(subprocess.call(cmd))\n",
        nseq = req.num_seq_per_target.clamp(1, 256),
    );

    let spec = ComputeJobSpec {
        schema_version: SPEC_SCHEMA_VERSION.to_string(),
        job_id: req.job_id.clone(),
        name: req.name.clone(),
        project_ref: req.project_ref.clone(),
        producer: "biomolcraft".to_string(),
        runbook: Some("proteinmpnn-ligandmpnn".to_string()),
        entrypoint: Some("run_mpnn.py".to_string()),
        command: vec![
            "python3".to_string(),
            "run_mpnn.py".to_string(),
            "--sampling_temp".to_string(),
            temps_arg,
        ],
        galaxy_tool_id: None,
        galaxy_tool_version: None,
        galaxy_workflow_id: None,
        galaxy_params: BTreeMap::new(),
        inputs: Vec::new(),
        expected_outputs: vec!["outputs/seqs/*.fa".to_string()],
        resources: ResourceHints {
            gpu: Some("T4".to_string()),
            vram_gb: Some(8),
            cpus: Some(4),
            ram_gb: Some(12),
            timeout_s: req.timeout_s,
            use_msa_server: false,
        },
        data_tier: DataTier::PublicOnly,
        ip_tier: req.ip_tier,
        created_at: "2026-11-27T00:00:00+00:00".to_string(),
    };
    validate_spec(&spec)?;

    let files = vec![
        StagedFile {
            rel_path: format!("inputs/{stem}.pdb"),
            role: "backbone_pdb".to_string(),
            contents: req.pdb_bytes.clone(),
            include_in_inputs: true,
        },
        StagedFile {
            rel_path: "inputs/fixed_positions.jsonl".to_string(),
            role: "fixed_positions".to_string(),
            contents: fixed_jsonl.into_bytes(),
            include_in_inputs: true,
        },
        StagedFile {
            rel_path: "run_mpnn.py".to_string(),
            role: "script".to_string(),
            contents: runner_script.into_bytes(),
            include_in_inputs: false,
        },
    ];
    Ok((spec, files))
}

#[derive(Debug, Clone)]
pub struct AntiSmashBundleRequest {
    pub job_id: String,
    pub name: String,
    pub project_ref: Option<String>,
    pub genbank_filename: String,
    pub genbank_bytes: Vec<u8>,
    pub timeout_s: u64,
}

/// Builds a `usegalaxy.eu` antiSMASH BGC annotation bundle for `GeneCraft` NRPS clusters.
pub fn build_genecraft_antismash_bundle(
    req: &AntiSmashBundleRequest,
) -> Result<(ComputeJobSpec, Vec<StagedFile>), ComputeError> {
    let fname = sanitize_single_line(&req.genbank_filename)?;
    if req.genbank_bytes.is_empty() {
        return Err(ComputeError::InvalidInput(
            "antiSMASH bundle requires non-empty GenBank/FASTA bytes".to_string(),
        ));
    }
    let spec = ComputeJobSpec {
        schema_version: SPEC_SCHEMA_VERSION.to_string(),
        job_id: req.job_id.clone(),
        name: req.name.clone(),
        project_ref: req.project_ref.clone(),
        producer: "genecraft".to_string(),
        runbook: Some("galaxy-antismash".to_string()),
        entrypoint: None,
        command: Vec::new(),
        galaxy_tool_id: Some(
            "toolshed.g2.bx.psu.edu/repos/bgruening/antismash/antismash/6.1.1+galaxy1".to_string(),
        ),
        galaxy_tool_version: Some("6.1.1+galaxy1".to_string()),
        galaxy_workflow_id: None,
        galaxy_params: BTreeMap::from([(
            "infile".to_string(),
            serde_json::Value::String(format!("inputs/{fname}")),
        )]),
        inputs: Vec::new(),
        expected_outputs: vec!["outputs/**".to_string()],
        resources: ResourceHints {
            gpu: None,
            vram_gb: None,
            cpus: Some(4),
            ram_gb: Some(8),
            timeout_s: req.timeout_s,
            use_msa_server: false,
        },
        data_tier: DataTier::PublicOnly,
        ip_tier: IpTier::Public,
        created_at: "2026-11-27T00:00:00+00:00".to_string(),
    };
    validate_spec(&spec)?;

    let files = vec![StagedFile {
        rel_path: format!("inputs/{fname}"),
        role: "cluster_genbank".to_string(),
        contents: req.genbank_bytes.clone(),
        include_in_inputs: true,
    }];
    Ok((spec, files))
}
