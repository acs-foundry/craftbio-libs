#![forbid(unsafe_code)]

//! Minimal CLI harness for `craftbio-compute` enabling cross-language Rust ↔ Python
//! verification against ResearchFW `fw compute`.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use craftbio_compute::adapters::{
    AntiSmashBundleRequest, CoFoldBundleRequest, CoFoldEntitySpec, build_dockcraft_cofold_bundle,
    build_genecraft_antismash_bundle,
};
use craftbio_compute::{
    DataTier, IpTier, emit_job_bundle, ingest_result_manifest, load_and_verify_bundle,
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(cmd) = args.get(1).map(String::as_str) else {
        eprintln!(
            "usage: craft-compute-cli <emit-cofold|emit-antismash|verify-bundle|ingest-results> <dir> [--job-id ID]"
        );
        return ExitCode::from(2);
    };
    let Some(dir_arg) = args.get(2) else {
        eprintln!("missing <dir> argument");
        return ExitCode::from(2);
    };
    let dir = PathBuf::from(dir_arg);
    let job_id = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--job-id"))
        .and_then(|w| w.get(1))
        .cloned()
        .unwrap_or_else(|| "J-rust-demo-001".to_string());

    let res: Result<String, Box<dyn std::error::Error>> = match cmd {
        "emit-cofold" => {
            let req = CoFoldBundleRequest {
                job_id,
                name: "dockcraft-boltz2-demo".to_string(),
                project_ref: Some("research/demo".to_string()),
                backend: "boltz".to_string(),
                entities: vec![
                    CoFoldEntitySpec::Protein {
                        chain_id: "A".to_string(),
                        sequence: "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ".to_string(),
                    },
                    CoFoldEntitySpec::LigandSmiles {
                        chain_id: "L".to_string(),
                        smiles: "CC(=O)Oc1ccccc1C(=O)O".to_string(),
                    },
                ],
                predict_affinity_binder: Some("L".to_string()),
                use_msa_server: false,
                diffusion_samples: 3,
                timeout_s: 1800,
                data_tier: DataTier::PublicOnly,
                ip_tier: IpTier::Public,
            };
            let (spec, files) = match build_dockcraft_cofold_bundle(&req) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(1);
                }
            };
            emit_job_bundle(&dir, spec, &files)
                .map(|p| {
                    format!(
                        "{{\"emitted\":{}}}",
                        serde_json::to_string(&p.display().to_string()).unwrap_or_default()
                    )
                })
                .map_err(Into::into)
        }
        "emit-antismash" => {
            let req = AntiSmashBundleRequest {
                job_id,
                name: "genecraft-nrps-antismash".to_string(),
                project_ref: Some("research/demo".to_string()),
                genbank_filename: "nrps_cluster.gbk".to_string(),
                genbank_bytes: b"LOCUS DEMO_NRPS 100 bp DNA\nORIGIN\n  1 atgcatgcat\n//\n".to_vec(),
                timeout_s: 3600,
            };
            let (spec, files) = match build_genecraft_antismash_bundle(&req) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(1);
                }
            };
            emit_job_bundle(&dir, spec, &files)
                .map(|p| {
                    format!(
                        "{{\"emitted\":{}}}",
                        serde_json::to_string(&p.display().to_string()).unwrap_or_default()
                    )
                })
                .map_err(Into::into)
        }
        "verify-bundle" => load_and_verify_bundle(&dir)
            .and_then(|s| Ok(serde_json::to_string_pretty(&s)?))
            .map_err(Into::into),
        "ingest-results" => ingest_result_manifest(&dir)
            .and_then(|v| Ok(serde_json::to_string_pretty(&v.manifest)?))
            .map_err(Into::into),
        other => {
            eprintln!("unknown subcommand: {other}");
            return ExitCode::from(2);
        }
    };

    match res {
        Ok(out) => {
            println!("{out}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(1)
        }
    }
}
