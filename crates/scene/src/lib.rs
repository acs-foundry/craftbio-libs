#![forbid(unsafe_code)]

//! `craftbio-scene` — Portable `.craftscene.json` Biomolecular Interchange, Scientific Provenance
//! Log (aligned with `craft-anything`), and **Craft Link v1** Inter-App IPC over `storytold/craft-libs`'s `craft-ipc`.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use craft_ipc::client::Client;
use craft_ipc::discovery::{Discovery, InstanceRecord, RuntimeDir};
use craft_ipc::error::LinkError;
use craft_ipc::server::{
    Activate, Activation, ControlSink, FrameOpen, Frames, Server, ServerConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Craft Link IPC error: {0}")]
    Link(#[from] LinkError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Primary 3D biomolecular representation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepresentationKind {
    Cartoon,
    NucleicLadder,
    Sticks,
    Spheres,
    Surface,
    Lines,
}

impl RepresentationKind {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cartoon => "Cartoon / Ribbon",
            Self::NucleicLadder => "RNA / DNA Ladder",
            Self::Sticks => "Sticks",
            Self::Spheres => "Spheres (VDW)",
            Self::Surface => "Molecular Surface",
            Self::Lines => "Lines",
        }
    }
}

/// Biomolecular color scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    Chain,
    SecondaryStructure,
    ElementCpk,
    Nucleotide,
    BfactorPlddt,
    Sap3D,
    ElectrostaticEsp,
    CustomHighlight,
}

impl ColorScheme {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Chain => "By Chain",
            Self::SecondaryStructure => "Secondary Structure (DSSP)",
            Self::ElementCpk => "Element (CPK)",
            Self::Nucleotide => "Nucleotide (A/U/G/C)",
            Self::BfactorPlddt => "pLDDT / B-Factor",
            Self::Sap3D => "3D SAP Hydrophobicity",
            Self::ElectrostaticEsp => "Electrostatic Potential (ESP)",
            Self::CustomHighlight => "Module Highlights",
        }
    }
}

/// Annotated 3D sphere overlay (Ala-Scan hotspot, HYDE corona, ΔMIF, catalytic site, clash disc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphereOverlay {
    pub label: String,
    pub category: String,
    pub coords: [f64; 3],
    pub radius_a: f64,
    pub rgba: [f32; 4],
}

/// Annotated 3D line/segment overlay (H-bond, salt bridge, RNA base-pair, distance measurement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineOverlay {
    pub label: String,
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub dashed: bool,
    pub rgba: [f32; 4],
}

/// Scientific epistemic provenance category (aligned with `craft-anything` evidence discipline).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceKind {
    ExperimentalStructure,
    ComputationalPrediction,
    MolecularDynamics,
    InteractiveEdit,
}

/// Immutable audit record of an action or data source attached to a biomolecular scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    pub kind: ProvenanceKind,
    pub source_app: String,
    pub summary: String,
}

/// Portable CraftBio scene snapshot exchanged between `BiomolCraft`, `BioCraft`, `DockCraft`,
/// `ChemCraft`, `AdmetCraft`, and ArtCraft apps (`.craftscene.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftSceneDocument {
    pub schema_version: String,
    pub title: String,
    pub source_app: String,
    pub pdb_path: Option<String>,
    pub inline_pdb: Option<String>,
    pub representation: RepresentationKind,
    pub color_scheme: ColorScheme,
    pub named_selections: HashMap<String, String>,
    pub residue_highlights: Vec<(char, i32, [f32; 3], String)>,
    pub sphere_overlays: Vec<SphereOverlay>,
    pub line_overlays: Vec<LineOverlay>,
    pub active_modules: Vec<String>,
    pub provenance: Vec<ProvenanceEntry>,
}

impl Default for CraftSceneDocument {
    fn default() -> Self {
        Self {
            schema_version: "craftscene/v1".to_string(),
            title: "Untitled Biomolecular Scene".to_string(),
            source_app: "biomolcraft".to_string(),
            pdb_path: None,
            inline_pdb: None,
            representation: RepresentationKind::Cartoon,
            color_scheme: ColorScheme::Chain,
            named_selections: HashMap::new(),
            residue_highlights: Vec::new(),
            sphere_overlays: Vec::new(),
            line_overlays: Vec::new(),
            active_modules: vec!["antibodies".to_string(), "ncrna".to_string()],
            provenance: Vec::new(),
        }
    }
}

/// Inbound Craft Link command received from a peer CraftBio / ArtCraft application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InboundLinkCommand {
    OpenPdb { path: String, source_app: String },
    LoadScene(Box<CraftSceneDocument>),
    RunDsl { command: String, source_app: String },
}

struct QueueSink {
    app_name: String,
    inbox: Arc<Mutex<Vec<InboundLinkCommand>>>,
}

impl ControlSink for QueueSink {
    fn submit(&self, method: String, params: Value) -> Option<Receiver<Value>> {
        let (tx, rx) = mpsc::channel();
        let reply = match method.as_str() {
            "biomol.ping" | "ping" => json!({
                "ok": true,
                "result": { "app": self.app_name, "protocol": "craftbio-link/v1" }
            }),
            "biomol.open_pdb" => {
                if let Some(path) = params.get("path").and_then(Value::as_str) {
                    let src = params
                        .get("source_app")
                        .and_then(Value::as_str)
                        .unwrap_or("peer")
                        .to_string();
                    if let Ok(mut q) = self.inbox.lock() {
                        q.push(InboundLinkCommand::OpenPdb {
                            path: path.to_string(),
                            source_app: src,
                        });
                    }
                    json!({ "ok": true, "result": { "queued": true, "path": path } })
                } else {
                    json!({ "ok": false, "error": "missing 'path' parameter" })
                }
            }
            "biomol.load_scene" => match serde_json::from_value::<CraftSceneDocument>(params) {
                Ok(doc) => {
                    let title = doc.title.clone();
                    if let Ok(mut q) = self.inbox.lock() {
                        q.push(InboundLinkCommand::LoadScene(Box::new(doc)));
                    }
                    json!({ "ok": true, "result": { "queued": true, "title": title } })
                }
                Err(e) => {
                    json!({ "ok": false, "error": format!("invalid CraftSceneDocument: {e}") })
                }
            },
            "biomol.run_dsl" => {
                if let Some(cmd) = params.get("command").and_then(Value::as_str) {
                    let src = params
                        .get("source_app")
                        .and_then(Value::as_str)
                        .unwrap_or("peer")
                        .to_string();
                    if let Ok(mut q) = self.inbox.lock() {
                        q.push(InboundLinkCommand::RunDsl {
                            command: cmd.to_string(),
                            source_app: src,
                        });
                    }
                    json!({ "ok": true, "result": { "queued": true, "command": cmd } })
                } else {
                    json!({ "ok": false, "error": "missing 'command' parameter" })
                }
            }
            other => {
                json!({ "ok": false, "error": format!("unsupported CraftBio link method '{other}'") })
            }
        };
        let _ = tx.send(reply);
        Some(rx)
    }
}

struct NoopActivate;
impl Activate for NoopActivate {
    fn activate(
        &self,
        _token: Option<String>,
        _from_pid: Option<u32>,
        _open: Option<Value>,
    ) -> Activation {
        Activation::Requested
    }
}

struct NoopFrames;
impl Frames for NoopFrames {
    fn open(&self, _params: &Map<String, Value>) -> Result<FrameOpen, LinkError> {
        Err(LinkError::Invalid(
            "bulk frames not active on control endpoint".to_string(),
        ))
    }

    fn render(&self, _ticket: &str, _header: &Value) -> Result<(Value, Vec<u8>), LinkError> {
        Err(LinkError::Invalid(
            "bulk frames not active on control endpoint".to_string(),
        ))
    }

    fn close_ticket(&self, _ticket: &str) {}
}

/// Running Craft Link v1 server for a CraftBio application (uses `storytold/craft-libs::craft-ipc`).
pub struct CraftBioLinkServer {
    server: Server,
    inbox: Arc<Mutex<Vec<InboundLinkCommand>>>,
}

impl CraftBioLinkServer {
    /// Start an authenticated local Unix-socket Craft Link v1 server (mode `0600` inside `0700` dir).
    pub fn start(
        app: &str,
        app_version: &str,
        runtime: Option<RuntimeDir>,
    ) -> Result<Self, SceneError> {
        let inbox = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::new(QueueSink {
            app_name: app.to_string(),
            inbox: Arc::clone(&inbox),
        });
        let server = Server::start(ServerConfig {
            app: app.to_string(),
            app_version: app_version.to_string(),
            instance: None,
            runtime,
            tcp_port: None,
            sink,
            activate: Arc::new(NoopActivate),
            frames: Arc::new(NoopFrames),
        })?;
        Ok(Self { server, inbox })
    }

    #[must_use]
    pub fn record(&self) -> &InstanceRecord {
        self.server.record()
    }

    /// Drain all queued inbound commands sent by peer CraftBio / ArtCraft apps.
    #[must_use]
    pub fn drain_inbox(&self) -> Vec<InboundLinkCommand> {
        if let Ok(mut q) = self.inbox.lock() {
            std::mem::take(&mut *q)
        } else {
            Vec::new()
        }
    }

    pub fn shutdown(self) {
        let _ = self.server.shutdown();
    }
}

/// Discover live CraftBio / ArtCraft app instances on this machine using `craft-ipc`.
pub fn discover_craft_instances(
    runtime: Option<RuntimeDir>,
    app_filter: Option<&str>,
) -> Result<Vec<InstanceRecord>, SceneError> {
    let dir = match runtime {
        Some(r) => r,
        None => RuntimeDir::resolve()?,
    };
    let mut records = Discovery::scan(&dir)?;
    if let Some(app) = app_filter {
        records.retain(|r| r.app == app);
    }
    Ok(records)
}

/// Send a `CraftSceneDocument` to a running `BiomolCraft` (or sibling CraftBio) instance via Craft Link v1.
pub fn push_scene_via_craft_link(
    target: &InstanceRecord,
    scene: &CraftSceneDocument,
) -> Result<Value, SceneError> {
    let mut client = Client::connect(target)?;
    let params = serde_json::to_value(scene)?;
    let reply = client.call("biomol.load_scene", params)?;
    Ok(reply)
}

/// Send a DSL command string to a running `BiomolCraft` instance via Craft Link v1.
pub fn send_dsl_via_craft_link(
    target: &InstanceRecord,
    source_app: &str,
    command: &str,
) -> Result<Value, SceneError> {
    let mut client = Client::connect(target)?;
    let reply = client.call(
        "biomol.run_dsl",
        json!({ "source_app": source_app, "command": command }),
    )?;
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn craft_link_v1_roundtrips_biomolecular_scene_over_unix_socket()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir_path = std::env::temp_dir().join(format!("craftbio_ipc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir_path);
        std::fs::create_dir_all(&dir_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir_path, std::fs::Permissions::from_mode(0o700))?;
        }

        let runtime = RuntimeDir::at(dir_path.clone())?;
        let srv = CraftBioLinkServer::start("biomolcraft", "0.1.0", Some(runtime.clone()))?;
        let found = discover_craft_instances(Some(runtime), Some("biomolcraft"))?;
        assert!(
            !found.is_empty(),
            "expected to discover live biomolcraft instance"
        );

        let doc = CraftSceneDocument {
            title: "Trastuzumab-HER2 Live Sync".to_string(),
            source_app: "biocraft".to_string(),
            ..Default::default()
        };
        let reply = push_scene_via_craft_link(&found[0], &doc)?;
        assert_eq!(reply.get("queued").and_then(Value::as_bool), Some(true));

        let cmds = srv.drain_inbox();
        assert_eq!(cmds.len(), 1);
        srv.shutdown();
        let _ = std::fs::remove_dir_all(&dir_path);
        Ok(())
    }
}
