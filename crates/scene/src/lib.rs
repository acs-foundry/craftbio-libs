#![forbid(unsafe_code)]

//! `craftbio-scene` — Portable `.craftscene.json` Biomolecular Interchange, Scientific Provenance
//! Log (aligned with `craft-anything`), and **Craft Link v1** Inter-App IPC over `storytold/craft-libs`'s `craft-ipc`.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, RwLock};

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
    Putty,
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
            Self::Putty => "B-Factor / Attribute Putty",
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
    LipophilicityMlp,
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
            Self::LipophilicityMlp => "Lipophilicity (MLP)",
            Self::CustomHighlight => "Module Highlights",
        }
    }
}

/// ChimeraX / PyMOL studio lighting preset shared across GPU & headless 4K renderers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LightingPreset {
    #[default]
    Soft,
    Flat,
    Full,
    Studio,
    PublicationWhite,
}

impl LightingPreset {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Soft => "ChimeraX Soft (SSAO)",
            Self::Flat => "Flat Diagram",
            Self::Full => "Full Specular",
            Self::Studio => "Dark Studio Rim",
            Self::PublicationWhite => "PyMOL Pub White",
        }
    }

    #[must_use]
    pub fn shader_id(self) -> f32 {
        match self {
            Self::Soft => 0.0,
            Self::Flat => 1.0,
            Self::Full => 2.0,
            Self::Studio => 3.0,
            Self::PublicationWhite => 4.0,
        }
    }
}

/// 3D Docking GridBox exchanged bidirectionally between `BiomolCraft` and `DockCraft`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridBox3D {
    pub center: [f64; 3],
    pub size: [f64; 3],
    pub spacing: f64,
}

/// Synchronous selection-to-GridBox snapshot served by `BiomolCraft` over `biomol.get_selection_box`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionBoxSnapshot {
    pub grid_box: GridBox3D,
    pub padding_a: f64,
    pub source_model: String,
    pub selected_residues: Vec<String>,
    pub heavy_atoms: usize,
}

/// Per-atom SeeSAR HYDE v2 free-energy corona for `ViewDockX` 3D visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydeAtomCorona {
    pub atom_index: usize,
    pub atom_name: String,
    pub element: String,
    pub coords: [f64; 3],
    pub delta_g_kcal: f64,
    pub corona_radius_a: f32,
    pub corona_rgba: [f32; 4],
    pub diagnostic: String,
}

/// Rotatable bond torsion traffic-light overlay (`Green` / `Yellow` / `Red`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorsionBondLight {
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub status: String,
    pub pattern: String,
    pub dihedral_deg: f64,
    pub strain_kcal: f64,
    pub rgba: [f32; 4],
}

/// Single docked pose entry with scores, HYDE coronas, torsion lights, and 3D interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockPoseEntry {
    pub pose_id: usize,
    pub name: String,
    pub canonical_smiles: Option<String>,
    pub sdf_block: String,
    pub vina_kcal: f64,
    pub hyde_kcal: Option<f64>,
    pub consensus_kcal: Option<f64>,
    pub cnn_score: Option<f64>,
    pub estimated_ki_nm: Option<f64>,
    pub ligand_efficiency: Option<f64>,
    pub mpo_score: Option<f64>,
    pub posebusters_valid: Option<bool>,
    pub posebusters_summary: Option<String>,
    /// `[green, yellow, red]` torsion counts.
    pub torsion_counts: Option<[usize; 3]>,
    #[serde(default)]
    pub hyde_coronas: Vec<HydeAtomCorona>,
    #[serde(default)]
    pub torsion_lights: Vec<TorsionBondLight>,
    #[serde(default)]
    pub interaction_lines: Vec<LineOverlay>,
}

/// Complete rich docking & SBDD payload exchanged between `DockCraft` and `BiomolCraft` (`ViewDockX`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockCraftSceneBundle {
    pub schema_version: String,
    pub title: String,
    pub source_app: String,
    pub receptor_name: String,
    pub receptor_pdb_path: Option<String>,
    pub receptor_pdb_inline: Option<String>,
    pub grid_box: Option<GridBox3D>,
    #[serde(default)]
    pub poses: Vec<DockPoseEntry>,
    #[serde(default)]
    pub pharmacophore_spheres: Vec<SphereOverlay>,
    #[serde(default)]
    pub pharmacophore_vectors: Vec<LineOverlay>,
    #[serde(default)]
    pub selectivity_dmif_hotspots: Vec<SphereOverlay>,
    #[serde(default)]
    pub bridging_waters: Vec<SphereOverlay>,
    /// Optional per-residue scalar override `(chain, res_seq, value, metric_label)` e.g. MD RMSF or MM/GBSA.
    #[serde(default)]
    pub residue_metric_overrides: Vec<(char, i32, f64, String)>,
    pub trajectory_dcd_path: Option<String>,
    #[serde(default)]
    pub provenance: Vec<ProvenanceEntry>,
}

impl Default for DockCraftSceneBundle {
    fn default() -> Self {
        Self {
            schema_version: "dockcraft-bundle/v1".to_string(),
            title: "DockCraft ViewDockX Session".to_string(),
            source_app: "dockcraft".to_string(),
            receptor_name: "receptor".to_string(),
            receptor_pdb_path: None,
            receptor_pdb_inline: None,
            grid_box: None,
            poses: Vec::new(),
            pharmacophore_spheres: Vec::new(),
            pharmacophore_vectors: Vec::new(),
            selectivity_dmif_hotspots: Vec::new(),
            bridging_waters: Vec::new(),
            residue_metric_overrides: Vec::new(),
            trajectory_dcd_path: None,
            provenance: Vec::new(),
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

/// 3D world-space text annotation projected cleanly onto the GUI & burned into 4K PNGs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label3D {
    pub text: String,
    pub coords: [f64; 3],
    pub rgba: [f32; 4],
    pub font_scale: f32,
}

/// 2D screen-space annotation at normalized `[0..1, 0..1]` coordinates (ChimeraX `2dlabels`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label2D {
    pub id: String,
    pub text: String,
    pub norm_pos: [f32; 2],
    pub rgba: [f32; 4],
    pub font_scale: f32,
}

/// Publication Color Key gradient legend bar (ChimeraX `key`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorKeyLegend {
    pub title: String,
    pub min_label: String,
    pub max_label: String,
    pub stops: Vec<[f32; 3]>,
}

/// Calibrated Ångström Scale Bar specification (ChimeraX `scalebar`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleBarSpec {
    pub length_a: f32,
    pub label: String,
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
    OpenPdb {
        path: String,
        source_app: String,
        module_hint: Option<String>,
    },
    LoadTrajectory {
        pdb_path: String,
        dcd_path: String,
        source_app: String,
    },
    LoadScene(Box<CraftSceneDocument>),
    LoadDockCraftBundle(Box<DockCraftSceneBundle>),
    RunDsl {
        command: String,
        source_app: String,
    },
}

struct QueueSink {
    app_name: String,
    inbox: Arc<Mutex<Vec<InboundLinkCommand>>>,
    selection_box: Arc<RwLock<Option<SelectionBoxSnapshot>>>,
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
                    let module_hint = params
                        .get("module_hint")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    if let Ok(mut q) = self.inbox.lock() {
                        q.push(InboundLinkCommand::OpenPdb {
                            path: path.to_string(),
                            source_app: src,
                            module_hint,
                        });
                    }
                    json!({ "ok": true, "result": { "queued": true, "path": path } })
                } else {
                    json!({ "ok": false, "error": "missing 'path' parameter" })
                }
            }
            "biomol.load_trajectory" => {
                if let (Some(pdb_path), Some(dcd_path)) = (
                    params.get("pdb_path").and_then(Value::as_str),
                    params.get("dcd_path").and_then(Value::as_str),
                ) {
                    let src = params
                        .get("source_app")
                        .and_then(Value::as_str)
                        .unwrap_or("peer")
                        .to_string();
                    if let Ok(mut q) = self.inbox.lock() {
                        q.push(InboundLinkCommand::LoadTrajectory {
                            pdb_path: pdb_path.to_string(),
                            dcd_path: dcd_path.to_string(),
                            source_app: src,
                        });
                    }
                    json!({ "ok": true, "result": { "queued": true, "pdb_path": pdb_path, "dcd_path": dcd_path } })
                } else {
                    json!({ "ok": false, "error": "missing 'pdb_path' or 'dcd_path' parameter" })
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
            "biomol.load_dockcraft_bundle" => {
                match serde_json::from_value::<DockCraftSceneBundle>(params) {
                    Ok(bundle) => {
                        let title = bundle.title.clone();
                        let pose_count = bundle.poses.len();
                        if let Ok(mut q) = self.inbox.lock() {
                            q.push(InboundLinkCommand::LoadDockCraftBundle(Box::new(bundle)));
                        }
                        json!({
                            "ok": true,
                            "result": { "queued": true, "title": title, "poses": pose_count }
                        })
                    }
                    Err(e) => {
                        json!({ "ok": false, "error": format!("invalid DockCraftSceneBundle: {e}") })
                    }
                }
            }
            "biomol.get_selection_box" => {
                let req_pad = params
                    .get("padding")
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 20.0);
                match self.selection_box.read() {
                    Ok(guard) => {
                        if let Some(mut snap) = guard.clone() {
                            if let Some(p) = req_pad {
                                let delta = 2.0 * (p - snap.padding_a);
                                for s in &mut snap.grid_box.size {
                                    *s = (*s + delta).clamp(12.0, 64.0);
                                }
                                snap.padding_a = p;
                            }
                            json!({ "ok": true, "result": snap })
                        } else {
                            json!({
                                "ok": false,
                                "error": "no active selection or GridBox in BiomolCraft (select binding-site residues or set GridBox in ViewDockX first)"
                            })
                        }
                    }
                    Err(_) => json!({ "ok": false, "error": "selection box state unavailable" }),
                }
            }
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
    selection_box: Arc<RwLock<Option<SelectionBoxSnapshot>>>,
}

impl CraftBioLinkServer {
    /// Start an authenticated local Unix-socket Craft Link v1 server (mode `0600` inside `0700` dir).
    pub fn start(
        app: &str,
        app_version: &str,
        runtime: Option<RuntimeDir>,
    ) -> Result<Self, SceneError> {
        let inbox = Arc::new(Mutex::new(Vec::new()));
        let selection_box = Arc::new(RwLock::new(None));
        let sink = Arc::new(QueueSink {
            app_name: app.to_string(),
            inbox: Arc::clone(&inbox),
            selection_box: Arc::clone(&selection_box),
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
        Ok(Self {
            server,
            inbox,
            selection_box,
        })
    }

    /// Update the live selection-derived `GridBox` snapshot served synchronously to `DockCraft`.
    pub fn set_selection_box(&self, snap: Option<SelectionBoxSnapshot>) {
        if let Ok(mut guard) = self.selection_box.write() {
            *guard = snap;
        }
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

/// Request a running `BiomolCraft` instance to open a PDB/mmCIF structure and optionally activate a module.
pub fn send_open_pdb_via_craft_link(
    target: &InstanceRecord,
    source_app: &str,
    pdb_path: &str,
    module_hint: Option<&str>,
) -> Result<Value, SceneError> {
    let mut client = Client::connect(target)?;
    let reply = client.call(
        "biomol.open_pdb",
        json!({
            "source_app": source_app,
            "path": pdb_path,
            "module_hint": module_hint,
        }),
    )?;
    Ok(reply)
}

/// Request a running `BiomolCraft` instance to load a structure + binary OpenMM `.dcd` trajectory.
pub fn send_trajectory_via_craft_link(
    target: &InstanceRecord,
    source_app: &str,
    pdb_path: &str,
    dcd_path: &str,
) -> Result<Value, SceneError> {
    let mut client = Client::connect(target)?;
    let reply = client.call(
        "biomol.load_trajectory",
        json!({
            "source_app": source_app,
            "pdb_path": pdb_path,
            "dcd_path": dcd_path,
        }),
    )?;
    Ok(reply)
}

/// Send a rich `DockCraftSceneBundle` (`ViewDockX`) to a live `BiomolCraft` instance via Craft Link v1.
pub fn push_dockcraft_bundle_via_craft_link(
    target: &InstanceRecord,
    bundle: &DockCraftSceneBundle,
) -> Result<Value, SceneError> {
    let mut client = Client::connect(target)?;
    let params = serde_json::to_value(bundle)?;
    let reply = client.call("biomol.load_dockcraft_bundle", params)?;
    Ok(reply)
}

/// Synchronously query the live 3D selection / `GridBox` from a running `BiomolCraft` instance.
pub fn request_selection_box_from_biomolcraft(
    target: &InstanceRecord,
    padding_a: Option<f64>,
) -> Result<SelectionBoxSnapshot, SceneError> {
    let mut client = Client::connect(target)?;
    let params = match padding_a {
        Some(p) => json!({ "padding": p }),
        None => json!({}),
    };
    let reply = client.call("biomol.get_selection_box", params)?;
    let snap: SelectionBoxSnapshot = serde_json::from_value(reply)?;
    Ok(snap)
}

/// Discover the first live `biomolcraft` instance and synchronously fetch its 3D selection `GridBox`.
pub fn discover_and_request_selection_box(
    runtime: Option<RuntimeDir>,
    padding_a: Option<f64>,
) -> Result<SelectionBoxSnapshot, SceneError> {
    let live = discover_craft_instances(runtime, Some("biomolcraft"))?;
    let target = live.first().ok_or_else(|| {
        SceneError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no running BiomolCraft instance found on Craft Link v1 (launch `biomolcraft` first)",
        ))
    })?;
    request_selection_box_from_biomolcraft(target, padding_a)
}

/// Convenience helper for sibling CraftBio apps (`dockcraft`, `chemcraft`, `admetcraft`):
/// discovers the first live `biomolcraft` instance over Craft Link v1 and sends a PDB + optional module hint.
pub fn push_pdb_to_live_biomolcraft(
    source_app: &str,
    pdb_path: &str,
    module_hint: Option<&str>,
) -> Result<Value, SceneError> {
    let live = discover_craft_instances(None, Some("biomolcraft"))?;
    let target = live.first().ok_or_else(|| {
        SceneError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no running BiomolCraft instance found on Craft Link v1 (launch `biomolcraft` first)",
        ))
    })?;
    send_open_pdb_via_craft_link(target, source_app, pdb_path, module_hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn craft_link_v1_roundtrips_biomolecular_scene_and_dockcraft_bidirectional_rpc()
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
        let srv = CraftBioLinkServer::start("biomolcraft", "0.4.0", Some(runtime.clone()))?;
        srv.set_selection_box(Some(SelectionBoxSnapshot {
            grid_box: GridBox3D {
                center: [12.5, 4.0, -3.2],
                size: [20.0, 20.0, 20.0],
                spacing: 0.375,
            },
            padding_a: 5.0,
            source_model: "1KE6".to_string(),
            selected_residues: vec!["A:83 LEU".to_string(), "A:81 GLU".to_string()],
            heavy_atoms: 18,
        }));

        let found = discover_craft_instances(Some(runtime.clone()), Some("biomolcraft"))?;
        assert!(
            !found.is_empty(),
            "expected to discover live biomolcraft instance"
        );

        let doc = CraftSceneDocument {
            title: "Trastuzumab-HER2 Live Sync".to_string(),
            source_app: "biomolcraft".to_string(),
            ..Default::default()
        };
        let reply = push_scene_via_craft_link(&found[0], &doc)?;
        assert_eq!(reply.get("queued").and_then(Value::as_bool), Some(true));
        let r2 = send_open_pdb_via_craft_link(
            &found[0],
            "chemcraft",
            "/tmp/1N8Z.pdb",
            Some("antibodies"),
        )?;
        assert_eq!(r2.get("queued").and_then(Value::as_bool), Some(true));
        let r3 = send_trajectory_via_craft_link(
            &found[0],
            "dockcraft",
            "/tmp/1N8Z.pdb",
            "/tmp/traj.dcd",
        )?;
        assert_eq!(r3.get("queued").and_then(Value::as_bool), Some(true));

        let dock_bundle = DockCraftSceneBundle {
            title: "CDK2 ViewDockX Test".to_string(),
            receptor_name: "1KE6".to_string(),
            poses: vec![DockPoseEntry {
                pose_id: 1,
                name: "Staurosporine_Mode1".to_string(),
                canonical_smiles: None,
                sdf_block: String::new(),
                vina_kcal: -9.8,
                hyde_kcal: Some(-10.4),
                consensus_kcal: Some(-10.0),
                cnn_score: None,
                estimated_ki_nm: Some(45.0),
                ligand_efficiency: Some(0.32),
                mpo_score: Some(4.8),
                posebusters_valid: Some(true),
                posebusters_summary: Some("5/5 PASS".to_string()),
                torsion_counts: Some([3, 1, 0]),
                hyde_coronas: Vec::new(),
                torsion_lights: Vec::new(),
                interaction_lines: Vec::new(),
            }],
            ..Default::default()
        };
        let r4 = push_dockcraft_bundle_via_craft_link(&found[0], &dock_bundle)?;
        assert_eq!(r4.get("queued").and_then(Value::as_bool), Some(true));
        assert_eq!(r4.get("poses").and_then(Value::as_u64), Some(1));

        // Synchronous selection-to-GridBox RPC from DockCraft -> BiomolCraft
        let snap = discover_and_request_selection_box(Some(runtime), Some(6.0))?;
        assert_eq!(snap.source_model, "1KE6");
        assert!((snap.grid_box.size[0] - 22.0).abs() < 1e-6);
        assert_eq!(snap.selected_residues.len(), 2);

        let cmds = srv.drain_inbox();
        assert_eq!(cmds.len(), 4);
        srv.shutdown();
        let _ = std::fs::remove_dir_all(&dir_path);
        Ok(())
    }
}
