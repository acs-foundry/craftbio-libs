//! Non-blocking native file & folder dialogs wired to [`egui::Context`] via [`craft_dialogs`].

use std::path::Path;
use std::sync::Arc;

use craft_dialogs::{Dialogs, FileDialog};

/// Create a [`craft_dialogs::Dialogs`] queue whose wakeup callback repaints `ctx`.
#[must_use]
pub fn dialogs_for_egui<T>(ctx: &egui::Context) -> Dialogs<T> {
    let wake_ctx = ctx.clone();
    Dialogs::new(Arc::new(move || {
        wake_ctx.request_repaint();
    }))
}

/// File picker for small-molecule / batch chemical inputs (`.smi`, `.smiles`, `.csv`, `.sdf`, `.mol`).
#[must_use]
pub fn open_chemical_file(start_dir: Option<&Path>) -> FileDialog {
    let mut dlg = FileDialog::open().title("Open chemical file").filter(
        "Chemical files",
        &["smi", "smiles", "csv", "sdf", "mol", "txt"],
    );
    if let Some(dir) = start_dir.filter(|p| p.is_dir()) {
        dlg = dlg.directory(dir);
    }
    dlg
}

/// File picker for 3D biomolecular structures (`.pdb`, `.cif`, `.mmcif`, `.sdf`, `.mol2`).
#[must_use]
pub fn open_structure_file(start_dir: Option<&Path>) -> FileDialog {
    let mut dlg = FileDialog::open().title("Open structure file").filter(
        "Structure files",
        &["pdb", "ent", "cif", "mmcif", "sdf", "mol2"],
    );
    if let Some(dir) = start_dir.filter(|p| p.is_dir()) {
        dlg = dlg.directory(dir);
    }
    dlg
}

/// File picker for sequence files (`.fa`, `.fasta`, `.faa`, `.fna`, `.gb`, `.gbk`).
#[must_use]
pub fn open_sequence_file(start_dir: Option<&Path>) -> FileDialog {
    let mut dlg = FileDialog::open().title("Open sequence file").filter(
        "Sequence files",
        &["fa", "fasta", "faa", "fna", "gb", "gbk"],
    );
    if let Some(dir) = start_dir.filter(|p| p.is_dir()) {
        dlg = dlg.directory(dir);
    }
    dlg
}

/// Folder picker for sandboxed exports or campaign directories.
#[must_use]
pub fn choose_folder(title: &str, start_dir: Option<&Path>) -> FileDialog {
    let mut dlg = FileDialog::folder().title(title);
    if let Some(dir) = start_dir.filter(|p| p.is_dir()) {
        dlg = dlg.directory(dir);
    }
    dlg
}

/// Save-file picker with suggested filename and extension filter.
#[must_use]
pub fn save_file(
    title: &str,
    default_name: &str,
    filter_name: &str,
    exts: &[&str],
    start_dir: Option<&Path>,
) -> FileDialog {
    let mut dlg = FileDialog::save()
        .title(title)
        .file_name(default_name)
        .filter(filter_name, exts);
    if let Some(dir) = start_dir.filter(|p| p.is_dir()) {
        dlg = dlg.directory(dir);
    }
    dlg
}
