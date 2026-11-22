#![forbid(unsafe_code)]

//! Universal Biomolecular Data Model supporting Proteins, Antibodies, Enzymes,
//! **ncRNA / mRNA / DNA**, Small-Molecule Ligands, Ions, Waters, and Multi-Frame Trajectories.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Safety cap for loaded biomolecular atoms to prevent unbounded allocations on hostile input.
pub const MAX_BIOMOL_ATOMS: usize = 500_000;

#[derive(Debug, Error)]
pub enum BioStructError {
    #[error("empty biomolecular input")]
    EmptyInput,
    #[error("structure parse error: {0}")]
    Parse(String),
    #[error("structure exceeds safety limit of {MAX_BIOMOL_ATOMS} atoms (got {0})")]
    TooManyAtoms(usize),
    #[error("selection algebra error: {0}")]
    Selection(String),
    #[error("alignment error: {0}")]
    Alignment(String),
}

/// Chemical polymer/entity class of a residue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidueClass {
    Protein,
    Rna,
    Dna,
    Ligand,
    Ion,
    Water,
}

/// Secondary structure classification (DSSP-class for protein + nucleic ribbon for RNA/DNA).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecondaryStructure {
    Helix,
    Sheet,
    Coil,
    NucleicRibbon,
}

/// Single atom inside a biomolecular model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Atom {
    pub serial: usize,
    pub name: String,
    pub element: String,
    pub res_name: String,
    pub res_code: char,
    pub res_class: ResidueClass,
    pub chain: char,
    pub res_seq: i32,
    pub icode: char,
    pub coords: [f64; 3],
    pub occupancy: f64,
    pub b_factor: f64,
    pub formal_charge: f64,
    pub is_hetatm: bool,
}

/// Covalent bond between two atom indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub order: u8,
}

/// Grouped residue (amino acid, nucleotide, ligand, ion, or water).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residue {
    pub chain: char,
    pub res_seq: i32,
    pub icode: char,
    pub res_name: String,
    pub code: char,
    pub class: ResidueClass,
    pub ss: SecondaryStructure,
    pub atom_indices: Vec<usize>,
    /// Representative backbone guide atom: `CA` for protein, `C4'` or `P` for RNA/DNA.
    pub guide_idx: Option<usize>,
    /// Secondary orientation atom: `O` for protein ribbon plane, `C1'`/`N1`/`N9` for RNA base rung.
    pub orient_idx: Option<usize>,
    /// Base tip atom (`N1`/`N3`/`C4` or ring centroid) for RNA/DNA base-pair ladders.
    pub base_tip_idx: Option<usize>,
    pub mean_bfactor: f64,
}

/// Polymer or heterogen chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chain {
    pub id: char,
    pub dominant_class: ResidueClass,
    pub sequence: String,
    pub residues: Vec<Residue>,
}

/// Covalent disulfide bridge (`SG–SG <= 2.65 Å`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disulfide {
    pub chain_a: char,
    pub res_seq_a: i32,
    pub chain_b: char,
    pub res_seq_b: i32,
    pub distance_a: f64,
}

/// Detected 3D RNA/DNA base pair (`A-U`, `G-C`, `G-U`, `A-T`, non-canonical).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasePair3D {
    pub chain_a: char,
    pub res_seq_a: i32,
    pub base_a: char,
    pub chain_b: char,
    pub res_seq_b: i32,
    pub base_b: char,
    pub kind: String,
    pub c1_distance_a: f64,
    pub hbond_count: usize,
}

/// Single trajectory / NMR model coordinate frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    pub frame_index: usize,
    pub time_ps: f64,
    pub coords: Vec<[f64; 3]>,
}

/// Complete universal biomolecular model (Proteins, Antibodies, Enzymes, ncRNA/mRNA, Ligands).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Biomolecule {
    pub id: String,
    pub title: String,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub chains: Vec<Chain>,
    pub disulfides: Vec<Disulfide>,
    pub base_pairs: Vec<BasePair3D>,
    pub frames: Vec<TrajectoryFrame>,
    pub active_frame: usize,
    pub is_alphafold_plddt: bool,
    #[serde(default)]
    pub unit_cell: Option<crate::symmetry::UnitCell>,
    #[serde(default)]
    pub biomt_ops: Vec<crate::symmetry::SymmetryOp>,
    #[serde(default)]
    pub altlocs: Vec<crate::symmetry::AltLocRecord>,
}

impl Biomolecule {
    #[must_use]
    pub fn chain(&self, id: char) -> Option<&Chain> {
        self.chains.iter().find(|c| c.id == id)
    }

    #[must_use]
    pub fn chain_ids(&self) -> Vec<char> {
        self.chains.iter().map(|c| c.id).collect()
    }

    /// Geometric center of all atoms in the biomolecule.
    #[must_use]
    pub fn centroid(&self) -> [f64; 3] {
        if self.atoms.is_empty() {
            return [0.0; 3];
        }
        let mut s = [0.0_f64; 3];
        for a in &self.atoms {
            s[0] += a.coords[0];
            s[1] += a.coords[1];
            s[2] += a.coords[2];
        }
        let inv = 1.0 / (self.atoms.len() as f64);
        [s[0] * inv, s[1] * inv, s[2] * inv]
    }

    /// Bounding box `(min, max, center, radius)` of all atoms.
    #[must_use]
    pub fn bounds(&self) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
        if self.atoms.is_empty() {
            return ([0.0; 3], [0.0; 3], [0.0; 3], 10.0);
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for a in &self.atoms {
            for k in 0..3 {
                lo[k] = lo[k].min(a.coords[k]);
                hi[k] = hi[k].max(a.coords[k]);
            }
        }
        let c = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        let r = (dist3(lo, hi) * 0.5).max(5.0);
        (lo, hi, c, r)
    }

    /// Switch active trajectory coordinates if multi-frame.
    pub fn set_active_frame(&mut self, idx: usize) -> bool {
        if let Some(frame) = self.frames.get(idx)
            && frame.coords.len() == self.atoms.len()
        {
            for (a, &pos) in self.atoms.iter_mut().zip(frame.coords.iter()) {
                a.coords = pos;
            }
            self.active_frame = idx;
            return true;
        }
        false
    }
}

/// Euclidean distance in Angstroms.
#[must_use]
pub fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Classify a PDB/mmCIF residue name into `(ResidueClass, 1-letter code)`.
#[must_use]
pub fn classify_residue(res_name: &str) -> (ResidueClass, char) {
    let u = res_name.trim().to_ascii_uppercase();
    match u.as_str() {
        // Canonical & protonation-state amino acids
        "ALA" => (ResidueClass::Protein, 'A'),
        "ARG" => (ResidueClass::Protein, 'R'),
        "ASN" => (ResidueClass::Protein, 'N'),
        "ASP" | "ASH" => (ResidueClass::Protein, 'D'),
        "CYS" | "CYX" | "CYM" => (ResidueClass::Protein, 'C'),
        "GLN" | "PCA" | "PYR" => (ResidueClass::Protein, 'Q'),
        "GLU" | "GLH" => (ResidueClass::Protein, 'E'),
        "GLY" => (ResidueClass::Protein, 'G'),
        "HIS" | "HID" | "HIE" | "HIP" => (ResidueClass::Protein, 'H'),
        "ILE" => (ResidueClass::Protein, 'I'),
        "LEU" => (ResidueClass::Protein, 'L'),
        "LYS" | "LYN" => (ResidueClass::Protein, 'K'),
        "MET" | "MSE" => (ResidueClass::Protein, 'M'),
        "PHE" => (ResidueClass::Protein, 'F'),
        "PRO" => (ResidueClass::Protein, 'P'),
        "SER" | "SEP" => (ResidueClass::Protein, 'S'),
        "THR" | "TPO" => (ResidueClass::Protein, 'T'),
        "TRP" => (ResidueClass::Protein, 'W'),
        "TYR" | "PTR" => (ResidueClass::Protein, 'Y'),
        "VAL" => (ResidueClass::Protein, 'V'),
        // RNA & modified ncRNA/mRNA ribonucleotides
        "A" | "RA" | "ADE" | "1MA" | "6MA" | "M6A" | "A2M" => (ResidueClass::Rna, 'A'),
        "U" | "RU" | "URA" | "PSU" | "H2U" | "5MU" | "M1U" | "OMU" | "4SU" => {
            (ResidueClass::Rna, 'U')
        }
        "G" | "RG" | "GUA" | "OMG" | "7MG" | "2MG" | "M2G" | "GTP" => (ResidueClass::Rna, 'G'),
        "C" | "RC" | "CYT" | "5MC" | "OMC" => (ResidueClass::Rna, 'C'),
        // DNA deoxyribonucleotides
        "DA" => (ResidueClass::Dna, 'A'),
        "DT" => (ResidueClass::Dna, 'T'),
        "DG" => (ResidueClass::Dna, 'G'),
        "DC" => (ResidueClass::Dna, 'C'),
        // Waters
        "HOH" | "WAT" | "SOL" | "TIP" | "TIP3" | "DOD" => (ResidueClass::Water, 'O'),
        // Common biological metal & buffer ions
        "MG" | "ZN" | "CA" | "FE" | "FE2" | "MN" | "CU" | "NA" | "K" | "CL" | "CO" | "NI"
        | "CD" | "BA" | "SR" => (ResidueClass::Ion, 'I'),
        // Everything else (ligands, cofactors, glycans, lipids, drugs)
        _ => (ResidueClass::Ligand, 'X'),
    }
}

/// Convert 1-letter amino acid to 3-letter PDB code.
#[must_use]
pub fn aa1_to_aa3(c: char) -> &'static str {
    match c.to_ascii_uppercase() {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "UNK",
    }
}

/// Approximate physiological (pH 7.4) partial charge for rapid 3D electrostatics (`ESP`).
#[must_use]
pub fn approx_atom_charge(res_class: ResidueClass, res_code: char, atom_name: &str) -> f64 {
    match res_class {
        ResidueClass::Protein => match (res_code, atom_name) {
            ('D', "OD1" | "OD2") | ('E', "OE1" | "OE2") => -0.5,
            ('K', "NZ") => 1.0,
            ('R', "NH1" | "NH2") => 0.5,
            ('H', "ND1" | "NE2") => 0.15,
            (_, "O") => -0.35,
            (_, "N") => 0.25,
            _ => 0.0,
        },
        ResidueClass::Rna | ResidueClass::Dna => match atom_name {
            "P" => 0.5,
            "OP1" | "OP2" | "O1P" | "O2P" => -0.75,
            "O3'" | "O5'" | "O2'" => -0.25,
            _ => 0.0,
        },
        ResidueClass::Ion => match atom_name {
            "MG" | "ZN" | "CA" | "FE" | "MN" | "CU" => 2.0,
            "NA" | "K" => 1.0,
            "CL" => -1.0,
            _ => 0.0,
        },
        _ => 0.0,
    }
}

/// Bondi Van der Waals radius (Angstroms).
#[must_use]
pub fn vdw_radius(elem: &str) -> f64 {
    match elem.to_ascii_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I" => 1.98,
        "MG" => 1.73,
        "ZN" => 1.39,
        "FE" => 1.40,
        "CA" => 2.00,
        _ => 1.70,
    }
}

/// Covalent radius (Angstroms) for bond perception.
#[must_use]
pub fn covalent_radius(elem: &str) -> f64 {
    match elem.to_ascii_uppercase().as_str() {
        "H" => 0.31,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        "P" => 1.07,
        "S" => 1.05,
        "CL" => 1.02,
        "BR" => 1.20,
        "I" => 1.39,
        _ => 0.77,
    }
}

/// Standard CPK RGB color `[0..1; 3]` for an element.
#[must_use]
pub fn cpk_color(elem: &str) -> [f32; 3] {
    match elem.to_ascii_uppercase().as_str() {
        "C" => [0.58, 0.64, 0.72],
        "N" => [0.22, 0.51, 0.96],
        "O" => [0.94, 0.27, 0.27],
        "S" => [0.92, 0.78, 0.16],
        "P" => [0.98, 0.45, 0.09],
        "F" | "CL" => [0.13, 0.77, 0.37],
        "BR" => [0.65, 0.22, 0.16],
        "MG" | "CA" => [0.18, 0.83, 0.75],
        "ZN" | "FE" => [0.66, 0.33, 0.97],
        "H" => [0.92, 0.94, 0.96],
        _ => [0.75, 0.75, 0.80],
    }
}

/// Distinct nucleotide color `[0..1; 3]` for RNA/ncRNA/DNA visualization.
#[must_use]
pub fn nucleotide_color(base: char) -> [f32; 3] {
    match base.to_ascii_uppercase() {
        'A' => [0.95, 0.36, 0.36],       // coral-red
        'U' | 'T' => [0.22, 0.74, 0.97], // sky-blue
        'G' => [0.20, 0.83, 0.60],       // emerald
        'C' => [0.98, 0.75, 0.14],       // amber
        _ => [0.75, 0.75, 0.82],
    }
}

/// Black & Mould (1991) hydrophobicity centered at Gly = 0.0 for 3D SAP.
#[must_use]
pub fn black_mould_sap_hydro(aa: char) -> f64 {
    let raw = match aa.to_ascii_uppercase() {
        'F' => 1.000,
        'L' | 'I' => 0.943,
        'Y' => 0.880,
        'W' => 0.878,
        'V' => 0.825,
        'M' => 0.738,
        'P' => 0.711,
        'C' => 0.680,
        'A' => 0.616,
        'G' => 0.501,
        'T' => 0.450,
        'S' => 0.359,
        'K' => 0.283,
        'Q' => 0.251,
        'N' => 0.236,
        'H' => 0.165,
        'E' => 0.043,
        'D' => 0.028,
        'R' => 0.000,
        _ => 0.501,
    };
    raw - 0.501
}

/// Tien (2013) reference fully-exposed residue SASA (Å²).
#[must_use]
pub fn max_residue_sasa(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'A' => 129.0,
        'R' => 274.0,
        'N' => 195.0,
        'D' => 193.0,
        'C' => 167.0,
        'E' => 223.0,
        'Q' => 225.0,
        'G' => 104.0,
        'H' => 224.0,
        'I' => 197.0,
        'L' => 201.0,
        'K' => 236.0,
        'M' => 224.0,
        'F' => 240.0,
        'P' => 159.0,
        'S' => 155.0,
        'T' => 172.0,
        'W' => 285.0,
        'Y' => 263.0,
        'V' => 174.0,
        _ => 220.0,
    }
}
