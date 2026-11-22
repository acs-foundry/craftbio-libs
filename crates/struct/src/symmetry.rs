#![forbid(unsafe_code)]

//! Crystallographic Symmetry (`CRYST1`, `REMARK 350 BIOMT`, `sym`, `symexp`, `unitcell`,
//! `crystalcontacts`), Alternate Conformation (`AltLoc`) Records, and **3D SNFG Glycan**
//! (Symbol Nomenclature for Glycans) Perception for Therapeutic Antibodies & Glycoproteins.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::io::finalize_biomolecule;
use crate::model::{Atom, BioStructError, Biomolecule, ResidueClass, dist3};

/// Crystallographic unit cell parsed from PDB `CRYST1` or mmCIF `_cell.*`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitCell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha_deg: f64,
    pub beta_deg: f64,
    pub gamma_deg: f64,
    pub space_group: String,
}

impl UnitCell {
    /// Fractional-to-Cartesian `3×3` orthogonalization matrix (PDB standard convention:
    /// `a` along X, `b` in XY plane).
    #[must_use]
    pub fn ortho_matrix(&self) -> [[f64; 3]; 3] {
        let al = self.alpha_deg.to_radians();
        let be = self.beta_deg.to_radians();
        let ga = self.gamma_deg.to_radians();
        let (ca, cb, cg) = (al.cos(), be.cos(), ga.cos());
        let sg = ga.sin().max(1e-6);
        let vol_factor = (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg)
            .max(1e-9)
            .sqrt();
        [
            [self.a, self.b * cg, self.c * cb],
            [0.0, self.b * sg, self.c * (ca - cb * cg) / sg],
            [0.0, 0.0, self.c * vol_factor / sg],
        ]
    }

    /// Cartesian-to-Fractional inverse `3×3` matrix.
    #[must_use]
    pub fn inv_ortho_matrix(&self) -> [[f64; 3]; 3] {
        let m = self.ortho_matrix();
        let m00 = m[0][0].max(1e-6);
        let m11 = m[1][1].max(1e-6);
        let m22 = m[2][2].max(1e-6);
        let inv00 = 1.0 / m00;
        let inv11 = 1.0 / m11;
        let inv22 = 1.0 / m22;
        let inv01 = -m[0][1] / (m00 * m11);
        let inv12 = -m[1][2] / (m11 * m22);
        let inv02 = (m[0][1] * m[1][2] - m[0][2] * m11) / (m00 * m11 * m22);
        [
            [inv00, inv01, inv02],
            [0.0, inv11, inv12],
            [0.0, 0.0, inv22],
        ]
    }

    #[must_use]
    pub fn frac_to_cart(&self, f: [f64; 3]) -> [f64; 3] {
        let m = self.ortho_matrix();
        [
            m[0][0] * f[0] + m[0][1] * f[1] + m[0][2] * f[2],
            m[1][0] * f[0] + m[1][1] * f[1] + m[1][2] * f[2],
            m[2][0] * f[0] + m[2][1] * f[1] + m[2][2] * f[2],
        ]
    }

    #[must_use]
    pub fn cart_to_frac(&self, p: [f64; 3]) -> [f64; 3] {
        let m = self.inv_ortho_matrix();
        [
            m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
        ]
    }

    /// 8 unit-cell corners in Cartesian Å and 12 wireframe edge index pairs.
    #[must_use]
    pub fn wireframe_edges(&self) -> ([[f64; 3]; 8], [(usize, usize); 12]) {
        let fracs = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let mut pts = [[0.0; 3]; 8];
        for (i, &f) in fracs.iter().enumerate() {
            pts[i] = self.frac_to_cart(f);
        }
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        (pts, edges)
    }
}

/// 3×4 Biological Assembly / Symmetry transformation (`REMARK 350 BIOMT`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryOp {
    pub id: usize,
    pub matrix: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl SymmetryOp {
    #[must_use]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        [
            self.matrix[0][0] * p[0]
                + self.matrix[0][1] * p[1]
                + self.matrix[0][2] * p[2]
                + self.translation[0],
            self.matrix[1][0] * p[0]
                + self.matrix[1][1] * p[1]
                + self.matrix[1][2] * p[2]
                + self.translation[1],
            self.matrix[2][0] * p[0]
                + self.matrix[2][1] * p[1]
                + self.matrix[2][2] * p[2]
                + self.translation[2],
        ]
    }

    #[must_use]
    pub fn is_identity(&self) -> bool {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut err = 0.0_f64;
        for r in 0..3 {
            err += self.translation[r].abs();
            for c in 0..3 {
                err += (self.matrix[r][c] - id[r][c]).abs();
            }
        }
        err < 1e-4
    }
}

/// Recorded alternate conformation (`altloc`) detected during PDB parsing (Altloc Explorer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AltLocRecord {
    pub chain: char,
    pub res_seq: i32,
    pub res_name: String,
    pub atom_name: String,
    pub alt_ids: Vec<(char, f64)>,
}

/// Official SNFG 3D geometric shape classification for glycans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnfgShape {
    /// GlcNAc (blue), GalNAc (yellow)
    Cube,
    /// Mannose (green), Galactose (yellow), Glucose (blue)
    Sphere,
    /// Fucose (red)
    Cone,
    /// Sialic acid / Neu5Ac (purple)
    Diamond,
    /// Xylose (orange)
    Star,
}

/// Single monosaccharide ring node ready for 3D SNFG glyph rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnfgGlycanNode {
    pub chain: char,
    pub res_seq: i32,
    pub res_name: String,
    pub canonical_name: String,
    pub shape: SnfgShape,
    pub rgb: [f32; 3],
    pub centroid: [f64; 3],
    pub radius_a: f64,
}

/// Covalent glycosidic linkage or N-/O-glycosylation attachment rod.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnfgGlycanLink {
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub label: String,
}

/// Full 3D SNFG Glycan perception report on a biomolecule.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SnfgGlycanReport {
    pub nodes: Vec<SnfgGlycanNode>,
    pub links: Vec<SnfgGlycanLink>,
}

fn classify_snfg_residue(res_name: &str) -> Option<(&'static str, SnfgShape, [f32; 3])> {
    match res_name.trim().to_ascii_uppercase().as_str() {
        "NAG" | "NDG" | "A2G" => Some(("GlcNAc", SnfgShape::Cube, [0.05, 0.44, 0.90])),
        "NGA" => Some(("GalNAc", SnfgShape::Cube, [0.97, 0.82, 0.14])),
        "MAN" | "BMA" => Some(("Man", SnfgShape::Sphere, [0.12, 0.68, 0.34])),
        "GAL" | "GLA" => Some(("Gal", SnfgShape::Sphere, [0.97, 0.82, 0.14])),
        "GLC" | "BGC" => Some(("Glc", SnfgShape::Sphere, [0.05, 0.44, 0.90])),
        "FUC" | "FUL" => Some(("Fuc", SnfgShape::Cone, [0.88, 0.16, 0.22])),
        "SIA" | "SLB" | "NGC" => Some(("Neu5Ac", SnfgShape::Diamond, [0.62, 0.22, 0.74])),
        "XYS" | "XYP" => Some(("Xyl", SnfgShape::Star, [0.96, 0.54, 0.14])),
        _ => None,
    }
}

/// Detect all N-/O-linked SNFG monosaccharide rings and glycosidic bonds in `mol`.
#[must_use]
pub fn detect_snfg_glycans(mol: &Biomolecule) -> SnfgGlycanReport {
    let mut nodes = Vec::new();
    let mut node_atoms: Vec<Vec<usize>> = Vec::new();

    for ch in &mol.chains {
        for r in &ch.residues {
            let Some((canon, shape, rgb)) = classify_snfg_residue(&r.res_name) else {
                continue;
            };
            // Compute centroid over pyranose ring atoms (or all heavy atoms if non-standard)
            let ring_pts: Vec<[f64; 3]> = r
                .atom_indices
                .iter()
                .copied()
                .filter(|&i| {
                    matches!(
                        mol.atoms[i].name.as_str(),
                        "C1" | "C2" | "C3" | "C4" | "C5" | "O5"
                    )
                })
                .map(|i| mol.atoms[i].coords)
                .collect();
            let pts = if ring_pts.len() >= 3 {
                ring_pts
            } else {
                r.atom_indices
                    .iter()
                    .copied()
                    .map(|i| mol.atoms[i].coords)
                    .collect()
            };
            if pts.is_empty() {
                continue;
            }
            let n = pts.len() as f64;
            let mut c = [0.0; 3];
            for p in &pts {
                c[0] += p[0] / n;
                c[1] += p[1] / n;
                c[2] += p[2] / n;
            }
            nodes.push(SnfgGlycanNode {
                chain: r.chain,
                res_seq: r.res_seq,
                res_name: r.res_name.clone(),
                canonical_name: canon.to_string(),
                shape,
                rgb,
                centroid: c,
                radius_a: 1.55,
            });
            node_atoms.push(r.atom_indices.clone());
        }
    }

    let mut links = Vec::new();
    // 1. Glycan-to-glycan glycosidic links (heavy-atom distance <= 2.35 A)
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            if dist3(nodes[i].centroid, nodes[j].centroid) > 9.5 {
                continue;
            }
            let linked = node_atoms[i].iter().any(|&ia| {
                node_atoms[j]
                    .iter()
                    .any(|&ib| dist3(mol.atoms[ia].coords, mol.atoms[ib].coords) <= 2.35)
            });
            if linked {
                links.push(SnfgGlycanLink {
                    from: nodes[i].centroid,
                    to: nodes[j].centroid,
                    label: format!(
                        "{}:{} — {}:{}",
                        nodes[i].canonical_name,
                        nodes[i].res_seq,
                        nodes[j].canonical_name,
                        nodes[j].res_seq
                    ),
                });
            }
        }
    }

    // 2. Covalent N-glycosylation (`ASN.ND2`) & O-glycosylation (`SER.OG` / `THR.OG1`) links
    for (ni, nd) in nodes.iter().enumerate() {
        let mut best: Option<(f64, [f64; 3], String)> = None;
        for a in &mol.atoms {
            if a.res_class != ResidueClass::Protein {
                continue;
            }
            let is_site = (a.res_name == "ASN" && a.name == "ND2")
                || (a.res_name == "SER" && a.name == "OG")
                || (a.res_name == "THR" && a.name == "OG1");
            if !is_site || dist3(a.coords, nd.centroid) > 6.5 {
                continue;
            }
            for &ga in &node_atoms[ni] {
                let d = dist3(a.coords, mol.atoms[ga].coords);
                if d <= 2.45 && best.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
                    best = Some((
                        d,
                        a.coords,
                        format!(
                            "{}:{}{} — {}{}",
                            a.chain, a.res_name, a.res_seq, nd.canonical_name, nd.res_seq
                        ),
                    ));
                }
            }
        }
        if let Some((_, p, lbl)) = best {
            links.push(SnfgGlycanLink {
                from: p,
                to: nd.centroid,
                label: lbl,
            });
        }
    }

    SnfgGlycanReport { nodes, links }
}

/// Curated fractional space-group symmetry operations `(R_frac, t_frac)` covering the most
/// common macromolecular crystallographic space groups in the PDB.
#[must_use]
pub fn space_group_fractional_ops(space_group: &str) -> Vec<([[f64; 3]; 3], [f64; 3])> {
    let sg = space_group.trim().to_ascii_uppercase().replace("  ", " ");
    let id = (
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
    );
    match sg.as_str() {
        "P 21 21 21" | "P212121" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.0, 0.5],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.5, 0.5],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.0],
            ),
        ],
        "P 21 21 2" | "P21212" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.0],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.0],
            ),
        ],
        "P 1 21 1" | "P 21" | "P21" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.5, 0.0],
            ),
        ],
        "C 1 2 1" | "C 2" | "C2" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.0],
            ),
        ],
        "C 2 2 21" | "C2221" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.5],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.5],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.5],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.5],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.0],
            ),
        ],
        "P 61" | "P61" => vec![
            id,
            (
                [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 1.0 / 3.0],
            ),
            (
                [[-1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 2.0 / 3.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.5],
            ),
            (
                [[0.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 5.0 / 6.0],
            ),
            (
                [[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 1.0 / 6.0],
            ),
        ],
        "P 65" | "P65" => vec![
            id,
            (
                [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 2.0 / 3.0],
            ),
            (
                [[-1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 1.0 / 3.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.5],
            ),
            (
                [[0.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 1.0 / 6.0],
            ),
            (
                [[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 5.0 / 6.0],
            ),
        ],
        "P 41 21 2" | "P41212" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.5],
            ),
            (
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.25],
            ),
            (
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.75],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.75],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.25],
            ),
            (
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.5],
            ),
        ],
        "P 43 21 2" | "P43212" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.5],
            ),
            (
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.75],
            ),
            (
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.25],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.25],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.75],
            ),
            (
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.5],
            ),
        ],
        "I 2 2 2" | "I222" => vec![
            id,
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.0, 0.0, 0.0],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.5],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5, 0.5, 0.5],
            ),
            (
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.5],
            ),
            (
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                [0.5, 0.5, 0.5],
            ),
        ],
        _ => vec![id],
    }
}

const CHAIN_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Build the full Biological Assembly from `REMARK 350 BIOMT` matrices (ChimeraX `sym`).
pub fn generate_biological_assembly(mol: &Biomolecule) -> Result<Biomolecule, BioStructError> {
    if mol.biomt_ops.is_empty() {
        return Err(BioStructError::Selection(
            "no REMARK 350 BIOMT operators present in structure".to_string(),
        ));
    }
    let mut all_atoms: Vec<Atom> = Vec::with_capacity(mol.atoms.len() * mol.biomt_ops.len());
    let mut used_chains: HashSet<char> = HashSet::new();
    let mut next_chain_idx = 0usize;

    for (copy_i, op) in mol.biomt_ops.iter().enumerate() {
        let mut chain_map = std::collections::HashMap::new();
        for ch in &mol.chains {
            let new_c = if copy_i == 0 && !used_chains.contains(&ch.id) {
                ch.id
            } else {
                let mut cand = 'A';
                while next_chain_idx < CHAIN_ALPHABET.len() {
                    let c = CHAIN_ALPHABET[next_chain_idx] as char;
                    next_chain_idx += 1;
                    if !used_chains.contains(&c) {
                        cand = c;
                        break;
                    }
                }
                cand
            };
            used_chains.insert(new_c);
            chain_map.insert(ch.id, new_c);
        }
        for a in &mol.atoms {
            let mut copy = a.clone();
            copy.coords = op.apply(a.coords);
            copy.chain = chain_map.get(&a.chain).copied().unwrap_or(a.chain);
            copy.serial = all_atoms.len() + 1;
            all_atoms.push(copy);
        }
    }

    let mut out = finalize_biomolecule(
        format!("{}_bioassembly", mol.id),
        format!(
            "Biological Assembly ({} copies) of {}",
            mol.biomt_ops.len(),
            mol.title
        ),
        all_atoms,
        Vec::new(),
        mol.is_alphafold_plddt,
    )?;
    out.unit_cell = mol.unit_cell.clone();
    out.biomt_ops = mol.biomt_ops.clone();
    Ok(out)
}

/// Generate crystallographic symmetry mates within `cutoff_a` of `mol` (PyMOL `symexp` / ChimeraX `crystalcontacts`).
/// Returns `(Vec<(mate_label, Biomolecule)>, total_crystal_contact_pairs_within_4A)`.
pub fn generate_crystal_mates(
    mol: &Biomolecule,
    cutoff_a: f64,
) -> Result<(Vec<(String, Biomolecule)>, usize), BioStructError> {
    let Some(ref uc) = mol.unit_cell else {
        return Err(BioStructError::Selection(
            "no CRYST1 / unit cell metadata present in structure".to_string(),
        ));
    };
    let ops = space_group_fractional_ops(&uc.space_group);
    let cutoff = cutoff_a.clamp(2.0, 25.0);
    let (_, _, orig_center, orig_rad) = mol.bounds();

    // Sample up to 350 guide/heavy anchor atoms from asymmetric unit for fast contact checks
    let step = (mol.atoms.len() / 350).max(1);
    let ref_pts: Vec<[f64; 3]> = mol
        .atoms
        .iter()
        .step_by(step)
        .filter(|a| a.res_class != ResidueClass::Water)
        .map(|a| a.coords)
        .collect();
    if ref_pts.is_empty() {
        return Ok((Vec::new(), 0));
    }

    // Center of ASU in fractional space to determine nearest lattice shifts cleanly
    let mut mates: Vec<(f64, String, Biomolecule)> = Vec::new();
    let mut total_contacts = 0usize;

    for (op_i, &(rot, tr)) in ops.iter().enumerate() {
        for tx in -1..=1 {
            for ty in -1..=1 {
                for tz in -1..=1 {
                    let is_id = op_i == 0 && tx == 0 && ty == 0 && tz == 0;
                    if is_id {
                        continue;
                    }
                    let xform = |p: [f64; 3]| -> [f64; 3] {
                        let f = uc.cart_to_frac(p);
                        let nf = [
                            rot[0][0] * f[0]
                                + rot[0][1] * f[1]
                                + rot[0][2] * f[2]
                                + tr[0]
                                + f64::from(tx),
                            rot[1][0] * f[0]
                                + rot[1][1] * f[1]
                                + rot[1][2] * f[2]
                                + tr[1]
                                + f64::from(ty),
                            rot[2][0] * f[0]
                                + rot[2][1] * f[1]
                                + rot[2][2] * f[2]
                                + tr[2]
                                + f64::from(tz),
                        ];
                        uc.frac_to_cart(nf)
                    };
                    let mate_center = xform(orig_center);
                    if dist3(mate_center, orig_center) > 2.0 * orig_rad + cutoff + 4.0 {
                        continue;
                    }
                    let mut min_d = f64::INFINITY;
                    let mut contact_pairs = 0usize;
                    for &rp in &ref_pts {
                        let mp = xform(rp);
                        for &op_p in &ref_pts {
                            let d = dist3(mp, op_p);
                            if d < min_d {
                                min_d = d;
                            }
                            if d <= 4.0 {
                                contact_pairs += 1;
                            }
                        }
                    }
                    if min_d <= cutoff && min_d > 0.25 {
                        total_contacts += contact_pairs;
                        let mut mate_atoms = Vec::with_capacity(mol.atoms.len());
                        for (ai, a) in mol.atoms.iter().enumerate() {
                            let mut na = a.clone();
                            na.coords = xform(a.coords);
                            na.serial = ai + 1;
                            mate_atoms.push(na);
                        }
                        let lbl = format!("sym{}_{tx:+}_{ty:+}_{tz:+}", op_i + 1);
                        if let Ok(mut mate) = finalize_biomolecule(
                            format!("{}_{lbl}", mol.id),
                            format!("Crystal Mate {lbl} of {}", mol.title),
                            mate_atoms,
                            Vec::new(),
                            mol.is_alphafold_plddt,
                        ) {
                            mate.unit_cell = Some(uc.clone());
                            mates.push((min_d, lbl, mate));
                        }
                    }
                }
            }
        }
    }

    mates.sort_by(|a, b| a.0.total_cmp(&b.0));
    mates.truncate(8);
    let out = mates.into_iter().map(|(_, lbl, m)| (lbl, m)).collect();
    Ok((out, total_contacts))
}
