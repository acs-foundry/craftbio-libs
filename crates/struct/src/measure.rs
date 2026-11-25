#![forbid(unsafe_code)]

//! Biomolecular Chemical Analysis, Interactive 3D Measurements, Axes/Planes/Centroids,
//! Synchronized Multi-Model MSA (`Match->Align`), and AlphaFold/Boltz PAE Error Matrices.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::align::{jacobi_sym3, smith_waterman_pairs};
use crate::analysis::compute_sasa;
use crate::model::{BioStructError, Biomolecule, ResidueClass, dist3, vdw_radius};

/// Detected hydrogen bond (ChimeraX `hbonds` / PyMOL polar contacts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HBondRecord {
    pub atom_a: usize,
    pub atom_b: usize,
    pub label_a: String,
    pub label_b: String,
    pub coords_a: [f64; 3],
    pub coords_b: [f64; 3],
    pub distance_a: f64,
    pub angle_deg: f64,
    pub energy_kcal: f64,
}

/// Detected steric clash (ChimeraX `clashes`: VDW overlap >= `overlap_cutoff_a`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClashRecord {
    pub atom_a: usize,
    pub atom_b: usize,
    pub label_a: String,
    pub label_b: String,
    pub coords_a: [f64; 3],
    pub coords_b: [f64; 3],
    pub distance_a: f64,
    pub overlap_a: f64,
}

/// Non-covalent atomic contact pair (ChimeraX `contacts`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactRecord {
    pub atom_a: usize,
    pub atom_b: usize,
    pub label_a: String,
    pub label_b: String,
    pub coords_a: [f64; 3],
    pub coords_b: [f64; 3],
    pub distance_a: f64,
}

/// Best-fit 3D principal axis (helix axis / domain vector — ChimeraX `define axis`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryAxis {
    pub label: String,
    pub centroid: [f64; 3],
    pub direction: [f64; 3],
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub length_a: f64,
    pub linearity_ratio: f64,
}

/// Best-fit 3D plane (aromatic ring / domain interface plane — ChimeraX `define plane`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryPlane {
    pub label: String,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub radius_a: f64,
    pub rms_deviation_a: f64,
}

/// Synchronized Multi-Model Sequence Alignment (`Match->Align`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsaRow {
    pub model_name: String,
    pub chain_id: char,
    /// One character per MSA column (`'-'` for gap).
    pub aligned_chars: Vec<char>,
    /// Corresponding PDB residue sequence number for each column (`None` on gap).
    pub res_seqs: Vec<Option<i32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsaAlignment {
    pub num_columns: usize,
    pub rows: Vec<MsaRow>,
    pub column_conservation: Vec<f32>,
}

/// Predicted Aligned Error (PAE) matrix from AlphaFold / Boltz-2 / Chai-1 (or structural proxy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaeMatrix {
    pub label: String,
    pub length: usize,
    /// Row-major `length * length` PAE values in Å (`0.0..=31.75`).
    pub values: Vec<f32>,
    pub max_pae: f32,
    pub mean_pae: f32,
    pub is_experimental_json: bool,
    /// Automatically segmented low-error rigid domain intervals `(start_res_idx, end_res_idx)`.
    pub domain_blocks: Vec<(usize, usize)>,
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3_f64(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3_f64(a: [f64; 3]) -> [f64; 3] {
    let l = dot3_f64(a, a).sqrt();
    if l > 1e-10 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

/// Exact Euclidean distance between 2 atoms (Å).
#[must_use]
pub fn measure_distance(p1: [f64; 3], p2: [f64; 3]) -> f64 {
    dist3(p1, p2)
}

/// Exact bond/geometric angle `p1 - p2 - p3` in degrees (`[0, 180]`).
#[must_use]
pub fn measure_angle_deg(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let v1 = norm3_f64(sub3(p1, p2));
    let v2 = norm3_f64(sub3(p3, p2));
    dot3_f64(v1, v2).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Exact IUPAC signed dihedral/torsion angle `p1 - p2 - p3 - p4` in degrees (`[-180, +180]`).
#[must_use]
pub fn measure_dihedral_deg(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
    let b0 = sub3(p1, p2);
    let b1 = norm3_f64(sub3(p3, p2));
    let b2 = sub3(p4, p3);
    let v = sub3(
        b0,
        [
            b1[0] * dot3_f64(b0, b1),
            b1[1] * dot3_f64(b0, b1),
            b1[2] * dot3_f64(b0, b1),
        ],
    );
    let w = sub3(
        b2,
        [
            b1[0] * dot3_f64(b2, b1),
            b1[1] * dot3_f64(b2, b1),
            b1[2] * dot3_f64(b2, b1),
        ],
    );
    let x = dot3_f64(v, w);
    let y = dot3_f64(cross3_f64(b1, v), w);
    y.atan2(x).to_degrees()
}

/// Angle in degrees between two 3D axes or plane normals (`[0, 90]` acute or `[0, 180]` directed).
#[must_use]
pub fn axis_angle_deg(dir_a: [f64; 3], dir_b: [f64; 3], acute: bool) -> f64 {
    let na = norm3_f64(dir_a);
    let nb = norm3_f64(dir_b);
    let d = dot3_f64(na, nb).clamp(-1.0, 1.0);
    let ang = d.acos().to_degrees();
    if acute && ang > 90.0 {
        180.0 - ang
    } else {
        ang
    }
}

fn atom_label(mol: &Biomolecule, idx: usize) -> String {
    let a = &mol.atoms[idx];
    format!("{}:{}{}.{}", a.chain, a.res_name, a.res_seq, a.name)
}

fn build_adjacency(mol: &Biomolecule) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); mol.atoms.len()];
    for b in &mol.bonds {
        if b.a < mol.atoms.len() && b.b < mol.atoms.len() {
            adj[b.a].push(b.b);
            adj[b.b].push(b.a);
        }
    }
    adj
}

fn within_bond_hops(adj: &[Vec<usize>], start: usize, target: usize, max_hops: usize) -> bool {
    if start == target {
        return true;
    }
    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    visited.insert(start);
    q.push_back((start, 0usize));
    while let Some((u, d)) = q.pop_front() {
        if d >= max_hops {
            continue;
        }
        for &v in &adj[u] {
            if v == target {
                return true;
            }
            if visited.insert(v) {
                q.push_back((v, d + 1));
            }
        }
    }
    false
}

fn is_polar_hbond_atom(elem: &str) -> bool {
    matches!(elem, "N" | "O" | "S" | "F")
}

/// Detect geometric hydrogen bonds (`N/O/S/F` heavy donor–acceptor pairs within `max_dist_a`,
/// default `3.5 Å`, checking base-donor-acceptor angle `>= min_angle_deg`, default `95°`).
/// When `mask_a` and `mask_b` are `None`, scans inter-chain and non-adjacent (`|Δres| >= 2`) pairs.
#[must_use]
pub fn detect_hbonds(
    mol: &Biomolecule,
    mask_a: Option<&[bool]>,
    mask_b: Option<&[bool]>,
    max_dist_a: f64,
    min_angle_deg: f64,
) -> Vec<HBondRecord> {
    let n = mol.atoms.len();
    if n < 2 {
        return Vec::new();
    }
    let adj = build_adjacency(mol);
    let cutoff = max_dist_a.clamp(2.2, 4.2);
    let cutoff2 = cutoff * cutoff;
    let cell = cutoff.max(3.5);
    let ckey = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell).floor() as i32,
            (c[1] / cell).floor() as i32,
            (c[2] / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        if is_polar_hbond_atom(&a.element) && a.res_class != ResidueClass::Water {
            grid.entry(ckey(a.coords)).or_default().push(i);
        }
    }

    let in_a = |i: usize| mask_a.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let in_b = |i: usize| mask_b.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let cross_mask = mask_a.is_some() && mask_b.is_some();

    let mut out = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();

    for (&(cx, cy, cz), bucket) in &grid {
        for &i in bucket {
            let ai = &mol.atoms[i];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(nbrs) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &j in nbrs {
                            if i >= j {
                                continue;
                            }
                            let pair_ok = if cross_mask {
                                (in_a(i) && in_b(j)) || (in_a(j) && in_b(i))
                            } else {
                                in_a(i) && in_b(j)
                            };
                            if !pair_ok {
                                continue;
                            }
                            let aj = &mol.atoms[j];
                            if ai.chain == aj.chain && (ai.res_seq - aj.res_seq).abs() <= 1 {
                                continue;
                            }
                            let d2 = (ai.coords[0] - aj.coords[0]).powi(2)
                                + (ai.coords[1] - aj.coords[1]).powi(2)
                                + (ai.coords[2] - aj.coords[2]).powi(2);
                            if d2 < 2.2 * 2.2 || d2 > cutoff2 {
                                continue;
                            }
                            if within_bond_hops(&adj, i, j, 3) {
                                continue;
                            }
                            let d = d2.sqrt();
                            // Base-donor-acceptor geometry check using first heavy neighbor of i or j
                            let ang_i = adj[i]
                                .first()
                                .map(|&bi| {
                                    measure_angle_deg(mol.atoms[bi].coords, ai.coords, aj.coords)
                                })
                                .unwrap_or(140.0);
                            let ang_j = adj[j]
                                .first()
                                .map(|&bj| {
                                    measure_angle_deg(mol.atoms[bj].coords, aj.coords, ai.coords)
                                })
                                .unwrap_or(140.0);
                            let best_ang = ang_i.min(ang_j);
                            if best_ang < min_angle_deg {
                                continue;
                            }
                            if seen.insert((i, j)) {
                                // Empirical H-bond score (strongest near 2.85 A & linear angle)
                                let dist_f = (1.0 - ((d - 2.85).abs() / 0.95)).clamp(0.15, 1.0);
                                let ang_f = ((best_ang - 90.0) / 80.0).clamp(0.2, 1.0);
                                let energy = -((2.8 * dist_f * ang_f) * 100.0).round() / 100.0;
                                out.push(HBondRecord {
                                    atom_a: i,
                                    atom_b: j,
                                    label_a: atom_label(mol, i),
                                    label_b: atom_label(mol, j),
                                    coords_a: ai.coords,
                                    coords_b: aj.coords,
                                    distance_a: (d * 100.0).round() / 100.0,
                                    angle_deg: (best_ang * 10.0).round() / 10.0,
                                    energy_kcal: energy,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.distance_a.total_cmp(&b.distance_a));
    out
}

/// Detect steric atomic clashes (`vdw(i) + vdw(j) - dist >= overlap_cutoff_a`, default `0.60 Å`,
/// skipping 1–2, 1–3, and 1–4 covalently connected neighbors — ChimeraX `clashes` equivalent).
#[must_use]
pub fn detect_clashes(
    mol: &Biomolecule,
    mask_a: Option<&[bool]>,
    mask_b: Option<&[bool]>,
    overlap_cutoff_a: f64,
) -> Vec<ClashRecord> {
    let n = mol.atoms.len();
    if n < 2 {
        return Vec::new();
    }
    let adj = build_adjacency(mol);
    let cell = 4.0_f64;
    let ckey = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell).floor() as i32,
            (c[1] / cell).floor() as i32,
            (c[2] / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        if a.res_class != ResidueClass::Water && a.element != "H" {
            grid.entry(ckey(a.coords)).or_default().push(i);
        }
    }
    let in_a = |i: usize| mask_a.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let in_b = |i: usize| mask_b.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let cross_mask = mask_a.is_some() && mask_b.is_some();

    let mut out = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();

    for (&(cx, cy, cz), bucket) in &grid {
        for &i in bucket {
            let ai = &mol.atoms[i];
            let ri = vdw_radius(&ai.element);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(nbrs) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &j in nbrs {
                            if i >= j {
                                continue;
                            }
                            let pair_ok = if cross_mask {
                                (in_a(i) && in_b(j)) || (in_a(j) && in_b(i))
                            } else {
                                in_a(i) && in_b(j)
                            };
                            if !pair_ok {
                                continue;
                            }
                            let aj = &mol.atoms[j];
                            if ai.chain == aj.chain && ai.res_seq == aj.res_seq {
                                continue;
                            }
                            let d = dist3(ai.coords, aj.coords);
                            let rsum = ri + vdw_radius(&aj.element);
                            let overlap = rsum - d;
                            if overlap < overlap_cutoff_a {
                                continue;
                            }
                            if within_bond_hops(&adj, i, j, 4) {
                                continue;
                            }
                            // Skip legitimate strong H-bonds (polar-polar >= 2.35 A)
                            if is_polar_hbond_atom(&ai.element)
                                && is_polar_hbond_atom(&aj.element)
                                && d >= 2.35
                            {
                                continue;
                            }
                            if seen.insert((i, j)) {
                                out.push(ClashRecord {
                                    atom_a: i,
                                    atom_b: j,
                                    label_a: atom_label(mol, i),
                                    label_b: atom_label(mol, j),
                                    coords_a: ai.coords,
                                    coords_b: aj.coords,
                                    distance_a: (d * 100.0).round() / 100.0,
                                    overlap_a: (overlap * 100.0).round() / 100.0,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.overlap_a.total_cmp(&a.overlap_a));
    out
}

/// Detect non-covalent heavy-atom contacts within `max_dist_a` (default `4.0 Å`, ChimeraX `contacts`).
#[must_use]
pub fn detect_contacts(
    mol: &Biomolecule,
    mask_a: Option<&[bool]>,
    mask_b: Option<&[bool]>,
    max_dist_a: f64,
    max_results: usize,
) -> Vec<ContactRecord> {
    let n = mol.atoms.len();
    if n < 2 {
        return Vec::new();
    }
    let cutoff = max_dist_a.clamp(2.0, 6.0);
    let cutoff2 = cutoff * cutoff;
    let cell = cutoff.max(3.5);
    let ckey = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell).floor() as i32,
            (c[1] / cell).floor() as i32,
            (c[2] / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        if a.res_class != ResidueClass::Water && a.element != "H" {
            grid.entry(ckey(a.coords)).or_default().push(i);
        }
    }
    let in_a = |i: usize| mask_a.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let in_b = |i: usize| mask_b.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let cross_mask = mask_a.is_some() && mask_b.is_some();

    let mut out = Vec::new();
    for (&(cx, cy, cz), bucket) in &grid {
        for &i in bucket {
            let ai = &mol.atoms[i];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(nbrs) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                            continue;
                        };
                        for &j in nbrs {
                            if i >= j {
                                continue;
                            }
                            let aj = &mol.atoms[j];
                            let pair_ok = if cross_mask {
                                (in_a(i) && in_b(j)) || (in_a(j) && in_b(i))
                            } else {
                                // Default global contacts: inter-chain or ligand-receptor contacts
                                ai.chain != aj.chain
                            };
                            if !pair_ok {
                                continue;
                            }
                            if ai.chain == aj.chain && (ai.res_seq - aj.res_seq).abs() <= 1 {
                                continue;
                            }
                            let d2 = (ai.coords[0] - aj.coords[0]).powi(2)
                                + (ai.coords[1] - aj.coords[1]).powi(2)
                                + (ai.coords[2] - aj.coords[2]).powi(2);
                            if d2 <= cutoff2 {
                                out.push(ContactRecord {
                                    atom_a: i,
                                    atom_b: j,
                                    label_a: atom_label(mol, i),
                                    label_b: atom_label(mol, j),
                                    coords_a: ai.coords,
                                    coords_b: aj.coords,
                                    distance_a: (d2.sqrt() * 100.0).round() / 100.0,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.distance_a.total_cmp(&b.distance_a));
    out.truncate(max_results.max(10));
    out
}

fn covariance_3x3(points: &[[f64; 3]]) -> ([f64; 3], [[f64; 3]; 3]) {
    let n = points.len().max(1) as f64;
    let mut c = [0.0; 3];
    for p in points {
        for k in 0..3 {
            c[k] += p[k] / n;
        }
    }
    let mut cov = [[0.0; 3]; 3];
    for p in points {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for r in 0..3 {
            for col in 0..3 {
                cov[r][col] += d[r] * d[col] / n;
            }
        }
    }
    (c, cov)
}

/// Fit the principal 3D axis to a selection of atoms (e.g. an α-helix or domain — ChimeraX `define axis`).
pub fn compute_geometry_axis(
    label: impl Into<String>,
    points: &[[f64; 3]],
) -> Result<GeometryAxis, BioStructError> {
    if points.len() < 3 {
        return Err(BioStructError::Selection(
            "define axis requires at least 3 atoms".to_string(),
        ));
    }
    let (centroid, cov) = covariance_3x3(points);
    let (evals, evecs) = jacobi_sym3(cov);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    let k0 = order[0];
    let dir = norm3_f64([evecs[0][k0], evecs[1][k0], evecs[2][k0]]);
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for p in points {
        let proj = dot3_f64(sub3(*p, centroid), dir);
        t_min = t_min.min(proj);
        t_max = t_max.max(proj);
    }
    let start = [
        centroid[0] + dir[0] * t_min,
        centroid[1] + dir[1] * t_min,
        centroid[2] + dir[2] * t_min,
    ];
    let end = [
        centroid[0] + dir[0] * t_max,
        centroid[1] + dir[1] * t_max,
        centroid[2] + dir[2] * t_max,
    ];
    let sum_ev = (evals[0].max(0.0) + evals[1].max(0.0) + evals[2].max(0.0)).max(1e-9);
    Ok(GeometryAxis {
        label: label.into(),
        centroid,
        direction: dir,
        start,
        end,
        length_a: ((t_max - t_min) * 100.0).round() / 100.0,
        linearity_ratio: ((evals[k0].max(0.0) / sum_ev) * 1000.0).round() / 1000.0,
    })
}

/// Fit the best-fit 3D plane to a selection of atoms (e.g. aromatic ring / interface — ChimeraX `define plane`).
pub fn compute_best_fit_plane(
    label: impl Into<String>,
    points: &[[f64; 3]],
) -> Result<GeometryPlane, BioStructError> {
    if points.len() < 3 {
        return Err(BioStructError::Selection(
            "define plane requires at least 3 atoms".to_string(),
        ));
    }
    let (centroid, cov) = covariance_3x3(points);
    let (evals, evecs) = jacobi_sym3(cov);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| evals[a].total_cmp(&evals[b]));
    let k_min = order[0];
    let normal = norm3_f64([evecs[0][k_min], evecs[1][k_min], evecs[2][k_min]]);
    let mut sq_dev = 0.0_f64;
    let mut max_r = 0.0_f64;
    for p in points {
        let rel = sub3(*p, centroid);
        let h = dot3_f64(rel, normal);
        sq_dev += h * h;
        let in_plane = (dot3_f64(rel, rel) - h * h).max(0.0).sqrt();
        max_r = max_r.max(in_plane);
    }
    let rms = (sq_dev / (points.len() as f64)).sqrt();
    Ok(GeometryPlane {
        label: label.into(),
        centroid,
        normal,
        radius_a: (max_r * 100.0).round() / 100.0,
        rms_deviation_a: (rms * 1000.0).round() / 1000.0,
    })
}

fn pick_primary_polymer_chain(mol: &Biomolecule) -> Option<&crate::model::Chain> {
    mol.chains
        .iter()
        .filter(|c| {
            matches!(
                c.dominant_class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            ) && !c.sequence.is_empty()
        })
        .max_by_key(|c| c.sequence.len())
}

/// Build a synchronized multi-model MSA (`Match->Align`) across loaded biomolecular models
/// (aligns each model's primary polymer chain onto the first model's primary polymer chain).
#[must_use]
pub fn build_synchronized_msa(models: &[(&str, &Biomolecule)]) -> MsaAlignment {
    let Some((ref_name, ref_mol)) = models.first() else {
        return MsaAlignment {
            num_columns: 0,
            rows: Vec::new(),
            column_conservation: Vec::new(),
        };
    };
    let Some(ref_ch) = pick_primary_polymer_chain(ref_mol) else {
        return MsaAlignment {
            num_columns: 0,
            rows: Vec::new(),
            column_conservation: Vec::new(),
        };
    };
    let ref_res: Vec<(char, i32)> = ref_ch
        .residues
        .iter()
        .filter(|r| {
            matches!(
                r.class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            )
        })
        .map(|r| (r.code, r.res_seq))
        .collect();
    let cols = ref_res.len();
    let ref_seq: String = ref_res.iter().map(|&(c, _)| c).collect();

    let mut rows = vec![MsaRow {
        model_name: (*ref_name).to_string(),
        chain_id: ref_ch.id,
        aligned_chars: ref_res.iter().map(|&(c, _)| c).collect(),
        res_seqs: ref_res.iter().map(|&(_, s)| Some(s)).collect(),
    }];

    for &(mname, mol) in models.iter().skip(1) {
        let Some(ch) = pick_primary_polymer_chain(mol) else {
            continue;
        };
        let m_res: Vec<(char, i32)> = ch
            .residues
            .iter()
            .filter(|r| {
                matches!(
                    r.class,
                    ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
                )
            })
            .map(|r| (r.code, r.res_seq))
            .collect();
        let m_seq: String = m_res.iter().map(|&(c, _)| c).collect();
        let pairs = smith_waterman_pairs(&m_seq, &ref_seq);
        let mut aligned_chars = vec!['-'; cols];
        let mut res_seqs = vec![None; cols];
        for (mi, ri) in pairs {
            if ri < cols && mi < m_res.len() {
                aligned_chars[ri] = m_res[mi].0;
                res_seqs[ri] = Some(m_res[mi].1);
            }
        }
        rows.push(MsaRow {
            model_name: mname.to_string(),
            chain_id: ch.id,
            aligned_chars,
            res_seqs,
        });
    }

    let mut column_conservation = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut counts: HashMap<char, usize> = HashMap::new();
        let mut non_gap = 0usize;
        for row in &rows {
            let ch = row.aligned_chars[c];
            if ch != '-' {
                *counts.entry(ch).or_insert(0) += 1;
                non_gap += 1;
            }
        }
        let top = counts.values().copied().max().unwrap_or(0);
        let frac = if non_gap > 0 {
            (top as f32) / (rows.len().max(1) as f32)
        } else {
            0.0
        };
        column_conservation.push(frac);
    }

    MsaAlignment {
        num_columns: cols,
        rows,
        column_conservation,
    }
}

fn segment_pae_domains(n: usize, vals: &[f32]) -> Vec<(usize, usize)> {
    if n < 8 {
        return if n > 0 { vec![(0, n - 1)] } else { Vec::new() };
    }
    let mut blocks = Vec::new();
    let mut start = 0usize;
    let win = 6usize.min(n / 2);
    let mut i = win;
    while i + win < n {
        // Mean cross-boundary PAE between [i-win..i] and [i..i+win]
        let mut sum = 0.0_f32;
        let mut cnt = 0usize;
        for a in (i - win)..i {
            for b in i..(i + win) {
                sum += vals[a * n + b];
                cnt += 1;
            }
        }
        let cross = sum / (cnt.max(1) as f32);
        if cross > 11.5 && i.saturating_sub(start) >= 15 {
            blocks.push((start, i - 1));
            start = i;
            i += 10;
        } else {
            i += 2;
        }
    }
    blocks.push((start, n - 1));
    blocks
}

/// Parse an AlphaFoldDB / AlphaFold 3 / Boltz-2 / Chai-1 PAE JSON file into a `PaeMatrix`.
pub fn parse_pae_json(label: &str, raw: &str) -> Result<PaeMatrix, BioStructError> {
    let val: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| BioStructError::Parse(format!("PAE JSON parse error: {e}")))?;

    // Support AlphaFoldDB `[ { "predicted_aligned_error": [[...]] } ]`, AF3 `{ "pae": [[...]] }`,
    // and Boltz/Chai `{ "predicted_aligned_error": [[...]] }`
    let matrix_val = if let Some(arr) = val.as_array() {
        arr.first().and_then(|o| {
            o.get("predicted_aligned_error")
                .or_else(|| o.get("pae"))
                .or_else(|| o.get("distance"))
        })
    } else {
        val.get("predicted_aligned_error")
            .or_else(|| val.get("pae"))
            .or_else(|| val.get("pae_matrix"))
    }
    .ok_or_else(|| {
        BioStructError::Parse("no 'predicted_aligned_error' or 'pae' 2D array found".to_string())
    })?;

    let rows = matrix_val
        .as_array()
        .ok_or_else(|| BioStructError::Parse("PAE field is not a 2D array".to_string()))?;
    let n = rows.len().min(2048);
    if n == 0 {
        return Err(BioStructError::EmptyInput);
    }
    let mut values = Vec::with_capacity(n * n);
    let mut max_v = 0.0_f32;
    let mut sum = 0.0_f64;
    for r in rows.iter().take(n) {
        let cols = r
            .as_array()
            .ok_or_else(|| BioStructError::Parse("PAE row is not an array".to_string()))?;
        for c in 0..n {
            let v = cols
                .get(c)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(30.0) as f32;
            max_v = max_v.max(v);
            sum += f64::from(v);
            values.push(v);
        }
    }
    let domain_blocks = segment_pae_domains(n, &values);
    Ok(PaeMatrix {
        label: label.to_string(),
        length: n,
        mean_pae: (sum / ((n * n) as f64)) as f32,
        max_pae: max_v.max(1.0),
        values,
        is_experimental_json: true,
        domain_blocks,
    })
}

/// Build a structural distance-uncertainty PAE proxy matrix directly from any loaded `Biomolecule`
/// (uses 3D Cα/C4' packing + B-factor/pLDDT confidence so domain architecture is immediately inspectable!).
#[must_use]
pub fn estimate_pae_proxy_from_structure(mol: &Biomolecule) -> PaeMatrix {
    let guides: Vec<([f64; 3], f64, char)> = mol
        .chains
        .iter()
        .flat_map(|ch| ch.residues.iter())
        .filter(|r| {
            matches!(
                r.class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            )
        })
        .filter_map(|r| {
            let gi = r.guide_idx?;
            Some((mol.atoms[gi].coords, r.mean_bfactor, r.chain))
        })
        .take(512)
        .collect();
    let n = guides.len();
    if n == 0 {
        return PaeMatrix {
            label: mol.id.clone(),
            length: 0,
            values: Vec::new(),
            max_pae: 30.0,
            mean_pae: 0.0,
            is_experimental_json: false,
            domain_blocks: Vec::new(),
        };
    }
    let mut values = Vec::with_capacity(n * n);
    let mut sum = 0.0_f64;
    let mut max_v = 0.0_f32;
    for i in 0..n {
        for j in 0..n {
            let d = dist3(guides[i].0, guides[j].0);
            let same_chain = guides[i].2 == guides[j].2;
            let seq_gap = i.abs_diff(j) as f64;
            let base_err = if i == j {
                0.5
            } else if same_chain && d < 12.0 {
                (1.0 + d * 0.28).min(5.5)
            } else if same_chain {
                (2.0 + d * 0.48 + (seq_gap * 0.03).min(6.0)).min(26.0)
            } else if d < 14.0 {
                (4.0 + d * 0.55).min(14.0)
            } else {
                (10.0 + d * 0.45).min(30.0)
            };
            let v = base_err as f32;
            max_v = max_v.max(v);
            sum += f64::from(v);
            values.push(v);
        }
    }
    let domain_blocks = segment_pae_domains(n, &values);
    PaeMatrix {
        label: format!("{} (3D Domain Packing Proxy)", mol.id),
        length: n,
        mean_pae: (sum / ((n * n) as f64)) as f32,
        max_pae: max_v.max(1.0),
        values,
        is_experimental_json: false,
        domain_blocks,
    }
}

/// Solvent-accessible surface area (SASA) & burial report for a selected subset of atoms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SasaBurialReport {
    pub atom_count: usize,
    pub isolated_sasa_a2: f64,
    pub complex_sasa_a2: f64,
    pub buried_sasa_a2: f64,
    pub buried_fraction_pct: f64,
}

/// Compute Shrake-Rupley SASA (`1.4 Å` probe, 92 Fibonacci points) of `target_mask` both
/// isolated (alone in solvent) and in the presence of `context_mask` (or all non-water atoms
/// of `mol` when `context_mask` is `None`), reporting buried surface area (`ΔSASA` & `% buried`).
#[must_use]
pub fn compute_selection_sasa(
    mol: &Biomolecule,
    target_mask: &[bool],
    context_mask: Option<&[bool]>,
) -> SasaBurialReport {
    let mut target_atoms = Vec::new();
    let mut complex_atoms = Vec::new();
    let mut target_in_complex_indices = Vec::new();

    for (i, a) in mol.atoms.iter().enumerate() {
        let is_target = target_mask.get(i).copied().unwrap_or(false);
        let is_ctx = match context_mask {
            Some(cm) => cm.get(i).copied().unwrap_or(false),
            None => a.res_class != ResidueClass::Water,
        };
        if is_target {
            target_atoms.push(a.clone());
            target_in_complex_indices.push(complex_atoms.len());
            complex_atoms.push(a.clone());
        } else if is_ctx {
            complex_atoms.push(a.clone());
        }
    }

    if target_atoms.is_empty() {
        return SasaBurialReport {
            atom_count: 0,
            isolated_sasa_a2: 0.0,
            complex_sasa_a2: 0.0,
            buried_sasa_a2: 0.0,
            buried_fraction_pct: 0.0,
        };
    }

    let sasa_iso: f64 = compute_sasa(&target_atoms, 1.4, 92).iter().sum();
    let sasa_all = compute_sasa(&complex_atoms, 1.4, 92);
    let sasa_cpx: f64 = target_in_complex_indices
        .iter()
        .filter_map(|&idx| sasa_all.get(idx).copied())
        .sum();
    let buried = (sasa_iso - sasa_cpx).max(0.0);
    let pct = if sasa_iso > 1e-6 {
        (buried / sasa_iso * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    let r2 = |v: f64| (v * 100.0).round() / 100.0;
    SasaBurialReport {
        atom_count: target_atoms.len(),
        isolated_sasa_a2: r2(sasa_iso),
        complex_sasa_a2: r2(sasa_cpx),
        buried_sasa_a2: r2(buried),
        buried_fraction_pct: (pct * 10.0).round() / 10.0,
    }
}

/// Detect geometric hydrogen bonds strictly BETWEEN two distinct models (`interModel true`).
#[must_use]
pub fn detect_hbonds_between(
    mol_a: &Biomolecule,
    mask_a: Option<&[bool]>,
    mol_b: &Biomolecule,
    mask_b: Option<&[bool]>,
    max_dist_a: f64,
    min_angle_deg: f64,
) -> Vec<HBondRecord> {
    if mol_a.atoms.is_empty() || mol_b.atoms.is_empty() {
        return Vec::new();
    }
    let adj_a = build_adjacency(mol_a);
    let adj_b = build_adjacency(mol_b);
    let cutoff = max_dist_a.clamp(2.2, 4.2);
    let cutoff2 = cutoff * cutoff;
    let cell = cutoff.max(3.5);
    let ckey = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell).floor() as i32,
            (c[1] / cell).floor() as i32,
            (c[2] / cell).floor() as i32,
        )
    };
    let in_a = |i: usize| mask_a.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let in_b = |i: usize| mask_b.is_none_or(|m| m.get(i).copied().unwrap_or(false));

    let mut grid_b: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (j, b) in mol_b.atoms.iter().enumerate() {
        if in_b(j) && is_polar_hbond_atom(&b.element) && b.res_class != ResidueClass::Water {
            grid_b.entry(ckey(b.coords)).or_default().push(j);
        }
    }

    let mut out = Vec::new();
    for (i, a) in mol_a.atoms.iter().enumerate() {
        if !in_a(i) || !is_polar_hbond_atom(&a.element) || a.res_class == ResidueClass::Water {
            continue;
        }
        let (cx, cy, cz) = ckey(a.coords);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(nbrs) = grid_b.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &j in nbrs {
                        let b = &mol_b.atoms[j];
                        let d2 = (a.coords[0] - b.coords[0]).powi(2)
                            + (a.coords[1] - b.coords[1]).powi(2)
                            + (a.coords[2] - b.coords[2]).powi(2);
                        if d2 < 2.2 * 2.2 || d2 > cutoff2 {
                            continue;
                        }
                        let d = d2.sqrt();
                        let ang_a = adj_a[i]
                            .first()
                            .map(|&bi| measure_angle_deg(mol_a.atoms[bi].coords, a.coords, b.coords))
                            .unwrap_or(140.0);
                        let ang_b = adj_b[j]
                            .first()
                            .map(|&bj| measure_angle_deg(mol_b.atoms[bj].coords, b.coords, a.coords))
                            .unwrap_or(140.0);
                        let best_ang = ang_a.min(ang_b);
                        if best_ang < min_angle_deg {
                            continue;
                        }
                        let dist_f = (1.0 - ((d - 2.85).abs() / 0.95)).clamp(0.15, 1.0);
                        let ang_f = ((best_ang - 90.0) / 80.0).clamp(0.2, 1.0);
                        let energy = -((2.8 * dist_f * ang_f) * 100.0).round() / 100.0;
                        out.push(HBondRecord {
                            atom_a: i,
                            atom_b: j,
                            label_a: atom_label(mol_a, i),
                            label_b: atom_label(mol_b, j),
                            coords_a: a.coords,
                            coords_b: b.coords,
                            distance_a: (d * 100.0).round() / 100.0,
                            angle_deg: (best_ang * 10.0).round() / 10.0,
                            energy_kcal: energy,
                        });
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.distance_a.total_cmp(&b.distance_a));
    out
}

/// Detect steric atomic clashes strictly BETWEEN two distinct models (`interModel true`).
#[must_use]
pub fn detect_clashes_between(
    mol_a: &Biomolecule,
    mask_a: Option<&[bool]>,
    mol_b: &Biomolecule,
    mask_b: Option<&[bool]>,
    overlap_cutoff_a: f64,
) -> Vec<ClashRecord> {
    if mol_a.atoms.is_empty() || mol_b.atoms.is_empty() {
        return Vec::new();
    }
    let cell = 4.0_f64;
    let ckey = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell).floor() as i32,
            (c[1] / cell).floor() as i32,
            (c[2] / cell).floor() as i32,
        )
    };
    let in_a = |i: usize| mask_a.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let in_b = |i: usize| mask_b.is_none_or(|m| m.get(i).copied().unwrap_or(false));

    let mut grid_b: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (j, b) in mol_b.atoms.iter().enumerate() {
        if in_b(j) && b.res_class != ResidueClass::Water && b.element != "H" {
            grid_b.entry(ckey(b.coords)).or_default().push(j);
        }
    }

    let mut out = Vec::new();
    for (i, a) in mol_a.atoms.iter().enumerate() {
        if !in_a(i) || a.res_class == ResidueClass::Water || a.element == "H" {
            continue;
        }
        let ra = vdw_radius(&a.element);
        let (cx, cy, cz) = ckey(a.coords);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(nbrs) = grid_b.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &j in nbrs {
                        let b = &mol_b.atoms[j];
                        let d = dist3(a.coords, b.coords);
                        let overlap = (ra + vdw_radius(&b.element)) - d;
                        if overlap < overlap_cutoff_a {
                            continue;
                        }
                        if is_polar_hbond_atom(&a.element)
                            && is_polar_hbond_atom(&b.element)
                            && d >= 2.35
                        {
                            continue;
                        }
                        out.push(ClashRecord {
                            atom_a: i,
                            atom_b: j,
                            label_a: atom_label(mol_a, i),
                            label_b: atom_label(mol_b, j),
                            coords_a: a.coords,
                            coords_b: b.coords,
                            distance_a: (d * 100.0).round() / 100.0,
                            overlap_a: (overlap * 100.0).round() / 100.0,
                        });
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.overlap_a.total_cmp(&a.overlap_a));
    out
}

/// Detect non-covalent heavy-atom contacts strictly BETWEEN two distinct models (`interModel true`).
#[must_use]
pub fn detect_contacts_between(
    mol_a: &Biomolecule,
    mask_a: Option<&[bool]>,
    mol_b: &Biomolecule,
    mask_b: Option<&[bool]>,
    max_dist_a: f64,
    max_results: usize,
) -> Vec<ContactRecord> {
    if mol_a.atoms.is_empty() || mol_b.atoms.is_empty() {
        return Vec::new();
    }
    let cutoff = max_dist_a.clamp(2.0, 6.0);
    let cutoff2 = cutoff * cutoff;
    let cell = cutoff.max(3.5);
    let ckey = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell).floor() as i32,
            (c[1] / cell).floor() as i32,
            (c[2] / cell).floor() as i32,
        )
    };
    let in_a = |i: usize| mask_a.is_none_or(|m| m.get(i).copied().unwrap_or(false));
    let in_b = |i: usize| mask_b.is_none_or(|m| m.get(i).copied().unwrap_or(false));

    let mut grid_b: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (j, b) in mol_b.atoms.iter().enumerate() {
        if in_b(j) && b.res_class != ResidueClass::Water && b.element != "H" {
            grid_b.entry(ckey(b.coords)).or_default().push(j);
        }
    }

    let mut out = Vec::new();
    for (i, a) in mol_a.atoms.iter().enumerate() {
        if !in_a(i) || a.res_class == ResidueClass::Water || a.element == "H" {
            continue;
        }
        let (cx, cy, cz) = ckey(a.coords);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(nbrs) = grid_b.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &j in nbrs {
                        let b = &mol_b.atoms[j];
                        let d2 = (a.coords[0] - b.coords[0]).powi(2)
                            + (a.coords[1] - b.coords[1]).powi(2)
                            + (a.coords[2] - b.coords[2]).powi(2);
                        if d2 <= cutoff2 {
                            out.push(ContactRecord {
                                atom_a: i,
                                atom_b: j,
                                label_a: atom_label(mol_a, i),
                                label_b: atom_label(mol_b, j),
                                coords_a: a.coords,
                                coords_b: b.coords,
                                distance_a: (d2.sqrt() * 100.0).round() / 100.0,
                            });
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.distance_a.total_cmp(&b.distance_a));
    out.truncate(max_results);
    out
}
