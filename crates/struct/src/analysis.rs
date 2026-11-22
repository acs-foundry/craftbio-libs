#![forbid(unsafe_code)]

//! Biomolecular Geometry & Biophysics:
//! - DSSP-class Secondary Structure Perception (`Helix`, `Sheet`, `Coil`, `NucleicRibbon`)
//! - Rayon-Parallel Shrake-Rupley 3D SASA
//! - Chennamsetty 3D SAP Aggregation Propensity
//! - Debye-Hückel Screened Coulomb 3D Electrostatic Potential (`ESP`)

use std::collections::HashMap;
use std::f64::consts::PI;

use rayon::prelude::*;

use crate::model::{
    Atom, Biomolecule, ResidueClass, SecondaryStructure, black_mould_sap_hydro, dist3,
    max_residue_sasa, vdw_radius,
};

/// Assign DSSP-class secondary structure (`Helix`, `Sheet`, `Coil`, `NucleicRibbon`) directly onto
/// `mol.chains[*].residues[*].ss` using backbone H-bond geometry and Cα spacing.
pub fn assign_secondary_structure(mol: &mut Biomolecule) {
    let atoms = &mol.atoms;
    for ch in &mut mol.chains {
        let n = ch.residues.len();
        for r in &mut ch.residues {
            r.ss = if matches!(r.class, ResidueClass::Rna | ResidueClass::Dna) {
                SecondaryStructure::NucleicRibbon
            } else {
                SecondaryStructure::Coil
            };
        }
        if ch.dominant_class != ResidueClass::Protein || n < 5 {
            continue;
        }

        // Gather N, CA, O coordinates per residue
        let mut n_pos: Vec<Option<[f64; 3]>> = vec![None; n];
        let mut ca_pos: Vec<Option<[f64; 3]>> = vec![None; n];
        let mut o_pos: Vec<Option<[f64; 3]>> = vec![None; n];
        for (i, r) in ch.residues.iter().enumerate() {
            if r.class != ResidueClass::Protein {
                continue;
            }
            for &ai in &r.atom_indices {
                match atoms[ai].name.as_str() {
                    "N" => n_pos[i] = Some(atoms[ai].coords),
                    "CA" => ca_pos[i] = Some(atoms[ai].coords),
                    "O" => o_pos[i] = Some(atoms[ai].coords),
                    _ => {}
                }
            }
        }

        // 1. Alpha-helix detection: i -> i+4 O..N H-bond <= 3.55 A OR tight helical CA(i)..CA(i+3) in 4.6..6.1 A
        let mut is_helix = vec![false; n];
        for i in 0..n.saturating_sub(4) {
            let hb4 =
                matches!((o_pos[i], n_pos[i + 4]), (Some(o), Some(ni)) if dist3(o, ni) <= 3.55);
            let ca3 = matches!((ca_pos[i], ca_pos[i + 3]), (Some(a), Some(b)) if (4.5..=6.0).contains(&dist3(a, b)));
            let ca4 = matches!((ca_pos[i], ca_pos[i + 4]), (Some(a), Some(b)) if (5.4..=6.9).contains(&dist3(a, b)));
            if hb4 && (ca3 || ca4) {
                for flag in &mut is_helix[i..=(i + 4).min(n - 1)] {
                    *flag = true;
                }
            }
        }

        // 2. Beta-sheet detection: non-local backbone O..N H-bond (|i-j| >= 4) + extended CA(i)..CA(i+2) >= 5.9 A
        let mut is_sheet = vec![false; n];
        for i in 0..n {
            if is_helix[i] {
                continue;
            }
            let extended = i + 2 < n
                && matches!((ca_pos[i], ca_pos[i + 2]), (Some(a), Some(b)) if dist3(a, b) >= 5.9);
            if !extended {
                continue;
            }
            let Some(oi) = o_pos[i] else { continue };
            let Some(ni) = n_pos[i] else { continue };
            for j in 0..n {
                if i.abs_diff(j) < 4 || is_helix[j] {
                    continue;
                }
                let hb_fwd = n_pos[j].is_some_and(|nj| dist3(oi, nj) <= 3.55);
                let hb_rev = o_pos[j].is_some_and(|oj| dist3(ni, oj) <= 3.55);
                if hb_fwd || hb_rev {
                    is_sheet[i] = true;
                    if i + 1 < n && !is_helix[i + 1] {
                        is_sheet[i + 1] = true;
                    }
                    break;
                }
            }
        }

        for i in 0..n {
            if ch.residues[i].class == ResidueClass::Protein {
                ch.residues[i].ss = if is_helix[i] {
                    SecondaryStructure::Helix
                } else if is_sheet[i] {
                    SecondaryStructure::Sheet
                } else {
                    SecondaryStructure::Coil
                };
            }
        }
    }
}

fn fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let y = 1.0 - (2.0 * (i as f64) + 1.0) / (n as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * (i as f64);
        pts.push([theta.cos() * r, y, theta.sin() * r]);
    }
    pts
}

/// Rayon-parallel Shrake-Rupley 3D SASA (Å²) per atom with spatial cell-list acceleration.
#[must_use]
pub fn compute_sasa(atoms: &[Atom], probe_radius: f64, n_points: usize) -> Vec<f64> {
    let n = atoms.len();
    if n == 0 {
        return Vec::new();
    }
    let sphere = fibonacci_sphere(n_points.max(16));
    let radii: Vec<f64> = atoms
        .iter()
        .map(|a| vdw_radius(&a.element) + probe_radius)
        .collect();
    let max_r = radii.iter().copied().fold(0.0_f64, f64::max);
    let cell_size = (2.0 * max_r).max(6.4);

    let cell_of = |c: [f64; 3]| -> (i32, i32, i32) {
        (
            (c[0] / cell_size).floor() as i32,
            (c[1] / cell_size).floor() as i32,
            (c[2] / cell_size).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(n / 4 + 1);
    for (idx, a) in atoms.iter().enumerate() {
        grid.entry(cell_of(a.coords)).or_default().push(idx);
    }

    (0..n)
        .into_par_iter()
        .map(|i| {
            let pi = atoms[i].coords;
            let ri = radii[i];
            let (cx, cy, cz) = cell_of(pi);
            let mut nbrs: Vec<(usize, f64)> = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in bucket {
                                if j == i {
                                    continue;
                                }
                                let rsum = ri + radii[j];
                                let d2 = (atoms[j].coords[0] - pi[0]).powi(2)
                                    + (atoms[j].coords[1] - pi[1]).powi(2)
                                    + (atoms[j].coords[2] - pi[2]).powi(2);
                                if d2 < rsum * rsum {
                                    nbrs.push((j, radii[j] * radii[j]));
                                }
                            }
                        }
                    }
                }
            }
            let mut exposed = 0usize;
            for &u in &sphere {
                let q = [pi[0] + ri * u[0], pi[1] + ri * u[1], pi[2] + ri * u[2]];
                let mut occluded = false;
                for &(j, rj2) in &nbrs {
                    let pj = atoms[j].coords;
                    let d2 =
                        (q[0] - pj[0]).powi(2) + (q[1] - pj[1]).powi(2) + (q[2] - pj[2]).powi(2);
                    if d2 < rj2 {
                        occluded = true;
                        break;
                    }
                }
                if !occluded {
                    exposed += 1;
                }
            }
            4.0 * PI * ri * ri * ((exposed as f64) / (sphere.len() as f64))
        })
        .collect()
}

/// Compute per-atom 3D SAP (Spatial Aggregation Propensity) values (`R = radius_a`, default `6.5 Å`).
#[must_use]
pub fn compute_per_atom_sap(mol: &Biomolecule, radius_a: f64) -> Vec<f32> {
    let atom_sasa = compute_sasa(&mol.atoms, 1.4, 32);
    struct RInfo {
        center: [f64; 3],
        weighted_hydro: f64,
        atom_indices: Vec<usize>,
    }
    let mut rinfos = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            let rs: f64 = r.atom_indices.iter().map(|&i| atom_sasa[i]).sum();
            let rel = (rs / max_residue_sasa(r.code)).clamp(0.0, 1.2);
            let h = if r.class == ResidueClass::Protein {
                black_mould_sap_hydro(r.code)
            } else {
                -0.25
            };
            let center = r
                .guide_idx
                .or_else(|| r.atom_indices.first().copied())
                .map_or([0.0; 3], |i| mol.atoms[i].coords);
            rinfos.push(RInfo {
                center,
                weighted_hydro: rel * h,
                atom_indices: r.atom_indices.clone(),
            });
        }
    }
    let mut out = vec![0.0_f32; mol.atoms.len()];
    let r2 = radius_a * radius_a;
    for ri in &rinfos {
        let mut sap = 0.0_f64;
        for rj in &rinfos {
            let dx = ri.center[0] - rj.center[0];
            let dy = ri.center[1] - rj.center[1];
            let dz = ri.center[2] - rj.center[2];
            if dx * dx + dy * dy + dz * dz <= r2 {
                sap += rj.weighted_hydro;
            }
        }
        for &ai in &ri.atom_indices {
            out[ai] = sap as f32;
        }
    }
    out
}

/// Evaluate Debye-Hückel screened Coulomb 3D Electrostatic Potential (`kT/e` approximate scale)
/// at arbitrary surface or atomic 3D points (`dielectric = 40.0`, `kappa = 0.125 Å^-1` ~150 mM NaCl).
#[must_use]
pub fn compute_esp_at_points(mol: &Biomolecule, points: &[[f64; 3]]) -> Vec<f32> {
    let charged: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .filter(|a| a.formal_charge.abs() > 1e-4)
        .map(|a| (a.coords, a.formal_charge))
        .collect();
    let kappa = 0.125_f64; // ~150 mM physiological ionic strength
    let scale = 332.0 / 28.0; // kcal/(mol*e) / effective boundary dielectric -> kT/e range
    points
        .par_iter()
        .map(|&p| {
            let mut phi = 0.0_f64;
            for &(c, q) in &charged {
                let d = dist3(p, c).max(1.8);
                if d <= 22.0 {
                    phi += scale * q * (-kappa * d).exp() / d;
                }
            }
            phi.clamp(-12.0, 12.0) as f32
        })
        .collect()
}

/// 3D electrostatic dipole moment and net charge summary (`1 e·Å = 4.8032 Debye`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DipoleMomentReport {
    pub net_charge_e: f64,
    pub dipole_magnitude_debye: f64,
    pub dipole_vector_debye: [f64; 3],
    pub center_of_geometry: [f64; 3],
    pub chain_charges: Vec<(char, f64)>,
}

/// Compute the 3D electrostatic dipole moment (Debye) referenced to geometric center
/// (relevant for antibody viscosity / colloidal self-association and electrostatics).
#[must_use]
pub fn compute_dipole_moment(mol: &Biomolecule) -> DipoleMomentReport {
    let center = mol.centroid();
    let mut vec_ea = [0.0_f64; 3];
    let mut net_q = 0.0_f64;
    for a in &mol.atoms {
        let q = a.formal_charge;
        net_q += q;
        if q.abs() > 1e-6 {
            vec_ea[0] += q * (a.coords[0] - center[0]);
            vec_ea[1] += q * (a.coords[1] - center[1]);
            vec_ea[2] += q * (a.coords[2] - center[2]);
        }
    }
    const E_ANGSTROM_TO_DEBYE: f64 = 4.80320425;
    let d_vec = [
        vec_ea[0] * E_ANGSTROM_TO_DEBYE,
        vec_ea[1] * E_ANGSTROM_TO_DEBYE,
        vec_ea[2] * E_ANGSTROM_TO_DEBYE,
    ];
    let mag = (d_vec[0] * d_vec[0] + d_vec[1] * d_vec[1] + d_vec[2] * d_vec[2]).sqrt();
    let chain_charges: Vec<(char, f64)> = mol
        .chains
        .iter()
        .map(|ch| {
            let q: f64 = ch
                .residues
                .iter()
                .flat_map(|r| r.atom_indices.iter())
                .map(|&idx| mol.atoms[idx].formal_charge)
                .sum();
            (ch.id, (q * 100.0).round() / 100.0)
        })
        .collect();
    DipoleMomentReport {
        net_charge_e: (net_q * 100.0).round() / 100.0,
        dipole_magnitude_debye: (mag * 100.0).round() / 100.0,
        dipole_vector_debye: [
            (d_vec[0] * 100.0).round() / 100.0,
            (d_vec[1] * 100.0).round() / 100.0,
            (d_vec[2] * 100.0).round() / 100.0,
        ],
        center_of_geometry: center,
        chain_charges,
    }
}

/// Atomic lipophilicity weight (Broto-Moreau / Wildman-Crippen inspired scale, negative = polar/hydrophilic,
/// positive = lipophilic/hydrophobic) for ChimeraX-style 3D Molecular Lipophilicity Potential (`MLP`).
#[must_use]
fn atom_lipophilicity_weight(a: &Atom) -> f64 {
    match a.res_class {
        ResidueClass::Protein => {
            let elem = a.element.as_str();
            match elem {
                "O" => -0.78,
                "N" => {
                    if a.formal_charge > 0.1 {
                        -1.15
                    } else {
                        -0.62
                    }
                }
                "S" => 0.45,
                "C" => {
                    // Distinguish aromatic / aliphatic sidechain carbons vs backbone carbonyl C
                    if a.name == "C" {
                        -0.20
                    } else if a.name == "CA" {
                        0.05
                    } else {
                        match a.res_code {
                            'F' | 'W' | 'I' | 'L' | 'V' | 'M' => 0.85,
                            'Y' | 'P' | 'A' | 'C' => 0.55,
                            'D' | 'E' | 'K' | 'R' | 'N' | 'Q' => -0.15,
                            _ => 0.25,
                        }
                    }
                }
                _ => 0.0,
            }
        }
        ResidueClass::Rna | ResidueClass::Dna => match a.element.as_str() {
            "P" | "O" => -0.95,
            "N" => -0.55,
            "C" => 0.28,
            _ => -0.2,
        },
        ResidueClass::Ion | ResidueClass::Water => -1.10,
        ResidueClass::Ligand => match a.element.as_str() {
            "C" => 0.65,
            "F" | "CL" | "BR" | "I" => 0.72,
            "S" => 0.35,
            "N" => -0.55,
            "O" => -0.72,
            "P" => -0.60,
            _ => 0.0,
        },
    }
}

/// Evaluate smooth 3D Molecular Lipophilicity Potential (`MLP`) at arbitrary 3D surface or atomic
/// points using an exponential distance kernel (`exp(-d / 2.0)` within `radius_a`, default `8.0 Å`).
/// Values typically span `[-1.0, +1.0]` (negative = hydrophilic cyan, positive = lipophilic gold).
#[must_use]
pub fn compute_mlp_at_points(mol: &Biomolecule, points: &[[f64; 3]], radius_a: f64) -> Vec<f32> {
    let srcs: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .filter(|a| a.res_class != ResidueClass::Water && a.element != "H")
        .map(|a| (a.coords, atom_lipophilicity_weight(a)))
        .collect();
    if srcs.is_empty() {
        return vec![0.0; points.len()];
    }
    let r_lim = radius_a.clamp(3.0, 14.0);
    let r_lim2 = r_lim * r_lim;
    points
        .par_iter()
        .map(|&p| {
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for &(c, f_i) in &srcs {
                let dx = p[0] - c[0];
                let dy = p[1] - c[1];
                let dz = p[2] - c[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= r_lim2 {
                    let w = (-d2.sqrt() * 0.5).exp();
                    num += f_i * w;
                    den += w;
                }
            }
            if den > 1e-8 {
                (num / den).clamp(-1.2, 1.2) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute per-atom 3D Molecular Lipophilicity Potential (`MLP`) for ChimeraX-style `color mlp`.
#[must_use]
pub fn compute_per_atom_mlp(mol: &Biomolecule, radius_a: f64) -> Vec<f32> {
    let pts: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
    compute_mlp_at_points(mol, &pts, radius_a)
}
