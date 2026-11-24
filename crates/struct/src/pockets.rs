#![forbid(unsafe_code)]

//! Pure-Rust 3D Cavity & Druggable Binding Pocket Detector (`fpocket` / `LIGSITE` / `CASTp`-style).
//!
//! Computes buried solvent-accessible grid cavities on the macromolecular receptor (excluding
//! waters and ligands so both apo and holo PDB/mmCIF complexes locate true active sites),
//! clusters 6-connected cavity voxels, scores volume (`Å³`), enclosure, hydrophobic/polar balance,
//! fpocket-like druggability (`0..1`), lining residues, and ready-to-dock `GridBox` coordinates.

use std::collections::{BTreeSet, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::model::{Biomolecule, ResidueClass, vdw_radius};

/// Detected 3D biomolecular cavity / druggable binding pocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingPocket3D {
    /// 1-based pocket rank ordered by composite druggability & volume score.
    pub pocket_id: usize,
    /// Geometric centroid of cavity voxels (Å).
    pub center: [f64; 3],
    /// Ready-to-dock bounding box dimensions `(sx, sy, sz)` in Ångströms (padded for DockCraft).
    pub size: [f64; 3],
    /// Estimated solvent-accessible cavity volume (`Å³`).
    pub volume_a3: f64,
    /// Mean 14-ray directional burial enclosure (`0.0 = flat`, `1.0 = fully enclosed`).
    pub enclosure: f64,
    /// Fraction of lining residues that are hydrophobic (`ALA/VAL/ILE/LEU/MET/PHE/TRP/PRO/TYR`).
    pub hydrophobic_fraction: f64,
    /// Fraction of lining residues that are polar/charged (`SER/THR/ASN/GLN/ASP/GLU/LYS/ARG/HIS/CYS`).
    pub polar_fraction: f64,
    /// Composite druggability score in `0.0..=1.0` (`>= 0.55` high druggability).
    pub druggability_score: f64,
    /// Lining macromolecular residues (e.g. `"A:LEU83"`, `"A:GLU81"`, `"A:ASP145"`).
    pub lining_residues: Vec<String>,
    /// Voxel spacing (`Å`) used when sampling `cavity_voxels`.
    pub voxel_step_a: f32,
    /// 3D world-space centers of cavity voxels for smooth Marching-Tetrahedra surface rendering.
    pub cavity_voxels: Vec<[f32; 3]>,
}

const DIRS_14: [[i32; 3]; 14] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 1],
    [-1, -1, -1],
    [1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
];

fn is_hydrophobic_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "ALA" | "VAL" | "ILE" | "LEU" | "MET" | "MSE" | "PHE" | "TRP" | "PRO" | "TYR"
    )
}

fn is_polar_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "SER"
            | "THR"
            | "ASN"
            | "GLN"
            | "ASP"
            | "GLU"
            | "LYS"
            | "ARG"
            | "HIS"
            | "HID"
            | "HIE"
            | "HIP"
            | "CYS"
    )
}

/// Detect and rank 3D binding cavities on `mol` (ignoring waters and small-molecule ligands).
#[must_use]
pub fn detect_binding_pockets(mol: &Biomolecule, max_pockets: usize) -> Vec<BindingPocket3D> {
    let limit = max_pockets.clamp(1, 16);
    // Collect macromolecular receptor atoms (Protein / RNA / DNA / catalytic Ions)
    let mut rec_indices: Vec<usize> = Vec::with_capacity(mol.atoms.len());
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (idx, a) in mol.atoms.iter().enumerate() {
        if matches!(a.res_class, ResidueClass::Water | ResidueClass::Ligand) {
            continue;
        }
        rec_indices.push(idx);
        for k in 0..3 {
            lo[k] = lo[k].min(a.coords[k]);
            hi[k] = hi[k].max(a.coords[k]);
        }
    }
    if rec_indices.len() < 12 {
        return Vec::new();
    }

    let pad = 2.0_f64;
    for k in 0..3 {
        lo[k] -= pad;
        hi[k] += pad;
    }
    let span = [
        (hi[0] - lo[0]).max(4.0),
        (hi[1] - lo[1]).max(4.0),
        (hi[2] - lo[2]).max(4.0),
    ];

    // Pick adaptive grid spacing targeting <= 340k voxels (1.15 Å on typical kinases/enzymes)
    let raw_vol = span[0] * span[1] * span[2];
    let step = (raw_vol / 320_000.0).cbrt().clamp(1.15, 2.25);
    let nx = ((span[0] / step).ceil() as usize).clamp(8, 128);
    let ny = ((span[1] / step).ceil() as usize).clamp(8, 128);
    let nz = ((span[2] / step).ceil() as usize).clamp(8, 128);
    let total = nx * ny * nz;

    // Grid state: 0 = bulk exterior, 1 = receptor occupied, 2 = solvent-accessible surface shell
    let mut grid = vec![0u8; total];
    let idx3 = |x: usize, y: usize, z: usize| -> usize { (z * ny + y) * nx + x };

    let probe_occ = 1.15_f64;
    let shell_pad = 3.25_f64;

    for &ai in &rec_indices {
        let a = &mol.atoms[ai];
        let vdw = vdw_radius(&a.element);
        let r_occ = vdw + probe_occ;
        let r_shell = vdw + shell_pad;
        let r_occ2 = r_occ * r_occ;
        let r_shell2 = r_shell * r_shell;

        let gx0 = (((a.coords[0] - r_shell - lo[0]) / step).floor() as isize)
            .clamp(0, nx as isize - 1) as usize;
        let gx1 = (((a.coords[0] + r_shell - lo[0]) / step).ceil() as isize)
            .clamp(0, nx as isize - 1) as usize;
        let gy0 = (((a.coords[1] - r_shell - lo[1]) / step).floor() as isize)
            .clamp(0, ny as isize - 1) as usize;
        let gy1 = (((a.coords[1] + r_shell - lo[1]) / step).ceil() as isize)
            .clamp(0, ny as isize - 1) as usize;
        let gz0 = (((a.coords[2] - r_shell - lo[2]) / step).floor() as isize)
            .clamp(0, nz as isize - 1) as usize;
        let gz1 = (((a.coords[2] + r_shell - lo[2]) / step).ceil() as isize)
            .clamp(0, nz as isize - 1) as usize;

        for z in gz0..=gz1 {
            let dz = (lo[2] + (z as f64 + 0.5) * step) - a.coords[2];
            let dz2 = dz * dz;
            if dz2 > r_shell2 {
                continue;
            }
            for y in gy0..=gy1 {
                let dy = (lo[1] + (y as f64 + 0.5) * step) - a.coords[1];
                let dyz2 = dz2 + dy * dy;
                if dyz2 > r_shell2 {
                    continue;
                }
                for x in gx0..=gx1 {
                    let dx = (lo[0] + (x as f64 + 0.5) * step) - a.coords[0];
                    let d2 = dyz2 + dx * dx;
                    if d2 <= r_occ2 {
                        grid[idx3(x, y, z)] = 1;
                    } else if d2 <= r_shell2 {
                        let cell = &mut grid[idx3(x, y, z)];
                        if *cell == 0 {
                            *cell = 2;
                        }
                    }
                }
            }
        }
    }

    // 14-direction LIGSITEcsc burial raycast on solvent-shell voxels (`grid == 2`)
    let max_ray_steps = ((9.5 / step).ceil() as i32).clamp(4, 11);
    let mut raw_cavity = vec![false; total];
    let mut voxel_enc = vec![0.0_f32; total];
    let mut voxel_psp = vec![0u8; total];

    for z in 1..(nz.saturating_sub(1)) {
        for y in 1..(ny.saturating_sub(1)) {
            for x in 1..(nx.saturating_sub(1)) {
                let id = idx3(x, y, z);
                if grid[id] != 2 {
                    continue;
                }
                let mut hits = [false; 14];
                let mut hit_count = 0u32;
                for (di, d) in DIRS_14.iter().enumerate() {
                    let mut cx = x as i32;
                    let mut cy = y as i32;
                    let mut cz = z as i32;
                    for _ in 0..max_ray_steps {
                        cx += d[0];
                        cy += d[1];
                        cz += d[2];
                        if cx < 0
                            || cy < 0
                            || cz < 0
                            || cx >= nx as i32
                            || cy >= ny as i32
                            || cz >= nz as i32
                        {
                            break;
                        }
                        if grid[idx3(cx as usize, cy as usize, cz as usize)] == 1 {
                            hits[di] = true;
                            hit_count += 1;
                            break;
                        }
                    }
                }
                let mut sandwiched_axes = 0u8;
                for pair in 0..7 {
                    if hits[pair * 2] && hits[pair * 2 + 1] {
                        sandwiched_axes += 1;
                    }
                }
                voxel_enc[id] = (hit_count as f32) / 14.0;
                voxel_psp[id] = sandwiched_axes;
                // Strict LIGSITEcsc PSP >= 4 + >= 71% directional burial isolates true deep pockets
                if (sandwiched_axes >= 4 && hit_count >= 10) || hit_count >= 12 {
                    raw_cavity[id] = true;
                }
            }
        }
    }

    // Morphological opening (require >= 6 cavity neighbors in 26-neighborhood) to sever
    // 1-voxel inter-cleft bridges and strip pinholes so distinct active sites stay crisp
    let mut cavity = vec![false; total];
    let mut kept_voxels = 0usize;
    for z in 1..(nz.saturating_sub(1)) {
        for y in 1..(ny.saturating_sub(1)) {
            for x in 1..(nx.saturating_sub(1)) {
                let id = idx3(x, y, z);
                if !raw_cavity[id] {
                    continue;
                }
                let mut nbrs = 0u32;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            if raw_cavity[idx3(
                                (x as i32 + dx) as usize,
                                (y as i32 + dy) as usize,
                                (z as i32 + dz) as usize,
                            )] {
                                nbrs += 1;
                            }
                        }
                    }
                }
                if nbrs >= 6 {
                    cavity[id] = true;
                    kept_voxels += 1;
                }
            }
        }
    }
    // Gentle fallback for shallow/open protein-protein clefts if strict PSP>=4 yielded < 8 voxels
    if kept_voxels < 8 {
        for z in 1..(nz.saturating_sub(1)) {
            for y in 1..(ny.saturating_sub(1)) {
                for x in 1..(nx.saturating_sub(1)) {
                    let id = idx3(x, y, z);
                    if voxel_psp[id] >= 3 && voxel_enc[id] >= 0.64 {
                        cavity[id] = true;
                    }
                }
            }
        }
    }

    // Build spatial hash of receptor atoms (5.0 Å cells) for fast lining-residue lookup
    let cell_sz = 5.0_f64;
    let mut atom_hash: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for &ai in &rec_indices {
        let c = mol.atoms[ai].coords;
        let key = (
            (c[0] / cell_sz).floor() as i32,
            (c[1] / cell_sz).floor() as i32,
            (c[2] / cell_sz).floor() as i32,
        );
        atom_hash.entry(key).or_default().push(ai);
    }

    // 6-connected flood-fill clustering of cavity voxels
    let mut visited = vec![false; total];
    let voxel_vol = step * step * step;
    let min_voxels = (45.0 / voxel_vol).ceil().max(6.0) as usize;
    let mut pockets: Vec<BindingPocket3D> = Vec::new();
    let nbr6: [[i32; 3]; 6] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];

    for z in 1..(nz.saturating_sub(1)) {
        for y in 1..(ny.saturating_sub(1)) {
            for x in 1..(nx.saturating_sub(1)) {
                let start = idx3(x, y, z);
                if !cavity[start] || visited[start] {
                    continue;
                }
                let mut queue = VecDeque::new();
                let mut cluster: Vec<(usize, usize, usize)> = Vec::new();
                visited[start] = true;
                queue.push_back((x, y, z));
                while let Some((cx, cy, cz)) = queue.pop_front() {
                    cluster.push((cx, cy, cz));
                    for d in &nbr6 {
                        let nx_i = cx as i32 + d[0];
                        let ny_i = cy as i32 + d[1];
                        let nz_i = cz as i32 + d[2];
                        if nx_i <= 0
                            || ny_i <= 0
                            || nz_i <= 0
                            || nx_i >= nx as i32 - 1
                            || ny_i >= ny as i32 - 1
                            || nz_i >= nz as i32 - 1
                        {
                            continue;
                        }
                        let (ux, uy, uz) = (nx_i as usize, ny_i as usize, nz_i as usize);
                        let nid = idx3(ux, uy, uz);
                        if cavity[nid] && !visited[nid] {
                            visited[nid] = true;
                            queue.push_back((ux, uy, uz));
                        }
                    }
                }

                if cluster.len() < min_voxels {
                    continue;
                }

                let mut sum_c = [0.0_f64; 3];
                let mut p_lo = [f64::INFINITY; 3];
                let mut p_hi = [f64::NEG_INFINITY; 3];
                let mut enc_sum = 0.0_f64;
                let mut lining_set: BTreeSet<(char, i32, String)> = BTreeSet::new();
                let mut world_pts: Vec<[f32; 3]> = Vec::with_capacity(cluster.len());

                for &(vx, vy, vz) in &cluster {
                    let wx = lo[0] + (vx as f64 + 0.5) * step;
                    let wy = lo[1] + (vy as f64 + 0.5) * step;
                    let wz = lo[2] + (vz as f64 + 0.5) * step;
                    sum_c[0] += wx;
                    sum_c[1] += wy;
                    sum_c[2] += wz;
                    p_lo[0] = p_lo[0].min(wx);
                    p_lo[1] = p_lo[1].min(wy);
                    p_lo[2] = p_lo[2].min(wz);
                    p_hi[0] = p_hi[0].max(wx);
                    p_hi[1] = p_hi[1].max(wy);
                    p_hi[2] = p_hi[2].max(wz);
                    enc_sum += f64::from(voxel_enc[idx3(vx, vy, vz)]);
                    world_pts.push([wx as f32, wy as f32, wz as f32]);

                    let hx = (wx / cell_sz).floor() as i32;
                    let hy = (wy / cell_sz).floor() as i32;
                    let hz = (wz / cell_sz).floor() as i32;
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if let Some(bucket) = atom_hash.get(&(hx + dx, hy + dy, hz + dz)) {
                                    for &ai in bucket {
                                        let a = &mol.atoms[ai];
                                        let ddx = a.coords[0] - wx;
                                        let ddy = a.coords[1] - wy;
                                        let ddz = a.coords[2] - wz;
                                        if ddx * ddx + ddy * ddy + ddz * ddz <= 23.04 {
                                            // 4.8 Å cutoff
                                            lining_set.insert((
                                                a.chain,
                                                a.res_seq,
                                                a.res_name.clone(),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let n_vox = cluster.len() as f64;
                let center = [sum_c[0] / n_vox, sum_c[1] / n_vox, sum_c[2] / n_vox];
                let size = [
                    (p_hi[0] - p_lo[0] + 8.0).clamp(16.0, 34.0),
                    (p_hi[1] - p_lo[1] + 8.0).clamp(16.0, 34.0),
                    (p_hi[2] - p_lo[2] + 8.0).clamp(16.0, 34.0),
                ];
                let volume_a3 = n_vox * voxel_vol;
                let enclosure = (enc_sum / n_vox).clamp(0.0, 1.0);

                let mut n_hydro = 0usize;
                let mut n_polar = 0usize;
                let mut lining_residues = Vec::with_capacity(lining_set.len());
                for (ch, seq, name) in &lining_set {
                    if is_hydrophobic_res(name) {
                        n_hydro += 1;
                    } else if is_polar_res(name) {
                        n_polar += 1;
                    }
                    lining_residues.push(format!("{ch}:{name}{seq}"));
                }
                let denom = lining_set.len().max(1) as f64;
                let hydrophobic_fraction = (n_hydro as f64) / denom;
                let polar_fraction = (n_polar as f64) / denom;

                // fpocket-style composite druggability score (0..1):
                // Ideal drug-like pockets have volume 220..1500 Å³, high enclosure (>= 0.62),
                // and strong apolar character (0.38..0.78) balanced with polar H-bond anchors.
                let vol_score = if volume_a3 < 200.0 {
                    (volume_a3 / 200.0).clamp(0.05, 1.0)
                } else if volume_a3 <= 1600.0 {
                    1.0
                } else {
                    (1600.0 / volume_a3).powf(0.45).clamp(0.35, 1.0)
                };
                let enc_score = ((enclosure - 0.50) / 0.35).clamp(0.0, 1.0);
                let hydro_score = if (0.32..=0.80).contains(&hydrophobic_fraction) {
                    1.0 - ((hydrophobic_fraction - 0.56).abs() / 0.55)
                } else if hydrophobic_fraction < 0.32 {
                    (hydrophobic_fraction / 0.32) * 0.72
                } else {
                    0.75
                }
                .clamp(0.1, 1.0);
                let raw_drug = 0.42 * vol_score + 0.34 * enc_score + 0.24 * hydro_score;
                let druggability_score =
                    (1.0 / (1.0 + (-5.5 * (raw_drug - 0.48)).exp())).clamp(0.01, 0.99);

                // Bound serialized voxel cloud cleanly if huge
                let cavity_voxels = if world_pts.len() > 2400 {
                    let stride = (world_pts.len() / 2400).max(1);
                    world_pts.into_iter().step_by(stride).collect()
                } else {
                    world_pts
                };

                pockets.push(BindingPocket3D {
                    pocket_id: 0,
                    center,
                    size,
                    volume_a3,
                    enclosure,
                    hydrophobic_fraction,
                    polar_fraction,
                    druggability_score,
                    lining_residues,
                    voxel_step_a: step as f32,
                    cavity_voxels,
                });
            }
        }
    }

    pockets.sort_by(|a, b| {
        let sa = a.druggability_score * (a.volume_a3.min(1800.0) / 400.0).sqrt().clamp(0.4, 2.0);
        let sb = b.druggability_score * (b.volume_a3.min(1800.0) / 400.0).sqrt().clamp(0.4, 2.0);
        sb.total_cmp(&sa)
    });
    pockets.truncate(limit);
    for (idx, p) in pockets.iter_mut().enumerate() {
        p.pocket_id = idx + 1;
    }
    pockets
}
