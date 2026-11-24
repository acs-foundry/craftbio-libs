#![forbid(unsafe_code)]

//! Pure-Rust Biomolecular 3D Mesh Engine:
//! - True Triangulated **Cartoon Ribbons** (carbonyl-oriented α-helices, flat β-sheets with
//!   directional arrowheads, smooth coils, and 3D rectangular RNA/DNA base slabs)
//! - Watertight Kuhn-6 **Marching Tetrahedra** Isosurface Extraction with smooth central-difference
//!   gradient normals & Laplacian smoothing for both **SES Molecular Surfaces** (ESP/MLP/SAP/pLDDT/Chain)
//!   and **Cryo-EM / APBS `VolumeGrid3D` Maps** (solid & wireframe `isomesh`)
//! - Zero-dependency **Binary glTF 2.0 (`.glb`)** and **Wavefront (`.obj`)** 3D Mesh Exporters

use std::collections::HashMap;
use std::f32::consts::TAU;
use std::fmt::Write as _;

use bytemuck::{Pod, Zeroable};
use craftbio_struct::{
    BindingPocket3D, Biomolecule, ResidueClass, SecondaryStructure, SnfgShape, VolumeGrid3D,
    detect_snfg_glycans, vdw_radius,
};
use rayon::prelude::*;
use serde_json::json;

use crate::camera::{cross3, dot3, normalize3};

/// GPU & headless indexed triangle vertex (`40` bytes, 4-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Indexed 3D triangle mesh (Cartoon ribbon, β-arrow, RNA base slab, SES surface, Cryo-EM map).
#[derive(Debug, Clone, Default)]
pub struct TriangleMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub is_transparent: bool,
    pub wireframe: bool,
}

impl TriangleMesh {
    pub fn extend_from(&mut self, other: &Self) {
        if other.vertices.is_empty() || other.indices.is_empty() {
            return;
        }
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices
            .extend(other.indices.iter().map(|&idx| base + idx));
    }

    /// Recompute smooth area-weighted vertex normals (optional post-pass).
    pub fn recompute_smooth_normals(&mut self) {
        let mut accum = vec![[0.0_f32; 3]; self.vertices.len()];
        for tri in self.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let p0 = self.vertices[i0].position;
            let p1 = self.vertices[i1].position;
            let p2 = self.vertices[i2].position;
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let fnrm = cross3(e1, e2);
            for &idx in &[i0, i1, i2] {
                accum[idx][0] += fnrm[0];
                accum[idx][1] += fnrm[1];
                accum[idx][2] += fnrm[2];
            }
        }
        for (v, &n) in self.vertices.iter_mut().zip(accum.iter()) {
            let l2 = dot3(n, n);
            if l2 > 1e-10 {
                let inv = 1.0 / l2.sqrt();
                v.normal = [n[0] * inv, n[1] * inv, n[2] * inv];
            }
        }
    }

    /// Single-pass Taubin-style mild Laplacian smoothing for silky ChimeraX-like molecular surfaces.
    pub fn smooth_surface(&mut self, lambda: f32) {
        let n = self.vertices.len();
        if n == 0 {
            return;
        }
        let mut sum = vec![[0.0_f32; 3]; n];
        let mut deg = vec![0u32; n];
        for tri in self.indices.as_chunks::<3>().0 {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if a >= n || b >= n || c >= n {
                continue;
            }
            for &(u, v) in &[(a, b), (b, c), (c, a)] {
                let pu = self.vertices[u].position;
                let pv = self.vertices[v].position;
                for k in 0..3 {
                    sum[u][k] += pv[k];
                    sum[v][k] += pu[k];
                }
                deg[u] += 1;
                deg[v] += 1;
            }
        }
        for i in 0..n {
            if deg[i] > 0 {
                let inv = 1.0 / (deg[i] as f32);
                for k in 0..3 {
                    let avg = sum[i][k] * inv;
                    self.vertices[i].position[k] += lambda * (avg - self.vertices[i].position[k]);
                }
            }
        }
    }
}

fn catmull_rom(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0_f32; 3];
    for k in 0..3 {
        out[k] = 0.5
            * ((2.0 * p1[k])
                + (-p0[k] + p2[k]) * t
                + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3);
    }
    out
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Emit a flat-shaded / crisp-edged 3D box (used for RNA/DNA rectangular base slabs).
fn push_oriented_slab(
    mesh: &mut TriangleMesh,
    center: [f32; 3],
    axis_u: [f32; 3],
    axis_v: [f32; 3],
    axis_w: [f32; 3],
    half_ext: [f32; 3],
    color: [f32; 4],
) {
    let corners: [[f32; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let mut pts = [[0.0_f32; 3]; 8];
    for (i, c) in corners.iter().enumerate() {
        pts[i] = add3(
            center,
            add3(
                scale3(axis_u, c[0] * half_ext[0]),
                add3(
                    scale3(axis_v, c[1] * half_ext[1]),
                    scale3(axis_w, c[2] * half_ext[2]),
                ),
            ),
        );
    }
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([4, 5, 6, 7], axis_w),
        ([1, 0, 3, 2], scale3(axis_w, -1.0)),
        ([1, 2, 6, 5], axis_u),
        ([3, 0, 4, 7], scale3(axis_u, -1.0)),
        ([2, 3, 7, 6], axis_v),
        ([0, 1, 5, 4], scale3(axis_v, -1.0)),
    ];
    for (quad, nrm) in faces {
        let base = mesh.vertices.len() as u32;
        for &ci in &quad {
            mesh.vertices.push(MeshVertex {
                position: pts[ci],
                normal: nrm,
                color,
            });
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Build true triangulated ChimeraX/PyMOL-style **Cartoon Ribbons** (oriented flat α-helices,
/// flat β-sheets with directional arrowheads, smooth coil tubes, and 3D rectangular RNA/DNA base slabs).
pub fn build_cartoon_mesh<F, G>(
    mol: &Biomolecule,
    color_of_atom: F,
    base_color_of_res: G,
    include_rna_slabs: bool,
) -> TriangleMesh
where
    F: Fn(usize, SecondaryStructure) -> [f32; 3],
    G: Fn(char, i32, char) -> [f32; 3],
{
    const RING_SEGS: usize = 8;
    const SUB_STEPS: usize = 6;
    let mut mesh = TriangleMesh {
        vertices: Vec::with_capacity(mol.atoms.len().min(60_000) * 2),
        indices: Vec::with_capacity(mol.atoms.len().min(60_000) * 6),
        is_transparent: false,
        wireframe: false,
    };

    struct GuideNode {
        atom_idx: usize,
        pos: [f32; 3],
        raw_up: [f32; 3],
        ss: SecondaryStructure,
        class: ResidueClass,
        color: [f32; 3],
        res_chain: char,
        res_seq: i32,
        res_code: char,
        c1_pos: Option<[f32; 3]>,
        tip_pos: Option<[f32; 3]>,
        ring_aux: Option<[f32; 3]>,
    }

    for ch in &mol.chains {
        let mut nodes: Vec<GuideNode> = Vec::with_capacity(ch.residues.len());
        for r in &ch.residues {
            if !matches!(
                r.class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            ) {
                continue;
            }
            let Some(gi) = r.guide_idx else { continue };
            let gc = mol.atoms[gi].coords;
            let pos = [gc[0] as f32, gc[1] as f32, gc[2] as f32];
            let raw_up = if let Some(oi) = r.orient_idx {
                let oc = mol.atoms[oi].coords;
                normalize3([
                    oc[0] as f32 - pos[0],
                    oc[1] as f32 - pos[1],
                    oc[2] as f32 - pos[2],
                ])
            } else {
                [0.0, 1.0, 0.0]
            };
            let c1_pos = r.orient_idx.map(|i| {
                let c = mol.atoms[i].coords;
                [c[0] as f32, c[1] as f32, c[2] as f32]
            });
            let tip_pos = r.base_tip_idx.or(r.orient_idx).map(|i| {
                let c = mol.atoms[i].coords;
                [c[0] as f32, c[1] as f32, c[2] as f32]
            });
            let ring_aux = if matches!(r.class, ResidueClass::Rna | ResidueClass::Dna) {
                r.atom_indices
                    .iter()
                    .copied()
                    .find(|&ai| matches!(mol.atoms[ai].name.as_str(), "C6" | "C8" | "C5" | "N7"))
                    .map(|ai| {
                        let c = mol.atoms[ai].coords;
                        [c[0] as f32, c[1] as f32, c[2] as f32]
                    })
            } else {
                None
            };
            nodes.push(GuideNode {
                atom_idx: gi,
                pos,
                raw_up,
                ss: r.ss,
                class: r.class,
                color: color_of_atom(gi, r.ss),
                res_chain: r.chain,
                res_seq: r.res_seq,
                res_code: r.code,
                c1_pos,
                tip_pos,
                ring_aux,
            });
        }
        if nodes.is_empty() {
            continue;
        }

        // Split chain into continuous segments across chain breaks
        let mut seg_start = 0usize;
        while seg_start < nodes.len() {
            let mut seg_end = seg_start + 1;
            while seg_end < nodes.len() {
                let max_jump = if nodes[seg_end - 1].class == ResidueClass::Protein {
                    5.3
                } else {
                    9.8
                };
                let d = sub3(nodes[seg_end].pos, nodes[seg_end - 1].pos);
                if dot3(d, d).sqrt() > max_jump {
                    break;
                }
                seg_end += 1;
            }
            let seg = &nodes[seg_start..seg_end];
            if seg.len() >= 2 {
                // Consistent orientation vectors along the segment (prevents 180-deg beta-strand twist)
                let mut ups: Vec<[f32; 3]> = Vec::with_capacity(seg.len());
                for (i, nd) in seg.iter().enumerate() {
                    let mut u = nd.raw_up;
                    if i > 0 && dot3(u, ups[i - 1]) < 0.0 {
                        u = scale3(u, -1.0);
                    }
                    ups.push(u);
                }
                // Gentle 3-point smoothing of ribbon normals for fluid ChimeraX sweep
                let mut smooth_ups = ups.clone();
                for i in 1..(seg.len() - 1) {
                    smooth_ups[i] = normalize3(add3(
                        add3(scale3(ups[i - 1], 0.25), scale3(ups[i], 0.5)),
                        scale3(ups[i + 1], 0.25),
                    ));
                }

                struct Station {
                    pos: [f32; 3],
                    up_hint: [f32; 3],
                    half_w: f32,
                    half_t: f32,
                    color: [f32; 3],
                    is_flat_sheet: bool,
                    is_helix: bool,
                }
                let mut stations: Vec<Station> =
                    Vec::with_capacity((seg.len() - 1) * SUB_STEPS + 1);

                let dims_for = |ss: SecondaryStructure| -> (f32, f32) {
                    match ss {
                        SecondaryStructure::Helix => (1.08, 0.22),
                        SecondaryStructure::Sheet => (1.02, 0.17),
                        SecondaryStructure::NucleicRibbon => (0.82, 0.22),
                        SecondaryStructure::Coil => (0.30, 0.30),
                    }
                };

                for i in 0..(seg.len() - 1) {
                    let p0 = seg[i.saturating_sub(1)].pos;
                    let p1 = seg[i].pos;
                    let p2 = seg[i + 1].pos;
                    let p3 = seg[(i + 2).min(seg.len() - 1)].pos;

                    let ss_a = seg[i].ss;
                    let ss_b = seg[i + 1].ss;
                    let is_arrow_span = ss_a == SecondaryStructure::Sheet
                        && (ss_b != SecondaryStructure::Sheet || i + 1 == seg.len() - 1);

                    let (wa, ta) = dims_for(ss_a);
                    let (wb, tb) = dims_for(ss_b);
                    let steps = if i + 1 == seg.len() - 1 {
                        SUB_STEPS
                    } else {
                        SUB_STEPS - 1
                    };
                    for s in 0..=steps {
                        let t = (s as f32) / (SUB_STEPS as f32);
                        let pos = catmull_rom(p0, p1, p2, p3, t);
                        let up_hint = normalize3(lerp3(smooth_ups[i], smooth_ups[i + 1], t));
                        let (half_w, half_t) = if is_arrow_span {
                            // Crisp PyMOL/ChimeraX beta-sheet directional arrowhead!
                            if t < 0.18 {
                                let k = t / 0.18;
                                (1.02 + 0.68 * k, 0.19)
                            } else {
                                let k = (t - 0.18) / 0.82;
                                ((1.70 * (1.0 - k)).max(0.18), 0.19 * (1.0 - 0.45 * k))
                            }
                        } else {
                            let blend = t * t * (3.0 - 2.0 * t);
                            (wa + (wb - wa) * blend, ta + (tb - ta) * blend)
                        };
                        let col = lerp3(seg[i].color, seg[i + 1].color, t);
                        stations.push(Station {
                            pos,
                            up_hint,
                            half_w,
                            half_t,
                            color: col,
                            is_flat_sheet: ss_a == SecondaryStructure::Sheet,
                            is_helix: ss_a == SecondaryStructure::Helix,
                        });
                    }
                }

                // Sweep elliptical/beveled cross-sections along stations with parallel-transport continuity
                let base_vert = mesh.vertices.len() as u32;
                let mut prev_n = [0.0_f32, 1.0, 0.0];
                for (si, st) in stations.iter().enumerate() {
                    let tang = if si == 0 {
                        normalize3(sub3(stations[1].pos, st.pos))
                    } else if si + 1 == stations.len() {
                        normalize3(sub3(st.pos, stations[si - 1].pos))
                    } else {
                        normalize3(sub3(stations[si + 1].pos, stations[si - 1].pos))
                    };
                    // Project up_hint orthogonal to tangent
                    let proj = sub3(st.up_hint, scale3(tang, dot3(st.up_hint, tang)));
                    let mut nrm = if dot3(proj, proj) > 1e-5 {
                        normalize3(proj)
                    } else {
                        let fb = sub3(prev_n, scale3(tang, dot3(prev_n, tang)));
                        if dot3(fb, fb) > 1e-5 {
                            normalize3(fb)
                        } else {
                            normalize3(cross3(tang, [0.0, 1.0, 0.0]))
                        }
                    };
                    if si > 0 && dot3(nrm, prev_n) < 0.0 {
                        nrm = scale3(nrm, -1.0);
                    }
                    prev_n = nrm;
                    let binrm = normalize3(cross3(tang, nrm));

                    for r in 0..RING_SEGS {
                        let ang = (r as f32) * TAU / (RING_SEGS as f32);
                        let ca = ang.cos();
                        let sa = ang.sin();
                        // Slightly squared superellipse for beta-sheets & flat ribbons
                        let (lx, ly) = if st.is_flat_sheet {
                            (
                                ca.signum() * ca.abs().powf(0.75),
                                sa.signum() * sa.abs().powf(0.75),
                            )
                        } else {
                            (ca, sa)
                        };
                        let offset =
                            add3(scale3(binrm, lx * st.half_w), scale3(nrm, ly * st.half_t));
                        let vpos = add3(st.pos, offset);
                        let vnrm = normalize3(add3(
                            scale3(binrm, ca / st.half_w.max(0.1)),
                            scale3(nrm, sa / st.half_t.max(0.1)),
                        ));
                        // Subtle warm inner-face tint on alpha-helices (ChimeraX / PyMOL fancy helix cue)
                        let tint = if st.is_helix && sa < -0.25 { 0.86 } else { 1.0 };
                        mesh.vertices.push(MeshVertex {
                            position: vpos,
                            normal: vnrm,
                            color: [
                                (st.color[0] * tint).clamp(0.0, 1.0),
                                (st.color[1] * tint).clamp(0.0, 1.0),
                                (st.color[2] * tint).clamp(0.0, 1.0),
                                1.0,
                            ],
                        });
                    }
                }

                // Stitch ring quads
                let stride = RING_SEGS as u32;
                for si in 0..(stations.len().saturating_sub(1)) {
                    let r0 = base_vert + (si as u32) * stride;
                    let r1 = r0 + stride;
                    for k in 0..stride {
                        let kn = (k + 1) % stride;
                        mesh.indices.extend_from_slice(&[
                            r0 + k,
                            r1 + k,
                            r1 + kn,
                            r0 + k,
                            r1 + kn,
                            r0 + kn,
                        ]);
                    }
                }

                // Watertight endcaps for start & finish of segment
                if let (Some(first), Some(last)) = (stations.first(), stations.last()) {
                    let start_c = mesh.vertices.len() as u32;
                    let start_n = normalize3(sub3(first.pos, stations[1].pos));
                    mesh.vertices.push(MeshVertex {
                        position: first.pos,
                        normal: start_n,
                        color: [first.color[0], first.color[1], first.color[2], 1.0],
                    });
                    for k in 0..stride {
                        let kn = (k + 1) % stride;
                        mesh.indices
                            .extend_from_slice(&[start_c, base_vert + kn, base_vert + k]);
                    }
                    let end_ring = base_vert + ((stations.len() - 1) as u32) * stride;
                    let end_c = mesh.vertices.len() as u32;
                    let end_n = normalize3(sub3(last.pos, stations[stations.len() - 2].pos));
                    mesh.vertices.push(MeshVertex {
                        position: last.pos,
                        normal: end_n,
                        color: [last.color[0], last.color[1], last.color[2], 1.0],
                    });
                    for k in 0..stride {
                        let kn = (k + 1) % stride;
                        mesh.indices
                            .extend_from_slice(&[end_c, end_ring + k, end_ring + kn]);
                    }
                }
            }

            // 3D oriented rectangular base slabs for RNA / DNA nucleotides (ChimeraX `nucleotides slab`)
            if include_rna_slabs {
                for nd in seg {
                    if !matches!(nd.class, ResidueClass::Rna | ResidueClass::Dna) {
                        continue;
                    }
                    let Some(tip) = nd.tip_pos else { continue };
                    let root = nd.c1_pos.unwrap_or(nd.pos);
                    let dir = sub3(tip, root);
                    let dlen = dot3(dir, dir).sqrt();
                    if dlen < 0.35 {
                        continue;
                    }
                    let axis_u = scale3(dir, 1.0 / dlen);
                    let aux = nd
                        .ring_aux
                        .map(|p| sub3(p, root))
                        .unwrap_or_else(|| sub3(nd.pos, root));
                    let mut axis_w = cross3(axis_u, aux);
                    if dot3(axis_w, axis_w) < 1e-4 {
                        axis_w = cross3(axis_u, [0.0, 1.0, 0.0]);
                    }
                    axis_w = normalize3(axis_w);
                    let axis_v = normalize3(cross3(axis_w, axis_u));

                    let is_purine = matches!(nd.res_code.to_ascii_uppercase(), 'A' | 'G');
                    let half_ext = if is_purine {
                        [1.85_f32, 1.05, 0.21]
                    } else {
                        [1.35_f32, 0.92, 0.21]
                    };
                    let center = add3(root, scale3(axis_u, half_ext[0] + 0.55));
                    let bc = base_color_of_res(nd.res_chain, nd.res_seq, nd.res_code);
                    push_oriented_slab(
                        &mut mesh,
                        center,
                        axis_u,
                        axis_v,
                        axis_w,
                        half_ext,
                        [bc[0], bc[1], bc[2], 1.0],
                    );
                }
            }

            let _ = seg[0].atom_idx;
            seg_start = seg_end;
        }
    }

    mesh
}

/// Build a smooth variable-radius **B-Factor / pLDDT / RMSF Putty (Sausage)** tube mesh
/// (PyMOL `cartoon putty` / ChimeraX `worm` equivalent): tube radius scales smoothly between
/// `0.24 Å` and `1.08 Å` based on each guide residue's scalar B-factor/attribute value.
pub fn build_putty_mesh<F>(mol: &Biomolecule, color_of_atom: F) -> TriangleMesh
where
    F: Fn(usize, SecondaryStructure) -> [f32; 3],
{
    const RING_SEGS: usize = 10;
    const SUB_STEPS: usize = 5;
    let mut mesh = TriangleMesh {
        vertices: Vec::with_capacity(mol.atoms.len().min(40_000) * 2),
        indices: Vec::with_capacity(mol.atoms.len().min(40_000) * 6),
        is_transparent: false,
        wireframe: false,
    };

    // Determine global B-factor range across polymer guide atoms
    let mut b_min = f64::INFINITY;
    let mut b_max = f64::NEG_INFINITY;
    let mut b_sum = 0.0_f64;
    let mut b_cnt = 0usize;
    for ch in &mol.chains {
        for r in &ch.residues {
            if matches!(
                r.class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            ) && let Some(gi) = r.guide_idx
            {
                let b = mol.atoms[gi].b_factor;
                if b.is_finite() {
                    b_min = b_min.min(b);
                    b_max = b_max.max(b);
                    b_sum += b;
                    b_cnt += 1;
                }
            }
        }
    }
    let span = (b_max - b_min).max(1e-3);
    // AlphaFold/Boltz pLDDT convention: high value (90-100) = rigid/slender, low value (<70) = flexible/thick
    let is_plddt_like = b_cnt > 0 && b_max <= 100.5 && (b_sum / b_cnt as f64) > 55.0;

    let radius_for_b = |b: f64| -> f32 {
        if (b_max - b_min) < 1e-3 {
            return 0.44;
        }
        let norm = ((b - b_min) / span).clamp(0.0, 1.0) as f32;
        let flex = if is_plddt_like { 1.0 - norm } else { norm };
        0.24 + 0.84 * flex
    };

    struct PuttyNode {
        pos: [f32; 3],
        radius: f32,
        color: [f32; 3],
        class: ResidueClass,
    }

    for ch in &mol.chains {
        let mut nodes = Vec::with_capacity(ch.residues.len());
        for r in &ch.residues {
            if !matches!(
                r.class,
                ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
            ) {
                continue;
            }
            let Some(gi) = r.guide_idx else { continue };
            let c = mol.atoms[gi].coords;
            nodes.push(PuttyNode {
                pos: [c[0] as f32, c[1] as f32, c[2] as f32],
                radius: radius_for_b(mol.atoms[gi].b_factor),
                color: color_of_atom(gi, r.ss),
                class: r.class,
            });
        }
        let mut seg_start = 0usize;
        while seg_start < nodes.len() {
            let mut seg_end = seg_start + 1;
            while seg_end < nodes.len() {
                let max_jump = if nodes[seg_end - 1].class == ResidueClass::Protein {
                    5.3
                } else {
                    9.8
                };
                let d = sub3(nodes[seg_end].pos, nodes[seg_end - 1].pos);
                if dot3(d, d).sqrt() > max_jump {
                    break;
                }
                seg_end += 1;
            }
            let seg = &nodes[seg_start..seg_end];
            if seg.len() >= 2 {
                let mut stations: Vec<([f32; 3], f32, [f32; 3])> =
                    Vec::with_capacity((seg.len() - 1) * SUB_STEPS + 1);
                for i in 0..(seg.len() - 1) {
                    let p0 = seg[i.saturating_sub(1)].pos;
                    let p1 = seg[i].pos;
                    let p2 = seg[i + 1].pos;
                    let p3 = seg[(i + 2).min(seg.len() - 1)].pos;
                    let steps = if i + 1 == seg.len() - 1 {
                        SUB_STEPS
                    } else {
                        SUB_STEPS - 1
                    };
                    for s in 0..=steps {
                        let t = (s as f32) / (SUB_STEPS as f32);
                        let pos = catmull_rom(p0, p1, p2, p3, t);
                        let rad = seg[i].radius + (seg[i + 1].radius - seg[i].radius) * t;
                        let col = lerp3(seg[i].color, seg[i + 1].color, t);
                        stations.push((pos, rad, col));
                    }
                }

                let base_vert = mesh.vertices.len() as u32;
                let mut prev_n = [0.0_f32, 1.0, 0.0];
                for (si, &(pos, rad, col)) in stations.iter().enumerate() {
                    let tang = if si == 0 {
                        normalize3(sub3(stations[1].0, pos))
                    } else if si + 1 == stations.len() {
                        normalize3(sub3(pos, stations[si - 1].0))
                    } else {
                        normalize3(sub3(stations[si + 1].0, stations[si - 1].0))
                    };
                    let fb = sub3(prev_n, scale3(tang, dot3(prev_n, tang)));
                    let nrm = if dot3(fb, fb) > 1e-5 {
                        normalize3(fb)
                    } else {
                        let alt = if tang[1].abs() < 0.8 {
                            [0.0, 1.0, 0.0]
                        } else {
                            [1.0, 0.0, 0.0]
                        };
                        normalize3(cross3(tang, alt))
                    };
                    prev_n = nrm;
                    let binrm = normalize3(cross3(tang, nrm));
                    for r in 0..RING_SEGS {
                        let ang = (r as f32) * TAU / (RING_SEGS as f32);
                        let ca = ang.cos();
                        let sa = ang.sin();
                        let vnrm = normalize3(add3(scale3(binrm, ca), scale3(nrm, sa)));
                        let vpos = add3(pos, scale3(vnrm, rad));
                        mesh.vertices.push(MeshVertex {
                            position: vpos,
                            normal: vnrm,
                            color: [col[0], col[1], col[2], 1.0],
                        });
                    }
                }
                let stride = RING_SEGS as u32;
                for si in 0..(stations.len().saturating_sub(1)) {
                    let r0 = base_vert + (si as u32) * stride;
                    let r1 = r0 + stride;
                    for k in 0..stride {
                        let kn = (k + 1) % stride;
                        mesh.indices.extend_from_slice(&[
                            r0 + k,
                            r1 + k,
                            r1 + kn,
                            r0 + k,
                            r1 + kn,
                            r0 + kn,
                        ]);
                    }
                }
                if let (Some(first), Some(last)) = (stations.first(), stations.last()) {
                    let start_c = mesh.vertices.len() as u32;
                    let start_n = normalize3(sub3(first.0, stations[1].0));
                    mesh.vertices.push(MeshVertex {
                        position: first.0,
                        normal: start_n,
                        color: [first.2[0], first.2[1], first.2[2], 1.0],
                    });
                    for k in 0..stride {
                        let kn = (k + 1) % stride;
                        mesh.indices
                            .extend_from_slice(&[start_c, base_vert + kn, base_vert + k]);
                    }
                    let end_ring = base_vert + ((stations.len() - 1) as u32) * stride;
                    let end_c = mesh.vertices.len() as u32;
                    let end_n = normalize3(sub3(last.0, stations[stations.len() - 2].0));
                    mesh.vertices.push(MeshVertex {
                        position: last.0,
                        normal: end_n,
                        color: [last.2[0], last.2[1], last.2[2], 1.0],
                    });
                    for k in 0..stride {
                        let kn = (k + 1) % stride;
                        mesh.indices
                            .extend_from_slice(&[end_c, end_ring + k, end_ring + kn]);
                    }
                }
            }
            seg_start = seg_end;
        }
    }
    mesh
}

/// Kuhn 6-tetrahedra decomposition of the unit cube `[0,1]^3` (100% crack-free watertight manifold).
const KUHN_TETS: [[usize; 4]; 6] = [
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
];

const CUBE_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0], // 0
    [1, 0, 0], // 1
    [1, 1, 0], // 2
    [0, 1, 0], // 3
    [0, 0, 1], // 4
    [1, 0, 1], // 5
    [1, 1, 1], // 6
    [0, 1, 1], // 7
];

/// Extract a watertight smooth triangulated isosurface at `iso_level` from any 3D scalar grid
/// using Kuhn-6 Marching Tetrahedra, shared edge vertices, and central-difference smooth normals.
#[must_use]
pub fn extract_isosurface_mesh<C>(
    grid: &VolumeGrid3D,
    iso_level: f32,
    color_fn: C,
    alpha: f32,
    wireframe: bool,
    smooth_passes: usize,
) -> TriangleMesh
where
    C: Fn([f32; 3]) -> [f32; 3],
{
    let [nx, ny, nz] = grid.dims;
    if nx < 3 || ny < 3 || nz < 3 {
        return TriangleMesh::default();
    }

    let sample = |x: usize, y: usize, z: usize| -> f32 { grid.get(x, y, z) };
    let grad_normal = |p: [f32; 3]| -> [f32; 3] {
        let eps_x = grid.voxel_size[0] as f32 * 0.5;
        let eps_y = grid.voxel_size[1] as f32 * 0.5;
        let eps_z = grid.voxel_size[2] as f32 * 0.5;
        let gx = grid.sample_trilinear([f64::from(p[0] + eps_x), f64::from(p[1]), f64::from(p[2])])
            - grid.sample_trilinear([f64::from(p[0] - eps_x), f64::from(p[1]), f64::from(p[2])]);
        let gy = grid.sample_trilinear([f64::from(p[0]), f64::from(p[1] + eps_y), f64::from(p[2])])
            - grid.sample_trilinear([f64::from(p[0]), f64::from(p[1] - eps_y), f64::from(p[2])]);
        let gz = grid.sample_trilinear([f64::from(p[0]), f64::from(p[1]), f64::from(p[2] + eps_z)])
            - grid.sample_trilinear([f64::from(p[0]), f64::from(p[1]), f64::from(p[2] - eps_z)]);
        // Outward normal points from higher density (interior) toward lower density (exterior)
        normalize3([-(gx as f32), -(gy as f32), -(gz as f32)])
    };

    let mut mesh = TriangleMesh {
        vertices: Vec::with_capacity(4096),
        indices: Vec::with_capacity(12_288),
        is_transparent: alpha < 0.98,
        wireframe,
    };
    type CornerKey = (u16, u16, u16);
    type EdgeKey = (CornerKey, CornerKey);
    let mut edge_cache: HashMap<EdgeKey, u32> = HashMap::with_capacity(8192);

    let mut get_edge_vertex =
        |verts: &mut Vec<MeshVertex>, c0: CornerKey, v0: f32, c1: CornerKey, v1: f32| -> u32 {
            let key = if c0 < c1 { (c0, c1) } else { (c1, c0) };
            if let Some(&idx) = edge_cache.get(&key) {
                return idx;
            }
            let denom = v1 - v0;
            let t = if denom.abs() > 1e-8 {
                ((iso_level - v0) / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let p0 = grid.voxel_world(c0.0 as usize, c0.1 as usize, c0.2 as usize);
            let p1 = grid.voxel_world(c1.0 as usize, c1.1 as usize, c1.2 as usize);
            let pos = [
                (p0[0] + (p1[0] - p0[0]) * f64::from(t)) as f32,
                (p0[1] + (p1[1] - p0[1]) * f64::from(t)) as f32,
                (p0[2] + (p1[2] - p0[2]) * f64::from(t)) as f32,
            ];
            let nrm = grad_normal(pos);
            let rgb = color_fn(pos);
            let idx = verts.len() as u32;
            verts.push(MeshVertex {
                position: pos,
                normal: nrm,
                color: [rgb[0], rgb[1], rgb[2], alpha],
            });
            edge_cache.insert(key, idx);
            idx
        };

    let push_oriented_tri =
        |verts: &[MeshVertex], indices: &mut Vec<u32>, a: u32, b: u32, c: u32| {
            let pa = verts[a as usize].position;
            let pb = verts[b as usize].position;
            let pc = verts[c as usize].position;
            let face_n = cross3(sub3(pb, pa), sub3(pc, pa));
            let avg_n = add3(
                add3(verts[a as usize].normal, verts[b as usize].normal),
                verts[c as usize].normal,
            );
            if dot3(face_n, avg_n) >= 0.0 {
                indices.extend_from_slice(&[a, b, c]);
            } else {
                indices.extend_from_slice(&[a, c, b]);
            }
        };

    for z in 0..(nz - 1) {
        for y in 0..(ny - 1) {
            for x in 0..(nx - 1) {
                let mut cvals = [0.0_f32; 8];
                let mut inside_mask = 0u8;
                for (ci, off) in CUBE_OFFSETS.iter().enumerate() {
                    let v = sample(x + off[0], y + off[1], z + off[2]);
                    cvals[ci] = v;
                    if v >= iso_level {
                        inside_mask |= 1 << ci;
                    }
                }
                if inside_mask == 0 || inside_mask == 0xFF {
                    continue;
                }
                for tet in &KUHN_TETS {
                    let mut inside = [0usize; 4];
                    let mut outside = [0usize; 4];
                    let mut n_in = 0usize;
                    let mut n_out = 0usize;
                    for &ci in tet {
                        if cvals[ci] >= iso_level {
                            inside[n_in] = ci;
                            n_in += 1;
                        } else {
                            outside[n_out] = ci;
                            n_out += 1;
                        }
                    }
                    let ckey = |ci: usize| -> CornerKey {
                        let o = CUBE_OFFSETS[ci];
                        ((x + o[0]) as u16, (y + o[1]) as u16, (z + o[2]) as u16)
                    };
                    match n_in {
                        1 => {
                            let i0 = inside[0];
                            let e0 = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i0),
                                cvals[i0],
                                ckey(outside[0]),
                                cvals[outside[0]],
                            );
                            let e1 = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i0),
                                cvals[i0],
                                ckey(outside[1]),
                                cvals[outside[1]],
                            );
                            let e2 = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i0),
                                cvals[i0],
                                ckey(outside[2]),
                                cvals[outside[2]],
                            );
                            push_oriented_tri(&mesh.vertices, &mut mesh.indices, e0, e1, e2);
                        }
                        3 => {
                            let o0 = outside[0];
                            let e0 = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(o0),
                                cvals[o0],
                                ckey(inside[0]),
                                cvals[inside[0]],
                            );
                            let e1 = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(o0),
                                cvals[o0],
                                ckey(inside[1]),
                                cvals[inside[1]],
                            );
                            let e2 = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(o0),
                                cvals[o0],
                                ckey(inside[2]),
                                cvals[inside[2]],
                            );
                            push_oriented_tri(&mesh.vertices, &mut mesh.indices, e0, e1, e2);
                        }
                        2 => {
                            let (i0, i1) = (inside[0], inside[1]);
                            let (o0, o1) = (outside[0], outside[1]);
                            let a = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i0),
                                cvals[i0],
                                ckey(o0),
                                cvals[o0],
                            );
                            let b = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i0),
                                cvals[i0],
                                ckey(o1),
                                cvals[o1],
                            );
                            let c = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i1),
                                cvals[i1],
                                ckey(o1),
                                cvals[o1],
                            );
                            let d = get_edge_vertex(
                                &mut mesh.vertices,
                                ckey(i1),
                                cvals[i1],
                                ckey(o0),
                                cvals[o0],
                            );
                            push_oriented_tri(&mesh.vertices, &mut mesh.indices, a, b, c);
                            push_oriented_tri(&mesh.vertices, &mut mesh.indices, a, c, d);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    for _ in 0..smooth_passes.min(3) {
        mesh.smooth_surface(0.32);
    }
    if smooth_passes > 0 {
        mesh.recompute_smooth_normals();
    }
    mesh
}

/// Build a smooth triangulated Solvent-Excluded / Gaussian Molecular Surface (`SES`) mesh colored
/// per-vertex by nearest atom color (ESP / MLP / SAP / pLDDT / Chain / Secondary Structure).
#[must_use]
pub fn build_ses_surface_mesh(
    mol: &Biomolecule,
    atom_colors: &[[f32; 3]],
    alpha: f32,
    wireframe: bool,
    show_waters: bool,
) -> TriangleMesh {
    let atoms: Vec<([f64; 3], f64, [f32; 3])> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| show_waters || a.res_class != ResidueClass::Water)
        .filter(|(_, a)| a.element != "H")
        .map(|(i, a)| {
            let r = vdw_radius(&a.element) + 0.65;
            let col = atom_colors.get(i).copied().unwrap_or([0.78, 0.82, 0.88]);
            (a.coords, r, col)
        })
        .collect();
    if atoms.is_empty() {
        return TriangleMesh::default();
    }

    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &(p, _, _) in &atoms {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let pad = 4.2_f64;
    let origin = [lo[0] - pad, lo[1] - pad, lo[2] - pad];
    let span = [
        (hi[0] - lo[0]) + 2.0 * pad,
        (hi[1] - lo[1]) + 2.0 * pad,
        (hi[2] - lo[2]) + 2.0 * pad,
    ];
    let max_span = span[0].max(span[1]).max(span[2]).max(8.0);
    // Adaptive grid capped cleanly at 72 voxels along longest axis (~0.9-1.6 A resolution, <120 ms)
    let vox = (max_span / 68.0).clamp(0.75, 2.2);
    let nx = ((span[0] / vox).ceil() as usize + 1).clamp(8, 76);
    let ny = ((span[1] / vox).ceil() as usize + 1).clamp(8, 76);
    let nz = ((span[2] / vox).ceil() as usize + 1).clamp(8, 76);

    let blobby_b = 2.15_f64;
    let slices: Vec<Vec<f32>> = (0..nz)
        .into_par_iter()
        .map(|z| {
            let wz = origin[2] + (z as f64) * vox;
            let mut slice = vec![0.0_f32; nx * ny];
            for &(p, r, _) in &atoms {
                let r_cut = r * 1.95;
                let dz = wz - p[2];
                if dz.abs() > r_cut {
                    continue;
                }
                let inv_r2 = 1.0 / (r * r);
                let r_cut2 = r_cut * r_cut;
                let x0 = (((p[0] - r_cut - origin[0]) / vox).floor() as isize).max(0) as usize;
                let x1 = (((p[0] + r_cut - origin[0]) / vox).ceil() as isize).min((nx - 1) as isize)
                    as usize;
                let y0 = (((p[1] - r_cut - origin[1]) / vox).floor() as isize).max(0) as usize;
                let y1 = (((p[1] + r_cut - origin[1]) / vox).ceil() as isize).min((ny - 1) as isize)
                    as usize;
                for y in y0..=y1 {
                    let dy = origin[1] + (y as f64) * vox - p[1];
                    let dyz2 = dy * dy + dz * dz;
                    if dyz2 > r_cut2 {
                        continue;
                    }
                    for x in x0..=x1 {
                        let dx = origin[0] + (x as f64) * vox - p[0];
                        let d2 = dx * dx + dyz2;
                        if d2 <= r_cut2 {
                            slice[y * nx + x] += (-blobby_b * (d2 * inv_r2 - 1.0)).exp() as f32;
                        }
                    }
                }
            }
            slice
        })
        .collect();

    let mut values = Vec::with_capacity(nx * ny * nz);
    for mut s in slices {
        values.append(&mut s);
    }
    let Ok(grid) =
        VolumeGrid3D::from_raw("ses_field", [nx, ny, nz], origin, [vox, vox, vox], values)
    else {
        return TriangleMesh::default();
    };

    // Spatial hash for O(1) smooth Gaussian-weighted surface vertex coloring
    let cell = 4.5_f64;
    let ckey = |p: [f64; 3]| -> (i32, i32, i32) {
        (
            (p[0] / cell).floor() as i32,
            (p[1] / cell).floor() as i32,
            (p[2] / cell).floor() as i32,
        )
    };
    let mut hash: HashMap<(i32, i32, i32), Vec<usize>> =
        HashMap::with_capacity(atoms.len() / 2 + 1);
    for (i, &(p, _, _)) in atoms.iter().enumerate() {
        hash.entry(ckey(p)).or_default().push(i);
    }

    let color_at = |pt: [f32; 3]| -> [f32; 3] {
        let p = [f64::from(pt[0]), f64::from(pt[1]), f64::from(pt[2])];
        let (cx, cy, cz) = ckey(p);
        let mut num = [0.0_f64; 3];
        let mut den = 0.0_f64;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = hash.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &ai in bucket {
                            let (ap, _, col) = atoms[ai];
                            let d2 = (p[0] - ap[0]).powi(2)
                                + (p[1] - ap[1]).powi(2)
                                + (p[2] - ap[2]).powi(2);
                            if d2 < 28.0 {
                                let w = (-d2 * 0.35).exp();
                                num[0] += f64::from(col[0]) * w;
                                num[1] += f64::from(col[1]) * w;
                                num[2] += f64::from(col[2]) * w;
                                den += w;
                            }
                        }
                    }
                }
            }
        }
        if den > 1e-8 {
            [
                (num[0] / den) as f32,
                (num[1] / den) as f32,
                (num[2] / den) as f32,
            ]
        } else {
            [0.78, 0.82, 0.88]
        }
    };

    extract_isosurface_mesh(&grid, 1.0, color_at, alpha, wireframe, 1)
}

/// Export one or more `TriangleMesh` instances to Wavefront `.obj` ASCII format (with normals).
#[must_use]
pub fn export_meshes_to_obj(meshes: &[&TriangleMesh], title: &str) -> String {
    let mut out = String::with_capacity(16_384);
    let _ = writeln!(out, "# BiomolCraft v0.4.0 Wavefront OBJ export: {title}");
    let mut v_offset = 1usize;
    for (mi, mesh) in meshes.iter().enumerate() {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        let _ = writeln!(out, "o biomol_mesh_{mi}");
        for v in &mesh.vertices {
            let _ = writeln!(
                out,
                "v {:.4} {:.4} {:.4} {:.3} {:.3} {:.3}",
                v.position[0], v.position[1], v.position[2], v.color[0], v.color[1], v.color[2]
            );
        }
        for v in &mesh.vertices {
            let _ = writeln!(
                out,
                "vn {:.4} {:.4} {:.4}",
                v.normal[0], v.normal[1], v.normal[2]
            );
        }
        for tri in mesh.indices.as_chunks::<3>().0 {
            let a = tri[0] as usize + v_offset;
            let b = tri[1] as usize + v_offset;
            let c = tri[2] as usize + v_offset;
            let _ = writeln!(out, "f {a}//{a} {b}//{b} {c}//{c}");
        }
        v_offset += mesh.vertices.len();
    }
    out
}

/// Export one or more `TriangleMesh` instances to a valid single-file **Binary glTF 2.0 (`.glb`)**
/// readable directly by Blender, ChimeraX, PyMOL, Three.js, and macOS/Linux 3D viewers.
#[must_use]
pub fn export_meshes_to_glb(meshes: &[&TriangleMesh], title: &str) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut buffer_views = Vec::new();
    let mut accessors = Vec::new();
    let mut primitives = Vec::new();
    let mut materials = Vec::new();

    for (mi, mesh) in meshes.iter().enumerate() {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        let mut pmin = [f32::INFINITY; 3];
        let mut pmax = [f32::NEG_INFINITY; 3];
        for v in &mesh.vertices {
            for k in 0..3 {
                pmin[k] = pmin[k].min(v.position[k]);
                pmax[k] = pmax[k].max(v.position[k]);
            }
        }

        // 1. POSITION buffer view
        let pos_off = bin.len();
        for v in &mesh.vertices {
            for &c in &v.position {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let pos_len = bin.len() - pos_off;
        let pos_bv = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0, "byteOffset": pos_off, "byteLength": pos_len, "target": 34962
        }));
        let pos_acc = accessors.len();
        accessors.push(json!({
            "bufferView": pos_bv, "componentType": 5126, "count": mesh.vertices.len(),
            "type": "VEC3", "min": pmin, "max": pmax
        }));

        // 2. NORMAL buffer view
        let nrm_off = bin.len();
        for v in &mesh.vertices {
            for &c in &v.normal {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let nrm_len = bin.len() - nrm_off;
        let nrm_bv = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0, "byteOffset": nrm_off, "byteLength": nrm_len, "target": 34962
        }));
        let nrm_acc = accessors.len();
        accessors.push(json!({
            "bufferView": nrm_bv, "componentType": 5126, "count": mesh.vertices.len(), "type": "VEC3"
        }));

        // 3. COLOR_0 buffer view
        let col_off = bin.len();
        for v in &mesh.vertices {
            for &c in &v.color {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let col_len = bin.len() - col_off;
        let col_bv = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0, "byteOffset": col_off, "byteLength": col_len, "target": 34962
        }));
        let col_acc = accessors.len();
        accessors.push(json!({
            "bufferView": col_bv, "componentType": 5126, "count": mesh.vertices.len(), "type": "VEC4"
        }));

        // 4. INDICES buffer view
        let idx_off = bin.len();
        for &idx in &mesh.indices {
            bin.extend_from_slice(&idx.to_le_bytes());
        }
        let idx_len = bin.len() - idx_off;
        let idx_bv = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0, "byteOffset": idx_off, "byteLength": idx_len, "target": 34963
        }));
        let idx_acc = accessors.len();
        accessors.push(json!({
            "bufferView": idx_bv, "componentType": 5125, "count": mesh.indices.len(), "type": "SCALAR"
        }));

        let mat_idx = materials.len();
        materials.push(json!({
            "name": format!("biomol_mat_{mi}"),
            "doubleSided": true,
            "alphaMode": if mesh.is_transparent { "BLEND" } else { "OPAQUE" },
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.05,
                "roughnessFactor": 0.45
            }
        }));

        primitives.push(json!({
            "attributes": {
                "POSITION": pos_acc,
                "NORMAL": nrm_acc,
                "COLOR_0": col_acc
            },
            "indices": idx_acc,
            "material": mat_idx
        }));
    }

    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let gltf_json = json!({
        "asset": {
            "version": "2.0",
            "generator": format!("BiomolCraft v0.4.0 ({title})")
        },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "name": title, "mesh": 0 }],
        "meshes": [{ "name": title, "primitives": primitives }],
        "materials": materials,
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": buffer_views,
        "accessors": accessors
    });

    let mut json_bytes = serde_json::to_vec(&gltf_json).unwrap_or_default();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_len);
    // GLB 12-byte header: magic "glTF", version 2, total length
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());
    // Chunk 0: JSON (0x4E4F534A)
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    glb.extend_from_slice(&json_bytes);
    // Chunk 1: BIN (0x004E4942)
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    glb.extend_from_slice(&bin);
    glb
}

fn push_uv_sphere(mesh: &mut TriangleMesh, center: [f32; 3], radius: f32, color: [f32; 4]) {
    const LAT: usize = 8;
    const LON: usize = 12;
    let base = mesh.vertices.len() as u32;
    for i in 0..=LAT {
        let theta = (i as f32) * std::f32::consts::PI / (LAT as f32);
        let st = theta.sin();
        let ct = theta.cos();
        for j in 0..=LON {
            let phi = (j as f32) * TAU / (LON as f32);
            let n = [st * phi.cos(), ct, st * phi.sin()];
            mesh.vertices.push(MeshVertex {
                position: add3(center, scale3(n, radius)),
                normal: n,
                color,
            });
        }
    }
    let row = (LON + 1) as u32;
    for i in 0..(LAT as u32) {
        for j in 0..(LON as u32) {
            let a = base + i * row + j;
            let b = a + row;
            mesh.indices
                .extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
}

fn push_octahedron(mesh: &mut TriangleMesh, center: [f32; 3], radius: f32, color: [f32; 4]) {
    let dirs: [[f32; 3]; 6] = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ];
    let pts: Vec<[f32; 3]> = dirs
        .iter()
        .map(|&d| add3(center, scale3(d, radius)))
        .collect();
    let faces = [
        (0, 2, 1),
        (0, 3, 2),
        (0, 4, 3),
        (0, 1, 4),
        (5, 1, 2),
        (5, 2, 3),
        (5, 3, 4),
        (5, 4, 1),
    ];
    for &(i0, i1, i2) in &faces {
        let n = normalize3(cross3(sub3(pts[i1], pts[i0]), sub3(pts[i2], pts[i0])));
        let base = mesh.vertices.len() as u32;
        for &idx in &[i0, i1, i2] {
            mesh.vertices.push(MeshVertex {
                position: pts[idx],
                normal: n,
                color,
            });
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

fn push_cone(mesh: &mut TriangleMesh, center: [f32; 3], radius: f32, color: [f32; 4]) {
    const SEGS: usize = 10;
    let apex = add3(center, [0.0, radius * 1.15, 0.0]);
    let base_y = center[1] - radius * 0.85;
    let base_center = [center[0], base_y, center[2]];
    let mut ring = Vec::with_capacity(SEGS);
    for s in 0..SEGS {
        let ang = (s as f32) * TAU / (SEGS as f32);
        ring.push([
            center[0] + radius * ang.cos(),
            base_y,
            center[2] + radius * ang.sin(),
        ]);
    }
    for s in 0..SEGS {
        let p0 = ring[s];
        let p1 = ring[(s + 1) % SEGS];
        let n = normalize3(cross3(sub3(p1, apex), sub3(p0, apex)));
        let b = mesh.vertices.len() as u32;
        for &p in &[apex, p1, p0] {
            mesh.vertices.push(MeshVertex {
                position: p,
                normal: n,
                color,
            });
        }
        mesh.indices.extend_from_slice(&[b, b + 1, b + 2]);
        // Base cap
        let bc = mesh.vertices.len() as u32;
        for &p in &[base_center, p0, p1] {
            mesh.vertices.push(MeshVertex {
                position: p,
                normal: [0.0, -1.0, 0.0],
                color,
            });
        }
        mesh.indices.extend_from_slice(&[bc, bc + 1, bc + 2]);
    }
}

/// 3D segment endpoint pair `(start, end)` in Ångströms.
pub type Segment3D = ([f32; 3], [f32; 3]);

/// Build official **SNFG 3D Glycan** geometric glyphs (Blue Cubes = GlcNAc, Green Spheres = Man,
/// Yellow Spheres = Gal, Red Cones = Fuc, Purple Diamonds = Neu5Ac) and covalent glycosidic rods.
#[must_use]
pub fn build_snfg_glycans_mesh(mol: &Biomolecule) -> (TriangleMesh, Vec<Segment3D>) {
    let report = detect_snfg_glycans(mol);
    let mut mesh = TriangleMesh::default();
    for nd in &report.nodes {
        let c = [
            nd.centroid[0] as f32,
            nd.centroid[1] as f32,
            nd.centroid[2] as f32,
        ];
        let r = nd.radius_a as f32;
        let col = [nd.rgb[0], nd.rgb[1], nd.rgb[2], 1.0];
        match nd.shape {
            SnfgShape::Cube => push_oriented_slab(
                &mut mesh,
                c,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [r * 0.92, r * 0.92, r * 0.92],
                col,
            ),
            SnfgShape::Sphere => push_uv_sphere(&mut mesh, c, r * 0.96, col),
            SnfgShape::Cone => push_cone(&mut mesh, c, r * 1.05, col),
            SnfgShape::Diamond | SnfgShape::Star => push_octahedron(&mut mesh, c, r * 1.15, col),
        }
    }
    let links: Vec<([f32; 3], [f32; 3])> = report
        .links
        .iter()
        .map(|l| {
            (
                [l.from[0] as f32, l.from[1] as f32, l.from[2] as f32],
                [l.to[0] as f32, l.to[1] as f32, l.to[2] as f32],
            )
        })
        .collect();
    (mesh, links)
}

/// Build 3 orthogonal translucent **Orthoplanes (XY / XZ / YZ)** slicing through a `VolumeGrid3D`
/// at normalized slice coordinates `frac = [fx, fy, fz]` in `[0.05..=0.95]` (ChimeraX Orthoplanes).
#[must_use]
pub fn build_orthoplanes_mesh(grid: &VolumeGrid3D, frac: [f32; 3], alpha: f32) -> TriangleMesh {
    let [nx, ny, nz] = grid.dims;
    if nx < 3 || ny < 3 || nz < 3 {
        return TriangleMesh::default();
    }
    let mut mesh = TriangleMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        is_transparent: true,
        wireframe: false,
    };
    let a = alpha.clamp(0.18, 0.92);
    let step_x = (nx / 28).max(1);
    let step_y = (ny / 28).max(1);
    let step_z = (nz / 28).max(1);

    let density_color = |val: f32| -> [f32; 4] {
        let sig = ((f64::from(val) - grid.mean) / grid.std_dev.max(1e-6)) as f32;
        let t = ((sig + 0.5) / 4.5).clamp(0.0, 1.0);
        if t < 0.5 {
            let u = t * 2.0;
            [0.08 + 0.14 * u, 0.16 + 0.62 * u, 0.34 + 0.60 * u, a]
        } else {
            let u = (t - 0.5) * 2.0;
            [0.22 + 0.76 * u, 0.78 + 0.14 * u, 0.94 * (1.0 - 0.72 * u), a]
        }
    };

    let push_quad = |m: &mut TriangleMesh,
                     p00: [f32; 3],
                     p10: [f32; 3],
                     p11: [f32; 3],
                     p01: [f32; 3],
                     c00: [f32; 4],
                     c10: [f32; 4],
                     c11: [f32; 4],
                     c01: [f32; 4],
                     nrm: [f32; 3]| {
        let base = m.vertices.len() as u32;
        m.vertices.push(MeshVertex {
            position: p00,
            normal: nrm,
            color: c00,
        });
        m.vertices.push(MeshVertex {
            position: p10,
            normal: nrm,
            color: c10,
        });
        m.vertices.push(MeshVertex {
            position: p11,
            normal: nrm,
            color: c11,
        });
        m.vertices.push(MeshVertex {
            position: p01,
            normal: nrm,
            color: c01,
        });
        m.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    let pt = |x: usize, y: usize, z: usize| -> [f32; 3] {
        let p = grid.voxel_world(x, y, z);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    // 1. XY slice at fixed z
    let iz = ((frac[2].clamp(0.05, 0.95)) * ((nz - 1) as f32)).round() as usize;
    let mut y = 0usize;
    while y + step_y < ny {
        let y1 = (y + step_y).min(ny - 1);
        let mut x = 0usize;
        while x + step_x < nx {
            let x1 = (x + step_x).min(nx - 1);
            push_quad(
                &mut mesh,
                pt(x, y, iz),
                pt(x1, y, iz),
                pt(x1, y1, iz),
                pt(x, y1, iz),
                density_color(grid.get(x, y, iz)),
                density_color(grid.get(x1, y, iz)),
                density_color(grid.get(x1, y1, iz)),
                density_color(grid.get(x, y1, iz)),
                [0.0, 0.0, 1.0],
            );
            x += step_x;
        }
        y += step_y;
    }

    // 2. XZ slice at fixed y
    let iy = ((frac[1].clamp(0.05, 0.95)) * ((ny - 1) as f32)).round() as usize;
    let mut z = 0usize;
    while z + step_z < nz {
        let z1 = (z + step_z).min(nz - 1);
        let mut x = 0usize;
        while x + step_x < nx {
            let x1 = (x + step_x).min(nx - 1);
            push_quad(
                &mut mesh,
                pt(x, iy, z),
                pt(x1, iy, z),
                pt(x1, iy, z1),
                pt(x, iy, z1),
                density_color(grid.get(x, iy, z)),
                density_color(grid.get(x1, iy, z)),
                density_color(grid.get(x1, iy, z1)),
                density_color(grid.get(x, iy, z1)),
                [0.0, 1.0, 0.0],
            );
            x += step_x;
        }
        z += step_z;
    }

    // 3. YZ slice at fixed x
    let ix = ((frac[0].clamp(0.05, 0.95)) * ((nx - 1) as f32)).round() as usize;
    let mut z = 0usize;
    while z + step_z < nz {
        let z1 = (z + step_z).min(nz - 1);
        let mut y = 0usize;
        while y + step_y < ny {
            let y1 = (y + step_y).min(ny - 1);
            push_quad(
                &mut mesh,
                pt(ix, y, z),
                pt(ix, y1, z),
                pt(ix, y1, z1),
                pt(ix, y, z1),
                density_color(grid.get(ix, y, z)),
                density_color(grid.get(ix, y1, z)),
                density_color(grid.get(ix, y1, z1)),
                density_color(grid.get(ix, y, z1)),
                [1.0, 0.0, 0.0],
            );
            y += step_y;
        }
        z += step_z;
    }

    mesh
}

/// Build smooth translucent Marching-Tetrahedra 3D cavity meshes for detected binding pockets.
///
/// The active/selected pocket (or Pocket #1 by default) is rendered in luminous emerald-teal;
/// secondary cavities use distinct gold/cyan/violet hues.
#[must_use]
pub fn build_pockets_mesh(
    pockets: &[BindingPocket3D],
    selected_pocket_id: Option<usize>,
) -> TriangleMesh {
    let mut combined = TriangleMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        is_transparent: true,
        wireframe: false,
    };
    const POCKET_PALETTE: &[[f32; 3]] = &[
        [0.14, 0.88, 0.58], // Pocket #1 / selected emerald
        [0.98, 0.74, 0.18], // Pocket #2 warm gold
        [0.24, 0.78, 0.98], // Pocket #3 sky cyan
        [0.74, 0.50, 0.98], // Pocket #4 soft violet
        [0.96, 0.46, 0.62], // Pocket #5 coral rose
    ];

    let active_id = selected_pocket_id.or_else(|| pockets.first().map(|p| p.pocket_id));

    for (idx, p) in pockets.iter().enumerate() {
        if p.cavity_voxels.is_empty() {
            continue;
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in &p.cavity_voxels {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        let step = p.voxel_step_a.clamp(0.95, 2.0);
        let pad = step * 2.5;
        for k in 0..3 {
            lo[k] -= pad;
            hi[k] += pad;
        }
        let gstep = (step * 0.88).clamp(0.85, 1.85);
        let nx = (((hi[0] - lo[0]) / gstep).ceil() as usize + 1).clamp(6, 48);
        let ny = (((hi[1] - lo[1]) / gstep).ceil() as usize + 1).clamp(6, 48);
        let nz = (((hi[2] - lo[2]) / gstep).ceil() as usize + 1).clamp(6, 48);
        let mut values = vec![0.0_f32; nx * ny * nz];
        let sigma = step * 0.95;
        let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
        let cut = sigma * 2.3;
        let cut2 = cut * cut;

        for v in &p.cavity_voxels {
            let x0 = (((v[0] - cut - lo[0]) / gstep).floor() as isize).clamp(0, nx as isize - 1)
                as usize;
            let x1 =
                (((v[0] + cut - lo[0]) / gstep).ceil() as isize).clamp(0, nx as isize - 1) as usize;
            let y0 = (((v[1] - cut - lo[1]) / gstep).floor() as isize).clamp(0, ny as isize - 1)
                as usize;
            let y1 =
                (((v[1] + cut - lo[1]) / gstep).ceil() as isize).clamp(0, ny as isize - 1) as usize;
            let z0 = (((v[2] - cut - lo[2]) / gstep).floor() as isize).clamp(0, nz as isize - 1)
                as usize;
            let z1 =
                (((v[2] + cut - lo[2]) / gstep).ceil() as isize).clamp(0, nz as isize - 1) as usize;
            for gz in z0..=z1 {
                let dz = (lo[2] + (gz as f32) * gstep) - v[2];
                let dz2 = dz * dz;
                for gy in y0..=y1 {
                    let dy = (lo[1] + (gy as f32) * gstep) - v[1];
                    let dyz2 = dz2 + dy * dy;
                    if dyz2 > cut2 {
                        continue;
                    }
                    let row = (gz * ny + gy) * nx;
                    for gx in x0..=x1 {
                        let dx = (lo[0] + (gx as f32) * gstep) - v[0];
                        let d2 = dyz2 + dx * dx;
                        if d2 <= cut2 {
                            values[row + gx] += (-d2 * inv_2s2).exp();
                        }
                    }
                }
            }
        }

        let grid = VolumeGrid3D {
            label: format!("pocket_{}", p.pocket_id),
            dims: [nx, ny, nz],
            origin: [f64::from(lo[0]), f64::from(lo[1]), f64::from(lo[2])],
            voxel_size: [f64::from(gstep), f64::from(gstep), f64::from(gstep)],
            values,
            min_val: 0.0,
            max_val: 1.0,
            mean: 0.0,
            std_dev: 1.0,
        };
        let is_selected = active_id == Some(p.pocket_id);
        let rgb = if is_selected {
            POCKET_PALETTE[0]
        } else {
            POCKET_PALETTE[(idx + 1).min(POCKET_PALETTE.len() - 1)]
        };
        let alpha = if is_selected { 0.50 } else { 0.34 };
        let sub = extract_isosurface_mesh(&grid, 0.62, |_| rgb, alpha, false, 1);
        combined.extend_from(&sub);
    }

    combined
}
