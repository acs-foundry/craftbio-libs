#![forbid(unsafe_code)]

//! Biomolecular Geometry Builder: converts `Biomolecule` + visual styles, volumetric maps,
//! `GridBox3D`, and module overlays into triangulated GPU meshes (true Cartoon ribbons with
//! β-sheet arrowheads, 3D RNA/DNA base slabs, Marching-Tetrahedra SES molecular surfaces &
//! Cryo-EM isosurfaces) plus instanced ray-cast spheres & capsules.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use craftbio_scene::{
    ColorScheme, GridBox3D, LightingPreset, LineOverlay, RepresentationKind, SphereOverlay,
};
use craftbio_struct::{
    Biomolecule, ResidueClass, SecondaryStructure, VolumeGrid3D, compute_esp_at_points,
    compute_per_atom_mlp, compute_per_atom_sap, cpk_color, dist3, nucleotide_color, vdw_radius,
};

use crate::mesh::{
    TriangleMesh, build_cartoon_mesh, build_ses_surface_mesh, export_meshes_to_glb,
    export_meshes_to_obj, extract_isosurface_mesh,
};

/// Single GPU ray-cast sphere impostor instance.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SphereInstance {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

/// Single GPU ray-cast capsule / cylinder segment instance (bonds, RNA stems, H-bonds, GridBox).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CapsuleInstance {
    pub a: [f32; 3],
    pub radius: f32,
    pub b: [f32; 3],
    pub dashed: f32,
    pub color: [f32; 4],
}

/// Built biomolecular visual scene ready for `wgpu 24` GPU or headless 4K PNG rasterization.
#[derive(Debug, Clone)]
pub struct RenderSceneData {
    pub opaque_meshes: Vec<TriangleMesh>,
    pub transparent_meshes: Vec<TriangleMesh>,
    pub spheres: Vec<SphereInstance>,
    pub capsules: Vec<CapsuleInstance>,
    pub background: [f32; 4],
    pub lighting: LightingPreset,
    pub ssao_strength: f32,
    pub show_silhouettes: bool,
}

impl Default for RenderSceneData {
    fn default() -> Self {
        Self {
            opaque_meshes: Vec::new(),
            transparent_meshes: Vec::new(),
            spheres: Vec::new(),
            capsules: Vec::new(),
            background: [0.06, 0.08, 0.12, 1.0],
            lighting: LightingPreset::Soft,
            ssao_strength: 0.55,
            show_silhouettes: true,
        }
    }
}

impl RenderSceneData {
    /// Merge another model's rendered geometry cleanly into this multi-model scene.
    pub fn merge_from(&mut self, mut other: Self) {
        self.opaque_meshes.append(&mut other.opaque_meshes);
        self.transparent_meshes
            .append(&mut other.transparent_meshes);
        self.spheres.append(&mut other.spheres);
        self.capsules.append(&mut other.capsules);
    }

    /// Export all triangulated meshes in this scene to binary glTF 2.0 (`.glb`).
    #[must_use]
    pub fn export_glb(&self, title: &str) -> Vec<u8> {
        let mut refs: Vec<&TriangleMesh> = Vec::new();
        for m in &self.opaque_meshes {
            refs.push(m);
        }
        for m in &self.transparent_meshes {
            refs.push(m);
        }
        export_meshes_to_glb(&refs, title)
    }

    /// Export all triangulated meshes in this scene to Wavefront `.obj`.
    #[must_use]
    pub fn export_obj(&self, title: &str) -> String {
        let mut refs: Vec<&TriangleMesh> = Vec::new();
        for m in &self.opaque_meshes {
            refs.push(m);
        }
        for m in &self.transparent_meshes {
            refs.push(m);
        }
        export_meshes_to_obj(&refs, title)
    }
}

/// Visual rendering options for `build_render_scene`.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub representation: RepresentationKind,
    pub color_scheme: ColorScheme,
    pub show_surface_shell: bool,
    pub surface_alpha: f32,
    pub surface_wireframe: bool,
    pub show_ligands_sticks: bool,
    pub show_waters: bool,
    pub show_rna_base_pairs: bool,
    pub show_silhouettes: bool,
    pub lighting: LightingPreset,
    pub ssao_strength: f32,
    pub background: [f32; 4],
    /// Optional per-residue custom highlight colors `(chain, res_seq) -> [r, g, b]` (e.g. IMGT CDRs, catalytic triads, RNA motifs).
    pub residue_highlights: HashMap<(char, i32), [f32; 3]>,
    pub sphere_overlays: Vec<SphereOverlay>,
    pub line_overlays: Vec<LineOverlay>,
    pub selected_atoms: Vec<bool>,
    /// Optional docking `GridBox3D` wireframe overlay.
    pub grid_box: Option<GridBox3D>,
    /// Optional volumetric Cryo-EM / APBS map rendered as a true Marching-Tetrahedra isosurface.
    pub volume_grid: Option<VolumeGrid3D>,
    pub volume_sigma: f64,
    pub volume_wireframe: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            representation: RepresentationKind::Cartoon,
            color_scheme: ColorScheme::Chain,
            show_surface_shell: false,
            surface_alpha: 0.72,
            surface_wireframe: false,
            show_ligands_sticks: true,
            show_waters: false,
            show_rna_base_pairs: true,
            show_silhouettes: true,
            lighting: LightingPreset::Soft,
            ssao_strength: 0.55,
            background: [0.06, 0.08, 0.12, 1.0],
            residue_highlights: HashMap::new(),
            sphere_overlays: Vec::new(),
            line_overlays: Vec::new(),
            selected_atoms: Vec::new(),
            grid_box: None,
            volume_grid: None,
            volume_sigma: 1.5,
            volume_wireframe: false,
        }
    }
}

const CHAIN_PALETTE: &[[f32; 3]] = &[
    [0.24, 0.74, 0.96], // sky blue
    [0.18, 0.82, 0.74], // teal
    [0.62, 0.68, 0.76], // cool slate (antigen/chain C)
    [0.97, 0.74, 0.18], // warm gold
    [0.94, 0.46, 0.70], // rose
    [0.66, 0.56, 0.96], // violet
    [0.32, 0.86, 0.52], // emerald
    [0.96, 0.58, 0.26], // orange
];

fn chain_color(chain_idx: usize) -> [f32; 3] {
    CHAIN_PALETTE[chain_idx % CHAIN_PALETTE.len()]
}

fn esp_colormap(phi: f32) -> [f32; 3] {
    let t = (phi / 5.0).clamp(-1.0, 1.0);
    if t < 0.0 {
        let u = -t; // negative -> red
        [0.95, 0.95 * (1.0 - 0.82 * u), 0.95 * (1.0 - 0.82 * u)]
    } else {
        let u = t; // positive -> blue
        [0.95 * (1.0 - 0.82 * u), 0.95 * (1.0 - 0.55 * u), 0.98]
    }
}

/// Official ChimeraX MLP colormap: dark cyan (hydrophilic) -> white (neutral) -> dark goldenrod (lipophilic).
fn mlp_colormap(mlp: f32) -> [f32; 3] {
    let t = (mlp / 0.75).clamp(-1.0, 1.0);
    if t < 0.0 {
        let u = -t;
        [
            0.95 * (1.0 - 0.96 * u),
            0.95 * (1.0 - 0.42 * u),
            0.96 * (1.0 - 0.40 * u),
        ]
    } else {
        let u = t;
        [
            0.95 * (1.0 - 0.24 * u),
            0.95 * (1.0 - 0.44 * u),
            0.95 * (1.0 - 0.94 * u),
        ]
    }
}

fn sap_colormap(sap: f32) -> [f32; 3] {
    let t = (sap / 0.85).clamp(-1.0, 1.0);
    if t > 0.0 {
        // hydrophobic patch -> warm red/amber
        [0.96, 0.94 * (1.0 - 0.78 * t), 0.94 * (1.0 - 0.85 * t)]
    } else {
        // hydrophilic -> soft blue
        let u = -t;
        [0.90 * (1.0 - 0.55 * u), 0.93 * (1.0 - 0.30 * u), 0.98]
    }
}

fn plddt_colormap(b: f64) -> [f32; 3] {
    if b >= 90.0 {
        [0.05, 0.35, 0.86]
    } else if b >= 70.0 {
        [0.39, 0.79, 0.96]
    } else if b >= 50.0 {
        [0.99, 0.84, 0.22]
    } else {
        [0.98, 0.48, 0.25]
    }
}

/// Emit a crisp 12-edge 3D wireframe docking `GridBox3D` with corner anchor spheres.
pub fn push_grid_box_wireframe(scene: &mut RenderSceneData, gb: &GridBox3D, rgba: [f32; 4]) {
    let hx = (gb.size[0] * 0.5) as f32;
    let hy = (gb.size[1] * 0.5) as f32;
    let hz = (gb.size[2] * 0.5) as f32;
    let cx = gb.center[0] as f32;
    let cy = gb.center[1] as f32;
    let cz = gb.center[2] as f32;
    let corners = [
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ];
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
    for &(u, v) in &edges {
        scene.capsules.push(CapsuleInstance {
            a: corners[u],
            radius: 0.14,
            b: corners[v],
            dashed: 0.0,
            color: rgba,
        });
    }
    for &c in &corners {
        scene.spheres.push(SphereInstance {
            center: c,
            radius: 0.24,
            color: rgba,
        });
    }
}

/// Build triangulated meshes + instanced spheres & capsules for GPU or headless 4K PNG rendering.
#[must_use]
pub fn build_render_scene(mol: &Biomolecule, opts: &RenderOptions) -> RenderSceneData {
    let mut scene = RenderSceneData {
        opaque_meshes: Vec::new(),
        transparent_meshes: Vec::new(),
        spheres: Vec::with_capacity(mol.atoms.len().min(4096)),
        capsules: Vec::with_capacity(mol.bonds.len().min(4096)),
        background: opts.background,
        lighting: opts.lighting,
        ssao_strength: opts.ssao_strength,
        show_silhouettes: opts.show_silhouettes,
    };

    let sap_vals = if opts.color_scheme == ColorScheme::Sap3D {
        Some(compute_per_atom_sap(mol, 6.5))
    } else {
        None
    };
    let esp_vals = if opts.color_scheme == ColorScheme::ElectrostaticEsp {
        let pts: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
        Some(compute_esp_at_points(mol, &pts))
    } else {
        None
    };
    let mlp_vals = if opts.color_scheme == ColorScheme::LipophilicityMlp {
        Some(compute_per_atom_mlp(mol, 8.0))
    } else {
        None
    };

    let mut chain_index_map: HashMap<char, usize> = HashMap::new();
    for (ci, ch) in mol.chains.iter().enumerate() {
        chain_index_map.insert(ch.id, ci);
    }

    let mut ss_by_atom = vec![SecondaryStructure::Coil; mol.atoms.len()];
    for ch in &mol.chains {
        for r in &ch.residues {
            for &ai in &r.atom_indices {
                ss_by_atom[ai] = r.ss;
            }
        }
    }

    let atom_color = |idx: usize, ss: SecondaryStructure| -> [f32; 3] {
        let a = &mol.atoms[idx];
        if let Some(&hi) = opts.residue_highlights.get(&(a.chain, a.res_seq)) {
            return hi;
        }
        if opts.selected_atoms.get(idx).copied().unwrap_or(false) {
            return [0.99, 0.90, 0.25];
        }
        match opts.color_scheme {
            ColorScheme::ElementCpk => cpk_color(&a.element),
            ColorScheme::Nucleotide => {
                if matches!(a.res_class, ResidueClass::Rna | ResidueClass::Dna) {
                    nucleotide_color(a.res_code)
                } else {
                    chain_color(chain_index_map.get(&a.chain).copied().unwrap_or(0))
                }
            }
            ColorScheme::SecondaryStructure => match ss {
                SecondaryStructure::Helix => [0.92, 0.28, 0.42],
                SecondaryStructure::Sheet => [0.96, 0.72, 0.16],
                SecondaryStructure::Coil => [0.35, 0.76, 0.88],
                SecondaryStructure::NucleicRibbon => nucleotide_color(a.res_code),
            },
            ColorScheme::BfactorPlddt => plddt_colormap(a.b_factor),
            ColorScheme::Sap3D => sap_vals
                .as_ref()
                .map_or([0.8, 0.8, 0.8], |v| sap_colormap(v[idx])),
            ColorScheme::ElectrostaticEsp => esp_vals
                .as_ref()
                .map_or([0.8, 0.8, 0.8], |v| esp_colormap(v[idx])),
            ColorScheme::LipophilicityMlp => mlp_vals
                .as_ref()
                .map_or([0.8, 0.8, 0.8], |v| mlp_colormap(v[idx])),
            ColorScheme::Chain | ColorScheme::CustomHighlight => {
                if a.res_class == ResidueClass::Ligand || a.res_class == ResidueClass::Ion {
                    cpk_color(&a.element)
                } else if matches!(a.res_class, ResidueClass::Rna | ResidueClass::Dna)
                    && opts.representation == RepresentationKind::NucleicLadder
                {
                    nucleotide_color(a.res_code)
                } else {
                    chain_color(chain_index_map.get(&a.chain).copied().unwrap_or(0))
                }
            }
        }
    };

    // 1. True Triangulated Cartoon Ribbons (flat α-helices, β-sheet directional arrowheads,
    // smooth coils, and 3D rectangular RNA/DNA base slabs!)
    let show_ribbons = matches!(
        opts.representation,
        RepresentationKind::Cartoon | RepresentationKind::NucleicLadder
    );
    if show_ribbons {
        let base_col = |chain: char, res_seq: i32, code: char| -> [f32; 3] {
            if let Some(&hi) = opts.residue_highlights.get(&(chain, res_seq)) {
                hi
            } else {
                nucleotide_color(code)
            }
        };
        let ribbon_mesh = build_cartoon_mesh(mol, atom_color, base_col, true);
        if !ribbon_mesh.indices.is_empty() {
            scene.opaque_meshes.push(ribbon_mesh);
        }
        // Short glycosidic stem linking backbone C4' to RNA/DNA base slab
        for ch in &mol.chains {
            for r in &ch.residues {
                if matches!(r.class, ResidueClass::Rna | ResidueClass::Dna)
                    && let (Some(gi), Some(ti)) = (r.guide_idx, r.orient_idx.or(r.base_tip_idx))
                {
                    let pa = mol.atoms[gi].coords;
                    let pb = mol.atoms[ti].coords;
                    let col = base_col(r.chain, r.res_seq, r.code);
                    scene.capsules.push(CapsuleInstance {
                        a: [pa[0] as f32, pa[1] as f32, pa[2] as f32],
                        radius: 0.22,
                        b: [pb[0] as f32, pb[1] as f32, pb[2] as f32],
                        dashed: 0.0,
                        color: [col[0], col[1], col[2], 1.0],
                    });
                }
            }
        }
    }

    // 2. Smooth Triangulated SES Molecular Surface (Marching Tetrahedra)
    if opts.representation == RepresentationKind::Surface || opts.show_surface_shell {
        let atom_cols: Vec<[f32; 3]> = (0..mol.atoms.len())
            .map(|i| atom_color(i, ss_by_atom[i]))
            .collect();
        let alpha = if opts.representation == RepresentationKind::Surface && !opts.surface_wireframe
        {
            1.0
        } else {
            opts.surface_alpha.clamp(0.18, 0.96)
        };
        let surf_mesh = build_ses_surface_mesh(
            mol,
            &atom_cols,
            alpha,
            opts.surface_wireframe,
            opts.show_waters,
        );
        if !surf_mesh.indices.is_empty() {
            if surf_mesh.is_transparent || surf_mesh.wireframe {
                scene.transparent_meshes.push(surf_mesh);
            } else {
                scene.opaque_meshes.push(surf_mesh);
            }
        }
    }

    // 3. Optional Volumetric Cryo-EM / APBS Isosurface Mesh (true Marching Tetrahedra)
    if let Some(ref vol) = opts.volume_grid {
        let iso = (vol.mean + opts.volume_sigma * vol.std_dev) as f32;
        let vol_mesh = extract_isosurface_mesh(
            vol,
            iso,
            |_| [0.28, 0.78, 0.96],
            if opts.volume_wireframe { 0.85 } else { 0.36 },
            opts.volume_wireframe,
            1,
        );
        if !vol_mesh.indices.is_empty() {
            scene.transparent_meshes.push(vol_mesh);
        }
    }

    // 4. RNA/DNA Base-Pair Dashed Rungs
    if opts.show_rna_base_pairs {
        for bp in &mol.base_pairs {
            let tip_a = mol
                .chain(bp.chain_a)
                .and_then(|c| c.residues.iter().find(|r| r.res_seq == bp.res_seq_a))
                .and_then(|r| r.base_tip_idx.or(r.orient_idx));
            let tip_b = mol
                .chain(bp.chain_b)
                .and_then(|c| c.residues.iter().find(|r| r.res_seq == bp.res_seq_b))
                .and_then(|r| r.base_tip_idx.or(r.orient_idx));
            if let (Some(ia), Some(ib)) = (tip_a, tip_b) {
                let pa = mol.atoms[ia].coords;
                let pb = mol.atoms[ib].coords;
                let col = if bp.kind.contains("G-C") {
                    [0.20, 0.88, 0.65, 0.95]
                } else if bp.kind.contains("Wobble") {
                    [0.98, 0.78, 0.22, 0.95]
                } else {
                    [0.45, 0.82, 0.98, 0.95]
                };
                scene.capsules.push(CapsuleInstance {
                    a: [pa[0] as f32, pa[1] as f32, pa[2] as f32],
                    radius: 0.13,
                    b: [pb[0] as f32, pb[1] as f32, pb[2] as f32],
                    dashed: 1.0,
                    color: col,
                });
            }
        }
    }

    // 5. Sticks / Spheres / Ligands / Highlighted Residues
    let atom_shown_as_sticks = |idx: usize| -> bool {
        let a = &mol.atoms[idx];
        if a.res_class == ResidueClass::Water && !opts.show_waters {
            return false;
        }
        if opts.representation == RepresentationKind::Sticks
            || opts.representation == RepresentationKind::Lines
        {
            return true;
        }
        if opts.show_ligands_sticks && a.res_class == ResidueClass::Ligand {
            return true;
        }
        opts.residue_highlights.contains_key(&(a.chain, a.res_seq))
            || opts.selected_atoms.get(idx).copied().unwrap_or(false)
    };

    for bond in &mol.bonds {
        if atom_shown_as_sticks(bond.a) && atom_shown_as_sticks(bond.b) {
            let aa = &mol.atoms[bond.a];
            let ab = &mol.atoms[bond.b];
            let pa = [
                aa.coords[0] as f32,
                aa.coords[1] as f32,
                aa.coords[2] as f32,
            ];
            let pb = [
                ab.coords[0] as f32,
                ab.coords[1] as f32,
                ab.coords[2] as f32,
            ];
            let mid = [
                0.5 * (pa[0] + pb[0]),
                0.5 * (pa[1] + pb[1]),
                0.5 * (pa[2] + pb[2]),
            ];
            let rad = if opts.representation == RepresentationKind::Lines {
                0.08
            } else {
                0.20
            };
            let ca = atom_color(bond.a, ss_by_atom[bond.a]);
            let cb = atom_color(bond.b, ss_by_atom[bond.b]);
            scene.capsules.push(CapsuleInstance {
                a: pa,
                radius: rad,
                b: mid,
                dashed: 0.0,
                color: [ca[0], ca[1], ca[2], 1.0],
            });
            scene.capsules.push(CapsuleInstance {
                a: mid,
                radius: rad,
                b: pb,
                dashed: 0.0,
                color: [cb[0], cb[1], cb[2], 1.0],
            });
        }
    }

    for (idx, a) in mol.atoms.iter().enumerate() {
        if a.res_class == ResidueClass::Water && !opts.show_waters {
            continue;
        }
        let pos = [a.coords[0] as f32, a.coords[1] as f32, a.coords[2] as f32];
        let col = atom_color(idx, ss_by_atom[idx]);
        if opts.representation == RepresentationKind::Spheres || a.res_class == ResidueClass::Ion {
            let r = (vdw_radius(&a.element) as f32)
                * if a.res_class == ResidueClass::Ion {
                    0.85
                } else {
                    0.95
                };
            scene.spheres.push(SphereInstance {
                center: pos,
                radius: r,
                color: [col[0], col[1], col[2], 1.0],
            });
        } else if atom_shown_as_sticks(idx) {
            scene.spheres.push(SphereInstance {
                center: pos,
                radius: 0.25,
                color: [col[0], col[1], col[2], 1.0],
            });
        }
    }

    // 6. Disulfide yellow bridges
    for ss in &mol.disulfides {
        let a_sg = mol
            .chain(ss.chain_a)
            .and_then(|c| c.residues.iter().find(|r| r.res_seq == ss.res_seq_a))
            .and_then(|r| {
                r.atom_indices
                    .iter()
                    .copied()
                    .find(|&i| mol.atoms[i].name == "SG")
            });
        let b_sg = mol
            .chain(ss.chain_b)
            .and_then(|c| c.residues.iter().find(|r| r.res_seq == ss.res_seq_b))
            .and_then(|r| {
                r.atom_indices
                    .iter()
                    .copied()
                    .find(|&i| mol.atoms[i].name == "SG")
            });
        if let (Some(ia), Some(ib)) = (a_sg, b_sg) {
            let pa = mol.atoms[ia].coords;
            let pb = mol.atoms[ib].coords;
            if dist3(pa, pb) <= 2.75 {
                scene.capsules.push(CapsuleInstance {
                    a: [pa[0] as f32, pa[1] as f32, pa[2] as f32],
                    radius: 0.25,
                    b: [pb[0] as f32, pb[1] as f32, pb[2] as f32],
                    dashed: 0.0,
                    color: [0.96, 0.82, 0.18, 1.0],
                });
            }
        }
    }

    // 7. Optional 3D GridBox Wireframe
    if let Some(ref gb) = opts.grid_box {
        push_grid_box_wireframe(&mut scene, gb, [0.98, 0.82, 0.22, 0.95]);
    }

    // 8. Module Overlays (Ala-Scan hotspots, HYDE coronas, ΔMIF, pharmacophores, H-bonds)
    for sp in &opts.sphere_overlays {
        scene.spheres.push(SphereInstance {
            center: [
                sp.coords[0] as f32,
                sp.coords[1] as f32,
                sp.coords[2] as f32,
            ],
            radius: sp.radius_a as f32,
            color: sp.rgba,
        });
    }
    for ln in &opts.line_overlays {
        scene.capsules.push(CapsuleInstance {
            a: [ln.from[0] as f32, ln.from[1] as f32, ln.from[2] as f32],
            radius: 0.12,
            b: [ln.to[0] as f32, ln.to[1] as f32, ln.to[2] as f32],
            dashed: if ln.dashed { 1.0 } else { 0.0 },
            color: ln.rgba,
        });
    }

    scene
}
