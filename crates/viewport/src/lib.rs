#![forbid(unsafe_code)]

//! `craftbio-viewport` — Pure-Rust `wgpu 30 + WGSL` Offscreen `Depth32Float` + SSAO Biomolecular
//! 3D Viewport, Triangulated Cartoon/SES/Cryo-EM Mesh Engine, `.glb`/`.obj` Exporter, and
//! Deterministic Headless 4K Publication PNG Renderer shared across the CraftBio Suite.

pub mod camera;
pub mod geometry;
pub mod gpu_renderer;
pub mod headless_png;
pub mod mesh;
pub mod movie;
pub mod overlay_2d;

pub use camera::{Camera3D, cross3, dot3, normalize3, transform_point_ndc};
pub use geometry::{
    CapsuleInstance, RenderOptions, RenderSceneData, SphereInstance, build_render_scene,
    push_grid_box_wireframe, push_unit_cell_wireframe,
};
pub use gpu_renderer::{SharedViewportScene, show_biomol_viewport, show_biomol_viewport_ex};
pub use headless_png::{RenderError, render_scene_png_bytes, render_scene_rgba, write_scene_png};
pub use mesh::{
    MeshVertex, TriangleMesh, build_cartoon_mesh, build_orthoplanes_mesh, build_pockets_mesh,
    build_putty_mesh, build_ses_surface_mesh, build_snfg_glycans_mesh, export_meshes_to_glb,
    export_meshes_to_obj, extract_isosurface_mesh,
};
pub use movie::{
    encode_gif89a, interpolate_camera, render_fly_frames, render_rock_frames,
    render_turntable_frames, write_movie_file,
};
pub use overlay_2d::{
    composite_2d_overlays, pixels_per_angstrom_at_target, project_world_to_screen,
};

#[cfg(test)]
mod tests {
    use super::*;
    use craftbio_scene::{ColorScheme, GridBox3D, LightingPreset, RepresentationKind};
    use craftbio_struct::{parse_pdb, simulate_cryoem_map};

    #[test]
    fn builds_triangulated_cartoon_ses_cryoem_glb_and_renders_ssao_png()
    -> Result<(), Box<dyn std::error::Error>> {
        let pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A   1       2.050   1.380   0.000  1.00 90.00           C
ATOM      4  O   ALA A   1       1.350   2.380   0.000  1.00 90.00           O
ATOM      5  N   GLY A   2       3.360   1.420   0.000  1.00 88.00           N
ATOM      6  CA  GLY A   2       4.100   2.680   0.200  1.00 88.00           C
ATOM      7  C   GLY A   2       5.580   2.500   0.000  1.00 88.00           C
ATOM      8  O   GLY A   2       6.120   1.420  -0.200  1.00 88.00           O
ATOM      9  N   TYR A   3       6.240   3.650   0.100  1.00 91.00           N
ATOM     10  CA  TYR A   3       7.680   3.750   0.000  1.00 91.00           C
ATOM     11  C   TYR A   3       8.280   5.120   0.100  1.00 91.00           C
ATOM     12  O   TYR A   3       7.600   6.120   0.200  1.00 91.00           O
ATOM     13  C4'   G B   1       0.000   7.000   0.000  1.00 80.00           C
ATOM     14  C1'   G B   1       1.800   7.800   0.000  1.00 80.00           C
ATOM     15  N1    G B   1       3.400   8.100   0.200  1.00 80.00           N
ATOM     16  C6    G B   1       2.900   9.200   0.100  1.00 80.00           C
ATOM     17  C4'   C B   2       5.800   7.200   0.000  1.00 80.00           C
ATOM     18  C1'   C B   2       5.100   8.100   0.000  1.00 80.00           C
ATOM     19  N3    C B   2       4.300   8.900   0.200  1.00 80.00           N
END
";
        let mol = parse_pdb("vp_v04_test", pdb)?;
        let vol = simulate_cryoem_map(&mol, 4.0, 1.4)?;
        let opts = RenderOptions {
            representation: RepresentationKind::Cartoon,
            color_scheme: ColorScheme::LipophilicityMlp,
            show_surface_shell: true,
            surface_alpha: 0.72,
            lighting: LightingPreset::PublicationWhite,
            ssao_strength: 0.65,
            background: [1.0, 1.0, 1.0, 1.0],
            grid_box: Some(GridBox3D {
                center: [4.0, 4.0, 0.0],
                size: [14.0, 14.0, 14.0],
                spacing: 0.375,
            }),
            volume_grid: Some(vol),
            volume_sigma: 1.5,
            ..RenderOptions::default()
        };
        let scene = build_render_scene(&mol, &opts);
        assert!(
            !scene.opaque_meshes.is_empty(),
            "expected triangulated cartoon + RNA slabs mesh"
        );
        assert!(
            scene.transparent_meshes.len() >= 2,
            "expected Marching-Tetrahedra SES surface + Cryo-EM volume mesh"
        );
        assert!(!scene.capsules.is_empty(), "expected GridBox wireframe");

        // Validate binary glTF 2.0 (.glb) & Wavefront (.obj) exports
        let glb = scene.export_glb("vp_v04_test");
        assert!(glb.len() > 256);
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes([glb[4], glb[5], glb[6], glb[7]]), 2);
        assert_eq!(
            u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]) as usize,
            glb.len()
        );
        let obj = scene.export_obj("vp_v04_test");
        assert!(obj.contains("v ") && obj.contains("vn ") && obj.contains("f "));

        // Validate deterministic headless PNG + clip planes
        let mut cam = Camera3D::default();
        cam.focus_on_biomolecule(&mol);
        let out = std::env::temp_dir().join(format!("craftbio_vp04_{}.png", std::process::id()));
        let bytes = write_scene_png(&scene, &cam, 320, 240, false, &out)?;
        assert!(bytes > 1000);
        let raw = std::fs::read(&out)?;
        assert_eq!(&raw[..8], b"\x89PNG\r\n\x1a\n");

        // Verify slab clip planes actually slice the scene cleanly
        cam.clip_slab_enabled = true;
        cam.clip_near_a = 1.5;
        cam.clip_far_a = 1.5;
        let clipped_bytes = render_scene_png_bytes(&scene, &cam, 320, 240, false)?;
        assert_ne!(raw, clipped_bytes);

        let _ = std::fs::remove_file(&out);
        Ok(())
    }

    #[test]
    fn wave5_putty_mesh_overlays_color_key_scalebar_and_gif89a_movie()
    -> Result<(), Box<dyn std::error::Error>> {
        use craftbio_scene::{ColorKeyLegend, Label2D, Label3D, ScaleBarSpec};
        let pdb = "\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 95.00           C
ATOM      2  CA  GLY A   2       3.800   0.400   0.200  1.00 82.00           C
ATOM      3  CA  SER A   3       7.500   1.200  -0.300  1.00 60.00           C
ATOM      4  CA  TYR A   4      11.200   0.800   0.500  1.00 42.00           C
END
";
        let mol = parse_pdb("w5_putty", pdb)?;
        let opts = RenderOptions {
            representation: RepresentationKind::Putty,
            color_scheme: ColorScheme::BfactorPlddt,
            labels_3d: vec![Label3D {
                text: "A:TYR4 3.1 A".to_string(),
                coords: [11.2, 0.8, 0.5],
                rgba: [1.0, 0.92, 0.25, 1.0],
                font_scale: 1.0,
            }],
            labels_2d: vec![Label2D {
                id: "title".to_string(),
                text: "WAVE 5 PUTTY TEST".to_string(),
                norm_pos: [0.04, 0.04],
                rgba: [0.95, 0.98, 1.0, 1.0],
                font_scale: 1.0,
            }],
            color_key: Some(ColorKeyLegend {
                title: "pLDDT".to_string(),
                min_label: "40".to_string(),
                max_label: "100".to_string(),
                stops: vec![[0.98, 0.48, 0.25], [0.99, 0.84, 0.22], [0.05, 0.35, 0.86]],
            }),
            scale_bar: Some(ScaleBarSpec {
                length_a: 10.0,
                label: "10 A".to_string(),
            }),
            ..RenderOptions::default()
        };
        let scene = build_render_scene(&mol, &opts);
        assert!(
            !scene.opaque_meshes.is_empty(),
            "expected triangulated Putty tube mesh"
        );
        let mut cam = Camera3D::default();
        cam.focus_on_biomolecule(&mol);
        let png_bytes = render_scene_png_bytes(&scene, &cam, 320, 240, false)?;
        assert!(png_bytes.len() > 2000 && &png_bytes[..8] == b"\x89PNG\r\n\x1a\n");

        // Render 6-frame turntable GIF89a movie (pure Rust, zero external crates!)
        let frames = render_turntable_frames(&scene, &cam, 160, 120, 6)?;
        assert_eq!(frames.len(), 6);
        let gif_path = std::env::temp_dir().join(format!("craftbio_w5_{}.gif", std::process::id()));
        let gif_len = write_movie_file(&gif_path, 160, 120, 10, &frames)?;
        assert!(gif_len > 1000);
        let gif_raw = std::fs::read(&gif_path)?;
        assert_eq!(&gif_raw[..6], b"GIF89a");
        assert_eq!(gif_raw.last().copied(), Some(0x3B));
        let _ = std::fs::remove_file(&gif_path);
        Ok(())
    }

    #[test]
    fn wave6_snfg_glycans_unitcell_and_orthoplanes_render() -> Result<(), Box<dyn std::error::Error>>
    {
        let pdb = "\
CRYST1   18.000   18.000   18.000  90.00  90.00  90.00 P 21 21 21    4
ATOM      1  CA  ASN A 297       4.000   4.000   4.000  1.00 22.00           C
ATOM      2  ND2 ASN A 297       5.500   4.000   4.000  1.00 22.00           N
HETATM    3  C1  NAG A 501       6.950   4.000   4.000  1.00 35.00           C
HETATM    4  C2  NAG A 501       7.700   5.100   4.000  1.00 35.00           C
HETATM    5  C3  NAG A 501       9.100   5.000   4.000  1.00 35.00           C
HETATM    6  C1  MAN A 502      10.900   5.000   4.000  1.00 38.00           C
HETATM    7  C1  FUC A 503       8.000   2.400   5.200  1.00 40.00           C
END
";
        let mol = parse_pdb("w6_vp", pdb)?;
        let vol = simulate_cryoem_map(&mol, 4.0, 1.4)?;
        let opts = RenderOptions {
            show_snfg_glycans: true,
            show_unit_cell: true,
            volume_grid: Some(vol),
            show_orthoplanes: true,
            ..RenderOptions::default()
        };
        let scene = build_render_scene(&mol, &opts);
        assert!(
            !scene.opaque_meshes.is_empty(),
            "expected 3D SNFG glycan meshes"
        );
        assert!(
            scene.transparent_meshes.len() >= 2,
            "expected isosurface + orthoplanes meshes"
        );
        assert!(
            scene.capsules.len() >= 12,
            "expected 12 CRYST1 unit-cell edges + glycan rods"
        );
        let mut cam = Camera3D::default();
        cam.focus_on_biomolecule(&mol);
        let png = render_scene_png_bytes(&scene, &cam, 240, 180, false)?;
        assert!(png.len() > 1000 && &png[..8] == b"\x89PNG\r\n\x1a\n");
        Ok(())
    }

    #[test]
    fn wave8_pcf_soft_shadows_pockets_mesh_and_drag_unproject()
    -> Result<(), Box<dyn std::error::Error>> {
        use craftbio_struct::detect_binding_pockets;

        // Build two overlapping stacked protein sheets so the upper sheet casts a clear shadow on the lower one
        let mut pdb = String::new();
        let mut serial = 1usize;
        let mut res_seq = 1i32;
        for iz in -5..=5i32 {
            for iy in -5..=5i32 {
                for ix in -5..=5i32 {
                    let x = ix as f64 * 2.2;
                    let y = iy as f64 * 2.2;
                    let z = iz as f64 * 2.2;
                    let r = (x * x + y * y + z * z).sqrt();
                    let is_mouth = z > 4.5 && (x * x + y * y).sqrt() < 2.8;
                    if (5.6..=10.0).contains(&r) && !is_mouth {
                        pdb.push_str(&format!(
                            "ATOM  {:5}  CA  LEU A{:4}    {:8.3}{:8.3}{:8.3}  1.00 20.00           C\n",
                            serial, res_seq, x, y, z
                        ));
                        serial += 1;
                        res_seq += 1;
                    }
                }
            }
        }
        pdb.push_str("END\n");
        let mol = parse_pdb("w8_shadow_pocket", &pdb)?;
        let pockets = detect_binding_pockets(&mol, 3);
        assert!(!pockets.is_empty());

        let mut opts_no_shadow = RenderOptions {
            representation: RepresentationKind::Spheres,
            shadow_strength: 0.0,
            ssao_strength: 0.0,
            show_silhouettes: false,
            binding_pockets: pockets.clone(),
            selected_pocket_id: Some(1),
            ..RenderOptions::default()
        };
        let scene_off = build_render_scene(&mol, &opts_no_shadow);
        assert!(
            !scene_off.transparent_meshes.is_empty(),
            "expected smooth Marching-Tetrahedra binding pocket mesh"
        );

        opts_no_shadow.shadow_strength = 0.85;
        let scene_on = build_render_scene(&mol, &opts_no_shadow);

        let mut cam = Camera3D::default();
        cam.focus_on_biomolecule(&mol);
        let rgba_off = render_scene_rgba(&scene_off, &cam, 240, 180, false)?;
        let rgba_on = render_scene_rgba(&scene_on, &cam, 240, 180, false)?;
        let sum_off: u64 = rgba_off.iter().map(|&b| u64::from(b)).sum();
        let sum_on: u64 = rgba_on.iter().map(|&b| u64::from(b)).sum();
        assert!(
            sum_on < sum_off,
            "expected directional PCF 3x3 shadow map to visibly darken occluded pixels (off={sum_off}, on={sum_on})"
        );

        // Verify exact screen-delta unprojection for mouse Drag-to-Sculpt
        let delta = cam.unproject_screen_delta_at_point([0.0, 0.0, 0.0], 20.0, -10.0, 600.0);
        let mag = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        assert!(
            mag > 0.1 && mag < 20.0,
            "unexpected unproject delta magnitude {mag}"
        );
        Ok(())
    }

    #[test]
    fn wave9_rainbow_bfactor_and_non_stick_residue_colors()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;
        let pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 12.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 12.00           C
ATOM      3  C   ALA A   1       2.050   1.380   0.000  1.00 12.00           C
ATOM      4  O   ALA A   1       1.350   2.380   0.000  1.00 12.00           O
ATOM      5  N   GLY A   2       3.360   1.420   0.000  1.00 45.00           N
ATOM      6  CA  GLY A   2       4.100   2.680   0.200  1.00 45.00           C
ATOM      7  C   GLY A   2       5.580   2.500   0.000  1.00 45.00           C
ATOM      8  O   GLY A   2       6.120   1.420  -0.200  1.00 45.00           O
ATOM      9  N   TYR A   3       6.240   3.650   0.100  1.00 85.00           N
ATOM     10  CA  TYR A   3       7.680   3.750   0.000  1.00 85.00           C
ATOM     11  C   TYR A   3       8.280   5.120   0.100  1.00 85.00           C
ATOM     12  O   TYR A   3       7.600   6.120   0.200  1.00 85.00           O
END
";
        let mol = parse_pdb("w9_pal", pdb)?;
        let mut rc = HashMap::new();
        rc.insert(('A', 1), [0.1, 0.3, 0.9]);
        rc.insert(('A', 2), [0.9, 0.9, 0.9]);
        rc.insert(('A', 3), [0.9, 0.2, 0.2]);

        let sc_rc = build_render_scene(
            &mol,
            &RenderOptions {
                representation: RepresentationKind::Cartoon,
                residue_colors: rc,
                ..RenderOptions::default()
            },
        );
        assert!(
            sc_rc.capsules.is_empty(),
            "residue_colors must NOT force sticks in Cartoon mode"
        );

        let sc_rb = build_render_scene(
            &mol,
            &RenderOptions {
                representation: RepresentationKind::Cartoon,
                color_scheme: ColorScheme::Rainbow,
                ..RenderOptions::default()
            },
        );
        assert!(!sc_rb.opaque_meshes.is_empty());

        let sc_bf = build_render_scene(
            &mol,
            &RenderOptions {
                representation: RepresentationKind::Surface,
                color_scheme: ColorScheme::CrystallographicBfactor,
                surface_alpha: 0.40,
                ..RenderOptions::default()
            },
        );
        assert!(
            !sc_bf.transparent_meshes.is_empty(),
            "surface_alpha < 0.99 must produce translucent surface even in Surface representation"
        );
        Ok(())
    }
}
