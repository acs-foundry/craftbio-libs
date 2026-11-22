#![forbid(unsafe_code)]

//! `craftbio-viewport` — Pure-Rust `wgpu 24 + WGSL` Offscreen `Depth32Float` + SSAO Biomolecular
//! 3D Viewport, Triangulated Cartoon/SES/Cryo-EM Mesh Engine, `.glb`/`.obj` Exporter, and
//! Deterministic Headless 4K Publication PNG Renderer shared across the CraftBio Suite.

pub mod camera;
pub mod geometry;
pub mod gpu_renderer;
pub mod headless_png;
pub mod mesh;

pub use camera::{Camera3D, cross3, dot3, normalize3};
pub use geometry::{
    CapsuleInstance, RenderOptions, RenderSceneData, SphereInstance, build_render_scene,
    push_grid_box_wireframe,
};
pub use gpu_renderer::{SharedViewportScene, show_biomol_viewport};
pub use headless_png::{RenderError, render_scene_png_bytes, render_scene_rgba, write_scene_png};
pub use mesh::{
    MeshVertex, TriangleMesh, build_cartoon_mesh, build_ses_surface_mesh, export_meshes_to_glb,
    export_meshes_to_obj, extract_isosurface_mesh,
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
}
