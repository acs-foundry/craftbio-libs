#![forbid(unsafe_code)]

//! `craftbio-viewport` — Pure-Rust `wgpu 24 + WGSL` Biomolecular 3D Viewport + Deterministic
//! Headless Publication PNG Renderer shared across `BiomolCraft`, `BioCraft`, `DockCraft`, and `ChemCraft`.

pub mod camera;
pub mod geometry;
pub mod gpu_renderer;
pub mod headless_png;

pub use camera::{Camera3D, cross3, dot3, normalize3};
pub use geometry::{
    CapsuleInstance, RenderOptions, RenderSceneData, SphereInstance, build_render_scene,
};
pub use gpu_renderer::{SharedViewportScene, show_biomol_viewport};
pub use headless_png::{RenderError, render_scene_png_bytes, render_scene_rgba, write_scene_png};

#[cfg(test)]
mod tests {
    use super::*;
    use craftbio_struct::parse_pdb;

    #[test]
    fn builds_biomolecular_scene_and_renders_valid_png_headlessly()
    -> Result<(), Box<dyn std::error::Error>> {
        let pdb = "\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00 88.00           C
ATOM      3  CA  TYR A   3       7.200   1.600   0.000  1.00 91.00           C
ATOM      4  C4'   G B   1       0.000   5.000   0.000  1.00 80.00           C
ATOM      5  C1'   G B   1       1.800   5.800   0.000  1.00 80.00           C
ATOM      6  N1    G B   1       3.200   6.000   0.000  1.00 80.00           N
ATOM      7  C4'   C B   2       5.800   5.200   0.000  1.00 80.00           C
ATOM      8  N3    C B   2       4.600   6.100   0.000  1.00 80.00           N
END
";
        let mol = parse_pdb("vp_test", pdb)?;
        let scene = build_render_scene(&mol, &RenderOptions::default());
        assert!(!scene.capsules.is_empty());
        let mut cam = Camera3D::default();
        cam.focus_on_biomolecule(&mol);
        let out = std::env::temp_dir().join(format!("craftbio_vp_{}.png", std::process::id()));
        let bytes = write_scene_png(&scene, &cam, 320, 240, false, &out)?;
        assert!(bytes > 500);
        let raw = std::fs::read(&out)?;
        assert_eq!(&raw[..8], b"\x89PNG\r\n\x1a\n");
        let _ = std::fs::remove_file(&out);
        Ok(())
    }
}
