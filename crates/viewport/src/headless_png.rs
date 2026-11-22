#![forbid(unsafe_code)]

//! Deterministic Headless Publication PNG Renderer (Ray-Cast Spheres, Smooth Ribbons/Capsules,
//! ChimeraX-Style Soft Lighting, Depth Fog, and Silhouette Outlines — up to 4K 3840×2160).

use std::path::Path;

use rayon::prelude::*;
use thiserror::Error;

use crate::camera::{Camera3D, dot3, normalize3};
use crate::geometry::RenderSceneData;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PNG encoding error: {0}")]
    Png(#[from] png::EncodingError),
    #[error("invalid render dimensions {0}x{1}")]
    InvalidDimensions(u32, u32),
}

/// Render `scene` from `camera` into an RGBA8 byte buffer (`width * height * 4`) using exact
/// ray-cast impostor shading + silhouette edge darkening (100% deterministic, zero GPU window needed).
pub fn render_scene_rgba(
    scene: &RenderSceneData,
    camera: &Camera3D,
    width: u32,
    height: u32,
    transparent_bg: bool,
) -> Result<Vec<u8>, RenderError> {
    if width < 16 || height < 16 || width > 4096 || height > 4096 {
        return Err(RenderError::InvalidDimensions(width, height));
    }
    let w = width as usize;
    let h = height as usize;
    let aspect = (width as f32) / (height as f32);
    let eye = camera.eye();
    let (right, up, fwd) = camera.basis();
    let half_h = (0.5 * camera.fov_y_rad).tan();

    // Pre-project spheres & capsules to screen-space bounding boxes for fast tile rasterization
    let project_pt = |p: [f32; 3]| -> Option<(f32, f32, f32, f32)> {
        let rel = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        let z = dot3(rel, fwd);
        if z <= 0.5 {
            return None;
        }
        let scale = if camera.orthographic {
            camera.distance * half_h
        } else {
            z * half_h
        };
        let sx = ((dot3(rel, right) / (scale * aspect)) * 0.5 + 0.5) * (width as f32);
        let sy = (0.5 - (dot3(rel, up) / scale) * 0.5) * (height as f32);
        let px_per_a = (height as f32) / (2.0 * scale);
        Some((sx, sy, z, px_per_a))
    };

    // Build projected sphere & capsule primitives for scanline rendering
    struct ProjSphere {
        sx: f32,
        sy: f32,
        z: f32,
        r_px: f32,
        color: [f32; 4],
    }
    struct ProjCapsule {
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
        r_px: f32,
        dashed: bool,
        color: [f32; 4],
        min_y: i32,
        max_y: i32,
    }

    let mut spheres = Vec::with_capacity(scene.spheres.len());
    for sp in &scene.spheres {
        if let Some((sx, sy, z, ppa)) = project_pt(sp.center) {
            let r_px = (sp.radius * ppa).max(1.0);
            if sx + r_px >= 0.0
                && sx - r_px < width as f32
                && sy + r_px >= 0.0
                && sy - r_px < height as f32
            {
                spheres.push(ProjSphere {
                    sx,
                    sy,
                    z,
                    r_px,
                    color: sp.color,
                });
            }
        }
    }

    let mut capsules = Vec::with_capacity(scene.capsules.len());
    for cap in &scene.capsules {
        if let (Some((ax, ay, az, ppa_a)), Some((bx, by, bz, ppa_b))) =
            (project_pt(cap.a), project_pt(cap.b))
        {
            let r_px = (cap.radius * 0.5 * (ppa_a + ppa_b)).max(0.9);
            let min_x = ax.min(bx) - r_px;
            let max_x = ax.max(bx) + r_px;
            let min_y = (ay.min(by) - r_px).floor() as i32;
            let max_y = (ay.max(by) + r_px).ceil() as i32;
            if max_x >= 0.0 && min_x < width as f32 && max_y >= 0 && min_y < height as i32 {
                capsules.push(ProjCapsule {
                    ax,
                    ay,
                    az,
                    bx,
                    by,
                    bz,
                    r_px,
                    dashed: cap.dashed > 0.5,
                    color: cap.color,
                    min_y,
                    max_y,
                });
            }
        }
    }

    // Key light + soft fill light in camera space (ChimeraX studio lighting)
    let light1 = normalize3([0.38, 0.58, 0.72]);
    let light2 = normalize3([-0.55, -0.25, 0.80]);
    let bg_rgba: [u8; 4] = if transparent_bg {
        [0, 0, 0, 0]
    } else {
        [
            (scene.background[0].clamp(0.0, 1.0) * 255.0) as u8,
            (scene.background[1].clamp(0.0, 1.0) * 255.0) as u8,
            (scene.background[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]
    };

    let z_ref = camera.distance.max(10.0);
    let shade = |nx: f32, ny: f32, nz: f32, z: f32, col: [f32; 4]| -> [u8; 4] {
        let d1 = (nx * light1[0] + ny * light1[1] + nz * light1[2]).max(0.0);
        let d2 = (nx * light2[0] + ny * light2[1] + nz * light2[2]).max(0.0);
        // Blinn-Phong specular highlight
        let h1 = normalize3([light1[0], light1[1], light1[2] + 1.0]);
        let spec = (nx * h1[0] + ny * h1[1] + nz * h1[2]).max(0.0).powi(28) * 0.32;
        // Subtle ChimeraX silhouette rim darkening
        let sil = if nz < 0.24 { 0.42 + 2.4 * nz } else { 1.0 };
        let fog = (1.0 - ((z - z_ref * 0.75) / (z_ref * 2.2)).clamp(0.0, 0.42)).clamp(0.58, 1.0);
        let lit = (0.24 + 0.68 * d1 + 0.20 * d2) * sil * fog;
        let r = ((col[0] * lit + spec * sil) * 255.0).clamp(0.0, 255.0) as u8;
        let g = ((col[1] * lit + spec * sil) * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((col[2] * lit + spec * sil) * 255.0).clamp(0.0, 255.0) as u8;
        let a = (col[3].clamp(0.0, 1.0) * 255.0) as u8;
        [r, g, b, a]
    };

    // Parallel per-row scanline rasterization (zero shared mutable state!)
    let mut rows: Vec<Vec<u8>> = (0..h)
        .into_par_iter()
        .map(|py| {
            let fy = py as f32 + 0.5;
            let mut row_rgba = vec![0u8; w * 4];
            let mut row_z = vec![f32::INFINITY; w];
            for px in 0..w {
                row_rgba[px * 4..px * 4 + 4].copy_from_slice(&bg_rgba);
            }

            // Rasterize spheres overlapping this scanline
            for sp in &spheres {
                let dy = fy - sp.sy;
                if dy.abs() > sp.r_px {
                    continue;
                }
                let span = (sp.r_px * sp.r_px - dy * dy).max(0.0).sqrt();
                let x0 = (sp.sx - span).floor().max(0.0) as usize;
                let x1 = (sp.sx + span).ceil().min((w - 1) as f32) as usize;
                for px in x0..=x1 {
                    let dx = (px as f32 + 0.5) - sp.sx;
                    let r2 = dx * dx + dy * dy;
                    let rad2 = sp.r_px * sp.r_px;
                    if r2 > rad2 {
                        continue;
                    }
                    let nz = (1.0 - r2 / rad2).max(0.0).sqrt();
                    let z = sp.z - nz * (sp.r_px * 0.04);
                    if z < row_z[px] {
                        let nx = dx / sp.r_px;
                        let ny = -dy / sp.r_px;
                        let c = shade(nx, ny, nz, z, sp.color);
                        if sp.color[3] >= 0.95 {
                            row_z[px] = z;
                            row_rgba[px * 4..px * 4 + 4].copy_from_slice(&c);
                        } else {
                            // Alpha blend translucent surface shell over background
                            let alpha = sp.color[3].clamp(0.0, 1.0);
                            for k in 0..3 {
                                let dst = row_rgba[px * 4 + k] as f32;
                                row_rgba[px * 4 + k] =
                                    (c[k] as f32 * alpha + dst * (1.0 - alpha)) as u8;
                            }
                            row_rgba[px * 4 + 3] = row_rgba[px * 4 + 3].max(c[3]);
                        }
                    }
                }
            }

            // Rasterize capsules overlapping this scanline
            let ipy = py as i32;
            for cap in &capsules {
                if ipy < cap.min_y || ipy > cap.max_y {
                    continue;
                }
                let abx = cap.bx - cap.ax;
                let aby = cap.by - cap.ay;
                let ab_len2 = (abx * abx + aby * aby).max(1e-5);
                let x0 = (cap.ax.min(cap.bx) - cap.r_px).floor().max(0.0) as usize;
                let x1 = (cap.ax.max(cap.bx) + cap.r_px).ceil().min((w - 1) as f32) as usize;
                let r2_lim = cap.r_px * cap.r_px;
                for px in x0..=x1 {
                    let fx = px as f32 + 0.5;
                    let apx = fx - cap.ax;
                    let apy = fy - cap.ay;
                    let t = ((apx * abx + apy * aby) / ab_len2).clamp(0.0, 1.0);
                    if cap.dashed && ((t * 9.0).floor() as i32) % 2 == 1 {
                        continue;
                    }
                    let cx = cap.ax + t * abx;
                    let cy = cap.ay + t * aby;
                    let dx = fx - cx;
                    let dy = fy - cy;
                    let d2 = dx * dx + dy * dy;
                    if d2 > r2_lim {
                        continue;
                    }
                    let nz = (1.0 - d2 / r2_lim).max(0.0).sqrt();
                    let z = cap.az + t * (cap.bz - cap.az) - nz * (cap.r_px * 0.03);
                    if z < row_z[px] {
                        let nx = dx / cap.r_px;
                        let ny = -dy / cap.r_px;
                        let c = shade(nx, ny, nz, z, cap.color);
                        row_z[px] = z;
                        row_rgba[px * 4..px * 4 + 4].copy_from_slice(&c);
                    }
                }
            }
            row_rgba
        })
        .collect();

    let mut out = Vec::with_capacity(w * h * 4);
    for r in &mut rows {
        out.append(r);
    }
    Ok(out)
}

/// Render `scene` from `camera` directly into encoded RGBA8 PNG bytes in memory.
pub fn render_scene_png_bytes(
    scene: &RenderSceneData,
    camera: &Camera3D,
    width: u32,
    height: u32,
    transparent_bg: bool,
) -> Result<Vec<u8>, RenderError> {
    let rgba = render_scene_rgba(scene, camera, width, height, transparent_bg)?;
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&rgba)?;
    }
    Ok(buf)
}

/// Write a publication-grade RGBA PNG image of `scene` to `out_path`.
pub fn write_scene_png(
    scene: &RenderSceneData,
    camera: &Camera3D,
    width: u32,
    height: u32,
    transparent_bg: bool,
    out_path: &Path,
) -> Result<usize, RenderError> {
    let buf = render_scene_png_bytes(scene, camera, width, height, transparent_bg)?;
    if let Some(p) = out_path.parent()
        && !p.as_os_str().is_empty()
    {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(out_path, &buf)?;
    Ok(buf.len())
}
