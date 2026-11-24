#![forbid(unsafe_code)]

//! Deterministic Headless Publication PNG Renderer (1:1 Parity with GPU `wgpu 24` Viewport):
//! - Smooth Indexed Triangle Meshes (Cartoon Ribbons, β-Sheet Arrowheads, RNA Base Slabs,
//!   Marching-Tetrahedra SES Molecular Surfaces, 3D Binding Pockets & Cryo-EM Isosurfaces)
//! - Directional `768×768` **3×3 PCF Soft Shadow Maps** aligned to the key studio light
//! - Analytical Ray-Cast Spheres & Capsules with Exact Near/Far Clip Planes
//! - Full-Frame Z-Buffer **SSAO (12-tap Poisson Ambient Occlusion)** & ChimeraX **Depth Silhouettes**
//! - Up to 4K `3840×2160`, 100% deterministic, zero GPU window required.

use std::path::Path;

use craftbio_scene::LightingPreset;
use rayon::prelude::*;
use thiserror::Error;

use crate::camera::{Camera3D, dot3, normalize3, transform_point_ndc};
use crate::geometry::RenderSceneData;
use crate::mesh::TriangleMesh;

const SHADOW_DIM: usize = 768;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PNG encoding error: {0}")]
    Png(#[from] png::EncodingError),
    #[error("invalid render dimensions {0}x{1}")]
    InvalidDimensions(u32, u32),
}

struct ProjTri {
    x: [f32; 3],
    y: [f32; 3],
    z: [f32; 3],
    wp: [[f32; 3]; 3],
    /// Camera-space normals `(nx, ny, nz)` per vertex
    n: [[f32; 3]; 3],
    col: [[f32; 4]; 3],
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    inv_area: f32,
}

struct ProjSphere {
    sx: f32,
    sy: f32,
    z: f32,
    r_px: f32,
    r_a: f32,
    center_world: [f32; 3],
    color: [f32; 4],
}

struct ProjCapsule {
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    a_world: [f32; 3],
    b_world: [f32; 3],
    r_px: f32,
    r_a: f32,
    dashed: bool,
    color: [f32; 4],
    min_y: i32,
    max_y: i32,
}

struct ProjLine {
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    color: [f32; 4],
    min_y: i32,
    max_y: i32,
}

/// Build an orthographic `SHADOW_DIM × SHADOW_DIM` depth buffer (`z in [0, 1]`) from the key studio
/// light across all opaque meshes, opaque spheres, and capsules for 1:1 GPU/headless PCF shadows.
fn build_shadow_depth_map(scene: &RenderSceneData, light_vp: &[[f32; 4]; 4]) -> Vec<f32> {
    let dim = SHADOW_DIM;
    let dim_f = dim as f32;
    let mut zbuf = vec![1.0_f32; dim * dim];

    let to_spx = |p: [f32; 3]| -> (f32, f32, f32) {
        let ndc = transform_point_ndc(light_vp, p);
        (
            (ndc[0] * 0.5 + 0.5) * dim_f,
            (0.5 - ndc[1] * 0.5) * dim_f,
            ndc[2],
        )
    };

    // 1. Opaque meshes
    for mesh in &scene.opaque_meshes {
        let pv: Vec<(f32, f32, f32)> = mesh.vertices.iter().map(|v| to_spx(v.position)).collect();
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (Some(&a), Some(&b), Some(&c)) = (
                pv.get(tri[0] as usize),
                pv.get(tri[1] as usize),
                pv.get(tri[2] as usize),
            ) else {
                continue;
            };
            let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
            if area.abs() < 0.05 {
                continue;
            }
            let inv_area = 1.0 / area;
            let x0 = a.0.min(b.0).min(c.0).floor().clamp(0.0, dim_f - 1.0) as usize;
            let x1 = a.0.max(b.0).max(c.0).ceil().clamp(0.0, dim_f - 1.0) as usize;
            let y0 = a.1.min(b.1).min(c.1).floor().clamp(0.0, dim_f - 1.0) as usize;
            let y1 = a.1.max(b.1).max(c.1).ceil().clamp(0.0, dim_f - 1.0) as usize;
            for py in y0..=y1 {
                let fy = py as f32 + 0.5;
                let row = py * dim;
                for px in x0..=x1 {
                    let fx = px as f32 + 0.5;
                    let w0 = ((b.0 - fx) * (c.1 - fy) - (c.0 - fx) * (b.1 - fy)) * inv_area;
                    if w0 < -1e-4 {
                        continue;
                    }
                    let w1 = ((c.0 - fx) * (a.1 - fy) - (a.0 - fx) * (c.1 - fy)) * inv_area;
                    if w1 < -1e-4 {
                        continue;
                    }
                    let w2 = 1.0 - w0 - w1;
                    if w2 < -1e-4 {
                        continue;
                    }
                    let z = w0 * a.2 + w1 * b.2 + w2 * c.2;
                    if z >= 0.0 && z < zbuf[row + px] {
                        zbuf[row + px] = z;
                    }
                }
            }
        }
    }

    // Light orthographic scales per Ångström (column-major matrix rows 0, 1, 2)
    let scale_x = (light_vp[0][0] * light_vp[0][0]
        + light_vp[1][0] * light_vp[1][0]
        + light_vp[2][0] * light_vp[2][0])
        .sqrt();
    let scale_z = (light_vp[0][2] * light_vp[0][2]
        + light_vp[1][2] * light_vp[1][2]
        + light_vp[2][2] * light_vp[2][2])
        .sqrt();
    let px_per_a = 0.5 * dim_f * scale_x;

    // 2. Opaque spheres
    for sp in &scene.spheres {
        if sp.color[3] < 0.95 {
            continue;
        }
        let (sx, sy, sz) = to_spx(sp.center);
        let r_px = (sp.radius * px_per_a).max(1.0);
        let r_z = sp.radius * scale_z;
        let x0 = (sx - r_px).floor().clamp(0.0, dim_f - 1.0) as usize;
        let x1 = (sx + r_px).ceil().clamp(0.0, dim_f - 1.0) as usize;
        let y0 = (sy - r_px).floor().clamp(0.0, dim_f - 1.0) as usize;
        let y1 = (sy + r_px).ceil().clamp(0.0, dim_f - 1.0) as usize;
        let rad2 = r_px * r_px;
        for py in y0..=y1 {
            let dy = (py as f32 + 0.5) - sy;
            let dy2 = dy * dy;
            let row = py * dim;
            for px in x0..=x1 {
                let dx = (px as f32 + 0.5) - sx;
                let d2 = dx * dx + dy2;
                if d2 > rad2 {
                    continue;
                }
                let nz = (1.0 - d2 / rad2).max(0.0).sqrt();
                let z = (sz - nz * r_z).clamp(0.0, 1.0);
                if z < zbuf[row + px] {
                    zbuf[row + px] = z;
                }
            }
        }
    }

    // 3. Solid opaque capsules (bonds / stems)
    for cap in &scene.capsules {
        if cap.dashed > 0.5 || cap.color[3] < 0.95 {
            continue;
        }
        let (ax, ay, az) = to_spx(cap.a);
        let (bx, by, bz) = to_spx(cap.b);
        let r_px = (cap.radius * px_per_a).max(0.8);
        let r_z = cap.radius * scale_z;
        let x0 = (ax.min(bx) - r_px).floor().clamp(0.0, dim_f - 1.0) as usize;
        let x1 = (ax.max(bx) + r_px).ceil().clamp(0.0, dim_f - 1.0) as usize;
        let y0 = (ay.min(by) - r_px).floor().clamp(0.0, dim_f - 1.0) as usize;
        let y1 = (ay.max(by) + r_px).ceil().clamp(0.0, dim_f - 1.0) as usize;
        let abx = bx - ax;
        let aby = by - ay;
        let ab_len2 = (abx * abx + aby * aby).max(1e-5);
        let rad2 = r_px * r_px;
        for py in y0..=y1 {
            let fy = py as f32 + 0.5;
            let row = py * dim;
            for px in x0..=x1 {
                let fx = px as f32 + 0.5;
                let t = (((fx - ax) * abx + (fy - ay) * aby) / ab_len2).clamp(0.0, 1.0);
                let dx = fx - (ax + t * abx);
                let dy = fy - (ay + t * aby);
                let d2 = dx * dx + dy * dy;
                if d2 > rad2 {
                    continue;
                }
                let nz = (1.0 - d2 / rad2).max(0.0).sqrt();
                let z = (az + t * (bz - az) - nz * r_z).clamp(0.0, 1.0);
                if z < zbuf[row + px] {
                    zbuf[row + px] = z;
                }
            }
        }
    }

    zbuf
}

/// Render `scene` from `camera` into an RGBA8 byte buffer (`width * height * 4`) using indexed
/// triangle rasterization + directional PCF 3×3 soft shadows + ray-cast impostor shading +
/// Z-buffer SSAO + ChimeraX silhouettes.
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
    let (clip_near, clip_far) = camera.clip_planes();

    let shadow_str = scene.shadow_strength.clamp(0.0, 1.0);
    let shadow_rad = scene.shadow_radius_around(camera.target, camera.distance * 0.65);
    let light_vp = camera.light_view_proj_matrix(shadow_rad);
    let shadow_zbuf: Option<Vec<f32>> = if shadow_str > 0.01 {
        Some(build_shadow_depth_map(scene, &light_vp))
    } else {
        None
    };

    let sample_shadow = |world_pos: [f32; 3], ndotl: f32| -> f32 {
        let Some(ref sz) = shadow_zbuf else {
            return 1.0;
        };
        let ndc = transform_point_ndc(&light_vp, world_pos);
        let u = ndc[0] * 0.5 + 0.5;
        let v = 0.5 - ndc[1] * 0.5;
        if u <= 0.002 || u >= 0.998 || v <= 0.002 || v >= 0.998 || ndc[2] <= 0.0 || ndc[2] >= 1.0 {
            return 1.0;
        }
        let dim = SHADOW_DIM as i32;
        let bx = (u * (SHADOW_DIM as f32)) as i32;
        let by = (v * (SHADOW_DIM as f32)) as i32;
        let bias = 0.0045 + 0.004 * (1.0 - ndotl.clamp(0.0, 1.0));
        let z_cmp = ndc[2] - bias;
        let mut occ = 0.0_f32;
        for dy in -1..=1 {
            let sy = (by + dy).clamp(0, dim - 1) as usize;
            let row = sy * SHADOW_DIM;
            for dx in -1..=1 {
                let sx = (bx + dx).clamp(0, dim - 1) as usize;
                if z_cmp > sz[row + sx] {
                    occ += 1.0;
                }
            }
        }
        1.0 - (occ / 9.0) * 0.56 * shadow_str
    };

    let project_pt = |p: [f32; 3]| -> Option<(f32, f32, f32, f32)> {
        let rel = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        let z = dot3(rel, fwd);
        if z < clip_near || z > clip_far {
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

    let cam_normal = |wn: [f32; 3]| -> [f32; 3] {
        let cn = [dot3(wn, right), dot3(wn, up), -dot3(wn, fwd)];
        let flip = if cn[2] < 0.0 { -1.0 } else { 1.0 };
        normalize3([cn[0] * flip, cn[1] * flip, (cn[2] * flip).max(0.02)])
    };

    type ProjVert = (f32, f32, f32, [f32; 3], [f32; 3], [f32; 4]);
    let project_meshes =
        |meshes: &[TriangleMesh], tris: &mut Vec<ProjTri>, wire_lines: &mut Vec<ProjLine>| {
            for mesh in meshes {
                let mut pv: Vec<Option<ProjVert>> = Vec::with_capacity(mesh.vertices.len());
                for v in &mesh.vertices {
                    pv.push(project_pt(v.position).map(|(sx, sy, z, _)| {
                        (sx, sy, z, v.position, cam_normal(v.normal), v.color)
                    }));
                }
                for tri in mesh.indices.as_chunks::<3>().0 {
                    let (Some(a), Some(b), Some(c)) = (
                        pv.get(tri[0] as usize).copied().flatten(),
                        pv.get(tri[1] as usize).copied().flatten(),
                        pv.get(tri[2] as usize).copied().flatten(),
                    ) else {
                        continue;
                    };
                    if mesh.wireframe {
                        for &(p, q) in &[(a, b), (b, c), (c, a)] {
                            let min_y = p.1.min(q.1).floor() as i32;
                            let max_y = p.1.max(q.1).ceil() as i32;
                            if max_y >= 0 && min_y < height as i32 {
                                wire_lines.push(ProjLine {
                                    ax: p.0,
                                    ay: p.1,
                                    az: p.2,
                                    bx: q.0,
                                    by: q.1,
                                    bz: q.2,
                                    color: p.5,
                                    min_y,
                                    max_y,
                                });
                            }
                        }
                        continue;
                    }
                    let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
                    if area.abs() < 0.05 {
                        continue;
                    }
                    let min_x = a.0.min(b.0).min(c.0).floor() as i32;
                    let max_x = a.0.max(b.0).max(c.0).ceil() as i32;
                    let min_y = a.1.min(b.1).min(c.1).floor() as i32;
                    let max_y = a.1.max(b.1).max(c.1).ceil() as i32;
                    if max_x < 0 || min_x >= width as i32 || max_y < 0 || min_y >= height as i32 {
                        continue;
                    }
                    tris.push(ProjTri {
                        x: [a.0, b.0, c.0],
                        y: [a.1, b.1, c.1],
                        z: [a.2, b.2, c.2],
                        wp: [a.3, b.3, c.3],
                        n: [a.4, b.4, c.4],
                        col: [a.5, b.5, c.5],
                        min_x: min_x.max(0),
                        max_x: max_x.min(width as i32 - 1),
                        min_y: min_y.max(0),
                        max_y: max_y.min(height as i32 - 1),
                        inv_area: 1.0 / area,
                    });
                }
            }
        };

    let mut opaque_tris = Vec::new();
    let mut transp_tris = Vec::new();
    let mut wire_lines = Vec::new();
    project_meshes(&scene.opaque_meshes, &mut opaque_tris, &mut wire_lines);
    project_meshes(&scene.transparent_meshes, &mut transp_tris, &mut wire_lines);

    let mut op_row_bins: Vec<Vec<u32>> = vec![Vec::new(); h];
    for (ti, t) in opaque_tris.iter().enumerate() {
        for bin in op_row_bins
            .iter_mut()
            .take(t.max_y as usize + 1)
            .skip(t.min_y as usize)
        {
            bin.push(ti as u32);
        }
    }
    let mut tr_row_bins: Vec<Vec<u32>> = vec![Vec::new(); h];
    for (ti, t) in transp_tris.iter().enumerate() {
        for bin in tr_row_bins
            .iter_mut()
            .take(t.max_y as usize + 1)
            .skip(t.min_y as usize)
        {
            bin.push(ti as u32);
        }
    }
    let mut wire_row_bins: Vec<Vec<u32>> = vec![Vec::new(); h];
    for (li, l) in wire_lines.iter().enumerate() {
        let y0 = l.min_y.max(0) as usize;
        let y1 = l.max_y.min(height as i32 - 1) as usize;
        for bin in wire_row_bins.iter_mut().take(y1 + 1).skip(y0) {
            bin.push(li as u32);
        }
    }

    let mut spheres_op = Vec::new();
    let mut spheres_tr = Vec::new();
    for sp in &scene.spheres {
        if let Some((sx, sy, z, ppa)) = project_pt(sp.center) {
            let r_px = (sp.radius * ppa).max(1.0);
            if sx + r_px >= 0.0
                && sx - r_px < width as f32
                && sy + r_px >= 0.0
                && sy - r_px < height as f32
            {
                let ps = ProjSphere {
                    sx,
                    sy,
                    z,
                    r_px,
                    r_a: sp.radius,
                    center_world: sp.center,
                    color: sp.color,
                };
                if sp.color[3] >= 0.95 {
                    spheres_op.push(ps);
                } else {
                    spheres_tr.push(ps);
                }
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
                    a_world: cap.a,
                    b_world: cap.b,
                    r_px,
                    r_a: cap.radius,
                    dashed: cap.dashed > 0.5,
                    color: cap.color,
                    min_y,
                    max_y,
                });
            }
        }
    }

    let light1 = normalize3([0.36, 0.56, 0.74]);
    let light2 = normalize3([-0.58, -0.22, 0.78]);
    let h1 = normalize3([light1[0], light1[1], light1[2] + 1.0]);
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

    let (amb, diff_k, fill_k, spec_k, shininess, rim_k) = match scene.lighting {
        LightingPreset::Soft => (0.28_f32, 0.64_f32, 0.22_f32, 0.24_f32, 28_i32, 0.08_f32),
        LightingPreset::Flat => (0.70, 0.28, 0.08, 0.02, 16, 0.0),
        LightingPreset::Full => (0.22, 0.72, 0.20, 0.46, 42, 0.10),
        LightingPreset::Studio => (0.18, 0.70, 0.26, 0.34, 34, 0.22),
        LightingPreset::PublicationWhite => (0.34, 0.62, 0.22, 0.18, 24, 0.06),
    };

    let z_ref = camera.distance.max(10.0);
    let shade =
        |nx: f32, ny: f32, nz: f32, z: f32, col: [f32; 4], wp: Option<[f32; 3]>| -> [u8; 4] {
            let d1 = (nx * light1[0] + ny * light1[1] + nz * light1[2]).max(0.0);
            let d2 = (nx * light2[0] + ny * light2[1] + nz * light2[2]).max(0.0);
            let sh = wp.map_or(1.0, |p| sample_shadow(p, d1));
            let spec = (nx * h1[0] + ny * h1[1] + nz * h1[2])
                .max(0.0)
                .powi(shininess)
                * spec_k
                * sh;
            let rim = (1.0 - nz).clamp(0.0, 1.0).powi(3) * rim_k;
            let fog =
                (1.0 - ((z - z_ref * 0.80) / (z_ref * 2.5)).clamp(0.0, 0.36)).clamp(0.64, 1.0);
            let lit = (amb + diff_k * d1 * sh + fill_k * d2) * fog;
            let r = ((col[0] * lit + spec + rim) * 255.0).clamp(0.0, 255.0) as u8;
            let g = ((col[1] * lit + spec + rim) * 255.0).clamp(0.0, 255.0) as u8;
            let b = ((col[2] * lit + spec + rim) * 255.0).clamp(0.0, 255.0) as u8;
            let a = (col[3].clamp(0.0, 1.0) * 255.0) as u8;
            [r, g, b, a]
        };

    // Pass 1: Parallel opaque scanline rasterization (with directional PCF 3×3 shadows)
    let opaque_rows: Vec<(Vec<u8>, Vec<f32>)> = (0..h)
        .into_par_iter()
        .map(|py| {
            let fy = py as f32 + 0.5;
            let mut row_rgba = vec![0u8; w * 4];
            let mut row_z = vec![f32::INFINITY; w];
            for px in 0..w {
                row_rgba[px * 4..px * 4 + 4].copy_from_slice(&bg_rgba);
            }

            // 1A. Opaque indexed triangles
            for &ti in &op_row_bins[py] {
                let t = &opaque_tris[ti as usize];
                for px in (t.min_x as usize)..=(t.max_x as usize) {
                    let fx = px as f32 + 0.5;
                    let w0 = ((t.x[1] - fx) * (t.y[2] - fy) - (t.x[2] - fx) * (t.y[1] - fy))
                        * t.inv_area;
                    if w0 < -1e-4 {
                        continue;
                    }
                    let w1 = ((t.x[2] - fx) * (t.y[0] - fy) - (t.x[0] - fx) * (t.y[2] - fy))
                        * t.inv_area;
                    if w1 < -1e-4 {
                        continue;
                    }
                    let w2 = 1.0 - w0 - w1;
                    if w2 < -1e-4 {
                        continue;
                    }
                    let z = w0 * t.z[0] + w1 * t.z[1] + w2 * t.z[2];
                    if z < row_z[px] {
                        let nx = w0 * t.n[0][0] + w1 * t.n[1][0] + w2 * t.n[2][0];
                        let ny = w0 * t.n[0][1] + w1 * t.n[1][1] + w2 * t.n[2][1];
                        let nz = w0 * t.n[0][2] + w1 * t.n[1][2] + w2 * t.n[2][2];
                        let nn = normalize3([nx, ny, nz.max(0.02)]);
                        let col = [
                            w0 * t.col[0][0] + w1 * t.col[1][0] + w2 * t.col[2][0],
                            w0 * t.col[0][1] + w1 * t.col[1][1] + w2 * t.col[2][1],
                            w0 * t.col[0][2] + w1 * t.col[1][2] + w2 * t.col[2][2],
                            1.0,
                        ];
                        let wp = [
                            w0 * t.wp[0][0] + w1 * t.wp[1][0] + w2 * t.wp[2][0],
                            w0 * t.wp[0][1] + w1 * t.wp[1][1] + w2 * t.wp[2][1],
                            w0 * t.wp[0][2] + w1 * t.wp[1][2] + w2 * t.wp[2][2],
                        ];
                        row_z[px] = z;
                        let c = shade(nn[0], nn[1], nn[2], z, col, Some(wp));
                        row_rgba[px * 4..px * 4 + 4].copy_from_slice(&c);
                    }
                }
            }

            // 1B. Opaque spheres
            for sp in &spheres_op {
                let dy = fy - sp.sy;
                if dy.abs() > sp.r_px {
                    continue;
                }
                let span = (sp.r_px * sp.r_px - dy * dy).max(0.0).sqrt();
                let x0 = (sp.sx - span).floor().max(0.0) as usize;
                let x1 = (sp.sx + span).ceil().min((w - 1) as f32) as usize;
                let rad2 = sp.r_px * sp.r_px;
                for px in x0..=x1 {
                    let dx = (px as f32 + 0.5) - sp.sx;
                    let r2 = dx * dx + dy * dy;
                    if r2 > rad2 {
                        continue;
                    }
                    let nz = (1.0 - r2 / rad2).max(0.0).sqrt();
                    let z = sp.z - nz * sp.r_a;
                    if z >= clip_near && z < row_z[px] {
                        let nx = dx / sp.r_px;
                        let ny = -dy / sp.r_px;
                        let wp = [
                            sp.center_world[0] + right[0] * (nx * sp.r_a) + up[0] * (ny * sp.r_a)
                                - fwd[0] * (nz * sp.r_a),
                            sp.center_world[1] + right[1] * (nx * sp.r_a) + up[1] * (ny * sp.r_a)
                                - fwd[1] * (nz * sp.r_a),
                            sp.center_world[2] + right[2] * (nx * sp.r_a) + up[2] * (ny * sp.r_a)
                                - fwd[2] * (nz * sp.r_a),
                        ];
                        row_z[px] = z;
                        let c = shade(nx, ny, nz, z, sp.color, Some(wp));
                        row_rgba[px * 4..px * 4 + 4].copy_from_slice(&c);
                    }
                }
            }

            // 1C. Capsules
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
                    let t = (((fx - cap.ax) * abx + (fy - cap.ay) * aby) / ab_len2).clamp(0.0, 1.0);
                    if cap.dashed && ((t * 9.0).floor() as i32) % 2 == 1 {
                        continue;
                    }
                    let dx = fx - (cap.ax + t * abx);
                    let dy = fy - (cap.ay + t * aby);
                    let d2 = dx * dx + dy * dy;
                    if d2 > r2_lim {
                        continue;
                    }
                    let nz = (1.0 - d2 / r2_lim).max(0.0).sqrt();
                    let z = cap.az + t * (cap.bz - cap.az) - nz * cap.r_a;
                    if z >= clip_near && z < row_z[px] {
                        let nx = dx / cap.r_px;
                        let ny = -dy / cap.r_px;
                        let wp = [
                            cap.a_world[0] + t * (cap.b_world[0] - cap.a_world[0])
                                - fwd[0] * (nz * cap.r_a),
                            cap.a_world[1] + t * (cap.b_world[1] - cap.a_world[1])
                                - fwd[1] * (nz * cap.r_a),
                            cap.a_world[2] + t * (cap.b_world[2] - cap.a_world[2])
                                - fwd[2] * (nz * cap.r_a),
                        ];
                        row_z[px] = z;
                        let c = shade(nx, ny, nz, z, cap.color, Some(wp));
                        row_rgba[px * 4..px * 4 + 4].copy_from_slice(&c);
                    }
                }
            }

            (row_rgba, row_z)
        })
        .collect();

    let mut zbuf = Vec::with_capacity(w * h);
    let mut rgba_base = Vec::with_capacity(w * h * 4);
    for (r, z) in opaque_rows {
        rgba_base.extend_from_slice(&r);
        zbuf.extend_from_slice(&z);
    }

    // Pass 2 + Pass 3: Screen-space 12-tap Poisson SSAO + Depth Silhouettes + Translucent Meshes/Wireframes/Overlays
    let ssao_str = scene.ssao_strength.clamp(0.0, 1.0);
    let do_sil = scene.show_silhouettes;
    let ssao_radius = ((w.min(h) as f32) / 140.0).clamp(2.5, 10.0);
    let r1 = ssao_radius.round() as i32;
    let r2 = (ssao_radius * 1.75).round() as i32;
    let r3 = (ssao_radius * 2.6).round() as i32;
    let offsets: [(i32, i32); 12] = [
        (r1, 0),
        (-r1, 0),
        (0, r1),
        (0, -r1),
        (r2, r1),
        (-r2, r1),
        (r2, -r1),
        (-r2, -r1),
        (r3, r1 / 2),
        (-r3, -r1 / 2),
        (r1 / 2, r3),
        (-r1 / 2, -r3),
    ];
    let z_thresh_sil = (z_ref * 0.045).clamp(1.6, 5.5);
    let z_occ_max = (z_ref * 0.24).clamp(6.0, 20.0);

    let final_rows: Vec<Vec<u8>> = (0..h)
        .into_par_iter()
        .map(|py| {
            let fy = py as f32 + 0.5;
            let row_off = py * w * 4;
            let mut row = rgba_base[row_off..row_off + w * 4].to_vec();
            let z_row = &zbuf[py * w..(py + 1) * w];

            if ssao_str > 0.01 || do_sil {
                for (px, &d0) in z_row.iter().enumerate().take(w) {
                    if !d0.is_finite() {
                        continue;
                    }
                    let mut factor = 1.0_f32;
                    if ssao_str > 0.01 {
                        let mut occ = 0.0_f32;
                        for &(ox, oy) in &offsets {
                            let sx = (px as i32 + ox).clamp(0, w as i32 - 1) as usize;
                            let sy = (py as i32 + oy).clamp(0, h as i32 - 1) as usize;
                            let ds = zbuf[sy * w + sx];
                            if ds.is_finite() {
                                let diff = d0 - ds;
                                if diff > 0.35 && diff < z_occ_max {
                                    occ += 1.0;
                                }
                            }
                        }
                        factor *= 1.0 - (occ / 12.0) * 0.44 * ssao_str;
                    }
                    if do_sil {
                        let mut max_jump = 0.0_f32;
                        for &(nx, ny) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                            let sx = (px as i32 + nx).clamp(0, w as i32 - 1) as usize;
                            let sy = (py as i32 + ny).clamp(0, h as i32 - 1) as usize;
                            let dn = zbuf[sy * w + sx];
                            let jump = if dn.is_finite() { dn - d0 } else { 999.0 };
                            if jump > max_jump {
                                max_jump = jump;
                            }
                        }
                        if max_jump > z_thresh_sil {
                            factor *= 0.44;
                        }
                    }
                    let p4 = px * 4;
                    row[p4] = ((f32::from(row[p4]) * factor).clamp(0.0, 255.0)) as u8;
                    row[p4 + 1] = ((f32::from(row[p4 + 1]) * factor).clamp(0.0, 255.0)) as u8;
                    row[p4 + 2] = ((f32::from(row[p4 + 2]) * factor).clamp(0.0, 255.0)) as u8;
                }
            }

            // 3A. Transparent meshes (smooth SES shell / Binding Pocket cavity blobs / Cryo-EM)
            if !tr_row_bins[py].is_empty() {
                let mut front_z = vec![f32::INFINITY; w];
                let mut front_col = vec![[0u8; 4]; w];
                for &ti in &tr_row_bins[py] {
                    let t = &transp_tris[ti as usize];
                    for px in (t.min_x as usize)..=(t.max_x as usize) {
                        let fx = px as f32 + 0.5;
                        let w0 = ((t.x[1] - fx) * (t.y[2] - fy) - (t.x[2] - fx) * (t.y[1] - fy))
                            * t.inv_area;
                        if w0 < -1e-4 {
                            continue;
                        }
                        let w1 = ((t.x[2] - fx) * (t.y[0] - fy) - (t.x[0] - fx) * (t.y[2] - fy))
                            * t.inv_area;
                        if w1 < -1e-4 {
                            continue;
                        }
                        let w2 = 1.0 - w0 - w1;
                        if w2 < -1e-4 {
                            continue;
                        }
                        let z = w0 * t.z[0] + w1 * t.z[1] + w2 * t.z[2];
                        if z < z_row[px] && z < front_z[px] {
                            let nx = w0 * t.n[0][0] + w1 * t.n[1][0] + w2 * t.n[2][0];
                            let ny = w0 * t.n[0][1] + w1 * t.n[1][1] + w2 * t.n[2][1];
                            let nz = w0 * t.n[0][2] + w1 * t.n[1][2] + w2 * t.n[2][2];
                            let nn = normalize3([nx, ny, nz.max(0.02)]);
                            let col = [
                                w0 * t.col[0][0] + w1 * t.col[1][0] + w2 * t.col[2][0],
                                w0 * t.col[0][1] + w1 * t.col[1][1] + w2 * t.col[2][1],
                                w0 * t.col[0][2] + w1 * t.col[1][2] + w2 * t.col[2][2],
                                w0 * t.col[0][3] + w1 * t.col[1][3] + w2 * t.col[2][3],
                            ];
                            front_z[px] = z;
                            front_col[px] = shade(nn[0], nn[1], nn[2], z, col, None);
                        }
                    }
                }
                for px in 0..w {
                    if front_z[px].is_finite() {
                        let c = front_col[px];
                        let alpha = f32::from(c[3]) / 255.0;
                        let p4 = px * 4;
                        for k in 0..3 {
                            row[p4 + k] = (f32::from(c[k]) * alpha
                                + f32::from(row[p4 + k]) * (1.0 - alpha))
                                as u8;
                        }
                        row[p4 + 3] = row[p4 + 3].max(c[3]);
                    }
                }
            }

            // 3B. Wireframe `isomesh` edges
            for &li in &wire_row_bins[py] {
                let l = &wire_lines[li as usize];
                let dy = l.by - l.ay;
                if dy.abs() < 1e-4 {
                    continue;
                }
                let t = (fy - l.ay) / dy;
                if !(0.0..=1.0).contains(&t) {
                    continue;
                }
                let x = l.ax + t * (l.bx - l.ax);
                let z = l.az + t * (l.bz - l.az);
                let px = x.round() as i32;
                if px >= 0 && (px as usize) < w && z < z_row[px as usize] {
                    let p4 = (px as usize) * 4;
                    let a = l.color[3].clamp(0.1, 1.0);
                    for k in 0..3 {
                        let sc = l.color[k].clamp(0.0, 1.0) * 255.0;
                        row[p4 + k] = (sc * a + f32::from(row[p4 + k]) * (1.0 - a)) as u8;
                    }
                    row[p4 + 3] = 255;
                }
            }

            // 3C. Translucent overlay spheres (HYDE coronas, pharmacophores, ΔMIF hotspots)
            for sp in &spheres_tr {
                let dy = fy - sp.sy;
                if dy.abs() > sp.r_px {
                    continue;
                }
                let span = (sp.r_px * sp.r_px - dy * dy).max(0.0).sqrt();
                let x0 = (sp.sx - span).floor().max(0.0) as usize;
                let x1 = (sp.sx + span).ceil().min((w - 1) as f32) as usize;
                let rad2 = sp.r_px * sp.r_px;
                for (px, &z_bg) in z_row.iter().enumerate().take(x1 + 1).skip(x0) {
                    let dx = (px as f32 + 0.5) - sp.sx;
                    let r2 = dx * dx + dy * dy;
                    if r2 > rad2 {
                        continue;
                    }
                    let nz = (1.0 - r2 / rad2).max(0.0).sqrt();
                    let z = sp.z - nz * sp.r_a;
                    if z >= clip_near && z < z_bg {
                        let nx = dx / sp.r_px;
                        let ny = -dy / sp.r_px;
                        let c = shade(nx, ny, nz, z, sp.color, None);
                        let alpha = (sp.color[3] * (0.55 + 0.45 * (1.0 - nz))).clamp(0.08, 0.92);
                        let p4 = px * 4;
                        for k in 0..3 {
                            row[p4 + k] = (f32::from(c[k]) * alpha
                                + f32::from(row[p4 + k]) * (1.0 - alpha))
                                as u8;
                        }
                        row[p4 + 3] = row[p4 + 3].max(c[3]);
                    }
                }
            }

            row
        })
        .collect();

    let mut out = Vec::with_capacity(w * h * 4);
    for mut r in final_rows {
        out.append(&mut r);
    }

    if !scene.labels_3d.is_empty()
        || !scene.labels_2d.is_empty()
        || scene.color_key.is_some()
        || scene.scale_bar.is_some()
    {
        crate::overlay_2d::composite_2d_overlays(
            &mut out,
            width,
            height,
            camera,
            &scene.labels_3d,
            &scene.labels_2d,
            scene.color_key.as_ref(),
            scene.scale_bar.as_ref(),
        );
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
