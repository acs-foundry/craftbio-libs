#![forbid(unsafe_code)]

//! Pure-Rust Biomolecular Movie Maker & Camera Timeline:
//! - Zero-dependency **Animated GIF89a LZW Encoder** (256-color RGB cube + grayscale palette,
//!   `NETSCAPE2.0` infinite loop, variable-width 9–12 bit LZW compression)
//! - Optional system **`ffmpeg`** pipe encoder for hardware-compatible H.264 **`.mp4`** / VP9 **`.webm`**
//! - Smoothstep Camera Interpolation (`fly` across saved Scenes `F1 -> F2 -> F3`),
//!   360° Turntable (`turn`), and Sinusoidal Presentation Rock (`rock`).

use std::f32::consts::TAU;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::camera::Camera3D;
use crate::geometry::RenderSceneData;
use crate::headless_png::{RenderError, render_scene_rgba};

/// Build a fixed 256-color Global Color Table:
/// `0..216` = 6×6×6 uniform RGB color cube, `216..256` = 40-step neutral grayscale ramp.
fn build_palette_256() -> [u8; 768] {
    let mut pal = [0u8; 768];
    let mut idx = 0usize;
    for r in 0..6u8 {
        for g in 0..6u8 {
            for b in 0..6u8 {
                pal[idx * 3] = r * 51;
                pal[idx * 3 + 1] = g * 51;
                pal[idx * 3 + 2] = b * 51;
                idx += 1;
            }
        }
    }
    for i in 0..40usize {
        let v = ((i * 255) / 39) as u8;
        pal[idx * 3] = v;
        pal[idx * 3 + 1] = v;
        pal[idx * 3 + 2] = v;
        idx += 1;
    }
    pal
}

#[inline]
fn quantize_rgb(r: u8, g: u8, b: u8) -> u8 {
    let ri = ((u16::from(r) * 5 + 127) / 255) as u8;
    let gi = ((u16::from(g) * 5 + 127) / 255) as u8;
    let bi = ((u16::from(b) * 5 + 127) / 255) as u8;
    let cr = ri * 51;
    let cg = gi * 51;
    let cb = bi * 51;
    let err_cube = (i32::from(r) - i32::from(cr)).pow(2)
        + (i32::from(g) - i32::from(cg)).pow(2)
        + (i32::from(b) - i32::from(cb)).pow(2);

    let lum = (u32::from(r) * 77 + u32::from(g) * 150 + u32::from(b) * 29) >> 8;
    let gray_i = ((lum * 39 + 127) / 255).min(39) as u8;
    let gv = ((u16::from(gray_i) * 255) / 39) as u8;
    let err_gray = (i32::from(r) - i32::from(gv)).pow(2)
        + (i32::from(g) - i32::from(gv)).pow(2)
        + (i32::from(b) - i32::from(gv)).pow(2);

    if err_gray < err_cube {
        216 + gray_i
    } else {
        ri * 36 + gi * 6 + bi
    }
}

/// GIF LSB-first variable-bit-width LZW encoder (`min_code_size = 8`).
fn lzw_encode_gif(indices: &[u8]) -> Vec<u8> {
    const MIN_CODE_SIZE: u8 = 8;
    const CLEAR: u16 = 256;
    const EOI: u16 = 257;
    const MAX_TABLE: usize = 4096;

    let mut bit_buf: u32 = 0;
    let mut bit_cnt: u32 = 0;
    let mut raw_bytes = Vec::with_capacity(indices.len() / 2 + 64);

    let emit = |code: u16, code_size: u32, buf: &mut u32, cnt: &mut u32, out: &mut Vec<u8>| {
        *buf |= u32::from(code) << *cnt;
        *cnt += code_size;
        while *cnt >= 8 {
            out.push((*buf & 0xFF) as u8);
            *buf >>= 8;
            *cnt -= 8;
        }
    };

    // Flat trie table: `next_code[prefix * 256 + byte]` would be 1M entries; use compact hash map
    let mut dict: std::collections::HashMap<u32, u16> =
        std::collections::HashMap::with_capacity(4096);
    let mut next_code: u16 = 258;
    let mut code_size: u32 = 9;

    emit(CLEAR, code_size, &mut bit_buf, &mut bit_cnt, &mut raw_bytes);

    if let Some(&first) = indices.first() {
        let mut prefix = u16::from(first);
        for &byte in &indices[1..] {
            let key = (u32::from(prefix) << 8) | u32::from(byte);
            if let Some(&found) = dict.get(&key) {
                prefix = found;
            } else {
                emit(
                    prefix,
                    code_size,
                    &mut bit_buf,
                    &mut bit_cnt,
                    &mut raw_bytes,
                );
                if (next_code as usize) < MAX_TABLE {
                    dict.insert(key, next_code);
                    if next_code == (1u16 << code_size) && code_size < 12 {
                        code_size += 1;
                    }
                    next_code += 1;
                } else {
                    emit(CLEAR, code_size, &mut bit_buf, &mut bit_cnt, &mut raw_bytes);
                    dict.clear();
                    next_code = 258;
                    code_size = 9;
                }
                prefix = u16::from(byte);
            }
        }
        emit(
            prefix,
            code_size,
            &mut bit_buf,
            &mut bit_cnt,
            &mut raw_bytes,
        );
    }
    emit(EOI, code_size, &mut bit_buf, &mut bit_cnt, &mut raw_bytes);
    if bit_cnt > 0 {
        raw_bytes.push((bit_buf & 0xFF) as u8);
    }

    // Wrap into GIF sub-blocks (max 255 bytes each) preceded by `MIN_CODE_SIZE`
    let mut out = Vec::with_capacity(raw_bytes.len() + raw_bytes.len() / 255 + 8);
    out.push(MIN_CODE_SIZE);
    for chunk in raw_bytes.chunks(255) {
        out.push(chunk.len() as u8);
        out.extend_from_slice(chunk);
    }
    out.push(0x00); // Block terminator
    out
}

/// Encode RGBA8 frames into a valid infinite-loop **GIF89a** animated byte buffer (100% pure Rust).
#[must_use]
pub fn encode_gif89a(width: u16, height: u16, delay_cs: u16, frames_rgba: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        frames_rgba.len() * (usize::from(width) * usize::from(height) / 4) + 1024,
    );
    let pal = build_palette_256();

    // Header + Logical Screen Descriptor (Global Color Table, 256 colors)
    out.extend_from_slice(b"GIF89a");
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.push(0b1111_0111); // GCT flag=1, color res=7, sort=0, size=7 (2^(7+1)=256)
    out.push(0); // Background index
    out.push(0); // Pixel aspect ratio
    out.extend_from_slice(&pal);

    // NETSCAPE2.0 Infinite Loop Application Extension
    out.extend_from_slice(&[
        0x21, 0xFF, 0x0B, b'N', b'E', b'T', b'S', b'C', b'A', b'P', b'E', b'2', b'.', b'0', 0x03,
        0x01, 0x00, 0x00, 0x00,
    ]);

    let n_px = usize::from(width) * usize::from(height);
    let mut indices = vec![0u8; n_px];

    for frame in frames_rgba {
        if frame.len() < n_px * 4 {
            continue;
        }
        for (i, dst) in indices.iter_mut().enumerate() {
            let p4 = i * 4;
            *dst = quantize_rgb(frame[p4], frame[p4 + 1], frame[p4 + 2]);
        }

        // Graphic Control Extension
        out.extend_from_slice(&[0x21, 0xF9, 0x04, 0x04]);
        out.extend_from_slice(&delay_cs.max(2).to_le_bytes());
        out.extend_from_slice(&[0x00, 0x00]);

        // Image Descriptor
        out.push(0x2C);
        out.extend_from_slice(&0u16.to_le_bytes()); // left
        out.extend_from_slice(&0u16.to_le_bytes()); // top
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.push(0x00); // Use GCT, not interlaced

        let compressed = lzw_encode_gif(&indices);
        out.extend_from_slice(&compressed);
    }

    out.push(0x3B); // GIF Trailer
    out
}

/// Write an animated biomolecular movie to `.gif` (pure-Rust GIF89a) or `.mp4`/`.webm` (via system `ffmpeg`).
pub fn write_movie_file(
    out_path: &Path,
    width: u32,
    height: u32,
    fps: u16,
    frames_rgba: &[Vec<u8>],
) -> Result<usize, RenderError> {
    if width < 16 || height < 16 || width > 4096 || height > 4096 || frames_rgba.is_empty() {
        return Err(RenderError::InvalidDimensions(width, height));
    }
    if let Some(p) = out_path.parent()
        && !p.as_os_str().is_empty()
    {
        std::fs::create_dir_all(p)?;
    }
    let ext = out_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("gif")
        .to_ascii_lowercase();

    if ext == "gif" {
        let delay_cs = (100u16 / fps.clamp(2, 50)).max(2);
        let bytes = encode_gif89a(width as u16, height as u16, delay_cs, frames_rgba);
        std::fs::write(out_path, &bytes)?;
        return Ok(bytes.len());
    }

    // H.264 MP4 or VP9 WebM via system `ffmpeg` (requires even dimensions)
    let ew = width & !1;
    let eh = height & !1;
    let codec_args: &[&str] = if ext == "webm" {
        &["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "2M"]
    } else {
        &[
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    };

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        &format!("{ew}x{eh}"),
        "-r",
        &fps.clamp(4, 60).to_string(),
        "-i",
        "-",
    ])
    .args(codec_args)
    .arg(out_path)
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        RenderError::Io(std::io::Error::other(format!(
            "failed to spawn ffmpeg for .{ext} export (use .gif for pure-Rust export): {e}"
        )))
    })?;

    if let Some(mut stdin) = child.stdin.take() {
        let row_bytes = (ew as usize) * 4;
        let stride_bytes = (width as usize) * 4;
        for frame in frames_rgba {
            if ew == width && eh == height {
                stdin.write_all(frame)?;
            } else {
                for y in 0..(eh as usize) {
                    let off = y * stride_bytes;
                    stdin.write_all(&frame[off..off + row_bytes])?;
                }
            }
        }
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RenderError::Io(std::io::Error::other(format!(
            "ffmpeg exited with {}: {}",
            output.status,
            stderr.trim()
        ))));
    }
    let meta = std::fs::metadata(out_path)?;
    Ok(meta.len() as usize)
}

/// Smoothstep spherical interpolation between two camera keyframes (`fly` across Scenes `F1 -> F2`).
#[must_use]
pub fn interpolate_camera(a: &Camera3D, b: &Camera3D, t: f32) -> Camera3D {
    let t = t.clamp(0.0, 1.0);
    let u = t * t * (3.0 - 2.0 * t);
    // Shortest angular delta for yaw
    let mut dyaw = (b.yaw - a.yaw) % TAU;
    if dyaw > std::f32::consts::PI {
        dyaw -= TAU;
    } else if dyaw < -std::f32::consts::PI {
        dyaw += TAU;
    }
    Camera3D {
        target: [
            a.target[0] + (b.target[0] - a.target[0]) * u,
            a.target[1] + (b.target[1] - a.target[1]) * u,
            a.target[2] + (b.target[2] - a.target[2]) * u,
        ],
        yaw: a.yaw + dyaw * u,
        pitch: a.pitch + (b.pitch - a.pitch) * u,
        distance: a.distance + (b.distance - a.distance) * u,
        fov_y_rad: a.fov_y_rad + (b.fov_y_rad - a.fov_y_rad) * u,
        orthographic: if u < 0.5 {
            a.orthographic
        } else {
            b.orthographic
        },
        clip_slab_enabled: a.clip_slab_enabled || b.clip_slab_enabled,
        clip_near_a: a.clip_near_a + (b.clip_near_a - a.clip_near_a) * u,
        clip_far_a: a.clip_far_a + (b.clip_far_a - a.clip_far_a) * u,
    }
}

/// Render a smooth 360° turntable orbit sequence (ChimeraX `turn y 360`).
pub fn render_turntable_frames(
    scene: &RenderSceneData,
    base_cam: &Camera3D,
    width: u32,
    height: u32,
    n_frames: usize,
) -> Result<Vec<Vec<u8>>, RenderError> {
    let n = n_frames.clamp(4, 180);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let mut cam = base_cam.clone();
        cam.yaw += TAU * (i as f32) / (n as f32);
        frames.push(render_scene_rgba(scene, &cam, width, height, false)?);
    }
    Ok(frames)
}

/// Render a smooth sinusoidal presentation rock sequence (ChimeraX `rock`).
pub fn render_rock_frames(
    scene: &RenderSceneData,
    base_cam: &Camera3D,
    width: u32,
    height: u32,
    n_frames: usize,
    amplitude_deg: f32,
) -> Result<Vec<Vec<u8>>, RenderError> {
    let n = n_frames.clamp(4, 180);
    let amp = amplitude_deg.clamp(5.0, 90.0).to_radians();
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let mut cam = base_cam.clone();
        let phase = TAU * (i as f32) / (n as f32);
        cam.yaw += amp * phase.sin();
        cam.pitch = (cam.pitch + 0.18 * amp * (2.0 * phase).sin()).clamp(-1.45, 1.45);
        frames.push(render_scene_rgba(scene, &cam, width, height, false)?);
    }
    Ok(frames)
}

/// Render a smooth multi-keyframe camera fly-through across saved Scenes (`fly F1 -> F2 -> ...`).
pub fn render_fly_frames(
    scene: &RenderSceneData,
    keyframes: &[Camera3D],
    width: u32,
    height: u32,
    frames_per_segment: usize,
) -> Result<Vec<Vec<u8>>, RenderError> {
    if keyframes.len() < 2 {
        return Err(RenderError::InvalidDimensions(width, height));
    }
    let seg_n = frames_per_segment.clamp(4, 90);
    let mut frames = Vec::with_capacity((keyframes.len() - 1) * seg_n + 1);
    for w in keyframes.windows(2) {
        for s in 0..seg_n {
            let t = (s as f32) / (seg_n as f32);
            let cam = interpolate_camera(&w[0], &w[1], t);
            frames.push(render_scene_rgba(scene, &cam, width, height, false)?);
        }
    }
    if let Some(last) = keyframes.last() {
        frames.push(render_scene_rgba(scene, last, width, height, false)?);
    }
    Ok(frames)
}
