#![forbid(unsafe_code)]

//! Zero-Dependency Crisp 2D/3D Publication Overlay Compositor:
//! - Projects 3D world-space `Label3D` annotations cleanly onto both live `egui` GUI & 4K PNGs
//! - Renders `Label2D` screen-space callouts (`2dlabels`)
//! - Renders publication gradient **Color Key** legends (`key`)
//! - Renders calibrated Ångström **Scale Bars** (`scalebar`)
//! - Embeds a crisp 5×7 ASCII + scientific symbol bitmap font (zero external font/image crates).

use craftbio_scene::{ColorKeyLegend, Label2D, Label3D, ScaleBarSpec};

use crate::camera::{Camera3D, dot3};

/// Project a 3D world coordinate `[x, y, z]` into screen pixel space `(sx, sy, depth_z)`.
/// Returns `None` if clipped outside active `(near, far)` planes.
#[must_use]
pub fn project_world_to_screen(
    camera: &Camera3D,
    world: [f32; 3],
    width: u32,
    height: u32,
) -> Option<(f32, f32, f32)> {
    let aspect = (width.max(1) as f32) / (height.max(1) as f32);
    let eye = camera.eye();
    let (right, up, fwd) = camera.basis();
    let half_h = (0.5 * camera.fov_y_rad).tan();
    let (clip_near, clip_far) = camera.clip_planes();
    let rel = [world[0] - eye[0], world[1] - eye[1], world[2] - eye[2]];
    let z = dot3(rel, fwd);
    if z < clip_near || z > clip_far {
        return None;
    }
    let scale = if camera.orthographic {
        camera.distance * half_h
    } else {
        z * half_h
    };
    if scale.abs() < 1e-6 {
        return None;
    }
    let sx = ((dot3(rel, right) / (scale * aspect)) * 0.5 + 0.5) * (width as f32);
    let sy = (0.5 - (dot3(rel, up) / scale) * 0.5) * (height as f32);
    Some((sx, sy, z))
}

/// Calibrated pixels-per-Ångström scale factor at the camera's focal target depth.
#[must_use]
pub fn pixels_per_angstrom_at_target(camera: &Camera3D, height: u32) -> f32 {
    let half_h = (0.5 * camera.fov_y_rad).tan();
    let scale = (camera.distance * half_h).max(1e-4);
    (height as f32) / (2.0 * scale)
}

/// Crisp 5×7 monochrome bitmap glyph (`bit 4` = left column .. `bit 0` = right column).
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        'A' | 'a' | 'Å' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' | 'b' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' | 'c' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' | 'd' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' | 'e' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' | 'f' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' | 'g' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
        ],
        'H' | 'h' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' | 'i' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' | 'j' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' | 'k' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' | 'l' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' | 'm' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' | 'n' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' | 'o' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' | 'p' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' | 'q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' | 'r' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' | 's' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' | 't' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' | 'u' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' | 'v' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' | 'w' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' | 'x' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' | 'y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' | 'z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
        ],
        ',' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000,
        ],
        ':' => [
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ],
        '-' | '–' | '—' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        '/' => [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        '[' => [
            0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110,
        ],
        ']' => [
            0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110,
        ],
        '#' => [
            0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000,
        ],
        '%' => [
            0b11001, 0b11010, 0b00100, 0b00100, 0b01011, 0b10011, 0b00000,
        ],
        '=' => [
            0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        '°' => [
            0b00110, 0b01001, 0b00110, 0b00000, 0b00000, 0b00000, 0b00000,
        ],
        'Δ' => [
            0b00100, 0b01010, 0b01010, 0b10001, 0b10001, 0b11111, 0b00000,
        ],
        '→' => [
            0b00000, 0b00100, 0b00010, 0b11111, 0b00010, 0b00100, 0b00000,
        ],
        _ => [0b00000; 7],
    }
}

struct Canvas<'a> {
    rgba: &'a mut [u8],
    w: usize,
    h: usize,
}

impl Canvas<'_> {
    fn blend_px(&mut self, x: i32, y: i32, col: [u8; 4]) {
        if x < 0 || y < 0 || (x as usize) >= self.w || (y as usize) >= self.h || col[3] == 0 {
            return;
        }
        let p4 = ((y as usize) * self.w + (x as usize)) * 4;
        let a = f32::from(col[3]) / 255.0;
        for k in 0..3 {
            self.rgba[p4 + k] =
                (f32::from(col[k]) * a + f32::from(self.rgba[p4 + k]) * (1.0 - a)) as u8;
        }
        self.rgba[p4 + 3] = self.rgba[p4 + 3].max(col[3]);
    }

    fn fill_rect(&mut self, x0: i32, y0: i32, w: i32, h: i32, col: [u8; 4]) {
        let xmin = x0.max(0);
        let xmax = (x0 + w).min(self.w as i32);
        let ymin = y0.max(0);
        let ymax = (y0 + h).min(self.h as i32);
        for y in ymin..ymax {
            for x in xmin..xmax {
                self.blend_px(x, y, col);
            }
        }
    }

    fn draw_text(&mut self, x0: i32, y0: i32, text: &str, scale: i32, col: [u8; 4]) {
        let s = scale.clamp(1, 5);
        let mut cx = x0;
        for ch in text.chars() {
            let g = glyph_5x7(ch);
            for (ry, &row_bits) in g.iter().enumerate() {
                for rx in 0..5 {
                    if (row_bits >> (4 - rx)) & 1 == 1 {
                        for dy in 0..s {
                            for dx in 0..s {
                                self.blend_px(cx + rx * s + dx, y0 + (ry as i32) * s + dy, col);
                            }
                        }
                    }
                }
            }
            cx += 6 * s;
        }
    }

    fn draw_pill_label(
        &mut self,
        x: i32,
        y: i32,
        text: &str,
        scale: i32,
        fg: [u8; 4],
        bg: [u8; 4],
    ) {
        let s = scale.clamp(1, 4);
        let n_chars = text.chars().count() as i32;
        let tw = (n_chars * 6 - 1).max(0) * s;
        let th = 7 * s;
        let pad_x = 5 * s;
        let pad_y = 3 * s;
        let bx = (x - pad_x).clamp(4, (self.w as i32 - tw - 2 * pad_x - 4).max(4));
        let by = (y - pad_y).clamp(4, (self.h as i32 - th - 2 * pad_y - 4).max(4));
        self.fill_rect(bx, by, tw + 2 * pad_x, th + 2 * pad_y, bg);
        let border = [fg[0], fg[1], fg[2], 140];
        self.fill_rect(bx, by, tw + 2 * pad_x, 1, border);
        self.fill_rect(bx, by + th + 2 * pad_y - 1, tw + 2 * pad_x, 1, border);
        self.draw_text(bx + pad_x, by + pad_y, text, s, fg);
    }
}

fn f32_to_u8_rgba(c: [f32; 4]) -> [u8; 4] {
    [
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
        (c[3].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

/// Composite 3D labels, 2D labels, Color Key legend, and calibrated Scale Bar crisply into RGBA8.
#[allow(clippy::too_many_arguments)]
pub fn composite_2d_overlays(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    camera: &Camera3D,
    labels_3d: &[Label3D],
    labels_2d: &[Label2D],
    color_key: Option<&ColorKeyLegend>,
    scale_bar: Option<&ScaleBarSpec>,
) {
    let w = width as usize;
    let h = height as usize;
    if rgba.len() < w * h * 4 || w < 32 || h < 32 {
        return;
    }
    let ui_scale = ((h as f32) / 520.0).round().clamp(1.0, 3.0) as i32;
    let mut cv = Canvas { rgba, w, h };

    // 1. 3D world-space projected labels
    for lbl in labels_3d {
        let p = [
            lbl.coords[0] as f32,
            lbl.coords[1] as f32,
            lbl.coords[2] as f32,
        ];
        if let Some((sx, sy, _)) = project_world_to_screen(camera, p, width, height) {
            let sc = ((ui_scale as f32) * lbl.font_scale.clamp(0.6, 2.5))
                .round()
                .clamp(1.0, 4.0) as i32;
            let fg = f32_to_u8_rgba(lbl.rgba);
            cv.draw_pill_label(
                sx.round() as i32 + 6,
                sy.round() as i32 - 10 * sc,
                &lbl.text,
                sc,
                fg,
                [14, 18, 28, 212],
            );
        }
    }

    // 2. 2D screen-space labels (`2dlabels`)
    for lbl in labels_2d {
        let x = (lbl.norm_pos[0].clamp(0.02, 0.94) * (width as f32)).round() as i32;
        let y = (lbl.norm_pos[1].clamp(0.02, 0.94) * (height as f32)).round() as i32;
        let sc = ((ui_scale as f32) * lbl.font_scale.clamp(0.7, 3.0))
            .round()
            .clamp(1.0, 4.0) as i32;
        let fg = f32_to_u8_rgba(lbl.rgba);
        cv.draw_pill_label(x, y, &lbl.text, sc, fg, [14, 18, 28, 215]);
    }

    // 3. Publication Color Key Legend (bottom-left)
    if let Some(key) = color_key
        && key.stops.len() >= 2
    {
        let bar_w = ((w as f32) * 0.24).round().clamp(140.0, 340.0) as i32;
        let bar_h = 12 * ui_scale;
        let pad = 10 * ui_scale;
        let panel_w = bar_w + 2 * pad;
        let panel_h = bar_h + 24 * ui_scale + 2 * pad;
        let x0 = 18 * ui_scale;
        let y0 = (h as i32) - panel_h - 16 * ui_scale;

        cv.fill_rect(x0, y0, panel_w, panel_h, [14, 18, 28, 216]);
        cv.draw_text(
            x0 + pad,
            y0 + pad,
            &key.title,
            ui_scale,
            [240, 245, 255, 255],
        );

        let gx0 = x0 + pad;
        let gy0 = y0 + pad + 11 * ui_scale;
        let segs = (key.stops.len() - 1) as f32;
        for dx in 0..bar_w {
            let u = (dx as f32) / ((bar_w - 1).max(1) as f32);
            let fidx = u * segs;
            let idx = (fidx.floor() as usize).min(key.stops.len() - 2);
            let frac = fidx - idx as f32;
            let ca = key.stops[idx];
            let cb = key.stops[idx + 1];
            let col = [
                ((ca[0] + (cb[0] - ca[0]) * frac).clamp(0.0, 1.0) * 255.0) as u8,
                ((ca[1] + (cb[1] - ca[1]) * frac).clamp(0.0, 1.0) * 255.0) as u8,
                ((ca[2] + (cb[2] - ca[2]) * frac).clamp(0.0, 1.0) * 255.0) as u8,
                255,
            ];
            for dy in 0..bar_h {
                cv.blend_px(gx0 + dx, gy0 + dy, col);
            }
        }
        cv.fill_rect(gx0, gy0, bar_w, 1, [220, 230, 245, 220]);
        cv.fill_rect(gx0, gy0 + bar_h - 1, bar_w, 1, [220, 230, 245, 220]);

        let ly = gy0 + bar_h + 4 * ui_scale;
        cv.draw_text(gx0, ly, &key.min_label, ui_scale, [215, 225, 240, 255]);
        let max_w = (key.max_label.chars().count() as i32) * 6 * ui_scale;
        cv.draw_text(
            (gx0 + bar_w - max_w).max(gx0),
            ly,
            &key.max_label,
            ui_scale,
            [215, 225, 240, 255],
        );
    }

    // 4. Calibrated Ångström Scale Bar (bottom-right)
    if let Some(sb) = scale_bar {
        let ppa = pixels_per_angstrom_at_target(camera, height);
        let bar_px = (sb.length_a.max(1.0) * ppa)
            .round()
            .clamp(42.0, (width as f32) * 0.34) as i32;
        let pad = 9 * ui_scale;
        let lbl_w = (sb.label.chars().count() as i32) * 6 * ui_scale;
        let box_w = bar_px.max(lbl_w) + 2 * pad;
        let box_h = 20 * ui_scale + 2 * pad;
        let x0 = (w as i32) - box_w - 18 * ui_scale;
        let y0 = (h as i32) - box_h - 16 * ui_scale;

        cv.fill_rect(x0, y0, box_w, box_h, [14, 18, 28, 214]);
        let tx = x0 + (box_w - lbl_w) / 2;
        cv.draw_text(tx, y0 + pad, &sb.label, ui_scale, [245, 248, 255, 255]);
        let bx = x0 + (box_w - bar_px) / 2;
        let by = y0 + pad + 12 * ui_scale;
        let thick = (3 * ui_scale).max(2);
        cv.fill_rect(bx, by, bar_px, thick, [245, 248, 255, 255]);
        cv.fill_rect(
            bx,
            by - 3 * ui_scale,
            ui_scale.max(2),
            thick + 6 * ui_scale,
            [245, 248, 255, 255],
        );
        cv.fill_rect(
            bx + bar_px - ui_scale.max(2),
            by - 3 * ui_scale,
            ui_scale.max(2),
            thick + 6 * ui_scale,
            [245, 248, 255, 255],
        );
    }
}
