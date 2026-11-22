#![forbid(unsafe_code)]

//! 3D Camera (Orbit, Pan, Zoom, Perspective/Orthographic Projection, and Exact Ray Picking).

use craftbio_struct::Biomolecule;
use serde::{Deserialize, Serialize};

/// Interactive biomolecular 3D camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera3D {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y_rad: f32,
    pub orthographic: bool,
    /// Enable ChimeraX Side-View / PyMOL slab near & far clipping planes centered on `target`.
    #[serde(default)]
    pub clip_slab_enabled: bool,
    /// Slab half-depth toward the camera from `target` (Å) when `clip_slab_enabled` is true.
    #[serde(default = "default_clip_near_a")]
    pub clip_near_a: f32,
    /// Slab half-depth behind `target` away from camera (Å) when `clip_slab_enabled` is true.
    #[serde(default = "default_clip_far_a")]
    pub clip_far_a: f32,
}

fn default_clip_near_a() -> f32 {
    35.0
}

fn default_clip_far_a() -> f32 {
    60.0
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            yaw: 0.65,
            pitch: 0.30,
            distance: 55.0,
            fov_y_rad: 42.0_f32.to_radians(),
            orthographic: false,
            clip_slab_enabled: false,
            clip_near_a: default_clip_near_a(),
            clip_far_a: default_clip_far_a(),
        }
    }
}

impl Camera3D {
    #[must_use]
    pub fn eye(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.target[0] + self.distance * cp * self.yaw.cos(),
            self.target[1] + self.distance * self.pitch.sin(),
            self.target[2] + self.distance * cp * self.yaw.sin(),
        ]
    }

    /// Returns orthonormal `(right, up, forward)` camera basis vectors.
    #[must_use]
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let eye = self.eye();
        let fwd = normalize3([
            self.target[0] - eye[0],
            self.target[1] - eye[1],
            self.target[2] - eye[2],
        ]);
        let world_up = [0.0_f32, 1.0, 0.0];
        let right = normalize3(cross3(fwd, world_up));
        let up = normalize3(cross3(right, fwd));
        (right, up, fwd)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.50, 1.50);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up, _) = self.basis();
        let scale = self.distance * 0.0016;
        for k in 0..3 {
            self.target[k] -= right[k] * dx * scale;
            self.target[k] += up[k] * dy * scale;
        }
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let factor = (1.0 - scroll_delta * 0.0015).clamp(0.65, 1.45);
        self.distance = (self.distance * factor).clamp(4.0, 450.0);
    }

    pub fn focus_on_biomolecule(&mut self, mol: &Biomolecule) {
        let (_, _, center, radius) = mol.bounds();
        self.focus_on_bounds(center, radius);
    }

    pub fn focus_on_bounds(&mut self, center: [f64; 3], radius: f64) {
        self.target = [center[0] as f32, center[1] as f32, center[2] as f32];
        self.distance = ((radius as f32) * 2.55).clamp(12.0, 360.0);
    }

    /// Active `(near, far)` view-space clipping planes in Ångströms.
    #[must_use]
    pub fn clip_planes(&self) -> (f32, f32) {
        if self.clip_slab_enabled {
            let near = (self.distance - self.clip_near_a.max(0.5)).max(0.5);
            let far = (self.distance + self.clip_far_a.max(0.5)).max(near + 1.0);
            (near, far)
        } else {
            let near = (self.distance * 0.02).clamp(0.5, 4.0);
            let far = (self.distance * 6.0).max(500.0);
            (near, far)
        }
    }

    /// Column-major 4×4 `View * Projection` matrix for WGSL (`z in [0, 1]`).
    #[must_use]
    pub fn view_proj_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let eye = self.eye();
        let (right, up, fwd) = self.basis();
        let view = [
            [right[0], up[0], -fwd[0], 0.0],
            [right[1], up[1], -fwd[1], 0.0],
            [right[2], up[2], -fwd[2], 0.0],
            [-dot3(right, eye), -dot3(up, eye), dot3(fwd, eye), 1.0],
        ];
        let (near, far) = self.clip_planes();
        let a = aspect.max(0.1);
        let proj = if self.orthographic {
            let half_h = self.distance * (0.5 * self.fov_y_rad).tan();
            let half_w = half_h * a;
            [
                [1.0 / half_w, 0.0, 0.0, 0.0],
                [0.0, 1.0 / half_h, 0.0, 0.0],
                [0.0, 0.0, -1.0 / (far - near), 0.0],
                [0.0, 0.0, -near / (far - near), 1.0],
            ]
        } else {
            let f = 1.0 / (0.5 * self.fov_y_rad).tan();
            [
                [f / a, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, far / (near - far), -1.0],
                [0.0, 0.0, (far * near) / (near - far), 0.0],
            ]
        };
        mat4_mul(proj, view)
    }

    /// Pick the closest atom index along normalized viewport coordinates `nx, ny in [-1, 1]`.
    #[must_use]
    pub fn pick_atom(&self, mol: &Biomolecule, nx: f32, ny: f32, aspect: f32) -> Option<usize> {
        let eye = self.eye();
        let (right, up, fwd) = self.basis();
        let half_h = (0.5 * self.fov_y_rad).tan();
        let half_w = half_h * aspect.max(0.1);
        let dir = normalize3([
            fwd[0] + right[0] * nx * half_w + up[0] * ny * half_h,
            fwd[1] + right[1] * nx * half_w + up[1] * ny * half_h,
            fwd[2] + right[2] * nx * half_w + up[2] * ny * half_h,
        ]);

        let mut best: Option<(usize, f32)> = None;
        for (i, a) in mol.atoms.iter().enumerate() {
            let oc = [
                a.coords[0] as f32 - eye[0],
                a.coords[1] as f32 - eye[1],
                a.coords[2] as f32 - eye[2],
            ];
            let t = dot3(oc, dir);
            if t <= 0.1 {
                continue;
            }
            let d2 = dot3(oc, oc) - t * t;
            let hit_r = 1.35_f32;
            if d2 <= hit_r * hit_r && best.is_none_or(|(_, bt)| t < bt) {
                best = Some((i, t));
            }
        }
        best.map(|(i, _)| i)
    }
}

#[must_use]
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[must_use]
pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[must_use]
pub fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = dot3(v, v).sqrt();
    if l > 1e-8 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}
