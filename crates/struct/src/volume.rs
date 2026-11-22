#![forbid(unsafe_code)]

//! Pure-Rust Cryo-EM Volumetric Density Maps (MRC2014 / CCP4 `.mrc`/`.map`),
//! APBS OpenDX (`.dx`) Electrostatic Grids, ChimeraX-style `molmap` Synthesis,
//! Rigid `fitmap` Cross-Correlation, and Volumetric Isosurface Contour Extraction.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::{BioStructError, Biomolecule, ResidueClass};

/// Maximum voxel count allowed in a single volume to prevent OOM DoS (`256^3`).
pub const MAX_VOLUME_VOXELS: usize = 16_777_216;

/// 3D regular scalar density / electrostatic potential volume grid (`values[z * nx * ny + y * nx + x]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeGrid3D {
    pub label: String,
    pub dims: [usize; 3],
    pub origin: [f64; 3],
    pub voxel_size: [f64; 3],
    pub values: Vec<f32>,
    pub mean: f64,
    pub std_dev: f64,
    pub min_val: f64,
    pub max_val: f64,
}

/// Result of evaluating or optimizing a biomolecular model inside a 3D Cryo-EM density map (`fitmap`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapFitReport {
    pub map_label: String,
    pub atoms_evaluated: usize,
    pub correlation_before: f64,
    pub correlation_after: f64,
    pub translation_applied_a: [f64; 3],
    pub mean_density_sigma: f64,
    pub atoms_inside_1sigma_pct: f64,
}

impl VolumeGrid3D {
    /// Construct a volume grid and compute exact summary statistics (`mean`, `std_dev`, `min`, `max`).
    pub fn from_raw(
        label: impl Into<String>,
        dims: [usize; 3],
        origin: [f64; 3],
        voxel_size: [f64; 3],
        values: Vec<f32>,
    ) -> Result<Self, BioStructError> {
        let total = dims[0]
            .checked_mul(dims[1])
            .and_then(|v| v.checked_mul(dims[2]))
            .ok_or_else(|| BioStructError::Parse("volume dimensions overflow".to_string()))?;
        if total == 0 || total > MAX_VOLUME_VOXELS || values.len() != total {
            return Err(BioStructError::Parse(format!(
                "invalid volume grid size {:?} (len={}, cap={MAX_VOLUME_VOXELS})",
                dims,
                values.len()
            )));
        }
        let mut sum = 0.0_f64;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &v in &values {
            let fv = f64::from(v);
            if fv.is_finite() {
                sum += fv;
                min_v = min_v.min(fv);
                max_v = max_v.max(fv);
            }
        }
        let mean = sum / (total as f64);
        let var: f64 = values
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / (total as f64);
        let std_dev = var.sqrt().max(1e-9);
        Ok(Self {
            label: label.into(),
            dims,
            origin,
            voxel_size,
            values,
            mean,
            std_dev,
            min_val: min_v,
            max_val: max_v,
        })
    }

    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.values[z * self.dims[0] * self.dims[1] + y * self.dims[0] + x]
    }

    /// Trilinearly sample volume density at arbitrary Cartesian 3D point `p` (Å).
    #[must_use]
    pub fn sample_trilinear(&self, p: [f64; 3]) -> f64 {
        let fx = (p[0] - self.origin[0]) / self.voxel_size[0].max(1e-6);
        let fy = (p[1] - self.origin[1]) / self.voxel_size[1].max(1e-6);
        let fz = (p[2] - self.origin[2]) / self.voxel_size[2].max(1e-6);
        if fx < 0.0
            || fy < 0.0
            || fz < 0.0
            || fx >= (self.dims[0] - 1) as f64
            || fy >= (self.dims[1] - 1) as f64
            || fz >= (self.dims[2] - 1) as f64
        {
            return 0.0;
        }
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let z0 = fz.floor() as usize;
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let tz = fz - z0 as f64;
        let c000 = f64::from(self.get(x0, y0, z0));
        let c100 = f64::from(self.get(x0 + 1, y0, z0));
        let c010 = f64::from(self.get(x0, y0 + 1, z0));
        let c110 = f64::from(self.get(x0 + 1, y0 + 1, z0));
        let c001 = f64::from(self.get(x0, y0, z0 + 1));
        let c101 = f64::from(self.get(x0 + 1, y0, z0 + 1));
        let c011 = f64::from(self.get(x0, y0 + 1, z0 + 1));
        let c111 = f64::from(self.get(x0 + 1, y0 + 1, z0 + 1));
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        c0 * (1.0 - tz) + c1 * tz
    }

    /// World coordinates (Å) of voxel center `(x, y, z)`.
    #[must_use]
    pub fn voxel_world(&self, x: usize, y: usize, z: usize) -> [f64; 3] {
        [
            self.origin[0] + (x as f64) * self.voxel_size[0],
            self.origin[1] + (y as f64) * self.voxel_size[1],
            self.origin[2] + (z as f64) * self.voxel_size[2],
        ]
    }

    /// 3D separable Gaussian smoothing filter (ChimeraX `vop gaussian`).
    pub fn volume_gaussian(&self, sigma_a: f64) -> Result<Self, BioStructError> {
        let [nx, ny, nz] = self.dims;
        let sig = sigma_a.clamp(0.4, 8.0);
        let kernel_1d = |vox: f64| -> Vec<f32> {
            let s_vox = (sig / vox.max(0.1)).clamp(0.4, 5.0);
            let rad = (s_vox * 2.5).ceil() as usize;
            let mut k = vec![0.0_f32; 2 * rad + 1];
            let mut sum = 0.0_f32;
            for (i, slot) in k.iter_mut().enumerate() {
                let d = (i as f64) - (rad as f64);
                let w = (-0.5 * (d / s_vox).powi(2)).exp() as f32;
                *slot = w;
                sum += w;
            }
            for w in &mut k {
                *w /= sum.max(1e-6);
            }
            k
        };
        let kx = kernel_1d(self.voxel_size[0]);
        let ky = kernel_1d(self.voxel_size[1]);
        let kz = kernel_1d(self.voxel_size[2]);
        let rx = kx.len() / 2;
        let ry = ky.len() / 2;
        let rz = kz.len() / 2;

        // Pass X
        let mut tmp_x = vec![0.0_f32; self.values.len()];
        for z in 0..nz {
            for y in 0..ny {
                let base = z * nx * ny + y * nx;
                for x in 0..nx {
                    let mut acc = 0.0_f32;
                    for (ki, &w) in kx.iter().enumerate() {
                        let sx = (x as isize + ki as isize - rx as isize).clamp(0, nx as isize - 1)
                            as usize;
                        acc += self.values[base + sx] * w;
                    }
                    tmp_x[base + x] = acc;
                }
            }
        }
        // Pass Y
        let mut tmp_y = vec![0.0_f32; self.values.len()];
        for z in 0..nz {
            let zoff = z * nx * ny;
            for y in 0..ny {
                for x in 0..nx {
                    let mut acc = 0.0_f32;
                    for (ki, &w) in ky.iter().enumerate() {
                        let sy = (y as isize + ki as isize - ry as isize).clamp(0, ny as isize - 1)
                            as usize;
                        acc += tmp_x[zoff + sy * nx + x] * w;
                    }
                    tmp_y[zoff + y * nx + x] = acc;
                }
            }
        }
        // Pass Z
        let mut out = vec![0.0_f32; self.values.len()];
        for z in 0..nz {
            for y in 0..ny {
                let yoff = y * nx;
                for x in 0..nx {
                    let mut acc = 0.0_f32;
                    for (ki, &w) in kz.iter().enumerate() {
                        let sz = (z as isize + ki as isize - rz as isize).clamp(0, nz as isize - 1)
                            as usize;
                        acc += tmp_y[sz * nx * ny + yoff + x] * w;
                    }
                    out[z * nx * ny + yoff + x] = acc;
                }
            }
        }
        Self::from_raw(
            format!("{}_gauss({sig:.1}A)", self.label),
            self.dims,
            self.origin,
            self.voxel_size,
            out,
        )
    }

    /// Unsharp-mask / B-factor Cryo-EM density sharpening (ChimeraX `vop sharpen`).
    pub fn volume_sharpen(&self, amount: f64) -> Result<Self, BioStructError> {
        let amt = amount.clamp(0.2, 4.0) as f32;
        let blurred = self.volume_gaussian(2.0)?;
        let values: Vec<f32> = self
            .values
            .iter()
            .zip(blurred.values.iter())
            .map(|(&orig, &blur)| orig + amt * (orig - blur))
            .collect();
        Self::from_raw(
            format!("{}_sharpen({amt:.1})", self.label),
            self.dims,
            self.origin,
            self.voxel_size,
            values,
        )
    }

    /// Difference density map `self - other` resampled onto `self` (ChimeraX `vop subtract` / `Fo-Fc`).
    pub fn volume_subtract(&self, other: &Self) -> Result<Self, BioStructError> {
        let [nx, ny, nz] = self.dims;
        let mut values = Vec::with_capacity(self.values.len());
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let p = self.voxel_world(x, y, z);
                    let diff = self.get(x, y, z) - (other.sample_trilinear(p) as f32);
                    values.push(diff);
                }
            }
        }
        Self::from_raw(
            format!("{}_minus_{}", self.label, other.label),
            self.dims,
            self.origin,
            self.voxel_size,
            values,
        )
    }

    /// Carve / zone density strictly within `radius_a` of `centers` (ChimeraX `volume zone` / PyMOL `carve`).
    pub fn volume_zone(&self, centers: &[[f64; 3]], radius_a: f64) -> Result<Self, BioStructError> {
        if centers.is_empty() {
            return Ok(self.clone());
        }
        let r = radius_a.clamp(1.5, 40.0);
        let r2 = r * r;
        let [nx, ny, nz] = self.dims;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in centers {
            for k in 0..3 {
                lo[k] = lo[k].min(c[k] - r);
                hi[k] = hi[k].max(c[k] + r);
            }
        }
        let slices: Vec<Vec<f32>> = (0..nz)
            .into_par_iter()
            .map(|z| {
                let wz = self.origin[2] + (z as f64) * self.voxel_size[2];
                let mut sl = vec![0.0_f32; nx * ny];
                if wz < lo[2] || wz > hi[2] {
                    return sl;
                }
                for y in 0..ny {
                    let wy = self.origin[1] + (y as f64) * self.voxel_size[1];
                    if wy < lo[1] || wy > hi[1] {
                        continue;
                    }
                    for x in 0..nx {
                        let wx = self.origin[0] + (x as f64) * self.voxel_size[0];
                        if wx < lo[0] || wx > hi[0] {
                            continue;
                        }
                        let inside = centers.iter().any(|c| {
                            let dx = wx - c[0];
                            let dy = wy - c[1];
                            let dz = wz - c[2];
                            dx * dx + dy * dy + dz * dz <= r2
                        });
                        if inside {
                            sl[y * nx + x] = self.get(x, y, z);
                        }
                    }
                }
                sl
            })
            .collect();
        let mut values = Vec::with_capacity(self.values.len());
        for mut s in slices {
            values.append(&mut s);
        }
        Self::from_raw(
            format!("{}_zone({r:.1}A)", self.label),
            self.dims,
            self.origin,
            self.voxel_size,
            values,
        )
    }

    /// 6-connected 3D flood-fill noise-island suppressor (ChimeraX `hide dust`).
    /// Zeroes all connected islands above `mean + sigma_level * std_dev` smaller than `min_voxels`.
    pub fn volume_hide_dust(
        &self,
        sigma_level: f64,
        min_voxels: usize,
    ) -> Result<(Self, usize), BioStructError> {
        let [nx, ny, nz] = self.dims;
        let total = self.values.len();
        let thr = (self.mean + sigma_level * self.std_dev) as f32;
        let min_v = min_voxels.max(2);
        let mut visited = vec![false; total];
        let mut values = self.values.clone();
        let mut removed_islands = 0usize;
        let mut queue = std::collections::VecDeque::new();

        for start in 0..total {
            if visited[start] || values[start] < thr {
                continue;
            }
            visited[start] = true;
            queue.clear();
            queue.push_back(start);
            let mut comp = vec![start];
            while let Some(u) = queue.pop_front() {
                let z = u / (nx * ny);
                let rem = u % (nx * ny);
                let y = rem / nx;
                let x = rem % nx;
                let nbrs: [(isize, isize, isize); 6] = [
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ];
                for &(dx, dy, dz) in &nbrs {
                    let nx_i = x as isize + dx;
                    let ny_i = y as isize + dy;
                    let nz_i = z as isize + dz;
                    if nx_i >= 0
                        && (nx_i as usize) < nx
                        && ny_i >= 0
                        && (ny_i as usize) < ny
                        && nz_i >= 0
                        && (nz_i as usize) < nz
                    {
                        let v_idx =
                            (nz_i as usize) * nx * ny + (ny_i as usize) * nx + (nx_i as usize);
                        if !visited[v_idx] && values[v_idx] >= thr {
                            visited[v_idx] = true;
                            queue.push_back(v_idx);
                            comp.push(v_idx);
                        }
                    }
                }
            }
            if comp.len() < min_v {
                removed_islands += 1;
                for idx in comp {
                    values[idx] = self.mean as f32;
                }
            }
        }
        let cleaned = Self::from_raw(
            format!("{}_nodust", self.label),
            self.dims,
            self.origin,
            self.voxel_size,
            values,
        )?;
        Ok((cleaned, removed_islands))
    }
}

/// Parse a standard binary MRC2014 / CCP4 `.mrc`/`.map` Cryo-EM density map (mode 2 `f32`) in safe Rust.
pub fn parse_mrc_map(label: &str, bytes: &[u8]) -> Result<VolumeGrid3D, BioStructError> {
    if bytes.len() < 1024 {
        return Err(BioStructError::Parse(
            "MRC file smaller than 1024-byte header".to_string(),
        ));
    }
    let read_i32 = |off: usize, le: bool| -> Result<i32, BioStructError> {
        let s = bytes
            .get(off..off + 4)
            .ok_or_else(|| BioStructError::Parse("MRC header EOF".to_string()))?;
        let arr: [u8; 4] = s
            .try_into()
            .map_err(|_| BioStructError::Parse("bad i32 slice".to_string()))?;
        Ok(if le {
            i32::from_le_bytes(arr)
        } else {
            i32::from_be_bytes(arr)
        })
    };
    let read_f32 = |off: usize, le: bool| -> Result<f32, BioStructError> {
        let s = bytes
            .get(off..off + 4)
            .ok_or_else(|| BioStructError::Parse("MRC float EOF".to_string()))?;
        let arr: [u8; 4] = s
            .try_into()
            .map_err(|_| BioStructError::Parse("bad f32 slice".to_string()))?;
        Ok(if le {
            f32::from_le_bytes(arr)
        } else {
            f32::from_be_bytes(arr)
        })
    };

    // Detect endianness via mode (word 3, offset 12): mode 2 = float32
    let mode_le = read_i32(12, true)?;
    let mode_be = read_i32(12, false)?;
    let le = if mode_le == 2 {
        true
    } else if mode_be == 2 {
        false
    } else {
        return Err(BioStructError::Parse(format!(
            "unsupported MRC mode (le={mode_le}, be={mode_be}; only mode 2 f32 supported)"
        )));
    };

    let nx = read_i32(0, le)? as usize;
    let ny = read_i32(4, le)? as usize;
    let nz = read_i32(8, le)? as usize;
    let nxstart = read_i32(16, le)?;
    let nystart = read_i32(20, le)?;
    let nzstart = read_i32(24, le)?;
    let mx = read_i32(28, le)?.max(1) as f64;
    let my = read_i32(32, le)?.max(1) as f64;
    let mz = read_i32(36, le)?.max(1) as f64;
    let cell_x = f64::from(read_f32(40, le)?);
    let cell_y = f64::from(read_f32(44, le)?);
    let cell_z = f64::from(read_f32(48, le)?);
    let nsymbt = read_i32(92, le)?.max(0) as usize;
    let orig_x = f64::from(read_f32(196, le)?);
    let orig_y = f64::from(read_f32(200, le)?);
    let orig_z = f64::from(read_f32(204, le)?);

    let vx = if cell_x > 1e-4 { cell_x / mx } else { 1.0 };
    let vy = if cell_y > 1e-4 { cell_y / my } else { 1.0 };
    let vz = if cell_z > 1e-4 { cell_z / mz } else { 1.0 };
    let origin = [
        if orig_x.abs() > 1e-5 {
            orig_x
        } else {
            f64::from(nxstart) * vx
        },
        if orig_y.abs() > 1e-5 {
            orig_y
        } else {
            f64::from(nystart) * vy
        },
        if orig_z.abs() > 1e-5 {
            orig_z
        } else {
            f64::from(nzstart) * vz
        },
    ];

    let total = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| BioStructError::Parse("MRC voxel count overflow".to_string()))?;
    if total == 0 || total > MAX_VOLUME_VOXELS {
        return Err(BioStructError::Parse(format!(
            "MRC dimensions {nx}x{ny}x{nz} exceed cap {MAX_VOLUME_VOXELS}"
        )));
    }
    let data_start = 1024 + nsymbt;
    let needed = data_start
        .checked_add(total * 4)
        .ok_or_else(|| BioStructError::Parse("MRC data size overflow".to_string()))?;
    if bytes.len() < needed {
        return Err(BioStructError::Parse(format!(
            "truncated MRC data ({} < {needed} bytes)",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(total);
    for i in 0..total {
        values.push(read_f32(data_start + i * 4, le)?);
    }
    VolumeGrid3D::from_raw(label, [nx, ny, nz], origin, [vx, vy, vz], values)
}

/// Write a standard MRC2014 little-endian mode-2 (`f32`) Cryo-EM density map readable by ChimeraX/Coot.
#[must_use]
pub fn write_mrc_map(grid: &VolumeGrid3D) -> Vec<u8> {
    let mut out = vec![0u8; 1024 + grid.values.len() * 4];
    let put_i32 = |buf: &mut [u8], off: usize, v: i32| {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put_f32 = |buf: &mut [u8], off: usize, v: f32| {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    };
    put_i32(&mut out, 0, grid.dims[0] as i32);
    put_i32(&mut out, 4, grid.dims[1] as i32);
    put_i32(&mut out, 8, grid.dims[2] as i32);
    put_i32(&mut out, 12, 2); // mode 2 = f32
    put_i32(&mut out, 28, grid.dims[0] as i32);
    put_i32(&mut out, 32, grid.dims[1] as i32);
    put_i32(&mut out, 36, grid.dims[2] as i32);
    put_f32(
        &mut out,
        40,
        (grid.dims[0] as f64 * grid.voxel_size[0]) as f32,
    );
    put_f32(
        &mut out,
        44,
        (grid.dims[1] as f64 * grid.voxel_size[1]) as f32,
    );
    put_f32(
        &mut out,
        48,
        (grid.dims[2] as f64 * grid.voxel_size[2]) as f32,
    );
    put_f32(&mut out, 52, 90.0);
    put_f32(&mut out, 56, 90.0);
    put_f32(&mut out, 60, 90.0);
    put_i32(&mut out, 64, 1); // MAPC = X
    put_i32(&mut out, 68, 2); // MAPR = Y
    put_i32(&mut out, 72, 3); // MAPS = Z
    put_f32(&mut out, 76, grid.min_val as f32);
    put_f32(&mut out, 80, grid.max_val as f32);
    put_f32(&mut out, 84, grid.mean as f32);
    put_f32(&mut out, 196, grid.origin[0] as f32);
    put_f32(&mut out, 200, grid.origin[1] as f32);
    put_f32(&mut out, 204, grid.origin[2] as f32);
    out[208..212].copy_from_slice(b"MAP ");
    out[212..216].copy_from_slice(&[0x44, 0x41, 0x00, 0x00]); // little-endian stamp
    put_f32(&mut out, 216, grid.std_dev as f32);
    for (i, &v) in grid.values.iter().enumerate() {
        put_f32(&mut out, 1024 + i * 4, v);
    }
    out
}

/// Parse an ASCII APBS / OpenDX `.dx` volumetric electrostatic potential grid.
pub fn parse_dx_map(label: &str, text: &str) -> Result<VolumeGrid3D, BioStructError> {
    let mut dims = [0usize; 3];
    let mut origin = [0.0_f64; 3];
    let mut delta = [1.0_f64; 3];
    let mut delta_idx = 0usize;
    let mut reading_data = false;
    let mut raw_zyx = Vec::new();

    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if l.starts_with("object 1 ") && l.contains("counts") {
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() >= 3 {
                let n = parts.len();
                dims[0] = parts[n - 3].parse().unwrap_or(0);
                dims[1] = parts[n - 2].parse().unwrap_or(0);
                dims[2] = parts[n - 1].parse().unwrap_or(0);
            }
        } else if let Some(rest) = l.strip_prefix("origin") {
            let nums: Vec<f64> = rest
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            if nums.len() == 3 {
                origin = [nums[0], nums[1], nums[2]];
            }
        } else if let Some(rest) = l.strip_prefix("delta") {
            let nums: Vec<f64> = rest
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            if nums.len() == 3 && delta_idx < 3 {
                delta[delta_idx] = nums[delta_idx].abs().max(1e-4);
                delta_idx += 1;
            }
        } else if l.starts_with("object 3 ") && l.contains("data follows") {
            reading_data = true;
        } else if reading_data {
            if l.starts_with("attribute") || l.starts_with("object") {
                break;
            }
            for tok in l.split_whitespace() {
                if let Ok(v) = tok.parse::<f32>() {
                    raw_zyx.push(v);
                }
            }
        }
    }
    let total = dims[0] * dims[1] * dims[2];
    if total == 0 || raw_zyx.len() < total {
        return Err(BioStructError::Parse(format!(
            "incomplete OpenDX grid {:?} (parsed {} of {total} values)",
            dims,
            raw_zyx.len()
        )));
    }
    // OpenDX stores Z fastest; convert to MRC X-fastest layout `(z * nx * ny + y * nx + x)`
    let mut values = vec![0.0_f32; total];
    for x in 0..dims[0] {
        for y in 0..dims[1] {
            for z in 0..dims[2] {
                let dx_idx = x * dims[1] * dims[2] + y * dims[2] + z;
                let mrc_idx = z * dims[0] * dims[1] + y * dims[0] + x;
                values[mrc_idx] = raw_zyx[dx_idx];
            }
        }
    }
    VolumeGrid3D::from_raw(label, dims, origin, delta, values)
}

/// Synthesize a resolution-filtered 3D Cryo-EM electron density map from atomic coordinates
/// (ChimeraX `molmap` equivalent: Gaussian atomic scatter `sigma = resolution / (pi * sqrt(2))`).
pub fn simulate_cryoem_map(
    mol: &Biomolecule,
    resolution_a: f64,
    voxel_a: f64,
) -> Result<VolumeGrid3D, BioStructError> {
    let heavy: Vec<([f64; 3], f64)> = mol
        .atoms
        .iter()
        .filter(|a| a.res_class != ResidueClass::Water && a.element != "H")
        .map(|a| {
            let z_num = match a.element.as_str() {
                "C" => 6.0,
                "N" => 7.0,
                "O" => 8.0,
                "P" => 15.0,
                "S" => 16.0,
                "MG" => 12.0,
                "ZN" => 30.0,
                "FE" => 26.0,
                _ => 7.0,
            };
            (a.coords, z_num)
        })
        .collect();
    if heavy.is_empty() {
        return Err(BioStructError::EmptyInput);
    }

    let (lo, hi, _, _) = mol.bounds();
    let res = resolution_a.clamp(1.5, 15.0);
    let sigma = res / (std::f64::consts::PI * std::f64::consts::SQRT_2);
    let pad = (3.2 * sigma).max(4.0);
    let origin = [lo[0] - pad, lo[1] - pad, lo[2] - pad];
    let span = [
        (hi[0] - lo[0]) + 2.0 * pad,
        (hi[1] - lo[1]) + 2.0 * pad,
        (hi[2] - lo[2]) + 2.0 * pad,
    ];
    // Cap each axis cleanly at 96 voxels so molmap stays crisp & sub-second even on large complexes
    let auto_vox = (span[0].max(span[1]).max(span[2]) / 80.0).max(voxel_a.clamp(0.6, 3.0));
    let nx = ((span[0] / auto_vox).ceil() as usize + 1).clamp(8, 96);
    let ny = ((span[1] / auto_vox).ceil() as usize + 1).clamp(8, 96);
    let nz = ((span[2] / auto_vox).ceil() as usize + 1).clamp(8, 96);

    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let r_cut = 3.0 * sigma;
    let r_cut2 = r_cut * r_cut;
    let slices: Vec<Vec<f32>> = (0..nz)
        .into_par_iter()
        .map(|z| {
            let wz = origin[2] + (z as f64) * auto_vox;
            let mut slice = vec![0.0_f32; nx * ny];
            for &(p, znum) in &heavy {
                let dz = wz - p[2];
                if dz.abs() > r_cut {
                    continue;
                }
                let x0 = (((p[0] - r_cut - origin[0]) / auto_vox).floor() as isize).max(0) as usize;
                let x1 = (((p[0] + r_cut - origin[0]) / auto_vox).ceil() as isize)
                    .min((nx - 1) as isize) as usize;
                let y0 = (((p[1] - r_cut - origin[1]) / auto_vox).floor() as isize).max(0) as usize;
                let y1 = (((p[1] + r_cut - origin[1]) / auto_vox).ceil() as isize)
                    .min((ny - 1) as isize) as usize;
                for y in y0..=y1 {
                    let dy = origin[1] + (y as f64) * auto_vox - p[1];
                    let dyz2 = dy * dy + dz * dz;
                    if dyz2 > r_cut2 {
                        continue;
                    }
                    for x in x0..=x1 {
                        let dx = origin[0] + (x as f64) * auto_vox - p[0];
                        let d2 = dx * dx + dyz2;
                        if d2 <= r_cut2 {
                            slice[y * nx + x] += (znum * (-d2 * inv_2s2).exp()) as f32;
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
    VolumeGrid3D::from_raw(
        format!("molmap({:.1}A)_{}", res, mol.id),
        [nx, ny, nz],
        origin,
        [auto_vox, auto_vox, auto_vox],
        values,
    )
}

/// Compute map-to-model correlation & optimize rigid translation inside a Cryo-EM map (`fitmap`).
pub fn fit_model_in_map(
    mol: &mut Biomolecule,
    grid: &VolumeGrid3D,
    optimize_steps: usize,
) -> MapFitReport {
    let heavy_idx: Vec<usize> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.res_class != ResidueClass::Water && a.element != "H")
        .map(|(i, _)| i)
        .collect();

    let eval_score = |shift: [f64; 3]| -> (f64, f64, f64) {
        if heavy_idx.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut sum_rho = 0.0_f64;
        let mut sum_rho2 = 0.0_f64;
        let mut inside_1s = 0usize;
        let thr_1s = grid.mean + grid.std_dev;
        for &i in &heavy_idx {
            let p = mol.atoms[i].coords;
            let val = grid.sample_trilinear([p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]);
            sum_rho += val;
            sum_rho2 += val * val;
            if val >= thr_1s {
                inside_1s += 1;
            }
        }
        let n = heavy_idx.len() as f64;
        let mean_rho = sum_rho / n;
        let mean_sigma = (mean_rho - grid.mean) / grid.std_dev.max(1e-6);
        let frac = 100.0 * (inside_1s as f64) / n;
        // Normalized correlation proxy against ideal unit atomic envelope
        let cc = (sum_rho / ((n * sum_rho2.max(1e-9)).sqrt())).clamp(0.0, 1.0);
        (cc, mean_sigma, frac)
    };

    let (cc_before, _, _) = eval_score([0.0, 0.0, 0.0]);
    let mut best_shift = [0.0_f64; 3];
    let mut best_score = eval_score(best_shift);
    let mut step = 1.0_f64;
    for _ in 0..optimize_steps.min(25) {
        let mut improved = false;
        for &(dx, dy, dz) in &[
            (step, 0.0, 0.0),
            (-step, 0.0, 0.0),
            (0.0, step, 0.0),
            (0.0, -step, 0.0),
            (0.0, 0.0, step),
            (0.0, 0.0, -step),
        ] {
            let cand = [best_shift[0] + dx, best_shift[1] + dy, best_shift[2] + dz];
            let sc = eval_score(cand);
            if sc.1 > best_score.1 + 1e-4 {
                best_score = sc;
                best_shift = cand;
                improved = true;
            }
        }
        if !improved {
            step *= 0.5;
            if step < 0.05 {
                break;
            }
        }
    }
    if best_shift.iter().any(|&v| v.abs() > 1e-4) {
        for a in &mut mol.atoms {
            a.coords[0] += best_shift[0];
            a.coords[1] += best_shift[1];
            a.coords[2] += best_shift[2];
        }
    }
    MapFitReport {
        map_label: grid.label.clone(),
        atoms_evaluated: heavy_idx.len(),
        correlation_before: (cc_before * 1000.0).round() / 1000.0,
        correlation_after: (best_score.0 * 1000.0).round() / 1000.0,
        translation_applied_a: [
            (best_shift[0] * 100.0).round() / 100.0,
            (best_shift[1] * 100.0).round() / 100.0,
            (best_shift[2] * 100.0).round() / 100.0,
        ],
        mean_density_sigma: (best_score.1 * 100.0).round() / 100.0,
        atoms_inside_1sigma_pct: (best_score.2 * 10.0).round() / 10.0,
    }
}

/// Extract 3D isosurface contour shell points at `mean + sigma_level * std_dev`
/// (returning `(world_pos, bead_radius)` pairs cleanly capped at `max_nodes` for smooth GPU & PNG rendering).
#[must_use]
pub fn extract_isosurface_nodes(
    grid: &VolumeGrid3D,
    sigma_level: f64,
    max_nodes: usize,
) -> Vec<([f64; 3], f64)> {
    let thresh = (grid.mean + sigma_level * grid.std_dev) as f32;
    let [nx, ny, nz] = grid.dims;
    if nx < 3 || ny < 3 || nz < 3 {
        return Vec::new();
    }
    let bead_r = (grid.voxel_size[0] + grid.voxel_size[1] + grid.voxel_size[2]) / 3.0 * 0.72;
    let mut surface = Vec::new();
    for z in 1..(nz - 1) {
        for y in 1..(ny - 1) {
            for x in 1..(nx - 1) {
                let v = grid.get(x, y, z);
                if v >= thresh {
                    let is_boundary = grid.get(x - 1, y, z) < thresh
                        || grid.get(x + 1, y, z) < thresh
                        || grid.get(x, y - 1, z) < thresh
                        || grid.get(x, y + 1, z) < thresh
                        || grid.get(x, y, z - 1) < thresh
                        || grid.get(x, y, z + 1) < thresh;
                    if is_boundary {
                        surface.push((grid.voxel_world(x, y, z), bead_r));
                    }
                }
            }
        }
    }
    let cap = max_nodes.max(64);
    if surface.len() > cap {
        let step = (surface.len() as f64) / (cap as f64);
        (0..cap)
            .map(|k| {
                let idx = ((k as f64) * step).floor() as usize;
                surface[idx.min(surface.len() - 1)]
            })
            .collect()
    } else {
        surface
    }
}
