#![forbid(unsafe_code)]

//! Exact Kabsch SVD 3D Superposition & Needleman-Wunsch Sequence-Guided Structural Alignment
//! (Pure-Rust ChimeraX `matchmaker` / PyMOL `align` equivalent for Proteins & RNA/DNA).

use serde::{Deserialize, Serialize};

use crate::model::{BioStructError, Biomolecule, ResidueClass, dist3};

/// Rigid-body 3D transformation `p' = R * (p - src_center) + dst_center`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigidTransform {
    pub rotation: [[f64; 3]; 3],
    pub src_center: [f64; 3],
    pub dst_center: [f64; 3],
    pub rmsd_a: f64,
    pub aligned_pairs: usize,
}

impl RigidTransform {
    #[must_use]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let q = [
            p[0] - self.src_center[0],
            p[1] - self.src_center[1],
            p[2] - self.src_center[2],
        ];
        [
            self.rotation[0][0] * q[0]
                + self.rotation[0][1] * q[1]
                + self.rotation[0][2] * q[2]
                + self.dst_center[0],
            self.rotation[1][0] * q[0]
                + self.rotation[1][1] * q[1]
                + self.rotation[1][2] * q[2]
                + self.dst_center[1],
            self.rotation[2][0] * q[0]
                + self.rotation[2][1] * q[1]
                + self.rotation[2][2] * q[2]
                + self.dst_center[2],
        ]
    }
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

fn mat3_transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}

fn mat3_det(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Jacobi symmetric eigensolver for a 3×3 symmetric matrix; returns `(eigenvalues, eigenvectors_as_columns)`.
fn jacobi_sym3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = a[0][1].abs();
        if a[0][2].abs() > max_off {
            p = 0;
            q = 2;
            max_off = a[0][2].abs();
        }
        if a[1][2].abs() > max_off {
            p = 1;
            q = 2;
            max_off = a[1][2].abs();
        }
        if max_off < 1e-13 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        for row in &mut v {
            let vp = row[p];
            let vq = row[q];
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Exact Kabsch optimal rigid-body superposition aligning `mobile` points onto `target` points.
pub fn kabsch_superpose(
    mobile: &[[f64; 3]],
    target: &[[f64; 3]],
) -> Result<RigidTransform, BioStructError> {
    let n = mobile.len();
    if n < 3 || n != target.len() {
        return Err(BioStructError::Alignment(format!(
            "Kabsch alignment requires equal-length point sets >= 3 (got {} vs {})",
            mobile.len(),
            target.len()
        )));
    }
    let inv_n = 1.0 / (n as f64);
    let mut cm = [0.0; 3];
    let mut ct = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            cm[k] += mobile[i][k];
            ct[k] += target[i][k];
        }
    }
    for k in 0..3 {
        cm[k] *= inv_n;
        ct[k] *= inv_n;
    }

    // Cross-covariance H = sum_i (p_i - cm) * (q_i - ct)^T
    let mut h = [[0.0; 3]; 3];
    for i in 0..n {
        let p = [
            mobile[i][0] - cm[0],
            mobile[i][1] - cm[1],
            mobile[i][2] - cm[2],
        ];
        let q = [
            target[i][0] - ct[0],
            target[i][1] - ct[1],
            target[i][2] - ct[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                h[r][c] += p[r] * q[c];
            }
        }
    }

    // Solve SVD of H via eigendecomposition of H^T H = V Sigma^2 V^T
    let ht = mat3_transpose(h);
    let hth = mat3_mul(ht, h);
    let (eigvals, v_raw) = jacobi_sym3(hth);

    // Sort eigenvalues descending
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| eigvals[b].total_cmp(&eigvals[a]));
    let mut v = [[0.0; 3]; 3];
    let mut sigma = [0.0; 3];
    for col in 0..3 {
        let orig = order[col];
        sigma[col] = eigvals[orig].max(0.0).sqrt();
        for row in 0..3 {
            v[row][col] = v_raw[row][orig];
        }
    }
    // Ensure right-handed V
    if mat3_det(v) < 0.0 {
        for row in &mut v {
            row[2] = -row[2];
        }
    }

    // Compute U = H * V * Sigma^-1 (with cross-product fallback for rank-2 planar inputs)
    let hv = mat3_mul(h, v);
    let mut u = [[0.0; 3]; 3];
    for col in 0..2 {
        let s = sigma[col];
        if s > 1e-10 {
            for row in 0..3 {
                u[row][col] = hv[row][col] / s;
            }
        } else {
            u[col][col] = 1.0;
        }
    }
    // Third column of U as u0 x u1 (ensuring det(U) = +1 -> det(R) = +1!)
    let u0 = [u[0][0], u[1][0], u[2][0]];
    let u1 = [u[0][1], u[1][1], u[2][1]];
    let mut u2 = [
        u0[1] * u1[2] - u0[2] * u1[1],
        u0[2] * u1[0] - u0[0] * u1[2],
        u0[0] * u1[1] - u0[1] * u1[0],
    ];
    let norm2 = (u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]).sqrt();
    if norm2 > 1e-12 {
        for x in &mut u2 {
            *x /= norm2;
        }
    }
    for row in 0..3 {
        u[row][2] = u2[row];
    }

    // Optimal rotation R = V * U^T mapping centered mobile -> centered target
    // Check sign on third axis if det(H) < 0 (pure rotation constraint)
    let mut rot = mat3_mul(v, mat3_transpose(u));
    if mat3_det(rot) < 0.0 {
        for row in &mut v {
            row[2] = -row[2];
        }
        rot = mat3_mul(v, mat3_transpose(u));
    }

    let mut tf = RigidTransform {
        rotation: rot,
        src_center: cm,
        dst_center: ct,
        rmsd_a: 0.0,
        aligned_pairs: n,
    };
    let mut sq_sum = 0.0_f64;
    for i in 0..n {
        let mapped = tf.apply(mobile[i]);
        let d = dist3(mapped, target[i]);
        sq_sum += d * d;
    }
    tf.rmsd_a = ((sq_sum * inv_n).sqrt() * 10_000.0).round() / 10_000.0;
    Ok(tf)
}

/// Needleman-Wunsch global sequence alignment returning matched `(idx_a, idx_b)` pairs.
#[must_use]
pub fn needleman_wunsch_pairs(seq_a: &str, seq_b: &str) -> Vec<(usize, usize)> {
    let a: Vec<char> = seq_a.chars().collect();
    let b: Vec<char> = seq_b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Vec::new();
    }
    let gap = -2_i32;
    let mut dp = vec![vec![0_i32; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = (i as i32) * gap;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
        *cell = (j as i32) * gap;
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub = if a[i - 1] == b[j - 1] { 3 } else { -1 };
            let diag = dp[i - 1][j - 1] + sub;
            let up = dp[i - 1][j] + gap;
            let left = dp[i][j - 1] + gap;
            dp[i][j] = diag.max(up).max(left);
        }
    }
    let mut pairs = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 && j > 0 {
        let sub = if a[i - 1] == b[j - 1] { 3 } else { -1 };
        if dp[i][j] == dp[i - 1][j - 1] + sub {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i][j] == dp[i - 1][j] + gap {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// Smith-Waterman local sequence alignment returning matched `(idx_a, idx_b)` pairs
/// (handles domain-to-multidomain alignment such as VHH -> Fab VH+CH1, just like ChimeraX MatchMaker).
#[must_use]
pub fn smith_waterman_pairs(seq_a: &str, seq_b: &str) -> Vec<(usize, usize)> {
    let a: Vec<char> = seq_a.chars().collect();
    let b: Vec<char> = seq_b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Vec::new();
    }
    let gap = -3_i32;
    let mut dp = vec![vec![0_i32; n + 1]; m + 1];
    let mut best_score = 0_i32;
    let mut best_pos = (0usize, 0usize);
    for i in 1..=m {
        for j in 1..=n {
            let sub = if a[i - 1] == b[j - 1] { 4 } else { -1 };
            let diag = dp[i - 1][j - 1] + sub;
            let up = dp[i - 1][j] + gap;
            let left = dp[i][j - 1] + gap;
            let val = 0.max(diag).max(up).max(left);
            dp[i][j] = val;
            if val > best_score {
                best_score = val;
                best_pos = (i, j);
            }
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = best_pos;
    while i > 0 && j > 0 && dp[i][j] > 0 {
        let sub = if a[i - 1] == b[j - 1] { 4 } else { -1 };
        if dp[i][j] == dp[i - 1][j - 1] + sub {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if dp[i][j] == dp[i - 1][j] + gap {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// Sequence-guided 3D MatchMaker superposition (works for both Protein `CA` and RNA/DNA `C4'`/`P`).
/// Uses Smith-Waterman local sequence alignment + ChimeraX-style iterative outlier pruning (2.5 Å core)
/// so single domains (e.g., VHH) superpose accurately onto multidomain chains (e.g., Fab VH+CH1).
/// Mutates `mobile` atom coordinates in place and returns the optimal `RigidTransform` + core RMSD.
pub fn matchmaker_align(
    mobile: &mut Biomolecule,
    mobile_chain_id: char,
    target: &Biomolecule,
    target_chain_id: char,
) -> Result<RigidTransform, BioStructError> {
    let mc = mobile.chain(mobile_chain_id).ok_or_else(|| {
        BioStructError::Alignment(format!("mobile chain '{mobile_chain_id}' not found"))
    })?;
    let tc = target.chain(target_chain_id).ok_or_else(|| {
        BioStructError::Alignment(format!("target chain '{target_chain_id}' not found"))
    })?;

    let is_polymer = |c: ResidueClass| {
        matches!(
            c,
            ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
        )
    };
    let m_res: Vec<_> = mc
        .residues
        .iter()
        .filter(|r| is_polymer(r.class) && r.guide_idx.is_some())
        .collect();
    let t_res: Vec<_> = tc
        .residues
        .iter()
        .filter(|r| is_polymer(r.class) && r.guide_idx.is_some())
        .collect();
    let m_seq: String = m_res.iter().map(|r| r.code).collect();
    let t_seq: String = t_res.iter().map(|r| r.code).collect();

    let mut matched = smith_waterman_pairs(&m_seq, &t_seq);
    if matched.len() < 3 {
        matched = needleman_wunsch_pairs(&m_seq, &t_seq);
    }
    let mut pts_m = Vec::with_capacity(matched.len());
    let mut pts_t = Vec::with_capacity(matched.len());
    for (im, it) in matched {
        if let (Some(gm), Some(gt)) = (m_res[im].guide_idx, t_res[it].guide_idx) {
            pts_m.push(mobile.atoms[gm].coords);
            pts_t.push(target.atoms[gt].coords);
        }
    }
    let mut tf = kabsch_superpose(&pts_m, &pts_t)?;
    // ChimeraX-style iterative core pruning: trim divergent loops (> 2.5 A) to lock onto the conserved fold
    let min_keep = (pts_m.len() * 2 / 5).max(8).min(pts_m.len());
    for _ in 0..5 {
        if pts_m.len() <= min_keep {
            break;
        }
        let mut indexed: Vec<(usize, f64)> = pts_m
            .iter()
            .zip(pts_t.iter())
            .enumerate()
            .map(|(idx, (&pm, &pt))| (idx, dist3(tf.apply(pm), pt)))
            .collect();
        let worst = indexed.iter().map(|&(_, d)| d).fold(0.0_f64, f64::max);
        if worst <= 2.5 {
            break;
        }
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
        let keep_n = ((pts_m.len() as f64) * 0.85).floor() as usize;
        let keep_n = keep_n.max(min_keep);
        let mut next_m = Vec::with_capacity(keep_n);
        let mut next_t = Vec::with_capacity(keep_n);
        for &(idx, _) in indexed.iter().take(keep_n) {
            next_m.push(pts_m[idx]);
            next_t.push(pts_t[idx]);
        }
        pts_m = next_m;
        pts_t = next_t;
        tf = kabsch_superpose(&pts_m, &pts_t)?;
    }
    for a in &mut mobile.atoms {
        a.coords = tf.apply(a.coords);
    }
    Ok(tf)
}
