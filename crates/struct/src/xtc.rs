#![forbid(unsafe_code)]

//! Pure-Rust GROMACS `.xtc` Compressed Trajectory Reader & Writer (`xdrfile` `3dfcoord` XDR codec).
//!
//! Zero C/FFI dependencies, `#![forbid(unsafe_code)]`, hard bounds against hostile inputs:
//! - Standard big-endian XDR headers (`magic = 1995`, `natoms`, `step`, `time` ps, `3×3 box` nm)
//! - Uncompressed coordinates (`natoms <= 9`) AND full GROMACS `MAGICINTS` bit-packed
//!   mixed-radix delta compression (`natoms > 9`)
//! - Automatic `nm <-> Å` (`× 10.0`) conversion into `Biomolecule::frames`

use crate::model::{BioStructError, Biomolecule, MAX_BIOMOL_ATOMS, TrajectoryFrame};

const XTC_MAGIC: i32 = 1995;
const FIRSTIDX: usize = 9;
const MAX_XTC_FRAMES: usize = 4_000;
const MAX_XTC_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Official GROMACS `xdrfile` 73-entry `magicints` quantization table (`floor(2^(i/3))`).
const MAGICINTS: [i32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203, 256,
    322, 406, 512, 645, 812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321,
    13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031, 131072, 165140, 208063,
    262144, 330280, 416127, 524287, 660561, 832255, 1048576, 1321122, 1664510, 2097152, 2642245,
    3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];
const LASTIDX: usize = MAGICINTS.len();

struct XdrReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> XdrReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_i32(&mut self) -> Result<i32, BioStructError> {
        let end = self
            .pos
            .checked_add(4)
            .ok_or_else(|| BioStructError::Parse("XTC cursor overflow".to_string()))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| BioStructError::Parse("unexpected EOF in XTC i32".to_string()))?;
        self.pos = end;
        Ok(i32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    fn read_f32(&mut self) -> Result<f32, BioStructError> {
        let end = self
            .pos
            .checked_add(4)
            .ok_or_else(|| BioStructError::Parse("XTC cursor overflow".to_string()))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| BioStructError::Parse("unexpected EOF in XTC f32".to_string()))?;
        self.pos = end;
        Ok(f32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
    }

    fn read_opaque_padded(&mut self, byte_len: usize) -> Result<&'a [u8], BioStructError> {
        if byte_len > MAX_XTC_FRAME_BYTES {
            return Err(BioStructError::Parse(format!(
                "XTC compressed payload too large ({byte_len} bytes)"
            )));
        }
        let pad = (4 - (byte_len % 4)) % 4;
        let end = self
            .pos
            .checked_add(byte_len)
            .ok_or_else(|| BioStructError::Parse("XTC payload overflow".to_string()))?;
        let padded_end = end
            .checked_add(pad)
            .ok_or_else(|| BioStructError::Parse("XTC padding overflow".to_string()))?;
        if padded_end > self.data.len() {
            return Err(BioStructError::Parse(
                "unexpected EOF in XTC compressed bitstream".to_string(),
            ));
        }
        let out = &self.data[self.pos..end];
        self.pos = padded_end;
        Ok(out)
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    cnt: usize,
    lastbits: u32,
    lastbyte: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            cnt: 0,
            lastbits: 0,
            lastbyte: 0,
        }
    }

    fn next_byte(&mut self) -> u32 {
        let b = self.buf.get(self.cnt).copied().unwrap_or(0);
        self.cnt = self.cnt.saturating_add(1);
        u32::from(b)
    }

    fn receivebits(&mut self, mut num_of_bits: u32) -> u32 {
        if num_of_bits == 0 {
            return 0;
        }
        let mask = if num_of_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << num_of_bits) - 1
        };
        let mut num = 0u32;
        while num_of_bits >= 8 {
            self.lastbyte = (self.lastbyte << 8) | self.next_byte();
            num |= (self.lastbyte >> self.lastbits) << (num_of_bits - 8);
            num_of_bits -= 8;
        }
        if num_of_bits > 0 {
            if self.lastbits < num_of_bits {
                self.lastbits += 8;
                self.lastbyte = (self.lastbyte << 8) | self.next_byte();
            }
            self.lastbits -= num_of_bits;
            num |= (self.lastbyte >> self.lastbits) & ((1u32 << num_of_bits) - 1);
        }
        num & mask
    }

    fn receiveints(
        &mut self,
        mut num_of_bits: u32,
        sizes: [u32; 3],
    ) -> Result<[i32; 3], BioStructError> {
        let mut bytes = [0u64; 32];
        let mut num_of_bytes = 0usize;
        while num_of_bits > 8 {
            if num_of_bytes >= bytes.len() {
                return Err(BioStructError::Parse(
                    "XTC receiveints bit width exceeded".to_string(),
                ));
            }
            bytes[num_of_bytes] = u64::from(self.receivebits(8));
            num_of_bytes += 1;
            num_of_bits -= 8;
        }
        if num_of_bits > 0 {
            if num_of_bytes >= bytes.len() {
                return Err(BioStructError::Parse(
                    "XTC receiveints bit width exceeded".to_string(),
                ));
            }
            bytes[num_of_bytes] = u64::from(self.receivebits(num_of_bits));
            num_of_bytes += 1;
        }

        let mut nums = [0i32; 3];
        for i in (1..3).rev() {
            let divisor = u64::from(sizes[i]).max(1);
            let mut rem = 0u64;
            for bytecnt in (0..num_of_bytes).rev() {
                let p = (rem << 8) | bytes[bytecnt];
                let q = p / divisor;
                bytes[bytecnt] = q;
                rem = p - q * divisor;
            }
            nums[i] = rem as i32;
        }
        let n0 = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
        nums[0] = n0 as i32;
        Ok(nums)
    }
}

struct BitWriter {
    buf: Vec<u8>,
    lastbits: u32,
    lastbyte: u32,
}

impl BitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            lastbits: 0,
            lastbyte: 0,
        }
    }

    fn sendbits(&mut self, mut num_of_bits: u32, num: u32) {
        while num_of_bits >= 8 {
            num_of_bits -= 8;
            let shift = num_of_bits.min(31);
            let byte = if num_of_bits >= 32 {
                0
            } else {
                (num >> shift) & 0xff
            };
            self.lastbyte = (self.lastbyte << 8) | byte;
            self.buf
                .push(((self.lastbyte >> self.lastbits) & 0xff) as u8);
        }
        if num_of_bits > 0 {
            let mask = (1u32 << num_of_bits) - 1;
            self.lastbyte = (self.lastbyte << num_of_bits) | (num & mask);
            self.lastbits += num_of_bits;
            if self.lastbits >= 8 {
                self.lastbits -= 8;
                self.buf
                    .push(((self.lastbyte >> self.lastbits) & 0xff) as u8);
            }
        }
    }

    fn sendints(&mut self, mut num_of_bits: u32, sizes: [u32; 3], nums: [u32; 3]) {
        let mut bytes = [0u64; 32];
        let mut num_of_bytes = 0usize;
        for i in 0..3 {
            let mut tmp = u64::from(nums[i]);
            let sz = u64::from(sizes[i]);
            let mut bytecnt = 0usize;
            while bytecnt < num_of_bytes {
                tmp += bytes[bytecnt] * sz;
                bytes[bytecnt] = tmp & 0xff;
                tmp >>= 8;
                bytecnt += 1;
            }
            while tmp != 0 && bytecnt < bytes.len() {
                bytes[bytecnt] = tmp & 0xff;
                tmp >>= 8;
                bytecnt += 1;
            }
            num_of_bytes = bytecnt;
        }
        if num_of_bits >= (num_of_bytes as u32) * 8 {
            for &b in bytes.iter().take(num_of_bytes) {
                self.sendbits(8, b as u32);
                num_of_bits -= 8;
            }
            if num_of_bits > 0 {
                self.sendbits(num_of_bits, 0);
            }
        } else {
            let lead = num_of_bytes.saturating_sub(1);
            for &b in bytes.iter().take(lead) {
                self.sendbits(8, b as u32);
                num_of_bits = num_of_bits.saturating_sub(8);
            }
            if num_of_bytes > 0 && num_of_bits > 0 {
                self.sendbits(num_of_bits, bytes[num_of_bytes - 1] as u32);
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.lastbits > 0 {
            let rem = (self.lastbyte << (8 - self.lastbits)) & 0xff;
            self.buf.push(rem as u8);
        }
        self.buf
    }
}

fn sizeofint(size: u32) -> u32 {
    let mut num = 1u64;
    let mut bits = 0u32;
    let target = u64::from(size);
    while target >= num && bits < 32 {
        bits += 1;
        num <<= 1;
    }
    bits
}

fn sizeofints(sizes: [u32; 3]) -> u32 {
    let mut bytes = [0u64; 32];
    bytes[0] = 1;
    let mut num_of_bytes = 1usize;
    for &s in &sizes {
        let mut tmp = 0u64;
        let sz = u64::from(s).max(1);
        let mut bytecnt = 0usize;
        while bytecnt < num_of_bytes {
            tmp += bytes[bytecnt] * sz;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
            bytecnt += 1;
        }
        while tmp != 0 && bytecnt < bytes.len() {
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
            bytecnt += 1;
        }
        num_of_bytes = bytecnt;
    }
    let mut num = 1u64;
    let mut num_of_bits = 0u32;
    let top = bytes[num_of_bytes.saturating_sub(1)];
    while top >= num && num_of_bits < 32 {
        num_of_bits += 1;
        num <<= 1;
    }
    num_of_bits + ((num_of_bytes.saturating_sub(1)) as u32) * 8
}

/// Parse a binary GROMACS `.xtc` trajectory into `mol.frames`, converting coordinates from `nm` to `Å`.
///
/// Supports both uncompressed frames (`<= 9` atoms) and full GROMACS `3dfcoord` bit-packed
/// compressed trajectories (`> 9` atoms). Updates `mol.atoms` to Frame 0 on success.
pub fn load_xtc_frames(mol: &mut Biomolecule, bytes: &[u8]) -> Result<usize, BioStructError> {
    if bytes.len() < 56 {
        return Err(BioStructError::Parse(
            "XTC file too short (< 56 bytes)".to_string(),
        ));
    }
    let mut r = XdrReader::new(bytes);
    let mut frames: Vec<TrajectoryFrame> = Vec::new();

    while r.remaining() >= 56 && frames.len() < MAX_XTC_FRAMES {
        let magic = r.read_i32()?;
        if magic != XTC_MAGIC {
            return Err(BioStructError::Parse(format!(
                "invalid XTC magic {magic} (expected {XTC_MAGIC})"
            )));
        }
        let natoms_i32 = r.read_i32()?;
        if natoms_i32 <= 0 || (natoms_i32 as usize) > MAX_BIOMOL_ATOMS {
            return Err(BioStructError::Parse(format!(
                "invalid XTC atom count {natoms_i32}"
            )));
        }
        let natoms = natoms_i32 as usize;
        if natoms != mol.atoms.len() {
            return Err(BioStructError::Parse(format!(
                "XTC atom count ({natoms}) does not match active structure ({})",
                mol.atoms.len()
            )));
        }
        let _step = r.read_i32()?;
        let time_ps = f64::from(r.read_f32()?);
        // 3×3 simulation box (nm)
        for _ in 0..9 {
            let _ = r.read_f32()?;
        }
        let lsize = r.read_i32()?;
        if lsize != natoms_i32 {
            return Err(BioStructError::Parse(format!(
                "XTC coordinate size mismatch ({lsize} != {natoms_i32})"
            )));
        }

        let mut coords: Vec<[f64; 3]> = Vec::with_capacity(natoms);
        if natoms <= 9 {
            for _ in 0..natoms {
                let x_nm = f64::from(r.read_f32()?);
                let y_nm = f64::from(r.read_f32()?);
                let z_nm = f64::from(r.read_f32()?);
                coords.push([x_nm * 10.0, y_nm * 10.0, z_nm * 10.0]);
            }
        } else {
            let precision = r.read_f32()?;
            if !precision.is_finite() || precision <= 0.0 {
                return Err(BioStructError::Parse(format!(
                    "invalid XTC precision {precision}"
                )));
            }
            let scale_nm_to_a = 10.0_f64 / f64::from(precision);
            let minint = [r.read_i32()?, r.read_i32()?, r.read_i32()?];
            let maxint = [r.read_i32()?, r.read_i32()?, r.read_i32()?];
            let smallidx_raw = r.read_i32()?;
            if smallidx_raw < 0 || (smallidx_raw as usize) >= LASTIDX {
                return Err(BioStructError::Parse(format!(
                    "XTC smallidx out of range ({smallidx_raw})"
                )));
            }
            let mut smallidx = smallidx_raw as usize;
            let byte_len_raw = r.read_i32()?;
            if byte_len_raw < 0 {
                return Err(BioStructError::Parse(format!(
                    "negative XTC bitstream length {byte_len_raw}"
                )));
            }
            let bitstream = r.read_opaque_padded(byte_len_raw as usize)?;

            let mut sizeint = [0u32; 3];
            for k in 0..3 {
                let span = i64::from(maxint[k]) - i64::from(minint[k]) + 1;
                if span <= 0 || span > i64::from(u32::MAX) {
                    return Err(BioStructError::Parse(
                        "XTC minint/maxint bounds invalid".to_string(),
                    ));
                }
                sizeint[k] = span as u32;
            }
            let (bitsize, bitsizeint) =
                if sizeint[0] > 0x00ff_ffff || sizeint[1] > 0x00ff_ffff || sizeint[2] > 0x00ff_ffff
                {
                    (
                        0u32,
                        [
                            sizeofint(sizeint[0]),
                            sizeofint(sizeint[1]),
                            sizeofint(sizeint[2]),
                        ],
                    )
                } else {
                    (sizeofints(sizeint), [0u32; 3])
                };

            let mut smaller = if smallidx > FIRSTIDX {
                MAGICINTS[smallidx - 1] / 2
            } else {
                0
            };
            let mut smallnum = MAGICINTS[smallidx] / 2;
            let mut sizesmall = [MAGICINTS[smallidx] as u32; 3];

            let mut br = BitReader::new(bitstream);
            while coords.len() < natoms {
                let mut thiscoord = if bitsize == 0 {
                    [
                        br.receivebits(bitsizeint[0]) as i32,
                        br.receivebits(bitsizeint[1]) as i32,
                        br.receivebits(bitsizeint[2]) as i32,
                    ]
                } else {
                    br.receiveints(bitsize, sizeint)?
                };
                for k in 0..3 {
                    thiscoord[k] = thiscoord[k].wrapping_add(minint[k]);
                }

                let flag = br.receivebits(1);
                let mut is_smaller: i32 = 0;
                let mut run: usize = 0;
                if flag == 1 {
                    let raw_run = br.receivebits(5) as usize;
                    let rem = raw_run % 3;
                    run = raw_run - rem;
                    is_smaller = (rem as i32) - 1;
                }

                if run > 0 {
                    let mut prevcoord = thiscoord;
                    let steps = run / 3;
                    for step_idx in 0..steps {
                        if coords.len() >= natoms {
                            break;
                        }
                        let mut delta = br.receiveints(smallidx as u32, sizesmall)?;
                        for k in 0..3 {
                            delta[k] = delta[k].wrapping_add(prevcoord[k]).wrapping_sub(smallnum);
                        }
                        if step_idx == 0 {
                            // GROMACS swaps first and second atom in a run for water H-O-H locality
                            std::mem::swap(&mut delta, &mut prevcoord);
                            coords.push([
                                f64::from(prevcoord[0]) * scale_nm_to_a,
                                f64::from(prevcoord[1]) * scale_nm_to_a,
                                f64::from(prevcoord[2]) * scale_nm_to_a,
                            ]);
                            if coords.len() < natoms {
                                coords.push([
                                    f64::from(delta[0]) * scale_nm_to_a,
                                    f64::from(delta[1]) * scale_nm_to_a,
                                    f64::from(delta[2]) * scale_nm_to_a,
                                ]);
                            }
                            prevcoord = delta;
                        } else {
                            prevcoord = delta;
                            coords.push([
                                f64::from(delta[0]) * scale_nm_to_a,
                                f64::from(delta[1]) * scale_nm_to_a,
                                f64::from(delta[2]) * scale_nm_to_a,
                            ]);
                        }
                    }
                } else {
                    coords.push([
                        f64::from(thiscoord[0]) * scale_nm_to_a,
                        f64::from(thiscoord[1]) * scale_nm_to_a,
                        f64::from(thiscoord[2]) * scale_nm_to_a,
                    ]);
                }

                let next_idx = (smallidx as i32 + is_smaller)
                    .clamp(FIRSTIDX as i32, (LASTIDX - 1) as i32)
                    as usize;
                smallidx = next_idx;
                if is_smaller < 0 {
                    smallnum = smaller;
                    smaller = if smallidx > FIRSTIDX {
                        MAGICINTS[smallidx - 1] / 2
                    } else {
                        0
                    };
                } else if is_smaller > 0 {
                    smaller = smallnum;
                    smallnum = MAGICINTS[smallidx] / 2;
                }
                sizesmall = [MAGICINTS[smallidx] as u32; 3];
            }
        }

        frames.push(TrajectoryFrame {
            frame_index: frames.len(),
            time_ps,
            coords,
        });
    }

    if frames.is_empty() {
        return Err(BioStructError::Parse(
            "XTC file contained 0 valid frames".to_string(),
        ));
    }
    let count = frames.len();
    mol.frames = frames;
    mol.set_active_frame(0);
    Ok(count)
}

/// Encode `mol.frames` (or current coordinates as a 1-frame trajectory) into standard GROMACS
/// binary `.xtc` format (`precision` defaults to `1000.0` = `0.001 nm` = `0.01 Å`).
pub fn write_xtc_frames(mol: &Biomolecule, precision: f32) -> Result<Vec<u8>, BioStructError> {
    if mol.atoms.is_empty() {
        return Err(BioStructError::EmptyInput);
    }
    let prec = if precision.is_finite() && precision > 1.0 {
        precision
    } else {
        1000.0
    };
    let natoms = mol.atoms.len();
    let frame_sources: Vec<(f64, Vec<[f64; 3]>)> = if mol.frames.is_empty() {
        vec![(0.0, mol.atoms.iter().map(|a| a.coords).collect())]
    } else {
        mol.frames
            .iter()
            .filter(|f| f.coords.len() == natoms)
            .map(|f| (f.time_ps, f.coords.clone()))
            .collect()
    };
    if frame_sources.is_empty() {
        return Err(BioStructError::Parse(
            "no trajectory frames matching atom count".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(frame_sources.len() * natoms * 6);
    for (step_idx, (time_ps, coords)) in frame_sources.iter().enumerate() {
        // Compute tight orthogonal box (nm)
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for c in coords {
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let bx = ((hi[0] - lo[0] + 10.0) * 0.1) as f32;
        let by = ((hi[1] - lo[1] + 10.0) * 0.1) as f32;
        let bz = ((hi[2] - lo[2] + 10.0) * 0.1) as f32;

        // XDR frame header
        out.extend_from_slice(&XTC_MAGIC.to_be_bytes());
        out.extend_from_slice(&(natoms as i32).to_be_bytes());
        out.extend_from_slice(&(step_idx as i32).to_be_bytes());
        out.extend_from_slice(&(*time_ps as f32).to_be_bytes());
        let box_mat = [bx, 0.0, 0.0, 0.0, by, 0.0, 0.0, 0.0, bz];
        for v in box_mat {
            out.extend_from_slice(&v.to_be_bytes());
        }
        out.extend_from_slice(&(natoms as i32).to_be_bytes());

        if natoms <= 9 {
            for c in coords {
                for &v in c {
                    out.extend_from_slice(&((v * 0.1) as f32).to_be_bytes());
                }
            }
            continue;
        }

        // Quantize coordinates to nm * precision
        let mut qcoords: Vec<[i32; 3]> = Vec::with_capacity(natoms);
        let mut minint = [i32::MAX; 3];
        let mut maxint = [i32::MIN; 3];
        let scale = f64::from(prec) * 0.1;
        for c in coords {
            let q = [
                (c[0] * scale).round() as i32,
                (c[1] * scale).round() as i32,
                (c[2] * scale).round() as i32,
            ];
            for k in 0..3 {
                minint[k] = minint[k].min(q[k]);
                maxint[k] = maxint[k].max(q[k]);
            }
            qcoords.push(q);
        }

        let mut sizeint = [0u32; 3];
        for k in 0..3 {
            let span = (i64::from(maxint[k]) - i64::from(minint[k]) + 1).max(1);
            sizeint[k] = span.min(i64::from(u32::MAX)) as u32;
        }
        let (bitsize, bitsizeint) =
            if sizeint[0] > 0x00ff_ffff || sizeint[1] > 0x00ff_ffff || sizeint[2] > 0x00ff_ffff {
                (
                    0u32,
                    [
                        sizeofint(sizeint[0]),
                        sizeofint(sizeint[1]),
                        sizeofint(sizeint[2]),
                    ],
                )
            } else {
                (sizeofints(sizeint), [0u32; 3])
            };

        let mut smallidx = FIRSTIDX;
        let max_range = sizeint[0].max(sizeint[1]).max(sizeint[2]) as i32;
        while smallidx + 1 < LASTIDX && MAGICINTS[smallidx] < max_range.min(1024) {
            smallidx += 1;
        }
        smallidx = smallidx.clamp(FIRSTIDX, 28);
        let init_smallidx = smallidx;
        let mut smallnum = MAGICINTS[smallidx] / 2;
        let mut smaller = if smallidx > FIRSTIDX {
            MAGICINTS[smallidx - 1] / 2
        } else {
            0
        };
        let mut sizesmall = [MAGICINTS[smallidx] as u32; 3];

        let mut bw = BitWriter::with_capacity(natoms * 5);
        let mut i = 0usize;
        while i < natoms {
            // Look ahead for a contiguous delta-compressible run (up to 8 delta atoms)
            let mut run_atoms = 0usize;
            if i + 1 < natoms {
                let mut prev = qcoords[i];
                for cand in qcoords.iter().skip(i + 1).take(8) {
                    let ok = (0..3).all(|k| (cand[k] - prev[k]).abs() < smallnum);
                    if ok {
                        run_atoms += 1;
                        prev = *cand;
                    } else {
                        break;
                    }
                }
            }

            if run_atoms > 0 {
                // GROMACS swaps atom i and i+1 on encode so decoder's k==0 swap restores exact order
                let anchor = qcoords[i + 1];
                let d0 = [
                    qcoords[i][0] - anchor[0],
                    qcoords[i][1] - anchor[1],
                    qcoords[i][2] - anchor[2],
                ];
                let can_swap = d0.iter().all(|&d| d.abs() < smallnum);
                if can_swap {
                    let abs_enc = [
                        (anchor[0] - minint[0]) as u32,
                        (anchor[1] - minint[1]) as u32,
                        (anchor[2] - minint[2]) as u32,
                    ];
                    if bitsize == 0 {
                        for k in 0..3 {
                            bw.sendbits(bitsizeint[k], abs_enc[k]);
                        }
                    } else {
                        bw.sendints(bitsize, sizeint, abs_enc);
                    }

                    // Check if deltas fit in smaller window
                    let mut all_fit_smaller = smallidx > FIRSTIDX;
                    let mut deltas: Vec<[i32; 3]> = Vec::with_capacity(run_atoms);
                    deltas.push(d0);
                    if d0.iter().any(|&d| d.abs() >= smaller) {
                        all_fit_smaller = false;
                    }
                    let mut prev = anchor;
                    for step in 1..run_atoms {
                        let cur = qcoords[i + 1 + step];
                        let d = [cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]];
                        if d.iter().any(|&v| v.abs() >= smallnum) {
                            break;
                        }
                        if d.iter().any(|&v| v.abs() >= smaller) {
                            all_fit_smaller = false;
                        }
                        deltas.push(d);
                        prev = cur;
                    }
                    let actual_run = deltas.len();
                    let is_smaller: i32 = if all_fit_smaller && smallidx > FIRSTIDX {
                        -1
                    } else {
                        0
                    };
                    bw.sendbits(1, 1);
                    let run_code = (actual_run as u32) * 3 + ((is_smaller + 1) as u32);
                    bw.sendbits(5, run_code);
                    for d in deltas {
                        let u = [
                            (d[0] + smallnum) as u32,
                            (d[1] + smallnum) as u32,
                            (d[2] + smallnum) as u32,
                        ];
                        bw.sendints(smallidx as u32, sizesmall, u);
                    }

                    if is_smaller < 0 && smallidx > FIRSTIDX {
                        smallidx -= 1;
                        smallnum = smaller;
                        smaller = if smallidx > FIRSTIDX {
                            MAGICINTS[smallidx - 1] / 2
                        } else {
                            0
                        };
                        sizesmall = [MAGICINTS[smallidx] as u32; 3];
                    }
                    i += 1 + actual_run;
                    continue;
                }
            }

            // Single uncompressed-delta atom (absolute encoded in bounding box)
            let cur = qcoords[i];
            let abs_enc = [
                (cur[0] - minint[0]) as u32,
                (cur[1] - minint[1]) as u32,
                (cur[2] - minint[2]) as u32,
            ];
            if bitsize == 0 {
                for k in 0..3 {
                    bw.sendbits(bitsizeint[k], abs_enc[k]);
                }
            } else {
                bw.sendints(bitsize, sizeint, abs_enc);
            }
            // Grow smallidx gently if next atom didn't fit in current window
            let grow = i + 1 < natoms && smallidx + 1 < LASTIDX - 2;
            if grow {
                bw.sendbits(1, 1);
                bw.sendbits(5, 2); // run = 0, is_smaller = +1
                smallidx += 1;
                smaller = smallnum;
                smallnum = MAGICINTS[smallidx] / 2;
                sizesmall = [MAGICINTS[smallidx] as u32; 3];
            } else {
                bw.sendbits(1, 0);
            }
            i += 1;
        }

        let packed = bw.finish();
        out.extend_from_slice(&prec.to_be_bytes());
        for v in minint {
            out.extend_from_slice(&v.to_be_bytes());
        }
        for v in maxint {
            out.extend_from_slice(&v.to_be_bytes());
        }
        out.extend_from_slice(&(init_smallidx as i32).to_be_bytes());
        out.extend_from_slice(&(packed.len() as i32).to_be_bytes());
        out.extend_from_slice(&packed);
        let pad = (4 - (packed.len() % 4)) % 4;
        out.extend(std::iter::repeat_n(0u8, pad));
    }

    Ok(out)
}
