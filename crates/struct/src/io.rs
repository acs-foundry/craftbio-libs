#![forbid(unsafe_code)]

//! PDB (single & multi-model trajectory), mmCIF, and FASTA parsers + PDB writer with
//! automatic covalent bond, disulfide, and 3D RNA/DNA base-pair perception.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use crate::analysis::assign_secondary_structure;
use crate::model::{
    Atom, BasePair3D, BioStructError, Biomolecule, Bond, Chain, Disulfide, MAX_BIOMOL_ATOMS,
    Residue, ResidueClass, SecondaryStructure, TrajectoryFrame, approx_atom_charge,
    classify_residue, covalent_radius, dist3,
};

fn infer_element(atom_name: &str, raw_elem: &str) -> String {
    let e = raw_elem.trim();
    if !e.is_empty() && e.chars().all(|c| c.is_ascii_alphabetic()) {
        return e.to_ascii_uppercase();
    }
    let trimmed = atom_name
        .trim()
        .trim_start_matches(|c: char| c.is_ascii_digit());
    let up = trimmed.to_ascii_uppercase();
    if up.starts_with("CL") {
        return "CL".to_string();
    }
    if up.starts_with("BR") {
        return "BR".to_string();
    }
    if up.starts_with("MG") {
        return "MG".to_string();
    }
    if up.starts_with("ZN") {
        return "ZN".to_string();
    }
    if up.starts_with("FE") {
        return "FE".to_string();
    }
    if up.starts_with("NA") {
        return "NA".to_string();
    }
    trimmed
        .chars()
        .next()
        .map_or_else(|| "C".to_string(), |c| c.to_ascii_uppercase().to_string())
}

/// Build chains, intra/inter-residue covalent bonds, disulfides, RNA/DNA base pairs, and DSSP.
pub fn finalize_biomolecule(
    id: String,
    title: String,
    atoms: Vec<Atom>,
    frames: Vec<TrajectoryFrame>,
    is_alphafold_plddt: bool,
) -> Result<Biomolecule, BioStructError> {
    if atoms.is_empty() {
        return Err(BioStructError::Parse(
            "no atoms found in structure".to_string(),
        ));
    }

    type ResKey = (char, i32, char);
    type ResAccum = (String, char, ResidueClass, Vec<usize>);
    let mut chain_order: Vec<char> = Vec::new();
    let mut res_map: BTreeMap<ResKey, ResAccum> = BTreeMap::new();
    let mut res_order: HashMap<char, Vec<(i32, char)>> = HashMap::new();

    for (idx, a) in atoms.iter().enumerate() {
        if !chain_order.contains(&a.chain) {
            chain_order.push(a.chain);
        }
        let key = (a.chain, a.res_seq, a.icode);
        let entry = res_map.entry(key).or_insert_with(|| {
            res_order
                .entry(a.chain)
                .or_default()
                .push((a.res_seq, a.icode));
            (a.res_name.clone(), a.res_code, a.res_class, Vec::new())
        });
        entry.3.push(idx);
    }

    let mut chains = Vec::with_capacity(chain_order.len());
    let mut bonds = Vec::with_capacity(atoms.len());

    for cid in chain_order {
        let mut residues = Vec::new();
        let mut seq = String::new();
        let mut class_counts = [0usize; 6];
        if let Some(keys) = res_order.get(&cid) {
            for &(rseq, icode) in keys {
                if let Some((rname, code, rclass, a_idxs)) = res_map.get(&(cid, rseq, icode)) {
                    let cidx = match rclass {
                        ResidueClass::Protein => 0,
                        ResidueClass::Rna => 1,
                        ResidueClass::Dna => 2,
                        ResidueClass::Ligand => 3,
                        ResidueClass::Ion => 4,
                        ResidueClass::Water => 5,
                    };
                    class_counts[cidx] += 1;
                    if matches!(
                        rclass,
                        ResidueClass::Protein | ResidueClass::Rna | ResidueClass::Dna
                    ) {
                        seq.push(*code);
                    }

                    // Guide & orientation atoms for ribbon / nucleic ladder rendering
                    let (guide_idx, orient_idx, base_tip_idx) = match rclass {
                        ResidueClass::Protein => {
                            let ca = a_idxs.iter().copied().find(|&i| atoms[i].name == "CA");
                            let o = a_idxs.iter().copied().find(|&i| atoms[i].name == "O");
                            let cb = a_idxs.iter().copied().find(|&i| atoms[i].name == "CB");
                            (ca, o, cb)
                        }
                        ResidueClass::Rna | ResidueClass::Dna => {
                            let c4p = a_idxs
                                .iter()
                                .copied()
                                .find(|&i| atoms[i].name == "C4'" || atoms[i].name == "C4*")
                                .or_else(|| a_idxs.iter().copied().find(|&i| atoms[i].name == "P"));
                            let c1p = a_idxs
                                .iter()
                                .copied()
                                .find(|&i| atoms[i].name == "C1'" || atoms[i].name == "C1*");
                            let tip = a_idxs.iter().copied().find(|&i| {
                                matches!(atoms[i].name.as_str(), "N1" | "N3" | "C4" | "C2" | "N9")
                            });
                            (c4p, c1p, tip)
                        }
                        _ => (a_idxs.first().copied(), None, None),
                    };

                    // Intra-residue covalent bond perception (bounded by residue atom count)
                    for m in 0..a_idxs.len() {
                        let ia = a_idxs[m];
                        let ra = covalent_radius(&atoms[ia].element);
                        for &ib in &a_idxs[(m + 1)..] {
                            let rb = covalent_radius(&atoms[ib].element);
                            let d = dist3(atoms[ia].coords, atoms[ib].coords);
                            if d > 0.45 && d <= (ra + rb) * 1.24 {
                                bonds.push(Bond {
                                    a: ia,
                                    b: ib,
                                    order: 1,
                                });
                            }
                        }
                    }

                    let mean_b = if a_idxs.is_empty() {
                        0.0
                    } else {
                        a_idxs.iter().map(|&i| atoms[i].b_factor).sum::<f64>()
                            / (a_idxs.len() as f64)
                    };
                    let ss = if matches!(rclass, ResidueClass::Rna | ResidueClass::Dna) {
                        SecondaryStructure::NucleicRibbon
                    } else {
                        SecondaryStructure::Coil
                    };
                    residues.push(Residue {
                        chain: cid,
                        res_seq: rseq,
                        icode,
                        res_name: rname.clone(),
                        code: *code,
                        class: *rclass,
                        ss,
                        atom_indices: a_idxs.clone(),
                        guide_idx,
                        orient_idx,
                        base_tip_idx,
                        mean_bfactor: mean_b,
                    });
                }
            }
        }

        // Inter-residue polymer backbone bonds (peptide C–N or nucleic O3'–P)
        for pair in residues.windows(2) {
            let r1 = &pair[0];
            let r2 = &pair[1];
            if r1.class == ResidueClass::Protein && r2.class == ResidueClass::Protein {
                if let (Some(c_idx), Some(n_idx)) = (
                    r1.atom_indices
                        .iter()
                        .copied()
                        .find(|&i| atoms[i].name == "C"),
                    r2.atom_indices
                        .iter()
                        .copied()
                        .find(|&i| atoms[i].name == "N"),
                ) && dist3(atoms[c_idx].coords, atoms[n_idx].coords) <= 1.85
                {
                    bonds.push(Bond {
                        a: c_idx,
                        b: n_idx,
                        order: 1,
                    });
                }
            } else if matches!(r1.class, ResidueClass::Rna | ResidueClass::Dna)
                && matches!(r2.class, ResidueClass::Rna | ResidueClass::Dna)
                && let (Some(o3), Some(p_idx)) = (
                    r1.atom_indices
                        .iter()
                        .copied()
                        .find(|&i| atoms[i].name == "O3'" || atoms[i].name == "O3*"),
                    r2.atom_indices
                        .iter()
                        .copied()
                        .find(|&i| atoms[i].name == "P"),
                )
                && dist3(atoms[o3].coords, atoms[p_idx].coords) <= 2.10
            {
                bonds.push(Bond {
                    a: o3,
                    b: p_idx,
                    order: 1,
                });
            }
        }

        let dominant_class = if class_counts[0] >= class_counts[1]
            && class_counts[0] >= class_counts[2]
            && class_counts[0] > 0
        {
            ResidueClass::Protein
        } else if class_counts[1] >= class_counts[2] && class_counts[1] > 0 {
            ResidueClass::Rna
        } else if class_counts[2] > 0 {
            ResidueClass::Dna
        } else if class_counts[3] > 0 {
            ResidueClass::Ligand
        } else if class_counts[4] > 0 {
            ResidueClass::Ion
        } else {
            ResidueClass::Water
        };

        chains.push(Chain {
            id: cid,
            dominant_class,
            sequence: seq,
            residues,
        });
    }

    // Disulfide bridges (SG–SG <= 2.65 Å)
    let sg_atoms: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            a.res_class == ResidueClass::Protein && a.res_code == 'C' && a.name == "SG"
        })
        .map(|(i, _)| i)
        .collect();
    let mut disulfides = Vec::new();
    for i in 0..sg_atoms.len() {
        for j in (i + 1)..sg_atoms.len() {
            let a = &atoms[sg_atoms[i]];
            let b = &atoms[sg_atoms[j]];
            let d = dist3(a.coords, b.coords);
            if (1.5..=2.65).contains(&d) {
                bonds.push(Bond {
                    a: sg_atoms[i],
                    b: sg_atoms[j],
                    order: 1,
                });
                disulfides.push(Disulfide {
                    chain_a: a.chain,
                    res_seq_a: a.res_seq,
                    chain_b: b.chain,
                    res_seq_b: b.res_seq,
                    distance_a: (d * 100.0).round() / 100.0,
                });
            }
        }
    }

    // Detect 3D RNA/DNA base pairs (Watson-Crick & Wobble G-U)
    let base_pairs = detect_nucleic_base_pairs(&atoms, &chains);

    let mut mol = Biomolecule {
        id,
        title,
        atoms,
        bonds,
        chains,
        disulfides,
        base_pairs,
        frames,
        active_frame: 0,
        is_alphafold_plddt,
    };
    assign_secondary_structure(&mut mol);
    Ok(mol)
}

fn detect_nucleic_base_pairs(atoms: &[Atom], chains: &[Chain]) -> Vec<BasePair3D> {
    let mut nucs: Vec<&Residue> = Vec::new();
    for ch in chains {
        for r in &ch.residues {
            if matches!(r.class, ResidueClass::Rna | ResidueClass::Dna) && r.orient_idx.is_some() {
                nucs.push(r);
            }
        }
    }
    let mut pairs = Vec::new();
    for i in 0..nucs.len() {
        let ra = nucs[i];
        let Some(c1a) = ra.orient_idx else { continue };
        for rb in &nucs[(i + 1)..] {
            if ra.chain == rb.chain && (ra.res_seq - rb.res_seq).abs() < 3 {
                continue;
            }
            let Some(c1b) = rb.orient_idx else { continue };
            let dc1 = dist3(atoms[c1a].coords, atoms[c1b].coords);
            if !(8.5..=12.8).contains(&dc1) {
                continue;
            }
            // Count nucleobase N/O..N/O H-bonds <= 3.45 A (exclude phosphate/sugar backbone)
            let mut hb = 0usize;
            for &ia in &ra.atom_indices {
                let aa = &atoms[ia];
                if !matches!(aa.element.as_str(), "N" | "O")
                    || aa.name.contains('\'')
                    || aa.name.starts_with("OP")
                {
                    continue;
                }
                for &ib in &rb.atom_indices {
                    let ab = &atoms[ib];
                    if !matches!(ab.element.as_str(), "N" | "O")
                        || ab.name.contains('\'')
                        || ab.name.starts_with("OP")
                    {
                        continue;
                    }
                    if dist3(aa.coords, ab.coords) <= 3.45 {
                        hb += 1;
                    }
                }
            }
            if hb >= 2 {
                let pair_str = format!("{}-{}", ra.code, rb.code);
                let kind = match pair_str.as_str() {
                    "G-C" | "C-G" => "Watson-Crick G-C",
                    "A-U" | "U-A" | "A-T" | "T-A" => "Watson-Crick A-U/T",
                    "G-U" | "U-G" => "Wobble G-U",
                    _ => "Non-Canonical",
                }
                .to_string();
                pairs.push(BasePair3D {
                    chain_a: ra.chain,
                    res_seq_a: ra.res_seq,
                    base_a: ra.code,
                    chain_b: rb.chain,
                    res_seq_b: rb.res_seq,
                    base_b: rb.code,
                    kind,
                    c1_distance_a: (dc1 * 100.0).round() / 100.0,
                    hbond_count: hb,
                });
            }
        }
    }
    pairs
}

/// Parse PDB text (including multi-model `MODEL`/`ENDMDL` trajectories and RNA/DNA/ligands/ions).
pub fn parse_pdb(id: &str, pdb_text: &str) -> Result<Biomolecule, BioStructError> {
    if pdb_text.trim().is_empty() {
        return Err(BioStructError::EmptyInput);
    }
    let mut title = String::new();
    let mut is_af = false;
    let mut model_idx = 0usize;
    let mut first_model_done = false;
    let mut best_atoms: HashMap<(char, i32, char, String), Atom> = HashMap::new();
    let mut order_keys: Vec<(char, i32, char, String)> = Vec::new();
    let mut frames: Vec<TrajectoryFrame> = Vec::new();
    let mut cur_frame_coords: Vec<[f64; 3]> = Vec::new();

    for line in pdb_text.lines() {
        if line.starts_with("MODEL ") {
            if first_model_done {
                cur_frame_coords.clear();
            }
            continue;
        }
        if line.starts_with("ENDMDL") {
            if !first_model_done && !order_keys.is_empty() {
                first_model_done = true;
                let coords: Vec<[f64; 3]> = order_keys
                    .iter()
                    .filter_map(|k| best_atoms.get(k).map(|a| a.coords))
                    .collect();
                frames.push(TrajectoryFrame {
                    frame_index: 0,
                    time_ps: 0.0,
                    coords,
                });
                model_idx = 1;
            } else if first_model_done && cur_frame_coords.len() == order_keys.len() {
                frames.push(TrajectoryFrame {
                    frame_index: model_idx,
                    time_ps: (model_idx as f64) * 10.0,
                    coords: std::mem::take(&mut cur_frame_coords),
                });
                model_idx += 1;
            }
            continue;
        }
        if line.starts_with("TITLE") {
            let t = line.get(10..).unwrap_or("").trim();
            if !t.is_empty() {
                if !title.is_empty() {
                    title.push(' ');
                }
                title.push_str(t);
            }
            if t.to_ascii_uppercase().contains("ALPHAFOLD") {
                is_af = true;
            }
            continue;
        }
        if line.starts_with("REMARK") && line.to_ascii_uppercase().contains("ALPHAFOLD") {
            is_af = true;
            continue;
        }
        let is_atom = line.starts_with("ATOM  ");
        let is_het = line.starts_with("HETATM");
        if (!is_atom && !is_het) || line.len() < 54 {
            continue;
        }
        let atom_name = line.get(12..16).unwrap_or("").trim().to_string();
        let raw_elem = line.get(76..78).unwrap_or("");
        let elem = infer_element(&atom_name, raw_elem);
        if elem == "H" || elem == "D" {
            continue;
        }
        let x: f64 = line
            .get(30..38)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);
        let y: f64 = line
            .get(38..46)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);
        let z: f64 = line
            .get(46..54)
            .unwrap_or("0")
            .trim()
            .parse()
            .unwrap_or(f64::NAN);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }

        if first_model_done {
            cur_frame_coords.push([x, y, z]);
            continue;
        }

        let res_name = line
            .get(17..20)
            .unwrap_or("UNK")
            .trim()
            .to_ascii_uppercase();
        let (res_class, res_code) = classify_residue(&res_name);
        let alt = line.as_bytes().get(16).copied().unwrap_or(b' ') as char;
        let chain = line.as_bytes().get(21).copied().unwrap_or(b'A') as char;
        let chain = if chain == ' ' { 'A' } else { chain };
        let res_seq: i32 = line.get(22..26).unwrap_or("0").trim().parse().unwrap_or(0);
        let icode = line.as_bytes().get(26).copied().unwrap_or(b' ') as char;
        let occ: f64 = line
            .get(54..60)
            .unwrap_or("1.0")
            .trim()
            .parse()
            .unwrap_or(1.0);
        let bf: f64 = line
            .get(60..66)
            .unwrap_or("0.0")
            .trim()
            .parse()
            .unwrap_or(0.0);
        let charge = approx_atom_charge(res_class, res_code, &atom_name);

        let key = (chain, res_seq, icode, atom_name.clone());
        let candidate = Atom {
            serial: 0,
            name: atom_name,
            element: elem,
            res_name,
            res_code,
            res_class,
            chain,
            res_seq,
            icode,
            coords: [x, y, z],
            occupancy: occ,
            b_factor: bf,
            formal_charge: charge,
            is_hetatm: is_het,
        };
        if let Some(prev) = best_atoms.get(&key) {
            if occ > prev.occupancy || (occ == prev.occupancy && (alt == ' ' || alt == 'A')) {
                best_atoms.insert(key, candidate);
            }
        } else {
            if order_keys.len() >= MAX_BIOMOL_ATOMS {
                return Err(BioStructError::TooManyAtoms(order_keys.len() + 1));
            }
            order_keys.push(key.clone());
            best_atoms.insert(key, candidate);
        }
    }

    let mut atoms = Vec::with_capacity(order_keys.len());
    for (i, k) in order_keys.into_iter().enumerate() {
        if let Some(mut a) = best_atoms.remove(&k) {
            a.serial = i + 1;
            atoms.push(a);
        }
    }
    if title.is_empty() {
        title = id.to_string();
    }
    finalize_biomolecule(id.to_string(), title, atoms, frames, is_af)
}

/// Parse mmCIF (`_atom_site` loop) into `Biomolecule`.
pub fn parse_mmcif(id: &str, cif_text: &str) -> Result<Biomolecule, BioStructError> {
    if cif_text.trim().is_empty() {
        return Err(BioStructError::EmptyInput);
    }
    let mut cols: Vec<String> = Vec::new();
    let mut in_atom_site = false;
    let mut atoms = Vec::new();

    for raw in cif_text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            if in_atom_site && !atoms.is_empty() {
                break;
            }
            continue;
        }
        if line.starts_with("loop_") {
            if in_atom_site && !atoms.is_empty() {
                break;
            }
            cols.clear();
            in_atom_site = false;
            continue;
        }
        if line.starts_with("_atom_site.") {
            in_atom_site = true;
            cols.push(line.to_string());
            continue;
        }
        if in_atom_site && line.starts_with('_') {
            if !atoms.is_empty() {
                break;
            }
            in_atom_site = false;
            continue;
        }
        if !in_atom_site {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < cols.len() {
            continue;
        }
        let col_val = |name: &str| -> Option<&str> {
            cols.iter()
                .position(|c| c == name)
                .and_then(|i| toks.get(i).copied())
        };
        let group = col_val("_atom_site.group_PDB").unwrap_or("ATOM");
        if group != "ATOM" && group != "HETATM" {
            continue;
        }
        let alt = col_val("_atom_site.label_alt_id").unwrap_or(".");
        if alt != "." && alt != "?" && alt != "A" {
            continue;
        }
        let res_name = col_val("_atom_site.label_comp_id")
            .or_else(|| col_val("_atom_site.auth_comp_id"))
            .unwrap_or("UNK")
            .to_ascii_uppercase();
        let (res_class, res_code) = classify_residue(&res_name);
        let aname = col_val("_atom_site.label_atom_id")
            .or_else(|| col_val("_atom_site.auth_atom_id"))
            .unwrap_or("CA")
            .trim_matches('"')
            .to_string();
        let elem = infer_element(&aname, col_val("_atom_site.type_symbol").unwrap_or(""));
        if elem == "H" || elem == "D" {
            continue;
        }
        let chain = col_val("_atom_site.auth_asym_id")
            .or_else(|| col_val("_atom_site.label_asym_id"))
            .and_then(|s| s.chars().next())
            .unwrap_or('A');
        let res_seq: i32 = col_val("_atom_site.auth_seq_id")
            .or_else(|| col_val("_atom_site.label_seq_id"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let x: f64 = col_val("_atom_site.Cartn_x")
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);
        let y: f64 = col_val("_atom_site.Cartn_y")
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);
        let z: f64 = col_val("_atom_site.Cartn_z")
            .and_then(|s| s.parse().ok())
            .unwrap_or(f64::NAN);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            continue;
        }
        let occ: f64 = col_val("_atom_site.occupancy")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let bf: f64 = col_val("_atom_site.B_iso_or_equiv")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        if atoms.len() >= MAX_BIOMOL_ATOMS {
            return Err(BioStructError::TooManyAtoms(atoms.len() + 1));
        }
        let serial = atoms.len() + 1;
        let formal_charge = approx_atom_charge(res_class, res_code, &aname);
        atoms.push(Atom {
            serial,
            name: aname,
            element: elem,
            res_name,
            res_code,
            res_class,
            chain,
            res_seq,
            icode: ' ',
            coords: [x, y, z],
            occupancy: occ,
            b_factor: bf,
            formal_charge,
            is_hetatm: group == "HETATM",
        });
    }
    finalize_biomolecule(id.to_string(), id.to_string(), atoms, Vec::new(), false)
}

/// Auto-detect PDB vs mmCIF and parse.
pub fn parse_structure_auto(id: &str, text: &str) -> Result<Biomolecule, BioStructError> {
    if text.contains("_atom_site.") {
        parse_mmcif(id, text)
    } else {
        parse_pdb(id, text)
    }
}

/// Serialize a `Biomolecule` back to valid standard PDB format (preserving edits & mutations).
#[must_use]
pub fn write_pdb(mol: &Biomolecule) -> String {
    let mut out = String::with_capacity(mol.atoms.len() * 82 + 256);
    let _ = writeln!(out, "TITLE     {}", mol.title);
    for (idx, a) in mol.atoms.iter().enumerate() {
        let rec = if a.is_hetatm { "HETATM" } else { "ATOM  " };
        let aname = if a.name.len() < 4 {
            format!(" {:<3}", a.name)
        } else {
            a.name.chars().take(4).collect()
        };
        let rname: String = a.res_name.chars().take(3).collect();
        let serial = (idx + 1) % 100_000;
        let _ = writeln!(
            out,
            "{rec}{serial:>5} {aname} {rname:>3} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
            a.chain,
            a.res_seq,
            a.icode,
            a.coords[0],
            a.coords[1],
            a.coords[2],
            a.occupancy,
            a.b_factor,
            a.element
        );
    }
    out.push_str("END\n");
    out
}

/// Parse single- or multi-entry FASTA text into `(header, sequence)` pairs.
pub fn parse_fasta(text: &str) -> Result<Vec<(String, String)>, BioStructError> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(BioStructError::EmptyInput);
    }
    if !trimmed.contains('>') {
        let seq: String = trimmed
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| c.to_ascii_uppercase())
            .collect();
        if seq.is_empty() {
            return Err(BioStructError::EmptyInput);
        }
        return Ok(vec![("Seq_1".to_string(), seq)]);
    }
    let mut out = Vec::new();
    let mut name: Option<String> = None;
    let mut seq = String::new();
    for line in trimmed.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        if let Some(h) = l.strip_prefix('>') {
            if let Some(n) = name.take()
                && !seq.is_empty()
            {
                out.push((n, std::mem::take(&mut seq)));
            }
            name = Some(h.trim().to_string());
        } else {
            for c in l.chars() {
                if c.is_ascii_alphabetic() {
                    seq.push(c.to_ascii_uppercase());
                }
            }
        }
    }
    if let Some(n) = name
        && !seq.is_empty()
    {
        out.push((n, seq));
    }
    if out.is_empty() {
        return Err(BioStructError::EmptyInput);
    }
    Ok(out)
}
