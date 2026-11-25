#![forbid(unsafe_code)]

//! PyMOL / ChimeraX-Compatible Biomolecular Selection Algebra Engine.
//! Supports: `all`, `none`, `protein`, `nucleic`, `rna`, `dna`, `ligand`, `water`, `ion`,
//! `backbone`, `sidechain`, `helix`, `sheet`, `coil`, `chain A,B`, `resi 50-120`, `resn TYR`,
//! `name CA`, `elem N,O`, `b > 80`, `within 5.0 of (...)`, `byres (...)`, `and`/`&`, `or`/`|`, `not`/`!`.

use std::collections::{HashMap, HashSet};

use crate::model::{BioStructError, Biomolecule, ResidueClass, SecondaryStructure, dist3};

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    LParen,
    RParen,
    And,
    Or,
    Not,
    Word(String),
}

fn tokenize(expr: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<Tok>| {
        if cur.is_empty() {
            return;
        }
        let w = std::mem::take(cur);
        match w.to_ascii_lowercase().as_str() {
            "and" | "&" => out.push(Tok::And),
            "or" | "|" => out.push(Tok::Or),
            "not" | "!" => out.push(Tok::Not),
            _ => out.push(Tok::Word(w)),
        }
    };
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        let next = chars.get(i + 1).copied();
        match ch {
            '(' => {
                flush(&mut cur, &mut out);
                out.push(Tok::LParen);
            }
            ')' => {
                flush(&mut cur, &mut out);
                out.push(Tok::RParen);
            }
            '&' => {
                flush(&mut cur, &mut out);
                out.push(Tok::And);
            }
            '|' => {
                flush(&mut cur, &mut out);
                out.push(Tok::Or);
            }
            '!' if next == Some('=') => {
                flush(&mut cur, &mut out);
                out.push(Tok::Word("!=".to_string()));
                i += 1;
            }
            '!' => {
                flush(&mut cur, &mut out);
                out.push(Tok::Not);
            }
            '>' | '<' | '=' => {
                flush(&mut cur, &mut out);
                if next == Some('=') {
                    out.push(Tok::Word(format!("{ch}=")));
                    i += 1;
                } else {
                    out.push(Tok::Word(ch.to_string()));
                }
            }
            c if c.is_whitespace() => flush(&mut cur, &mut out),
            _ => cur.push(ch),
        }
        i += 1;
    }
    flush(&mut cur, &mut out);
    out
}

/// Deepest nesting the selection parser will descend into before giving up.
///
/// `parse_unary` and `parse_primary` each recurse, so nesting depth is stack depth: without
/// a budget a long enough run of `(` or `not` overflows the stack and **aborts the process**.
/// That is not a panic, so `catch_unwind` cannot contain it, and every consumer of this
/// crate — including servers that take selections from the network — inherits the fault.
/// Hand-written selections nest a handful of levels; 128 is far beyond any real query.
pub const MAX_SELECTION_DEPTH: usize = 128;

struct Parser<'a> {
    mol: &'a Biomolecule,
    named: &'a HashMap<String, Vec<bool>>,
    toks: Vec<Tok>,
    pos: usize,
    res_ss_by_atom: Vec<SecondaryStructure>,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    /// Descend one nesting level, refusing to go deeper than [`MAX_SELECTION_DEPTH`].
    fn descend(&mut self) -> Result<(), BioStructError> {
        self.depth += 1;
        if self.depth > MAX_SELECTION_DEPTH {
            return Err(BioStructError::Selection(format!(
                "selection nests deeper than {MAX_SELECTION_DEPTH} levels"
            )));
        }
        Ok(())
    }

    fn ascend(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn next(&mut self) -> Option<Tok> {
        if self.pos < self.toks.len() {
            let t = self.toks[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn parse_or(&mut self) -> Result<Vec<bool>, BioStructError> {
        let mut lhs = self.parse_and()?;
        while self.peek() == Some(&Tok::Or) {
            self.next();
            let rhs = self.parse_and()?;
            for (a, b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = *a || *b;
            }
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> Result<Vec<bool>, BioStructError> {
        let mut lhs = self.parse_unary()?;
        while self.peek() == Some(&Tok::And) {
            self.next();
            let rhs = self.parse_unary()?;
            for (a, b) in lhs.iter_mut().zip(rhs.iter()) {
                *a = *a && *b;
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Vec<bool>, BioStructError> {
        if self.peek() == Some(&Tok::Not) {
            self.next();
            self.descend()?;
            let mut m = self.parse_unary()?;
            self.ascend();
            for v in &mut m {
                *v = !*v;
            }
            return Ok(m);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Vec<bool>, BioStructError> {
        let n = self.mol.atoms.len();
        match self.next() {
            Some(Tok::LParen) => {
                self.descend()?;
                let inner = self.parse_or()?;
                self.ascend();
                if self.next() != Some(Tok::RParen) {
                    return Err(BioStructError::Selection("unclosed '('".to_string()));
                }
                Ok(inner)
            }
            Some(Tok::Word(raw)) => {
                let kw = raw.to_ascii_lowercase();
                if let Some(saved) = self.named.get(&raw).or_else(|| self.named.get(&kw))
                    && saved.len() == n
                {
                    return Ok(saved.clone());
                }
                match kw.as_str() {
                    "all" | "*" => Ok(vec![true; n]),
                    "none" => Ok(vec![false; n]),
                    "protein" | "polymer.protein" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| a.res_class == ResidueClass::Protein)
                        .collect()),
                    "nucleic" | "polymer.nucleic" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| matches!(a.res_class, ResidueClass::Rna | ResidueClass::Dna))
                        .collect()),
                    "rna" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| a.res_class == ResidueClass::Rna)
                        .collect()),
                    "dna" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| a.res_class == ResidueClass::Dna)
                        .collect()),
                    "ligand" | "organic" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| a.res_class == ResidueClass::Ligand)
                        .collect()),
                    "hetatm" => Ok(self.mol.atoms.iter().map(|a| a.is_hetatm).collect()),
                    "water" | "solvent" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| a.res_class == ResidueClass::Water)
                        .collect()),
                    "ion" | "inorganic" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| a.res_class == ResidueClass::Ion)
                        .collect()),
                    "backbone" | "bb" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| match a.res_class {
                            ResidueClass::Protein => {
                                matches!(a.name.as_str(), "N" | "CA" | "C" | "O")
                            }
                            ResidueClass::Rna | ResidueClass::Dna => matches!(
                                a.name.as_str(),
                                "P" | "OP1"
                                    | "OP2"
                                    | "O5'"
                                    | "C5'"
                                    | "C4'"
                                    | "O4'"
                                    | "C3'"
                                    | "O3'"
                                    | "C2'"
                                    | "O2'"
                                    | "C1'"
                            ),
                            _ => false,
                        })
                        .collect()),
                    "sidechain" | "sc" => Ok(self
                        .mol
                        .atoms
                        .iter()
                        .map(|a| {
                            a.res_class == ResidueClass::Protein
                                && !matches!(a.name.as_str(), "N" | "CA" | "C" | "O")
                        })
                        .collect()),
                    "helix" => Ok(self
                        .res_ss_by_atom
                        .iter()
                        .map(|&ss| ss == SecondaryStructure::Helix)
                        .collect()),
                    "sheet" => Ok(self
                        .res_ss_by_atom
                        .iter()
                        .map(|&ss| ss == SecondaryStructure::Sheet)
                        .collect()),
                    "coil" | "loop" => Ok(self
                        .res_ss_by_atom
                        .iter()
                        .map(|&ss| ss == SecondaryStructure::Coil)
                        .collect()),
                    "chain" | "c." => {
                        let arg = self.expect_word("chain")?;
                        let set: HashSet<char> = arg
                            .split([',', '+'])
                            .filter_map(|s| s.trim().chars().next())
                            .collect();
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| set.contains(&a.chain))
                            .collect())
                    }
                    "resi" | "i." => {
                        let arg = self.expect_word("resi")?;
                        let ranges = parse_resi_ranges(&arg)?;
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| {
                                ranges
                                    .iter()
                                    .any(|&(lo, hi)| a.res_seq >= lo && a.res_seq <= hi)
                            })
                            .collect())
                    }
                    "resn" | "r." => {
                        let arg = self.expect_word("resn")?;
                        let set: HashSet<String> = arg
                            .split([',', '+'])
                            .map(|s| s.trim().to_ascii_uppercase())
                            .collect();
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| set.contains(&a.res_name))
                            .collect())
                    }
                    "name" | "n." => {
                        let arg = self.expect_word("name")?;
                        let set: HashSet<String> = arg
                            .split([',', '+'])
                            .map(|s| s.trim().to_ascii_uppercase())
                            .collect();
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| set.contains(&a.name.to_ascii_uppercase()))
                            .collect())
                    }
                    "elem" | "e." => {
                        let arg = self.expect_word("elem")?;
                        let set: HashSet<String> = arg
                            .split([',', '+'])
                            .map(|s| s.trim().to_ascii_uppercase())
                            .collect();
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| set.contains(&a.element))
                            .collect())
                    }
                    "b" | "bfactor" | "plddt" => {
                        let op = self.expect_word("b operator")?;
                        let val: f64 = self.expect_word("b value")?.parse().map_err(|_| {
                            BioStructError::Selection("invalid numeric b-factor".to_string())
                        })?;
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| match op.as_str() {
                                ">" => a.b_factor > val,
                                ">=" => a.b_factor >= val,
                                "<" => a.b_factor < val,
                                "<=" => a.b_factor <= val,
                                "!=" => (a.b_factor - val).abs() >= 1e-3,
                                _ => (a.b_factor - val).abs() < 1e-3,
                            })
                            .collect())
                    }
                    "within" => {
                        let dist_s = self.expect_word("within distance")?;
                        let cutoff: f64 = dist_s.parse().map_err(|_| {
                            BioStructError::Selection(format!("invalid within distance '{dist_s}'"))
                        })?;
                        let of_kw = self.expect_word("of")?;
                        if !of_kw.eq_ignore_ascii_case("of") {
                            return Err(BioStructError::Selection(
                                "expected 'of' after 'within <dist>'".to_string(),
                            ));
                        }
                        let target = self.parse_unary()?;
                        let t_coords: Vec<[f64; 3]> = self
                            .mol
                            .atoms
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| target[i])
                            .map(|(_, a)| a.coords)
                            .collect();
                        let mask = self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| t_coords.iter().any(|&tc| dist3(a.coords, tc) <= cutoff))
                            .collect();
                        Ok(mask)
                    }
                    "byres" => {
                        let inner = self.parse_unary()?;
                        let mut hit_res: HashSet<(char, i32, char)> = HashSet::new();
                        for (i, a) in self.mol.atoms.iter().enumerate() {
                            if inner[i] {
                                hit_res.insert((a.chain, a.res_seq, a.icode));
                            }
                        }
                        Ok(self
                            .mol
                            .atoms
                            .iter()
                            .map(|a| hit_res.contains(&(a.chain, a.res_seq, a.icode)))
                            .collect())
                    }
                    other => Err(BioStructError::Selection(format!(
                        "unknown selection keyword or named selection '{other}'"
                    ))),
                }
            }
            other => Err(BioStructError::Selection(format!(
                "unexpected token {other:?} in selection"
            ))),
        }
    }

    fn expect_word(&mut self, ctx: &str) -> Result<String, BioStructError> {
        match self.next() {
            Some(Tok::Word(w)) => Ok(w),
            _ => Err(BioStructError::Selection(format!(
                "expected argument after '{ctx}'"
            ))),
        }
    }
}

fn parse_resi_ranges(spec: &str) -> Result<Vec<(i32, i32)>, BioStructError> {
    let mut out = Vec::new();
    for part in spec.split([',', '+']) {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Some((a, b)) = p.split_once('-') {
            let lo: i32 = a
                .parse()
                .map_err(|_| BioStructError::Selection(format!("invalid resi '{p}'")))?;
            let hi: i32 = b
                .parse()
                .map_err(|_| BioStructError::Selection(format!("invalid resi '{p}'")))?;
            out.push((lo.min(hi), lo.max(hi)));
        } else {
            let v: i32 = p
                .parse()
                .map_err(|_| BioStructError::Selection(format!("invalid resi '{p}'")))?;
            out.push((v, v));
        }
    }
    if out.is_empty() {
        return Err(BioStructError::Selection("empty resi range".to_string()));
    }
    Ok(out)
}

/// Evaluate a PyMOL/ChimeraX selection expression against `mol`, returning a per-atom boolean mask.
pub fn evaluate_selection(
    mol: &Biomolecule,
    expr: &str,
    named: &HashMap<String, Vec<bool>>,
) -> Result<Vec<bool>, BioStructError> {
    let toks = tokenize(expr);
    if toks.is_empty() {
        return Ok(vec![true; mol.atoms.len()]);
    }
    let mut res_ss = vec![SecondaryStructure::Coil; mol.atoms.len()];
    for ch in &mol.chains {
        for r in &ch.residues {
            for &ai in &r.atom_indices {
                res_ss[ai] = r.ss;
            }
        }
    }
    let mut parser = Parser {
        mol,
        named,
        toks,
        pos: 0,
        res_ss_by_atom: res_ss,
        depth: 0,
    };
    let mask = parser.parse_or()?;
    if parser.pos < parser.toks.len() {
        return Err(BioStructError::Selection(format!(
            "unexpected trailing token {:?} in '{}'",
            parser.toks[parser.pos], expr
        )));
    }
    Ok(mask)
}
