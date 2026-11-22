#![forbid(unsafe_code)]

//! `craftbio-struct` — Universal Biomolecular Structure IR, Parsers (PDB/mmCIF/FASTA/Trajectories),
//! DSSP, Shrake-Rupley SASA, 3D SAP & ESP, Kabsch/MatchMaker Alignment, and Selection Algebra.

pub mod align;
pub mod analysis;
pub mod io;
pub mod model;
pub mod selection;

pub use align::{
    RigidTransform, kabsch_superpose, matchmaker_align, needleman_wunsch_pairs,
    smith_waterman_pairs,
};
pub use analysis::{
    DipoleMomentReport, assign_secondary_structure, compute_dipole_moment, compute_esp_at_points,
    compute_per_atom_sap, compute_sasa,
};
pub use io::{
    finalize_biomolecule, load_dcd_frames, parse_fasta, parse_mmcif, parse_pdb,
    parse_structure_auto, write_mmcif, write_multi_model_pdb, write_pdb,
};
pub use model::{
    Atom, BasePair3D, BioStructError, Biomolecule, Bond, Chain, Disulfide, MAX_BIOMOL_ATOMS,
    Residue, ResidueClass, SecondaryStructure, TrajectoryFrame, aa1_to_aa3, approx_atom_charge,
    black_mould_sap_hydro, classify_residue, covalent_radius, cpk_color, dist3, max_residue_sasa,
    nucleotide_color, vdw_radius,
};
pub use selection::evaluate_selection;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn kabsch_recovers_exact_3d_rotation_and_translation() -> Result<(), BioStructError> {
        let src = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [1.5, 2.5, -1.0],
        ];
        // Rotate 90 deg around Z and translate by (10, -5, 3)
        let dst: Vec<[f64; 3]> = src
            .iter()
            .map(|p| [-p[1] + 10.0, p[0] - 5.0, p[2] + 3.0])
            .collect();
        let tf = kabsch_superpose(&src, &dst)?;
        assert!(
            tf.rmsd_a < 1e-4,
            "expected exact Kabsch RMSD ~0, got {}",
            tf.rmsd_a
        );
        Ok(())
    }

    #[test]
    fn parses_protein_plus_rna_and_evaluates_selection_algebra() -> Result<(), BioStructError> {
        let pdb = "\
TITLE     PROTEIN-RNA COMPLEX TEST
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 88.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 92.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00 91.00           C
ATOM      4  O   ALA A   1       1.250   2.350   0.000  1.00 90.00           O
ATOM      5  P     G B   1       4.500   0.000   0.000  1.00 72.00           P
ATOM      6  C4'   G B   1       5.500   1.000   0.000  1.00 75.00           C
ATOM      7  C1'   G B   1       6.800   1.200   0.000  1.00 74.00           C
ATOM      8  N1    G B   1       7.900   1.500   0.000  1.00 76.00           N
HETATM    9 MG    MG C 101       4.200  -1.800   0.000  1.00 99.00          MG
END
";
        let mol = parse_pdb("test_rna", pdb)?;
        assert_eq!(mol.chains.len(), 3);
        assert_eq!(mol.chains[0].dominant_class, ResidueClass::Protein);
        assert_eq!(mol.chains[1].dominant_class, ResidueClass::Rna);
        assert_eq!(mol.chains[2].dominant_class, ResidueClass::Ion);

        let named = HashMap::new();
        let m_rna = evaluate_selection(&mol, "rna and chain B", &named)?;
        assert_eq!(m_rna.iter().filter(|&&b| b).count(), 4);

        let m_near = evaluate_selection(&mol, "protein and within 4.5 of (rna or ion)", &named)?;
        assert!(m_near.iter().any(|&b| b));

        let m_plddt = evaluate_selection(&mol, "b > 85", &named)?;
        assert_eq!(m_plddt.iter().filter(|&&b| b).count(), 5);

        // Roundtrip write_pdb & write_mmcif & dipole moment
        let out_pdb = write_pdb(&mol);
        let mol2 = parse_pdb("rt", &out_pdb)?;
        assert_eq!(mol2.atoms.len(), mol.atoms.len());

        let out_cif = write_mmcif(&mol);
        let mol3 = parse_mmcif("rt_cif", &out_cif)?;
        assert_eq!(mol3.atoms.len(), mol.atoms.len());

        let dip = compute_dipole_moment(&mol);
        assert!(dip.dipole_magnitude_debye >= 0.0);

        // Synthetic little-endian CHARMM/OpenMM DCD binary roundtrip test (2 frames)
        let mut dcd = Vec::new();
        let mut push_rec = |payload: &[u8]| {
            let n = payload.len() as u32;
            dcd.extend_from_slice(&n.to_le_bytes());
            dcd.extend_from_slice(payload);
            dcd.extend_from_slice(&n.to_le_bytes());
        };
        let mut hdr = vec![0u8; 84];
        hdr[0..4].copy_from_slice(b"CORD");
        hdr[4..8].copy_from_slice(&2u32.to_le_bytes());
        push_rec(&hdr);
        let mut title = vec![0u8; 84];
        title[0..4].copy_from_slice(&1u32.to_le_bytes());
        push_rec(&title);
        push_rec(&(mol.atoms.len() as u32).to_le_bytes());
        for f in 0..2 {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut zs = Vec::new();
            for a in &mol.atoms {
                xs.extend_from_slice(&((a.coords[0] + f as f64) as f32).to_le_bytes());
                ys.extend_from_slice(&(a.coords[1] as f32).to_le_bytes());
                zs.extend_from_slice(&(a.coords[2] as f32).to_le_bytes());
            }
            push_rec(&xs);
            push_rec(&ys);
            push_rec(&zs);
        }
        let mut mol_traj = mol.clone();
        let loaded = load_dcd_frames(&mut mol_traj, &dcd)?;
        assert_eq!(loaded, 2);
        let multi = write_multi_model_pdb(&mol_traj);
        assert!(multi.contains("MODEL        2"));
        Ok(())
    }
}
