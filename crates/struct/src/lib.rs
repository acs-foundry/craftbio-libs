#![forbid(unsafe_code)]

//! `craftbio-struct` — Universal Biomolecular Structure IR, Parsers (PDB/mmCIF/FASTA/Trajectories),
//! DSSP, Shrake-Rupley SASA, 3D SAP & ESP, Kabsch/MatchMaker Alignment, and Selection Algebra.

pub mod align;
pub mod analysis;
pub mod io;
pub mod measure;
pub mod model;
pub mod selection;
pub mod volume;

pub use align::{
    RigidTransform, cealign_superpose, kabsch_superpose, matchmaker_align, morph_biomolecules,
    needleman_wunsch_pairs, smith_waterman_pairs,
};
pub use analysis::{
    DipoleMomentReport, assign_secondary_structure, compute_dipole_moment, compute_esp_at_points,
    compute_mlp_at_points, compute_per_atom_mlp, compute_per_atom_sap, compute_sasa,
};
pub use io::{
    SdfPoseRecord, finalize_biomolecule, load_dcd_frames, parse_fasta, parse_mmcif, parse_pdb,
    parse_sdf_poses, parse_structure_auto, write_mmcif, write_multi_model_pdb, write_pdb,
};
pub use measure::{
    ClashRecord, ContactRecord, GeometryAxis, GeometryPlane, HBondRecord, MsaAlignment, MsaRow,
    PaeMatrix, axis_angle_deg, build_synchronized_msa, compute_best_fit_plane,
    compute_geometry_axis, detect_clashes, detect_contacts, detect_hbonds,
    estimate_pae_proxy_from_structure, measure_angle_deg, measure_dihedral_deg, measure_distance,
    parse_pae_json,
};
pub use model::{
    Atom, BasePair3D, BioStructError, Biomolecule, Bond, Chain, Disulfide, MAX_BIOMOL_ATOMS,
    Residue, ResidueClass, SecondaryStructure, TrajectoryFrame, aa1_to_aa3, approx_atom_charge,
    black_mould_sap_hydro, classify_residue, covalent_radius, cpk_color, dist3, max_residue_sasa,
    nucleotide_color, vdw_radius,
};
pub use selection::evaluate_selection;
pub use volume::{
    MAX_VOLUME_VOXELS, MapFitReport, VolumeGrid3D, extract_isosurface_nodes, fit_model_in_map,
    parse_dx_map, parse_mrc_map, simulate_cryoem_map, write_mrc_map,
};

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

        // Cryo-EM MRC2014 molmap -> write_mrc_map -> parse_mrc_map -> fit_model_in_map -> isosurface
        let vol = simulate_cryoem_map(&mol, 4.0, 1.5)?;
        let mrc_bytes = write_mrc_map(&vol);
        let vol2 = parse_mrc_map("rt_mrc", &mrc_bytes)?;
        assert_eq!(vol2.dims, vol.dims);
        let mut mol_fit = mol.clone();
        let fit = fit_model_in_map(&mut mol_fit, &vol2, 6);
        assert!(fit.correlation_after > 0.7);
        let iso = extract_isosurface_nodes(&vol2, 1.5, 400);
        assert!(!iso.is_empty());

        // OpenDX .dx grid parse test
        let dx = "object 1 class gridpositions counts 2 2 2\norigin 0 0 0\ndelta 1 0 0\ndelta 0 1 0\ndelta 0 0 1\nobject 3 class array type double rank 0 items 8 data follows\n1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0\n";
        let gdx = parse_dx_map("dx_test", dx)?;
        assert_eq!(gdx.dims, [2, 2, 2]);

        // MLP lipophilicity potential check
        let mlp = compute_per_atom_mlp(&mol, 8.0);
        assert_eq!(mlp.len(), mol.atoms.len());

        // Multi-pose V2000 SDF parsing test
        let sdf = "\
pose_alpha
  DockCraft

  2  1  0  0  0  0  0  0  0  0999 V2000
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
M  END
>  <Vina_kcal>
-8.45

$$$$
pose_beta
  DockCraft

  2  1  0  0  0  0  0  0  0  0999 V2000
    3.0000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.3000    1.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
>  <Vina_kcal>
-7.90

$$$$
";
        let poses = parse_sdf_poses(sdf)?;
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].title, "pose_alpha");
        assert_eq!(
            poses[0].properties.get("Vina_kcal").map(String::as_str),
            Some("-8.45")
        );
        assert_eq!(poses[0].molecule.bonds.len(), 1);
        assert_eq!(poses[0].molecule.bonds[0].order, 2);
        Ok(())
    }

    #[test]
    fn wave5_measurements_hbonds_clashes_axes_planes_pae_and_msa() -> Result<(), BioStructError> {
        let pdb = "\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 92.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 92.00           C
ATOM      3  C   ALA A   1       2.050   1.380   0.000  1.00 92.00           C
ATOM      4  O   ALA A   1       1.400   2.400   0.000  1.00 92.00           O
ATOM      5  N   GLY A   2       3.350   1.420   0.000  1.00 88.00           N
ATOM      6  CA  GLY A   2       4.100   2.680   0.000  1.00 88.00           C
ATOM      7  N   SER B   1       1.400   5.250   0.000  1.00 85.00           N
ATOM      8  CA  SER B   1       2.500   6.150   0.000  1.00 85.00           C
ATOM      9  CB  SER B   1       2.500   2.400   0.000  1.00 85.00           C
END
";
        let mol = parse_pdb("w5_meas", pdb)?;

        // Distance, Angle, Dihedral
        assert!((measure_distance([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) - 5.0).abs() < 1e-6);
        assert!(
            (measure_angle_deg([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]) - 90.0).abs()
                < 1e-4
        );
        let dih = measure_dihedral_deg(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        );
        assert!((dih.abs() - 90.0).abs() < 1e-4);

        // H-bond (A:ALA1.O at (1.4,2.4,0) to B:SER1.N at (1.4,5.25,0) => 2.85 A)
        let hbs = detect_hbonds(&mol, None, None, 3.5, 90.0);
        assert!(!hbs.is_empty(), "expected inter-chain H-bond near 2.85 A");

        // Steric clash (A:ALA1.O at (1.4,2.4,0) vs B:SER1.CB at (2.5,2.4,0) => 1.1 A < vdw sum 3.22 A)
        let clashes = detect_clashes(&mol, None, None, 0.6);
        assert!(
            !clashes.is_empty(),
            "expected severe clash between A:1.O and B:1.CB"
        );

        // Contacts
        let contacts = detect_contacts(&mol, None, None, 4.0, 50);
        assert!(!contacts.is_empty());

        // Axis & Plane PCA
        let line_pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.1],
            [8.0, 0.0, 0.0],
        ];
        let ax = compute_geometry_axis("helix1", &line_pts)?;
        assert!(ax.length_a > 7.5 && ax.linearity_ratio > 0.98);
        let plane_pts = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let pl = compute_best_fit_plane("ring1", &plane_pts)?;
        assert!(pl.normal[2].abs() > 0.99 && pl.rms_deviation_a < 1e-4);
        assert!((axis_angle_deg(ax.direction, pl.normal, true) - 90.0).abs() < 2.0);

        // Synchronized MSA
        let msa = build_synchronized_msa(&[("m1", &mol), ("m2", &mol)]);
        assert_eq!(msa.rows.len(), 2);
        assert!(msa.num_columns >= 2);
        assert!(msa.column_conservation.iter().all(|&c| c > 0.99));

        // AlphaFold / Boltz PAE JSON parser + structural proxy
        let pae_json = r#"[{"predicted_aligned_error":[[0.5,1.2,18.0,19.0],[1.2,0.5,17.5,18.5],[18.0,17.5,0.6,1.1],[19.0,18.5,1.1,0.6]]}]"#;
        let pae = parse_pae_json("af_test", pae_json)?;
        assert_eq!(pae.length, 4);
        assert!(pae.is_experimental_json && pae.max_pae >= 19.0);
        let proxy = estimate_pae_proxy_from_structure(&mol);
        assert!(proxy.length >= 2 && !proxy.is_experimental_json);
        Ok(())
    }

    #[test]
    fn wave5_cealign_zero_identity_and_conformational_morph() -> Result<(), BioStructError> {
        // Build a 16-residue synthetic curved helix in mol_a, and a 0%-sequence-identity
        // rotated+translated copy in mol_b to prove sequence-independent CE alignment!
        let mut pdb_a = String::new();
        let mut pdb_b = String::new();
        let mut pdb_bent = String::new();
        let aa_a = [
            "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY", "SER", "THR", "CYS",
            "TYR", "ASN", "GLN", "ASP",
        ];
        let aa_b = [
            "GLU", "LYS", "ARG", "HIS", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO",
            "GLY", "SER", "THR", "CYS",
        ];
        for i in 0..16usize {
            let t = i as f64;
            let x = t * 1.5;
            let y = 2.3 * (t * 1.745).cos();
            let z = 2.3 * (t * 1.745).sin();
            pdb_a.push_str(&format!(
                "ATOM  {:5}  CA  {} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 90.00           C\n",
                i + 1,
                aa_a[i],
                i + 1,
                x,
                y,
                z
            ));
            // Rotate 90 deg around Z and translate by (25, -14, 9) with 0% sequence identity!
            let rx = -y + 25.0;
            let ry = x - 14.0;
            let rz = z + 9.0;
            pdb_b.push_str(&format!(
                "ATOM  {:5}  CA  {} B{:4}    {:8.3}{:8.3}{:8.3}  1.00 85.00           C\n",
                i + 1,
                aa_b[i],
                i + 101,
                rx,
                ry,
                rz
            ));
            // Bent conformation for morph (second half swings in Y)
            let by = if i >= 8 { y + (i - 7) as f64 * 1.4 } else { y };
            pdb_bent.push_str(&format!(
                "ATOM  {:5}  CA  {} A{:4}    {:8.3}{:8.3}{:8.3}  1.00 90.00           C\n",
                i + 1,
                aa_a[i],
                i + 1,
                x,
                by,
                z
            ));
        }
        let mol_a = parse_pdb("helix_a", &pdb_a)?;
        let mut mol_b = parse_pdb("helix_b_0id", &pdb_b)?;
        let tf = cealign_superpose(&mut mol_b, 'B', &mol_a, 'A')?;
        assert!(
            tf.rmsd_a < 0.05 && tf.aligned_pairs >= 14,
            "expected near-exact sequence-independent cealign, got RMSD={} on {} atoms",
            tf.rmsd_a,
            tf.aligned_pairs
        );

        let mol_bent = parse_pdb("helix_bent", &pdb_bent)?;
        let morphed = morph_biomolecules(&mol_a, &mol_bent, 12)?;
        assert_eq!(morphed.frames.len(), 12);
        assert!(dist3(morphed.frames[0].coords[15], mol_a.atoms[15].coords) < 1e-6);
        assert!(dist3(morphed.frames[11].coords[15], morphed.frames[0].coords[15]) > 2.0);
        Ok(())
    }
}
