# CraftBio Shared Libraries (`craftbio-libs`)

Shared pure-Rust bio-chemical foundation libraries for the **CraftBio** suite (`BiomolCraft`, `BioCraft`, `ChemCraft`, `AdmetCraft`, `DockCraft`), aligned with [storytold/craft-libs](https://github.com/storytold/craft-libs) and [storytold/craft-contrib](https://github.com/storytold/craft-contrib).

## Crates
- **`craftbio-sandbox`**: Kernel-confined `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` + `O_NOFOLLOW` `PathSandbox`.
- **`craftbio-struct`**: Universal biomolecular structure IR (Proteins, Antibodies, Enzymes, **ncRNA/mRNA/DNA**, Ligands, Ions, Waters, Trajectories), PDB/mmCIF/FASTA parsers, DSSP, Shrake-Rupley SASA, 3D SAP & ESP, Kabsch/MatchMaker alignment, and PyMOL/ChimeraX selection algebra.
- **`craftbio-scene`**: `.craftscene.json` interchange, provenance/evidence graph, and **Craft Link v1** inter-app IPC over `storytold/craft-libs`'s `craft-ipc`.
- **`craftbio-viewport`**: Pure-Rust `wgpu 24 + WGSL` biomolecular 3D renderer (GPU ray-cast impostors, DSSP protein ribbons, RNA/ncRNA ribbons + base-pair rungs, SES surfaces, SSAO, silhouettes) + deterministic headless 4K PNG renderer.
- **`craftbio-report`**: XSS-safe HTML/XML escaping, SVG helpers, and offline Mol\* asset resolver.

## License
Dual-licensed under `MIT` or `Apache-2.0`.
