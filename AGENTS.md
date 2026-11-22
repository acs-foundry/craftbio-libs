# AGENTS.md — CraftBio Rust House Rules (Aligned with storytold/craft-contrib)

1. **Never crash.** Non-test production code forbids `unsafe` (`#![forbid(unsafe_code)]`) and denies `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::unimplemented`, `clippy::todo`, `clippy::unreachable`. Bound allocations and recursion; treat all external molecular/script input as hostile.
2. **Clean room.** Implement algorithms from published scientific specifications and observed behaviour. Never copy code or proprietary assets from PyMOL, ChimeraX, or Schrödinger.
3. **Rust only for native apps & libraries.** Zero Electron, Tauri, or handwritten JavaScript shells in native desktop binaries.
4. **Scientific honesty.** Clearly separate experimental crystal/cryo-EM data, computational predictions, and MD trajectories in provenance records.
5. **Reproducible builds.** Commit `Cargo.lock` and verify with `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace`.
