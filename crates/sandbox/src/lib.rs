#![forbid(unsafe_code)]

//! `craftbio-sandbox` — Kernel-confined `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`
//! with `O_NOFOLLOW` dirfd-descent fallback shared across all CraftBio applications.

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Component, Path, PathBuf};

use rustix::fs::{Mode, OFlags, ResolveFlags, fstat, mkdirat, openat, openat2};
use rustix::io::Errno;
use thiserror::Error;

/// Default maximum readable biomolecular structure / trajectory file size (64 MiB).
pub const DEFAULT_MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("path rejected: {0}")]
    RejectedPath(String),
    #[error("sandbox root not configured for {0}")]
    RootUnconfigured(&'static str),
    #[error("file exceeds byte cap ({size} > {cap})")]
    SizeLimitExceeded { size: u64, cap: u64 },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("OS confinement error: {0}")]
    Os(String),
}

#[derive(Debug, Clone)]
pub struct PathSandbox {
    read_root: Option<PathBuf>,
    write_root: Option<PathBuf>,
    max_read_bytes: u64,
}

impl PathSandbox {
    pub fn new(
        read_root: Option<PathBuf>,
        write_root: Option<PathBuf>,
        max_read_bytes: u64,
    ) -> Result<Self, SandboxError> {
        let rr = read_root.map(|p| std::fs::canonicalize(&p)).transpose()?;
        let wr = write_root.map(|p| std::fs::canonicalize(&p)).transpose()?;
        Ok(Self {
            read_root: rr,
            write_root: wr,
            max_read_bytes,
        })
    }

    #[must_use]
    pub fn read_root(&self) -> Option<&Path> {
        self.read_root.as_deref()
    }

    #[must_use]
    pub fn write_root(&self) -> Option<&Path> {
        self.write_root.as_deref()
    }

    fn relative_segments<'a>(
        &self,
        root: &Path,
        candidate: &'a Path,
    ) -> Result<Vec<&'a std::ffi::OsStr>, SandboxError> {
        let rel = if candidate.is_absolute() {
            candidate.strip_prefix(root).map_err(|_| {
                SandboxError::RejectedPath(format!(
                    "path '{}' is outside sandbox root '{}'",
                    candidate.display(),
                    root.display()
                ))
            })?
        } else {
            candidate
        };

        let mut segs = Vec::new();
        for comp in rel.components() {
            match comp {
                Component::Normal(s) => {
                    if s.to_string_lossy().contains('\0') {
                        return Err(SandboxError::RejectedPath("NUL byte in path".to_string()));
                    }
                    segs.push(s);
                }
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(SandboxError::RejectedPath(format!(
                        "disallowed path traversal component in '{}'",
                        candidate.display()
                    )));
                }
            }
        }
        if segs.is_empty() {
            return Err(SandboxError::RejectedPath(
                "empty relative path target".to_string(),
            ));
        }
        Ok(segs)
    }

    fn open_root_fd(root: &Path) -> Result<OwnedFd, SandboxError> {
        openat(
            rustix::fs::CWD,
            root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| SandboxError::Os(format!("open root '{}': {e}", root.display())))
    }

    pub fn read_bytes(&self, candidate: &Path) -> Result<Vec<u8>, SandboxError> {
        let root = self
            .read_root
            .as_deref()
            .ok_or(SandboxError::RootUnconfigured("read"))?;
        let segs = self.relative_segments(root, candidate)?;
        let root_fd = Self::open_root_fd(root)?;
        let rel_joined: PathBuf = segs.iter().collect();
        let oflags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW;

        let fd = match openat2(
            root_fd.as_fd(),
            &rel_joined,
            oflags,
            Mode::empty(),
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
        ) {
            Ok(fd) => fd,
            Err(Errno::NOSYS) => Self::open_read_dirfd_descent(root_fd, &segs)?,
            Err(e) => return Err(SandboxError::Os(format!("openat2 read blocked: {e}"))),
        };

        let st = fstat(fd.as_fd()).map_err(|e| SandboxError::Os(format!("fstat failed: {e}")))?;
        let size = st.st_size.max(0) as u64;
        if size > self.max_read_bytes {
            return Err(SandboxError::SizeLimitExceeded {
                size,
                cap: self.max_read_bytes,
            });
        }
        let file = File::from(fd);
        let mut buf = Vec::with_capacity(size.min(self.max_read_bytes) as usize);
        file.take(self.max_read_bytes + 1).read_to_end(&mut buf)?;
        if buf.len() as u64 > self.max_read_bytes {
            return Err(SandboxError::SizeLimitExceeded {
                size: buf.len() as u64,
                cap: self.max_read_bytes,
            });
        }
        Ok(buf)
    }

    pub fn read_string(&self, candidate: &Path) -> Result<String, SandboxError> {
        let bytes = self.read_bytes(candidate)?;
        String::from_utf8(bytes)
            .map_err(|e| SandboxError::RejectedPath(format!("non-UTF-8 text file: {e}")))
    }

    pub fn write_bytes(&self, candidate: &Path, data: &[u8]) -> Result<PathBuf, SandboxError> {
        let root = self
            .write_root
            .as_deref()
            .ok_or(SandboxError::RootUnconfigured("write"))?;
        let segs = self.relative_segments(root, candidate)?;
        let root_fd = Self::open_root_fd(root)?;
        let last = segs.len() - 1;

        let parent_fd = Self::ensure_dir_chain(root_fd, &segs[..last])?;
        let leaf_fd = match openat2(
            parent_fd.as_fd(),
            segs[last],
            OFlags::WRONLY | OFlags::CREATE | OFlags::TRUNC | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o644),
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
        ) {
            Ok(fd) => fd,
            Err(Errno::NOSYS) => openat(
                parent_fd.as_fd(),
                segs[last],
                OFlags::WRONLY
                    | OFlags::CREATE
                    | OFlags::TRUNC
                    | OFlags::NOFOLLOW
                    | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o644),
            )
            .map_err(|e| SandboxError::Os(format!("write openat failed: {e}")))?,
            Err(e) => return Err(SandboxError::Os(format!("openat2 write blocked: {e}"))),
        };

        let mut file = File::from(leaf_fd);
        file.write_all(data)?;
        file.flush()?;

        let mut resolved = root.to_path_buf();
        for s in segs {
            resolved.push(s);
        }
        Ok(resolved)
    }

    pub fn write_string(&self, candidate: &Path, text: &str) -> Result<PathBuf, SandboxError> {
        self.write_bytes(candidate, text.as_bytes())
    }

    fn open_read_dirfd_descent(
        mut dir_fd: OwnedFd,
        segs: &[&std::ffi::OsStr],
    ) -> Result<OwnedFd, SandboxError> {
        let last = segs.len() - 1;
        for seg in &segs[..last] {
            dir_fd = openat(
                dir_fd.as_fd(),
                *seg,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| SandboxError::Os(format!("dirfd descent failed: {e}")))?;
        }
        openat(
            dir_fd.as_fd(),
            segs[last],
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| SandboxError::Os(format!("leaf openat failed: {e}")))
    }

    fn ensure_dir_chain(
        mut dir_fd: OwnedFd,
        parents: &[&std::ffi::OsStr],
    ) -> Result<OwnedFd, SandboxError> {
        for seg in parents {
            match mkdirat(dir_fd.as_fd(), *seg, Mode::from_raw_mode(0o755)) {
                Ok(()) | Err(Errno::EXIST) => {}
                Err(e) => return Err(SandboxError::Os(format!("mkdirat failed: {e}"))),
            }
            dir_fd = openat(
                dir_fd.as_fd(),
                *seg,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| SandboxError::Os(format!("dirfd descend failed: {e}")))?;
        }
        Ok(dir_fd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn sandbox_enforces_openat2_no_symlinks_and_no_traversal()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("craftbio_sbx_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("root");
        let outside = base.join("outside");
        std::fs::create_dir_all(&root)?;
        std::fs::create_dir_all(&outside)?;
        std::fs::write(root.join("ok.pdb"), b"HEADER TEST\n")?;
        std::fs::write(outside.join("secret.txt"), b"SECRET")?;
        symlink(outside.join("secret.txt"), root.join("escape.txt"))?;

        let sbx = PathSandbox::new(
            Some(root.clone()),
            Some(root.clone()),
            DEFAULT_MAX_FILE_BYTES,
        )?;
        assert!(sbx.read_string(Path::new("ok.pdb"))?.contains("HEADER"));
        assert!(sbx.read_string(Path::new("../outside/secret.txt")).is_err());
        assert!(sbx.read_string(Path::new("escape.txt")).is_err());
        assert!(
            sbx.write_string(Path::new("../outside/bad.txt"), "x")
                .is_err()
        );

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}
