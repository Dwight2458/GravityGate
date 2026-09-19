//! Persistence for the account pool.
//!
//! Two properties matter here and neither is negotiable:
//!
//! - **No torn writes.** The pool is rewritten on every rate-limit event, so a
//!   crash mid-write would otherwise corrupt every credential at once. Writes go
//!   to a sibling temp file and are moved into place with `rename`, which is
//!   atomic on both Unix and Windows.
//! - **No lost updates.** Two processes editing the pool — a running gateway and
//!   an operator running `account add` — must not clobber each other. An
//!   advisory lock file serialises them, with staleness detection so a killed
//!   process cannot wedge the pool forever.
//!
//! Credentials are stored in plaintext, matching every reference implementation.
//! The mitigation is filesystem permissions: `0700` on the directory and `0600`
//! on the file. On Windows those bits are not applied; protection comes from the
//! user profile's ACL instead, which is why the file lives under `%APPDATA%`.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use crate::accounts::account::{AccountStorage, SCHEMA_VERSION};

/// How long to wait for the advisory lock before giving up.
const LOCK_TIMEOUT: Duration = Duration::from_secs(10);

/// A lock file older than this is assumed to be abandoned by a dead process.
const LOCK_STALE_AFTER: Duration = Duration::from_secs(30);

/// Poll interval while waiting for the lock.
const LOCK_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("could not read account store at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not write account store at {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Deliberately fail-closed: a pool we cannot parse must not be silently
    /// replaced with an empty one, which would look like "no accounts".
    #[error("account store at {path} is not valid JSON: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("account store at {path} has unsupported version {found} (this build supports {SCHEMA_VERSION})")]
    UnsupportedVersion { path: PathBuf, found: u32 },

    #[error("timed out waiting for the account store lock at {path}")]
    LockTimeout { path: PathBuf },
}

/// A held advisory lock, released on drop.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(
        target: &Path,
        timeout: Duration,
        stale_after: Duration,
    ) -> Result<Self, StoreError> {
        let path = lock_path(target);
        let deadline = Instant::now() + timeout;

        loop {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    // Record the owner for diagnostics; contents are advisory.
                    let _ = writeln!(file, "pid={}", std::process::id());
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if is_stale(&path, stale_after) {
                        tracing::warn!(path = %path.display(), "removing stale account store lock");
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(StoreError::LockTimeout { path });
                    }
                    std::thread::sleep(LOCK_POLL_INTERVAL);
                }
                Err(source) => {
                    return Err(StoreError::Write { path, source });
                }
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn lock_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_os_string();
    name.push(".lock");
    PathBuf::from(name)
}

/// Whether an existing lock file looks abandoned.
///
/// A missing file counts as stale, which also covers the race where the holder
/// released between our failed create and this check.
fn is_stale(path: &Path, stale_after: Duration) -> bool {
    match fs::metadata(path).and_then(|meta| meta.modified()) {
        Ok(modified) => SystemTime::now()
            .duration_since(modified)
            .map(|age| age > stale_after)
            .unwrap_or(false),
        Err(_) => true,
    }
}

/// The account pool, with its file location and in-process lock.
pub struct AccountStore {
    path: PathBuf,
    inner: Mutex<AccountStorage>,
    lock_timeout: Duration,
    lock_stale_after: Duration,
}

impl std::fmt::Debug for AccountStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Omit the pool contents: they hold refresh tokens, and `Account`'s
        // redacting Debug only protects the top level.
        f.debug_struct("AccountStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl AccountStore {
    /// Load the pool, treating a missing file as an empty pool.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();

        let storage = if path.exists() {
            let text = fs::read_to_string(&path).map_err(|source| StoreError::Read {
                path: path.clone(),
                source,
            })?;
            if text.trim().is_empty() {
                AccountStorage::default()
            } else {
                let mut parsed: AccountStorage =
                    serde_json::from_str(&text).map_err(|source| StoreError::Parse {
                        path: path.clone(),
                        source,
                    })?;
                if parsed.version > SCHEMA_VERSION {
                    return Err(StoreError::UnsupportedVersion {
                        path,
                        found: parsed.version,
                    });
                }
                parsed.version = SCHEMA_VERSION;
                parsed.clamp_indices();
                parsed
            }
        } else {
            AccountStorage::default()
        };

        Ok(Self {
            path,
            inner: Mutex::new(storage),
            lock_timeout: LOCK_TIMEOUT,
            lock_stale_after: LOCK_STALE_AFTER,
        })
    }

    /// Override the lock timing. Test-only: production values are tuned for a
    /// pool that is written rarely and read constantly.
    #[cfg(test)]
    fn with_lock_timing(mut self, timeout: Duration, stale_after: Duration) -> Self {
        self.lock_timeout = timeout;
        self.lock_stale_after = stale_after;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read the pool without holding the lock beyond the clone.
    pub fn snapshot(&self) -> AccountStorage {
        self.inner.lock().expect("account store poisoned").clone()
    }

    /// Apply a change and persist it.
    ///
    /// The closure returns `true` when the pool changed, which avoids rewriting
    /// the file on every read-only observation. Persisting happens while the
    /// in-process lock is held so two concurrent mutations cannot interleave
    /// their writes.
    pub fn mutate<F>(&self, change: F) -> Result<(), StoreError>
    where
        F: FnOnce(&mut AccountStorage) -> bool,
    {
        let mut guard = self.inner.lock().expect("account store poisoned");
        if !change(&mut guard) {
            return Ok(());
        }
        guard.clamp_indices();
        self.persist(&guard)
    }

    /// Write the pool to disk atomically, under the advisory lock.
    fn persist(&self, storage: &AccountStorage) -> Result<(), StoreError> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|source| StoreError::Write {
                path: parent.to_path_buf(),
                source,
            })?;
            restrict_directory(parent);
        }

        let _lock = FileLock::acquire(&self.path, self.lock_timeout, self.lock_stale_after)?;

        let serialised = serde_json::to_vec_pretty(storage).map_err(|source| StoreError::Parse {
            path: self.path.clone(),
            source,
        })?;

        // Same directory, so the rename stays within one filesystem.
        let temp_path = self.path.with_extension("json.tmp");
        {
            let mut file = fs::File::create(&temp_path).map_err(|source| StoreError::Write {
                path: temp_path.clone(),
                source,
            })?;
            file.write_all(&serialised)
                .map_err(|source| StoreError::Write {
                    path: temp_path.clone(),
                    source,
                })?;
            file.sync_all().map_err(|source| StoreError::Write {
                path: temp_path.clone(),
                source,
            })?;
        }
        restrict_file(&temp_path);

        fs::rename(&temp_path, &self.path).map_err(|source| StoreError::Write {
            path: self.path.clone(),
            source,
        })?;

        Ok(())
    }

    /// Re-read the file, discarding in-memory state. Used after an external
    /// process may have changed the pool.
    pub fn reload(&self) -> Result<(), StoreError> {
        let fresh = Self::load(self.path.clone())?;
        let mut guard = self.inner.lock().expect("account store poisoned");
        *guard = fresh.snapshot();
        Ok(())
    }
}

/// Restrict a directory to its owner. No-op on Windows, where protection comes
/// from the profile ACL.
fn restrict_directory(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// Restrict a file to its owner. No-op on Windows.
fn restrict_file(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accounts::account::Account;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gravitygate-test-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_file_loads_as_empty_pool() {
        let dir = temp_dir("missing");
        let store = AccountStore::load(dir.join("accounts.json")).unwrap();
        assert!(store.snapshot().is_empty());
        // Loading must not create the file.
        assert!(!dir.join("accounts.json").exists());
    }

    #[test]
    fn mutate_persists_and_reloads() {
        let dir = temp_dir("persist");
        let path = dir.join("accounts.json");

        let store = AccountStore::load(&path).unwrap();
        store
            .mutate(|storage| {
                storage.accounts.push(Account::new("1//token"));
                true
            })
            .unwrap();

        let reloaded = AccountStore::load(&path).unwrap();
        assert_eq!(reloaded.snapshot().accounts.len(), 1);
        assert_eq!(reloaded.snapshot().accounts[0].refresh_token, "1//token");
    }

    #[test]
    fn mutate_returning_false_does_not_write() {
        let dir = temp_dir("noop");
        let path = dir.join("accounts.json");
        let store = AccountStore::load(&path).unwrap();
        store.mutate(|_| false).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn corrupted_file_fails_closed_rather_than_resetting() {
        let dir = temp_dir("corrupt");
        let path = dir.join("accounts.json");
        fs::write(&path, "{ this is not json").unwrap();

        let error = AccountStore::load(&path).unwrap_err();
        assert!(matches!(error, StoreError::Parse { .. }));
        // The original content must survive: silently starting from an empty
        // pool would look like "all accounts vanished".
        assert!(fs::read_to_string(&path).unwrap().contains("not json"));
    }

    #[test]
    fn empty_file_loads_as_empty_pool() {
        let dir = temp_dir("empty");
        let path = dir.join("accounts.json");
        fs::write(&path, "   \n").unwrap();
        assert!(AccountStore::load(&path).unwrap().snapshot().is_empty());
    }

    #[test]
    fn future_schema_version_is_rejected() {
        let dir = temp_dir("future");
        let path = dir.join("accounts.json");
        fs::write(&path, r#"{"version": 99, "accounts": []}"#).unwrap();
        let error = AccountStore::load(&path).unwrap_err();
        assert!(matches!(error, StoreError::UnsupportedVersion { found: 99, .. }));
    }

    #[test]
    fn missing_version_is_rejected() {
        let dir = temp_dir("noversion");
        let path = dir.join("accounts.json");
        fs::write(&path, r#"{"accounts": []}"#).unwrap();
        assert!(matches!(
            AccountStore::load(&path).unwrap_err(),
            StoreError::Parse { .. }
        ));
    }

    #[test]
    fn no_temp_file_is_left_behind_after_a_write() {
        let dir = temp_dir("notemp");
        let path = dir.join("accounts.json");
        let store = AccountStore::load(&path).unwrap();
        store
            .mutate(|s| {
                s.accounts.push(Account::new("t"));
                true
            })
            .unwrap();

        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp") || name.ends_with(".lock"))
            .collect();
        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    }

    #[test]
    fn lock_is_released_after_a_write() {
        let dir = temp_dir("lockrelease");
        let path = dir.join("accounts.json");
        let store = AccountStore::load(&path).unwrap();
        store
            .mutate(|s| {
                s.accounts.push(Account::new("t"));
                true
            })
            .unwrap();
        assert!(!lock_path(&path).exists());
    }

    #[test]
    fn stale_lock_is_reclaimed() {
        let dir = temp_dir("stalelock");
        let path = dir.join("accounts.json");
        // A zero staleness window makes any existing lock reclaimable, which is
        // what an abandoned lock looks like after the timeout has passed.
        let store = AccountStore::load(&path)
            .unwrap()
            .with_lock_timing(Duration::from_millis(200), Duration::ZERO);

        fs::write(lock_path(&path), "pid=999999").unwrap();

        store
            .mutate(|s| {
                s.accounts.push(Account::new("t"));
                true
            })
            .expect("a stale lock must not block the write");
        assert_eq!(AccountStore::load(&path).unwrap().snapshot().accounts.len(), 1);
    }

    #[test]
    fn active_lock_times_out_rather_than_clobbering() {
        let dir = temp_dir("activelock");
        let path = dir.join("accounts.json");
        // A long staleness window means the lock below looks live, so the write
        // must refuse rather than silently overwrite another process's work.
        let store = AccountStore::load(&path)
            .unwrap()
            .with_lock_timing(Duration::from_millis(50), Duration::from_secs(300));

        fs::write(lock_path(&path), "pid=1").unwrap();

        let error = store
            .mutate(|s| {
                s.accounts.push(Account::new("t"));
                true
            })
            .unwrap_err();
        assert!(matches!(error, StoreError::LockTimeout { .. }));
        assert!(!path.exists(), "nothing should have been written");

        let _ = fs::remove_file(lock_path(&path));
    }

    #[test]
    fn reload_picks_up_external_changes() {
        let dir = temp_dir("reload");
        let path = dir.join("accounts.json");

        let store_a = AccountStore::load(&path).unwrap();
        store_a
            .mutate(|s| {
                s.accounts.push(Account::new("a"));
                true
            })
            .unwrap();

        let store_b = AccountStore::load(&path).unwrap();
        store_b
            .mutate(|s| {
                s.accounts.push(Account::new("b"));
                true
            })
            .unwrap();

        assert_eq!(store_a.snapshot().accounts.len(), 1, "stale view");
        store_a.reload().unwrap();
        assert_eq!(store_a.snapshot().accounts.len(), 2);
    }

    #[test]
    fn indices_are_clamped_on_load() {
        let dir = temp_dir("clamp");
        let path = dir.join("accounts.json");
        fs::write(
            &path,
            r#"{"version":1,"accounts":[],"active_index":5}"#,
        )
        .unwrap();
        let store = AccountStore::load(&path).unwrap();
        assert_eq!(store.snapshot().active_index, 0);
    }

    #[test]
    fn concurrent_mutations_do_not_lose_updates() {
        let dir = temp_dir("concurrent");
        let path = dir.join("accounts.json");
        let store = std::sync::Arc::new(AccountStore::load(&path).unwrap());

        let threads: Vec<_> = (0..8)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    store
                        .mutate(|s| {
                            s.accounts.push(Account::new(format!("token-{i}")));
                            true
                        })
                        .unwrap();
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        let final_count = AccountStore::load(&path).unwrap().snapshot().accounts.len();
        assert_eq!(final_count, 8);
    }

    #[cfg(unix)]
    #[test]
    fn file_is_owner_only_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("permissions");
        let path = dir.join("accounts.json");
        let store = AccountStore::load(&path).unwrap();
        store
            .mutate(|s| {
                s.accounts.push(Account::new("t"));
                true
            })
            .unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "credentials must not be world readable");
    }

}
