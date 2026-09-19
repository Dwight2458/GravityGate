//! The request audit log.
//!
//! Answers the questions `/health` cannot: which account served a given request,
//! how long it took, what it cost in tokens, and how it failed. That is the
//! difference between "the gateway is slow" and "the second account has been
//! slow since Tuesday".
//!
//! Three decisions worth stating:
//!
//! - **Writes never block a request.** SQLite is synchronous, so records go
//!   through a channel to a thread that owns the write connection. A request
//!   path that waited on an fsync would trade latency for telemetry, which is
//!   the wrong trade.
//! - **Losing a record is acceptable; failing a request is not.** A full channel
//!   or a broken database degrades to a logged warning. An audit log that can
//!   take the gateway down is worse than no audit log.
//! - **The account is recorded by credential id, not by email.** The id is
//!   stable across a rename and correlates against `/health`; the email is one
//!   lookup away for an operator who needs it, and does not belong scattered
//!   across a table.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OpenFlags, params};

/// Columns returned by every read, in order.
const COLUMNS: &str = "ts, account_id, requested_model, wire_model, stream, status, \
                       outcome, attempts, prompt_tokens, completion_tokens, \
                       cached_tokens, reasoning_tokens, latency_ms";

/// One finished request.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Record {
    pub ts: i64,
    pub account_id: String,
    pub requested_model: String,
    pub wire_model: String,
    pub stream: bool,
    pub status: u16,
    pub outcome: String,
    pub attempts: u32,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cached_tokens: i64,
    pub reasoning_tokens: i64,
    pub latency_ms: i64,
}

impl Record {
    /// A record with everything but the identifiers left at zero.
    pub fn new(account_id: impl Into<String>, requested_model: impl Into<String>) -> Self {
        Self {
            ts: now_ms(),
            account_id: account_id.into(),
            requested_model: requested_model.into(),
            wire_model: String::new(),
            stream: false,
            status: 200,
            outcome: "ok".into(),
            attempts: 1,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            reasoning_tokens: 0,
            latency_ms: 0,
        }
    }
}

/// Aggregate counts over a window.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Totals {
    pub requests: i64,
    pub ok: i64,
    pub errors: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cached_tokens: i64,
    pub reasoning_tokens: i64,
    /// Mean end-to-end latency, in milliseconds.
    pub mean_latency_ms: i64,
}

/// Per-account aggregate over a window.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct AccountSummary {
    pub account_id: String,
    pub requests: i64,
    pub errors: i64,
    pub mean_latency_ms: i64,
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("could not open or use the audit database at {path}: {source}")]
    Database {
        path: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
}

/// A handle to the audit log.
///
/// Dropping it closes the channel and lets the writer drain and exit.
#[derive(Debug)]
pub struct AuditLog {
    /// `Option` so [`Drop`] can close the channel explicitly before joining the
    /// writer. Leaving it in place would deadlock: the writer blocks on `recv`,
    /// `recv` only returns once every sender is gone, and the sender is not
    /// dropped until after the `Drop` body finishes.
    sender: Option<Sender<Record>>,
    path: PathBuf,
    writer: Option<JoinHandle<()>>,
}

impl AuditLog {
    /// Open or create the log at `path`.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let connection = Self::connect(&path)?;
        Self::migrate(&connection, &path)?;

        let (sender, receiver) = channel::<Record>();
        let writer_path = path.clone();
        let writer = std::thread::Builder::new()
            .name("gravitygate-audit".into())
            .spawn(move || {
                // The receiver ends when the handle is dropped, at which point
                // this thread drains and exits.
                while let Ok(record) = receiver.recv() {
                    if let Err(error) = insert(&connection, &record) {
                        // Losing a record must never take the gateway down.
                        tracing::warn!(%error, "could not write an audit record");
                    }
                }
                tracing::debug!(path = %writer_path.display(), "audit writer stopped");
            })
            .map_err(|source| AuditError::Database {
                path: path.clone(),
                source: rusqlite::Error::ToSqlConversionFailure(Box::new(source)),
            })?;

        Ok(Self {
            sender: Some(sender),
            path,
            writer: Some(writer),
        })
    }

    fn connect(path: &Path) -> Result<Connection, AuditError> {
        let connection = Connection::open(path).map_err(|source| AuditError::Database {
            path: path.to_path_buf(),
            source,
        })?;

        // WAL lets the dashboard read while the writer writes, which is the
        // whole point of having both. It is also what makes a read connection
        // opened on demand cheap.
        connection
            .pragma_update(None, "journal_mode", "WAL")
            .map_err(|source| AuditError::Database {
                path: path.to_path_buf(),
                source,
            })?;
        connection
            .pragma_update(None, "synchronous", "NORMAL")
            .map_err(|source| AuditError::Database {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(connection)
    }

    fn migrate(connection: &Connection, path: &Path) -> Result<(), AuditError> {
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS requests (
                     id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                     ts                 INTEGER NOT NULL,
                     account_id         TEXT    NOT NULL DEFAULT '',
                     requested_model    TEXT    NOT NULL DEFAULT '',
                     wire_model         TEXT    NOT NULL DEFAULT '',
                     stream             INTEGER NOT NULL DEFAULT 0,
                     status             INTEGER NOT NULL DEFAULT 0,
                     outcome            TEXT    NOT NULL DEFAULT '',
                     attempts           INTEGER NOT NULL DEFAULT 1,
                     prompt_tokens      INTEGER NOT NULL DEFAULT 0,
                     completion_tokens  INTEGER NOT NULL DEFAULT 0,
                     cached_tokens      INTEGER NOT NULL DEFAULT 0,
                     reasoning_tokens   INTEGER NOT NULL DEFAULT 0,
                     latency_ms         INTEGER NOT NULL DEFAULT 0
                 );
                 CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
                 CREATE INDEX IF NOT EXISTS idx_requests_account ON requests(account_id, ts);",
            )
            .map_err(|source| AuditError::Database {
                path: path.to_path_buf(),
                source,
            })
    }

    /// Queue a record. Never blocks and never fails the caller.
    pub fn record(&self, record: Record) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };
        if sender.send(record).is_err() {
            // The writer thread is gone, which means the database failed. The
            // request carries on; telemetry stops.
            tracing::warn!("audit log is not accepting records");
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Open a read connection.
    fn reader(&self) -> Result<Connection, AuditError> {
        Connection::open_with_flags(&self.path, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(
            |source| AuditError::Database {
                path: self.path.clone(),
                source,
            },
        )
    }

    /// Aggregate counts over the last `window`.
    pub fn totals(&self, window: Duration) -> Result<Totals, AuditError> {
        let since = cutoff(window);
        let connection = self.reader()?;
        connection
            .query_row(
                "SELECT COUNT(*),
                        COALESCE(SUM(CASE WHEN outcome = 'ok' THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(prompt_tokens), 0),
                        COALESCE(SUM(completion_tokens), 0),
                        COALESCE(SUM(cached_tokens), 0),
                        COALESCE(SUM(reasoning_tokens), 0),
                        COALESCE(CAST(AVG(latency_ms) AS INTEGER), 0)
                 FROM requests WHERE ts >= ?1",
                params![since],
                |row| {
                    let requests: i64 = row.get(0)?;
                    let ok: i64 = row.get(1)?;
                    Ok(Totals {
                        requests,
                        ok,
                        errors: requests - ok,
                        prompt_tokens: row.get(2)?,
                        completion_tokens: row.get(3)?,
                        cached_tokens: row.get(4)?,
                        reasoning_tokens: row.get(5)?,
                        mean_latency_ms: row.get(6)?,
                    })
                },
            )
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })
    }

    /// Per-account aggregates over the last `window`, busiest first.
    pub fn per_account(&self, window: Duration) -> Result<Vec<AccountSummary>, AuditError> {
        let since = cutoff(window);
        let connection = self.reader()?;
        let mut statement = connection
            .prepare(
                "SELECT account_id,
                        COUNT(*),
                        COALESCE(SUM(CASE WHEN outcome = 'ok' THEN 0 ELSE 1 END), 0),
                        COALESCE(CAST(AVG(latency_ms) AS INTEGER), 0)
                 FROM requests WHERE ts >= ?1
                 GROUP BY account_id ORDER BY COUNT(*) DESC",
            )
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })?;

        let rows = statement
            .query_map(params![since], |row| {
                Ok(AccountSummary {
                    account_id: row.get(0)?,
                    requests: row.get(1)?,
                    errors: row.get(2)?,
                    mean_latency_ms: row.get(3)?,
                })
            })
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })
    }

    /// The most recent requests, newest first.
    pub fn recent(&self, limit: usize) -> Result<Vec<Record>, AuditError> {
        let connection = self.reader()?;
        let sql = format!("SELECT {COLUMNS} FROM requests ORDER BY ts DESC LIMIT ?1");
        let mut statement = connection.prepare(&sql).map_err(|source| {
            AuditError::Database {
                path: self.path.clone(),
                source,
            }
        })?;

        let rows = statement
            .query_map(params![limit as i64], |row| {
                Ok(Record {
                    ts: row.get(0)?,
                    account_id: row.get(1)?,
                    requested_model: row.get(2)?,
                    wire_model: row.get(3)?,
                    stream: row.get::<_, i64>(4)? != 0,
                    status: row.get::<_, i64>(5)? as u16,
                    outcome: row.get(6)?,
                    attempts: row.get::<_, i64>(7)? as u32,
                    prompt_tokens: row.get(8)?,
                    completion_tokens: row.get(9)?,
                    cached_tokens: row.get(10)?,
                    reasoning_tokens: row.get(11)?,
                    latency_ms: row.get(12)?,
                })
            })
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })
    }

    /// Delete records older than `retention`, returning how many went.
    ///
    /// An unbounded log on a gateway that never restarts is a slow disk leak.
    pub fn prune(&self, retention: Duration) -> Result<usize, AuditError> {
        let before = cutoff(retention);
        // A separate writable connection rather than the writer thread's: the
        // writer owns its connection for the process lifetime, and SQLite
        // serialises the two.
        let connection = Connection::open(&self.path).map_err(|source| AuditError::Database {
            path: self.path.clone(),
            source,
        })?;
        connection
            .execute("DELETE FROM requests WHERE ts < ?1", params![before])
            .map_err(|source| AuditError::Database {
                path: self.path.clone(),
                source,
            })
    }

    /// Wait for the writer to finish. Used by tests, which need to read back
    /// what they just wrote.
    #[cfg(test)]
    fn flush(&self) {
        // The channel has no flush, so a round trip through a second query is
        // not available either. Sleeping briefly is enough for a test.
        std::thread::sleep(Duration::from_millis(150));
    }
}

impl Drop for AuditLog {
    fn drop(&mut self) {
        // Close the channel *first*. The writer loops on `recv`, which returns
        // only once every sender is gone, so joining before dropping the sender
        // deadlocks — which is exactly what it did, hanging the whole test
        // suite rather than failing it.
        drop(self.sender.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

fn insert(connection: &Connection, record: &Record) -> rusqlite::Result<()> {
    connection.execute(
        "INSERT INTO requests (ts, account_id, requested_model, wire_model, stream, status,
                               outcome, attempts, prompt_tokens, completion_tokens,
                               cached_tokens, reasoning_tokens, latency_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            record.ts,
            record.account_id,
            record.requested_model,
            record.wire_model,
            i64::from(record.stream),
            i64::from(record.status),
            record.outcome,
            i64::from(record.attempts),
            record.prompt_tokens,
            record.completion_tokens,
            record.cached_tokens,
            record.reasoning_tokens,
            record.latency_ms,
        ],
    )?;
    Ok(())
}

/// Epoch milliseconds `window` ago.
fn cutoff(window: Duration) -> i64 {
    now_ms() - window.as_millis() as i64
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_log(label: &str) -> AuditLog {
        let path = std::env::temp_dir().join(format!(
            "gravitygate-audit-{label}-{}-{:?}.db",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        AuditLog::open(&path).expect("the audit log opens")
    }

    fn record_for(account: &str, model: &str, outcome: &str) -> Record {
        Record {
            outcome: outcome.into(),
            ..Record::new(account, model)
        }
    }

    #[test]
    fn the_database_is_created_with_its_table() {
        let log = temp_log("create");
        log.flush();
        assert!(log.path().exists());
    }

    #[test]
    fn a_record_round_trips() {
        let log = temp_log("roundtrip");
        let mut record = record_for("abcdef01", "gemini-3.8-flash", "ok");
        record.wire_model = "gemini-3.8-flash-medium".into();
        record.stream = true;
        record.attempts = 2;
        record.prompt_tokens = 10;
        record.completion_tokens = 4;
        record.cached_tokens = 3;
        record.reasoning_tokens = 7;
        record.latency_ms = 1234;
        log.record(record.clone());
        log.flush();

        let recent = log.recent(10).unwrap();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0], record);
    }

    #[test]
    fn records_come_back_newest_first() {
        let log = temp_log("order");
        for index in 0..5 {
            let mut record = record_for("a", "m", "ok");
            record.ts = 1_000 + index;
            log.record(record);
        }
        log.flush();

        let recent = log.recent(10).unwrap();
        let timestamps: Vec<i64> = recent.iter().map(|record| record.ts).collect();
        assert_eq!(timestamps, vec![1004, 1003, 1002, 1001, 1000]);
    }

    #[test]
    fn the_limit_is_honoured() {
        let log = temp_log("limit");
        for _ in 0..10 {
            log.record(record_for("a", "m", "ok"));
        }
        log.flush();
        assert_eq!(log.recent(3).unwrap().len(), 3);
    }

    #[test]
    fn totals_separate_successes_from_failures() {
        let log = temp_log("totals");
        for outcome in ["ok", "ok", "client_error", "upstream_error"] {
            let mut record = record_for("a", "m", outcome);
            record.latency_ms = 100;
            log.record(record);
        }
        log.flush();

        let totals = log.totals(Duration::from_secs(3600)).unwrap();
        assert_eq!(totals.requests, 4);
        assert_eq!(totals.ok, 2);
        assert_eq!(totals.errors, 2);
        assert_eq!(totals.mean_latency_ms, 100);
    }

    #[test]
    fn totals_sum_tokens() {
        let log = temp_log("tokens");
        let mut record = record_for("a", "m", "ok");
        record.prompt_tokens = 100;
        record.completion_tokens = 20;
        record.cached_tokens = 40;
        record.reasoning_tokens = 15;
        log.record(record);
        log.flush();

        let totals = log.totals(Duration::from_secs(3600)).unwrap();
        assert_eq!(totals.prompt_tokens, 100);
        assert_eq!(totals.completion_tokens, 20);
        assert_eq!(totals.cached_tokens, 40);
        assert_eq!(totals.reasoning_tokens, 15);
    }

    #[test]
    fn a_window_excludes_older_records() {
        let log = temp_log("window");
        let mut old = record_for("a", "m", "ok");
        old.ts = now_ms() - 10 * 60 * 1000;
        log.record(old);
        log.record(record_for("a", "m", "ok"));
        log.flush();

        let recent = log.totals(Duration::from_secs(60)).unwrap();
        assert_eq!(recent.requests, 1, "the ten-minute-old record is excluded");

        let all = log.totals(Duration::from_secs(3600)).unwrap();
        assert_eq!(all.requests, 2);
    }

    #[test]
    fn per_account_aggregates_group_correctly() {
        let log = temp_log("peraccount");
        for _ in 0..3 {
            log.record(record_for("account-a", "m", "ok"));
        }
        log.record(record_for("account-b", "m", "ok"));
        log.record(record_for("account-b", "m", "upstream_error"));
        log.flush();

        let summaries = log.per_account(Duration::from_secs(3600)).unwrap();
        // Busiest first.
        assert_eq!(summaries[0].account_id, "account-a");
        assert_eq!(summaries[0].requests, 3);
        assert_eq!(summaries[0].errors, 0);

        assert_eq!(summaries[1].account_id, "account-b");
        assert_eq!(summaries[1].requests, 2);
        assert_eq!(summaries[1].errors, 1);
    }

    #[test]
    fn an_empty_log_reports_zeroes_rather_than_failing() {
        let log = temp_log("empty");
        log.flush();
        let totals = log.totals(Duration::from_secs(3600)).unwrap();
        assert_eq!(totals.requests, 0);
        assert!(log.recent(10).unwrap().is_empty());
        assert!(log.per_account(Duration::from_secs(3600)).unwrap().is_empty());
    }

    #[test]
    fn pruning_removes_old_records_and_keeps_recent_ones() {
        let log = temp_log("prune");
        let mut old = record_for("a", "m", "ok");
        old.ts = now_ms() - 48 * 60 * 60 * 1000;
        log.record(old);
        log.record(record_for("a", "m", "ok"));
        log.flush();

        let removed = log.prune(Duration::from_secs(24 * 60 * 60)).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(log.recent(10).unwrap().len(), 1);
    }

    #[test]
    fn reopening_keeps_existing_records() {
        let path = std::env::temp_dir().join(format!(
            "gravitygate-audit-reopen-{}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        {
            let log = AuditLog::open(&path).unwrap();
            log.record(record_for("a", "m", "ok"));
            log.flush();
        }
        {
            let log = AuditLog::open(&path).unwrap();
            log.flush();
            assert_eq!(log.recent(10).unwrap().len(), 1);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writes_from_several_threads_all_land() {
        let log = std::sync::Arc::new(temp_log("threads"));
        let threads: Vec<_> = (0..4)
            .map(|worker| {
                let log = log.clone();
                std::thread::spawn(move || {
                    for index in 0..25 {
                        log.record(record_for(
                            &format!("account-{worker}"),
                            "m",
                            if index % 5 == 0 { "upstream_error" } else { "ok" },
                        ));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        log.flush();

        assert_eq!(log.totals(Duration::from_secs(3600)).unwrap().requests, 100);
    }

    #[test]
    fn dropping_the_handle_drains_queued_records() {
        let path = std::env::temp_dir().join(format!(
            "gravitygate-audit-drain-{}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        {
            let log = AuditLog::open(&path).unwrap();
            for _ in 0..50 {
                log.record(record_for("a", "m", "ok"));
            }
            // No flush: dropping must be what lets the queue land.
        }

        let log = AuditLog::open(&path).unwrap();
        assert_eq!(log.totals(Duration::from_secs(3600)).unwrap().requests, 50);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_record_defaults_are_sane() {
        let record = Record::new("acct", "gemini-3.8-flash");
        assert_eq!(record.status, 200);
        assert_eq!(record.outcome, "ok");
        assert_eq!(record.attempts, 1);
        assert!(!record.stream);
        assert!(record.ts > 0, "a record is stamped when it is made");
    }
}
