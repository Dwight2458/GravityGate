//! Build stamp.
//!
//! A stale binary is indistinguishable from a bug when the version string never
//! changes. That happened: a five-day-old `cargo install` produced a 404 that
//! looked like an upstream fault, because the binary predated model tier
//! resolution and sent `gemini-3.8-flash` where the upstream wants
//! `gemini-3.8-flash-medium`. Nothing in its output said it was old.
//!
//! Stamping the build time and revision into the binary makes that a one-line
//! diagnosis instead of an investigation.
//!
//! `rerun-if-changed` is deliberately not declared for the source tree: cargo
//! then reruns this script whenever the package is rebuilt, which is exactly when
//! the stamp should change. Declaring it would pin the stamp to `build.rs`'s own
//! mtime and let it go stale.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let epoch = build_epoch();
    println!("cargo::rustc-env=GRAVITYGATE_BUILD_EPOCH={epoch}");
    println!("cargo::rustc-env=GRAVITYGATE_BUILD_TIME={}", iso8601(epoch));
    println!("cargo::rustc-env=GRAVITYGATE_REVISION={}", revision());
}

/// Format an epoch as an ISO 8601 UTC timestamp.
///
/// Done here rather than at runtime so the binary needs no date handling at all,
/// and by arithmetic rather than by shelling out to `date`, which does not exist
/// in the same form on every host that might build this.
///
/// The day-to-civil conversion is Howard Hinnant's, which is exact for the whole
/// range and needs no lookup tables.
fn iso8601(epoch: u64) -> String {
    let days = (epoch / 86_400) as i64;
    let seconds = epoch % 86_400;
    let (hour, minute, second) = (seconds / 3600, (seconds % 3600) / 60, seconds % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// When this build was made, in seconds since the epoch.
///
/// `SOURCE_DATE_EPOCH` is honoured when set, so a reproducible build produces a
/// stable stamp rather than a different one on every invocation.
fn build_epoch() -> u64 {
    if let Ok(value) = std::env::var("SOURCE_DATE_EPOCH")
        && let Ok(seconds) = value.trim().parse()
    {
        return seconds;
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// The short revision, or a marker when there is not one.
///
/// A repository with no commits yet is normal during early development, and a
/// dirty tree matters more than a clean one when diagnosing a report.
fn revision() -> String {
    let short = git(&["rev-parse", "--short", "HEAD"]);
    let Some(short) = short else {
        return "unknown".into();
    };

    // Untracked build output would make every build look dirty, so only tracked
    // modifications are considered.
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .is_some_and(|status| !status.trim().is_empty());

    if dirty {
        format!("{short}-dirty")
    } else {
        short
    }
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}
