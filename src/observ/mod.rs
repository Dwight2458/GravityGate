//! Observability: metrics, the audit log, and the dashboard they feed.

pub mod audit;
pub mod metrics;

/// Everything the server needs to observe itself.
///
/// Held together rather than separately because they are configured together and
/// because both are optional: a deployment that wants neither should pay nothing
/// for them.
#[derive(Debug)]
pub struct Observability {
    /// `None` when a recorder is already installed, which happens when more than
    /// one engine exists in a process.
    pub metrics: Option<metrics::Metrics>,
    /// `None` when the audit log is disabled or could not be opened.
    pub audit: Option<audit::AuditLog>,
}

impl Observability {
    /// Set up observability from configuration.
    ///
    /// Neither component is allowed to be fatal. A gateway that refuses to start
    /// because it cannot write a log file is a worse outcome than one running
    /// without telemetry, and the operator gets a loud warning either way.
    pub fn setup(config: &crate::config::Config) -> Self {
        let metrics = if config.metrics.enabled {
            match metrics::Metrics::install() {
                Ok(metrics) => Some(metrics),
                Err(error) => {
                    tracing::warn!(%error, "metrics are unavailable");
                    None
                }
            }
        } else {
            None
        };

        let audit = if config.audit.enabled {
            let path = config.audit.resolved_path();
            match audit::AuditLog::open(&path) {
                Ok(log) => {
                    tracing::info!(path = %log.path().display(), "audit log open");
                    Some(log)
                }
                Err(error) => {
                    tracing::warn!(%error, "audit log is unavailable");
                    None
                }
            }
        } else {
            None
        };

        Self { metrics, audit }
    }

    /// Render metrics, when enabled.
    pub fn render_metrics(&self) -> Option<String> {
        self.metrics.as_ref().map(metrics::Metrics::render)
    }

    pub fn audit(&self) -> Option<&audit::AuditLog> {
        self.audit.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observability_can_be_disabled_entirely() {
        let mut config = crate::config::Config::default();
        config.metrics.enabled = false;
        config.audit.enabled = false;

        let observability = Observability::setup(&config);
        assert!(observability.metrics.is_none());
        assert!(observability.audit.is_none());
        assert!(observability.render_metrics().is_none());
    }

    #[test]
    fn a_missing_audit_directory_is_created() {
        let mut config = crate::config::Config::default();
        config.metrics.enabled = false;
        config.audit.path = Some(
            std::env::temp_dir()
                .join(format!("gg-observ-{}", std::process::id()))
                .join("nested")
                .join("audit.db")
                .to_string_lossy()
                .into_owned(),
        );

        let observability = Observability::setup(&config);
        let log = observability.audit().expect("the log opens");
        assert!(log.path().exists());

        let _ = std::fs::remove_dir_all(
            std::env::temp_dir().join(format!("gg-observ-{}", std::process::id())),
        );
    }
}
