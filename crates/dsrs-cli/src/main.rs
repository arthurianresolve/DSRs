//! The `dsrs` binary: argument parsing + exit codes over the library
//! functions in [`dsrs_cli`]. Keep logic out of here — tests exercise the
//! library directly.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dsrs_cli::fmt::FmtOutcome;
use dsrs_cli::serve::ServeConfig;

#[derive(Parser)]
#[command(
    name = "dsrs",
    version,
    about = "DSRs .dsrs program toolchain: check, fmt, serve (RFC 0002 IR-7)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse + validate a .dsrs artifact; errors carry line/column positions.
    Check {
        /// Path to the .dsrs artifact.
        program: PathBuf,
    },
    /// Print (or rewrite) the canonical form of a .dsrs artifact.
    Fmt {
        /// Path to the .dsrs artifact.
        program: PathBuf,
        /// Rewrite the file in place instead of printing to stdout.
        #[arg(long)]
        write: bool,
        /// Explicitly allow model URL redaction when rewriting the file.
        #[arg(long, requires = "write")]
        redact_model_urls: bool,
    },
    /// Serve a .dsrs program over loopback HTTP. Use a TLS reverse proxy for
    /// remote access. Set DSRS_AUTH_TOKEN for a stable bearer token.
    Serve {
        /// Path to the .dsrs artifact.
        program: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Overlay JSON in the named form ({"<param path>": <value>, ...}).
        #[arg(long)]
        overlay: Option<PathBuf>,
        /// Capability grant (repeatable): --allow net:search --allow fs:read.
        #[arg(long = "allow", value_name = "CAP")]
        allow: Vec<String>,
        /// Permit an exact model base_url from the artifact (repeatable).
        #[arg(long = "allow-model-base-url", value_name = "URL")]
        allow_model_base_urls: Vec<String>,
        /// Bind a model to an endpoint URL held in an environment variable.
        #[arg(long = "model-base-url-env", value_name = "MODEL=ENV_VAR", value_parser = parse_model_base_url_env)]
        model_base_url_envs: Vec<(String, String)>,
    },
}

fn parse_model_base_url_env(value: &str) -> Result<(String, String), String> {
    match value.split_once('=') {
        Some((model, var)) if !model.is_empty() && !var.is_empty() => {
            Ok((model.to_string(), var.to_string()))
        }
        _ => Err("expected MODEL=ENV_VAR".to_string()),
    }
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Cmd::Check { program } => match dsrs_cli::check::check_file(&program) {
            Ok(report) => {
                println!("{report}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Cmd::Fmt {
            program,
            write,
            redact_model_urls,
        } => match dsrs_cli::fmt::fmt_file_with_options(&program, write, redact_model_urls) {
            Ok(FmtOutcome::Canonical(text)) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Ok(FmtOutcome::Rewrote) => {
                eprintln!("formatted `{}`", program.display());
                ExitCode::SUCCESS
            }
            Ok(FmtOutcome::Unchanged) => {
                eprintln!("`{}` already canonical", program.display());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err:#}");
                ExitCode::FAILURE
            }
        },
        Cmd::Serve {
            program,
            host,
            port,
            overlay,
            allow,
            allow_model_base_urls,
            model_base_url_envs,
        } => {
            let config = ServeConfig {
                program,
                overlay,
                allow,
                allowed_model_base_urls: allow_model_base_urls,
                auth_token: std::env::var("DSRS_AUTH_TOKEN").ok(),
            };
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(err) => {
                    eprintln!("failed to start async runtime: {err}");
                    return ExitCode::FAILURE;
                }
            };
            match runtime.block_on(dsrs_cli::serve::serve_with_model_base_url_envs(
                &config,
                &host,
                port,
                &model_base_url_envs,
            )) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("{err:#}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
