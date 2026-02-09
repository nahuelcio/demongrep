use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub async fn run() -> Result<()> {
    println!("Checking demongrep installation...");

    let mut failures: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    check_binary(&mut failures)?;
    check_path_conflicts(&mut warnings)?;
    check_local_index(&mut warnings)?;
    check_agent_config(
        "Claude Code",
        get_claude_config_path()?,
        &mut failures,
        &mut warnings,
    )?;
    check_agent_config(
        "Codex",
        get_codex_config_path()?,
        &mut failures,
        &mut warnings,
    )?;
    check_agent_config(
        "OpenCode",
        get_opencode_config_path()?,
        &mut failures,
        &mut warnings,
    )?;

    if !warnings.is_empty() {
        println!("Warnings:");
        for warning in &warnings {
            println!("  - {}", warning);
        }
    }

    if !failures.is_empty() {
        println!("Failures:");
        for failure in &failures {
            println!("  - {}", failure);
        }
        bail!("doctor found {} blocking issue(s)", failures.len());
    }

    println!("All checks passed.");
    Ok(())
}

fn check_binary(failures: &mut Vec<String>) -> Result<()> {
    let binary = env::current_exe().context("Failed to locate current executable")?;

    let meta = fs::metadata(&binary)
        .with_context(|| format!("Failed to read binary metadata: {}", binary.display()))?;

    if !meta.is_file() {
        failures.push(format!("Binary path is not a file: {}", binary.display()));
        return Ok(());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = meta.permissions().mode();
        if mode & 0o111 == 0 {
            failures.push(format!("Binary is not executable: {}", binary.display()));
        }
    }

    Ok(())
}

fn check_path_conflicts(warnings: &mut Vec<String>) -> Result<()> {
    let path = env::var("PATH").unwrap_or_default();
    if path.is_empty() {
        return Ok(());
    }

    let mut matches: Vec<PathBuf> = Vec::new();
    let mut seen = HashSet::new();
    for segment in env::split_paths(&path) {
        if segment.as_os_str().is_empty() {
            continue;
        }
        let candidate = segment.join("demongrep");
        if candidate.is_file() {
            let key = candidate.to_string_lossy().to_string();
            if !seen.insert(key) {
                continue;
            }
            matches.push(candidate);
        }
    }

    if matches.len() <= 1 {
        return Ok(());
    }

    let listed = matches
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    warnings.push(format!(
        "Multiple demongrep binaries detected in PATH: {}. Keep only one first in PATH to avoid version skew.",
        listed
    ));

    Ok(())
}

fn check_local_index(warnings: &mut Vec<String>) -> Result<()> {
    let cwd = env::current_dir().context("Failed to determine current directory")?;
    let local_index = cwd.join(".demongrep.db");

    if !local_index.exists() {
        warnings.push(format!(
            "No local index found at {} (run `demongrep index` in your project)",
            local_index.display()
        ));
    }

    Ok(())
}

fn get_claude_config_path() -> Result<PathBuf> {
    if cfg!(target_os = "windows") {
        let appdata =
            env::var("APPDATA").context("APPDATA environment variable not set on Windows")?;
        Ok(PathBuf::from(appdata).join("Claude").join("settings.json"))
    } else {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
        Ok(home.join(".config").join("claude-code").join("config.json"))
    }
}

fn get_codex_config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    Ok(home.join(".codex").join("config.json"))
}

fn get_opencode_config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    Ok(home.join(".config").join("opencode").join("config.json"))
}

fn check_agent_config(
    name: &str,
    config_path: PathBuf,
    failures: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !config_path.exists() {
        warnings.push(format!(
            "{} config not found at {}",
            name,
            config_path.display()
        ));
        return Ok(());
    }

    let content = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read {} config", name))?;

    let parsed: Value = serde_json::from_str(&content).with_context(|| {
        format!(
            "Failed to parse {} config as JSON: {}",
            name,
            config_path.display()
        )
    })?;

    if let Some(mcp_servers) = parsed.get("mcpServers") {
        if !mcp_servers.is_object() {
            failures.push(format!(
                "{} config has non-object `mcpServers`: {}",
                name,
                config_path.display()
            ));
            return Ok(());
        }

        let demongrep = &parsed["mcpServers"]["demongrep"];
        if demongrep.is_null() {
            warnings.push(format!("{} has no `mcpServers.demongrep` entry", name));
            return Ok(());
        }

        validate_demongrep_legacy_entry(name, &config_path, demongrep, failures);
        return Ok(());
    }

    if let Some(mcp) = parsed.get("mcp") {
        if !mcp.is_object() {
            failures.push(format!(
                "{} config has non-object `mcp`: {}",
                name,
                config_path.display()
            ));
            return Ok(());
        }

        let demongrep = &parsed["mcp"]["demongrep"];
        if demongrep.is_null() {
            warnings.push(format!("{} has no `mcp.demongrep` entry", name));
            return Ok(());
        }

        validate_demongrep_new_entry(name, &config_path, demongrep, failures);
        return Ok(());
    }

    warnings.push(format!(
        "{} config has neither `mcpServers` nor `mcp` key: {}",
        name,
        config_path.display()
    ));
    Ok(())
}

fn validate_demongrep_legacy_entry(
    name: &str,
    config_path: &Path,
    entry: &Value,
    failures: &mut Vec<String>,
) {
    let command_ok = entry
        .get("command")
        .and_then(Value::as_str)
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    if !command_ok {
        failures.push(format!(
            "{} demongrep MCP entry has invalid `command`: {}",
            name,
            config_path.display()
        ));
    }

    let args = entry.get("args").and_then(Value::as_array);
    let args_ok = args
        .map(|arr| arr.len() >= 2 && arr.first().and_then(Value::as_str) == Some("mcp"))
        .unwrap_or(false);

    if !args_ok {
        failures.push(format!(
            "{} demongrep MCP entry has invalid `args` (expected [\"mcp\", \"<project>\"]): {}",
            name,
            config_path.display()
        ));
    }
}

fn validate_demongrep_new_entry(
    name: &str,
    config_path: &Path,
    entry: &Value,
    failures: &mut Vec<String>,
) {
    let command_ok = entry
        .get("command")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.len() >= 2
                && arr
                    .iter()
                    .all(|v| v.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false))
                && arr.get(1).and_then(Value::as_str) == Some("mcp")
        })
        .unwrap_or(false);

    if !command_ok {
        failures.push(format!(
            "{} demongrep MCP entry has invalid `command` array (expected [\"<bin>\", \"mcp\", \"<project>\", ...]): {}",
            name,
            config_path.display()
        ));
    }
}
