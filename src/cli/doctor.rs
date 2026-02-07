use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub async fn run() -> Result<()> {
    println!("Checking demongrep installation...");

    let mut failures: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    check_binary(&mut failures)?;
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

    let mcp_servers = parsed.get("mcpServers");
    if mcp_servers.is_none() {
        warnings.push(format!(
            "{} config has no `mcpServers` key: {}",
            name,
            config_path.display()
        ));
        return Ok(());
    }

    if !mcp_servers.and_then(Value::as_object).is_some() {
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

    validate_demongrep_entry(name, &config_path, demongrep, failures);
    Ok(())
}

fn validate_demongrep_entry(
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
