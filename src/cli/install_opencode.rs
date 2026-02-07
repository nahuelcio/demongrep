use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Get the OpenCode config path
fn get_config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    Ok(home.join(".config").join("opencode").join("config.json"))
}

/// Get the absolute path to the demongrep binary
fn get_demongrep_binary_path() -> Result<PathBuf> {
    env::current_exe().context("Failed to get current executable path")
}

/// Read existing config or create a new one
fn load_config(config_path: &Path) -> Result<Value> {
    if config_path.exists() {
        let content =
            fs::read_to_string(config_path).context("Failed to read existing config file")?;
        serde_json::from_str(&content).context("Failed to parse existing config as JSON")
    } else {
        Ok(json!({ "mcpServers": {} }))
    }
}

/// Update or create the demongrep MCP server configuration
fn update_mcp_config(
    mut config: Value,
    demongrep_path: &Path,
    project_path: &Path,
) -> Result<Value> {
    if !config["mcpServers"].is_object() {
        config["mcpServers"] = json!({});
    }

    let cmd_str = demongrep_path
        .to_str()
        .ok_or_else(|| anyhow!("Failed to convert demongrep path to string"))?;

    let project_str = project_path
        .to_str()
        .ok_or_else(|| anyhow!("Failed to convert project path to string"))?;

    config["mcpServers"]["demongrep"] = json!({
        "command": cmd_str,
        "args": ["mcp", project_str]
    });

    Ok(config)
}

/// Write updated config back to file
fn write_config(config_path: &Path, config: &Value) -> Result<()> {
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).context(format!(
            "Failed to create config directory: {}",
            parent.display()
        ))?;
    }

    let json_string =
        serde_json::to_string_pretty(config).context("Failed to serialize config to JSON")?;

    fs::write(config_path, json_string).context(format!(
        "Failed to write config to {}",
        config_path.display()
    ))?;

    Ok(())
}

/// Install OpenCode MCP integration
pub fn run() -> Result<()> {
    let config_path = get_config_path().context("Failed to determine config file location")?;

    let demongrep_path =
        get_demongrep_binary_path().context("Failed to get demongrep binary path")?;

    let project_path = env::current_dir().context("Failed to get current directory")?;

    let config = load_config(&config_path).context("Failed to load or initialize config")?;

    let updated_config = update_mcp_config(config, &demongrep_path, &project_path)
        .context("Failed to update MCP configuration")?;

    write_config(&config_path, &updated_config)
        .context("Failed to write updated config to file")?;

    println!("OpenCode MCP integration installed successfully!");
    println!();
    println!("Configuration details:");
    println!("  Config file: {}", config_path.display());
    println!("  Demongrep binary: {}", demongrep_path.display());
    println!("  Project path: {}", project_path.display());
    println!();
    println!("MCP server configuration:");
    println!("  Command: {}", demongrep_path.display());
    println!("  Args: mcp {}", project_path.display());
    println!();
    println!("Please restart OpenCode for changes to take effect.");

    Ok(())
}
