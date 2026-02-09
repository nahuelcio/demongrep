use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub fn get_demongrep_binary_path() -> Result<PathBuf> {
    let current = env::current_exe().context("Failed to get current executable path")?;

    // If invoked from `cargo run` (target/{debug,release}), prefer installed user binary when present.
    let current_str = current.to_string_lossy();
    if current_str.contains("/target/debug/") || current_str.contains("/target/release/") {
        if let Some(home) = dirs::home_dir() {
            let preferred = home.join(".local").join("bin").join("demongrep-bin");
            if preferred.is_file() {
                return Ok(preferred);
            }
        }
    }

    Ok(current)
}

pub fn resolve_project_path(project_path: Option<PathBuf>) -> Result<PathBuf> {
    let path = match project_path {
        Some(path) => path,
        None => env::current_dir().context("Failed to get current directory")?,
    };

    let canonical = fs::canonicalize(&path)
        .with_context(|| format!("Failed to resolve project path: {}", path.display()))?;

    Ok(canonical)
}

pub fn load_config(config_path: &Path) -> Result<Value> {
    if config_path.exists() {
        let content =
            fs::read_to_string(config_path).context("Failed to read existing config file")?;
        serde_json::from_str(&content).context("Failed to parse existing config as JSON")
    } else {
        Ok(json!({ "mcpServers": {} }))
    }
}

fn is_opencode(agent_name: &str) -> bool {
    agent_name.eq_ignore_ascii_case("OpenCode")
}

pub fn update_mcp_config(
    mut config: Value,
    demongrep_path: &Path,
    project_path: &Path,
    agent_name: &str,
) -> Result<Value> {
    let cmd_str = demongrep_path
        .to_str()
        .ok_or_else(|| anyhow!("Failed to convert demongrep path to string"))?;

    let project_str = project_path
        .to_str()
        .ok_or_else(|| anyhow!("Failed to convert project path to string"))?;

    if is_opencode(agent_name) {
        if !config["mcp"].is_object() {
            config["mcp"] = json!({});
        }
        config["mcp"]["demongrep"] = json!({
            "type": "local",
            "command": [cmd_str, "mcp", project_str],
            "enabled": true
        });
    } else {
        if !config["mcpServers"].is_object() {
            config["mcpServers"] = json!({});
        }
        config["mcpServers"]["demongrep"] = json!({
            "command": cmd_str,
            "args": ["mcp", project_str]
        });
    }

    Ok(config)
}

pub fn write_config_with_backup(config_path: &Path, config: &Value) -> Result<Option<PathBuf>> {
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).context(format!(
            "Failed to create config directory: {}",
            parent.display()
        ))?;
    }

    let mut backup_path = None;
    if config_path.exists() {
        let backup = config_path.with_extension("json.bak");
        fs::copy(config_path, &backup).with_context(|| {
            format!(
                "Failed to create backup file from {} to {}",
                config_path.display(),
                backup.display()
            )
        })?;
        backup_path = Some(backup);
    }

    let json_string =
        serde_json::to_string_pretty(config).context("Failed to serialize config to JSON")?;

    fs::write(config_path, json_string)
        .with_context(|| format!("Failed to write config to {}", config_path.display()))?;

    Ok(backup_path)
}

pub fn install_agent_mcp(
    agent_name: &str,
    config_path: &Path,
    project_path: Option<PathBuf>,
    dry_run: bool,
) -> Result<()> {
    let demongrep_path =
        get_demongrep_binary_path().context("Failed to get demongrep binary path")?;
    let project_path = resolve_project_path(project_path)?;

    let config = load_config(config_path).context("Failed to load or initialize config")?;
    let updated_config = update_mcp_config(config, &demongrep_path, &project_path, agent_name)
        .context("Failed to update MCP configuration")?;

    println!("{} MCP integration", agent_name);
    println!("  Config file : {}", config_path.display());
    println!("  Binary      : {}", demongrep_path.display());
    println!("  Project     : {}", project_path.display());

    if dry_run {
        println!("Dry run mode: no changes written.");
        let path = if is_opencode(agent_name) {
            "mcp.demongrep"
        } else {
            "mcpServers.demongrep"
        };
        println!("Planned {} entry:", path);
        let preview = if is_opencode(agent_name) {
            &updated_config["mcp"]["demongrep"]
        } else {
            &updated_config["mcpServers"]["demongrep"]
        };
        println!(
            "{}",
            serde_json::to_string_pretty(preview).unwrap_or_else(|_| "{}".to_string())
        );
        return Ok(());
    }

    let backup_path = write_config_with_backup(config_path, &updated_config)
        .context("Failed to write updated config to file")?;

    println!("Installed successfully for {}.", agent_name);
    if let Some(backup_path) = backup_path {
        println!("Backup created: {}", backup_path.display());
    }
    println!("Restart {} for changes to take effect.", agent_name);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn update_mcp_config_preserves_existing_keys() {
        let config = json!({
            "theme": "dark",
            "mcpServers": {
                "other": {
                    "command": "other"
                }
            }
        });

        let updated = update_mcp_config(
            config,
            Path::new("/tmp/demongrep"),
            Path::new("/tmp/project"),
            "Codex",
        )
        .unwrap();

        assert_eq!(updated["theme"], "dark");
        assert!(updated["mcpServers"]["other"].is_object());
        assert_eq!(
            updated["mcpServers"]["demongrep"]["command"],
            "/tmp/demongrep"
        );
        assert_eq!(updated["mcpServers"]["demongrep"]["args"][0], "mcp");
        assert_eq!(
            updated["mcpServers"]["demongrep"]["args"][1],
            "/tmp/project"
        );
    }

    #[test]
    fn update_mcp_config_creates_mcp_servers_if_missing() {
        let config = json!({"foo": "bar"});
        let updated = update_mcp_config(
            config,
            Path::new("/tmp/demongrep"),
            Path::new("/tmp/project"),
            "Codex",
        )
        .unwrap();

        assert!(updated["mcpServers"].is_object());
        assert_eq!(
            updated["mcpServers"]["demongrep"]["command"],
            "/tmp/demongrep"
        );
    }

    #[test]
    fn write_config_with_backup_creates_backup_on_existing_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.json");
        fs::write(&path, "{\"mcpServers\":{}}\n").unwrap();

        let config = json!({"mcpServers": {"demongrep": {"command": "x", "args": ["mcp", "/p"]}}});
        let backup = write_config_with_backup(&path, &config).unwrap();

        assert!(backup.is_some());
        let backup = backup.unwrap();
        assert!(backup.exists());
        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("demongrep"));
    }

    #[test]
    fn update_mcp_config_uses_new_format_for_opencode() {
        let config = json!({"theme": "dark"});
        let updated = update_mcp_config(
            config,
            Path::new("/tmp/demongrep"),
            Path::new("/tmp/project"),
            "OpenCode",
        )
        .unwrap();

        assert_eq!(updated["theme"], "dark");
        assert!(updated["mcp"].is_object());
        assert_eq!(updated["mcp"]["demongrep"]["type"], "local");
        assert_eq!(updated["mcp"]["demongrep"]["command"][0], "/tmp/demongrep");
        assert_eq!(updated["mcp"]["demongrep"]["command"][1], "mcp");
        assert_eq!(updated["mcp"]["demongrep"]["command"][2], "/tmp/project");
        assert_eq!(updated["mcp"]["demongrep"]["enabled"], true);
    }
}
