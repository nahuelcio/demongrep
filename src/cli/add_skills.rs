use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const REPO_OWNER: &str = "nahuelcio";
const REPO_NAME: &str = "demongrep";

pub fn run(skill: String, ref_name: Option<String>, dest: Option<PathBuf>) -> Result<()> {
    ensure_command("curl")?;
    ensure_command("tar")?;

    let ref_name = ref_name.unwrap_or_else(|| format!("v{}", env!("CARGO_PKG_VERSION")));
    let destinations = resolve_destinations(dest)?;
    let temp_root = create_temp_root()?;

    let tarball = temp_root.join("repo.tar.gz");
    download_repo_archive(&ref_name, &tarball)?;
    extract_archive(&tarball, &temp_root)?;

    let repo_dir = find_extracted_repo_dir(&temp_root)?;
    let skill_src = repo_dir.join("skills").join(&skill);
    if !skill_src.exists() {
        return Err(anyhow!("Skill '{}' not found in ref '{}'", skill, ref_name));
    }

    let mut installed_paths = Vec::new();
    for skills_root in destinations {
        fs::create_dir_all(&skills_root)
            .with_context(|| format!("Failed to create {}", skills_root.display()))?;

        let skill_dest = skills_root.join(&skill);
        if skill_dest.exists() {
            fs::remove_dir_all(&skill_dest)
                .with_context(|| format!("Failed to remove existing {}", skill_dest.display()))?;
        }

        copy_dir_recursive(&skill_src, &skill_dest)?;
        installed_paths.push(skill_dest);
    }

    let _ = fs::remove_dir_all(&temp_root);

    println!("Installed skill: {}", skill);
    for path in installed_paths {
        println!("Destination: {}", path.display());
    }
    println!("Restart your agent(s) to pick up new skills.");
    Ok(())
}

fn ensure_command(cmd: &str) -> Result<()> {
    let status = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {} >/dev/null 2>&1", cmd))
        .status()
        .context("Failed to check required command")?;

    if !status.success() {
        return Err(anyhow!("Missing required command: {}", cmd));
    }

    Ok(())
}

fn resolve_destinations(dest: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    if let Some(d) = dest {
        return Ok(vec![d]);
    }

    let cwd = std::env::current_dir().context("Failed to determine current directory")?;
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;

    Ok(vec![
        cwd.join(".agents").join("skills"),
        home.join(".agents").join("skills"),
    ])
}

fn create_temp_root() -> Result<PathBuf> {
    let pid = std::process::id();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("System clock error")?
        .as_millis();
    let root = std::env::temp_dir().join(format!("demongrep-skill-{}-{}", pid, ts));
    fs::create_dir_all(&root).with_context(|| format!("Failed to create {}", root.display()))?;
    Ok(root)
}

fn download_repo_archive(ref_name: &str, tarball: &Path) -> Result<()> {
    let tag_url = format!(
        "https://github.com/{}/{}/archive/refs/tags/{}.tar.gz",
        REPO_OWNER, REPO_NAME, ref_name
    );
    let branch_url = format!(
        "https://github.com/{}/{}/archive/{}.tar.gz",
        REPO_OWNER, REPO_NAME, ref_name
    );

    let tag_status = Command::new("curl")
        .arg("-fsSL")
        .arg(&tag_url)
        .arg("-o")
        .arg(tarball)
        .status()
        .context("Failed to run curl for tag archive")?;

    if tag_status.success() {
        return Ok(());
    }

    let branch_status = Command::new("curl")
        .arg("-fsSL")
        .arg(&branch_url)
        .arg("-o")
        .arg(tarball)
        .status()
        .context("Failed to run curl for branch archive")?;

    if !branch_status.success() {
        return Err(anyhow!(
            "Failed to download skill source from '{}' or '{}'",
            tag_url,
            branch_url
        ));
    }

    Ok(())
}

fn extract_archive(tarball: &Path, out_dir: &Path) -> Result<()> {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(tarball)
        .arg("-C")
        .arg(out_dir)
        .status()
        .context("Failed to run tar")?;

    if !status.success() {
        return Err(anyhow!("Failed to extract archive"));
    }

    Ok(())
}

fn find_extracted_repo_dir(temp_root: &Path) -> Result<PathBuf> {
    let entries = fs::read_dir(temp_root)
        .with_context(|| format!("Failed to read {}", temp_root.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&format!("{}-", REPO_NAME)))
                .unwrap_or(false)
        {
            return Ok(path);
        }
    }

    Err(anyhow!("Could not find extracted repository directory"))
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst).with_context(|| format!("Failed to create {}", dst.display()))?;

    for entry in fs::read_dir(src).with_context(|| format!("Failed to read {}", src.display()))? {
        let entry = entry?;
        let entry_path = entry.path();
        let target_path = dst.join(entry.file_name());

        if entry_path.is_dir() {
            copy_dir_recursive(&entry_path, &target_path)?;
        } else {
            fs::copy(&entry_path, &target_path).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    entry_path.display(),
                    target_path.display()
                )
            })?;
        }
    }

    Ok(())
}
