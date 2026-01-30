"""
Load and apply environment configuration from dataset JSON files.

This module provides utilities to:
1. Load dataset configuration (version, paths, environment requirements)
2. Resolve paths (relative to apr_new/ or absolute)
3. Build environment variables from configuration
4. Apply environment to current process or subprocess
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

APR_ROOT = Path(__file__).resolve().parent.parent


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Load dataset configuration from JSON file.
    
    Args:
        dataset_name: Name of dataset (e.g., "defects4j", "swebench_verified")
    
    Returns:
        Dictionary containing dataset configuration
    """
    config_path = APR_ROOT / "dataset" / f"{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str: str, base: Optional[Path] = None) -> Path:
    """
    Resolve a path (relative to apr_new/ or absolute).
    
    Args:
        path_str: Path string (may be relative or absolute)
        base: Base directory for relative paths (default: APR_ROOT)
    
    Returns:
        Resolved absolute Path
    """
    if base is None:
        base = APR_ROOT
    
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_path_template(
    template: str,
    pid: Optional[str] = None,
    bid: Optional[int] = None,
    scratch_base: Optional[str] = None,
    base: Optional[Path] = None,
) -> Path:
    """
    Resolve a path template with placeholders.
    
    Supported placeholders:
    - {scratch_base}: Base directory for scratch (temporary) files
    - {pid}: Project ID
    - {bid}: Bug ID (will be formatted as {bid}b for Defects4J)
    - {APR_DIR}: apr_new/ directory
    
    Args:
        template: Path template string
        pid: Project ID (for {pid} placeholder)
        bid: Bug ID (for {bid} placeholder, formatted as {bid}b)
        scratch_base: Scratch base directory (for {scratch_base} placeholder)
        base: Base directory for relative paths (default: APR_ROOT)
    
    Returns:
        Resolved absolute Path
    """
    if base is None:
        base = APR_ROOT
    
    # Replace placeholders
    path_str = template
    path_str = path_str.replace("{APR_DIR}", str(base))
    
    if scratch_base:
        path_str = path_str.replace("{scratch_base}", scratch_base)
    elif "{scratch_base}" in path_str:
        # Default scratch base if not provided
        default_scratch = os.environ.get("APR_SCRATCH_BASE", "/tmp/apr_scratch")
        path_str = path_str.replace("{scratch_base}", default_scratch)
    
    if pid:
        path_str = path_str.replace("{pid}", pid)
    
    if bid is not None:
        # Replace {bid} - template may already have 'b' suffix (e.g., {pid}-{bid}b)
        # So we replace {bid} with just the number
        path_str = path_str.replace("{bid}", str(bid))
    
    # Resolve the final path
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def get_dataset_paths(
    dataset_name: str,
    pid: Optional[str] = None,
    bid: Optional[int] = None,
    scratch_base: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Get all paths for a dataset instance.
    
    Args:
        dataset_name: Name of dataset
        pid: Project ID
        bid: Bug ID
        scratch_base: Scratch base directory (optional, uses default if not provided)
    
    Returns:
        Dictionary of path name -> Path object
    """
    config = load_dataset_config(dataset_name)
    paths_config = config.get("paths", {})
    
    result = {}
    
    # Get scratch_base from config if not provided
    if scratch_base is None:
        scratch_base = paths_config.get("scratch_base", os.environ.get("APR_SCRATCH_BASE", "/tmp/apr_scratch"))
    
    # Resolve each path template
    for key, template in paths_config.items():
        if key == "scratch_base":
            # Skip scratch_base itself, it's just a config value
            continue
        if isinstance(template, str) and ("{" in template or "/" in template):
            result[key] = resolve_path_template(template, pid=pid, bid=bid, scratch_base=scratch_base)
        else:
            # Simple path (no template)
            result[key] = resolve_path(str(template))
    
    return result


def build_env_vars(config: Dict[str, Any], overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build environment variables from dataset configuration.
    
    Args:
        config: Dataset configuration dictionary
        overrides: Optional overrides for specific environment variables
    
    Returns:
        Dictionary of environment variable name -> value
    """
    env = {}
    env_config = config.get("environment", {})
    env_vars_config = env_config.get("env_vars", {})
    
    # Resolve base paths
    apr_dir = APR_ROOT
    defects4j_home = None
    if "defects4j_home" in env_config:
        defects4j_home = resolve_path(env_config["defects4j_home"]["default"])
    # Allow callers to override DEFECTS4J_HOME (common in cluster setups)
    if overrides and overrides.get("DEFECTS4J_HOME"):
        try:
            defects4j_home = resolve_path(overrides["DEFECTS4J_HOME"], base=Path("/"))
        except Exception:
            defects4j_home = Path(overrides["DEFECTS4J_HOME"])
    
    # Build each environment variable
    for var_name, var_config in env_vars_config.items():
        if isinstance(var_config, dict):
            # Handle different source types
            if "source" in var_config:
                # Simple source mapping
                source_path = var_config["source"]
                if source_path.startswith("environment."):
                    key = source_path.replace("environment.", "")
                    if key in env_config:
                        value = resolve_path(env_config[key]["default"])
                        env[var_name] = str(value)
                else:
                    # Direct path
                    env[var_name] = str(resolve_path(source_path))
            elif "prepend" in var_config:
                # PATH-like variable (prepend)
                prepend_paths = var_config["prepend"]
                # IMPORTANT: prefer overrides' existing value if provided (do NOT clobber PATH later)
                existing = (overrides or {}).get(var_name) or os.environ.get(var_name, "")
                new_parts = []
                for p in prepend_paths:
                    # Replace placeholders
                    p = p.replace("{APR_DIR}", str(apr_dir))
                    p = p.replace("{DEFECTS4J_HOME}", str(defects4j_home or ""))
                    p = p.replace("{JAVA_HOME}", os.environ.get("JAVA_HOME", ""))
                    if p:
                        new_parts.append(str(resolve_path(p)))
                if existing:
                    new_parts.append(existing)
                env[var_name] = ":".join(new_parts)
            elif "auto_build" in var_config and var_config["auto_build"]:
                # Auto-built variable (e.g., PERL5LIB)
                if "includes" in var_config:
                    parts = []
                    for include_path in var_config["includes"]:
                        # Replace placeholders
                        include_path = include_path.replace("{DEFECTS4J_HOME}", str(defects4j_home or ""))
                        if include_path:
                            resolved = resolve_path(include_path)
                            if resolved.exists():
                                # Special case: local perl vendor dir usually needs arch-specific subdir(s) too.
                                # Example: .../.perl5/lib/perl5/x86_64-linux-thread-multi/DBI.pm
                                if (
                                    resolved.is_dir()
                                    and resolved.name == "perl5"
                                    and resolved.parent.name == "lib"
                                    and resolved.parent.parent.name == ".perl5"
                                ):
                                    try:
                                        for sub in sorted(resolved.iterdir()):
                                            if sub.is_dir():
                                                parts.append(str(sub))
                                    except Exception:
                                        pass
                                parts.append(str(resolved))
                    if parts:
                        existing = (overrides or {}).get(var_name) or os.environ.get(var_name, "")
                        if existing:
                            parts.append(existing)
                        env[var_name] = ":".join(parts)
    
    # Apply overrides for keys NOT managed by this dataset config.
    # For managed keys like PATH, we already incorporated the override value above
    # (as the "existing" suffix for prepend/auto_build).
    if overrides:
        managed = set(env_vars_config.keys())
        for k, v in overrides.items():
            if k not in managed and k not in env:
                env[k] = v
    
    return env


def apply_defects4j_env(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Load and apply Defects4J environment configuration.
    
    Args:
        overrides: Optional overrides for specific environment variables
    
    Returns:
        Dictionary of environment variables to set
    """
    config = load_dataset_config("defects4j")
    env = build_env_vars(config, overrides)
    
    # Set DEFECTS4J_HOME if not already set
    if "DEFECTS4J_HOME" not in env:
        env_config = config.get("environment", {})
        if "defects4j_home" in env_config:
            env["DEFECTS4J_HOME"] = str(resolve_path(env_config["defects4j_home"]["default"]))
    
    # Set TZ if not already set
    if "TZ" not in env:
        env_config = config.get("environment", {})
        if "timezone" in env_config:
            env["TZ"] = env_config["timezone"]["default"]
    
    # Java setup (priority: DEFECTS4J_JAVA_HOME > .jdks/java8 > auto-detect)
    java_home = None
    if overrides and "JAVA_HOME" in overrides:
        java_home = overrides["JAVA_HOME"]
    elif "DEFECTS4J_JAVA_HOME" in os.environ:
        java_home = os.environ["DEFECTS4J_JAVA_HOME"]
    else:
        # Check for repo-local Java 8 (try both apr_new/.jdks and tdel/.jdks)
        jdk8_path = None
        # Try apr_new/.jdks first
        apr_jdk8 = APR_ROOT / ".jdks" / "java8"
        if apr_jdk8.exists():
            jdk8_path = apr_jdk8
        else:
            # Try tdel/.jdks (parent directory)
            tdel_jdk8 = APR_ROOT.parent / ".jdks" / "java8"
            if tdel_jdk8.exists():
                jdk8_path = tdel_jdk8
        if jdk8_path.exists():
            # Support both symlink and directory
            java_home = str(jdk8_path.resolve())
        else:
            # Auto-detect Java 8 or Java 11
            import glob
            java8_dirs = glob.glob("/usr/lib/jvm/java-1.8.0-openjdk*") + glob.glob("/usr/lib/jvm/java-8-openjdk*")
            if java8_dirs:
                java_home = java8_dirs[0]
            else:
                # Try additional common paths for Java 8
                additional_paths = [
                    "/usr/java/jdk1.8.0",
                    "/opt/java/jdk1.8.0",
                    "/usr/local/java/jdk1.8.0",
                    "/opt/jdk/jdk1.8.0",
                    "/usr/local/jdk/jdk1.8.0",
                    str(APR_ROOT / ".jdks" / "java8"),  # Current repo
                ]
                for path_str in additional_paths:
                    path = Path(path_str)
                    if path.exists():
                        java_bin = path / "bin" / "java"
                        if java_bin.exists():
                            java_home = str(path.resolve())
                            break
                
                # If still no Java 8, try Java 11
                if not java_home:
                    java11_dirs = glob.glob("/usr/lib/jvm/java-11-openjdk*")
                    if java11_dirs:
                        java_home = java11_dirs[0]
    
    if java_home and os.path.exists(java_home):
        env["JAVA_HOME"] = java_home
        # Update PATH to include Java bin
        java_bin = os.path.join(java_home, "bin")
        existing_path = env.get("PATH", os.environ.get("PATH", ""))
        # Remove Java 17/21 from PATH
        path_parts = [p for p in existing_path.split(":") if "java-17" not in p and "java-21" not in p and "java-1.17" not in p and "java-1.21" not in p]
        env["PATH"] = f"{java_bin}:{':'.join(path_parts)}"
    
    return env


def apply_swebench_env(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Load and apply SWE-bench environment configuration.
    
    Args:
        overrides: Optional overrides for specific environment variables
    
    Returns:
        Dictionary of environment variables to set
    """
    config = load_dataset_config("swebench_verified")
    env = build_env_vars(config, overrides)
    
    # Set APR_SWEBENCH_RUNTIME to apptainer by default (this project always uses apptainer)
    if "APR_SWEBENCH_RUNTIME" not in env:
        env["APR_SWEBENCH_RUNTIME"] = "apptainer"
    
    return env


def get_dataset_version(dataset_name: str) -> str:
    """
    Get version string for a dataset.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Version string (e.g., "2.0.0", "4.1.0")
    """
    config = load_dataset_config(dataset_name)
    return config.get("version", "unknown")
