from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agent.adapters.base import DatasetAdapter


APR_ROOT = Path(__file__).resolve().parents[2]

def _swe_runtime() -> str:
    """
    Select runtime for SWE-bench execution.

    - docker/podman (default): use docker SDK + swebench harness
    - apptainer: use apptainer exec (no docker daemon required)
    """
    return (os.environ.get("APR_SWEBENCH_RUNTIME") or "").strip().lower() or "docker"


def _swebench_sif_path() -> str:
    """
    Optional fast-path for Apptainer: if this env var points to an existing .sif file,
    we run `apptainer exec <sif>` instead of `docker://...` (avoids long image pull).
    """
    p = (os.environ.get("APR_SWEBENCH_SIF_PATH") or "").strip()
    if p and Path(p).is_file():
        return p
    return ""


def _parse_json_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x)]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # many SWE-bench fields are JSON-encoded strings like ["test_a", "test_b"]
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        except Exception:
            pass
        return [s]
    return [str(val)]


def _ensure_hf_project_cache() -> None:
    """
    Force HF/datasets caches to live under apr_new/, not $HOME.
    Safe to call repeatedly.
    """
    hf_root = APR_ROOT / ".hf"
    (hf_root / "hub").mkdir(parents=True, exist_ok=True)
    (hf_root / "datasets").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_root / "datasets"))


_SWE_DATASET_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_verified_dataset_map() -> Dict[str, Dict[str, Any]]:
    global _SWE_DATASET_CACHE
    if _SWE_DATASET_CACHE is not None:
        return _SWE_DATASET_CACHE

    _ensure_hf_project_cache()

    # Import lazily so this adapter doesn't affect Defects4J runs unless used.
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in ds:
        iid = row.get("instance_id")
        if isinstance(iid, str) and iid:
            mapping[iid] = dict(row)
    _SWE_DATASET_CACHE = mapping
    return mapping


def _github_https_url(repo: str) -> str:
    # repo in dataset is typically like "psf/requests"
    return f"https://github.com/{repo}.git"


def _using_workdir_archives() -> bool:
    # Archive-extract mode: workdir in local tmp, removed after run; no per-workdir cleanup
    return os.environ.get("APR_USE_WORKDIR_ARCHIVES", "0") == "1"


def _run(cmd: list[str], *, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=merged_env, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired as e:
        return {"rc": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s: {' '.join(cmd)}", "timeout": True}
    except FileNotFoundError as e:
        return {"rc": -1, "stdout": "", "stderr": f"Command not found: {cmd[0] if cmd else 'unknown'}. Error: {e}", "error": str(e)}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        # Include detailed error (first 1000 chars: exception type and key info)
        error_detail = f"Command execution failed: {' '.join(cmd[:10])}...\nError type: {type(e).__name__}\nError: {e}\n{traceback.format_exc()[:1000]}"
        return {"rc": -1, "stdout": "", "stderr": error_detail, "error": str(e)}


def _run_apptainer(*, image: str, argv: list[str], bind: str, pwd: str, timeout: int) -> Dict[str, Any]:
    """
    Run a command inside an Apptainer container.

    Note: Caller should ensure APPTAINER_CACHEDIR/APPTAINER_TMPDIR are set to non-$HOME
    (e.g. source apr_new/bin/apptainer_project_env.sh).
    """
    # NOTE: apptainer exec does NOT use `--` as an argv separator (unlike some CLIs).
    #
    # IMPORTANT (cluster compatibility):
    # Some HPC Apptainer configs bind-mount host paths like /opt into the container
    # (via apptainer.conf bind-paths / hostfs). That can *overwrite* the image's own
    # conda/testbed environment and cause "pytest not importable" / wrong Python.
    #
    # Strategy: Conditionally mount /opt based on whether testbed Python exists in image.
    # - If image has testbed Python: use --no-mount /opt to avoid host /opt overwriting it
    # - If image lacks testbed Python: explicitly bind host /opt (fallback to host testbed)
    #
    # Check if image has testbed Python (quick check with --no-mount to see image content)
    # Strategy for /opt mounting:
    # - For SWE-bench SIF images, ALWAYS use --no-mount /opt to preserve image's own environment
    # - This matches prefetch verification behavior: prefetch always uses --no-mount /opt
    # - The script inside will find testbed Python if it exists in the image
    # - If testbed Python is not found, script will fallback to system Python and bootstrap required version
    cmd = ["apptainer", "exec", "--cleanenv", "--bind", bind]
    # IMPORTANT:
    # - Binding host /opt into container can overwrite the image's own /opt (miniconda/testbed),
    #   causing "Testbed Python not found" and forcing slow/fragile dynamic miniconda+pytest installs.
    # - For SWE-bench images (both pre-built .sif and docker:// swebench/*), we want the image's
    #   own environment, so ALWAYS disable /opt mounting.
    is_swebench_image = (
        image.startswith("docker://swebench/")
        or "/sweb.eval." in image
        or "sweb.eval." in image
    )
    if (image.endswith(".sif") and Path(image).exists()) or is_swebench_image:
        cmd.extend(["--no-mount", "/opt"])
        print("[APPTAINER] Using --no-mount /opt (preserve image environment)", flush=True)
    else:
        # For non-SWE-bench images, try to bind host /opt as fallback
        try:
            if Path("/opt").exists():
                cmd.extend(["--bind", "/opt:/opt"])
                print("[APPTAINER] Binding host /opt:/opt (fallback for non-SWE-bench image)", flush=True)
            else:
                print("[APPTAINER] Host /opt not present; cannot bind fallback testbed", flush=True)
        except Exception as e:
            print(f"[APPTAINER] Failed to bind host /opt (continuing without): {e}", flush=True)
    cmd.extend(["--pwd", pwd, image])
    cmd.extend(argv)

    # Use APPTAINER_* if the caller set them (preferred).
    # Otherwise use APPTAINER_BASE env or generic default (no hardcoded project paths).
    cache_base = Path(os.environ.get("APPTAINER_BASE", "/tmp/apptainer"))
    env_cache = os.environ.get("APPTAINER_CACHEDIR") or os.environ.get("SINGULARITY_CACHEDIR")
    env_tmp = os.environ.get("APPTAINER_TMPDIR") or os.environ.get("SINGULARITY_TMPDIR")

    cache_dir = Path(env_cache) if env_cache else (cache_base / "cache")
    primary_tmp_dir = Path(env_tmp) if env_tmp else (cache_base / "tmp")

    def _ensure_dir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)
        try:
            p.chmod(0o700)
        except Exception:
            pass

    try:
        _ensure_dir(cache_dir)
        _ensure_dir(primary_tmp_dir)
    except Exception:
        pass

    def _env_for(tmp_dir: Path) -> Dict[str, str]:
        return {
            "APPTAINER_CACHEDIR": str(cache_dir),
            "APPTAINER_TMPDIR": str(tmp_dir),
            "SINGULARITY_CACHEDIR": str(cache_dir),
            "SINGULARITY_TMPDIR": str(tmp_dir),
        }

    print(f"[APPTAINER] Executing: {' '.join(cmd[:6])}... [bash script]", flush=True)
    print(f"[APPTAINER] Cache dir: {cache_dir}", flush=True)
    print(f"[APPTAINER] Tmp dir: {primary_tmp_dir}", flush=True)
    print(f"[APPTAINER] Timeout: {timeout} seconds", flush=True)
    
    r = _run(cmd, env=_env_for(primary_tmp_dir), timeout=timeout)
    
    print(f"[APPTAINER] Execution completed: rc={r.get('rc', 'N/A')}, timeout={r.get('timeout', False)}", flush=True)
    
    stderr_lower = (r.get("stderr") or "").lower()
    if r.get("rc") not in (0, None) and "disk quota exceeded" in stderr_lower:
        # Fallback: use node-local /tmp for the large unpack step.
        try:
            uid = os.getuid()
            alt_tmp = Path(f"/tmp/apptainer-tmp-{uid}")
            _ensure_dir(alt_tmp)
            r2 = _run(cmd, env=_env_for(alt_tmp), timeout=timeout)
            r2["stderr"] = "[RETRY] APPTAINER_TMPDIR=/tmp (quota fallback)\n" + (r2.get("stderr") or "")
            return r2
        except Exception:
            return r
    return r


def _swebench_instance_image(*, instance_id: str, arch: str = "x86_64", tag: str = "latest", namespace: str = "swebench") -> str:
    """
    SWE-bench instance image naming convention (see swebench.harness.test_spec.TestSpec.instance_image_key):
      {namespace}/sweb.eval.{arch}.{instance_id_lower}:{tag}
    where instance_id_lower replaces "__" with "_1776_" for remote namespace images.
    """
    iid = instance_id.strip().lower()
    iid = iid.replace("__", "_1776_")
    return f"{namespace}/sweb.eval.{arch}.{iid}:{tag}"


def _parse_test_directives_from_patch(test_patch: str) -> list[str]:
    """
    Minimal version of swebench.harness.test_spec.utils.get_test_directives for Python repos:
    Extract the b/<path> targets from "diff --git a/... b/<path>" lines.
    """
    import re

    diff_pat = r"^diff --git a/.* b/(.*)$"
    directives = re.findall(diff_pat, test_patch or "", flags=re.MULTILINE)
    # Keep only likely test files (Python)
    directives = [d for d in directives if d.endswith(".py")]
    # De-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for d in directives:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Django sitecustomize.py source: reused in validation and run_one_test heredocs; edit here to apply in both
_DJANGO_SITECUSTOMIZE_PY_SRC = r"""import os
import sys
try:
    # Ensure DJANGO_SETTINGS_MODULE is set
    settings_module = os.environ.get("DJANGO_SETTINGS_MODULE", "test_sqlite")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    
    # Change to testbed directory to ensure proper module resolution
    if '/testbed' not in sys.path:
        sys.path.insert(0, '/testbed')
    if '/testbed/tests' not in sys.path:
        sys.path.insert(0, '/testbed/tests')
    
    # Change to testbed directory (important for relative imports)
    try:
        os.chdir('/testbed')
    except OSError:
        pass  # Ignore if /testbed doesn't exist yet
    
    # When APR_DJANGO_USE_RUNTESTS=1 runtests.py does django.setup/setup_databases; skip import here to avoid "populate() isn't reentrant"
    if os.environ.get("APR_DJANGO_USE_RUNTESTS") == "1":
        pass
    else:
        # Import Django
        import django
        from django.conf import settings

        # CRITICAL: Import the settings module BEFORE django.setup()
        # This ensures INSTALLED_APPS and other settings are loaded
        settings_loaded = False
        try:
            import importlib
            if settings_module:
                # Import the settings module explicitly
                settings_mod = importlib.import_module(settings_module)
                settings_loaded = True
                # Verify that settings module was loaded and has INSTALLED_APPS
                if hasattr(settings_mod, 'INSTALLED_APPS'):
                    # Settings module loaded successfully with INSTALLED_APPS
                    pass
                else:
                    # Settings module exists but no INSTALLED_APPS - will use Django defaults
                    pass
        except (ImportError, ModuleNotFoundError) as e:
            # Settings module not found - Django will use default settings
            import sys
            print(f'[WARN] Failed to import settings module {settings_module}: {e}', file=sys.stderr, flush=True)
        except Exception as e:
            # Other errors during settings import
            import sys
            print(f'[WARN] Error importing settings module {settings_module}: {e}', file=sys.stderr, flush=True)

        # CRITICAL: Ensure common Django apps are in INSTALLED_APPS before django.setup()
        # Some Django projects' test_sqlite.py don't define INSTALLED_APPS, causing
        # "Model class doesn't declare an explicit app_label" errors when models are imported
        # (e.g. test_utils.models.Car). Also discover test apps from /testbed/tests/ so that
        # Django's own test suite can be collected when running full regression via pytest.
        try:
            if settings_loaded:
                # Discover Django test apps from /testbed/tests/ (mirrors runtests.py)
                _django_test_apps = []
                try:
                    import os as _os
                    _skip = {'data', 'import_error_package', 'test_runner_apps', 'gis_tests', 'admin_autodiscover'}
                    if _os.path.isdir('/testbed/tests'):
                        for _f in _os.scandir('/testbed/tests'):
                            if (_f.is_dir() and '.' not in _f.name and _f.name not in _skip and
                                _os.path.exists(_os.path.join(_f.path, '__init__.py'))):
                                _django_test_apps.append(_f.name)
                except Exception:
                    pass
                if hasattr(settings_mod, 'INSTALLED_APPS'):
                    installed_apps = getattr(settings_mod, 'INSTALLED_APPS', [])
                    if not isinstance(installed_apps, list):
                        installed_apps = list(installed_apps)
                    if 'django.contrib.sites' not in installed_apps:
                        installed_apps.append('django.contrib.sites')
                    for _a in _django_test_apps:
                        if _a not in installed_apps:
                            installed_apps.append(_a)
                    setattr(settings_mod, 'INSTALLED_APPS', installed_apps)
                else:
                    installed_apps = [
                        'django.contrib.contenttypes',
                        'django.contrib.auth',
                        'django.contrib.sessions',
                        'django.contrib.admin',
                        'django.contrib.messages',
                        'django.contrib.staticfiles',
                        'django.contrib.sites',
                    ] + _django_test_apps
                    setattr(settings_mod, 'INSTALLED_APPS', installed_apps)
            else:
                pass
        except Exception as e:
            import sys
            print(f'[WARN] Failed to modify settings module INSTALLED_APPS: {e}', file=sys.stderr, flush=True)

        # Ensure sqlite3 DATABASES have NAME (e.g. test_sqlite.py omits it; backend requires it)
        try:
            if settings_loaded and hasattr(settings_mod, 'DATABASES') and getattr(settings_mod, 'DATABASES'):
                _db = getattr(settings_mod, 'DATABASES')
                if isinstance(_db, dict):
                    for _k in ('default', 'other'):
                        if _k in _db and isinstance(_db[_k], dict):
                            _e = _db[_k].get('ENGINE') or ''
                            if 'sqlite3' in str(_e) and not _db[_k].get('NAME'):
                                _db[_k]['NAME'] = ':memory:'
        except Exception as _e:
            import sys
            print(f'[WARN] Failed to set DATABASES NAME for sqlite3: {_e}', file=sys.stderr, flush=True)

        # MIGRATION_MODULES: fix "Cannot resolve bases" and "no such table".
        # auth_tests/admin_views: use injected placeholder migrations (below), depend on auth so UserProxy graph resolves.
        # queries/aggregation: set None so migrate --run-syncdb creates tables from models, avoid "no such table".
        try:
            if settings_loaded:
                _mm = getattr(settings_mod, 'MIGRATION_MODULES', None)
                if _mm is None or not isinstance(_mm, dict):
                    _mm = {}
                    setattr(settings_mod, 'MIGRATION_MODULES', _mm)
                _mm['auth_tests'] = 'tests.auth_tests.migrations'
                _mm['admin_views'] = 'tests.admin_views.migrations'
                _mm['queries'] = None
                _mm['aggregation'] = None
        except Exception as _e:
            import sys
            print(f'[WARN] Failed to set MIGRATION_MODULES: {_e}', file=sys.stderr, flush=True)

        django.setup()
        # Inject placeholder migrations for auth_tests/admin_views so migrate resolves UserProxy -> auth.User
        try:
            _tests = '/testbed/tests'
            os.makedirs(_tests, exist_ok=True)
            if not os.path.exists(_tests + '/__init__.py'):
                open(_tests + '/__init__.py', 'w').close()
            _auth_first = '0001_initial'
            try:
                import pkgutil
                import django.contrib.auth.migrations as _am
                _names = [m.name for m in pkgutil.iter_modules(_am.__path__) if m.name and m.name[0].isdigit()]
                if _names:
                    _auth_first = '0001_initial' if '0001_initial' in _names else min(_names)
            except Exception:
                pass
            for _mig_name in ('auth_tests', 'admin_views'):
                _parent = '/testbed/tests/' + _mig_name
                _mdir = _parent + '/migrations'
                os.makedirs(_mdir, exist_ok=True)
                _parent_init = _parent + '/__init__.py'
                if not os.path.exists(_parent_init):
                    open(_parent_init, 'w').close()
                _init = _mdir + '/__init__.py'
                if not os.path.exists(_init):
                    open(_init, 'w').close()
                _py = _mdir + '/0001_initial.py'
                if os.path.exists(_py):
                    continue
                _body = '''# Generated by APR sitecustomize to fix "Cannot resolve bases" for UserProxy
from django.db import migrations
class Migration(migrations.Migration):
    initial = True
    dependencies = [('auth', ''' + repr(_auth_first) + ''')]
    operations = []
'''
                with open(_py, 'w') as _f:
                    _f.write(_body)
        except Exception as _e:
            import sys
            print(f'[WARN] Failed to inject auth_tests/admin_views migrations: {_e}', file=sys.stderr, flush=True)
        # Create test DB tables for pytest (runtests.py does setup_databases; raw pytest does not)
        try:
            from django.test.utils import setup_databases
            setup_databases(verbosity=0, interactive=False, keepdb=False, debug_sql=False, parallel=0)
        except Exception as _e:
            import sys
            print(f'[WARN] setup_databases in sitecustomize failed: {_e}', file=sys.stderr, flush=True)
            try:
                from django.db import connection
                connection.creation.create_test_db(verbosity=0, autoclobber=True)
            except Exception as _e2:
                print(f'[WARN] create_test_db fallback also failed: {_e2}', file=sys.stderr, flush=True)
except Exception as e:
    import sys
    print(f'[WARN] Django setup in sitecustomize.py failed: {e}', file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    pass
"""

_DJANGO_SITECUSTOMIZE_HEREDOC = (
    '      cat >"$SITE_DIR/sitecustomize.py" <<\'PY_APR_SITE\'\n'
    + _DJANGO_SITECUSTOMIZE_PY_SRC
    + '\nPY_APR_SITE\n'
)


def _build_test_environment_script_base() -> str:
    """
    Build the common test-environment script base (Python lookup, conda, pytest, etc.).
    The returned fragment is inserted at the start of test scripts, before project-specific config.
    """
    return r"""
# ============================================================================
# Common test environment script (from _build_test_environment_script_base)
# Same logic used by _verify_test_suite and _validate_apptainer
# ============================================================================

set -euo pipefail
export HOME=/tmp
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1

# pylint-dev__pylint-4661: Fix PYLINT_HOME path mismatch
if [ "${APR_IS_PYLINTDEV:-0}" = "1" ] && [ "${APR_INSTANCE_ID:-}" = "pylint-dev__pylint-4661" ]; then
  export XDG_CACHE_HOME=/tmp/.cache
  mkdir -p /tmp/.cache
fi

# Ensure pytest is available in-container.
# IMPORTANT: do NOT reuse a shared /tmp/apr_site across runs.
SITE_DIR="/tmp/apr_site_$$"
mkdir -p "$SITE_DIR"

# pylint-dev: provide _distutils_hack stub
if [ "${APR_IS_PYLINTDEV:-0}" = "1" ] && [ ! -f "$SITE_DIR/_distutils_hack.py" ]; then
  cat > "$SITE_DIR/_distutils_hack.py" <<'EOF_APR_DISTUTILS_HACK'
# Auto-generated by APR verification harness.
def add_shim():
    pass
def ensure_shim():
    pass
EOF_APR_DISTUTILS_HACK
fi

# Special case: pytest-dev projects
PYTEST_SRC_DIR=""
if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ -f "/testbed/src/pytest.py" ]; then
  PYTEST_SRC_DIR="/testbed/src"
  export PYTHONPATH="$PYTEST_SRC_DIR:$SITE_DIR:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SITE_DIR:${PYTHONPATH:-}"
fi

# Helper functions for Python discovery
pick_python_with_pytest() {
  local c
  for c in \
    /opt/miniconda3/envs/testbed/bin/python \
    /opt/conda/envs/testbed/bin/python \
    /opt/miniconda/envs/testbed/bin/python \
    /opt/conda/bin/python \
    /usr/bin/python3 \
    /usr/local/bin/python3 \
    python3 \
    python \
  ; do
    if [ -x "$c" ]; then
      if "$c" -c "import pytest" >/dev/null 2>&1; then
        echo "$c"
        return 0
      fi
    elif command -v "$c" >/dev/null 2>&1; then
      if "$c" -c "import pytest" >/dev/null 2>&1; then
        command -v "$c"
        return 0
      fi
    fi
  done
  return 1
}

pick_python_with_pip() {
  local c
  for c in \
    /opt/miniconda3/envs/testbed/bin/python \
    /opt/conda/envs/testbed/bin/python \
    /opt/miniconda/envs/testbed/bin/python \
    /opt/conda/bin/python \
    /usr/bin/python3 \
    /usr/local/bin/python3 \
    python3 \
    python \
  ; do
    if [ -x "$c" ]; then
      if "$c" -m pip --version >/dev/null 2>&1; then
        echo "$c"
        return 0
      fi
    elif command -v "$c" >/dev/null 2>&1; then
      if "$c" -m pip --version >/dev/null 2>&1; then
        command -v "$c"
        return 0
      fi
    fi
  done
  return 1
}

# Prefer SWE-bench "testbed" python for stability
PY=""
for c in \
  /opt/miniconda3/envs/testbed/bin/python \
  /opt/conda/envs/testbed/bin/python \
  /opt/miniconda/envs/testbed/bin/python \
; do
  if [ -x "$c" ]; then
    PY="$c"
    echo "[INFO] Found testbed Python in image: $PY" >&2
    break
  fi
done

# If testbed Python not found but /miniconda.sh exists, try to create testbed environment dynamically
if [ -z "$PY" ] && [ -f "/miniconda.sh" ]; then
  echo "[INFO] Testbed Python not found, but /miniconda.sh exists. Attempting to create testbed environment..." >&2
  CONDA_BASE=""
  if [ -x "/opt/miniconda3/bin/conda" ]; then
    CONDA_BASE="/opt/miniconda3"
  elif [ -d "/opt/miniconda3" ] && [ ! -x "/opt/miniconda3/bin/conda" ]; then
    echo "[WARN] /opt/miniconda3 exists but conda is not executable, will try /tmp instead" >&2
    CONDA_BASE="/tmp/apr_miniconda3"
  else
    echo "[INFO] Attempting to install miniconda to /opt/miniconda3..." >&2
    if bash /miniconda.sh -b -p /opt/miniconda3 >/tmp/apr_miniconda_install_testbed.log 2>&1; then
      CONDA_BASE="/opt/miniconda3"
      echo "[INFO] Successfully installed miniconda to $CONDA_BASE" >&2
    else
      echo "[INFO] /opt/miniconda3 not writable, installing to /tmp/apr_miniconda3 instead..." >&2
      CONDA_BASE="/tmp/apr_miniconda3"
      bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_miniconda_install_testbed.log 2>&1 || true
    fi
  fi
  if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
    TESTBED_PYTHON_VERSION="${APR_REQUIRED_PYTHON_VERSION:-3.10}"
    TESTBED_PY="${CONDA_BASE}/envs/testbed/bin/python"
    if [ -x "$TESTBED_PY" ]; then
      PY="$TESTBED_PY"
      echo "[INFO] Found existing testbed environment: PY=$PY" >&2
      "$PY" -V >&2 || true
      # scikit-learn-specific: ensure pip/numpy/scipy are available
      if [ "${APR_IS_SCIKITLEARN:-0}" = "1" ]; then
        if ! "$PY" -m pip --version >/dev/null 2>&1; then
          echo "[INFO] scikit-learn: pip not found in testbed environment, installing..." >&2
          "${CONDA_BASE}/bin/conda" install -n testbed pip -y >/tmp/apr_conda_install_pip.log 2>&1 || true
        fi
        if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
          echo "[INFO] scikit-learn: numpy not found in testbed environment, installing..." >&2
          "$PY" -m pip install numpy >/tmp/apr_pip_install_numpy.log 2>&1 || true
        fi
        if ! "$PY" -c "import scipy" >/dev/null 2>&1; then
          echo "[INFO] scikit-learn: scipy not found in testbed environment, installing..." >&2
          "$PY" -m pip install scipy >/tmp/apr_pip_install_scipy.log 2>&1 || true
        fi
      fi
    else
      echo "[INFO] Creating testbed conda environment with Python ${TESTBED_PYTHON_VERSION}..." >&2
      if ! "${CONDA_BASE}/bin/conda" create -n testbed "python=${TESTBED_PYTHON_VERSION}" pip -y >/tmp/apr_conda_create_testbed.log 2>&1; then
        if grep -qE "prefix already exists|CondaValueError.*prefix" /tmp/apr_conda_create_testbed.log 2>/dev/null; then
          echo "[WARN] Conda create failed due to existing prefix, removing and retrying..." >&2
          rm -rf "${CONDA_BASE}/envs/testbed" 2>/dev/null || true
          "${CONDA_BASE}/bin/conda" create -n testbed "python=${TESTBED_PYTHON_VERSION}" pip -y >/tmp/apr_conda_create_testbed_retry.log 2>&1 || true
        fi
      fi
      if [ -x "$TESTBED_PY" ]; then
        PY="$TESTBED_PY"
        echo "[INFO] Successfully created testbed environment: PY=$PY" >&2
        "$PY" -V >&2 || true
        # scikit-learn-specific: ensure pip/numpy/scipy are available
        if [ "${APR_IS_SCIKITLEARN:-0}" = "1" ]; then
          if ! "$PY" -m pip --version >/dev/null 2>&1; then
            "${CONDA_BASE}/bin/conda" install -n testbed pip -y >/tmp/apr_conda_install_pip.log 2>&1 || true
          fi
          if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
            "$PY" -m pip install numpy >/tmp/apr_pip_install_numpy.log 2>&1 || true
          fi
          if ! "$PY" -c "import scipy" >/dev/null 2>&1; then
            "$PY" -m pip install scipy >/tmp/apr_pip_install_scipy.log 2>&1 || true
          fi
        fi
      fi
    fi
  fi
fi

# Astropy-specific: handle existing but broken testbed environments
if [ "${APR_IS_ASTROPY:-0}" = "1" ] && [ -z "$PY" ] && [ -f "/miniconda.sh" ]; then
  CONDA_BASE=""
  if [ -x "/opt/miniconda3/bin/conda" ]; then
    CONDA_BASE="/opt/miniconda3"
  elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
    CONDA_BASE="/tmp/apr_miniconda3"
  fi
  if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
    TESTBED_PY="${CONDA_BASE}/envs/testbed/bin/python"
    if [ -d "${CONDA_BASE}/envs/testbed" ] && [ ! -x "$TESTBED_PY" ]; then
      echo "[INFO] Astropy: Found broken testbed environment, removing and recreating..." >&2
      rm -rf "${CONDA_BASE}/envs/testbed" 2>/dev/null || true
      source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
      TESTBED_PYTHON_VERSION="${APR_REQUIRED_PYTHON_VERSION:-3.10}"
      "${CONDA_BASE}/bin/conda" create -n testbed "python=${TESTBED_PYTHON_VERSION}" pip -y >/tmp/apr_conda_create_testbed_astropy.log 2>&1 || true
      if [ -x "$TESTBED_PY" ]; then
        PY="$TESTBED_PY"
        echo "[INFO] Astropy: Successfully recreated testbed environment: PY=$PY" >&2
        "$PY" -V >&2 || true
      fi
    elif [ -x "$TESTBED_PY" ]; then
      PY="$TESTBED_PY"
      echo "[INFO] Astropy: Found existing testbed environment: PY=$PY" >&2
      "$PY" -V >&2 || true
      # Check and upgrade numpy if version is too old
      if "$PY" -c "import numpy" >/dev/null 2>&1; then
        NUMPY_VER=$("$PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
        if [ -n "$NUMPY_VER" ]; then
          NUMPY_MAJOR=$(echo "$NUMPY_VER" | cut -d. -f1)
          NUMPY_MINOR=$(echo "$NUMPY_VER" | cut -d. -f2)
          NUMPY_PATCH=$(echo "$NUMPY_VER" | cut -d. -f3 | sed 's/[^0-9].*//')
          NEED_UPGRADE=0
          if [ "$NUMPY_MAJOR" -lt 1 ] 2>/dev/null || \
             ([ "$NUMPY_MAJOR" -eq 1 ] && [ "$NUMPY_MINOR" -lt 14 ] 2>/dev/null) || \
             ([ "$NUMPY_MAJOR" -eq 1 ] && [ "$NUMPY_MINOR" -eq 14 ] && [ "${NUMPY_PATCH:-0}" -lt 5 ] 2>/dev/null); then
            NEED_UPGRADE=1
          fi
          if [ "$NEED_UPGRADE" -eq 1 ]; then
            PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            echo "[INFO] Astropy: numpy version $NUMPY_VER is too old (need >= 1.14.5), upgrading..." >&2
            if [ "$PY_VER" = "3.6" ]; then
              "$PY" -m pip install --upgrade --no-cache-dir "numpy>=1.14.5,<1.20" >/tmp/apr_astropy_upgrade_numpy_$$.log 2>&1 || true
            else
              "$PY" -m pip install --upgrade --no-cache-dir "numpy>=1.14.5" >/tmp/apr_astropy_upgrade_numpy_$$.log 2>&1 || true
            fi
          fi
        fi
      fi
    fi
  fi
fi

# If testbed Python is required (Astropy) but not found, fail fast
if [ "${APR_IS_ASTROPY:-0}" = "1" ] && [ -z "$PY" ]; then
  echo "[ERROR] Astropy: testbed conda python not found in image. Refusing to use /usr/bin/python3." >&2
  exit 2
fi

if [ -z "$PY" ]; then
  PY="$(pick_python_with_pytest || true)"
fi
if [ -z "$PY" ]; then
  PY="$(pick_python_with_pip || true)"
fi
if [ -z "$PY" ]; then
  echo "[ERROR] No usable python found in container." >&2
  exit 2
fi

echo "[INFO] Selected PY=$PY" >&2
"$PY" -V >&2 || true
"$PY" -c "import sys; print('sys.executable=', sys.executable)" >&2 || true

# scikit-learn: rebuild extensions if needed (shared helper)
apr_scikit_rebuild_if_needed() {
  if [ "${APR_IS_SCIKITLEARN:-0}" != "1" ]; then
    return 0
  fi
  # IMPORTANT: verify imports FROM THE SOURCE TREE (/testbed), not from a preinstalled sklearn
  _APR_ORIG_PYTHONPATH="${PYTHONPATH:-}"
  if [ -n "${_APR_ORIG_PYTHONPATH}" ]; then
    _APR_ORIG_PYTHONPATH=$(echo "${_APR_ORIG_PYTHONPATH}" | tr ':' '\n' | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
  fi
  export PYTHONPATH="/testbed${_APR_ORIG_PYTHONPATH:+:${_APR_ORIG_PYTHONPATH}}"
  echo "[INFO] scikit-learn: checking source-tree importability (PYTHONPATH starts with /testbed)..." >&2
  if (cd /tmp && "$PY" -c "import sklearn; import sklearn.utils.murmurhash; from sklearn.__check_build._check_build import check_build" >/dev/null 2>&1); then
    return 0
  fi
  if [ ! -f "/testbed/setup.py" ]; then
    echo "[ERROR] scikit-learn: /testbed/setup.py missing; cannot rebuild extensions" >&2
    return 1
  fi
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "[ERROR] scikit-learn: pip not available in $PY; cannot rebuild extensions" >&2
    return 1
  fi

  PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
  echo "[INFO] scikit-learn: compiled extensions missing; attempting in-place rebuild (PY=$PY PY_VER=$PY_VER)..." >&2

  TOOL_LOG="/tmp/apr_scikit_toolchain_$$.log"
  if [ "$PY_VER" = "3.6" ]; then
    if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pip==21.3.1" "setuptools==59.6.0" "wheel" "Cython<3" "numpy<1.20" "scipy<1.6" >"$TOOL_LOG" 2>&1; then
      echo "[ERROR] scikit-learn: failed to install build toolchain for Python 3.6" >&2
      tail -50 "$TOOL_LOG" >&2 || true
      return 1
    fi
    CYTHON_SPEC="Cython<3"
  elif [ "$PY_VER" = "3.9" ]; then
    if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "setuptools<60" "wheel" "Cython==3.0.10" "numpy<1.23" >"$TOOL_LOG" 2>&1; then
      echo "[ERROR] scikit-learn: failed to install build toolchain for Python 3.9" >&2
      tail -50 "$TOOL_LOG" >&2 || true
      return 1
    fi
    CYTHON_SPEC="Cython==3.0.10"
  else
    if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "setuptools<60" "wheel" "Cython<3.1" "numpy<1.23" >"$TOOL_LOG" 2>&1; then
      echo "[ERROR] scikit-learn: failed to install build toolchain for Python $PY_VER" >&2
      tail -50 "$TOOL_LOG" >&2 || true
      return 1
    fi
    CYTHON_SPEC="Cython<3.1"
  fi
  
  CYTHON_VER=$("$PY" -c "import sys; sys.path.insert(0, '$SITE_DIR'); import Cython; print(Cython.__version__)" 2>/dev/null || echo "")
  if [ -z "$CYTHON_VER" ]; then
    echo "[ERROR] scikit-learn: Cython not found in $SITE_DIR after installation" >&2
    return 1
  fi
  echo "[INFO] scikit-learn: installed Cython version: $CYTHON_VER (expected: $CYTHON_SPEC)" >&2
  
  export NPY_NUM_BUILD_JOBS=1
  export SETUPTOOLS_USE_DISTUTILS=stdlib
  
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$SITE_DIR:/testbed${PYTHONPATH:+:${PYTHONPATH}}"
  else
    export PYTHONPATH="$SITE_DIR:/testbed"
  fi

  BUILD_LOG="/tmp/apr_scikit_build_ext_$$.log"
  if ! (cd /testbed && "$PY" setup.py build_ext --inplace >"$BUILD_LOG" 2>&1); then
    echo "[ERROR] scikit-learn: build_ext failed" >&2
    tail -200 "$BUILD_LOG" >&2 || true
    return 1
  fi
  if ! (cd /tmp && "$PY" -c "import sklearn; import sklearn.utils.murmurhash; from sklearn.__check_build._check_build import check_build" >/dev/null 2>&1); then
    echo "[ERROR] scikit-learn: rebuild failed; tail of toolchain log:" >&2
    tail -200 "$TOOL_LOG" >&2 || true
    echo "[ERROR] scikit-learn: tail of build_ext log:" >&2
    tail -200 "$BUILD_LOG" >&2 || true
    return 1
  fi
  echo "[INFO] scikit-learn: rebuild succeeded; compiled extensions are now importable" >&2
  return 0
}

if [ "${APR_IS_SCIKITLEARN:-0}" = "1" ]; then
  if ! apr_scikit_rebuild_if_needed; then
    exit 2
  fi
fi

# ============================================================================
# End of common env script; project-specific config follows
# ============================================================================
"""


class SWEbenchVerifiedAdapter(DatasetAdapter):
    """
    Adapter that lets apr_new's G* loop operate on SWE-bench Verified instances.

    Design constraints (per user request):
    - Do not modify existing Defects4J code paths.
    - Use containerized evaluation (SWE-bench harness) for validation.
    - Avoid writing to $HOME (set HF caches under apr_new/).

    Important semantic mapping:
    - `pid` is treated as SWE-bench `instance_id` (string).
    - `bid` is ignored (kept only to satisfy the existing interface).
    """

    def _get_instance(self, instance_id: str) -> Dict[str, Any]:
        m = _load_verified_dataset_map()
        if instance_id not in m:
            raise KeyError(f"instance_id not found in SWE-bench Verified split=test: {instance_id}")
        return m[instance_id]

    def checkout(self, pid: str, bid: int, workdir: str) -> Dict[str, Any]:
        # Remove any leftover git lock file (e.g. from parallel runs)
        wd = Path(workdir)
        if wd.exists() and (wd / ".git" / "index.lock").exists():
            try:
                (wd / ".git" / "index.lock").unlink()
            except Exception:
                pass
        
        # Set git temp dir to avoid quota issues (GIT_TMPDIR / TMPDIR)
        git_tmpdir = os.environ.get("GIT_TMPDIR") or os.environ.get("TMPDIR")
        if not git_tmpdir:
            # Prefer TRACE_WORK_ROOT or /tmp (env or generic paths only)
            work_root = os.environ.get("TRACE_WORK_ROOT") or "/tmp/trace_work"
            git_tmpdir_candidates = [
                f"{work_root}/tmp",
                "/tmp",
            ]
            for candidate in git_tmpdir_candidates:
                try:
                    Path(candidate).mkdir(parents=True, exist_ok=True)
                    git_tmpdir = candidate
                    break
                except Exception:
                    continue
        
        # Set temp dir env for all git commands
        git_env = {}
        if git_tmpdir:
            git_env["GIT_TMPDIR"] = git_tmpdir
            git_env["TMPDIR"] = git_tmpdir
            git_env["TMP"] = git_tmpdir
            git_env["TEMP"] = git_tmpdir
        
        instance_id = pid
        inst = self._get_instance(instance_id)

        repo = inst["repo"]
        base_commit = inst["base_commit"]
        test_patch = inst.get("test_patch") or ""

        wd = Path(workdir)
        if _using_workdir_archives():
            # Archive mode: workdir must already be extracted by caller and contain .git
            if not wd.exists():
                return {"ok": False, "step": "workdir_missing", "workdir": workdir, "error": f"Archived workdir not found: {workdir}", "rc": 1}
            if not (wd / ".git").exists():
                return {"ok": False, "step": "workdir_not_git", "workdir": workdir, "error": f"Archived workdir is not a git repo (missing .git): {workdir}", "rc": 1}
        else:
            wd.mkdir(parents=True, exist_ok=True)

        # Clone if .git doesn't exist (debug/non-archive mode only)
        if not _using_workdir_archives() and not (wd / ".git").exists():
            # Check if workdir is empty (except for meta directory which is OK)
            if wd.exists():
                contents = list(wd.iterdir())
                # Filter out meta directory and hidden files that are OK
                non_git_contents = [item for item in contents if item.name != "meta" and not item.name.startswith(".")]
                if non_git_contents:
                    # Workdir exists but is not empty (and not a git repo)
                    # This can happen if a previous run failed or was interrupted
                    error_msg = f"Workdir exists but is not a git repository and contains non-meta files: {[str(c.name) for c in non_git_contents[:5]]}"
                    print(f"[ERROR] {error_msg}", flush=True)
                    return {"ok": False, "step": "git_clone", "repo": repo, "workdir": workdir, "error": error_msg, "rc": 1}
            # Strategy: Use blobless clone to reduce initial transfer size
            # This fetches only tree and commit objects, not file contents (fetched on-demand)
            # Much faster for large repos, especially with many binary files
            print(f"[CHECKOUT] Cloning {repo} with blobless filter (faster for large repos)...", flush=True)
            r = _run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", _github_https_url(repo), str(wd)],
                timeout=600,
                env=git_env
            )
            if r["rc"] != 0:
                # Fallback to shallow clone if blobless fails
                print(f"[WARN] Blobless clone failed, falling back to shallow clone...", flush=True)
                r = _run(
                    ["git", "clone", "--depth", "1", "--no-single-branch", _github_https_url(repo), str(wd)],
                    timeout=600,
                    env=git_env
                )
                if r["rc"] != 0:
                    return {"ok": False, "step": "git_clone", "repo": repo, "workdir": workdir, **r}

        # Skip checkout if workdir already at correct commit
        if (wd / ".git").exists():
            # Check if current commit equals base_commit
            r_current = _run(["git", "rev-parse", "HEAD"], cwd=str(wd), timeout=10, env=git_env)
            if r_current["rc"] == 0:
                current_commit = r_current["stdout"].strip()
                if current_commit == base_commit:
                    # Check if test_patch already applied (via .swebench_test_patch.diff existence and content)
                    test_patch_file = wd / ".swebench_test_patch.diff"
                    if test_patch_file.exists():
                        existing_patch = test_patch_file.read_text(encoding="utf-8")
                        if existing_patch.strip() == test_patch.strip():
                            print(f"[CHECKOUT] Workdir already at correct commit {base_commit[:8]}, skipping checkout", flush=True)
                            r_status = _run(["git", "status", "--porcelain"], cwd=str(wd), timeout=10, env=git_env)
                            if r_status["rc"] == 0 and not r_status["stdout"].strip():
                                # Workdir clean, skip checkout
                                return {"ok": True, "step": "checkout_skipped", "workdir": workdir, "base_commit": base_commit, "note": "Workdir already at correct commit, checkout skipped"}
                            else:
                                print(f"[CHECKOUT] Workdir has uncommitted changes, will reset and re-checkout", flush=True)
                        else:
                            print(f"[CHECKOUT] Test patch mismatch, will re-apply", flush=True)
                    else:
                        print(f"[CHECKOUT] Test patch file missing, will re-apply", flush=True)
                else:
                    print(f"[CHECKOUT] Current commit ({current_commit[:8]}) != base_commit ({base_commit[:8]}), will checkout", flush=True)
        
        # Fetch and checkout specific commit
        # For blobless clone, we need to fetch the commit to get the tree, then checkout
        print(f"[CHECKOUT] Fetching commit {base_commit[:8]}...", flush=True)
        r = _run(["git", "fetch", "origin", base_commit], cwd=str(wd), timeout=300, env=git_env)
        if r["rc"] != 0:
            # Try fetching with depth to get commit history if needed
            print(f"[WARN] Direct commit fetch failed, trying with depth...", flush=True)
            r = _run(["git", "fetch", "origin", base_commit, "--depth", "100"], cwd=str(wd), timeout=300, env=git_env)
            if r["rc"] != 0:
                # Last resort: unshallow and fetch all (slow but should work)
                print(f"[WARN] Depth fetch failed, unshallowing and fetching all (this may take a while)...", flush=True)
                _run(["git", "fetch", "--unshallow"], cwd=str(wd), timeout=600, env=git_env)
                r = _run(["git", "fetch", "origin", base_commit], cwd=str(wd), timeout=300, env=git_env)
                if r["rc"] != 0:
                    return {"ok": False, "step": "git_fetch", "workdir": workdir, **r}
        
        print(f"[CHECKOUT] Checking out commit {base_commit[:8]}...", flush=True)
        # Remove lock file again before checkout (may have been created in previous steps)
        git_dir = wd / ".git"
        if git_dir.exists():
            lock_file = git_dir / "index.lock"
            if lock_file.exists():
                try:
                    lock_file.unlink()
                    print(f"[CHECKOUT] Removed existing index.lock before checkout", flush=True)
                except Exception:
                    pass
        
        # Retry on disk quota or lock file errors; large repos (e.g. blobless) may need long checkout
        max_retries = 3
        checkout_timeout = 600  # 10 min for large-repo checkout
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"[CHECKOUT] Retry attempt {attempt + 1}/{max_retries} for checkout...", flush=True)
            r = _run(["git", "checkout", "--force", base_commit], cwd=str(wd), timeout=checkout_timeout, env=git_env)
            if r["rc"] == 0:
                break
            
            # Check for disk quota or lock file (retriable)
            stderr_lower = (r.get("stderr") or "").lower()
            if ("disk quota exceeded" in stderr_lower or "index.lock" in stderr_lower) and attempt < max_retries - 1:
                print(f"[WARN] Checkout failed (attempt {attempt + 1}/{max_retries}): {stderr_lower[:200]}", flush=True)
                # Remove lock file
                if git_dir.exists():
                    lock_file = git_dir / "index.lock"
                    if lock_file.exists():
                        try:
                            lock_file.unlink()
                            print(f"[CHECKOUT] Removed index.lock", flush=True)
                        except Exception:
                            pass
                
                time.sleep(2)  # Wait before retry
                continue
            
            # Non-retriable error
            return {"ok": False, "step": "git_checkout", "base_commit": base_commit, "workdir": workdir, **r}

        # Apply test_patch locally for agent context (so tests are visible to read_file/grep).
        # We do NOT run tests here; validation uses swebench harness in containers.
        #
        # IMPORTANT: We commit test_patch into a local commit so later `git diff` only
        # contains the model's fix patch (not test additions). The evaluation harness
        # applies `test_patch` internally, so we must avoid sending it in `model_patch`.
        if test_patch.strip():
            patch_file = wd / ".swebench_test_patch.diff"
            patch_file.write_text(test_patch, encoding="utf-8")
            # Try apply; if already applied, allow "patch does not apply" style failures by returning details.
            r_apply = _run(["git", "apply", "--verbose", str(patch_file)], cwd=str(wd), env=git_env)
            if r_apply["rc"] != 0:
                # Try with --reject (best-effort). Still ok for read-only context in some cases.
                r_apply2 = _run(["git", "apply", "--verbose", "--reject", str(patch_file)], cwd=str(wd), env=git_env)
                if r_apply2["rc"] != 0:
                    return {
                        "ok": False,
                        "step": "apply_test_patch",
                        "workdir": workdir,
                        "note": "test_patch failed to apply cleanly; cannot guarantee local context matches harness",
                        "first_try": r_apply,
                        "second_try": r_apply2,
                    }

            # Commit test_patch if there are changes
            # Add only modified tracked files (-u); avoid meta/, .bak, etc.
            r_status = _run(["git", "status", "--porcelain"], cwd=str(wd), env=git_env)
            if r_status["rc"] == 0 and (r_status.get("stdout") or "").strip():
                r_add = _run(["git", "add", "-u"], cwd=str(wd), env=git_env)
                if r_add["rc"] != 0:
                    print(f"[WARN] git add -u failed: {r_add.get('stderr', '')[:200]}", flush=True)
                    # Fallback: add only files modified by test_patch (from git status)
                    status_lines = (r_status.get("stdout") or "").strip().split("\n")
                    modified_files = []
                    for line in status_lines:
                        # porcelain format: " M file", "MM file", etc.
                        if line.startswith(" M") or line.startswith("MM") or line.startswith("M "):
                            parts = line.split(None, 1)
                            if len(parts) > 1:
                                modified_files.append(parts[1])
                    if modified_files:
                        for f in modified_files:
                            # Skip temp paths
                            if not any(f.startswith(exclude) for exclude in ["meta/", ".swebench_test_patch.diff", ".bak"]):
                                _run(["git", "add", f], cwd=str(wd), env=git_env)
                
                # Check if anything was staged
                r_status_after = _run(["git", "status", "--porcelain"], cwd=str(wd), env=git_env)
                staged_changes = [line for line in (r_status_after.get("stdout") or "").strip().split("\n") 
                                 if line and line[0] in "MADRC"]
                
                if not staged_changes:
                    print(f"[WARN] No changes staged for commit after git add", flush=True)
                    # No staged changes; test_patch may already be applied, continue
                else:
                    r_commit = _run(
                        [
                            "git",
                            "-c",
                            "user.name=swebench",
                            "-c",
                            "user.email=swebench@example.com",
                            "commit",
                            "-m",
                            "Apply SWE-bench test_patch (local)",
                            "--no-gpg-sign",
                        ],
                        cwd=str(wd),
                        env=git_env,
                    )
                    if r_commit["rc"] != 0:
                        return {
                            "ok": False,
                            "step": "commit_test_patch",
                            "workdir": workdir,
                            "stderr": r_commit.get("stderr", "")[-2000:],
                            "stdout": r_commit.get("stdout", "")[-2000:],
                        }

        # Try to find ABCoder index if USE_ABCODER_INDEX is enabled
        index_path = None
        if os.environ.get("USE_ABCODER_INDEX") == "1":
            try:
                from dataset.env_config import load_dataset_config, resolve_path_template
                dataset_cfg = load_dataset_config("swebench_verified")
                asts_dir_template = dataset_cfg.get("paths", {}).get("abcoder_asts_dir", "{scratch_base}/abcoder_asts/swebench_verified")
                scratch_base = dataset_cfg.get("paths", {}).get("scratch_base") or os.environ.get("TRACE_WORK_ROOT", "/tmp/trace_work")
                asts_dir = resolve_path_template(asts_dir_template, scratch_base=scratch_base)
                
                # Flat format: {instance_id}_index.json
                abcoder_index_path = asts_dir / f"{instance_id}_index.json"
                
                if abcoder_index_path.exists():
                    index_path = str(abcoder_index_path)
                    print(f"[CHECKOUT] ✓ ABCoder index found: {index_path}", flush=True)
            except Exception as e:
                print(f"[CHECKOUT] WARN: Failed to locate ABCoder index: {e}", flush=True)
        
        result = {"ok": True, "instance_id": instance_id, "repo": repo, "base_commit": base_commit, "workdir": workdir}
        if index_path:
            result["index_path"] = index_path
        return result

    def harness(
        self,
        pid: str,
        bid: int,
        workdir: str,
        meta_dir: str,
        full_log: str,
        trig_log: str,
        index_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run SWE-bench evaluation harness for this single instance.
        
        If APR_VERIFY_TEST_SUITE=1 is set, will also verify that the test suite
        can be collected and executed successfully in the container environment.
        This is useful for validating Django initialization fixes and ensuring
        test suites are executable before running the actual repair process.
        """
        # For apptainer runtime we do not require docker connectivity.
        if _swe_runtime() != "apptainer":
            # Early Docker connection test to catch configuration issues before checkout
            print("[HARNESS] Testing Docker connection...", flush=True)
            try:
                import docker
                docker_client = docker.from_env()
                docker_client.api.version()
                print("[HARNESS] Docker connection test passed", flush=True)
            except docker.errors.DockerException as e:
                error_msg = f"Docker connection failed: {e}. Please check DOCKER_HOST environment variable."
                print(f"[ERROR] {error_msg}", flush=True)
                return {
                    "ok": False,
                    "error": error_msg,
                    "step": "docker_connection_test",
                    "workdir": workdir,
                }
            except Exception as e:
                error_msg = f"Docker connection test failed: {e}"
                print(f"[ERROR] {error_msg}", flush=True)
                return {
                    "ok": False,
                    "error": error_msg,
                    "step": "docker_connection_test",
                    "workdir": workdir,
                }
        
        instance_id = pid
        print(f"[HARNESS] Loading instance metadata for {instance_id}...", flush=True)
        inst = self._get_instance(instance_id)

        Path(meta_dir).mkdir(parents=True, exist_ok=True)
        _write_json(Path(meta_dir) / "instance.json", inst)
        print(f"[HARNESS] Instance metadata saved to {meta_dir}/instance.json", flush=True)

        print(f"[HARNESS] Starting checkout (this may take a while for large repos)...", flush=True)
        checkout_start = time.time()
        co = self.checkout(pid, bid, workdir)
        checkout_elapsed = time.time() - checkout_start
        print(f"[HARNESS] Checkout completed in {checkout_elapsed:.1f}s", flush=True)
        
        if not co.get("ok"):
            # Print to stderr so stdout stays clean for verify_swe_environment JSON
            print(f"[HARNESS] Checkout failed: {co}", file=sys.stderr, flush=True)
            return {"pid": pid, "bid": bid, "workdir": workdir, "meta_dir": meta_dir, "ok": False, "checkout": co}

        # Provide context to the agent loop via HARNESS_RESULT
        result = {
            "pid": pid,
            "bid": bid,
            "workdir": workdir,
            "meta_dir": meta_dir,
            "dataset": "princeton-nlp/SWE-bench_Verified",
            "instance_id": instance_id,
            "repo": inst.get("repo"),
            "base_commit": inst.get("base_commit"),
            "problem_statement": inst.get("problem_statement"),
            "hints_text": inst.get("hints_text"),
            "note": "Validation is performed via swebench.harness.run_evaluation (containerized).",
        }
        
        # Include index_path if available from checkout
        if "index_path" in co:
            result["index_path"] = co["index_path"]
            print(f"[HARNESS] Index path included: {co['index_path']}", flush=True)
        
        # Optional: Verify PASS_TO_PASS tests can be executed (if APR_VERIFY_TEST_SUITE=1)
        # This runs BEFORE RED gate to ensure environment is functional.
        # Only runs PASS_TO_PASS (not FAIL_TO_PASS) to avoid flakiness before RED gate.
        # Full regression (_validate_apptainer) runs FAIL_TO_PASS + PASS_TO_PASS after GREEN gate.
        if os.environ.get("APR_VERIFY_TEST_SUITE", "0") in ("1", "true", "yes"):
            print(f"[HARNESS] APR_VERIFY_TEST_SUITE=1: Verifying instance tests can be collected...", flush=True)
            test_suite_result = self._verify_test_suite(instance_id=instance_id, workdir=workdir, inst=inst)
            result["test_suite_verification"] = test_suite_result
            if not test_suite_result.get("ok", False):
                print(f"[HARNESS] WARN: Test suite verification failed: {test_suite_result.get('error', 'Unknown error')}", flush=True)
                # Don't fail harness, just warn - this is for diagnostic purposes
            else:
                print(f"[HARNESS] ✓ Test suite verification passed: {test_suite_result.get('test_count', 0)} instance tests collectible", flush=True)
        
        return result

    def validate(self, pid: str, bid: int, workdir: str, meta_dir: str, full_log: str, trig_log: str) -> Dict[str, Any]:
        """
        Run SWE-bench evaluation harness for this single instance.

        We assume `apply_patch_fn` already applied the candidate patch to the local repo.
        For evaluation harness, we need to pass the patch text as `model_patch`.

        NOTE: main_ablation currently does not pass `patch_text` into adapter.validate.
        To avoid modifying existing code, we derive `model_patch` from `git diff` of the
        local checkout. Since `checkout()` commits the SWE-bench `test_patch` into HEAD,
        this diff should contain only the model's fix patch.
        """
        if _swe_runtime() == "apptainer":
            return self._validate_apptainer(pid=pid, workdir=workdir, meta_dir=meta_dir)

        # Early Docker connection test to catch configuration issues early
        try:
            import docker
            docker_client = docker.from_env()
            docker_client.api.version()
            print("[HARNESS] Docker connection test passed", flush=True)
        except docker.errors.DockerException as e:
            error_msg = f"Docker connection failed: {e}. Please check DOCKER_HOST environment variable."
            print(f"[ERROR] {error_msg}", flush=True)
            return {
                "passed": False,
                "error": error_msg,
                "instance_id": pid,
                "rc": -1,
                "stdout": "",
                "stderr": str(e),
            }
        except Exception as e:
            error_msg = f"Docker connection test failed: {e}"
            print(f"[ERROR] {error_msg}", flush=True)
            return {
                "passed": False,
                "error": error_msg,
                "instance_id": pid,
                "rc": -1,
                "stdout": "",
                "stderr": str(e),
            }
        
        instance_id = pid
        inst = self._get_instance(instance_id)

        wd = Path(workdir)
        if not (wd / ".git").exists():
            return {"passed": False, "error": f"workdir is not a git repo: {workdir}", "instance_id": instance_id}

        r_diff = _run(["git", "-c", "core.fileMode=false", "diff"], cwd=str(wd))
        if r_diff["rc"] != 0:
            return {"passed": False, "error": "failed to compute git diff", "instance_id": instance_id, **r_diff}
        patch_text = r_diff.get("stdout") or ""
        if not patch_text.strip():
            return {"passed": False, "error": "empty patch (git diff is empty)", "instance_id": instance_id}

        # Create predictions file for swebench harness.
        preds = [
            {
                "instance_id": instance_id,
                "model_name_or_path": "apr_new_g5",
                "model_patch": patch_text,
            }
        ]
        meta = Path(meta_dir)
        meta.mkdir(parents=True, exist_ok=True)
        preds_path = meta / "predictions.json"
        _write_json(preds_path, preds)

        # Run harness in a dedicated cwd so it writes logs under that directory.
        # IMPORTANT (Podman rootless on shared FS):
        # If cwd is under a setgid directory (e.g. group=prjsXXXX), files created by
        # SWE-bench (patch.diff/eval.sh) inherit that GID. The docker SDK then tries
        # to set that GID inside the container during put_archive, which fails with:
        #   lchown ... invalid argument
        # because rootless podman often has a 1-entry gidmap.
        #
        # Workaround: keep harness cwd under a non-setgid directory whose group is
        # the user's primary group.
        base = Path(os.environ.get("SWEBENCH_HARNESS_CWD", str(APR_ROOT / ".swebench_harness")))
        harness_cwd = base / instance_id
        harness_cwd.mkdir(parents=True, exist_ok=True)

        # The harness uses docker SDK (docker.from_env). We rely on the caller to
        # export DOCKER_HOST to podman socket and to have swebench installed in venv.
        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "princeton-nlp/SWE-bench_Verified",
            "--split",
            "test",
            "--instance_ids",
            instance_id,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            "1",
            "--run_id",
            f"apr_new_g5_{instance_id}",
        ]
        r = _run(cmd, cwd=str(harness_cwd))

        # Swebench writes a run report to logs/run_evaluation/...; we don't parse it deeply here.
        passed = r["rc"] == 0
        return {
            "passed": passed,
            "instance_id": instance_id,
            "repo": inst.get("repo"),
            "harness_cwd": str(harness_cwd),
            "cmd": cmd,
            "rc": r["rc"],
            "stdout": (r.get("stdout") or "")[-2000:],
            "stderr": (r.get("stderr") or "")[-2000:],
        }

    def check_compile(self, workdir: str) -> Dict[str, Any]:
        # SWE-bench supports many languages; we do not provide a fast compile-only gate here.
        return {"ok": True, "skipped": True, "reason": "compile gate not supported for swebench_verified"}

    def _verify_test_suite(self, *, instance_id: str, workdir: str, inst: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that the instance's PASS_TO_PASS tests can be executed in the container environment.
        This runs BEFORE the RED gate (in harness phase) to ensure the environment is functional.
        
        IMPORTANT: Only runs PASS_TO_PASS (not FAIL_TO_PASS) to avoid:
        - Running failing tests before RED gate (which could introduce flakiness)
        - If PASS tests can run, regression (which runs FAIL_TO_PASS + PASS_TO_PASS) won't hang
        
        Full validation (_validate_apptainer) runs after GREEN gate and executes both
        FAIL_TO_PASS + PASS_TO_PASS for complete regression testing.
        
        Returns a dict with:
        - ok: bool - whether test suite verification passed
        - test_count: int - number of tests that ran successfully
        - error: str - error message if verification failed
        - duration: float - time taken for verification
        """
        start_time = time.time()

        # ------------------------------------------------------------------
        # NEW (robust): verify FAIL_TO_PASS + PASS_TO_PASS can actually RUN
        # using the SAME runner as full validation (green/regression path).
        #
        # Rationale:
        # - pytest collection is not equivalent to Django's own test runner and
        #   can trigger unrelated collection-time failures (GIS/GDAL, app configs).
        # - pip/pytest installs during "verification" can hang; full validation
        #   must not depend on that.
        #
        # So we execute each instance test via `run_one_test` (apptainer path),
        # and treat test failures as OK (env works), but treat runner failures
        # (ran=False / dependency_error / timeout) as environment failures.
        # ------------------------------------------------------------------
        try:
            fail_to_pass = _parse_json_list(inst.get("FAIL_TO_PASS", "[]"))
            pass_to_pass = _parse_json_list(inst.get("PASS_TO_PASS", "[]"))
            all_tests = fail_to_pass + pass_to_pass
            if not all_tests:
                return {
                    "ok": False,
                    "error": "No tests specified in instance (FAIL_TO_PASS + PASS_TO_PASS)",
                    "duration": time.time() - start_time,
                }

            suite_dir = APR_ROOT / "logs" / "swebench_verified" / instance_id / "suite_verify"
            suite_dir.mkdir(parents=True, exist_ok=True)

            old_timeout = os.environ.get("APR_TEST_TIMEOUT_SECONDS")
            os.environ["APR_TEST_TIMEOUT_SECONDS"] = os.environ.get("APR_VERIFY_TEST_SUITE_TIMEOUT_SECONDS", "300")

            # Keep verification cheap: always include FAIL_TO_PASS, plus a limited number
            # of PASS_TO_PASS to sanity-check the environment. Full validation will run all.
            max_tests = None
            try:
                max_tests = int(os.environ.get("APR_VERIFY_TEST_SUITE_MAX_TESTS", "10"))
            except Exception:
                max_tests = 10
            # Avoid running FAIL_TO_PASS here (it runs before RED gate and can introduce flakiness).
            # For env sanity, PASS_TO_PASS is sufficient: if PASS tests can run, regression won't hang.
            if max_tests is not None and max_tests > 0:
                tests_to_run = pass_to_pass[:max_tests]
            else:
                tests_to_run = pass_to_pass
            if not tests_to_run:
                # Fallback: if no PASS_TO_PASS exists, run a single FAIL_TO_PASS as a smoke test.
                tests_to_run = fail_to_pass[:1]
            env_failures: list[dict[str, Any]] = []
            ran_count = 0
            try:
                for i, t in enumerate(tests_to_run):
                    log_file = str(suite_dir / f"test_{i}.log")
                    r = self.run_one_test(workdir=workdir, test_name=t, log_file=log_file)
                    if not r.get("ran", False):
                        env_failures.append({"test": t, "error": r.get("error") or r.get("stderr") or r.get("stdout"), "logfile": log_file})
                        continue
                    if r.get("timeout", False):
                        env_failures.append({"test": t, "error": "timeout", "logfile": log_file})
                        continue
                    if r.get("dependency_error", False):
                        env_failures.append({"test": t, "error": r.get("error") or "dependency/module import error", "logfile": log_file})
                        continue
                    ran_count += 1
            finally:
                if old_timeout is None:
                    os.environ.pop("APR_TEST_TIMEOUT_SECONDS", None)
                else:
                    os.environ["APR_TEST_TIMEOUT_SECONDS"] = old_timeout

            duration = time.time() - start_time
            if env_failures:
                return {
                    "ok": False,
                    "error": f"Environment errors in {len(env_failures)}/{len(tests_to_run)} verified tests. See logs in {suite_dir}",
                    "duration": duration,
                    "test_count": ran_count,
                    "total": len(tests_to_run),
                    "log_dir": str(suite_dir),
                    "failures": env_failures[:5],
                }
            return {"ok": True, "test_count": ran_count, "duration": duration, "log_dir": str(suite_dir), "total": len(tests_to_run)}
        except Exception as e:
            return {"ok": False, "error": str(e), "duration": time.time() - start_time}

    def _validate_apptainer(self, *, pid: str, workdir: str, meta_dir: str) -> Dict[str, Any]:
        """
        Best-effort Apptainer validation for a single instance.

        We run only the tests specified in the instance's FAIL_TO_PASS and PASS_TO_PASS
        fields, not the entire test suite.
        """
        # Auto-cleanup Apptainer cache if enabled (prevent disk space explosion)
        if os.environ.get("APR_APPTAINER_AUTO_CLEANUP", "1") == "1":
            try:
                cleanup_script = APR_ROOT / "bin" / "cleanup_apptainer_cache_auto.sh"
                if cleanup_script.exists():
                    subprocess.run(
                        ["bash", str(cleanup_script)],
                        capture_output=True,
                        timeout=60,
                        check=False
                    )
            except Exception:
                pass  # Ignore cleanup errors, don't block validation
        
        instance_id = pid
        inst = self._get_instance(instance_id)

        wd = Path(workdir)
        if not (wd / ".git").exists():
            return {"passed": False, "error": f"workdir is not a git repo: {workdir}", "instance_id": instance_id, "rc": 1, "stdout": "", "stderr": ""}

        repo = inst.get("repo") or ""

        # Parse test names from instance
        fail_to_pass = _parse_json_list(inst.get("FAIL_TO_PASS", "[]"))
        pass_to_pass = _parse_json_list(inst.get("PASS_TO_PASS", "[]"))
        
        all_tests = fail_to_pass + pass_to_pass
        if not all_tests:
            return {"passed": False, "error": "No tests specified in instance", "instance_id": instance_id, "rc": 1, "stdout": "", "stderr": ""}

        # Use SWE-bench testbed instance image (prefer pre-pulled SIF if available)
        image_name = _swebench_instance_image(instance_id=instance_id, arch="x86_64", tag="latest", namespace="swebench")
        sif = _swebench_sif_path()
        if sif:
            image = sif
            print(f"[RUN_TEST] Using pre-pulled SIF image: {sif}", flush=True)
        else:
            image = f"docker://{image_name}"
        bind = f"{wd}:/testbed"

        test_patch = inst.get("test_patch", "") or ""
        base_commit = inst.get("base_commit") or ""
        directives = _parse_test_directives_from_patch(test_patch)
        test_files = "\n".join(directives)

        # Run each expected-to-pass test individually; fail fast on first failure.
        tests_list_str = ", ".join(all_tests)

        # Use shared test-environment script base
        script = (_build_test_environment_script_base() + r"""

# pylint-dev: per-instance Python switch.
# For pylint-dev__pylint-8898 specifically, tests require Python >= 3.7
# (e.g. `from __future__ import annotations`). If current PY is too old, bootstrap
# a separate miniconda env and switch PY to it. Keep it instance-scoped to avoid
# affecting other pylint-dev instances that fail for non-version reasons.
if [ "${APR_IS_PYLINTDEV:-0}" = "1" ] && [ "${APR_INSTANCE_ID:-}" = "pylint-dev__pylint-8898" ] && [ -f "/miniconda.sh" ]; then
  if [ -z "${APR_REQUIRED_PYTHON_VERSION:-}" ]; then
    APR_REQUIRED_PYTHON_VERSION="3.7"
  fi
  CUR_MM=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
  CUR_MAJ=${CUR_MM%%.*}
  CUR_MIN=${CUR_MM#*.}
  REQ_MAJ=${APR_REQUIRED_PYTHON_VERSION%%.*}
  REQ_MIN=${APR_REQUIRED_PYTHON_VERSION#*.}
  if [ -n "${CUR_MAJ:-}" ] && [ -n "${CUR_MIN:-}" ] && [ -n "${REQ_MAJ:-}" ] && [ -n "${REQ_MIN:-}" ] && \
     ( [ "$CUR_MAJ" -lt "$REQ_MAJ" ] 2>/dev/null || ( [ "$CUR_MAJ" -eq "$REQ_MAJ" ] 2>/dev/null && [ "$CUR_MIN" -lt "$REQ_MIN" ] 2>/dev/null ) ); then
    echo "[INFO] pylint-dev: Python too old ($CUR_MM), bootstrapping ${APR_REQUIRED_PYTHON_VERSION} via miniconda..." >&2
    CONDA_BASE="/tmp/apr_miniconda3"
    PY_REQ="${CONDA_BASE}/envs/apr_py${APR_REQUIRED_PYTHON_VERSION//./}/bin/python"
    if [ ! -x "$PY_REQ" ]; then
      if [ ! -x "${CONDA_BASE}/bin/conda" ]; then
        bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_miniconda_install.log 2>&1 || true
      fi
      if [ -x "${CONDA_BASE}/bin/conda" ]; then
        "${CONDA_BASE}/bin/conda" create -y -p "${CONDA_BASE}/envs/apr_py${APR_REQUIRED_PYTHON_VERSION//./}" "python=${APR_REQUIRED_PYTHON_VERSION}" pip >/tmp/apr_miniconda_create_py${APR_REQUIRED_PYTHON_VERSION//./}.log 2>&1 || true
      fi
    fi
    if [ -x "$PY_REQ" ]; then
      PY="$PY_REQ"
      echo "[INFO] pylint-dev: switched to PY=$PY" >&2
      "$PY" -V >&2 || true
      "$PY" -c "import sys; print('sys.executable=', sys.executable)" >&2 || true
      if "$PY" -m pip --version >/dev/null 2>&1; then
        echo "[INFO] pylint-dev: upgrading pip..." >&2
        "$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
      fi
    else
      echo "[WARN] pylint-dev: failed to bootstrap Python ${APR_REQUIRED_PYTHON_VERSION}; continuing with PY=$PY ($CUR_MM)" >&2
      tail -200 /tmp/apr_miniconda_create_py${APR_REQUIRED_PYTHON_VERSION//./}.log 2>/dev/null || true
    fi
  fi
fi

# Check if selected Python version matches required version (from SWE-bench spec)
# If version mismatch and we're using system Python, try to bootstrap correct version
if [ -n "${APR_REQUIRED_PYTHON_VERSION:-}" ] && [ "$PY" = "/usr/bin/python3" ] && [ -f "/miniconda.sh" ]; then
  CURRENT_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
  if [ -n "$CURRENT_PY_VER" ] && [ "$CURRENT_PY_VER" != "${APR_REQUIRED_PYTHON_VERSION}" ]; then
    echo "[INFO] Python version mismatch: required=${APR_REQUIRED_PYTHON_VERSION}, current=$CURRENT_PY_VER, bootstrapping..." >&2
    CONDA_BASE="/tmp/apr_miniconda3"
    PY_REQ="${CONDA_BASE}/envs/apr_py${APR_REQUIRED_PYTHON_VERSION//./}/bin/python"
    if [ ! -x "$PY_REQ" ]; then
      echo "[INFO] Bootstrapping miniconda Python ${APR_REQUIRED_PYTHON_VERSION} env (first-time)..." >&2
      if [ ! -x "${CONDA_BASE}/bin/conda" ]; then
        bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_miniconda_install.log 2>&1 || true
      fi
      if [ -x "${CONDA_BASE}/bin/conda" ]; then
        "${CONDA_BASE}/bin/conda" create -y -p "${CONDA_BASE}/envs/apr_py${APR_REQUIRED_PYTHON_VERSION//./}" "python=${APR_REQUIRED_PYTHON_VERSION}" pip >/tmp/apr_miniconda_create_py${APR_REQUIRED_PYTHON_VERSION//./}.log 2>&1 || true
      fi
    fi
    if [ -x "$PY_REQ" ]; then
      PY="$PY_REQ"
      echo "[INFO] Switched to required Python version: PY=$PY" >&2
      "$PY" -V >&2 || true
      "$PY" -c "import sys; print('sys.executable=', sys.executable)" >&2 || true
      # Upgrade pip to ensure compatibility with modern packages
      if "$PY" -m pip --version >/dev/null 2>&1; then
        echo "[INFO] Upgrading pip for Python ${APR_REQUIRED_PYTHON_VERSION}..." >&2
        "$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
      fi
    else
      echo "[WARN] Failed to bootstrap Python ${APR_REQUIRED_PYTHON_VERSION}; continuing with system Python $CURRENT_PY_VER" >&2
      tail -200 /tmp/apr_miniconda_create_py${APR_REQUIRED_PYTHON_VERSION//./}.log 2>/dev/null || true
    fi
  fi
fi

# pytest-dev at older commits is incompatible with Python 3.10's import hook API
# (AssertionRewritingHook missing find_spec). When the image lacks a conda testbed,
# bootstrap a local miniconda env with Python 3.9 and use it for pytest-dev only.
# Note: This is a fallback if APR_REQUIRED_PYTHON_VERSION was not set or bootstrap failed.
if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ "$PY" = "/usr/bin/python3" ] && [ -f "/miniconda.sh" ]; then
  CONDA_BASE="/tmp/apr_miniconda3"
  PY39="${CONDA_BASE}/envs/apr_py39/bin/python"
  if [ ! -x "$PY39" ]; then
    echo "[INFO] pytest-dev: bootstrapping miniconda Python 3.9 env (first-time)..." >&2
    if [ ! -x "${CONDA_BASE}/bin/conda" ]; then
      bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_miniconda_install.log 2>&1 || true
    fi
    if [ -x "${CONDA_BASE}/bin/conda" ]; then
      "${CONDA_BASE}/bin/conda" create -y -p "${CONDA_BASE}/envs/apr_py39" python=3.9 pip >/tmp/apr_miniconda_create.log 2>&1 || true
    fi
  fi
  if [ -x "$PY39" ]; then
    PY="$PY39"
    echo "[INFO] pytest-dev: switched to PY=$PY" >&2
    "$PY" -V >&2 || true
    "$PY" -c "import sys; print('sys.executable=', sys.executable)" >&2 || true
    # Upgrade pip to ensure compatibility
    if "$PY" -m pip --version >/dev/null 2>&1; then
      echo "[INFO] pytest-dev: upgrading pip..." >&2
      "$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
    fi
  else
    echo "[WARN] pytest-dev: failed to bootstrap Python 3.9 env; continuing with system Python" >&2
    tail -200 /tmp/apr_miniconda_create.log 2>/dev/null || true
  fi
fi

# Install SWE-bench spec pip packages deterministically.
# For Astropy and pylint-dev, these deps are NOT optional; if this step fails we treat it as env error.
if "$PY" -m pip --version >/dev/null 2>&1; then
  # CRITICAL: Support both file-based and direct variable formats
  if [ -n "${APR_PIP_PACKAGES_FILE:-}" ] && [ -f "$APR_PIP_PACKAGES_FILE" ]; then
    # File-based format: already one package per line
    cp "$APR_PIP_PACKAGES_FILE" /tmp/apr_pip_pkgs.txt || true
  elif [ -n "${APR_PIP_PACKAGES:-}" ]; then
    # Direct variable format: space-separated, convert to lines
    echo "$APR_PIP_PACKAGES" | tr ' ' '\n' | sed '/^$/d' > /tmp/apr_pip_pkgs.txt || true
  fi
  if [ -s /tmp/apr_pip_pkgs.txt ]; then
    PIP_PKGS_LOG="/tmp/apr_pip_pkgs_install_$$.log"
    echo "[INFO] Installing pip packages from APR_PIP_PACKAGES into $SITE_DIR..." >&2
    echo "[DEBUG] Pip packages file content (first 10 lines):" >&2
    head -10 /tmp/apr_pip_pkgs.txt >&2 || true
    if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" -r /tmp/apr_pip_pkgs.txt >"$PIP_PKGS_LOG" 2>&1; then
      if [ "${APR_IS_ASTROPY:-0}" = "1" ] || [ "${APR_IS_PYLINTDEV:-0}" = "1" ]; then
        echo "[ERROR] Failed to install required pip_packages into $SITE_DIR. Tail:" >&2
        tail -200 "$PIP_PKGS_LOG" >&2 || true
        exit 2
      fi
      # Non-astropy: best-effort only.
      true
    fi
  fi
fi

# Ensure pytest import works (prefer preinstalled; otherwise install into /tmp/apr_site).
if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "[ERROR] pytest is missing and pip is not available in $PY" >&2
    exit 2
  fi
  if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ -n "${PYTEST_SRC_DIR:-}" ]; then
    # For pytest-dev: do NOT install external pytest; instead satisfy in-tree deps.
    echo "[INFO] pytest-dev: pytest missing, bootstrapping in-tree deps into $SITE_DIR..." >&2
    # Ensure PYTHONPATH includes SITE_DIR so newly installed modules can be found
    if [ -n "${PYTHONPATH:-}" ]; then
      PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
      export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
    else
      export PYTHONPATH="$SITE_DIR"
    fi
    # Pre-install common pytest core dependencies to speed up bootstrap
    echo "[INFO] pytest-dev: pre-installing common pytest dependencies..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "atomicwrites" "iniconfig" "pluggy" "py" "packaging" "attrs" "more-itertools" "tomli" "exceptiongroup" >/tmp/apr_pytest_preinstall_$$.log 2>&1 || true
    # Now iteratively install missing dependencies until pytest can be imported
    MAX_ITER=20
    INSTALLED_MODS=""
    for _i in $(seq 1 $MAX_ITER); do
      OUT=$("$PY" -c "import pytest" 2>&1 || true)
      if [ -z "$OUT" ]; then
        echo "[INFO] pytest-dev: pytest import successful after $_i iteration(s)" >&2
        break
      fi
      MOD=$("$PY" - <<'PY'
import re,sys
t=sys.stdin.read()
m=re.search(r"No module named ['\\\"]([^'\\\"]+)['\\\"]", t)
print(m.group(1) if m else "")
PY
<<<"$OUT" 2>/dev/null || true)
      if [ -z "${MOD:-}" ]; then
        echo "[WARN] pytest-dev: could not resolve missing module from import error (iteration $_i):" >&2
        echo "$OUT" >&2
        # If we can't extract module name, try to continue anyway
        if [ "$_i" -ge 10 ]; then
          echo "[ERROR] pytest-dev: too many iterations without progress, giving up" >&2
          break
        fi
        continue
      fi
      # Skip if we already tried to install this module
      if echo "$INSTALLED_MODS" | grep -q "^${MOD}$"; then
        echo "[WARN] pytest-dev: module $MOD already installed but still missing, may have dependency issue" >&2
        if [ "$_i" -ge 15 ]; then
          echo "[ERROR] pytest-dev: stuck in dependency loop, giving up" >&2
          break
        fi
        continue
      fi
      INSTALLED_MODS="${INSTALLED_MODS}${INSTALLED_MODS:+$'\n'}${MOD}"
      echo "[INFO] pytest-dev: installing missing module: $MOD (iteration $_i)" >&2
      INSTALL_LOG="/tmp/apr_pytest_install_${MOD}_$$.log"
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "$MOD" >"$INSTALL_LOG" 2>&1; then
        echo "[WARN] pytest-dev: failed to install $MOD, log:" >&2
        tail -20 "$INSTALL_LOG" >&2 || true
      fi
      # Ensure PYTHONPATH is still set correctly after installation
      if [ -n "${PYTHONPATH:-}" ]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
        export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
      else
        export PYTHONPATH="$SITE_DIR"
      fi
    done
  else
    # Select pytest version based on Python version
    # Python 3.6: pytest 7.0.1 (last version supporting Python 3.6)
    # Python 3.7+: pytest 7.4.4
    PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    PYTEST_VER="7.4.4"
    if [ "$PY_VER" = "3.6" ]; then
      PYTEST_VER="7.0.1"
      echo "[INFO] Python 3.6 detected, using pytest ${PYTEST_VER} (compatible version)..." >&2
    fi
    INSTALL_LOG="/tmp/pytest_install_$$.log"
    echo "[INFO] Installing pytest==${PYTEST_VER} into $SITE_DIR (container /tmp)..." >&2
    # For Python 3.6, pytest 7.0.1 requires specific dependencies
    if [ "$PY_VER" = "3.6" ]; then
      # Install pytest dependencies explicitly for Python 3.6 compatibility
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "packaging>=17.1" "attrs>=17.4.0" "more-itertools>=4.0.0" "pluggy>=0.12,<1.0" "py>=1.5.0" "setuptools>=40.0" "six>=1.10.0" "toml>=0.9.4" "pytest==${PYTEST_VER}" >"$INSTALL_LOG" 2>&1; then
        echo "[ERROR] Failed to install pytest==${PYTEST_VER} with dependencies. Tail of pip log:" >&2
        tail -200 "$INSTALL_LOG" >&2 || true
        exit 2
      fi
    else
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pytest==${PYTEST_VER}" >"$INSTALL_LOG" 2>&1; then
        echo "[ERROR] Failed to install pytest==${PYTEST_VER}. Tail of pip log:" >&2
        tail -200 "$INSTALL_LOG" >&2 || true
        exit 2
      fi
    fi
    # Ensure PYTHONPATH is set immediately after installation (avoid duplicates)
    if [ -n "${PYTHONPATH:-}" ]; then
      # Remove duplicates and ensure SITE_DIR is first
      PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
      export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
    else
      export PYTHONPATH="$SITE_DIR"
    fi
  fi
fi

# Re-export to ensure our chosen ordering persists after any installs above (avoid duplicates)
if [ -n "${PYTEST_SRC_DIR:-}" ]; then
  # Remove duplicates and ensure PYTEST_SRC_DIR is first, then SITE_DIR
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$PYTEST_SRC_DIR$" | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$PYTEST_SRC_DIR:$SITE_DIR:${PYTHONPATH}"
  else
    export PYTHONPATH="$PYTEST_SRC_DIR:$SITE_DIR"
  fi
else
  # Remove duplicates and ensure SITE_DIR is first
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
  else
    export PYTHONPATH="$SITE_DIR"
  fi
fi

# Debug: check if pytest files exist after installation
if [ -d "$SITE_DIR" ]; then
  if [ -f "$SITE_DIR/pytest.py" ] || [ -d "$SITE_DIR/pytest" ] || [ -d "$SITE_DIR/_pytest" ]; then
    echo "[DEBUG] pytest files found in $SITE_DIR" >&2
  else
    echo "[DEBUG] pytest files NOT found in $SITE_DIR, listing contents:" >&2
    ls -la "$SITE_DIR" | head -20 >&2 || true
  fi
fi

# Try importing pytest with detailed error output
PYTEST_IMPORT_OUT="/tmp/pytest_import_check_$$.log"
if ! "$PY" -c "import pytest; print('pytest_version=', getattr(pytest,'__version__','unknown'))" >"$PYTEST_IMPORT_OUT" 2>&1; then
  echo "[ERROR] CRITICAL: pytest still not importable after setup. PY=$PY SITE_DIR=$SITE_DIR PYTHONPATH=$PYTHONPATH" >&2
  echo "[ERROR] Import error output:" >&2
  cat "$PYTEST_IMPORT_OUT" >&2 || true
  # Try to diagnose: check if pytest is in site-packages
  "$PY" -c "import sys; print('sys.path:', sys.path)" >&2 || true
  exit 2
fi
rm -f "$PYTEST_IMPORT_OUT" 2>/dev/null || true

install_missing_module_from_file() {
  # $1: path to captured pytest output
  local out="$1"
  local mod=""
  mod=$("$PY" -c "import re,sys; t=open(sys.argv[1],'r',encoding='utf-8',errors='ignore').read(); m=re.search(r\"No module named ['\\\"]([^'\\\"]+)['\\\"]\", t); print(m.group(1) if m else '')" "$out" 2>/dev/null || true)
  if [ -z "$mod" ]; then
    if "$PY" -c "import sys; t=open(sys.argv[1],'r',encoding='utf-8',errors='ignore').read().lower(); sys.exit(0 if 'depends on mpmath' in t else 1)" "$out" 2>/dev/null; then
      mod="mpmath"
    fi
  fi
  if [ -n "$mod" ]; then
    echo "[INFO] Installing missing module: $mod" >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "$mod" >/dev/null 2>&1 || return 1
    return 0
  fi
  return 1
}

# Fix for pylint and other projects that need 'py' package (pytest's legacy dependency).
# Some test modules import 'py._path.local' which requires the 'py' package.
if ! "$PY" -c "import py._path" >/dev/null 2>&1; then
  echo "[INFO] Installing 'py' package (required by some test modules like pylint)..." >&2
  "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" py >/dev/null 2>&1 || true
fi

# pytest-dev__pytest-5262: Install hypothesis dependency
# Some pytest-dev tests require hypothesis (e.g., testing/python/metafunc.py)
if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ "${APR_INSTANCE_ID:-}" = "pytest-dev__pytest-5262" ]; then
  if ! "$PY" -c "import hypothesis" >/dev/null 2>&1; then
    echo "[INFO] pytest-dev__pytest-5262: Installing hypothesis (required by testing/python/metafunc.py)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" hypothesis >/dev/null 2>&1 || true
  fi
fi

# Fix for Pallets/Flask projects that need flask module.
# For pallets/flask, flask is the project itself, so we should use the source checkout.
# However, if flask is missing or incompatible, we need to install a compatible version.
# Flask 2.0-2.3 has request_ctx in flask.globals, but Flask 3.0+ removed it.
# Strategy: Try to use flask from /testbed first (via pip install -e .), then install compatible version if needed.
if [ "${APR_IS_PALLETS:-0}" = "1" ]; then
  # First, try to use flask from /testbed (the project itself)
  # This should work after "pip install -e ." is run later in the script
  # But we check early to see if we need to install a fallback version
  FLASK_AVAILABLE=0
  if "$PY" -c "import flask" >/dev/null 2>&1; then
    # Check if request_ctx is available (required by test code)
    if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
      FLASK_AVAILABLE=1
      echo "[INFO] Flask with request_ctx available (from /testbed or existing install)" >&2
    else
      echo "[INFO] Flask found but request_ctx not available, may need compatible version..." >&2
    fi
  fi
  
  # If flask is not available or request_ctx is missing, install a compatible version
  # Flask version compatibility:
  # - Flask 2.3.x: requires Python 3.8+, has request_ctx
  # - Flask 2.0.x: requires Python 3.6+, has _request_ctx_stack (deprecated) but may not have request_ctx
  # - Flask 2.1+: has request_ctx, requires Python 3.7+
  # Strategy: Detect Python version and install appropriate Flask version
  if [ "$FLASK_AVAILABLE" -eq 0 ]; then
    PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    PY_MAJOR=${PY_VER%%.*}
    PY_MINOR=${PY_VER#*.}
    
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 8 ]; then
      # Python 3.8+: Install Flask 2.3.x (has request_ctx)
      echo "[INFO] Installing Flask 2.3.x (Python $PY_VER, has request_ctx)..." >&2
      if "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "flask>=2.3.0,<2.4.0" >/tmp/apr_flask_install_$$.log 2>&1; then
        if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
          echo "[INFO] Flask 2.3.x installed successfully with request_ctx support" >&2
        else
          echo "[WARN] Flask 2.3.x installed but request_ctx still not available" >&2
        fi
      else
        echo "[WARN] Failed to install Flask 2.3.x, log:" >&2
        tail -20 /tmp/apr_flask_install_$$.log >&2 || true
      fi
    elif [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -eq 7 ]; then
      # Python 3.7: Install Flask 2.1+ (has request_ctx)
      echo "[INFO] Installing Flask 2.1+ (Python $PY_VER, has request_ctx)..." >&2
      if "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "flask>=2.1.0,<2.4.0" >/tmp/apr_flask_install_$$.log 2>&1; then
        if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
          echo "[INFO] Flask 2.1+ installed successfully with request_ctx support" >&2
        else
          echo "[WARN] Flask 2.1+ installed but request_ctx still not available" >&2
        fi
      else
        echo "[WARN] Failed to install Flask 2.1+, log:" >&2
        tail -20 /tmp/apr_flask_install_$$.log >&2 || true
      fi
    else
      # Python 3.6: Install Flask 2.0.x (may not have request_ctx, but has _request_ctx_stack)
      # Note: Flask 2.0.x may not have request_ctx, but test code might need it
      # We'll install Flask 2.0.x and create a compatibility shim for request_ctx
      echo "[INFO] Installing Flask 2.0.x (Python $PY_VER, may not have request_ctx)..." >&2
      if "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "flask>=2.0.0,<2.1.0" >/tmp/apr_flask_install_$$.log 2>&1; then
        # Check if request_ctx is available (unlikely for Flask 2.0.x)
        if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
          echo "[INFO] Flask 2.0.x installed with request_ctx support" >&2
        else
          echo "[WARN] Flask 2.0.x installed but request_ctx not available (expected for Flask 2.0.x)" >&2
          echo "[INFO] Creating compatibility shim for request_ctx using _request_ctx_stack..." >&2
          # Create a compatibility shim in SITE_DIR to provide request_ctx for Flask 2.0.x
          FLASK_GLOBALS_SHIM="$SITE_DIR/flask/globals.py"
          if [ -f "$FLASK_GLOBALS_SHIM" ]; then
            # Backup original file
            cp "$FLASK_GLOBALS_SHIM" "$FLASK_GLOBALS_SHIM.bak" 2>/dev/null || true
            # Add request_ctx compatibility shim if not already present
            if ! grep -q "APR_FLASK_REQUEST_CTX_SHIM" "$FLASK_GLOBALS_SHIM" 2>/dev/null; then
              # Use a temporary Python script file to avoid heredoc issues with arguments
              SHIM_SCRIPT="/tmp/apr_flask_shim_$$.py"
              cat > "$SHIM_SCRIPT" <<'PY_SHIM_EOF'
import sys
import os
if len(sys.argv) < 2:
    print("Error: shim_file path required", file=sys.stderr)
    sys.exit(1)
shim_file = sys.argv[1]
try:
    if not os.path.exists(shim_file):
        print("Error: file not found: {0}".format(shim_file), file=sys.stderr)
        sys.exit(1)
    
    with open(shim_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if our shim is already present
    if 'APR_FLASK_REQUEST_CTX_SHIM' in content:
        print("Shim already present")
        sys.exit(0)
    
    # For Flask 2.0.x, request_ctx does NOT exist natively
    # Always add the shim regardless of what's in the file
    # (Flask 2.0.x may mention request_ctx in comments/docs, but it's not actually defined)
    
    # Add compatibility shim at the end of the file
    shim_code = '''
# APR_FLASK_REQUEST_CTX_SHIM: Compatibility shim for Flask 2.0.x
# Flask 2.0.x uses _request_ctx_stack, but test code expects request_ctx (introduced in Flask 2.3)
# This shim provides request_ctx as a property that accesses _request_ctx_stack.top
try:
    from flask import _request_ctx_stack
    # Create a property-like object that mimics request_ctx behavior
    # In Flask 2.3+, request_ctx is a LocalProxy that wraps _request_ctx_stack
    # In Flask 2.0.x, we need to provide a compatible interface
    class _RequestCtxShim:
        @property
        def top(self):
            return _request_ctx_stack.top if _request_ctx_stack else None
        
        def _get_current_object(self):
            # Flask 2.0.x uses _request_ctx_stack.top directly
            # This method is called by conftest.py to get the current request context
            return _request_ctx_stack.top if _request_ctx_stack else None
        
        def pop(self):
            # Flask 2.0.x: pop from _request_ctx_stack
            if _request_ctx_stack:
                return _request_ctx_stack.pop()
            return None
        
        def __bool__(self):
            # Check if there's a current context
            return _request_ctx_stack.top is not None if _request_ctx_stack else False
        
        def __getattr__(self, name):
            ctx = _request_ctx_stack.top if _request_ctx_stack else None
            if ctx:
                return getattr(ctx, name)
            raise AttributeError("'{0}' object has no attribute '{1}'".format(type(self).__name__, name))
    
    request_ctx = _RequestCtxShim()
except (ImportError, AttributeError):
    # If _request_ctx_stack is not available, create a minimal stub
    class _RequestCtxStub:
        @property
        def top(self):
            return None
        
        def _get_current_object(self):
            return None
        
        def pop(self):
            return None
        
        def __bool__(self):
            return False
    request_ctx = _RequestCtxStub()
'''
    
    # Append shim if not already present
    if 'APR_FLASK_REQUEST_CTX_SHIM' not in content:
        with open(shim_file, 'a', encoding='utf-8') as f:
            f.write(shim_code)
        print("Shim added successfully")
    else:
        print("Shim already present")
except Exception as e:
    print("Error adding shim: {0}".format(e), file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PY_SHIM_EOF
              if "$PY" "$SHIM_SCRIPT" "$FLASK_GLOBALS_SHIM" >/tmp/apr_flask_shim_$$.log 2>&1; then
                echo "[INFO] request_ctx shim script executed successfully" >&2
                cat /tmp/apr_flask_shim_$$.log >&2 || true
              else
                echo "[WARN] Failed to add request_ctx shim, log:" >&2
                cat /tmp/apr_flask_shim_$$.log >&2 || true
              fi
              rm -f "$SHIM_SCRIPT" 2>/dev/null || true
            fi
            # Verify shim works
            if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
              echo "[INFO] request_ctx compatibility shim created successfully" >&2
            else
              echo "[WARN] request_ctx shim created but import still fails" >&2
            fi
          else
            echo "[WARN] Flask globals.py not found at $FLASK_GLOBALS_SHIM, cannot create shim" >&2
          fi
        fi
      else
        echo "[WARN] Failed to install Flask 2.0.x, log:" >&2
        tail -20 /tmp/apr_flask_install_$$.log >&2 || true
      fi
    fi
  fi
fi

# Fix for Django projects that need common dependencies (asgiref, pytz, etc.).
# Django requires several packages that may not be installed in some SIF images.
if [ "${APR_IS_DJANGO:-0}" = "1" ]; then
  # Install asgiref (required for Django's ASGI support)
  if ! "$PY" -c "import asgiref" >/dev/null 2>&1; then
    echo "[INFO] Installing 'asgiref' package (required by Django)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" asgiref >/dev/null 2>&1 || true
  fi
  # Install pytz (required for Django's timezone support)
  if ! "$PY" -c "import pytz" >/dev/null 2>&1; then
    echo "[INFO] Installing 'pytz' package (required by Django)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" pytz >/dev/null 2>&1 || true
  fi
  # Install sqlparse (required for Django's database support)
  if ! "$PY" -c "import sqlparse" >/dev/null 2>&1; then
    echo "[INFO] Installing 'sqlparse' package (required by Django)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" sqlparse >/dev/null 2>&1 || true
  fi
  # Install PostgreSQL client (psql) for Django dbshell tests
  # Some Django tests require psql command-line tool (e.g., test_postgresql.PostgreSqlDbshellCommandTestCase)
  if ! command -v psql >/dev/null 2>&1; then
    echo "[INFO] Django: psql not found, attempting to install PostgreSQL client..." >&2
    PSQL_INSTALLED=0
    # Strategy 1: Try conda install (works in writable conda environments)
    # Detect CONDA_BASE from PY path or common locations
    DJANGO_CONDA_BASE=""
    if [ -n "${CONDA_BASE:-}" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
      DJANGO_CONDA_BASE="${CONDA_BASE}"
    elif [ -x "/opt/miniconda3/bin/conda" ]; then
      DJANGO_CONDA_BASE="/opt/miniconda3"
    elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
      DJANGO_CONDA_BASE="/tmp/apr_miniconda3"
    fi
    if [ -n "$DJANGO_CONDA_BASE" ] && [ -x "${DJANGO_CONDA_BASE}/bin/conda" ]; then
      echo "[INFO] Django: trying to install psql via conda (CONDA_BASE=${DJANGO_CONDA_BASE})..." >&2
      source "${DJANGO_CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
      # Try installing into testbed environment first
      if "${DJANGO_CONDA_BASE}/bin/conda" install -n testbed -c conda-forge postgresql -y >/tmp/apr_conda_install_psql.log 2>&1; then
        # Activate testbed environment to ensure psql is in PATH
        source "${DJANGO_CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate testbed 2>/dev/null || true
        # Also add testbed bin to PATH explicitly
        export PATH="${DJANGO_CONDA_BASE}/envs/testbed/bin:${PATH}"
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via conda" >&2
        else
          # Check if psql exists in testbed bin directory
          if [ -x "${DJANGO_CONDA_BASE}/envs/testbed/bin/psql" ]; then
            export PATH="${DJANGO_CONDA_BASE}/envs/testbed/bin:${PATH}"
            PSQL_INSTALLED=1
            echo "[INFO] Django: psql found in testbed bin, added to PATH" >&2
          else
            echo "[WARN] Django: conda install completed but psql not found" >&2
          fi
        fi
      else
        # If testbed environment is not writable, try installing to a writable location
        echo "[WARN] Django: conda install to testbed failed (may be read-only), trying writable location..." >&2
        tail -20 /tmp/apr_conda_install_psql.log >&2 || true
        # Try installing to a temporary conda environment in /tmp
        PSQL_ENV_DIR="/tmp/apr_psql_env_$$"
        if "${DJANGO_CONDA_BASE}/bin/conda" create -y -p "$PSQL_ENV_DIR" -c conda-forge postgresql >/tmp/apr_conda_install_psql_tmp.log 2>&1; then
          # Add psql from temporary environment to PATH
          if [ -x "${PSQL_ENV_DIR}/bin/psql" ]; then
            export PATH="${PSQL_ENV_DIR}/bin:${PATH}"
            PSQL_INSTALLED=1
            echo "[INFO] Django: psql installed to writable location $PSQL_ENV_DIR and added to PATH" >&2
          fi
        else
          echo "[WARN] Django: conda install to writable location also failed" >&2
          tail -20 /tmp/apr_conda_install_psql_tmp.log >&2 || true
        fi
      fi
    fi
    # Strategy 2: Try system package manager (may fail in read-only containers)
    if [ "$PSQL_INSTALLED" -eq 0 ]; then
      if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Django: trying to install psql via apt-get..." >&2
        DEBIAN_FRONTEND=noninteractive apt-get update >/tmp/apr_apt_update.log 2>&1 && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends postgresql-client >/tmp/apr_apt_install_psql.log 2>&1 || true
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via apt-get" >&2
        fi
      elif command -v yum >/dev/null 2>&1; then
        echo "[INFO] Django: trying to install psql via yum..." >&2
        yum install -y postgresql >/tmp/apr_yum_install_psql.log 2>&1 || true
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via yum" >&2
        fi
      elif command -v apk >/dev/null 2>&1; then
        echo "[INFO] Django: trying to install psql via apk..." >&2
        apk add --no-cache postgresql-client >/tmp/apr_apk_install_psql.log 2>&1 || true
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via apk" >&2
        fi
      fi
    fi
    # Verify installation
    if [ "$PSQL_INSTALLED" -eq 0 ]; then
      echo "[WARN] Django: failed to install psql (tests requiring psql may fail)" >&2
      echo "[WARN] Django: checked conda and system package managers, but installation failed" >&2
    fi
  fi
  # Set Django environment variables BEFORE any Django imports or test execution
  # This must be set early to avoid ImproperlyConfigured errors during test collection
  # Note: This is in the validation script, so it applies to full test suite validation
  # Dynamically detect Django settings module (test_sqlite is common but not always present)
  # Try common Django settings modules in order of preference
  # Strategy: Check files first (more reliable), then try imports
  DJANGO_SETTINGS=""
  # Priority 1: Check if test_sqlite.py file exists (most reliable)
  if [ -f "/testbed/tests/test_sqlite.py" ] || [ -d "/testbed/tests/test_sqlite" ]; then
    DJANGO_SETTINGS="test_sqlite"
  # Priority 2: Check if tests/settings.py exists
  elif [ -f "/testbed/tests/settings.py" ]; then
    DJANGO_SETTINGS="tests.settings"
  # Priority 3: Try importing test_sqlite (may work if in PYTHONPATH)
  elif "$PY" -c "import test_sqlite" >/dev/null 2>&1; then
    DJANGO_SETTINGS="test_sqlite"
  # Priority 4: Fallback to test_sqlite (Django projects in SWE-bench typically use this)
  else
    # Django projects in SWE-bench typically use test_sqlite, so use it as default
    # Even if import fails now, it might work after pip install -e . adds /testbed to PYTHONPATH
    DJANGO_SETTINGS="test_sqlite"
  fi
  # Always set DJANGO_SETTINGS_MODULE (Django requires it)
  export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS"
  export LANG=en_US.UTF-8
  export LC_ALL=en_US.UTF-8
  export PYTHONIOENCODING=utf8
  export LANGUAGE=en_US:en
fi

cd /testbed
git config --global --add safe.directory /testbed || true

# scikit-learn: DO NOT do editable install from /testbed.
# Editable installs add /testbed to sys.path (egg-link), which causes Python to import the unbuilt
# source checkout and fail with missing compiled extensions (e.g., __check_build, murmurhash).
if [ "${APR_IS_SCIKITLEARN:-0}" != "1" ]; then
  "$PY" -m pip install -e . >/dev/null 2>&1 || true
fi

revert_tests() {
  if [ -n "${APR_BASE_COMMIT:-}" ] && [ -n "${APR_TEST_FILES:-}" ]; then
    IFS=$'\n'
    for f in ${APR_TEST_FILES}; do
      git checkout "${APR_BASE_COMMIT}" -- "${f}" >/dev/null 2>&1 || true
    done
    unset IFS
  fi
}
trap revert_tests EXIT

# Load test patch from file if APR_TEST_PATCH_FILE is set (for large patches)
# Otherwise use APR_TEST_PATCH variable directly
APR_TEST_PATCH_CONTENT=""
if [ -n "${APR_TEST_PATCH_FILE:-}" ] && [ -f "$APR_TEST_PATCH_FILE" ]; then
  APR_TEST_PATCH_CONTENT=$(cat "$APR_TEST_PATCH_FILE" 2>/dev/null || echo "")
elif [ -n "${APR_TEST_PATCH:-}" ]; then
  APR_TEST_PATCH_CONTENT="${APR_TEST_PATCH}"
fi

if [ -n "$APR_TEST_PATCH_CONTENT" ]; then
  # Align with SWE-bench eval.sh: reset the touched test files to base_commit first
  if [ -n "${APR_BASE_COMMIT:-}" ] && [ -n "${APR_TEST_FILES:-}" ]; then
    IFS=$'\n'
    for f in ${APR_TEST_FILES}; do
      git checkout "${APR_BASE_COMMIT}" -- "${f}" >/dev/null 2>&1 || true
    done
    unset IFS
  fi
  # Apply test patch (use stdin; do NOT embed literal content)
  printf "%s\n" "$APR_TEST_PATCH_CONTENT" | git apply -v -
fi

echo "=== RUN_VALIDATE ==="
echo "Tests: ${APR_TEST_LIST}"

# For Django: initialize Django apps before test collection to avoid AppRegistryNotReady
# This must be done once before any pytest collection starts
# IMPORTANT: Do this AFTER pip install -e . so that /testbed is in PYTHONPATH
if [ "${APR_IS_DJANGO:-0}" = "1" ]; then
  echo "[INFO] Django: Initializing Django apps before test collection..." >&2
  DJANGO_INIT_LOG="/tmp/django_init_$$.log"
  # Use the same settings module detection logic as above
  # But now /testbed should be in PYTHONPATH (from pip install -e .)
  DJANGO_SETTINGS_INIT=""
  if [ -n "${DJANGO_SETTINGS_MODULE:-}" ]; then
    DJANGO_SETTINGS_INIT="${DJANGO_SETTINGS_MODULE}"
  else
    # Try to detect settings module dynamically (same priority as above)
    # Now that /testbed is in PYTHONPATH, test_sqlite should be importable
    if "$PY" -c "import test_sqlite" >/dev/null 2>&1; then
      DJANGO_SETTINGS_INIT="test_sqlite"
    elif [ -f "/testbed/tests/test_sqlite.py" ] || [ -d "/testbed/tests/test_sqlite" ]; then
      DJANGO_SETTINGS_INIT="test_sqlite"
    elif [ -f "/testbed/tests/settings.py" ]; then
      DJANGO_SETTINGS_INIT="tests.settings"
    else
      # Fallback to test_sqlite (Django projects in SWE-bench typically use this)
      DJANGO_SETTINGS_INIT="test_sqlite"
    fi
  fi
  # Ensure /testbed and /testbed/tests are in PYTHONPATH for test_sqlite import
  # Django projects typically have test_sqlite in tests/ directory
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | grep -v "^/testbed/tests$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="/testbed:/testbed/tests${PYTHONPATH:+:${PYTHONPATH}}"
  else
    export PYTHONPATH="/testbed:/testbed/tests"
  fi
  
  # CRITICAL: Create sitecustomize.py so django.setup() runs inside *pytest process*
  # (fixes AppRegistryNotReady during collection).
  # This must be done BEFORE any pytest collection starts, and SITE_DIR must be in PYTHONPATH
  # Use the same detection logic as _verify_test_suite for consistency
  if [ -f "/testbed/django/__init__.py" ] || [ -f "/testbed/tests/test_sqlite.py" ]; then
    if [ -n "${SITE_DIR:-}" ] && [ -d "$SITE_DIR" ]; then
      __DJANGO_SITECUSTOMIZE_HEREDOC__
      echo "[INFO] Django: Created sitecustomize.py in $SITE_DIR for automatic Django initialization in pytest process" >&2
      CREATED_DJANGO_SITECUSTOMIZE=1
    fi
  fi
  # Skip the init probe when sitecustomize was created: sitecustomize runs in every Python
  # (including this -c), and if its django.setup() fails it leaves apps.loading=True, so
  # a second django.setup() would raise \"populate() isn't reentrant\".
  if [ "${CREATED_DJANGO_SITECUSTOMIZE:-0}" != "1" ]; then
  if "$PY" -c "
import os
import sys
os.chdir('/testbed')
django_settings_init = \"${DJANGO_SETTINGS_INIT}\"
if django_settings_init:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', django_settings_init)
try:
    import django
    if not django.apps.apps.ready:
        django.setup()
    print('[INFO] Django apps initialized successfully', flush=True)
except Exception as e:
    print(f'[ERROR] Django initialization failed: {e}', file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
" >"$DJANGO_INIT_LOG" 2>&1; then
    cat "$DJANGO_INIT_LOG" >&2
    rm -f "$DJANGO_INIT_LOG"
  else
    echo "[ERROR] Django initialization failed. Error output:" >&2
    cat "$DJANGO_INIT_LOG" >&2
    rm -f "$DJANGO_INIT_LOG"
    echo "[WARN] Continuing despite Django initialization failure (may cause AppRegistryNotReady)..." >&2
  fi
  fi
fi

OUT_FILE="$(mktemp -p /tmp)"

run_pytest_one() {
  local f="$1"
  local t="$2"
  t=$(printf '%s' "$t" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  # For pytest-dev: ignore unknown config option warnings (e.g., rsyncdirs)
  # These warnings can cause pytest to fail in strict mode (rc=3)
  # For matplotlib: ignore pyparsing deprecation warnings (e.g., enablePackrat)
  # These warnings can cause pytest collection to fail (rc=2)
  local pytest_extra_args=""
  if [ "${APR_IS_PYTESTDEV:-0}" = "1" ]; then
    pytest_extra_args="-W ignore::pytest.PytestConfigWarning"
  fi
  if [ "${APR_IS_MATPLOTLIB:-0}" = "1" ]; then
    pytest_extra_args="${pytest_extra_args} -W ignore::pyparsing.warnings.PyparsingDeprecationWarning"
  fi
  # Django: run all Django tests via runtests.py to avoid INSTALLED_APPS/migration issues under pytest
  # Use runtests when test name format is test_method (class.path) and runtests.py exists
  if [ "${APR_IS_DJANGO:-0}" = "1" ] && [ -f /testbed/tests/runtests.py ]; then
    if echo "$t" | grep -qE '^[a-zA-Z0-9_]+ *\([a-zA-Z0-9_.]+\)'; then
      class_path=$(echo "$t" | sed -n 's/.*(\([^)]*\)).*/\1/p')
      method_name=$(echo "$t" | sed 's/ *([^)]*).*//')
      # All Django test modules use runtests; class_path format usually tests.module.ClassName
      if echo "$class_path" | grep -qE '\.(tests|test_|tests\.)'; then
        # If class_path ends with method_name use as-is; else append method_name
        if echo "$class_path" | grep -qE "\.${method_name}$"; then
          runtests_label="$class_path"
        else
          runtests_label="${class_path}.${method_name}"
        fi
        export APR_DJANGO_USE_RUNTESTS=1
        if [ -f /testbed/tests/test_sqlite.py ] || [ -d /testbed/tests/test_sqlite ]; then
          DJANGO_SETTINGS_RUN="test_sqlite"
        elif [ -f /testbed/tests/settings.py ]; then
          DJANGO_SETTINGS_RUN="tests.settings"
        else
          DJANGO_SETTINGS_RUN="test_sqlite"
        fi
        export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS_RUN"
        (cd /testbed && "$PY" tests/runtests.py --noinput --failfast --settings="$DJANGO_SETTINGS_RUN" "$runtests_label") 2>&1 | tee "$OUT_FILE"
        return "${PIPESTATUS[0]}"
      fi
    fi
  fi
  if [[ "$t" == *"::"* ]]; then
    "$PY" -m pytest $pytest_extra_args -q -x "$t" 2>&1 | tee "$OUT_FILE"
    return $?
  fi
  # For test names with special characters (commas, asterisks, parentheses, ampersands), handle them carefully
  # pytest -k uses Python expression syntax, so special chars can cause ModuleNotFoundError or bash syntax errors
  # If test name contains special chars, extract a simpler pattern that pytest can safely match
  local test_pattern="$t"
  if echo "$t" | grep -qE '[(),*&]'; then
    # Strategy 0: For format "test_name (class_name)", extract just the test_name part
    # This is common in Django and other frameworks where test names include class context
    # Pattern: test_name followed by space and opening parenthesis
    if echo "$t" | grep -qE '^[a-zA-Z0-9_][a-zA-Z0-9_ ]* \('; then
      # Extract test name before the opening parenthesis
      test_pattern=$(echo "$t" | sed 's/ (.*//')
    # Strategy 1: Extract the description part (after " - ") which usually doesn't have special chars
    elif echo "$t" | grep -qE ' - '; then
      local desc_part=$(echo "$t" | sed 's/.* - //')
      if [ -n "$desc_part" ] && [ "$desc_part" != "$t" ] && ! echo "$desc_part" | grep -qE '[(),*&]'; then
        # Use the description part (e.g., "new function with partial application")
        test_pattern="$desc_part"
      else
        # Fall through to Strategy 2
        test_pattern=""
      fi
    fi
    # Strategy 2: If Strategy 0/1 didn't work, extract first meaningful word (test function name)
    if [ -z "$test_pattern" ] || [ "$test_pattern" = "$t" ]; then
      # Extract first meaningful word (longer than 4 chars) from the test name
      # This is usually the test function name
      local first_word=$(echo "$t" | sed 's/[(),*&]//g' | sed "s/'//g" | tr ' ' '\n' | grep -E '^.{5,}' | head -1)
      if [ -n "$first_word" ]; then
        test_pattern="$first_word"
      else
        # Fallback: try to match a common pattern
        if echo "$t" | grep -qi "partial"; then
          test_pattern="partial"
        elif echo "$t" | grep -qi "output"; then
          test_pattern="output"
        elif echo "$t" | grep -qi "form"; then
          test_pattern="form"
        else
          # Last resort: use first 20 chars without special chars
          test_pattern=$(echo "$t" | sed 's/[(),*&]//g' | sed "s/'//g" | cut -c1-20 | sed 's/ $//')
        fi
      fi
    fi
  fi
  if [ -n "$f" ]; then
    "$PY" -m pytest $pytest_extra_args -q -x "$f" -k "$test_pattern" 2>&1 | tee "$OUT_FILE"
    return $?
  fi
  "$PY" -m pytest $pytest_extra_args -q -x -k "$test_pattern" 2>&1 | tee "$OUT_FILE"
  return $?
}

IFS=$'\n'
for t in ${APR_TEST_LIST}; do
  FOUND=0
  RC=0
  if [ -n "${APR_TEST_FILES:-}" ]; then
    for f in ${APR_TEST_FILES}; do
      set +e
      run_pytest_one "$f" "$t"
      RC=$?
      if [ "$RC" -ne 0 ] && install_missing_module_from_file "$OUT_FILE"; then
        run_pytest_one "$f" "$t"
        RC=$?
      fi
      set -e
      if [ "$RC" -ne 5 ]; then
        FOUND=1
        break
      fi
    done
  fi
  if [ "$FOUND" -eq 0 ]; then
    set +e
    run_pytest_one "" "$t"
    RC=$?
    if [ "$RC" -ne 0 ] && install_missing_module_from_file "$OUT_FILE"; then
      run_pytest_one "" "$t"
      RC=$?
    fi
    set -e
  fi
  if [ "$RC" -eq 5 ]; then
    echo "ERROR: pytest collected 0 tests for '$t'" >&2
    exit 2
  fi
  if [ "$RC" -ne 0 ]; then
    echo "ERROR: test '$t' failed (rc=$RC)" >&2
    exit "$RC"
  fi
done
unset IFS
""").replace('      __DJANGO_SITECUSTOMIZE_HEREDOC__', _DJANGO_SITECUSTOMIZE_HEREDOC)

        env_lines = []
        # Use heredoc for all dynamic variables to avoid bash syntax errors with special characters
        env_lines.append("export APR_BASE_COMMIT=$(cat <<'EOF_APR_BC'\n" + base_commit + "\nEOF_APR_BC\n)")
        # Root fix: install deterministic pip deps from SWE-bench spec (e.g. mpmath for sympy)
        # Also extract required Python version for version matching
        required_python_version = None
        try:
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # type: ignore
            version = (inst.get("version") or "") if inst else ""
            spec = (MAP_REPO_VERSION_TO_SPECS.get(repo, {}) or {}).get(version, {})  # type: ignore
            pip_pkgs = spec.get("pip_packages") or []
            test_cmd = spec.get("test_cmd") or ""
            required_python_version = spec.get("python")
        except Exception:
            pip_pkgs = []
            test_cmd = ""
            required_python_version = None
        # CRITICAL: Use temporary files for large environment variables to avoid "Argument list too long" error
        # Store large variables for later file writing (after bind is defined)
        _apr_temp_pip_pkgs = pip_pkgs if pip_pkgs else None
        _apr_temp_test_patch = test_patch if test_patch else None
        
        if pip_pkgs:
            # Will write to file after bind is defined
            # CRITICAL: Only set file path, do NOT read file content into environment variable
            # Reading file content into APR_PIP_PACKAGES would still cause "Argument list too long"
            env_lines.append("export APR_PIP_PACKAGES_FILE=/testbed/.apr_env/apr_pip_packages.txt")
            # Do NOT set APR_PIP_PACKAGES variable - script will read from file directly
        if test_cmd:
            # Use heredoc to avoid bash syntax errors when test_cmd contains special characters
            env_lines.append("export APR_TEST_CMD=$(cat <<'EOF_APR_TC'\n" + test_cmd + "\nEOF_APR_TC\n)")
        if required_python_version:
            # Use heredoc to avoid bash syntax errors (though version numbers are usually safe)
            env_lines.append("export APR_REQUIRED_PYTHON_VERSION=$(cat <<'EOF_APR_RPV'\n" + str(required_python_version) + "\nEOF_APR_RPV\n)")
        # Mark project types for special handling
        is_django = repo.startswith("django/")
        is_astropy = repo.startswith("astropy/")
        is_pytestdev = instance_id.startswith("pytest-dev__")
        is_scikitlearn = instance_id.startswith("scikit-learn__scikit-learn-")
        is_pallets = instance_id.startswith("pallets__")
        is_seaborn = instance_id.startswith("mwaskom__seaborn-")
        if is_django:
            env_lines.append("export APR_IS_DJANGO=1")
        if is_astropy:
            env_lines.append("export APR_IS_ASTROPY=1")
        if is_pytestdev:
            env_lines.append("export APR_IS_PYTESTDEV=1")
        if is_scikitlearn:
            env_lines.append("export APR_IS_SCIKITLEARN=1")
        if is_pallets:
            env_lines.append("export APR_IS_PALLETS=1")
        if is_seaborn:
            env_lines.append("export APR_IS_SEABORN=1")
        if test_patch:
            # Will write to file after bind is defined
            # CRITICAL: Only set file path, do NOT read file content into environment variable
            # Reading file content into APR_TEST_PATCH would still cause "Argument list too long"
            env_lines.append("export APR_TEST_PATCH_FILE=/testbed/.apr_env/apr_test_patch.txt")
            # Do NOT set APR_TEST_PATCH variable - script will read from file directly
        if test_files:
            env_lines.append("export APR_TEST_FILES=$(cat <<'EOF_APR_TF'\n" + test_files + "\nEOF_APR_TF\n)")
        env_lines.append("export APR_TEST_LIST=$(cat <<'EOF_APR_TL'\n" + "\n".join(all_tests) + "\nEOF_APR_TL\n)")
        # Use heredoc to avoid bash syntax errors when test names contain special characters (parentheses, etc.)
        env_lines.append("export APR_TEST_LIST_STR=$(cat <<'EOF_APR_TLS'\n" + tests_list_str + "\nEOF_APR_TLS\n)")
        full_script = "\n".join(env_lines) + "\n" + script

        # CRITICAL: Write script to temp file to avoid "Argument list too long" error
        # The entire script (env vars + script) is passed as a single argument to bash -lc
        # Writing to file and sourcing it avoids the argument length limit
        temp_dir = Path(bind.split(":")[0]) / ".apr_env" if ":" in bind else Path("/tmp") / f"apr_env_{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write large environment variables to files
        if _apr_temp_pip_pkgs:
            pip_file = temp_dir / "apr_pip_packages.txt"
            pip_file.write_text("\n".join(_apr_temp_pip_pkgs) + "\n")
        if _apr_temp_test_patch:
            patch_file = temp_dir / "apr_test_patch.txt"
            patch_file.write_text(_apr_temp_test_patch)
        
        # Write full script to file
        script_file = temp_dir / "apr_script.sh"
        script_file.write_text(full_script)
        script_file.chmod(0o755)
        
        # Add temp dir bind to existing bind string
        bind_with_temp = f"{bind},{temp_dir}:/testbed/.apr_env"

        print(f"[VALIDATE] Starting Apptainer validation for {instance_id}...", flush=True)
        print(f"[VALIDATE] Image: {image}", flush=True)
        print(f"[VALIDATE] Tests: {tests_list_str}", flush=True)
        print(f"[VALIDATE] Timeout: 3600s", flush=True)
        
        try:
            r = _run_apptainer(
                image=image,
                argv=["bash", "-lc", "source /testbed/.apr_env/apr_script.sh"],
                bind=bind_with_temp,
                pwd="/testbed",
                timeout=3600,
            )
            print(f"[VALIDATE] Apptainer validation completed: rc={r.get('rc')}, timeout={r.get('timeout', False)}", flush=True)
            if r.get("error"):
                print(f"[VALIDATE] Validation error: {r.get('error')}", flush=True)
            if r.get("stderr") and len(r.get("stderr", "")) > 0:
                stderr_preview = r.get("stderr", "")[-500:]
                print(f"[VALIDATE] Validation stderr (last 500 chars):\n{stderr_preview}", flush=True)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[ERROR] _run_apptainer raised exception: {e}", flush=True)
            print(f"[ERROR] Traceback:\n{error_trace}", flush=True)
            r = {
                "rc": -1,
                "stdout": "",
                "stderr": f"Validation exception: {e}\n{error_trace[:1000]}",
                "error": str(e)
            }
        
        passed = r["rc"] == 0
        # Write a short log into meta_dir for debugging
        meta = Path(meta_dir)
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "apptainer_validate.log").write_text((r.get("stdout") or "") + "\n" + (r.get("stderr") or ""), encoding="utf-8")
        return {
            "passed": passed,
            "instance_id": instance_id,
            "repo": repo,
            "cmd": f"pytest validate ({tests_list_str})",
            "rc": r["rc"],
            "stdout": (r.get("stdout") or "")[-2000:],
            "stderr": (r.get("stderr") or "")[-2000:],
        }

    def run_one_test(self, workdir: str, test_name: str, log_file: str) -> Dict[str, Any]:
        """
        Run a single test in the workdir using SWE-bench container environment.
        
        For SWE-bench, tests should run in containers with all dependencies.
        Strategy:
        1. Try to use SWE-bench harness container (if available from previous runs)
        2. Fallback to direct execution (for compatibility, but may have dependency issues)
        
        Test name formats:
        - "test_immutable" (function name)
        - "sympy/core/tests/test_basic.py::test_immutable" (full path)
        - "test_basic.py::test_immutable" (relative path)
        """
        if _swe_runtime() == "apptainer":
            return self._run_one_test_apptainer(workdir=workdir, test_name=test_name, log_file=log_file)

        wd = Path(workdir)
        if not wd.exists():
            return {
                "ran": False,
                "rc": 1,
                "error": f"workdir does not exist: {workdir}",
                "test_name": test_name,
                "logfile": log_file,
            }
        
        # Try to get instance info to find test file
        instance_id = None
        test_file = None
        inst = None
        try:
            # Try to infer instance_id from workdir path
            # workdir structures:
            # 1. Normal: .../runs/swebench_verified/{instance_id}/workdir
            # 2. Archive extracted: .../apr_extracted/swebench_verified/{job_tag}/{instance_id}/{instance_id}
            parts = Path(workdir).parts
            if "swebench_verified" in parts:
                idx = parts.index("swebench_verified")
                if "apr_extracted" in parts:
                    # Archive mode: workdir ends with {instance_id}/{instance_id}, so the last part is instance_id
                    instance_id = parts[-1]
                elif idx + 1 < len(parts):
                    # Normal mode: next part after "swebench_verified" is instance_id
                    instance_id = parts[idx + 1]
                
                if instance_id:
                    inst = self._get_instance(instance_id)
                    # Get test file from FAIL_TO_PASS or test_patch
                    fail_to_pass = inst.get("FAIL_TO_PASS", [])
                    # Handle JSON string format
                    if isinstance(fail_to_pass, str):
                        import json
                        try:
                            fail_to_pass = json.loads(fail_to_pass)
                        except:
                            pass
                    if fail_to_pass and test_name in fail_to_pass:
                        # Try to find test file from test_patch
                        test_patch = inst.get("test_patch", "")
                        if test_patch:
                            import re
                            # Extract file path from test_patch diff
                            match = re.search(r'^\+\+\+ b/(.+\.py)', test_patch, re.MULTILINE)
                            if match:
                                test_file = match.group(1)
        except Exception:
            pass  # Continue with fallback strategies
        
        # Prepare log file
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Strategy 1: Try to use SWE-bench harness container (if available)
        # This ensures tests run with all dependencies in the containerized environment
        if instance_id and inst:
            try:
                import sys
                import docker
                # Add SWE-bench to path if not already there
                swebench_path = APR_ROOT.parent / "third_party" / "SWE-bench"
                if str(swebench_path) not in sys.path:
                    sys.path.insert(0, str(swebench_path))
                
                from swebench.harness.test_spec.test_spec import make_test_spec
                from swebench.harness.docker_build import (
                    build_container,
                    build_env_images,
                    setup_logger,
                )
                
                docker_client = docker.from_env()
                
                # Create TestSpec for this instance
                test_spec = make_test_spec(inst)
                
                # Try to build and start container
                # Use a temporary run_id for this test
                run_id = f"tdd_gate_{instance_id}"
                logger = setup_logger(instance_id, log_path.parent / "container.log")
                
                try:
                    # IMPORTANT (rootless podman + setgid project dirs):
                    # SWE-bench harness builds images under relative dirs like
                    #   logs/build_images/...
                    # If current cwd is under a setgid project directory (gid=prjsXXXX),
                    # the build context files inherit that GID and rootless userns may
                    # fail with lchown invalid argument (insufficient gid mappings).
                    #
                    # Workaround: run image build from a non-setgid cwd under PODMAN_DIR
                    # (exported by podman_project_env.sh).
                    build_base = Path(os.environ.get("PODMAN_DIR", str(APR_ROOT))) / "swebench_build_cwd"
                    build_cwd = build_base / instance_id
                    build_cwd.mkdir(parents=True, exist_ok=True)
                    old_cwd = os.getcwd()
                    os.chdir(str(build_cwd))

                    # Ensure base/env images exist (otherwise build_container will fail)
                    # This can be slow the first time, but is required to get a real test result
                    # (instead of missing-dependency errors in local execution).
                    build_env_images(
                        docker_client,
                        [test_spec],
                        force_rebuild=False,
                        max_workers=1,
                    )

                    container = build_container(
                        test_spec, docker_client, run_id, logger, nocache=False, force_rebuild=False
                    )
                    container.start()
                    logger.info(f"Container for {instance_id} started: {container.id}")
                    
                    # Build test command
                    # Check if this is Django and should use runtests.py
                    is_django_container = instance_id and instance_id.startswith("django__")
                    runtests_path_container = "/testbed/tests/runtests.py"
                    
                    if is_django_container:
                        # Django test name format: "test_method (tests.module.ClassName)"
                        # Convert to runtests.py format: "tests.module.ClassName.test_method"
                        import re
                        runtests_label = None
                        
                        # Pattern 1: "test_method (tests.module.ClassName)"
                        match = re.match(r'^([a-zA-Z0-9_]+)\s*\(([a-zA-Z0-9_.]+)\)', test_name)
                        if match:
                            method_name = match.group(1)
                            class_path = match.group(2)
                            if class_path.endswith(f".{method_name}"):
                                runtests_label = class_path
                            else:
                                runtests_label = f"{class_path}.{method_name}"
                        # Pattern 2: Already in runtests format
                        elif re.match(r'^tests\.[a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+', test_name):
                            runtests_label = test_name
                        
                        if runtests_label:
                            # Determine Django settings module
                            django_settings = "test_sqlite"
                            # Use runtests.py
                            test_cmd = f"cd /testbed && python3 {runtests_path_container} --noinput --failfast --settings={django_settings} {runtests_label}"
                        else:
                            # Fallback to pytest if runtests_label couldn't be determined
                            if "::" in test_name:
                                test_path = test_name
                            elif test_file:
                                test_path = f"{test_file}::{test_name}"
                            else:
                                test_path = test_name
                            if "::" in test_path:
                                test_cmd = f"python3 -m pytest -v {test_path} --tb=short -x"
                            else:
                                test_cmd = f"python3 -m pytest -v -k {test_name} --tb=short -x"
                    else:
                        # Non-Django: use pytest
                        if "::" in test_name:
                            test_path = test_name
                        elif test_file:
                            test_path = f"{test_file}::{test_name}"
                        else:
                            test_path = test_name
                        
                        if "::" in test_path:
                            test_cmd = f"python3 -m pytest -v {test_path} --tb=short -x"
                        else:
                            test_cmd = f"python3 -m pytest -v -k {test_name} --tb=short -x"
                    
                    # Execute test in container
                    # Use exec_run to get proper exit code
                    # SWE-bench containers use DOCKER_WORKDIR and DOCKER_USER constants
                    from swebench.harness.constants import DOCKER_WORKDIR, DOCKER_USER
                    try:
                        exec_result = container.exec_run(
                            test_cmd,
                            workdir=DOCKER_WORKDIR,
                            user=DOCKER_USER,
                        )
                        output = exec_result.output.decode("utf-8") if exec_result.output else ""
                        rc = exec_result.exit_code
                    finally:
                        # Always cleanup container
                        try:
                            container.stop(timeout=10)
                        except Exception:
                            pass
                        try:
                            container.remove(force=True)
                        except Exception:
                            pass
                    
                    # Write to log
                    with open(log_path, "w", encoding="utf-8") as log_f:
                        log_f.write(f"Command (in container): {test_cmd}\n")
                        log_f.write(f"Container: {container.name}\n")
                        log_f.write(f"Return code: {rc}\n")
                        log_f.write("\n")
                        log_f.write("=== OUTPUT ===\n")
                        log_f.write(output)
                        log_f.write("\n")
                    
                    return {
                        "ran": True,
                        "rc": rc,
                        "test_name": test_name,
                        "logfile": log_file,
                        "stdout": output,
                        "stderr": "",
                        "cmd": test_cmd,
                        "container": container.name,
                    }
                except Exception as e:
                    logger.error(f"Failed to run test in container: {e}")
                    print(f"[WARN] [run_one_test] Container approach failed: {e}, falling back to direct execution", flush=True)
                    # Fall through to direct execution
                finally:
                    try:
                        os.chdir(old_cwd)
                    except Exception:
                        pass
            except (ImportError, Exception) as e:
                # SWE-bench harness not available or container creation failed
                print(f"[WARN] [run_one_test] Cannot use container (SWE-bench harness may not be available): {e}", flush=True)
                # Fall through to direct execution
        
        # Strategy 2: Fallback to direct execution (original logic)
        # If test_name contains "::", use as-is (pytest format)
        if "::" in test_name:
            test_path = test_name
        elif test_file:
            # Strategy 2: Use test_file from instance metadata
            test_path = f"{test_file}::{test_name}"
        else:
            # Strategy 3: Try to find test file by searching
            # For now, use test_name as-is and let the test framework figure it out
            test_path = test_name
        
        # Try different test runners
        cmd_options = []
        
        # Option 0: Django runtests.py (priority for Django projects to avoid pytest config issues)
        # Check if this is a Django project by:
        # 1. instance_id starts with "django__"
        # 2. or runtests.py exists in workdir/tests/
        is_django = False
        runtests_path = wd / "tests" / "runtests.py"
        if instance_id and instance_id.startswith("django__"):
            is_django = True
        elif runtests_path.exists():
            is_django = True
        
        if is_django and runtests_path.exists():
            # Django test name format: "test_method (tests.module.ClassName)"
            # Convert to runtests.py format: "tests.module.ClassName.test_method"
            import re
            runtests_label = None
            
            # Pattern 1: "test_method (tests.module.ClassName)"
            match = re.match(r'^([a-zA-Z0-9_]+)\s*\(([a-zA-Z0-9_.]+)\)', test_name)
            if match:
                method_name = match.group(1)
                class_path = match.group(2)
                # Check if class_path already ends with method_name
                if class_path.endswith(f".{method_name}"):
                    runtests_label = class_path
                else:
                    runtests_label = f"{class_path}.{method_name}"
            # Pattern 2: "tests.module.ClassName.test_method" (already in runtests format)
            elif re.match(r'^tests\.[a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+', test_name):
                runtests_label = test_name
            # Pattern 3: "test_method" only - try to find from test_file
            elif test_file:
                # Extract module path from test_file (e.g., "tests/module/test_file.py" -> "tests.module")
                test_file_path = Path(test_file)
                if test_file_path.parts[0] == "tests":
                    module_parts = list(test_file_path.parts[:-1])  # Remove filename
                    module_path = ".".join(module_parts)
                    # Try to find class name from test_file content
                    test_file_full = wd / test_file
                    if test_file_full.exists():
                        try:
                            content = test_file_full.read_text(encoding='utf-8', errors='ignore')
                            # Look for class definition
                            class_match = re.search(r'class\s+([a-zA-Z0-9_]+)\s*\(', content)
                            if class_match:
                                class_name = class_match.group(1)
                                runtests_label = f"{module_path}.{class_name}.{test_name}"
                        except Exception:
                            pass
            
            if runtests_label:
                # Determine Django settings module
                django_settings = "test_sqlite"
                if (wd / "tests" / "test_sqlite.py").exists() or (wd / "tests" / "test_sqlite").exists():
                    django_settings = "test_sqlite"
                elif (wd / "tests" / "settings.py").exists():
                    django_settings = "tests.settings"
                
                # Use runtests.py with proper settings
                env = os.environ.copy()
                env["APR_DJANGO_USE_RUNTESTS"] = "1"
                env["DJANGO_SETTINGS_MODULE"] = django_settings
                cmd_options.append([
                    "python3", str(runtests_path),
                    "--noinput", "--failfast",
                    "--settings", django_settings,
                    runtests_label
                ])
        
        # Option 1: pytest (most common for Python projects, but skip for Django if runtests was added)
        if not (is_django and runtests_path.exists() and cmd_options):
            if "::" in test_path:
                # pytest format: file::test_name
                cmd_options.append(["python3", "-m", "pytest", "-v", test_path, "--tb=short", "-x"])
            else:
                # Try pytest with test name pattern
                cmd_options.append(["python3", "-m", "pytest", "-v", "-k", test_name, "--tb=short", "-x"])
        
        # Option 2: unittest (Python standard library) - try file path format
        if "::" in test_path:
            # Convert pytest format to unittest format
            # sympy/core/tests/test_basic.py::test_immutable -> sympy.core.tests.test_basic.test_immutable
            unittest_path = test_path.replace("/", ".").replace("::", ".")
            # Remove .py extension if present
            unittest_path = unittest_path.replace(".py", "")
            cmd_options.append(["python3", "-m", "unittest", "-v", unittest_path])
            
            # Also try running the test file directly and filtering by test name
            test_file_part = test_path.split("::")[0]
            if (wd / test_file_part).exists():
                # Run the test file and let unittest discover the test
                cmd_options.append(["python3", "-m", "unittest", "-v", test_file_part.replace("/", ".").replace(".py", ""), "-k", test_name])
        else:
            # Try unittest discovery with test name
            cmd_options.append(["python3", "-m", "unittest", "discover", "-v", "-k", test_name])
        
        # Option 3: Try running test file directly (if it's a standalone script)
        if test_file and (wd / test_file).exists():
            cmd_options.append(["python3", test_file])
        
        # Try each command until one works
        last_error = None
        last_stderr = None
        for cmd in cmd_options:
            try:
                r = _run(cmd, cwd=str(wd), timeout=300)
                
                # Write output to log file
                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"Command: {' '.join(cmd)}\n")
                    log_f.write(f"Return code: {r['rc']}\n\n")
                    if r.get("stdout"):
                        log_f.write("=== STDOUT ===\n")
                        log_f.write(r["stdout"])
                        log_f.write("\n")
                    if r.get("stderr"):
                        log_f.write("=== STDERR ===\n")
                        log_f.write(r["stderr"])
                        log_f.write("\n")
                
                # If command executed (not "command not found"), consider it ran
                # Even if test fails (rc != 0), we still return the result
                stderr_lower = (r.get("stderr") or "").lower()
                stdout_lower = (r.get("stdout") or "").lower()
                
                # Check if it's a real execution (not import/module errors)
                is_module_error = (
                    "no module named" in stderr_lower or
                    "failed to import" in stderr_lower or
                    "import error" in stderr_lower or
                    r["rc"] == 127  # command not found
                )
                
                if not is_module_error:
                    # Command executed, return result (even if test failed)
                    return {
                        "ran": True,
                        "rc": r["rc"],
                        "test_name": test_name,
                        "logfile": log_file,
                        "stdout": r.get("stdout", ""),
                        "stderr": r.get("stderr", ""),
                        "cmd": cmd,
                    }
                else:
                    # Dependency error - try to install dependencies if requirements.txt exists
                    if (wd / "requirements.txt").exists() and "no module named" in stderr_lower:
                        print(f"[INFO] [run_one_test] Dependency error detected, attempting to install from requirements.txt", flush=True)
                        install_r = _run(["pip", "install", "-r", "requirements.txt"], cwd=str(wd), timeout=300)
                        if install_r["rc"] == 0:
                            # Retry the test command after installing dependencies
                            print(f"[INFO] [run_one_test] Dependencies installed, retrying test", flush=True)
                            r = _run(cmd, cwd=str(wd), timeout=300)
                            # Update log file
                            with open(log_path, "a", encoding="utf-8") as log_f:
                                log_f.write("\n=== AFTER DEPENDENCY INSTALLATION ===\n")
                                log_f.write(f"Command: {' '.join(cmd)}\n")
                                log_f.write(f"Return code: {r['rc']}\n\n")
                                if r.get("stdout"):
                                    log_f.write("=== STDOUT ===\n")
                                    log_f.write(r["stdout"])
                                    log_f.write("\n")
                                if r.get("stderr"):
                                    log_f.write("=== STDERR ===\n")
                                    log_f.write(r["stderr"])
                                    log_f.write("\n")
                            # Check if still module error
                            stderr_lower_retry = (r.get("stderr") or "").lower()
                            is_module_error_retry = (
                                "no module named" in stderr_lower_retry or
                                "failed to import" in stderr_lower_retry
                            )
                            if not is_module_error_retry:
                                # Success after dependency installation
                                return {
                                    "ran": True,
                                    "rc": r["rc"],
                                    "test_name": test_name,
                                    "logfile": log_file,
                                    "stdout": r.get("stdout", ""),
                                    "stderr": r.get("stderr", ""),
                                    "cmd": cmd,
                                }
                    last_error = r.get("stderr", "") or r.get("stdout", "")
                    last_stderr = r.get("stderr", "")
            except Exception as e:
                last_error = str(e)
                continue  # Try next command
        
        # If all commands failed, return error with details
        # Check if it was a dependency error
        dependency_error = False
        if last_stderr:
            stderr_lower = last_stderr.lower()
            dependency_error = (
                "no module named" in stderr_lower or
                "failed to import" in stderr_lower or
                "import error" in stderr_lower
            )
        
        error_msg = f"Could not run test {test_name}: all test runners failed"
        if dependency_error:
            error_msg += " (dependency/module import error - test may require containerized environment with full dependencies)"
        if last_error:
            error_msg += f"\nLast error: {last_error[:500]}"
        
        return {
            "ran": False,
            "rc": 1,
            "error": error_msg,
            "test_name": test_name,
            "logfile": log_file,
            "stderr": last_stderr or "",
            "dependency_error": dependency_error,
        }

    def _run_one_test_apptainer(self, *, workdir: str, test_name: str, log_file: str) -> Dict[str, Any]:
        """
        Run a single (RED/GREEN) test via Apptainer inside the SWE-bench testbed container.

        IMPORTANT: This must not rely on creating a venv or pip-installing arbitrary deps at runtime.
        Instead we use the SWE-bench per-instance eval image which contains a conda env ("testbed")
        matching the task's dependency snapshot.
        """
        wd = Path(workdir)
        if not wd.exists():
            return {"ran": False, "rc": 1, "error": f"workdir does not exist: {workdir}", "test_name": test_name, "logfile": log_file}

        # Try to infer instance metadata (for test_patch + base_commit)
        instance_id = None
        inst = None
        test_patch = ""
        base_commit = None
        directives: list[str] = []
        try:
            parts = Path(workdir).parts
            if "swebench_verified" in parts:
                idx = parts.index("swebench_verified")
                # Handle different workdir path structures:
                # 1. Normal: .../swebench_verified/{instance_id}/workdir
                # 2. Archive extracted: .../apr_extracted/swebench_verified/{job_tag}/{instance_id}
                #    In this case, the last component is the instance_id
                if "apr_extracted" in parts:
                    # Archive mode: workdir ends with {instance_id}, so the last part is instance_id
                    instance_id = parts[-1]
                elif idx + 1 < len(parts):
                    # Normal mode: next part after "swebench_verified" is instance_id
                    instance_id = parts[idx + 1]
                
                if instance_id:
                    inst = self._get_instance(instance_id)
                    test_patch = inst.get("test_patch", "") if inst else ""
                    base_commit = inst.get("base_commit") if inst else None
                    directives = _parse_test_directives_from_patch(test_patch)
        except Exception:
            pass

        # Default to instance image if we know the instance_id; otherwise fall back to the old minimal image.
        if instance_id:
            instance_id_l = instance_id.lower()
            is_scikitlearn = instance_id_l.startswith("scikit-learn__scikit-learn-")

            # Some scikit-learn instances have broken DockerHub images lacking compiled extensions
            # (e.g. sklearn.__check_build._check_build). In that case, prefer GHCR images.
            # Keep this project-scoped to avoid impacting other repos.
            if is_scikitlearn and os.environ.get("APR_SCIKIT_PREFER_GHCR", "1") == "1":
                image = f"docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
                print(f"[RUN_TEST] scikit-learn: using GHCR image: {image}", flush=True)
            else:
                image_name = _swebench_instance_image(instance_id=instance_id, arch="x86_64", tag="latest", namespace="swebench")
                sif = _swebench_sif_path()
                if sif:
                    image = sif
                    print(f"[RUN_TEST] Using pre-pulled SIF image: {sif}", flush=True)
                else:
                    image = f"docker://{image_name}"
                    print(f"[RUN_TEST] Using SWE-bench instance image: {image_name}", flush=True)
        else:
            image = "docker://python:3.11"
            print(f"[RUN_TEST] Using fallback image: {image}", flush=True)

        # Bind workdir as /testbed (SWE-bench convention)
        bind = f"{wd}:/testbed"
        print(f"[RUN_TEST] Workdir: {workdir}", flush=True)
        print(f"[RUN_TEST] Test name: {test_name}", flush=True)
        print(f"[RUN_TEST] Log file: {log_file}", flush=True)
        print(f"[RUN_TEST] Container bind: {bind}", flush=True)
        print(f"[RUN_TEST] Preparing test execution script...", flush=True)

        # Use shared env script base then add run_one_test-specific project config
        script = (_build_test_environment_script_base() + r"""

# ============================================================================
# run_one_test-specific project config (after shared env)
# ============================================================================

# Special case: pytest-dev projects (run_one_test needs finer detection)
PYTEST_SRC_DIR=""
if [ "${APR_IS_PYTESTDEV:-0}" = "1" ]; then
  # pytest repo layout is typically:
  # - newer: /testbed/src/pytest/ + /testbed/src/_pytest/
  # - older: /testbed/src/pytest.py + /testbed/src/_pytest/
  # So support both layouts.
  if [ -f "/testbed/src/pytest.py" ] || [ -d "/testbed/src/_pytest" ] || [ -d "/testbed/src/pytest" ]; then
    PYTEST_SRC_DIR="/testbed/src"
  elif [ -d "/testbed/_pytest" ] || [ -d "/testbed/pytest" ]; then
    # Fallback: some layouts may have packages at repo root.
    PYTEST_SRC_DIR="/testbed"
  fi
  if [ -n "$PYTEST_SRC_DIR" ]; then
    export PYTHONPATH="$PYTEST_SRC_DIR:$SITE_DIR:${PYTHONPATH:-}"
  fi
  # pytest-dev (older commits): some trees reference _pytest._version but don't ship it.
  if [ -n "${PYTEST_SRC_DIR:-}" ] && [ -d "${PYTEST_SRC_DIR}/_pytest" ] && [ ! -f "${PYTEST_SRC_DIR}/_pytest/_version.py" ]; then
    echo "[WARN] pytest-dev: missing ${PYTEST_SRC_DIR}/_pytest/_version.py; creating minimal stub for import compatibility" >&2
    cat > "${PYTEST_SRC_DIR}/_pytest/_version.py" <<'EOF_APR_PYTEST_VERSION'
# Auto-generated by APR verification harness for compatibility.
version = "0.0"
__version__ = version
version_tuple = (0, 0, 0)
EOF_APR_PYTEST_VERSION
  fi
fi

# xarray/matplotlib: avoid shadowing matplotlib with any pip-installed copy in SITE_DIR
if [ "${APR_IS_XARRAY:-0}" = "1" ] || [ "${APR_IS_MATPLOTLIB:-0}" = "1" ]; then
  rm -rf "$SITE_DIR/matplotlib" "$SITE_DIR"/matplotlib-*.dist-info "$SITE_DIR"/matplotlib-*.egg-info 2>/dev/null || true
fi

# matplotlib: Fix pyparsing.warnings import error
if [ "${APR_IS_MATPLOTLIB:-0}" = "1" ] && [ ! -d "$SITE_DIR/pyparsing" ]; then
  mkdir -p "$SITE_DIR/pyparsing"
  cat > "$SITE_DIR/pyparsing/__init__.py" <<'EOF_APR_PYPARSING_INIT'
# Auto-generated by APR verification harness for matplotlib compatibility.
EOF_APR_PYPARSING_INIT
  cat > "$SITE_DIR/pyparsing/warnings.py" <<'EOF_APR_PYPARSING_WARNINGS'
# Auto-generated by APR verification harness for matplotlib compatibility.
import warnings
class PyparsingDeprecationWarning(DeprecationWarning):
    pass
__all__ = ['PyparsingDeprecationWarning']
EOF_APR_PYPARSING_WARNINGS
  echo "[INFO] matplotlib: Created pyparsing.warnings stub in $SITE_DIR" >&2
fi

# SymPy: some tests import `raises` from sympy.utilities.pytest
if [ "${APR_IS_SYMPY:-0}" = "1" ] && [ -f "/testbed/sympy/utilities/pytest.py" ]; then
  if ! grep -q "APR_SYMPY_RAISES_SHIM" "/testbed/sympy/utilities/pytest.py" 2>/dev/null; then
    cat >> "/testbed/sympy/utilities/pytest.py" <<'EOF_APR_SYMPY_RAISES'
# APR_SYMPY_RAISES_SHIM: added by APR verification harness for backwards compatibility.
try:
    import pytest as _apr_pytest  # type: ignore[import]
except Exception:  # pragma: no cover
    _apr_pytest = None
if _apr_pytest is not None and 'raises' not in globals():
    raises = _apr_pytest.raises  # type: ignore[assignment]
EOF_APR_SYMPY_RAISES
  fi
fi

# matplotlib: fix missing _c_internal_utils (C extension module).
# Some matplotlib instances fail because _c_internal_utils is not compiled.
# We rebuild matplotlib's C extensions in-place to fix this.
if [ "${APR_IS_MATPLOTLIB:-0}" = "1" ] && [ -n "$PY" ]; then
  if [ ! -f "/testbed/setup.py" ]; then
    echo "[WARN] matplotlib: /testbed/setup.py missing; cannot rebuild C extensions" >&2
  elif ! "$PY" -c "import matplotlib._c_internal_utils" >/dev/null 2>&1; then
    echo "[INFO] matplotlib: _c_internal_utils missing, rebuilding C extensions..." >&2
    if ! "$PY" -m pip --version >/dev/null 2>&1; then
      echo "[WARN] matplotlib: pip not available, cannot install build dependencies" >&2
    else
      # Install build dependencies if needed
      if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
        echo "[INFO] matplotlib: installing numpy (required for build)..." >&2
        "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" numpy >/tmp/apr_matplotlib_numpy_$$.log 2>&1 || true
      fi
      # Ensure /testbed is in PYTHONPATH for in-place build
      if [ -n "${PYTHONPATH:-}" ]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
        export PYTHONPATH="/testbed:${PYTHONPATH}"
      else
        export PYTHONPATH="/testbed"
      fi
      # Build C extensions in-place
      BUILD_LOG="/tmp/apr_matplotlib_build_$$.log"
      if (cd /testbed && "$PY" setup.py build_ext --inplace >"$BUILD_LOG" 2>&1); then
        # Verify _c_internal_utils is now available (use /testbed in PYTHONPATH for verification)
        if (cd /tmp && PYTHONPATH="/testbed:${PYTHONPATH:-}" "$PY" -c "import matplotlib._c_internal_utils" >/dev/null 2>&1); then
          echo "[INFO] matplotlib: C extensions rebuilt successfully" >&2
        else
          echo "[WARN] matplotlib: build completed but _c_internal_utils verification failed (may be import path issue)" >&2
          tail -30 "$BUILD_LOG" >&2 || true
        fi
      else
        echo "[WARN] matplotlib: build_ext failed" >&2
        tail -50 "$BUILD_LOG" >&2 || true
      fi
    fi
  else
    echo "[INFO] matplotlib: _c_internal_utils already available" >&2
  fi
fi

# Check if selected Python version matches required version (from SWE-bench spec)
# If version mismatch and we're using system Python, try to bootstrap correct version
if [ -n "${APR_REQUIRED_PYTHON_VERSION:-}" ] && [ "$PY" = "/usr/bin/python3" ] && [ -f "/miniconda.sh" ]; then
  CURRENT_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
  if [ -n "$CURRENT_PY_VER" ] && [ "$CURRENT_PY_VER" != "${APR_REQUIRED_PYTHON_VERSION}" ]; then
    echo "[INFO] Python version mismatch: required=${APR_REQUIRED_PYTHON_VERSION}, current=$CURRENT_PY_VER, bootstrapping..." >&2
    CONDA_BASE="/tmp/apr_miniconda3"
    PY_REQ="${CONDA_BASE}/envs/apr_py${APR_REQUIRED_PYTHON_VERSION//./}/bin/python"
    if [ ! -x "$PY_REQ" ]; then
      echo "[INFO] Bootstrapping miniconda Python ${APR_REQUIRED_PYTHON_VERSION} env (first-time)..." >&2
      if [ ! -x "${CONDA_BASE}/bin/conda" ]; then
        bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_miniconda_install.log 2>&1 || true
      fi
      if [ -x "${CONDA_BASE}/bin/conda" ]; then
        "${CONDA_BASE}/bin/conda" create -y -p "${CONDA_BASE}/envs/apr_py${APR_REQUIRED_PYTHON_VERSION//./}" "python=${APR_REQUIRED_PYTHON_VERSION}" pip >/tmp/apr_miniconda_create_py${APR_REQUIRED_PYTHON_VERSION//./}.log 2>&1 || true
      fi
    fi
    if [ -x "$PY_REQ" ]; then
      PY="$PY_REQ"
      echo "[INFO] Switched to required Python version: PY=$PY" >&2
      "$PY" -V >&2 || true
      "$PY" -c "import sys; print('sys.executable=', sys.executable)" >&2 || true
      # Upgrade pip to ensure compatibility with modern packages
      if "$PY" -m pip --version >/dev/null 2>&1; then
        echo "[INFO] Upgrading pip for Python ${APR_REQUIRED_PYTHON_VERSION}..." >&2
        "$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
      fi
    else
      echo "[WARN] Failed to bootstrap Python ${APR_REQUIRED_PYTHON_VERSION}; continuing with system Python $CURRENT_PY_VER" >&2
      tail -200 /tmp/apr_miniconda_create_py${APR_REQUIRED_PYTHON_VERSION//./}.log 2>/dev/null || true
    fi
  fi
fi

# pytest-dev at older commits is incompatible with Python 3.10's import hook API
# (AssertionRewritingHook missing find_spec). When the image lacks a conda testbed,
# bootstrap a local miniconda env with Python 3.9 and use it for pytest-dev only.
if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ "$PY" = "/usr/bin/python3" ] && [ -f "/miniconda.sh" ]; then
  CONDA_BASE="/tmp/apr_miniconda3"
  PY39="${CONDA_BASE}/envs/apr_py39/bin/python"
  if [ ! -x "$PY39" ]; then
    echo "[INFO] pytest-dev: bootstrapping miniconda Python 3.9 env (first-time)..." >&2
    if [ ! -x "${CONDA_BASE}/bin/conda" ]; then
      bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_miniconda_install.log 2>&1 || true
    fi
    if [ -x "${CONDA_BASE}/bin/conda" ]; then
      "${CONDA_BASE}/bin/conda" create -y -p "${CONDA_BASE}/envs/apr_py39" python=3.9 pip >/tmp/apr_miniconda_create.log 2>&1 || true
    fi
  fi
  if [ -x "$PY39" ]; then
    PY="$PY39"
    echo "[INFO] pytest-dev: switched to PY=$PY" >&2
    "$PY" -V >&2 || true
    "$PY" -c "import sys; print('sys.executable=', sys.executable)" >&2 || true
    # Upgrade pip to ensure compatibility
    if "$PY" -m pip --version >/dev/null 2>&1; then
      echo "[INFO] pytest-dev: upgrading pip..." >&2
      "$PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
    fi
  else
    echo "[WARN] pytest-dev: failed to bootstrap Python 3.9 env; continuing with system Python" >&2
    tail -200 /tmp/apr_miniconda_create.log 2>/dev/null || true
  fi
fi

# Install SWE-bench spec pip packages deterministically (best-effort; helps missing deps like mpmath).
if "$PY" -m pip --version >/dev/null 2>&1; then
  # CRITICAL: Support both file-based and direct variable formats
  if [ -n "${APR_PIP_PACKAGES_FILE:-}" ] && [ -f "$APR_PIP_PACKAGES_FILE" ]; then
    # File-based format: already one package per line
    cp "$APR_PIP_PACKAGES_FILE" /tmp/apr_pip_pkgs.txt || true
  elif [ -n "${APR_PIP_PACKAGES:-}" ]; then
    # Direct variable format: space-separated, convert to lines
    echo "$APR_PIP_PACKAGES" | tr ' ' '\n' | sed '/^$/d' > /tmp/apr_pip_pkgs.txt || true
  fi
  if [ -s /tmp/apr_pip_pkgs.txt ]; then
    echo "[INFO] Installing pip packages from APR_PIP_PACKAGES into $SITE_DIR..." >&2
    PIP_INSTALL_LOG="/tmp/apr_pip_install_$$.log"
    if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" -r /tmp/apr_pip_pkgs.txt >"$PIP_INSTALL_LOG" 2>&1; then
      echo "[WARN] Some pip packages failed to install. Log:" >&2
      tail -50 "$PIP_INSTALL_LOG" >&2 || true
      # Continue anyway (best-effort installation)
    else
      echo "[INFO] Pip packages installed successfully" >&2
    fi
  fi
fi

# Ensure pytest import works (prefer preinstalled; otherwise install into /tmp/apr_site).
if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "[ERROR] pytest is missing and pip is not available in $PY" >&2
    exit 2
  fi
  if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ -n "${PYTEST_SRC_DIR:-}" ]; then
    echo "[INFO] pytest-dev: pytest missing, bootstrapping in-tree deps into $SITE_DIR..." >&2
    # Ensure PYTHONPATH includes SITE_DIR so newly installed modules can be found
    if [ -n "${PYTHONPATH:-}" ]; then
      PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
      export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
    else
      export PYTHONPATH="$SITE_DIR"
    fi
    # Pre-install common pytest core dependencies to speed up bootstrap
    echo "[INFO] pytest-dev: pre-installing common pytest dependencies..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "atomicwrites" "iniconfig" "pluggy" "py" "packaging" "attrs" "more-itertools" "tomli" "exceptiongroup" >/tmp/apr_pytest_preinstall_$$.log 2>&1 || true
    # Now iteratively install missing dependencies until pytest can be imported
    MAX_ITER=20
    INSTALLED_MODS=""
    for _i in $(seq 1 $MAX_ITER); do
      OUT=$("$PY" -c "import pytest" 2>&1 || true)
      if [ -z "$OUT" ]; then
        echo "[INFO] pytest-dev: pytest import successful after $_i iteration(s)" >&2
        break
      fi
      MOD=$("$PY" - <<'PY'
import re,sys
t=sys.stdin.read()
m=re.search(r"No module named ['\\\"]([^'\\\"]+)['\\\"]", t)
print(m.group(1) if m else "")
PY
<<<"$OUT" 2>/dev/null || true)
      if [ -z "${MOD:-}" ]; then
        echo "[WARN] pytest-dev: could not resolve missing module from import error (iteration $_i):" >&2
        echo "$OUT" >&2
        # If we can't extract module name, try to continue anyway
        if [ "$_i" -ge 10 ]; then
          echo "[ERROR] pytest-dev: too many iterations without progress, giving up" >&2
          break
        fi
        continue
      fi
      # Skip if we already tried to install this module
      if echo "$INSTALLED_MODS" | grep -q "^${MOD}$"; then
        echo "[WARN] pytest-dev: module $MOD already installed but still missing, may have dependency issue" >&2
        if [ "$_i" -ge 15 ]; then
          echo "[ERROR] pytest-dev: stuck in dependency loop, giving up" >&2
          break
        fi
        continue
      fi
      INSTALLED_MODS="${INSTALLED_MODS}${INSTALLED_MODS:+$'\n'}${MOD}"
      echo "[INFO] pytest-dev: installing missing module: $MOD (iteration $_i)" >&2
      INSTALL_LOG="/tmp/apr_pytest_install_${MOD}_$$.log"
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "$MOD" >"$INSTALL_LOG" 2>&1; then
        echo "[WARN] pytest-dev: failed to install $MOD, log:" >&2
        tail -20 "$INSTALL_LOG" >&2 || true
      fi
      # Ensure PYTHONPATH is still set correctly after installation
      if [ -n "${PYTHONPATH:-}" ]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
        export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
      else
        export PYTHONPATH="$SITE_DIR"
      fi
    done
  else
    # Select pytest version based on Python version
    # Python 3.6: pytest 7.0.1 (last version supporting Python 3.6)
    # Python 3.7+: pytest 7.4.4
    PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    PYTEST_VER="7.4.4"
    if [ "$PY_VER" = "3.6" ]; then
      PYTEST_VER="7.0.1"
      echo "[INFO] Python 3.6 detected, using pytest ${PYTEST_VER} (compatible version)..." >&2
    fi
    INSTALL_LOG="/tmp/pytest_install_$$.log"
    echo "[INFO] Installing pytest==${PYTEST_VER} into $SITE_DIR (container /tmp)..." >&2
    # For Python 3.6, pytest 7.0.1 requires specific dependencies
    if [ "$PY_VER" = "3.6" ]; then
      # Install pytest dependencies explicitly for Python 3.6 compatibility
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "packaging>=17.1" "attrs>=17.4.0" "more-itertools>=4.0.0" "pluggy>=0.12,<1.0" "py>=1.5.0" "setuptools>=40.0" "six>=1.10.0" "toml>=0.9.4" "pytest==${PYTEST_VER}" >"$INSTALL_LOG" 2>&1; then
        echo "[ERROR] Failed to install pytest==${PYTEST_VER} with dependencies. Tail of pip log:" >&2
        tail -200 "$INSTALL_LOG" >&2 || true
        exit 2
      fi
    else
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pytest==${PYTEST_VER}" >"$INSTALL_LOG" 2>&1; then
        echo "[ERROR] Failed to install pytest==${PYTEST_VER}. Tail of pip log:" >&2
        tail -200 "$INSTALL_LOG" >&2 || true
        exit 2
      fi
    fi
    # Ensure PYTHONPATH is set immediately after installation (avoid duplicates)
    if [ -n "${PYTHONPATH:-}" ]; then
      # Remove duplicates and ensure SITE_DIR is first
      PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
      export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
    else
      export PYTHONPATH="$SITE_DIR"
    fi
  fi
fi

# Re-export to ensure ordering persists after installs above (avoid duplicates)
if [ -n "${PYTEST_SRC_DIR:-}" ]; then
  # Remove duplicates and ensure PYTEST_SRC_DIR is first, then SITE_DIR
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$PYTEST_SRC_DIR$" | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$PYTEST_SRC_DIR:$SITE_DIR:${PYTHONPATH}"
  else
    export PYTHONPATH="$PYTEST_SRC_DIR:$SITE_DIR"
  fi
else
  # Remove duplicates and ensure SITE_DIR is first
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
  else
    export PYTHONPATH="$SITE_DIR"
  fi
fi

# Debug: check if pytest files exist after installation
if [ -d "$SITE_DIR" ]; then
  if [ -f "$SITE_DIR/pytest.py" ] || [ -d "$SITE_DIR/pytest" ] || [ -d "$SITE_DIR/_pytest" ]; then
    echo "[DEBUG] pytest files found in $SITE_DIR" >&2
  else
    echo "[DEBUG] pytest files NOT found in $SITE_DIR, listing contents:" >&2
    ls -la "$SITE_DIR" | head -20 >&2 || true
  fi
fi

# Try importing pytest with detailed error output
PYTEST_IMPORT_OUT="/tmp/pytest_import_check_$$.log"
if ! "$PY" -c "import pytest; print('pytest_version=', getattr(pytest,'__version__','unknown'))" >"$PYTEST_IMPORT_OUT" 2>&1; then
  echo "[ERROR] CRITICAL: pytest still not importable after setup. PY=$PY SITE_DIR=$SITE_DIR PYTHONPATH=$PYTHONPATH" >&2
  echo "[ERROR] Import error output:" >&2
  cat "$PYTEST_IMPORT_OUT" >&2 || true
  # Try to diagnose: check if pytest is in site-packages
  "$PY" -c "import sys; print('sys.path:', sys.path)" >&2 || true
  exit 2
fi
rm -f "$PYTEST_IMPORT_OUT" 2>/dev/null || true

install_missing_module_from_file() {
  # $1: path to captured pytest output
  local out="$1"
  local mod=""
  mod=$("$PY" -c "import re,sys; t=open(sys.argv[1],'r',encoding='utf-8',errors='ignore').read(); m=re.search(r\"No module named ['\\\"]([^'\\\"]+)['\\\"]\", t); print(m.group(1) if m else '')" "$out" 2>/dev/null || true)
  if [ -z "$mod" ]; then
    if "$PY" -c "import sys; t=open(sys.argv[1],'r',encoding='utf-8',errors='ignore').read().lower(); sys.exit(0 if 'depends on mpmath' in t else 1)" "$out" 2>/dev/null; then
      mod="mpmath"
    fi
  fi
  if [ -n "$mod" ]; then
    echo "[INFO] Installing missing module: $mod" >&2
  "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "$mod" >/dev/null 2>&1 || return 1
    return 0
  fi
  return 1
}

# Fix for pylint and other projects that need 'py' package (pytest's legacy dependency).
# Some test modules import 'py._path.local' which requires the 'py' package.
if ! "$PY" -c "import py._path" >/dev/null 2>&1; then
  echo "[INFO] Installing 'py' package (required by some test modules like pylint)..." >&2
  "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" py >/dev/null 2>&1 || true
fi

# pytest-dev__pytest-5262: Install hypothesis dependency
# Some pytest-dev tests require hypothesis (e.g., testing/python/metafunc.py)
if [ "${APR_IS_PYTESTDEV:-0}" = "1" ] && [ "${APR_INSTANCE_ID:-}" = "pytest-dev__pytest-5262" ]; then
  if ! "$PY" -c "import hypothesis" >/dev/null 2>&1; then
    echo "[INFO] pytest-dev__pytest-5262: Installing hypothesis (required by testing/python/metafunc.py)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" hypothesis >/dev/null 2>&1 || true
  fi
fi

# Fix for Pallets/Flask projects that need flask module.
# For pallets/flask, flask is the project itself, so we should use the source checkout.
# However, if flask is missing or incompatible, we need to install a compatible version.
# Flask 2.0-2.3 has request_ctx in flask.globals, but Flask 3.0+ removed it.
# Strategy: Try to use flask from /testbed first (via pip install -e .), then install compatible version if needed.
if [ "${APR_IS_PALLETS:-0}" = "1" ]; then
  # First, try to use flask from /testbed (the project itself)
  # This should work after "pip install -e ." is run later in the script
  # But we check early to see if we need to install a fallback version
  FLASK_AVAILABLE=0
  if "$PY" -c "import flask" >/dev/null 2>&1; then
    # Check if request_ctx is available (required by test code)
    if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
      FLASK_AVAILABLE=1
      echo "[INFO] Flask with request_ctx available (from /testbed or existing install)" >&2
    else
      echo "[INFO] Flask found but request_ctx not available, may need compatible version..." >&2
    fi
  fi
  
  # If flask is not available or request_ctx is missing, install a compatible version
  # Flask version compatibility:
  # - Flask 2.3.x: requires Python 3.8+, has request_ctx
  # - Flask 2.0.x: requires Python 3.6+, has _request_ctx_stack (deprecated) but may not have request_ctx
  # - Flask 2.1+: has request_ctx, requires Python 3.7+
  # Strategy: Detect Python version and install appropriate Flask version
  if [ "$FLASK_AVAILABLE" -eq 0 ]; then
    PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    PY_MAJOR=${PY_VER%%.*}
    PY_MINOR=${PY_VER#*.}
    
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 8 ]; then
      # Python 3.8+: Install Flask 2.3.x (has request_ctx)
      echo "[INFO] Installing Flask 2.3.x (Python $PY_VER, has request_ctx)..." >&2
      if "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "flask>=2.3.0,<2.4.0" >/tmp/apr_flask_install_$$.log 2>&1; then
        if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
          echo "[INFO] Flask 2.3.x installed successfully with request_ctx support" >&2
        else
          echo "[WARN] Flask 2.3.x installed but request_ctx still not available" >&2
        fi
      else
        echo "[WARN] Failed to install Flask 2.3.x, log:" >&2
        tail -20 /tmp/apr_flask_install_$$.log >&2 || true
      fi
    elif [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -eq 7 ]; then
      # Python 3.7: Install Flask 2.1+ (has request_ctx)
      echo "[INFO] Installing Flask 2.1+ (Python $PY_VER, has request_ctx)..." >&2
      if "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "flask>=2.1.0,<2.4.0" >/tmp/apr_flask_install_$$.log 2>&1; then
        if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
          echo "[INFO] Flask 2.1+ installed successfully with request_ctx support" >&2
        else
          echo "[WARN] Flask 2.1+ installed but request_ctx still not available" >&2
        fi
      else
        echo "[WARN] Failed to install Flask 2.1+, log:" >&2
        tail -20 /tmp/apr_flask_install_$$.log >&2 || true
      fi
    else
      # Python 3.6: Install Flask 2.0.x (may not have request_ctx, but has _request_ctx_stack)
      # Note: Flask 2.0.x may not have request_ctx, but test code might need it
      # We'll install Flask 2.0.x and create a compatibility shim for request_ctx
      echo "[INFO] Installing Flask 2.0.x (Python $PY_VER, may not have request_ctx)..." >&2
      if "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "flask>=2.0.0,<2.1.0" >/tmp/apr_flask_install_$$.log 2>&1; then
        # Check if request_ctx is available (unlikely for Flask 2.0.x)
        if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
          echo "[INFO] Flask 2.0.x installed with request_ctx support" >&2
        else
          echo "[WARN] Flask 2.0.x installed but request_ctx not available (expected for Flask 2.0.x)" >&2
          echo "[INFO] Creating compatibility shim for request_ctx using _request_ctx_stack..." >&2
          # Create a compatibility shim in SITE_DIR to provide request_ctx for Flask 2.0.x
          FLASK_GLOBALS_SHIM="$SITE_DIR/flask/globals.py"
          if [ -f "$FLASK_GLOBALS_SHIM" ]; then
            # Backup original file
            cp "$FLASK_GLOBALS_SHIM" "$FLASK_GLOBALS_SHIM.bak" 2>/dev/null || true
            # Add request_ctx compatibility shim if not already present
            if ! grep -q "APR_FLASK_REQUEST_CTX_SHIM" "$FLASK_GLOBALS_SHIM" 2>/dev/null; then
              # Use a temporary Python script file to avoid heredoc issues with arguments
              SHIM_SCRIPT="/tmp/apr_flask_shim_$$.py"
              cat > "$SHIM_SCRIPT" <<'PY_SHIM_EOF'
import sys
import os
if len(sys.argv) < 2:
    print("Error: shim_file path required", file=sys.stderr)
    sys.exit(1)
shim_file = sys.argv[1]
try:
    if not os.path.exists(shim_file):
        print("Error: file not found: {0}".format(shim_file), file=sys.stderr)
        sys.exit(1)
    
    with open(shim_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if our shim is already present
    if 'APR_FLASK_REQUEST_CTX_SHIM' in content:
        print("Shim already present")
        sys.exit(0)
    
    # For Flask 2.0.x, request_ctx does NOT exist natively
    # Always add the shim regardless of what's in the file
    # (Flask 2.0.x may mention request_ctx in comments/docs, but it's not actually defined)
    
    # Add compatibility shim at the end of the file
    shim_code = '''
# APR_FLASK_REQUEST_CTX_SHIM: Compatibility shim for Flask 2.0.x
# Flask 2.0.x uses _request_ctx_stack, but test code expects request_ctx (introduced in Flask 2.3)
# This shim provides request_ctx as a property that accesses _request_ctx_stack.top
try:
    from flask import _request_ctx_stack
    # Create a property-like object that mimics request_ctx behavior
    # In Flask 2.3+, request_ctx is a LocalProxy that wraps _request_ctx_stack
    # In Flask 2.0.x, we need to provide a compatible interface
    class _RequestCtxShim:
        @property
        def top(self):
            return _request_ctx_stack.top if _request_ctx_stack else None
        
        def _get_current_object(self):
            # Flask 2.0.x uses _request_ctx_stack.top directly
            # This method is called by conftest.py to get the current request context
            return _request_ctx_stack.top if _request_ctx_stack else None
        
        def pop(self):
            # Flask 2.0.x: pop from _request_ctx_stack
            if _request_ctx_stack:
                return _request_ctx_stack.pop()
            return None
        
        def __bool__(self):
            # Check if there's a current context
            return _request_ctx_stack.top is not None if _request_ctx_stack else False
        
        def __getattr__(self, name):
            ctx = _request_ctx_stack.top if _request_ctx_stack else None
            if ctx:
                return getattr(ctx, name)
            raise AttributeError("'{0}' object has no attribute '{1}'".format(type(self).__name__, name))
    
    request_ctx = _RequestCtxShim()
except (ImportError, AttributeError):
    # If _request_ctx_stack is not available, create a minimal stub
    class _RequestCtxStub:
        @property
        def top(self):
            return None
        
        def _get_current_object(self):
            return None
        
        def pop(self):
            return None
        
        def __bool__(self):
            return False
    request_ctx = _RequestCtxStub()
'''
    
    # Append shim if not already present
    if 'APR_FLASK_REQUEST_CTX_SHIM' not in content:
        with open(shim_file, 'a', encoding='utf-8') as f:
            f.write(shim_code)
        print("Shim added successfully")
    else:
        print("Shim already present")
except Exception as e:
    print("Error adding shim: {0}".format(e), file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PY_SHIM_EOF
              if "$PY" "$SHIM_SCRIPT" "$FLASK_GLOBALS_SHIM" >/tmp/apr_flask_shim_$$.log 2>&1; then
                echo "[INFO] request_ctx shim script executed successfully" >&2
                cat /tmp/apr_flask_shim_$$.log >&2 || true
              else
                echo "[WARN] Failed to add request_ctx shim, log:" >&2
                cat /tmp/apr_flask_shim_$$.log >&2 || true
              fi
              rm -f "$SHIM_SCRIPT" 2>/dev/null || true
            fi
            # Verify shim works
            if "$PY" -c "from flask.globals import request_ctx" >/dev/null 2>&1; then
              echo "[INFO] request_ctx compatibility shim created successfully" >&2
            else
              echo "[WARN] request_ctx shim created but import still fails" >&2
            fi
          else
            echo "[WARN] Flask globals.py not found at $FLASK_GLOBALS_SHIM, cannot create shim" >&2
          fi
        fi
      else
        echo "[WARN] Failed to install Flask 2.0.x, log:" >&2
        tail -20 /tmp/apr_flask_install_$$.log >&2 || true
      fi
    fi
  fi
fi

# Fix for Django projects that need common dependencies (asgiref, pytz, etc.).
# Django requires several packages that may not be installed in some SIF images.
if [ "${APR_IS_DJANGO:-0}" = "1" ]; then
  # Install asgiref (required for Django's ASGI support)
  if ! "$PY" -c "import asgiref" >/dev/null 2>&1; then
    echo "[INFO] Installing 'asgiref' package (required by Django)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" asgiref >/dev/null 2>&1 || true
  fi
  # Install pytz (required for Django's timezone support)
  if ! "$PY" -c "import pytz" >/dev/null 2>&1; then
    echo "[INFO] Installing 'pytz' package (required by Django)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" pytz >/dev/null 2>&1 || true
  fi
  # Install sqlparse (required for Django's database support)
  if ! "$PY" -c "import sqlparse" >/dev/null 2>&1; then
    echo "[INFO] Installing 'sqlparse' package (required by Django)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" sqlparse >/dev/null 2>&1 || true
  fi
  # Install PostgreSQL client (psql) for Django dbshell tests
  # Some Django tests require psql command-line tool (e.g., test_postgresql.PostgreSqlDbshellCommandTestCase)
  if ! command -v psql >/dev/null 2>&1; then
    echo "[INFO] Django: psql not found, attempting to install PostgreSQL client..." >&2
    PSQL_INSTALLED=0
    # Strategy 1: Try conda install (works in writable conda environments)
    # Detect CONDA_BASE from PY path or common locations
    DJANGO_CONDA_BASE=""
    if [ -n "${CONDA_BASE:-}" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
      DJANGO_CONDA_BASE="${CONDA_BASE}"
    elif [ -x "/opt/miniconda3/bin/conda" ]; then
      DJANGO_CONDA_BASE="/opt/miniconda3"
    elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
      DJANGO_CONDA_BASE="/tmp/apr_miniconda3"
    fi
    if [ -n "$DJANGO_CONDA_BASE" ] && [ -x "${DJANGO_CONDA_BASE}/bin/conda" ]; then
      echo "[INFO] Django: trying to install psql via conda (CONDA_BASE=${DJANGO_CONDA_BASE})..." >&2
      source "${DJANGO_CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
      # Try installing into testbed environment first
      if "${DJANGO_CONDA_BASE}/bin/conda" install -n testbed -c conda-forge postgresql -y >/tmp/apr_conda_install_psql.log 2>&1; then
        # Activate testbed environment to ensure psql is in PATH
        source "${DJANGO_CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate testbed 2>/dev/null || true
        # Also add testbed bin to PATH explicitly
        export PATH="${DJANGO_CONDA_BASE}/envs/testbed/bin:${PATH}"
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via conda" >&2
        else
          # Check if psql exists in testbed bin directory
          if [ -x "${DJANGO_CONDA_BASE}/envs/testbed/bin/psql" ]; then
            export PATH="${DJANGO_CONDA_BASE}/envs/testbed/bin:${PATH}"
            PSQL_INSTALLED=1
            echo "[INFO] Django: psql found in testbed bin, added to PATH" >&2
          else
            echo "[WARN] Django: conda install completed but psql not found" >&2
          fi
        fi
      else
        # If testbed environment is not writable, try installing to a writable location
        echo "[WARN] Django: conda install to testbed failed (may be read-only), trying writable location..." >&2
        tail -20 /tmp/apr_conda_install_psql.log >&2 || true
        # Try installing to a temporary conda environment in /tmp
        PSQL_ENV_DIR="/tmp/apr_psql_env_$$"
        if "${DJANGO_CONDA_BASE}/bin/conda" create -y -p "$PSQL_ENV_DIR" -c conda-forge postgresql >/tmp/apr_conda_install_psql_tmp.log 2>&1; then
          # Add psql from temporary environment to PATH
          if [ -x "${PSQL_ENV_DIR}/bin/psql" ]; then
            export PATH="${PSQL_ENV_DIR}/bin:${PATH}"
            PSQL_INSTALLED=1
            echo "[INFO] Django: psql installed to writable location $PSQL_ENV_DIR and added to PATH" >&2
          fi
        else
          echo "[WARN] Django: conda install to writable location also failed" >&2
          tail -20 /tmp/apr_conda_install_psql_tmp.log >&2 || true
        fi
      fi
    fi
    # Strategy 2: Try system package manager (may fail in read-only containers)
    if [ "$PSQL_INSTALLED" -eq 0 ]; then
      if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Django: trying to install psql via apt-get..." >&2
        DEBIAN_FRONTEND=noninteractive apt-get update >/tmp/apr_apt_update.log 2>&1 && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends postgresql-client >/tmp/apr_apt_install_psql.log 2>&1 || true
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via apt-get" >&2
        fi
      elif command -v yum >/dev/null 2>&1; then
        echo "[INFO] Django: trying to install psql via yum..." >&2
        yum install -y postgresql >/tmp/apr_yum_install_psql.log 2>&1 || true
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via yum" >&2
        fi
      elif command -v apk >/dev/null 2>&1; then
        echo "[INFO] Django: trying to install psql via apk..." >&2
        apk add --no-cache postgresql-client >/tmp/apr_apk_install_psql.log 2>&1 || true
        if command -v psql >/dev/null 2>&1; then
          PSQL_INSTALLED=1
          echo "[INFO] Django: psql installed successfully via apk" >&2
        fi
      fi
    fi
    # Verify installation
    if [ "$PSQL_INSTALLED" -eq 0 ]; then
      echo "[WARN] Django: failed to install psql (tests requiring psql may fail)" >&2
      echo "[WARN] Django: checked conda and system package managers, but installation failed" >&2
    fi
  fi
fi

# Fix for Seaborn projects that need matplotlib and other dependencies.
# Seaborn requires Python 3.9+ and matplotlib, numpy, pandas, scipy.
if [ "${APR_IS_SEABORN:-0}" = "1" ]; then
  # Ensure Python version is 3.9+ (seaborn requirement)
  PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
  PY_MAJOR=${PY_VER%%.*}
  PY_MINOR=${PY_VER#*.}
  
  # If Python version is < 3.9, bootstrap Python 3.9
  if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]) && [ -f "/miniconda.sh" ]; then
    echo "[INFO] Seaborn: Python version $PY_VER < 3.9, bootstrapping Python 3.9..." >&2
    CONDA_BASE=""
    if [ -x "/opt/miniconda3/bin/conda" ]; then
      CONDA_BASE="/opt/miniconda3"
    elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
      CONDA_BASE="/tmp/apr_miniconda3"
    else
      if bash /miniconda.sh -b -p /opt/miniconda3 >/tmp/apr_seaborn_conda_install_$$.log 2>&1; then
        CONDA_BASE="/opt/miniconda3"
      else
        CONDA_BASE="/tmp/apr_miniconda3"
        bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_seaborn_conda_install_$$.log 2>&1 || true
      fi
    fi
    
    if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
      PY39_ENV="${CONDA_BASE}/envs/apr_seaborn_py39"
      PY39="${PY39_ENV}/bin/python"
      
      if [ ! -x "$PY39" ]; then
        echo "[INFO] Seaborn: creating Python 3.9 environment..." >&2
        source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
        "${CONDA_BASE}/bin/conda" create -y -p "$PY39_ENV" python=3.9 pip >/tmp/apr_seaborn_create_py39_$$.log 2>&1 || true
      fi
      
      if [ -x "$PY39" ]; then
        PY="$PY39"
        echo "[INFO] Seaborn: switched to Python 3.9: PY=$PY" >&2
        "$PY" -V >&2 || true
        # Ensure pip is available
        if ! "$PY" -m pip --version >/dev/null 2>&1; then
          "${CONDA_BASE}/bin/conda" install -y -p "$PY39_ENV" pip >/tmp/apr_seaborn_install_pip_$$.log 2>&1 || true
        fi
        # Reinstall pytest for Python 3.9
        if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
          "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pytest==7.4.4" >/tmp/apr_seaborn_pytest_$$.log 2>&1 || true
        fi
        # Update PYTHONPATH
        if [ -n "${PYTHONPATH:-}" ]; then
          PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
          export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
        else
          export PYTHONPATH="$SITE_DIR"
        fi
      else
        echo "[WARN] Seaborn: failed to bootstrap Python 3.9, continuing with current Python $PY_VER" >&2
      fi
    fi
  fi
  
  # Install matplotlib and dependencies if not available
  if ! "$PY" -c "import matplotlib" >/dev/null 2>&1; then
    echo "[INFO] Seaborn: installing matplotlib (required dependency)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" matplotlib >/tmp/apr_seaborn_matplotlib_$$.log 2>&1 || true
    if "$PY" -c "import matplotlib" >/dev/null 2>&1; then
      echo "[INFO] Seaborn: matplotlib installed successfully" >&2
    else
      echo "[WARN] Seaborn: matplotlib installation failed, log:" >&2
      tail -30 /tmp/apr_seaborn_matplotlib_$$.log >&2 || true
    fi
  fi
  
  # Install numpy if not available
  if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "[INFO] Seaborn: installing numpy (required dependency)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" numpy >/tmp/apr_seaborn_numpy_$$.log 2>&1 || true
  fi
  
  # Install pandas if not available
  if ! "$PY" -c "import pandas" >/dev/null 2>&1; then
    echo "[INFO] Seaborn: installing pandas (required dependency)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" pandas >/tmp/apr_seaborn_pandas_$$.log 2>&1 || true
  fi
  
  # Install scipy if not available
  if ! "$PY" -c "import scipy" >/dev/null 2>&1; then
    echo "[INFO] Seaborn: installing scipy (required dependency)..." >&2
    "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" scipy >/tmp/apr_seaborn_scipy_$$.log 2>&1 || true
  fi
fi

cd /testbed
git config --global --add safe.directory /testbed || true

# Django: set up environment variables and initialize apps before any test execution
# This must be done early to avoid AppRegistryNotReady errors during test collection
if [ "${APR_IS_DJANGO:-0}" = "1" ]; then
  # Dynamically detect Django settings module (test_sqlite is common but not always present)
  # Try common Django settings modules in order of preference
  # Strategy: Check files first (more reliable), then try imports
  DJANGO_SETTINGS=""
  # Priority 1: Check if test_sqlite.py file exists (most reliable)
  if [ -f "/testbed/tests/test_sqlite.py" ] || [ -d "/testbed/tests/test_sqlite" ]; then
    DJANGO_SETTINGS="test_sqlite"
  # Priority 2: Check if tests/settings.py exists
  elif [ -f "/testbed/tests/settings.py" ]; then
    DJANGO_SETTINGS="tests.settings"
  # Priority 3: Try importing test_sqlite (may work if in PYTHONPATH)
  elif "$PY" -c "import test_sqlite" >/dev/null 2>&1; then
    DJANGO_SETTINGS="test_sqlite"
  # Priority 4: Fallback to test_sqlite (Django projects in SWE-bench typically use this)
  else
    # Django projects in SWE-bench typically use test_sqlite, so use it as default
    # Even if import fails now, it might work after pip install -e . adds /testbed to PYTHONPATH
    DJANGO_SETTINGS="test_sqlite"
  fi
  # Always set DJANGO_SETTINGS_MODULE (Django requires it)
  export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS"
  export LANG=en_US.UTF-8
  export LC_ALL=en_US.UTF-8
  export PYTHONIOENCODING=utf8
  export LANGUAGE=en_US:en
  # Initialize Django apps before test execution to avoid AppRegistryNotReady
  # This is done early so all tests (both run_one_test and validate) can use it
  # IMPORTANT: Do this AFTER pip install -e . so that /testbed is in PYTHONPATH
  echo "[INFO] Django: Initializing Django apps before test execution..." >&2
  DJANGO_INIT_LOG="/tmp/django_init_$$.log"
  # Use the same settings module detection logic as above
  # But now /testbed should be in PYTHONPATH (from pip install -e .)
  DJANGO_SETTINGS_INIT=""
  if [ -n "${DJANGO_SETTINGS_MODULE:-}" ]; then
    DJANGO_SETTINGS_INIT="${DJANGO_SETTINGS_MODULE}"
  else
    # Try to detect settings module dynamically (same priority as above)
    # Now that /testbed is in PYTHONPATH, test_sqlite should be importable
    if "$PY" -c "import test_sqlite" >/dev/null 2>&1; then
      DJANGO_SETTINGS_INIT="test_sqlite"
    elif [ -f "/testbed/tests/test_sqlite.py" ] || [ -d "/testbed/tests/test_sqlite" ]; then
      DJANGO_SETTINGS_INIT="test_sqlite"
    elif [ -f "/testbed/tests/settings.py" ]; then
      DJANGO_SETTINGS_INIT="tests.settings"
    else
      # Fallback to test_sqlite (Django projects in SWE-bench typically use this)
      DJANGO_SETTINGS_INIT="test_sqlite"
    fi
  fi
  # Ensure /testbed and /testbed/tests are in PYTHONPATH for test_sqlite import
  # Django projects typically have test_sqlite in tests/ directory
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | grep -v "^/testbed/tests$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="/testbed:/testbed/tests${PYTHONPATH:+:${PYTHONPATH}}"
  else
    export PYTHONPATH="/testbed:/testbed/tests"
  fi
  
  # CRITICAL: Create sitecustomize.py so django.setup() runs inside *pytest process*
  # (fixes AppRegistryNotReady during collection).
  # This must be done BEFORE any pytest collection starts, and SITE_DIR must be in PYTHONPATH
  if [ -f "/testbed/django/__init__.py" ] || [ -f "/testbed/tests/test_sqlite.py" ]; then
    if [ -n "${SITE_DIR:-}" ] && [ -d "$SITE_DIR" ]; then
      __DJANGO_SITECUSTOMIZE_HEREDOC__
      echo "[INFO] Django: Created sitecustomize.py in $SITE_DIR for automatic Django initialization in pytest process" >&2
      CREATED_DJANGO_SITECUSTOMIZE=1
    fi
  fi
  if [ "${CREATED_DJANGO_SITECUSTOMIZE:-0}" != "1" ]; then
  if "$PY" -c "
import os
import sys
os.chdir('/testbed')
django_settings_init = \"${DJANGO_SETTINGS_INIT}\"
if django_settings_init:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', django_settings_init)
try:
    import django
    if not django.apps.apps.ready:
        django.setup()
    print('[INFO] Django apps initialized successfully', flush=True)
except Exception as e:
    print(f'[ERROR] Django initialization failed: {e}', file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
" >"$DJANGO_INIT_LOG" 2>&1; then
    cat "$DJANGO_INIT_LOG" >&2
    rm -f "$DJANGO_INIT_LOG"
  else
    echo "[ERROR] Django initialization failed. Error output:" >&2
    cat "$DJANGO_INIT_LOG" >&2
    rm -f "$DJANGO_INIT_LOG"
    echo "[WARN] Continuing despite Django initialization failure (may cause AppRegistryNotReady)..." >&2
  fi
  fi
fi

# Astropy: fix warnings logging issue without touching repo files.
# We observed failures like:
#   astropy.logger.LoggingError: Cannot disable warnings logging: warnings.showwarning was not set by this logger, or has been overridden
#
# Per Astropy logger implementation, this can happen during import-time default logger setup.
# The most robust approach here is to set Astropy's logger config explicitly via astropy.cfg
# in HOME (we already set HOME=/tmp for container runs), avoiding any workdir modifications.
if [ "${APR_IS_ASTROPY:-0}" = "1" ]; then
  export ASTROPY_LOGGER_LEVEL="WARNING"
  mkdir -p "${HOME}/.astropy/config" 2>/dev/null || true
  cat > "${HOME}/.astropy/config/astropy.cfg" <<'EOF_APR_ASTROPY_CFG'
[logger]
# Ensure warnings logging is enabled (prevents disable_warnings_logging path raising LoggingError
# when warnings.showwarning was overridden by the test runner).
log_warnings = True
EOF_APR_ASTROPY_CFG
fi

# Best-effort install (some repos require editable install for tests)
if [ "${APR_IS_ASTROPY:-0}" = "1" ]; then
  # Astropy must be importable from the workdir; editable install is required and may build extensions.
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "[ERROR] Astropy: pip is not available in $PY" >&2
    exit 2
  fi
  
  # CRITICAL FIX: Check Python version requirement from pyproject.toml or setup.py
  # Some astropy versions require Python 3.8+, but the image may have Python 3.6.
  # If version mismatch, bootstrap the required Python version.
  ASTROPY_REQUIRED_PY_VER=""
  if [ -f "/testbed/pyproject.toml" ]; then
    echo "[DEBUG] Astropy: checking pyproject.toml for requires-python..." >&2
    # First try simple grep (works even if Python can't parse TOML)
    _TMP_REQ=$(grep -i "requires-python" /testbed/pyproject.toml 2>/dev/null | head -1 | sed -n 's/.*>=\([0-9]\+\.[0-9]\+\).*/\1/p' || echo "")
    if [ -z "$_TMP_REQ" ]; then
      # Try Python parsing as fallback (if tomli/tomllib available)
      _TMP_REQ=$("$PY" -c "
try:
    import tomllib as _toml
except:
    try:
        import tomli as _toml
    except:
        _toml = None
if _toml:
    with open('/testbed/pyproject.toml', 'rb') as f:
        d = _toml.load(f)
        rp = d.get('project', {}).get('requires-python', '')
        if rp:
            import re
            m = re.search(r'>=?(\d+\.\d+)', rp)
            if m:
                print(m.group(1))
" 2>/dev/null || echo "")
    fi
    if [ -n "$_TMP_REQ" ]; then
      ASTROPY_REQUIRED_PY_VER="$_TMP_REQ"
      echo "[DEBUG] Astropy: detected requires-python from pyproject.toml: ${ASTROPY_REQUIRED_PY_VER}" >&2
    else
      echo "[DEBUG] Astropy: no requires-python found in pyproject.toml" >&2
    fi
  fi
  
  # If pyproject.toml doesn't specify, check setup.py
  if [ -z "$ASTROPY_REQUIRED_PY_VER" ] && [ -f "/testbed/setup.py" ]; then
    echo "[DEBUG] Astropy: checking setup.py for python_requires..." >&2
    ASTROPY_REQUIRED_PY_VER=$(grep -i "python_requires\|requires.*python" /testbed/setup.py 2>/dev/null | head -1 | sed -n 's/.*>=\([0-9]\+\.[0-9]\+\).*/\1/p' || echo "")
    if [ -n "$ASTROPY_REQUIRED_PY_VER" ]; then
      echo "[DEBUG] Astropy: detected python_requires from setup.py: ${ASTROPY_REQUIRED_PY_VER}" >&2
    fi
  fi
  
  # If we found a Python version requirement, check if current Python meets it
  if [ -n "$ASTROPY_REQUIRED_PY_VER" ]; then
    CURRENT_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    if [ -z "$CURRENT_PY_VER" ]; then
      CURRENT_PY_VER="0.0"
    fi
    CURRENT_PY_MAJOR=${CURRENT_PY_VER%%.*}
    CURRENT_PY_MINOR=${CURRENT_PY_VER#*.}
    REQUIRED_PY_MAJOR=${ASTROPY_REQUIRED_PY_VER%%.*}
    REQUIRED_PY_MINOR=${ASTROPY_REQUIRED_PY_VER#*.}
    
    # Ensure variables are numeric (use default 0 if empty)
    if [ -z "$CURRENT_PY_MAJOR" ]; then
      CURRENT_PY_MAJOR=0
    fi
    if [ -z "$CURRENT_PY_MINOR" ]; then
      CURRENT_PY_MINOR=0
    fi
    if [ -z "$REQUIRED_PY_MAJOR" ]; then
      REQUIRED_PY_MAJOR=0
    fi
    if [ -z "$REQUIRED_PY_MINOR" ]; then
      REQUIRED_PY_MINOR=0
    fi
    
    # Compare versions (use numeric comparison with error suppression)
    NEEDS_UPGRADE=0
    if [ "$CURRENT_PY_MAJOR" -lt "$REQUIRED_PY_MAJOR" ] 2>/dev/null; then
      NEEDS_UPGRADE=1
    elif [ "$CURRENT_PY_MAJOR" -eq "$REQUIRED_PY_MAJOR" ] 2>/dev/null && [ "$CURRENT_PY_MINOR" -lt "$REQUIRED_PY_MINOR" ] 2>/dev/null; then
      NEEDS_UPGRADE=1
    fi
    
    if [ "$NEEDS_UPGRADE" = "1" ] && [ -f "/miniconda.sh" ]; then
      echo "[INFO] Astropy: Python version mismatch detected: required>=${ASTROPY_REQUIRED_PY_VER}, current=${CURRENT_PY_VER}" >&2
      echo "[INFO] Astropy: bootstrapping Python ${ASTROPY_REQUIRED_PY_VER} environment..." >&2
      
      CONDA_BASE=""
      if [ -x "/opt/miniconda3/bin/conda" ]; then
        CONDA_BASE="/opt/miniconda3"
      elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
        CONDA_BASE="/tmp/apr_miniconda3"
      else
        # Install miniconda if not available
        if [ -f "/miniconda.sh" ]; then
          if bash /miniconda.sh -b -p /opt/miniconda3 >/tmp/apr_astropy_conda_install_$$.log 2>&1; then
            CONDA_BASE="/opt/miniconda3"
          else
            CONDA_BASE="/tmp/apr_miniconda3"
            bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_astropy_conda_install_$$.log 2>&1 || true
          fi
        fi
      fi
      
      if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
        PY_REQ_ENV="${CONDA_BASE}/envs/apr_astropy_py${ASTROPY_REQUIRED_PY_VER//./}"
        PY_REQ="${PY_REQ_ENV}/bin/python"
        
        if [ ! -x "$PY_REQ" ]; then
          echo "[INFO] Astropy: creating Python ${ASTROPY_REQUIRED_PY_VER} environment..." >&2
          source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
          "${CONDA_BASE}/bin/conda" create -y -p "$PY_REQ_ENV" "python=${ASTROPY_REQUIRED_PY_VER}" pip >/tmp/apr_astropy_create_py${ASTROPY_REQUIRED_PY_VER//./}.log 2>&1 || true
        fi
        
          if [ -x "$PY_REQ" ]; then
            PY="$PY_REQ"
            echo "[INFO] Astropy: switched to Python ${ASTROPY_REQUIRED_PY_VER}: PY=$PY" >&2
            "$PY" -V >&2 || true
            # Re-detect Python version after switching (needed for venv creation logic)
            ASTROPY_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            ASTROPY_PY_MAJOR=${ASTROPY_PY_VER%%.*}
            ASTROPY_PY_MINOR=${ASTROPY_PY_VER#*.}
            if [ -z "${ASTROPY_PY_MAJOR:-}" ] || [ -z "${ASTROPY_PY_MINOR:-}" ]; then
              ASTROPY_PY_MAJOR=0
              ASTROPY_PY_MINOR=0
            fi
          # Ensure pip is available
          if ! "$PY" -m pip --version >/dev/null 2>&1; then
            "${CONDA_BASE}/bin/conda" install -y -p "$PY_REQ_ENV" pip >/tmp/apr_astropy_install_pip_py${ASTROPY_REQUIRED_PY_VER//./}.log 2>&1 || true
          fi
        else
          echo "[WARN] Astropy: failed to bootstrap Python ${ASTROPY_REQUIRED_PY_VER}, continuing with current Python ${CURRENT_PY_VER}" >&2
        fi
      fi
    fi
  fi
  # Astropy images sometimes default to /usr/bin/python3 with distro site-packages
  # (/usr/lib/python3/dist-packages), which can contain an incompatible setuptools.
  # Also: older Astropy sources require setuptools.dep_util (removed in setuptools>=60.9, last version with dep_util is 60.5.0).
  #
  # Fix strategy (robust and deterministic):
  # - Keep build toolchain in its OWN writable dir ($ASTROPY_TOOLCHAIN_DIR)
  # - Keep pyproject build requirements in its OWN writable dir ($ASTROPY_BUILDREQ_DIR)
  # - Install runtime deps + Astropy into $SITE_DIR
  # - Put PYTHONPATH order as: toolchain -> buildreq -> site -> (rest)

  ASTROPY_TOOLCHAIN_DIR="/tmp/apr_astropy_toolchain"
  ASTROPY_BUILDREQ_DIR="/tmp/apr_astropy_buildreq"
  # IMPORTANT: /tmp can persist across repeated runs within the same container image/execution context.
  # If a previous attempt wrote an incompatible pip/setuptools into these dirs (e.g. py>=3.7-only),
  # then subsequent runs can fail before we even install (python -m pip will import from PYTHONPATH).
  # Always start with a clean slate for Astropy toolchain dirs.
  # CRITICAL FIX: More thorough cleanup to avoid file deletion conflicts during pip install
  rm -rf "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
  sleep 0.1  # Ensure filesystem sync to avoid race conditions
  mkdir -p "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
  # Ensure directory permissions are correct
  chmod 755 "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
  
  # CRITICAL: Fix SSL certificate issues for pip toolchain
  # Some pip installations in toolchain dirs have broken certifi paths
  # Set SSL_CERT_FILE and REQUESTS_CA_BUNDLE to use system certs or testbed Python's certifi
  if [ -n "$PY" ] && "$PY" -c "import certifi" >/dev/null 2>&1; then
    CERT_PATH=$("$PY" -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
    if [ -n "$CERT_PATH" ] && [ -f "$CERT_PATH" ]; then
      export SSL_CERT_FILE="$CERT_PATH"
      export REQUESTS_CA_BUNDLE="$CERT_PATH"
      export CURL_CA_BUNDLE="$CERT_PATH"
    fi
  fi
  # Fallback: try to find system certs
  if [ -z "${SSL_CERT_FILE:-}" ] || [ ! -f "${SSL_CERT_FILE:-}" ]; then
    for cert_path in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /usr/share/ca-certificates/mozilla/*.crt; do
      if [ -f "$cert_path" ]; then
        export SSL_CERT_FILE="$cert_path"
        export REQUESTS_CA_BUNDLE="$cert_path"
        export CURL_CA_BUNDLE="$cert_path"
        break
      fi
    done
  fi

  NEEDS_DEP_UTIL=0
  if [ -f "/testbed/astropy/wcs/setup_package.py" ] && grep -q "setuptools\\.dep_util" "/testbed/astropy/wcs/setup_package.py" 2>/dev/null; then
    NEEDS_DEP_UTIL=1
  fi

  # Helper function to create venv for Astropy (called from multiple places)
  _apr_astropy_create_venv() {
    local _PY_FOR_VENV="$1"
    local _APR_ASTROPY_USE_VENV="${APR_ASTROPY_USE_VENV:-1}"
    local _APR_ASTROPY_ORIG_PY_SAVED="${_APR_ASTROPY_ORIG_PY:-$_PY_FOR_VENV}"
    
    # Detect Python version
    local _PY_VER=$("$_PY_FOR_VENV" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    local _PY_MAJOR=${_PY_VER%%.*}
    local _PY_MINOR=${_PY_VER#*.}
    if [ -z "${_PY_MAJOR:-}" ] || [ -z "${_PY_MINOR:-}" ]; then
      _PY_MAJOR=0
      _PY_MINOR=0
    fi
    
    if [ "$_APR_ASTROPY_USE_VENV" = "1" ] && [ "$_PY_MAJOR" -eq 3 ] && [ "$_PY_MINOR" -ge 7 ]; then
      echo "[DEBUG] Astropy: venv creation condition met: Python ${_PY_MAJOR}.${_PY_MINOR} >= 3.7" >&2
      local _VENV_DIR="/tmp/apr_astropy_venv_py${_PY_MAJOR}${_PY_MINOR}"
      local _VENV_PY="${_VENV_DIR}/bin/python"
      local _VENV_MARKER="${_VENV_DIR}/.apr_venv_ready"
      
      # Check if venv already exists and is usable
      if [ -f "$_VENV_PY" ] && [ -f "$_VENV_MARKER" ] && "$_VENV_PY" -m pip --version >/dev/null 2>&1; then
        echo "$_VENV_PY"
        return 0
      fi
      
      # Create venv if it doesn't exist or is broken
      if [ ! -f "$_VENV_PY" ]; then
        rm -rf "$_VENV_DIR" 2>/dev/null || true
        if ! "$_PY_FOR_VENV" -m venv --without-pip "$_VENV_DIR" >/tmp/apr_astropy_venv_create_$$.log 2>&1; then
          echo "[WARN] Astropy: failed to create venv; continuing with base python. Tail:" >&2
          tail -120 /tmp/apr_astropy_venv_create_$$.log >&2 || true
          echo "$_PY_FOR_VENV"
          return 1
        fi
      fi
      
      if [ -f "$_VENV_PY" ]; then
        local _VENV_PY_ACTUAL="$_VENV_PY"
        echo "[INFO] Astropy: using dedicated venv PY=$_VENV_PY_ACTUAL" >&2
        "$_VENV_PY_ACTUAL" -V >&2 || true
        
        # Install pip manually using get-pip.py
        if ! "$_VENV_PY_ACTUAL" -m pip --version >/dev/null 2>&1; then
          local _GET_PIP_VENV="/tmp/get-pip_venv_$$.py"
          if command -v curl >/dev/null 2>&1; then
            curl -sSL https://bootstrap.pypa.io/get-pip.py >"$_GET_PIP_VENV" 2>/dev/null || true
          elif command -v wget >/dev/null 2>&1; then
            wget -q -O "$_GET_PIP_VENV" https://bootstrap.pypa.io/get-pip.py 2>/dev/null || true
          fi
          if [ -f "$_GET_PIP_VENV" ] && [ -s "$_GET_PIP_VENV" ]; then
            if "$_VENV_PY_ACTUAL" "$_GET_PIP_VENV" "pip==23.3.2" >/tmp/apr_astropy_venv_pip_$$.log 2>&1; then
              echo "[INFO] Astropy: pip installed in venv" >&2
            else
              echo "[WARN] Astropy: get-pip.py failed in venv" >&2
              tail -30 /tmp/apr_astropy_venv_pip_$$.log >&2 || true
            fi
            rm -f "$_GET_PIP_VENV" 2>/dev/null || true
          fi
        fi
        
        # Bootstrap venv (only if marker doesn't exist)
        if [ ! -f "$_VENV_MARKER" ]; then
          if "$_VENV_PY_ACTUAL" -m pip --version >/dev/null 2>&1; then
            # CRITICAL FIX: Uninstall existing setuptools first to avoid file deletion conflicts
            # If venv was created with --without-pip but base Python has setuptools, it may be copied into venv
            # Uninstalling first ensures clean installation of setuptools==60.5.0 (last version with dep_util)
            echo "[INFO] Astropy: checking for existing setuptools in venv..." >&2
            if "$_VENV_PY_ACTUAL" -c "import setuptools" >/dev/null 2>&1; then
              SETUPTOOLS_VER=$("$_VENV_PY_ACTUAL" -c "import setuptools; print(setuptools.__version__)" 2>/dev/null || echo "")
              if [ -n "$SETUPTOOLS_VER" ]; then
                SETUPTOOLS_MAJOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f1)
                # Check if version is >= 60.9 (dep_util removed) or >= 61
                SETUPTOOLS_MINOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f2)
                NEED_UNINSTALL=0
                if [ "$SETUPTOOLS_MAJOR" -ge 61 ] 2>/dev/null; then
                  NEED_UNINSTALL=1
                elif [ "$SETUPTOOLS_MAJOR" -eq 60 ] 2>/dev/null && [ "$SETUPTOOLS_MINOR" -ge 9 ] 2>/dev/null; then
                  NEED_UNINSTALL=1
                fi
                if [ "$NEED_UNINSTALL" -eq 1 ]; then
                  echo "[INFO] Astropy: venv has setuptools $SETUPTOOLS_VER >= 60.9 (dep_util removed), uninstalling first..." >&2
                  # CRITICAL: Create missing __pycache__ directories before uninstall to avoid OSError
                  VENV_SITE_PACKAGES=$("$_VENV_PY_ACTUAL" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
                  if [ -n "$VENV_SITE_PACKAGES" ]; then
                    mkdir -p "${VENV_SITE_PACKAGES}/pkg_resources/__pycache__" 2>/dev/null || true
                    mkdir -p "${VENV_SITE_PACKAGES}/setuptools/__pycache__" 2>/dev/null || true
                  fi
                  "$_VENV_PY_ACTUAL" -m pip uninstall -y setuptools >/tmp/apr_astropy_venv_uninstall_setuptools_$$.log 2>&1 || true
                  # Wait a moment for filesystem sync
                  sleep 0.1
                fi
              fi
            fi
            # CRITICAL: Install numpy in venv for wheel building and in-place builds
            # numpy is required for building astropy extensions (numpy/arrayobject.h)
            # Use --ignore-installed to avoid file deletion conflicts during installation
            # CRITICAL: Use setuptools==60.5.0 (last version with dep_util, removed in 60.9.0+)
            if "$_VENV_PY_ACTUAL" -m pip install -U --no-cache-dir --ignore-installed \
                "pip==23.3.2" "setuptools==60.5.0" "wheel" "setuptools_scm==7.1.0" "pytest==7.4.4" "tomli" "extension-helpers" "numpy" \
                >/tmp/apr_astropy_venv_bootstrap_$$.log 2>&1; then
              echo "[INFO] Astropy: venv bootstrap successful (including numpy)" >&2
              touch "$_VENV_MARKER" 2>/dev/null || true
              echo "$_VENV_PY_ACTUAL"
              return 0
            else
              echo "[WARN] Astropy: venv bootstrap failed; falling back to base python. Tail:" >&2
              tail -120 /tmp/apr_astropy_venv_bootstrap_$$.log >&2 || true
              echo "$_PY_FOR_VENV"
              return 1
            fi
          else
            echo "[WARN] Astropy: venv pip not available; falling back to base python" >&2
            echo "$_PY_FOR_VENV"
            return 1
          fi
        else
          echo "[INFO] Astropy: venv already bootstrapped (marker exists)" >&2
          # CRITICAL: Ensure numpy is available even if venv was bootstrapped before
          if ! "$_VENV_PY_ACTUAL" -c "import numpy" >/dev/null 2>&1; then
            echo "[INFO] Astropy: numpy not found in venv, installing..." >&2
            "$_VENV_PY_ACTUAL" -m pip install -U --no-cache-dir "numpy" >/tmp/apr_astropy_venv_numpy_$$.log 2>&1 || true
          fi
          # CRITICAL: Also ensure setuptools version is correct (may have been upgraded)
          if "$_VENV_PY_ACTUAL" -c "import setuptools" >/dev/null 2>&1; then
            SETUPTOOLS_VER=$("$_VENV_PY_ACTUAL" -c "import setuptools; print(setuptools.__version__)" 2>/dev/null || echo "")
            if [ -n "$SETUPTOOLS_VER" ]; then
              SETUPTOOLS_MAJOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f1)
                if [ "$SETUPTOOLS_MAJOR" -ge 61 ] 2>/dev/null; then
                  echo "[INFO] Astropy: venv setuptools $SETUPTOOLS_VER >= 61, downgrading to 60.5.0 (last version with dep_util)..." >&2
                  "$_VENV_PY_ACTUAL" -m pip install --no-cache-dir --force-reinstall --ignore-installed "setuptools==60.5.0" >/tmp/apr_astropy_venv_setuptools_fix_$$.log 2>&1 || true
                fi
            fi
          fi
          echo "$_VENV_PY_ACTUAL"
          return 0
        fi
      else
        echo "[WARN] Astropy: venv creation failed; using base python" >&2
        echo "$_PY_FOR_VENV"
        return 1
      fi
    else
      echo "[DEBUG] Astropy: venv creation skipped (Python ${_PY_MAJOR}.${_PY_MINOR} < 3.7 or disabled)" >&2
      echo "$_PY_FOR_VENV"
      return 0
    fi
  }

  # Ensure our dirs take precedence over distro site-packages.
  #
  # BUT: during toolchain bootstrap we must NOT have $ASTROPY_TOOLCHAIN_DIR on PYTHONPATH, otherwise
  # "$PY -m pip" may import a stale/incompatible pip from that directory and crash (py3.6 case).
  # CRITICAL FIX: Remove distro site-packages paths that may contain incompatible setuptools (e.g., 68.0.0)
  _APR_ASTROPY_ORIG_PYTHONPATH="${PYTHONPATH:-}"
  # Remove common distro site-packages paths that may shadow our toolchain
  _APR_ASTROPY_CLEANED_PYTHONPATH=$(echo "$_APR_ASTROPY_ORIG_PYTHONPATH" | tr ':' '\n' | \
    grep -vE '^/usr/lib/python[0-9.]+/dist-packages$|^/usr/lib/python[0-9.]+/site-packages$|^/usr/local/lib/python[0-9.]+/dist-packages$|^/usr/local/lib/python[0-9.]+/site-packages$' | \
    tr '\n' ':' | sed 's/:$//')
  _APR_ASTROPY_BOOTSTRAP_PYTHONPATH="$ASTROPY_BUILDREQ_DIR:$SITE_DIR:${_APR_ASTROPY_CLEANED_PYTHONPATH}"
  export PYTHONPATH="$_APR_ASTROPY_BOOTSTRAP_PYTHONPATH"

  TOOLCHAIN_LOG="/tmp/apr_astropy_toolchain_$$.log"
  # CRITICAL: Re-detect Python version here (after potential Python switching) for venv creation logic
  # If Python was switched earlier, ASTROPY_PY_MAJOR/MINOR may have been set, but we need to ensure
  # they reflect the current $PY value (which may have changed after switching)
  # Only detect if not already set (to preserve value from Python switching)
  if [ -z "${ASTROPY_PY_MAJOR:-}" ] || [ -z "${ASTROPY_PY_MINOR:-}" ] || [ "${ASTROPY_PY_MAJOR:-0}" = "0" ]; then
    ASTROPY_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    ASTROPY_PY_MAJOR=${ASTROPY_PY_VER%%.*}
    ASTROPY_PY_MINOR=${ASTROPY_PY_VER#*.}
    if [ -z "${ASTROPY_PY_MAJOR:-}" ] || [ -z "${ASTROPY_PY_MINOR:-}" ]; then
      ASTROPY_PY_MAJOR=0
      ASTROPY_PY_MINOR=0
    fi
  fi
  # Debug: log detected Python version for venv logic
  echo "[DEBUG] Astropy: detected Python version for venv logic: ${ASTROPY_PY_MAJOR}.${ASTROPY_PY_MINOR} (PY=$PY)" >&2

  # CRITICAL FIX: Ensure base pip is functional before installing toolchain pip.
  # Some Python 3.6 environments have broken/incomplete pip installations.
  # Test pip functionality by trying to import it and run a simple command.
  if ! "$PY" -m pip --version >/dev/null 2>&1 || ! "$PY" -c "import pip._internal.cli.main" >/dev/null 2>&1; then
    echo "[INFO] Astropy: base pip is broken or incomplete, attempting to repair..." >&2
    # Try to repair pip using get-pip.py (most reliable for Python 3.6)
    GET_PIP="/tmp/get-pip_$$.py"
    if command -v curl >/dev/null 2>&1; then
      curl -sSL https://bootstrap.pypa.io/get-pip.py >"$GET_PIP" 2>/dev/null || true
    elif command -v wget >/dev/null 2>&1; then
      wget -q -O "$GET_PIP" https://bootstrap.pypa.io/get-pip.py 2>/dev/null || true
    fi
    if [ -f "$GET_PIP" ] && [ -s "$GET_PIP" ]; then
      # For Python 3.6, get-pip.py needs specific pip version
      if [ "$ASTROPY_PY_MAJOR" -eq 3 ] && [ "$ASTROPY_PY_MINOR" -lt 7 ]; then
        if "$PY" "$GET_PIP" "pip==21.3.1" >/tmp/apr_astropy_repair_pip_$$.log 2>&1; then
          echo "[INFO] Astropy: pip repaired successfully using get-pip.py" >&2
        else
          echo "[WARN] Astropy: get-pip.py failed, will try conda if available" >&2
          # Fallback: try conda to install pip
          if command -v conda >/dev/null 2>&1; then
            conda install -y pip >/tmp/apr_astropy_conda_pip_$$.log 2>&1 || true
          fi
        fi
      else
        if "$PY" "$GET_PIP" >/tmp/apr_astropy_repair_pip_$$.log 2>&1; then
          echo "[INFO] Astropy: pip repaired successfully using get-pip.py" >&2
        fi
      fi
      rm -f "$GET_PIP" 2>/dev/null || true
    else
      # Fallback: try conda to install pip
      if command -v conda >/dev/null 2>&1; then
        echo "[INFO] Astropy: attempting to install pip via conda..." >&2
        conda install -y pip >/tmp/apr_astropy_conda_pip_$$.log 2>&1 || true
      fi
    fi
    # Verify pip is now functional
    if ! "$PY" -m pip --version >/dev/null 2>&1 || ! "$PY" -c "import pip._internal.cli.main" >/dev/null 2>&1; then
      echo "[ERROR] Astropy: failed to repair base pip. Cannot proceed with toolchain bootstrap." >&2
      exit 2
    fi
    echo "[INFO] Astropy: base pip is now functional" >&2
  fi

  # Scheme2: create a shared, dedicated venv for Astropy runs (reused across all instances).
  # Goal: eliminate PYTHONPATH shadowing issues (setuptools=68) and broken toolchain pip (pip._vendor.rich),
  # and ensure build-system requirements (extension-helpers, tomli) are available deterministically.
  # Only enable for Python>=3.7 (pip modern, venv reliable). Keep legacy path for py3.6.
  # CRITICAL: venv is shared across instances (same Python version) to avoid repeated creation overhead.
  APR_ASTROPY_USE_VENV="${APR_ASTROPY_USE_VENV:-1}"
  _APR_ASTROPY_ORIG_PY="$PY"  # Save original PY for fallback
  # Ensure ASTROPY_PY_MAJOR/MINOR are up-to-date (in case Python was switched)
  ASTROPY_PY_VER_CURRENT=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
  ASTROPY_PY_MAJOR_CURRENT=${ASTROPY_PY_VER_CURRENT%%.*}
  ASTROPY_PY_MINOR_CURRENT=${ASTROPY_PY_VER_CURRENT#*.}
  if [ -n "${ASTROPY_PY_MAJOR_CURRENT:-}" ] && [ -n "${ASTROPY_PY_MINOR_CURRENT:-}" ]; then
    ASTROPY_PY_MAJOR="$ASTROPY_PY_MAJOR_CURRENT"
    ASTROPY_PY_MINOR="$ASTROPY_PY_MINOR_CURRENT"
    echo "[DEBUG] Astropy: updated Python version for venv check: ${ASTROPY_PY_MAJOR}.${ASTROPY_PY_MINOR}" >&2
  fi
  # Try to create venv (may not succeed if Python < 3.7, but that's OK)
  PY_NEW=$(_apr_astropy_create_venv "$PY")
  if [ -n "$PY_NEW" ] && [ "$PY_NEW" != "$PY" ]; then
    PY="$PY_NEW"
  fi

  if [ "$NEEDS_DEP_UTIL" = "1" ]; then
    echo "[INFO] Astropy: bootstrapping toolchain (setuptools==60.5.0 for dep_util) into $ASTROPY_TOOLCHAIN_DIR..." >&2
    # NOTE: Some Astropy images use Python 3.6 in testbed. Modern pip/setuptools require Python>=3.7.
    if [ "$ASTROPY_PY_MAJOR" -eq 3 ] && [ "$ASTROPY_PY_MINOR" -lt 7 ]; then
      # Last pip supporting Python 3.6 is in the 21.x series. Setuptools 59.6.0 is a safe ceiling.
      # CRITICAL: pip 21.3.1 requires charset_normalizer or chardet for requests compatibility
      # Also install certifi and requests to fix certificate and module issues
      if ! "$PY" -m pip install -U --no-cache-dir -t "$ASTROPY_TOOLCHAIN_DIR" "pip==21.3.1" "wheel" "setuptools==59.6.0" "setuptools_scm==6.4.2" "charset-normalizer<3" "urllib3<2" "certifi" "requests<3" >"$TOOLCHAIN_LOG" 2>&1; then
        echo "[ERROR] Astropy: failed to bootstrap toolchain (py<3.7) into $ASTROPY_TOOLCHAIN_DIR. Tail:" >&2
        tail -200 "$TOOLCHAIN_LOG" >&2 || true
        exit 2
      fi
      # Verify pip is functional after installation
      if ! "$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_TOOLCHAIN_DIR'); import pip._internal.cli.main" >/dev/null 2>&1; then
        echo "[WARN] Astropy: pip toolchain installed but not fully functional, attempting repair..." >&2
        # Try reinstalling pip with --force-reinstall and --ignore-installed
        "$PY" -m pip install --force-reinstall --no-cache-dir --ignore-installed -t "$ASTROPY_TOOLCHAIN_DIR" "pip==21.3.1" "certifi" "requests<3" >/tmp/apr_astropy_pip_repair_$$.log 2>&1 || true
      fi
    else
      # IMPORTANT: do NOT install pip into $ASTROPY_TOOLCHAIN_DIR (PYTHONPATH would shadow venv/base pip).
      if ! "$PY" -m pip install -U --no-cache-dir --ignore-installed -t "$ASTROPY_TOOLCHAIN_DIR" "wheel" "setuptools==60.5.0" "setuptools_scm==7.1.0" "certifi" >"$TOOLCHAIN_LOG" 2>&1; then
        echo "[ERROR] Astropy: failed to bootstrap toolchain into $ASTROPY_TOOLCHAIN_DIR. Tail:" >&2
        tail -200 "$TOOLCHAIN_LOG" >&2 || true
        exit 2
      fi
    fi
    # CRITICAL: Check dep_util from toolchain directory specifically (not from PYTHONPATH)
    # Use sys.path.insert to ensure we check the toolchain's setuptools, not any pre-imported one
    if ! "$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_TOOLCHAIN_DIR'); from setuptools.dep_util import newer_group" >/dev/null 2>&1; then
      echo "[ERROR] Astropy: toolchain sanity check failed (missing setuptools.dep_util). PYTHONPATH=$PYTHONPATH" >&2
      "$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_TOOLCHAIN_DIR'); import setuptools; print('toolchain setuptools_version=', getattr(setuptools,'__version__','<missing>'))" >&2 || true
      exit 2
    fi
  else
    echo "[INFO] Astropy: bootstrapping modern toolchain into $ASTROPY_TOOLCHAIN_DIR..." >&2
    if [ "$ASTROPY_PY_MAJOR" -eq 3 ] && [ "$ASTROPY_PY_MINOR" -lt 7 ]; then
      # Python 3.6: must use legacy toolchain versions.
      # Base pip should already be repaired above, but verify again
      if ! "$PY" -m pip --version >/dev/null 2>&1; then
        echo "[ERROR] Astropy: base pip not functional, cannot install toolchain" >&2
        exit 2
      fi
      # CRITICAL: pip 21.3.1 requires charset_normalizer or chardet for requests compatibility
      # Also install certifi and requests to fix certificate and module issues
      # CRITICAL FIX: Add --ignore-installed to avoid file deletion conflicts
      if ! "$PY" -m pip install -U --no-cache-dir --ignore-installed -t "$ASTROPY_TOOLCHAIN_DIR" "pip==21.3.1" "wheel" "setuptools==59.6.0" "setuptools_scm==6.4.2" "charset-normalizer<3" "urllib3<2" "certifi" "requests<3" >"$TOOLCHAIN_LOG" 2>&1; then
        echo "[ERROR] Astropy: failed to bootstrap toolchain (py<3.7) into $ASTROPY_TOOLCHAIN_DIR. Tail:" >&2
        tail -200 "$TOOLCHAIN_LOG" >&2 || true
        exit 2
      fi
      # Verify pip is functional after installation
      if ! "$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_TOOLCHAIN_DIR'); import pip._internal.cli.main" >/dev/null 2>&1; then
        echo "[WARN] Astropy: pip toolchain installed but not fully functional, attempting repair..." >&2
        # Try reinstalling pip with --force-reinstall
        "$PY" -m pip install --force-reinstall --no-cache-dir --ignore-installed -t "$ASTROPY_TOOLCHAIN_DIR" "pip==21.3.1" "certifi" "requests<3" >/tmp/apr_astropy_pip_repair_$$.log 2>&1 || true
      fi
    else
      # Keep versions reasonably modern but not overly aggressive (avoid requiring very new Python).
      # IMPORTANT: do NOT install pip into $ASTROPY_TOOLCHAIN_DIR (PYTHONPATH would shadow venv/base pip).
      if ! "$PY" -m pip install -U --no-cache-dir --ignore-installed -t "$ASTROPY_TOOLCHAIN_DIR" "wheel" "setuptools==60.5.0" "setuptools_scm==7.1.0" >"$TOOLCHAIN_LOG" 2>&1; then
        echo "[ERROR] Astropy: failed to bootstrap toolchain into $ASTROPY_TOOLCHAIN_DIR. Tail:" >&2
        tail -200 "$TOOLCHAIN_LOG" >&2 || true
        exit 2
      fi
    fi
    # Verify we are NOT using the distro/testbed setuptools anymore (skip strict check for Python 3.6).
    # IMPORTANT: do NOT rely on PYTHONPATH here; always force sys.path to prefer our toolchain dir.
    if [ "$ASTROPY_PY_MAJOR" -eq 3 ] && [ "$ASTROPY_PY_MINOR" -ge 7 ]; then
      if ! "$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_TOOLCHAIN_DIR'); import setuptools; v=getattr(setuptools,'__version__',''); sys.exit(0 if v and (v.startswith('60.') or v.startswith('59.')) else 2)" >/dev/null 2>&1; then
        echo "[ERROR] Astropy: setuptools not in expected range after bootstrap (toolchain setuptools may be shadowed). PYTHONPATH=$PYTHONPATH" >&2
        "$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_TOOLCHAIN_DIR'); import setuptools; print('setuptools_version=', getattr(setuptools,'__version__','<missing>'))" >&2 || true
        exit 2
      fi
    fi
  fi

  # Now that toolchain is bootstrapped, put it back at the front.
  export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:$SITE_DIR:${_APR_ASTROPY_ORIG_PYTHONPATH}"

  # Build+install Astropy into $SITE_DIR. Prefer no-deps and no-build-isolation to avoid downloads and
  # conflicts; rely on spec deps already installed into $SITE_DIR.
  #
  # However, we MUST still satisfy PEP517 build-system requirements from pyproject.toml when present
  # (e.g. extension-helpers), since --no-build-isolation disables pip's isolated env that would
  # otherwise install them automatically.
  if [ -f "/testbed/pyproject.toml" ]; then
    BUILD_REQS_FILE="/tmp/apr_astropy_build_requires_$$.txt"
    # Extract build-system.requires safely (tomllib is available in py>=3.11; use tomli fallback).
    "$PY" - <<'PY_APR_BUILDREQS' >"$BUILD_REQS_FILE" 2>/dev/null || true
import sys
try:
    import tomllib as _toml  # py>=3.11
except Exception:
    try:
        import tomli as _toml  # type: ignore
    except Exception:
        _toml = None

if _toml is None:
    sys.exit(0)

with open("/testbed/pyproject.toml", "rb") as f:
    data = _toml.load(f)
reqs = (((data or {}).get("build-system") or {}).get("requires") or [])
for r in reqs:
    if isinstance(r, str) and r.strip():
        print(r.strip())
PY_APR_BUILDREQS

    if [ -s "$BUILD_REQS_FILE" ]; then
      BUILD_REQS_LOG="/tmp/apr_astropy_build_requires_install_$$.log"
      echo "[INFO] Astropy: installing pyproject build-system.requires into $ASTROPY_BUILDREQ_DIR..." >&2
      # CRITICAL: Replace setuptools with setuptools==60.5.0 in build requirements to preserve dep_util
      # setuptools 60.5.0 is the last version that contains dep_util (removed in 60.9.0+)
      BUILD_REQS_FILE_FIXED="/tmp/apr_astropy_buildreqs_fixed_$$.txt"
      sed 's/^setuptools.*$/setuptools==60.5.0/' "$BUILD_REQS_FILE" | grep -v "^setuptools>=" | grep -v "^setuptools<" | grep -v "^setuptools!=" > "$BUILD_REQS_FILE_FIXED" 2>/dev/null || cp "$BUILD_REQS_FILE" "$BUILD_REQS_FILE_FIXED"
      # Ensure setuptools==60.5.0 is in the list
      if ! grep -q "setuptools==60.5.0" "$BUILD_REQS_FILE_FIXED" 2>/dev/null; then
        # Remove any existing setuptools line and add our version
        grep -v "^setuptools" "$BUILD_REQS_FILE_FIXED" > "${BUILD_REQS_FILE_FIXED}.tmp" 2>/dev/null || cp "$BUILD_REQS_FILE_FIXED" "${BUILD_REQS_FILE_FIXED}.tmp"
        echo "setuptools==60.5.0" >> "${BUILD_REQS_FILE_FIXED}.tmp"
        mv "${BUILD_REQS_FILE_FIXED}.tmp" "$BUILD_REQS_FILE_FIXED"
      fi
      _APR_ASTROPY_PYTHONPATH_SAVED="$PYTHONPATH"
      export PYTHONPATH="$_APR_ASTROPY_BOOTSTRAP_PYTHONPATH"
      # CRITICAL FIX: Use --ignore-installed to avoid file deletion conflicts
      if ! "$PY" -m pip install -U --no-cache-dir --ignore-installed -t "$ASTROPY_BUILDREQ_DIR" -r "$BUILD_REQS_FILE_FIXED" >"$BUILD_REQS_LOG" 2>&1; then
        echo "[ERROR] Astropy: failed to install pyproject build requirements. Tail:" >&2
        tail -200 "$BUILD_REQS_LOG" >&2 || true
        export PYTHONPATH="$_APR_ASTROPY_PYTHONPATH_SAVED"
        exit 2
      fi
      # Also explicitly ensure setuptools==60.5.0 is installed (in case it wasn't in requirements)
      # CRITICAL: Use setuptools==60.5.0 (last version with dep_util, removed in 60.9.0+)
      # Force reinstall with --ignore-installed to ensure setuptools==60.5.0 is used and avoid file conflicts
      "$PY" -m pip install --no-cache-dir --force-reinstall --ignore-installed -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_ensure_$$.log 2>&1 || true
      # CRITICAL: ensure extension-helpers (and tomli for parsing) exist for in-place builds.
      "$PY" -m pip install -U --no-cache-dir -t "$ASTROPY_BUILDREQ_DIR" "extension-helpers" "tomli" >/tmp/apr_astropy_buildreq_extras_$$.log 2>&1 || true
      export PYTHONPATH="$_APR_ASTROPY_PYTHONPATH_SAVED"
      # Verify setuptools version and dep_util availability
      SETUPTOOLS_VER=$("$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_BUILDREQ_DIR'); import setuptools; print(setuptools.__version__)" 2>/dev/null || echo "")
      if [ -n "$SETUPTOOLS_VER" ]; then
        SETUPTOOLS_MAJOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f1)
        SETUPTOOLS_MINOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f2)
        # Check if version is >= 60.9 (dep_util removed) or >= 61
        NEED_DOWNGRADE=0
        if [ "$SETUPTOOLS_MAJOR" -ge 61 ] 2>/dev/null; then
          NEED_DOWNGRADE=1
        elif [ "$SETUPTOOLS_MAJOR" -eq 60 ] 2>/dev/null && [ "$SETUPTOOLS_MINOR" -ge 9 ] 2>/dev/null; then
          NEED_DOWNGRADE=1
        fi
        if [ "$NEED_DOWNGRADE" -eq 1 ]; then
          echo "[WARN] Astropy: setuptools version $SETUPTOOLS_VER >= 60.9, forcing downgrade to 60.5.0 (last version with dep_util)..." >&2
          "$PY" -m pip install --no-cache-dir --force-reinstall --no-deps -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_downgrade_$$.log 2>&1 || true
        fi
        # Verify dep_util is available
        # CRITICAL: Use a fresh Python process and ensure buildreq is first in sys.path before any import
        if ! "$PY" -c "import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; from setuptools.dep_util import newer_group" >/dev/null 2>&1; then
          echo "[WARN] Astropy: setuptools.dep_util still not available after install, forcing reinstall of 60.5.0..." >&2
          "$PY" -m pip install --no-cache-dir --force-reinstall --ignore-installed -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_force_$$.log 2>&1 || true
          # Verify again after reinstall
          if ! "$PY" -c "import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; from setuptools.dep_util import newer_group" >/dev/null 2>&1; then
            echo "[ERROR] Astropy: setuptools.dep_util still not available after reinstall. Checking installation..." >&2
            "$PY" -c "import os; dep_util_path = os.path.join('$ASTROPY_BUILDREQ_DIR', 'setuptools', 'dep_util.py'); print('dep_util.py exists:', os.path.exists(dep_util_path)); import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; import setuptools; print('setuptools version:', setuptools.__version__); print('setuptools file:', setuptools.__file__)" >&2 || true
          fi
        fi
      fi
    fi
  else
    # Minimal fallback for older pyproject-less trees or toml parser missing:
    # extension-helpers is required by many Astropy builds.
    "$PY" -m pip install -U --no-cache-dir -t "$ASTROPY_BUILDREQ_DIR" extension-helpers >/dev/null 2>&1 || true
    # Also install setuptools==60.5.0 for dep_util (last version with dep_util, removed in 60.9.0+)
    "$PY" -m pip install --no-cache-dir -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_fallback_$$.log 2>&1 || true
  fi
  
  # CRITICAL: Ensure PYTHONPATH has buildreq BEFORE any other setuptools locations
  # This ensures setuptools==60.5.0 from buildreq is used instead of newer versions
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$ASTROPY_BUILDREQ_DIR$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$ASTROPY_BUILDREQ_DIR:${PYTHONPATH}"
  else
    export PYTHONPATH="$ASTROPY_BUILDREQ_DIR"
  fi

  # CRITICAL FIX: Try to install astropy first, and if it fails due to Python version mismatch,
  # extract the required version from the error message and bootstrap that Python version.
  # This is more reliable than parsing pyproject.toml/setup.py upfront.
  # OPTIMIZATION: For test suite verification (APR_VERIFY_TEST_SUITE=1), skip the slow pip install -e step
  # and go directly to wheel build or build_ext, as we don't need to detect Python version requirements
  # (they should already be known from the instance metadata).
  WHEEL_BUILD_LOG="/tmp/apr_astropy_wheel_build_$$.log"
  SKIP_PIP_INSTALL_E=0
  if [ "${APR_VERIFY_TEST_SUITE:-0}" = "1" ]; then
    echo "[INFO] Astropy: APR_VERIFY_TEST_SUITE=1, skipping slow pip install -e step..." >&2
    SKIP_PIP_INSTALL_E=1
  fi
  if [ "$SKIP_PIP_INSTALL_E" -eq 0 ]; then
    echo "[INFO] Astropy: attempting wheel build to detect Python version requirements..." >&2
    if ! timeout 120 "$PY" -m pip install --no-cache-dir --no-build-isolation -e /testbed >"$WHEEL_BUILD_LOG" 2>&1; then
    # Check if the error is due to Python version mismatch
    PY_VERSION_ERROR=$(grep -i "requires a different Python\|not in.*>=" "$WHEEL_BUILD_LOG" 2>/dev/null | head -1 || echo "")
    if [ -n "$PY_VERSION_ERROR" ]; then
      echo "[INFO] Astropy: detected Python version requirement error: $PY_VERSION_ERROR" >&2
      # Extract required version from error message
      # Patterns: "3.6.13 not in '>=3.8'" -> "3.8"
      #           "3.6.13 not in '>=3.9'" -> "3.9"
      #           "requires Python >=3.8" -> "3.8"
      REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*not in.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
      if [ -z "$REQUIRED_PY_VER" ]; then
        # Try alternative pattern: "requires Python >=3.8" or "requires Python >=3.9"
        REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*Python.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
      fi
      if [ -z "$REQUIRED_PY_VER" ]; then
        # Try pattern: ">=3.8" or ">=3.9" directly
        REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
      fi
      
      if [ -n "$REQUIRED_PY_VER" ] && [ -f "/miniconda.sh" ]; then
        echo "[INFO] Astropy: Python version mismatch detected, bootstrapping Python ${REQUIRED_PY_VER}..." >&2
        
        CONDA_BASE=""
        if [ -x "/opt/miniconda3/bin/conda" ]; then
          CONDA_BASE="/opt/miniconda3"
        elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
          CONDA_BASE="/tmp/apr_miniconda3"
        else
          # Install miniconda if not available
          if [ -f "/miniconda.sh" ]; then
            if bash /miniconda.sh -b -p /opt/miniconda3 >/tmp/apr_astropy_conda_install_$$.log 2>&1; then
              CONDA_BASE="/opt/miniconda3"
            else
              CONDA_BASE="/tmp/apr_miniconda3"
              bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_astropy_conda_install_$$.log 2>&1 || true
            fi
          fi
        fi
        
        if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
          PY_REQ_ENV="${CONDA_BASE}/envs/apr_astropy_py${REQUIRED_PY_VER//./}"
          PY_REQ="${PY_REQ_ENV}/bin/python"
          
          if [ ! -x "$PY_REQ" ]; then
            echo "[INFO] Astropy: creating Python ${REQUIRED_PY_VER} environment..." >&2
            source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
            "${CONDA_BASE}/bin/conda" create -y -p "$PY_REQ_ENV" "python=${REQUIRED_PY_VER}" pip >/tmp/apr_astropy_create_py${REQUIRED_PY_VER//./}.log 2>&1 || true
          fi
          
          if [ -x "$PY_REQ" ]; then
            PY="$PY_REQ"
            echo "[INFO] Astropy: switched to Python ${REQUIRED_PY_VER}: PY=$PY" >&2
            "$PY" -V >&2 || true
            # CRITICAL: After Python switching, immediately update ASTROPY_PY_MAJOR/MINOR for venv creation logic
            ASTROPY_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            ASTROPY_PY_MAJOR=${ASTROPY_PY_VER%%.*}
            ASTROPY_PY_MINOR=${ASTROPY_PY_VER#*.}
            if [ -z "${ASTROPY_PY_MAJOR:-}" ] || [ -z "${ASTROPY_PY_MINOR:-}" ]; then
              ASTROPY_PY_MAJOR=0
              ASTROPY_PY_MINOR=0
            fi
            echo "[DEBUG] Astropy: after Python switch (wheel build), updated version for venv: ${ASTROPY_PY_MAJOR}.${ASTROPY_PY_MINOR}" >&2
            # Ensure pip is available
            if ! "$PY" -m pip --version >/dev/null 2>&1; then
              "${CONDA_BASE}/bin/conda" install -y -p "$PY_REQ_ENV" pip >/tmp/apr_astropy_install_pip_py${REQUIRED_PY_VER//./}.log 2>&1 || true
            fi
            # Reinstall pytest for the new Python version
            if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
              PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
              PYTEST_VER="7.4.4"
              if [ "$PY_VER" = "3.6" ]; then
                PYTEST_VER="7.0.1"
              fi
              "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pytest==${PYTEST_VER}" >/tmp/apr_astropy_pytest_reinstall_$$.log 2>&1 || true
            fi
            # Re-export PYTHONPATH with SITE_DIR
            if [ -n "${PYTHONPATH:-}" ]; then
              PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
              export PYTHONPATH="$SITE_DIR:${PYTHONPATH}"
            else
              export PYTHONPATH="$SITE_DIR"
            fi
            echo "[INFO] Astropy: retrying installation with Python ${REQUIRED_PY_VER}..." >&2
          else
            echo "[WARN] Astropy: failed to bootstrap Python ${REQUIRED_PY_VER}, continuing with current Python" >&2
          fi
        fi
      fi
    fi
  fi

  # IMPORTANT:
  # Astropy tests run from the source checkout (/testbed). Python will import "astropy" from the
  # current working directory first (sys.path[0] == ''), so installing into $SITE_DIR alone is not
  # sufficient. We must build extension modules in-place so imports from /testbed succeed.
  BUILD_LOG="/tmp/apr_astropy_build_ext_$$.log"

  # Prefer building a wheel and installing into $SITE_DIR (writable) to avoid:
  # - read-only /opt/miniconda3/envs/testbed/site-packages
  # - setuptools "develop" trying to write into the conda env
  #
  # Then run pytest from /tmp (not /testbed) so imports resolve from $SITE_DIR.
  WHEEL_DIR="/tmp/apr_astropy_wheels_$$"
  rm -rf "$WHEEL_DIR" 2>/dev/null || true
  mkdir -p "$WHEEL_DIR" 2>/dev/null || true

  # Reduce flakiness for old Astropy + old Python images.
  # - Limit build parallelism (OOM/cc1plus crashes can look like random segfaults).
  # - Prefer using pre-generated C sources when possible (avoid Cython at build time).
  export NPY_NUM_BUILD_JOBS=1
  export ASTROPY_USE_SYSTEM_CYTHON=0
  export ASTROPY_USE_CYTHON=0
  export SETUPTOOLS_USE_DISTUTILS=stdlib

  echo "[INFO] Astropy: building wheel (then install into SITE_DIR=$SITE_DIR)..." >&2
  WHEEL_LOG="/tmp/apr_astropy_wheel_$$.log"
  # CRITICAL: Ensure PYTHONPATH has buildreq BEFORE venv site-packages for wheel build
  # This ensures setuptools==60.5.0 from buildreq is used instead of venv's setuptools
  # Re-export PYTHONPATH to ensure buildreq is first (may have been modified)
  if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$ASTROPY_TOOLCHAIN_DIR$" | grep -v "^$ASTROPY_BUILDREQ_DIR$" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:${PYTHONPATH}"
  else
    export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR"
  fi
  # Verify setuptools.dep_util is available before wheel build
  # CRITICAL: Use a fresh Python process and ensure buildreq is first in sys.path before any import
  # This prevents importing setuptools from PYTHONPATH before we can check buildreq's version
  if ! "$PY" -c "import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; from setuptools.dep_util import newer_group" >/dev/null 2>&1; then
    echo "[ERROR] Astropy: setuptools.dep_util not available in buildreq before wheel build. PYTHONPATH=$PYTHONPATH" >&2
    "$PY" -c "import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; import setuptools; print('buildreq setuptools version:', setuptools.__version__); print('buildreq setuptools file:', setuptools.__file__)" >&2 || true
    # Also check if dep_util file exists
    "$PY" -c "import os; dep_util_path = os.path.join('$ASTROPY_BUILDREQ_DIR', 'setuptools', 'dep_util.py'); print('dep_util.py exists:', os.path.exists(dep_util_path))" >&2 || true
    exit 2
  fi
  # OPTIMIZATION: For test suite verification, skip wheel build and go directly to build_ext
  # Wheel build can take a long time (>300s) and is not necessary for verification
  if [ "${APR_VERIFY_TEST_SUITE:-0}" = "1" ]; then
    echo "[INFO] Astropy: APR_VERIFY_TEST_SUITE=1, skipping wheel build, using build_ext --inplace directly..." >&2
    WHEEL_BUILD_SUCCESS=0
  else
    # Use timeout for wheel build to avoid hanging (max 180 seconds)
    if timeout 180 "$PY" -m pip wheel --no-deps --no-build-isolation -w "$WHEEL_DIR" . >"$WHEEL_LOG" 2>&1; then
      WHEEL_BUILD_SUCCESS=1
    else
      WHEEL_BUILD_SUCCESS=0
    fi
  fi
  if [ "$WHEEL_BUILD_SUCCESS" -eq 1 ]; then
    WHEEL_FILE="$(ls -1 "$WHEEL_DIR"/astropy-*.whl 2>/dev/null | head -1 || true)"
    if [ -n "$WHEEL_FILE" ] && [ -f "$WHEEL_FILE" ]; then
      echo "[INFO] Astropy: installing wheel into $SITE_DIR: $WHEEL_FILE" >&2
      INSTALL_LOG="/tmp/apr_astropy_wheel_install_$$.log"
      # CRITICAL: Install wheel with dependencies to ensure runtime deps (e.g., erfa/pyerfa) are installed
      # Use --no-deps=false (default) to install dependencies, but ensure PYTHONPATH is correct
      _APR_ASTROPY_PYTHONPATH_SAVED="$PYTHONPATH"
      export PYTHONPATH="$ASTROPY_BUILDREQ_DIR:$SITE_DIR:${_APR_ASTROPY_CLEANED_PYTHONPATH}"
      if ! "$PY" -m pip install --no-cache-dir --no-deps --no-build-isolation -t "$SITE_DIR" "$WHEEL_FILE" >"$INSTALL_LOG" 2>&1; then
        echo "[ERROR] Astropy: wheel install into SITE_DIR failed. Tail:" >&2
        tail -200 "$INSTALL_LOG" >&2 || true
        exit 2
      fi
      # CRITICAL: Install runtime dependencies (erfa/pyerfa) that are required by astropy
      # The wheel was installed with --no-deps, so we need to install dependencies separately
      echo "[INFO] Astropy: installing runtime dependencies (erfa/pyerfa) into $SITE_DIR..." >&2
      if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pyerfa" >/tmp/apr_astropy_runtime_deps_$$.log 2>&1; then
        # Fallback: try erfa (older name) or pyerfa with different versions
        echo "[WARN] Astropy: pyerfa install failed, trying erfa... Tail:" >&2
        tail -50 /tmp/apr_astropy_runtime_deps_$$.log >&2 || true
        "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "erfa" >/tmp/apr_astropy_runtime_deps_erfa_$$.log 2>&1 || true
      fi
      # CRITICAL: Ensure PYTHONPATH prioritizes SITE_DIR over /testbed to import from installed wheel
      # Remove /testbed from PYTHONPATH to prevent importing from source checkout
      if [ -n "${PYTHONPATH:-}" ]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
        export PYTHONPATH="$SITE_DIR:$ASTROPY_BUILDREQ_DIR:${PYTHONPATH}"
      else
        export PYTHONPATH="$SITE_DIR:$ASTROPY_BUILDREQ_DIR"
      fi
      export APR_ASTROPY_RUN_FROM_TMP=1
      echo "[INFO] Astropy: wheel installed; PYTHONPATH configured to import from SITE_DIR (not /testbed)" >&2
    else
      echo "[WARN] Astropy: wheel build succeeded but no wheel file found in $WHEEL_DIR. Tail:" >&2
      tail -200 "$WHEEL_LOG" >&2 || true
    fi
  else
    if [ "${APR_VERIFY_TEST_SUITE:-0}" = "1" ]; then
      echo "[INFO] Astropy: APR_VERIFY_TEST_SUITE=1, wheel build skipped, will use build_ext --inplace..." >&2
    else
      echo "[WARN] Astropy: wheel build failed, checking for Python version mismatch..." >&2
    # CRITICAL FIX: Check if failure is due to Python version mismatch
    PY_VERSION_ERROR=$(grep -i "requires a different Python\|not in.*>=\|requires Python.*>=" "$WHEEL_LOG" 2>/dev/null | head -1 || echo "")
    if [ -n "$PY_VERSION_ERROR" ]; then
      echo "[INFO] Astropy: detected Python version requirement error: $PY_VERSION_ERROR" >&2
      # Extract required version from error message
      # Patterns: "3.6.13 not in '>=3.8'" -> "3.8"
      #           "3.6.13 not in '>=3.9'" -> "3.9"
      #           "requires Python >=3.8" -> "3.8"
      REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*not in.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
      if [ -z "$REQUIRED_PY_VER" ]; then
        # Try alternative pattern: "requires Python >=3.8" or "requires Python >=3.9"
        REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*Python.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
      fi
      if [ -z "$REQUIRED_PY_VER" ]; then
        # Try pattern: ">=3.8" or ">=3.9" directly
        REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
      fi
      
      if [ -n "$REQUIRED_PY_VER" ] && [ -f "/miniconda.sh" ]; then
        echo "[INFO] Astropy: switching to Python ${REQUIRED_PY_VER}..." >&2
        
        CONDA_BASE=""
        if [ -x "/opt/miniconda3/bin/conda" ]; then
          CONDA_BASE="/opt/miniconda3"
        elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
          CONDA_BASE="/tmp/apr_miniconda3"
        else
          # Install miniconda if not available
          if [ -f "/miniconda.sh" ]; then
            if bash /miniconda.sh -b -p /opt/miniconda3 >/tmp/apr_astropy_conda_install_$$.log 2>&1; then
              CONDA_BASE="/opt/miniconda3"
            else
              CONDA_BASE="/tmp/apr_miniconda3"
              bash /miniconda.sh -b -p "$CONDA_BASE" >/tmp/apr_astropy_conda_install_$$.log 2>&1 || true
            fi
          fi
        fi
        
        if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
          PY_REQ_ENV="${CONDA_BASE}/envs/apr_astropy_py${REQUIRED_PY_VER//./}"
          PY_REQ="${PY_REQ_ENV}/bin/python"
          
          if [ ! -x "$PY_REQ" ]; then
            echo "[INFO] Astropy: creating Python ${REQUIRED_PY_VER} environment..." >&2
            source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
            "${CONDA_BASE}/bin/conda" create -y -p "$PY_REQ_ENV" "python=${REQUIRED_PY_VER}" pip >/tmp/apr_astropy_create_py${REQUIRED_PY_VER//./}.log 2>&1 || true
          fi
          
          if [ -x "$PY_REQ" ]; then
            PY="$PY_REQ"
            echo "[INFO] Astropy: switched to Python ${REQUIRED_PY_VER}: PY=$PY" >&2
            "$PY" -V >&2 || true
            # CRITICAL: After Python switching, immediately update ASTROPY_PY_MAJOR/MINOR for venv creation logic
            ASTROPY_PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            ASTROPY_PY_MAJOR=${ASTROPY_PY_VER%%.*}
            ASTROPY_PY_MINOR=${ASTROPY_PY_VER#*.}
            if [ -z "${ASTROPY_PY_MAJOR:-}" ] || [ -z "${ASTROPY_PY_MINOR:-}" ]; then
              ASTROPY_PY_MAJOR=0
              ASTROPY_PY_MINOR=0
            fi
            echo "[DEBUG] Astropy: after Python switch (wheel build), updated version for venv: ${ASTROPY_PY_MAJOR}.${ASTROPY_PY_MINOR}" >&2
            # Ensure pip is available
            if ! "$PY" -m pip --version >/dev/null 2>&1; then
              "${CONDA_BASE}/bin/conda" install -y -p "$PY_REQ_ENV" pip >/tmp/apr_astropy_install_pip_py${REQUIRED_PY_VER//./}.log 2>&1 || true
            fi
            # CRITICAL: After Python switching (wheel build failure), create venv if Python >= 3.7
            echo "[INFO] Astropy: attempting to create venv after Python switch (wheel build)..." >&2
            PY_NEW=$(_apr_astropy_create_venv "$PY")
            if [ -n "$PY_NEW" ] && [ "$PY_NEW" != "$PY" ]; then
              PY="$PY_NEW"
              echo "[INFO] Astropy: switched to venv Python after wheel build failure: PY=$PY" >&2
            fi
            # CRITICAL: Reinstall toolchain and build requirements for the new Python version
            # The old toolchain and build requirements were compiled for Python 3.6
            # and are incompatible with Python 3.8
            echo "[INFO] Astropy: reinstalling toolchain and build requirements for Python ${REQUIRED_PY_VER}..." >&2
            rm -rf "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
            mkdir -p "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
            # Reinstall toolchain (pip, setuptools) for new Python version
              # CRITICAL: Use setuptools==60.5.0 to preserve setuptools.dep_util (required by old astropy code, last version with dep_util)
            TOOLCHAIN_LOG="/tmp/apr_astropy_toolchain_reinstall_$$.log"
            PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
            if [ "$PY_VER" = "3.6" ]; then
              "$PY" -m pip install --no-cache-dir -t "$ASTROPY_TOOLCHAIN_DIR" "pip==21.3.1" "setuptools==59.6.0" "wheel" >"$TOOLCHAIN_LOG" 2>&1 || true
            else
              # For Python 3.8+, use setuptools==60.5.0 to keep dep_util (removed in setuptools>=60.9)
              "$PY" -m pip install --no-cache-dir -t "$ASTROPY_TOOLCHAIN_DIR" "setuptools==60.5.0" "wheel" >"$TOOLCHAIN_LOG" 2>&1 || true
            fi
            # Update PYTHONPATH to include new toolchain
            if [ -n "${PYTHONPATH:-}" ]; then
              PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$ASTROPY_TOOLCHAIN_DIR$" | tr '\n' ':' | sed 's/:$//')
              export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:${PYTHONPATH}"
            else
              export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR"
            fi
            # Reinstall build requirements
            if [ -f "/testbed/pyproject.toml" ]; then
              BUILD_REQS_FILE="/tmp/apr_astropy_buildreqs_$$.txt"
              "$PY" - <<'PY_APR_BUILDREQS' >"$BUILD_REQS_FILE" 2>/dev/null || true
try:
    import tomllib as _toml
except:
    try:
        import tomli as _toml
    except:
        _toml = None

if _toml:
    with open("/testbed/pyproject.toml", "rb") as f:
        data = _toml.load(f)
    reqs = (((data or {}).get("build-system") or {}).get("requires") or [])
    for r in reqs:
        if isinstance(r, str) and r.strip():
            # Replace setuptools with setuptools==60.5.0 to preserve dep_util
            r_clean = r.strip()
            if r_clean.startswith("setuptools") and "60.5.0" not in r_clean and "==" not in r_clean:
                print("setuptools==60.5.0")
            else:
                print(r_clean)
PY_APR_BUILDREQS
              if [ -s "$BUILD_REQS_FILE" ]; then
                BUILD_REQS_LOG="/tmp/apr_astropy_build_requires_reinstall_$$.log"
                "$PY" -m pip install -U --no-cache-dir -t "$ASTROPY_BUILDREQ_DIR" -r "$BUILD_REQS_FILE" >"$BUILD_REQS_LOG" 2>&1 || true
              fi
            fi
            # CRITICAL: Also explicitly ensure setuptools==60.5.0 is installed in buildreq
            # Force reinstall to ensure we have setuptools==60.5.0 with dep_util
            echo "[INFO] Astropy: ensuring setuptools==60.5.0 in buildreq for dep_util..." >&2
            "$PY" -m pip install --no-cache-dir --force-reinstall -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_switch_$$.log 2>&1 || true
            # Verify setuptools version in buildreq
            SETUPTOOLS_VER=$("$PY" -c "import sys; sys.path.insert(0, '$ASTROPY_BUILDREQ_DIR'); import setuptools; print(setuptools.__version__)" 2>/dev/null || echo "")
            if [ -n "$SETUPTOOLS_VER" ]; then
              SETUPTOOLS_MAJOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f1)
              SETUPTOOLS_MINOR=$(echo "$SETUPTOOLS_VER" | cut -d. -f2)
              # Check if version is >= 60.9 (dep_util removed) or >= 61
              NEED_DOWNGRADE=0
              if [ "$SETUPTOOLS_MAJOR" -ge 61 ] 2>/dev/null; then
                NEED_DOWNGRADE=1
              elif [ "$SETUPTOOLS_MAJOR" -eq 60 ] 2>/dev/null && [ "$SETUPTOOLS_MINOR" -ge 9 ] 2>/dev/null; then
                NEED_DOWNGRADE=1
              fi
              if [ "$NEED_DOWNGRADE" -eq 1 ]; then
                echo "[WARN] Astropy: setuptools version $SETUPTOOLS_VER >= 60.9 in buildreq, forcing downgrade to 60.5.0..." >&2
                "$PY" -m pip install --no-cache-dir --force-reinstall --no-deps -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_downgrade_switch_$$.log 2>&1 || true
              else
                echo "[INFO] Astropy: setuptools version $SETUPTOOLS_VER < 66 in buildreq (OK)" >&2
              fi
            fi
            # Reinstall pytest for the new Python version
            if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
              PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
              PYTEST_VER="7.4.4"
              if [ "$PY_VER" = "3.6" ]; then
                PYTEST_VER="7.0.1"
              fi
              "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pytest==${PYTEST_VER}" >/tmp/apr_astropy_pytest_reinstall_$$.log 2>&1 || true
            fi
            # CRITICAL: Ensure PYTHONPATH has buildreq BEFORE any other setuptools locations
            # This ensures setuptools==60.5.0 from buildreq is used instead of newer versions
            if [ -n "${PYTHONPATH:-}" ]; then
              PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$ASTROPY_TOOLCHAIN_DIR$" | grep -v "^$ASTROPY_BUILDREQ_DIR$" | grep -v "^$SITE_DIR$" | tr '\n' ':' | sed 's/:$//')
              export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:$SITE_DIR:${PYTHONPATH}"
            else
              export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:$SITE_DIR"
            fi
            # Retry wheel build with new Python
            echo "[INFO] Astropy: retrying wheel build with Python ${REQUIRED_PY_VER}..." >&2
            if "$PY" -m pip wheel --no-deps --no-build-isolation -w "$WHEEL_DIR" . >"$WHEEL_LOG" 2>&1; then
              WHEEL_FILE="$(ls -1 "$WHEEL_DIR"/astropy-*.whl 2>/dev/null | head -1 || true)"
              if [ -n "$WHEEL_FILE" ] && [ -f "$WHEEL_FILE" ]; then
                echo "[INFO] Astropy: installing wheel into $SITE_DIR: $WHEEL_FILE" >&2
                INSTALL_LOG="/tmp/apr_astropy_wheel_install_$$.log"
                _APR_ASTROPY_PYTHONPATH_SAVED="$PYTHONPATH"
                export PYTHONPATH="$ASTROPY_BUILDREQ_DIR:$SITE_DIR:${_APR_ASTROPY_CLEANED_PYTHONPATH}"
                if ! "$PY" -m pip install --no-cache-dir --no-deps --no-build-isolation -t "$SITE_DIR" "$WHEEL_FILE" >"$INSTALL_LOG" 2>&1; then
                  echo "[ERROR] Astropy: wheel install into SITE_DIR failed. Tail:" >&2
                  tail -200 "$INSTALL_LOG" >&2 || true
                  export PYTHONPATH="$_APR_ASTROPY_PYTHONPATH_SAVED"
                  exit 2
                fi
                # CRITICAL: Install runtime dependencies (erfa/pyerfa) that are required by astropy
                # The wheel was installed with --no-deps, so we need to install dependencies separately
                echo "[INFO] Astropy: installing runtime dependencies (erfa/pyerfa) into $SITE_DIR..." >&2
                if ! "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "pyerfa" >/tmp/apr_astropy_runtime_deps_retry_$$.log 2>&1; then
                  # Fallback: try erfa (older name) or pyerfa with different versions
                  echo "[WARN] Astropy: pyerfa install failed, trying erfa... Tail:" >&2
                  tail -50 /tmp/apr_astropy_runtime_deps_retry_$$.log >&2 || true
                  "$PY" -m pip install --no-cache-dir -t "$SITE_DIR" "erfa" >/tmp/apr_astropy_runtime_deps_erfa_retry_$$.log 2>&1 || true
                fi
                # CRITICAL: Ensure PYTHONPATH prioritizes SITE_DIR over /testbed to import from installed wheel
                # Remove /testbed from PYTHONPATH to prevent importing from source checkout
                if [ -n "${PYTHONPATH:-}" ]; then
                  PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
                  export PYTHONPATH="$SITE_DIR:$ASTROPY_BUILDREQ_DIR:${PYTHONPATH}"
                else
                  export PYTHONPATH="$SITE_DIR:$ASTROPY_BUILDREQ_DIR"
                fi
                export APR_ASTROPY_RUN_FROM_TMP=1
                echo "[INFO] Astropy: wheel installed; PYTHONPATH configured to import from SITE_DIR (not /testbed)" >&2
              fi
            fi
          else
            echo "[WARN] Astropy: failed to bootstrap Python ${REQUIRED_PY_VER}, will fall back to in-place build" >&2
          fi
            fi
            fi
        fi
      fi
    fi
    
    # If wheel build still failed (or version switch didn't work), fall back to in-place build
    if [ "${APR_ASTROPY_RUN_FROM_TMP:-0}" != "1" ]; then
      echo "[WARN] Astropy: wheel build failed, will fall back to in-place build. Tail:" >&2
      tail -200 "$WHEEL_LOG" >&2 || true
    fi
  fi

  # Fallback path: try in-place build_ext (needed for some source-checkout import patterns).
  # Before falling back, check if there was a Python version mismatch that we should handle
  if [ "${APR_ASTROPY_RUN_FROM_TMP:-0}" != "1" ]; then
    # Check if build_ext failure might be due to Python version mismatch
    if [ -f "$BUILD_LOG" ]; then
      PY_VERSION_ERROR=$(grep -i "requires a different Python\|not in.*>=\|requires Python.*>=" "$BUILD_LOG" 2>/dev/null | head -1 || echo "")
      if [ -n "$PY_VERSION_ERROR" ] && [ -f "/miniconda.sh" ]; then
        echo "[INFO] Astropy: detected Python version requirement in build_ext error, switching Python..." >&2
        # Extract and switch version (reuse the same logic as above)
        REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*not in.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
        if [ -z "$REQUIRED_PY_VER" ]; then
          REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*Python.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
        fi
        if [ -z "$REQUIRED_PY_VER" ]; then
          REQUIRED_PY_VER=$(echo "$PY_VERSION_ERROR" | sed -n "s/.*>=\([0-9]\+\.[0-9]\+\).*/\1/p" || echo "")
        fi
        # Switch Python version if detected (same logic as wheel build failure)
        if [ -n "$REQUIRED_PY_VER" ]; then
          CONDA_BASE=""
          if [ -x "/opt/miniconda3/bin/conda" ]; then
            CONDA_BASE="/opt/miniconda3"
          elif [ -x "/tmp/apr_miniconda3/bin/conda" ]; then
            CONDA_BASE="/tmp/apr_miniconda3"
          fi
          if [ -n "$CONDA_BASE" ] && [ -x "${CONDA_BASE}/bin/conda" ]; then
            PY_REQ_ENV="${CONDA_BASE}/envs/apr_astropy_py${REQUIRED_PY_VER//./}"
            PY_REQ="${PY_REQ_ENV}/bin/python"
            if [ ! -x "$PY_REQ" ]; then
              source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
              "${CONDA_BASE}/bin/conda" create -y -p "$PY_REQ_ENV" "python=${REQUIRED_PY_VER}" pip >/tmp/apr_astropy_create_py${REQUIRED_PY_VER//./}.log 2>&1 || true
            fi
            if [ -x "$PY_REQ" ]; then
              PY="$PY_REQ"
              echo "[INFO] Astropy: switched to Python ${REQUIRED_PY_VER} for in-place build: PY=$PY" >&2
              # Reinstall toolchain and build requirements
              rm -rf "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
              mkdir -p "$ASTROPY_TOOLCHAIN_DIR" "$ASTROPY_BUILDREQ_DIR" 2>/dev/null || true
              PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
              if [ "$PY_VER" = "3.6" ]; then
                "$PY" -m pip install --no-cache-dir -t "$ASTROPY_TOOLCHAIN_DIR" "pip==21.3.1" "setuptools==59.6.0" "wheel" >/tmp/apr_astropy_toolchain_reinstall_$$.log 2>&1 || true
              else
                # CRITICAL: Use setuptools==60.5.0 to preserve setuptools.dep_util (required by old astropy code, last version with dep_util)
                "$PY" -m pip install --no-cache-dir -t "$ASTROPY_TOOLCHAIN_DIR" "setuptools==60.5.0" "wheel" >/tmp/apr_astropy_toolchain_reinstall_$$.log 2>&1 || true
              fi
            # Reinstall build requirements
            # CRITICAL: Ensure setuptools==60.5.0 is installed in buildreq to preserve dep_util
            if [ -f "/testbed/pyproject.toml" ]; then
              BUILD_REQS_FILE="/tmp/apr_astropy_buildreqs_$$.txt"
              "$PY" - <<'PY_APR_BUILDREQS' >"$BUILD_REQS_FILE" 2>/dev/null || true
try:
    import tomllib as _toml
except:
    try:
        import tomli as _toml
    except:
        _toml = None
if _toml:
    with open("/testbed/pyproject.toml", "rb") as f:
        d = _toml.load(f)
        reqs = d.get("build-system", {}).get("requires", [])
        for r in reqs:
            if isinstance(r, str) and r.strip():
                # Replace setuptools with setuptools==60.5.0 to preserve dep_util
                r_clean = r.strip()
                if r_clean.startswith("setuptools") and "60.5.0" not in r_clean and "==" not in r_clean:
                    print("setuptools==60.5.0")
                else:
                    print(r_clean)
PY_APR_BUILDREQS
              if [ -s "$BUILD_REQS_FILE" ]; then
                "$PY" -m pip install -U --no-cache-dir -t "$ASTROPY_BUILDREQ_DIR" -r "$BUILD_REQS_FILE" >/tmp/apr_astropy_build_requires_reinstall_$$.log 2>&1 || true
              fi
            fi
            # Also ensure setuptools==60.5.0 is explicitly installed in buildreq
            "$PY" -m pip install --no-cache-dir -t "$ASTROPY_BUILDREQ_DIR" "setuptools==60.5.0" >/tmp/apr_astropy_buildreq_setuptools_$$.log 2>&1 || true
            # Update PYTHONPATH
            if [ -n "${PYTHONPATH:-}" ]; then
              PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$ASTROPY_TOOLCHAIN_DIR$" | grep -v "^$ASTROPY_BUILDREQ_DIR$" | tr '\n' ':' | sed 's/:$//')
              export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:${PYTHONPATH}"
            else
              export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR"
            fi
            fi
          fi
        fi
      fi
    fi
    
    echo "[INFO] Astropy: building extension modules in-place (setup.py build_ext --inplace)..." >&2
    if [ -f "/testbed/setup.py" ]; then
      # CRITICAL: Ensure PYTHONPATH has buildreq BEFORE any other setuptools locations
      # This ensures setuptools==60.5.0 from buildreq is used instead of newer versions
      # Re-export PYTHONPATH to ensure buildreq is first (may have been modified during wheel build attempts)
      if [ -n "${PYTHONPATH:-}" ]; then
        PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^$ASTROPY_TOOLCHAIN_DIR$" | grep -v "^$ASTROPY_BUILDREQ_DIR$" | grep -v "^$SITE_DIR$" | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
        export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:$SITE_DIR:/testbed${PYTHONPATH:+:${PYTHONPATH}}"
      else
        export PYTHONPATH="$ASTROPY_TOOLCHAIN_DIR:$ASTROPY_BUILDREQ_DIR:$SITE_DIR:/testbed"
      fi
      # Verify setuptools.dep_util is available before build_ext
      # CRITICAL: Use a fresh Python process and ensure buildreq is first in sys.path before any import
      if ! "$PY" -c "import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; from setuptools.dep_util import newer_group" >/dev/null 2>&1; then
        echo "[ERROR] Astropy: setuptools.dep_util not available in buildreq before build_ext. PYTHONPATH=$PYTHONPATH" >&2
        "$PY" -c "import sys; sys.path = ['$ASTROPY_BUILDREQ_DIR'] + [p for p in sys.path if p != '$ASTROPY_BUILDREQ_DIR']; import setuptools; print('buildreq setuptools version:', setuptools.__version__); print('buildreq setuptools file:', setuptools.__file__)" >&2 || true
        # Also check if dep_util file exists
        "$PY" -c "import os; dep_util_path = os.path.join('$ASTROPY_BUILDREQ_DIR', 'setuptools', 'dep_util.py'); print('dep_util.py exists:', os.path.exists(dep_util_path))" >&2 || true
        exit 2
      fi
      if ! "$PY" /testbed/setup.py build_ext --inplace >"$BUILD_LOG" 2>&1; then
        echo "[ERROR] Astropy: build_ext --inplace failed. Tail:" >&2
        tail -200 "$BUILD_LOG" >&2 || true
        exit 2
      fi
    else
      echo "[ERROR] Astropy: /testbed/setup.py missing; cannot build extensions" >&2
      exit 2
    fi
  fi

  # Sanity-check that Astropy is importable and has a version (prevents "broken installation").
  # CRITICAL: For wheel installations, ensure PYTHONPATH prioritizes SITE_DIR over /testbed
  if [ "${APR_ASTROPY_RUN_FROM_TMP:-0}" = "1" ]; then
    # Wheel installation: ensure SITE_DIR is first in PYTHONPATH (already set above)
    echo "[INFO] Astropy: sanity check for wheel installation (PYTHONPATH=$PYTHONPATH)..." >&2
  else
    # In-place build: ensure /testbed is first for source checkout
    if [ -n "${PYTHONPATH:-}" ]; then
      PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | tr '\n' ':' | sed 's/:$//')
      export PYTHONPATH="/testbed:${PYTHONPATH}"
    else
      export PYTHONPATH="/testbed"
    fi
    echo "[INFO] Astropy: sanity check for in-place build (PYTHONPATH=$PYTHONPATH)..." >&2
  fi
  if ! "$PY" - <<'PY_APR_ASTROPY_SANITY' >/tmp/apr_astropy_sanity_out_$$.log 2>&1; then
import sys, traceback, os
# CRITICAL: For wheel installation, ensure /testbed is NOT in sys.path
# Remove /testbed from sys.path to prevent importing from source checkout
if os.environ.get("APR_ASTROPY_RUN_FROM_TMP", "0") == "1":
    # CRITICAL: Change working directory first to avoid empty string '' in sys.path pointing to /testbed
    # The empty string '' in sys.path represents the current working directory
    original_cwd = os.getcwd()
    if original_cwd == "/testbed":
        os.chdir("/tmp")
        print(f"DEBUG: Changed CWD from /testbed to /tmp", file=sys.stderr)
    # Remove /testbed from sys.path if present (more aggressive removal)
    original_path = list(sys.path)
    sys.path = [p for p in sys.path if p and p != "/testbed" and not p.endswith("/testbed") and "/testbed" not in p]
    # Also check if /testbed was in original path and log it
    if "/testbed" in original_path or "" in original_path:
        print(f"DEBUG: Removed /testbed and empty string from sys.path. Original: {original_path[:3]}...", file=sys.stderr)
    # Force remove /testbed from any site-packages .pth files by clearing site cache
    import site
    # Clear site-packages cache to force re-scan
    site._init_pathinfo()
    # Also try to remove /testbed from sys.modules if it was already imported
    if "astropy" in sys.modules:
        del sys.modules["astropy"]
try:
    import astropy  # noqa: F401
    v = getattr(astropy, "__version__", "")
    print(f"astropy_version={v!r}")
    # Verify astropy is imported from SITE_DIR, not /testbed
    astropy_path = astropy.__file__ if hasattr(astropy, "__file__") else ""
    if "/testbed" in astropy_path:
        print(f"ERROR: astropy imported from /testbed: {astropy_path}", file=sys.stderr)
        print(f"ERROR: sys.path was: {sys.path[:5]}", file=sys.stderr)
        print(f"ERROR: CWD was: {os.getcwd()}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if v else 2)
except Exception:
    traceback.print_exc()
    sys.exit(2)
PY_APR_ASTROPY_SANITY
    echo "[ERROR] Astropy: import/version sanity check failed after install (broken installation)" >&2
    if [ "${APR_ASTROPY_RUN_FROM_TMP:-0}" = "1" ]; then
      echo "[ERROR] Astropy: wheel installation sanity check failed. PYTHONPATH=$PYTHONPATH" >&2
    else
      echo "[ERROR] Astropy: tail of build log ($BUILD_LOG):" >&2
      tail -200 "$BUILD_LOG" >&2 || true
    fi
    echo "[ERROR] Astropy: sanity traceback/output:" >&2
    tail -200 /tmp/apr_astropy_sanity_out_$$.log >&2 || true
    exit 2
  fi
else
  # scikit-learn: DO NOT do editable install from /testbed; it adds /testbed to sys.path (egg-link)
  # and forces importing the unbuilt source checkout, causing missing compiled extension errors.
  if [ "${APR_IS_SCIKITLEARN:-0}" != "1" ]; then
    "$PY" -m pip install -e . >/dev/null 2>&1 || true
  fi
fi

apply_patch() {
  # Unused: test patch is applied by APR_TEST_PATCH_CONTENT block via printf|git apply.
  # Empty heredoc can cause "unexpected end of file" in some bash; use : no-op.
  :
}

revert_tests() {
  # Revert only files touched by the test patch, to avoid clobbering agent code changes.
  if [ -n "${APR_BASE_COMMIT:-}" ] && [ -n "${APR_TEST_FILES:-}" ]; then
    IFS=$'\n'
    for f in ${APR_TEST_FILES}; do
      git checkout "${APR_BASE_COMMIT}" -- "${f}" >/dev/null 2>&1 || true
    done
    unset IFS
  fi
}

trap revert_tests EXIT

# Load test patch from file if APR_TEST_PATCH_FILE is set (for large patches)
# Otherwise use APR_TEST_PATCH variable directly
APR_TEST_PATCH_CONTENT=""
if [ -n "${APR_TEST_PATCH_FILE:-}" ] && [ -f "$APR_TEST_PATCH_FILE" ]; then
  APR_TEST_PATCH_CONTENT=$(cat "$APR_TEST_PATCH_FILE" 2>/dev/null || echo "")
elif [ -n "${APR_TEST_PATCH:-}" ]; then
  APR_TEST_PATCH_CONTENT="${APR_TEST_PATCH}"
fi

if [ -n "$APR_TEST_PATCH_CONTENT" ]; then
  # Align with SWE-bench eval.sh: reset the touched test files to base_commit first
  if [ -n "${APR_BASE_COMMIT:-}" ] && [ -n "${APR_TEST_FILES:-}" ]; then
    IFS=$'\n'
    for f in ${APR_TEST_FILES}; do
      git checkout "${APR_BASE_COMMIT}" -- "${f}" >/dev/null 2>&1 || true
    done
    unset IFS
  fi
  # Apply test patch (use stdin; do NOT embed literal content)
  # Try normal apply first (most patches should work this way)
  if ! printf "%s\n" "$APR_TEST_PATCH_CONTENT" | git apply -v - 2>&1; then
    # Only for specific cases where patch fails due to whitespace/minor differences:
    # Try --3way (allows fuzzy matching using 3-way merge) as fallback
    echo "[WARN] Normal patch apply failed, trying --3way (fuzzy matching)..." >&2
    if ! printf "%s\n" "$APR_TEST_PATCH_CONTENT" | git apply -v --3way - 2>&1; then
      # Last resort: --reject (creates .rej files for failed hunks)
      # This is best-effort and may result in incomplete test setup
      echo "[WARN] --3way also failed, trying --reject (best-effort, may create .rej files)..." >&2
      if ! printf "%s\n" "$APR_TEST_PATCH_CONTENT" | git apply -v --reject - 2>&1; then
        echo "[ERROR] All patch apply strategies failed. Test patch may be incompatible." >&2
        # Check if any .rej files were created (indicates partial failure)
        if find /testbed -name "*.rej" -type f 2>/dev/null | head -1 | grep -q .; then
          echo "[WARN] Patch partially applied (some hunks rejected). Tests may not run correctly." >&2
        fi
        # Continue anyway (let test framework handle it, may fail with rc=4 which is expected)
      else
        # --reject succeeded, but check for rejected hunks
        if find /testbed -name "*.rej" -type f 2>/dev/null | head -1 | grep -q .; then
          echo "[WARN] Patch applied with some rejected hunks (.rej files present). Tests may be incomplete." >&2
        fi
      fi
    fi
  fi
fi

RUN_RC=0
OUT_FILE="$(mktemp -p /tmp)"

run_pytest_one() {
  # $1: file (optional, may be empty)
  # $2: test selector (name or nodeid)
  local f="$1"
  local t="$2"
  t=$(printf '%s' "$t" | tr -d '\r\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

  # Django: run all tests via runtests.py to avoid INSTALLED_APPS/migration issues under pytest
  if [ "${APR_IS_DJANGO:-0}" = "1" ]; then
    # Debug: why runtests was not used (Django only)
    echo "[DEBUG runtests] APR_IS_DJANGO=${APR_IS_DJANGO:-0}, test_name='$t', runtests.py exists: $([ -f /testbed/tests/runtests.py ] && echo yes || echo no)" >&2
    # Prefer runtests for all Django tests (format test_method (class.path))
    if [ -f /testbed/tests/runtests.py ]; then
      if echo "$t" | grep -qE '^[a-zA-Z0-9_]+ *\([a-zA-Z0-9_.]+\)'; then
        class_path=$(echo "$t" | sed -n 's/.*(\([^)]*\)).*/\1/p')
        method_name=$(echo "$t" | sed 's/ *([^)]*).*//')
        echo "[DEBUG runtests] regex matched, class_path='$class_path', method_name='$method_name'" >&2
        # All Django test modules use runtests; class_path format as above
        if echo "$class_path" | grep -qE '\.(tests|test_|tests\.)'; then
          # If class_path ends with method_name use as-is; else append
          if echo "$class_path" | grep -qE "\.${method_name}$"; then
            runtests_label="$class_path"
          else
            runtests_label="${class_path}.${method_name}"
          fi
          echo "[DEBUG runtests] class_path is Django test module, using runtests with label='$runtests_label'" >&2
          export APR_DJANGO_USE_RUNTESTS=1
          if [ -f /testbed/tests/test_sqlite.py ] || [ -d /testbed/tests/test_sqlite ]; then
            DJANGO_SETTINGS_RUN="test_sqlite"
          elif [ -f /testbed/tests/settings.py ]; then
            DJANGO_SETTINGS_RUN="tests.settings"
          else
            DJANGO_SETTINGS_RUN="test_sqlite"
          fi
          export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS_RUN"
          echo "[DEBUG runtests] Executing: cd /testbed && $PY tests/runtests.py --noinput --failfast --settings=$DJANGO_SETTINGS_RUN $runtests_label" >&2
          echo "[DEBUG runtests] About to execute runtests, OUT_FILE=$OUT_FILE" >&2
          (cd /testbed && "$PY" tests/runtests.py --noinput --failfast --settings="$DJANGO_SETTINGS_RUN" "$runtests_label") 2>&1 | tee "$OUT_FILE"
          RUNTESTS_RC="${PIPESTATUS[0]}"
          echo "[DEBUG runtests] runtests command exited with rc=$RUNTESTS_RC" >&2
          echo "[DEBUG runtests] OUT_FILE content (first 20 lines):" >&2
          head -20 "$OUT_FILE" >&2 || echo "[DEBUG runtests] Failed to read OUT_FILE" >&2
          echo "[DEBUG runtests] OUT_FILE contains 'pytest': $(grep -q pytest "$OUT_FILE" && echo yes || echo no)" >&2
          echo "[DEBUG runtests] OUT_FILE contains 'runtests': $(grep -q runtests "$OUT_FILE" && echo yes || echo no)" >&2
          echo "[DEBUG runtests] Returning from run_pytest_one with rc=$RUNTESTS_RC" >&2
          return "$RUNTESTS_RC"
        else
          echo "[DEBUG runtests] class_path '$class_path' does not appear to be a Django test module" >&2
        fi
      else
        echo "[DEBUG runtests] regex '^[a-zA-Z0-9_]+ *\([a-zA-Z0-9_.]+\)' did not match test_name='$t'" >&2
        # If test name has special chars, find actual test method from file or run via runtests path
        if [ -n "$f" ] && [ -f "/testbed/$f" ]; then
          echo "[DEBUG runtests] Test name contains special chars, trying to find test method from file '$f'" >&2
          # Find test method in file (test name may be docstring)
          local test_method=$( "$PY" -c "
import ast
import sys
import re
try:
    test_name = '''$t'''
    with open('/testbed/$f', 'r') as file:
        tree = ast.parse(file.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            # Check if docstring contains test name (case/space insensitive)
            if node.body and isinstance(node.body[0], ast.Expr):
                if isinstance(node.body[0].value, ast.Str):
                    docstring = node.body[0].value.s
                elif isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    docstring = node.body[0].value.value
                else:
                    continue
                # Match if test name keywords appear in docstring
                test_keywords = re.sub(r'[^a-zA-Z0-9\s]', ' ', test_name).split()
                docstring_lower = docstring.lower()
                if all(kw.lower() in docstring_lower for kw in test_keywords if len(kw) > 3):
                    print(node.name)
                    sys.exit(0)
except Exception as e:
    pass
" 2>/dev/null )
          
          # If test method found, build runtests label
          if [ -n "$test_method" ]; then
            local module_path="${f%.py}"
            module_path="${module_path#tests/}"
            module_path="${module_path//\//.}"
            # Find test class
            local test_class=$( "$PY" -c "
import ast
import sys
try:
    with open('/testbed/$f', 'r') as file:
        tree = ast.parse(file.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '$test_method':
                    print(node.name)
                    sys.exit(0)
except:
    pass
" 2>/dev/null )
            
            if [ -n "$test_class" ]; then
              runtests_label="${module_path}.${test_class}.${test_method}"
            else
              runtests_label="${module_path}.${test_method}"
            fi
            
            echo "[DEBUG runtests] Found test method '$test_method' in class '${test_class:-N/A}', using runtests label='$runtests_label'" >&2
            export APR_DJANGO_USE_RUNTESTS=1
            if [ -f /testbed/tests/test_sqlite.py ] || [ -d /testbed/tests/test_sqlite ]; then
              DJANGO_SETTINGS_RUN="test_sqlite"
            elif [ -f /testbed/tests/settings.py ]; then
              DJANGO_SETTINGS_RUN="tests.settings"
            else
              DJANGO_SETTINGS_RUN="test_sqlite"
            fi
            export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS_RUN"
            echo "[DEBUG runtests] Executing: cd /testbed && $PY tests/runtests.py --noinput --failfast --settings=$DJANGO_SETTINGS_RUN $runtests_label" >&2
            (cd /testbed && "$PY" tests/runtests.py --noinput --failfast --settings="$DJANGO_SETTINGS_RUN" "$runtests_label") 2>&1 | tee "$OUT_FILE"
            return "${PIPESTATUS[0]}"
          else
            echo "[DEBUG runtests] Could not find test method matching '$t' in file '$f', will try runtests with module path" >&2
            # If no specific method found, run whole test file by module path
            local module_path="${f%.py}"
            module_path="${module_path#tests/}"
            module_path="${module_path//\//.}"
            if [ -n "$module_path" ]; then
              echo "[DEBUG runtests] Trying runtests with module path '$module_path'" >&2
              export APR_DJANGO_USE_RUNTESTS=1
              if [ -f /testbed/tests/test_sqlite.py ] || [ -d /testbed/tests/test_sqlite ]; then
                DJANGO_SETTINGS_RUN="test_sqlite"
              elif [ -f /testbed/tests/settings.py ]; then
                DJANGO_SETTINGS_RUN="tests.settings"
              else
                DJANGO_SETTINGS_RUN="test_sqlite"
              fi
              export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS_RUN"
              echo "[DEBUG runtests] Executing: cd /testbed && $PY tests/runtests.py --noinput --failfast --settings=$DJANGO_SETTINGS_RUN $module_path" >&2
              (cd /testbed && "$PY" tests/runtests.py --noinput --failfast --settings="$DJANGO_SETTINGS_RUN" "$module_path") 2>&1 | tee "$OUT_FILE"
              return "${PIPESTATUS[0]}"
            fi
          fi
        fi
      fi
    else
      echo "[DEBUG runtests] /testbed/tests/runtests.py does not exist" >&2
    fi
    # Non-runtests Django cases (e.g. queryset_pickle): use pytest; for special-char test names use file path not -k
    local django_test_module=""
    if echo "$t" | grep -q '('; then
      django_test_module=$(echo "$t" | sed -n 's/.*(\([^)]*\)).*/\1/p')
    fi
    if [ -z "$django_test_module" ] && [ -n "$f" ]; then
      django_test_module="${f%.py}"
      django_test_module="${django_test_module#tests/}"
      django_test_module="${django_test_module//\//.}"
    fi
    if [ -n "$django_test_module" ]; then
      DJANGO_SETTINGS_RUN=""
      if [ -f /testbed/tests/test_sqlite.py ] || [ -d /testbed/tests/test_sqlite ]; then
        DJANGO_SETTINGS_RUN="test_sqlite"
      elif [ -f /testbed/tests/settings.py ]; then
        DJANGO_SETTINGS_RUN="tests.settings"
      elif "$PY" -c "import test_sqlite" >/dev/null 2>&1; then
        DJANGO_SETTINGS_RUN="test_sqlite"
      else
        DJANGO_SETTINGS_RUN="test_sqlite"
      fi
      if [ -n "$DJANGO_SETTINGS_RUN" ]; then
        export DJANGO_SETTINGS_MODULE="$DJANGO_SETTINGS_RUN"
      fi
      # If test name has special chars, use file path only, not -k
      if echo "$t" | grep -qE '[().:\[\]{}]'; then
        echo "[DEBUG runtests] Test name contains special chars, running entire test file '$f' instead of using -k" >&2
        if [ -n "$f" ] && [ -f "/testbed/$f" ]; then
          "$PY" -m pytest -q -x "/testbed/$f" 2>&1 | tee "$OUT_FILE"
        else
          "$PY" -m pytest -q -x -k "$t" 2>&1 | tee "$OUT_FILE"
        fi
      else
        local method_name=$(echo "$t" | sed 's/ *([^)]*).*//')
        if [ -n "$f" ]; then
          "$PY" -m pytest -q -x "$f" -k "$method_name" 2>&1 | tee "$OUT_FILE"
        else
          "$PY" -m pytest -q -x -k "$method_name" 2>&1 | tee "$OUT_FILE"
        fi
      fi
      return "${PIPESTATUS[0]}"
    fi
  fi
  
  # Non-Django: use pytest as before.
  # For Astropy: disable pytest's warnings plugin to avoid it overriding warnings.showwarning,
  # which can trigger astropy.logger.LoggingError at import time.
  local -a PYTEST_CMD
  local -a PYTEST_EXTRA_ARGS
  PYTEST_CMD=("$PY" -m pytest)
  PYTEST_EXTRA_ARGS=()
  if [ "${APR_IS_ASTROPY:-0}" = "1" ]; then
    PYTEST_EXTRA_ARGS+=(-p no:warnings)
  fi
  # For pytest-dev: ignore unknown config option warnings (e.g., rsyncdirs)
  # These warnings can cause pytest to fail in strict mode (rc=3)
  if [ "${APR_IS_PYTESTDEV:-0}" = "1" ]; then
    PYTEST_EXTRA_ARGS+=(-W ignore::pytest.PytestConfigWarning)
  fi
  # For matplotlib: ignore pyparsing deprecation warnings (e.g., enablePackrat)
  # These warnings can cause pytest collection to fail (rc=2)
  if [ "${APR_IS_MATPLOTLIB:-0}" = "1" ]; then
    PYTEST_EXTRA_ARGS+=(-W ignore::pyparsing.warnings.PyparsingDeprecationWarning)
  fi

  # Astropy: if we installed a wheel into $SITE_DIR, run pytest from /tmp (not /testbed) so that
  # imports resolve from $SITE_DIR rather than the source checkout.
  # CRITICAL FIX: Resolve ImportPathMismatchError by ensuring /testbed takes precedence and removing conftest.py from SITE_DIR
  local astropy_run_from_tmp=0
  if [ "${APR_IS_ASTROPY:-0}" = "1" ] && [ "${APR_ASTROPY_RUN_FROM_TMP:-0}" = "1" ]; then
    astropy_run_from_tmp=1
    # CRITICAL: Ensure /testbed is first (for conftest.py) but SITE_DIR is also in PYTHONPATH (for hypothesis and other deps)
    # The order is: /testbed (for conftest.py) : SITE_DIR (for deps like hypothesis) : ... (rest of PYTHONPATH)
    if [ -n "${PYTHONPATH:-}" ]; then
      # Remove /testbed and SITE_DIR if already present, then add them in correct order
      PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^/testbed$" | grep -v "^$SITE_DIR$" | tr '\n' ':')
      export PYTHONPATH="/testbed:$SITE_DIR:${PYTHONPATH%:}"
    else
      export PYTHONPATH="/testbed:$SITE_DIR"
    fi
    echo "[INFO] Astropy: pytest PYTHONPATH set to: $PYTHONPATH (for conftest.py and deps like hypothesis)" >&2
    # Remove conftest.py from SITE_DIR to avoid path mismatch
    if [ -d "$SITE_DIR/astropy" ] && [ -f "$SITE_DIR/astropy/conftest.py" ]; then
      echo "[INFO] Astropy: removing conftest.py from SITE_DIR to avoid ImportPathMismatchError" >&2
      rm -f "$SITE_DIR/astropy/conftest.py" 2>/dev/null || true
      # Also remove __pycache__ versions
      find "$SITE_DIR/astropy" -name "conftest*.pyc" -delete 2>/dev/null || true
      find "$SITE_DIR/astropy" -path "*/__pycache__/conftest*" -delete 2>/dev/null || true
    fi
  fi

  local run_from_tmp=0
  if [ "$astropy_run_from_tmp" = "1" ]; then
    run_from_tmp=1
  fi
  
  if [[ "$t" == *"::"* ]]; then
    # Check if test name contains parameterized test marker but might be incomplete
    # e.g., "test_unparse[(1," should fallback to using -k pattern matching
    if echo "$t" | grep -qE '\[.*[^]]$'; then
      # Parameterized test name appears incomplete (ends with [ but no closing ])
      # Extract file + base test function and use -k pattern matching instead.
      # NOTE: pytest -k matches keywords (usually function/class names), NOT full nodeids.
      local base_nodeid
      base_nodeid=$(echo "$t" | sed 's/\[.*$//')
      local base_kw
      base_kw=$(echo "$base_nodeid" | sed 's/.*:://')
      echo "[INFO] Parameterized test name appears incomplete, using -k keyword: $base_kw (from $base_nodeid)" >&2
      if [ -n "$f" ]; then
        "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$f" -k "$base_kw" 2>&1 | tee "$OUT_FILE"
      else
        # Extract file path from test name if available
        local test_file_part=$(echo "$t" | sed 's/::.*$//')
        if [ -n "$test_file_part" ] && [ -f "$test_file_part" ]; then
          "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$test_file_part" -k "$base_kw" 2>&1 | tee "$OUT_FILE"
        else
          "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x -k "$base_kw" 2>&1 | tee "$OUT_FILE"
        fi
      fi
      return "${PIPESTATUS[0]}"
    fi
    # Try direct execution first
    local t_exec="$t"
    if [ "$run_from_tmp" = "1" ] && [[ "$t_exec" != /* ]]; then
      t_exec="/testbed/$t_exec"
    fi
    if [ "$run_from_tmp" = "1" ]; then
      (cd /tmp && "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$t_exec" 2>&1 | tee "$OUT_FILE")
    else
      "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$t_exec" 2>&1 | tee "$OUT_FILE"
    fi
    local direct_rc="${PIPESTATUS[0]}"
    # If direct execution fails with "not found", try -k pattern matching as fallback
    if [ "$direct_rc" -ne 0 ] && grep -qi "not found\|no collectors" "$OUT_FILE" 2>/dev/null; then
      local base_test=$(echo "$t" | sed 's/.*::\([^[]*\).*/\1/')
      echo "[INFO] Test not found with direct name, trying -k pattern: $base_test" >&2
      local test_file_part=$(echo "$t" | sed 's/::.*$//')
      if [ -n "$test_file_part" ] && [ -f "$test_file_part" ]; then
        local tf_exec="$test_file_part"
        if [ "$run_from_tmp" = "1" ] && [[ "$tf_exec" != /* ]]; then
          tf_exec="/testbed/$tf_exec"
        fi
        if [ "$run_from_tmp" = "1" ]; then
          (cd /tmp && "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$tf_exec" -k "$base_test" 2>&1 | tee "$OUT_FILE")
        else
          "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$tf_exec" -k "$base_test" 2>&1 | tee "$OUT_FILE"
        fi
      else
        if [ "$run_from_tmp" = "1" ]; then
          (cd /tmp && "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x -k "$base_test" 2>&1 | tee "$OUT_FILE")
        else
          "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x -k "$base_test" 2>&1 | tee "$OUT_FILE"
        fi
      fi
      return "${PIPESTATUS[0]}"
    fi
    return "$direct_rc"
  fi
  
  # For test names with parentheses (like Django), extract just the method name part
  local test_pattern="$t"
  if echo "$t" | grep -q '('; then
    test_pattern=$(echo "$t" | sed 's/ *([^)]*).*//')
  fi
  
  if [ -n "$f" ]; then
    local f_exec="$f"
    if [ "$run_from_tmp" = "1" ] && [[ "$f_exec" != /* ]]; then
      f_exec="/testbed/$f_exec"
    fi
    if [ "$run_from_tmp" = "1" ]; then
      (cd /tmp && "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$f_exec" -k "$test_pattern" 2>&1 | tee "$OUT_FILE")
    else
      "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x "$f_exec" -k "$test_pattern" 2>&1 | tee "$OUT_FILE"
    fi
    return "${PIPESTATUS[0]}"
  fi
  if [ "$run_from_tmp" = "1" ]; then
    (cd /tmp && "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x -k "$test_pattern" 2>&1 | tee "$OUT_FILE")
  else
    "${PYTEST_CMD[@]}" "${PYTEST_EXTRA_ARGS[@]}" -q -x -k "$test_pattern" 2>&1 | tee "$OUT_FILE"
  fi
  return "${PIPESTATUS[0]}"
}

# Try directives derived from test_patch first; if nothing collected, fall back to repo-wide -k.
FOUND=0
if [ -n "${APR_TEST_FILES:-}" ]; then
  echo "[DEBUG] APR_TEST_FILES is set, trying test files first" >&2
  IFS=$'\n'
  for f in ${APR_TEST_FILES}; do
    echo "[DEBUG] Calling run_pytest_one with file='$f', test_name='${APR_TEST_NAME}'" >&2
    set +e
    run_pytest_one "$f" "${APR_TEST_NAME}"
    RUN_RC=$?
    set -e
    echo "[DEBUG] run_pytest_one returned rc=$RUN_RC" >&2
    if [ "$RUN_RC" -ne 5 ]; then
      FOUND=1
      break
    fi
  done
  unset IFS
fi

if [ "$FOUND" -eq 0 ]; then
  echo "[DEBUG] FOUND=0, calling run_pytest_one without file, test_name='${APR_TEST_NAME}'" >&2
  set +e
  run_pytest_one "" "${APR_TEST_NAME}"
  RUN_RC=$?
  set -e
  echo "[DEBUG] run_pytest_one (no file) returned rc=$RUN_RC" >&2
fi

# rc=5 means "no tests collected" (not acceptable for TDD gates)
if [ "$RUN_RC" -eq 5 ]; then
  echo "ERROR: pytest collected 0 tests for '${APR_TEST_NAME}'" >&2
  exit 2
fi

exit "$RUN_RC"
""").replace('      __DJANGO_SITECUSTOMIZE_HEREDOC__', _DJANGO_SITECUSTOMIZE_HEREDOC)

        # Inject small parameters via environment variables (avoid quoting issues)
        test_files = "\n".join(directives)
        env_lines = []
        if base_commit:
            # Use heredoc to avoid bash syntax errors with special characters
            env_lines.append("export APR_BASE_COMMIT=$(cat <<'EOF_APR_BC'\n" + base_commit + "\nEOF_APR_BC\n)")
        
        # Detect Django project and handle test execution differently
        is_django = instance_id and "django__django" in instance_id.lower()
        # Detect Astropy project for warnings logging fix
        is_astropy = instance_id and "astropy__astropy" in instance_id.lower()
        # Detect pytest-dev (project is pytest itself; prefer importing from source /testbed/src)
        is_pytestdev = instance_id and instance_id.lower().startswith("pytest-dev__pytest-")
        # Detect pylint-dev (project is pylint; some images ship broken distutils-precedence.pth)
        is_pylintdev = instance_id and instance_id.lower().startswith("pylint-dev__pylint-")
        # Detect xarray (pydata__xarray-*) to avoid shadowing matplotlib in /tmp site
        is_xarray = instance_id and instance_id.lower().startswith("pydata__xarray-")
        # Detect sphinx-doc (HTML builder) for html5lib dependency
        is_sphinx = instance_id and instance_id.lower().startswith("sphinx-doc__sphinx-")
        # Detect sympy project for backwards-compatible sympy.utilities.pytest helpers
        is_sympy = instance_id and instance_id.lower().startswith("sympy__sympy-")
        # Detect matplotlib project to avoid shadowing matplotlib in /tmp site
        is_matplotlib = instance_id and instance_id.lower().startswith("matplotlib__matplotlib-")
        is_pallets = instance_id and instance_id.lower().startswith("pallets__")
        # Detect seaborn project (mwaskom__seaborn-*) for matplotlib dependencies
        is_seaborn = instance_id and instance_id.lower().startswith("mwaskom__seaborn-")
        # Detect scikit-learn project for testbed pip installation fix
        is_scikitlearn = instance_id and instance_id.lower().startswith("scikit-learn__scikit-learn-")
        
        # Root fix: install deterministic pip deps from SWE-bench spec (e.g. mpmath for sympy)
        # Also extract required Python version for version matching
        required_python_version = None
        try:
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # type: ignore
            version = (inst.get("version") or "") if inst else ""
            spec = (MAP_REPO_VERSION_TO_SPECS.get(inst.get("repo") or "", {}) or {}).get(version, {})  # type: ignore
            pip_pkgs = list(spec.get("pip_packages") or [])
            test_cmd = spec.get("test_cmd") or ""
            required_python_version = spec.get("python")
        except Exception:
            pip_pkgs = []
            test_cmd = ""
            required_python_version = None
        # PYLINT-DEV ENV FIX (project-scoped):
        # - appdirs: real runtime dependency for some old commits
        # - setuptools: for distutils/tooling compatibility
        # Note: _distutils_hack is NOT a standalone pip package; we will stub it in /tmp instead.
        if is_pylintdev:
            extra = ["appdirs", "setuptools>=50.0.0"]
            # Deduplicate while preserving order
            seen = set()
            merged = []
            for name in list(pip_pkgs) + extra:
                if name in seen:
                    continue
                seen.add(name)
                merged.append(name)
            pip_pkgs = merged
        # SPHINX-DOC ENV FIX (project-scoped):
        # Some Sphinx HTML/latex tests require html5lib to be present.
        if is_sphinx:
            extra = ["html5lib"]
            seen = set()
            merged = []
            for name in list(pip_pkgs) + extra:
                if name in seen:
                    continue
                seen.add(name)
                merged.append(name)
            pip_pkgs = merged
        # SEABORN ENV FIX (project-scoped):
        # Seaborn requires matplotlib and other dependencies, and needs Python 3.9+
        # Ensure required Python version is set for seaborn
        if is_seaborn:
            if required_python_version is None:
                required_python_version = "3.9"  # Seaborn requires Python 3.9
            # Ensure matplotlib and dependencies are in pip_pkgs
            seaborn_deps = ["matplotlib", "numpy", "pandas", "scipy"]
            seen = set(pip_pkgs)
            for dep in seaborn_deps:
                # Check if any version of this package is already in the list
                if not any(dep.split("==")[0].split(">=")[0].split("<=")[0].strip() == dep for d in pip_pkgs):
                    pip_pkgs.append(dep)
        # CRITICAL: Use temporary files for large environment variables to avoid "Argument list too long" error
        # Store pip_pkgs and test_patch for later file writing (after bind is defined)
        _apr_temp_pip_pkgs = pip_pkgs if pip_pkgs else None
        _apr_temp_test_patch = test_patch if test_patch else None
        
        if pip_pkgs:
            # Will write to file after bind is defined
            # CRITICAL: Only set file path, do NOT read file content into environment variable
            # Reading file content into APR_PIP_PACKAGES would still cause "Argument list too long"
            env_lines.append("export APR_PIP_PACKAGES_FILE=/testbed/.apr_env/apr_pip_packages.txt")
            # Do NOT set APR_PIP_PACKAGES variable - script will read from file directly
        if test_cmd:
            # Use heredoc to avoid bash syntax errors when test_cmd contains special characters
            env_lines.append("export APR_TEST_CMD=$(cat <<'EOF_APR_TC'\n" + test_cmd + "\nEOF_APR_TC\n)")
        if test_patch:
            # Will write to file after bind is defined
            # CRITICAL: Only set file path, do NOT read file content into environment variable
            # Reading file content into APR_TEST_PATCH would still cause "Argument list too long"
            env_lines.append("export APR_TEST_PATCH_FILE=/testbed/.apr_env/apr_test_patch.txt")
            # Do NOT set APR_TEST_PATCH variable - script will read from file directly
        # Use heredoc for APR_TEST_NAME to safely handle special characters (quotes, commas, etc.)
        # This avoids bash syntax errors when test_name contains single quotes like "doesn't"
        env_lines.append("export APR_TEST_NAME=$(cat <<'EOF_APR_TN'\n" + test_name + "\nEOF_APR_TN\n)")
        if test_files:
            env_lines.append("export APR_TEST_FILES=$(cat <<'EOF_APR_TF'\n" + test_files + "\nEOF_APR_TF\n)")
        # Always pass instance id into the container script (for per-instance fixes).
        # Use heredoc to avoid bash syntax errors with special characters
        env_lines.append("export APR_INSTANCE_ID=$(cat <<'EOF_APR_IID'\n" + instance_id + "\nEOF_APR_IID\n)")
        # Mark Django project for special handling
        if is_django:
            env_lines.append("export APR_IS_DJANGO=1")
        # Mark Astropy project for special handling (warnings logging fix)
        if is_astropy:
            env_lines.append("export APR_IS_ASTROPY=1")
        if is_pytestdev:
            env_lines.append("export APR_IS_PYTESTDEV=1")
        if is_pylintdev:
            env_lines.append("export APR_IS_PYLINTDEV=1")
        if is_xarray:
            env_lines.append("export APR_IS_XARRAY=1")
        if is_pallets:
            env_lines.append("export APR_IS_PALLETS=1")
        if is_seaborn:
            env_lines.append("export APR_IS_SEABORN=1")
        if is_sphinx:
            env_lines.append("export APR_IS_SPHINX=1")
        if is_sympy:
            env_lines.append("export APR_IS_SYMPY=1")
        if is_matplotlib:
            env_lines.append("export APR_IS_MATPLOTLIB=1")
        if is_scikitlearn:
            env_lines.append("export APR_IS_SCIKITLEARN=1")
        # Pass required Python version to container script for version matching
        if required_python_version:
            # Use heredoc to avoid bash syntax errors (though version numbers are usually safe)
            env_lines.append("export APR_REQUIRED_PYTHON_VERSION=$(cat <<'EOF_APR_RPV'\n" + str(required_python_version) + "\nEOF_APR_RPV\n)")
        prefix = "\n".join(env_lines) + "\n"

        full_script = prefix + script

        # CRITICAL: Write script to temp file to avoid "Argument list too long" error
        # The entire script (env vars + script) is passed as a single argument to bash -lc
        # Writing to file and sourcing it avoids the argument length limit
        temp_dir = Path(bind.split(":")[0]) / ".apr_env" if ":" in bind else Path("/tmp") / f"apr_env_{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write large environment variables to files
        if _apr_temp_pip_pkgs:
            pip_file = temp_dir / "apr_pip_packages.txt"
            pip_file.write_text("\n".join(_apr_temp_pip_pkgs) + "\n")
        if _apr_temp_test_patch:
            patch_file = temp_dir / "apr_test_patch.txt"
            patch_file.write_text(_apr_temp_test_patch)
        
        # Write full script to file
        script_file = temp_dir / "apr_script.sh"
        script_file.write_text(full_script)
        script_file.chmod(0o755)
        
        # Add temp dir bind to existing bind string
        bind_with_temp = f"{bind},{temp_dir}:/testbed/.apr_env"

        print(f"[RUN_TEST] Executing test in Apptainer container...", flush=True)
        print(f"[RUN_TEST] Image: {image}", flush=True)
        print(f"[RUN_TEST] Test command: pytest -q -x -k '{test_name}' (or via test file directives)", flush=True)
        # Allow callers to override test timeout (e.g., for faster TRACE verification).
        # Default remains 1800s to preserve existing behavior unless explicitly set.
        timeout_s = 1800
        try:
            timeout_s = int(os.environ.get("APR_TRACE_TIMEOUT_SECONDS") or os.environ.get("APR_TEST_TIMEOUT_SECONDS") or "1800")
        except Exception:
            timeout_s = 1800
        print(f"[RUN_TEST] Timeout: {timeout_s} seconds", flush=True)
        print(f"[RUN_TEST] This may take a while - container needs to:", flush=True)
        print(f"[RUN_TEST]   1. Pull/load image (if not cached)", flush=True)
        print(f"[RUN_TEST]   2. Setup Python environment (conda testbed)", flush=True)
        print(f"[RUN_TEST]   3. Install dependencies (if needed)", flush=True)
        print(f"[RUN_TEST]   4. Apply test patch", flush=True)
        print(f"[RUN_TEST]   5. Run pytest for test: {test_name}", flush=True)
        
        r = _run_apptainer(
            image=image,
            argv=["bash", "-lc", "source /testbed/.apr_env/apr_script.sh"],
            bind=bind_with_temp,
            pwd="/testbed",
            timeout=timeout_s,
        )
        
        print(f"[RUN_TEST] Test execution completed: rc={r.get('rc', 'N/A')}, timeout={r.get('timeout', False)}", flush=True)

        # Write log file
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"Command (apptainer): pytest one ({test_name})\nImage: {image}\nBind: {bind}\nReturn code: {r['rc']}\n\n=== STDOUT ===\n{r.get('stdout','')}\n\n=== STDERR ===\n{r.get('stderr','')}\n",
            encoding="utf-8",
        )

        return {
            "ran": True,
            "rc": r["rc"],
            "test_name": test_name,
            "logfile": log_file,
            "stdout": r.get("stdout", ""),
            "stderr": r.get("stderr", ""),
            "cmd": "pytest one",
            "runtime": "apptainer",
        }
