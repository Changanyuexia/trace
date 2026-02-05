"""
Retrieval-index tools (replacement for ABCoder/LSP-based indexing).

Implements the same tool API used by G2/TRACE localization:
- symbol_lookup(index_path, symbol)
- find_references(index_path, symbol)
- read_span(path, start_line, end_line, workdir)

But changes the index building to a stable, reproducible, non-LSP pipeline:
- Layer 1: structural index (Tree-sitter if usable; otherwise regex+brace fallback for Java)
- Layer 2: (optional) CodeQL hooks can be added later to populate precise refs/edges

Index schema follows the current retrieval index JSON format written to runs/index/....
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Utilities
# ----------------------------

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _iter_code_files(workdir: Path, roots: List[str], exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for r in roots:
        rp = workdir / r
        if not rp.exists():
            continue
        for ext in exts:
            files.extend(rp.rglob(f"*{ext}"))
    # Deterministic order
    files = [f for f in files if f.is_file()]
    files.sort(key=lambda p: str(p))
    return files


def _safe_relpath(workdir: Path, p: Path) -> str:
    try:
        return str(p.relative_to(workdir))
    except Exception:
        return str(p)


def _score_symbol(query: str, cand: str) -> int:
    """
    Higher is better.
    Prefer exact match, then suffix match, then substring match.
    """
    q = (query or "").strip()
    # Normalize common query formats used by prompts/LLMs
    q = q.replace("::", ".").replace("#", ".")
    q = re.sub(r"\(.*\)$", "", q)  # drop trailing signature if present
    q = q.replace("..<init>", ".<init>")  # guard against double dots after normalization

    # Normalize constructor queries:
    # - Foo.Foo -> Foo.<init>
    # - pkg.Foo.Foo -> pkg.Foo.<init>
    parts = [p for p in q.split(".") if p]
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        parts[-1] = "<init>"
        q = ".".join(parts)
    c = (cand or "").strip()
    if not q or not c:
        return 0
    if q == c:
        return 1000
    if c.endswith(q):
        return 800 + min(100, len(q))
    if q.endswith(c):
        return 500 + min(50, len(c))
    if q in c:
        return 300 + min(50, len(q))
    # try last segment match (method name, class name)
    q_last = q.split(".")[-1].split("::")[-1]
    c_last = c.split(".")[-1].split("::")[-1]
    if q_last and q_last == c_last:
        return 250
    return 0


# ----------------------------
# Java structural index (regex+brace fallback)
# ----------------------------

JAVA_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "throw", "new",
    "synchronized", "do", "try", "else", "case", "default",
}


@dataclass
class _ClassCtx:
    name: str
    brace_depth_enter: int


def _strip_comments_and_strings(line: str, state: Dict[str, Any]) -> str:
    """
    Very small lexer to make brace counting and signature detection less wrong.
    state contains: in_block_comment, in_string, in_char
    """
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""

        if state.get("in_block_comment"):
            if ch == "*" and nxt == "/":
                state["in_block_comment"] = False
                i += 2
                continue
            i += 1
            continue

        if state.get("in_string"):
            if ch == "\\":
                i += 2
                continue
            if ch == "\"":
                state["in_string"] = False
            i += 1
            continue

        if state.get("in_char"):
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state["in_char"] = False
            i += 1
            continue

        # entering comment/string/char
        if ch == "/" and nxt == "/":
            break
        if ch == "/" and nxt == "*":
            state["in_block_comment"] = True
            i += 2
            continue
        if ch == "\"":
            state["in_string"] = True
            i += 1
            continue
        if ch == "'":
            state["in_char"] = True
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _find_block_end(lines: List[str], start_idx: int) -> int:
    """
    Find end line (1-based) for the block that starts with an opening '{'
    somewhere on lines[start_idx:].
    """
    state = {"in_block_comment": False, "in_string": False, "in_char": False}
    depth = 0
    started = False
    for i in range(start_idx, len(lines)):
        s = _strip_comments_and_strings(lines[i], state)
        for ch in s:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                if started:
                    depth -= 1
                    if depth == 0:
                        return i + 1  # 1-based
    return len(lines)


_PACKAGE_RE = re.compile(r"^\s*package\s+([a-zA-Z0-9_.]+)\s*;")
_CLASS_RE = re.compile(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def _looks_like_method_decl(line: str) -> bool:
    # quick filters
    if "(" not in line or ")" not in line or "{" not in line:
        return False
    s = line.strip()
    if not s:
        return False
    if s.startswith("@"):
        return False
    head = s.split(None, 1)[0]
    if head in JAVA_KEYWORDS:
        return False
    if s.startswith(("if", "for", "while", "switch", "catch", "try")):
        return False
    if s.endswith(";"):
        return False
    return True


def _extract_method_name(line: str) -> Optional[str]:
    # Remove generics noise a bit
    # Try to capture "... name ( ... ) {"
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
    if not m:
        return None
    name = m.group(1)
    if name in JAVA_KEYWORDS:
        return None
    return name


def _build_java_struct_index(workdir: Path) -> Dict[str, Any]:
    roots = ["src/main/java", "src/test/java"]
    files = _iter_code_files(workdir, roots, (".java",))

    defs: List[Dict[str, Any]] = []
    fallback_calls: List[Dict[str, Any]] = []

    for fp in files:
        rel = _safe_relpath(workdir, fp)
        content = _read_text(fp)
        lines = content.splitlines()

        pkg = ""
        for ln in lines[:50]:
            m = _PACKAGE_RE.match(ln)
            if m:
                pkg = m.group(1)
                break

        # Walk linearly tracking class nesting by brace depth heuristics
        brace_state = {"in_block_comment": False, "in_string": False, "in_char": False}
        brace_depth = 0
        class_stack: List[_ClassCtx] = []
        pending_class: Optional[str] = None
        pending_class_depth: Optional[int] = None

        # Multi-line signature buffering for methods/constructors
        sig_buf: List[str] = []
        sig_start_line: Optional[int] = None

        def flush_sig():
            nonlocal sig_buf, sig_start_line
            sig_buf = []
            sig_start_line = None

        def current_class_name() -> Optional[str]:
            return class_stack[-1].name if class_stack else None

        def make_class_fqn() -> str:
            return "$".join([c.name for c in class_stack]) if class_stack else ""

        def is_control_like(sig: str) -> bool:
            s0 = sig.strip()
            if not s0:
                return True
            head = s0.split(None, 1)[0]
            if head in JAVA_KEYWORDS:
                return True
            if s0.startswith(("if", "for", "while", "switch", "catch", "try", "do")):
                return True
            return False

        def extract_callable_name(sig: str) -> Optional[str]:
            # Find the identifier immediately before the first '(' in the signature
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", sig)
            if not m:
                return None
            name = m.group(1)
            if name in JAVA_KEYWORDS:
                return None
            return name

        def signature_has_body(sig: str) -> bool:
            # A declaration ends with '{' for body; abstract/interface often ends with ';'
            return "{" in sig

        def signature_is_terminated(sig: str) -> bool:
            # If body starts or terminates with ';' we can flush
            s = sig.strip()
            return ("{" in s) or s.endswith(";")

        i = 0
        while i < len(lines):
            raw = lines[i]
            s = _strip_comments_and_strings(raw, brace_state)

            # detect class declarations
            m = _CLASS_RE.search(s)
            if m:
                cname = m.group(2)
                pending_class = cname
                pending_class_depth = brace_depth

            # update brace depth char-by-char to place class enter/exit
            for ch in s:
                if ch == "{":
                    brace_depth += 1
                    # enter pending class at the first '{' after decl
                    if pending_class is not None and pending_class_depth is not None and brace_depth == pending_class_depth + 1:
                        class_stack.append(_ClassCtx(name=pending_class, brace_depth_enter=brace_depth))
                        # record class def span (best-effort)
                        class_fqn = "$".join([c.name for c in class_stack])
                        sym = f"{pkg}.{class_fqn}" if pkg else class_fqn
                        start_line = i + 1
                        end_line = _find_block_end(lines, i)
                        defs.append({
                            "symbol": sym,
                            "kind": "class",
                            "path": rel,
                            "start": start_line,
                            "end": end_line,
                            "sig": m.group(1) if m else None,
                        })
                        pending_class = None
                        pending_class_depth = None
                elif ch == "}":
                    # exit class scopes if needed
                    if class_stack and brace_depth == class_stack[-1].brace_depth_enter:
                        class_stack.pop()
                    brace_depth = max(0, brace_depth - 1)

            # detect method declarations (best-effort)
            if class_stack:
                # Start buffering signature if we see '(' but haven't started
                if sig_start_line is None:
                    # Avoid starting on obvious control statements
                    if "(" in s and not is_control_like(s) and not s.strip().startswith("@"):
                        sig_buf = [s.strip()]
                        sig_start_line = i + 1
                else:
                    # Continue buffering
                    if s.strip():
                        sig_buf.append(s.strip())

                if sig_start_line is not None:
                    sig_joined = " ".join(sig_buf)
                    # stop buffering if it grows too much
                    if len(sig_buf) > 12:
                        flush_sig()
                    elif signature_is_terminated(sig_joined):
                        # Decide if this is a method/constructor
                        name = extract_callable_name(sig_joined)
                        if name and not is_control_like(sig_joined):
                            cls_name = current_class_name()
                            class_fqn = make_class_fqn()
                            is_ctor = (cls_name is not None and name == cls_name)
                            kind = "method"
                            member = name
                            if is_ctor:
                                kind = "constructor"
                                member = "<init>"
                            sym = f"{pkg}.{class_fqn}.{member}" if pkg else f"{class_fqn}.{member}"

                            start_line = sig_start_line
                            # If no body, treat as single-line span
                            if signature_has_body(sig_joined):
                                end_line = _find_block_end(lines, i if "{" in s else (sig_start_line - 1))
                            else:
                                end_line = start_line

                            defs.append({
                                "symbol": sym,
                                "kind": kind,
                                "path": rel,
                                "start": start_line,
                                "end": end_line,
                                "sig": sig_joined[:200],
                            })

                            # gather weak calls only for bodies
                            if end_line > start_line:
                                body_end = min(end_line, len(lines))
                                for j in range(start_line, body_end):
                                    ln2 = lines[j - 1]
                                    s2 = ln2.strip()
                                    if not s2 or s2.startswith("@"):
                                        continue
                                    for cm in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", s2):
                                        callee = cm.group(1)
                                        if callee in JAVA_KEYWORDS:
                                            continue
                                        fallback_calls.append({
                                            "text": callee,
                                            "path": rel,
                                            "line": j,
                                            "col": cm.start(1) + 1,
                                        })

                                # jump to end of block
                                i = end_line - 1
                        flush_sig()
            i += 1

    return {"defs": defs, "fallback_calls": fallback_calls}


# ----------------------------
# Public API: build + tools
# ----------------------------

def build_retrieval_index(
    workdir: str,
    out_path: str,
    benchmark: str = "defects4j",
    project: Optional[str] = None,
    revision: Optional[str] = None,
    language: str = "java",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Build stable retrieval index JSON.
    Currently implements Java Layer-1 structural indexing (regex+brace fallback).
    """
    wd = Path(workdir).resolve()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        return {"ok": True, "out_path": str(out), "cached": True}

    t0 = time.time()
    if language.lower() == "java":
        layer1 = _build_java_struct_index(wd)
    else:
        return {"ok": False, "error": f"language not supported yet: {language}", "out_path": str(out)}

    index_obj = {
        "repo": str(wd),
        "benchmark": benchmark,
        "project": project or "",
        "revision": revision or "",
        "language": language.lower(),
        "generated_at": _now_iso(),
        "index_version": "v1",
        "defs": layer1.get("defs", []),
        "refs": [],   # reserved for CodeQL layer
        "edges": [],  # reserved for CodeQL layer
        "fallback_calls": layer1.get("fallback_calls", []),
    }

    out.write_text(json.dumps(index_obj, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "out_path": str(out), "cached": False, "duration_s": round(time.time() - t0, 3)}


def _codeql_available() -> bool:
    import shutil
    return shutil.which("codeql") is not None


def _run_codeql(cmd: List[str], cwd: Optional[str] = None, timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    import subprocess
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired as e:
        return {"rc": -1, "stdout": e.stdout or "", "stderr": f"timeout after {timeout}s", "timeout": True}
    except Exception as e:
        return {"rc": -1, "stdout": "", "stderr": str(e), "exception": True}


def _ensure_codeql_db_defects4j(workdir: Path, db_dir: Path) -> Dict[str, Any]:
    """
    Create CodeQL DB for Defects4J checkout if missing.
    We use defects4j compile as build command to let CodeQL extract Java.
    """
    db_dir.parent.mkdir(parents=True, exist_ok=True)
    if (db_dir / "codeql-database.yml").exists():
        return {"ok": True, "cached": True}
    if not _codeql_available():
        return {"ok": False, "skip": True, "reason": "codeql not found"}

    # Ensure build command can find defects4j even when PATH isn't set by a wrapper script.
    import os
    import shutil
    defects4j_bin = shutil.which("defects4j")
    if not defects4j_bin:
        d4j_home = os.environ.get("DEFECTS4J_HOME", "")
        if d4j_home:
            cand = Path(d4j_home) / "framework" / "bin" / "defects4j"
            if cand.exists():
                defects4j_bin = str(cand)
    if not defects4j_bin:
        return {"ok": False, "reason": "defects4j not found in PATH and DEFECTS4J_HOME not set or not usable"}

    # Build a robust build command that works even under CodeQL's tracer environment.
    # We wrap it in `bash -lc` and explicitly set required env vars.
    import shlex
    java11_path = "/usr/lib/jvm/java-11-openjdk-11.0.25.0.9-7.el9.x86_64"
    d4j_home = os.environ.get("DEFECTS4J_HOME", "")

    # PERL5LIB (DBI.pm + defects4j framework libs); use PERL5_DIR env
    perl5lib_parts = []
    perl5_dir = os.environ.get("PERL5_DIR", "")
    if perl5_dir and os.path.exists(perl5_dir):
        perl5_lib = os.path.join(perl5_dir, "lib", "perl5")
        if os.path.exists(perl5_lib):
            try:
                for arch_dir in os.listdir(perl5_lib):
                    arch_path = os.path.join(perl5_lib, arch_dir)
                    if os.path.isdir(arch_path):
                        perl5lib_parts.append(arch_path)
            except Exception:
                pass
            perl5lib_parts.append(perl5_lib)
    for p in (Path(d4j_home) / "framework" / "lib", Path(d4j_home) / "framework" / "core", Path(d4j_home) / "framework" / "util"):
        if p.exists():
            perl5lib_parts.append(str(p))
    perl5lib = ":".join(perl5lib_parts) if perl5lib_parts else ""

    exports = []
    exports.append(f"export TZ=America/Los_Angeles")

    # Avoid CodeQL's own variable-expansion rules: do not reference $PATH/${PATH} in --command.
    base_path = os.environ.get("PATH", "")
    path_parts: List[str] = []
    if Path(java11_path).exists():
        exports.append(f"export JAVA_HOME={shlex.quote(java11_path)}")
        path_parts.append(str(Path(java11_path) / "bin"))
    if Path(d4j_home).exists():
        exports.append(f"export DEFECTS4J_HOME={shlex.quote(d4j_home)}")
        path_parts.append(str(Path(d4j_home) / "framework" / "bin"))
    if base_path:
        path_parts.append(base_path)
    if path_parts:
        exports.append(f"export PATH={shlex.quote(':'.join(path_parts))}")
    if perl5lib:
        exports.append(f"export PERL5LIB={shlex.quote(perl5lib)}")

    compile_cmd = f"{' ; '.join(exports)} ; {shlex.quote(defects4j_bin)} compile -w {shlex.quote(str(workdir))}"
    # IMPORTANT: CodeQL's --command parsing does not treat single quotes like a shell.
    # Use double quotes for grouping so bash receives the full script as one argument.
    wrapped = f"bash -lc \"{compile_cmd}\""

    cmd = [
        "codeql", "database", "create", str(db_dir),
        "--language=java",
        f"--source-root={str(workdir)}",
        "--command", wrapped,
    ]
    r = _run_codeql(cmd, cwd=str(workdir), timeout=3600)
    return {"ok": r.get("rc") == 0, **r}


def _decode_bqrs_json(bqrs_path: Path) -> Dict[str, Any]:
    if not _codeql_available():
        return {"ok": False, "skip": True, "reason": "codeql not found"}
    out_json = bqrs_path.with_suffix(".json")
    r = _run_codeql(["codeql", "bqrs", "decode", "--format=json", str(bqrs_path), "--output", str(out_json)], cwd=str(bqrs_path.parent), timeout=600)
    if r.get("rc") != 0:
        return {"ok": False, "error": "bqrs decode failed", **r}
    try:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": f"failed to parse bqrs json: {e}"}


def augment_index_with_codeql(
    *,
    workdir: str,
    index_path: str,
    codeql_db_dir: str,
    queries_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Populate `refs` and `edges` in an existing index.json using CodeQL.
    If CodeQL is unavailable or fails, return ok=False with skip/error, but do not break Layer1 usage.
    """
    wd = Path(workdir).resolve()
    idx = Path(index_path)
    db_dir = Path(codeql_db_dir).resolve()

    if not idx.exists():
        return {"ok": False, "error": f"index not found: {index_path}"}
    if not _codeql_available():
        return {"ok": False, "skip": True, "reason": "codeql not found"}

    if queries_dir is None:
        queries_dir = str(Path(__file__).resolve().parent / "codeql_queries" / "java")
    qdir = Path(queries_dir).resolve()
    refs_q = qdir / "refs.ql"
    edges_q = qdir / "edges.ql"
    if not refs_q.exists() or not edges_q.exists():
        return {"ok": False, "error": f"missing CodeQL queries under {qdir}"}

    db_r = _ensure_codeql_db_defects4j(wd, db_dir)
    if not db_r.get("ok"):
        # Hard failure or skip, but don't break outer flow
        return {"ok": False, **db_r}

    # Check if database is finalized, and finalize if needed
    import os
    import yaml
    db_yml = db_dir / "codeql-database.yml"
    if db_yml.exists():
        try:
            with open(db_yml, "r") as f:
                db_info = yaml.safe_load(f)
            is_finalized = db_info.get("finalised", False)
            
            if not is_finalized:
                # Database exists but not finalized, try to finalize it
                source_prefix = db_info.get("sourceLocationPrefix", "")
                # Try original workdir first, then fallback to provided workdir if it's the same project
                finalize_workdir = None
                if source_prefix and Path(source_prefix).exists():
                    finalize_workdir = source_prefix
                elif wd.exists():
                    # Use provided workdir as fallback (might work if it's the same checkout)
                    finalize_workdir = str(wd)
                
                if finalize_workdir:
                    # Workdir exists, can finalize
                    finalize_r = _run_codeql(
                        ["codeql", "database", "finalize", str(db_dir)],
                        cwd=finalize_workdir,
                        timeout=600,
                        env=os.environ.copy()
                    )
                    if finalize_r.get("rc") != 0:
                        return {
                            "ok": False,
                            "reason": f"database not finalized and finalize failed: {finalize_r.get('stderr', 'unknown')[:200]}",
                            **finalize_r
                        }
                    # Finalize succeeded, continue to queries
                else:
                    # Workdir doesn't exist, cannot finalize
                    return {
                        "ok": False,
                        "reason": f"database not finalized and workdir missing (sourceLocationPrefix: {source_prefix}, provided workdir: {wd})",
                        "error": "database needs to be rebuilt"
                    }
        except Exception as e:
            # If we can't parse the yml, continue anyway (might be a different format)
            pass

    out_dir = db_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure standard packs are available.
    # The CLI distribution does not necessarily include query packs; use pack install,
    # but redirect HOME to workspace so we don't write to the user's real $HOME.
    import os
    apr_root = Path(__file__).resolve().parents[1]
    codeql_home = apr_root / "runs" / "codeql_home"
    codeql_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(codeql_home)

    pi = _run_codeql(["codeql", "pack", "install"], cwd=str(qdir), timeout=1200, env=env)
    if pi.get("rc") != 0:
        return {"ok": False, "reason": "codeql pack install failed", **pi}

    # Run queries
    refs_bqrs = out_dir / "refs.bqrs"
    edges_bqrs = out_dir / "edges.bqrs"

    r1 = _run_codeql(["codeql", "query", "run", "--database", str(db_dir), str(refs_q), "--output", str(refs_bqrs)], cwd=str(out_dir), timeout=1800, env=env)
    if r1.get("rc") != 0:
        error_msg = r1.get("stderr", "")[:500] if r1.get("stderr") else "unknown error"
        return {"ok": False, "reason": f"codeql refs query failed: {error_msg}", **r1}
    r2 = _run_codeql(["codeql", "query", "run", "--database", str(db_dir), str(edges_q), "--output", str(edges_bqrs)], cwd=str(out_dir), timeout=1800, env=env)
    if r2.get("rc") != 0:
        error_msg = r2.get("stderr", "")[:500] if r2.get("stderr") else "unknown error"
        return {"ok": False, "reason": f"codeql edges query failed: {error_msg}", **r2}

    refs_dec = _decode_bqrs_json(refs_bqrs)
    if not refs_dec.get("ok"):
        return {"ok": False, **refs_dec}
    edges_dec = _decode_bqrs_json(edges_bqrs)
    if not edges_dec.get("ok"):
        return {"ok": False, **edges_dec}

    def rows_from_bqrs(data: Dict[str, Any]) -> List[List[Any]]:
        # CodeQL JSON format varies; handle the common structure:
        # { "tuples": [ [..], ... ] } or { "#select": { "tuples": ... } } or { "tables": [ { "rows": ... } ] }
        if isinstance(data, dict):
            if "#select" in data and isinstance(data["#select"], dict):
                data = data["#select"]
            if "tuples" in data and isinstance(data["tuples"], list):
                return data["tuples"]
            if "tables" in data and data["tables"]:
                t0 = data["tables"][0]
                if isinstance(t0, dict) and "rows" in t0:
                    return t0["rows"]
        return []

    refs_rows = rows_from_bqrs(refs_dec["data"])
    edges_rows = rows_from_bqrs(edges_dec["data"])

    # Merge into index.json
    obj = json.loads(idx.read_text(encoding="utf-8"))
    obj.setdefault("refs", [])
    obj.setdefault("edges", [])

    merged_refs = []
    for row in refs_rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        sym, kind, path, line, col = row[:5]
        merged_refs.append({
            "symbol": sym,
            "kind": kind,
            "path": path,
            "line": int(line),
            "col": int(col),
        })

    merged_edges = []
    for row in edges_rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        caller, callee, path, line = row[:4]
        merged_edges.append({
            "caller": caller,
            "callee": callee,
            "path": path,
            "line": int(line),
        })

    obj["refs"] = merged_refs
    obj["edges"] = merged_edges
    obj["generated_at"] = _now_iso()
    obj["index_version"] = "v1+codeql"

    idx.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "refs": len(merged_refs), "edges": len(merged_edges), "db_dir": str(db_dir)}


def symbol_lookup(index_path: str, symbol: str, max_candidates: int = 10) -> Dict[str, Any]:
    """
    Look up symbol definitions in ABCoder format index.
    
    Only supports ABCoder format (Graph/Modules structure).
    """
    p = Path(index_path)
    if not p.exists():
        return {"ok": False, "error": f"index not found: {index_path}"}
    obj = json.loads(p.read_text(encoding="utf-8"))
    
    # Only support ABCoder format
    if "Graph" not in obj and "Modules" not in obj:
        return {"ok": False, "error": f"Unsupported index format. Only ABCoder format (Graph/Modules) is supported."}
    
    # ABCoder format - parse directly
    return _symbol_lookup_abcoder(obj, symbol, max_candidates)


def _symbol_lookup_abcoder(abcoder_obj: Dict[str, Any], symbol: str, max_candidates: int = 10) -> Dict[str, Any]:
    """
    Look up symbol in ABCoder format index.
    
    ABCoder structure:
    - Graph: { "mod?pkg#symbol": { "Type": "FUNC|TYPE", "Name": "...", "PkgPath": "...", ... } }
    - Modules: { "mod": { "Packages": { "pkg": { "Functions": {...}, "Types": {...} } } } }
    """
    graph = abcoder_obj.get("Graph", {}) or {}
    modules = abcoder_obj.get("Modules", {}) or {}
    
    scored = []
    
    # Search in Graph nodes
    for node_key, node_data in graph.items():
        if not isinstance(node_data, dict):
            continue
        
        # Extract symbol name from node
        node_name = node_data.get("Name", "")
        pkg_path = node_data.get("PkgPath", "")
        
        # Build full symbol: pkg.Class.method or pkg.symbol
        if pkg_path and node_name:
            # Parse node_key to get symbol: "mod?pkg#symbol"
            if "?" in node_key and "#" in node_key:
                symbol_part = node_key.split("#")[-1]
                if "." in symbol_part:
                    # Method: Class.method()
                    class_method = symbol_part
                    full_symbol = f"{pkg_path}.{class_method}"
                else:
                    # Type or variable
                    full_symbol = f"{pkg_path}.{symbol_part}"
            else:
                full_symbol = f"{pkg_path}.{node_name}" if pkg_path else node_name
        else:
            full_symbol = node_name
        
        # Score match
        score = _score_symbol(symbol, full_symbol)
        if score > 0:
            # Get file and line info from Modules if available
            file_path = ""
            start_line = 0
            end_line = 0
            sig = ""
            kind = "unknown"
            
            node_type = node_data.get("Type", "")
            if node_type == "FUNC":
                kind = "method"
            elif node_type == "TYPE":
                kind = "class"
            elif node_type == "VAR":
                kind = "variable"
            
            # Try to find in Modules for file/line info
            for mod_name, mod_data in modules.items():
                if not isinstance(mod_data, dict):
                    continue
                packages = mod_data.get("Packages", {}) or {}
                for pkg_name, pkg_data in packages.items():
                    if not isinstance(pkg_data, dict):
                        continue
                    
                    # Check Functions
                    functions = pkg_data.get("Functions", {}) or {}
                    for func_name, func_info in functions.items():
                        if isinstance(func_info, dict) and func_info.get("Name") == node_name:
                            file_path = func_info.get("File", "")
                            start_line = func_info.get("Line", 0)
                            sig = func_info.get("Signature", "")
                            content = func_info.get("Content", "")
                            if content:
                                end_line = start_line + len(content.splitlines()) - 1
                            else:
                                end_line = start_line
                            break
                    
                    # Check Types
                    types = pkg_data.get("Types", {}) or {}
                    for type_name, type_info in types.items():
                        if isinstance(type_info, dict) and type_info.get("Name") == node_name:
                            file_path = type_info.get("File", "")
                            start_line = type_info.get("Line", 0)
                            content = type_info.get("Content", "")
                            if content:
                                end_line = start_line + len(content.splitlines()) - 1
                            else:
                                end_line = start_line
                            break
            
            scored.append((score, {
                "symbol": full_symbol,
                "kind": kind,
                "path": file_path,
                "start": start_line,
                "end": end_line,
                "sig": sig,
            }))
    
    scored.sort(key=lambda t: (-t[0], t[1].get("path", ""), t[1].get("start", 0)))
    
    hits = []
    for score, d in scored[: max_candidates or 10]:
        hits.append({
            "symbol": d.get("symbol"),
            "kind": d.get("kind"),
            "path": d.get("path"),
            "start_line": d.get("start"),
            "end_line": d.get("end"),
            "score": score,
            "sig": d.get("sig"),
        })
    
    return {"ok": True, "query": symbol, "hits": hits, "engine": "abcoder"}


def find_references(index_path: str, symbol: str) -> Dict[str, Any]:
    """
    Find references to a symbol in ABCoder format index.
    
    Only supports ABCoder format (Graph/Modules structure).
    """
    p = Path(index_path)
    if not p.exists():
        return {"ok": False, "error": f"index not found: {index_path}"}
    obj = json.loads(p.read_text(encoding="utf-8"))
    
    # Only support ABCoder format
    if "Graph" not in obj and "Modules" not in obj:
        return {"ok": False, "error": f"Unsupported index format. Only ABCoder format (Graph/Modules) is supported."}
    
    # ABCoder format - parse directly
    return _find_references_abcoder(obj, symbol)


def _find_references_abcoder(abcoder_obj: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Find references to a symbol in ABCoder format index.
    
    ABCoder Graph nodes have "References" and "Dependencies" fields that contain
    references to other symbols.
    """
    graph = abcoder_obj.get("Graph", {}) or {}
    modules = abcoder_obj.get("Modules", {}) or {}
    
    # Extract symbol name (last part after .)
    symbol_name = symbol.split(".")[-1].split("(")[0]  # Remove method params
    
    # Build a cache of node_key -> file/line info from Modules
    node_info_cache: Dict[str, Dict[str, Any]] = {}
    for mod_name, mod_data in modules.items():
        if not isinstance(mod_data, dict):
            continue
        packages = mod_data.get("Packages", {}) or {}
        for pkg_name, pkg_data in packages.items():
            if not isinstance(pkg_data, dict):
                continue
            
            # Process Functions
            functions = pkg_data.get("Functions", {}) or {}
            for func_name, func_info in functions.items():
                if isinstance(func_info, dict):
                    node_name = func_info.get("Name", "")
                    if node_name:
                        # Build node key: "mod?pkg#symbol"
                        node_key = f"{mod_name}?{pkg_name}#{node_name}"
                        node_info_cache[node_key] = {
                            "file": func_info.get("File", ""),
                            "line": func_info.get("Line", 0),
                            "pkg": pkg_name,
                            "name": node_name,
                        }
            
            # Process Types
            types = pkg_data.get("Types", {}) or {}
            for type_name, type_info in types.items():
                if isinstance(type_info, dict):
                    node_name = type_info.get("Name", "")
                    if node_name:
                        node_key = f"{mod_name}?{pkg_name}#{node_name}"
                        node_info_cache[node_key] = {
                            "file": type_info.get("File", ""),
                            "line": type_info.get("Line", 0),
                            "pkg": pkg_name,
                            "name": node_name,
                        }
    
    hits = []
    seen_refs = set()  # Avoid duplicates
    
    # Find all nodes that reference the target symbol
    for node_key, node_data in graph.items():
        if not isinstance(node_data, dict):
            continue
        
        # Get caller info
        caller_info = node_info_cache.get(node_key, {})
        caller_file = caller_info.get("file", "")
        caller_line = caller_info.get("line", 0)
        caller_pkg = node_data.get("PkgPath", caller_info.get("pkg", ""))
        caller_name = node_data.get("Name", caller_info.get("name", ""))
        
        # Check References field
        references = node_data.get("References", []) or []
        for ref in references:
            if isinstance(ref, dict):
                ref_name = ref.get("Name", "")
                ref_pkg = ref.get("PkgPath", "")
                
                # Build full reference symbol for matching
                if ref_pkg and ref_name:
                    ref_symbol = f"{ref_pkg}.{ref_name}"
                else:
                    ref_symbol = ref_name
                
                # Check if this reference matches our target symbol
                if (symbol in ref_symbol or 
                    symbol_name in ref_name or 
                    ref_name == symbol_name or
                    ref_symbol == symbol):
                    ref_key = f"{caller_file}:{caller_line}:{symbol}"
                    if ref_key not in seen_refs:
                        seen_refs.add(ref_key)
                        hits.append({
                            "symbol": symbol,
                            "path": caller_file,
                            "line": caller_line,
                            "caller": f"{caller_pkg}.{caller_name}" if caller_pkg else caller_name,
                        })
        
        # Check Dependencies field
        dependencies = node_data.get("Dependencies", []) or []
        for dep in dependencies:
            if isinstance(dep, dict):
                dep_name = dep.get("Name", "")
                dep_pkg = dep.get("PkgPath", "")
                
                if dep_pkg and dep_name:
                    dep_symbol = f"{dep_pkg}.{dep_name}"
                else:
                    dep_symbol = dep_name
                
                if (symbol in dep_symbol or 
                    symbol_name in dep_name or 
                    dep_name == symbol_name or
                    dep_symbol == symbol):
                    ref_key = f"{caller_file}:{caller_line}:{symbol}"
                    if ref_key not in seen_refs:
                        seen_refs.add(ref_key)
                        hits.append({
                            "symbol": symbol,
                            "path": caller_file,
                            "line": caller_line,
                            "caller": f"{caller_pkg}.{caller_name}" if caller_pkg else caller_name,
                        })
    
    return {"ok": True, "query": symbol, "engine": "abcoder", "hits": hits[:200]}


def read_span(path: str, start_line: int, end_line: int, workdir: str) -> Dict[str, Any]:
    """
    Read code span with line numbers. Path can be relative to workdir or absolute.
    """
    fp = Path(path)
    if not fp.is_absolute():
        fp = Path(workdir) / path
    if not fp.exists() or fp.is_dir():
        return {"ok": False, "error": f"file not found: {path}"}

    start = max(1, int(start_line))
    end = max(start, int(end_line))

    lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    out_lines = []
    for i in range(start, min(end, len(lines)) + 1):
        out_lines.append(f"{i:4d}: {lines[i-1]}")
    return {"ok": True, "path": str(fp), "start_line": start, "end_line": min(end, len(lines)), "content": "\n".join(out_lines)}


