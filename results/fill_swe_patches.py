#!/usr/bin/env python3
"""SWE jsonl: 打乱顺序（避免 patch 空成片）、去掉占位符，只保留真实 patch。"""
import json
import hashlib
import re
from pathlib import Path


def is_our_placeholder(patch: str) -> bool:
    if not patch or "diff --git" not in patch:
        return True
    if "index " in patch:
        m = re.search(r"index ([0-9a-f]+)\.\.([0-9a-f]+)", patch)
        if m and len(m.group(1)) >= 8 and len(m.group(2)) >= 8:
            return False
    return True


def shuffle_key(rec: dict) -> tuple:
    pid = rec.get("pid") or ""
    h = hashlib.md5(pid.encode()).hexdigest()
    return (int(h[:8], 16), pid)


def main():
    base = Path(__file__).parent
    swe_dir = base / "swe" if (base / "swe").exists() else base
    for jf in swe_dir.glob("swe_*.jsonl"):
        with open(jf, "r", encoding="utf-8") as f:
            recs = [json.loads(L) for L in f if L.strip()]
        cleared = 0
        for rec in recs:
            patch = (rec.get("patch") or "").strip()
            if patch and is_our_placeholder(patch):
                rec["patch"] = ""
                cleared += 1
        recs.sort(key=shuffle_key)
        with open(jf, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{jf.name}: cleared {cleared} placeholders, shuffled by hash(pid)")


if __name__ == "__main__":
    main()
