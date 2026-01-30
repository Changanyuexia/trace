"""
Dataset adapter interface.

Goal: decouple core ablation logic from a specific benchmark (e.g., Defects4J).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class DatasetAdapter(ABC):
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        pid: str,
        bid: int,
        workdir: str,
        meta_dir: str,
        full_log: str,
        trig_log: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def check_compile(self, workdir: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_one_test(self, workdir: str, test_name: str, log_file: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def checkout(self, pid: str, bid: int, workdir: str) -> Dict[str, Any]:
        raise NotImplementedError




