"""BaseStage: the contract every phase follows (Section 12.1 of the guide).

input artifacts in -> run() -> one saved artifact -> evaluate() -> gate() ->
metrics JSON saved next to the artifact. Results are cached by config hash.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from src.common.utils import config_hash


class BaseStage:

    name = "base"

    requires: list[str] = []


    def __init__(
        self,
        cfg: dict,
        video_id: str,
        logger=None
    ):
        self.cfg = cfg
        self.video_id = video_id
        self.log = logger


    @property
    def artifact(self) -> Path:
        """Path of the main artifact this stage writes."""
        raise NotImplementedError


    def run(self) -> None:
        raise NotImplementedError


    def evaluate(self, gt=None) -> dict:
        raise NotImplementedError


    def gate(self, metrics: dict) -> tuple[bool, str]:
        raise NotImplementedError


    @property
    def metrics_path(self) -> Path:
        return (
            self.artifact.parent
            / f"{self.video_id}.metrics.json"
        )


    def cache_key(self) -> str:
        return config_hash(self.cfg)


    def is_cached(self) -> bool:
        """True if artifact + metrics exist,
        the config is unchanged and the gate passed."""

        if not (
            self.artifact.exists()
            and self.metrics_path.exists()
        ):
            return False

        try:
            saved = json.loads(
                self.metrics_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError:
            return False

        return (
            saved.get("cfg") == self.cache_key()
            and saved.get("passed") is True
        )


    def execute(
        self,
        gt=None,
        force: bool = False
    ) -> bool:

        if not force and self.is_cached():

            if self.log:
                self.log.info(
                    "[%s] cached (config unchanged), skipping",
                    self.name
                )

            return True

        t0 = time.time()

        self.run()

        metrics = self.evaluate(gt)

        ok, reason = self.gate(metrics)

        metrics.update(
            stage=self.name,
            video_id=self.video_id,
            runtime_s=round(
                time.time() - t0,
                3
            ),
            passed=ok,
            reason=reason,
            cfg=self.cache_key(),
            python=platform.python_version(),
        )

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )

        if self.log:
            self.log.info(
                "[%s] gate %s: %s",
                self.name,
                "PASSED" if ok else "FAILED",
                reason
            )

        return ok