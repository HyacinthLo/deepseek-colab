"""Append-only local CU billing ledger (usage.jsonl) for the Control Console and CLI."""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class UsageLedger:
    """Thread-safe JSONL session billing ledger stored in state/usage.jsonl."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(
        self,
        *,
        session: str,
        started_ts: float,
        stopped_ts: float | None = None,
        reason: str,
        ran_s: float,
        cu_est: float,
        balance_start: float | None = None,
        balance_end: float | None = None,
        model: str = "",
        basalt_ref: str = "",
        served: int = 0,
    ) -> dict[str, Any]:
        stopped_ts = stopped_ts if stopped_ts is not None else time.time()
        entry = {
            "session": session,
            "started_ts": round(started_ts, 1),
            "stopped_ts": round(stopped_ts, 1),
            "date": time.strftime("%Y-%m-%d", time.localtime(started_ts)),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_ts)),
            "reason": reason,
            "ran_s": round(max(0.0, ran_s), 1),
            "cu_est": round(max(0.0, cu_est), 3),
            "balance_start": round(balance_start, 2) if balance_start is not None else None,
            "balance_end": round(balance_end, 2) if balance_end is not None else None,
            "model": model,
            "basalt_ref": basalt_ref or "pinned",
            "served": int(served),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        with self._lock:
            try:
                raw_lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "cu_est" in obj:
                    out.append(obj)
            except ValueError:
                continue
        return out

    def summary(self, days: int = 30, active_ran_s: float | None = None, active_cu: float | None = None) -> dict[str, Any]:
        entries = self.read_all()
        today = dt.date.today()
        today_str = today.isoformat()
        month_prefix = today_str[:7]

        daily_map: dict[str, dict[str, Any]] = defaultdict(lambda: {"cu": 0.0, "ran_s": 0.0, "sessions": 0})
        today_cu = 0.0
        month_cu = 0.0
        total_cu = 0.0
        total_s = 0.0

        for e in entries:
            d = str(e.get("date") or "")
            cu = float(e.get("cu_est") or 0.0)
            rs = float(e.get("ran_s") or 0.0)
            total_cu += cu
            total_s += rs
            if d == today_str:
                today_cu += cu
            if d.startswith(month_prefix):
                month_cu += cu
            if d:
                bucket = daily_map[d]
                bucket["cu"] += cu
                bucket["ran_s"] += rs
                bucket["sessions"] += 1

        if active_cu is not None and active_cu > 0:
            today_cu += active_cu
            month_cu += active_cu
            total_cu += active_cu
            total_s += active_ran_s or 0.0
            bucket = daily_map[today_str]
            bucket["cu"] += active_cu
            bucket["ran_s"] += active_ran_s or 0.0

        daily: list[dict[str, Any]] = []
        for offset in range(max(1, days) - 1, -1, -1):
            d = (today - dt.timedelta(days=offset)).isoformat()
            b = daily_map.get(d) or {"cu": 0.0, "ran_s": 0.0, "sessions": 0}
            daily.append({
                "date": d,
                "cu": round(b["cu"], 3),
                "hours": round(b["ran_s"] / 3600.0, 2),
                "sessions": int(b["sessions"]),
            })

        return {
            "today_cu": round(today_cu, 2),
            "month_cu": round(month_cu, 2),
            "total_cu": round(total_cu, 2),
            "total_hours": round(total_s / 3600.0, 2),
            "sessions_count": len(entries),
            "daily": daily,
            "recent": list(reversed(entries[-25:])),
        }
