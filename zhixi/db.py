"""工单数据库（SQLite）。

智能体执行闭环的“落地证据”：所有维修工单都真实写入数据库，
可以在答辩现场直接展示新增记录。
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    code            TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    lot_id          TEXT NOT NULL,
    station         TEXT NOT NULL,
    priority        TEXT NOT NULL,
    risk            REAL NOT NULL,
    action          TEXT NOT NULL,
    owner           TEXT NOT NULL,
    spare_parts     TEXT,
    due_at          TEXT,
    status          TEXT NOT NULL DEFAULT '待执行',
    evidence        TEXT,
    closed_at       TEXT,
    close_note      TEXT
);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
    except (OSError, sqlite3.OperationalError):
        # 云端只读文件系统时的兜底：退到临时目录，保证演示不中断
        import tempfile

        path = Path(tempfile.gettempdir()) / "zhixi_workorders.db"
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def _next_code(conn: sqlite3.Connection, now: datetime) -> str:
    prefix = f"WO-{now:%Y%m%d}-"
    row = conn.execute(
        "SELECT code FROM work_orders WHERE code LIKE ? ORDER BY code DESC LIMIT 1", (prefix + "%",)
    ).fetchone()
    seq = int(row["code"].split("-")[-1]) + 1 if row else 1
    return f"{prefix}{seq:03d}"


def create_work_order(
    *,
    lot_id: str,
    station: str,
    priority: str,
    risk: float,
    action: str,
    owner: str = "设备维护组",
    spare_parts: str = "",
    evidence: str = "",
    repair_hours: float = 1.5,
    path: Path | str = DB_PATH,
    now: datetime | None = None,
) -> dict:
    now = now or datetime.now()
    with closing(connect(path)) as conn:
        code = _next_code(conn, now)
        due = now + timedelta(hours=repair_hours)
        conn.execute(
            """INSERT INTO work_orders
               (code, created_at, lot_id, station, priority, risk, action, owner,
                spare_parts, due_at, status, evidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                code,
                now.isoformat(timespec="seconds"),
                lot_id,
                station,
                priority,
                round(float(risk), 4),
                action,
                owner,
                spare_parts,
                due.isoformat(timespec="seconds"),
                "待执行",
                evidence,
            ),
        )
        conn.commit()
        return dict(conn.execute("SELECT * FROM work_orders WHERE code=?", (code,)).fetchone())


def list_work_orders(path: Path | str = DB_PATH, limit: int = 100) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM work_orders ORDER BY created_at DESC LIMIT ?", conn, params=(limit,)
        )
    return df


def close_work_order(code: str, note: str = "", path: Path | str = DB_PATH) -> dict | None:
    with closing(connect(path)) as conn:
        conn.execute(
            "UPDATE work_orders SET status='已完成', closed_at=?, close_note=? WHERE code=?",
            (datetime.now().isoformat(timespec="seconds"), note, code),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM work_orders WHERE code=?", (code,)).fetchone()
        return dict(row) if row else None


def clear_all(path: Path | str = DB_PATH) -> None:
    with closing(connect(path)) as conn:
        conn.execute("DELETE FROM work_orders")
        conn.commit()
