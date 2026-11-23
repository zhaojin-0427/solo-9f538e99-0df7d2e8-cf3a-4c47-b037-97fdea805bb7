"""Tests for the pilot vs non-pilot comparison: segments, roster imports,
weighted rates, migration of pre-existing follow-ups, and concurrency."""

from __future__ import annotations

import io
import sqlite3
import threading
from datetime import date

import pytest

from analysis import (
    AGE_LAYER_THRESHOLD_DAYS,
    compare_groups,
    compute_segments,
)
from app import create_app
from db import Database
from importer import (
    ImportValidationError,
    apply_import,
    apply_roster_import,
    parse_csv,
    parse_roster_csv,
)

FOLLOWUP_HEADER = (
    "battery_no,activation_date,followup_start,followup_end,outcome,action\n"
)
ROSTER_HEADER = "battery_no,pilot_start,action\n"


def followup_csv(*lines: str) -> bytes:
    return (FOLLOWUP_HEADER + "\n".join(lines) + "\n").encode("utf-8")


def roster_csv(*lines: str) -> bytes:
    return (ROSTER_HEADER + "\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "battery.db"


@pytest.fixture()
def database(db_path):
    database = Database(db_path)
    database.initialize()
    return database


@pytest.fixture()
def client(database, db_path):
    app = create_app({"TESTING": True, "DATABASE": str(db_path), "SECRET_KEY": "t"})
    with app.test_client() as client:
        yield client


def import_followups(database, *lines: str, filename: str = "followups.csv"):
    records = parse_csv(followup_csv(*lines))
    conn = database.connect()
    try:
        return apply_import(conn, filename, records)
    finally:
        conn.close()


def import_roster(database, *lines: str, filename: str = "roster.csv"):
    records = parse_roster_csv(roster_csv(*lines))
    conn = database.connect()
    try:
        return apply_roster_import(conn, filename, records)
    finally:
        conn.close()


def segment_rows(database, battery_no=None):
    conn = database.connect()
    try:
        sql = (
            "SELECT battery_no, seg_start, seg_end, seg_group, age_layer, "
            "failure_on_end FROM battery_segments"
        )
        params: tuple = ()
        if battery_no is not None:
            sql += " WHERE battery_no = ?"
            params = (battery_no,)
        sql += " ORDER BY battery_no, seg_start"
        return [tuple(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def compare_via_db(database, start: str, end: str):
    """Run the same summary queries the /pilot page uses."""
    conn = database.connect()
    try:
        day_rows = list(
            conn.execute(
                "SELECT seg_group, age_layer, "
                "SUM(CAST(julianday(MIN(seg_end, ?)) "
                "- julianday(MAX(seg_start, ?)) AS INTEGER) + 1) "
                "FROM battery_segments WHERE seg_start <= ? AND seg_end >= ? "
                "GROUP BY seg_group, age_layer",
                (end, start, end, start),
            )
        )
        failure_rows = list(
            conn.execute(
                "SELECT seg_group, age_layer, SUM(failure_on_end) "
                "FROM battery_segments "
                "WHERE failure_on_end = 1 AND seg_end >= ? AND seg_end <= ? "
                "GROUP BY seg_group, age_layer",
                (start, end),
            )
        )
    finally:
        conn.close()
    return compare_groups(
        [tuple(row) for row in day_rows],
        [tuple(row) for row in failure_rows],
        date.fromisoformat(start),
        date.fromisoformat(end),
    )


def cell(result, group, layer):
    return result.cells[(group, layer)]


# --------------------------------------------------------------------------
# Segment computation (pure date arithmetic, no day-by-day expansion)
# --------------------------------------------------------------------------


def test_segments_single_segment_without_roster():
    segments = compute_segments(
        date(2026, 1, 1), date(2026, 1, 10), date(2026, 2, 1), "censored", None
    )
    assert len(segments) == 1
    segment = segments[0]
    assert (segment.seg_group, segment.age_layer) == ("control", "lt180")
    assert segment.days == 23
    assert not segment.failure_on_end


def test_segments_split_at_age_boundary():
    # Age 180 is reached on 2026-06-30 for a 2026-01-01 activation.
    segments = compute_segments(
        date(2026, 1, 1), date(2026, 1, 1), date(2026, 8, 31), "censored", None
    )
    assert [(s.seg_start.isoformat(), s.seg_end.isoformat(), s.age_layer) for s in segments] == [
        ("2026-01-01", "2026-06-29", "lt180"),
        ("2026-06-30", "2026-08-31", "ge180"),
    ]
    assert [s.days for s in segments] == [180, 63]


def test_segments_split_at_pilot_start():
    segments = compute_segments(
        date(2026, 1, 1),
        date(2026, 1, 1),
        date(2026, 3, 31),
        "censored",
        date(2026, 2, 1),
    )
    assert [(s.seg_group, s.days) for s in segments] == [("control", 31), ("pilot", 59)]


def test_segments_split_at_both_boundaries():
    segments = compute_segments(
        date(2026, 1, 1),
        date(2026, 1, 1),
        date(2026, 8, 31),
        "censored",
        date(2026, 4, 1),
    )
    assert [(s.seg_group, s.age_layer, s.days) for s in segments] == [
        ("control", "lt180", 90),
        ("pilot", "lt180", 90),
        ("pilot", "ge180", 63),
    ]


def test_segments_pilot_before_followup_is_all_pilot():
    segments = compute_segments(
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 1),
        "censored",
        date(2026, 1, 15),
    )
    assert len(segments) == 1
    assert segments[0].seg_group == "pilot"


def test_segments_pilot_after_followup_is_all_control():
    segments = compute_segments(
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 1),
        "censored",
        date(2026, 4, 1),
    )
    assert len(segments) == 1
    assert segments[0].seg_group == "control"


def test_segments_failure_belongs_to_last_segment():
    # Fails exactly on the day it turns 180 days old: the failure (and that
    # one day) land in the ge180 layer.
    segments = compute_segments(
        date(2026, 1, 1), date(2026, 1, 1), date(2026, 6, 30), "failure", None
    )
    assert [(s.age_layer, s.days, s.failure_on_end) for s in segments] == [
        ("lt180", 180, False),
        ("ge180", 1, True),
    ]
    # One day earlier the whole follow-up (and the failure) is lt180.
    segments = compute_segments(
        date(2026, 1, 1), date(2026, 1, 1), date(2026, 6, 29), "failure", None
    )
    assert [(s.age_layer, s.days, s.failure_on_end) for s in segments] == [
        ("lt180", 180, True),
    ]


def test_age_threshold_is_180_days():
    assert AGE_LAYER_THRESHOLD_DAYS == 180


# --------------------------------------------------------------------------
# Weighted comparison math
# --------------------------------------------------------------------------


def test_compare_groups_weighted_rates():
    day_rows = [
        ("pilot", "lt180", 100),
        ("control", "lt180", 300),
        ("pilot", "ge180", 200),
        ("control", "ge180", 100),
    ]
    failure_rows = [
        ("pilot", "lt180", 1),
        ("control", "lt180", 3),
        ("pilot", "ge180", 1),
        ("control", "ge180", 1),
    ]
    result = compare_groups(
        day_rows, failure_rows, date(2026, 1, 1), date(2026, 12, 31)
    )
    assert result.comparable
    assert result.common_layers == ("lt180", "ge180")
    weights = {w.layer: w.weight for w in result.layer_weights}
    assert weights["lt180"] == pytest.approx(400 / 700)
    assert weights["ge180"] == pytest.approx(300 / 700)
    # pilot: 4/7 * 1/100 + 3/7 * 1/200 = 11/1400 per day
    assert result.pilot_rate_per_1000 == pytest.approx(11000 / 1400)
    # control: 4/7 * 3/300 + 3/7 * 1/100 = 1/100 per day
    assert result.control_rate_per_1000 == pytest.approx(10.0)
    assert result.difference_per_1000 == pytest.approx(11000 / 1400 - 10.0)


def test_compare_groups_single_common_layer_gets_full_weight():
    day_rows = [("pilot", "ge180", 50), ("control", "ge180", 40)]
    failure_rows = [("control", "ge180", 2)]
    result = compare_groups(
        day_rows, failure_rows, date(2026, 1, 1), date(2026, 1, 31)
    )
    assert result.common_layers == ("ge180",)
    assert result.layer_weights[0].weight == pytest.approx(1.0)
    assert result.pilot_rate_per_1000 == 0.0
    assert result.control_rate_per_1000 == pytest.approx(50.0)
    assert result.difference_per_1000 == pytest.approx(-50.0)


def test_compare_groups_no_common_layer_is_not_comparable():
    day_rows = [("pilot", "lt180", 100), ("control", "ge180", 100)]
    result = compare_groups(day_rows, [], date(2026, 1, 1), date(2026, 1, 31))
    assert not result.comparable
    assert result.pilot_rate_per_1000 is None
    assert result.control_rate_per_1000 is None
    assert result.difference_per_1000 is None


# --------------------------------------------------------------------------
# End-to-end: follow-up import builds segments; roster drives the grouping
# --------------------------------------------------------------------------

# B1: cross-layer, cross-group follow-up (censored).  B2: fails at age 242.
B1_LINE = "B1,2026-01-01,2026-01-01,2026-08-31,censored"
B2_LINE = "B2,2026-01-01,2026-01-01,2026-08-31,failure"


def test_followup_import_builds_segments_without_roster(database):
    import_followups(database, B1_LINE, B2_LINE)
    assert segment_rows(database) == [
        ("B1", "2026-01-01", "2026-06-29", "control", "lt180", 0),
        ("B1", "2026-06-30", "2026-08-31", "control", "ge180", 0),
        ("B2", "2026-01-01", "2026-06-29", "control", "lt180", 0),
        ("B2", "2026-06-30", "2026-08-31", "control", "ge180", 1),
    ]


def test_cross_layer_followup_counts_days_per_cell(database):
    """跨层随访：一段随访同时跨过 180 天机龄界和试点开始日。"""
    import_followups(database, B1_LINE, B2_LINE)
    import_roster(database, "B1,2026-04-01")

    result = compare_via_db(database, "2026-01-01", "2026-08-31")
    assert cell(result, "control", "lt180").days == 90 + 180
    assert cell(result, "control", "lt180").failures == 0
    assert cell(result, "control", "ge180").days == 63
    assert cell(result, "control", "ge180").failures == 1
    assert cell(result, "pilot", "lt180").days == 90
    assert cell(result, "pilot", "ge180").days == 63

    # Both layers are shared; weights are the combined-day shares of 486.
    assert result.comparable
    weights = {w.layer: w.weight for w in result.layer_weights}
    assert weights["lt180"] == pytest.approx(360 / 486)
    assert weights["ge180"] == pytest.approx(126 / 486)
    assert result.pilot_rate_per_1000 == 0.0
    assert result.control_rate_per_1000 == pytest.approx((126 / 486) * (1 / 63) * 1000)
    assert result.difference_per_1000 == pytest.approx(
        -(126 / 486) * (1 / 63) * 1000
    )


def test_date_range_clips_segments_without_daily_expansion(database):
    import_followups(database, B1_LINE, B2_LINE)
    import_roster(database, "B1,2026-04-01")

    result = compare_via_db(database, "2026-04-01", "2026-06-30")
    assert cell(result, "pilot", "lt180").days == 90
    assert cell(result, "pilot", "ge180").days == 1  # only 2026-06-30
    assert cell(result, "control", "lt180").days == 90
    assert cell(result, "control", "ge180").days == 1
    # B2's failure on 2026-08-31 is outside the range.
    assert result.total_failures == 0


def test_failure_on_range_boundary_day_is_counted(database):
    import_followups(database, B2_LINE)
    result = compare_via_db(database, "2026-08-31", "2026-08-31")
    # Single day range: the failure day itself, counted as one day.
    assert cell(result, "control", "ge180").days == 1
    assert cell(result, "control", "ge180").failures == 1


def test_roster_correction_moves_days_between_groups(database):
    """名单更正：同一电池号重新上传后，分段汇总按新开始日重算。"""
    import_followups(database, B1_LINE, B2_LINE)
    first = import_roster(database, "B1,2026-04-01")
    assert first.inserted == 1

    corrected = import_roster(database, "B1,2026-07-01")
    assert corrected.updated == 1
    assert segment_rows(database, "B1") == [
        ("B1", "2026-01-01", "2026-06-29", "control", "lt180", 0),
        ("B1", "2026-06-30", "2026-06-30", "control", "ge180", 0),
        ("B1", "2026-07-01", "2026-08-31", "pilot", "ge180", 0),
    ]

    result = compare_via_db(database, "2026-01-01", "2026-08-31")
    assert cell(result, "pilot", "lt180").days == 0
    assert cell(result, "pilot", "ge180").days == 62
    # Only the ge180 layer is shared now, so it gets the full weight.
    assert result.common_layers == ("ge180",)
    assert result.control_rate_per_1000 == pytest.approx(1000 / 64)
    assert result.difference_per_1000 == pytest.approx(-1000 / 64)

    # Identical re-upload is a no-op.
    again = import_roster(database, "B1,2026-07-01")
    assert again.unchanged == 1


def test_roster_delete_returns_battery_to_control(database):
    import_followups(database, B1_LINE)
    import_roster(database, "B1,2026-04-01")
    removed = import_roster(database, "B1,,delete")
    assert removed.deleted == 1
    assert segment_rows(database, "B1") == [
        ("B1", "2026-01-01", "2026-06-29", "control", "lt180", 0),
        ("B1", "2026-06-30", "2026-08-31", "control", "ge180", 0),
    ]


def test_roster_can_precede_followup_registration(database):
    """名单先于随访登记：随访导入时按已有名单分组。"""
    import_roster(database, "B3,2026-02-01")
    import_followups(database, "B3,2026-01-01,2026-01-15,2026-03-15,censored")
    assert segment_rows(database, "B3") == [
        ("B3", "2026-01-15", "2026-01-31", "control", "lt180", 0),
        ("B3", "2026-02-01", "2026-03-15", "pilot", "lt180", 0),
    ]


def test_followup_correction_and_delete_rebuild_segments(database):
    import_followups(database, "B4,2026-01-01,2026-01-01,2026-03-01,censored")
    import_followups(database, "B4,2026-01-01,2026-01-01,2026-08-31,failure")
    assert segment_rows(database, "B4") == [
        ("B4", "2026-01-01", "2026-06-29", "control", "lt180", 0),
        ("B4", "2026-06-30", "2026-08-31", "control", "ge180", 1),
    ]
    import_followups(database, "B4,,,,,delete")
    assert segment_rows(database, "B4") == []


def test_import_only_recomputes_batch_batteries(database):
    """导入只重算本批涉及的电池，其它电池的分段保持不变。"""
    import_followups(database, B1_LINE, B2_LINE)
    before_b2 = segment_rows(database, "B2")
    import_roster(database, "B1,2026-04-01")
    assert segment_rows(database, "B2") == before_b2
    # And the batch battery was recomputed.
    assert any(row[3] == "pilot" for row in segment_rows(database, "B1"))


def test_no_common_layer_message(database, client):
    # Only control-group data: no shared layer, cannot compare.
    import_followups(database, B2_LINE)
    page = client.get("/pilot?start=2026-01-01&end=2026-08-31")
    assert page.status_code == 200
    assert "无法比较" in page.get_data(as_text=True)


# --------------------------------------------------------------------------
# Roster CSV parsing and validation
# --------------------------------------------------------------------------


def test_roster_parse_accepts_chinese_headers(database):
    content = "电池号,试点开始日\nB1,2026-03-01\n".encode("utf-8")
    records = parse_roster_csv(content)
    assert records[0].battery_no == "B1"
    assert records[0].pilot_start == date(2026, 3, 1)


def test_roster_parse_rejects_duplicate_battery():
    with pytest.raises(ImportValidationError) as excinfo:
        parse_roster_csv(roster_csv("B1,2026-03-01", "B1,2026-04-01"))
    assert any("重复" in message for message in excinfo.value.messages)


def test_roster_parse_requires_start_date():
    with pytest.raises(ImportValidationError) as excinfo:
        parse_roster_csv(roster_csv("B1,"))
    assert any("试点开始日" in message for message in excinfo.value.messages)


def test_roster_parse_rejects_missing_column():
    content = "battery_no,start\nB1,2026-03-01\n".encode("utf-8")
    with pytest.raises(ImportValidationError) as excinfo:
        parse_roster_csv(content)
    assert any("缺少列" in message for message in excinfo.value.messages)


def test_roster_explicit_actions(database):
    import_roster(database, "B1,2026-03-01")
    with pytest.raises(ImportValidationError):
        import_roster(database, "B1,2026-04-01,insert")  # already exists
    with pytest.raises(ImportValidationError):
        import_roster(database, "B2,2026-04-01,update")  # does not exist
    with pytest.raises(ImportValidationError):
        import_roster(database, "B2,,delete")  # does not exist
    result = import_roster(database, "B1,2026-05-01,update")
    assert result.updated == 1


def test_failed_roster_import_is_atomic(database, client):
    """失败的名单导入整体回滚，原有对比结果仍可查询。"""
    import_followups(database, B1_LINE, B2_LINE)
    import_roster(database, "B1,2026-04-01")
    before_segments = segment_rows(database)
    before = compare_via_db(database, "2026-01-01", "2026-08-31")

    with pytest.raises(ImportValidationError):
        import_roster(database, "B2,2026-05-01", "B3,not-a-date")

    assert segment_rows(database) == before_segments
    after = compare_via_db(database, "2026-01-01", "2026-08-31")
    assert after.cells == before.cells
    assert after.difference_per_1000 == before.difference_per_1000

    conn = database.connect()
    try:
        roster_count = conn.execute("SELECT COUNT(*) FROM pilot_roster").fetchone()[0]
        batch_count = conn.execute("SELECT COUNT(*) FROM roster_batches").fetchone()[0]
    finally:
        conn.close()
    assert roster_count == 1
    assert batch_count == 1

    page = client.get("/pilot?start=2026-01-01&end=2026-08-31")
    assert page.status_code == 200
    assert "无法比较" not in page.get_data(as_text=True)


# --------------------------------------------------------------------------
# Pre-existing follow-ups in the SQLite volume join the statistics directly
# --------------------------------------------------------------------------

OLD_SCHEMA = """
CREATE TABLE battery_ledger (
    battery_no      TEXT PRIMARY KEY,
    activation_date TEXT NOT NULL,
    followup_start  TEXT NOT NULL,
    followup_end    TEXT NOT NULL,
    outcome         TEXT NOT NULL CHECK (outcome IN ('failure', 'censored')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE battery_summary (
    battery_no      TEXT PRIMARY KEY,
    activation_date TEXT NOT NULL,
    entry_age       INTEGER NOT NULL,
    end_age         INTEGER NOT NULL,
    event           INTEGER NOT NULL,
    revision        INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE battery_summary_usage (
    battery_no      TEXT PRIMARY KEY,
    summary_records INTEGER NOT NULL
);
CREATE TABLE summary_counts (
    activation_date TEXT NOT NULL,
    entry_age       INTEGER NOT NULL,
    end_age         INTEGER NOT NULL,
    event           INTEGER NOT NULL,
    subject_count   INTEGER NOT NULL,
    PRIMARY KEY (activation_date, entry_age, end_age, event)
);
CREATE TABLE import_batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    imported_at     TEXT NOT NULL,
    total_rows      INTEGER NOT NULL,
    inserted        INTEGER NOT NULL,
    updated         INTEGER NOT NULL,
    unchanged       INTEGER NOT NULL,
    summary_records INTEGER NOT NULL
);
"""


def test_existing_volume_followups_are_migrated(db_path):
    """原 SQLite 卷里的随访直接纳入统计（启动时一次性回填分段汇总）。"""
    conn = sqlite3.connect(db_path)
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        "INSERT INTO battery_ledger VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "B-OLD",
            "2026-01-01",
            "2026-01-02",
            "2026-02-01",
            "failure",
            "2026-02-02T00:00:00+00:00",
            "2026-02-02T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    database = Database(db_path)
    database.initialize()

    assert segment_rows(database, "B-OLD") == [
        ("B-OLD", "2026-01-02", "2026-02-01", "control", "lt180", 1)
    ]
    result = compare_via_db(database, "2026-01-01", "2026-12-31")
    assert cell(result, "control", "lt180").days == 31
    assert cell(result, "control", "lt180").failures == 1

    # The migration is marked done and is idempotent on restart.
    conn = database.connect()
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        deleted_column = "deleted" in {
            row["name"] for row in conn.execute("PRAGMA table_info(import_batches)")
        }
    finally:
        conn.close()
    assert version == 1
    assert deleted_column

    database.initialize()  # second startup keeps the segments
    assert segment_rows(database, "B-OLD") == [
        ("B-OLD", "2026-01-02", "2026-02-01", "control", "lt180", 1)
    ]


# --------------------------------------------------------------------------
# Concurrent roster and follow-up imports
# --------------------------------------------------------------------------


def test_concurrent_roster_and_followup_imports(database):
    """名单和随访并发导入：每批原数据和汇总一起生效，最终结果一致。"""
    failures: list[BaseException] = []

    def followup_worker(index: int):
        try:
            for n in range(4):
                import_followups(
                    database,
                    f"F{index}-{n},2026-01-01,2026-01-01,2026-03-31,censored",
                    filename=f"f{index}-{n}.csv",
                )
        except BaseException as exc:  # pragma: no cover - failure path
            failures.append(exc)

    def roster_worker(index: int):
        try:
            for n in range(4):
                import_roster(
                    database,
                    f"F{index}-{n},2026-02-01",
                    filename=f"r{index}-{n}.csv",
                )
        except BaseException as exc:  # pragma: no cover - failure path
            failures.append(exc)

    threads = [
        threading.Thread(target=followup_worker, args=(i,)) for i in range(2)
    ] + [threading.Thread(target=roster_worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    # Whatever the interleaving, every battery ends up in the pilot group
    # from 2026-02-01: 31 control days + 59 pilot days.
    result = compare_via_db(database, "2026-01-01", "2026-03-31")
    assert cell(result, "control", "lt180").days == 8 * 31
    assert cell(result, "pilot", "lt180").days == 8 * 59
    assert result.total_days == 8 * 90


# --------------------------------------------------------------------------
# Web pages
# --------------------------------------------------------------------------


def upload_followups(client, data: bytes, filename="followups.csv"):
    return client.post(
        "/upload",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def upload_roster(client, data: bytes, filename="roster.csv"):
    return client.post(
        "/roster",
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_pilot_page_end_to_end(client):
    assert upload_followups(client, followup_csv(B1_LINE, B2_LINE)).status_code == 302
    assert upload_roster(client, roster_csv("B1,2026-04-01")).status_code == 302

    page = client.get("/pilot?start=2026-01-01&end=2026-08-31")
    text = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "270" in text  # control lt180 days
    assert "4.115" in text  # control weighted rate per 1000 days
    assert "-4.115" in text  # difference pilot - control
    assert "0.7407" in text  # lt180 weight 360/486
    assert "无法比较" not in text


def test_pilot_page_validates_date_range(client):
    page = client.get("/pilot?start=2026-08-31&end=2026-01-01")
    assert "开始日不能晚于结束日" in page.get_data(as_text=True)

    page = client.get("/pilot?start=2026-01-01")
    assert "开始日和结束日都要选择" in page.get_data(as_text=True)

    page = client.get("/pilot?start=not-a-date&end=2026-01-01")
    assert "YYYY-MM-DD" in page.get_data(as_text=True)

    page = client.get("/pilot")
    assert page.status_code == 200
    assert "请选择含首尾的 UTC 日期范围" in page.get_data(as_text=True)


def test_pilot_page_reads_only_summaries(client, db_path):
    """统计只读汇总表：台账表不可用时对比页照常工作。"""
    assert upload_followups(client, followup_csv(B1_LINE, B2_LINE)).status_code == 302
    assert upload_roster(client, roster_csv("B1,2026-04-01")).status_code == 302

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE battery_ledger")
        conn.commit()

    page = client.get("/pilot?start=2026-01-01&end=2026-08-31")
    assert page.status_code == 200
    assert "4.115" in page.get_data(as_text=True)


def test_roster_page_upload_list_and_correct(client, database):
    page = client.get("/roster")
    assert page.status_code == 200
    assert "名单为空" in page.get_data(as_text=True)

    response = upload_roster(client, roster_csv("B1,2026-03-01", "B2,2026-03-15"))
    assert response.status_code == 302

    page = client.get("/roster")
    text = page.get_data(as_text=True)
    assert "B1" in text and "2026-03-01" in text
    assert "B2" in text and "2026-03-15" in text

    # Correction of the same battery number is shown on the page.
    response = upload_roster(client, roster_csv("B1,2026-04-01"))
    assert response.status_code == 302
    page = client.get("/roster")
    text = page.get_data(as_text=True)
    assert "2026-04-01" in text and "2026-03-01" not in text

    conn = database.connect()
    try:
        rows = list(conn.execute("SELECT battery_no, pilot_start FROM pilot_roster"))
        batches = conn.execute("SELECT COUNT(*) FROM roster_batches").fetchone()[0]
    finally:
        conn.close()
    assert len(rows) == 2
    assert batches == 2


def test_roster_page_rejects_bad_file_and_keeps_old_results(client):
    assert upload_roster(client, roster_csv("B1,2026-03-01")).status_code == 302
    response = upload_roster(
        client, b"battery_no,pilot_start\nB2,not-a-date\n", "bad.csv"
    )
    assert response.status_code == 400
    assert "不是有效日期" in response.get_data(as_text=True)

    page = client.get("/roster")
    text = page.get_data(as_text=True)
    assert "B1" in text and "B2" not in text


def test_roster_template_download(client):
    response = client.get("/roster-template.csv")
    assert response.status_code == 200
    assert "pilot_start" in response.get_data(as_text=True)
