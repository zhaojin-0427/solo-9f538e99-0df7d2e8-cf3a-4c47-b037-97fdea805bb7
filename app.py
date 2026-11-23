"""Flask application: battery follow-up CSV import, KM survival page, and
pilot vs non-pilot failure-rate comparison."""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from analysis import (
    GROUP_LABELS,
    GROUPS,
    LAYER_LABELS,
    LAYERS,
    compare_groups,
    summarize,
)
from db import MAX_SUMMARY_RECORDS_PER_BATTERY, Database, ledger_rows, roster_rows
from importer import (
    ImportValidationError,
    apply_import,
    apply_roster_import,
    parse_csv,
    parse_roster_csv,
)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        DATABASE=os.environ.get(
            "BATTERY_DB", str(Path(app.root_path) / "data" / "battery.db")
        ),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        SECRET_KEY=os.environ.get("SECRET_KEY", "battery-survival-local-dev"),
    )
    if config:
        app.config.update(config)

    database = Database(app.config["DATABASE"])
    database.initialize()

    @app.context_processor
    def inject_globals() -> dict:
        return {
            "max_summary_records": MAX_SUMMARY_RECORDS_PER_BATTERY,
            "today_utc": _utc_today().isoformat(),
        }

    @app.get("/")
    def index():
        return redirect(url_for("survival_page"))

    @app.get("/upload")
    def upload_page():
        return render_template(
            "upload.html",
            result=None,
            errors=[],
            filename="",
            batches=_recent_batches(database),
        )

    @app.post("/upload")
    def upload():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return (
                render_template(
                    "upload.html",
                    result=None,
                    errors=["请选择一个 CSV 文件再上传。"],
                    filename="",
                    batches=_recent_batches(database),
                ),
                400,
            )

        filename = secure_filename(uploaded.filename) or "upload.csv"
        errors: list[str] = []
        try:
            data = uploaded.read()
            records = parse_csv(data)
            conn = database.connect()
            try:
                result = apply_import(conn, filename, records)
            finally:
                conn.close()
        except ImportValidationError as exc:
            errors = exc.shown_messages
            result = None
        except sqlite3.Error as exc:
            app.logger.exception("import failed")
            errors = [f"写入数据库失败，未做任何更改：{exc}"]
            result = None
        except Exception:  # pragma: no cover - defensive; transaction rolled back
            app.logger.exception("unexpected import failure")
            errors = ["导入过程中发生未预期错误，未做任何更改，原有曲线仍可查询。"]
            result = None
        if result is not None:
            flash(
                f"导入成功：新增 {result.inserted}，更正 {result.updated}，"
                f"删除 {result.deleted}，未变化 {result.unchanged}；"
                f"本次写入汇总记录 {result.summary_records} 条。"
            )
            return redirect(url_for("upload_page"))

        return (
            render_template(
                "upload.html",
                result=None,
                errors=errors,
                filename=filename,
                batches=_recent_batches(database),
            ),
            400,
        )

    @app.get("/survival")
    def survival_page():
        cutoff_text = request.args.get("cutoff", "").strip()
        cutoff: date | None = None
        errors: list[str] = []
        if cutoff_text:
            try:
                cutoff = date.fromisoformat(cutoff_text)
            except ValueError:
                errors.append("截止日必须是 YYYY-MM-DD 格式的 UTC 日期。")
        else:
            cutoff = _utc_today()

        rows: list = []
        result = None
        if cutoff is not None:
            conn = database.connect()
            try:
                rows = _load_summary(conn)
            finally:
                conn.close()
            result = summarize(rows, cutoff)

        return render_template(
            "survival.html",
            cutoff=cutoff,
            cutoff_text=cutoff_text,
            result=result,
            rows=rows,
            errors=errors,
            total_subjects=sum(row[4] for row in rows),
        )

    @app.get("/ledger")
    def ledger_page():
        conn = database.connect()
        try:
            rows = list(ledger_rows(conn))
        finally:
            conn.close()
        return render_template("ledger.html", rows=rows)

    @app.get("/pilot")
    def pilot_page():
        start_text = request.args.get("start", "").strip()
        end_text = request.args.get("end", "").strip()
        errors: list[str] = []
        start: date | None = None
        end: date | None = None
        if start_text or end_text:
            if not start_text or not end_text:
                errors.append("开始日和结束日都要选择（日期范围含首尾两天）。")
            else:
                try:
                    start = date.fromisoformat(start_text)
                except ValueError:
                    errors.append("开始日必须是 YYYY-MM-DD 格式的 UTC 日期。")
                try:
                    end = date.fromisoformat(end_text)
                except ValueError:
                    errors.append("结束日必须是 YYYY-MM-DD 格式的 UTC 日期。")
                if start is not None and end is not None and start > end:
                    errors.append("开始日不能晚于结束日。")

        result = None
        cell_rows: list[dict] = []
        weight_rows: list[dict] = []
        if start is not None and end is not None and not errors:
            conn = database.connect()
            try:
                day_rows = list(
                    conn.execute(
                        "SELECT seg_group, age_layer, "
                        "SUM(CAST(julianday(MIN(seg_end, ?)) "
                        "- julianday(MAX(seg_start, ?)) AS INTEGER) + 1) "
                        "FROM battery_segments "
                        "WHERE seg_start <= ? AND seg_end >= ? "
                        "GROUP BY seg_group, age_layer",
                        (
                            end.isoformat(),
                            start.isoformat(),
                            end.isoformat(),
                            start.isoformat(),
                        ),
                    )
                )
                failure_rows = list(
                    conn.execute(
                        "SELECT seg_group, age_layer, SUM(failure_on_end) "
                        "FROM battery_segments "
                        "WHERE failure_on_end = 1 AND seg_end >= ? AND seg_end <= ? "
                        "GROUP BY seg_group, age_layer",
                        (start.isoformat(), end.isoformat()),
                    )
                )
            finally:
                conn.close()
            result = compare_groups(day_rows, failure_rows, start, end)
            for layer in LAYERS:
                for group in GROUPS:
                    cell = result.cells[(group, layer)]
                    cell_rows.append(
                        {
                            "layer_label": LAYER_LABELS[layer],
                            "group_label": GROUP_LABELS[group],
                            "days": cell.days,
                            "failures": cell.failures,
                            "rate_per_1000": (
                                cell.failures * 1000.0 / cell.days
                                if cell.days
                                else None
                            ),
                        }
                    )
            weight_rows = [
                {
                    "layer_label": LAYER_LABELS[layer_weight.layer],
                    "days": layer_weight.days,
                    "weight": layer_weight.weight,
                }
                for layer_weight in result.layer_weights
            ]

        return render_template(
            "pilot.html",
            start_text=start_text,
            end_text=end_text,
            result=result,
            cell_rows=cell_rows,
            weight_rows=weight_rows,
            errors=errors,
        )

    @app.get("/roster")
    def roster_page():
        return render_template(
            "roster.html", errors=[], filename="", **_roster_context(database)
        )

    @app.post("/roster")
    def roster_upload():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return (
                render_template(
                    "roster.html",
                    errors=["请选择一个 CSV 文件再上传。"],
                    filename="",
                    **_roster_context(database),
                ),
                400,
            )

        filename = secure_filename(uploaded.filename) or "roster.csv"
        errors: list[str] = []
        try:
            data = uploaded.read()
            records = parse_roster_csv(data)
            conn = database.connect()
            try:
                result = apply_roster_import(conn, filename, records)
            finally:
                conn.close()
        except ImportValidationError as exc:
            errors = exc.shown_messages
            result = None
        except sqlite3.Error as exc:
            app.logger.exception("roster import failed")
            errors = [f"写入数据库失败，未做任何更改：{exc}"]
            result = None
        except Exception:  # pragma: no cover - defensive; transaction rolled back
            app.logger.exception("unexpected roster import failure")
            errors = ["导入过程中发生未预期错误，未做任何更改，原有对比结果仍可查询。"]
            result = None
        if result is not None:
            flash(
                f"名单导入成功：新增 {result.inserted}，更正 {result.updated}，"
                f"删除 {result.deleted}，未变化 {result.unchanged}。"
            )
            return redirect(url_for("roster_page"))

        return (
            render_template(
                "roster.html",
                errors=errors,
                filename=filename,
                **_roster_context(database),
            ),
            400,
        )

    @app.get("/roster-template.csv")
    def roster_template_csv():
        path = Path(app.root_path) / "static" / "roster_template.csv"
        return send_file(path, mimetype="text/csv", as_attachment=True,
                         download_name="pilot_roster_template.csv")

    @app.get("/template.csv")
    def template_csv():
        path = Path(app.root_path) / "static" / "template.csv"
        return send_file(path, mimetype="text/csv", as_attachment=True,
                         download_name="battery_followup_template.csv")

    @app.get("/healthz")
    def healthz():
        conn = database.connect()
        try:
            conn.execute("SELECT 1")
        finally:
            conn.close()
        return {"status": "ok"}

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        message = f"文件超过大小上限（{MAX_UPLOAD_BYTES // (1024 * 1024)} MB）。"
        if request.path.startswith("/roster"):
            return (
                render_template(
                    "roster.html",
                    errors=[message],
                    filename="",
                    **_roster_context(database),
                ),
                413,
            )
        return (
            render_template(
                "upload.html",
                result=None,
                errors=[message],
                filename="",
                batches=_recent_batches(database),
            ),
            413,
        )

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("error.html", code=404, message="页面不存在。"), 404

    return app


def _load_summary(conn: sqlite3.Connection):
    return [
        (
            row["activation_date"],
            int(row["entry_age"]),
            int(row["end_age"]),
            int(row["event"]),
            int(row["subject_count"]),
        )
        for row in conn.execute(
            "SELECT activation_date, entry_age, end_age, event, subject_count "
            "FROM summary_counts"
        )
    ]


def _recent_batches(database: Database):
    conn = database.connect()
    try:
        return list(
            conn.execute(
                "SELECT id, filename, imported_at, total_rows, inserted, updated, "
                "deleted, unchanged, summary_records FROM import_batches "
                "ORDER BY id DESC LIMIT 10"
            )
        )
    finally:
        conn.close()


def _roster_context(database: Database) -> dict:
    conn = database.connect()
    try:
        entries = list(roster_rows(conn))
        batches = list(
            conn.execute(
                "SELECT id, filename, imported_at, total_rows, inserted, updated, "
                "deleted, unchanged FROM roster_batches ORDER BY id DESC LIMIT 10"
            )
        )
    finally:
        conn.close()
    return {"entries": entries, "batches": batches}


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
