"""Kaplan–Meier failure-free probability and pilot-group comparison.

Only pre-aggregated summary tables are read; ``battery_ledger`` is not
touched by queries, and follow-ups are never expanded day by day.

Counting rules
--------------
* Age is whole days from ``activation_date`` to the follow-up date.
* A subject enters at its follow-up start age and is exposed from that age.
* The follow-up window is clipped at the chosen cutoff date.  A failure after
  the cutoff is treated as censoring at the cutoff; a follow-up end after the
  cutoff becomes the cutoff day itself.
* Subjects whose follow-up start is after the cutoff are not yet enrolled and
  contribute nothing.
* At a given age, subjects entering that day are included in the risk set and
  subjects whose follow-up ends that day remain in the risk set, so failures
  are counted after entry and after the end-of-day censoring rule.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence

Row = tuple[str, int, int, int, int]  # activation_date, entry, end, event, count


@dataclass(frozen=True)
class SurvivalPoint:
    age_days: int
    at_risk: int
    failures: int
    censored: int
    survival: float


@dataclass(frozen=True)
class SurvivalResult:
    cutoff: date
    points: list[SurvivalPoint]
    enrolled_subjects: int
    risk_set_at_cutoff: int

    @property
    def has_failures(self) -> bool:
        return bool(self.points)


def summarize(rows: Iterable[Row], cutoff: date) -> SurvivalResult:
    """Return KM statistics at every age where a failure was observed."""

    failures_by_age: dict[int, int] = {}
    censored_by_age: dict[int, int] = {}
    # Compressed risk-set contributions: one (entry, end, count) interval per
    # summary row, with the follow-up window clipped to the cutoff.
    intervals: list[tuple[int, int, int]] = []
    enrolled = 0
    risk_at_cutoff = 0

    for activation_text, entry_age, end_age, event, count in rows:
        if count <= 0:
            continue
        activation = date.fromisoformat(activation_text)
        cutoff_age = (cutoff - activation).days

        # Not yet enrolled at the cutoff.
        if entry_age > cutoff_age:
            continue

        enrolled += count
        if entry_age <= cutoff_age <= end_age:
            risk_at_cutoff += count

        if end_age > cutoff_age:
            end_age = cutoff_age
            event = 0  # a failure after the cutoff is censored at the cutoff
        if end_age < entry_age:
            end_age = entry_age

        intervals.append((entry_age, end_age, count))
        if event:
            failures_by_age[end_age] = failures_by_age.get(end_age, 0) + count
        else:
            censored_by_age[end_age] = censored_by_age.get(end_age, 0) + count

    failure_ages = sorted(failures_by_age)

    # Difference array over failure ages: each subject is at risk on every
    # failure age from its entry age through its end age (inclusive).
    size = len(failure_ages)
    diff = [0] * (size + 1)
    for entry_age, end_age, count in intervals:
        left = bisect_left(failure_ages, entry_age)
        right = bisect_right(failure_ages, end_age)
        if left < right:
            diff[left] += count
            diff[right] -= count

    points: list[SurvivalPoint] = []
    at_risk = 0
    survival = 1.0
    for index, age in enumerate(failure_ages):
        at_risk += diff[index]
        failures = failures_by_age[age]
        censored = censored_by_age.get(age, 0)
        survival *= 1.0 - failures / at_risk if at_risk > 0 else 1.0
        points.append(
            SurvivalPoint(
                age_days=age,
                at_risk=at_risk,
                failures=failures,
                censored=censored,
                survival=survival,
            )
        )

    return SurvivalResult(
        cutoff=cutoff,
        points=points,
        enrolled_subjects=enrolled,
        risk_set_at_cutoff=risk_at_cutoff,
    )


# --------------------------------------------------------------------------
# Pilot vs non-pilot comparison, stratified by battery age
# --------------------------------------------------------------------------

#: A follow-up day belongs to the "old" layer once the battery's age on that
#: day reaches this many whole days.
AGE_LAYER_THRESHOLD_DAYS = 180

GROUP_PILOT = "pilot"
GROUP_CONTROL = "control"
LAYER_YOUNG = "lt180"
LAYER_OLD = "ge180"
GROUPS = (GROUP_PILOT, GROUP_CONTROL)
LAYERS = (LAYER_YOUNG, LAYER_OLD)

GROUP_LABELS = {GROUP_PILOT: "试点组", GROUP_CONTROL: "非试点组"}
LAYER_LABELS = {
    LAYER_YOUNG: f"机龄 < {AGE_LAYER_THRESHOLD_DAYS} 天",
    LAYER_OLD: f"机龄 ≥ {AGE_LAYER_THRESHOLD_DAYS} 天",
}


@dataclass(frozen=True)
class Segment:
    """A homogeneous slice of a follow-up window: one group, one age layer."""

    seg_start: date
    seg_end: date
    seg_group: str
    age_layer: str
    failure_on_end: bool

    @property
    def days(self) -> int:
        return (self.seg_end - self.seg_start).days + 1


def compute_segments(
    activation: date,
    followup_start: date,
    followup_end: date,
    outcome: str,
    pilot_start: date | None,
) -> list[Segment]:
    """Split a follow-up window into homogeneous (group, age-layer) segments.

    The inclusive window ``[followup_start, followup_end]`` is cut at the day
    the battery turns ``AGE_LAYER_THRESHOLD_DAYS`` old and at the pilot start
    day, so every segment lies entirely in one group and one age layer.  Pure
    day arithmetic — the follow-up is never expanded day by day.

    A failure belongs to the segment ending on the follow-up end day, i.e. it
    is recorded only in the group and layer of its own day; that day still
    counts as one follow-up day.
    """

    cuts: set[date] = set()
    age_boundary = activation + timedelta(days=AGE_LAYER_THRESHOLD_DAYS)
    if followup_start < age_boundary <= followup_end:
        cuts.add(age_boundary)
    if pilot_start is not None and followup_start < pilot_start <= followup_end:
        cuts.add(pilot_start)

    boundaries = [followup_start, *sorted(cuts), followup_end + timedelta(days=1)]
    segments: list[Segment] = []
    for seg_start, next_start in zip(boundaries, boundaries[1:]):
        seg_end = next_start - timedelta(days=1)
        age_days = (seg_start - activation).days
        layer = LAYER_OLD if age_days >= AGE_LAYER_THRESHOLD_DAYS else LAYER_YOUNG
        group = (
            GROUP_PILOT
            if pilot_start is not None and seg_start >= pilot_start
            else GROUP_CONTROL
        )
        segments.append(
            Segment(
                seg_start=seg_start,
                seg_end=seg_end,
                seg_group=group,
                age_layer=layer,
                failure_on_end=outcome == "failure" and seg_end == followup_end,
            )
        )
    return segments


@dataclass(frozen=True)
class CellStat:
    """Follow-up battery-days and failures for one (group, layer) cell."""

    days: int
    failures: int


@dataclass(frozen=True)
class LayerWeight:
    layer: str
    days: int  # combined follow-up days of both groups in this layer
    weight: float


@dataclass(frozen=True)
class ComparisonResult:
    start: date
    end: date
    cells: dict[tuple[str, str], CellStat]
    common_layers: tuple[str, ...]
    layer_weights: tuple[LayerWeight, ...]
    pilot_rate_per_1000: float | None
    control_rate_per_1000: float | None
    difference_per_1000: float | None

    @property
    def comparable(self) -> bool:
        return bool(self.common_layers)

    @property
    def total_days(self) -> int:
        return sum(cell.days for cell in self.cells.values())

    @property
    def total_failures(self) -> int:
        return sum(cell.failures for cell in self.cells.values())


def compare_groups(
    day_rows: Iterable[Sequence],
    failure_rows: Iterable[Sequence],
    start: date,
    end: date,
) -> ComparisonResult:
    """Weighted per-1000-battery-day failure rates, pilot minus control.

    ``day_rows`` and ``failure_rows`` are ``(group, layer, value)`` triples
    aggregated from the segment summary for the inclusive range
    ``[start, end]``.  Only layers where both groups have follow-up days are
    used; each such layer is weighted by its share of the combined follow-up
    days across those layers.  When no layer has follow-up in both groups the
    result is not comparable and the rates are ``None``.
    """

    days = {(group, layer): 0 for group in GROUPS for layer in LAYERS}
    failures = dict(days)
    for group, layer, value in day_rows:
        if (group, layer) in days and value is not None:
            days[(group, layer)] += int(value)
    for group, layer, value in failure_rows:
        if (group, layer) in failures and value is not None:
            failures[(group, layer)] += int(value)

    cells = {key: CellStat(days=days[key], failures=failures[key]) for key in days}
    common = tuple(
        layer
        for layer in LAYERS
        if cells[(GROUP_PILOT, layer)].days > 0
        and cells[(GROUP_CONTROL, layer)].days > 0
    )
    if not common:
        return ComparisonResult(
            start=start,
            end=end,
            cells=cells,
            common_layers=(),
            layer_weights=(),
            pilot_rate_per_1000=None,
            control_rate_per_1000=None,
            difference_per_1000=None,
        )

    total = sum(cells[(group, layer)].days for layer in common for group in GROUPS)
    weights = tuple(
        LayerWeight(
            layer=layer,
            days=sum(cells[(group, layer)].days for group in GROUPS),
            weight=sum(cells[(group, layer)].days for group in GROUPS) / total,
        )
        for layer in common
    )

    rates: dict[str, float] = {}
    for group in GROUPS:
        per_day = sum(
            layer_weight.weight
            * cells[(group, layer_weight.layer)].failures
            / cells[(group, layer_weight.layer)].days
            for layer_weight in weights
        )
        rates[group] = per_day * 1000.0

    return ComparisonResult(
        start=start,
        end=end,
        cells=cells,
        common_layers=common,
        layer_weights=weights,
        pilot_rate_per_1000=rates[GROUP_PILOT],
        control_rate_per_1000=rates[GROUP_CONTROL],
        difference_per_1000=rates[GROUP_PILOT] - rates[GROUP_CONTROL],
    )
