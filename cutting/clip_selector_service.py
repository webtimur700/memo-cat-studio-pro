"""Выбор финальных моментов для нарезки (Функция 4).

Вход — оценённые скользящие окна по всему видео (WindowScore), выход —
непересекающиеся Moment длиной min..max секунд (по умолчанию 15–60),
отсортированные по времени.

Отбор ОТНОСИТЕЛЬНО самого видео, а не по одному абсолютному порогу:
  1. "сильное" окно — score не ниже абсолютного минимума (queue_threshold) И
     входит в лучшие relative_top_ratio окон этого видео. У сборок с ровно
     хорошим материалом абсолютный порог пропускал бы почти всё (75% видео
     в клипах), относительный оставляет действительно лучшее.
  2. соседние сильные окна (допускается одно слабое между ними) склеиваются
     в один момент — длинная удачная сцена не рвётся на куски по 15 с.
  3. момент подгоняется под min..max: короткий расширяется в сторону более
     сильных окон, слишком длинный сжимается до лучшего отрезка max секунд.
  4. отбор по убыванию оценки с ограничениями: не больше max_moments и не
     больше max_coverage_ratio исходного видео суммарно (первый момент
     берётся всегда).
  5. границы подтягиваются к естественным точкам разреза в пределах
     snap_tolerance_sec: смены сцен (сильнее всего) и паузы в речи — чтобы не
     резать посреди фразы или действия.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from core.entities.detection import Detection
from core.entities.moment import Moment
from core.entities.settings import UserSettings
from core.entities.score import ScoreBreakdown, SignalPart
from scoring.viral_score_service import ScoreInputs

MIN_WINDOW_SEC = 1.0
GAP_BRIDGE_SCORE_RATIO = 0.85       # слабое окно между двумя сильными склеивается, если оно не хуже 85% порога
PEAK_WEIGHT = 0.6                   # оценка момента = 0.6 * лучшее окно + 0.4 * среднее
PAUSE_PENALTY_SEC = 0.5             # пауза речи чуть "дороже" смены сцены при равном расстоянии
SCENE_DISTANCE_WEIGHT = 0.5         # смена сцены "притягивает" вдвое сильнее, чем пауза

PauseFinder = Callable[[float, float], list[float]]
"""(начало, конец) в секундах видео -> моменты пауз речи (секунды видео) внутри диапазона."""


@dataclass(frozen=True, slots=True)
class WindowScore:
    start_sec: float
    end_sec: float
    viral_score: int
    motion_intensity: float
    detections: tuple[Detection, ...] = ()
    inputs: ScoreInputs | None = None            # из чего сложилась оценка (для пересчёта и для объяснения пользователю)
    breakdown: ScoreBreakdown | None = None      # вклад каждого сигнала в очках


@dataclass(frozen=True, slots=True)
class _Run:
    windows: tuple[WindowScore, ...]
    score: int

    @property
    def start(self) -> float:
        return self.windows[0].start_sec

    @property
    def end(self) -> float:
        return self.windows[-1].end_sec


def _quantile(sorted_values: Sequence[int], q: float) -> float:
    """Nearest-rank квантиль отсортированного списка (q в 0..1)."""
    index = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return sorted_values[index]


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _range_score(windows: Sequence[WindowScore], start: float, end: float) -> float:
    """Сумма score, взвешенная длиной пересечения отрезка с окнами."""
    total = 0.0
    for w in windows:
        overlap = min(end, w.end_sec) - max(start, w.start_sec)
        if overlap > 0:
            total += overlap * w.viral_score
    return total


def _build_runs(windows: Sequence[WindowScore], threshold: float) -> list[_Run]:
    strong = [w.viral_score >= threshold for w in windows]
    bridge_floor = threshold * GAP_BRIDGE_SCORE_RATIO
    runs: list[_Run] = []
    i = 0
    while i < len(windows):
        if not strong[i]:
            i += 1
            continue
        j = i
        while True:
            if j + 1 < len(windows) and strong[j + 1]:
                j += 1
            elif (
                j + 2 < len(windows)
                and not strong[j + 1]
                and strong[j + 2]
                and windows[j + 1].viral_score >= bridge_floor
            ):
                j += 2
            else:
                break
        members = tuple(windows[i:j + 1])
        scores = [w.viral_score for w in members]
        combined = PEAK_WEIGHT * max(scores) + (1 - PEAK_WEIGHT) * (sum(scores) / len(scores))
        runs.append(_Run(members, round(combined)))
        i = j + 1
    return runs


def _moment_breakdown(run: _Run) -> tuple[SignalPart, ...]:
    """Из чего сложилась оценка момента: те же 0.6·лучшее окно + 0.4·среднее, но по вкладам сигналов, поэтому сумма очков
    совпадает с viral_score (до округления). Пусто, если у окон нет разложения (оценка считалась без него)."""
    if any(w.breakdown is None for w in run.windows):
        return ()
    peak = max(run.windows, key=lambda w: w.viral_score)
    parts: list[SignalPart] = []
    for peak_part in peak.breakdown.parts:
        same = [(w, w.breakdown.part(peak_part.key)) for w in run.windows]
        same = [(w, p) for w, p in same if p is not None]
        mean_points = sum(p.points for _, p in same) / len(same)
        details = [p.detail for _, p in sorted(same, key=lambda wp: -wp[1].points) if p.detail]
        parts.append(SignalPart(
            key=peak_part.key,
            label=peak_part.label,
            value=sum(p.value for _, p in same) / len(same),
            weight=peak_part.weight,
            points=PEAK_WEIGHT * peak_part.points + (1 - PEAK_WEIGHT) * mean_points,
            detail=_merge_details(peak_part.key, details),
            bonus=peak_part.bonus,
        ))
    return tuple(parts)


def _merge_details(key: str, details: list[str]) -> str:
    if not details:
        return ""
    if key == "scene":
        total = sum(int(m.group()) for d in details if (m := re.match(r"\d+", d)))
        return f"{total} смен(ы)" if total else ""
    if key == "audio":   # «лай, смех» из разных окон: без повторов, самые заметные первыми
        seen: list[str] = []
        for detail in details:
            seen += [label for label in detail.split(", ") if label not in seen]
        return ", ".join(seen[:3])
    unique = list(dict.fromkeys(details))
    return "; ".join(unique[:3])


def _slide_best(
    windows: Sequence[WindowScore], lo: float, hi: float, length: float
) -> float:
    """Начало отрезка длиной length внутри [lo, hi] с максимальной суммой score (шаг 1 с)."""
    best_start, best_score = lo, -1.0
    start = lo
    while start + length <= hi + 1e-6:
        score = _range_score(windows, start, start + length)
        if score > best_score + 1e-9:
            best_start, best_score = start, score
        start += 1.0
    return best_start


def _fit_duration(
    run: _Run, windows: Sequence[WindowScore], settings: UserSettings, video_duration_sec: float
) -> tuple[float, float]:
    min_d = float(settings.shorts.min_duration_sec)
    max_d = float(settings.shorts.max_duration_sec)
    a, b = run.start, min(run.end, video_duration_sec)
    length = b - a

    if length > max_d:
        start = _slide_best(windows, a, b, max_d)
        return start, start + max_d

    if length < min_d:
        allowed = sorted(d for d in settings.shorts.allowed_durations_sec if min_d <= d <= max_d)
        target = next((float(d) for d in allowed if d >= length), max_d)
        target = min(target, video_duration_sec)
        # отрезок target секунд, обязательно содержащий сам момент, с лучшим окружением
        lo = max(0.0, b - target)
        hi = min(video_duration_sec, a + target)
        start = _slide_best(windows, lo, max(hi, lo + target), target)
        start = min(max(start, lo), max(0.0, video_duration_sec - target))
        return start, min(video_duration_sec, start + target)

    return a, b


def _snap_boundary(
    target: float,
    scene_boundaries: Sequence[float],
    pauses: Sequence[float],
    tolerance: float,
    accept: Callable[[float], bool],
) -> float:
    """Ближайшая "естественная" точка разреза к target (смена сцены или пауза
    речи) в пределах tolerance; если такой нет или она ломает допустимую
    длительность (accept) — сама target."""
    best_point, best_cost = target, float("inf")
    for point in scene_boundaries:
        if abs(point - target) <= tolerance and accept(point):
            cost = abs(point - target) * SCENE_DISTANCE_WEIGHT
            if cost < best_cost:
                best_point, best_cost = point, cost
    for point in pauses:
        if abs(point - target) <= tolerance and accept(point):
            cost = abs(point - target) + PAUSE_PENALTY_SEC
            if cost < best_cost:
                best_point, best_cost = point, cost
    return best_point


def _snap_moment(
    start: float,
    end: float,
    settings: UserSettings,
    video_duration_sec: float,
    scene_boundaries: Sequence[float],
    pause_finder: PauseFinder | None,
) -> tuple[float, float]:
    tolerance = settings.shorts.snap_tolerance_sec
    min_d, max_d = float(settings.shorts.min_duration_sec), float(settings.shorts.max_duration_sec)
    min_d = min(min_d, video_duration_sec)
    pauses: list[float] = []
    if pause_finder is not None:
        pauses = pause_finder(max(0.0, start - tolerance), min(video_duration_sec, end + tolerance))

    def valid_length(a: float, b: float) -> bool:
        return min_d - 1e-6 <= b - a <= max_d + 1e-6

    new_start = _snap_boundary(
        start, scene_boundaries, pauses, tolerance,
        lambda p: 0.0 <= p < end and valid_length(p, end),
    )
    new_end = _snap_boundary(
        end, scene_boundaries, pauses, tolerance,
        lambda p: p <= video_duration_sec and valid_length(new_start, p),
    )
    return new_start, new_end


def select_moments(
    window_scores: list[WindowScore],
    settings: UserSettings,
    video_duration_sec: float,
    max_moments: int | None = None,
    scene_boundaries: Sequence[float] = (),
    pause_finder: PauseFinder | None = None,
) -> list[Moment]:
    windows = [w for w in window_scores if w.end_sec - w.start_sec >= MIN_WINDOW_SEC]
    if not windows:
        return []

    max_moments = max_moments if max_moments is not None else settings.shorts.max_moments

    sorted_scores = sorted(w.viral_score for w in windows)
    relative_cutoff = _quantile(sorted_scores, 1.0 - settings.viral_score.relative_top_ratio)
    threshold = max(settings.viral_score.queue_threshold, relative_cutoff)

    runs = _build_runs(windows, threshold)
    runs.sort(key=lambda r: r.score, reverse=True)

    coverage_budget = settings.shorts.max_coverage_ratio * video_duration_sec
    chosen: list[tuple[_Run, float, float]] = []
    covered = 0.0
    for run in runs:
        if len(chosen) >= max_moments:
            break
        start, end = _fit_duration(run, windows, settings, video_duration_sec)
        if any(_overlaps((start, end), (s, e)) for _, s, e in chosen):
            continue
        if chosen and covered + (end - start) > coverage_budget:
            continue
        chosen.append((run, start, end))
        covered += end - start

    moments: list[Moment] = []
    final_ranges: list[tuple[float, float]] = []
    for run, start, end in chosen:
        snapped = _snap_moment(start, end, settings, video_duration_sec, scene_boundaries, pause_finder)
        if any(_overlaps(snapped, r) for r in final_ranges):
            snapped = (start, end)
            if any(_overlaps(snapped, r) for r in final_ranges):
                continue
        final_ranges.append(snapped)
        peak = max(run.windows, key=lambda w: w.viral_score)
        moments.append(
            Moment(
                start_sec=snapped[0],
                end_sec=snapped[1],
                viral_score=run.score,
                motion_intensity=sum(w.motion_intensity for w in run.windows) / len(run.windows),
                detections=peak.detections,
                breakdown=_moment_breakdown(run),
            )
        )

    moments.sort(key=lambda m: m.start_sec)
    return moments
