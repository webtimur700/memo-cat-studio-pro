"""Динамический crop-фильтр FFmpeg для покадрового автокадрирования
(Функция 5: "следить за животным"). vision/smart_crop.SmartCropPlanner уже
даёт сглаженную (EMA) последовательность CropWindow — здесь она сжимается до
разреженных временных сэмплов и превращается в кусочно-линейное
ffmpeg-выражение для x/y (`crop` фильтр с `eval=frame`), чтобы кроп реально
двигался внутри одного вызова ffmpeg, а не был статичным кадром на весь клип.

Почему кусочно-линейно, а не просто передать все кадры: ffmpeg `crop` умеет
вычислять x/y как выражение от `t` (секунды) на КАЖДЫЙ кадр — параметры x/y
этого фильтра помечены флагом "T" (time-varying), то есть перевычисляются
покадрово автоматически, без отдельной опции eval (проверено реальным
запуском ffmpeg на этой машине — в более старых/новых сборках ffmpeg
поведение опций `crop` отличалось, поэтому это стоит перепроверять командой
`ffmpeg -h filter=crop` при переходе на другую версию ffmpeg). Вложенные
`if(lt(t,T), A, B)` дают ровно кусочно-линейную интерполяцию между сэмплами
и остаются читаемым плоским выражением даже на sparse-сэмплах (каждые ~0.5с
достаточно плавно при уже сглаженном EMA-треке).
"""

from __future__ import annotations

from dataclasses import dataclass

from vision.smart_crop import CropWindow


@dataclass(frozen=True, slots=True)
class CropSample:
    timestamp_sec: float
    window: CropWindow


def _build_piecewise_linear_expr(samples: list[tuple[float, float]]) -> str:
    """samples: [(t0, v0), (t1, v1), ...], отсортированы по t.
    Возвращает ffmpeg-выражение, линейно интерполирующее v между сэмплами,
    константное до первого и после последнего.
    """
    if not samples:
        raise ValueError("Нужен хотя бы один сэмпл")
    if len(samples) == 1:
        return f"{samples[0][1]:.2f}"

    # Собираем с конца: каждый следующий уровень вложенности — "else"-ветка.
    expr = f"{samples[-1][1]:.2f}"  # после последнего сэмпла — держим последнее значение
    for i in range(len(samples) - 2, -1, -1):
        t0, v0 = samples[i]
        t1, v1 = samples[i + 1]
        segment = f"({v0:.2f}+({v1:.2f}-{v0:.2f})*(t-{t0:.3f})/{max(t1 - t0, 1e-6):.3f})"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"

    # До первого сэмпла — держим первое значение.
    t0 = samples[0][0]
    expr = f"if(lt(t,{t0:.3f}),{samples[0][1]:.2f},{expr})"
    return expr


def build_dynamic_crop_filter(
    samples: list[CropSample],
    sample_stride_sec: float = 0.5,
) -> str:
    """Возвращает готовый ffmpeg filter-фрагмент вида
    "crop=W:H:x='EXPR':y='EXPR':eval=frame" на основе разреженной выборки
    из полного (покадрового) списка CropSample.
    """
    if not samples:
        raise ValueError("samples не может быть пустым")

    ordered = sorted(samples, key=lambda s: s.timestamp_sec)

    sparse: list[CropSample] = [ordered[0]]
    for sample in ordered[1:]:
        if sample.timestamp_sec - sparse[-1].timestamp_sec >= sample_stride_sec:
            sparse.append(sample)
    if sparse[-1] is not ordered[-1]:
        sparse.append(ordered[-1])

    # Ширина/высота кропа берутся из первого сэмпла — SmartCropPlanner уже
    # обеспечивает постоянный аспект-рейшо, размер меняется только вместе с
    # zoom, который мы тоже сэмплируем через x/y (упрощение: используем
    # медианный размер, чтобы избежать скачков итогового разрешения).
    width = round(sum(s.window.width for s in sparse) / len(sparse))
    height = round(sum(s.window.height for s in sparse) / len(sparse))

    x_expr = _build_piecewise_linear_expr([(s.timestamp_sec, s.window.x1) for s in sparse])
    y_expr = _build_piecewise_linear_expr([(s.timestamp_sec, s.window.y1) for s in sparse])

    return f"crop={width}:{height}:x='{x_expr}':y='{y_expr}'"
