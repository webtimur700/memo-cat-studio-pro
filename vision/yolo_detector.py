"""YOLO11 инференс через ONNX Runtime.

Модель ожидается в формате, который получается стандартным экспортом
Ultralytics: `yolo export model=yolo11n.pt format=onnx` — выход формы
(1, 4 + num_classes, num_boxes) для одного изображения (box-координаты в
формате cx,cy,w,h в пикселях letterbox-кадра, дальше по каналам —
per-class confidence, без отдельного objectness, как в YOLOv8/11).

Предобработка (letterbox) и постобработка (decode + NMS) вынесены в чистые
функции без побочных эффектов — это позволяет тестировать их синтетическими
тензорами без реального .onnx файла и реального GPU/CPU инференса (что и
сделано в этой песочнице, см. пояснение в ответе).

ЧЕСТНОЕ ПРИМЕЧАНИЕ: сам вызов ONNX-модели (session.run) в этой песочнице не
протестирован — здесь нет весов yolo11n.onnx и нет сети, чтобы их скачать.
Протестирована и подтверждена вся математика вокруг него (letterbox,
инверсия координат, NMS) на синтетических данных той же формы, что выдаёт
реальная модель.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.entities.detection import COCO_ANIMAL_NAMES, BoundingBox, Detection
from core.exceptions import MemoCatError
from vision.compute_backend import ComputeBackend

# Индексы классов COCO, которые релевантны проекту (полный список из 80 классов
# Ultralytics YOLO11, обученной на COCO — не обучаем свою модель с нуля).
COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    **COCO_ANIMAL_NAMES,   # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
}


class ModelNotFoundError(MemoCatError):
    pass


def letterbox(
    frame: np.ndarray, target_size: int = 640
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Приводит кадр к квадрату target_size x target_size с сохранением
    пропорций (паддинг серым по краям) — стандартная предобработка YOLO.

    Возвращает (letterboxed_frame, scale, (pad_x, pad_y)) — эти три числа
    нужны, чтобы потом перевести координаты боксов обратно в систему
    координат оригинального кадра.
    """
    height, width = frame.shape[:2]
    scale = min(target_size / height, target_size / width)
    new_height, new_width = int(round(height * scale)), int(round(width * scale))

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_size - new_width) / 2
    pad_y = (target_size - new_height) / 2

    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    # copyMakeBorder может дать размер +-1px из-за округления — подрезаем/дополняем точно.
    padded = cv2.resize(padded, (target_size, target_size))

    return padded, scale, (pad_x, pad_y)


def preprocess_for_inference(frame: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    letterboxed, scale, pad = letterbox(frame, target_size)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)
    return batched, scale, pad


def postprocess_predictions(
    raw_output: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    allowed_class_ids: dict[int, str],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Декодирует сырой выход YOLO11 (1, 4+num_classes, num_boxes) в список Detection
    в системе координат ОРИГИНАЛЬНОГО кадра (не letterbox-кадра).
    """
    predictions = raw_output[0]  # (4 + num_classes, num_boxes)
    num_channels, num_boxes = predictions.shape
    num_classes = num_channels - 4

    boxes_cxcywh = predictions[0:4, :].T          # (num_boxes, 4)
    class_scores = predictions[4:, :].T           # (num_boxes, num_classes)

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(num_boxes), class_ids]

    pad_x, pad_y = pad

    candidate_boxes: list[list[float]] = []
    candidate_scores: list[float] = []
    candidate_class_ids: list[int] = []

    for i in range(num_boxes):
        class_id = int(class_ids[i])
        confidence = float(confidences[i])

        if class_id not in allowed_class_ids:
            continue
        if confidence < conf_threshold:
            continue

        cx, cy, w, h = boxes_cxcywh[i]

        # Обратное преобразование letterbox -> оригинальные координаты.
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        candidate_boxes.append([x1, y1, x2 - x1, y2 - y1])  # NMSBoxes ждёт (x, y, w, h)
        candidate_scores.append(confidence)
        candidate_class_ids.append(class_id)

    if not candidate_boxes:
        return []

    keep_indices = cv2.dnn.NMSBoxes(
        candidate_boxes, candidate_scores, conf_threshold, iou_threshold
    )
    keep_indices = np.array(keep_indices).flatten() if len(keep_indices) else []

    detections: list[Detection] = []
    for idx in keep_indices:
        x, y, w, h = candidate_boxes[idx]
        class_id = candidate_class_ids[idx]
        detections.append(
            Detection(
                class_id=class_id,
                class_name=allowed_class_ids[class_id],
                confidence=candidate_scores[idx],
                bbox=BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h),
            )
        )
    return detections


class YoloDetector:
    def __init__(
        self,
        model_path: Path,
        compute_backend: ComputeBackend | None = None,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        allowed_class_ids: dict[int, str] | None = None,
    ) -> None:
        if not model_path.exists():
            raise ModelNotFoundError(
                f"YOLO11 ONNX-модель не найдена: {model_path}. "
                f"Экспортируйте: yolo export model=yolo11n.pt format=onnx"
            )

        self._compute_backend = compute_backend or ComputeBackend()
        self._session = self._compute_backend.create_session(str(model_path))
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._allowed_class_ids = allowed_class_ids or COCO_CLASS_NAMES

    def detect(self, frame: np.ndarray) -> list[Detection]:
        input_tensor, scale, pad = preprocess_for_inference(frame, self._input_size)
        raw_output = self._session.run(None, {self._input_name: input_tensor})[0]
        return postprocess_predictions(
            raw_output,
            scale,
            pad,
            self._allowed_class_ids,
            self._conf_threshold,
            self._iou_threshold,
        )
