# API — Memo Cat AI Studio Pro

Этот документ описывает внутренние API, которые нужны для (а) написания
плагинов и (б) прямого использования сервисов пайплайна из кода (например,
из тестов или скриптов автоматизации).

## Plugin SDK

### Структура плагина

```
plugins/
  my_plugin/
    plugin.py   # обязательный файл, точка входа
```

`plugin.py` должен экспортировать объект `PLUGIN`, реализующий
`core.interfaces.plugin_base.PluginBase`:

```python
from core.interfaces.plugin_base import PluginMetadata
from plugins.sdk.registry import PluginRegistry

class MyPlugin:
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_effect",
            version="1.0.0",
            description="Что делает эффект",
            author="Ваше имя",
        )

    def register(self, registry: PluginRegistry) -> None:
        registry.register_effect("my_effect", my_effect_function)

PLUGIN = MyPlugin()
```

Рабочий пример: `plugins/examples/example_effect_plugin/plugin.py`
(сепия-тонирование).

### PluginRegistry — что можно регистрировать

| Метод | Назначение | Сигнатура регистрируемой функции |
|---|---|---|
| `register_effect(name, fn)` | Эффект обработки кадра | `(frame_bgr: np.ndarray) -> np.ndarray` |
| `register_template(name, template)` | Шаблон (dict с параметрами) | — |
| `register_animation_style(name, fn)` | Стиль анимации | зависит от типа анимации |
| `register_scoring_strategy(name, fn)` | Альтернативная формула Viral Score | `(features: dict) -> float` |

### Загрузка плагинов

```python
from pathlib import Path
from plugins.loader import load_plugins
from plugins.sdk.registry import PluginRegistry

registry = PluginRegistry()
loaded = load_plugins(Path("plugins/examples"), registry)
# loaded: list[PluginMetadata]
```

Сломанный плагин не прерывает загрузку остальных — ошибка логируется через
`loguru` и загрузчик переходит к следующему.

---

## Основные сервисы (для прямого использования вне UI)

### Video Engine

```python
from pathlib import Path
from video.ingestion_service import IngestionService

service = IngestionService()
source = service.ingest(Path("video.mp4"))  # -> VideoSource
```

### AI Engine

```python
from pathlib import Path
from vision.yolo_detector import YoloDetector
from vision.compute_backend import ComputeBackend

detector = YoloDetector(Path("models/yolo11n.onnx"), compute_backend=ComputeBackend())
detections = detector.detect(frame_bgr)  # -> list[Detection]
```

### Subtitle Engine

```python
from pathlib import Path
from subtitles.subtitle_service import WhisperTranscriber
from subtitles.ass_renderer import render_ass

transcriber = WhisperTranscriber(model_size="small", compute_type="int8")
segments = transcriber.transcribe_to_segments(Path("audio.wav"))
ass_content = render_ass(segments, style_preset="modern_bold")
```

### LLM (заголовки/описание/хештеги)

```python
from llm.lm_studio_provider import LMStudioProvider
from llm.prompts.titles_prompt import generate_titles
from llm.prompts.description_prompt import generate_description
from llm.prompts.hashtags_prompt import generate_hashtags

provider = LMStudioProvider()  # использует LM_STUDIO_BASE_URL из .env
titles = generate_titles(provider, "Кот прыгает на шкаф и сбивает вазу")
description = generate_description(provider, "...")
hashtags = generate_hashtags(provider, "...")
```

### Export

```python
from export.export_service import ExportService, ExportPlan

plan = ExportPlan(
    source_path=..., output_path=..., crop_samples=..., subtitle_ass_path=...,
    banner_rect=..., banner_text_lines=[...], banner_appear_at_sec=4.0,
    banner_duration_sec=5.0, settings=...,
)
result_path = ExportService().export_clip(plan)
```

Полные сигнатуры и dataclass-поля — смотрите исходники соответствующих
модулей, каждый снабжён подробным docstring с обоснованием решений.
