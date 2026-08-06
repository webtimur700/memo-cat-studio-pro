from pathlib import Path

from core.entities.settings import ExportSettings
from core.entities.subtitle import SubtitleSegment, WordTiming
from export.dynamic_crop import CropSample
from export.export_service import ExportPlan, ExportService
from subtitles.ass_renderer import render_ass
from video.ffmpeg_wrapper import FFmpegWrapper
from vision.smart_crop import CropWindow


def test_full_export_pipeline_produces_valid_mp4(sample_video: Path, tmp_path: Path):
    crop_samples = [
        CropSample(float(i), CropWindow(x1=100 + i * 15, y1=0, x2=100 + i * 15 + 405, y2=720, zoom_factor=1.0, timestamp_sec=float(i)))
        for i in range(9)
    ]

    segments = [SubtitleSegment(words=[WordTiming("Тест", 1.0, 1.5), WordTiming("экспорта", 1.5, 2.2)])]
    ass_path = tmp_path / "subs.ass"
    ass_path.write_text(render_ass(segments, style_preset="modern_bold"), encoding="utf-8")

    plan = ExportPlan(
        source_path=sample_video,
        output_path=tmp_path / "final.mp4",
        crop_samples=crop_samples,
        subtitle_ass_path=ass_path,
        banner_rect=(90, 1500, 990, 1780),
        banner_text_lines=["100zarplat.ru", "Подбор вакансий"],
        banner_appear_at_sec=2.0,
        banner_duration_sec=5.0,
        settings=ExportSettings(width=1080, height=1920, fps=30, quality_preset="low"),
    )

    result = ExportService().export_clip(plan)

    assert result.exists()
    wrapper = FFmpegWrapper()
    output_source = wrapper.get_video_source(result)
    assert output_source.width == 1080
    assert output_source.height == 1920
    assert abs(output_source.duration_sec - 8.0) < 0.3
