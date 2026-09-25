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


def _plan(sample_video, tmp_path, name, seconds=9, **export):
    crop_samples = [
        CropSample(float(i), CropWindow(x1=100 + i * 15, y1=0, x2=100 + i * 15 + 405, y2=720, zoom_factor=1.0, timestamp_sec=float(i)))
        for i in range(seconds)
    ]
    return ExportPlan(
        source_path=sample_video, output_path=tmp_path / name, crop_samples=crop_samples, subtitle_ass_path=None,
        banner_rect=(90, 1500, 990, 1780), banner_text_lines=["100zarplat.ru"], banner_appear_at_sec=2.0, banner_duration_sec=5.0,
        settings=ExportSettings(width=1080, height=1920, quality_preset="high", **export),
    )


def _probe(path: Path, entries: str) -> str:
    import subprocess

    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={entries}:format=bit_rate",
                           "-of", "default=nw=1", str(path)], capture_output=True, text=True, check=True).stdout


def test_fps_and_bitrate_settings_reach_the_final_clip(tmp_path: Path):
    """Настройки FPS и битрейта должны влиять на итоговый клип (он получается во втором проходе, с оверлеями)."""
    import subprocess

    sample_video = tmp_path / "noisy.mp4"     # шумный кадр: без потолка он не уложится в 1 Мбит/с
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:duration=9:rate=30,noise=alls=60:allf=t+u",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=9", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", "-pix_fmt", "yuv420p",
                    "-shortest", str(sample_video)], check=True)
    service = ExportService()
    low = service.export_clip(_plan(sample_video, tmp_path, "low.mp4", seconds=4, fps=24, bitrate_mbps=1))
    high = service.export_clip(_plan(sample_video, tmp_path, "high.mp4", seconds=4, fps=30, bitrate_mbps=20))

    assert "r_frame_rate=24/1" in _probe(low, "r_frame_rate") and "r_frame_rate=30/1" in _probe(high, "r_frame_rate")
    rate = lambda p: int(next(l for l in _probe(p, "codec_name").splitlines() if l.startswith("bit_rate=")).split("=")[1])
    assert rate(low) < 1_500_000                       # потолок 1 Мбит/с (+ звук) соблюдён
    assert rate(high) > 1.5 * rate(low)                # без потолка тот же кадр весит заметно больше
