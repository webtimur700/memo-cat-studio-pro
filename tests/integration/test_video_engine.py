from pathlib import Path

from video.ffmpeg_wrapper import FFmpegWrapper, TranscodeOptions
from video.frame_extractor import FrameExtractor
from video.ingestion_service import IngestionService


def test_ingestion_reads_real_metadata(sample_video: Path):
    service = IngestionService()
    source = service.ingest(sample_video)

    assert source.width == 1280
    assert source.height == 720
    assert abs(source.duration_sec - 20.0) < 0.5
    assert source.video_codec == "h264"
    assert source.has_audio is True


def test_ingest_many_separates_valid_and_invalid(sample_video: Path, tmp_path: Path):
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("not a video")
    missing_file = tmp_path / "missing.mp4"

    service = IngestionService()
    accepted, rejected = service.ingest_many([sample_video, bad_file, missing_file])

    assert len(accepted) == 1
    assert accepted[0].path == sample_video
    assert len(rejected) == 2


def test_extract_segment_produces_correct_resolution(sample_video: Path, tmp_path: Path):
    wrapper = FFmpegWrapper()
    output = wrapper.extract_segment(
        sample_video,
        start_sec=3.0,
        end_sec=8.0,
        destination=tmp_path / "segment.mp4",
        options=TranscodeOptions(fps=30, width=1080, height=1920, bitrate_mbps=6),
    )

    assert output.exists()
    result_source = wrapper.get_video_source(output)
    assert result_source.width == 1080
    assert result_source.height == 1920
    assert abs(result_source.duration_sec - 5.0) < 0.3


def test_frame_extractor_reads_expected_frames(sample_video: Path):
    with FrameExtractor(sample_video) as extractor:
        assert extractor.width == 1280
        assert extractor.height == 720

        frame = extractor.frame_at(5.0)  # в синем сегменте
        assert frame.shape == (720, 1280, 3)
        # BGR: синий канал должен доминировать в первые 10 секунд
        assert frame[:, :, 0].mean() > frame[:, :, 2].mean()

        frames = list(extractor.frames_in_range(2.0, 4.0, sample_fps=5.0))
        assert len(frames) == 10
