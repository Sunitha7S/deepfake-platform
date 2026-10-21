"""Preprocessing tests: header probing, caps, decoding, video metadata.

The header-probing tests matter more than they look. The pixel cap is only
meaningful if it fires *before* ``cv2.imdecode`` allocates, and the only way to
prove that is to hand the decoder a header claiming enormous dimensions with no
pixel data behind it. If the cap were checked after decode, that input would
either allocate or crash rather than raising cleanly.

VP8L and VP8X headers are hand-constructed rather than produced by cv2. That
isolates the parser from whatever WebP features the local OpenCV build happens
to support, and makes the test assert the byte layout rather than a round trip.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.preprocess import (  # noqa: E402
    DEFAULT_CAPS,
    MediaCaps,
    batch_clips,
    crop_face,
    decode_image,
    detect_faces,
    extract_frames,
    normalize,
    prepare_image,
    probe_image_header,
    probe_video,
    resize,
    sample_frame_indices,
    suffix_for,
    temp_media_file,
    to_rgb,
)
from inference.protocol import UnreadableMedia, UnsupportedMedia  # noqa: E402
from tests.detector_fixtures import haar_selection  # noqa: E402


def _encode(ext: str, size: int = 64, params=None) -> bytes:
    img = np.zeros((size, size, 3), np.uint8)
    img[:, : size // 2] = (10, 200, 30)
    ok, buf = cv2.imencode(ext, img, params or [])
    assert ok, f"cv2 could not encode {ext}"
    return buf.tobytes()


def _make_video(frames: int = 60, size: int = 128, fps: int = 25) -> bytes:
    path = Path(tempfile.mkdtemp()) / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(frames):
        frame = np.full((size, size, 3), (i * 3) % 255, np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


# --------------------------------------------------------------------------
# Header probing
# --------------------------------------------------------------------------


def test_probe_jpeg_header():
    header = probe_image_header(_encode(".jpg", 96))
    assert (header.width, header.height, header.format) == (96, 96, "jpeg")


def test_probe_png_header():
    header = probe_image_header(_encode(".png", 72))
    assert (header.width, header.height, header.format) == (72, 72, "png")


def test_probe_bmp_header():
    header = probe_image_header(_encode(".bmp", 48))
    assert (header.width, header.height, header.format) == (48, 48, "bmp")


def test_probe_webp_vp8_header():
    """Lossy WebP, produced by cv2 if the build supports it."""
    try:
        data = _encode(".webp", 80)
    except AssertionError:
        return  # build without WebP encode support; the hand-built cases still run
    header = probe_image_header(data)
    assert header.format == "webp"
    assert (header.width, header.height) == (80, 80)


def test_probe_webp_vp8l_header():
    """Hand-built VP8L: 14-bit width and height packed into a little-endian word."""
    w, h = 300, 200
    bits = (w - 1) | ((h - 1) << 14)
    chunk = b"VP8L" + struct.pack("<I", 5) + b"\x2f" + struct.pack("<I", bits)[:4]
    data = b"RIFF" + struct.pack("<I", len(chunk) + 4) + b"WEBP" + chunk
    header = probe_image_header(data)
    assert (header.width, header.height, header.format) == (w, h, "webp")


def test_probe_webp_vp8x_header():
    """Hand-built VP8X: dimensions are 24-bit, stored minus one."""
    w, h = 1024, 768
    chunk = (
        b"VP8X"
        + struct.pack("<I", 10)
        + b"\x00\x00\x00\x00"
        + (w - 1).to_bytes(3, "little")
        + (h - 1).to_bytes(3, "little")
    )
    data = b"RIFF" + struct.pack("<I", len(chunk) + 4) + b"WEBP" + chunk
    header = probe_image_header(data)
    assert (header.width, header.height, header.format) == (w, h, "webp")


def test_probe_gif_header():
    data = b"GIF89a" + struct.pack("<HH", 320, 240) + b"\x00" * 8
    header = probe_image_header(data)
    assert (header.width, header.height, header.format) == (320, 240, "gif")


def test_probe_rejects_unknown_format():
    try:
        probe_image_header(b"\x00\x01\x02\x03" * 8)
    except UnreadableMedia as exc:
        assert "Unrecognised" in str(exc)
        return
    raise AssertionError("an unknown format was probed successfully")


def test_probe_rejects_a_truncated_file():
    try:
        probe_image_header(b"\x89PNG")
    except UnreadableMedia:
        return
    raise AssertionError("a four-byte file was probed successfully")


def test_probe_rejects_a_truncated_png_ihdr():
    try:
        probe_image_header(b"\x89PNG\r\n\x1a\n" + b"\x00" * 6)
    except UnreadableMedia:
        return
    raise AssertionError("a PNG with no IHDR was accepted")


# --------------------------------------------------------------------------
# Caps applied before decode
# --------------------------------------------------------------------------


def _png_claiming(width: int, height: int) -> bytes:
    """A real PNG with its IHDR dimensions overwritten.

    The pixel data behind it is tiny, so anything that tried to decode this
    would fail rather than allocate — which is exactly what makes it a proof
    that the cap fires on the header.
    """
    data = bytearray(_encode(".png", 32))
    data[16:24] = struct.pack(">II", width, height)
    return bytes(data)


def test_pixel_cap_is_enforced_before_decode():
    bomb = _png_claiming(60_000, 60_000)
    header = probe_image_header(bomb)
    assert header.pixels > DEFAULT_CAPS.max_pixels
    try:
        decode_image(bomb)
    except UnsupportedMedia as exc:
        assert "before decoding" in (exc.detail or "")
        return
    raise AssertionError("an oversized image was decoded")


def test_pixel_cap_is_configurable():
    data = _encode(".png", 64)
    decode_image(data, caps=MediaCaps(max_pixels=64 * 64))
    try:
        decode_image(data, caps=MediaCaps(max_pixels=64 * 64 - 1))
    except UnsupportedMedia:
        return
    raise AssertionError("a tighter cap was ignored")


def test_byte_cap_is_enforced():
    data = _encode(".png", 64)
    try:
        decode_image(data, caps=MediaCaps(max_bytes=len(data) - 1))
    except UnsupportedMedia as exc:
        assert "larger than" in str(exc)
        return
    raise AssertionError("the byte cap was ignored")


def test_zero_dimension_header_is_rejected():
    try:
        decode_image(_png_claiming(0, 100))
    except UnreadableMedia as exc:
        assert "zero dimensions" in str(exc)
        return
    raise AssertionError("a zero-width image was accepted")


# --------------------------------------------------------------------------
# Image decoding
# --------------------------------------------------------------------------


def test_decode_returns_rgb_uint8():
    img = decode_image(_encode(".png", 40))
    assert img.shape == (40, 40, 3)
    assert img.dtype == np.uint8


def test_decode_converts_bgr_to_rgb():
    """A channel-order bug is invisible on grey test images and catastrophic on
    faces, so it is checked with a colour that distinguishes them."""
    pure_red = np.zeros((16, 16, 3), np.uint8)
    pure_red[:, :, 2] = 255  # BGR red
    data = cv2.imencode(".png", pure_red)[1].tobytes()
    rgb = decode_image(data)
    assert rgb[0, 0, 0] > 200 and rgb[0, 0, 2] < 50, "channels were not swapped"


def test_decode_rejects_empty_input():
    try:
        decode_image(b"")
    except UnreadableMedia as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("empty input was decoded")


def test_decode_rejects_a_valid_header_with_corrupt_body():
    data = bytearray(_encode(".png", 64))
    data[40:] = b"\x00" * (len(data) - 40)
    try:
        decode_image(bytes(data))
    except UnreadableMedia as exc:
        assert "could not be decoded" in str(exc)
        return
    raise AssertionError("a corrupt PNG body was decoded")


def test_grayscale_and_alpha_are_normalised_to_three_channels():
    grey = np.full((20, 20), 128, np.uint8)
    assert to_rgb(grey).shape == (20, 20, 3)
    rgba = np.zeros((20, 20, 4), np.uint8)
    assert to_rgb(rgba).shape == (20, 20, 3)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def test_resize_produces_a_square():
    img = np.zeros((100, 250, 3), np.uint8)
    assert resize(img, 64).shape == (64, 64, 3)


def test_normalize_produces_float32_centred_near_zero():
    out = normalize(np.full((8, 8, 3), 128, np.uint8))
    assert out.dtype == np.float32
    assert abs(float(out.mean())) < 1.0


def test_prepare_image_returns_a_single_item_batch():
    batch = prepare_image(np.zeros((100, 100, 3), np.uint8), size=32)
    assert batch.shape == (1, 32, 32, 3)
    assert batch.dtype == np.float32


def test_crop_face_stays_inside_the_image_and_is_never_empty():
    """Renamed in Batch 15. It was ``..._is_square_and_clipped_to_bounds``.

    **The crop is not always square**, and this test never checked that it was:
    the body asserts non-emptiness and containment, and the word "square" lived
    only in the name. The fixture below is the one corner where both axes clip by
    the same amount, so the crop came out square by coincidence — move the same
    box to ``(90, 40)`` and it is 16x52, a 3.25x stretch onto the model's square
    input. That is R23; ``tests/test_crop_geometry.py`` measures and pins it.
    """
    from inference.preprocess import FaceBox

    img = np.zeros((100, 100, 3), np.uint8)
    crop = crop_face(img, FaceBox(90, 90, 40, 40))
    assert crop.size > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_detect_faces_returns_empty_on_noise_rather_than_raising():
    """No face is a normal answer, not an error."""
    noise = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert isinstance(detect_faces(noise, policy=haar_selection()), list)


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------


def test_temp_media_file_is_removed_even_on_failure():
    seen: list[Path] = []
    try:
        with temp_media_file(b"data", suffix=".mp4") as path:
            seen.append(path)
            assert path.exists()
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not seen[0].exists(), "a temp file leaked when the body raised"


def test_suffix_is_chosen_from_the_mime_type():
    assert suffix_for("video/mp4") == ".mp4"
    assert suffix_for("video/webm") == ".webm"
    assert suffix_for("video/quicktime") == ".mov"
    assert suffix_for("application/unknown") == ".mp4"


def test_video_metadata_is_extracted():
    data = _make_video(frames=50, size=96, fps=25)
    with temp_media_file(data, suffix=".mp4") as path:
        meta = probe_video(path)
    assert meta.width == 96 and meta.height == 96
    assert meta.frame_count == 50
    assert abs(meta.fps - 25.0) < 1.0
    assert abs(meta.duration_s - 2.0) < 0.2


def test_duration_cap_is_enforced_from_metadata():
    data = _make_video(frames=200, fps=25)  # 8 seconds
    with temp_media_file(data, suffix=".mp4") as path:
        try:
            probe_video(path, caps=MediaCaps(max_duration_s=1.0))
        except UnsupportedMedia as exc:
            assert "beyond the" in str(exc)
            return
    raise AssertionError("the duration cap was ignored")


def test_resolution_cap_is_enforced_from_metadata():
    data = _make_video(frames=10, size=128)
    with temp_media_file(data, suffix=".mp4") as path:
        try:
            probe_video(path, caps=MediaCaps(max_pixels=64 * 64))
        except UnsupportedMedia as exc:
            assert "resolution" in str(exc)
            return
    raise AssertionError("the resolution cap was ignored")


def test_frame_count_cap_is_enforced():
    data = _make_video(frames=40)
    with temp_media_file(data, suffix=".mp4") as path:
        try:
            probe_video(path, caps=MediaCaps(max_frames=10))
        except UnsupportedMedia as exc:
            assert "frames" in str(exc)
            return
    raise AssertionError("the frame cap was ignored")


def test_malformed_video_is_rejected():
    try:
        extract_frames(b"\x00\x01\x02 not a video", mime="video/mp4")
    except UnreadableMedia:
        return
    raise AssertionError("garbage was accepted as a video")


def test_empty_video_is_rejected():
    try:
        extract_frames(b"", mime="video/mp4")
    except UnreadableMedia as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("empty input was accepted as a video")


def test_frames_are_extracted_and_shaped():
    data = _make_video(frames=60, size=96)
    meta, frames, indices, stamps = extract_frames(data, n_frames=16)
    assert frames.shape[0] <= 16
    assert frames.shape[1:] == (96, 96, 3)
    assert len(indices) == frames.shape[0]
    assert len(stamps) == frames.shape[0]
    assert meta.frame_count == 60


def test_frame_sampling_is_uniform_and_deterministic():
    a = sample_frame_indices(100, 10)
    assert a == sample_frame_indices(100, 10), "sampling is not reproducible"
    assert a == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert sample_frame_indices(5, 10) == [0, 1, 2, 3, 4], "fewer frames than asked for"
    assert sample_frame_indices(0, 10) == []


def test_frame_indices_never_exceed_the_frame_count():
    for count in (1, 7, 48, 999):
        for n in (1, 8, 48):
            assert max(sample_frame_indices(count, n), default=0) < count


def test_batch_clips_covers_every_frame_exactly_once():
    frames = np.arange(20).reshape(20, 1, 1, 1)
    batches = list(batch_clips(frames, 8))
    assert [b.shape[0] for b in batches] == [8, 8, 4]
    assert np.array_equal(np.concatenate(batches), frames)


def test_batch_clips_rejects_a_non_positive_size():
    try:
        list(batch_clips(np.zeros((4, 1, 1, 1)), 0))
    except ValueError:
        return
    raise AssertionError("a zero batch size was accepted")


def _run() -> int:
    tests = sorted((n, o) for n, o in globals().items() if n.startswith("test_") and callable(o))
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
