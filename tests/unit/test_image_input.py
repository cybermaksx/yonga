"""Decoding the data-URL images ``yonga serve`` accepts for a VLM artifact.

Nothing here touches the network or a device: the URLs are refused or decoded
in memory, and the images are made by Pillow or written byte by byte when they
must lie about their size.
"""

from __future__ import annotations

import base64
import io
import struct
import sys
import zlib

import pytest

from yonga.adapters.runtime_genai import ChatImage, ChatMessage
from yonga.api.images import ImageRefused, decode_data_url, image_bytes, pillow_available
from yonga.domain.diagnostics import ConfigurationError, FailureCategory

Image = pytest.importorskip("PIL.Image")

LIMIT = 4096 * 4096
RED = (255, 0, 0)


def encoded(
    image_format: str,
    *,
    size: tuple[int, int] = (3, 2),
    color: object = RED,
    mode: str = "RGB",
    **save: object,
) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=image_format, **save)
    return buffer.getvalue()


def data_url(data: bytes, media: str = "png") -> str:
    return f"data:image/{media};base64,{base64.b64encode(data).decode('ascii')}"


def png_header(width: int, height: int) -> bytes:
    """A PNG that declares ``width`` x ``height`` but holds no pixel data at all."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


# --- what is accepted -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("image_format", "media"), [("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp"), ("GIF", "gif")]
)
def test_each_accepted_format_decodes_to_rgb_pixels(image_format: str, media: str) -> None:
    image = decode_data_url(data_url(encoded(image_format), media), max_pixels=LIMIT)

    assert (image.width, image.height) == (3, 2)
    assert len(image.pixels) == 3 * 2 * 3
    red, green, blue = image.pixels[:3]
    assert red > 200 and green < 60 and blue < 60, "lossy formats come close to pure red"


def test_the_media_type_is_matched_without_regard_to_case() -> None:
    url = "DATA:IMAGE/PNG;BASE64," + base64.b64encode(encoded("PNG")).decode("ascii")

    assert decode_data_url(url, max_pixels=LIMIT).width == 3


def test_transparency_is_shown_on_white() -> None:
    clear = encoded("PNG", mode="RGBA", color=(0, 0, 0, 0))

    image = decode_data_url(data_url(clear), max_pixels=LIMIT)

    assert image.pixels[:3] == b"\xff\xff\xff"


def test_an_animation_gives_its_first_frame() -> None:
    frames = [Image.new("RGB", (2, 2), color) for color in (RED, (0, 0, 255))]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])

    image = decode_data_url(data_url(buffer.getvalue(), "gif"), max_pixels=LIMIT)

    assert image.pixels[:3] == bytes(RED)


def test_an_exif_rotation_is_applied_as_a_browser_shows_it() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    photo = encoded("JPEG", size=(4, 2), exif=exif)

    image = decode_data_url(data_url(photo, "jpeg"), max_pixels=LIMIT)

    assert (image.width, image.height) == (2, 4)


def test_a_declared_type_that_does_not_match_the_bytes_still_decodes() -> None:
    """The media type only gates the URL; the bytes must be one of the four formats."""
    image = decode_data_url(data_url(encoded("JPEG"), "png"), max_pixels=LIMIT)

    assert image.width == 3


# --- what is refused ------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/cat.png",
        "http://127.0.0.1:8000/v1/models",
        "HTTPS://example.invalid/cat.png",
        "file:///etc/passwd",
        "ftp://example.invalid/cat.png",
        "//example.invalid/cat.png",
        "cat.png",
        "",
    ],
)
def test_anything_but_a_data_url_is_refused_without_being_fetched(url: str) -> None:
    with pytest.raises(ImageRefused) as refused:
        image_bytes(url)

    assert refused.value.unsupported is True
    assert "never fetches" in str(refused.value)
    if url:
        assert url not in str(refused.value), "request content is never echoed"


@pytest.mark.parametrize(
    "url",
    [
        "data:image/svg+xml;base64,PHN2Zz4=",
        "data:image/bmp;base64,Qk0=",
        "data:image/jpg;base64,AAAA",
        "data:text/plain;base64,aGk=",
        "data:image/png,rawbytes",
        "data:image/png;charset=utf-8;base64,AAAA",
    ],
)
def test_other_data_urls_are_refused(url: str) -> None:
    with pytest.raises(ImageRefused) as refused:
        image_bytes(url)

    assert refused.value.unsupported is True
    assert "PNG, JPEG, WebP or GIF" in str(refused.value)


@pytest.mark.parametrize(
    "url",
    ["data:image/png;base64,not base64!", "data:image/png;base64,AAA", "data:image/png;base64,"],
)
def test_broken_or_empty_base64_is_refused(url: str) -> None:
    with pytest.raises(ImageRefused) as refused:
        image_bytes(url)

    assert refused.value.unsupported is False


def test_an_image_over_the_pixel_limit_is_refused_before_it_is_decoded() -> None:
    """The header says 5000 x 5000 and there are no pixels: only the size was read."""
    with pytest.raises(ImageRefused, match="5000x5000 pixels, more than 4096x4096 pixels"):
        decode_data_url(data_url(png_header(5000, 5000)), max_pixels=LIMIT)


def test_a_decompression_bomb_is_refused() -> None:
    """Past Pillow's own ceiling, opening the image raises; that is a refusal too."""
    with pytest.raises(ImageRefused, match="larger than"):
        decode_data_url(data_url(png_header(20000, 20000)), max_pixels=LIMIT)


def test_an_image_at_the_limit_is_accepted() -> None:
    image = decode_data_url(data_url(encoded("PNG", size=(4, 4))), max_pixels=16)

    assert image.width == 4


@pytest.mark.parametrize(
    "data",
    [
        b"definitely not an image",
        encoded("PNG")[:40],  # truncated
        encoded("BMP"),  # a real image, in a format no decoder is offered
        encoded("TIFF"),
    ],
)
def test_bytes_that_are_not_an_accepted_image_are_refused(data: bytes) -> None:
    with pytest.raises(ImageRefused, match="could not be decoded"):
        decode_data_url(data_url(data), max_pixels=LIMIT)


def test_without_pillow_an_image_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PIL", "PIL.Image", "PIL.ImageOps"):
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ConfigurationError) as missing:
        decode_data_url(data_url(b"\x89PNG"), max_pixels=LIMIT)

    report = missing.value.diagnostic
    assert report.code == "serve.images.dependency_missing"
    assert report.category is FailureCategory.PACKAGE_ABI_MISMATCH
    assert "yonga[serve]" in " ".join(report.remediation)


def test_pillow_is_reported_available_here() -> None:
    assert pillow_available() is True


# --- the domain type ------------------------------------------------------------------


def test_an_image_must_hold_exactly_its_pixels() -> None:
    with pytest.raises(ValueError, match="RGB image"):
        ChatImage(width=2, height=2, pixels=b"\x00" * 11)


def test_pixels_never_show_in_a_repr_or_a_dump() -> None:
    image = ChatImage(width=1, height=1, pixels=b"\x01\x02\x03")
    message = ChatMessage(role="user", content="hi", images=[image])

    assert "pixels" not in repr(image)
    assert "images" not in message.model_dump()
    assert message.as_payload() == {"role": "user", "content": "hi"}
