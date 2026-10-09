"""Images sent inline as ``data:`` URLs, decoded for a VLM pipeline.

Security posture: an image arrives inside the request or not at all. Only
``data:image/(png|jpeg|webp|gif);base64,`` URLs are accepted; ``http``,
``https``, ``file`` and every other scheme are refused, permanently and with no
option to allow them, because fetching an address the client chose would let
anyone who can reach the server make it connect to services on its own host or
network. Nothing in this module opens a socket or a file.

Decoding is bounded before it starts. Pillow is offered only the four formats
above, so no other decoder ever sees client bytes. An image's size is read from
its header and anything over
:data:`~yonga.application.serve.MAX_IMAGE_PIXELS` is refused before its pixels
are decoded; a larger one is not scaled down, so the model never sees something
other than what was sent. The encoded size is already bounded by the request
body limit. :func:`check_data_url` reads only that header, so the wire layer can
refuse a request before it waits for the pipeline; :func:`decode_data_url`
decodes the pixels, which the server does only once the request holds the
pipeline, so at most one request at a time holds decoded images. Both take CPU
time, so callers run them off the event loop. Any failure inside Pillow is a
refusal, never an error escaping to the server.

Pillow is part of the ``serve`` extra but imported only here, when an image is
actually decoded: without it, text requests are served as before and an image
request fails with a diagnostic that names the extra.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any

from yonga.application.chat import ChatImage
from yonga.domain.diagnostics import ConfigurationError, FailureCategory, diagnostic

__all__ = [
    "ImageRefused",
    "check_data_url",
    "check_image",
    "decode_data_url",
    "decode_image",
    "image_bytes",
    "pillow_available",
]

#: The prefix an accepted image URL starts with; the media type is captured.
_DATA_URL = re.compile(r"data:image/(png|jpeg|webp|gif);base64,", re.IGNORECASE)

#: The Pillow decoders an image may reach; no other plugin ever reads client bytes.
_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")

#: Image modes that carry transparency, flattened onto white rather than black.
_ALPHA_MODES = frozenset({"RGBA", "RGBa", "LA", "La", "PA"})

#: Greyscale modes wider than 8 bits (a 16-bit PNG opens as ``I;16``). Converting
#: one straight to RGB clips every value over 255 to white, so it is scaled first.
_WIDE_GREY_MODES = frozenset({"I", "I;16", "I;16L", "I;16B", "I;16N"})

#: The EXIF orientation tag, and the transposition each of its values asks for so
#: that the image is shown upright, as a browser shows it.
_ORIENTATION_TAG = 0x0112
_UPRIGHT = {
    2: "FLIP_LEFT_RIGHT",
    3: "ROTATE_180",
    4: "FLIP_TOP_BOTTOM",
    5: "TRANSPOSE",
    6: "ROTATE_270",
    7: "TRANSVERSE",
    8: "ROTATE_90",
}


class ImageRefused(ValueError):
    """An image the request cannot use, with the reason as its message.

    ``unsupported`` marks what yonga never accepts (a remote URL, another media
    type) apart from an accepted kind of image that is broken or too large. The
    wire layer adds the parameter name.
    """

    def __init__(self, message: str, *, unsupported: bool = False) -> None:
        super().__init__(message)
        self.unsupported = unsupported


def pillow_available() -> bool:
    """Whether Pillow is installed, so images can be decoded at all."""
    return find_spec("PIL") is not None


def image_bytes(url: str) -> bytes:
    """The encoded image inside a ``data:`` URL; refused unless it is one yonga accepts.

    The URL itself never appears in a message: it is request content.
    """
    match = _DATA_URL.match(url)
    if match is None:
        if url[:5].lower() != "data:":
            raise ImageRefused(
                "Only images embedded as data: URLs are accepted "
                "(data:image/png|jpeg|webp|gif;base64,...). yonga never fetches an image "
                "from a URL.",
                unsupported=True,
            )
        raise ImageRefused(
            "Only base64 data: URLs of PNG, JPEG, WebP or GIF images are accepted "
            "(data:image/png|jpeg|webp|gif;base64,...).",
            unsupported=True,
        )
    try:
        data = base64.b64decode(url[match.end() :], validate=True)
    except (binascii.Error, ValueError):
        raise ImageRefused("The image data is not valid base64.") from None
    if not data:
        raise ImageRefused("The image data is empty.")
    return data


def check_image(data: bytes, *, max_pixels: int) -> None:
    """Refuse ``data`` unless its header shows a PNG, JPEG, WebP or GIF within ``max_pixels``.

    Only the header is read; no pixel is decoded. A broken image can still pass
    and be refused by :func:`decode_image` later.
    """
    with _opened(_pillow(), data, max_pixels=max_pixels):
        pass


def decode_image(data: bytes, *, max_pixels: int) -> ChatImage:
    """Decode ``data`` (PNG, JPEG, WebP or GIF) into RGB pixels.

    The first frame of an animation is used, an EXIF rotation is applied the
    way a browser shows the image, transparency is flattened onto white and a
    16-bit greyscale image is scaled down to 8 bits.
    """
    image_module = _pillow()
    with _opened(image_module, data, max_pixels=max_pixels) as image:
        rgb = _flatten(image_module, _upright(image_module, image))
        return ChatImage(width=rgb.width, height=rgb.height, pixels=rgb.tobytes())


def check_data_url(url: str, *, max_pixels: int) -> None:
    """:func:`image_bytes` then :func:`check_image`."""
    check_image(image_bytes(url), max_pixels=max_pixels)


def decode_data_url(url: str, *, max_pixels: int) -> ChatImage:
    """:func:`image_bytes` then :func:`decode_image`."""
    return decode_image(image_bytes(url), max_pixels=max_pixels)


@contextmanager
def _opened(image_module: Any, data: bytes, *, max_pixels: int) -> Iterator[Any]:
    """``data`` opened by Pillow, its size checked; any failure inside is a refusal.

    Pillow's decoders raise more than ``OSError`` and ``ValueError`` on hostile
    input (a corrupt EXIF block has given ``struct.error``), so every exception
    but running out of memory becomes an :class:`ImageRefused`. Pillow's message
    may quote the bytes it choked on; it is not repeated.
    """
    try:
        with image_module.open(io.BytesIO(data), formats=_FORMATS) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise _too_large(width, height, max_pixels)
            yield image
    except ImageRefused:
        raise
    except image_module.DecompressionBombError:
        raise ImageRefused(f"The image is larger than {_limit(max_pixels)}.") from None
    except MemoryError:
        raise
    except Exception:
        raise ImageRefused("The image could not be decoded as PNG, JPEG, WebP or GIF.") from None


def _upright(image_module: Any, image: Any) -> Any:
    """``image`` turned the way its EXIF orientation says.

    The tag is read directly rather than through ``ImageOps.exif_transpose``,
    which writes the whole EXIF block back out and fails on a corrupt one. An
    EXIF block that cannot be read leaves the image as stored, which is what a
    browser shows too.
    """
    try:
        orientation = image.getexif().get(_ORIENTATION_TAG)
    except Exception:
        return image
    method = _UPRIGHT.get(orientation) if isinstance(orientation, int) else None
    if method is None:
        return image
    return image.transpose(getattr(image_module.Transpose, method))


def _flatten(image_module: Any, image: Any) -> Any:
    """``image`` as an RGB image, transparent areas shown on white."""
    if image.mode in _ALPHA_MODES or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = image_module.new("RGBA", rgba.size, (255, 255, 255, 255))
        return image_module.alpha_composite(background, rgba).convert("RGB")
    if image.mode in _WIDE_GREY_MODES:
        image = image.convert("I").point(lambda value: value * (1 / 257)).convert("L")
    return image.convert("RGB")


def _too_large(width: int, height: int, max_pixels: int) -> ImageRefused:
    return ImageRefused(
        f"The image is {width}x{height} pixels, more than {_limit(max_pixels)}. "
        "Scale it down before sending it."
    )


def _limit(max_pixels: int) -> str:
    """The pixel limit in words: ``4096x4096 pixels`` when it is a square."""
    side = int(max_pixels**0.5)
    return f"{side}x{side} pixels" if side * side == max_pixels else f"{max_pixels} pixels"


def _pillow() -> Any:
    """Pillow's ``Image`` module, or a diagnostic naming the extra."""
    try:
        from PIL import Image
    except ImportError as error:
        raise ConfigurationError(
            diagnostic(
                "serve.images.dependency_missing",
                FailureCategory.PACKAGE_ABI_MISMATCH,
                "Image input needs Pillow",
                f"Pillow is not installed ({error}); the serve extra provides it.",
                remediation=["pip install 'yonga[serve]'"],
            )
        ) from error
    return Image
