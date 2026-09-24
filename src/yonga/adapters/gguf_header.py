"""Reading a GGUF file's header: its metadata and the type of every tensor.

Whether OpenVINO GenAI can read a GGUF file is not a property of its name.
A ``Q4_K_M`` file is a mixture: most tensors are ``Q4_K``, but some are
``Q6_K``, ``Q5_0`` or ``F32`` depending on the model and the quantizer. The
GenAI reader converts only some tensor types, so on the reference host a
Llama ``Q4_K_M`` imports and a Qwen2.5 ``Q4_K_M`` fails with
``gguf_tensor_to_f16 failed``. The only honest answer comes from the tensor
table, which sits in the header, ahead of the weights.

This module parses that header from any ``read(offset, length)`` function, so
the same code reads a local file or a remote one through HTTP range requests
without downloading gigabytes of weights. It keeps scalar metadata, skips the
contents of arrays (a tokenizer vocabulary is an array of a hundred thousand
strings), and counts tensors per type. Every length it reads is bounded, so a
malformed or hostile file cannot make it allocate without limit.
"""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GGML_TYPES",
    "GGUFFormatError",
    "GGUFHeader",
    "parse_gguf_header",
    "read_local_header",
]

#: ``ggml_type`` ids to their names, as the GGUF specification numbers them.
GGML_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}

_MAGIC = b"GGUF"
_MAX_HEADER_BYTES = 256 * 2**20
_MAX_STRING = 16 * 2**20
_MAX_TENSORS = 1_000_000
_MAX_KV = 100_000
_MAX_DIMS = 8

# value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SCALAR: dict[int, str] = {
    _U8: "<B",
    _I8: "<b",
    _U16: "<H",
    _I16: "<h",
    _U32: "<I",
    _I32: "<i",
    _F32: "<f",
    _BOOL: "<?",
    _U64: "<Q",
    _I64: "<q",
    _F64: "<d",
}

Reader = Callable[[int, int], bytes]
"""``read(offset, length)``: up to ``length`` bytes starting at ``offset``."""


class GGUFFormatError(ValueError):
    """The bytes are not a GGUF header this parser can read."""


class GGUFHeader(BaseModel):
    """What the header says about a GGUF file."""

    model_config = ConfigDict(frozen=True)

    version: int
    tensor_count: int
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    """Scalar key-value pairs; arrays are recorded only as ``<key>.length``."""
    tensor_types: dict[str, int] = Field(default_factory=dict)
    """Tensor type name to how many tensors have it."""
    header_bytes: int = 0

    @property
    def architecture(self) -> str | None:
        """``general.architecture``: the name the GGUF model builder dispatches on."""
        value = self.metadata.get("general.architecture")
        return str(value) if value is not None else None

    def unsupported_types(self, supported: set[str] | frozenset[str]) -> list[str]:
        """Tensor types present in the file but not in ``supported``, sorted."""
        return sorted(name for name in self.tensor_types if name not in supported)


class _Cursor:
    """Sequential reads over a :data:`Reader`, fetching in chunks."""

    def __init__(self, read: Reader, chunk: int) -> None:
        self._read = read
        self._chunk = chunk
        self._buffer = b""
        self._start = 0
        self.position = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.position + size > _MAX_HEADER_BYTES:
            raise GGUFFormatError("the header is larger than any real GGUF header")
        end = self.position + size
        while self._start + len(self._buffer) < end:
            fetch_from = self._start + len(self._buffer)
            data = self._read(fetch_from, max(self._chunk, end - fetch_from))
            if not data:
                raise GGUFFormatError("the file ends inside its own header")
            self._buffer += data
        offset = self.position - self._start
        piece = self._buffer[offset : offset + size]
        self.position = end
        if offset > 4 * self._chunk:  # drop consumed bytes; headers can be tens of MB
            self._buffer = self._buffer[offset + size :]
            self._start = end
        return piece

    def unpack(self, fmt: str) -> int | float | bool:
        (value,) = struct.unpack(fmt, self.take(struct.calcsize(fmt)))
        return value  # type: ignore[no-any-return]

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def string(self) -> str:
        length = self.u64()
        if length > _MAX_STRING:
            raise GGUFFormatError("a header string is implausibly long")
        return self.take(length).decode("utf-8", errors="replace")


class _Array:
    """An array value that was skipped, remembered only by its length."""

    def __init__(self, count: int) -> None:
        self.count = count


def _value(cursor: _Cursor, kind: int) -> str | int | float | bool | _Array:
    """Read one value; an array's contents are skipped and only its length kept."""
    if kind in _SCALAR:
        return cursor.unpack(_SCALAR[kind])
    if kind == _STR:
        return cursor.string()
    if kind == _ARR:
        item = cursor.u32()
        count = cursor.u64()
        if item in _SCALAR:
            cursor.take(struct.calcsize(_SCALAR[item]) * count)
        elif item == _STR:
            for _ in range(count):
                length = cursor.u64()
                if length > _MAX_STRING:
                    raise GGUFFormatError("a header string is implausibly long")
                cursor.take(length)
        else:
            for _ in range(count):
                _value(cursor, item)
        return _Array(count)
    raise GGUFFormatError(f"unknown metadata value type {kind}")


def parse_gguf_header(read: Reader, *, chunk: int = 2 * 2**20) -> GGUFHeader:
    """Parse the header behind ``read`` and count the tensors of each type."""
    cursor = _Cursor(read, chunk)
    if cursor.take(4) != _MAGIC:
        raise GGUFFormatError("not a GGUF file (bad magic)")
    version = cursor.u32()
    if version not in (2, 3):
        raise GGUFFormatError(f"GGUF version {version} is not supported")
    tensor_count = cursor.u64()
    kv_count = cursor.u64()
    if tensor_count > _MAX_TENSORS or kv_count > _MAX_KV:
        raise GGUFFormatError("implausible tensor or metadata count")
    metadata: dict[str, str | int | float | bool] = {}
    for _ in range(kv_count):
        key = cursor.string()
        value = _value(cursor, cursor.u32())
        if isinstance(value, _Array):
            metadata[f"{key}.length"] = value.count
        else:
            metadata[key] = value
    counts: Counter[str] = Counter()
    for _ in range(tensor_count):
        cursor.string()  # name
        dims = cursor.u32()
        if dims > _MAX_DIMS:
            raise GGUFFormatError("a tensor declares an implausible number of dimensions")
        cursor.take(8 * dims)
        type_id = cursor.u32()
        cursor.u64()  # data offset
        counts[GGML_TYPES.get(type_id, f"type_{type_id}")] += 1
    return GGUFHeader(
        version=version,
        tensor_count=tensor_count,
        metadata=metadata,
        tensor_types=dict(sorted(counts.items())),
        header_bytes=cursor.position,
    )


def read_local_header(path: Path) -> GGUFHeader:
    """Parse the header of a GGUF file on disk."""
    with path.open("rb") as handle:

        def read(offset: int, length: int) -> bytes:
            handle.seek(offset)
            return handle.read(length)

        return parse_gguf_header(read)
