"""Next-token log-probabilities straight from an artifact's IR, for ``yonga eval``.

OpenVINO GenAI generates text but does not expose logits, so evaluation runs
the exported IR directly with ``openvino.Core``. Two layouts are understood,
both as optimum-intel writes them:

* **LLM** -- ``openvino_model.xml`` taking ``input_ids``;
* **VLM** -- ``openvino_text_embeddings_model.xml`` turning ids into
  embeddings and ``openvino_language_model.xml`` taking ``inputs_embeds``.
  Text-only position ids are the plain sequence, repeated once per rotary
  section when the model declares several (Qwen3.5 declares four), exactly as
  optimum-intel feeds them.

Each call is one prefill from an empty state: the model reads the whole
passage at once and returns logits for every position, which is what a
teacher-forced comparison needs. Tokenisation uses the artifact's own
``openvino_tokenizer.xml``, never a Hugging Face tokenizer, so the ids are the
ones the runtime would use.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from yonga.adapters.native_output import capture_native_output, with_native_output
from yonga.domain.diagnostics import CompileError, FailureCategory, diagnostic
from yonga.util.redaction import redact_text

__all__ = ["LogitsModel", "log_softmax"]

#: CPU inference in full precision, so the measured difference is the
#: quantization's and not the arithmetic's.
CPU_PROPERTIES: dict[str, str] = {"INFERENCE_PRECISION_HINT": "f32"}

_LLM = "openvino_model.xml"
_VLM_LANGUAGE = "openvino_language_model.xml"
_VLM_EMBEDDINGS = "openvino_text_embeddings_model.xml"
_TOKENIZER = "openvino_tokenizer.xml"


def log_softmax(logits: npt.NDArray[Any], *, rows: int = 32) -> npt.NDArray[np.float32]:
    """Row-wise log-softmax in float64, returned as float32, ``rows`` at a time."""
    result = np.empty(logits.shape, dtype=np.float32)
    for start in range(0, logits.shape[0], rows):
        block = logits[start : start + rows].astype(np.float64)
        block -= block.max(axis=-1, keepdims=True)
        block -= np.log(np.exp(block).sum(axis=-1, keepdims=True))
        result[start : start + rows] = block
    return result


class LogitsModel:
    """One artifact, compiled for teacher-forced scoring on one device."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        device: str,
        properties: Mapping[str, str] | None = None,
    ) -> None:
        """Compile the artifact's tokenizer and numeric submodels on ``device``."""
        import openvino as ov
        import openvino_tokenizers  # noqa: F401  (registers the tokenizer operations)

        self._dir = Path(artifact_dir)
        self.device = device.upper()
        self.properties = dict(properties or (CPU_PROPERTIES if self.device == "CPU" else {}))
        core = ov.Core()
        failure: Exception | None = None
        # A plugin's compiler can write its real reason to the native
        # descriptors, which would otherwise land on the terminal -- and on
        # stdout, in the middle of a --json document.
        with capture_native_output() as native:
            try:
                self._compile(core)
            except Exception as error:
                failure = error
        if failure is not None:
            raw = redact_text(with_native_output(str(failure), native))
            reason = _last_line(redact_text(str(failure))) or type(failure).__name__
            raise CompileError(
                diagnostic(
                    "eval.load_failed",
                    FailureCategory.UNKNOWN_RUNTIME_ERROR,
                    f"Could not load the artifact on {self.device} for evaluation",
                    f"OpenVINO could not compile {self._dir.name} on {self.device}: {reason}",
                    remediation=["Try --device cpu, or check the artifact with 'yonga probe'."],
                    raw_excerpt=raw,
                )
            ) from failure
        self._inputs = {port.get_any_name(): port for port in self._model.inputs}
        self._request = self._model.create_infer_request()

    def _compile(self, core: Any) -> None:
        """Compile the tokenizer on CPU and the numeric submodels on the device."""
        self._tokenizer = core.compile_model(str(self._dir / _TOKENIZER), "CPU")
        if (self._dir / _VLM_LANGUAGE).is_file():
            self._embeddings: Any = core.compile_model(
                str(self._dir / _VLM_EMBEDDINGS), self.device, self.properties
            )
            self._model = core.compile_model(
                str(self._dir / _VLM_LANGUAGE), self.device, self.properties
            )
            self._ids_input = "inputs_embeds"
        elif (self._dir / _LLM).is_file():
            self._embeddings = None
            self._model = core.compile_model(str(self._dir / _LLM), self.device, self.properties)
            self._ids_input = "input_ids"
        else:
            raise FileNotFoundError(f"no {_LLM} or {_VLM_LANGUAGE} in {self._dir}")

    def tokenize(self, text: str) -> npt.NDArray[np.int64]:
        """The artifact's own token ids for ``text``, as a 1-D array."""
        ids = self._tokenizer([text])["input_ids"]
        return np.asarray(ids, dtype=np.int64).reshape(-1)

    def log_probs(self, ids: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """Log-probabilities ``[len(ids), vocab]`` of the next token at every position."""
        length = int(ids.shape[0])
        batch = ids.reshape(1, length)
        feed: dict[str, Any] = {"attention_mask": np.ones((1, length), dtype=np.int64)}
        if self._embeddings is not None:
            feed[self._ids_input] = self._embeddings([batch])[0]
        else:
            feed[self._ids_input] = batch
        if "position_ids" in self._inputs:
            feed["position_ids"] = self._positions(length)
        if "beam_idx" in self._inputs:
            feed["beam_idx"] = np.zeros(1, dtype=np.int32)
        self._request.reset_state()
        logits = self._request.infer(feed)["logits"]
        return log_softmax(np.asarray(logits)[0])

    def _positions(self, length: int) -> npt.NDArray[np.int64]:
        shape = self._inputs["position_ids"].get_partial_shape()
        sequence = np.arange(length, dtype=np.int64).reshape(1, length)
        if shape.rank.get_length() == 3:
            sections = shape[0].get_length() if shape[0].is_static else 3
            return np.repeat(sequence[None, :, :], sections, axis=0)
        return sequence

    def close(self) -> None:
        """Release the compiled models."""
        self._request = None
        self._model = None
        self._embeddings = None
        self._tokenizer = None


def _last_line(text: str) -> str:
    """The last non-blank line of ``text``: where an OpenVINO error states its reason."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""
