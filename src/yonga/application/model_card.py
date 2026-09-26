"""The model card ``yonga publish`` writes: generated from the manifest, never embellished.

Every sentence here is derived from a recorded fact -- the manifest's lineage,
export arguments, repair records and probe results -- and the wording is
constrained by ``SPEC.md`` section 21:

* a quantized conversion is described as one, and the card says in plain words
  that no weights were trained;
* a device appears as supported only when its probe passed; a failed probe is
  listed as failed with its diagnostic code, and a device never probed is
  listed as not tested, never as unsupported and never as working;
* every repair is disclosed with the files it changed and their digests;
* every file is listed with its SHA-256.

No local path appears anywhere: the card is public text.
"""

from __future__ import annotations

from collections.abc import Sequence

from yonga import __version__
from yonga.domain.artifacts import ArtifactManifest
from yonga.domain.models import PipelineKind, SourceRelationship
from yonga.domain.publication import (
    DeviceEvidence,
    LicenseAssessment,
    LicenseClass,
    PublishFile,
)

__all__ = ["render_model_card"]

#: The Hub's pipeline tags for the tasks optimum-intel exports.
_PIPELINE_TAGS: dict[str, str] = {
    "text-generation": "text-generation",
    "text-generation-with-past": "text-generation",
    "image-text-to-text": "image-text-to-text",
}

_RELATIONSHIP_WORDS: dict[SourceRelationship, str] = {
    SourceRelationship.ORIGINAL: "the requested model itself",
    SourceRelationship.CONVERTED: "a format conversion of the requested model",
    SourceRelationship.LOSSLESS_CONVERSION: (
        "a lossless format conversion of the requested model (the same weights)"
    ),
    SourceRelationship.QUANTIZED: (
        "an already quantized derivative of the requested model, so this artifact is "
        "compressed from reduced-precision weights, not from the original ones"
    ),
}

_SOFTWARE = (
    "openvino",
    "openvino-tokenizers",
    "openvino-genai",
    "optimum-intel",
    "optimum",
    "nncf",
    "transformers",
    "torch",
)


def render_model_card(
    manifest: ArtifactManifest,
    *,
    repo_id: str,
    licenses: Sequence[LicenseAssessment],
    devices: Sequence[DeviceEvidence],
    files: Sequence[PublishFile],
    license_files: Sequence[str] = (),
    profile_device: str | None = None,
    license_reviewed: bool = False,
) -> str:
    """Render the ``README.md`` for a publication of ``manifest``'s artifact."""
    source = manifest.resolved_source
    requested = manifest.requested_model
    arguments = manifest.export.arguments
    supported = [d.device for d in devices if d.status == "ok"]
    source_license = next((a.license for a in licenses if a.role == "source"), None)

    lines: list[str] = ["---"]
    lines.append(f"license: {source_license or 'other'}")
    lines.append(f"base_model: {source.repo_id}")
    lines.append("base_model_relation: quantized")
    lines.append("library_name: openvino")
    tag = _PIPELINE_TAGS.get(manifest.task or "")
    if tag:
        lines.append(f"pipeline_tag: {tag}")
    lines.append("tags:")
    for value in ("openvino", "openvino-genai", "nncf", _weight_tag(arguments), "yonga"):
        if value:
            lines.append(f"  - {value}")
    lines += ["---", ""]

    name = repo_id.rsplit("/", 1)[-1]
    lines += [f"# {name}", ""]
    lines.append(
        f"OpenVINO IR of [{source.repo_id}](https://huggingface.co/{source.repo_id}) "
        f"at revision `{_short(source.resolved_revision)}`, weight-compressed to "
        f"{_quantization_words(arguments)} with NNCF through optimum-intel."
    )
    lines.append("")
    lines.append(
        "**This is a quantized conversion, not a fine-tune.** No weights were trained or "
        "otherwise changed beyond compression, so the model's behaviour, abilities and "
        "limitations are those of the source model, reduced by quantization."
    )
    lines.append("")
    if supported:
        lines.append(f"Tested on: **{', '.join(supported)}** (see below for the evidence).")
    lines.append("")
    lines.append(f"Prepared with [yonga](https://github.com/ooguz/yonga) {__version__}.")
    lines.append("")

    lines += [
        "## Source lineage",
        "",
        "| Role | Repository | Revision | License |",
        "|---|---|---|---|",
    ]
    by_role = {assessment.role: assessment for assessment in licenses}
    lines.append(
        f"| Requested model | [{requested.repo_id}](https://huggingface.co/{requested.repo_id}) "
        f"| `{requested.resolved_revision or 'unknown'}` "
        f"| {_license_cell(by_role.get('requested'))} |"
    )
    if source.repo_id != requested.repo_id:
        lines.append(
            f"| Converted weights | [{source.repo_id}](https://huggingface.co/{source.repo_id}) "
            f"| `{source.resolved_revision or 'unknown'}` "
            f"| {_license_cell(by_role.get('source'))} |"
        )
    lines.append("")
    relationship = source.relationship
    if source.repo_id != requested.repo_id and relationship is not None:
        words = _RELATIONSHIP_WORDS.get(relationship, relationship.value.replace("_", " "))
        lines.append(f"The converted repository is {words}.")
        lines.append("")

    lines += ["## Export", "", "| Setting | Value |", "|---|---|"]
    rows = [
        ("Profile", f"`{manifest.export.profile}`"),
        ("Weight format", str(arguments.get("weight_format", "unknown"))),
        ("Symmetric", _flag(arguments.get("symmetric"))),
        ("Group size", _group(arguments.get("group_size"))),
        ("Ratio", str(arguments.get("ratio", "unknown"))),
        ("AWQ", _flag(arguments.get("awq"))),
        ("Scale estimation", _flag(arguments.get("scale_estimation"))),
        ("Calibration dataset", str(arguments.get("dataset") or "none (data-free)")),
        ("Task", f"`{manifest.task or arguments.get('task') or 'unknown'}`"),
        ("Remote code", "used" if arguments.get("trust_remote_code") else "not used"),
        ("Exporter", manifest.export.backend),
    ]
    lines += [f"| {key} | {value} |" for key, value in rows]
    lines.append("")
    versions = manifest.export.package_versions
    known = [(package, versions[package]) for package in _SOFTWARE if versions.get(package)]
    if known:
        lines += ["Exported with:", ""]
        lines += [f"- `{package}` {version}" for package, version in known]
        lines.append("")
    if manifest.export.environment_key:
        lines.append(
            "The export ran in an isolated environment with the package versions above, "
            "because the source architecture needs them."
        )
        lines.append("")

    lines += ["## Repairs applied", ""]
    applied = [record for record in manifest.repairs if record.applied]
    if not applied:
        lines += ["None. The files are exactly what the exporter wrote.", ""]
    else:
        lines.append(
            "The exporter's output was changed after export, by these narrow, versioned "
            "repairs. Each is listed with the digests of the files before and after."
        )
        lines.append("")
        for record in applied:
            lines.append(f"### `{record.repair_id}`")
            lines.append("")
            if record.description:
                lines += [record.description, ""]
            lines += ["| File | SHA-256 before | SHA-256 after |", "|---|---|---|"]
            for filename in record.files:
                before = record.before_hashes.get(filename, "")
                after = record.after_hashes.get(filename, "")
                if before or after:
                    lines.append(f"| `{filename}` | `{before}` | `{after}` |")
            lines.append("")

    lines += ["## Tested hardware and software", ""]
    lines += ["| Device | Result | Evidence | Date |", "|---|---|---|---|"]
    for evidence in devices:
        result = {"ok": "works", "failed": "**failed**", "not_tested": "not tested"}.get(
            evidence.status, evidence.status
        )
        lines.append(
            f"| {evidence.device} | {result} | {evidence.detail or '—'} "
            f"| {evidence.attempted_at or '—'} |"
        )
    lines.append("")
    host = manifest.host
    host_rows = [
        ("CPU", host.get("cpu")),
        ("Devices present", host.get("devices")),
        ("OS", host.get("distro")),
        ("Kernel", host.get("kernel")),
        ("OpenVINO", host.get("openvino")),
        ("OpenVINO Tokenizers", host.get("openvino_tokenizers")),
    ]
    present = [(key, value) for key, value in host_rows if value]
    if present:
        lines += ["Probed on:", ""]
        lines += [f"- {key}: {value}" for key, value in present]
        lines.append("")
    failed = [d for d in devices if d.status == "failed"]
    untested = [d for d in devices if d.status == "not_tested"]
    if failed or untested:
        lines += ["### Not supported or not tested", ""]
        lines += [f"- **{d.device}: failed.** {d.detail}" for d in failed]
        lines += [
            f"- **{d.device}: not tested.** No probe was run; nothing is claimed." for d in untested
        ]
        lines.append("")
    if profile_device and profile_device not in supported:
        lines.append(
            f"The `{manifest.export.profile}` profile was designed for {profile_device}, but "
            f"this artifact is only verified on {', '.join(supported) or 'no device'}."
        )
        lines.append("")

    lines += ["## Usage", "", "```python"]
    pipeline = "VLMPipeline" if manifest.pipeline is PipelineKind.VLM else "LLMPipeline"
    device = supported[0] if supported else "CPU"
    lines += [
        "from huggingface_hub import snapshot_download",
        "import openvino_genai",
        "",
        f'path = snapshot_download("{repo_id}")',
        f'pipe = openvino_genai.{pipeline}(path, "{device}")',
        "history = openvino_genai.ChatHistory()",
        'history.append({"role": "user", "content": "Hello!"})',
        "print(pipe.generate(history, max_new_tokens=128))",
        "```",
        "",
        "Or with yonga, which applies the same chat profile the artifact was probed with:",
        "",
        "```bash",
        f"hf download {repo_id} --local-dir {name}",
        f"yonga chat ./{name} --device {device.lower()}",
        "```",
        "",
    ]

    lines += ["## Files", "", "| File | Size | SHA-256 |", "|---|---|---|"]
    lines += [f"| `{f.path_in_repo}` | {f.size} | `{f.sha256}` |" for f in files]
    lines.append("")
    lines.append(
        "`yonga-manifest.json` is the complete preparation record: export command, "
        "validation, probes and repairs. Local paths in it are anonymised."
    )
    lines.append("")

    lines += ["## License", ""]
    worst = _worst(licenses)
    lines.append(
        f"The weights are distributed under the source model's license "
        f"(`{source_license or 'unspecified'}`)."
    )
    if worst is LicenseClass.CONDITIONAL:
        lines.append(
            "That license attaches terms to redistribution and use; they apply to this "
            "conversion exactly as to the original."
        )
    if license_files:
        lines.append(
            "The source's license files are included unchanged: "
            + ", ".join(f"`{name}`" for name in license_files)
            + "."
        )
    if license_reviewed:
        lines.append(
            "yonga could not classify every license involved; the publisher stated that they "
            "reviewed them and that they permit redistributing a quantized derivative."
        )
    lines.append("")
    return "\n".join(lines)


def _short(revision: str | None) -> str:
    return (revision or "unknown")[:12]


def _flag(value: object) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _group(value: object) -> str:
    if value in (None, -1):
        return "per channel"
    return str(value)


def _weight_tag(arguments: dict[str, object]) -> str:
    return str(arguments.get("weight_format") or "").strip().lower()


def _quantization_words(arguments: dict[str, object]) -> str:
    weight = str(arguments.get("weight_format") or "unknown format").upper()
    parts = [weight]
    symmetric = arguments.get("symmetric")
    if symmetric is not None:
        parts.append("symmetric" if symmetric else "asymmetric")
    parts.append(f"group size {_group(arguments.get('group_size'))}")
    return f"{parts[0]} ({', '.join(parts[1:])})"


def _license_cell(assessment: LicenseAssessment | None) -> str:
    if assessment is None:
        return "unknown"
    return f"`{assessment.license}`" if assessment.license else "not declared"


def _worst(licenses: Sequence[LicenseAssessment]) -> LicenseClass:
    order = [
        LicenseClass.PERMISSIVE,
        LicenseClass.CONDITIONAL,
        LicenseClass.UNKNOWN,
        LicenseClass.RESTRICTED,
    ]
    worst = LicenseClass.PERMISSIVE
    for assessment in licenses:
        if order.index(assessment.classification) > order.index(worst):
            worst = assessment.classification
    return worst
