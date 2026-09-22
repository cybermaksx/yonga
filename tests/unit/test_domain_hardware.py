"""Unit tests for yonga.domain.hardware."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from yonga.domain.hardware import (
    DeviceKind,
    FilesystemInfo,
    HardwareInventory,
    HostInfo,
    HuggingFaceAuthInfo,
    MemoryInfo,
    NPUNodeInfo,
    OpenVINODevice,
    PackageInfo,
    device_kind,
    normalise_device,
)


def _inventory(*names: str) -> HardwareInventory:
    return HardwareInventory(devices=[OpenVINODevice(name=name) for name in names])


class TestNormaliseDevice:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("GPU.0", "GPU"),
            ("gpu", "GPU"),
            ("NPU", "NPU"),
            ("npu.0", "NPU"),
            ("CPU", "CPU"),
            (" cpu ", "CPU"),
            ("GPU.1", "GPU"),
            ("GPU.0.1", "GPU"),
            ("vpux.3", "VPUX"),
            ("some-device", "SOME-DEVICE"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert normalise_device(raw) == expected

    def test_is_idempotent(self) -> None:
        assert normalise_device(normalise_device("GPU.0")) == "GPU"

    def test_does_not_strip_a_non_index_suffix(self) -> None:
        assert normalise_device("AUTO:GPU,CPU") == "AUTO:GPU,CPU"


class TestDeviceKind:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("CPU", DeviceKind.CPU),
            ("gpu", DeviceKind.GPU),
            ("GPU.0", DeviceKind.GPU),
            ("NPU", DeviceKind.NPU),
            ("HETERO", DeviceKind.OTHER),
            ("", DeviceKind.OTHER),
        ],
    )
    def test_classification(self, raw: str, expected: DeviceKind) -> None:
        assert device_kind(raw) is expected


class TestOpenVINODevice:
    def test_kind_is_derived_from_the_name(self) -> None:
        device = OpenVINODevice(name="GPU.0", full_name="Intel Arc Graphics")
        assert device.kind is DeviceKind.GPU
        assert device.normalised_name == "GPU"
        assert device.is_available is True

    def test_explicit_kind_is_preserved(self) -> None:
        device = OpenVINODevice(name="MYDEV", kind=DeviceKind.NPU)
        assert device.kind is DeviceKind.NPU

    def test_unknown_names_stay_other(self) -> None:
        assert OpenVINODevice(name="HETERO").kind is DeviceKind.OTHER

    def test_defaults(self) -> None:
        device = OpenVINODevice(name="CPU")
        assert device.full_name == "unknown"
        assert device.architecture is None
        assert device.properties == {}


class TestAvailableDeviceNames:
    def test_canonical_ordering(self) -> None:
        inventory = _inventory("NPU", "GPU.0", "CPU")
        assert inventory.available_device_names == ["CPU", "GPU", "NPU"]

    def test_duplicates_are_collapsed(self) -> None:
        inventory = _inventory("GPU.0", "GPU.1", "CPU", "gpu")
        assert inventory.available_device_names == ["CPU", "GPU"]

    def test_other_devices_sort_alphabetically_after_the_known_ones(self) -> None:
        inventory = _inventory("ZETA", "NPU", "AUTO", "CPU", "MYRIAD")
        assert inventory.available_device_names == ["CPU", "NPU", "AUTO", "MYRIAD", "ZETA"]

    def test_missing_known_devices_are_skipped(self) -> None:
        assert _inventory("NPU").available_device_names == ["NPU"]

    def test_empty_inventory(self) -> None:
        assert HardwareInventory().available_device_names == []


class TestInventoryLookups:
    def test_device_lookup_is_normalised(self) -> None:
        inventory = _inventory("GPU.0", "CPU")
        found = inventory.device("gpu")
        assert found is not None
        assert found.name == "GPU.0"
        assert inventory.has_device("GPU.1") is True
        assert inventory.has_device("NPU") is False
        assert inventory.device("NPU") is None

    def test_exact_name_wins_over_normalised_match(self) -> None:
        inventory = _inventory("GPU.0", "GPU.1")
        found = inventory.device("GPU.1")
        assert found is not None
        assert found.name == "GPU.1"

    def test_blank_device_name_is_not_found(self) -> None:
        assert _inventory("CPU").device("  ") is None

    def test_package_lookup_tolerates_separators_and_case(self) -> None:
        inventory = HardwareInventory(
            packages=[
                PackageInfo(name="openvino", version="2026.4.0", importable=True),
                PackageInfo(name="openvino-genai", version="2026.4.0.0", importable=True),
            ]
        )
        found = inventory.package("OpenVINO_GenAI")
        assert found is not None
        assert found.version == "2026.4.0.0"
        assert inventory.package("nncf") is None

    def test_package_present_property(self) -> None:
        assert PackageInfo(name="nncf", version="3.4.0").present is True
        assert PackageInfo(name="nncf", importable=True).present is True
        assert PackageInfo(name="nncf", import_error="No module named nncf").present is False


class TestFilesystemInfo:
    def test_tmpfs_is_ram_backed(self) -> None:
        info = FilesystemInfo(path=Path("/tmp"), fstype="tmpfs", total_bytes=16 * 1024**3)
        assert info.is_tmpfs is True
        assert info.is_ram_backed is True

    def test_ext4_is_not_ram_backed(self) -> None:
        info = FilesystemInfo(path=Path("/home"), fstype="ext4")
        assert info.is_tmpfs is False
        assert info.is_ram_backed is False

    def test_devtmpfs_is_ram_backed_but_not_tmpfs(self) -> None:
        info = FilesystemInfo(path=Path("/dev"), fstype="DevTmpFS")
        assert info.is_tmpfs is False
        assert info.is_ram_backed is True

    def test_unknown_fstype_defaults(self) -> None:
        info = FilesystemInfo(path=Path("/"))
        assert info.fstype == "unknown"
        assert info.is_ram_backed is False
        assert info.free_bytes == 0


class TestNPUNodeInfo:
    def test_accessible_requires_read_and_write(self) -> None:
        node = NPUNodeInfo(
            directory_exists=True,
            node_path=Path("/dev/accel/accel0"),
            node_exists=True,
            readable=True,
            writable=True,
            group_name="render",
            user_in_group=True,
        )
        assert node.accessible is True

    def test_not_accessible_when_write_is_denied(self) -> None:
        node = NPUNodeInfo(node_exists=True, readable=True, writable=False)
        assert node.accessible is False

    def test_defaults_are_negative(self) -> None:
        node = NPUNodeInfo()
        assert node.accessible is False
        assert node.node_path is None
        assert node.user_groups == []


class TestHuggingFaceAuthInfo:
    def test_has_no_token_field(self) -> None:
        assert "token" not in HuggingFaceAuthInfo.model_fields
        leaky = {
            name
            for name in HuggingFaceAuthInfo.model_fields
            if "token" in name and name != "token_source"
        }
        assert leaky == set()

    def test_rejects_an_unexpected_token_keyword(self) -> None:
        with pytest.raises(PydanticValidationError):
            HuggingFaceAuthInfo(logged_in=True, token="hf_averyrealsecretvalue")  # type: ignore[call-arg]

    def test_rejects_a_credential_as_token_source(self) -> None:
        with pytest.raises(PydanticValidationError):
            HuggingFaceAuthInfo(logged_in=True, token_source="hf_averyrealsecretvalue")

    def test_accepts_a_source_label(self) -> None:
        info = HuggingFaceAuthInfo(logged_in=True, user="someone", token_source="environment")
        assert info.token_source == "environment"
        assert info.offline is False

    def test_blank_token_source_becomes_none(self) -> None:
        assert HuggingFaceAuthInfo(token_source="  ").token_source is None

    def test_serialisation_contains_no_secret(self) -> None:
        info = HuggingFaceAuthInfo(
            logged_in=True, user="someone", token_source="hf cache", cache_dir=Path("/tmp/hf")
        )
        dumped = info.model_dump(mode="json")
        assert set(dumped) == {"logged_in", "user", "token_source", "offline", "cache_dir"}


class TestInventoryDefaults:
    def test_inventory_is_constructible_without_arguments(self) -> None:
        inventory = HardwareInventory()
        assert inventory.host.platform == "unknown"
        assert inventory.memory.total_bytes == 0
        assert inventory.openvino_available is False
        assert inventory.npu is None
        assert inventory.hf_auth is None

    def test_full_inventory_round_trips_through_json(self) -> None:
        inventory = HardwareInventory(
            host=HostInfo(platform="Linux", machine="x86_64", kernel="7.2.6", distro_id="ubuntu"),
            memory=MemoryInfo(total_bytes=32 * 1024**3, available_bytes=16 * 1024**3),
            devices=[OpenVINODevice(name="CPU"), OpenVINODevice(name="NPU")],
            filesystems=[FilesystemInfo(path=Path("/tmp"), fstype="tmpfs")],
            openvino_available=True,
            device_tags=["intel.cpu", "intel.npu"],
            hf_auth=HuggingFaceAuthInfo(logged_in=False),
        )
        restored = HardwareInventory.model_validate(inventory.model_dump(mode="json"))
        assert restored == inventory
        assert restored.available_device_names == ["CPU", "NPU"]
