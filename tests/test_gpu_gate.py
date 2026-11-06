"""Resource gate behavior with known desktop and unknown compute processes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

SPEC = importlib.util.spec_from_file_location("gpu_gate", Path(__file__).resolve().parents[1] / "scripts/run_when_idle.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def check(processes, free_mib=24000, available_ram=32 * 1024 ** 3, utilization=0):
    rows = [["0", "GPU-test", "A5000", "24564", str(free_mib), str(utilization)]]
    with mock.patch.object(gate, "query_csv", side_effect=[rows, processes]), mock.patch.object(gate.psutil, "virtual_memory", return_value=SimpleNamespace(available=available_ram)):
        return gate.snapshot("0", 8, 8, 5)


def test_known_desktop_only_is_eligible():
    result = check([["213186", gate.DESKTOP_EXECUTABLE, "11"]])
    assert result["eligible"]
    assert result["ignored_known_desktop_processes"][0]["pid"] == 213186


@pytest.mark.parametrize("processes", [
    [["123", "/home/user/python", "100"]],
    [["123", "/tmp/snapd-desktop-integration", "11"]],
    [["123", gate.DESKTOP_EXECUTABLE, "1000"]],
    [["123", gate.DESKTOP_EXECUTABLE, "11"], ["124", gate.DESKTOP_EXECUTABLE, "11"]],
])
def test_unknown_or_abnormal_compute_blocks_launch(processes):
    result = check(processes)
    assert not result["eligible"]
    assert "compute_process_present" in result["blockers"]


def test_insufficient_memory_and_gpu_activity_block_launch():
    result = check([], free_mib=100, available_ram=1024 ** 3, utilization=99)
    assert set(result["blockers"]) == {"gpu_memory", "host_memory", "gpu_utilization"}


def test_fully_busy_cpu_keeps_inherited_affinity():
    process = SimpleNamespace(cpu_affinity=lambda: [0, 1])
    with mock.patch.object(gate.psutil, "Process", return_value=process), mock.patch.object(gate.psutil, "cpu_percent", return_value=[100.0, 100.0]):
        policy = gate.cpu_policy()
    assert policy["selected_affinity"] is None
    assert policy["threads"] == 1
    assert policy["inherited_affinity"] == [0, 1]
