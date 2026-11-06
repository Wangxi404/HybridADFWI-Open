"""Per-worker GPU audit: preserve outputs, reject competing compute workloads."""
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybridadfwi import resources


def snapshot(processes=()):
    return {"compute_processes": list(processes), "eligible": not processes, "blockers": []}


def test_cpu_worker_needs_no_nvidia_and_keeps_exit_code(tmp_path):
    with (tmp_path / "worker.log").open("w") as out, mock.patch.object(resources, "gpu_snapshot", side_effect=AssertionError("CPU must not query NVIDIA")):
        evidence = resources.run_checked([sys.executable, "-c", "print('saved output')"], stdout=out, device="cpu", sample_interval=.01)
    assert evidence["status"] == "complete"
    assert evidence["returncode"] == 0
    assert (tmp_path / "worker.log").read_text().strip() == "saved output"


def test_only_launched_worker_and_descendants_are_excluded():
    item = snapshot([{"pid": 100}, {"pid": 101}, {"pid": 200}])
    assert resources._external_processes(item, {100, 101}) == [{"pid": 200}]


def test_unknown_compute_during_worker_marks_contaminated_without_killing_it(tmp_path):
    calls = 0
    def sampled(*args):
        nonlocal calls
        calls += 1
        return snapshot() if calls == 1 else snapshot([{"pid": 99999999, "executable": "/other/python"}])
    output = tmp_path / "finished"
    with mock.patch.object(resources, "gpu_snapshot", side_effect=sampled):
        evidence = resources.run_checked([sys.executable, "-c", f"import time,pathlib; time.sleep(.05); pathlib.Path({str(output)!r}).write_text('finished')"], device="cuda", stable_checks=1, sample_interval=.01)
    assert evidence["returncode"] == 0
    assert output.read_text() == "finished"
    assert evidence["status"] == "contaminated"
    assert evidence["contaminated"]
    assert any(s.get("external_compute_processes") for s in evidence["snapshots"])


def test_query_failure_prevents_launch():
    with mock.patch.object(resources, "gpu_snapshot", side_effect=RuntimeError("NVIDIA unavailable")), mock.patch.object(resources.subprocess, "Popen") as launch:
        evidence = resources.run_checked(["must-not-launch"], device="cuda", stable_checks=1)
    assert evidence["status"] == "monitor_error"
    assert evidence["returncode"] is None
    launch.assert_not_called()


def test_busy_gpu_timeout_does_not_launch_worker():
    with mock.patch.object(resources, "gpu_snapshot", side_effect=lambda *args: snapshot([{"pid": 42, "executable": "/other/python"}])), mock.patch.object(resources.subprocess, "Popen") as launch:
        evidence = resources.run_checked(["must-not-launch"], device="cuda", stable_checks=1, sample_interval=.001, idle_timeout=.001)
    assert evidence["status"] == "blocked"
    assert evidence["returncode"] is None
    launch.assert_not_called()
