#!/usr/bin/env python3
"""Audit full-shot orchestration metadata, not gradient mathematics.

SPDX-License-Identifier: MIT
JSONL schema: references/full-shot-inversion.md. No solver/framework dependency.
"""
import argparse
import json
from pathlib import Path
import re


class AuditError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _fingerprint(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "model fingerprint must be a lowercase SHA-256")
    return value


def _shots(values):
    _require(isinstance(values, list) and bool(values), "shot_ids must be nonempty")
    _require(all(type(v) in (str, int) for v in values), "shot IDs must be strings or integers")
    keys = {(type(v).__name__, v) for v in values}
    _require(len(keys) == len(values), "duplicate shot IDs")
    return keys


def audit_events(events):
    iterations = {}
    dataset = None
    current_model = None
    for index, row in enumerate(events):
        try:
            event, iteration_id = row["event"], str(row["iteration_id"])
            if event == "iteration_start":
                _require(iteration_id not in iterations, "iteration/attempt ID reused")
                _require(all(s["ended"] for s in iterations.values()), "next iteration started before the current one ended")
                shots = _shots(row["shot_ids"])
                if dataset is None:
                    dataset = shots
                _require(shots == dataset, "declared full dataset changed between iterations")
                _require(row["reduction"] in ("sum", "mean"), "unknown reduction")
                _require(isinstance(row["objective_id"], str) and bool(row["objective_id"]), "objective identifier is required")
                model = _fingerprint(row["model_sha256"])
                if current_model is None:
                    current_model = model
                _require(model == current_model, "iteration does not start from the last accepted model")
                iterations[iteration_id] = dict(shots=shots, model=model,
                    reduction=row["reduction"], objective=row["objective_id"], evaluations={}, updates=0, ended=False)
                continue
            _require(iteration_id in iterations, "event precedes iteration_start")
            state = iterations[iteration_id]
            _require(not state["ended"], "event follows iteration_end")
            if event == "evaluation_start":
                eid = str(row["evaluation_id"])
                _require(eid not in state["evaluations"], "evaluation ID reused")
                _require(row["phase"] in ("gradient", "closure", "trial"), "unknown evaluation phase")
                _require(row["objective_id"] == state["objective"], "objective changed within an iteration")
                state["evaluations"][eid] = dict(model=_fingerprint(row["model_sha256"]), phase=row["phase"],
                    seen=set(), complete=False, aborted=False)
            elif event in ("shot_batch", "evaluation_complete", "evaluation_abort"):
                eid = str(row["evaluation_id"])
                _require(eid in state["evaluations"], "unknown evaluation")
                evaluation = state["evaluations"][eid]
                _require(not evaluation["complete"] and not evaluation["aborted"], "evaluation already closed")
                if event == "evaluation_abort":
                    _require(bool(row.get("reason")), "aborted evaluation needs a reason")
                    evaluation["aborted"] = True
                    continue
                _require(_fingerprint(row["model_sha256"]) == evaluation["model"], "model changed between shots")
                if event == "shot_batch":
                    shots = _shots(row["shot_ids"])
                    _require(shots <= state["shots"], "undeclared shot")
                    _require(not shots & evaluation["seen"], "shot contributed more than once")
                    evaluation["seen"].update(shots)
                else:
                    _require(evaluation["seen"] == state["shots"], "full reduction completed before all shots")
                    expected = 1 if state["reduction"] == "sum" else len(state["shots"])
                    _require(type(row["normalizer"]) in (int, float) and row["normalizer"] == expected, "sum/mean normalization mismatch")
                    evaluation["complete"] = True
            elif event == "accepted_update":
                _require(state["updates"] == 0, "multiple accepted updates in one iteration")
                _require(_fingerprint(row["model_before_sha256"]) == state["model"], "update anchor changed")
                eid = str(row["gradient_evaluation_id"])
                _require(eid in state["evaluations"], "update has no gradient evaluation")
                evaluation = state["evaluations"][eid]
                _require(evaluation["complete"] and evaluation["phase"] in ("gradient", "closure") and
                         evaluation["model"] == state["model"], "update precedes the complete current-model gradient")
                _require(all(e["complete"] or e["aborted"] for e in state["evaluations"].values()),
                         "update while another evaluation is incomplete")
                after = _fingerprint(row["model_after_sha256"])
                if "candidate_evaluation_id" in row:
                    candidate = state["evaluations"].get(str(row["candidate_evaluation_id"]))
                    _require(candidate is not None and candidate["complete"] and candidate["model"] == after,
                             "accepted candidate lacks its complete full-shot evaluation")
                state["updates"] += 1
                current_model = after
            elif event == "iteration_end":
                status = row["status"]
                _require(status in ("accepted", "converged", "rejected", "failed", "interrupted"), "unknown iteration status")
                _require(state["updates"] == (1 if status == "accepted" else 0), "status disagrees with update count")
                _require(all(e["complete"] or e["aborted"] for e in state["evaluations"].values()), "open evaluation at iteration_end")
                if status in ("converged", "rejected"):
                    _require(any(e["complete"] and e["phase"] in ("gradient", "closure") and e["model"] == state["model"]
                                 for e in state["evaluations"].values()), "stopping without a complete current-model gradient")
                state.update(ended=True, status=status)
            else:
                raise AuditError("unknown event " + str(event))
        except (KeyError, TypeError, AuditError) as error:
            raise AuditError(f"event {index}: {error}") from error
    _require(bool(iterations), "no iterations recorded")
    complete = all(s["ended"] for s in iterations.values())
    return dict(passed=complete, status="PASS" if complete else "INCOMPLETE", complete=complete,
                iterations=len(iterations), accepted_updates=sum(s["updates"] for s in iterations.values()),
                shot_count=len(dataset), iteration_statuses={key:sum(s.get("status")==key for s in iterations.values())
                    for key in ("accepted", "converged", "rejected", "failed", "interrupted")},
                scope="Orchestration metadata only; not a numerical-gradient or inversion-convergence certificate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    try:
        events = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
        report = audit_events(events)
    except (AuditError, ValueError, OSError) as error:
        print(json.dumps(dict(passed=False, status="FAIL", error=str(error))))
        return 1
    print(json.dumps(report))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
