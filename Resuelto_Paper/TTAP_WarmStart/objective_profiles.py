"""Explicit, fingerprinted task-value definitions for the TTAP study.

The multidimensional definition assumes med, pers and load are three delivered
outputs. Feasibility still consumes exactly the original resource vector.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math


DIMENSIONS = ("medical", "personnel", "cargo")
FIELDS = {"medical": "med", "personnel": "pers", "cargo": "load"}
DEFAULT_WEIGHTS = {"medical": 2.0, "personnel": 1.5, "cargo": 1.0}
DEFAULT_REFERENCES = {"medical": 3.0, "personnel": 3.0, "cargo": 500.0}


@dataclass(frozen=True)
class ObjectiveSpec:
    mode: str = "multidimensional"
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    references: dict = field(default_factory=lambda: DEFAULT_REFERENCES.copy())

    def __post_init__(self):
        if self.mode not in ("primary", "multidimensional"):
            raise ValueError("mode must be primary or multidimensional")
        for name, values in (("weights", self.weights), ("references", self.references)):
            if set(values) != set(DIMENSIONS):
                raise ValueError(f"{name} must specify {DIMENSIONS}")
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
                raise ValueError(f"Every {name} entry must be finite and positive")
        object.__setattr__(self, "weights", {d: float(self.weights[d]) for d in DIMENSIONS})
        object.__setattr__(self, "references", {d: float(self.references[d]) for d in DIMENSIONS})

    def as_dict(self):
        return {"schema": "ttap-value-v2", "mode": self.mode,
                "weights": self.weights, "references": self.references,
                "semantics": "distinct-medical-nonmedical-cargo-outputs" if self.mode == "multidimensional"
                             else "one-primary-output-with-auxiliary-resource-requirements"}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    @property
    def historical_compatible(self):
        return (self.mode == "primary" and self.weights == DEFAULT_WEIGHTS
                and self.references == DEFAULT_REFERENCES)

    @property
    def version(self):
        return f"{self.mode}-v2-{self.fingerprint[:12]}"


def primary_class(task):
    for d in DIMENSIONS:
        if task[FIELDS[d]] > 0:
            return d
    raise ValueError("A task must have at least one positive requirement")


def task_components(task, spec):
    if any(not math.isfinite(float(task[FIELDS[d]])) or task[FIELDS[d]] < 0 for d in DIMENSIONS):
        raise ValueError("Task quantities must be finite and nonnegative")
    return {d: spec.weights[d] * task[FIELDS[d]] / spec.references[d] for d in DIMENSIONS}


def apply_objective(scenario, spec):
    """Update value metadata only; leave locations, fleet and resource needs intact."""
    for task in scenario["tasks"].values():
        task_class = primary_class(task)
        components = task_components(task, spec)
        if spec.mode == "primary":
            value = components[task_class]
            task["class_weight"] = spec.weights[task_class]
            task["impact"] = task[FIELDS[task_class]] / spec.references[task_class]
        else:
            value = sum(components.values())
            # A weighted average keeps the legacy validation invariant 0 < impact <= 1
            # for the original generated demand ranges. The common factor cancels
            # in the instance-normalized objective; no second class weight is used.
            task["class_weight"] = sum(spec.weights.values())
            task["impact"] = value / task["class_weight"]
        task["priority_class"] = task_class  # descriptive category, not strict precedence
        task["raw_impact"] = float(task[FIELDS[task_class]])
        task["impact_reference"] = spec.references[task_class]
        task["potential_value"] = float(value)
        task["value_components"] = components
        task["objective_mode"] = spec.mode
        if value <= 0:
            raise ValueError("Every generated task must have positive potential value")
    scenario["total_potential_value"] = sum(t["potential_value"] for t in scenario["tasks"].values())
    scenario["params"]["PRIORITY_WEIGHTS"] = spec.weights.copy()
    scenario["params"]["IMPACT_REFERENCES"] = spec.references.copy()
    scenario["params"]["OBJECTIVE_SPEC"] = spec.as_dict()
    scenario["objective_version"] = spec.version
    scenario["objective_sha256"] = spec.fingerprint
    return scenario


def compute_metrics(kernel, scenario, routes, summary, spec):
    """Count all delivered dimensions in multidimensional mode, not only task classes."""
    row = kernel["compute_ttap_metrics"](scenario, routes, summary)
    row.update({"Objective Version": spec.version, "Objective SHA256": spec.fingerprint,
                "Objective Mode": spec.mode})
    if spec.mode == "primary" or not bool(summary.get("Has Incumbent", False)):
        return row
    ids = [] if routes is None or routes.empty else routes.loc[
        routes["Status"].eq("Task Executed"), "Task ID"].astype(int).tolist()
    done = [scenario["tasks"][i] for i in ids]
    tasks = list(scenario["tasks"].values())
    for d, output, coverage in [
            ("medical", "Medical Evacuees", "Medical Impact Coverage"),
            ("personnel", "Personnel Transported", "Personnel Impact Coverage"),
            ("cargo", "Cargo Delivered kg", "Cargo Impact Coverage")]:
        total = sum(t[FIELDS[d]] for t in tasks)
        delivered = sum(t[FIELDS[d]] for t in done)
        row[output] = float(delivered)
        row[coverage] = delivered / total if total else float("nan")
    return row


def professor_examples(spec=None):
    spec = spec or ObjectiveSpec()
    mixed = {"med": 1, "pers": 3, "load": 500}
    medical_only = {"med": 1, "pers": 0, "load": 0}
    return {"mixed": sum(task_components(mixed, spec).values()),
            "medical_only": sum(task_components(medical_only, spec).values())}
