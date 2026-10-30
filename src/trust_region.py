from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .auto_surrogate import (
    LocalSmallSignalSurrogate,
    ParsedNetlist,
    assess_surrogate_confidence,
    parse_netlist,
    run_operating_point,
    run_spice_ac_reference,
)


DEFAULT_CALIBRATION_STEPS = {
    "rd": 200.0,
    "win": 2e-6,
    "wtail": 5e-6,
    "vbias": 0.01,
}

DEVICE_KEYS = [
    "id", "gm", "gds", "gmbs", "vgs", "vds", "vdsat", "vth", "von",
    "cgs", "cgd", "cgb", "cbd", "cbs",
]


def set_netlist_params(text: str, values: Dict[str, float]) -> str:
    """Replace selected .param values while leaving all other deck text intact."""
    out = text
    for key, val in values.items():
        pat = re.compile(rf"(?im)(\b{re.escape(key)}\s*=\s*)([^\s]+)")
        repl = rf"\g<1>{float(val):.12g}"
        out, n = pat.subn(repl, out, count=1)
        if n == 0:
            raise ValueError(f".param {key} was not found in the netlist")
    return out


def _finite(v: Optional[float]) -> bool:
    return v is not None and math.isfinite(float(v))


@dataclass
class LocalSensitivityModel:
    baseline_text: str
    baseline_parsed: ParsedNetlist
    baseline_params: Dict[str, float]
    baseline_op: Dict[str, Dict[str, Optional[float]]]
    calibration_steps: Dict[str, float]
    sensitivities: Dict[str, Dict[str, Dict[str, float]]]
    calibration_runs: int
    warnings: List[str] = field(default_factory=list)

    def normalized_distance(self, params: Dict[str, float]) -> float:
        total = 0.0
        for key, step in self.calibration_steps.items():
            if step <= 0 or key not in params or key not in self.baseline_params:
                continue
            total += ((float(params[key]) - float(self.baseline_params[key])) / float(step)) ** 2
        return math.sqrt(total)

    def predict_op(self, params: Dict[str, float]) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
        pred: Dict[str, Dict[str, Optional[float]]] = {}
        notes: List[str] = []
        for dev, basevals in self.baseline_op.items():
            pred[dev] = {}
            for key in DEVICE_KEYS:
                base = basevals.get(key)
                if not _finite(base):
                    pred[dev][key] = base
                    continue
                val = float(base)
                for p, jac_by_dev in self.sensitivities.items():
                    if p not in params or p not in self.baseline_params:
                        continue
                    deriv = jac_by_dev.get(dev, {}).get(key)
                    if deriv is None or not math.isfinite(float(deriv)):
                        continue
                    val += float(deriv) * (float(params[p]) - float(self.baseline_params[p]))
                # The linearized model is local.  Keep essential conductances
                # numerically solvable, but record clipping as a trust warning.
                if key == "gds" and val <= 1e-12:
                    notes.append(f"{dev}[gds] linear prediction clipped at 1e-12 S")
                    val = 1e-12
                elif key == "gm" and val < 0.0:
                    notes.append(f"{dev}[gm] linear prediction clipped at 0 S")
                    val = 0.0
                elif key in {"cgs", "cgd", "cgb", "cbd", "cbs"} and val < 0.0:
                    notes.append(f"{dev}[{key}] negative linear prediction clipped at 0 F")
                    val = 0.0
                pred[dev][key] = val
        return pred, notes

    def predict_response(
        self,
        params: Dict[str, float],
        output_p: str,
        output_n: str,
        input_amplitude: float,
        f_start: float,
        f_stop: float,
        points_per_decade: int = 60,
    ) -> Tuple[dict, LocalSmallSignalSurrogate, List[str]]:
        candidate_text = set_netlist_params(self.baseline_text, params)
        parsed = parse_netlist(candidate_text)
        pred_op, notes = self.predict_op(params)
        surrogate = LocalSmallSignalSurrogate(parsed=parsed, op=pred_op, warnings=list(notes))
        response = surrogate.frequency_response(
            output_p=output_p,
            output_n=output_n,
            input_amplitude=input_amplitude,
            f_start=f_start,
            f_stop=f_stop,
            points_per_decade=points_per_decade,
        )
        return response, surrogate, notes


def calibrate_local_sensitivities(
    text: str,
    ngspice_path: str,
    calibration_steps: Optional[Dict[str, float]] = None,
) -> LocalSensitivityModel:
    steps = dict(DEFAULT_CALIBRATION_STEPS)
    if calibration_steps:
        steps.update({k.lower(): float(v) for k, v in calibration_steps.items()})

    parsed0, op0, _ = run_operating_point(text, ngspice_path)
    baseline_params = dict(parsed0.params)
    required = ["rd", "win", "wtail", "vbias"]
    missing = [k for k in required if k not in baseline_params]
    if missing:
        raise ValueError("Phase 2 demo requires .param values: " + ", ".join(missing))

    sens: Dict[str, Dict[str, Dict[str, float]]] = {}
    runs = 1
    warnings: List[str] = []

    for p in required:
        step = float(steps[p])
        if step <= 0:
            raise ValueError(f"Calibration step for {p} must be positive")
        x0 = float(baseline_params[p])
        lo = x0 - step
        hi = x0 + step
        if p in {"win", "wtail", "rd"} and lo <= 0:
            lo = x0
        plus_text = set_netlist_params(text, {p: hi})
        _, op_plus, _ = run_operating_point(plus_text, ngspice_path)
        runs += 1
        if lo == x0:
            op_minus = op0
            denom = hi - x0
            warnings.append(f"Used one-sided sensitivity for {p}")
        else:
            minus_text = set_netlist_params(text, {p: lo})
            _, op_minus, _ = run_operating_point(minus_text, ngspice_path)
            runs += 1
            denom = hi - lo

        sens[p] = {}
        for dev in op0:
            sens[p][dev] = {}
            for key in DEVICE_KEYS:
                vp = op_plus.get(dev, {}).get(key)
                vm = op_minus.get(dev, {}).get(key)
                if _finite(vp) and _finite(vm) and denom != 0:
                    sens[p][dev][key] = (float(vp) - float(vm)) / denom

    return LocalSensitivityModel(
        baseline_text=text,
        baseline_parsed=parsed0,
        baseline_params=baseline_params,
        baseline_op=op0,
        calibration_steps=steps,
        sensitivities=sens,
        calibration_runs=runs,
        warnings=warnings,
    )


def compact_validation_points(base: Dict[str, float]) -> List[Dict[str, float]]:
    """Unseen sizing points.  None are identical to the ± calibration probes."""
    b = {k: float(v) for k, v in base.items()}
    pts: List[Dict[str, float]] = []

    def add(label: str, **changes: float):
        p = {k: b[k] for k in ("rd", "win", "wtail", "vbias")}
        p.update(changes)
        p["label"] = label  # type: ignore[assignment]
        pts.append(p)

    add("RD 4.7 kΩ", rd=4700.0)
    add("RD 6.8 kΩ", rd=6800.0)
    add("RD 8.2 kΩ", rd=8200.0)
    add("WIN 15 µm", win=15e-6)
    add("WIN 25 µm", win=25e-6)
    add("WIN 30 µm", win=30e-6)
    add("WIN 40 µm", win=40e-6)
    add("WTAIL 40 µm", wtail=40e-6)
    add("WTAIL 60 µm", wtail=60e-6)
    add("VBIAS 0.68 V", vbias=0.68)
    add("VBIAS 0.72 V", vbias=0.72)
    add("VBIAS 0.75 V", vbias=0.75)
    add("Search path A", rd=6800.0, win=25e-6)
    add("Search path B", rd=6800.0, win=30e-6)
    add("Search path C", rd=6800.0, win=40e-6)
    return pts


def evaluate_unseen_points(
    model: LocalSensitivityModel,
    ngspice_path: str,
    points: List[Dict[str, float]],
    output_p: str,
    output_n: str,
    input_amplitude: float,
    f_start: float,
    f_stop: float,
    points_per_decade: int = 60,
) -> List[dict]:
    rows: List[dict] = []
    for raw in points:
        label = str(raw.get("label", "candidate"))
        params = {k: float(raw[k]) for k in ("rd", "win", "wtail", "vbias")}
        predicted, _, notes = model.predict_response(
            params, output_p, output_n, input_amplitude, f_start, f_stop, points_per_decade
        )
        candidate_text = set_netlist_params(model.baseline_text, params)
        actual = run_spice_ac_reference(
            candidate_text, ngspice_path, output_p, output_n, input_amplitude,
            f_start, f_stop, points_per_decade,
        )
        agreement = assess_surrogate_confidence(predicted, actual)
        rows.append({
            "Design": label,
            "Distance": model.normalized_distance(params),
            "RD_kOhm": params["rd"] / 1e3,
            "WIN_um": params["win"] * 1e6,
            "WTAIL_um": params["wtail"] * 1e6,
            "VBIAS_V": params["vbias"],
            "Pred_Gain_dB": float(predicted["low_gain_db"]),
            "SPICE_Gain_dB": float(actual["low_gain_db"]),
            "Gain_Error_dB": float(agreement["gain_error_db"]),
            "Pred_BW_MHz": float(predicted["bandwidth_hz"]) / 1e6,
            "SPICE_BW_MHz": float(actual["bandwidth_hz"]) / 1e6,
            "BW_Error_pct": float(agreement["bw_error_pct"]),
            "Agreement": agreement["level"],
            "Prediction_notes": "; ".join(notes),
        })
    return rows


def empirical_trust_radius(rows: List[dict]) -> dict:
    if not rows:
        return {"radius": 0.0, "status": "NO DATA", "high_points": 0, "total": 0}
    ordered = sorted(rows, key=lambda r: float(r["Distance"]))
    radius = 0.0
    high_points = 0
    first_failure = None
    for row in ordered:
        d = float(row["Distance"])
        if row["Agreement"] == "HIGH":
            if first_failure is None:
                radius = d
            high_points += 1
        elif first_failure is None:
            first_failure = d
    if first_failure is None:
        status = "HIGH across all tested points"
    elif radius == 0.0:
        status = f"No conservative HIGH region established; first non-HIGH at distance {first_failure:.2f}"
    else:
        status = f"Conservative HIGH region through distance {radius:.2f}; first non-HIGH at {first_failure:.2f}"
    return {"radius": radius, "status": status, "high_points": high_points, "total": len(rows), "first_failure": first_failure}
