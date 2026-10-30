from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .auto_surrogate import parse_netlist, run_operating_point, run_spice_ac_reference, spice_number
from .stress_test import physical_status
from .trust_region import (
    DEVICE_KEYS,
    LocalSensitivityModel,
    calibrate_local_sensitivities,
    set_netlist_params,
)

PARAM_KEYS = ("rd", "win", "wtail", "vbias")


def _finite(v) -> bool:
    return v is not None and math.isfinite(float(v))


def _detect_supply_voltage(text: str) -> float:
    """Best-effort VDD extraction for the current prototype."""
    for raw in text.splitlines():
        toks = raw.strip().split()
        if len(toks) >= 4 and toks[0].lower() == "vdd" and toks[2].lower() in {"0", "gnd"}:
            try:
                return abs(spice_number(toks[3]))
            except Exception:
                pass
    return 1.8


def _tail_device_name(text: str) -> str:
    parsed = parse_netlist(text)
    for m in parsed.mosfets:
        if "tail" in m.name.lower():
            return m.name.lower()
    for m in parsed.mosfets:
        if m.s in {"0", "gnd"} and m.d not in {"0", "gnd"}:
            return m.name.lower()
    if not parsed.mosfets:
        raise ValueError("No MOSFET found for power estimate")
    return parsed.mosfets[-1].name.lower()


def power_from_op(text: str, op: Dict[str, Dict[str, Optional[float]]]) -> float:
    vdd = _detect_supply_voltage(text)
    tail = _tail_device_name(text)
    ids = op.get(tail, {}).get("id")
    if not _finite(ids):
        # Fallback: use the largest absolute MOS drain current as a supply-current proxy.
        vals = [abs(float(v.get("id"))) for v in op.values() if _finite(v.get("id"))]
        if not vals:
            return float("nan")
        ids = max(vals)
    return vdd * abs(float(ids)) * 1e3


def _calibrate_reusing_baseline(
    text: str,
    ngspice_path: str,
    parsed0,
    op0,
    steps: Dict[str, float],
) -> LocalSensitivityModel:
    """Build local Jacobian while reusing a baseline OP already paid for.

    Requires 8 additional OP runs for the four central-difference variables.
    """
    baseline_params = dict(parsed0.params)
    sens: Dict[str, Dict[str, Dict[str, float]]] = {}
    warnings: List[str] = []
    runs = 0
    for p in PARAM_KEYS:
        step = float(steps[p])
        x0 = float(baseline_params[p])
        lo, hi = x0 - step, x0 + step
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
        calibration_steps=dict(steps),
        sensitivities=sens,
        calibration_runs=runs,
        warnings=warnings,
    )


def _predicted_metrics(model: LocalSensitivityModel, params: Dict[str, float], output_p: str, output_n: str,
                       input_amp: float, f_start: float, f_stop: float, sat_margin_v: float) -> dict:
    response, surrogate, notes = model.predict_response(
        params, output_p, output_n, input_amp, f_start, f_stop, points_per_decade=45
    )
    phys = physical_status(surrogate.op, sat_margin_v)
    candidate_text = set_netlist_params(model.baseline_text, params)
    return {
        "gain_db": float(response["low_gain_db"]),
        "bw_mhz": float(response["bandwidth_hz"]) / 1e6,
        "power_mw": power_from_op(candidate_text, surrogate.op),
        "physical_valid": bool(phys["valid"]),
        "min_sat_margin_mv": float(phys["min_margin_v"]) * 1e3,
        "notes": "; ".join(notes),
    }


def _actual_metrics(text: str, ngspice_path: str, output_p: str, output_n: str, input_amp: float,
                    f_start: float, f_stop: float, sat_margin_v: float):
    parsed, op, _ = run_operating_point(text, ngspice_path)
    ac = run_spice_ac_reference(text, ngspice_path, output_p, output_n, input_amp, f_start, f_stop, 60)
    phys = physical_status(op, sat_margin_v)
    return parsed, op, {
        "gain_db": float(ac["low_gain_db"]),
        "bw_mhz": float(ac["bandwidth_hz"]) / 1e6,
        "power_mw": power_from_op(text, op),
        "physical_valid": bool(phys["valid"]),
        "min_sat_margin_mv": float(phys["min_margin_v"]) * 1e3,
        "limiting_device": phys["limiting_device"],
    }


def meets_spec(m: dict, spec: dict) -> bool:
    return bool(
        m["physical_valid"]
        and m["gain_db"] >= spec["min_gain_db"]
        and m["bw_mhz"] >= spec["min_bw_mhz"]
        and m["power_mw"] <= spec["max_power_mw"]
    )


def _score(m: dict, spec: dict, distance: float) -> float:
    g = max(0.0, (spec["min_gain_db"] - m["gain_db"]) / max(spec["min_gain_db"], 1e-9))
    b = max(0.0, (spec["min_bw_mhz"] - m["bw_mhz"]) / max(spec["min_bw_mhz"], 1e-9))
    p = max(0.0, (m["power_mw"] - spec["max_power_mw"]) / max(spec["max_power_mw"], 1e-9))
    phys = 0.0 if m["physical_valid"] else 10.0
    # Prefer lower violation first, then slightly prefer farther useful progress
    # inside the already validated region to reduce anchor rebuilds.
    return phys + 5.0 * g + 5.0 * b + 3.0 * p - 0.002 * distance


def local_candidates(model: LocalSensitivityModel, safe_radius: float, bounds: dict) -> List[Tuple[Dict[str, float], float]]:
    """Generate deterministic multi-variable candidates in calibration-step units."""
    max_step = int(math.ceil(safe_radius))
    base = {k: float(model.baseline_params[k]) for k in PARAM_KEYS}
    out = []
    seen = set()
    for offs in itertools.product(range(-max_step, max_step + 1), repeat=4):
        if offs == (0, 0, 0, 0):
            continue
        d = math.sqrt(sum(float(x) ** 2 for x in offs))
        if d > safe_radius + 1e-12:
            continue
        p = {k: base[k] + offs[i] * float(model.calibration_steps[k]) for i, k in enumerate(PARAM_KEYS)}
        if not (bounds["rd"][0] <= p["rd"] <= bounds["rd"][1]):
            continue
        if not (bounds["win"][0] <= p["win"] <= bounds["win"][1]):
            continue
        if not (bounds["wtail"][0] <= p["wtail"] <= bounds["wtail"][1]):
            continue
        if not (bounds["vbias"][0] <= p["vbias"] <= bounds["vbias"][1]):
            continue
        key = tuple(round(p[k], 15) for k in PARAM_KEYS)
        if key not in seen:
            seen.add(key)
            out.append((p, d))
    return out


@dataclass
class AdaptiveRunResult:
    success: bool
    final_params: Dict[str, float]
    final_metrics: dict
    trace: List[dict]
    leaderboard: List[dict]
    op_simulations: int
    ac_simulations: int
    surrogate_evaluations: int
    anchor_rebuilds: int
    message: str


def run_adaptive_optimizer(
    baseline_text: str,
    ngspice_path: str,
    spec: dict,
    output_p: str = "outp",
    output_n: str = "outn",
    input_amp: float = 1.0,
    f_start: float = 10.0,
    f_stop: float = 1e9,
    calibration_steps: Optional[Dict[str, float]] = None,
    safe_radius: float = 3.0,
    sat_margin_v: float = 0.05,
    max_design_moves: int = 8,
    bounds: Optional[dict] = None,
) -> AdaptiveRunResult:
    steps = calibration_steps or {"rd": 200.0, "win": 2e-6, "wtail": 5e-6, "vbias": 0.01}
    bounds = bounds or {
        "rd": (3000.0, 11000.0),
        "win": (6e-6, 80e-6),
        "wtail": (15e-6, 100e-6),
        "vbias": (0.58, 0.82),
    }
    trace: List[dict] = []
    leaderboard: List[dict] = []
    op_sims = 0
    ac_sims = 0
    surrogate_evals = 0
    anchor_rebuilds = 0

    # Initial anchor: 9 OP simulations (baseline + 8 central perturbations).
    model = calibrate_local_sensitivities(baseline_text, ngspice_path, steps)
    op_sims += model.calibration_runs
    anchor_rebuilds += 1
    anchor_text = baseline_text
    anchor_params = {k: float(model.baseline_params[k]) for k in PARAM_KEYS}
    anchor_ac = run_spice_ac_reference(anchor_text, ngspice_path, output_p, output_n, input_amp, f_start, f_stop, 60)
    ac_sims += 1
    anchor_phys = physical_status(model.baseline_op, sat_margin_v)
    anchor_metrics = {
        "gain_db": float(anchor_ac["low_gain_db"]),
        "bw_mhz": float(anchor_ac["bandwidth_hz"]) / 1e6,
        "power_mw": power_from_op(anchor_text, model.baseline_op),
        "physical_valid": bool(anchor_phys["valid"]),
        "min_sat_margin_mv": float(anchor_phys["min_margin_v"]) * 1e3,
    }
    trace.append({
        "Step": 0, "Action": "Initial SPICE anchor", "Distance": 0.0,
        "RD_kOhm": anchor_params["rd"] / 1e3, "WIN_um": anchor_params["win"] * 1e6,
        "WTAIL_um": anchor_params["wtail"] * 1e6, "VBIAS_V": anchor_params["vbias"],
        "Pred_Gain_dB": anchor_metrics["gain_db"], "Pred_BW_MHz": anchor_metrics["bw_mhz"],
        "Pred_Power_mW": anchor_metrics["power_mw"], "Pred_MinSatMargin_mV": anchor_metrics["min_sat_margin_mv"],
        "SPICE_Gain_dB": anchor_metrics["gain_db"], "SPICE_BW_MHz": anchor_metrics["bw_mhz"],
        "SPICE_Power_mW": anchor_metrics["power_mw"], "SPICE_MinSatMargin_mV": anchor_metrics["min_sat_margin_mv"],
        "Physical_Valid": anchor_metrics["physical_valid"], "PASS": meets_spec(anchor_metrics, spec),
    })
    if meets_spec(anchor_metrics, spec):
        return AdaptiveRunResult(True, anchor_params, anchor_metrics, trace, leaderboard, op_sims, ac_sims,
                                 surrogate_evals, anchor_rebuilds, "Initial anchor already satisfies the specification.")

    for move in range(1, max_design_moves + 1):
        candidates = local_candidates(model, safe_radius, bounds)
        if not candidates:
            return AdaptiveRunResult(False, anchor_params, anchor_metrics, trace, leaderboard, op_sims, ac_sims,
                                     surrogate_evals, anchor_rebuilds, "No candidate exists inside the configured trust region.")

        scored = []
        for params, d in candidates:
            try:
                pred = _predicted_metrics(model, params, output_p, output_n, input_amp, f_start, f_stop, sat_margin_v)
                surrogate_evals += 1
                score = _score(pred, spec, d)
                scored.append((score, params, d, pred))
            except Exception:
                continue
        if not scored:
            return AdaptiveRunResult(False, anchor_params, anchor_metrics, trace, leaderboard, op_sims, ac_sims,
                                     surrogate_evals, anchor_rebuilds, "All local surrogate candidates failed numerically.")
        scored.sort(key=lambda x: x[0])
        for rank, (score, p, d, pred) in enumerate(scored[:5], start=1):
            leaderboard.append({
                "Move": move, "Rank": rank, "Score": score, "Distance": d,
                "RD_kOhm": p["rd"] / 1e3, "WIN_um": p["win"] * 1e6,
                "WTAIL_um": p["wtail"] * 1e6, "VBIAS_V": p["vbias"],
                "Pred_Gain_dB": pred["gain_db"], "Pred_BW_MHz": pred["bw_mhz"],
                "Pred_Power_mW": pred["power_mw"], "Pred_MinSatMargin_mV": pred["min_sat_margin_mv"],
                "Pred_PASS": meets_spec(pred, spec),
            })

        _, chosen, d, pred = scored[0]
        candidate_text = set_netlist_params(anchor_text, chosen)

        # Verify the selected design with one OP + one AC.  Validation is not used
        # in the prediction that selected this candidate.
        parsed_actual, op_actual, _ = run_operating_point(candidate_text, ngspice_path)
        op_sims += 1
        actual_ac = run_spice_ac_reference(candidate_text, ngspice_path, output_p, output_n, input_amp, f_start, f_stop, 60)
        ac_sims += 1
        actual_phys = physical_status(op_actual, sat_margin_v)
        actual = {
            "gain_db": float(actual_ac["low_gain_db"]),
            "bw_mhz": float(actual_ac["bandwidth_hz"]) / 1e6,
            "power_mw": power_from_op(candidate_text, op_actual),
            "physical_valid": bool(actual_phys["valid"]),
            "min_sat_margin_mv": float(actual_phys["min_margin_v"]) * 1e3,
        }
        passed = meets_spec(actual, spec)
        trace.append({
            "Step": move,
            "Action": "SPICE PASS" if passed else "Verify → re-anchor",
            "Distance": d,
            "RD_kOhm": chosen["rd"] / 1e3, "WIN_um": chosen["win"] * 1e6,
            "WTAIL_um": chosen["wtail"] * 1e6, "VBIAS_V": chosen["vbias"],
            "Pred_Gain_dB": pred["gain_db"], "Pred_BW_MHz": pred["bw_mhz"],
            "Pred_Power_mW": pred["power_mw"], "Pred_MinSatMargin_mV": pred["min_sat_margin_mv"],
            "SPICE_Gain_dB": actual["gain_db"], "SPICE_BW_MHz": actual["bw_mhz"],
            "SPICE_Power_mW": actual["power_mw"], "SPICE_MinSatMargin_mV": actual["min_sat_margin_mv"],
            "Physical_Valid": actual["physical_valid"], "PASS": passed,
        })
        if passed:
            return AdaptiveRunResult(True, chosen, actual, trace, leaderboard, op_sims, ac_sims,
                                     surrogate_evals, anchor_rebuilds, "SPICE-verified specification satisfied.")

        # New local anchor. Reuse the OP just paid for, then spend only the 8
        # perturbation OP simulations needed for the new Jacobian.
        model = _calibrate_reusing_baseline(candidate_text, ngspice_path, parsed_actual, op_actual, steps)
        op_sims += model.calibration_runs
        anchor_rebuilds += 1
        anchor_text = candidate_text
        anchor_params = dict(chosen)
        anchor_metrics = actual

    return AdaptiveRunResult(False, anchor_params, anchor_metrics, trace, leaderboard, op_sims, ac_sims,
                             surrogate_evals, anchor_rebuilds,
                             f"Stopped after {max_design_moves} verified design moves without satisfying the specification.")
