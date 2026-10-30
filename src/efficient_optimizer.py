from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .adaptive_optimizer import (
    PARAM_KEYS,
    _actual_metrics,
    _calibrate_reusing_baseline,
    _predicted_metrics,
    _score,
    local_candidates,
    meets_spec,
    power_from_op,
)
from .auto_surrogate import parse_netlist, run_operating_point, run_spice_ac_reference
from .stress_test import physical_status
from .trust_region import DEVICE_KEYS, LocalSensitivityModel, calibrate_local_sensitivities, set_netlist_params


STRATEGY_LABELS = {
    "full": "Full finite-difference",
    "active": "Active-variable recalibration",
    "broyden": "Jacobian reuse · Broyden",
}


def _finite(v) -> bool:
    return v is not None and math.isfinite(float(v))


def _copy_sens(sens):
    return {
        p: {dev: dict(vals) for dev, vals in by_dev.items()}
        for p, by_dev in sens.items()
    }


def _active_parameters(old_model: LocalSensitivityModel, new_params: Dict[str, float], k: int = 2) -> List[str]:
    """Choose variables that moved most in calibration-step coordinates.

    If fewer than k variables moved, fill deterministically using a simple
    aggregate-Jacobian importance score from the previous local model.
    """
    moved = []
    for p in PARAM_KEYS:
        step = max(abs(float(old_model.calibration_steps[p])), 1e-30)
        dz = abs(float(new_params[p]) - float(old_model.baseline_params[p])) / step
        moved.append((dz, p))
    moved.sort(key=lambda x: (-x[0], PARAM_KEYS.index(x[1])))
    selected = [p for dz, p in moved if dz > 1e-12][:k]

    if len(selected) < k:
        importance = []
        for p in PARAM_KEYS:
            if p in selected:
                continue
            step = abs(float(old_model.calibration_steps[p]))
            s = 0.0
            for _, vals in old_model.sensitivities.get(p, {}).items():
                for key in ("gm", "gds", "id", "vds", "vdsat"):
                    d = vals.get(key)
                    if d is not None and math.isfinite(float(d)):
                        s += abs(float(d) * step)
            importance.append((s, p))
        importance.sort(key=lambda x: (-x[0], PARAM_KEYS.index(x[1])))
        for _, p in importance:
            if p not in selected:
                selected.append(p)
            if len(selected) >= k:
                break
    return selected[:k]


def _active_recalibrate(
    text: str,
    ngspice_path: str,
    parsed0,
    op0,
    old_model: LocalSensitivityModel,
    new_params: Dict[str, float],
    active_k: int = 2,
) -> Tuple[LocalSensitivityModel, List[str]]:
    """Recompute only the most relevant k parameter columns of the Jacobian.

    Other columns are reused from the previous anchor. The already-paid actual
    operating point becomes the new baseline, so only 2*k perturbation OP runs
    are added for central differences.
    """
    selected = _active_parameters(old_model, new_params, active_k)
    sens = _copy_sens(old_model.sensitivities)
    runs = 0
    warnings: List[str] = ["Reused non-active Jacobian columns from previous anchor"]

    for p in selected:
        step = float(old_model.calibration_steps[p])
        x0 = float(new_params[p])
        lo, hi = x0 - step, x0 + step
        if p in {"rd", "win", "wtail"} and lo <= 0:
            lo = x0
        plus_text = set_netlist_params(text, {p: hi})
        _, op_plus, _ = run_operating_point(plus_text, ngspice_path)
        runs += 1
        if lo == x0:
            op_minus = op0
            denom = hi - x0
            warnings.append(f"Used one-sided active sensitivity for {p}")
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

    model = LocalSensitivityModel(
        baseline_text=text,
        baseline_parsed=parsed0,
        baseline_params=dict(new_params),
        baseline_op=op0,
        calibration_steps=dict(old_model.calibration_steps),
        sensitivities=sens,
        calibration_runs=runs,
        warnings=warnings,
    )
    return model, selected


def _broyden_reanchor(
    text: str,
    parsed_actual,
    op_actual,
    old_model: LocalSensitivityModel,
    new_params: Dict[str, float],
) -> LocalSensitivityModel:
    """Rank-one Broyden update in normalized design coordinates.

    The previous finite-difference Jacobian is reused. The actual OP at the new
    verified design supplies one secant equation; no additional perturbation
    simulation is required.
    """
    steps = {p: float(old_model.calibration_steps[p]) for p in PARAM_KEYS}
    dz = [
        (float(new_params[p]) - float(old_model.baseline_params[p])) / max(abs(steps[p]), 1e-30)
        for p in PARAM_KEYS
    ]
    denom = sum(v * v for v in dz)
    sens_new = _copy_sens(old_model.sensitivities)
    warnings = ["Jacobian updated by normalized-coordinate Broyden rank-one secant update"]

    if denom <= 1e-24:
        warnings.append("Broyden update skipped because anchor displacement was negligible")
    else:
        for dev, old_vals in old_model.baseline_op.items():
            new_vals = op_actual.get(dev, {})
            for key in DEVICE_KEYS:
                y0 = old_vals.get(key)
                y1 = new_vals.get(key)
                if not (_finite(y0) and _finite(y1)):
                    continue
                # Convert old raw-coordinate derivatives to normalized-coordinate slopes.
                jz = []
                valid = True
                for p in PARAM_KEYS:
                    d_raw = old_model.sensitivities.get(p, {}).get(dev, {}).get(key)
                    if d_raw is None or not math.isfinite(float(d_raw)):
                        valid = False
                        break
                    jz.append(float(d_raw) * steps[p])
                if not valid:
                    continue
                pred_delta = sum(jz[i] * dz[i] for i in range(len(PARAM_KEYS)))
                residual = (float(y1) - float(y0)) - pred_delta
                jz_new = [jz[i] + residual * dz[i] / denom for i in range(len(PARAM_KEYS))]
                for i, p in enumerate(PARAM_KEYS):
                    sens_new.setdefault(p, {}).setdefault(dev, {})[key] = jz_new[i] / steps[p]

    return LocalSensitivityModel(
        baseline_text=text,
        baseline_parsed=parsed_actual,
        baseline_params=dict(new_params),
        baseline_op=op_actual,
        calibration_steps=dict(old_model.calibration_steps),
        sensitivities=sens_new,
        calibration_runs=0,
        warnings=warnings,
    )


@dataclass
class EfficientRunResult:
    strategy: str
    success: bool
    final_params: Dict[str, float]
    final_metrics: dict
    trace: List[dict]
    leaderboard: List[dict]
    op_simulations: int
    perturbation_op_simulations: int
    anchor_op_simulations: int
    ac_simulations: int
    surrogate_evaluations: int
    anchor_rebuilds: int
    wall_clock_s: float
    message: str


def run_efficient_optimizer(
    baseline_text: str,
    ngspice_path: str,
    spec: dict,
    strategy: str = "full",
    active_k: int = 2,
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
) -> EfficientRunResult:
    if strategy not in STRATEGY_LABELS:
        raise ValueError(f"Unknown strategy: {strategy}")
    start_time = time.perf_counter()
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
    perturb_ops = 0
    anchor_ops = 0
    ac_sims = 0
    surrogate_evals = 0
    rebuilds = 0

    # Initial full calibration is common to all strategies: 1 baseline OP + 8 probes.
    model = calibrate_local_sensitivities(baseline_text, ngspice_path, steps)
    op_sims += model.calibration_runs
    anchor_ops += 1
    perturb_ops += max(0, model.calibration_runs - 1)
    rebuilds += 1
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
        "Step": 0, "Action": "Initial SPICE anchor", "Calibration": "Full ± finite difference",
        "Active_vars": "RD, WIN, WTAIL, VBIAS", "Distance": 0.0,
        "RD_kOhm": anchor_params["rd"] / 1e3, "WIN_um": anchor_params["win"] * 1e6,
        "WTAIL_um": anchor_params["wtail"] * 1e6, "VBIAS_V": anchor_params["vbias"],
        "Pred_Gain_dB": anchor_metrics["gain_db"], "Pred_BW_MHz": anchor_metrics["bw_mhz"],
        "SPICE_Gain_dB": anchor_metrics["gain_db"], "SPICE_BW_MHz": anchor_metrics["bw_mhz"],
        "SPICE_Power_mW": anchor_metrics["power_mw"], "Gain_Error_dB": 0.0, "BW_Error_pct": 0.0,
        "Physical_Valid": anchor_metrics["physical_valid"], "PASS": meets_spec(anchor_metrics, spec),
    })
    if meets_spec(anchor_metrics, spec):
        return EfficientRunResult(strategy, True, anchor_params, anchor_metrics, trace, leaderboard, op_sims,
                                  perturb_ops, anchor_ops, ac_sims, surrogate_evals, rebuilds,
                                  time.perf_counter() - start_time, "Initial anchor already satisfies specification.")

    for move in range(1, max_design_moves + 1):
        candidates = local_candidates(model, safe_radius, bounds)
        scored = []
        for params, d in candidates:
            try:
                pred = _predicted_metrics(model, params, output_p, output_n, input_amp, f_start, f_stop, sat_margin_v)
                surrogate_evals += 1
                scored.append((_score(pred, spec, d), params, d, pred))
            except Exception:
                continue
        if not scored:
            return EfficientRunResult(strategy, False, anchor_params, anchor_metrics, trace, leaderboard, op_sims,
                                      perturb_ops, anchor_ops, ac_sims, surrogate_evals, rebuilds,
                                      time.perf_counter() - start_time, "All local candidates failed numerically.")
        scored.sort(key=lambda x: x[0])
        for rank, (score, p, d, pred) in enumerate(scored[:5], 1):
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
        parsed_actual, op_actual, _ = run_operating_point(candidate_text, ngspice_path)
        op_sims += 1
        anchor_ops += 1
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
        gain_err = abs(float(pred["gain_db"]) - actual["gain_db"])
        bw_err = 100.0 * abs(float(pred["bw_mhz"]) - actual["bw_mhz"]) / max(abs(actual["bw_mhz"]), 1e-12)

        calibration_note = "Final verification"
        active_vars = ""
        if not passed:
            if strategy == "full":
                new_model = _calibrate_reusing_baseline(candidate_text, ngspice_path, parsed_actual, op_actual, steps)
                op_sims += new_model.calibration_runs
                perturb_ops += new_model.calibration_runs
                calibration_note = "Full ± finite difference"
                active_vars = "RD, WIN, WTAIL, VBIAS"
            elif strategy == "active":
                new_model, selected = _active_recalibrate(
                    candidate_text, ngspice_path, parsed_actual, op_actual, model, chosen, active_k=active_k
                )
                op_sims += new_model.calibration_runs
                perturb_ops += new_model.calibration_runs
                calibration_note = f"Active {len(selected)} vars"
                active_vars = ", ".join(p.upper() for p in selected)
            else:
                new_model = _broyden_reanchor(candidate_text, parsed_actual, op_actual, model, chosen)
                calibration_note = "Broyden secant update"
                active_vars = "No perturbation OP"
            rebuilds += 1
        else:
            new_model = None

        trace.append({
            "Step": move, "Action": "SPICE PASS" if passed else "Verify → re-anchor",
            "Calibration": calibration_note, "Active_vars": active_vars, "Distance": d,
            "RD_kOhm": chosen["rd"] / 1e3, "WIN_um": chosen["win"] * 1e6,
            "WTAIL_um": chosen["wtail"] * 1e6, "VBIAS_V": chosen["vbias"],
            "Pred_Gain_dB": pred["gain_db"], "Pred_BW_MHz": pred["bw_mhz"],
            "SPICE_Gain_dB": actual["gain_db"], "SPICE_BW_MHz": actual["bw_mhz"],
            "SPICE_Power_mW": actual["power_mw"], "Gain_Error_dB": gain_err, "BW_Error_pct": bw_err,
            "Physical_Valid": actual["physical_valid"], "PASS": passed,
        })

        if passed:
            return EfficientRunResult(strategy, True, chosen, actual, trace, leaderboard, op_sims,
                                      perturb_ops, anchor_ops, ac_sims, surrogate_evals, rebuilds,
                                      time.perf_counter() - start_time, "SPICE-verified specification satisfied.")

        model = new_model
        anchor_text = candidate_text
        anchor_params = dict(chosen)
        anchor_metrics = actual

    return EfficientRunResult(strategy, False, anchor_params, anchor_metrics, trace, leaderboard, op_sims,
                              perturb_ops, anchor_ops, ac_sims, surrogate_evals, rebuilds,
                              time.perf_counter() - start_time,
                              f"Stopped after {max_design_moves} verified moves without satisfying specification.")
