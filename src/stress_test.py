from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .auto_surrogate import assess_surrogate_confidence, run_operating_point, run_spice_ac_reference
from .trust_region import LocalSensitivityModel, set_netlist_params


DEFAULT_SAT_MARGIN_V = 0.05


def _finite(v: Optional[float]) -> bool:
    return v is not None and math.isfinite(float(v))


def physical_status(op: Dict[str, Dict[str, Optional[float]]], margin_v: float = DEFAULT_SAT_MARGIN_V) -> dict:
    """Simple NMOS local physical-validity check used by the current prototype.

    A device is considered safely saturated when VDS - VDSAT >= margin_v.
    This is intentionally a local prototype rule, not a general foundry signoff rule.
    """
    margins: List[Tuple[str, float]] = []
    essential_ok = True
    for dev, vals in op.items():
        gm, gds = vals.get("gm"), vals.get("gds")
        if not _finite(gm) or not _finite(gds) or float(gm) < 0 or float(gds) <= 0:
            essential_ok = False
        vds, vdsat = vals.get("vds"), vals.get("vdsat")
        if _finite(vds) and _finite(vdsat):
            margins.append((dev, float(vds) - float(vdsat)))
    if not margins:
        return {"valid": False, "min_margin_v": float("nan"), "limiting_device": "unknown"}
    limiting_device, min_margin = min(margins, key=lambda x: x[1])
    return {
        "valid": bool(essential_ok and min_margin >= float(margin_v)),
        "min_margin_v": float(min_margin),
        "limiting_device": limiting_device,
    }


def _point_from_offsets(
    base: Dict[str, float],
    steps: Dict[str, float],
    offsets: Tuple[float, float, float, float],
    label: str,
    tier: str,
) -> Dict[str, float]:
    keys = ("rd", "win", "wtail", "vbias")
    p = {k: float(base[k]) + float(offsets[i]) * float(steps[k]) for i, k in enumerate(keys)}
    # Keep the generated stress deck numerically meaningful. These are only
    # safety clamps; the profile is designed so they normally do not trigger.
    p["rd"] = max(p["rd"], 200.0)
    p["win"] = max(p["win"], 0.5e-6)
    p["wtail"] = max(p["wtail"], 0.5e-6)
    p["vbias"] = min(max(p["vbias"], 0.45), 1.10)
    p["label"] = label  # type: ignore[assignment]
    p["tier"] = tier  # type: ignore[assignment]
    p["off_rd"] = offsets[0]  # type: ignore[assignment]
    p["off_win"] = offsets[1]  # type: ignore[assignment]
    p["off_wtail"] = offsets[2]  # type: ignore[assignment]
    p["off_vbias"] = offsets[3]  # type: ignore[assignment]
    return p


def stress_validation_points(
    base: Dict[str, float], steps: Dict[str, float], profile: str = "Quick · 36 points"
) -> List[Dict[str, float]]:
    """Deterministic multi-variable stress points, expressed in calibration-step units."""
    near = [
        (2, 2, 0, 0), (-2, 2, 0, 0), (2, -2, 0, 0), (-2, -2, 0, 0),
        (0, 2, 2, 1), (0, -2, -2, -1), (2, 0, -2, 1), (-2, 0, 2, -1),
    ]
    mid = [
        (4, 4, 2, 2), (-4, 4, -2, 2), (4, -4, 2, -2), (-4, -4, -2, -2),
        (6, 3, 0, 2), (-6, 3, 0, -2), (3, 6, 2, 0), (3, -6, -2, 0),
        (0, 5, 4, 3), (0, -5, -4, -3), (5, 0, 4, -3), (-5, 0, -4, 3),
    ]
    far = [
        (8, 6, 4, 4), (-8, 6, -4, 4), (8, -6, 4, -4), (-8, -6, -4, -4),
        (10, 8, 5, 5), (-10, 8, -5, 5), (10, -8, 5, -5), (-10, -8, -5, -5),
        (12, 4, 6, 7), (-10, 4, -6, -7), (4, 12, 6, -6), (4, -8, -6, 8),
    ]
    extreme = [
        (12, 10, 6, 8), (-12, 10, -6, 8), (12, -8, 7, -8), (-10, -8, -7, -8),
    ]
    deep_extra = [
        (16, 10, 8, 10), (-16, 10, -8, 10), (16, -9, 8, -10), (-16, -9, -8, -10),
        (20, 5, 10, 12), (-20, 5, -9, -12), (8, 18, 8, -10), (-8, 18, -8, 10),
        (14, 15, 0, 12), (-14, 15, 0, -12), (14, -9, 8, 8), (-14, -9, -8, -8),
        (18, 12, 6, -12), (-18, 12, -6, 12), (18, -8, 10, 6), (-18, -8, -9, -6),
        (5, 20, 10, 10), (-5, 20, -9, -10), (10, 16, -8, 12), (-10, 16, 8, -12),
        (20, 12, 10, 0), (-20, 12, -9, 0), (20, -8, 0, 12), (-20, -8, 0, -12),
    ]

    groups: List[Tuple[str, List[Tuple[float, float, float, float]]]] = [
        ("Near", near), ("Mid", mid), ("Far", far), ("Extreme", extreme)
    ]
    if profile.startswith("Deep"):
        groups.append(("Deep", deep_extra))

    points: List[Dict[str, float]] = []
    idx = 1
    for tier, vectors in groups:
        for vec in vectors:
            points.append(_point_from_offsets(base, steps, vec, f"{tier}-{idx:02d}", tier))
            idx += 1
    return points


def evaluate_stress_points(
    model: LocalSensitivityModel,
    ngspice_path: str,
    points: List[Dict[str, float]],
    output_p: str,
    output_n: str,
    input_amplitude: float,
    f_start: float,
    f_stop: float,
    points_per_decade: int = 50,
    sat_margin_v: float = DEFAULT_SAT_MARGIN_V,
) -> List[dict]:
    rows: List[dict] = []
    for raw in points:
        label = str(raw.get("label", "candidate"))
        tier = str(raw.get("tier", "Stress"))
        params = {k: float(raw[k]) for k in ("rd", "win", "wtail", "vbias")}
        row = {
            "Design": label,
            "Tier": tier,
            "Distance": model.normalized_distance(params),
            "RD_kOhm": params["rd"] / 1e3,
            "WIN_um": params["win"] * 1e6,
            "WTAIL_um": params["wtail"] * 1e6,
            "VBIAS_V": params["vbias"],
            "dRD_steps": float(raw.get("off_rd", 0.0)),
            "dWIN_steps": float(raw.get("off_win", 0.0)),
            "dWTAIL_steps": float(raw.get("off_wtail", 0.0)),
            "dVBIAS_steps": float(raw.get("off_vbias", 0.0)),
            "Error": "",
        }
        try:
            predicted, pred_surrogate, notes = model.predict_response(
                params, output_p, output_n, input_amplitude, f_start, f_stop, points_per_decade
            )
            pred_phys = physical_status(pred_surrogate.op, sat_margin_v)

            candidate_text = set_netlist_params(model.baseline_text, params)
            _, actual_op, _ = run_operating_point(candidate_text, ngspice_path)
            actual_phys = physical_status(actual_op, sat_margin_v)
            actual = run_spice_ac_reference(
                candidate_text, ngspice_path, output_p, output_n, input_amplitude,
                f_start, f_stop, points_per_decade,
            )
            agreement = assess_surrogate_confidence(predicted, actual)
            level = str(agreement["level"])
            # Leaving the physical operating domain is itself a trust-boundary event.
            if pred_phys["valid"] != actual_phys["valid"]:
                level = "LOW"
            elif not actual_phys["valid"]:
                level = "LOW"
            elif notes and level == "HIGH":
                # Numerical clipping is a warning that first-order extrapolation is strained.
                level = "MEDIUM"

            row.update({
                "Pred_Gain_dB": float(predicted["low_gain_db"]),
                "SPICE_Gain_dB": float(actual["low_gain_db"]),
                "Gain_Error_dB": float(agreement["gain_error_db"]),
                "Pred_BW_MHz": float(predicted["bandwidth_hz"]) / 1e6,
                "SPICE_BW_MHz": float(actual["bandwidth_hz"]) / 1e6,
                "BW_Error_pct": float(agreement["bw_error_pct"]),
                "Pred_MinSatMargin_mV": float(pred_phys["min_margin_v"]) * 1e3,
                "SPICE_MinSatMargin_mV": float(actual_phys["min_margin_v"]) * 1e3,
                "Pred_Physical_Valid": bool(pred_phys["valid"]),
                "SPICE_Physical_Valid": bool(actual_phys["valid"]),
                "Physical_Match": bool(pred_phys["valid"] == actual_phys["valid"]),
                "Limiting_Device": actual_phys["limiting_device"],
                "Agreement": level,
                "Prediction_notes": "; ".join(notes),
            })
        except Exception as exc:
            row.update({
                "Pred_Gain_dB": float("nan"), "SPICE_Gain_dB": float("nan"),
                "Gain_Error_dB": float("nan"), "Pred_BW_MHz": float("nan"),
                "SPICE_BW_MHz": float("nan"), "BW_Error_pct": float("nan"),
                "Pred_MinSatMargin_mV": float("nan"), "SPICE_MinSatMargin_mV": float("nan"),
                "Pred_Physical_Valid": False, "SPICE_Physical_Valid": False,
                "Physical_Match": False, "Limiting_Device": "error",
                "Agreement": "LOW", "Prediction_notes": "",
                "Error": f"{type(exc).__name__}: {exc}",
            })
        rows.append(row)
    return rows


def trust_boundary_summary(rows: List[dict]) -> dict:
    usable = [r for r in rows if math.isfinite(float(r.get("Distance", float("nan"))))]
    if not usable:
        return {
            "conservative_radius": 0.0, "first_non_high": None, "first_low": None,
            "status": "NO DATA", "validated_lower_bound": False,
        }
    ordered = sorted(usable, key=lambda r: float(r["Distance"]))
    radius = 0.0
    first_non_high = None
    first_low = None
    for row in ordered:
        d = float(row["Distance"])
        level = str(row.get("Agreement", "LOW"))
        if level == "HIGH" and first_non_high is None:
            radius = d
        elif first_non_high is None:
            first_non_high = d
        if level == "LOW" and first_low is None:
            first_low = d
    if first_non_high is None:
        max_d = max(float(r["Distance"]) for r in ordered)
        return {
            "conservative_radius": max_d,
            "first_non_high": None,
            "first_low": None,
            "status": f"No boundary found: HIGH through all tested points (validated radius ≥ {max_d:.2f})",
            "validated_lower_bound": True,
        }
    status = (
        f"Conservative all-HIGH radius = {radius:.2f}; first non-HIGH at {first_non_high:.2f}"
        + (f"; first LOW at {first_low:.2f}" if first_low is not None else "")
    )
    return {
        "conservative_radius": radius,
        "first_non_high": first_non_high,
        "first_low": first_low,
        "status": status,
        "validated_lower_bound": False,
    }


def tier_summary(rows: List[dict]) -> List[dict]:
    tiers = []
    order = ["Near", "Mid", "Far", "Extreme", "Deep"]
    for tier in order:
        subset = [r for r in rows if r.get("Tier") == tier]
        if not subset:
            continue
        gain_errors = [float(r["Gain_Error_dB"]) for r in subset if math.isfinite(float(r["Gain_Error_dB"]))]
        bw_errors = [float(r["BW_Error_pct"]) for r in subset if math.isfinite(float(r["BW_Error_pct"]))]
        tiers.append({
            "Tier": tier,
            "Points": len(subset),
            "HIGH": sum(r.get("Agreement") == "HIGH" for r in subset),
            "MEDIUM": sum(r.get("Agreement") == "MEDIUM" for r in subset),
            "LOW": sum(r.get("Agreement") == "LOW" for r in subset),
            "Physical_Invalid": sum(not bool(r.get("SPICE_Physical_Valid", False)) for r in subset),
            "Max_Gain_Error_dB": max(gain_errors) if gain_errors else float("nan"),
            "Max_BW_Error_pct": max(bw_errors) if bw_errors else float("nan"),
        })
    return tiers


def generator_self_test() -> dict:
    base = {"rd": 5600.0, "win": 20e-6, "wtail": 50e-6, "vbias": 0.70}
    steps = {"rd": 200.0, "win": 2e-6, "wtail": 5e-6, "vbias": 0.01}
    q = stress_validation_points(base, steps, "Quick · 36 points")
    d = stress_validation_points(base, steps, "Deep · 60 points")
    assert len(q) == 36
    assert len(d) == 60
    assert all(p["rd"] > 0 and p["win"] > 0 and p["wtail"] > 0 for p in d)
    return {"quick_points": len(q), "deep_points": len(d)}
