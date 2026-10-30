from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .autocircuit_core import resolve_ngspice

VDD = 1.8
KP = 200e-6
VTH = 0.5
LAMBDA = 0.02
L_UM = 1.0
CL_F = 5e-12
SAT_MARGIN = 0.05

RD_GRID = [3300, 4700, 5600, 6800, 8200, 10000]
WIN_GRID = [10, 15, 20, 25, 30, 40, 50]
WTAIL_GRID = [20, 30, 40, 50, 60, 80]
VBIAS_GRID = [0.60, 0.65, 0.70, 0.75, 0.80]

VCM_CORNERS = [0.95, 1.00, 1.05]

PROCESS_LIKE = {
    "SS": {"KP": 180e-6, "VTO": 0.53},
    "TT": {"KP": 200e-6, "VTO": 0.50},
    "FF": {"KP": 220e-6, "VTO": 0.47},
}
TEMP_CORNERS = [-40, 27, 125]

PVT_RD_GRID = [6800, 8200, 9000, 10000]
PVT_WIN_GRID = [60, 80, 100, 120, 140, 160, 180]
PVT_WTAIL_GRID = [20, 30, 40, 50]
PVT_VBIAS_GRID = [0.70, 0.72, 0.75]

EXP_RD = [6800, 7500, 8200, 9000, 10000]
EXP_WIN = [40, 50, 60, 70, 80]
EXP_WTAIL = [20, 30, 40, 50, 60]
EXP_VBIAS = [0.65, 0.67, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75]

_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def _extract(stdout: str, label: str) -> float:
    pat = rf"{re.escape(label)}\s*=\s*{_NUMBER}"
    found = re.findall(pat, stdout, flags=re.IGNORECASE)
    if not found:
        raise ValueError(f"Could not parse {label} from ngspice output")
    return float(found[-1])


def _write_real_tail_netlist(
    path: Path,
    rd_ohm: float,
    win_um: float,
    wtail_um: float,
    vbias_v: float,
    vcm_v: float,
    kp: float = KP,
    vto: float = VTH,
    temp_c: float = 27,
):
    text = f"""* SynapseFlow real-tail NMOS differential pair
VDD vdd 0 {VDD}
VINP inp 0 DC {vcm_v} AC 0.5 0
VINN inn 0 DC {vcm_v} AC 0.5 180
VBIAS vbias 0 DC {vbias_v}
RDP vdd outp {rd_ohm}
RDN vdd outn {rd_ohm}
CLP outp 0 5p
CLN outn 0 5p
M1 outp inp tail 0 NMOS W={win_um}u L=1u
M2 outn inn tail 0 NMOS W={win_um}u L=1u
MTAIL tail vbias 0 0 NMOS W={wtail_um}u L=1u
.model NMOS NMOS LEVEL=1 VTO={vto} KP={kp} LAMBDA={LAMBDA}
.temp {temp_c}
.control
op
print v(outp) v(outn) v(tail)
print @m1[id] @m2[id] @mtail[id]
print @m1[vgs] @m1[vds] @m1[vdsat]
print @m2[vgs] @m2[vds] @m2[vdsat]
print @mtail[vgs] @mtail[vds] @mtail[vdsat]
ac dec 100 10 1G
set wr_singlescale
wrdata diff_pair_ac.txt db(v(outp,outn))
quit
.endc
.end
"""
    path.write_text(text, encoding="utf-8")


def evaluate_diff_pair(
    RD_ohm: float,
    WIN_um: float,
    WTAIL_um: float,
    VBIAS_V: float,
    vcm_v: float,
    ngspice_path: str,
    kp: float = KP,
    vto: float = VTH,
    temp_c: float = 27,
    sat_margin_v: float = SAT_MARGIN,
):
    ngspice = resolve_ngspice(ngspice_path)

    with tempfile.TemporaryDirectory(prefix="autocircuit_diff_") as td:
        td_path = Path(td)
        circuit_path = td_path / "candidate.cir"
        ac_path = td_path / "diff_pair_ac.txt"
        _write_real_tail_netlist(
            circuit_path,
            RD_ohm,
            WIN_um,
            WTAIL_um,
            VBIAS_V,
            vcm_v,
            kp,
            vto,
            temp_c,
        )
        proc = subprocess.run(
            [ngspice, "-b", str(circuit_path)],
            cwd=td_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(stdout[-5000:])

        voutp = _extract(stdout, "v(outp)")
        voutn = _extract(stdout, "v(outn)")
        vtail = _extract(stdout, "v(tail)")
        id1 = _extract(stdout, "@m1[id]")
        id2 = _extract(stdout, "@m2[id]")
        idtail = _extract(stdout, "@mtail[id]")
        m1_vgs = _extract(stdout, "@m1[vgs]")
        m1_vds = _extract(stdout, "@m1[vds]")
        m1_vdsat = _extract(stdout, "@m1[vdsat]")
        m2_vds = _extract(stdout, "@m2[vds]")
        m2_vdsat = _extract(stdout, "@m2[vdsat]")
        tail_vgs = _extract(stdout, "@mtail[vgs]")
        tail_vds = _extract(stdout, "@mtail[vds]")
        tail_vdsat = _extract(stdout, "@mtail[vdsat]")

        arr = np.loadtxt(ac_path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        freq = arr[:, 0].astype(float)
        gain_vec = arr[:, -1].astype(float)
        gain_db = float(np.median(gain_vec[: min(5, len(gain_vec))]))
        target = gain_db - 3.0
        crossings = np.where(gain_vec <= target)[0]
        if len(crossings) == 0:
            bw_hz = float(freq[-1])
        else:
            i = int(crossings[0])
            if i == 0:
                bw_hz = float(freq[0])
            else:
                f1, f2 = float(freq[i - 1]), float(freq[i])
                g1, g2 = float(gain_vec[i - 1]), float(gain_vec[i])
                if abs(g2 - g1) < 1e-15:
                    bw_hz = f2
                else:
                    log_bw = math.log10(f1) + (target - g1) * (
                        math.log10(f2) - math.log10(f1)
                    ) / (g2 - g1)
                    bw_hz = 10**log_bw

        m1_sat = bool(m1_vds >= m1_vdsat + sat_margin_v)
        m2_sat = bool(m2_vds >= m2_vdsat + sat_margin_v)
        tail_sat = bool(tail_vds >= tail_vdsat + sat_margin_v)
        rail_valid = bool(
            0 <= vtail <= VDD and 0 <= voutp <= VDD and 0 <= voutn <= VDD
        )
        physical_valid = bool(m1_sat and m2_sat and tail_sat and rail_valid)

        return {
            "RD_ohm": float(RD_ohm),
            "WIN_um": float(WIN_um),
            "WTAIL_um": float(WTAIL_um),
            "VBIAS_V": float(VBIAS_V),
            "VCM_V": float(vcm_v),
            "KP": float(kp),
            "VTO": float(vto),
            "Temp_C": float(temp_c),
            "Gain_dB": gain_db,
            "Bandwidth_MHz": bw_hz / 1e6,
            "Power_mW": VDD * abs(idtail) * 1000.0,
            "VOUTP_V": voutp,
            "VOUTN_V": voutn,
            "VTAIL_V": vtail,
            "ID1_uA": id1 * 1e6,
            "ID2_uA": id2 * 1e6,
            "IDTAIL_uA": idtail * 1e6,
            "M1_VGS_V": m1_vgs,
            "M1_VDS_V": m1_vds,
            "M1_VDSAT_V": m1_vdsat,
            "M2_VDS_V": m2_vds,
            "M2_VDSAT_V": m2_vdsat,
            "Tail_VGS_V": tail_vgs,
            "Tail_VDS_V": tail_vds,
            "Tail_VDSAT_V": tail_vdsat,
            "M1_Saturation": m1_sat,
            "M2_Saturation": m2_sat,
            "Tail_Saturation": tail_sat,
            "Physical_Valid": physical_valid,
        }


def meets_spec(actual: dict, spec: dict) -> bool:
    return bool(
        actual["Physical_Valid"]
        and actual["Gain_dB"] >= spec["min_gain_db"]
        and actual["Bandwidth_MHz"] >= spec["min_bandwidth_mhz"]
        and actual["Power_mW"] <= spec["max_power_mw"]
    )


def analytic_real_tail(
    RD_ohm,
    WIN_um,
    WTAIL_um,
    VBIAS_V,
    vcm_v=1.0,
    kp=KP,
    vth=VTH,
):
    beta_in = kp * (WIN_um / L_UM)
    beta_tail = kp * (WTAIL_um / L_UM)
    vov_tail = VBIAS_V - vth
    if vov_tail <= 0:
        return {
            "Gain_dB": -999.0,
            "Bandwidth_MHz": 0.0,
            "Power_mW": 0.0,
            "Physical_Valid": False,
            "VTAIL_V": -999.0,
            "VOUT_V": -999.0,
        }

    itail = 0.5 * beta_tail * vov_tail**2
    id_each = itail / 2
    vov_in = math.sqrt(max(2 * id_each / beta_in, 0.0))
    vtail = vcm_v - vth - vov_in

    for _ in range(100):
        old = vtail
        itail = 0.5 * beta_tail * vov_tail**2 * max(1 + LAMBDA * vtail, 0.0)
        id_each = itail / 2
        vout = VDD - RD_ohm * id_each
        vds_in = vout - vtail
        lam = max(1 + LAMBDA * vds_in, 1e-12)
        vov_in = math.sqrt(max(2 * id_each / (beta_in * lam), 0.0))
        vtail = vcm_v - vth - vov_in
        if abs(vtail - old) < 1e-12:
            break

    itail = 0.5 * beta_tail * vov_tail**2 * max(1 + LAMBDA * vtail, 0.0)
    id_each = itail / 2
    vout = VDD - RD_ohm * id_each
    vds_in = vout - vtail
    gm = 2 * id_each / max(vov_in, 1e-12)
    ro = 1 / max(LAMBDA * id_each, 1e-18)
    rout = 1 / (1 / RD_ohm + 1 / ro)
    gain_db = 20 * math.log10(max(gm * rout, 1e-30))
    bw_mhz = 1 / (2 * math.pi * rout * CL_F) / 1e6
    power_mw = VDD * itail * 1000
    input_sat = bool(vov_in > 0 and vds_in >= vov_in + SAT_MARGIN)
    tail_sat = bool(vov_tail > 0 and vtail >= vov_tail + SAT_MARGIN)
    rail_valid = bool(0 <= vtail <= VDD and 0 <= vout <= VDD)
    return {
        "Gain_dB": gain_db,
        "Bandwidth_MHz": bw_mhz,
        "Power_mW": power_mw,
        "Physical_Valid": bool(input_sat and tail_sat and rail_valid),
        "VTAIL_V": vtail,
        "VOUT_V": vout,
    }


def _nearest_local_values(value, grid):
    idx = min(range(len(grid)), key=lambda i: abs(grid[i] - value))
    ids = sorted({idx, max(0, idx - 1), min(len(grid) - 1, idx + 1)})
    return [grid[i] for i in ids]


def _local_candidates(current, visited, grids=None):
    grids = grids or (RD_GRID, WIN_GRID, WTAIL_GRID, VBIAS_GRID)
    rg, wg, tg, bg = grids
    rds = _nearest_local_values(current["RD_ohm"], rg)
    wins = _nearest_local_values(current["WIN_um"], wg)
    tails = _nearest_local_values(current["WTAIL_um"], tg)
    biases = _nearest_local_values(current["VBIAS_V"], bg)
    current_key = (
        current["RD_ohm"],
        current["WIN_um"],
        current["WTAIL_um"],
        round(current["VBIAS_V"], 4),
    )
    out, seen = [], set()
    for rd in rds:
        for win in wins:
            for wt in tails:
                for vb in biases:
                    key = (rd, win, wt, round(vb, 4))
                    if key == current_key or key in visited or key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {"RD_ohm": rd, "WIN_um": win, "WTAIL_um": wt, "VBIAS_V": vb}
                    )
    return out


def _calibrated_predict(current_design, current_actual, candidate, vcm_v=1.0):
    ca = analytic_real_tail(**current_design, vcm_v=vcm_v)
    na = analytic_real_tail(**candidate, vcm_v=vcm_v)
    actual_gain_lin = 10 ** (current_actual["Gain_dB"] / 20)
    ca_gain_lin = 10 ** (ca["Gain_dB"] / 20)
    na_gain_lin = 10 ** (na["Gain_dB"] / 20)
    gain_lin = actual_gain_lin * na_gain_lin / max(ca_gain_lin, 1e-30)
    return {
        "Gain_dB": 20 * math.log10(max(gain_lin, 1e-30)),
        "Bandwidth_MHz": current_actual["Bandwidth_MHz"]
        * na["Bandwidth_MHz"]
        / max(ca["Bandwidth_MHz"], 1e-30),
        "Power_mW": current_actual["Power_mW"]
        * na["Power_mW"]
        / max(ca["Power_mW"], 1e-30),
        "Physical_Valid": na["Physical_Valid"],
    }


def _constraint_score(pred, spec):
    if not pred["Physical_Valid"]:
        return (1000.0, -999.0)
    gg = max(0.0, (spec["min_gain_db"] - pred["Gain_dB"]) / spec["min_gain_db"])
    bg = max(
        0.0,
        (spec["min_bandwidth_mhz"] - pred["Bandwidth_MHz"])
        / spec["min_bandwidth_mhz"],
    )
    pg = max(0.0, (pred["Power_mW"] - spec["max_power_mw"]) / spec["max_power_mw"])
    penalty = gg**2 + bg**2 + pg**2
    margins = [
        (pred["Gain_dB"] - spec["min_gain_db"]) / spec["min_gain_db"],
        (pred["Bandwidth_MHz"] - spec["min_bandwidth_mhz"])
        / spec["min_bandwidth_mhz"],
        (spec["max_power_mw"] - pred["Power_mW"]) / spec["max_power_mw"],
    ]
    return penalty, min(margins)


def run_nominal_agent(
    spec,
    ngspice_path,
    start_design=None,
    max_iterations=20,
):
    current = start_design or {
        "RD_ohm": 5600,
        "WIN_um": 20,
        "WTAIL_um": 50,
        "VBIAS_V": 0.70,
    }
    current = dict(current)
    visited, history = set(), []

    for iteration in range(1, max_iterations + 1):
        key = (
            current["RD_ohm"],
            current["WIN_um"],
            current["WTAIL_um"],
            round(current["VBIAS_V"], 4),
        )
        visited.add(key)
        actual = evaluate_diff_pair(**current, vcm_v=1.0, ngspice_path=ngspice_path)
        passed = meets_spec(actual, spec)
        history.append(
            {
                "Iteration": iteration,
                **current,
                "Gain_dB": actual["Gain_dB"],
                "Bandwidth_MHz": actual["Bandwidth_MHz"],
                "Power_mW": actual["Power_mW"],
                "VTAIL_V": actual["VTAIL_V"],
                "Physical_Valid": actual["Physical_Valid"],
                "PASS": passed,
            }
        )
        if passed:
            return {
                "success": True,
                "spice_evaluations": iteration,
                "final_design": current,
                "final_result": actual,
                "history": history,
            }
        candidates = _local_candidates(current, visited)
        if not candidates:
            break
        ranked = []
        for cand in candidates:
            pred = _calibrated_predict(current, actual, cand, vcm_v=1.0)
            penalty, robustness = _constraint_score(pred, spec)
            ranked.append((penalty, -robustness, cand))
        ranked.sort(key=lambda x: (x[0], x[1]))
        current = dict(ranked[0][2])

    return {
        "success": False,
        "spice_evaluations": len(history),
        "final_design": current,
        "final_result": actual,
        "history": history,
    }


def evaluate_vcm_corners(design, spec, ngspice_path, corners=VCM_CORNERS):
    rows = []
    for vcm in corners:
        actual = evaluate_diff_pair(**design, vcm_v=vcm, ngspice_path=ngspice_path)
        rows.append({**actual, "PASS": meets_spec(actual, spec)})
    df = pd.DataFrame(rows)
    return {
        "all_pass": bool(df["PASS"].all()),
        "all_physical": bool(df["Physical_Valid"].all()),
        "corner_results": df,
    }


def _score_candidate_all_vcm(candidate, spec, corners=VCM_CORNERS):
    penalties, margins = [], []
    for vcm in corners:
        pred = analytic_real_tail(**candidate, vcm_v=vcm)
        penalty, robustness = _constraint_score(pred, spec)
        penalties.append(penalty)
        margins.append(robustness)
    return max(penalties), float(np.mean(penalties)), min(margins)


def _expanded_robust_search(spec, ngspice_path, corners=VCM_CORNERS):
    candidates = []
    for rd in EXP_RD:
        for win in EXP_WIN:
            for wt in EXP_WTAIL:
                for vb in EXP_VBIAS:
                    cand = {"RD_ohm": rd, "WIN_um": win, "WTAIL_um": wt, "VBIAS_V": vb}
                    p, mp, r = _score_candidate_all_vcm(cand, spec, corners)
                    if p == 0:
                        candidates.append((mp, -r, cand))
    candidates.sort(key=lambda x: (x[0], x[1]))
    for rank, (_, _, cand) in enumerate(candidates[:20], start=1):
        actual = evaluate_vcm_corners(cand, spec, ngspice_path, corners)
        if actual["all_pass"]:
            return {
                "success": True,
                "rank": rank,
                "candidate_count": len(candidates),
                "design": cand,
                "corner_results": actual["corner_results"],
                "spice_evaluations": rank * len(corners),
            }
    return {
        "success": False,
        "rank": None,
        "candidate_count": len(candidates),
        "design": None,
        "corner_results": None,
        "spice_evaluations": min(20, len(candidates)) * len(corners),
    }


def run_vcm_robust_agent(
    spec,
    ngspice_path,
    start_design,
    max_iterations=20,
    allow_expansion=True,
):
    current = dict(start_design)
    visited, history = set(), []
    total_spice = 0
    for iteration in range(1, max_iterations + 1):
        key = (
            current["RD_ohm"],
            current["WIN_um"],
            current["WTAIL_um"],
            round(current["VBIAS_V"], 4),
        )
        visited.add(key)
        actual = evaluate_vcm_corners(current, spec, ngspice_path)
        total_spice += len(VCM_CORNERS)
        df = actual["corner_results"]
        history.append(
            {
                "Iteration": iteration,
                **current,
                "Corners_PASS": int(df["PASS"].sum()),
                "Worst_Gain_dB": float(df["Gain_dB"].min()),
                "Worst_BW_MHz": float(df["Bandwidth_MHz"].min()),
                "Worst_Power_mW": float(df["Power_mW"].max()),
                "Min_VTAIL_V": float(df["VTAIL_V"].min()),
                "All_Physical": actual["all_physical"],
                "All_Corners_PASS": actual["all_pass"],
            }
        )
        if actual["all_pass"]:
            return {
                "success": True,
                "expanded": False,
                "iterations": iteration,
                "spice_evaluations": total_spice,
                "final_design": current,
                "final_corner_results": df,
                "history": history,
            }
        candidates = _local_candidates(current, visited)
        if not candidates:
            break
        ranked = []
        for cand in candidates:
            p, mp, r = _score_candidate_all_vcm(cand, spec)
            ranked.append((p, mp, -r, cand))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        current = dict(ranked[0][3])

    if allow_expansion:
        expanded = _expanded_robust_search(spec, ngspice_path)
        total_spice += expanded["spice_evaluations"]
        if expanded["success"]:
            df = expanded["corner_results"]
            history.append(
                {
                    "Iteration": "Expanded",
                    **expanded["design"],
                    "Corners_PASS": int(df["PASS"].sum()),
                    "Worst_Gain_dB": float(df["Gain_dB"].min()),
                    "Worst_BW_MHz": float(df["Bandwidth_MHz"].min()),
                    "Worst_Power_mW": float(df["Power_mW"].max()),
                    "Min_VTAIL_V": float(df["VTAIL_V"].min()),
                    "All_Physical": bool(df["Physical_Valid"].all()),
                    "All_Corners_PASS": True,
                }
            )
            return {
                "success": True,
                "expanded": True,
                "expanded_candidate_count": expanded["candidate_count"],
                "iterations": len(history),
                "spice_evaluations": total_spice,
                "final_design": expanded["design"],
                "final_corner_results": df,
                "history": history,
            }

    return {
        "success": False,
        "expanded": False,
        "iterations": len(history),
        "spice_evaluations": total_spice,
        "final_design": current,
        "final_corner_results": actual["corner_results"],
        "history": history,
    }


def evaluate_pvt27(design, spec, ngspice_path):
    rows = []
    for process, model in PROCESS_LIKE.items():
        for temp in TEMP_CORNERS:
            for vcm in VCM_CORNERS:
                r = evaluate_diff_pair(
                    **design,
                    vcm_v=vcm,
                    ngspice_path=ngspice_path,
                    kp=model["KP"],
                    vto=model["VTO"],
                    temp_c=temp,
                )
                rows.append({"Process": process, **r, "PASS": meets_spec(r, spec)})
    df = pd.DataFrame(rows)
    return {
        "all_pass": bool(df["PASS"].all()),
        "pass_count": int(df["PASS"].sum()),
        "physical_count": int(df["Physical_Valid"].sum()),
        "worst_gain_db": float(df["Gain_dB"].min()),
        "worst_bw_mhz": float(df["Bandwidth_MHz"].min()),
        "worst_power_mw": float(df["Power_mW"].max()),
        "corner_results": df,
    }


def _infer_effective_params(actual_row, current_design):
    vov_in = float(actual_row["M1_VDSAT_V"])
    vds_in = float(actual_row["M1_VDS_V"])
    id_in = float(actual_row["ID1_uA"]) * 1e-6
    vgs_in = float(actual_row["M1_VGS_V"])
    vth_in = vgs_in - vov_in
    kp_in = 2 * id_in / max(
        (current_design["WIN_um"] / L_UM) * vov_in**2 * (1 + LAMBDA * vds_in),
        1e-30,
    )

    vov_tail = float(actual_row["Tail_VDSAT_V"])
    vds_tail = float(actual_row["Tail_VDS_V"])
    id_tail = float(actual_row["IDTAIL_uA"]) * 1e-6
    vth_tail = current_design["VBIAS_V"] - vov_tail
    kp_tail = 2 * id_tail / max(
        (current_design["WTAIL_um"] / L_UM)
        * vov_tail**2
        * (1 + LAMBDA * vds_tail),
        1e-30,
    )
    return float(np.mean([kp_in, kp_tail])), float(np.mean([vth_in, vth_tail]))


def _predict_effective(design, vcm, kp_eff, vth_eff):
    return analytic_real_tail(**design, vcm_v=vcm, kp=kp_eff, vth=vth_eff)


def _calibrated_pvt_predict(current_design, actual_row, candidate):
    kp_eff, vth_eff = _infer_effective_params(actual_row, current_design)
    vcm = float(actual_row["VCM_V"])
    ca = _predict_effective(current_design, vcm, kp_eff, vth_eff)
    na = _predict_effective(candidate, vcm, kp_eff, vth_eff)
    if not na["Physical_Valid"]:
        return {**na, "Physical_Valid": False}
    actual_gain_lin = 10 ** (float(actual_row["Gain_dB"]) / 20)
    ca_gain_lin = 10 ** (ca["Gain_dB"] / 20)
    na_gain_lin = 10 ** (na["Gain_dB"] / 20)
    gain_lin = actual_gain_lin * na_gain_lin / max(ca_gain_lin, 1e-30)
    return {
        "Gain_dB": 20 * math.log10(max(gain_lin, 1e-30)),
        "Bandwidth_MHz": float(actual_row["Bandwidth_MHz"])
        * na["Bandwidth_MHz"]
        / max(ca["Bandwidth_MHz"], 1e-30),
        "Power_mW": float(actual_row["Power_mW"])
        * na["Power_mW"]
        / max(ca["Power_mW"], 1e-30),
        "Physical_Valid": True,
    }


def _score_pvt_candidate(current_design, current_corner_df, candidate, spec):
    invalid_count = 0
    fail_count = 0
    penalties, margins = [], []
    for _, row in current_corner_df.iterrows():
        pred = _calibrated_pvt_predict(current_design, row, candidate)
        if not pred["Physical_Valid"]:
            invalid_count += 1
            fail_count += 1
            penalties.append(1000.0)
            margins.append(-999.0)
            continue
        penalty, robustness = _constraint_score(pred, spec)
        if penalty > 0:
            fail_count += 1
        penalties.append(penalty)
        margins.append(robustness)
    move_cost = sum(current_design[k] != candidate[k] for k in current_design)
    return (
        invalid_count,
        fail_count,
        max(penalties),
        float(np.mean(penalties)),
        -min(margins),
        move_cost,
    )


def run_pvt_aware_agent(
    spec,
    ngspice_path,
    start_design=None,
    max_iterations=8,
):
    current = start_design or {
        "RD_ohm": 9000,
        "WIN_um": 80,
        "WTAIL_um": 20,
        "VBIAS_V": 0.75,
    }
    current = dict(current)
    visited, history = set(), []
    total_spice = 0

    for iteration in range(1, max_iterations + 1):
        key = (
            current["RD_ohm"],
            current["WIN_um"],
            current["WTAIL_um"],
            round(current["VBIAS_V"], 4),
        )
        visited.add(key)
        actual = evaluate_pvt27(current, spec, ngspice_path)
        total_spice += 27
        history.append(
            {
                "Iteration": iteration,
                **current,
                "PASS_Count": actual["pass_count"],
                "Physical_Count": actual["physical_count"],
                "Worst_Gain_dB": actual["worst_gain_db"],
                "Worst_BW_MHz": actual["worst_bw_mhz"],
                "Worst_Power_mW": actual["worst_power_mw"],
                "Total_SPICE_Evals": total_spice,
                "All_27_PASS": actual["all_pass"],
            }
        )
        if actual["all_pass"]:
            return {
                "success": True,
                "iterations": iteration,
                "spice_evaluations": total_spice,
                "final_design": current,
                "final_corner_results": actual["corner_results"],
                "history": history,
            }
        candidates = _local_candidates(
            current,
            visited,
            grids=(PVT_RD_GRID, PVT_WIN_GRID, PVT_WTAIL_GRID, PVT_VBIAS_GRID),
        )
        if not candidates:
            break
        ranked = [
            (_score_pvt_candidate(current, actual["corner_results"], cand, spec), cand)
            for cand in candidates
        ]
        ranked.sort(key=lambda x: x[0])
        current = dict(ranked[0][1])

    return {
        "success": False,
        "iterations": len(history),
        "spice_evaluations": total_spice,
        "final_design": current,
        "final_corner_results": actual["corner_results"],
        "history": history,
    }
