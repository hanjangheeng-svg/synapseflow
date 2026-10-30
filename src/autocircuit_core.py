from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Design space / model constants
# ---------------------------------------------------------------------

RD_GRID = [2200, 3300, 4700, 5600, 6800, 8200, 10000]
W_GRID = [10, 15, 20, 25, 30, 40, 50]
VG_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]

VDD = 1.8
VTH = 0.5
KP = 200e-6
LAMBDA = 0.02
DEFAULT_L_UM = 1.0
DEFAULT_CL_PF = 5.0


def resolve_ngspice(explicit_path: str | None = None) -> str:
    candidates = [
        explicit_path,
        os.getenv("NGSPICE_PATH"),
        shutil.which("ngspice_con.exe"),
        shutil.which("ngspice.exe"),
        shutil.which("ngspice_con"),
        shutil.which("ngspice"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    raise FileNotFoundError(
        "ngspice executable not found. Set NGSPICE_PATH or provide the executable path."
    )


# ---------------------------------------------------------------------
# LLM parser: raw extraction only
# ---------------------------------------------------------------------

def _deterministic_raw_spec_fallback(user_request: str) -> dict:
    """Parse explicit value+unit tokens without doing unit conversion.

    This is only a resilience fallback for local-LLM outages. It deliberately
    accepts only the same explicit units used by SynapseFlow's normal parser.
    """
    text = user_request.replace("µ", "u").replace("μ", "u")
    token_re = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*(dB|MHz|kHz|Hz|mW|uW|W)\b",
        re.IGNORECASE,
    )

    found = {}
    canonical = {
        "db": "dB",
        "mhz": "MHz",
        "khz": "kHz",
        "hz": "Hz",
        "mw": "mW",
        "uw": "uW",
        "w": "W",
    }

    for match in token_re.finditer(text):
        value = float(match.group(1))
        unit = canonical[match.group(2).lower()]

        if unit == "dB" and "gain_value" not in found:
            found["gain_value"] = value
            found["gain_unit"] = unit
        elif unit in {"Hz", "kHz", "MHz"} and "bandwidth_value" not in found:
            found["bandwidth_value"] = value
            found["bandwidth_unit"] = unit
        elif unit in {"W", "mW", "uW"} and "power_value" not in found:
            found["power_value"] = value
            found["power_unit"] = unit

    required = {
        "gain_value",
        "gain_unit",
        "bandwidth_value",
        "bandwidth_unit",
        "power_value",
        "power_unit",
    }
    missing = sorted(required - set(found))
    if missing:
        raise ValueError(
            "Deterministic fallback could not find all explicit value/unit tokens: "
            + ", ".join(missing)
        )

    found["_parser_source"] = "deterministic_fallback"
    found["_parser_note"] = (
        "Local LLM request failed, so SynapseFlow copied explicit value/unit "
        "tokens deterministically. No arithmetic or unit conversion was done by the fallback."
    )
    return found


def extract_raw_circuit_spec(
    user_request: str,
    ollama_url: str = "",
    model: str = "qwen3:4b",
):
    schema = {
        "type": "object",
        "properties": {
            "gain_value": {"type": "number"},
            "gain_unit": {"type": "string", "enum": ["dB"]},
            "bandwidth_value": {"type": "number"},
            "bandwidth_unit": {
                "type": "string",
                "enum": ["Hz", "kHz", "MHz"],
            },
            "power_value": {"type": "number"},
            "power_unit": {
                "type": "string",
                "enum": ["W", "mW", "uW"],
            },
        },
        "required": [
            "gain_value",
            "gain_unit",
            "bandwidth_value",
            "bandwidth_unit",
            "power_value",
            "power_unit",
        ],
    }

    normalized_text = user_request.replace("µ", "u").replace("μ", "u")

    # Public/cloud deployments may intentionally run without Ollama. In that
    # case, skip the network call entirely and use the explicit value/unit
    # parser. This keeps the app self-contained while preserving optional
    # Ollama support when OLLAMA_URL is configured.
    if not str(ollama_url or "").strip():
        fallback = _deterministic_raw_spec_fallback(normalized_text)
        fallback["_parser_warning"] = (
            "No Ollama endpoint is configured; deterministic value/unit parsing was used."
        )
        return fallback

    prompt = f"""
Extract the numerical circuit requirements from the user's request.

User request:
"{normalized_text}"

IMPORTANT:
- COPY the numerical values exactly as written.
- DO NOT convert units.
- DO NOT perform arithmetic.
- DO NOT change numbers.
- Only identify each value and its unit.
- Treat uW as uW.

Examples:

"bandwidth 4500 kHz"
-> bandwidth_value = 4500
-> bandwidth_unit = "kHz"

"power 300 uW"
-> power_value = 300
-> power_unit = "uW"

"bandwidth 2.5 MHz"
-> bandwidth_value = 2.5
-> bandwidth_unit = "MHz"
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract circuit specification values and units. "
                    "Never convert units or perform calculations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 120},
        "keep_alive": "10m",
    }

    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    last_error = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))

            raw = data.get("message", {}).get("content", "")
            if not raw.strip():
                raise RuntimeError("Qwen returned an empty specification.")

            parsed = json.loads(raw)
            parsed["_parser_source"] = "local_llm"
            return parsed

        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            detail = body[:700] if body else str(exc.reason)
            last_error = f"Ollama HTTP {exc.code}: {detail}"
        except urllib.error.URLError as exc:
            last_error = f"Ollama connection error: {exc.reason}"
        except (TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = f"Ollama parser error: {exc}"

        if attempt == 0:
            time.sleep(1.0)

    try:
        fallback = _deterministic_raw_spec_fallback(normalized_text)
        fallback["_parser_warning"] = last_error or "Local LLM parser unavailable."
        return fallback
    except Exception as fallback_error:
        raise RuntimeError(
            f"{last_error or 'Local LLM parser failed.'} "
            f"Deterministic fallback also failed: {fallback_error}"
        ) from fallback_error


def normalize_circuit_spec(raw_spec: dict) -> dict:
    gain_db = float(raw_spec["gain_value"])

    bw_value = float(raw_spec["bandwidth_value"])
    bw_unit = raw_spec["bandwidth_unit"]
    bw_multiplier = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}
    if bw_unit not in bw_multiplier:
        raise ValueError(f"Unsupported bandwidth unit: {bw_unit}")
    bandwidth_hz = bw_value * bw_multiplier[bw_unit]

    power_value = float(raw_spec["power_value"])
    power_unit = raw_spec["power_unit"]
    power_to_mw = {"W": 1000.0, "mW": 1.0, "uW": 0.001}
    if power_unit not in power_to_mw:
        raise ValueError(f"Unsupported power unit: {power_unit}")
    power_mw = power_value * power_to_mw[power_unit]

    return {
        "min_gain_db": gain_db,
        "min_bandwidth_hz": bandwidth_hz,
        "max_power_mw": power_mw,
    }


def parse_circuit_spec_v2(
    user_request: str,
    ollama_url: str = "",
    model: str = "qwen3:4b",
):
    raw = extract_raw_circuit_spec(
        user_request,
        ollama_url=ollama_url,
        model=model,
    )
    return raw, normalize_circuit_spec(raw)


# ---------------------------------------------------------------------
# SPICE
# ---------------------------------------------------------------------

def _write_common_source_netlist(
    path: Path,
    rd_ohm: float,
    w_um: float,
    vg_bias: float,
    l_um: float = DEFAULT_L_UM,
    cl_pf: float = DEFAULT_CL_PF,
):
    text = f"""* SynapseFlow common-source amplifier
VDD vdd 0 DC {VDD}
VIN in 0 DC {vg_bias} AC 1
RD vdd out {rd_ohm}
M1 out in 0 0 NMOS W={w_um}u L={l_um}u
CL out 0 {cl_pf}p

.model NMOS NMOS LEVEL=1 VTO={VTH} KP={KP} LAMBDA={LAMBDA}

.control
op
print v(out)
print @m1[id]
ac dec 100 1k 100Meg
wrdata ac_result.txt db(v(out))
quit
.endc

.end
"""
    path.write_text(text, encoding="utf-8")


_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def _extract_stdout_value(stdout: str, patterns: list[str]) -> float:
    for pat in patterns:
        matches = re.findall(pat, stdout, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    raise ValueError(f"Could not parse value from ngspice output. Patterns: {patterns}")


def run_and_evaluate(
    rd_ohm: float,
    w_um: float,
    vg_bias: float,
    ngspice_path: str,
    l_um: float = DEFAULT_L_UM,
    cl_pf: float = DEFAULT_CL_PF,
):
    ngspice = resolve_ngspice(ngspice_path)

    with tempfile.TemporaryDirectory(prefix="autocircuit_") as td:
        td_path = Path(td)
        circuit_path = td_path / "candidate.cir"
        ac_path = td_path / "ac_result.txt"

        _write_common_source_netlist(
            circuit_path,
            rd_ohm=rd_ohm,
            w_um=w_um,
            vg_bias=vg_bias,
            l_um=l_um,
            cl_pf=cl_pf,
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
            return {
                "success": False,
                "error": stdout[-4000:],
            }

        try:
            vout = _extract_stdout_value(
                stdout,
                [
                    rf"v\(out\)\s*=\s*{_NUMBER}",
                    rf"vout\s*=\s*{_NUMBER}",
                ],
            )
            idrain = abs(
                _extract_stdout_value(
                    stdout,
                    [
                        rf"@m1\[id\]\s*=\s*{_NUMBER}",
                        rf"m1#branch\s*=\s*{_NUMBER}",
                    ],
                )
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"DC parse error: {exc}\n{stdout[-4000:]}",
            }

        if not ac_path.exists():
            return {
                "success": False,
                "error": f"AC output file not created.\n{stdout[-4000:]}",
            }

        try:
            arr = np.loadtxt(ac_path)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            freqs = arr[:, 0].astype(float)
            gain_db_vec = arr[:, -1].astype(float)

            if len(freqs) < 2:
                raise ValueError("Not enough AC points.")

            n_low = min(5, len(gain_db_vec))
            midband_gain_db = float(np.mean(gain_db_vec[:n_low]))
            target_db = midband_gain_db - 3.0

            below = np.where(gain_db_vec <= target_db)[0]
            bandwidth_hz = float(freqs[below[0]]) if len(below) else float(freqs[-1])

        except Exception as exc:
            return {
                "success": False,
                "error": f"AC parse error: {exc}\n{stdout[-4000:]}",
            }

        power_mw = VDD * idrain * 1000.0

        return {
            "success": True,
            "vout_dc": float(vout),
            "idrain_a": float(idrain),
            "power_mw": float(power_mw),
            "gain_db": float(midband_gain_db),
            "bandwidth_hz": float(bandwidth_hz),
        }


def check_operating_region(metrics: dict, vg_bias: float, vth: float = VTH, margin: float = 0.05):
    vds = float(metrics["vout_dc"])
    vgs = float(vg_bias)
    vov = vgs - vth
    required_vds = vov + margin
    valid = bool(vov > 0 and vds >= required_vds)

    return {
        "valid": valid,
        "vgs_v": vgs,
        "vds_v": vds,
        "vov_v": vov,
        "required_vds_v": required_vds,
        "region": "SATURATION" if valid else "TRIODE / INVALID",
        "reason": "Operating point is valid." if valid else "Insufficient drain headroom.",
    }


def evaluate_candidate(
    rd_ohm: float,
    w_um: float,
    vg_bias: float,
    spec: dict,
    ngspice_path: str,
    l_um: float = DEFAULT_L_UM,
    cl_pf: float = DEFAULT_CL_PF,
):
    sim = run_and_evaluate(
        rd_ohm=rd_ohm,
        w_um=w_um,
        vg_bias=vg_bias,
        ngspice_path=ngspice_path,
        l_um=l_um,
        cl_pf=cl_pf,
    )

    if not sim.get("success"):
        return {
            "success": False,
            "pass": False,
            "failures": ["SPICE_ERROR"],
            "error": sim.get("error", "Unknown SPICE error"),
        }

    op = check_operating_region(sim, vg_bias)

    failures = []
    if not op["valid"]:
        failures.append("OPERATING_REGION")
    if sim["gain_db"] < spec["min_gain_db"]:
        failures.append("GAIN")
    if sim["bandwidth_hz"] < spec["min_bandwidth_hz"]:
        failures.append("BANDWIDTH")
    if sim["power_mw"] > spec["max_power_mw"]:
        failures.append("POWER")

    return {
        "success": True,
        "pass": len(failures) == 0,
        "failures": failures,
        "parameters": {
            "RD_ohm": float(rd_ohm),
            "W_um": float(w_um),
            "VG_V": float(vg_bias),
            "L_um": float(l_um),
            "CL_pF": float(cl_pf),
        },
        "metrics": {
            "VOUT_V": sim["vout_dc"],
            "Gain_dB": sim["gain_db"],
            "BW_MHz": sim["bandwidth_hz"] / 1e6,
            "Power_mW": sim["power_mw"],
        },
        "operating_point": op,
    }


# ---------------------------------------------------------------------
# Physics surrogate / search
# ---------------------------------------------------------------------

def analytic_common_source(
    rd_ohm,
    w_um,
    vg_v,
    l_um=DEFAULT_L_UM,
    cl_pf=DEFAULT_CL_PF,
    vdd=VDD,
    vth=VTH,
    kp=KP,
    lamb=LAMBDA,
):
    vov = vg_v - vth

    if vov <= 0:
        return {
            "valid": False,
            "gain_db": -999.0,
            "gain_linear": 0.0,
            "bw_hz": 0.0,
            "power_mw": 0.0,
            "vds_v": vdd,
            "id_a": 0.0,
        }

    i0 = 0.5 * kp * (w_um / l_um) * (vov ** 2)
    id_a = i0 * (1 + lamb * vdd) / (1 + i0 * lamb * rd_ohm)
    vds = vdd - id_a * rd_ohm
    required_vds = vov + 0.05
    valid = bool(vds >= required_vds)

    gm = kp * (w_um / l_um) * vov * (1 + lamb * vds)
    ro = 1 / (lamb * id_a) if lamb > 0 and id_a > 0 else 1e99
    rout = 1 / (1 / rd_ohm + 1 / ro)
    gain_linear = gm * rout if valid else max(gm * rout, 1e-12)
    gain_db = 20 * math.log10(max(gain_linear, 1e-12))
    cl_f = cl_pf * 1e-12
    bw_hz = 1 / (2 * math.pi * rout * cl_f)
    power_mw = vdd * id_a * 1000

    return {
        "valid": valid,
        "gain_db": gain_db,
        "gain_linear": gain_linear,
        "bw_hz": bw_hz,
        "power_mw": power_mw,
        "vds_v": vds,
        "id_a": id_a,
    }


def calibrated_predict(current_result: dict, candidate: dict):
    p = current_result["parameters"]
    m = current_result["metrics"]

    current_ana = analytic_common_source(
        p["RD_ohm"],
        p["W_um"],
        p["VG_V"],
        l_um=p["L_um"],
        cl_pf=p["CL_pF"],
    )

    candidate_ana = analytic_common_source(
        candidate["RD_ohm"],
        candidate["W_um"],
        candidate["VG_V"],
        l_um=p["L_um"],
        cl_pf=p["CL_pF"],
    )

    measured_gain_linear = 10 ** (m["Gain_dB"] / 20)
    gain_ratio = (
        candidate_ana["gain_linear"] / current_ana["gain_linear"]
        if current_ana["gain_linear"] > 0
        else 1.0
    )
    predicted_gain_linear = measured_gain_linear * gain_ratio
    predicted_gain_db = 20 * math.log10(max(predicted_gain_linear, 1e-12))

    current_bw_hz = m["BW_MHz"] * 1e6
    bw_ratio = (
        candidate_ana["bw_hz"] / current_ana["bw_hz"]
        if current_ana["bw_hz"] > 0
        else 1.0
    )
    predicted_bw_hz = current_bw_hz * bw_ratio

    power_ratio = (
        candidate_ana["power_mw"] / current_ana["power_mw"]
        if current_ana["power_mw"] > 0
        else 1.0
    )
    predicted_power_mw = m["Power_mW"] * power_ratio

    return {
        "gain_db": predicted_gain_db,
        "bw_hz": predicted_bw_hz,
        "power_mw": predicted_power_mw,
        "valid": candidate_ana["valid"],
        "vds_v": candidate_ana["vds_v"],
    }


def grid_index(grid, value):
    return min(range(len(grid)), key=lambda i: abs(grid[i] - value))


def generate_local_candidates(current_result: dict, visited: set):
    p = current_result["parameters"]

    rd_idx = grid_index(RD_GRID, p["RD_ohm"])
    w_idx = grid_index(W_GRID, p["W_um"])
    vg_idx = grid_index(VG_GRID, p["VG_V"])

    candidates = []
    seen = set()

    for dr in (-1, 0, 1):
        for dw in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if dr == 0 and dw == 0 and dv == 0:
                    continue

                new_rd = RD_GRID[max(0, min(len(RD_GRID) - 1, rd_idx + dr))]
                new_w = W_GRID[max(0, min(len(W_GRID) - 1, w_idx + dw))]
                new_vg = VG_GRID[max(0, min(len(VG_GRID) - 1, vg_idx + dv))]

                key = (new_rd, new_w, round(float(new_vg), 2))
                current_key = (
                    p["RD_ohm"],
                    p["W_um"],
                    round(float(p["VG_V"]), 2),
                )

                if key == current_key or key in visited or key in seen:
                    continue

                seen.add(key)
                candidates.append(
                    {
                        "RD_ohm": new_rd,
                        "W_um": new_w,
                        "VG_V": float(new_vg),
                    }
                )

    return candidates


def predicted_penalty(prediction: dict, spec: dict):
    gain_target = spec["min_gain_db"]
    bw_target = spec["min_bandwidth_hz"]
    power_limit = spec["max_power_mw"]

    gain_gap = max(
        0.0,
        (gain_target - prediction["gain_db"]) / max(gain_target, 1.0),
    )
    bw_gap = max(
        0.0,
        (bw_target - prediction["bw_hz"]) / bw_target,
    )
    power_gap = max(
        0.0,
        (prediction["power_mw"] - power_limit) / power_limit,
    )
    op_penalty = 0.0 if prediction["valid"] else 2.0

    return gain_gap ** 2 + bw_gap ** 2 + power_gap ** 2 + op_penalty


def choose_physics_candidate_v2(current_result: dict, spec: dict, visited: set):
    candidates = generate_local_candidates(current_result, visited)
    if not candidates:
        return None

    ranked = []
    for candidate in candidates:
        prediction = calibrated_predict(current_result, candidate)
        penalty = predicted_penalty(prediction, spec)
        ranked.append(
            {
                **candidate,
                "pred_gain_db": prediction["gain_db"],
                "pred_bw_mhz": prediction["bw_hz"] / 1e6,
                "pred_power_mw": prediction["power_mw"],
                "pred_valid": prediction["valid"],
                "penalty": penalty,
            }
        )

    ranked.sort(key=lambda x: x["penalty"])
    return ranked[0], ranked


def run_physics_guided_agent(
    spec: dict,
    ngspice_path: str,
    start_rd=4700,
    start_w=20,
    start_vg=0.80,
    max_iterations=20,
):
    current = evaluate_candidate(
        rd_ohm=start_rd,
        w_um=start_w,
        vg_bias=start_vg,
        spec=spec,
        ngspice_path=ngspice_path,
    )

    if not current.get("success"):
        return {
            "success": False,
            "result": current,
            "history": [],
            "spice_runs": 1,
        }

    visited = set()
    history = []
    spice_runs = 1

    for iteration in range(1, max_iterations + 1):
        p = current["parameters"]
        m = current["metrics"]
        op = current["operating_point"]

        key = (
            p["RD_ohm"],
            p["W_um"],
            round(float(p["VG_V"]), 2),
        )
        visited.add(key)

        row = {
            "Iteration": iteration,
            "RD_ohm": p["RD_ohm"],
            "W_um": p["W_um"],
            "VG_V": p["VG_V"],
            "Gain_dB": m["Gain_dB"],
            "BW_MHz": m["BW_MHz"],
            "Power_mW": m["Power_mW"],
            "Region": op["region"],
            "Failures": ",".join(current["failures"]),
            "PASS": bool(current["pass"]),
        }

        if current["pass"]:
            history.append(row)
            return {
                "success": True,
                "result": current,
                "history": history,
                "spice_runs": spice_runs,
            }

        choice = choose_physics_candidate_v2(current, spec, visited)
        if choice is None:
            history.append(row)
            break

        next_candidate, ranking = choice
        row.update(
            {
                "Pred_next_RD": next_candidate["RD_ohm"],
                "Pred_next_W": next_candidate["W_um"],
                "Pred_next_VG": next_candidate["VG_V"],
                "Pred_next_gain_dB": next_candidate["pred_gain_db"],
                "Pred_next_BW_MHz": next_candidate["pred_bw_mhz"],
                "Pred_next_power_mW": next_candidate["pred_power_mw"],
                "Pred_next_penalty": next_candidate["penalty"],
            }
        )
        history.append(row)

        current = evaluate_candidate(
            rd_ohm=next_candidate["RD_ohm"],
            w_um=next_candidate["W_um"],
            vg_bias=next_candidate["VG_V"],
            spec=spec,
            ngspice_path=ngspice_path,
        )
        spice_runs += 1

        if not current.get("success"):
            break

    return {
        "success": False,
        "result": current,
        "history": history,
        "spice_runs": spice_runs,
    }


def design_from_natural_language(
    user_request: str,
    ngspice_path: str,
    ollama_url: str = "",
    model: str = "qwen3:4b",
    start_rd=4700,
    start_w=20,
    start_vg=0.80,
):
    raw_spec, spec = parse_circuit_spec_v2(
        user_request,
        ollama_url=ollama_url,
        model=model,
    )
    result = run_physics_guided_agent(
        spec=spec,
        ngspice_path=ngspice_path,
        start_rd=start_rd,
        start_w=start_w,
        start_vg=start_vg,
    )
    return {
        "request": user_request,
        "raw_spec": raw_spec,
        "spec": spec,
        "design_result": result,
    }
