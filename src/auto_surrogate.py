from __future__ import annotations

import cmath
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .autocircuit_core import resolve_ngspice


_SUFFIX = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

_NUM_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)?\s*$"
)


def spice_number(value: str, params: Optional[Dict[str, float]] = None) -> float:
    """Parse a basic SPICE numeric value or a simple {PARAM} reference."""
    params = params or {}
    s = value.strip()
    if s.startswith("{") and s.endswith("}"):
        key = s[1:-1].strip().lower()
        if key not in params:
            raise ValueError(f"Unknown .param reference: {value}")
        return float(params[key])
    if s.lower() in params:
        return float(params[s.lower()])

    m = _NUM_RE.match(s)
    if not m:
        raise ValueError(f"Unsupported SPICE numeric expression: {value}")
    base = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if not suffix:
        return base
    if suffix in _SUFFIX:
        return base * _SUFFIX[suffix]
    # SPICE commonly ignores characters after a recognized scale suffix.
    if suffix.startswith("meg"):
        return base * 1e6
    if suffix[0] in _SUFFIX:
        return base * _SUFFIX[suffix[0]]
    return base


def _logical_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+") and out:
            out[-1] += " " + stripped[1:].strip()
        else:
            # $ starts an inline comment in many SPICE decks.
            if "$" in line:
                line = line.split("$", 1)[0].rstrip()
            if line.strip():
                out.append(line.strip())
    return out


@dataclass
class Resistor:
    name: str
    n1: str
    n2: str
    value: float


@dataclass
class Capacitor:
    name: str
    n1: str
    n2: str
    value: float


@dataclass
class VSource:
    name: str
    nplus: str
    nminus: str
    ac: complex = 0j


@dataclass
class Mosfet:
    name: str
    d: str
    g: str
    s: str
    b: str
    model: str
    w: Optional[float] = None
    l: Optional[float] = None


@dataclass
class ParsedNetlist:
    text: str
    params: Dict[str, float] = field(default_factory=dict)
    model_types: Dict[str, str] = field(default_factory=dict)
    resistors: List[Resistor] = field(default_factory=list)
    capacitors: List[Capacitor] = field(default_factory=list)
    vsources: List[VSource] = field(default_factory=list)
    mosfets: List[Mosfet] = field(default_factory=list)
    unsupported: List[str] = field(default_factory=list)

    @property
    def nodes(self) -> List[str]:
        seen = set()
        order = []
        for e in [*self.resistors, *self.capacitors]:
            for n in (e.n1, e.n2):
                if n not in ("0", "gnd") and n not in seen:
                    seen.add(n); order.append(n)
        for v in self.vsources:
            for n in (v.nplus, v.nminus):
                if n not in ("0", "gnd") and n not in seen:
                    seen.add(n); order.append(n)
        for m in self.mosfets:
            for n in (m.d, m.g, m.s, m.b):
                if n not in ("0", "gnd") and n not in seen:
                    seen.add(n); order.append(n)
        return order


def parse_netlist(text: str) -> ParsedNetlist:
    lines = _logical_lines(text)
    parsed = ParsedNetlist(text=text)

    # First pass: params and model polarity.
    for line in lines:
        low = line.lower()
        if low.startswith(".param"):
            body = line[len(".param"):].strip()
            for key, val in re.findall(r"([A-Za-z_]\w*)\s*=\s*([^\s]+)", body):
                try:
                    parsed.params[key.lower()] = spice_number(val, parsed.params)
                except Exception:
                    pass
        elif low.startswith(".model"):
            toks = line.split()
            if len(toks) >= 3:
                mname = toks[1].lower()
                mtype = toks[2].lower()
                if "pmos" in mtype or "pchan" in mtype:
                    parsed.model_types[mname] = "pmos"
                elif "nmos" in mtype or "nchan" in mtype:
                    parsed.model_types[mname] = "nmos"
                else:
                    parsed.model_types[mname] = mtype

    for line in lines:
        if line.startswith("."):
            continue
        toks = line.split()
        if not toks:
            continue
        name = toks[0]
        lead = name[0].upper()
        try:
            if lead == "R" and len(toks) >= 4:
                parsed.resistors.append(Resistor(name, toks[1].lower(), toks[2].lower(), spice_number(toks[3], parsed.params)))
            elif lead == "C" and len(toks) >= 4:
                parsed.capacitors.append(Capacitor(name, toks[1].lower(), toks[2].lower(), spice_number(toks[3], parsed.params)))
            elif lead == "V" and len(toks) >= 4:
                ac = 0j
                lowt = [t.lower() for t in toks]
                if "ac" in lowt:
                    i = lowt.index("ac")
                    mag = spice_number(toks[i + 1], parsed.params) if i + 1 < len(toks) else 1.0
                    phase_deg = spice_number(toks[i + 2], parsed.params) if i + 2 < len(toks) else 0.0
                    ac = mag * cmath.exp(1j * math.radians(phase_deg))
                parsed.vsources.append(VSource(name, toks[1].lower(), toks[2].lower(), ac))
            elif lead == "M" and len(toks) >= 6:
                kv = {}
                for tok in toks[6:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        kv[k.lower()] = v
                w = spice_number(kv["w"], parsed.params) if "w" in kv else None
                l = spice_number(kv["l"], parsed.params) if "l" in kv else None
                parsed.mosfets.append(Mosfet(name, toks[1].lower(), toks[2].lower(), toks[3].lower(), toks[4].lower(), toks[5].lower(), w, l))
            elif lead not in {"I", "D", "Q", "X", "L", "B", "E", "F", "G", "H"}:
                parsed.unsupported.append(line)
        except Exception:
            parsed.unsupported.append(line)
    return parsed


def _strip_control_and_end(text: str) -> str:
    lines = text.splitlines()
    out = []
    in_control = False
    for line in lines:
        s = line.strip().lower()
        if s == ".control":
            in_control = True
            continue
        if in_control:
            if s == ".endc":
                in_control = False
            continue
        if s == ".end":
            continue
        # Remove analyses so probe deck only runs what we request.
        if s.startswith((".ac ", ".tran ", ".dc ", ".noise ", ".tf ")):
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


OP_KEYS_ESSENTIAL = ["id", "gm", "gds", "gmbs", "vgs", "vds", "vdsat"]
OP_KEYS_OPTIONAL = ["vth", "von", "cgs", "cgd", "cgb", "cbd", "cbs"]


def make_op_probe_deck(text: str, mos_names: Iterable[str]) -> str:
    """Build an OP probe deck.

    Probe each device parameter with a separate ``print`` command.  Some
    ngspice device models do not expose every optional capacitance keyword;
    keeping the probes separate prevents one unavailable optional vector from
    hiding essential values such as gm/gds.
    """
    base = _strip_control_and_end(text)
    print_lines = []
    for name in mos_names:
        for key in [*OP_KEYS_ESSENTIAL, *OP_KEYS_OPTIONAL]:
            print_lines.append(f"print @{name}[{key}]")
    return base + ".control\nop\n" + "\n".join(print_lines) + "\nquit\n.endc\n.end\n"


def _extract_param(stdout: str, device: str, key: str) -> Optional[float]:
    # ngspice may lowercase names; accept several spacing variants.
    pat = re.compile(
        rf"@?{re.escape(device)}\[{re.escape(key)}\]\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    vals = pat.findall(stdout)
    return float(vals[-1]) if vals else None


def _essential_op_missing(parsed: ParsedNetlist, data: Dict[str, Dict[str, Optional[float]]]) -> List[str]:
    missing: List[str] = []
    for m in parsed.mosfets:
        d = data.get(m.name.lower(), {})
        for key in ("gm", "gds"):
            val = d.get(key)
            if val is None or not math.isfinite(float(val)):
                missing.append(f"{m.name}[{key}]")
    return missing


def run_operating_point(text: str, ngspice_path: str) -> Tuple[ParsedNetlist, Dict[str, Dict[str, Optional[float]]], str]:
    parsed = parse_netlist(text)
    if not parsed.mosfets:
        raise ValueError("No MOSFET instances were detected in the netlist.")
    ngspice = resolve_ngspice(ngspice_path)
    deck = make_op_probe_deck(text, [m.name for m in parsed.mosfets])
    with tempfile.TemporaryDirectory(prefix="autocircuit_surrogate_op_") as td:
        path = Path(td) / "probe.cir"
        path.write_text(deck, encoding="utf-8")
        proc = subprocess.run([ngspice, "-b", str(path)], cwd=td, capture_output=True, text=True, timeout=90)
        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(stdout[-7000:])
    data: Dict[str, Dict[str, Optional[float]]] = {}
    for m in parsed.mosfets:
        data[m.name.lower()] = {k: _extract_param(stdout, m.name, k) for k in [*OP_KEYS_ESSENTIAL, *OP_KEYS_OPTIONAL]}

    missing = _essential_op_missing(parsed, data)
    if missing:
        tail = stdout[-9000:]
        raise RuntimeError(
            "ngspice operating-point probe did not return essential small-signal parameters: "
            + ", ".join(missing)
            + ". The automatic surrogate will not silently replace missing gm/gds with zero.\n\n"
            + tail
        )
    return parsed, data, stdout


def device_physics_rows(parsed: ParsedNetlist, op: Dict[str, Dict[str, Optional[float]]]) -> List[dict]:
    rows = []
    for m in parsed.mosfets:
        d = op.get(m.name.lower(), {})
        gm = d.get("gm")
        gds = d.get("gds")
        vds = d.get("vds")
        vdsat = d.get("vdsat")
        ro = (1.0 / gds) if gds not in (None, 0) else None
        intrinsic_gain = (abs(gm / gds)) if gm is not None and gds not in (None, 0) else None
        margin = (abs(vds) - abs(vdsat)) if vds is not None and vdsat is not None else None
        rows.append({
            "Device": m.name,
            "Model": m.model,
            "Type": parsed.model_types.get(m.model.lower(), "unknown"),
            "W_um": None if m.w is None else m.w * 1e6,
            "L_um": None if m.l is None else m.l * 1e6,
            "Id_uA": None if d.get("id") is None else d["id"] * 1e6,
            "gm_mS": None if gm is None else gm * 1e3,
            "gds_uS": None if gds is None else gds * 1e6,
            "ro_kOhm": None if ro is None else ro / 1e3,
            "gm_over_gds": intrinsic_gain,
            "VGS_V": d.get("vgs"),
            "VDS_V": vds,
            "VDSAT_V": vdsat,
            "Sat_margin_V": margin,
            "Cgs_fF": None if d.get("cgs") is None else d["cgs"] * 1e15,
            "Cgd_fF": None if d.get("cgd") is None else d["cgd"] * 1e15,
            "Cgb_fF": None if d.get("cgb") is None else d["cgb"] * 1e15,
        })
    return rows


def _stamp_admittance(A: np.ndarray, node_index: Dict[str, int], a: str, b: str, y: complex):
    ia = node_index.get(a)
    ib = node_index.get(b)
    if ia is not None:
        A[ia, ia] += y
    if ib is not None:
        A[ib, ib] += y
    if ia is not None and ib is not None:
        A[ia, ib] -= y
        A[ib, ia] -= y


def _stamp_vccs(A: np.ndarray, node_index: Dict[str, int], p: str, n: str, cp: str, cn: str, g: complex):
    # Current from p -> n equals g * (V(cp) - V(cn)).
    ip, inn = node_index.get(p), node_index.get(n)
    icp, icn = node_index.get(cp), node_index.get(cn)
    if ip is not None and icp is not None: A[ip, icp] += g
    if ip is not None and icn is not None: A[ip, icn] -= g
    if inn is not None and icp is not None: A[inn, icp] -= g
    if inn is not None and icn is not None: A[inn, icn] += g


@dataclass
class LocalSmallSignalSurrogate:
    parsed: ParsedNetlist
    op: Dict[str, Dict[str, Optional[float]]]
    warnings: List[str] = field(default_factory=list)

    def solve(self, freq_hz: float) -> Dict[str, complex]:
        nodes = self.parsed.nodes
        nidx = {n: i for i, n in enumerate(nodes)}
        nv = len(self.parsed.vsources)
        N = len(nodes) + nv
        A = np.zeros((N, N), dtype=complex)
        z = np.zeros(N, dtype=complex)
        s = 1j * 2 * math.pi * max(freq_hz, 0.0)

        for r in self.parsed.resistors:
            if r.value == 0:
                continue
            _stamp_admittance(A, nidx, r.n1, r.n2, 1.0 / r.value)
        for c in self.parsed.capacitors:
            _stamp_admittance(A, nidx, c.n1, c.n2, s * c.value)

        for m in self.parsed.mosfets:
            d = self.op.get(m.name.lower(), {})
            # gm/gds were validated when the OP data were imported.  Never
            # silently turn a missing essential parameter into zero.
            gm = float(d["gm"])
            gds = float(d["gds"])
            gmb = float(d.get("gmbs") or 0.0)
            _stamp_admittance(A, nidx, m.d, m.s, gds)
            _stamp_vccs(A, nidx, m.d, m.s, m.g, m.s, gm)
            _stamp_vccs(A, nidx, m.d, m.s, m.b, m.s, gmb)
            # Local pairwise capacitance approximation around the OP.
            for key, a, b in [
                ("cgs", m.g, m.s), ("cgd", m.g, m.d), ("cgb", m.g, m.b),
                ("cbd", m.b, m.d), ("cbs", m.b, m.s),
            ]:
                cv = d.get(key)
                if cv is not None and math.isfinite(float(cv)) and float(cv) > 0:
                    _stamp_admittance(A, nidx, a, b, s * float(cv))

        base = len(nodes)
        for k, v in enumerate(self.parsed.vsources):
            row = base + k
            ip = nidx.get(v.nplus)
            im = nidx.get(v.nminus)
            if ip is not None:
                A[ip, row] += 1
                A[row, ip] += 1
            if im is not None:
                A[im, row] -= 1
                A[row, im] -= 1
            z[row] = v.ac

        # First try the exact small-signal matrix.  If it is structurally or
        # numerically rank deficient, retry with a tiny conductance from every
        # electrical node to AC ground.  This is analogous to SPICE gmin and
        # does not materially change normal amplifier results (1 Tohm shunt).
        try:
            return {node: val for node, val in zip(nodes, np.linalg.solve(A, z)[:len(nodes)])}
        except np.linalg.LinAlgError:
            A2 = A.copy()
            gmin = 1e-12
            for i in range(len(nodes)):
                A2[i, i] += gmin
            try:
                x = np.linalg.solve(A2, z)
                if "Applied 1e-12 S gmin stabilization to a rank-deficient local MNA matrix." not in self.warnings:
                    self.warnings.append("Applied 1e-12 S gmin stabilization to a rank-deficient local MNA matrix.")
                return {node: x[i] for node, i in nidx.items()}
            except np.linalg.LinAlgError as exc:
                rank = int(np.linalg.matrix_rank(A2))
                diag = []
                for node, i in nidx.items():
                    row_norm = float(np.linalg.norm(A2[i, :]))
                    col_norm = float(np.linalg.norm(A2[:, i]))
                    if row_norm < 1e-18 or col_norm < 1e-18:
                        diag.append(node)
                detail = f" matrix rank={rank}/{N}"
                if diag:
                    detail += "; weak/floating nodes=" + ", ".join(diag)
                raise RuntimeError(
                    "Automatic small-signal matrix remains singular after gmin stabilization;"
                    + detail
                    + ". Check the raw OP probe and unsupported element families."
                ) from exc

    def frequency_response(self, output_p: str, output_n: str = "0", input_amplitude: float = 1.0,
                           f_start: float = 10.0, f_stop: float = 1e9, points_per_decade: int = 60) -> dict:
        output_p, output_n = output_p.lower(), output_n.lower()
        decades = max(math.log10(f_stop) - math.log10(f_start), 0.1)
        npts = max(20, int(decades * points_per_decade) + 1)
        freqs = np.logspace(math.log10(f_start), math.log10(f_stop), npts)
        gains = []
        for f in freqs:
            sol = self.solve(float(f))
            vp = sol.get(output_p, 0j) if output_p not in ("0", "gnd") else 0j
            vn = sol.get(output_n, 0j) if output_n not in ("0", "gnd") else 0j
            gain = abs(vp - vn) / max(abs(input_amplitude), 1e-30)
            gains.append(20 * math.log10(max(gain, 1e-30)))
        gains = np.array(gains)
        low = float(np.median(gains[: min(5, len(gains))]))
        target = low - 3.0
        idx = np.where(gains <= target)[0]
        if len(idx) == 0:
            bw = float(freqs[-1])
        else:
            i = int(idx[0])
            if i == 0:
                bw = float(freqs[0])
            else:
                f1, f2 = float(freqs[i - 1]), float(freqs[i])
                g1, g2 = float(gains[i - 1]), float(gains[i])
                if abs(g2 - g1) < 1e-15:
                    bw = f2
                else:
                    log_bw = math.log10(f1) + (target - g1) * (math.log10(f2) - math.log10(f1)) / (g2 - g1)
                    bw = 10 ** log_bw
        return {"freq_hz": freqs, "gain_db": gains, "low_gain_db": low, "bandwidth_hz": bw}



def run_spice_ac_reference(text: str, ngspice_path: str, output_p: str, output_n: str = "0",
                           input_amplitude: float = 1.0, f_start: float = 10.0, f_stop: float = 1e9,
                           points_per_decade: int = 60) -> dict:
    """Run a direct ngspice AC sweep for validation of the automatic surrogate."""
    ngspice = resolve_ngspice(ngspice_path)
    base = _strip_control_and_end(text)
    output_p, output_n = output_p.lower(), output_n.lower()
    expr = f"db(v({output_p},{output_n}))" if output_n not in ("0", "gnd") else f"db(v({output_p}))"
    deck = (
        base
        + ".control\n"
        + f"ac dec {int(points_per_decade)} {float(f_start)} {float(f_stop)}\n"
        + "set wr_singlescale\n"
        + f"wrdata ac_reference.txt {expr}\n"
        + "quit\n.endc\n.end\n"
    )
    with tempfile.TemporaryDirectory(prefix="autocircuit_surrogate_ac_") as td:
        td_path = Path(td)
        path = td_path / "reference.cir"
        data_path = td_path / "ac_reference.txt"
        path.write_text(deck, encoding="utf-8")
        proc = subprocess.run([ngspice, "-b", str(path)], cwd=td_path, capture_output=True, text=True, timeout=90)
        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or not data_path.exists():
            raise RuntimeError(stdout[-7000:])
        arr = np.loadtxt(data_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    freq = arr[:, 0].astype(float)
    gain_db = arr[:, -1].astype(float) - 20.0 * math.log10(max(abs(input_amplitude), 1e-30))
    low = float(np.median(gain_db[: min(5, len(gain_db))]))
    target = low - 3.0
    ids = np.where(gain_db <= target)[0]
    if len(ids) == 0:
        bw = float(freq[-1])
    else:
        i = int(ids[0])
        if i == 0:
            bw = float(freq[0])
        else:
            f1, f2 = float(freq[i - 1]), float(freq[i])
            g1, g2 = float(gain_db[i - 1]), float(gain_db[i])
            if abs(g2 - g1) < 1e-15:
                bw = f2
            else:
                log_bw = math.log10(f1) + (target - g1) * (math.log10(f2)-math.log10(f1)) / (g2-g1)
                bw = 10 ** log_bw
    return {"freq_hz": freq, "gain_db": gain_db, "low_gain_db": low, "bandwidth_hz": bw, "stdout": stdout}


def assess_surrogate_confidence(predicted: dict, actual: dict) -> dict:
    gain_error_db = abs(float(predicted["low_gain_db"]) - float(actual["low_gain_db"]))
    actual_bw = max(abs(float(actual["bandwidth_hz"])), 1e-30)
    bw_error_pct = abs(float(predicted["bandwidth_hz"]) - float(actual["bandwidth_hz"])) / actual_bw * 100.0
    if gain_error_db <= 0.75 and bw_error_pct <= 10.0:
        level = "HIGH"
    elif gain_error_db <= 2.0 and bw_error_pct <= 25.0:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"level": level, "gain_error_db": gain_error_db, "bw_error_pct": bw_error_pct}

def build_local_surrogate(text: str, ngspice_path: str) -> Tuple[LocalSmallSignalSurrogate, List[dict], str]:
    parsed, op, stdout = run_operating_point(text, ngspice_path)
    warnings = []
    unsupported_prefixes = sorted({line.split()[0][0].upper() for line in parsed.unsupported if line.split()})
    if unsupported_prefixes:
        warnings.append("Unsupported/ignored element families in Phase 1: " + ", ".join(unsupported_prefixes))
    missing = []
    for m in parsed.mosfets:
        d = op.get(m.name.lower(), {})
        for k in ("gm", "gds"):
            if d.get(k) is None:
                missing.append(f"{m.name}[{k}]")
    if missing:
        warnings.append("Missing essential OP parameters: " + ", ".join(missing))
    surrogate = LocalSmallSignalSurrogate(parsed=parsed, op=op, warnings=warnings)
    return surrogate, device_physics_rows(parsed, op), stdout


DEFAULT_DIFF_PAIR_NETLIST = r"""* SynapseFlow automatic surrogate demo netlist
.param RD=5600 WIN=20u WTAIL=50u VBIAS=0.70 VCM=1.00
VDD vdd 0 1.8
VINP inp 0 DC {VCM} AC 0.5 0
VINN inn 0 DC {VCM} AC 0.5 180
VBIAS vbias 0 DC {VBIAS}
RDP vdd outp {RD}
RDN vdd outn {RD}
CLP outp 0 5p
CLN outn 0 5p
M1 outp inp tail 0 NMOS W={WIN} L=1u
M2 outn inn tail 0 NMOS W={WIN} L=1u
MTAIL tail vbias 0 0 NMOS W={WTAIL} L=1u
.model NMOS NMOS LEVEL=1 VTO=0.5 KP=200u LAMBDA=0.02
.end
"""


def parser_self_test() -> dict:
    p = parse_netlist(DEFAULT_DIFF_PAIR_NETLIST)
    assert len(p.mosfets) == 3
    assert len(p.resistors) == 2
    assert len(p.capacitors) == 2
    assert len(p.vsources) == 4
    assert abs(p.params["rd"] - 5600.0) < 1e-9
    assert abs((p.mosfets[0].w or 0) - 20e-6) < 1e-15
    return {
        "mosfets": len(p.mosfets), "resistors": len(p.resistors), "capacitors": len(p.capacitors),
        "vsources": len(p.vsources), "nodes": len(p.nodes), "params": dict(p.params),
    }
