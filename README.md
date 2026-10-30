# SynapseFlow · Integrated Automatic Surrogate Engine

## What changed from V4

V5 keeps the existing Common-Source, Differential Pair Nominal, VCM Robust and illustrative PVT-like flows, and adds a new Differential Pair Nominal engine:

**Specification → netlist + ngspice operating point → automatic local small-signal surrogate → trust-region candidate ranking → ngspice verification → local model update**

The automatic engine does not call the old differential-pair-specific analytical gain equation for its local surrogate. It parses R/C/V/MOS connectivity, imports device operating-point quantities from ngspice, assembles a local MNA small-signal model, and updates the local device-parameter Jacobian while searching.

## Surrogate update methods

- **Jacobian reuse · Broyden**: default. Initial full local calibration, then rank-one secant updates from verified operating points without extra perturbation OP calls.
- **Active-variable recalibration**: recalibrates selected Jacobian columns after re-anchoring.
- **Full finite-difference**: high-cost reference; recalibrates all four design variables.

Final PASS is always based on ngspice verification.

## Current scope

The current automatic optimizer is validated for the real-tail NMOS differential-pair parameterization using `.param RD`, `WIN`, `WTAIL`, and `VBIAS`, with differential output nodes `outp/outn`. The underlying automatic small-signal builder handles the current Phase-1 R/C/V/MOS network scope, but arbitrary-topology optimization-variable discovery is not yet implemented.

The VCM/PVT robust flows remain the reference research engine for now.

## Run

```bat
cd "C:\Users\'장희'\Desktop\SynapseFlow"
py -m streamlit run app.py
```

or run `run_app.bat` from the project folder.
