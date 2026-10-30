# SynapseFlow cloud deployment

This package is prepared for Streamlit Community Cloud and generic Docker hosting.

## What changed for cloud use

- `ngspice` is installed on the Linux host with `packages.txt`.
- The app no longer requires the developer's Windows ngspice path.
- `NGSPICE_PATH` is optional; when blank, SynapseFlow auto-detects `ngspice` from `PATH`.
- Ollama is optional. Set `OLLAMA_URL` and `OLLAMA_MODEL` only if a remotely reachable Ollama server is available.
- If `OLLAMA_URL` is blank, natural-language requests that explicitly contain gain/bandwidth/power values and units use the deterministic value/unit parser immediately.

Example supported public-demo request:

`차동 증폭기를 설계해줘. 전압 이득은 최소 18 dB, 대역폭은 4.5 MHz 이상, 소비전력은 0.4 mW 이하로 해줘.`

## Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this folder to the repository root.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create an app and select `app.py` as the entrypoint.
4. Recommended Python version: 3.12.
5. Deploy. `requirements.txt` installs Python dependencies and `packages.txt` installs Linux `ngspice`.

### Optional Ollama endpoint

If you have a public/private remotely reachable Ollama server, add these to Streamlit app Secrets or environment configuration:

```toml
OLLAMA_URL = "https://your-ollama-host.example"
OLLAMA_MODEL = "qwen3:4b"
```

The app does not need these settings to run the public demo because deterministic parsing is built in.

## Docker deployment

Build and run locally or on a Docker-capable host:

```bash
docker build -t synapseflow .
docker run --rm -p 8501:8501 synapseflow
```

Then open `http://localhost:8501`.
