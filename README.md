# bareboneLLM

A minimal GPT-style (decoder-only) language model you can train from scratch.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_sample_data.py
PYTHONPATH=src python -m barebonellm.train --config configs/tiny.json
PYTHONPATH=src python -m barebonellm.generate --checkpoint checkpoints/model.pt --prompt "Hello"
PYTHONPATH=src python -m barebonellm.server --checkpoint checkpoints/model.pt
```

## API

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":80,"temperature":1.0,"top_k":64}'
```
