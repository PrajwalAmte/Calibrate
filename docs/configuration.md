# Configuration

Precedence for every setting: **command-line flag > environment variable >
config file > built-in default**.

## Config file

`--config <FILE>` or `$CALIBRATE_CONFIG`, otherwise the default location
(`~/.config/calibrate/config.toml` on Linux,
`~/Library/Application Support/calibrate/config.toml` on macOS). Unknown keys
are rejected so typos never pass silently.

```toml
[general]
offline = false              # never touch the network
http_timeout_secs = 10
specs_url = "https://raw.githubusercontent.com/PrajwalAmte/Calibrate/main/assets/fallback_specs.json"

[watch]
interval = 2.0
cost_per_hour = 23.92
target_mfu = 45.0
metrics_source = "auto"      # auto | gpm | dcgm | nvml
prometheus_listen = "0.0.0.0:9400"
otlp_endpoint = "http://otel-collector:4318"
hook_dir = "/var/run/calibrate"

[plan]
providers = ["runpod", "vastai"]
seq_len = 4096
price_cache_ttl_secs = 900

[credentials]               # prefer environment variables or a secret store
hf_token = "hf_..."
lambda_api_key = "..."
runpod_api_key = "..."
```

## Environment variables

| Variable | Purpose |
|---|---|
| `CALIBRATE_CONFIG` | config file path |
| `CALIBRATE_OFFLINE` | `1`/`true`/`yes` = offline |
| `CALIBRATE_SPECS_URL` | mirror of the GPU spec database |
| `CALIBRATE_HOOK_DIR` | Python hook socket directory (default `/tmp`) |
| `CALIBRATE_PYTHON` | interpreter for `bench` drivers (default `python3`) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | gated/private Hugging Face models (also read from `huggingface-cli login`'s token file) |
| `HF_ENDPOINT` | Hugging Face mirror |
| `LAMBDA_API_KEY`, `RUNPOD_API_KEY` | provider pricing APIs |
| `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY` | outbound proxy |
| `RUST_LOG` | diagnostic verbosity (`warn` default), with `--log-format json` for structured logs |

TLS uses the operating system trust store, so corporate CAs installed on the
host are honoured.

## Air-gapped clusters

* `--offline` (or `CALIBRATE_OFFLINE=1`): no network at all. `watch` uses the
  built-in spec table; `plan` needs `--params-b` or a local model directory
  and a `--prices-file`; cached prices are used with an age warning.
* Point `CALIBRATE_SPECS_URL` and `HF_ENDPOINT` at internal mirrors to keep
  online features without internet egress.

## Prices file

For on-prem clusters, reserved capacity or negotiated rates:

```json
[
  {"provider": "on-prem", "gpu_model": "H100 SXM", "gpu_count": 8,
   "vram_per_gpu_gib": 80, "hourly_usd": 18.40},
  {"provider": "reserved", "gpu_model": "L40S", "vram_per_gpu_gib": 48,
   "hourly_usd": 0.95, "availability": "Available", "flags": []}
]
```

`gpu_count` defaults to 1 and `hourly_usd` is the price of the whole
instance. `{"listings": [...]}` is also accepted. GPU names are matched
against the spec database (aliases and NVML spellings work).

## Caches

Under the platform cache directory (`~/.cache/calibrate` on Linux):

* `gpu-specs.json` — remote spec database, refreshed daily; a failed fetch is
  retried after 6 h.
* `prices-<provider>.json` — provider prices, `price_cache_ttl_secs`
  (default 15 min); `plan --refresh-prices` bypasses it.
