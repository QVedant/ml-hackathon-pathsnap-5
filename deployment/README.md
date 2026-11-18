# Deployment (two laptops)

| Laptop | Runs | Needs |
|---|---|---|
| 1 — Model + API | `api/main.py` (FastAPI): preprocessing → candidates → features → LightGBM → HMM/Viterbi | `uv`, repo, `data/` link, evaluator trips folder |
| 2 — UI | `ui/index.html` (static page, Leaflet + fonts bundled) | Python 3 (only for `http.server`) |

Both laptops must be on the same network. College Wi-Fi often blocks device-to-device traffic, so use a phone hotspot if `/health` does not load.

## Laptop 1
```bash
uv sync
ln -s "/path/to/Student folder" data                 # Windows: mklink /D data "C:\path\to\Student folder"
deployment/run_api.sh /path/to/evaluator_test        # Windows: deployment\run_api.bat C:\path\to\evaluator_test
sudo ufw allow 8000/tcp                              # only if the firewall is on (Windows: allow Python on the prompt)
```
The script prints the LAN IP. Keep its terminal visible: each request is logged with the trip id, classifier/HMM Top-1, continuity and time.
Without `uv`: `pip install -r deployment/requirements.txt` then `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

## Laptop 2
```bash
deployment/run_ui.sh                                 # Windows: deployment\run_ui.bat
```
Open `http://localhost:5500`, type laptop 1's IP in the API box, press Connect.

## Check
- `http://<laptop1-ip>:8000/health` in laptop 2's browser must return `{"status":"ok",...}`.
- Backup UI served by the API itself: `http://<laptop1-ip>:8000/ui/`.

## API
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, model file, eval folder |
| GET | `/trips` | – | trip ids found in `PATHSNAP_EVAL_DIR` |
| POST | `/predict` | `{"trip_id": "00000015"}` | GPS points, matched route (HMM) and classifier-only geometry, arc ids, metrics |

The reference `.route` is read only after prediction, on the server, to compute accuracy; it is never sent to the UI or used by the model.
