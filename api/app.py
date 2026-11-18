"""PathSnap inference API (runs on Laptop 1).

    uvicorn api.app:app --host 0.0.0.0 --port 8000

Env vars:
  PATHSNAP_DATA       folder containing the partition folders (default: parent of the repo)
  PATHSNAP_PARTITION  partition served to the UI (default: evaluator_test, falls back to student_test)

The UI only ever sends a trip id.  The model sees the GPS track and the road
network; the reference route is read afterwards, server-side, purely to score
the prediction, and is never returned to the client.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsnap.data import DEFAULT_DATA_ROOT, list_trips, load_trip, partition_dir  # noqa: E402
from pathsnap.pipeline import PathSnapModel, score_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

DATA = os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_ROOT)
PARTITION = os.environ.get("PATHSNAP_PARTITION", "evaluator_test")
if not os.path.isdir(partition_dir(PARTITION, DATA)):
    log.warning("partition %s not found under %s, falling back to student_test", PARTITION, DATA)
    PARTITION = "student_test"

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
MODEL = PathSnapModel()
log.info("model loaded: %s | HMM params: %s", type(MODEL.clf).__name__, MODEL.hmm.to_dict())
log.info("serving partition '%s' from %s", PARTITION, DATA)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)  # browsers request this automatically; avoid 404 noise in the log


@app.get("/health")
def health():
    return {"status": "ok", "partition": PARTITION, "model": type(MODEL.clf).__name__}


@app.get("/trips")
def trips():
    ids = list_trips(PARTITION, DATA)
    log.info("GET /trips -> %d trips", len(ids))
    return {"partition": PARTITION, "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in list_trips(PARTITION, DATA):
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("POST /predict trip=%s : running pipeline", tid)
    t0 = time.perf_counter()
    trip = load_trip(tid, PARTITION, DATA, with_route=False)   # model input: no reference
    out = MODEL.predict(trip)
    total = time.perf_counter() - t0

    # ---- server-side scoring (reference route loaded only now)
    scored = load_trip(tid, PARTITION, DATA, with_route=True)
    metrics = score_trip(scored, out["net"], out) if scored.route is not None else {}

    nl = trip.node_lonlat
    route_lines = [[[nl[u, 1], nl[u, 0]], [nl[v, 1], nl[v, 0]]] for u, v in trip.arcs[out["route"]]]
    gps = trip.lonlat[:, ::-1].round(7).tolist()
    # matched point for each fix: projection onto its predicted arc
    net, arcs = out["net"], out["pred_hmm"]
    pxy = trip.xy[out["obs"]]
    d = net.p1[arcs] - net.p0[arcs]
    tt = np.clip(((pxy - net.p0[arcs]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-9), 0, 1)
    q = net.p0[arcs] + tt[:, None] * d
    qlon, qlat = trip.proj.inv(q[:, 0], q[:, 1])
    matched = np.column_stack([qlat, qlon]).round(7).tolist()

    log.info("trip=%s fixes=%d candidates=%d breaks=%d time=%.2fs | top1(HMM)=%s top1(clf)=%s route_f1=%s",
             tid, trip.n_obs, out["n_candidates"], out["n_breaks"], total,
             _f(metrics.get("top1_acc_hmm")), _f(metrics.get("top1_acc_classifier")), _f(metrics.get("route_f1")))
    return {
        "trip_id": tid,
        "n_fixes": int(trip.n_obs),
        "gps": gps,
        "matched_points": matched,
        "predicted_route": route_lines,
        "predicted_arc_ids": out["route"].tolist(),
        "n_route_arcs": int(len(out["route"])),
        "hmm_breaks": int(out["n_breaks"]),
        "timings_s": {k: round(v, 3) for k, v in out["timings"].items()} | {"total": round(total, 3)},
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
    }


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"
