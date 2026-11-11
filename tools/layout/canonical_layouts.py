"""One public-only layout selector for every restored HFX experiment.

The candidate family and numerical model are frozen from full_restore.  Unlike
the previous entry point this function never returns a profile by matching an
experiment's parameters to a confirmed case.  Historical public shapes are
ordinary candidates, scored under the requested public parameters.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
from . import public_layout as _model
probabilities, risks, sigma = _model.probabilities, _model.risks, _model.sigma
GEOM = pd.read_csv(SOURCE / "data/measured_geometry.csv").set_index("w")
CORE = json.loads((SOURCE / "data/frozen_core_layouts.json").read_text())
TIE_TOLERANCE = 1e-6
RULE_ID = "hfx-public-candidates-v20261102-no-case-override"
CACHE = {}


def execution_signature(p):
    """Fields that determine encoding, decoding and Gaussian noise."""
    return dict(w=int(p["w"]), q16=int(p["q16"]), rows=list(map(int, p["rows"])),
                gmod_width=int(p["gmod_width"]), gmod_rows=int(p["gmod_rows"]),
                capacity=int(p["capacity"]), K=int(p["K"]), epsilon=float(p["epsilon"]),
                mode=p.get("mode", p.get("memory_mode", "full")),
                memory_kib=int(p["memory_kib"]), sigma=float(p["sigma"]))


def profile(mem, U, K, eps=1., mode="payload"):
    mem, U, K, eps = int(mem), int(U), int(K), float(eps)
    if mode not in {"payload", "full"}:
        raise ValueError("mode must be 'payload' or 'full'")
    if mem < 1 or U < 2 or K < 1 or eps <= 0:
        raise ValueError("invalid public parameters")
    key = (mem, U, K, eps, mode)
    if key in CACHE:
        return CACHE[key]
    cap = mem * 1024
    sig = sigma(K, eps)
    grid = np.unique(np.rint(np.geomspace(1, U, 257)))
    dense = np.unique(np.rint(np.geomspace(1, U, 4097)))

    def bits_for(w, method):
        return ((cap - int(GEOM.loc[w, method + "_base_bytes"])) // 8 * 64
                if mode == "full" else cap * 8)

    grows, gcurves, slopes = [], [], []
    for w in range(1, 33):
        m = bits_for(w, "gmod") // w
        if m < 4:
            raise ValueError("budget cannot hold the frozen width scan")
        grows.append(m)
        p = probabilities(w)
        gcurves.append(risks(grid, .5 * p, np.full(w, m), sig, False))
        slopes.append(np.sum(.5 * p * np.exp(-U * p / m)))
    gcurves = np.asarray(gcurves)
    wg = int(np.argmin(logsumexp(2 * (np.log(gcurves) - np.log(gcurves.min(axis=0))), axis=1))) + 1
    wc = int(np.argmax(slopes)) + 1

    def reference(xs):
        return np.min([risks(xs, .5 * probabilities(w), np.full(w, grows[w-1]), sig, False)
                       for w in sorted({wg, wc})], axis=0)

    ref = reference(grid)
    candidates = []

    def add(w, q, row, origin):
        row = np.asarray(row, dtype=np.int64)
        if np.any(row < 4):
            raise ValueError("invalid public candidate")
        r = risks(grid, probabilities(w, q), row, sig)
        candidates.append(dict(w=w, q16=q, rows=row.tolist(), origin=origin,
                               max_ratio=float(np.max(r / ref))))

    for w in range(1, 33):
        bits = bits_for(w, "lean")
        add(w, 16, np.full(w, (bits // 64 // w) * 64), "dyadic public allocation")
        if w == wg + 1 and bits - wg * grows[wg-1] >= 4:
            add(w, 8, np.r_[bits - wg * grows[wg-1], np.full(wg, grows[wg-1])],
                "retained GMod coordinates")
    for c in CORE:
        w = c["w"]
        bits = bits_for(w, "lean")
        free = bits - 64 * w
        shape = np.maximum(np.asarray(c["rows"]) - 64, 1.)
        real_rows = 64 + free * shape / shape.sum()
        words = np.floor(real_rows / 64).astype(np.int64)
        left = bits // 64 - words.sum()
        order = np.argsort(-(real_rows / 64 - words), kind="stable")
        words[order[:left]] += 1
        add(w, c["q16"], words * 64, "scaled frozen public shape")

    # Keep the original two-stage public scorer and its deterministic ordering.
    leading = sorted(candidates, key=lambda c: c["max_ratio"])[:8]
    leading += [c for c in candidates if c["origin"] == "retained GMod coordinates"]
    rd = reference(dense)
    for c in leading:
        c["dense_ratio"] = float(np.max(risks(dense, probabilities(c["w"], c["q16"]),
                                               np.asarray(c["rows"]), sig) / rd))
    low = min(c["dense_ratio"] for c in leading)
    tied = [c for c in leading if c["dense_ratio"] <= low * (1 + TIE_TOLERANCE)]
    best = min(tied, key=lambda c: (c["origin"] != "retained GMod coordinates",
                                   c["w"], c["q16"])).copy()
    best.update(gmod_width=wg, gmod_rows=grows[wg-1],
                gmod_state_bytes=int(GEOM.loc[wg, "gmod_base_bytes"]) + (grows[wg-1]*wg+63)//64*8,
                memory_kib=mem, capacity=U, K=K, epsilon=eps, mode=mode, sigma=sig,
                selector_rule=RULE_ID, tie_tolerance=TIE_TOLERANCE)
    best["state_bytes"] = int(GEOM.loc[best["w"], "lean_base_bytes"]) + (sum(best["rows"])+63)//64*8
    best["profile_id"] = hashlib.sha256(json.dumps(best, sort_keys=True).encode()).hexdigest()[:16]
    if mode == "full":
        assert max(best["state_bytes"], best["gmod_state_bytes"]) <= cap
    CACHE[key] = best
    return best


def save(path=None):
    target = Path(path) if path else ROOT / "CANONICAL_LAYOUT_PROFILES.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(CACHE.values()), indent=2), encoding="utf-8")


def provenance():
    files = [Path(__file__), SOURCE / "public_layout.py",
             SOURCE / "data/measured_geometry.csv", SOURCE / "data/frozen_core_layouts.json"]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
