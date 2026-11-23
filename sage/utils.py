"""Device, graines, chronométrage et écritures atomiques (un run interrompu ne laisse pas de
fichier tronqué)."""
import json
import os
import pickle
import random
import time

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def latency_ms(fn, warmup=1, iters=3):
    """Durée moyenne d'un appel (événements CUDA sur GPU)."""
    if DEVICE != "cuda":
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1e3
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_pickle(path, obj):
    def write(tmp):
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
    _atomic(path, write)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(path, obj):
    _atomic(path, lambda t: t.write_text(json.dumps(obj, indent=2)))


def save_torch(path, obj):
    _atomic(path, lambda t: torch.save(obj, t))
