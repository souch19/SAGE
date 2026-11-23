"""Coût d'encodage d'une page : ColBERT (tous les patchs) contre AGC (k vecteurs produits pendant
le forward). FLOPs, latence du forward et stockage ; --carbon ajoute le CO2 de l'encodage du split.
Environnement omni-col-press.

    python 05_analysis/cost_encode.py --split econ --k 64
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.data import Split
from sage.encoders import build_encoder
from sage.utils import DEVICE, latency_ms, save_json


def measure(name, enc, data, docs, k, carbon, out_dir):
    flops, lat, vec = [], [], []
    for did in docs:
        if k is not None:
            enc.model.num_select_tokens = k
        inputs = enc.doc_inputs(data.image(did))
        with FlopCounterMode(display=False) as fc:
            vec.append(enc.embed(inputs).shape[0])
        flops.append(fc.get_total_flops())
        lat.append(latency_ms(lambda: enc.embed(inputs)))
    r = {"tflops": np.mean(flops) / 1e12, "ms": float(np.mean(lat)), "vectors": float(np.mean(vec))}
    if carbon:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(project_name=f"encode_{name}", measure_power_secs=1,
                                   tracking_mode="process", log_level="error", output_dir=out_dir)
        tracker.start()
        t0 = time.time()
        for did in data.doc_ids:
            enc.encode_doc(data.image(did), k)
        r.update(encode_s=time.time() - t0, mg_co2_per_page=tracker.stop() * 1e6 / len(data.doc_ids))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--split", default="econ")
    ap.add_argument("--k", type=int, default=64, help="vecteurs par page pour AGC")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--carbon", action="store_true")
    ap.add_argument("--out", default=str(C.REPORTS))
    args = ap.parse_args()

    data = Split(args.bench, C.BENCHMARKS[args.bench][args.split])
    docs = data.doc_ids[:args.n_docs]
    res = {}
    for name, k in (("colbert", None), ("agc", args.k)):
        enc = build_encoder(name)
        res[name] = measure(name, enc, data, docs, k, args.carbon, args.out)
        del enc
        gc.collect()
        torch.cuda.empty_cache()

    print(f"{'encodeur':<10}{'TFLOPs':>9}{'ms/page':>10}{'vecteurs':>10}{'Ko/page':>10}{'mgCO2/page':>12}")
    for name, r in res.items():
        co2 = f"{r['mg_co2_per_page']:.3f}" if "mg_co2_per_page" in r else "-"
        print(f"{name:<10}{r['tflops']:>9.3f}{r['ms']:>10.1f}{r['vectors']:>10.1f}"
              f"{r['vectors'] * 2048 * 2 / 1e3:>10.1f}{co2:>12}")
    extra = res["agc"]["ms"] - res["colbert"]["ms"]
    print(f"surcoût AGC : {extra:+.1f} ms/page ({res['agc']['ms'] / res['colbert']['ms']:.2f}x)")
    save_json(Path(args.out) / f"cost_encode_{args.split}_k{args.k}.json",
              {"split": args.split, "agc_k": args.k, "device": DEVICE, **res})


if __name__ == "__main__":
    main()
