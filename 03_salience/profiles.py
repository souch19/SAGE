"""Tire 64 requêtes du pool d'entraînement : profils de réponse zero-shot, indépendants des
requêtes de test (utilisés par S-Group, S-Xr et les baselines sans apprentissage).

    python 03_salience/profiles.py --model colqwen [--seed 1]
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.utils import load_pickle, save_pickle

N_PROFILES = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    args = ap.parse_args()

    P = C.Paths(args.model, args.seed)
    if P.profiles().exists():
        print(f"[{args.model}] {P.profiles().name} existe déjà")
        return
    q = load_pickle(P.train("queries.pkl"))["queries"]
    keep = random.Random(args.seed).sample(list(q), min(N_PROFILES, len(q)))
    save_pickle(P.profiles(), {"queries": {k: q[k] for k in keep}})
    print(f"[{args.model}] {len(keep)} profils -> {P.profiles()}")


if __name__ == "__main__":
    main()
