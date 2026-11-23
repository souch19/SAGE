"""AGC (OmniColPress) à budget variable, comparé sur le backbone ColBERT.

AGC compresse pendant l'encodage : chaque page est ré-encodée avec k vecteurs
(num_select_tokens = k, sans réentraînement). k suit la règle des méthodes post-hoc avec
N = nombre de patchs ColBERT de la page ; pages, requêtes et pertinence sont celles des
embeddings ColBERT, donc la rétention a le même dénominateur.
Sortie : colbert/results/{banc}/{split}/{budget}/agc.pkl (environnement omni-col-press).

    python 04_evaluate/agc.py --bench vidore_v2
"""
import argparse
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.data import Split
from sage.encoders import build_encoder
from sage.retrieval import evaluate, native_ndcg
from sage.utils import DEVICE, load_pickle, save_pickle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default="vidore_v2", choices=list(C.BENCHMARKS))
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", default=C.BUDGETS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    P, enc = C.Paths("colbert"), None
    budgets = [C.parse_budget(b) for b in args.budgets]
    for split, repo in C.splits(args.bench, args.splits).items():
        todo = [(t, k_of) for t, k_of in budgets
                if args.force or not P.result(args.bench, split, t, "agc").exists()]
        if not todo:
            print(f"[agc/{split}] déjà évalué")
            continue
        if not P.emb(args.bench, split).exists():
            print(f"[agc/{split}] embeddings ColBERT absents ({P.emb(args.bench, split)})")
            continue
        cb = load_pickle(P.emb(args.bench, split))
        rel, sizes = cb["rel"], {d: e.shape[0] for d, e in cb["doc_emb"].items()}
        nat = native_ndcg(P, args.bench, split, cb, DEVICE)
        del cb
        enc = enc or build_encoder("agc")
        data = Split(args.bench, repo)
        print(f"\n=== agc / {args.bench} / {split} : {len(sizes)} pages, natif ColBERT {nat:.1f} ===")
        qe = {q: enc.encode_query(data.queries[q]) for q in tqdm(rel, desc="requêtes")}
        for tag, k_of in todo:
            t0 = time.time()
            de = {d: enc.encode_doc(data.image(d), k_of(n)) for d, n in tqdm(sizes.items(), desc=tag)}
            res = evaluate(qe, de, rel, DEVICE)
            n_kept = sum(e.shape[0] for e in de.values())
            save_pickle(P.result(args.bench, split, tag, "agc"), {
                "split": split, "method": "agc", "budget": tag, **res,
                "avg_vectors": n_kept / len(de), "compression": n_kept / sum(sizes.values()),
                "time": time.time() - t0})
            print(f"  agc {tag:<6} nDCG@5={res['ndcg5']:5.1f}  rétention={100 * res['ndcg5'] / nat:5.1f} %"
                  f"  vecteurs/page={n_kept / len(de):6.1f}")


if __name__ == "__main__":
    main()
