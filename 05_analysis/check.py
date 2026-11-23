"""Contrôle des résultats : chaque .pkl doit se lire et contenir un nDCG@5 dans [0, 100].
Avec --ref, compare le nDCG@5 au même fichier d'un autre arbre d'artefacts, y compris
l'ancienne disposition (agc/, kmeans/, {modèle}_xp/).

    python 05_analysis/check.py [--delete]
    python 05_analysis/check.py --ref ~/STRIDE/artifacts
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage import config as C
from sage.utils import load_pickle


def ndcg(path):
    try:
        v = load_pickle(path)["ndcg5"]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return (v, None) if isinstance(v, (int, float)) and 0 <= v <= 100 else (None, f"nDCG@5 aberrant : {v}")


def ref_candidates(ref, rel):
    """Emplacements possibles du même résultat dans l'arbre de référence."""
    model, kind, *rest = rel.parts
    out = [ref / rel]
    method = rel.stem
    if model == "colbert" and method in ("agc", "kmeans"):
        out.append(ref / method / kind / Path(*rest))
    if method == "XP_wkmeans":
        out.append(ref / f"{model}_xp" / kind / Path(*rest))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(C.ARTIFACTS))
    ap.add_argument("--ref", default=None, help="autre racine d'artefacts à comparer")
    ap.add_argument("--tol", type=float, default=0.05, help="écart toléré en points de nDCG@5")
    ap.add_argument("--delete", action="store_true", help="supprime les fichiers illisibles")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(f for f in root.glob("*/results*/**/*.pkl"))
    bad, same, diff = [], 0, []
    for f in files:
        v, err = ndcg(f)
        if err:
            bad.append((f, err))
            continue
        if args.ref:
            for g in ref_candidates(Path(args.ref).expanduser(), f.relative_to(root)):
                if g.exists():
                    w, _ = ndcg(g)
                    if w is not None and abs(v - w) <= args.tol:
                        same += 1
                    elif w is not None:
                        diff.append((f.relative_to(root), v, w))
                    break
    print(f"{len(files)} fichiers de résultats sous {root}, {len(bad)} illisibles")
    for f, err in bad:
        print(f"  [!] {f.relative_to(root)} : {err}")
        if args.delete:
            f.unlink()
    if args.ref:
        print(f"comparaison à {args.ref} : {same} identiques (écart <= {args.tol}), {len(diff)} différents")
        for rel, v, w in diff:
            print(f"  {rel}  {v:.2f} (ici) vs {w:.2f} (réf.)  écart {v - w:+.2f}")


if __name__ == "__main__":
    main()
