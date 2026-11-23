"""Bancs d'évaluation (ViDoRe v2, JinaVDR) et pool d'entraînement 10k."""
import json
from collections import defaultdict
from pathlib import Path


class Split:
    """Un split d'évaluation : ids des pages (ordre du dataset), textes des requêtes et
    pertinence {qid: {page}}. Les images sont décodées à la demande."""

    def __init__(self, bench, repo):
        import datasets
        if bench == "vidore_v2":
            queries = datasets.load_dataset(repo, "queries", split="test")
            qrels = datasets.load_dataset(repo, "qrels", split="test")
            self._ds = datasets.load_dataset(repo, "corpus", split="test")
            self.doc_ids = [str(d) for d in self._ds["corpus-id"]]
            self._row = {d: i for i, d in enumerate(self.doc_ids)}
            self.queries = {str(q): t for q, t in zip(queries["query-id"], queries["query"])}
            rel = defaultdict(set)
            for q, d, s in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
                if int(s) > 0:
                    rel[str(q)].add(str(d))
            self.rel = dict(rel)
        else:
            # JinaVDR : une ligne = (requête, image) ; le corpus est l'ensemble des images uniques
            self._ds = datasets.load_dataset(repo, split="test")
            n, cols = len(self._ds), self._ds.column_names
            names = self._ds["image_filename"] if "image_filename" in cols else [None] * n
            texts = self._ds["query"] if "query" in cols else [None] * n
            self._row, self.queries, self.rel = {}, {}, {}
            for i, (fn, q) in enumerate(zip(names, texts)):
                did = str(fn) if fn else f"img_{i}"
                self._row.setdefault(did, i)
                if q is None or (isinstance(q, str) and not q.strip()):
                    continue
                self.queries[f"q_{i}"] = q
                self.rel[f"q_{i}"] = {did}
            self.doc_ids = list(self._row)

    def image(self, did):
        return self._ds[self._row[did]]["image"]


def resolve_image(data, imf, manifest_base):
    """Chemin de l'image d'une page du pool (nom à plat, puis manifeste, puis basename)."""
    for p in (data / f"{imf}.jpg",
              data / manifest_base[imf] if imf in manifest_base else None,
              data / Path(imf).name):
        if p is not None and p.exists():
            return p
    return None


def train_pool(data_dir):
    """Pool produit par 01_data/build_train10k.py.
    Renvoie (pages, {qid: requête}, {qid: [pages pertinentes]}, résolveur page -> chemin image)."""
    import pandas as pd
    data = Path(data_dir)
    manifest = json.loads((data / "s00_top10k_image_manifest.json").read_text())
    base = {k: Path(v).name for k, v in manifest.items()}
    pairs = pd.read_parquet(data / "s00_top10k_pairs_indexed.parquet")
    relevance = json.loads((data / "s00_train_relevance.json").read_text())
    qid2query = {int(r.qid): r.query for r in pairs.itertuples()}
    rel = {int(q): list(v) for q, v in relevance.items()}
    pages = list(pairs["image_filename"].unique())
    return pages, qid2query, rel, lambda imf: resolve_image(data, imf, base)
