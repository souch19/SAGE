"""Registres (modèles, bancs, budgets) et arborescence des artefacts.

    $SAGE_ARTIFACTS/{modèle}/
        embeddings/{banc}/{split}.pkl                  doc_emb, q_emb, rel, ndcg5, npatch
        train/docs/shard_{i}.pkl, queries.pkl, meta.pkl, targets.pkl
        train/regressor[_s{seed}].pt, regressor[_s{seed}].meta.json, profiles64[_s{seed}].pkl
        results[_s{seed}]/{banc}/{split}/{budget}/{méthode}.pkl
        results/{banc}/{split}/native.pkl
"""
import os
from pathlib import Path

MODELS = {
    "colpali":  {"loader": "colpali", "id": "vidore/colpali-v1.3",
                 "cls": "ColPali", "proc": "ColPaliProcessor", "dim": 128},
    "colqwen":  {"loader": "colpali", "id": "vidore/colqwen2.5-v0.2",
                 "cls": "ColQwen2_5", "proc": "ColQwen2_5_Processor", "dim": 128},
    "colnomic": {"loader": "colpali", "id": "nomic-ai/colnomic-embed-multimodal-3b",
                 "cls": "ColQwen2_5", "proc": "ColQwen2_5_Processor", "dim": 128},
    "jinav4":   {"loader": "jina", "id": "jinaai/jina-embeddings-v4", "dim": 128},
    "colbert":  {"loader": "omni", "id": "hltcoe/ColBERT_qwen2.5-vl_colpali", "dim": 2048},
}
AGC_ID = "hltcoe/AGC_qwen2.5-vl_colpali"

BENCHMARKS = {
    "vidore_v2": {
        "bio":   "vidore/biomedical_lectures_v2",
        "econ":  "vidore/economics_reports_v2",
        "esg_r": "vidore/esg_reports_v2",
        "esg_h": "vidore/esg_reports_human_labeled_v2",
    },
    "jinavdr": {
        "slide":       "jinaai/stanford_slide",
        "openai":      "jinaai/openai-news",
        "auto_jp":     "jinaai/automobile_catalogue_jp",
        "docqa_ai":    "jinaai/docqa_artificial_intelligence",
        "jina_book":   "jinaai/jina_2024_yearly_book",
        "mmtab":       "jinaai/MMTab",
        "charxiv":     "jinaai/CharXiv-en",
        "de_news":     "jinaai/europeana-de-news",
        "ru_bev":      "jinaai/beverages_catalogue_ru",
        "zh_shanghai": "jinaai/shanghai_master_plan",
        "nl_legal":    "jinaai/europeana-nl-legal",
        "es_news":     "jinaai/europeana-es-news",
        "it_scans":    "jinaai/europeana-it-scans",
        "ar_info":     "jinaai/arabic_infographicsvqa_ar",
        "hi_gov":      "jinaai/hindi-gov-vqa",
    },
}

# budgets : fraction de vecteurs gardés par page, plus un budget absolu (comparaison AGC@64)
BUDGETS = ["05pct", "10pct", "15pct", "20pct", "25pct", "30pct", "64abs"]
DEFAULT_SEED = 42

CODE = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("SAGE_ARTIFACTS", CODE / "artifacts")).expanduser()
REPORTS = Path(os.environ.get("SAGE_REPORTS", CODE / "reports")).expanduser()
DATA = Path(os.environ.get("SAGE_DATA", Path.home() / "STRIDE" / "data_10k")).expanduser()


def parse_budget(spec):
    """'05pct' | '5pct' | '0.05' | '64abs' -> (tag, k_of) avec k_of(N) = nb de vecteurs gardés."""
    s = str(spec).strip().lower()
    if s.endswith("abs"):
        k = int(s[:-3])
        return f"{k:02d}abs", lambda n: min(k, n)
    r = float(s[:-3]) / 100 if s.endswith("pct") else float(s)
    return f"{round(r * 100):02d}pct", lambda n: max(int(n * r), 1)


def splits(bench, only=None):
    """{split: repo} d'un banc, restreint à `only` si fourni."""
    known = BENCHMARKS[bench]
    if not only:
        return dict(known)
    bad = [s for s in only if s not in known]
    if bad:
        raise SystemExit(f"split(s) inconnu(s) pour {bench} : {bad} (choix : {', '.join(known)})")
    return {s: known[s] for s in only}


class Paths:
    """Chemins des artefacts d'un modèle. Un seed autre que 42 suffixe régresseur, profils et résultats."""

    def __init__(self, model, seed=DEFAULT_SEED, root=None):
        if model not in MODELS:
            raise SystemExit(f"modèle inconnu '{model}' (choix : {', '.join(MODELS)})")
        self.model, self.seed = model, seed
        self.dim = MODELS[model]["dim"]
        self.root = Path(root or ARTIFACTS) / model
        self.sfx = "" if seed == DEFAULT_SEED else f"_s{seed}"

    def emb(self, bench, split):
        return self.root / "embeddings" / bench / f"{split}.pkl"

    def train(self, name):
        return self.root / "train" / name

    def train_docs(self, i):
        return self.root / "train" / "docs" / f"shard_{i}.pkl"

    def regressor(self):
        return self.train(f"regressor{self.sfx}.pt")

    def regressor_meta(self):
        return self.train(f"regressor{self.sfx}.meta.json")

    def profiles(self):
        return self.train(f"profiles64{self.sfx}.pkl")

    def result_dir(self, bench, split, tag):
        return self.root / f"results{self.sfx}" / bench / split / tag

    def result(self, bench, split, tag, method):
        return self.result_dir(bench, split, tag) / f"{method}.pkl"

    def native(self, bench, split):
        return self.root / "results" / bench / split / "native.pkl"
