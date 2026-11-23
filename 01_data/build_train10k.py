"""Pool d'entraînement : les 10 000 pages de vidore/colpali_train_set qui portent le plus de
requêtes distinctes (au moins 2), toutes leurs paires (requête, page) et une image par page.

Sorties dans --out ($SAGE_DATA) :
  s00_top10k_pages.csv               source, image_filename, nq
  s00_top10k_pairs_raw.parquet       paires (source, image_filename, query)
  <image_filename>.jpg               une image par page ('/' remplacés par '__')
  s00_top10k_image_manifest.json     {image_filename: chemin}
  s00_top10k_pairs_indexed.parquet   qid, source, image_filename, query
  s00_train_relevance.json           {qid: [image_filename]}
  s00_top10k_summary.md              composition par source

La sélection ne lit que les métadonnées (DuckDB) ; les images sont lues shard par shard.
Chaque étape est mise en cache. Les pages à égalité de nq sont départagées par nom de fichier :
pour retrouver exactement le pool du rapport (10 000 pages, 55 399 paires), repartir de la
liste figée avec --from-pairs.

    python 01_data/build_train10k.py
    python 01_data/build_train10k.py --from-pairs ref/s00_top10k_pairs_indexed.parquet
    python 01_data/build_train10k.py --check
"""
import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sage.config import DATA
from sage.data import resolve_image

REPORT = {"pages": 10_000, "pairs": 55_399, "q_mean": 5.54, "q_min": 3, "q_max": 28}


def safe_name(fn):
    return fn.replace("/", "__").replace("\\", "__") + ".jpg"


def write_atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def connect(memory_limit, threads, remote, proxy=None, keep_order=True):
    import duckdb
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    if not keep_order:
        con.execute("SET preserve_insertion_order=false")
    if remote:
        if proxy:
            con.execute(f"SET http_proxy='{proxy.split('://', 1)[-1].rstrip('/')}'")
        try:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
        except Exception as e:
            raise SystemExit(f"DuckDB ne charge pas httpfs ({type(e).__name__}) : relancer avec "
                             f"--download-shards DIR ou --parquet-glob vers des shards locaux")
    return con


def select_pages(con, glob, out, top_k, min_q, force):
    f = out / "s00_top10k_pages.csv"
    if f.exists() and not force:
        return pd.read_csv(f)
    sel = con.execute(f"""
        SELECT source, image_filename, nq FROM (
            SELECT source, image_filename, COUNT(DISTINCT query) AS nq
            FROM read_parquet('{glob}') GROUP BY source, image_filename)
        WHERE nq >= {min_q}
        ORDER BY nq DESC, image_filename, source
        LIMIT {top_k}""").df()
    sel.to_csv(f, index=False)
    print(f"[1/4] {len(sel)} pages, {int(sel.nq.sum())} requêtes, nq dans [{sel.nq.min()}, {sel.nq.max()}]")
    return sel


def collect_pairs(con, glob, out, sel, force):
    f = out / "s00_top10k_pairs_raw.parquet"
    if f.exists() and not force:
        return pd.read_parquet(f)
    con.register("sel", sel[["image_filename"]])
    pairs = con.execute(f"""
        SELECT t.source, t.image_filename, t.query
        FROM read_parquet('{glob}') t SEMI JOIN sel s ON t.image_filename = s.image_filename""").df()
    pairs = (pairs.drop_duplicates(["image_filename", "query"])
                  .sort_values(["image_filename", "query"], kind="stable").reset_index(drop=True))
    write_atomic(f, lambda t: pairs.to_parquet(t, index=False))
    print(f"[2/4] {len(pairs)} paires (requête, page)")
    return pairs


def write_images(con, glob, out, filenames):
    """Une image par page ; les images déjà écrites ne sont pas retéléchargées."""
    need, manifest = set(filenames), {}
    for fn in list(need):
        p = out / safe_name(fn)
        if p.exists() and p.stat().st_size > 0:
            manifest[fn] = str(p.resolve())
            need.discard(fn)
    files = con.execute(f"SELECT file FROM glob('{glob}')").df()["file"].tolist() if need else []
    for i, shard in enumerate(files):
        if not need:
            break
        con.register("sel", pd.DataFrame({"image_filename": sorted(need)}))
        df = con.execute(f"""
            SELECT t.image_filename, t.image.bytes AS b
            FROM read_parquet('{shard}') t SEMI JOIN sel s ON t.image_filename = s.image_filename""").df()
        for fn, b in zip(df.image_filename, df.b):
            if fn in need and b is not None:
                p = out / safe_name(fn)
                write_atomic(p, lambda t: t.write_bytes(b))
                manifest[fn] = str(p.resolve())
                need.discard(fn)
        del df
        gc.collect()
        print(f"  shard {i + 1}/{len(files)} : {len(manifest)} images, {len(need)} restantes")
    write_atomic(out / "s00_top10k_image_manifest.json", lambda t: t.write_text(json.dumps(manifest)))
    if need:
        print(f"[!] {len(need)} pages sans image (exclues)")
    print(f"[3/4] {len(manifest)} images")
    return manifest


def index_pairs(out, pairs, manifest):
    df = pairs[pairs["image_filename"].isin(manifest)].reset_index(drop=True)
    if "qid" not in df.columns:
        df.insert(0, "qid", df.index)
    write_atomic(out / "s00_top10k_pairs_indexed.parquet", lambda t: df.to_parquet(t, index=False))
    rel = {str(int(q)): [fn] for q, fn in zip(df["qid"], df["image_filename"])}
    write_atomic(out / "s00_train_relevance.json", lambda t: t.write_text(json.dumps(rel)))
    print(f"[4/4] {len(df)} requêtes, {df.image_filename.nunique()} pages")
    return df


def summary(out, df):
    g = df.groupby("image_filename")
    pages = pd.DataFrame({"nq": g["query"].nunique(),
                          "source": g["source"].first() if "source" in df.columns else "?"})
    md = ["| source | pages | requêtes | q moy | q min | q max |", "|---|---|---|---|---|---|"]
    for src, s in sorted(pages.groupby("source"), key=lambda x: -len(x[1])):
        md.append(f"| {src} | {len(s)} | {int(s.nq.sum())} | {s.nq.mean():.2f} | {s.nq.min()} | {s.nq.max()} |")
    md.append(f"| **total** | **{len(pages)}** | **{int(pages.nq.sum())}** | **{pages.nq.mean():.2f}** "
              f"| {pages.nq.min()} | {pages.nq.max()} |")
    md += ["", f"Rapport : {REPORT['pages']} pages, {REPORT['pairs']} paires, q moy {REPORT['q_mean']}, "
               f"min {REPORT['q_min']}, max {REPORT['q_max']}."]
    (out / "s00_top10k_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def check(out, n_open=20):
    names = ["s00_top10k_image_manifest.json", "s00_top10k_pairs_indexed.parquet", "s00_train_relevance.json"]
    missing = [n for n in names if not (out / n).exists()]
    if missing:
        print(f"absent(s) de {out} : {', '.join(missing)}")
        return False
    base = {k: Path(v).name for k, v in json.loads((out / names[0]).read_text()).items()}
    pairs = pd.read_parquet(out / names[1])
    rel = json.loads((out / names[2]).read_text())
    images = list(pairs["image_filename"].unique())
    lost = [im for im in images if resolve_image(out, im, base) is None]
    qids = set(pairs["qid"].astype(int))
    unknown = [q for q in rel if int(q) not in qids]
    from PIL import Image
    unreadable = 0
    for im in images[:n_open]:
        try:
            Image.open(resolve_image(out, im, base)).convert("RGB")
        except Exception:
            unreadable += 1
    print(f"pages={len(images)} requêtes={len(pairs)} images introuvables={len(lost)} "
          f"qid inconnus={len(unknown)} illisibles (sur {n_open})={unreadable}")
    summary(out, pairs)
    return not (lost or unknown or unreadable)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--repo", default="vidore/colpali_train_set")
    ap.add_argument("--parquet-glob", default=None, help="défaut : hf://datasets/<repo>/data/train-*.parquet")
    ap.add_argument("--download-shards", default=None, metavar="DIR",
                    help="télécharge les shards (huggingface_hub) puis lit en local")
    ap.add_argument("--top-k", type=int, default=10_000)
    ap.add_argument("--min-queries", type=int, default=2)
    ap.add_argument("--from-pairs", default=None, metavar="PARQUET", help="liste figée de paires indexées")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--check", action="store_true", help="vérifie --out sans rien télécharger")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if args.check:
        raise SystemExit(0 if check(out) else 1)

    glob = args.parquet_glob or f"hf://datasets/{args.repo}/data/train-*.parquet"
    if args.download_shards:
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=args.repo, repo_type="dataset",
                                  allow_patterns=["data/train-*.parquet"],
                                  local_dir=str(Path(args.download_shards).expanduser()))
        glob = str(Path(local) / "data" / "train-*.parquet")
    remote = "://" in glob
    proxy = next((os.environ[k] for k in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY")
                  if os.environ.get(k)), None)
    image_con = lambda: connect("4GB", 1, remote, proxy, keep_order=False)

    if args.from_pairs:
        src = Path(args.from_pairs).expanduser().resolve()
        pairs = pd.read_parquet(src)
        for f, name in ((src, "s00_top10k_pairs_indexed.parquet"),
                        (src.parent / "s00_train_relevance.json", "s00_train_relevance.json")):
            if f.exists() and f.resolve() != (out / name).resolve():
                shutil.copy2(f, out / name)
        if not args.skip_images:
            write_images(image_con(), glob, out, pairs["image_filename"].unique())
        if not (out / "s00_train_relevance.json").exists():
            rel = {str(int(q)): [fn] for q, fn in zip(pairs["qid"], pairs["image_filename"])}
            (out / "s00_train_relevance.json").write_text(json.dumps(rel))
        summary(out, pairs)
        return

    con = connect(args.memory_limit, args.threads, remote, proxy)
    sel = select_pages(con, glob, out, args.top_k, args.min_queries, args.force)
    pairs = collect_pairs(con, glob, out, sel, args.force)
    if args.skip_images:
        summary(out, pairs)
        return
    manifest = write_images(image_con(), glob, out, sel["image_filename"])
    summary(out, index_pairs(out, pairs, manifest))


if __name__ == "__main__":
    main()
