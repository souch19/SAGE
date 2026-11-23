# SAGE-I

Compression post-hoc d'index multi-vecteurs (late interaction) guidée par la saillance.
Un petit MLP prédit l'importance de chaque patch d'une page, appris sur des cibles
d'attribution MaxSim calculées sur 10 000 pages d'entraînement ; un opérateur à budget k
compresse ensuite l'index. L'opérateur principal, Sage, est un k-means pondéré par cette
saillance. Évaluation sur ViDoRe v2 (4 splits) et JinaVDR (15 splits), avec cinq backbones :
ColPali, ColQwen2.5, ColNomic, Jina-v4 et ColBERT Qwen2.5-VL (2048-D, backbone d'AGC).
Le projet s'appelait STRIDE : la méthode `stride` (S-Group) et le dossier de données par
défaut (`~/STRIDE/data_10k`) en gardent le nom.

## Organisation

```
code/
├── run.sh                    enchaîne les étapes (./run.sh help)
├── env.sh                    chemins par défaut (artefacts, données, venv)
├── sage/                     bibliothèque
│   ├── config.py             modèles, bancs, budgets, arborescence des artefacts
│   ├── data.py               chargement de ViDoRe v2, JinaVDR et du pool d'entraînement
│   ├── encoders.py           ColPali/ColQwen/ColNomic, Jina-v4, ColBERT et AGC
│   ├── salience.py           régresseur de saillance, profils de requêtes
│   ├── kmeans.py             k-means pondéré : sklearn (CPU) et torch (GPU)
│   ├── operators.py          opérateurs de compression d'une page
│   ├── methods.py            registre des méthodes évaluées
│   ├── retrieval.py          MaxSim, nDCG@5, succès@k
│   └── utils.py              device, graines, écritures atomiques
├── 01_data/                  pool d'entraînement 10k (vidore/colpali_train_set)
├── 02_encode/                encodage du pool et des bancs d'évaluation
├── 03_salience/              cibles d'attribution, régresseur, profils
├── 04_evaluate/              compression et nDCG@5 ; AGC ; borne haute query-aware
├── 05_analysis/              tableaux, intervalles de confiance, graines, coûts, contrôle
├── scripts/                  job OAR, rapatriement des résultats, migration d'anciens artefacts
└── requirements/             environnements figés (pip freeze, Python 3.12, CUDA 12.4)
```

## Installation

Les environnements figés exigent **Python 3.12 ou plus récent** (`scipy==1.18.0` n'existe pas
pour 3.11). Si la machine n'a qu'un Python plus ancien, `uv` en installe un dans le home, sans
droits administrateur :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv --python 3.12 ~/venv-mm      # puis : uv pip install --python ~/venv-mm/bin/python ...
```

Trois environnements pour les modèles, à cause de versions de `transformers` incompatibles,
et un quatrième, léger, pour construire le pool d'entraînement :

```bash
python3.12 -m venv ~/venv-mm && ~/venv-mm/bin/pip install --no-deps -r requirements/colpali.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124       # colpali, colqwen, colnomic
python3.12 -m venv ~/venv-jina && ~/venv-jina/bin/pip install --no-deps -r requirements/jina.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124       # jinav4
python3.12 -m venv ~/venv-stride && ~/venv-stride/bin/pip install -r requirements/omni.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124       # colbert, AGC
python3.12 -m venv ~/venv-data && ~/venv-data/bin/pip install -r requirements/data.txt   # pool 10k
git clone https://github.com/HanxiangQin/omni-col-press ~/omni-col-press
git -C ~/omni-col-press checkout 4a559677bbc8a3ea0c10322a721b52bb70d382ec
```

Avec `uv pip install`, ajouter `--index-strategy unsafe-best-match` pour que l'index PyTorch
et PyPI soient consultés ensemble.

`--no-deps` installe les freezes tels quels (Jina-v4 exige une version de `transformers`
au-delà de celle que déclare colpali-engine). `run.sh` active de lui-même l'environnement de
chaque modèle (`VENV_COLPALI`, `VENV_JINA`, `VENV_OMNI` dans `env.sh`), sauf pour
`./run.sh data`, qui utilise le Python courant : activer `~/venv-data` avant. Sans accès
réseau sur les nœuds, les modèles et datasets déjà en cache suffisent avec
`export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

### Jeton Hugging Face

Sur une machine partagée (cluster), l'IP sortante est commune à tous les utilisateurs et
Hugging Face limite vite les requêtes anonymes (erreur `HTTP 429 Too Many Requests`). Se
connecter une fois avec un jeton en lecture :

```bash
~/venv-data/bin/hf auth login                 # jeton créé sur huggingface.co/settings/tokens
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

Le jeton suffit pour les modèles et les bancs d'évaluation. En revanche, `duckdb` lit les
fichiers `hf://` de façon anonyme : pour le pool d'entraînement, passer par
`--download-shards` (voir plus bas).

## Utilisation

```bash
./run.sh check                 # environnements, omni-col-press, GPU, pool d'entraînement
./run.sh model colqwen         # un modèle : entraînement, encodage, évaluation
./run.sh baselines             # AGC reproduit sur ColQwen-2048
./run.sh report                # tableaux et intervalles de confiance dans reports/
./run.sh all                   # tout, pour les cinq modèles
```

Pool d'entraînement (une fois, environ 30 min, dans `$SAGE_DATA`). Les 82 shards de
`vidore/colpali_train_set` sont d'abord téléchargés avec `huggingface_hub`, qui utilise le
jeton, puis lus en local ; le dossier des shards peut être supprimé ensuite. Un pool
interrompu reprend là où il s'était arrêté.

```bash
source ~/venv-data/bin/activate
./run.sh data --download-shards ~/hf_data/colpali_train_set
./run.sh data --check          # 10000 pages, 55399 paires attendues
```

Pour réutiliser les régresseurs du papier sans réentraîner, extraire la release v1.0 (voir
plus bas), puis lancer `./run.sh encode <modèle>` et `./run.sh eval <modèle>` au lieu de
`./run.sh model`.

### Grid'5000 (OAR)

```bash
mkdir -p logs && oarsub -q default -t exotic -p grouille -l gpu=1,walltime=14:00:00 \
       -O logs/%jobid%.out -E logs/%jobid%.err "scripts/oar_job.sh model colqwen"
```

### MesoNET Juliet (SLURM)

L'option `--account` est obligatoire (le projet `default` est refusé). Les nœuds de calcul ont
accès à Internet. Un job par modèle, en tableau :

```bash
#!/bin/bash
#SBATCH --job-name=sage
#SBATCH --account=mXXXXX
#SBATCH --partition=mesonet
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "$SLURM_SUBMIT_DIR"
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
MODELS=(colpali colqwen colnomic jinav4 colbert)
./run.sh model "${MODELS[$SLURM_ARRAY_TASK_ID]}"
```

Puis `./run.sh baselines && ./run.sh report` dans un second job soumis avec
`sbatch --dependency=afterok:<jobid>`. Sur une A100 80 Go, l'encodage et l'évaluation d'un
modèle prennent de 10 à 14 h, les baselines et le rapport environ 7 h.

Variables utiles : `MODELS`, `BENCHES`, `BUDGETS`, `KMEANS` (`auto`, `sklearn` ou `gpu`),
`CLEANUP=1`, `CARBON=1`. Chaque script se lance aussi seul (`python 04_evaluate/evaluate.py -h`) ;
un résultat déjà présent est sauté, `--force` le recalcule.

## Vérifier une installation

```bash
./run.sh smoke colqwen     # tout le pipeline sur 64 pages d'entraînement et 60 pages par split, dans /tmp
./run.sh smoke colbert     # idem pour ColBERT, avec AGC

# recalculer quelques résultats à partir d'embeddings existants et les comparer à l'ancien arbre
T=/tmp/sage_cmp && mkdir -p $T/colqwen && ln -s ~/STRIDE/artifacts/colqwen/{embeddings,train} $T/colqwen/
SAGE_ARTIFACTS=$T python 04_evaluate/evaluate.py --model colqwen --splits econ --budgets 05pct 64abs
SAGE_ARTIFACTS=$T python 05_analysis/check.py --ref ~/STRIDE/artifacts
```

## Méthodes

Par défaut, seuls les opérateurs du papier (groupe `paper`) et AGC sont lancés. Les autres
méthodes restent disponibles (`--methods all`, ou décommenter les lignes de `run.sh baselines`).

| code | papier | groupe | principe |
|---|---|---|---|
| `select` | S-Top-K | paper | k patchs les plus saillants |
| `stride` | S-Group | paper | regroupement glouton (voisinage spatial et de réponse), fusion softmax |
| `wkmeans` | Sage | paper | k-means pondéré par la saillance |
| `XR_wkmeans` | S-Xr | paper | k-means pondéré sur [position, réponse aux profils] |
| `XP_wkmeans` | S-Xp | paper | k-means pondéré sur [position, embedding] |
| `kmeans` | k-means | ablation | k-means uniforme (sans saillance) |
| `ptm_mean`, `ptm_wavg` | Prune-then-Merge | ablation | seuil de saillance puis Ward |
| `random`, `bottomk_red`, `norm_sel`, `profile_sel`, `profile_wkm`, `dyn_wkmeans` | | trainfree | signaux sans apprentissage (type OmniSelect) |
| `omni_select`, `omni_merge` | | adaptive | budget par page selon la redondance |
| `agc` | AGC | paper | compression pendant l'encodage (`04_evaluate/agc.py`) |
| `topk_query` | | | sélection conditionnée par la requête, borne haute (`04_evaluate/query_conditioned.py`) |

Budgets : 5 à 30 % des vecteurs de la page (`05pct` ... `30pct`) et 64 vecteurs (`64abs`).
Rétention = 100 x nDCG@5 (méthode) / nDCG@5 (index complet), moyennée sur les splits.

## Artefacts

```
$SAGE_ARTIFACTS/{modèle}/
    embeddings/{banc}/{split}.pkl                     doc_emb, q_emb, rel, ndcg5, npatch
    train/docs/shard_{i}.pkl, queries.pkl, meta.pkl, targets.pkl
    train/regressor.pt, regressor.meta.json, profiles64.pkl
    results/{banc}/{split}/native.pkl
    results/{banc}/{split}/{budget}/{méthode}.pkl     ndcg5, recall, per_query, avg_vectors...
```

Les résultats d'AGC et du k-means uniforme sont rangés sous `colbert/` ; les anciens dossiers
`agc/`, `kmeans/` et `{modèle}_xp/` s'y intègrent avec `scripts/migrate_artifacts.sh`.
Une graine autre que 42 (`--seed`, `./run.sh seeds`) écrit `regressor_s{graine}.pt`,
`profiles64_s{graine}.pkl` et `results_s{graine}/`.

## Analyses

- `05_analysis/aggregate.py` : tableaux Markdown par banc et `results.csv`.
- `05_analysis/significance.py` : IC 95 % de la rétention par bootstrap apparié sur les
  requêtes, et écart à une méthode de référence (`--ref`) avec sa p-valeur.
- `05_analysis/seed_variance.py` : moyenne et écart-type de la rétention entre graines
  (régresseur, profils, initialisations des k-means).
- `05_analysis/cost.py`, `cost_encode.py`, `kmeans_backends.py` : FLOPs, latences, stockage,
  CO2 (CodeCarbon), comparaison des deux k-means.
- `05_analysis/check.py` : contrôle des fichiers de résultats et comparaison à un autre arbre.

## Reproductibilité

Les graines sont fixées (42). Sur CPU, deux exécutions donnent des résultats identiques au
bit près ; sur GPU, les sommes atomiques et l'encodage en bf16 peuvent créer de petits écarts.
Un run complet sur une autre machine (MesoNET Juliet, A100) a reproduit tous les chiffres de
`results/` sauf un : Sage sur ColQwen2.5 / JinaVDR (94,1 contre 94,9 à K=64, 91,6 contre
91,3 à 5 %). Le calibrage du
seuil de S-Group tire ses paires de patchs avec un générateur local : il ne dépend plus des
splits traités avant dans le même processus, ce qui peut décaler légèrement les chiffres de
S-Group par rapport à l'ancien code ; toutes les autres méthodes reproduisent ses résultats à
l'identique. Pour retrouver exactement le pool d'entraînement du rapport, partir de la liste
figée (`01_data/build_train10k.py --from-pairs`).

## Résultats du papier

`results/` contient les tableaux du papier EGC 2027 (`vidore_v2.md`, `jinavdr.md`, `results.csv`),
générés par `05_analysis/aggregate.py` à partir de nos runs. Pour les recalculer : `./run.sh all`.

Les poids des régresseurs, les profils et les résultats bruts sont dans la release v1.0 :
`tar xzf sage_i_weights.tar.gz` à la racine du dépôt les remet dans `artifacts/`.
