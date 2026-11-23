"""Régresseur de saillance par patch et profils de requêtes zero-shot."""
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 256


class SaliencyMLP(nn.Module):
    """MLP peu profond avec connexion résiduelle ; sortie sigmoïde par patch."""

    def __init__(self, dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.skip = nn.Linear(dim, hidden) if dim != hidden else nn.Identity()
        self.head = nn.Linear(hidden, 1)

    def forward(self, p):
        return self.head(self.net(p) + self.skip(p)).squeeze(-1).sigmoid()


def _require(path, script):
    if not path.exists():
        raise SystemExit(f"{path} absent : lancer d'abord {script}")


def load_regressor(paths, device):
    _require(paths.regressor(), "03_salience/train_regressor.py")
    meta = json.loads(paths.regressor_meta().read_text())
    model = SaliencyMLP(meta["embed_dim"], meta.get("hidden", HIDDEN)).to(device)
    model.load_state_dict(torch.load(paths.regressor(), map_location=device))
    return model.eval()


@torch.no_grad()
def predict(model, de, device):
    """Saillance de chaque patch, pour toutes les pages : {page: [N]} (sur device)."""
    return {d: model(p.to(device).float().unsqueeze(0)).squeeze(0) for d, p in de.items()}


def load_profiles(paths):
    """Tokens des 64 requêtes profils, concaténés et L2-normalisés : [T, dim]."""
    _require(paths.profiles(), "03_salience/profiles.py")
    with open(paths.profiles(), "rb") as f:
        q = pickle.load(f)["queries"]
    return F.normalize(torch.cat([q[k].float() for k in q], 0), dim=-1)
