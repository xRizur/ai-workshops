"""
run_full_benchmark.py
Full benchmark: MLP + GIN on real ChEMBL data for one protein.

Key design decisions:
  - Scaffold split (Bemis-Murcko) instead of random: tests true generalisation
    to new chemical series, not just interpolation within known scaffolds.
    Random split is optimistic; scaffold split is the standard in drug discovery.
  - MLP features: 8 RDKit descriptors + 2048-bit Morgan fingerprints (radius=2)
  - GIN: molecular graph directly from SMILES
  - All IC50 measurements for top protein (up to 40k, not capped at 10k)
  - 300 epochs, early stopping patience=25
"""

import os, warnings, time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "."
RESULTS_DIR = "results_benchmark"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_SAMPLES  = None   # None = use ALL data for top protein
BATCH_SIZE   = 256
GNN_BATCH    = 64
LR           = 1e-3
MAX_EPOCHS   = 300
PATIENCE     = 25
HIDDEN_DIMS  = [512, 256, 128]
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4
MORGAN_BITS  = 2048
MORGAN_RADIUS = 2
RDKIT_FEATS   = ["MW", "LogP", "HBD", "HBA", "TPSA", "QED", "RotBonds", "AromaticRings"]

print(f"Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Load and join ChEMBL silver parquets")
print("=" * 65)

df_act = pd.read_parquet(os.path.join(DATA_DIR, "activities_silver.parquet"))
df_ass = pd.read_parquet(os.path.join(DATA_DIR, "assays_silver.parquet"))
df_tgt = pd.read_parquet(os.path.join(DATA_DIR, "targets_silver.parquet"))
df_mol = pd.read_parquet(os.path.join(DATA_DIR, "molecules_silver.parquet"))

df = (df_act
      .merge(df_ass, on="assay_id", how="inner")
      .merge(df_tgt, on="tid",      how="inner")
      .merge(df_mol, on="molregno", how="inner"))
df = df.dropna(subset=["canonical_smiles", "pIC50"])
df = df[(df["pIC50"] >= 2) & (df["pIC50"] <= 12)]
df_hs = df[df["organism"] == "Homo sapiens"]

top_counts = df_hs.groupby("tid").size().sort_values(ascending=False)
TOP_TID    = top_counts.index[0]
N_TOTAL    = top_counts.iloc[0]
print(f"  Top protein TID={TOP_TID}  ({N_TOTAL:,} measurements in HS)")

df_p = df_hs[df_hs["tid"] == TOP_TID].copy()
df_p = df_p.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
print(f"  After dedup by SMILES: {len(df_p):,}")

if MAX_SAMPLES and len(df_p) > MAX_SAMPLES:
    df_p = df_p.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
print(f"  Working set: {len(df_p):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RDKIT FEATURES + MORGAN FINGERPRINTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Compute RDKit descriptors + Morgan fingerprints")
print("=" * 65)

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED as QED_module
from rdkit.Chem.Scaffolds import MurckoScaffold

def rdkit_features(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, None, None
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            QED_module.qed(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
        ]
        fp = list(AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS))
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
        return desc, fp, scaffold
    except Exception:
        return None, None, None

print(f"  Processing {len(df_p):,} molecules...")
t0 = time.time()
descs, fps, scaffolds = [], [], []
valid_mask = []
for smi in df_p["canonical_smiles"]:
    d, f, s = rdkit_features(smi)
    if d is None:
        valid_mask.append(False)
    else:
        descs.append(d); fps.append(f); scaffolds.append(s)
        valid_mask.append(True)

df_p = df_p[valid_mask].reset_index(drop=True)
print(f"  Done in {time.time()-t0:.1f}s  |  valid: {len(df_p):,}")

feat_df = pd.DataFrame(descs, columns=RDKIT_FEATS)
fp_cols  = [f"fp{i}" for i in range(MORGAN_BITS)]
fp_df    = pd.DataFrame(fps, columns=fp_cols, dtype=np.float32)
df_p     = pd.concat([df_p.reset_index(drop=True), feat_df, fp_df], axis=1)
df_p["scaffold"] = scaffolds

print(f"  Morgan fingerprints: {MORGAN_BITS}-bit, radius={MORGAN_RADIUS}")
print(f"  Total MLP features : {len(RDKIT_FEATS)} descriptors + {MORGAN_BITS} fps = {len(RDKIT_FEATS)+MORGAN_BITS}")

smiles_list = df_p["canonical_smiles"].tolist()
y_all       = df_p["pIC50"].values.astype(np.float32)

print(f"\n  pIC50: mean={y_all.mean():.2f}  std={y_all.std():.2f}  "
      f"range=[{y_all.min():.2f}, {y_all.max():.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SCAFFOLD SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Scaffold split  (70 / 15 / 15)")
print("=" * 65)
print("  Bemis-Murcko scaffold split: molecules with the same scaffold")
print("  go into the same fold -> tests generalisation to new scaffolds")
print("  (random split is too optimistic for drug discovery)")

scaffold_to_idx = defaultdict(list)
for i, sc in enumerate(df_p["scaffold"]):
    scaffold_to_idx[sc].append(i)

# Sort scaffolds largest first so train gets well-represented scaffolds
scaffold_groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)

n      = len(df_p)
n_tr   = int(0.70 * n)
n_va   = int(0.15 * n)

idx_tr, idx_va, idx_te = [], [], []
for grp in scaffold_groups:
    if len(idx_tr) < n_tr:
        idx_tr.extend(grp)
    elif len(idx_va) < n_va:
        idx_va.extend(grp)
    else:
        idx_te.extend(grp)

idx_tr = np.array(idx_tr); idx_va = np.array(idx_va); idx_te = np.array(idx_te)
print(f"  Train : {len(idx_tr):,}  ({len(idx_tr)/n*100:.1f}%)")
print(f"  Val   : {len(idx_va):,}  ({len(idx_va)/n*100:.1f}%)")
print(f"  Test  : {len(idx_te):,}  ({len(idx_te)/n*100:.1f}%)")

n_sc_tr = len(set(df_p.iloc[idx_tr]["scaffold"]))
n_sc_va = len(set(df_p.iloc[idx_va]["scaffold"]))
n_sc_te = len(set(df_p.iloc[idx_te]["scaffold"]))
print(f"  Unique scaffolds: train={n_sc_tr}  val={n_sc_va}  test={n_sc_te}")

# check overlap
sc_tr = set(df_p.iloc[idx_tr]["scaffold"])
sc_te = set(df_p.iloc[idx_te]["scaffold"])
overlap = sc_tr & sc_te - {""}
print(f"  Scaffold overlap train/test: {len(overlap)} (should be 0)")

# ─────────────────────────────────────────────────────────────────────────────
# MLP DATA PREP
# ─────────────────────────────────────────────────────────────────────────────
MLP_FEAT_COLS = RDKIT_FEATS + fp_cols
X_all = df_p[MLP_FEAT_COLS].values.astype(np.float32)

X_tr_raw, y_tr_raw = X_all[idx_tr], y_all[idx_tr]
X_va_raw, y_va_raw = X_all[idx_va], y_all[idx_va]
X_te_raw, y_te_raw = X_all[idx_te], y_all[idx_te]

# Scale (fit on train only)
fs = StandardScaler().fit(X_tr_raw)
ts = StandardScaler().fit(y_tr_raw.reshape(-1, 1))

def to_tensor(a): return torch.tensor(a, dtype=torch.float32).to(DEVICE)

X_tr = to_tensor(fs.transform(X_tr_raw))
y_tr = to_tensor(ts.transform(y_tr_raw.reshape(-1,1)).ravel())
X_va = to_tensor(fs.transform(X_va_raw))
y_va = to_tensor(ts.transform(y_va_raw.reshape(-1,1)).ravel())
X_te = to_tensor(fs.transform(X_te_raw))

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MLP MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: MLP training")
print(f"  Features: {len(MLP_FEAT_COLS)} ({len(RDKIT_FEATS)} RDKit + {MORGAN_BITS} Morgan fps)")
print(f"  Architecture: {len(RDKIT_FEATS)+MORGAN_BITS} -> {' -> '.join(map(str,HIDDEN_DIMS))} -> 1")
print("=" * 65)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers, in_d = [], input_dim
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout if i < len(hidden_dims)-1 else dropout*0.5),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grad_norm(model):
    return sum(p.grad.data.norm(2).item()**2
               for p in model.parameters() if p.grad is not None) ** 0.5


def dead_pct(model, Xp):
    stats, hooks = {}, []
    def mk(name):
        def h(mod, inp, out):
            with torch.no_grad():
                stats[name] = (out <= 0).float().mean().item()
        return h
    for nm, mod in model.named_modules():
        if isinstance(mod, nn.LeakyReLU):
            hooks.append(mod.register_forward_hook(mk(nm)))
    model.eval()
    with torch.no_grad(): model(Xp)
    model.train()
    for h in hooks: h.remove()
    return stats


def train_eval(model, loader, crit, optimizer=None):
    if optimizer:
        model.train()
        total = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(Xb)
        return total / len(loader.dataset)
    else:
        model.eval()
        with torch.no_grad():
            preds = torch.cat([model(Xb) for Xb,_ in loader]).cpu().numpy()
            loss  = crit(torch.tensor(preds),
                         loader.dataset.tensors[1].cpu()).item()
        return loss, preds


def run_mlp():
    model = MLP(input_dim=X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

    hist = {k: [] for k in ["tr","va","va_r2","gn","gd"]}
    best_va, best_wts, pat, prev_g = float("inf"), None, 0, 0.0
    X_probe = X_tr[:128]
    t0 = time.time()

    for ep in range(1, MAX_EPOCHS+1):
        tr = train_eval(model, train_loader, crit, opt)
        gn = grad_norm(model)
        gd = abs(gn - prev_g); prev_g = gn

        va, vp_s = train_eval(model, val_loader, crit)
        vp   = ts.inverse_transform(vp_s.reshape(-1,1)).ravel()
        vr2  = r2_score(y_va_raw, vp)

        sched.step(va)
        for k,v in zip(["tr","va","va_r2","gn","gd"],[tr,va,vr2,gn,gd]):
            hist[k].append(v)

        if va < best_va:
            best_va = va
            best_wts = {k:v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"  Early stop @ ep {ep}")
                break

        if ep % 30 == 0 or ep == 1:
            dead = dead_pct(model, X_probe)
            dstr = "  ".join(f"L{i+1}:{v*100:.0f}%" for i,(k,v) in enumerate(dead.items()))
            print(f"  ep{ep:03d} | tr={tr:.4f}  va={va:.4f}  R2={vr2:.4f}"
                  f"  | grad={gn:.3f}  D={gd:.3f}")
            print(f"           dead: {dstr}")

    print(f"  Total time: {time.time()-t0:.1f}s")
    model.load_state_dict(best_wts)
    model.eval()
    with torch.no_grad():
        tp_s = model(X_te).cpu().numpy()
    tp = ts.inverse_transform(tp_s.reshape(-1,1)).ravel()

    mse  = mean_squared_error(y_te_raw, tp)
    rmse = mse**0.5
    r2   = r2_score(y_te_raw, tp)
    print(f"\n  +-- MLP TEST RESULTS (scaffold split) {'-'*20}")
    print(f"  |   N train/val/test = {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")
    print(f"  |   MSE  = {mse:.4f}")
    print(f"  |   RMSE = {rmse:.4f}  pIC50")
    print(f"  |   R2   = {r2:.4f}")
    print(f"  +{'-'*55}")

    return {"label":"MLP (RDKit+Morgan, scaffold split)",
            "N":len(df_p),"mse":mse,"rmse":rmse,"r2":r2,
            "hist":hist,"yte":y_te_raw,"ypred":tp,"epochs":len(hist["tr"])}

mlp_res = run_mlp()


# ─────────────────────────────────────────────────────────────────────────────
# 5. GIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: GIN (Graph Isomorphism Network)")
print("  Building molecular graphs from SMILES...")
print("=" * 65)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool

ATOM_SYMS   = ['C','N','O','S','F','Si','P','Cl','Br','I','B','other']
HYBR        = ['S','SP','SP2','SP3','SP3D','SP3D2','other']
DEGREE_MAX  = 10
CHARGE_VALS = list(range(-3, 4))

def one_hot(v, lst):
    enc = [0]*len(lst)
    enc[lst.index(v) if v in lst else -1] = 1
    return enc

def atom_feat(atom):
    sym = atom.GetSymbol() if atom.GetSymbol() in ATOM_SYMS else 'other'
    hyb = str(atom.GetHybridization()).split('.')[-1]
    if hyb not in HYBR: hyb = 'other'
    ch  = max(CHARGE_VALS[0], min(CHARGE_VALS[-1], atom.GetFormalCharge()))
    return (one_hot(sym, ATOM_SYMS) +
            one_hot(min(atom.GetDegree(), DEGREE_MAX), list(range(DEGREE_MAX+1))) +
            one_hot(ch, CHARGE_VALS) +
            one_hot(hyb, HYBR) +
            [int(atom.GetIsAromatic())])   # 12+11+7+7+1 = 38

BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

NODE_DIM = len(atom_feat(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
print(f"  Node feature dim: {NODE_DIM}")

def smi_to_graph(smi, y_scaled, y_orig):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ef = (one_hot(b.GetBondType(), BOND_TYPES) +
              [int(b.GetIsConjugated()), int(b.IsInRing())])
        ei += [[i,j],[j,i]]; ea += [ef,ef]
    if not ei: return None
    ei = torch.tensor(ei, dtype=torch.long).t().contiguous()
    ea = torch.tensor(ea, dtype=torch.float)
    return Data(x=x, edge_index=ei, edge_attr=ea,
                y=torch.tensor([y_scaled], dtype=torch.float),
                y_orig=torch.tensor([y_orig], dtype=torch.float))

# Scale targets for GNN (fit on same train split)
ts_gin = StandardScaler().fit(y_all[idx_tr].reshape(-1,1))

print(f"  Converting {len(df_p):,} SMILES to graphs...")
t0 = time.time()
graphs = []
for smi, y_orig in zip(smiles_list, y_all):
    y_sc = ts_gin.transform([[y_orig]])[0,0]
    g = smi_to_graph(smi, y_sc, y_orig)
    graphs.append(g)   # None entries handled below

# Re-index splits (same indices as before)
def make_gin_split(indices):
    return [graphs[i] for i in indices if graphs[i] is not None]

gin_tr = make_gin_split(idx_tr)
gin_va = make_gin_split(idx_va)
gin_te = make_gin_split(idx_te)
y_te_gin = np.array([g.y_orig.item() for g in gin_te])

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Graphs: train={len(gin_tr)}  val={len(gin_va)}  test={len(gin_te)}")

gin_train_loader = PyGLoader(gin_tr, batch_size=GNN_BATCH, shuffle=True)
gin_val_loader   = PyGLoader(gin_va, batch_size=GNN_BATCH)
gin_test_loader  = PyGLoader(gin_te, batch_size=GNN_BATCH)


class GINLayer(nn.Module):
    def __init__(self, dim, dropout=DROPOUT):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(dim, dim*2), nn.BatchNorm1d(dim*2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        )
        self.conv = GINConv(mlp, train_eps=True)

    def forward(self, x, ei):
        return self.conv(x, ei)


class GINModel(nn.Module):
    def __init__(self, node_dim, hidden=128, n_layers=4, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.convs = nn.ModuleList([GINLayer(hidden, dropout) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.proj(x)
        for conv in self.convs:
            x = x + conv(x, ei)   # residual connection
        x = torch.cat([global_mean_pool(x, batch),
                        global_add_pool(x, batch)], dim=-1)
        return self.readout(x).squeeze(-1)


def eval_gin(model, loader):
    model.eval()
    preds_s, trues_o = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            preds_s.extend(model(b).cpu().numpy())
            trues_o.extend(b.y_orig.cpu().numpy().ravel())
    preds_o = ts_gin.inverse_transform(
        np.array(preds_s).reshape(-1,1)).ravel()
    return np.array(trues_o), preds_o


def run_gin():
    model = GINModel(node_dim=NODE_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GIN params: {n_params:,}")
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)

    hist = {k: [] for k in ["tr","va","va_r2","gn","gd"]}
    best_va, best_wts, pat, prev_g = float("inf"), None, 0, 0.0
    t0 = time.time()

    for ep in range(1, MAX_EPOCHS+1):
        model.train()
        tr = 0.0
        for b in gin_train_loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(b), b.y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item() * b.num_graphs
        tr /= len(gin_tr)

        gn = grad_norm(model)
        gd = abs(gn - prev_g); prev_g = gn

        # val
        model.eval()
        va_loss = 0.0
        vp_s, vy_s = [], []
        with torch.no_grad():
            for b in gin_val_loader:
                b = b.to(DEVICE)
                p = model(b)
                va_loss += crit(p, b.y.view(-1)).item() * b.num_graphs
                vp_s.extend(p.cpu().numpy())
                vy_s.extend(b.y.cpu().numpy().ravel())
        va_loss /= len(gin_va)
        vp_orig = ts_gin.inverse_transform(np.array(vp_s).reshape(-1,1)).ravel()
        vy_orig = ts_gin.inverse_transform(np.array(vy_s).reshape(-1,1)).ravel()
        vr2 = r2_score(vy_orig, vp_orig)

        sched.step(va_loss)
        for k,v in zip(["tr","va","va_r2","gn","gd"],[tr,va_loss,vr2,gn,gd]):
            hist[k].append(v)

        if va_loss < best_va:
            best_va = va_loss
            best_wts = {k:v.clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"  Early stop @ ep {ep}")
                break

        if ep % 30 == 0 or ep == 1:
            print(f"  ep{ep:03d} | tr={tr:.4f}  va={va_loss:.4f}  R2={vr2:.4f}"
                  f"  | grad={gn:.3f}  D={gd:.3f}")

    print(f"  Total time: {time.time()-t0:.1f}s")
    model.load_state_dict(best_wts)
    te_true, te_pred = eval_gin(model, gin_test_loader)

    mse  = mean_squared_error(te_true, te_pred)
    rmse = mse**0.5
    r2   = r2_score(te_true, te_pred)
    print(f"\n  +-- GIN TEST RESULTS (scaffold split) {'-'*21}")
    print(f"  |   N train/val/test = {len(gin_tr)}/{len(gin_va)}/{len(gin_te)}")
    print(f"  |   MSE  = {mse:.4f}")
    print(f"  |   RMSE = {rmse:.4f}  pIC50")
    print(f"  |   R2   = {r2:.4f}")
    print(f"  +{'-'*55}")
    return {"label":"GIN (molecular graph, scaffold split)",
            "N":len(gin_tr)+len(gin_va)+len(gin_te),
            "mse":mse,"rmse":rmse,"r2":r2,
            "hist":hist,"yte":te_true,"ypred":te_pred,"epochs":len(hist["tr"])}

gin_res = run_gin()


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_result(res, fname):
    hist = res["hist"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(f"{res['label']}  (N={res['N']}, R2={res['r2']:.3f})", fontsize=11)

    axes[0].plot(hist["tr"], label="Train")
    axes[0].plot(hist["va"], label="Val")
    axes[0].set_title("Loss (MSE, scaled)"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(hist["va_r2"], color="green")
    axes[1].axhline(res["r2"], color="red", ls="--", label=f"Test R2={res['r2']:.3f}")
    axes[1].set_title("Val R2"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(hist["gn"],  label="||grad||")
    axes[2].plot(hist["gd"], label="delta", alpha=0.7)
    axes[2].set_title("Gradient norm"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    mn = min(res["yte"].min(), res["ypred"].min())
    mx = max(res["yte"].max(), res["ypred"].max())
    axes[3].scatter(res["yte"], res["ypred"], alpha=0.3, s=8)
    axes[3].plot([mn,mx],[mn,mx],"r--",label="Perfect")
    axes[3].set_xlabel("True pIC50"); axes[3].set_ylabel("Pred pIC50")
    axes[3].set_title("Actual vs Predicted"); axes[3].legend()

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, fname)
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

plot_result(mlp_res, "mlp_scaffold.png")
plot_result(gin_res, "gin_scaffold.png")

# comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"MLP vs GIN — TID={TOP_TID}  scaffold split", fontsize=12)

for ax, res, color in zip(axes, [mlp_res, gin_res], ["steelblue","darkorange"]):
    mn = min(res["yte"].min(), res["ypred"].min())
    mx = max(res["yte"].max(), res["ypred"].max())
    ax.scatter(res["yte"], res["ypred"], alpha=0.3, s=8, color=color)
    ax.plot([mn,mx],[mn,mx],"r--")
    ax.set_xlabel("True pIC50"); ax.set_ylabel("Pred pIC50")
    ax.set_title(f"{res['label'].split('(')[0].strip()}\n"
                 f"RMSE={res['rmse']:.3f}  R2={res['r2']:.3f}")

plt.tight_layout()
cp = os.path.join(RESULTS_DIR, "comparison.png")
plt.savefig(cp, dpi=130, bbox_inches="tight"); plt.close()
print(f"  Saved: {cp}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'#'*65}")
print("  FINAL BENCHMARK SUMMARY")
print(f"  Protein TID={TOP_TID}  |  scaffold split  |  {len(df_p):,} unique molecules")
print(f"{'#'*65}")
print(f"  {'Model':<42} {'Epochs':>7}  {'RMSE':>7}  {'R2':>7}")
print("  " + "-"*62)
for r in [mlp_res, gin_res]:
    print(f"  {r['label']:<42} {r['epochs']:>7}  {r['rmse']:>7.4f}  {r['r2']:>7.4f}")

print()
diff_r2   = gin_res["r2"]   - mlp_res["r2"]
diff_rmse = mlp_res["rmse"] - gin_res["rmse"]
print(f"  GIN improvement over MLP:  R2 +{diff_r2:.3f}  |  RMSE -{diff_rmse:.3f} pIC50")
print()
print("  Note: scaffold split is harder than random split.")
print("  Expected R2 with scaffold split: 0.2-0.5 (MLP) / 0.3-0.6 (GIN)")
print(f"\n  Plots -> {RESULTS_DIR}/")
