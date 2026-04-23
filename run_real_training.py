"""
run_real_training.py
Full MLP pipeline on real ChEMBL silver parquets.
  1. JOIN silver tables, filter Homo sapiens, pick top protein by IC50 count
  2. Compute RDKit descriptors
  3. TINY (N=50, no regularisation)  -> must overfit  -> sanity check
  4. FULL (N<=10k, with Dropout)     -> generalisation -> real metrics
"""

import os, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
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
DATA_DIR     = "."          # parquets are in repo root
RESULTS_DIR  = "results_real"
MAX_SAMPLES  = 10_000
BATCH_SIZE   = 256
LR           = 1e-3
LR_TINY      = 3e-4
MAX_EPOCHS   = 150
PATIENCE     = 20
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4
FEATURES     = ["MW", "LogP", "HBD", "HBA", "TPSA", "QED", "RotBonds", "AromaticRings"]

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}\n")

# ── 1. LOAD & JOIN ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and joining ChEMBL silver parquets")
print("=" * 60)

df_act = pd.read_parquet(os.path.join(DATA_DIR, "activities_silver.parquet"))
df_ass = pd.read_parquet(os.path.join(DATA_DIR, "assays_silver.parquet"))
df_tgt = pd.read_parquet(os.path.join(DATA_DIR, "targets_silver.parquet"))
df_mol = pd.read_parquet(os.path.join(DATA_DIR, "molecules_silver.parquet"))

print(f"  activities : {df_act.shape}")
print(f"  assays     : {df_ass.shape}")
print(f"  targets    : {df_tgt.shape}")
print(f"  molecules  : {df_mol.shape}")

# join
df = df_act.merge(df_ass, on="assay_id", how="inner")
df = df.merge(df_tgt,    on="tid",      how="inner")
df = df.merge(df_mol,    on="molregno", how="inner")
df = df.dropna(subset=["canonical_smiles", "pIC50"])
df = df[(df["pIC50"] >= 2) & (df["pIC50"] <= 12)]

print(f"\n  After join + pIC50 filter: {len(df):,} rows")

# keep only Homo sapiens
df_hs = df[df["organism"] == "Homo sapiens"].copy()
print(f"  Homo sapiens only       : {len(df_hs):,} rows")

# top protein
top_counts = df_hs.groupby("tid").size().sort_values(ascending=False)
TOP_TID    = top_counts.index[0]
N_TOTAL    = top_counts.iloc[0]
print(f"\n  Top protein TID={TOP_TID}  ({N_TOTAL:,} measurements)")
print(f"  Top 5 proteins by count:")
for tid, n in top_counts.head(5).items():
    print(f"    TID={tid:6d}  n={n:7,}")

df_protein = df_hs[df_hs["tid"] == TOP_TID].copy()

# sample
if len(df_protein) > MAX_SAMPLES:
    df_protein = df_protein.sample(n=MAX_SAMPLES, random_state=SEED)
df_protein = df_protein.reset_index(drop=True)

print(f"\n  Working dataset: {len(df_protein):,} samples")
print(f"  pIC50 stats: mean={df_protein['pIC50'].mean():.2f}  "
      f"std={df_protein['pIC50'].std():.2f}  "
      f"range=[{df_protein['pIC50'].min():.2f}, {df_protein['pIC50'].max():.2f}]")

# ── 2. RDKit DESCRIPTORS ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Computing RDKit descriptors")
print("=" * 60)

from rdkit import Chem
from rdkit.Chem import Descriptors, QED as QED_module

def compute_rdkit(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * 8
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            QED_module.qed(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
        ]
    except Exception:
        return [np.nan] * 8

print(f"  Computing for {len(df_protein):,} SMILES...")
feats = [compute_rdkit(s) for s in df_protein["canonical_smiles"]]
feat_df = pd.DataFrame(feats, columns=FEATURES)
df_protein = pd.concat([df_protein.reset_index(drop=True), feat_df], axis=1)
n_bad = df_protein[FEATURES].isnull().any(axis=1).sum()
df_protein = df_protein.dropna(subset=FEATURES).reset_index(drop=True)
print(f"  Failed SMILES removed: {n_bad}")
print(f"  Final dataset: {len(df_protein):,}")

print("\n  Descriptor statistics:")
print(df_protein[FEATURES].describe().round(2).to_string())
print(f"\n  pIC50 distribution:")
for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
    print(f"    p{int(q*100):02d} = {df_protein['pIC50'].quantile(q):.2f}")

X_all = df_protein[FEATURES].values.astype(np.float32)
y_all = df_protein["pIC50"].values.astype(np.float32)

# ── MODEL ──────────────────────────────────────────────────────────────────────
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


def gradient_norm(model):
    total = sum(p.grad.data.norm(2).item()**2
                for p in model.parameters() if p.grad is not None)
    return total ** 0.5


def dead_pct(model, X_probe):
    stats, hooks = {}, []
    def make_hook(name):
        def h(mod, inp, out):
            with torch.no_grad():
                stats[name] = (out <= 0).float().mean().item()
        return h
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LeakyReLU):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        _ = model(X_probe)
    model.train()
    for h in hooks: h.remove()
    return stats


def prepare(X, y, batch_size=BATCH_SIZE):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    Xva, Xte, yva, yte   = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=SEED)
    fs = StandardScaler().fit(Xtr)
    ts = StandardScaler().fit(ytr.reshape(-1,1))
    t  = lambda a: torch.tensor(a, dtype=torch.float32).to(DEVICE)
    def ld(Xa, ya, sh):
        Xs = fs.transform(Xa)
        ys = ts.transform(ya.reshape(-1,1)).ravel()
        return DataLoader(TensorDataset(t(Xs), t(ys)), batch_size=batch_size, shuffle=sh)
    return (ld(Xtr,ytr,True), ld(Xva,yva,False),
            t(fs.transform(Xte)), yte, yva, ts)


def run_training(X, y, label, n_epochs, patience, batch_size=BATCH_SIZE,
                 lr=LR, wd=WEIGHT_DECAY, dropout=DROPOUT):
    print(f"\n{'='*60}")
    print(f"  {label}  (N={len(X)})")
    print(f"{'='*60}")

    train_l, val_l, Xte, yte, yva, ts = prepare(X, y, batch_size)
    ntr, nva = len(train_l.dataset), len(val_l.dataset)
    print(f"  Split: train={ntr}  val={nva}  test={len(yte)}")

    model = MLP(X.shape[1], dropout=dropout).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit  = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    hist = {k: [] for k in ["tr","va","va_r2","gnorm","gdelta"]}
    best_va, best_wts, pat, prev_g = float("inf"), None, 0, 0.0
    X_probe = next(iter(train_l))[0][:64]

    for ep in range(1, n_epochs+1):
        model.train()
        tr = 0.0
        for Xb, yb in train_l:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr += loss.item() * len(Xb)
        tr /= ntr

        gn = gradient_norm(model)
        gd = abs(gn - prev_g); prev_g = gn

        model.eval()
        with torch.no_grad():
            vp_s = torch.cat([model(Xb) for Xb,_ in val_l]).cpu().numpy()
            va   = crit(torch.tensor(vp_s),
                        torch.tensor(val_l.dataset.tensors[1].cpu().numpy())).item()
        vp   = ts.inverse_transform(vp_s.reshape(-1,1)).ravel()
        vr2  = r2_score(yva, vp)

        sched.step(va)
        for k,v in zip(["tr","va","va_r2","gnorm","gdelta"],[tr,va,vr2,gn,gd]):
            hist[k].append(v)

        if va < best_va:
            best_va, best_wts, pat = va, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= patience:
                print(f"  Early stop @ ep {ep}")
                break

        if ep % max(1, n_epochs//10) == 0 or ep == 1:
            dead = dead_pct(model, X_probe)
            dstr = "  ".join(f"{k}:{v*100:.0f}%" for k,v in dead.items())
            print(f"  ep{ep:04d} | tr={tr:.4f}  va={va:.4f}  R2={vr2:.4f}"
                  f"  | grad={gn:.3f}  D={gd:.3f}")
            print(f"          dead activations: {dstr}")

    model.load_state_dict(best_wts)
    model.eval()
    with torch.no_grad():
        tp_s = model(Xte).cpu().numpy()
    tp = ts.inverse_transform(tp_s.reshape(-1,1)).ravel()

    mse  = mean_squared_error(yte, tp)
    rmse = mse**0.5
    r2   = r2_score(yte, tp)

    print(f"\n  +-- TEST RESULTS {'-'*30}")
    print(f"  |   MSE  = {mse:.4f}")
    print(f"  |   RMSE = {rmse:.4f}  pIC50")
    print(f"  |   R2   = {r2:.4f}")
    print(f"  +{'-'*46}")

    return {"label":label,"N":len(X),"mse":mse,"rmse":rmse,"r2":r2,
            "hist":hist,"yte":yte,"ypred":tp,"epochs":len(hist["tr"])}


def plot_run(res):
    hist = res["hist"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(f"{res['label']}  (N={res['N']}, R2={res['r2']:.3f})", fontsize=12)

    axes[0].plot(hist["tr"], label="Train")
    axes[0].plot(hist["va"], label="Val")
    axes[0].set_title("Loss (MSE, scaled pIC50)")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(hist["va_r2"], color="green")
    axes[1].axhline(res["r2"], color="red", ls="--", label=f"Test R2={res['r2']:.3f}")
    axes[1].set_title("Val R2"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(hist["gnorm"],  label="||grad||")
    axes[2].plot(hist["gdelta"], label="delta||grad||", alpha=0.7)
    axes[2].set_title("Gradient norm per epoch")
    axes[2].set_xlabel("Epoch"); axes[2].legend()

    mn = min(res["yte"].min(), res["ypred"].min())
    mx = max(res["yte"].max(), res["ypred"].max())
    axes[3].scatter(res["yte"], res["ypred"], alpha=0.35, s=10)
    axes[3].plot([mn,mx],[mn,mx],"r--", label="Perfect")
    axes[3].set_xlabel("True pIC50"); axes[3].set_ylabel("Pred pIC50")
    axes[3].set_title("Actual vs Predicted"); axes[3].legend()

    plt.tight_layout()
    name = res["label"].replace(" ","_").replace("/","_")
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 3. TINY OVERFIT CHECK ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Tiny overfit check  (N=50, no regularisation)")
print("  Purpose: confirm the model CAN learn the data.")
print("  Expected: train_loss -> ~0, train >> val gap (memorisation)")
print("=" * 60)

X_tiny = X_all[:50]; y_tiny = y_all[:50]
res_tiny = run_training(X_tiny, y_tiny, "Tiny N=50 overfit check",
                         n_epochs=500, patience=500, batch_size=16,
                         lr=LR_TINY, wd=0.0, dropout=0.0)
plot_run(res_tiny)

# ── 4. FULL TRAINING ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 4: Full training  (N={len(X_all)}, TID={TOP_TID})")
print("  Purpose: real generalisation test with regularisation.")
print("  Expected: smooth convergence, R2 > 0, RMSE < 1.5 pIC50")
print("=" * 60)

res_full = run_training(X_all, y_all, f"Full N={len(X_all)} TID={TOP_TID}",
                         n_epochs=MAX_EPOCHS, patience=PATIENCE,
                         lr=LR, wd=WEIGHT_DECAY, dropout=DROPOUT)
plot_run(res_full)

# ── 5. RESIDUAL ANALYSIS ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Residual analysis")
print("=" * 60)

residuals = res_full["yte"] - res_full["ypred"]
print(f"  Residual mean  : {residuals.mean():.4f}  (should be ~0)")
print(f"  Residual std   : {residuals.std():.4f}")
print(f"  |residual| > 2 : {(np.abs(residuals) > 2).sum()} samples  "
      f"({(np.abs(residuals) > 2).mean()*100:.1f}%)")
print(f"  |residual| > 1 : {(np.abs(residuals) > 1).sum()} samples  "
      f"({(np.abs(residuals) > 1).mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(f"Residual analysis — TID={TOP_TID}", fontsize=12)

axes[0].hist(residuals, bins=40, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", ls="--")
axes[0].set_title("Residual distribution")
axes[0].set_xlabel("True - Predicted pIC50")

axes[1].scatter(res_full["ypred"], residuals, alpha=0.3, s=10)
axes[1].axhline(0, color="red", ls="--")
axes[1].set_xlabel("Predicted pIC50"); axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Predicted")

axes[2].scatter(res_full["yte"], res_full["ypred"], alpha=0.35, s=10)
mn = min(res_full["yte"].min(), res_full["ypred"].min())
mx = max(res_full["yte"].max(), res_full["ypred"].max())
axes[2].plot([mn,mx],[mn,mx],"r--", label="Perfect")
axes[2].set_xlabel("True pIC50"); axes[2].set_ylabel("Predicted pIC50")
axes[2].set_title(f"Actual vs Predicted  R2={res_full['r2']:.3f}")
axes[2].legend()

plt.tight_layout()
rpath = os.path.join(RESULTS_DIR, "residual_analysis.png")
plt.savefig(rpath, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {rpath}")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print(f"\n{'#'*60}")
print("  FINAL SUMMARY")
print(f"{'#'*60}")
print(f"  Protein TID : {TOP_TID}  ({N_TOTAL:,} total IC50 measurements)")
print(f"  Working set : {len(X_all):,} samples  (8 RDKit features)")
print()
print(f"  {'Run':<35} {'N':>6}  {'RMSE':>7}  {'R2':>7}")
print("  " + "-"*55)
for r in [res_tiny, res_full]:
    print(f"  {r['label']:<35} {r['N']:>6}  {r['rmse']:>7.4f}  {r['r2']:>7.4f}")

print(f"\n  Plots saved to: {RESULTS_DIR}/")
print()
print("  Interpretation:")
print(f"  - Tiny: train_loss={min(res_tiny['hist']['tr']):.3f} (should be <<1 = model CAN learn)")
print(f"  - Full: RMSE={res_full['rmse']:.3f} pIC50  R2={res_full['r2']:.3f}")
if res_full["r2"] >= 0.5:
    print("    -> Good result for 8 molecular descriptors")
elif res_full["r2"] >= 0.3:
    print("    -> Moderate — descriptors miss 3D/conformational information")
    print("    -> GNN should improve this significantly")
else:
    print("    -> Weak — this protein may be hard to predict from 2D descriptors alone")
    print("    -> Consider: more features, graph-based model (GNN)")
