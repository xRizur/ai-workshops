"""
train_models.py — MLP training + evaluation for pIC50 prediction.

Modes:
  --mode demo    : synthetic data (no ChEMBL needed, runs anywhere)
  --mode real    : load from silver parquets (needs chembl_work/ or GCS data)

Usage:
  python train_models.py --mode demo
  python train_models.py --mode real --silver_dir chembl_work/parquet_silver

Steps:
  1. TINY  (N=50)  -> model should overfit  -> confirms architecture works
  2. SMALL (N=200) -> overfit check
  3. FULL  (N<=10k) -> real generalisation test
"""

import argparse, os, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")      # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -- reproducibility ----------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- hyper-parameters ---------------------------------------------------------
BATCH_SIZE   = 64
LR           = 1e-3
MAX_EPOCHS   = 200
PATIENCE     = 20
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4
MAX_SAMPLES  = 10_000
FEATURES     = ["MW", "LogP", "HBD", "HBA", "TPSA", "QED", "RotBonds", "AromaticRings"]


# -- DATA ---------------------------------------------------------------------

def make_synthetic_data(n: int, noise: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic molecular-like data (no RDKit / ChEMBL needed).

    pIC50 is a noisy linear function of the descriptors:
      pIC50 ~ 7.0 + 0.3*LogP - 0.005*MW + 4*QED - 0.01*TPSA + N(0, noise)
    """
    rng = np.random.RandomState(SEED + n)  # different seed per size so sets differ
    MW            = rng.normal(350, 100, n).clip(100, 800)
    LogP          = rng.normal(2.5, 1.5,  n).clip(-3, 7)
    HBD           = rng.randint(0, 6,     n).astype(float)
    HBA           = rng.randint(0, 11,    n).astype(float)
    TPSA          = rng.normal(80, 40,    n).clip(0, 200)
    QED           = rng.beta(5, 2,        n)
    RotBonds      = rng.randint(0, 12,    n).astype(float)
    AromaticRings = rng.randint(0, 5,     n).astype(float)

    X = np.column_stack([MW, LogP, HBD, HBA, TPSA, QED, RotBonds, AromaticRings])

    y = (7.0
         + 0.30 * LogP
         - 0.005 * MW
         + 4.0  * QED
         - 0.01 * TPSA
         + 0.05 * AromaticRings
         + rng.normal(0, noise, n)).clip(3, 11).astype(np.float32)

    return X.astype(np.float32), y


def load_real_data(silver_dir: str, bronze_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ChEMBL silver parquets, filter to top Homo sapiens protein, compute RDKit features."""
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QED_module

    print("  Loading silver parquets...")
    df_act = pd.read_parquet(os.path.join(silver_dir, "activities_silver.parquet"))
    df_ass = pd.read_parquet(os.path.join(silver_dir, "assays_silver.parquet"))
    df_mol = pd.read_parquet(os.path.join(silver_dir, "molecules_silver.parquet"))
    df_tgt = pd.read_parquet(os.path.join(bronze_dir, "targets.parquet"),
                              columns=["tid", "pref_name", "organism"])
    df_tgt = df_tgt[df_tgt["organism"] == "Homo sapiens"]
    df_tgt["tid"] = df_tgt["tid"].astype(int)

    df = df_act.merge(df_ass, on="assay_id", how="inner")
    df = df.merge(df_tgt[["tid", "pref_name"]], on="tid", how="inner")
    df = df.merge(df_mol, on="molregno", how="inner")
    df = df.dropna(subset=["canonical_smiles", "pIC50"])

    top = df.groupby(["tid", "pref_name"]).size().sort_values(ascending=False)
    top_tid, top_name = top.index[0]
    print(f"  Protein: {top_name}  (tid={top_tid}, total={top.iloc[0]:,})")

    df = df[df["tid"] == top_tid].copy()
    df = df[(df["pIC50"] >= 2) & (df["pIC50"] <= 12)]
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=SEED)
    df = df.reset_index(drop=True)

    def rdkit_feats(smi):
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

    print(f"  Computing RDKit descriptors for {len(df):,} molecules...")
    feats = [rdkit_feats(s) for s in df["canonical_smiles"]]
    import pandas as pd
    feat_df = pd.DataFrame(feats, columns=FEATURES)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1).dropna(subset=FEATURES)

    X = df[FEATURES].values.astype(np.float32)
    y = df["pIC50"].values.astype(np.float32)
    print(f"  Final: {X.shape},  pIC50 [{y.min():.2f}, {y.max():.2f}]")
    return X, y


# -- MODEL ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        in_d = input_dim
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -- TRAINING UTILS ------------------------------------------------------------

def gradient_norm(model: nn.Module) -> float:
    total = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return total ** 0.5


def dead_neuron_pct(model: nn.Module, X_probe: torch.Tensor) -> dict:
    stats = {}
    hooks = []

    def make_hook(name):
        def _h(module, inp, out):
            with torch.no_grad():
                stats[name] = (out <= 0).float().mean().item()
        return _h

    for name, module in model.named_modules():
        if isinstance(module, nn.LeakyReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(X_probe)
    model.train()
    for h in hooks:
        h.remove()
    return stats


def prepare_loaders(X: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE):
    """Split, scale, convert to DataLoaders. Returns loaders + scalers + raw test arrays."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_va, X_te, y_va, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    fs = StandardScaler().fit(X_tr)
    ts = StandardScaler().fit(y_tr.reshape(-1, 1))

    def scale(Xa, ya=None):
        Xs = fs.transform(Xa)
        if ya is None:
            return Xs
        return Xs, ts.transform(ya.reshape(-1, 1)).ravel()

    X_tr_s, y_tr_s = scale(X_tr, y_tr)
    X_va_s, y_va_s = scale(X_va, y_va)
    X_te_s         = scale(X_te)

    def loader(Xa, ya, shuffle=False):
        t = lambda a: torch.tensor(a, dtype=torch.float32).to(DEVICE)
        return DataLoader(TensorDataset(t(Xa), t(ya)), batch_size=batch_size, shuffle=shuffle)

    return (
        loader(X_tr_s, y_tr_s, shuffle=True),
        loader(X_va_s, y_va_s),
        torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE),
        y_te,         # original scale for final R2
        y_va,         # original scale for tracking
        ts,
    )


def train_one_run(
    X: np.ndarray,
    y: np.ndarray,
    label: str,
    n_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    overfit_mode: bool = False,   # True: no dropout/wd -> model should memorise
) -> dict:
    """Full train / val / test loop. Returns metrics dict.

    overfit_mode=True: removes regularisation so the model CAN memorise a tiny
    dataset. Use only to confirm the architecture can learn at all (sanity check).
    """
    print(f"\n{'='*60}")
    print(f"  {label}  (N={len(X)}, device={DEVICE})")
    if overfit_mode:
        print("  [overfit mode: no dropout, no weight decay]")
    print(f"{'='*60}")

    actual_lr = 1e-3 if not overfit_mode else 5e-4   # lower lr for tiny set

    train_loader, val_loader, X_te, y_te, y_va, ts = prepare_loaders(X, y, batch_size)
    # sizes
    n_tr = len(train_loader.dataset)
    n_va = len(val_loader.dataset)
    print(f"  Train={n_tr}, Val={n_va}, Test={len(y_te)}")

    dropout = 0.0 if overfit_mode else DROPOUT
    wd      = 0.0 if overfit_mode else WEIGHT_DECAY

    model = MLP(input_dim=X.shape[1], dropout=dropout).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=actual_lr, weight_decay=wd)
    crit  = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    hist = {"train_loss": [], "val_loss": [], "val_r2": [],
            "grad_norm": [], "grad_delta": []}
    best_val  = float("inf")
    best_wts  = None
    pat_ctr   = 0
    prev_gnorm = 0.0

    # Probe for dead-neuron check
    X_probe = next(iter(train_loader))[0][:64]

    for ep in range(1, n_epochs + 1):
        # -- train --
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= n_tr

        # -- gradient stats --
        gnorm  = gradient_norm(model)
        gdelta = abs(gnorm - prev_gnorm)
        prev_gnorm = gnorm

        # -- val --
        model.eval()
        with torch.no_grad():
            va_preds_s = torch.cat([model(Xb) for Xb, _ in val_loader]).cpu().numpy()
            va_loss    = crit(
                torch.tensor(va_preds_s),
                torch.tensor(val_loader.dataset.tensors[1].cpu().numpy())
            ).item()
        va_preds = ts.inverse_transform(va_preds_s.reshape(-1, 1)).ravel()
        va_r2    = r2_score(y_va, va_preds)

        sched.step(va_loss)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_r2"].append(va_r2)
        hist["grad_norm"].append(gnorm)
        hist["grad_delta"].append(gdelta)

        if va_loss < best_val:
            best_val = va_loss
            best_wts = {k: v.clone() for k, v in model.state_dict().items()}
            pat_ctr  = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f"  Early stop @ epoch {ep}")
                break

        if ep % max(1, n_epochs // 10) == 0 or ep == 1:
            dead = dead_neuron_pct(model, X_probe)
            dead_str = "  ".join(f"{k}: {v*100:.0f}%" for k, v in dead.items())
            print(f"  ep{ep:04d} | tr={tr_loss:.4f}  va={va_loss:.4f}  R2={va_r2:.4f} "
                  f"| grad={gnorm:.4f}  delta={gdelta:.4f}")
            print(f"          dead -> {dead_str}")

    model.load_state_dict(best_wts)
    model.eval()
    with torch.no_grad():
        te_preds_s = model(X_te).cpu().numpy()
    te_preds = ts.inverse_transform(te_preds_s.reshape(-1, 1)).ravel()

    mse  = mean_squared_error(y_te, te_preds)
    rmse = mse ** 0.5
    r2   = r2_score(y_te, te_preds)

    print(f"\n  +- TEST RESULTS {'-'*30}")
    print(f"  |  MSE  = {mse:.4f}")
    print(f"  |  RMSE = {rmse:.4f}  (pIC50 units)")
    print(f"  |  R2   = {r2:.4f}")
    print(f"  +{'-'*45}")

    return {
        "label":    label,
        "N":        len(X),
        "mse":      mse,
        "rmse":     rmse,
        "r2":       r2,
        "history":  hist,
        "y_te":     y_te,
        "y_pred":   te_preds,
        "epochs":   len(hist["train_loss"]),
    }


# -- PLOTTING ------------------------------------------------------------------

def plot_run(res: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    label = res["label"].replace(" ", "_")
    hist  = res["history"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(f"{res['label']}  (N={res['N']}, R2={res['r2']:.3f})", fontsize=12)

    # Loss
    ax = axes[0]
    ax.plot(hist["train_loss"], label="Train")
    ax.plot(hist["val_loss"],   label="Val")
    ax.set_title("Loss (MSE, scaled)"); ax.set_xlabel("Epoch"); ax.legend()

    # R2
    ax = axes[1]
    ax.plot(hist["val_r2"], color="green")
    ax.axhline(res["r2"], color="red", ls="--", label=f"Test R2={res['r2']:.3f}")
    ax.set_title("Val R2"); ax.set_xlabel("Epoch"); ax.legend()

    # Gradient norm
    ax = axes[2]
    ax.plot(hist["grad_norm"],  label="||grad||")
    ax.plot(hist["grad_delta"], label="delta||grad||", alpha=0.7)
    ax.set_title("Gradient norm"); ax.set_xlabel("Epoch"); ax.legend()

    # Actual vs Predicted
    ax = axes[3]
    ax.scatter(res["y_te"], res["y_pred"], alpha=0.4, s=12)
    mn = min(res["y_te"].min(), res["y_pred"].min())
    mx = max(res["y_te"].max(), res["y_pred"].max())
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect")
    ax.set_xlabel("True pIC50"); ax.set_ylabel("Predicted pIC50")
    ax.set_title("Actual vs Predicted"); ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"{label}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# -- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="demo", choices=["demo", "real"])
    parser.add_argument("--silver_dir", default=os.path.join("chembl_work", "parquet_silver"))
    parser.add_argument("--bronze_dir", default=os.path.join("chembl_work", "parquet_bronze"))
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  pIC50 MLP Training   mode={args.mode}   device={DEVICE}")
    print(f"{'#'*60}\n")

    # -- load data --
    if args.mode == "demo":
        print("[DEMO] Generating synthetic molecular data...")
        X_big, y_big = make_synthetic_data(MAX_SAMPLES)
        print(f"  Synthetic dataset: {X_big.shape},  pIC50 [{y_big.min():.2f}, {y_big.max():.2f}]")
    else:
        print("[REAL] Loading ChEMBL silver parquets...")
        X_big, y_big = load_real_data(args.silver_dir, args.bronze_dir)

    results = []

    # -- Step 1: TINY — overfit check ------------------------------------------
    print("\n" + "-"*60)
    print("STEP 1 / 3 - Tiny dataset (N=50): overfit check")
    print("Expected: train_loss -> ~0.0, train R2 -> ~1.0  (model memorises data)")
    print("-"*60)
    X_tiny = X_big[:50]
    y_tiny = y_big[:50]
    res_tiny = train_one_run(X_tiny, y_tiny, "Tiny N=50 (overfit check)",
                              n_epochs=500, patience=500, batch_size=16,
                              overfit_mode=True)
    results.append(res_tiny)
    plot_run(res_tiny)

    # -- Step 2: SMALL — early overfitting -------------------------------------
    print("\n" + "-"*60)
    print("STEP 2 / 3 - Small dataset (N=200): overfit check with val gap")
    print("Expected: train << val, gap visible")
    print("-"*60)
    X_small = X_big[:200]
    y_small = y_big[:200]
    res_small = train_one_run(X_small, y_small, "Small N=200 (overfit check)",
                               n_epochs=300, patience=300, batch_size=32,
                               overfit_mode=True)
    results.append(res_small)
    plot_run(res_small)

    # -- Step 3: FULL — generalisation -----------------------------------------
    print("\n" + "-"*60)
    print("STEP 3 / 3 - Full dataset (N<=10k): generalisation")
    print("Expected: train ~ val, R2 > 0 (higher with real data)")
    print("-"*60)
    res_full = train_one_run(X_big, y_big, f"Full N={len(X_big)} (generalisation)",
                              n_epochs=MAX_EPOCHS, patience=PATIENCE)
    results.append(res_full)
    plot_run(res_full)

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'#'*60}")
    print("  SUMMARY")
    print(f"{'#'*60}")
    print(f"  {'Run':<35} {'N':>6} {'Epochs':>7} {'RMSE':>8} {'R2':>8}")
    print("  " + "-"*60)
    for r in results:
        print(f"  {r['label']:<35} {r['N']:>6} {r['epochs']:>7} {r['rmse']:>8.4f} {r['r2']:>8.4f}")

    print("\nPlots saved to results/")


if __name__ == "__main__":
    main()
