# ════════════════════════════════════════════════════════
# train.py — GITHUB ACTIONS VERSION (Fully Corrected)
# ════════════════════════════════════════════════════════

import gc, os, sys, psutil, shutil, json
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path setup ────────────────────────────────────────
sys.path.insert(0, os.getcwd())
from trainer import (
    SystemConfig, DataIngestion, StateBuilder,
    PortfolioManager, RiskManager, RewardFunction,
    WalkForward, PerfMonitor, PrioritizedReplayBuffer,
    EnsembleDoubleQAgent, build_scaler_params,
    SumTree, train_and_export
)

# Import ONNX tools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# ── GPU Detection ─────────────────────────────────────
def get_xgb_gpu_params():
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    try:
        m = XGBClassifier(
            n_estimators=5,
            device="cuda",
            tree_method="hist",
            verbosity=0)
        m.fit(X, y)
        print("✔ Using device=cuda")
        return {"device": "cuda", "tree_method": "hist"}
    except:
        pass
    try:
        m = XGBClassifier(
            n_estimators=5,
            tree_method="gpu_hist",
            verbosity=0)
        m.fit(X, y)
        print("✔ Using tree_method=gpu_hist")
        return {"device": None, "tree_method": "gpu_hist"}
    except:
        pass
    print("⚠ Using CPU")
    return {"device": None, "tree_method": "hist"}

GPU_PARAMS = get_xgb_gpu_params()
print(f"GPU params: {GPU_PARAMS}\n")


def ram():
    mb = psutil.Process(
        os.getpid()).memory_info().rss / 1024 / 1024
    return f"{mb:.0f} MB"


# ════════════════════════════════════════════════════════
# OVERRIDE 1: FeatureSelector
# Simplified + memory efficient for GitHub runner
# ════════════════════════════════════════════════════════
class FeatureSelector:
    BASE_EXCL = {
        "timestamp", "open", "high", "low",
        "close", "volume", "returns", "log_returns"
    }

    def __init__(self, cfg):
        self.cfg = cfg

    def select(self, data, target, top_k=30):
        fcols = [c for c in data.columns
                 if c not in self.BASE_EXCL]
        X = data[fcols].copy()
        y = target.copy()
        idx = X.index.intersection(y.index)
        X, y = X.loc[idx], y.loc[idx]

        # Correlation filter
        corr  = X.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop  = [c for c in upper.columns
                 if (upper[c] >
                     self.cfg.CORRELATION_THRESHOLD).any()]
        X.drop(columns=drop, inplace=True)
        del corr, upper
        gc.collect()
        print(f"  Corr filter: removed {len(drop)} features")
        print(f"  RAM: {ram()}")

        rem  = list(X.columns)
        n    = min(8000, len(X))
        sidx = np.random.choice(len(X), n, replace=False)
        Xa   = X.iloc[sidx].values.astype(np.float32)
        ya   = y.iloc[sidx].values
        del X
        gc.collect()

        # XGBoost importance with GPU
        params = {
            "n_estimators":     200,
            "max_depth":        5,
            "verbosity":        0,
            "random_state":     42,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            **GPU_PARAMS
        }
        params = {k: v for k, v in params.items()
                  if v is not None}

        tmp = XGBClassifier(**params)
        tmp.fit(Xa, ya)
        xgb_imp = np.array(tmp.feature_importances_)
        del tmp
        gc.collect()
        print(f"  XGB importance done | RAM: {ram()}")

        # Mutual information
        try:
            mi = mutual_info_classif(
                Xa, ya,
                random_state=42,
                n_neighbors=5)
        except:
            mi = xgb_imp.copy()
        del Xa, ya
        gc.collect()

        xr = pd.Series(
            xgb_imp, index=rem).rank(ascending=False)
        mr = pd.Series(
            mi, index=rem).rank(ascending=False)
        combined = (xr + mr) / 2
        combined.sort_values(inplace=True)
        sel = combined.head(
            min(top_k, len(combined))).index.tolist()

        print(f"  Selected {len(sel)} features")
        print(f"  Top 10: {sel[:10]}")
        return sel

    def adversarial_check(self, X_tr, X_te):
        # Skip on GitHub runner — saves ~5 min
        print("  Skipping adversarial check (CI mode)")
        return [], False


# ════════════════════════════════════════════════════════
# OVERRIDE 2: FeatureEngine
# Full feature set matching v4.03 EA ComputeRaw()
# ════════════════════════════════════════════════════════
class FeatureEngine:
    EPS = 1e-10

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _rsi(s, p):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / (l + FeatureEngine.EPS))

    @staticmethod
    def _macd(s, f=12, sl=26, sg=9):
        ef  = s.ewm(span=f,  adjust=False).mean()
        es  = s.ewm(span=sl, adjust=False).mean()
        ml  = ef - es
        sig = ml.ewm(span=sg, adjust=False).mean()
        return ml, sig, ml - sig

    @staticmethod
    def _atr(h, l, c, p=14):
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def _stoch(h, l, c, kp=14, dp=3):
        ll = l.rolling(kp).min()
        hh = h.rolling(kp).max()
        k  = 100 * (c - ll) / (hh - ll + FeatureEngine.EPS)
        return k, k.rolling(dp).mean()

    @staticmethod
    def _obv(c, v):
        return (np.sign(c.diff()).fillna(0) * v).cumsum()

    @staticmethod
    def _williams_r(h, l, c, p=14):
        hh = h.rolling(p).max()
        ll = l.rolling(p).min()
        return -100 * (hh - c) / (hh - ll + FeatureEngine.EPS)

    @staticmethod
    def _cci(h, l, c, p=20):
        tp = (h + l + c) / 3
        ma = tp.rolling(p).mean()
        md = tp.rolling(p).apply(
            lambda x: np.mean(np.abs(x - x.mean())),
            raw=True)
        return (tp - ma) / (0.015 * md + FeatureEngine.EPS)

    def build(self, data):
        print(f"  RAM at feature build: {ram()}")
        df  = data.copy()
        eps = self.EPS
        c   = df["close"]
        h   = df["high"]
        l   = df["low"]
        v   = df["volume"]
        ret = df["returns"]

        # ── Price / MA ────────────────────────────────
        for p in [10, 20, 50, 100]:
            sma = c.rolling(p).mean()
            df[f"close_sma_{p}"] = c / (sma + eps) - 1

        for p in [10, 20, 50]:
            ema = c.ewm(span=p, adjust=False).mean()
            df[f"close_ema_{p}"] = c / (ema + eps) - 1

        # EMA crossovers
        ema10 = c.ewm(span=10, adjust=False).mean()
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        df["ema10_20_cross"] = ema10 / (ema20 + eps) - 1
        df["ema20_50_cross"] = ema20 / (ema50 + eps) - 1

        # ── Momentum ──────────────────────────────────
        for p in [7, 14, 21]:
            df[f"rsi_{p}"] = self._rsi(c, p)

        # RSI derived signals
        df["rsi_14_slope"]   = df["rsi_14"].diff(3)
        df["price_slope"]    = c.pct_change(3)
        df["rsi_divergence"] = (
            df["rsi_14_slope"] -
            df["price_slope"] * 100)

        df["macd"], df["macd_sig"], df["macd_hist"] = \
            self._macd(c)
        df["macd_cross"] = (
            np.sign(df["macd_hist"]) *
            np.sign(df["macd_hist"].shift(1)))

        df["williams_r"] = self._williams_r(h, l, c)
        df["cci"]        = self._cci(h, l, c)
        df["stoch_k"], df["stoch_d"] = self._stoch(h, l, c)
        df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

        # Returns
        for p in [1, 5, 10, 20]:
            df[f"ret_{p}"] = c.pct_change(p)

        # ── Volatility ────────────────────────────────
        for w in [10, 20, 60]:
            df[f"vol_{w}"] = ret.rolling(w).std()
        df["vol_ratio"] = (
            df["vol_20"] / (df["vol_60"] + eps))

        df["atr_14"]    = self._atr(h, l, c, 14)
        df["atr_ratio"] = df["atr_14"] / (c + eps)
        df["atr_pct"]   = df["atr_14"].rolling(100).rank(
            pct=True)

        bm = c.rolling(20).mean()
        bs = c.rolling(20).std()
        df["bb_pos"]     = (c - (bm - 2*bs)) / (4*bs + eps)
        df["bb_width"]   = 4 * bs / (bm + eps)
        df["bb_squeeze"] = (
            df["bb_width"] <
            df["bb_width"].rolling(50).mean()
        ).astype(float)

        # ── Volume ────────────────────────────────────
        vsma = v.rolling(20).mean()
        df["vol_ratio_20"] = v / (vsma + eps)
        obv = self._obv(c, v)
        df["obv_norm"]  = obv / (obv.abs() + 1)
        df["obv_slope"] = obv.pct_change(5)

        # ── Microstructure ────────────────────────────
        spread = (h - l).clip(lower=eps)
        df["spread_pct"] = spread / (c + eps)
        df["body"] = (
            (c - df["open"]).abs() / (spread + eps))

        top = pd.concat(
            [df["open"], c], axis=1).max(axis=1)
        bot = pd.concat(
            [df["open"], c], axis=1).min(axis=1)
        df["upper_wick"] = (h - top) / (spread + eps)
        df["lower_wick"] = (bot - l) / (spread + eps)

        # Candle patterns
        df["bullish_engulf"] = (
            (df["open"] > c.shift(1)) &
            (c > df["open"].shift(1)) &
            (df["body"] > 0.6)
        ).astype(float)
        df["bearish_engulf"] = (
            (df["open"] < c.shift(1)) &
            (c < df["open"].shift(1)) &
            (df["body"] > 0.6)
        ).astype(float)

        # ── Statistical ───────────────────────────────
        rm = c.rolling(20).mean()
        rs = c.rolling(20).std()
        df["zscore_20"]     = (c - rm) / (rs + eps)
        df["zscore_extreme"] = (
            df["zscore_20"].abs() > 2.0
        ).astype(float)

        for w in [20, 50]:
            df[f"skew_{w}"] = ret.rolling(w).skew()
            df[f"kurt_{w}"] = ret.rolling(w).kurt()

        # ── Support / Resistance ──────────────────────
        for p in [20, 50]:
            df[f"high_{p}"]      = h.rolling(p).max()
            df[f"low_{p}"]       = l.rolling(p).min()
            df[f"dist_high_{p}"] = (
                c - df[f"high_{p}"]) / (c + eps)
            df[f"dist_low_{p}"]  = (
                c - df[f"low_{p}"]) / (c + eps)

        # ── ADX ───────────────────────────────────────
        dm_pos = (h - h.shift(1)).clip(lower=0)
        dm_neg = (l.shift(1) - l).clip(lower=0)
        tr14   = df["atr_14"]
        df["adx_pos"] = (
            dm_pos.rolling(14).mean() / (tr14 + eps))
        df["adx_neg"] = (
            dm_neg.rolling(14).mean() / (tr14 + eps))
        df["adx"] = (
            df["adx_pos"] - df["adx_neg"]).abs()

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        gc.collect()

        n_f = len([
            col for col in df.columns
            if col not in {
                "timestamp", "open", "high", "low",
                "close", "volume", "returns",
                "log_returns"}])
        print(f"  Features: {n_f} | RAM: {ram()}")
        return df


# ════════════════════════════════════════════════════════
# OVERRIDE 3: HPOptimizer
# Injects GPU params into Optuna trials
# ════════════════════════════════════════════════════════
class HPOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def optimize(self, X, y):
        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int(
                    "n_est", 200, 800),
                "max_depth":        trial.suggest_int(
                    "depth", 3, 8),
                "learning_rate":    trial.suggest_float(
                    "lr", 0.005, 0.2, log=True),
                "min_child_weight": trial.suggest_int(
                    "mcw", 1, 15),
                "subsample":        trial.suggest_float(
                    "ss", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "cs", 0.4, 1.0),
                "reg_alpha":        trial.suggest_float(
                    "ra", 1e-4, 10, log=True),
                "reg_lambda":       trial.suggest_float(
                    "rl", 1e-4, 10, log=True),
                "gamma":            trial.suggest_float(
                    "gm", 0, 5),
                "random_state":     42,
                "verbosity":        0,
                **GPU_PARAMS,
            }
            params = {k: v for k, v in params.items()
                      if v is not None}

            folds  = self._cv_folds(
                len(X),
                self.cfg.OPTUNA_CV_SPLITS,
                self.cfg.EMBARGO_PCT)
            scores = []

            for fi, (tri, tei) in enumerate(folds):
                m = XGBClassifier(**params)
                try:
                    m.fit(X[tri], y[tri],
                          eval_set=[(X[tei], y[tei])],
                          verbose=False)
                    auc = roc_auc_score(
                        y[tei],
                        m.predict_proba(X[tei])[:, 1])
                    scores.append(auc)
                except:
                    scores.append(0.5)
                trial.report(np.mean(scores), fi)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner())
        study.optimize(
            objective,
            n_trials=self.cfg.OPTUNA_N_TRIALS,
            n_jobs=1,
            show_progress_bar=True)
        print(f"Best AUC: {study.best_value:.4f}")
        return study.best_params

    @staticmethod
    def _cv_folds(n, ns, ep):
        emb  = int(n * ep)
        fsz  = n // ns
        folds = []
        for i in range(ns):
            ts = i * fsz
            te = (i+1) * fsz if i < ns-1 else n
            tr = np.concatenate([
                np.arange(0, max(0, ts-emb)),
                np.arange(min(n, te+emb), n)
            ]).astype(int)
            tv = np.arange(ts, te).astype(int)
            if len(tr) >= 10 and len(tv) >= 5:
                folds.append((tr, tv))
        return folds


# ════════════════════════════════════════════════════════
# OVERRIDE 4: EnsembleDoubleQAgent
# Injects GPU params into model creation
# Cannot be deleted — GPU injection is required here
# Safe pattern: import base class under alias first
# ════════════════════════════════════════════════════════
_BaseAgent = EnsembleDoubleQAgent

class EnsembleDoubleQAgent(_BaseAgent):
    """
    Extends base agent with GPU-accelerated model creation.
    Uses alias _BaseAgent to avoid self-inheritance loop.
    """
    def _make(self, i: int) -> XGBRegressor:
        p = {
            **self.cfg.XGB_BASE,
            **self.cfg.ENSEMBLE_CONFIGS[i],
            **GPU_PARAMS
        }
        p = {k: v for k, v in p.items()
             if v is not None}
        return XGBRegressor(**p)


# ════════════════════════════════════════════════════════
# OVERRIDE 5: ONNXExporter
# Critical: Static [1, n] shape — required by MT5
# ════════════════════════════════════════════════════════
_BaseExporter = __import__(
    'trainer',
    fromlist=['ONNXExporter']).ONNXExporter

class ONNXExporter(_BaseExporter):

    def _export_one(self, mdl, net_name: str,
                    a: int, mi: int,
                    in_dim: int) -> str:
        import onnx
        from onnx import shape_inference

        # Static [1, n] — NOT [None, n]
        init_types = [
            ("input",
             FloatTensorType([1, in_dim]))
        ]
        try:
            onnx_model = convert_sklearn(
                mdl,
                initial_types=init_types,
                target_opset=15,
                options={type(mdl): {"nocopy": True}})
        except:
            onnx_model = convert_sklearn(
                mdl,
                initial_types=init_types,
                target_opset=15)

        # Force static dims on graph nodes
        graph = onnx_model.graph

        inp = graph.input[0]
        inp.type.tensor_type.shape.ClearField("dim")
        d0 = inp.type.tensor_type.shape.dim.add()
        d0.dim_value = 1
        d1 = inp.type.tensor_type.shape.dim.add()
        d1.dim_value = in_dim

        out = graph.output[0]
        out.type.tensor_type.shape.ClearField("dim")
        od0 = out.type.tensor_type.shape.dim.add()
        od0.dim_value = 1
        od1 = out.type.tensor_type.shape.dim.add()
        od1.dim_value = 1

        onnx_model = shape_inference.infer_shapes(
            onnx_model)

        fname = f"{net_name}_a{a}_m{mi}.onnx"
        fpath = os.path.join(self.out, fname)
        with open(fpath, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Verify
        sess    = ort.InferenceSession(fpath)
        dummy   = np.zeros((1, in_dim), dtype=np.float32)
        in_name = sess.get_inputs()[0].name
        result  = sess.run(None, {in_name: dummy})
        in_sh   = sess.get_inputs()[0].shape
        out_sh  = sess.get_outputs()[0].shape
        val     = float(result[0].flatten()[0])
        print(f"  ✔ {fname}: "
              f"in{in_sh} → out{out_sh} = {val:.6f}")
        return fpath

    def export_agent(self, agent, selected,
                     scaler_params, state_dim, report):
        os.makedirs(self.out, exist_ok=True)
        exported = []

        for net_name, ens in [("q1", agent.q1),
                               ("q2", agent.q2)]:
            for a in range(agent.n_a):
                for mi in range(agent.n_m):
                    m      = ens[a][mi]
                    mdl    = m["model"]
                    fi     = m["fi"]
                    in_dim = (len(fi) if fi is not None
                              else state_dim)
                    try:
                        self._export_one(
                            mdl, net_name,
                            a, mi, in_dim)
                        exported.append(
                            f"{net_name}_a{a}_m{mi}.onnx")
                    except Exception as e:
                        print(f"  ✘ {net_name} "
                              f"a={a} m={mi}: {e}")
                        mdl.save_model(
                            os.path.join(
                                self.out,
                                f"{net_name}_a{a}"
                                f"_m{mi}.json"))

        # Save all metadata
        fi_map = {}
        for a in range(agent.n_a):
            for mi in range(agent.n_m):
                fi  = agent.q1[a][mi]["fi"]
                key = f"a{a}_m{mi}"
                fi_map[key] = (
                    fi.tolist() if fi is not None
                    else list(range(state_dim)))

        with open(os.path.join(
                self.out,
                "feature_indices.json"), "w") as f:
            json.dump(fi_map, f, indent=2)

        with open(os.path.join(
                self.out,
                "selected_features.json"), "w") as f:
            json.dump(selected, f, indent=2)

        with open(os.path.join(
                self.out,
                "scaler_params.json"), "w") as f:
            json.dump(scaler_params, f, indent=2)

        with open(os.path.join(
                self.out,
                "ensemble_weights.json"), "w") as f:
            json.dump(
                {"weights": agent.ew.tolist()},
                f, indent=2)

        with open(os.path.join(
                self.out, "config.json"), "w") as f:
            json.dump({
                "N_ACTIONS":           agent.n_a,
                "N_ENSEMBLE_MODELS":   agent.n_m,
                "GAMMA":               agent.gamma,
                "STATE_DIM":           state_dim,
                "N_MARKET_FEATURES":   len(selected),
                "N_PORTFOLIO_FEATURES":9,
                "N_REGIME_FEATURES":   8,
                "SELECTED_FEATURES":   selected,
                "CONFIDENCE_THRESHOLD":agent.conf_thr,
            }, f, indent=2)

        with open(os.path.join(
                self.out,
                "training_report.json"), "w") as f:
            json.dump(
                {k: float(v)
                 if isinstance(v, (float, np.floating))
                 else v
                 for k, v in report.items()},
                f, indent=2)

        print(f"\nExported {len(exported)} ONNX models")
        print("\n=== SCALER SUMMARY (first 10) ===")
        for i, (feat, params) in enumerate(
                scaler_params.items()):
            if i >= 10:
                break
            print(f"  {feat}: "
                  f"mean={params['mean']:.6f} "
                  f"std={params['std']:.6f}")
        return exported


# ════════════════════════════════════════════════════════
# MAIN — Config & Execution
# ════════════════════════════════════════════════════════
gc.collect()
print(f"Starting GitHub Actions Job | RAM: {ram()}\n")

# Cache directory for workflow relay
cache_dir = os.path.join(os.getcwd(), "training_cache")
os.makedirs(cache_dir, exist_ok=True)
print(f"Cache dir: {cache_dir}")

cfg = SystemConfig(
    HISTORICAL_BARS        = 50000,
    OPTUNA_N_TRIALS        = 40,
    OPTUNA_CV_SPLITS       = 5,
    TRAIN_WINDOW           = 8000,
    RETRAIN_INTERVAL       = 800,
    MIN_TRAIN_SAMPLES      = 1000,
    BATCH_SIZE             = 256,
    MIN_BUFFER_SIZE        = 1000,
    REPLAY_BUFFER_CAPACITY = 50000,
    N_ENSEMBLE_MODELS      = 3,
    MAX_FEATURES_SELECTED  = 30,
    CORRELATION_THRESHOLD  = 0.90,
    NOISE_INJECTION_LEVEL  = 0.005,
    FEATURE_DROPOUT_RATE   = 0.05,
    OUTPUT_DIR             = cache_dir,
    TRADING_PAIR           = "EURUSD",

    # Updated XGB base — tree_method
    # overridden by GPU_PARAMS at runtime
    XGB_BASE = {
        "n_estimators":  800,
        "learning_rate": 0.03,
        "reg_alpha":     0.5,
        "reg_lambda":    1.0,
        "tree_method":   "hist",
        "verbosity":     0,
        "random_state":  42,
    },

    STOP_LOSS_PCT      = 0.015,
    TAKE_PROFIT_PCT    = 0.045,
    MAX_DRAWDOWN_LIMIT = 0.15,
    INITIAL_CAPITAL    = 100000.0,
    MAX_POSITION_SIZE  = 0.15,
)

# Run full pipeline
metrics = train_and_export(
    cfg      = cfg,
    source   = "csv",
    filepath = "EURUSD_H1.csv"
)

# Bundle for workflow upload
archive_path = os.path.join(cache_dir, "xgb_rl_artifacts")
shutil.make_archive(
    archive_path,
    "zip",
    cache_dir)

print(f"\nArchive created: {archive_path}.zip")

files = os.listdir(cache_dir)
print(f"\nTotal files in cache: {len(files)}")
for f in sorted(files):
    print(f"  {f}")
print("\nDone! Workflow handoff ready.")
