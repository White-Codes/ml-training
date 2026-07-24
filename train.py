# ════════════════════════════════════════════════════════
# train.py — GITHUB ACTIONS VERSION (Session Resume Fixed)
# ════════════════════════════════════════════════════════

import gc, os, sys, psutil, shutil, json, joblib
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.getcwd())
from trainer import (
    SystemConfig, DataIngestion, StateBuilder,
    PortfolioManager, RiskManager, RewardFunction,
    WalkForward, PerfMonitor, PrioritizedReplayBuffer,
    EnsembleDoubleQAgent, build_scaler_params,
    SumTree, train_and_export
)

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
# SESSION STATE — Save & Load Between Runs
# ════════════════════════════════════════════════════════

def save_session_state(agent, port, reward, wf,
                       selected, scaler_params,
                       start_idx, all_results,
                       output_dir):
    """Save training state for next session to resume."""
    state_path = os.path.join(
        output_dir, "session_state.joblib")
    try:
        state = {
            "epsilon":      agent.epsilon,
            "step":         agent.step,
            "fitted":       agent.fitted,
            "ew":           agent.ew,
            "buf_size":     agent.buf.size,
            "equity":       port.equity,
            "cash":         port.cash,
            "pos":          port.pos,
            "trades":       port.trades,
            "wins":         port.wins,
            "gross_p":      port.gross_p,
            "gross_l":      port.gross_l,
            "reward_hist":  reward.hist[-200:],
            "wf_last":      wf.last,
            "wf_cnt":       wf.cnt,
            "start_idx":    start_idx,
            "selected":     selected,
            "scaler_params":scaler_params,
            "n_results":    len(all_results),
        }
        joblib.dump(state, state_path)
        print(f"[RELAY] Session state saved → "
              f"{state_path}")
    except Exception as e:
        print(f"[WARN] Could not save state: {e}")


def load_session_state(output_dir):
    """Load previous session state if it exists."""
    state_path = os.path.join(
        output_dir, "session_state.joblib")
    if not os.path.exists(state_path):
        print("[RELAY] No previous state found "
              "— starting fresh")
        return None
    try:
        state = joblib.load(state_path)
        print(f"[RELAY] Loaded session state: "
              f"step={state.get('step',0)} "
              f"epsilon={state.get('epsilon',1.0):.4f} "
              f"trades={state.get('trades',0)}")
        return state
    except Exception as e:
        print(f"[WARN] Could not load state: {e}")
        return None


def save_agent_models(agent, output_dir):
    """Save XGBoost model files for resume."""
    models_dir = os.path.join(output_dir, "agent_models")
    os.makedirs(models_dir, exist_ok=True)
    try:
        for a in range(agent.n_a):
            for mi in range(agent.n_m):
                # Q1
                fi1 = agent.q1[a][mi]["fi"]
                mdl1 = agent.q1[a][mi]["model"]
                path1 = os.path.join(
                    models_dir,
                    f"q1_a{a}_m{mi}.json")
                mdl1.save_model(path1)
                # Q2
                mdl2 = agent.q2[a][mi]["model"]
                path2 = os.path.join(
                    models_dir,
                    f"q2_a{a}_m{mi}.json")
                mdl2.save_model(path2)
                # Feature indices
                if fi1 is not None:
                    fi_path = os.path.join(
                        models_dir,
                        f"fi_a{a}_m{mi}.npy")
                    np.save(fi_path, fi1)
        print(f"[RELAY] Agent models saved to "
              f"{models_dir}/")
    except Exception as e:
        print(f"[WARN] Model save failed: {e}")


def load_agent_models(agent, output_dir):
    """Load XGBoost models from previous session."""
    models_dir = os.path.join(output_dir, "agent_models")
    if not os.path.exists(models_dir):
        return False
    try:
        loaded = 0
        for a in range(agent.n_a):
            for mi in range(agent.n_m):
                path1 = os.path.join(
                    models_dir,
                    f"q1_a{a}_m{mi}.json")
                path2 = os.path.join(
                    models_dir,
                    f"q2_a{a}_m{mi}.json")
                fi_path = os.path.join(
                    models_dir,
                    f"fi_a{a}_m{mi}.npy")

                if os.path.exists(path1):
                    agent.q1[a][mi]["model"].load_model(
                        path1)
                    loaded += 1
                if os.path.exists(path2):
                    agent.q2[a][mi]["model"].load_model(
                        path2)
                    loaded += 1
                if os.path.exists(fi_path):
                    fi = np.load(fi_path)
                    agent.q1[a][mi]["fi"] = fi
                    agent.q2[a][mi]["fi"] = fi

        if loaded > 0:
            agent.fitted = True
            print(f"[RELAY] Loaded {loaded} "
                  f"model files from {models_dir}/")
            return True
        return False
    except Exception as e:
        print(f"[WARN] Model load failed: {e}")
        return False


# ════════════════════════════════════════════════════════
# OVERRIDE 1: FeatureSelector
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

        corr  = X.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape),
                    k=1).astype(bool))
        drop  = [c for c in upper.columns
                 if (upper[c] >
                     self.cfg.CORRELATION_THRESHOLD
                     ).any()]
        X.drop(columns=drop, inplace=True)
        del corr, upper
        gc.collect()
        print(f"  Corr filter: -{len(drop)} features")

        rem  = list(X.columns)
        n    = min(8000, len(X))
        sidx = np.random.choice(len(X), n, replace=False)
        Xa   = X.iloc[sidx].values.astype(np.float32)
        ya   = y.iloc[sidx].values
        del X
        gc.collect()

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
            min(top_k,
                len(combined))).index.tolist()
        print(f"  Selected {len(sel)} features")
        print(f"  Top 10: {sel[:10]}")
        return sel

    def adversarial_check(self, X_tr, X_te):
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

        # Price / MA
        for p in [10, 20, 50, 100]:
            sma = c.rolling(p).mean()
            df[f"close_sma_{p}"] = c / (sma + eps) - 1
        for p in [10, 20, 50]:
            ema = c.ewm(span=p, adjust=False).mean()
            df[f"close_ema_{p}"] = c / (ema + eps) - 1

        ema10 = c.ewm(span=10, adjust=False).mean()
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        df["ema10_20_cross"] = ema10 / (ema20 + eps) - 1
        df["ema20_50_cross"] = ema20 / (ema50 + eps) - 1

        # Momentum
        for p in [7, 14, 21]:
            df[f"rsi_{p}"] = self._rsi(c, p)
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

        for p in [1, 5, 10, 20]:
            df[f"ret_{p}"] = c.pct_change(p)

        # Volatility
        for w in [10, 20, 60]:
            df[f"vol_{w}"] = ret.rolling(w).std()
        df["vol_ratio"] = df["vol_20"] / (df["vol_60"] + eps)
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

        # Volume
        vsma = v.rolling(20).mean()
        df["vol_ratio_20"] = v / (vsma + eps)
        obv = self._obv(c, v)
        df["obv_norm"]  = obv / (obv.abs() + 1)
        df["obv_slope"] = obv.pct_change(5)

        # Microstructure
        spread = (h - l).clip(lower=eps)
        df["spread_pct"] = spread / (c + eps)
        df["body"] = (c - df["open"]).abs() / (spread + eps)
        top = pd.concat(
            [df["open"], c], axis=1).max(axis=1)
        bot = pd.concat(
            [df["open"], c], axis=1).min(axis=1)
        df["upper_wick"] = (h - top) / (spread + eps)
        df["lower_wick"] = (bot - l) / (spread + eps)

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

        # Statistical
        rm = c.rolling(20).mean()
        rs = c.rolling(20).std()
        df["zscore_20"]      = (c - rm) / (rs + eps)
        df["zscore_extreme"] = (
            df["zscore_20"].abs() > 2.0
        ).astype(float)
        for w in [20, 50]:
            df[f"skew_{w}"] = ret.rolling(w).skew()
            df[f"kurt_{w}"] = ret.rolling(w).kurt()

        # Support / Resistance
        for p in [20, 50]:
            df[f"high_{p}"]      = h.rolling(p).max()
            df[f"low_{p}"]       = l.rolling(p).min()
            df[f"dist_high_{p}"] = (
                c - df[f"high_{p}"]) / (c + eps)
            df[f"dist_low_{p}"]  = (
                c - df[f"low_{p}"]) / (c + eps)

        # ADX
        dm_pos = (h - h.shift(1)).clip(lower=0)
        dm_neg = (l.shift(1) - l).clip(lower=0)
        tr14   = df["atr_14"]
        df["adx_pos"] = (
            dm_pos.rolling(14).mean() / (tr14 + eps))
        df["adx_neg"] = (
            dm_neg.rolling(14).mean() / (tr14 + eps))
        df["adx"] = (df["adx_pos"] - df["adx_neg"]).abs()

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        gc.collect()

        n_f = len([col for col in df.columns
                   if col not in {
                       "timestamp", "open", "high",
                       "low", "close", "volume",
                       "returns", "log_returns"}])
        print(f"  Features: {n_f} | RAM: {ram()}")
        return df


# ════════════════════════════════════════════════════════
# OVERRIDE 3: HPOptimizer
# Persists Optuna study to SQLite for session resume
# ════════════════════════════════════════════════════════
class HPOptimizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def optimize(self, X, y, storage_dir=None):
        # SQLite persistence — survives across sessions
        if storage_dir:
            db_path = os.path.join(
                storage_dir, "optuna_study.db")
            storage = f"sqlite:///{db_path}"
            print(f"  [Optuna] Using SQLite: {db_path}")
        else:
            storage = None

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

        # load_if_exists=True is the key resume mechanism
        study = optuna.create_study(
            study_name="xgb_rl_optimization",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner())

        existing = len(study.trials)
        remaining = max(0,
            self.cfg.OPTUNA_N_TRIALS - existing)
        print(f"  [Optuna] Existing trials: {existing} "
              f"| Running: {remaining} more")

        if remaining > 0:
            study.optimize(
                objective,
                n_trials=remaining,
                n_jobs=1,
                show_progress_bar=True)

        print(f"  Best AUC: {study.best_value:.4f}")
        return study.best_params

    @staticmethod
    def _cv_folds(n, ns, ep):
        emb   = int(n * ep)
        fsz   = n // ns
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
# OVERRIDE 4: EnsembleDoubleQAgent — GPU injection
# ════════════════════════════════════════════════════════
_BaseAgent = EnsembleDoubleQAgent

class EnsembleDoubleQAgent(_BaseAgent):
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
# OVERRIDE 5: ONNXExporter — Static [1, n] shape
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

        graph = onnx_model.graph
        inp   = graph.input[0]
        inp.type.tensor_type.shape.ClearField("dim")
        d0 = inp.type.tensor_type.shape.dim.add()
        d0.dim_value = 1
        d1 = inp.type.tensor_type.shape.dim.add()
        d1.dim_value = in_dim

        out  = graph.output[0]
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

        sess    = ort.InferenceSession(fpath)
        dummy   = np.zeros(
            (1, in_dim), dtype=np.float32)
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
        return exported


# ════════════════════════════════════════════════════════
# MAIN PIPELINE — with session resume support
# ════════════════════════════════════════════════════════
gc.collect()
print(f"Starting GitHub Actions Job | RAM: {ram()}\n")

# Output dir aligned with train.yml cache path
cache_dir = os.path.join(os.getcwd(), "xgb_rl_artifacts")
os.makedirs(cache_dir, exist_ok=True)
print(f"Artifacts dir: {cache_dir}")

# Check for previous session
prev_state = load_session_state(cache_dir)

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

# ── Load data & features ──────────────────────────────
print("Phase 1: Data")
from trainer import DataIngestion
data = DataIngestion.load("csv", "EURUSD_H1.csv",
                          cfg.HISTORICAL_BARS)
print(f"  {len(data):,} bars loaded\n")

print("Phase 2: Features")
fe        = FeatureEngine(cfg)
feat_data = fe.build(data)
print(f"  {feat_data.shape[1]} columns\n")

# ── Feature selection ─────────────────────────────────
# Reuse selected features from previous session if available
if (prev_state and
        "selected" in prev_state and
        len(prev_state["selected"]) > 0):
    selected      = prev_state["selected"]
    scaler_params = prev_state["scaler_params"]
    print(f"Phase 3: Reusing {len(selected)} features "
          f"from previous session")
else:
    print("Phase 3: Feature Selection")
    sel    = FeatureSelector(cfg)
    target = (feat_data["returns"].shift(-1) > 0
              ).astype(int).iloc[:-1]
    feat_sub  = feat_data.iloc[:-1].copy()
    selected  = sel.select(
        feat_sub, target, cfg.MAX_FEATURES_SELECTED)
    cut       = int(0.6 * len(feat_sub))
    scaler_params = build_scaler_params(
        feat_sub[selected].iloc[:cut], selected)

    # ── Optuna with SQLite persistence ────────────────
    print("\nPhase 4: Hyperparameter Optimization")
    Xopt = (feat_sub[selected].iloc[:cut]
            .fillna(0).values.astype(np.float32))
    yopt = target.iloc[:cut].values
    hp   = HPOptimizer(cfg)
    best = hp.optimize(Xopt, yopt,
                       storage_dir=cache_dir)
    cfg.XGB_BASE.update(best)
    print(f"  Best params: {best}\n")

# ── Initialize components ─────────────────────────────
print("Phase 5: Initialize Agent")
from trainer import (PortfolioManager, RiskManager,
                     RewardFunction, WalkForward,
                     StateBuilder, PerfMonitor)

agent  = EnsembleDoubleQAgent(cfg)
port   = PortfolioManager(cfg)
risk   = RiskManager(cfg)
reward = RewardFunction(cfg)
wf     = WalkForward(cfg)
sb     = StateBuilder()
perf   = PerfMonitor(cfg)

# ── Restore previous state if available ───────────────
if prev_state:
    agent.epsilon = prev_state.get("epsilon", 1.0)
    agent.step    = prev_state.get("step", 0)
    agent.ew      = prev_state.get(
        "ew", np.ones(cfg.N_ENSEMBLE_MODELS) /
              cfg.N_ENSEMBLE_MODELS)
    port.equity   = prev_state.get("equity",
                                    cfg.INITIAL_CAPITAL)
    port.cash     = prev_state.get("cash",
                                    cfg.INITIAL_CAPITAL)
    port.trades   = prev_state.get("trades", 0)
    port.wins     = prev_state.get("wins", 0)
    port.gross_p  = prev_state.get("gross_p", 0.0)
    port.gross_l  = prev_state.get("gross_l", 0.0)
    reward.hist   = prev_state.get("reward_hist", [])
    wf.last       = prev_state.get("wf_last", 0)
    wf.cnt        = prev_state.get("wf_cnt", 0)

    # Load XGBoost model weights
    if load_agent_models(agent, cache_dir):
        print("[RELAY] ✔ Agent models restored")

# Determine start index
start_idx = prev_state.get(
    "start_idx",
    max(cfg.MIN_TRAIN_SAMPLES, 200)
) if prev_state else max(cfg.MIN_TRAIN_SAMPLES, 200)

# State dim
test_pi    = port.info(
    float(feat_data["close"].iloc[200]))
test_state = sb.build(
    feat_data, 200, selected, test_pi)
state_dim  = len(test_state)
print(f"  State dim: {state_dim}")
print(f"  Resume from step: {start_idx}\n")

# ── Walk-forward training loop ────────────────────────
print("Phase 6: Walk-Forward Training")
from tqdm import tqdm
import signal, time

end_idx     = len(feat_data) - 1
selected    = [f for f in selected
               if f in feat_data.columns]
all_results = []
train_log   = []

# 5-hour time limit with 25-min buffer for save
SESSION_LIMIT = 5 * 3600 + 5 * 60
session_start = time.time()

for t in tqdm(range(start_idx, end_idx),
              desc="Training", unit="step"):

    # Time check — save state before timeout
    elapsed = time.time() - session_start
    if elapsed > SESSION_LIMIT:
        print(f"\n[RELAY] Time limit reached at "
              f"step {t} — saving state")
        save_session_state(
            agent, port, reward, wf,
            selected, scaler_params,
            t, all_results, cache_dir)
        save_agent_models(agent, cache_dir)
        break

    if wf.should_retrain(t, feat_data):
        if wf.regime_change(feat_data, t):
            agent  = EnsembleDoubleQAgent(cfg)
            reward.reset()
        if len(all_results) > 100:
            rs = np.array(
                [r["state"] for r in all_results[-100:]])
            rt = np.array(
                [r["reward"] for r in all_results[-100:]])
            agent.update_ew(rs, rt)
            # Save checkpoint at each retrain
            save_agent_models(agent, cache_dir)

    pi    = port.info(
        float(feat_data["close"].iloc[t]),
        feat_data["timestamp"].iloc[t])
    state = sb.build(feat_data, t, selected, pi)
    raw_a = agent.select(state, training=True)

    row = feat_data.iloc[t]
    mkt = {
        "vol_20": float(row.get("vol_20", 0.02)),
        "vol_60": float(row.get("vol_60", 0.02)),
        "date":   str(feat_data["timestamp"].iloc[t])[:10]
    }
    val_a = risk.validate(raw_a, pi, mkt)
    conf  = agent.confidence(state)
    size  = risk.position_size(pi, mkt, conf)
    price = float(feat_data["close"].iloc[t])
    time_ = feat_data["timestamp"].iloc[t]
    res   = port.execute(val_a, price, time_, size)
    if res["executed"]:
        risk.record()

    upi = port.info(price, time_)
    rwd = reward.calc(val_a, res["port_ret"], upi)
    wf.record(rwd)

    if t < end_idx - 1:
        ns   = sb.build(feat_data, t+1, selected, upi)
        done = False
    else:
        ns = state; done = True

    agent.store(state, val_a, rwd, ns, done)
    tm = agent.train()
    if tm:
        train_log.append(tm)

    all_results.append({
        "t":      t,
        "price":  price,
        "action": val_a,
        "equity": res["equity"],
        "dd":     res["dd"],
        "reward": rwd,
        "state":  state,
    })

    if (res["equity"] <
            cfg.INITIAL_CAPITAL *
            (1 - cfg.MAX_DRAWDOWN_LIMIT * 1.5)):
        print(f"\nEmergency stop at step {t}")
        break

# ── Final session save ────────────────────────────────
save_session_state(
    agent, port, reward, wf,
    selected, scaler_params,
    end_idx, all_results, cache_dir)
save_agent_models(agent, cache_dir)

# ── Evaluation & ONNX export ──────────────────────────
print("\nPhase 7: Evaluation")
metrics = perf.evaluate(port.eq_curve, port.trade_log)

print("Phase 8: ONNX Export")
exporter = ONNXExporter(cfg)
exported = exporter.export_agent(
    agent, selected, scaler_params,
    state_dim, metrics)
print(f"  Exported {len(exported)} ONNX files\n")

# ── Final file listing ────────────────────────────────
files = os.listdir(cache_dir)
print(f"\nTotal files in cache: {len(files)}")
for f in sorted(files):
    print(f"  {f}")
print("\nDone! Workflow handoff ready.")
