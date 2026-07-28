# ════════════════════════════════════════════════════════
# train.py — v2.1 SUPERVISED CLASSIFICATION & ONNX PIPELINE
# ════════════════════════════════════════════════════════

import gc, os, sys, psutil, shutil, json, joblib
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.getcwd())

# ── ONNX imports ──
import onnx
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Pin opset to 3 for ai.onnx.ml domain to prevent ONNX export errors
TARGET_OPSET = {"": 15, "ai.onnx.ml": 3}


# ── GPU Detection ──────────────────────────────────────
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
    except Exception:
        pass
    try:
        m = XGBClassifier(
            n_estimators=5,
            tree_method="gpu_hist",
            verbosity=0)
        m.fit(X, y)
        print("✔ Using tree_method=gpu_hist")
        return {"device": None, "tree_method": "gpu_hist"}
    except Exception:
        pass
    print("⚠ Using CPU")
    return {"device": None, "tree_method": "hist"}


GPU_PARAMS = get_xgb_gpu_params()


def ram():
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return f"{mb:.0f} MB"


# ════════════════════════════════════════════════════════
# CONFIG & DATA INGESTION
# ════════════════════════════════════════════════════════
class SystemConfig:
    HISTORICAL_BARS = 50000
    OPTUNA_N_TRIALS = 30
    OPTUNA_CV_SPLITS = 5
    MAX_FEATURES_SELECTED = 30
    CORRELATION_THRESHOLD = 0.90
    OUTPUT_DIR = os.path.join(os.getcwd(), "xgb_trader_artifacts")
    TRADING_PAIR = "EURUSD"
    
    # Target Parameters
    TP_PCT = 0.015  # 1.5% Take Profit
    SL_PCT = 0.010  # 1.0% Stop Loss
    LOOKAHEAD_BARS = 24
    PROB_THRESHOLD_BUY = 0.63
    PROB_THRESHOLD_SELL = 0.63

    XGB_BASE = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }


class DataIngestion:
    @staticmethod
    def load_data(filepath, max_bars):
        if not os.path.exists(filepath):
            # Generate synthetic data if file not present
            np.random.seed(42)
            prices = 1.1000 + np.cumsum(np.random.randn(max_bars) * 0.0005)
            opens = np.empty_like(prices)
            opens[0] = prices[0]
            opens[1:] = prices[:-1]  # Fix open price gap
            highs = np.maximum(opens, prices) + np.abs(np.random.randn(max_bars) * 0.0003)
            lows = np.minimum(opens, prices) - np.abs(np.random.randn(max_bars) * 0.0003)
            volumes = np.random.randint(100, 5000, size=max_bars)
            timestamps = pd.date_range(end="2026-07-01", periods=max_bars, freq="1h")

            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
                "returns": pd.Series(prices).pct_change().fillna(0)
            })
            return df
        
        df = pd.read_csv(filepath)
        df["returns"] = df["close"].pct_change().fillna(0)
        return df.iloc[-max_bars:].reset_index(drop=True)


# ════════════════════════════════════════════════════════
# TARGET BUILDER (Fixes Same-Bar Ambiguity)
# ════════════════════════════════════════════════════════
class TargetBuilder:
    @staticmethod
    def build_targets(df, tp_pct, sl_pct, lookahead):
        n = len(df)
        y_buy = np.zeros(n, dtype=int)
        y_sell = np.zeros(n, dtype=int)
        
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        for i in range(n - lookahead):
            entry = closes[i]
            tp_buy_price = entry * (1 + tp_pct)
            sl_buy_price = entry * (1 - sl_pct)
            
            tp_sell_price = entry * (1 - tp_pct)
            sl_sell_price = entry * (1 + sl_pct)

            # Check BUY Target
            for j in range(1, lookahead + 1):
                hit_tp = highs[i + j] >= tp_buy_price
                hit_sl = lows[i + j] <= sl_buy_price

                if hit_tp and hit_sl:
                    y_buy[i] = 0  # Conservative Same-Bar SL default
                    break
                elif hit_sl:
                    y_buy[i] = 0
                    break
                elif hit_tp:
                    y_buy[i] = 1
                    break

            # Check SELL Target
            for j in range(1, lookahead + 1):
                hit_tp = lows[i + j] <= tp_sell_price
                hit_sl = highs[i + j] >= sl_sell_price

                if hit_tp and hit_sl:
                    y_sell[i] = 0  # Conservative Same-Bar SL default
                    break
                elif hit_sl:
                    y_sell[i] = 0
                    break
                elif hit_tp:
                    y_sell[i] = 1
                    break

        return y_buy, y_sell


# ════════════════════════════════════════════════════════
# FEATURE ENGINE
# ════════════════════════════════════════════════════════
class FeatureEngine:
    EPS = 1e-10

    def build(self, data):
        df = data.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        ret = df["returns"]

        for p in [10, 20, 50]:
            df[f"close_sma_{p}"] = c / (c.rolling(p).mean() + self.EPS) - 1
            df[f"rsi_{p}"] = self._rsi(c, p)

        for w in [10, 20, 60]:
            df[f"vol_{w}"] = ret.rolling(w).std()

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean() / (c + self.EPS)

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.dropna(inplace=True)
        return df.reset_index(drop=True)

    @staticmethod
    def _rsi(s, p):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / (l + 1e-10))


# ════════════════════════════════════════════════════════
# MODEL TRAINER & CALIBRATION
# ════════════════════════════════════════════════════════
class ModelTrainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train_calibrated_model(self, X_train, y_train):
        params = {**self.cfg.XGB_BASE, **GPU_PARAMS}
        params = {k: v for k, v in params.items() if v is not None}
        
        base_xgb = XGBClassifier(**params)
        
        # Calibrate raw outputs using Isotonic Regression
        calibrated_model = CalibratedClassifierCV(
            estimator=base_xgb,
            method="isotonic",
            cv=3
        )
        calibrated_model.fit(X_train, y_train)
        return calibrated_model


# ════════════════════════════════════════════════════════
# ONNX EXPORTER
# ════════════════════════════════════════════════════════
class ONNXExporter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def export_model(self, model, name, num_features):
        initial_types = [("input", FloatTensorType([1, num_features]))]
        
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_types,
            target_opset=TARGET_OPSET
        )

        fpath = os.path.join(self.out_dir, f"{name}.onnx")
        with open(fpath, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # ONNX Runtime Validation
        sess = ort.InferenceSession(fpath)
        dummy = np.zeros((1, num_features), dtype=np.float32)
        in_name = sess.get_inputs()[0].name
        probs = sess.run(None, {in_name: dummy})[1]
        print(f"  ✔ Exported {name}.onnx | Prob outputs shape: {probs.shape}")
        return fpath


# ════════════════════════════════════════════════════════
# SESSION STATE & RELAY HELPERS
# ════════════════════════════════════════════════════════
def write_progress_flag(output_dir, steps_run: int):
    flag_path = os.path.join(output_dir, "session_progress.json")
    with open(flag_path, "w") as fh:
        json.dump({"steps_run": steps_run}, fh)


# ════════════════════════════════════════════════════════
# MAIN EXECUTION PIPELINE
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    gc.collect()
    print(f"Starting GitHub Actions Run | Initial RAM: {ram()}")

    cfg = SystemConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\nPhase 1: Loading Data...")
    df = DataIngestion.load_data("EURUSD_H1.csv", cfg.HISTORICAL_BARS)

    print("Phase 2: Extracting Features...")
    fe = FeatureEngine()
    df_feat = fe.build(df)

    print("Phase 3: Labeling Targets (TP/SL same-bar ambiguity resolved)...")
    y_buy, y_sell = TargetBuilder.build_targets(
        df_feat, cfg.TP_PCT, cfg.SL_PCT, cfg.LOOKAHEAD_BARS
    )

    feature_cols = [c for c in df_feat.columns if c not in ["timestamp", "open", "high", "low", "close", "volume", "returns"]]
    X = df_feat[feature_cols].values.astype(np.float32)

    # Save canonical feature names and indices
    features_meta = {
        "feature_cols": feature_cols,
        "feature_index": {col: idx for idx, col in enumerate(feature_cols)}
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(features_meta, f, indent=2)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_buy_tr, y_buy_te = y_buy[:train_size], y_buy[train_size:]
    y_sell_tr, y_sell_te = y_sell[:train_size], y_sell[train_size:]

    print("\nPhase 4: Training & Calibrating Models...")
    trainer = ModelTrainer(cfg)
    
    print("  -> Calibrating BUY Model...")
    model_buy = trainer.train_calibrated_model(X_train, y_buy_tr)
    
    print("  -> Calibrating SELL Model...")
    model_sell = trainer.train_calibrated_model(X_train, y_sell_tr)

    print("\nPhase 5: Evaluating Out-of-Sample Calibrated Probabilities...")
    p_buy_te = model_buy.predict_proba(X_test)[:, 1]
    p_sell_te = model_sell.predict_proba(X_test)[:, 1]

    auc_buy = roc_auc_score(y_buy_te, p_buy_te)
    auc_sell = roc_auc_score(y_sell_te, p_sell_te)
    
    print(f"  OOS BUY AUC:  {auc_buy:.4f}")
    print(f"  OOS SELL AUC: {auc_sell:.4f}")

    metrics = {
        "oos_buy_auc": float(auc_buy),
        "oos_sell_auc": float(auc_sell),
        "buy_threshold": cfg.PROB_THRESHOLD_BUY,
        "sell_threshold": cfg.PROB_THRESHOLD_SELL,
        "total_test_bars": len(X_test)
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nPhase 6: Exporting ONNX Artifacts...")
    exporter = ONNXExporter(cfg.OUTPUT_DIR)
    exporter.export_model(model_buy, "model_buy", len(feature_cols))
    exporter.export_model(model_sell, "model_sell", len(feature_cols))

    write_progress_flag(cfg.OUTPUT_DIR, steps_run=len(X))
    print(f"\nExecution Finished Successfully | Final RAM: {ram()}")
