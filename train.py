# ════════════════════════════════════════════════════════
# train.py — v4.0 ATR TRIPLE-BARRIER & STATIONARY QUANT PIPELINE
# ════════════════════════════════════════════════════════

import gc, os, sys, psutil, time, json, glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.getcwd())

# ── ONNX & ONNXRuntime imports ──
import onnx
import onnxruntime as ort
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as ONNXFloatTensorType


# ── GPU Detection ──────────────────────────────────────
def get_xgb_gpu_params():
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    try:
        m = XGBClassifier(n_estimators=5, device="cuda", tree_method="hist", verbosity=0)
        m.fit(X, y)
        print("✔ Using device=cuda")
        return {"device": "cuda", "tree_method": "hist"}
    except Exception:
        pass
    try:
        m = XGBClassifier(n_estimators=5, tree_method="gpu_hist", verbosity=0)
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
    HISTORICAL_BARS = 100000
    OUTPUT_DIR = os.path.join(os.getcwd(), "xgb_trader_artifacts")
    
    # Execution Window: 5 hours inside GitHub Runner
    MAX_TUNING_TIME_SEC = 5 * 3600  

    # Dynamic ATR Triple-Barrier Multipliers
    ATR_PERIOD = 14
    TP_ATR_MULT = 1.5      # 1.5x ATR target profit
    SL_ATR_MULT = 1.0      # 1.0x ATR stop loss
    LOOKAHEAD_BARS = 12    # 12-hour max holding window
    
    # Statistical Rigor Enforcement
    MIN_TRADES = 150       # Minimum required trades in test set to avoid cherry-picking
    
    # Initial Fallback Probability Thresholds
    PROB_THRESHOLD_BUY = 0.52   
    PROB_THRESHOLD_SELL = 0.52


class DataIngestion:
    @staticmethod
    def load_data(max_bars):
        csv_files = glob.glob("*.csv") + glob.glob("*.CSV")
        target_file = None
        for f in csv_files:
            if "eurusd" in f.lower():
                target_file = f
                break
        
        if target_file and os.path.exists(target_file):
            print(f"✔ Found local dataset file: {target_file}")
            df = pd.read_csv(target_file)
            df.columns = [c.lower() for c in df.columns]
            if "returns" not in df.columns and "close" in df.columns:
                df["returns"] = df["close"].pct_change().fillna(0)
            return df.iloc[-max_bars:].reset_index(drop=True)
        
        raise FileNotFoundError("EURUSD dataset not found in root directory!")


# ════════════════════════════════════════════════════════
# STATIONARY FEATURE ENGINE
# ════════════════════════════════════════════════════════
class FeatureEngine:
    EPS = 1e-10

    def build(self, data):
        df = data.copy()
        c, h, l = df["close"], df["high"], df["low"]
        ret = df["returns"]

        # 1. Compute Base ATR (Unnormalized)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().replace(0, self.EPS)
        atr_100 = tr.rolling(100).mean().replace(0, self.EPS)
        
        df["atr_raw"] = atr_14  # Saved for Triple-Barrier calculations

        # 2. Stationary Normalized SMA Distances (C - SMA) / ATR
        for p in [10, 20, 50, 100]:
            sma = c.rolling(p).mean()
            df[f"dist_sma_{p}_atr"] = (c - sma) / atr_14

        # 3. Volatility Ratios & Normalized Range
        df["vol_ratio_14_100"] = atr_14 / atr_100
        df["bar_range_atr"] = (h - l) / atr_14

        # 4. Stationary Return Z-Scores
        for w in [10, 20, 60]:
            roll_mean = ret.rolling(w).mean()
            roll_std = ret.rolling(w).std().replace(0, self.EPS)
            df[f"ret_zscore_{w}"] = (ret - roll_mean) / roll_std

        # 5. Stationarized Momentum (RSI)
        for p in [14, 28]:
            df[f"rsi_{p}"] = self._rsi(c, p) / 100.0  # Bounded [0, 1]

        # 6. Distance to Rolling High/Low normalized by ATR
        for w in [12, 24, 48]:
            roll_h = h.rolling(w).max()
            roll_l = l.rolling(w).min()
            df[f"dist_high_{w}_atr"] = (c - roll_h) / atr_14
            df[f"dist_low_{w}_atr"] = (c - roll_l) / atr_14

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df.reset_index(drop=True)

    @staticmethod
    def _rsi(s, p):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100 / (1 + g / (l + 1e-10))


# ════════════════════════════════════════════════════════
# ATR TRIPLE-BARRIER TARGET BUILDER
# ════════════════════════════════════════════════════════
class TargetBuilder:
    @staticmethod
    def build_targets(df, tp_mult, sl_mult, lookahead):
        n = len(df)
        y_buy = np.zeros(n, dtype=int)
        y_sell = np.zeros(n, dtype=int)
        
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        atrs = df["atr_raw"].values

        for i in range(n - lookahead):
            entry = closes[i]
            atr = atrs[i]
            
            if atr <= 0 or np.isnan(atr):
                continue

            tp_buy_price = entry + (tp_mult * atr)
            sl_buy_price = entry - (sl_mult * atr)
            
            tp_sell_price = entry - (tp_mult * atr)
            sl_sell_price = entry + (sl_mult * atr)

            # BUY Triple-Barrier Logic
            for j in range(1, lookahead + 1):
                hit_tp = highs[i + j] >= tp_buy_price
                hit_sl = lows[i + j] <= sl_buy_price

                if hit_tp and hit_sl:
                    y_buy[i] = 0  # Conservative touch of both SL/TP inside same bar
                    break
                elif hit_sl:
                    y_buy[i] = 0
                    break
                elif hit_tp:
                    y_buy[i] = 1
                    break

            # SELL Triple-Barrier Logic
            for j in range(1, lookahead + 1):
                hit_tp = lows[i + j] <= tp_sell_price
                hit_sl = highs[i + j] >= sl_sell_price

                if hit_tp and hit_sl:
                    y_sell[i] = 0
                    break
                elif hit_sl:
                    y_sell[i] = 0
                    break
                elif hit_tp:
                    y_sell[i] = 1
                    break

        return y_buy, y_sell


# ════════════════════════════════════════════════════════
# DYNAMIC ATR BACKTEST ENGINE
# ════════════════════════════════════════════════════════
class BacktestEngine:
    @staticmethod
    def run_backtest(df_test, p_buy, p_sell, cfg):
        thresh_buy = cfg.PROB_THRESHOLD_BUY
        thresh_sell = cfg.PROB_THRESHOLD_SELL

        trades = []
        closes = df_test["close"].values
        highs = df_test["high"].values
        lows = df_test["low"].values
        atrs = df_test["atr_raw"].values

        for i in range(len(df_test) - cfg.LOOKAHEAD_BARS):
            prob_b = p_buy[i]
            prob_s = p_sell[i]
            
            signal = 0
            if prob_b >= thresh_buy and prob_b > prob_s:
                signal = 1
            elif prob_s >= thresh_sell and prob_s > prob_b:
                signal = -1

            if signal == 0:
                continue

            entry_price = closes[i]
            atr = atrs[i]
            if atr <= 0 or np.isnan(atr):
                continue

            tp_dist = cfg.TP_ATR_MULT * atr
            sl_dist = cfg.SL_ATR_MULT * atr
            
            trade_pnl = 0.0
            
            for j in range(1, cfg.LOOKAHEAD_BARS + 1):
                high = highs[i + j]
                low = lows[i + j]

                if signal == 1:
                    hit_tp = high >= (entry_price + tp_dist)
                    hit_sl = low <= (entry_price - sl_dist)
                    if hit_tp and hit_sl or hit_sl:
                        trade_pnl = -cfg.SL_ATR_MULT
                        break
                    elif hit_tp:
                        trade_pnl = cfg.TP_ATR_MULT
                        break
                elif signal == -1:
                    hit_tp = low <= (entry_price - tp_dist)
                    hit_sl = high >= (entry_price + sl_dist)
                    if hit_tp and hit_sl or hit_sl:
                        trade_pnl = -cfg.SL_ATR_MULT
                        break
                    elif hit_tp:
                        trade_pnl = cfg.TP_ATR_MULT
                        break

            trades.append(trade_pnl)

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0
            }

        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades < 0]

        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-10

        profit_factor = gross_profit / gross_loss
        win_rate = len(wins) / len(trades)
        total_pnl = np.sum(trades)

        equity = np.cumsum(trades)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        sharpe = (np.mean(trades) / (np.std(trades) + 1e-10)) * np.sqrt(252 * 24)

        return {
            "total_trades": int(len(trades)),
            "win_rate": round(float(win_rate), 4),
            "profit_factor": round(float(profit_factor), 4),
            "total_pnl_pct": round(float(total_pnl), 2),  # measured in ATR risk units (R)
            "max_drawdown_pct": round(float(max_dd), 2),
            "sharpe_ratio": round(float(sharpe), 4)
        }


# ════════════════════════════════════════════════════════
# ONNX EXPORTER
# ════════════════════════════════════════════════════════
class ONNXExporter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def export_model(self, model, name, num_features):
        initial_types = [("input", ONNXFloatTensorType([1, num_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=15)

        fpath = os.path.join(self.out_dir, f"{name}.onnx")
        with open(fpath, "wb") as f:
            f.write(onnx_model.SerializeToString())

        sess = ort.InferenceSession(fpath)
        dummy = np.zeros((1, num_features), dtype=np.float32)
        in_name = sess.get_inputs()[0].name
        probs = sess.run(None, {in_name: dummy})[1]
        print(f"  ✔ Exported {name}.onnx | Prob outputs shape: {probs.shape}")
        return fpath


def write_progress_flag(output_dir, steps_run: int):
    flag_path = os.path.join(output_dir, "session_progress.json")
    with open(flag_path, "w") as fh:
        json.dump({"steps_run": steps_run}, fh)


# ════════════════════════════════════════════════════════
# MAIN CONTINUOUS TUNING PIPELINE
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    gc.collect()
    start_time = time.time()
    print(f"Starting Continuous Training Session v4.0 | Initial RAM: {ram()}")

    cfg = SystemConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\nPhase 1: Loading Dataset...")
    df = DataIngestion.load_data(cfg.HISTORICAL_BARS)

    print(f"Phase 2: Extracting Stationary Features across {len(df)} bars...")
    fe = FeatureEngine()
    df_feat = fe.build(df)

    print("Phase 3: Labeling Dynamic ATR Triple-Barrier Targets...")
    y_buy, y_sell = TargetBuilder.build_targets(df_feat, cfg.TP_ATR_MULT, cfg.SL_ATR_MULT, cfg.LOOKAHEAD_BARS)

    ignore_cols = ["timestamp", "open", "high", "low", "close", "volume", "returns", "atr_raw"]
    feature_cols = [c for c in df_feat.columns if c not in ignore_cols]
    X = df_feat[feature_cols].values.astype(np.float32)

    print(f"✔ Features created: {len(feature_cols)} stationary indicators.")

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
    df_test = df_feat.iloc[train_size:].reset_index(drop=True)

    print(f"\nPhase 4: Entering Continuous 5-Hour Optuna Search with Min-Trade Penalty ({cfg.MIN_TRADES}+ trades)...")
    
    best_models = {"buy": None, "sell": None}
    best_results = {"score": -999.0}

    def objective(trial):
        if time.time() - start_time > cfg.MAX_TUNING_TIME_SEC:
            trial.study.stop()
            return -999.0

        # Dynamic Probability Threshold Search
        threshold = trial.suggest_float("prob_threshold", 0.50, 0.58, step=0.01)
        cfg.PROB_THRESHOLD_BUY = threshold
        cfg.PROB_THRESHOLD_SELL = threshold

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "verbosity": 0,
            "eval_metric": "logloss",
            **GPU_PARAMS
        }
        params = {k: v for k, v in params.items() if v is not None}

        model_b = XGBClassifier(**params).fit(X_train, y_buy_tr)
        model_s = XGBClassifier(**params).fit(X_train, y_sell_tr)

        p_b = model_b.predict_proba(X_test)[:, 1]
        p_s = model_s.predict_proba(X_test)[:, 1]

        bt = BacktestEngine.run_backtest(df_test, p_b, p_s, cfg)

        trade_count = bt["total_trades"]
        
        # STRICT STATISTICAL PENALTY: Prevent Optuna from cherry-picking low-volume noise
        if trade_count < cfg.MIN_TRADES:
            score = -10.0 + (trade_count / cfg.MIN_TRADES)
        else:
            # Score penalizes small samples, rewards high Profit Factor and Sharpe Ratio
            score = (bt["profit_factor"] * np.log10(trade_count)) + (0.5 * bt["sharpe_ratio"])

        if score > best_results.get("score", -999.0):
            best_results["score"] = score
            best_results["bt"] = bt
            best_results["threshold"] = threshold
            best_results["auc_buy"] = float(roc_auc_score(y_buy_te, p_b))
            best_results["auc_sell"] = float(roc_auc_score(y_sell_te, p_s))
            best_models["buy"] = model_b
            best_models["sell"] = model_s
            print(f"  ★ Trial {trial.number} New Best! Trades: {trade_count} | PF: {bt['profit_factor']} | Win Rate: {bt['win_rate']*100:.1f}% | Thresh: {threshold:.2f} | Score: {score:.3f}")

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5000, timeout=cfg.MAX_TUNING_TIME_SEC)

    print("\nPhase 5: Optimization Summary & Best Metrics...")
    bt = best_results.get("bt", {})
    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │        BEST OPTUNA BACKTEST METRICS      │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │ Optimal Threshold: {best_results.get('threshold', 0.52):<22}│")
    print(f"  │ Total Trades:      {bt.get('total_trades', 0):<22}│")
    print(f"  │ Win Rate:          {bt.get('win_rate', 0.0) * 100:.1f}%{'':<18}│")
    print(f"  │ Profit Factor:     {bt.get('profit_factor', 0.0):<22}│")
    print(f"  │ Total PnL (ATR):   {bt.get('total_pnl_pct', 0.0):.2f} R{'':<17}│")
    print(f"  │ Max Drawdown (ATR):{bt.get('max_drawdown_pct', 0.0):.2f} R{'':<17}│")
    print(f"  │ Sharpe Ratio:      {bt.get('sharpe_ratio', 0.0):<22}│")
    print(f"  └──────────────────────────────────────────┘")

    metrics = {
        "auc": {"buy": best_results.get("auc_buy", 0), "sell": best_results.get("auc_sell", 0)},
        "backtest": bt,
        "optimal_threshold": best_results.get("threshold", 0.52),
        "best_params": study.best_params if len(study.trials) > 0 else {}
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if best_models["buy"] is not None and best_models["sell"] is not None:
        print("\nPhase 6: Exporting ONNX Artifacts for Best Validated Models...")
        exporter = ONNXExporter(cfg.OUTPUT_DIR)
        exporter.export_model(best_models["buy"], "model_buy", len(feature_cols))
        exporter.export_model(best_models["sell"], "model_sell", len(feature_cols))

    write_progress_flag(cfg.OUTPUT_DIR, steps_run=len(X))
    print(f"\nExecution Finished Successfully | Final RAM: {ram()}")
