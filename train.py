# ════════════════════════════════════════════════════════
# train.py — v5.0 STRICT 3-WAY HOLDOUT QUANT PIPELINE
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
    MIN_VAL_TRADES = 100   # Minimum trades required on Validation Set
    MIN_HOLD_TRADES = 30   # Minimum trades required on Strict Holdout Set
    
    # Initial Fallback Probability Threshold
    PROB_THRESHOLD = 0.52   


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
        
        df["atr_raw"] = atr_14  # Saved for Triple-Barrier & PnL evaluation

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
            df[f"rsi_{p}"] = self._rsi(c, p) / 100.0

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

            # BUY Triple-Barrier
            for j in range(1, lookahead + 1):
                hit_tp = highs[i + j] >= tp_buy_price
                hit_sl = lows[i + j] <= sl_buy_price

                if hit_tp and hit_sl or hit_sl:
                    y_buy[i] = 0
                    break
                elif hit_tp:
                    y_buy[i] = 1
                    break

            # SELL Triple-Barrier
            for j in range(1, lookahead + 1):
                hit_tp = lows[i + j] <= tp_sell_price
                hit_sl = highs[i + j] >= sl_sell_price

                if hit_tp and hit_sl or hit_sl:
                    y_sell[i] = 0
                    break
                elif hit_tp:
                    y_sell[i] = 1
                    break

        return y_buy, y_sell


# ════════════════════════════════════════════════════════
# ADVANCED QUANT PERFORMANCE EVALUATION ENGINE
# ════════════════════════════════════════════════════════
class QuantPerformanceEngine:
    @staticmethod
    def evaluate(df_sub, p_buy, p_sell, thresh, cfg, min_trades=50):
        trades = []
        closes = df_sub["close"].values
        highs = df_sub["high"].values
        lows = df_sub["low"].values
        atrs = df_sub["atr_raw"].values

        for i in range(len(df_sub) - cfg.LOOKAHEAD_BARS):
            prob_b = p_buy[i]
            prob_s = p_sell[i]
            
            signal = 0
            if prob_b >= thresh and prob_b > prob_s:
                signal = 1
            elif prob_s >= thresh and prob_s > prob_b:
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
                    if (hit_tp and hit_sl) or hit_sl:
                        trade_pnl = -cfg.SL_ATR_MULT
                        break
                    elif hit_tp:
                        trade_pnl = cfg.TP_ATR_MULT
                        break
                elif signal == -1:
                    hit_tp = low <= (entry_price - tp_dist)
                    hit_sl = high >= (entry_price + sl_dist)
                    if (hit_tp and hit_sl) or hit_sl:
                        trade_pnl = -cfg.SL_ATR_MULT
                        break
                    elif hit_tp:
                        trade_pnl = cfg.TP_ATR_MULT
                        break

            trades.append(trade_pnl)

        total_trades = len(trades)
        if total_trades < min_trades:
            return {
                "total_trades": total_trades, "win_rate": 0.0, "profit_factor": 0.0,
                "total_pnl_r": 0.0, "max_drawdown_r": 100.0, "sharpe_ratio": -99.0,
                "expectancy_r": 0.0, "score": -1000.0
            }

        trades = np.array(trades)
        wins = trades[trades > 0]
        losses = trades[trades < 0]

        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-10

        profit_factor = gross_profit / gross_loss
        win_rate = len(wins) / total_trades
        total_pnl = np.sum(trades)
        expectancy = np.mean(trades)

        equity = np.cumsum(trades)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        std_pnl = np.std(trades, ddof=1)
        sharpe = (np.mean(trades) / (std_pnl + 1e-10)) * np.sqrt(252 * 24)

        # Optimization Score Function
        score = (profit_factor * np.log10(total_trades)) + (0.5 * sharpe) - (0.1 * max_dd)

        return {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "profit_factor": round(float(profit_factor), 4),
            "total_pnl_r": round(float(total_pnl), 2),
            "max_drawdown_r": round(float(max_dd), 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "expectancy_r": round(float(expectancy), 4),
            "score": float(score)
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
# MAIN TUNING & HOLDOUT PIPELINE
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    gc.collect()
    start_time = time.time()
    print(f"Starting Continuous Training Session v5.0 (Strict 3-Way Split) | RAM: {ram()}")

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

    # Save feature mapping
    features_meta = {
        "feature_cols": feature_cols,
        "feature_index": {col: idx for idx, col in enumerate(feature_cols)}
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "features.json"), "w") as f:
        json.dump(features_meta, f, indent=2)

    # ── STRICT 3-WAY SPLIT: 70% Train / 15% Validation / 15% Holdout ──
    n = len(X)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    X_train, X_val, X_hold = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_buy_tr, y_buy_val, y_buy_hold = y_buy[:train_idx], y_buy[train_idx:val_idx], y_buy[val_idx:]
    y_sell_tr, y_sell_val, y_sell_hold = y_sell[:train_idx], y_sell[train_idx:val_idx], y_sell[val_idx:]

    df_val = df_feat.iloc[train_idx:val_idx].reset_index(drop=True)
    df_hold = df_feat.iloc[val_idx:].reset_index(drop=True)

    print(f"✔ Dataset Split Breakdown:")
    print(f"   • Train Set (70%):      {len(X_train)} bars (Model fitting)")
    print(f"   • Validation Set (15%): {len(X_val)} bars (Optuna tuning & threshold optimization)")
    print(f"   • Holdout Set (15%):    {len(X_hold)} bars (100% UNTOUCHED Final Test of Truth)")

    print(f"\nPhase 4: Entering 5-Hour Optuna Search (Tuning exclusively on Validation Set)...")
    
    best_models_val = {"buy": None, "sell": None}
    best_val_results = {"score": -999.0}

    def objective(trial):
        if time.time() - start_time > cfg.MAX_TUNING_TIME_SEC:
            trial.study.stop()
            return -999.0

        threshold = trial.suggest_float("prob_threshold", 0.50, 0.58, step=0.01)

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

        # Fit on Training Set
        model_b = XGBClassifier(**params).fit(X_train, y_buy_tr)
        model_s = XGBClassifier(**params).fit(X_train, y_sell_tr)

        # Predict ONLY on Validation Set
        p_b_val = model_b.predict_proba(X_val)[:, 1]
        p_s_val = model_s.predict_proba(X_val)[:, 1]

        val_eval = QuantPerformanceEngine.evaluate(df_val, p_b_val, p_s_val, threshold, cfg, min_trades=cfg.MIN_VAL_TRADES)
        score = val_eval["score"]

        if score > best_val_results.get("score", -999.0):
            best_val_results["score"] = score
            best_val_results["eval"] = val_eval
            best_val_results["threshold"] = threshold
            best_val_results["params"] = params
            best_val_results["auc_buy"] = float(roc_auc_score(y_buy_val, p_b_val))
            best_val_results["auc_sell"] = float(roc_auc_score(y_sell_val, p_s_val))
            best_models_val["buy"] = model_b
            best_models_val["sell"] = model_s
            print(f"  ★ Trial {trial.number} New Best (Val)! Trades: {val_eval['total_trades']} | PF: {val_eval['profit_factor']} | Win Rate: {val_eval['win_rate']*100:.1f}% | Thresh: {threshold:.2f} | Sharpe: {val_eval['sharpe_ratio']:.2f}")

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5000, timeout=cfg.MAX_TUNING_TIME_SEC)

    # ════════════════════════════════════════════════════════
    # Phase 5: RETRAIN FINAL MODEL ON COMBINED (TRAIN + VAL)
    # ════════════════════════════════════════════════════════
    print("\nPhase 5: Retraining Final Winning Architecture on Combined (Train + Validation)...")
    best_params = best_val_results.get("params", {})
    optimal_thresh = best_val_results.get("threshold", 0.52)

    X_train_val = np.vstack((X_train, X_val))
    y_buy_tr_val = np.concatenate((y_buy_tr, y_buy_val))
    y_sell_tr_val = np.concatenate((y_sell_tr, y_sell_val))

    final_model_buy = XGBClassifier(**best_params).fit(X_train_val, y_buy_tr_val)
    final_model_sell = XGBClassifier(**best_params).fit(X_train_val, y_sell_tr_val)

    # ════════════════════════════════════════════════════════
    # Phase 6: THE FINAL TEST OF TRUTH (STRICT HOLDOUT SET)
    # ════════════════════════════════════════════════════════
    print("\nPhase 6: Executing Final Test of Truth on 100% Untouched Holdout Data...")
    
    p_b_hold = final_model_buy.predict_proba(X_hold)[:, 1]
    p_s_hold = final_model_sell.predict_proba(X_hold)[:, 1]

    hold_eval = QuantPerformanceEngine.evaluate(df_hold, p_b_hold, p_s_hold, optimal_thresh, cfg, min_trades=cfg.MIN_HOLD_TRADES)

    print("\n" + "═"*60)
    print("         FINAL TEST OF TRUTH (STRICT OUT-OF-SAMPLE HOLDOUT)       ")
    print("═"*60)
    print(f" Optimal Threshold:      {optimal_thresh:.2f}")
    print(f" Total Holdout Trades:   {hold_eval['total_trades']}")
    print(f" Holdout Win Rate:       {hold_eval['win_rate'] * 100:.1f}%")
    print(f" Holdout Profit Factor:  {hold_eval['profit_factor']:.4f}")
    print(f" Holdout Sharpe Ratio:   {hold_eval['sharpe_ratio']:.4f}")
    print(f" Holdout Max Drawdown:   {hold_eval['max_drawdown_r']:.2f} R")
    print(f" Trade Expectancy:       {hold_eval['expectancy_r']:.4f} R / trade")
    print(f" Total Return:           {hold_eval['total_pnl_r']:.2f} R")
    print("─"*60)

    # Validation Pass/Fail Gate
    passed = (hold_eval['profit_factor'] >= 1.20) and (hold_eval['sharpe_ratio'] >= 0.8) and (hold_eval['total_trades'] >= cfg.MIN_HOLD_TRADES)
    if passed:
        print(" STATUS: ✔ PASSED (Valid Out-of-Sample Quant Edge Confirmed)")
    else:
        print(" STATUS: ✖ FAILED (Edge Did Not Hold on Pure Holdout Data)")
    print("═"*60 + "\n")

    # Export metrics json
    metrics = {
        "val_auc": {"buy": best_val_results.get("auc_buy", 0), "sell": best_val_results.get("auc_sell", 0)},
        "val_backtest": best_val_results.get("eval", {}),
        "holdout_backtest": hold_eval,
        "optimal_threshold": optimal_thresh,
        "passed_holdout_check": passed,
        "best_params": best_params
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Export ONNX artifacts
    print("Phase 7: Exporting Final ONNX Artifacts...")
    exporter = ONNXExporter(cfg.OUTPUT_DIR)
    exporter.export_model(final_model_buy, "model_buy", len(feature_cols))
    exporter.export_model(final_model_sell, "model_sell", len(feature_cols))

    write_progress_flag(cfg.OUTPUT_DIR, steps_run=len(X))
    print(f"\nExecution Finished Successfully | Final RAM: {ram()}")
