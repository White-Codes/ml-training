import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import onnx
import tf2onnx

# Disable Optuna verbose logging for cleaner CI outputs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# ------------------------------------------------------------------
DATA_PATH = "EURUSD_H1.csv"  # Update path if needed
df = pd.read_csv(DATA_PATH)

# Assume 'X' features and 'y' target labels are pre-calculated
# y: 1 = BUY, -1 = SELL, 0 = NEUTRAL (or separate models for Buy/Sell)
X = df.drop(columns=["target", "time"], errors="ignore").values
y = df["target"].values
pnl_returns = df["pnl_points"].values  # Point return per bar/trade for performance engine

# ------------------------------------------------------------------
# 2. STRICT 3-WAY SPLIT (70% Train / 15% Val / 15% Holdout)
# ------------------------------------------------------------------
train_idx = int(len(df) * 0.70)
val_idx   = int(len(df) * 0.85)

X_train, y_train = X[:train_idx], y[:train_idx]
X_val,   y_val   = X[train_idx:val_idx], y[train_idx:val_idx]
X_hold,  y_hold  = X[val_idx:], y[val_idx:]

pnl_val  = pnl_returns[train_idx:val_idx]
pnl_hold = pnl_returns[val_idx:]

# ------------------------------------------------------------------
# 3. ADVANCED QUANTITATIVE EVALUATION ENGINE
# ------------------------------------------------------------------
def calculate_quant_metrics(predictions, actual_pnl, threshold=0.5, min_trades=50):
    """
    Calculates institutional trading metrics:
    - Profit Factor, Win Rate, Total Trades
    - Max Drawdown (%), Sharpe Ratio, Expectancy
    """
    # Filter trades above probability threshold
    trade_mask = predictions >= threshold
    trades_pnl = actual_pnl[trade_mask]
    total_trades = len(trades_pnl)
    
    # Min-trade penalty enforcement
    if total_trades < min_trades:
        return {
            "pf": 0.0, "trades": total_trades, "win_rate": 0.0,
            "max_dd": 100.0, "sharpe": -99.0, "expectancy": 0.0,
            "score": -1000.0
        }
    
    wins = trades_pnl[trades_pnl > 0]
    losses = trades_pnl[trades_pnl < 0]
    
    gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-6
    
    pf = gross_profit / gross_loss
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    expectancy = np.mean(trades_pnl) if total_trades > 0 else 0.0
    
    # Calculate Equity Curve & Max Drawdown (%)
    equity_curve = np.cumsum(trades_pnl) + 10000.0  # Assumes $10,000 starting equity
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = abs(np.min(drawdowns)) * 100.0
    
    # Calculate Annualized Sharpe Ratio (assuming H1 timeframe, ~6200 bars/yr)
    std_pnl = np.std(trades_pnl, ddof=1)
    if std_pnl > 0:
        sharpe = (np.mean(trades_pnl) / std_pnl) * np.sqrt(total_trades)
    else:
        sharpe = 0.0
        
    # Composite score for Optuna objective
    score = (pf * 5.0) + (sharpe * 2.0) - (max_dd * 0.1)
    
    return {
        "pf": pf,
        "trades": total_trades,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "expectancy": expectancy,
        "score": score
    }

# ------------------------------------------------------------------
# 4. OPTUNA HYPERPARAMETER SEARCH (VALIDATION SET ONLY)
# ------------------------------------------------------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predict on 15% Validation Set ONLY
    val_probs = model.predict_proba(X_val)[:, 1]
    thresh = trial.suggest_float('thresh', 0.45, 0.65)
    
    metrics = calculate_quant_metrics(val_probs, pnl_val, threshold=thresh, min_trades=100)
    return metrics["score"]

print("Starting 5-Hour Optuna Optimization (Tuning on Validation Set)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, timeout=18000)  # 5 Hours limit

print(f"Best Trial Score: {study.best_value:.4f}")
best_params = study.best_params
best_thresh = best_params.pop('thresh')

# ------------------------------------------------------------------
# 5. RETRAIN FINAL MODEL ON TRAIN + VAL
# ------------------------------------------------------------------
print("\nRetraining final model on Combined Train + Validation data...")
X_comb = np.vstack((X_train, X_val))
y_comb = np.concatenate((y_train, y_val))

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_comb, y_comb)

# ------------------------------------------------------------------
# 6. THE FINAL TEST OF TRUTH (100% UNTOUCHED HOLDOUT SET)
# ------------------------------------------------------------------
hold_probs = final_model.predict_proba(X_hold)[:, 1]
hold_metrics = calculate_quant_metrics(hold_probs, pnl_hold, threshold=best_thresh, min_trades=30)

print("\n" + "=" * 60)
print("       FINAL TEST OF TRUTH (STRICT OUT-OF-SAMPLE HOLDOUT)       ")
print("=" * 60)
print(f"Probability Threshold:    {best_thresh:.2f}")
print(f"Total Holdout Trades:     {hold_metrics['trades']}")
print(f"Holdout Win Rate:         {hold_metrics['win_rate'] * 100:.2f}%")
print(f"Holdout Profit Factor:    {hold_metrics['pf']:.3f}")
print(f"Holdout Sharpe Ratio:     {hold_metrics['sharpe']:.2f}")
print(f"Holdout Max Drawdown:     {hold_metrics['max_dd']:.2f}%")
print(f"Trade Expectancy:         ${hold_metrics['expectancy']:.2f} / trade")
print("-" * 60)

# Validation Guardrail Thresholds
if hold_metrics['pf'] >= 1.25 and hold_metrics['sharpe'] >= 1.0 and hold_metrics['max_dd'] <= 15.0:
    print("STATUS: ✔ PASSED (Institutional-Grade Quant Edge Confirmed)")
else:
    print("STATUS: ✖ FAILED (Model Failed Out-of-Sample Risk Checks)")
print("=" * 60 + "\n")

# ------------------------------------------------------------------
# 7. EXPORT MODEL ARTIFACTS
# ------------------------------------------------------------------
if hold_metrics['pf'] >= 1.25:
    final_model.save_model("model_buy.json")
    print("Artifact successfully exported: model_buy.json")
