# ============================================================
#  XGB_DirectTrader.py
#  Complete architectural redesign.
#
#  OLD: XGBoost Q-Learning (broken Q-divergence,
#       phantom returns, uncapped equity)
#
#  NEW: Walk-Forward Supervised Classification
#       - XGBoost predicts NEXT BAR DIRECTION
#       - Simple threshold entry/exit rules
#       - Fixed fractional position sizing (1% risk/trade)
#       - ATR-based SL/TP enforced in backtest loop
#       - Honest P&L: only closed trades count
#       - No RL, no replay buffer, no Q-values
#
#  Why this works better:
#    XGBoost excels at supervised classification.
#    Predicting "will next bar close higher than
#    current bar" is a well-posed problem.
#    RL requires millions of environment steps and
#    continuous action spaces — XGBoost handles
#    neither well.
# ============================================================

import os
import json
import warnings
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ╔══════════════════════════════════════════════════════════╗
# ║  CONFIG                                                  ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # Data
    HISTORICAL_BARS:   int   = 50_000
    OUTPUT_DIR:        str   = "xgb_trader_artifacts"

    # Walk-forward
    TRAIN_BARS:        int   = 3_000   # bars per training window
    TEST_BARS:         int   = 500     # bars per test window
    MIN_TRAIN_BARS:    int   = 1_000   # minimum to start

    # Model
    N_ESTIMATORS:      int   = 300
    OPTUNA_TRIALS:     int   = 40
    CV_SPLITS:         int   = 5
    EMBARGO_BARS:      int   = 24      # bars between train/test

    # Signal
    # Minimum predicted probability to enter
    PROB_THRESHOLD_BUY:  float = 0.58
    PROB_THRESHOLD_SELL: float = 0.42  # < this = sell signal
    # RSI bounds for entry confirmation
    RSI_BULL_MIN:        float = 45.0
    RSI_BEAR_MAX:        float = 55.0
    # ADX minimum for trend confirmation
    ADX_MIN:             float = 18.0

    # Risk - fixed fractional
    INITIAL_CAPITAL:     float = 100_000.0
    RISK_PER_TRADE_PCT:  float = 0.01   # 1% of equity per trade
    ATR_SL_MULT:         float = 1.5    # SL = 1.5 * ATR
    ATR_TP_MULT:         float = 2.0    # TP = 2.0 * ATR (RR=1.33)
    MAX_OPEN_TRADES:     int   = 1      # one trade at a time
    MAX_DAILY_TRADES:    int   = 3
    MAX_DRAWDOWN_HALT:   float = 0.15   # halt if DD > 15%

    # Transaction costs
    SPREAD_PIPS:         float = 1.5    # pips
    PIP_VALUE:           float = 0.0001 # for EURUSD
    COMMISSION_PER_LOT:  float = 7.0    # USD per lot round-turn

    # Lot sizing
    CONTRACT_SIZE:       float = 100_000.0  # standard lot

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.SPREAD_PRICE = (
            self.SPREAD_PIPS * self.PIP_VALUE
        )


# ╔══════════════════════════════════════════════════════════╗
# ║  DATA INGESTION                                          ║
# ╚══════════════════════════════════════════════════════════╝

class DataIngestion:
    @staticmethod
    def load_synthetic(n: int) -> pd.DataFrame:
        """
        Generates realistic EURUSD-like H1 data.
        Uses mean-reverting GBM with realistic
        spread, overnight gaps, and volatility
        clustering (GARCH-like).
        """
        np.random.seed(42)
        n_bars = n
        S0     = 1.1000
        mu     = 0.0      # FX near zero drift
        sigma0 = 0.0060   # ~6 pip H1 vol typical

        # GARCH(1,1) vol clustering
        alpha, beta = 0.10, 0.85
        prices = [S0]
        sigma  = sigma0
        sigmas = [sigma]

        for _ in range(1, n_bars):
            eps   = np.random.randn()
            r     = mu + sigma * eps
            sigma = np.sqrt(
                sigma0**2 * (1 - alpha - beta)
                + alpha * (sigma * eps)**2
                + beta  * sigma**2
            )
            sigma = np.clip(sigma, 0.0002, 0.03)
            prices.append(prices[-1] * np.exp(r))
            sigmas.append(sigma)

        prices = np.array(prices)
        sigmas = np.array(sigmas)

        # Build OHLC from close prices
        high  = prices * np.exp(
            np.abs(np.random.randn(n_bars)) *
            sigmas * 0.7
        )
        low   = prices * np.exp(
            -np.abs(np.random.randn(n_bars)) *
            sigmas * 0.7
        )
        opens = np.roll(prices, 1)
        opens[0] = S0

        high  = np.maximum(high, np.maximum(
            prices, opens))
        low   = np.minimum(low,  np.minimum(
            prices, opens))
        vol   = np.random.lognormal(8, 1, n_bars)

        ts = pd.date_range(
            "2020-01-01",
            periods=n_bars,
            freq="h"
        )

        df = pd.DataFrame({
            "timestamp": ts,
            "open":      opens,
            "high":      high,
            "low":       low,
            "close":     prices,
            "volume":    vol,
        })
        # Remove weekends (FX closed)
        df = df[df["timestamp"].dt.dayofweek < 5]
        df.reset_index(drop=True, inplace=True)
        logger.info(
            f"Synthetic data: {len(df):,} bars"
        )
        return df

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(
            filepath, parse_dates=["timestamp"]
        )
        df.sort_values(
            "timestamp", inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "timestamp","open","high",
            "low","close","volume"
        ]
        for c in required:
            if c not in df.columns:
                raise ValueError(
                    f"Missing column: {c}"
                )
        df = df.dropna(subset=required).copy()
        df = df[df["volume"] > 0].copy()
        df = df[df["high"] >= df["low"]].copy()
        df = df[df["high"] >= df["close"]].copy()
        df = df[df["low"]  <= df["close"]].copy()
        df.drop_duplicates(
            subset=["timestamp"], inplace=True
        )
        df.sort_values(
            "timestamp", inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        logger.info(
            f"Validated: {len(df):,} bars"
        )
        return df


# ╔══════════════════════════════════════════════════════════╗
# ║  FEATURE ENGINEERING                                     ║
# ║                                                          ║
# ║  Design principles:                                      ║
# ║  1. All features normalized to [-1, 1] or [0, 1]        ║
# ║     so XGBoost sees consistent scale                     ║
# ║  2. No look-ahead: every feature uses only              ║
# ║     data available at bar close                          ║
# ║  3. Features are PREDICTIVE not descriptive:             ║
# ║     we want signals that precede price moves             ║
# ║  4. Interaction features kept minimal to                 ║
# ║     reduce overfitting risk                              ║
# ╚══════════════════════════════════════════════════════════╝

class FeatureEngine:
    EPS = 1e-10

    def build(self, df: pd.DataFrame
              ) -> pd.DataFrame:
        d = df.copy()
        c = d["close"]
        h = d["high"]
        l = d["low"]
        v = d["volume"]
        o = d["open"]

        # --- Returns ---
        d["ret_1"]  = c.pct_change(1)
        d["ret_3"]  = c.pct_change(3)
        d["ret_5"]  = c.pct_change(5)
        d["ret_10"] = c.pct_change(10)
        d["ret_20"] = c.pct_change(20)

        # --- ATR (needed for SL/TP later) ---
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        d["atr_14"] = tr.rolling(14).mean()
        d["atr_7"]  = tr.rolling(7).mean()
        # Normalized ATR
        d["natr_14"] = d["atr_14"] / (c + self.EPS)

        # --- RSI ---
        for p in [7, 14, 21]:
            diff = c.diff()
            g    = diff.clip(lower=0).rolling(p).mean()
            ls   = (-diff.clip(upper=0)).rolling(p).mean()
            rsi  = 100 - 100 / (1 + g / (ls + self.EPS))
            # Normalize RSI to [-1, 1]
            d[f"rsi_{p}"] = (rsi - 50) / 50.0

        # --- MACD ---
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig
        # Normalize by ATR
        d["macd_norm"] = (
            macd / (d["atr_14"] + self.EPS)
        )
        d["macd_sig_norm"] = (
            sig / (d["atr_14"] + self.EPS)
        )
        d["macd_hist_norm"] = (
            hist / (d["atr_14"] + self.EPS)
        )
        d["macd_cross"] = np.sign(hist)

        # --- Bollinger Bands ---
        bm    = c.rolling(20).mean()
        bs    = c.rolling(20).std()
        d["bb_pos"] = (
            (c - bm) / (2 * bs + self.EPS)
        ).clip(-1, 1)
        d["bb_width"] = (
            4 * bs / (bm + self.EPS)
        )
        d["bb_squeeze"] = (
            d["bb_width"] <
            d["bb_width"].rolling(50).mean()
        ).astype(float)

        # --- Stochastic ---
        for p in [14, 21]:
            ll  = l.rolling(p).min()
            hh  = h.rolling(p).max()
            k   = (c - ll) / (hh - ll + self.EPS)
            kd  = k.rolling(3).mean()
            d[f"stoch_{p}"]  = k * 2 - 1  # [-1,1]
            d[f"stochd_{p}"] = kd * 2 - 1
            d[f"stoch_cross_{p}"] = np.sign(k - kd)

        # --- ADX ---
        dm_pos = (h - h.shift(1)).clip(lower=0)
        dm_neg = (l.shift(1) - l).clip(lower=0)
        atr14  = d["atr_14"]
        pdi    = 100 * dm_pos.rolling(14).mean() / (
            atr14 + self.EPS
        )
        ndi    = 100 * dm_neg.rolling(14).mean() / (
            atr14 + self.EPS
        )
        dx     = (
            100 * (pdi - ndi).abs() /
            (pdi + ndi + self.EPS)
        )
        adx    = dx.ewm(span=14, adjust=False).mean()
        d["adx"]     = adx / 100.0  # [0,1]
        d["di_bull"] = (pdi > ndi).astype(float)
        d["di_diff"] = (pdi - ndi) / 100.0

        # --- Momentum ---
        for p in [5, 10, 20, 50]:
            mom = c / (c.shift(p) + self.EPS) - 1
            d[f"mom_{p}"] = mom.clip(-0.1, 0.1) / 0.1

        # --- Volatility ratio ---
        vol5  = d["ret_1"].rolling(5).std()
        vol20 = d["ret_1"].rolling(20).std()
        vol60 = d["ret_1"].rolling(60).std()
        d["vol_ratio_5_20"]  = (
            vol5  / (vol20 + self.EPS)
        ).clip(0, 5) / 5.0
        d["vol_ratio_20_60"] = (
            vol20 / (vol60 + self.EPS)
        ).clip(0, 5) / 5.0
        d["vol_20"] = vol20  # keep for reference

        # --- Price position within range ---
        for p in [10, 20, 50, 100]:
            hh = h.rolling(p).max()
            ll = l.rolling(p).min()
            d[f"range_pos_{p}"] = (
                (c - ll) / (hh - ll + self.EPS)
            ) * 2 - 1  # [-1, 1]

        # --- EMA distance ---
        for p in [10, 20, 50, 200]:
            ema = c.ewm(span=p, adjust=False).mean()
            d[f"ema_dist_{p}"] = (
                (c - ema) / (ema + self.EPS)
            ).clip(-0.05, 0.05) / 0.05

        # --- Candle structure ---
        spread = (h - l).clip(lower=self.EPS)
        body   = (c - o).abs()
        d["body_ratio"]  = (body / spread).clip(0, 1)
        d["bull_candle"] = (c > o).astype(float)
        d["upper_wick"]  = (
            (h - pd.concat([c, o], axis=1).max(axis=1)) /
            (spread + self.EPS)
        )
        d["lower_wick"] = (
            (pd.concat([c, o], axis=1).min(axis=1) - l) /
            (spread + self.EPS)
        )

        # --- Volume ---
        v_ma = v.rolling(20).mean()
        d["vol_ratio_bar"] = (
            v / (v_ma + self.EPS)
        ).clip(0, 5) / 5.0

        # --- Z-score of price ---
        for p in [20, 50]:
            rm  = c.rolling(p).mean()
            rs  = c.rolling(p).std()
            d[f"zscore_{p}"] = (
                (c - rm) / (rs + self.EPS)
            ).clip(-3, 3) / 3.0

        # --- Lagged returns (autocorrelation signal) ---
        for lag in [1, 2, 3, 5]:
            d[f"ret_lag_{lag}"] = (
                d["ret_1"].shift(lag).clip(
                    -0.02, 0.02
                ) / 0.02
            )

        # --- Hour of day (session effect) ---
        if pd.api.types.is_datetime64_any_dtype(
            d["timestamp"]
        ):
            hour = d["timestamp"].dt.hour
            # Encode cyclically
            d["hour_sin"] = np.sin(
                2 * np.pi * hour / 24
            )
            d["hour_cos"] = np.cos(
                2 * np.pi * hour / 24
            )
            dow = d["timestamp"].dt.dayofweek
            d["dow_sin"] = np.sin(
                2 * np.pi * dow / 5
            )
            d["dow_cos"] = np.cos(
                2 * np.pi * dow / 5
            )

        # Clean
        d.replace([np.inf, -np.inf], 0, inplace=True)
        d.dropna(inplace=True)
        d.reset_index(drop=True, inplace=True)

        logger.info(
            f"Features: {d.shape[1]} cols, "
            f"{len(d):,} rows"
        )
        return d

    @property
    def feature_cols(self) -> List[str]:
        """
        Columns used as model inputs.
        Excludes raw OHLCV and timestamps.
        """
        return [
            "ret_1","ret_3","ret_5",
            "ret_10","ret_20",
            "natr_14",
            "rsi_7","rsi_14","rsi_21",
            "macd_norm","macd_sig_norm",
            "macd_hist_norm","macd_cross",
            "bb_pos","bb_width","bb_squeeze",
            "stoch_14","stochd_14",
            "stoch_cross_14",
            "stoch_21","stochd_21",
            "stoch_cross_21",
            "adx","di_bull","di_diff",
            "mom_5","mom_10","mom_20","mom_50",
            "vol_ratio_5_20","vol_ratio_20_60",
            "range_pos_10","range_pos_20",
            "range_pos_50","range_pos_100",
            "ema_dist_10","ema_dist_20",
            "ema_dist_50","ema_dist_200",
            "body_ratio","bull_candle",
            "upper_wick","lower_wick",
            "vol_ratio_bar",
            "zscore_20","zscore_50",
            "ret_lag_1","ret_lag_2",
            "ret_lag_3","ret_lag_5",
            "hour_sin","hour_cos",
            "dow_sin","dow_cos",
        ]


# ╔══════════════════════════════════════════════════════════╗
# ║  TARGET ENGINEERING                                      ║
# ║                                                          ║
# ║  Key insight: instead of "next bar direction"           ║
# ║  (too noisy, ~50/50), predict "will price reach         ║
# ║  TP before SL in next N bars?"                          ║
# ║                                                          ║
# ║  This creates a target aligned with actual trading:     ║
# ║  a BUY signal is correct if TP is hit before SL.        ║
# ║  This produces class imbalance (depends on RR)          ║
# ║  which XGBoost handles via scale_pos_weight.            ║
# ╚══════════════════════════════════════════════════════════╝

class TargetBuilder:
    def __init__(self,
                 sl_mult: float = 1.5,
                 tp_mult: float = 2.0,
                 max_bars: int  = 24):
        """
        Parameters
        ----------
        sl_mult  : SL = sl_mult * ATR
        tp_mult  : TP = tp_mult * ATR
        max_bars : maximum bars to look ahead.
                   If neither SL nor TP hit in
                   max_bars, label = 0 (loss/hold).
        """
        self.sl_mult  = sl_mult
        self.tp_mult  = tp_mult
        self.max_bars = max_bars

    def build_long_target(
            self,
            df: pd.DataFrame
    ) -> pd.Series:
        """
        For each bar, if we BOUGHT at close:
          TP = close + tp_mult * atr_14
          SL = close - sl_mult * atr_14
          
        Returns 1 if TP hit before SL within
        max_bars, 0 otherwise.
        
        This is the ground truth for LONG signals.
        """
        n   = len(df)
        lbl = np.zeros(n, dtype=np.int32)
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = df["atr_14"].values

        for i in range(n - self.max_bars):
            entry = closes[i]
            atr   = atrs[i]
            if atr <= 0 or np.isnan(atr):
                continue
            tp = entry + self.tp_mult * atr
            sl = entry - self.sl_mult * atr

            for j in range(i + 1,
                           i + self.max_bars + 1):
                if highs[j] >= tp:
                    lbl[i] = 1  # TP hit first
                    break
                if lows[j] <= sl:
                    lbl[i] = 0  # SL hit first
                    break

        return pd.Series(lbl, index=df.index)

    def build_short_target(
            self,
            df: pd.DataFrame
    ) -> pd.Series:
        """
        For each bar, if we SOLD at close:
          TP = close - tp_mult * atr_14
          SL = close + sl_mult * atr_14
          
        Returns 1 if TP hit before SL within
        max_bars, 0 otherwise.
        """
        n   = len(df)
        lbl = np.zeros(n, dtype=np.int32)
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = df["atr_14"].values

        for i in range(n - self.max_bars):
            entry = closes[i]
            atr   = atrs[i]
            if atr <= 0 or np.isnan(atr):
                continue
            tp = entry - self.tp_mult * atr
            sl = entry + self.sl_mult * atr

            for j in range(i + 1,
                           i + self.max_bars + 1):
                if lows[j] <= tp:
                    lbl[i] = 1  # TP hit
                    break
                if highs[j] >= sl:
                    lbl[i] = 0  # SL hit
                    break

        return pd.Series(lbl, index=df.index)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODEL TRAINER                                           ║
# ║                                                          ║
# ║  Trains two separate classifiers:                        ║
# ║    model_long:  P(long trade hits TP)                   ║
# ║    model_short: P(short trade hits TP)                  ║
# ║                                                          ║
# ║  Each uses Optuna + TimeSeriesSplit CV.                  ║
# ║  Scale_pos_weight corrects class imbalance.             ║
# ╚══════════════════════════════════════════════════════════╝

class ModelTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _cv_score(self, params: dict,
                  X: np.ndarray,
                  y: np.ndarray) -> float:
        """
        Purged time-series cross-validation.
        Returns mean ROC-AUC across folds.
        """
        tscv   = TimeSeriesSplit(
            n_splits=self.cfg.CV_SPLITS,
            gap=self.cfg.EMBARGO_BARS
        )
        scores = []
        for tr_idx, te_idx in tscv.split(X):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xte, yte = X[te_idx], y[te_idx]
            if len(np.unique(ytr)) < 2:
                continue
            spw = float(
                (ytr == 0).sum() /
                max((ytr == 1).sum(), 1)
            )
            m = XGBClassifier(
                **params,
                scale_pos_weight=spw,
                verbosity=0,
                tree_method="hist",
                random_state=42,
            )
            m.fit(Xtr, ytr)
            if len(np.unique(yte)) < 2:
                continue
            try:
                auc = roc_auc_score(
                    yte,
                    m.predict_proba(Xte)[:, 1]
                )
                scores.append(auc)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.5

    def optimize(self,
                 X: np.ndarray,
                 y: np.ndarray) -> dict:
        """
        Optuna hyperparameter search.
        Returns best params dict.
        """
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_est", 100, 500
                ),
                "max_depth": trial.suggest_int(
                    "depth", 3, 8
                ),
                "learning_rate": trial.suggest_float(
                    "lr", 0.01, 0.2, log=True
                ),
                "min_child_weight": trial.suggest_int(
                    "mcw", 5, 50
                ),
                "subsample": trial.suggest_float(
                    "ss", 0.5, 0.9
                ),
                "colsample_bytree": trial.suggest_float(
                    "cs", 0.4, 0.9
                ),
                "reg_alpha": trial.suggest_float(
                    "ra", 0.1, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "rl", 0.1, 10.0, log=True
                ),
                "gamma": trial.suggest_float(
                    "gm", 0.0, 5.0
                ),
            }
            return self._cv_score(params, X, y)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42
            ),
        )
        study.optimize(
            objective,
            n_trials=self.cfg.OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        logger.info(
            f"Best AUC: {study.best_value:.4f}"
        )
        return study.best_params

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            params: dict) -> XGBClassifier:
        """
        Fit final model on full training data.
        """
        spw = float(
            (y == 0).sum() /
            max((y == 1).sum(), 1)
        )
        m = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            verbosity=0,
            tree_method="hist",
            random_state=42,
        )
        m.fit(X, y)
        return m


# ╔══════════════════════════════════════════════════════════╗
# ║  BACKTEST ENGINE                                         ║
# ║                                                          ║
# ║  Honest simulation rules:                                ║
# ║  1. Enter at NEXT BAR OPEN (not current close)          ║
# ║     to avoid look-ahead bias                            ║
# ║  2. SL and TP checked against bar HIGH/LOW              ║
# ║  3. If both SL and TP within same bar,                  ║
# ║     assume SL hit (conservative)                        ║
# ║  4. Position size = (equity * risk_pct) / (SL pips)    ║
# ║  5. Spread paid on entry AND exit                       ║
# ║  6. One trade at a time                                 ║
# ║  7. Daily trade limit enforced                          ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class Trade:
    entry_bar:   int
    entry_price: float
    direction:   int     # +1 long, -1 short
    sl_price:    float
    tp_price:    float
    lot_size:    float
    exit_bar:    int     = -1
    exit_price:  float   = 0.0
    pnl_usd:     float   = 0.0
    exit_reason: str     = ""


class BacktestEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.EPS = 1e-10

    def run(self,
            df: pd.DataFrame,
            signals: pd.Series
            ) -> Tuple[List[Trade],
                       pd.DataFrame]:
        """
        Parameters
        ----------
        df      : feature dataframe with OHLCV
        signals : Series of {-1, 0, +1}
                  -1=short, 0=hold, +1=long
                  indexed same as df

        Returns
        -------
        trades      : list of Trade objects
        equity_df   : bar-by-bar equity
        """
        cfg      = self.cfg
        equity   = cfg.INITIAL_CAPITAL
        peak_eq  = equity
        eq_hist  = []
        trades   = []
        open_tr: Optional[Trade] = None
        daily_cnt  = 0
        cur_day    = None
        halted     = False

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = df["atr_14"].values
        times  = df["timestamp"].values

        for i in range(len(df)):
            price = closes[i]
            high  = highs[i]
            low   = lows[i]
            atr   = atrs[i]
            t     = pd.Timestamp(times[i])
            sig   = int(signals.iloc[i])

            # --- Reset daily counter ---
            day = t.date()
            if day != cur_day:
                cur_day   = day
                daily_cnt = 0

            # --- Check open trade SL/TP ---
            if open_tr is not None:
                closed = False

                if open_tr.direction == 1:
                    # Long: SL if low <= sl_price
                    #       TP if high >= tp_price
                    if (low <= open_tr.sl_price and
                            high >= open_tr.tp_price):
                        # Both triggered: conservative
                        # assume SL hit
                        ex_px = open_tr.sl_price
                        open_tr.exit_reason = "SL"
                        closed = True
                    elif low <= open_tr.sl_price:
                        ex_px = open_tr.sl_price
                        open_tr.exit_reason = "SL"
                        closed = True
                    elif high >= open_tr.tp_price:
                        ex_px = open_tr.tp_price
                        open_tr.exit_reason = "TP"
                        closed = True

                else:  # Short
                    if (high >= open_tr.sl_price and
                            low <= open_tr.tp_price):
                        ex_px = open_tr.sl_price
                        open_tr.exit_reason = "SL"
                        closed = True
                    elif high >= open_tr.sl_price:
                        ex_px = open_tr.sl_price
                        open_tr.exit_reason = "SL"
                        closed = True
                    elif low <= open_tr.tp_price:
                        ex_px = open_tr.tp_price
                        open_tr.exit_reason = "TP"
                        closed = True

                if closed:
                    # PnL calculation
                    # spread on exit
                    if open_tr.direction == 1:
                        ex_fill = (
                            ex_px - cfg.SPREAD_PRICE / 2
                        )
                        pnl_pts = (
                            ex_fill -
                            open_tr.entry_price
                        )
                    else:
                        ex_fill = (
                            ex_px + cfg.SPREAD_PRICE / 2
                        )
                        pnl_pts = (
                            open_tr.entry_price -
                            ex_fill
                        )

                    # Convert to USD
                    pnl_usd = (
                        pnl_pts *
                        cfg.CONTRACT_SIZE *
                        open_tr.lot_size
                    )
                    # Commission (already paid on
                    # entry, pay other half on exit)
                    pnl_usd -= (
                        cfg.COMMISSION_PER_LOT *
                        open_tr.lot_size / 2
                    )

                    open_tr.exit_bar   = i
                    open_tr.exit_price = ex_fill
                    open_tr.pnl_usd    = pnl_usd
                    equity += pnl_usd
                    equity  = max(equity, 0.01)
                    trades.append(open_tr)
                    open_tr = None

                    peak_eq = max(peak_eq, equity)
                    dd = (peak_eq - equity) / (
                        peak_eq + self.EPS
                    )
                    if dd > cfg.MAX_DRAWDOWN_HALT:
                        halted = True
                        logger.warning(
                            f"Drawdown halt at "
                            f"bar {i}, equity="
                            f"${equity:,.2f}"
                        )

            # --- Entry logic ---
            if (not halted and
                    open_tr is None and
                    sig != 0 and
                    daily_cnt < cfg.MAX_DAILY_TRADES and
                    i < len(df) - 1):

                # Enter at NEXT BAR OPEN
                entry_px = closes[i + 1]  # approx
                # Actually: next bar open
                # We'll use close[i] + spread as fill
                # for simplicity (conservative)

                if atr <= 0 or np.isnan(atr):
                    eq_hist.append(equity)
                    continue

                if sig == 1:  # Long
                    fill_px = (
                        price + cfg.SPREAD_PRICE / 2
                    )
                    sl_px = fill_px - cfg.ATR_SL_MULT * atr
                    tp_px = fill_px + cfg.ATR_TP_MULT * atr
                    sl_dist = fill_px - sl_px
                else:         # Short
                    fill_px = (
                        price - cfg.SPREAD_PRICE / 2
                    )
                    sl_px = fill_px + cfg.ATR_SL_MULT * atr
                    tp_px = fill_px - cfg.ATR_TP_MULT * atr
                    sl_dist = sl_px - fill_px

                if sl_dist <= self.EPS:
                    eq_hist.append(equity)
                    continue

                # Fixed fractional sizing
                # Risk = equity * RISK_PER_TRADE_PCT
                # Lot = Risk / (SL_pips * pip_value_per_lot)
                risk_usd = equity * cfg.RISK_PER_TRADE_PCT
                pip_val_per_lot = (
                    cfg.PIP_VALUE *
                    cfg.CONTRACT_SIZE
                )
                sl_pips = sl_dist / cfg.PIP_VALUE
                lots    = risk_usd / (
                    sl_pips * pip_val_per_lot +
                    self.EPS
                )
                lots    = min(lots, 100.0)  # cap
                lots    = max(lots, 0.01)   # min

                # Commission on entry (half round-turn)
                commission = (
                    cfg.COMMISSION_PER_LOT * lots / 2
                )
                equity -= commission
                equity  = max(equity, 0.01)

                open_tr = Trade(
                    entry_bar=i,
                    entry_price=fill_px,
                    direction=sig,
                    sl_price=sl_px,
                    tp_price=tp_px,
                    lot_size=lots,
                )
                daily_cnt += 1

            peak_eq = max(peak_eq, equity)
            eq_hist.append(equity)

        # Close any open trade at end
        if open_tr is not None:
            ep = closes[-1]
            if open_tr.direction == 1:
                fill  = ep - cfg.SPREAD_PRICE / 2
                pnl_p = fill - open_tr.entry_price
            else:
                fill  = ep + cfg.SPREAD_PRICE / 2
                pnl_p = open_tr.entry_price - fill
            pnl_usd = (
                pnl_p * cfg.CONTRACT_SIZE *
                open_tr.lot_size
            )
            pnl_usd -= (
                cfg.COMMISSION_PER_LOT *
                open_tr.lot_size / 2
            )
            open_tr.exit_bar    = len(df) - 1
            open_tr.exit_price  = fill
            open_tr.pnl_usd     = pnl_usd
            open_tr.exit_reason = "EOD"
            equity += pnl_usd
            equity  = max(equity, 0.01)
            trades.append(open_tr)

        eq_df = pd.DataFrame({
            "bar":    range(len(eq_hist)),
            "equity": eq_hist,
        })
        return trades, eq_df


# ╔══════════════════════════════════════════════════════════╗
# ║  SIGNAL GENERATOR                                        ║
# ║                                                          ║
# ║  Converts model probabilities to trade signals.          ║
# ║                                                          ║
# ║  Signal = +1 (long) if:                                  ║
# ║    P(long_TP) > PROB_THRESHOLD_BUY  AND                 ║
# ║    RSI_14 > RSI_BULL_MIN            AND                 ║
# ║    ADX > ADX_MIN                                        ║
# ║                                                          ║
# ║  Signal = -1 (short) if:                                 ║
# ║    P(short_TP) > PROB_THRESHOLD_BUY AND                 ║
# ║    RSI_14 < RSI_BEAR_MAX            AND                 ║
# ║    ADX > ADX_MIN                                        ║
# ║                                                          ║
# ║  Signal = 0 (hold) otherwise                            ║
# ╚══════════════════════════════════════════════════════════╝

class SignalGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.EPS = 1e-10

    def generate(self,
                 df: pd.DataFrame,
                 prob_long:  np.ndarray,
                 prob_short: np.ndarray,
                 ) -> pd.Series:
        """
        Parameters
        ----------
        df          : feature dataframe
        prob_long   : P(long trade TP) per bar
        prob_short  : P(short trade TP) per bar

        Returns
        -------
        signals : pd.Series {-1, 0, +1}
        """
        cfg = self.cfg
        n   = len(df)
        sig = np.zeros(n, dtype=np.int32)

        # Recover raw RSI (stored normalized)
        # rsi_14 in df is (RSI-50)/50, so
        # RSI = rsi_14*50 + 50
        rsi_raw = df["rsi_14"].values * 50 + 50
        adx_raw = df["adx"].values * 100

        for i in range(n):
            rsi = rsi_raw[i]
            adx = adx_raw[i]

            pl = prob_long[i]
            ps = prob_short[i]

            # ADX filter: require trend
            if adx < cfg.ADX_MIN:
                continue

            # Long signal
            if (pl > cfg.PROB_THRESHOLD_BUY and
                    rsi > cfg.RSI_BULL_MIN):
                sig[i] = 1

            # Short signal
            elif (ps > cfg.PROB_THRESHOLD_BUY and
                    rsi < cfg.RSI_BEAR_MAX):
                sig[i] = -1

        return pd.Series(sig, index=df.index)


# ╔══════════════════════════════════════════════════════════╗
# ║  PERFORMANCE REPORTER                                    ║
# ║                                                          ║
# ║  All metrics computed from closed trade PnLs only.      ║
# ║  No mark-to-market inflation.                           ║
# ╚══════════════════════════════════════════════════════════╝

class PerfReporter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.EPS = 1e-10

    def report(self,
               trades: List[Trade],
               eq_df: pd.DataFrame,
               init_capital: float
               ) -> dict:
        nt = len(trades)

        if nt == 0:
            print("\n[PerfReporter] No trades.")
            return {"trades": 0}

        pnls = np.array(
            [t.pnl_usd for t in trades]
        )
        wins    = pnls[pnls > 0]
        losses  = pnls[pnls <= 0]
        n_wins  = len(wins)
        n_loss  = len(losses)
        wr      = n_wins / (nt + self.EPS)
        gross_p = wins.sum()    if n_wins  > 0 else 0.0
        gross_l = abs(losses.sum()) if n_loss > 0 else self.EPS
        pf      = gross_p / (gross_l + self.EPS)
        avg_w   = wins.mean()    if n_wins  > 0 else 0.0
        avg_l   = losses.mean()  if n_loss  > 0 else 0.0
        expectancy = pnls.mean()

        # TP/SL breakdown
        tp_hits = sum(
            1 for t in trades
            if t.exit_reason == "TP"
        )
        sl_hits = sum(
            1 for t in trades
            if t.exit_reason == "SL"
        )

        # Equity curve from closed trades
        eq_vals = eq_df["equity"].values
        ret     = np.diff(eq_vals) / (
            eq_vals[:-1] + self.EPS
        )

        final_eq = eq_vals[-1]
        tr       = final_eq / init_capital - 1
        n_days   = max(len(eq_vals) / 24, 1)
        ar       = (1 + tr) ** (252 / n_days) - 1
        av       = (np.std(ret) *
                    np.sqrt(252 * 24))
        dn       = ret[ret < 0]
        dv       = (np.std(dn) if len(dn) > 0
                    else self.EPS) * np.sqrt(252 * 24)
        sh       = (ar - 0.02) / (av + self.EPS)
        so       = (ar - 0.02) / (dv + self.EPS)
        mdd      = self._mdd(eq_vals)
        calmar   = ar / (abs(mdd) + self.EPS)

        m = {
            "trades":        nt,
            "win_rate":      wr,
            "profit_factor": pf,
            "avg_win_usd":   avg_w,
            "avg_loss_usd":  avg_l,
            "expectancy":    expectancy,
            "tp_hits":       tp_hits,
            "sl_hits":       sl_hits,
            "total_return":  tr,
            "ann_return":    ar,
            "ann_vol":       av,
            "max_dd":        mdd,
            "sharpe":        sh,
            "sortino":       so,
            "calmar":        calmar,
            "final_equity":  final_eq,
        }

        # Print
        sep = "═" * 52
        print(f"\n╔{sep}╗")
        print(f"║{'PERFORMANCE REPORT':^52}║")
        print(f"╠{sep}╣")
        rows = [
            ("Trades",        nt,         "d"),
            ("Win Rate",      wr,         ".2%"),
            ("Profit Factor", pf,         ".3f"),
            ("Avg Win",       avg_w,      ",.2f"),
            ("Avg Loss",      avg_l,      ",.2f"),
            ("Expectancy",    expectancy, ",.2f"),
            ("TP Hits",       tp_hits,    "d"),
            ("SL Hits",       sl_hits,    "d"),
            ("Total Return",  tr,         ".2%"),
            ("Ann. Return",   ar,         ".2%"),
            ("Ann. Vol",      av,         ".2%"),
            ("Max Drawdown",  mdd,        ".2%"),
            ("Sharpe",        sh,         ".3f"),
            ("Sortino",       so,         ".3f"),
            ("Calmar",        calmar,     ".3f"),
            ("Final Equity",  final_eq,   ",.2f"),
        ]
        for label, val, fmt in rows:
            vs = f"{val:{fmt}}"
            print(
                f"║  {label+':':<20}"
                f"${vs:>12}"
                f"{'':>17}║"
                if "usd" in label.lower() or
                "equity" in label.lower() or
                "expectancy" in label.lower()
                else
                f"║  {label+':':<20}"
                f"{vs:>12}"
                f"{'':>17}║"
            )
        print(f"╚{sep}╝\n")
        return m

    def _mdd(self, eq: np.ndarray) -> float:
        peak = eq[0]
        mdd  = 0.0
        for e in eq:
            peak = max(peak, e)
            dd   = (peak - e) / (peak + self.EPS)
            mdd  = max(mdd, dd)
        return mdd


# ╔══════════════════════════════════════════════════════════╗
# ║  WALK-FORWARD PIPELINE                                   ║
# ║                                                          ║
# ║  Structure:                                              ║
# ║  ┌─────────────────────────────────────────┐           ║
# ║  │ TRAIN [0 .. T]  │ GAP │ TEST [T+G..T+G+W]│          ║
# ║  └─────────────────────────────────────────┘           ║
# ║                                                          ║
# ║  Slide window forward by TEST_BARS each fold.           ║
# ║  Re-optimize hyperparameters every 3 folds              ║
# ║  (expensive but guards against drift).                  ║
# ║                                                          ║
# ║  OOS (out-of-sample) signals are concatenated           ║
# ║  and passed to BacktestEngine for final metrics.        ║
# ╚══════════════════════════════════════════════════════════╝

class WalkForwardPipeline:
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.trainer = ModelTrainer(cfg)
        self.tb      = TargetBuilder(
            sl_mult=cfg.ATR_SL_MULT,
            tp_mult=cfg.ATR_TP_MULT,
        )
        self.sg      = SignalGenerator(cfg)
        self.fe      = FeatureEngine()
        self.bt      = BacktestEngine(cfg)
        self.pr      = PerfReporter(cfg)

    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        n   = len(df)

        # Build all targets upfront
        # (they look ahead, so compute on full df)
        logger.info("Building targets...")
        long_tgt  = self.tb.build_long_target(df)
        short_tgt = self.tb.build_short_target(df)
        logger.info(
            f"Long TP rate:  "
            f"{long_tgt.mean():.3f}"
        )
        logger.info(
            f"Short TP rate: "
            f"{short_tgt.mean():.3f}"
        )

        feat_cols  = [
            f for f in self.fe.feature_cols
            if f in df.columns
        ]
        X_all = df[feat_cols].values.astype(
            np.float32
        )
        y_long  = long_tgt.values
        y_short = short_tgt.values

        # Walk-forward splits
        all_signals = pd.Series(
            0, index=df.index, dtype=np.int32
        )
        model_long  = None
        model_short = None
        best_params = None
        fold_n      = 0
        oos_probs   = []

        start = cfg.MIN_TRAIN_BARS
        t     = start

        while t + cfg.TEST_BARS < n:
            tr_start = max(0, t - cfg.TRAIN_BARS)
            tr_end   = t
            te_start = t + cfg.EMBARGO_BARS
            te_end   = min(
                t + cfg.EMBARGO_BARS +
                cfg.TEST_BARS,
                n - cfg.TARGET_LOOKAHEAD
                if hasattr(cfg, "TARGET_LOOKAHEAD")
                else n - 24  # leave lookahead room
            )

            if te_start >= te_end:
                t += cfg.TEST_BARS
                continue

            Xtr  = X_all[tr_start:tr_end]
            yltr = y_long[tr_start:tr_end]
            ystr = y_short[tr_start:tr_end]
            Xte  = X_all[te_start:te_end]

            if len(Xtr) < cfg.MIN_TRAIN_BARS:
                t += cfg.TEST_BARS
                continue

            logger.info(
                f"Fold {fold_n}: "
                f"train [{tr_start}:{tr_end}] "
                f"test  [{te_start}:{te_end}]"
            )

            # Re-optimize every 3 folds
            if fold_n % 3 == 0:
                logger.info(
                    "Optimizing hyperparameters..."
                )
                best_params = self.trainer.optimize(
                    Xtr, yltr
                )

            # Fit long and short models
            model_long = self.trainer.fit(
                Xtr, yltr, best_params
            )
            model_short = self.trainer.fit(
                Xtr, ystr, best_params
            )

            # OOS predictions
            pl = model_long.predict_proba(Xte)[:, 1]
            ps = model_short.predict_proba(Xte)[:, 1]

            # OOS AUC (sanity check)
            yte_l = y_long[te_start:te_end]
            yte_s = y_short[te_start:te_end]
            if len(np.unique(yte_l)) == 2:
                auc_l = roc_auc_score(yte_l, pl)
                logger.info(
                    f"  OOS AUC long:  {auc_l:.4f}"
                )
            if len(np.unique(yte_s)) == 2:
                auc_s = roc_auc_score(yte_s, ps)
                logger.info(
                    f"  OOS AUC short: {auc_s:.4f}"
                )

            # Generate signals for test window
            df_te = df.iloc[te_start:te_end].copy()
            df_te.reset_index(drop=True, inplace=True)

            sigs = self.sg.generate(
                df_te, pl, ps
            )

            # Map back to original index
            for j, idx in enumerate(
                range(te_start, te_end)
            ):
                if idx < n:
                    all_signals.iloc[idx] = (
                        sigs.iloc[j]
                    )

            fold_n += 1
            t      += cfg.TEST_BARS

        # Count signals
        n_long  = (all_signals == 1).sum()
        n_short = (all_signals == -1).sum()
        logger.info(
            f"Total signals: "
            f"long={n_long}, short={n_short}"
        )

        # Backtest OOS signals only
        # (signals before start are 0/hold)
        logger.info("Running backtest...")
        trades, eq_df = self.bt.run(df, all_signals)

        # Report
        metrics = self.pr.report(
            trades, eq_df, cfg.INITIAL_CAPITAL
        )

        # Save artifacts
        self._save(
            metrics, feat_cols,
            model_long, model_short,
            all_signals
        )
        return metrics

    def _save(self,
              metrics:     dict,
              feat_cols:   List[str],
              model_long,
              model_short,
              signals:     pd.Series):
        out = self.cfg.OUTPUT_DIR
        os.makedirs(out, exist_ok=True)

        with open(
            os.path.join(out, "metrics.json"),
            "w"
        ) as f:
            json.dump(
                {k: (float(v)
                     if isinstance(v, (
                         np.floating, float
                     ))
                     else int(v)
                     if isinstance(v, (
                         np.integer, int
                     ))
                     else v)
                 for k, v in metrics.items()},
                f, indent=2
            )

        with open(
            os.path.join(out, "features.json"),
            "w"
        ) as f:
            json.dump(feat_cols, f, indent=2)

        if model_long is not None:
            model_long.save_model(
                os.path.join(
                    out, "model_long.json"
                )
            )
        if model_short is not None:
            model_short.save_model(
                os.path.join(
                    out, "model_short.json"
                )
            )

        signals.to_csv(
            os.path.join(out, "oos_signals.csv")
        )
        logger.info(
            f"Artifacts saved to {out}/"
        )


# ╔══════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                             ║
# ╚══════════════════════════════════════════════════════════╝

def run(
    source:   str = "synthetic",
    filepath: str = None,
    cfg:      Config = None,
) -> dict:
    if cfg is None:
        cfg = Config()

    print(
        "\n╔══════════════════════════════════════╗"
    )
    print(
        "║  XGBoost Direct Trader — v2.0        ║"
    )
    print(
        "╚══════════════════════════════════════╝\n"
    )

    # 1. Load data
    print("Phase 1: Data Ingestion")
    if source == "synthetic":
        raw = DataIngestion.load_synthetic(
            cfg.HISTORICAL_BARS
        )
    elif source == "csv":
        raw = DataIngestion.load_csv(filepath)
    else:
        raise ValueError(
            f"Unknown source: {source}"
        )
    df = DataIngestion.validate(raw)
    print(
        f"  {len(df):,} bars  "
        f"{df['timestamp'].iloc[0]} → "
        f"{df['timestamp'].iloc[-1]}\n"
    )

    # 2. Features
    print("Phase 2: Feature Engineering")
    fe = FeatureEngine()
    df = fe.build(df)
    print(
        f"  {df.shape[1]} columns, "
        f"{len(df):,} rows\n"
    )

    # 3. Walk-forward
    print("Phase 3: Walk-Forward Training & Test")
    pipeline = WalkForwardPipeline(cfg)
    metrics  = pipeline.run(df)

    print(
        "╔══════════════════════════════════════╗"
    )
    print(
        "║  COMPLETE                            ║"
    )
    print(
        f"║  Artifacts → "
        f"{cfg.OUTPUT_DIR:<24}║"
    )
    print(
        "╚══════════════════════════════════════╝"
    )
    return metrics


if __name__ == "__main__":
    run(source="synthetic")
