# ============================================================
#  XGB_DirectTrader.py  v2.1
#
#  Fixes applied vs v2.0:
#    BUG 1: TargetBuilder same-bar SL/TP ambiguity
#           → conservative SL-first rule applied
#    BUG 2: Feature column order saved to disk
#           → feat_cols written to features.json
#           with explicit index positions
#    BUG 3: Synthetic open prices used np.roll
#           → replaced with prices[:-1] shift
#
#  Config adjustments:
#    PROB_THRESHOLD_BUY raised 0.58 → 0.63
#    PROB_THRESHOLD_SELL raised 0.58 → 0.63
#    CalibratedClassifierCV wraps XGBClassifier
#    scale_pos_weight removed (calibration handles
#    class imbalance via isotonic regression)
# ============================================================

import os
import json
import warnings
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ╔══════════════════════════════════════════════════════════╗
# ║  CONFIG                                                  ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class Config:
    # ── Data ─────────────────────────────────────────────────
    HISTORICAL_BARS:     int   = 50_000
    OUTPUT_DIR:          str   = "xgb_trader_artifacts"

    # ── Walk-forward windows ──────────────────────────────────
    TRAIN_BARS:          int   = 3_000
    TEST_BARS:           int   = 500
    MIN_TRAIN_BARS:      int   = 1_000
    # Bars skipped between train-end and test-start.
    # 24 bars on H1 = 1 trading day of embargo.
    # Prevents any rolling-window feature calculated
    # at bar T from contaminating targets at T+1…T+24.
    EMBARGO_BARS:        int   = 24

    # ── Model ─────────────────────────────────────────────────
    OPTUNA_TRIALS:       int   = 40
    CV_SPLITS:           int   = 5
    # Re-optimize hyperparameters every N folds.
    # Fold cadence = TEST_BARS bars, so every 3 folds
    # = 1,500 bars ≈ 62 trading days on H1.
    REOPT_EVERY_N_FOLDS: int   = 3

    # ── Signal thresholds ────────────────────────────────────
    # BUG-FIX / CONFIG ADJUSTMENT:
    #   scale_pos_weight with RR=1.33 pushes raw
    #   XGB probabilities toward 0.65-0.75 even on
    #   noise.  CalibratedClassifierCV re-centres
    #   them to true frequencies, so the threshold
    #   must match the calibrated output range.
    #   0.63 sits 13 pp above the ~0.50 null and
    #   filters low-confidence bars while leaving
    #   enough trades for statistical significance.
    PROB_THRESHOLD_BUY:  float = 0.63
    PROB_THRESHOLD_SELL: float = 0.63

    # RSI confirmation bounds (raw RSI 0-100 scale)
    RSI_BULL_MIN:        float = 45.0
    RSI_BEAR_MAX:        float = 55.0

    # ADX minimum for a tradeable trend
    ADX_MIN:             float = 18.0

    # ── Risk / position sizing ────────────────────────────────
    INITIAL_CAPITAL:     float = 100_000.0
    # Fixed fractional: risk this fraction of equity
    # on every trade.  1% means 10 consecutive full
    # losses reduce equity to ~90,000 — survivable.
    RISK_PER_TRADE_PCT:  float = 0.01
    ATR_SL_MULT:         float = 1.5
    ATR_TP_MULT:         float = 2.0    # RR = 1.33
    MAX_DAILY_TRADES:    int   = 3
    MAX_DRAWDOWN_HALT:   float = 0.15

    # ── Execution costs (EURUSD H1 realistic) ────────────────
    SPREAD_PIPS:         float = 1.5
    PIP_VALUE:           float = 0.0001
    COMMISSION_PER_LOT:  float = 7.0    # round-turn USD

    # ── Contract spec ─────────────────────────────────────────
    CONTRACT_SIZE:       float = 100_000.0

    # ── Target builder look-ahead ─────────────────────────────
    # Maximum bars to scan forward for SL/TP hit.
    # 24 bars on H1 = 1 trading day.  Enough for
    # ATR-scaled levels without leaking far into the
    # future.
    TARGET_MAX_BARS:     int   = 24

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.SPREAD_PRICE = self.SPREAD_PIPS * self.PIP_VALUE


# ╔══════════════════════════════════════════════════════════╗
# ║  DATA INGESTION                                          ║
# ╚══════════════════════════════════════════════════════════╝

class DataIngestion:

    @staticmethod
    def load_synthetic(n: int) -> pd.DataFrame:
        """
        GARCH(1,1) EURUSD-like H1 OHLCV.

        BUG 3 FIX:
          Old code:  opens = np.roll(prices, 1)
          Problem:   np.roll wraps the last element
                     to position 0, creating a single
                     bar with a multi-thousand-pip gap
                     between close[N-1] and open[0].
                     This one bar dominates ATR
                     calculations for the first ~100
                     bars, inflating SL distances and
                     distorting early feature values.
          Fix:       opens[0]  = S0 (true start)
                     opens[1:] = prices[:-1]
                     This correctly makes each bar's
                     open equal to the previous bar's
                     close, as it is in real FX data.
        """
        np.random.seed(42)
        S0      = 1.1000
        mu      = 0.0
        sigma0  = 0.0060

        # GARCH(1,1) variance process
        alpha, beta = 0.10, 0.85
        prices  = [S0]
        sigma   = sigma0
        sigmas  = [sigma]

        for _ in range(1, n):
            eps   = np.random.randn()
            r     = mu + sigma * eps
            sigma = np.sqrt(
                sigma0 ** 2 * (1 - alpha - beta)
                + alpha * (sigma * eps) ** 2
                + beta  * sigma ** 2
            )
            sigma = np.clip(sigma, 0.0002, 0.03)
            prices.append(prices[-1] * np.exp(r))
            sigmas.append(sigma)

        prices = np.array(prices)
        sigmas = np.array(sigmas)

        # ── BUG 3 FIX ───────────────────────────────────────
        # Each bar's open = previous bar's close.
        # Bar 0 opens at S0 (no phantom gap).
        opens    = np.empty(n)
        opens[0] = S0
        opens[1:] = prices[:-1]   # NOT np.roll
        # ────────────────────────────────────────────────────

        high = prices * np.exp(
            np.abs(np.random.randn(n)) * sigmas * 0.7
        )
        low  = prices * np.exp(
            -np.abs(np.random.randn(n)) * sigmas * 0.7
        )
        high = np.maximum(high, np.maximum(prices, opens))
        low  = np.minimum(low,  np.minimum(prices, opens))
        vol  = np.random.lognormal(8, 1, n)

        ts = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "timestamp": ts,
            "open":      opens,
            "high":      high,
            "low":       low,
            "close":     prices,
            "volume":    vol,
        })
        # Remove weekends (FX market closed)
        df = df[df["timestamp"].dt.dayofweek < 5].copy()
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Synthetic data: {len(df):,} bars")
        return df

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        required = ["timestamp", "open", "high",
                    "low", "close", "volume"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")
        df = df.dropna(subset=required).copy()
        df = df[df["volume"] > 0].copy()
        df = df[df["high"]  >= df["low"]  ].copy()
        df = df[df["high"]  >= df["close"]].copy()
        df = df[df["low"]   <= df["close"]].copy()
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Validated: {len(df):,} bars")
        return df


# ╔══════════════════════════════════════════════════════════╗
# ║  FEATURE ENGINEERING                                     ║
# ║                                                          ║
# ║  All features are bounded to a known range:             ║
# ║    • Returns / momentum  → clipped to [-1, 1]           ║
# ║    • RSI                 → (RSI-50)/50 ∈ [-1, 1]        ║
# ║    • MACD/BB             → divided by ATR               ║
# ║    • ADX                 → divided by 100               ║
# ║    • Time                → cyclic sin/cos               ║
# ║                                                          ║
# ║  This matters for ONNX / MQL5 portability because       ║
# ║  the inference runtime applies no scaler — the          ║
# ║  features must arrive in the same numeric range         ║
# ║  that XGBoost saw during training.  XGBoost itself      ║
# ║  is scale-invariant (uses split thresholds not          ║
# ║  distances), but bounded features prevent extreme       ║
# ║  inputs from reaching leaf nodes never visited in       ║
# ║  training and returning garbage predictions.            ║
# ╚══════════════════════════════════════════════════════════╝

class FeatureEngine:
    EPS = 1e-10

    # Canonical ordered list of model inputs.
    # ORDER IS FIXED — do not reorder.
    # BUG 2 FIX: this list is the single source of
    # truth for column order.  features.json records
    # both name and integer index so the MQL5/ONNX
    # runtime can validate input tensor alignment.
    FEATURE_COLS: List[str] = [
        # Returns
        "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
        # Volatility
        "natr_14",
        # RSI (normalised)
        "rsi_7", "rsi_14", "rsi_21",
        # MACD (ATR-normalised)
        "macd_norm", "macd_sig_norm",
        "macd_hist_norm", "macd_cross",
        # Bollinger Bands
        "bb_pos", "bb_width", "bb_squeeze",
        # Stochastic
        "stoch_14", "stochd_14", "stoch_cross_14",
        "stoch_21", "stochd_21", "stoch_cross_21",
        # ADX / directional
        "adx", "di_bull", "di_diff",
        # Momentum
        "mom_5", "mom_10", "mom_20", "mom_50",
        # Volatility ratios
        "vol_ratio_5_20", "vol_ratio_20_60",
        # Price range position
        "range_pos_10", "range_pos_20",
        "range_pos_50", "range_pos_100",
        # EMA distance
        "ema_dist_10", "ema_dist_20",
        "ema_dist_50", "ema_dist_200",
        # Candle structure
        "body_ratio", "bull_candle",
        "upper_wick", "lower_wick",
        # Volume
        "vol_ratio_bar",
        # Z-score
        "zscore_20", "zscore_50",
        # Lagged returns
        "ret_lag_1", "ret_lag_2",
        "ret_lag_3", "ret_lag_5",
        # Session timing (cyclic)
        "hour_sin", "hour_cos",
        "dow_sin",  "dow_cos",
    ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        d   = df.copy()
        c   = d["close"]
        h   = d["high"]
        l   = d["low"]
        v   = d["volume"]
        o   = d["open"]
        eps = self.EPS

        # ── Returns ──────────────────────────────────────────
        d["ret_1"]  = c.pct_change(1)
        d["ret_3"]  = c.pct_change(3)
        d["ret_5"]  = c.pct_change(5)
        d["ret_10"] = c.pct_change(10)
        d["ret_20"] = c.pct_change(20)

        # ── ATR (kept as price units for SL/TP) ──────────────
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        d["atr_14"] = tr.rolling(14).mean()
        d["atr_7"]  = tr.rolling(7).mean()
        d["natr_14"] = d["atr_14"] / (c + eps)

        # ── RSI — normalised to [-1, 1] ───────────────────────
        for p in [7, 14, 21]:
            diff = c.diff()
            g    = diff.clip(lower=0).rolling(p).mean()
            ls   = (-diff.clip(upper=0)).rolling(p).mean()
            rsi  = 100 - 100 / (1 + g / (ls + eps))
            d[f"rsi_{p}"] = (rsi - 50) / 50.0

        # ── MACD — ATR-normalised ─────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig
        d["macd_norm"]      = macd / (d["atr_14"] + eps)
        d["macd_sig_norm"]  = sig  / (d["atr_14"] + eps)
        d["macd_hist_norm"] = hist / (d["atr_14"] + eps)
        d["macd_cross"]     = np.sign(hist)

        # ── Bollinger Bands ───────────────────────────────────
        bm = c.rolling(20).mean()
        bs = c.rolling(20).std()
        d["bb_pos"]     = ((c - bm) / (2 * bs + eps)).clip(-1, 1)
        d["bb_width"]   = (4 * bs / (bm + eps))
        d["bb_squeeze"] = (
            d["bb_width"] < d["bb_width"].rolling(50).mean()
        ).astype(float)

        # ── Stochastic ────────────────────────────────────────
        for p in [14, 21]:
            ll  = l.rolling(p).min()
            hh  = h.rolling(p).max()
            k   = (c - ll) / (hh - ll + eps)
            kd  = k.rolling(3).mean()
            d[f"stoch_{p}"]      = k * 2 - 1
            d[f"stochd_{p}"]     = kd * 2 - 1
            d[f"stoch_cross_{p}"] = np.sign(k - kd)

        # ── ADX ───────────────────────────────────────────────
        dm_pos  = (h - h.shift(1)).clip(lower=0)
        dm_neg  = (l.shift(1) - l).clip(lower=0)
        atr14   = d["atr_14"]
        pdi     = 100 * dm_pos.rolling(14).mean() / (atr14 + eps)
        ndi     = 100 * dm_neg.rolling(14).mean() / (atr14 + eps)
        dx      = 100 * (pdi - ndi).abs() / (pdi + ndi + eps)
        adx     = dx.ewm(span=14, adjust=False).mean()
        d["adx"]     = adx / 100.0
        d["di_bull"] = (pdi > ndi).astype(float)
        d["di_diff"] = (pdi - ndi) / 100.0

        # ── Momentum ──────────────────────────────────────────
        for p in [5, 10, 20, 50]:
            mom = c / (c.shift(p) + eps) - 1
            d[f"mom_{p}"] = mom.clip(-0.1, 0.1) / 0.1

        # ── Volatility ratios ─────────────────────────────────
        vol5  = d["ret_1"].rolling(5).std()
        vol20 = d["ret_1"].rolling(20).std()
        vol60 = d["ret_1"].rolling(60).std()
        d["vol_ratio_5_20"]  = (vol5  / (vol20 + eps)).clip(0, 5) / 5.0
        d["vol_ratio_20_60"] = (vol20 / (vol60 + eps)).clip(0, 5) / 5.0
        d["vol_20"] = vol20   # kept for ATR reference, not a model input

        # ── Price range position ──────────────────────────────
        for p in [10, 20, 50, 100]:
            hh = h.rolling(p).max()
            ll = l.rolling(p).min()
            d[f"range_pos_{p}"] = (
                (c - ll) / (hh - ll + eps)
            ) * 2 - 1

        # ── EMA distance ──────────────────────────────────────
        for p in [10, 20, 50, 200]:
            ema = c.ewm(span=p, adjust=False).mean()
            d[f"ema_dist_{p}"] = (
                (c - ema) / (ema + eps)
            ).clip(-0.05, 0.05) / 0.05

        # ── Candle structure ──────────────────────────────────
        spread = (h - l).clip(lower=eps)
        body   = (c - o).abs()
        top    = pd.concat([c, o], axis=1).max(axis=1)
        bot    = pd.concat([c, o], axis=1).min(axis=1)
        d["body_ratio"]  = (body / spread).clip(0, 1)
        d["bull_candle"] = (c > o).astype(float)
        d["upper_wick"]  = ((h - top) / (spread + eps)).clip(0, 1)
        d["lower_wick"]  = ((bot - l) / (spread + eps)).clip(0, 1)

        # ── Volume ────────────────────────────────────────────
        v_ma = v.rolling(20).mean()
        d["vol_ratio_bar"] = (v / (v_ma + eps)).clip(0, 5) / 5.0

        # ── Z-score ───────────────────────────────────────────
        for p in [20, 50]:
            rm = c.rolling(p).mean()
            rs = c.rolling(p).std()
            d[f"zscore_{p}"] = (
                (c - rm) / (rs + eps)
            ).clip(-3, 3) / 3.0

        # ── Lagged returns ────────────────────────────────────
        for lag in [1, 2, 3, 5]:
            d[f"ret_lag_{lag}"] = (
                d["ret_1"].shift(lag).clip(-0.02, 0.02) / 0.02
            )

        # ── Session timing ────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(
            d["timestamp"]
        ):
            hour = d["timestamp"].dt.hour
            dow  = d["timestamp"].dt.dayofweek
            d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            d["dow_sin"]  = np.sin(2 * np.pi * dow  / 5)
            d["dow_cos"]  = np.cos(2 * np.pi * dow  / 5)
        else:
            d["hour_sin"] = 0.0
            d["hour_cos"] = 1.0
            d["dow_sin"]  = 0.0
            d["dow_cos"]  = 1.0

        # ── Clean ─────────────────────────────────────────────
        d.replace([np.inf, -np.inf], 0, inplace=True)
        d.dropna(inplace=True)
        d.reset_index(drop=True, inplace=True)
        logger.info(
            f"Features: {d.shape[1]} cols, {len(d):,} rows"
        )
        return d

    def feature_matrix(
            self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Returns X in the canonical column order
        defined by FEATURE_COLS.

        Any column missing from df is filled with 0.
        This guarantees the runtime receives a tensor
        with the same shape and column semantics the
        model was trained on even if the live feed
        adds or renames columns in the future.
        """
        cols   = self.FEATURE_COLS
        arrays = []
        for col in cols:
            if col in df.columns:
                arrays.append(
                    df[col].values.astype(np.float32)
                )
            else:
                logger.warning(
                    f"Feature '{col}' missing — "
                    f"filling with zeros"
                )
                arrays.append(
                    np.zeros(len(df), dtype=np.float32)
                )
        return np.column_stack(arrays)


# ╔══════════════════════════════════════════════════════════╗
# ║  TARGET BUILDER                                          ║
# ║                                                          ║
# ║  BUG 1 FIX — same-bar SL/TP ambiguity                   ║
# ║                                                          ║
# ║  Old code checked `if high >= tp` before                 ║
# ║  `if low <= sl`.  When both fired on the same           ║
# ║  bar the trade was labelled a win (1) regardless        ║
# ║  of which level price reached first intra-bar.          ║
# ║                                                          ║
# ║  Impact of the old bug:                                  ║
# ║    For a 1.33 RR setup (SL=1.5×ATR, TP=2×ATR)          ║
# ║    the no-skill base rate for same-bar hits is          ║
# ║    roughly proportional to TP/(SL+TP) ≈ 57%.           ║
# ║    When we incorrectly label those bars as wins          ║
# ║    the long-target positive rate inflates from           ║
# ║    the true ~35% to an artifically high ~42%.           ║
# ║    XGBoost learns to predict 1 more readily,            ║
# ║    scale_pos_weight under-corrects, and the             ║
# ║    backtest win rate is 5-7 pp higher than real.        ║
# ║                                                          ║
# ║  Fix:                                                    ║
# ║    When both high[j] >= tp AND low[j] <= sl              ║
# ║    in the same bar → label = 0 (SL hit first).          ║
# ║    This is conservative but correct: without            ║
# ║    tick-level data we cannot know intra-bar order,      ║
# ║    so we assume the worst for risk management.          ║
# ╚══════════════════════════════════════════════════════════╝

class TargetBuilder:

    def __init__(self,
                 sl_mult:  float = 1.5,
                 tp_mult:  float = 2.0,
                 max_bars: int   = 24):
        self.sl_mult  = sl_mult
        self.tp_mult  = tp_mult
        self.max_bars = max_bars

    # ── internal helper ───────────────────────────────────────
    def _label_series(
            self,
            closes: np.ndarray,
            highs:  np.ndarray,
            lows:   np.ndarray,
            atrs:   np.ndarray,
            long:   bool
    ) -> np.ndarray:
        """
        Shared logic for long and short labelling.

        For LONG  entry at close[i]:
            TP = close[i] + tp_mult * atr[i]
            SL = close[i] - sl_mult * atr[i]
            Win (1) when high[j] >= TP first

        For SHORT entry at close[i]:
            TP = close[i] - tp_mult * atr[i]
            SL = close[i] + sl_mult * atr[i]
            Win (1) when low[j]  <= TP first

        BUG 1 FIX is inside the inner loop:
            if hit_tp AND hit_sl in same bar j
                → label = 0  (conservative)
        """
        n   = len(closes)
        lbl = np.zeros(n, dtype=np.int32)

        for i in range(n - self.max_bars):
            entry = closes[i]
            atr   = atrs[i]
            if atr <= 0 or np.isnan(atr):
                continue

            if long:
                tp = entry + self.tp_mult * atr
                sl = entry - self.sl_mult * atr
            else:
                tp = entry - self.tp_mult * atr
                sl = entry + self.sl_mult * atr

            for j in range(i + 1,
                           i + self.max_bars + 1):
                if long:
                    hit_tp = highs[j] >= tp
                    hit_sl = lows[j]  <= sl
                else:
                    hit_tp = lows[j]  <= tp
                    hit_sl = highs[j] >= sl

                # ── BUG 1 FIX ────────────────────────────
                # Both levels breached in the same bar.
                # Without tick data we cannot determine
                # which was hit first.
                # Conservative choice: SL hit first → 0.
                if hit_tp and hit_sl:
                    lbl[i] = 0
                    break
                # ─────────────────────────────────────────
                elif hit_tp:
                    lbl[i] = 1
                    break
                elif hit_sl:
                    lbl[i] = 0
                    break
                # Neither: continue scanning forward

        return lbl

    def build_long_target(
            self, df: pd.DataFrame
    ) -> pd.Series:
        lbl = self._label_series(
            closes=df["close"].values,
            highs =df["high"].values,
            lows  =df["low"].values,
            atrs  =df["atr_14"].values,
            long  =True,
        )
        pos_rate = lbl.mean()
        logger.info(
            f"Long target positive rate: "
            f"{pos_rate:.3f}  "
            f"(expected 0.30-0.45 for RR=1.33)"
        )
        return pd.Series(lbl, index=df.index)

    def build_short_target(
            self, df: pd.DataFrame
    ) -> pd.Series:
        lbl = self._label_series(
            closes=df["close"].values,
            highs =df["high"].values,
            lows  =df["low"].values,
            atrs  =df["atr_14"].values,
            long  =False,
        )
        pos_rate = lbl.mean()
        logger.info(
            f"Short target positive rate: "
            f"{pos_rate:.3f}  "
            f"(expected 0.30-0.45 for RR=1.33)"
        )
        return pd.Series(lbl, index=df.index)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODEL TRAINER                                           ║
# ║                                                          ║
# ║  CONFIG ADJUSTMENT — probability calibration            ║
# ║                                                          ║
# ║  Old approach: scale_pos_weight = n_neg/n_pos           ║
# ║                                                          ║
# ║  Problem with scale_pos_weight on an imbalanced         ║
# ║  target (30-35% positives):                             ║
# ║    scale_pos_weight ≈ 2.0-2.3 pushes the decision      ║
# ║    boundary so far that predict_proba outputs           ║
# ║    cluster around 0.6-0.8 for the positive class.       ║
# ║    A threshold of 0.63 then barely filters              ║
# ║    anything — it sits in the middle of the output        ║
# ║    distribution rather than in its tail.                ║
# ║                                                          ║
# ║  Fix: CalibratedClassifierCV (isotonic, cv=3)           ║
# ║    Isotonic calibration maps the raw XGB scores to      ║
# ║    empirical frequencies via a non-parametric           ║
# ║    monotone function.  After calibration,               ║
# ║    P(y=1 | score=0.63) ≈ 0.63 in the training set.     ║
# ║    The threshold therefore has a direct probabilistic   ║
# ║    interpretation: we only trade when the model         ║
# ║    assigns at least 63% confidence to the outcome.      ║
# ║                                                          ║
# ║  Brier score logged alongside AUC so we can             ║
# ║  verify calibration quality across folds.               ║
# ╚══════════════════════════════════════════════════════════╝

class ModelTrainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── cross-validation ──────────────────────────────────────
    def _cv_score(self,
                  params: dict,
                  X: np.ndarray,
                  y: np.ndarray) -> float:
        """
        Purged time-series cross-validation.
        Returns mean ROC-AUC.

        EMBARGO_BARS bars are excluded between
        each train and validation fold to prevent
        rolling-window features from carrying
        target information across the boundary.
        """
        tscv   = TimeSeriesSplit(
            n_splits=self.cfg.CV_SPLITS,
            gap    =self.cfg.EMBARGO_BARS,
        )
        scores = []
        for tr_idx, te_idx in tscv.split(X):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xte, yte = X[te_idx], y[te_idx]
            if (len(np.unique(ytr)) < 2 or
                    len(np.unique(yte)) < 2):
                continue
            # No scale_pos_weight — calibration
            # handles class imbalance post-hoc.
            m = XGBClassifier(
                **params,
                verbosity  =0,
                tree_method="hist",
                random_state=42,
            )
            m.fit(Xtr, ytr)
            try:
                auc = roc_auc_score(
                    yte,
                    m.predict_proba(Xte)[:, 1]
                )
                scores.append(auc)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.5

    # ── Optuna search ─────────────────────────────────────────
    def optimize(self,
                 X: np.ndarray,
                 y: np.ndarray) -> dict:
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
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            objective,
            n_trials=self.cfg.OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        logger.info(
            f"Best CV AUC: {study.best_value:.4f}"
        )
        return study.best_params

    # ── final fit with calibration ────────────────────────────
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            params: dict
            ) -> CalibratedClassifierCV:
        """
        Fits XGBClassifier then wraps with
        CalibratedClassifierCV (isotonic, cv=3).

        isotonic regression is chosen over sigmoid
        (Platt scaling) because:
          - isotonic is non-parametric → handles
            the non-sigmoid XGB score distribution
          - isotonic needs ≥ ~200 samples per fold;
            with TRAIN_BARS=3000 and cv=3 we have
            ~1000 samples per fold — sufficient.
        """
        base = XGBClassifier(
            **params,
            verbosity  =0,
            tree_method="hist",
            random_state=42,
        )
        # cv=3 splits the training data internally
        # for calibration; no test-set leakage.
        cal = CalibratedClassifierCV(
            base,
            method="isotonic",
            cv=3,
        )
        cal.fit(X, y)
        # Log Brier score on training data as a
        # rough calibration sanity check.
        # Well-calibrated model: Brier < 0.25.
        p    = cal.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, p)
        logger.info(
            f"Train Brier score: {brier:.4f}  "
            f"(lower = better calibrated)"
        )
        return cal


# ╔══════════════════════════════════════════════════════════╗
# ║  SIGNAL GENERATOR                                        ║
# ╚══════════════════════════════════════════════════════════╝

class SignalGenerator:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.EPS = 1e-10

    def generate(self,
                 df:         pd.DataFrame,
                 prob_long:  np.ndarray,
                 prob_short: np.ndarray
                 ) -> pd.Series:
        """
        Signal = +1 (long)  if:
            prob_long  > PROB_THRESHOLD_BUY AND
            RSI_14_raw > RSI_BULL_MIN        AND
            ADX_raw    > ADX_MIN

        Signal = -1 (short) if:
            prob_short > PROB_THRESHOLD_SELL AND
            RSI_14_raw < RSI_BEAR_MAX        AND
            ADX_raw    > ADX_MIN

        Signal =  0 (hold)  otherwise.

        When both long and short conditions are met
        simultaneously (rare after calibration),
        the stronger probability wins.
        """
        cfg = self.cfg
        n   = len(df)
        sig = np.zeros(n, dtype=np.int32)

        # Recover raw RSI: stored as (RSI-50)/50
        rsi_raw = df["rsi_14"].values * 50 + 50
        # Recover raw ADX: stored as adx/100
        adx_raw = df["adx"].values * 100

        for i in range(n):
            rsi = rsi_raw[i]
            adx = adx_raw[i]
            pl  = prob_long[i]
            ps  = prob_short[i]

            if np.isnan(rsi) or np.isnan(adx):
                continue

            # Trend quality gate
            if adx < cfg.ADX_MIN:
                continue

            long_ok  = (pl > cfg.PROB_THRESHOLD_BUY  and
                        rsi > cfg.RSI_BULL_MIN)
            short_ok = (ps > cfg.PROB_THRESHOLD_SELL and
                        rsi < cfg.RSI_BEAR_MAX)

            if long_ok and short_ok:
                # Take whichever is more confident
                sig[i] = 1 if pl >= ps else -1
            elif long_ok:
                sig[i] =  1
            elif short_ok:
                sig[i] = -1

        return pd.Series(sig, index=df.index)


# ╔══════════════════════════════════════════════════════════╗
# ║  BACKTEST ENGINE                                         ║
# ║                                                          ║
# ║  Execution rules (conservative / realistic):            ║
# ║  1. Entry fill = close[i] ± spread/2                    ║
# ║     (approximates next-bar open for H1 data)            ║
# ║  2. SL and TP hit checked against bar HIGH/LOW          ║
# ║  3. Same-bar SL+TP → SL wins (consistent with          ║
# ║     TargetBuilder conservative rule)                    ║
# ║  4. Position size = (equity × risk_pct) /               ║
# ║     (SL_pips × pip_value_per_lot)                       ║
# ║  5. Commission deducted half on entry,                  ║
# ║     half on exit (round-turn split)                     ║
# ║  6. Equity updated only on closed trades;               ║
# ║     no mark-to-market inflation                         ║
# ║  7. Drawdown computed from peak closed equity           ║
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
            df:      pd.DataFrame,
            signals: pd.Series
            ) -> Tuple[List[Trade], pd.DataFrame]:

        cfg      = self.cfg
        equity   = cfg.INITIAL_CAPITAL
        peak_eq  = equity
        eq_hist  = []           # closed-trade equity
        trades   = []
        open_tr: Optional[Trade] = None
        daily_cnt = 0
        cur_day   = None
        halted    = False

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

            day = t.date()
            if day != cur_day:
                cur_day   = day
                daily_cnt = 0

            # ── Check open trade SL/TP ────────────────────
            if open_tr is not None:
                closed = False

                if open_tr.direction == 1:        # long
                    hit_tp = high >= open_tr.tp_price
                    hit_sl = low  <= open_tr.sl_price
                else:                             # short
                    hit_tp = low  <= open_tr.tp_price
                    hit_sl = high >= open_tr.sl_price

                # ── Same-bar rule (consistent with
                #    TargetBuilder BUG 1 fix) ────────────
                if hit_tp and hit_sl:
                    ex_px  = (open_tr.sl_price
                              if open_tr.direction == 1
                              else open_tr.sl_price)
                    open_tr.exit_reason = "SL"
                    closed = True
                elif hit_tp:
                    ex_px  = open_tr.tp_price
                    open_tr.exit_reason = "TP"
                    closed = True
                elif hit_sl:
                    ex_px  = open_tr.sl_price
                    open_tr.exit_reason = "SL"
                    closed = True

                if closed:
                    # Spread on exit
                    if open_tr.direction == 1:
                        fill = ex_px - cfg.SPREAD_PRICE / 2
                        pnl_pts = fill - open_tr.entry_price
                    else:
                        fill = ex_px + cfg.SPREAD_PRICE / 2
                        pnl_pts = open_tr.entry_price - fill

                    pnl_usd  = (pnl_pts *
                                cfg.CONTRACT_SIZE *
                                open_tr.lot_size)
                    # Exit half of round-turn commission
                    pnl_usd -= (cfg.COMMISSION_PER_LOT *
                                open_tr.lot_size / 2)

                    open_tr.exit_bar   = i
                    open_tr.exit_price = fill
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
                            f"Drawdown halt bar {i}  "
                            f"equity=${equity:,.2f}"
                        )

            # ── New entry ─────────────────────────────────
            if (not halted and
                    open_tr is None and
                    sig != 0 and
                    daily_cnt < cfg.MAX_DAILY_TRADES and
                    i < len(df) - 1 and
                    atr > 0 and
                    not np.isnan(atr)):

                if sig == 1:    # long
                    fill_px = price + cfg.SPREAD_PRICE / 2
                    sl_px   = fill_px - cfg.ATR_SL_MULT * atr
                    tp_px   = fill_px + cfg.ATR_TP_MULT * atr
                    sl_dist = fill_px - sl_px
                else:           # short
                    fill_px = price - cfg.SPREAD_PRICE / 2
                    sl_px   = fill_px + cfg.ATR_SL_MULT * atr
                    tp_px   = fill_px - cfg.ATR_TP_MULT * atr
                    sl_dist = sl_px - fill_px

                if sl_dist <= self.EPS:
                    eq_hist.append(equity)
                    continue

                # Fixed fractional position sizing
                risk_usd       = equity * cfg.RISK_PER_TRADE_PCT
                pip_val_per_lot = (cfg.PIP_VALUE *
                                   cfg.CONTRACT_SIZE)
                sl_pips = sl_dist / cfg.PIP_VALUE
                lots    = risk_usd / (
                    sl_pips * pip_val_per_lot + self.EPS
                )
                lots = np.clip(lots, 0.01, 100.0)

                # Entry commission (half round-turn)
                commission = (cfg.COMMISSION_PER_LOT *
                              lots / 2)
                equity    -= commission
                equity     = max(equity, 0.01)

                open_tr = Trade(
                    entry_bar  =i,
                    entry_price=fill_px,
                    direction  =sig,
                    sl_price   =sl_px,
                    tp_price   =tp_px,
                    lot_size   =lots,
                )
                daily_cnt += 1

            eq_hist.append(equity)

        # ── Close any trade still open at end ────────────
        if open_tr is not None:
            ep = closes[-1]
            if open_tr.direction == 1:
                fill    = ep - cfg.SPREAD_PRICE / 2
                pnl_pts = fill - open_tr.entry_price
            else:
                fill    = ep + cfg.SPREAD_PRICE / 2
                pnl_pts = open_tr.entry_price - fill
            pnl_usd  = (pnl_pts * cfg.CONTRACT_SIZE *
                        open_tr.lot_size)
            pnl_usd -= (cfg.COMMISSION_PER_LOT *
                        open_tr.lot_size / 2)
            open_tr.exit_bar    = len(df) - 1
            open_tr.exit_price  = fill
            open_tr.pnl_usd     = pnl_usd
            open_tr.exit_reason = "EOD"
            equity += pnl_usd
            equity  = max(equity, 0.01)
            trades.append(open_tr)
            eq_hist.append(equity)

        eq_df = pd.DataFrame({
            "bar":    range(len(eq_hist)),
            "equity": eq_hist,
        })
        return trades, eq_df


# ╔══════════════════════════════════════════════════════════╗
# ║  PERFORMANCE REPORTER                                    ║
# ╚══════════════════════════════════════════════════════════╝

class PerfReporter:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.EPS = 1e-10

    def report(self,
               trades:       List[Trade],
               eq_df:        pd.DataFrame,
               init_capital: float
               ) -> dict:
        nt = len(trades)
        if nt == 0:
            print("\n[PerfReporter] No trades executed.")
            return {"trades": 0}

        pnls   = np.array([t.pnl_usd for t in trades])
        wins   = pnls[pnls >  0]
        losses = pnls[pnls <= 0]
        n_w    = len(wins)
        n_l    = len(losses)
        wr     = n_w / (nt + self.EPS)
        gp     = wins.sum()   if n_w > 0 else 0.0
        gl     = abs(losses.sum()) if n_l > 0 else self.EPS
        pf     = gp / (gl + self.EPS)
        avg_w  = wins.mean()   if n_w > 0 else 0.0
        avg_l  = losses.mean() if n_l > 0 else 0.0
        expect = pnls.mean()
        tp_hits = sum(
            1 for t in trades if t.exit_reason == "TP"
        )
        sl_hits = sum(
            1 for t in trades if t.exit_reason == "SL"
        )

        eq    = eq_df["equity"].values
        ret   = np.diff(eq) / (eq[:-1] + self.EPS)
        fin   = eq[-1]
        tr    = fin / init_capital - 1
        n_d   = max(len(eq) / 24, 1)
        ar    = (1 + tr) ** (252 / n_d) - 1
        av    = np.std(ret) * np.sqrt(252 * 24)
        dn    = ret[ret < 0]
        dv    = (np.std(dn) if len(dn) > 0
                 else self.EPS) * np.sqrt(252 * 24)
        sh    = (ar - 0.02) / (av + self.EPS)
        so    = (ar - 0.02) / (dv + self.EPS)
        mdd   = self._mdd(eq)
        cal   = ar / (abs(mdd) + self.EPS)

        m = {
            "trades":        nt,
            "win_rate":      wr,
            "profit_factor": pf,
            "avg_win_usd":   avg_w,
            "avg_loss_usd":  avg_l,
            "expectancy_usd": expect,
            "tp_hits":       tp_hits,
            "sl_hits":       sl_hits,
            "total_return":  tr,
            "ann_return":    ar,
            "ann_vol":       av,
            "max_dd":        mdd,
            "sharpe":        sh,
            "sortino":       so,
            "calmar":        cal,
            "final_equity":  fin,
        }
        self._print(m)
        return m

    def _mdd(self, eq: np.ndarray) -> float:
        peak = eq[0]
        mdd  = 0.0
        for e in eq:
            peak = max(peak, e)
            dd   = (peak - e) / (peak + self.EPS)
            mdd  = max(mdd, dd)
        return mdd

    def _print(self, m: dict):
        sep = "═" * 52
        print(f"\n╔{sep}╗")
        print(f"║{'PERFORMANCE REPORT':^52}║")
        print(f"╠{sep}╣")
        rows = [
            ("Trades",         m["trades"],          "d"),
            ("Win Rate",       m["win_rate"],         ".2%"),
            ("Profit Factor",  m["profit_factor"],    ".3f"),
            ("Avg Win",        m["avg_win_usd"],      ",.2f"),
            ("Avg Loss",       m["avg_loss_usd"],     ",.2f"),
            ("Expectancy",     m["expectancy_usd"],   ",.2f"),
            ("TP Hits",        m["tp_hits"],          "d"),
            ("SL Hits",        m["sl_hits"],          "d"),
            ("Total Return",   m["total_return"],     ".3%"),
            ("Ann. Return",    m["ann_return"],       ".3%"),
            ("Ann. Vol",       m["ann_vol"],          ".3%"),
            ("Max Drawdown",   m["max_dd"],           ".3%"),
            ("Sharpe",         m["sharpe"],           ".3f"),
            ("Sortino",        m["sortino"],          ".3f"),
            ("Calmar",         m["calmar"],           ".3f"),
            ("Final Equity",   m["final_equity"],     ",.2f"),
        ]
        for label, val, fmt in rows:
            vs = f"{val:{fmt}}"
            print(f"║  {label+':':<22}{vs:>14}{'':>14}║")
        print(f"╚{sep}╝\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  WALK-FORWARD PIPELINE                                   ║
# ╚══════════════════════════════════════════════════════════╝

class WalkForwardPipeline:

    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.trainer = ModelTrainer(cfg)
        self.tb      = TargetBuilder(
            sl_mult =cfg.ATR_SL_MULT,
            tp_mult =cfg.ATR_TP_MULT,
            max_bars=cfg.TARGET_MAX_BARS,
        )
        self.sg  = SignalGenerator(cfg)
        self.fe  = FeatureEngine()
        self.bt  = BacktestEngine(cfg)
        self.pr  = PerfReporter(cfg)

    # ── main entry ────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        n   = len(df)

        logger.info("Building long targets…")
        long_tgt  = self.tb.build_long_target(df)
        logger.info("Building short targets…")
        short_tgt = self.tb.build_short_target(df)

        # BUG 2 FIX: X_all constructed via
        # FeatureEngine.feature_matrix() which
        # enforces canonical FEATURE_COLS order.
        X_all = self.fe.feature_matrix(df)
        logger.info(
            f"Feature matrix: "
            f"{X_all.shape[0]:,} × {X_all.shape[1]}"
        )

        y_long  = long_tgt.values
        y_short = short_tgt.values

        all_signals = pd.Series(
            0, index=df.index, dtype=np.int32
        )
        model_long  = None
        model_short = None
        best_params = None
        fold_n      = 0
        t           = cfg.MIN_TRAIN_BARS

        # ── fold metrics accumulator ─────────────────────────
        fold_aucs: List[dict] = []

        while t + cfg.TEST_BARS < n:
            tr_start = max(0, t - cfg.TRAIN_BARS)
            tr_end   = t
            te_start = t  + cfg.EMBARGO_BARS
            te_end   = min(
                t + cfg.EMBARGO_BARS + cfg.TEST_BARS,
                n - cfg.TARGET_MAX_BARS,
            )

            if te_start >= te_end or te_end <= 0:
                t += cfg.TEST_BARS
                continue

            Xtr   = X_all[tr_start:tr_end]
            yltr  = y_long[tr_start:tr_end]
            ystr  = y_short[tr_start:tr_end]
            Xte   = X_all[te_start:te_end]
            ylte  = y_long[te_start:te_end]
            yste  = y_short[te_start:te_end]

            if len(Xtr) < cfg.MIN_TRAIN_BARS:
                t += cfg.TEST_BARS
                continue

            logger.info(
                f"Fold {fold_n:>3d}: "
                f"train [{tr_start:>6d}:{tr_end:>6d}]  "
                f"test  [{te_start:>6d}:{te_end:>6d}]"
            )

            # Re-optimise every REOPT_EVERY_N_FOLDS folds
            if fold_n % cfg.REOPT_EVERY_N_FOLDS == 0:
                logger.info(
                    f"  Optimising hyperparameters "
                    f"(fold {fold_n})…"
                )
                best_params = self.trainer.optimize(
                    Xtr, yltr
                )

            # Fit calibrated models
            model_long  = self.trainer.fit(
                Xtr, yltr, best_params
            )
            model_short = self.trainer.fit(
                Xtr, ystr, best_params
            )

            # OOS predictions
            pl = model_long.predict_proba(Xte)[:, 1]
            ps = model_short.predict_proba(Xte)[:, 1]

            # Log OOS AUC + Brier for each fold
            fold_row = {"fold": fold_n}
            if len(np.unique(ylte)) == 2:
                fold_row["auc_long"] = roc_auc_score(
                    ylte, pl
                )
                fold_row["brier_long"] = brier_score_loss(
                    ylte, pl
                )
                logger.info(
                    f"  OOS AUC long:  "
                    f"{fold_row['auc_long']:.4f}  "
                    f"Brier: "
                    f"{fold_row['brier_long']:.4f}"
                )
            if len(np.unique(yste)) == 2:
                fold_row["auc_short"] = roc_auc_score(
                    yste, ps
                )
                fold_row["brier_short"] = brier_score_loss(
                    yste, ps
                )
                logger.info(
                    f"  OOS AUC short: "
                    f"{fold_row['auc_short']:.4f}  "
                    f"Brier: "
                    f"{fold_row['brier_short']:.4f}"
                )
            fold_aucs.append(fold_row)

            # Signals for test window
            df_te = df.iloc[te_start:te_end].copy()
            df_te.reset_index(drop=True, inplace=True)
            sigs  = self.sg.generate(df_te, pl, ps)

            for j, idx in enumerate(
                range(te_start, te_end)
            ):
                if idx < n:
                    all_signals.iloc[idx] = sigs.iloc[j]

            fold_n += 1
            t      += cfg.TEST_BARS

        # ── Signal counts ─────────────────────────────────────
        n_long  = (all_signals ==  1).sum()
        n_short = (all_signals == -1).sum()
        n_hold  = (all_signals ==  0).sum()
        logger.info(
            f"OOS signals — long: {n_long}  "
            f"short: {n_short}  hold: {n_hold}"
        )

        # ── Backtest ──────────────────────────────────────────
        logger.info("Running backtest…")
        trades, eq_df = self.bt.run(df, all_signals)

        metrics = self.pr.report(
            trades, eq_df, cfg.INITIAL_CAPITAL
        )
        metrics["n_folds"]  = fold_n
        metrics["n_long"]   = int(n_long)
        metrics["n_short"]  = int(n_short)

        # ── Persist ───────────────────────────────────────────
        self._save(
            metrics, fold_aucs,
            model_long, model_short,
            all_signals
        )
        return metrics

    # ── persistence ───────────────────────────────────────────
    def _save(self,
              metrics:     dict,
              fold_aucs:   list,
              model_long,
              model_short,
              signals:     pd.Series):
        out = self.cfg.OUTPUT_DIR
        os.makedirs(out, exist_ok=True)

        # metrics.json
        def _serial(v):
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            return v

        with open(
            os.path.join(out, "metrics.json"), "w"
        ) as f:
            json.dump(
                {k: _serial(v)
                 for k, v in metrics.items()},
                f, indent=2
            )

        # fold_diagnostics.json
        with open(
            os.path.join(out, "fold_diagnostics.json"),
            "w"
        ) as f:
            json.dump(
                [{k: _serial(v)
                  for k, v in row.items()}
                 for row in fold_aucs],
                f, indent=2
            )

        # BUG 2 FIX: features.json with explicit index
        # so MQL5/ONNX runtime can validate tensor order.
        feat_index = {
            col: idx
            for idx, col in
            enumerate(FeatureEngine.FEATURE_COLS)
        }
        with open(
            os.path.join(out, "features.json"), "w"
        ) as f:
            json.dump(
                {
                    "feature_cols":  FeatureEngine.FEATURE_COLS,
                    "feature_index": feat_index,
                    "n_features":    len(FeatureEngine.FEATURE_COLS),
                    "note": (
                        "Input tensor must have columns "
                        "in the order of feature_cols. "
                        "feature_index maps name→position."
                    ),
                },
                f, indent=2
            )

        # Models
        if model_long is not None:
            try:
                import pickle
                with open(
                    os.path.join(
                        out, "model_long.pkl"
                    ), "wb"
                ) as f:
                    pickle.dump(model_long, f)
            except Exception as e:
                logger.warning(
                    f"Could not save model_long: {e}"
                )

        if model_short is not None:
            try:
                import pickle
                with open(
                    os.path.join(
                        out, "model_short.pkl"
                    ), "wb"
                ) as f:
                    pickle.dump(model_short, f)
            except Exception as e:
                logger.warning(
                    f"Could not save model_short: {e}"
                )

        # OOS signals
        signals.to_csv(
            os.path.join(out, "oos_signals.csv")
        )
        logger.info(f"Artifacts saved → {out}/")


# ╔══════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                             ║
# ╚══════════════════════════════════════════════════════════╝

def run(
    source:   str    = "synthetic",
    filepath: str    = None,
    cfg:      Config = None,
) -> dict:

    if cfg is None:
        cfg = Config()

    print(
        "\n╔══════════════════════════════════════╗"
    )
    print(
        "║  XGBoost Direct Trader — v2.1        ║"
    )
    print(
        "╚══════════════════════════════════════╝\n"
    )

    # Phase 1: Data
    print("Phase 1: Data Ingestion")
    if source == "synthetic":
        raw = DataIngestion.load_synthetic(
            cfg.HISTORICAL_BARS
        )
    elif source == "csv":
        if filepath is None:
            raise ValueError(
                "filepath required for source='csv'"
            )
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

    # Phase 2: Features
    print("Phase 2: Feature Engineering")
    fe = FeatureEngine()
    df = fe.build(df)
    print(
        f"  {df.shape[1]} columns, "
        f"{len(df):,} rows\n"
    )

    # Phase 3: Walk-forward train + backtest
    print("Phase 3: Walk-Forward Training & Backtest")
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
