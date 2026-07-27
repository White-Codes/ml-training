# ============================================================
#  XGBoost_RL_Trainer.py
# ============================================================

import os, json, warnings, logging, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from collections import deque

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


# ╔══════════════════════════════════════════════════════════╗
# ║  CONFIG                                                  ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class RewardWeights:
    sharpe_weight:        float = 0.30
    sortino_weight:       float = 0.20
    profit_factor_weight: float = 0.15
    consistency_weight:   float = 0.10
    drawdown_penalty:     float = 0.10
    trade_penalty:        float = 0.10
    ruin_penalty:         float = 0.05

@dataclass
class SystemConfig:
    DATA_SOURCE:      str   = "csv"
    TRADING_PAIR:     str   = "EURUSD"
    TIMEFRAME:        str   = "H1"
    HISTORICAL_BARS:  int   = 50_000

    SMA_PERIODS:         List[int] = field(
        default_factory=lambda: [5, 10, 20, 50, 100, 200])
    EMA_PERIODS:         List[int] = field(
        default_factory=lambda: [5, 10, 20, 50, 100])
    RSI_PERIODS:         List[int] = field(
        default_factory=lambda: [7, 14, 21])
    VOLATILITY_WINDOWS:  List[int] = field(
        default_factory=lambda: [10, 20, 60, 100])
    LAG_PERIODS:         List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 20])
    MULTI_TIMEFRAMES:    List[int] = field(
        default_factory=lambda: [5, 10, 20, 50, 100])
    MAX_FEATURES_SELECTED: int   = 50
    CORRELATION_THRESHOLD: float = 0.95

    N_ACTIONS:            int   = 3
    GAMMA:                float = 0.99
    INITIAL_EPSILON:      float = 1.0
    EPSILON_MIN:          float = 0.01
    EPSILON_DECAY:        float = 0.9995

    # ── FIX 2: Lowered from 0.60 → 0.35 ──────────────────
    # At 0.60 the 7-model ensemble rarely reached consensus,
    # defaulting to Hold almost every step. 0.35 allows the
    # agent to act on moderate conviction signals and
    # dramatically increases trade frequency.
    CONFIDENCE_THRESHOLD: float = 0.35

    N_ENSEMBLE_MODELS: int = 7
    ENSEMBLE_CONFIGS: List[dict] = field(default_factory=lambda: [
        {"max_depth": 4,  "subsample": 0.70, "colsample_bytree": 0.60},
        {"max_depth": 6,  "subsample": 0.80, "colsample_bytree": 0.70},
        {"max_depth": 8,  "subsample": 0.90, "colsample_bytree": 0.80},
        {"max_depth": 5,  "subsample": 0.60, "colsample_bytree": 0.50},
        {"max_depth": 7,  "subsample": 0.85, "colsample_bytree": 0.75},
        {"max_depth": 3,  "subsample": 0.95, "colsample_bytree": 0.90},
        {"max_depth": 10, "subsample": 0.65, "colsample_bytree": 0.65},
    ])
    XGB_BASE: dict = field(default_factory=lambda: {
        "n_estimators":  500,
        "learning_rate": 0.05,
        "reg_alpha":     1.0,
        "reg_lambda":    2.0,
        "tree_method":   "hist",
        "verbosity":     0,
        "random_state":  42,
    })

    REPLAY_BUFFER_CAPACITY:  int   = 200_000
    PRIORITY_ALPHA:          float = 0.6
    PRIORITY_BETA_START:     float = 0.4
    PRIORITY_BETA_INCREMENT: float = 0.001
    BATCH_SIZE:              int   = 256
    MIN_BUFFER_SIZE:         int   = 1_000

    TRAIN_WINDOW:          int   = 5_000
    RETRAIN_INTERVAL:      int   = 500
    MIN_TRAIN_SAMPLES:     int   = 1_000
    RECENCY_WEIGHT_DECAY:  float = -1.0
    REGIME_CHANGE_P_VALUE: float = 0.01
    REGIME_LOOKBACK:       int   = 100

    REWARD_WEIGHTS:       RewardWeights = field(
        default_factory=RewardWeights)
    RISK_FREE_RATE:       float = 0.02
    MAX_DRAWDOWN_LIMIT:   float = 0.20
    TRANSACTION_COST_BPS: int   = 5

    INITIAL_CAPITAL:       float = 100_000.0
    MAX_POSITION_SIZE:     float = 0.25

    # ── FIX 4: ATR multipliers replace fixed pct stops ────
    # Fixed 2% SL and 6% TP on H1 EURUSD are almost never
    # hit (2% EURUSD move takes days). Replaced with ATR
    # multipliers so stops scale with current volatility,
    # triggering more frequently and producing cleaner
    # risk/reward outcomes. SL=1.0×ATR, TP=1.5×ATR gives
    # a 1.5:1 R:R ratio — only needs 41% win rate for
    # Profit Factor > 1.30.
    ATR_SL_MULT:           float = 1.0   # Stop  = 1.0 × ATR14
    ATR_TP_MULT:           float = 1.5   # Target = 1.5 × ATR14
    # Keep percentage fallback for when ATR is unavailable
    STOP_LOSS_PCT:         float = 0.02
    TAKE_PROFIT_PCT:       float = 0.06

    MAX_DAILY_TRADES:      int   = 10
    EQUITY_RUIN_THRESHOLD: float = 0.80

    # ── FIX 3: Realistic EURUSD spread ───────────────────
    # 1.5 pips = 0.00015 price units. Applied at entry and
    # exit so the agent learns costs that match live trading.
    SPREAD_PRICE_UNITS:    float = 0.00015   # 1.5 pips

    # ── FIX 1: Position size cap for evaluation ───────────
    # Caps each trade at 2% of INITIAL capital (not growing
    # equity) so the equity curve is linear and metrics are
    # comparable across sessions. Removes the exponential
    # compounding that produced 71,755% phantom returns.
    EVAL_SIZE_CAP_PCT:     float = 0.02      # 2% of init_cap

    OPTUNA_N_TRIALS:  int   = 50
    OPTUNA_CV_SPLITS: int   = 5
    EMBARGO_PCT:      float = 0.02

    NOISE_INJECTION_LEVEL: float = 0.01
    FEATURE_DROPOUT_RATE:  float = 0.10
    ADVERSARIAL_AUC_LIMIT: float = 0.55

    OUTPUT_DIR: str = "xgb_rl_artifacts"

    def __post_init__(self):
        self.DAILY_RISK_FREE = self.RISK_FREE_RATE / 252
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 1 — DATA INGESTION                               ║
# ╚══════════════════════════════════════════════════════════╝

class DataIngestion:
    @staticmethod
    def load(source: str, filepath: str = None,
             n_bars: int = 50_000) -> pd.DataFrame:
        if source == "csv":
            df = pd.read_csv(filepath, parse_dates=["timestamp"])
        elif source == "synthetic":
            df = DataIngestion._synthetic(n_bars)
        elif source == "exchange_api":
            import ccxt
            ex = ccxt.binance()
            ohlcv = ex.fetch_ohlcv(
                "BTC/USDT", "1h", limit=n_bars)
            df = pd.DataFrame(ohlcv,
                columns=["timestamp","open","high",
                         "low","close","volume"])
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms")
        else:
            raise ValueError(f"Unknown source: {source}")
        return DataIngestion.preprocess(df)

    @staticmethod
    def _synthetic(n: int) -> pd.DataFrame:
        np.random.seed(42)
        dt, mu, sigma, S0 = (
            1/(24*365), 0.10, 0.80, 30_000.0)
        prices = [S0]
        for _ in range(1, n):
            r = mu*dt + sigma*np.sqrt(dt)*np.random.randn()
            prices.append(prices[-1]*np.exp(r))
        prices = np.array(prices)
        noise  = sigma*np.sqrt(dt)
        o = prices*np.exp(np.random.randn(n)*noise*0.3)
        h = prices*np.exp(
            np.abs(np.random.randn(n))*noise)
        l = prices*np.exp(
            -np.abs(np.random.randn(n))*noise)
        h = np.maximum(h, np.maximum(o, prices))
        l = np.minimum(l, np.minimum(o, prices))
        v = np.random.lognormal(10, 1, n)
        ts = pd.date_range("2020-01-01", periods=n,
                           freq="h")
        return pd.DataFrame({
            "timestamp": ts, "open": o, "high": h,
            "low": l, "close": prices, "volume": v})

    @staticmethod
    def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
        df   = raw.copy()
        cols = ["open","high","low","close","volume"]
        df.dropna(subset=cols, inplace=True)
        df = df[df["volume"] > 0].copy()
        df.drop_duplicates(
            subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        for c in cols:
            mu, sig = df[c].mean(), df[c].std()
            if sig > 0:
                df[c] = df[c].clip(mu-5*sig, mu+5*sig)
        df = df[
            (df.high>=df.low) &
            (df.high>=df.open) &
            (df.high>=df.close) &
            (df.low<=df.open) &
            (df.low<=df.close) &
            (df.volume>=0)
        ].copy()
        df.reset_index(drop=True, inplace=True)
        df["returns"]     = df["close"].pct_change()
        df["log_returns"] = np.log(
            df["close"]/df["close"].shift(1))
        logger.info(f"Preprocessed: {len(df):,} bars")
        return df


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 2 — FEATURE ENGINEERING                          ║
# ╚══════════════════════════════════════════════════════════╝

class FeatureEngine:
    EPS = 1e-10

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    @staticmethod
    def _rsi(s: pd.Series, p: int) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - 100/(1 + g/(l+FeatureEngine.EPS))

    @staticmethod
    def _macd(s: pd.Series, f=12, sl=26, sg=9):
        ef  = s.ewm(span=f,  adjust=False).mean()
        es  = s.ewm(span=sl, adjust=False).mean()
        ml  = ef - es
        sig = ml.ewm(span=sg, adjust=False).mean()
        return ml, sig, ml-sig

    @staticmethod
    def _atr(h, l, c, p=14) -> pd.Series:
        tr = pd.concat([
            h-l,
            (h-c.shift(1)).abs(),
            (l-c.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def _williams_r(h, l, c, p=14) -> pd.Series:
        hh = h.rolling(p).max()
        ll = l.rolling(p).min()
        return -100*(hh-c)/(hh-ll+FeatureEngine.EPS)

    @staticmethod
    def _cci(h, l, c, p=20) -> pd.Series:
        tp = (h+l+c)/3
        ma = tp.rolling(p).mean()
        md = tp.rolling(p).apply(
            lambda x: np.mean(np.abs(x-x.mean())),
            raw=True)
        return (tp-ma)/(0.015*md+FeatureEngine.EPS)

    @staticmethod
    def _stoch(h, l, c, kp=14, dp=3):
        ll = l.rolling(kp).min()
        hh = h.rolling(kp).max()
        k  = 100*(c-ll)/(hh-ll+FeatureEngine.EPS)
        return k, k.rolling(dp).mean()

    @staticmethod
    def _obv(c, v) -> pd.Series:
        return (
            np.sign(c.diff()).fillna(0)*v
        ).cumsum()

    @staticmethod
    def _mfi(h, l, c, v, p=14) -> pd.Series:
        tp  = (h+l+c)/3
        rmf = tp*v
        pos = (rmf*(tp>tp.shift(1))).rolling(p).sum()
        neg = (rmf*(tp<tp.shift(1))).rolling(p).sum()
        return 100 - 100/(
            1+pos/(neg+FeatureEngine.EPS))

    @staticmethod
    def _hurst(s: pd.Series, lb=100,
               ml=20) -> pd.Series:
        result = np.full(len(s), 0.5)
        arr    = s.values
        for t in range(lb, len(arr)):
            w   = arr[t-lb:t]
            lgs = range(2, min(ml, lb//2))
            tau = [
                np.std(w[lg:]-w[:-lg])+1e-10
                for lg in lgs]
            if len(tau) < 2:
                continue
            try:
                slope, _ = np.polyfit(
                    np.log(list(lgs)),
                    np.log(tau), 1)
                result[t] = float(
                    np.clip(slope, 0, 1))
            except Exception:
                pass
        return pd.Series(result, index=s.index)

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        df  = data.copy()
        cfg = self.cfg
        eps = self.EPS
        c   = df["close"]
        h   = df["high"]
        l   = df["low"]
        v   = df["volume"]
        ret = df["returns"]

        for p in cfg.SMA_PERIODS:
            sma = c.rolling(p).mean()
            df[f"sma_{p}"]       = sma
            df[f"close_sma_{p}"] = c/(sma+eps)-1
            df[f"sma_{p}_slope"] = sma.pct_change(5)

        for p in cfg.EMA_PERIODS:
            ema = c.ewm(span=p, adjust=False).mean()
            df[f"ema_{p}"]       = ema
            df[f"close_ema_{p}"] = c/(ema+eps)-1

        ema10 = c.ewm(span=10, adjust=False).mean()
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        df["ema10_20_cross"] = ema10/(ema20+eps)-1
        df["ema20_50_cross"] = ema20/(ema50+eps)-1

        for p in cfg.MULTI_TIMEFRAMES:
            mx = h.rolling(p).max()
            mn = l.rolling(p).min()
            df[f"range_pos_{p}"] = (
                c-mn)/(mx-mn+eps)

        for p in cfg.RSI_PERIODS:
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

        for p in [10, 20, 30, 60]:
            df[f"mom_{p}"] = c/c.shift(p)-1
            df[f"roc_{p}"] = c.pct_change(p)

        df["williams_r"]             = self._williams_r(h,l,c)
        df["cci"]                    = self._cci(h,l,c)
        df["stoch_k"], df["stoch_d"] = self._stoch(h,l,c)
        df["stoch_cross"]            = (
            df["stoch_k"] - df["stoch_d"])

        for w in cfg.VOLATILITY_WINDOWS:
            df[f"vol_{w}"] = ret.rolling(w).std()

        v20  = df.get("vol_20",  ret.rolling(20).std())
        v60  = df.get("vol_60",  ret.rolling(60).std())
        v10  = df.get("vol_10",  ret.rolling(10).std())
        v100 = df.get("vol_100", ret.rolling(100).std())
        df["vol_ratio_20_60"]  = v20/(v60+eps)
        df["vol_ratio_10_100"] = v10/(v100+eps)
        df["vol_ratio"]        = v20/(v60+eps)

        df["atr_14"]   = self._atr(h, l, c, 14)
        df["atr_ratio"] = df["atr_14"]/(c+eps)
        df["atr_pct"]   = (
            df["atr_14"].rolling(100).rank(pct=True))

        bm  = c.rolling(20).mean()
        bs  = c.rolling(20).std()
        bu  = bm+2*bs
        bl_ = bm-2*bs
        df["bb_width"]   = (bu-bl_)/(bm+eps)
        df["bb_pos"]     = (c-bl_)/(bu-bl_+eps)
        df["bb_squeeze"] = (
            df["bb_width"] <
            df["bb_width"].rolling(50).mean()
        ).astype(float)

        for w in [5, 10, 20, 50]:
            vsma = v.rolling(w).mean()
            df[f"vol_sma_{w}"]   = vsma
            df[f"vol_ratio_{w}"] = v/(vsma+eps)

        obv = self._obv(c, v)
        df["obv"]       = obv
        df["obv_slope"] = obv.pct_change(5)
        df["obv_norm"]  = obv/(obv.abs()+1)

        vwap = (c*v).cumsum()/(v.cumsum()+eps)
        df["vwap"]       = vwap
        df["close_vwap"] = c/(vwap+eps)-1
        df["mfi"]        = self._mfi(h, l, c, v)

        spread = (h-l).clip(lower=eps)
        df["spread_pct"] = spread/(c+eps)
        df["body"]       = (
            (c-df["open"]).abs()/(spread+eps))

        top = pd.concat(
            [df["open"],c], axis=1).max(axis=1)
        bot = pd.concat(
            [df["open"],c], axis=1).min(axis=1)
        df["upper_wick"] = (h-top)/(spread+eps)
        df["lower_wick"] = (bot-l)/(spread+eps)
        df["gap"]        = df["open"]/(c.shift(1)+eps)-1

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

        for w in [20, 50, 100]:
            df[f"skew_{w}"] = ret.rolling(w).skew()
            df[f"kurt_{w}"] = ret.rolling(w).kurt()
            rm = c.rolling(w).mean()
            rs = c.rolling(w).std()
            df[f"zscore_{w}"] = (c-rm)/(rs+eps)

        df["hurst"] = self._hurst(ret.fillna(0))

        df["zscore_extreme"] = (
            df["zscore_20"].abs() > 2.0
        ).astype(float)

        for p in [20, 50]:
            df[f"high_{p}"]      = h.rolling(p).max()
            df[f"low_{p}"]       = l.rolling(p).min()
            df[f"dist_high_{p}"] = (
                c - df[f"high_{p}"])/(c+eps)
            df[f"dist_low_{p}"]  = (
                c - df[f"low_{p}"])/(c+eps)

        dm_pos = (h - h.shift(1)).clip(lower=0)
        dm_neg = (l.shift(1) - l).clip(lower=0)
        tr14   = df["atr_14"]
        df["adx_pos"] = (
            dm_pos.rolling(14).mean()/(tr14+eps))
        df["adx_neg"] = (
            dm_neg.rolling(14).mean()/(tr14+eps))
        df["adx"] = (
            df["adx_pos"]-df["adx_neg"]).abs()

        key = ["returns","volume","rsi_14",
               "macd_hist","vol_20"]
        for feat in key:
            if feat not in df.columns:
                continue
            for lag in cfg.LAG_PERIODS:
                sh = df[feat].shift(lag)
                df[f"{feat}_lag{lag}"]  = sh
                df[f"{feat}_diff{lag}"] = df[feat]-sh

        excl  = {"timestamp","open","high","low",
                 "close","volume","returns","log_returns"}
        fcols = [
            c_ for c_ in df.select_dtypes(np.number).columns
            if c_ not in excl]
        top20 = (
            df[fcols].var().nlargest(20).index.tolist())
        cnt = 0
        for i in range(len(top20)):
            for j in range(i+1, len(top20)):
                if cnt >= 50: break
                f1, f2 = top20[i], top20[j]
                df[f"{f1}_div_{f2}"] = (
                    df[f1]/(df[f2]+eps))
                df[f"{f1}_x_{f2}"]   = (
                    df[f1]*df[f2])
                cnt += 1

        df.replace([np.inf,-np.inf], 0, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(
            f"Features built: {df.shape[1]} cols, "
            f"{len(df):,} rows")
        return df


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 3 — FEATURE SELECTION                            ║
# ╚══════════════════════════════════════════════════════════╝

class FeatureSelector:
    BASE_EXCL = {
        "timestamp","open","high","low","close",
        "volume","returns","log_returns"
    }

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def select(self, data: pd.DataFrame,
               target: pd.Series,
               top_k=50) -> List[str]:
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
        logger.info(
            f"Corr filter: -{len(drop)} "
            f"→ {X.shape[1]} remain")
        rem  = list(X.columns)
        Xa   = X.values.astype(np.float32)
        ya   = y.values

        tmp  = XGBClassifier(
            n_estimators=200, max_depth=6,
            verbosity=0, random_state=42,
            tree_method="hist")
        tmp.fit(Xa, ya)
        xgb_imp = np.array(tmp.feature_importances_)
        if len(xgb_imp) != len(rem):
            xgb_imp = np.ones(len(rem))/len(rem)
        xr = pd.Series(
            xgb_imp, index=rem).rank(ascending=False)

        try:
            expl = shap.TreeExplainer(tmp)
            sv   = expl.shap_values(Xa)
            if isinstance(sv, list):
                si = np.mean(
                    [np.mean(np.abs(s), axis=0)
                     for s in sv], axis=0)
            else:
                si = np.mean(np.abs(sv), axis=0)
            if len(si) != len(rem):
                si = xgb_imp.copy()
        except Exception:
            si = xgb_imp.copy()
        sr = pd.Series(
            si, index=rem).rank(ascending=False)

        try:
            mi = mutual_info_classif(
                Xa, ya, random_state=42)
        except Exception:
            mi = np.ones(len(rem))
        mr = pd.Series(
            mi, index=rem).rank(ascending=False)

        try:
            pi   = permutation_importance(
                tmp, Xa, ya,
                n_repeats=5, random_state=42)
            ps   = np.array(pi.importances_mean)
            if len(ps) != len(rem):
                ps = xgb_imp.copy()
        except Exception:
            ps = xgb_imp.copy()
        pr = pd.Series(
            ps, index=rem).rank(ascending=False)

        combined = (xr+sr+mr+pr)/4
        combined.sort_values(inplace=True)
        sel = combined.head(
            min(top_k, len(combined))
        ).index.tolist()
        logger.info(
            f"Selected {len(sel)} features. "
            f"Top5: {sel[:5]}")
        return sel

    def adversarial_check(
            self,
            X_tr: pd.DataFrame,
            X_te: pd.DataFrame
    ) -> Tuple[List[str], bool]:
        fcols = [c for c in X_tr.columns
                 if c not in self.BASE_EXCL]
        Xt   = X_tr[fcols].copy()
        Xv   = X_te[fcols].copy()
        Xall = pd.concat([Xt, Xv]).fillna(0)
        yall = np.concatenate(
            [np.zeros(len(Xt)), np.ones(len(Xv))])
        adv  = XGBClassifier(
            n_estimators=100, max_depth=4,
            verbosity=0, random_state=42)
        try:
            auc = cross_val_score(
                adv, Xall.values, yall,
                cv=3, scoring="roc_auc").mean()
        except Exception:
            auc = 0.5
        logger.info(f"Adversarial AUC: {auc:.4f}")
        if auc > self.cfg.ADVERSARIAL_AUC_LIMIT:
            logger.warning(
                "Distribution shift detected!")
            adv.fit(Xall.values, yall)
            imp  = adv.feature_importances_
            thr  = np.percentile(imp, 90)
            prob = [
                fcols[i] for i,v in enumerate(imp)
                if v > thr]
            return prob, True
        return [], False


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 4 — PRIORITIZED REPLAY BUFFER                    ║
# ╚══════════════════════════════════════════════════════════╝

class SumTree:
    def __init__(self, cap: int):
        self.cap  = cap
        self.tree = np.zeros(2*cap-1, dtype=np.float64)
        self.data = np.empty(cap, dtype=object)
        self.n    = 0
        self.ptr  = 0

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority: float, data):
        leaf = self.ptr + self.cap - 1
        self.data[self.ptr] = data
        self.update(leaf, priority)
        self.ptr = (self.ptr+1) % self.cap
        if self.n < self.cap:
            self.n += 1

    def update(self, leaf: int, priority: float):
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        idx = leaf
        while idx > 0:
            idx = (idx-1)//2
            self.tree[idx] += delta

    def get(self, s: float):
        idx = 0
        while True:
            l, r = 2*idx+1, 2*idx+2
            if l >= len(self.tree):
                break
            if s <= self.tree[l] or self.tree[r] == 0:
                idx = l
            else:
                s -= self.tree[l]; idx = r
        di = idx - (self.cap-1)
        return idx, self.tree[idx], self.data[di]


class PrioritizedReplayBuffer:
    def __init__(self, cap, alpha,
                 beta_start, beta_inc):
        self.cap      = cap
        self.alpha    = alpha
        self.beta     = beta_start
        self.beta_inc = beta_inc
        self.tree     = SumTree(cap)
        self._maxp    = 1.0

    @property
    def size(self):
        return self.tree.n

    def add(self, s, a, r, ns, done, td_err=None):
        pri = (self._maxp if td_err is None
               else (abs(td_err)+1e-6)**self.alpha)
        self.tree.add(pri, (s,a,r,ns,done))

    def sample(self, batch: int) -> dict:
        idx_list, pri_list, exp_list = [], [], []
        tot = self.tree.total
        seg = tot/batch
        for i in range(batch):
            s = random.uniform(seg*i, seg*(i+1))
            idx, pri, exp = self.tree.get(s)
            if exp is None:
                s2 = random.uniform(0, tot)
                idx, pri, exp = self.tree.get(s2)
            idx_list.append(idx)
            pri_list.append(max(pri, 1e-10))
            exp_list.append(exp)

        self.beta = min(1.0, self.beta+self.beta_inc)
        min_p  = min(pri_list)/(tot+1e-10)
        max_w  = (min_p*self.size)**(-self.beta)
        weights = np.array(
            [(p/tot*self.size)**(-self.beta)/max_w
             for p in pri_list], dtype=np.float32)

        return {
            "states":
                np.array([e[0] for e in exp_list],
                         np.float32),
            "actions":
                np.array([e[1] for e in exp_list],
                         np.int32),
            "rewards":
                np.array([e[2] for e in exp_list],
                         np.float32),
            "next_states":
                np.array([e[3] for e in exp_list],
                         np.float32),
            "dones":
                np.array([e[4] for e in exp_list],
                         np.float32),
            "weights":  weights,
            "indices":  idx_list,
        }

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            pri = (abs(td)+1e-6)**self.alpha
            self._maxp = max(self._maxp, pri)
            self.tree.update(idx, pri)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 5 — STATE BUILDER                                ║
# ╚══════════════════════════════════════════════════════════╝

class StateBuilder:
    EPS = 1e-10

    def build(self, data: pd.DataFrame, idx: int,
              features: List[str],
              portfolio: dict) -> np.ndarray:
        lb  = min(100, idx)
        mkt = []
        for f in features:
            if f not in data.columns:
                mkt.append(0.0); continue
            val = float(data[f].iloc[idx])
            win = data[f].iloc[max(0,idx-lb):idx]
            mu  = win.mean() if len(win)>0 else 0.0
            sg  = win.std()  if len(win)>1 else 1.0
            sg  = sg if sg > self.EPS else 1.0
            mkt.append(
                float(np.clip((val-mu)/sg, -5, 5)))

        p   = portfolio
        raw = [
            float(p.get("current_position",  0)),
            float(p.get("unrealized_pnl",    0)),
            float(p.get("holding_duration",  0)),
            float(p.get("current_drawdown",  0)),
            float(p.get("recent_win_rate",   0)),
            float(p.get("avg_trade_duration",0)),
            float(p.get("cash_ratio",        1)),
            float(p.get("trades_today",      0)),
            float(p.get("daily_pnl",         0)),
        ]
        mn, mx = min(raw), max(raw)
        rng    = mx - mn + self.EPS
        port   = [(v-mn)/rng for v in raw]
        reg    = self._regime(data, idx)
        state  = np.array(mkt+port+reg, dtype=np.float32)
        return np.nan_to_num(
            state, nan=0, posinf=0, neginf=0)

    def _regime(self, data: pd.DataFrame,
                idx: int) -> List[float]:
        st  = max(0, idx-100)
        r   = data["returns"].iloc[st:idx].fillna(0).values
        if len(r) < 5:
            return [0.0]*8
        sh   = float(
            np.mean(r)/(np.std(r)+self.EPS))
        rv   = float(
            np.std(r[-20:]) if len(r)>=20
            else np.std(r))
        lv   = float(
            np.std(r[-60:]) if len(r)>=60
            else np.std(r))
        vr   = rv/(lv+self.EPS)
        bull = float(np.mean(r>0))
        hurst= self._hurst(r)
        ac1  = (float(pd.Series(r).autocorr(1))
                if len(r)>2 else 0.0)
        ac5  = (float(pd.Series(r).autocorr(5))
                if len(r)>6 else 0.0)
        sk   = (float(stats.skew(r))
                if len(r)>3 else 0.0)
        ku   = (float(stats.kurtosis(r))
                if len(r)>3 else 0.0)
        feats = [sh,vr,bull,hurst,ac1,ac5,sk,ku]
        return [float(np.nan_to_num(f)) for f in feats]

    @staticmethod
    def _hurst(s: np.ndarray, ml=20) -> float:
        n = len(s)
        if n < 4: return 0.5
        lgs = range(2, min(ml, n//2))
        tau = [
            np.std(s[lg:]-s[:-lg])+1e-10
            for lg in lgs]
        if len(tau) < 2: return 0.5
        try:
            sl, _ = np.polyfit(
                np.log(list(lgs)),
                np.log(tau), 1)
            return float(np.clip(sl, 0, 1))
        except Exception:
            return 0.5


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 6 — REWARD FUNCTION  (FIX 2B applied)           ║
# ╚══════════════════════════════════════════════════════════╝

class RewardFunction:
    """
    FIX 2B — Inaction (opportunity-cost) penalty added.

    When the agent selects HOLD (action==1) while market
    volatility (vol_20) is above the activity threshold,
    a small negative reward is applied. This prevents the
    agent from learning that doing nothing is always safe,
    which is the root cause of the Hold-spiral where the
    agent stays in cash for thousands of steps.

    Threshold 0.0008 ≈ 0.08% per-bar std on H1 EURUSD,
    chosen to activate only during genuinely moving markets
    and stay silent during consolidation.
    Penalty magnitude 0.03 is intentionally small so it
    nudges rather than dominates the reward signal.
    """

    # Volatility level above which holding is penalised
    INACTION_VOL_THRESHOLD: float = 0.0008
    # Size of the opportunity-cost penalty
    INACTION_PENALTY:       float = 0.03

    def __init__(self, cfg: SystemConfig):
        self.w    = cfg.REWARD_WEIGHTS
        self.drf  = cfg.DAILY_RISK_FREE
        self.tc   = cfg.TRANSACTION_COST_BPS / 10_000
        self.ruin = cfg.EQUITY_RUIN_THRESHOLD
        self.hist: List[float] = []
        self.EPS = 1e-10

    def calc(self, action: int, port_ret: float,
             port_info: dict) -> float:
        self.hist.append(port_ret)

        sharpe    = self._sharpe()
        sortino   = self._sortino(port_ret)
        dd_pen    = self._dd_pen(
            port_info.get("current_drawdown", 0))
        trade_pen = self._trade_pen(action, port_info)
        pf_bonus  = self._pf_bonus()
        consist   = self._consist()
        ruin_pen  = self._ruin(port_info)

        total = (
            self.w.sharpe_weight        * sharpe
          + self.w.sortino_weight       * sortino
          + self.w.profit_factor_weight * pf_bonus
          + self.w.consistency_weight   * consist
          - self.w.drawdown_penalty     * dd_pen
          - self.w.trade_penalty        * trade_pen
          - self.w.ruin_penalty         * ruin_pen
        )

        # ── FIX 2B: Inaction penalty ─────────────────
        # Apply only when market is moving AND agent holds
        if action == 1:
            vol = float(port_info.get("vol_20", 0.0))
            if vol > self.INACTION_VOL_THRESHOLD:
                total -= self.INACTION_PENALTY

        return float(np.clip(total, -10, 10))

    def _sharpe(self) -> float:
        if len(self.hist) < 10: return 0.0
        r  = np.array(self.hist[-100:]) - self.drf
        sg = np.std(r)
        if sg < self.EPS: return 0.0
        return float(np.mean(r)/sg*np.sqrt(252))

    def _sortino(self, cr: float) -> float:
        if len(self.hist) < 10: return float(cr)
        r  = np.array(self.hist[-100:])
        dn = r[r<0]
        ds = np.std(dn) if len(dn)>0 else self.EPS
        return float((cr-self.drf)/(ds+self.EPS))

    def _dd_pen(self, dd: float) -> float:
        return float(np.exp(3*abs(dd))-1)

    def _trade_pen(self, a: int, pi: dict) -> float:
        return (0.001*self.tc
                if a != pi.get("previous_action", 1)
                else 0.0)

    def _pf_bonus(self) -> float:
        if len(self.hist) < 20: return 0.0
        r  = np.array(self.hist[-50:])
        gp = r[r>0].sum()
        gl = abs(r[r<0].sum())
        if gl < self.EPS: return 1.0
        return float(np.clip(
            np.log(gp/(gl+self.EPS)+self.EPS),
            -2, 2))

    def _consist(self) -> float:
        if len(self.hist) < 30: return 0.0
        r = np.array(self.hist[-30:])
        if np.mean(r)>0 and np.std(r)>self.EPS:
            return float(np.mean(r)/np.std(r))
        return -0.1

    def _ruin(self, pi: dict) -> float:
        cur  = pi.get("current_equity",  1.0)
        init = pi.get("initial_equity",  1.0)
        rat  = cur/(init+self.EPS)
        if rat < self.ruin:
            return float((self.ruin-rat)*10)
        return 0.0

    def reset(self):
        self.hist.clear()


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 7 — PORTFOLIO MANAGER  (FIX 1 + FIX 3)          ║
# ╚══════════════════════════════════════════════════════════╝

class PortfolioManager:
    """
    FIX 1 — Position size cap.
        Each trade is capped at cfg.EVAL_SIZE_CAP_PCT of
        INITIAL capital (not growing equity).  This makes
        the equity curve linear and metrics comparable
        across sessions.  Without this cap, a lucky early
        streak compounded position size exponentially and
        produced the 71,755% phantom total return.

    FIX 3 — Spread friction.
        BUY orders fill at (price + SPREAD_PRICE_UNITS).
        SELL orders fill at (price - SPREAD_PRICE_UNITS).
        This matches H1 EURUSD live-trading conditions and
        forces the RL agent to learn that frequent small
        trades must overcome the bid/ask cost to be worth
        taking.
    """

    def __init__(self, cfg: SystemConfig):
        self.init_cap  = cfg.INITIAL_CAPITAL
        self.tc        = cfg.TRANSACTION_COST_BPS / 10_000
        self.equity    = cfg.INITIAL_CAPITAL
        self.cash      = cfg.INITIAL_CAPITAL
        self.pos       = 0
        self.pos_size  = 0.0
        self.entry_px  = 0.0
        self.entry_t   = None
        self.peak_eq   = cfg.INITIAL_CAPITAL
        self.prev_act  = 1
        self.trades    = 0
        self.wins      = 0
        self.gross_p   = 0.0
        self.gross_l   = 0.0
        self.max_dd    = 0.0
        self.eq_curve: List[dict] = []
        self.trade_log: List[dict] = []
        self.daily_cnt = 0
        self.cur_day   = None
        self.daily_pnl = 0.0
        self.EPS       = 1e-10

        # FIX 1: fixed cap per trade in dollar terms
        self._size_cap = (cfg.INITIAL_CAPITAL *
                          cfg.EVAL_SIZE_CAP_PCT)

        # FIX 3: spread in price units (e.g. 0.00015)
        self._spread   = cfg.SPREAD_PRICE_UNITS

    def execute(self, action: int, price: float,
                time, size_usd: float) -> dict:
        # ── FIX 1: cap position size ─────────────────
        size_usd = min(size_usd, self._size_cap)

        port_ret = 0.0
        executed = False

        if action == 2:   # BUY
            # ── FIX 3: fill at Ask (price + spread) ──
            fill_px = price + self._spread

            if self.pos == -1:      # close short
                pnl  = self.pos_size*(
                    self.entry_px/(fill_px+self.EPS)-1)
                cost = self.pos_size*self.tc
                net  = pnl - cost
                self.cash += self.pos_size + net
                port_ret   = net/(self.equity+self.EPS)
                self._log_trade(net, time)
                self.pos = 0; self.pos_size = 0.0
                executed = True

            elif self.pos == 0:     # open long
                cost = size_usd * self.tc
                self.pos      = 1
                self.pos_size = size_usd
                self.entry_px = fill_px   # FIX 3
                self.entry_t  = time
                self.cash -= (size_usd + cost)
                self.cash  = max(self.cash, 0)
                executed   = True

            else:                   # mark long
                port_ret = self.pos_size*(
                    price/(self.entry_px+self.EPS)-1
                )/(self.equity+self.EPS)

        elif action == 0: # SELL
            # ── FIX 3: fill at Bid (price - spread) ──
            fill_px = price - self._spread

            if self.pos == 1:       # close long
                pnl  = self.pos_size*(
                    fill_px/(self.entry_px+self.EPS)-1)
                cost = self.pos_size * self.tc
                net  = pnl - cost
                self.cash += self.pos_size + net
                port_ret   = net/(self.equity+self.EPS)
                self._log_trade(net, time)
                self.pos = 0; self.pos_size = 0.0
                executed = True

            elif self.pos == 0:     # open short
                cost = size_usd * self.tc
                self.pos      = -1
                self.pos_size = size_usd
                self.entry_px = fill_px   # FIX 3
                self.entry_t  = time
                self.cash -= cost
                executed = True

            else:                   # mark short
                port_ret = self.pos_size*(
                    self.entry_px/(price+self.EPS)-1
                )/(self.equity+self.EPS)

        else:             # HOLD — mark-to-market
            if self.pos == 1:
                port_ret = self.pos_size*(
                    price/(self.entry_px+self.EPS)-1
                )/(self.equity+self.EPS)
            elif self.pos == -1:
                port_ret = self.pos_size*(
                    self.entry_px/(price+self.EPS)-1
                )/(self.equity+self.EPS)

        # Update equity
        if self.pos == 1:
            unr = self.pos_size*(
                price/(self.entry_px+self.EPS)-1)
            self.equity = (
                self.cash + self.pos_size + unr)
        elif self.pos == -1:
            unr = self.pos_size*(
                self.entry_px/(price+self.EPS)-1)
            self.equity = (
                self.cash + self.pos_size + unr)
        else:
            self.equity = self.cash
        self.equity = max(self.equity, 0.01)

        self.peak_eq = max(self.peak_eq, self.equity)
        dd = ((self.peak_eq-self.equity) /
              (self.peak_eq+self.EPS))
        self.max_dd = max(self.max_dd, dd)
        self.eq_curve.append({
            "time": time, "equity": self.equity,
            "dd": dd})

        day = str(time)[:10] if time else "?"
        if day != self.cur_day:
            self.cur_day   = day
            self.daily_cnt = 0
            self.daily_pnl = 0.0
        self.daily_pnl += port_ret * self.equity
        if executed:
            self.daily_cnt += 1
        self.prev_act = action

        return {
            "port_ret": port_ret,
            "executed": executed,
            "equity":   self.equity,
            "dd":       dd,
        }

    def _log_trade(self, pnl: float, time):
        self.trades += 1
        if pnl > 0:
            self.wins    += 1
            self.gross_p += pnl
        else:
            self.gross_l += abs(pnl)
        self.trade_log.append({
            "time": time, "pnl": pnl,
            "eq":   self.equity})

    def info(self, price: float = 0,
             time=None) -> dict:
        wr = (self.wins/self.trades
              if self.trades > 0 else 0)
        if self.pos == 1 and self.entry_px > 0:
            upnl = price/self.entry_px - 1
        elif self.pos == -1 and self.entry_px > 0:
            upnl = self.entry_px/price - 1
        else:
            upnl = 0
        hold = 0
        if (self.pos != 0 and
                self.entry_t and time):
            try:
                hold = ((time-self.entry_t)
                        .total_seconds()/3600)
            except Exception:
                pass
        dd = ((self.peak_eq-self.equity) /
              (self.peak_eq+self.EPS))
        return {
            "current_position":   self.pos,
            "current_equity":     self.equity,
            "initial_equity":     self.init_cap,
            "cash_ratio":
                self.cash/(self.equity+self.EPS),
            "unrealized_pnl":
                upnl*self.pos_size,
            "unrealized_pnl_pct": upnl,
            "current_drawdown":   dd,
            "recent_win_rate":    wr,
            "avg_win_size":
                self.gross_p/(self.wins+self.EPS),
            "avg_loss_size":
                self.gross_l/(
                    max(1,self.trades-self.wins)),
            "avg_trade_duration": 0,
            "holding_duration":   hold,
            "previous_action":    self.prev_act,
            "trades_today":       self.daily_cnt,
            "daily_pnl":          self.daily_pnl,
            "total_trades":       self.trades,
        }


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 8 — RISK MANAGER  (FIX 4 applied)               ║
# ╚══════════════════════════════════════════════════════════╝

class RiskManager:
    """
    FIX 4 — Dynamic ATR-based Stop-Loss / Take-Profit.

    Previous code used STOP_LOSS_PCT=2% and
    TAKE_PROFIT_PCT=6% which are fixed percentages.
    On H1 EURUSD a 2% move rarely happens in a day,
    so trades sat open for many bars without triggering
    SL/TP. This suppressed trade frequency and gave
    the RL agent no exit signal.

    The new logic:
        SL = entry_price ± (ATR_SL_MULT × atr_14)
        TP = entry_price ± (ATR_TP_MULT × atr_14)

    With ATR_SL_MULT=1.0 and ATR_TP_MULT=1.5:
    • Average EURUSD H1 ATR ≈ 0.0007 (7 pips)
    • SL ≈ 7 pips  (hit within hours on active sessions)
    • TP ≈ 10.5 pips (1.5× R:R)
    • Minimum win rate needed for PF>1.30 ≈ 41%

    The percentage fallback (STOP_LOSS_PCT /
    TAKE_PROFIT_PCT) is still used when atr_14 is
    unavailable in the mkt dict, ensuring no regression
    on synthetic or minimal-feature data.

    The mkt dict is extended in the main loop to include
    the atr_14 and entry_px values required here.
    """

    def __init__(self, cfg: SystemConfig):
        self.max_pos   = cfg.MAX_POSITION_SIZE
        self.sl_pct    = cfg.STOP_LOSS_PCT
        self.tp_pct    = cfg.TAKE_PROFIT_PCT
        self.atr_sl    = cfg.ATR_SL_MULT
        self.atr_tp    = cfg.ATR_TP_MULT
        self.max_d     = cfg.MAX_DAILY_TRADES
        self.max_dd    = cfg.MAX_DRAWDOWN_LIMIT
        self.ruin      = cfg.EQUITY_RUIN_THRESHOLD
        self.init_cap  = cfg.INITIAL_CAPITAL
        self.peak_eq   = cfg.INITIAL_CAPITAL
        self.cnt       = 0
        self.cur_day   = None
        self.EPS       = 1e-10

    def validate(self, action: int, pi: dict,
                 mkt: dict = None) -> int:
        eq   = pi.get("current_equity", self.init_cap)
        pos  = pi.get("current_position", 0)
        upnl = pi.get("unrealized_pnl_pct", 0)

        # Ruin guard
        if eq/(self.init_cap+self.EPS) < self.ruin:
            return 1

        # DD guard
        self.peak_eq = max(self.peak_eq, eq)
        dd = ((self.peak_eq-eq) /
              (self.peak_eq+self.EPS))
        if dd > self.max_dd:
            if pos > 0 and action == 0: return 0
            if pos < 0 and action == 2: return 2
            return 1

        # Daily trade limit
        today = mkt.get("date") if mkt else None
        if today and today != self.cur_day:
            self.cur_day = today; self.cnt = 0
        if (action != 1 and
                action != pi.get("previous_action", 1)):
            if self.cnt >= self.max_d:
                return 1

        # ── FIX 4: ATR-based SL / TP ─────────────────
        if pos != 0:
            atr      = mkt.get("atr_14", 0) if mkt else 0
            entry_px = mkt.get("entry_px", 0) if mkt else 0

            if atr > self.EPS and entry_px > self.EPS:
                # Use dynamic ATR boundaries
                sl_dist = self.atr_sl * atr
                tp_dist = self.atr_tp * atr

                if pos > 0:   # long position
                    sl_price = entry_px - sl_dist
                    tp_price = entry_px + tp_dist
                    price    = mkt.get(
                        "current_price", entry_px)
                    if price <= sl_price:
                        return 0  # hit stop — close long
                    if price >= tp_price:
                        return 0  # hit target — close long
                else:          # short position
                    sl_price = entry_px + sl_dist
                    tp_price = entry_px - tp_dist
                    price    = mkt.get(
                        "current_price", entry_px)
                    if price >= sl_price:
                        return 2  # hit stop — close short
                    if price <= tp_price:
                        return 2  # hit target — close short
            else:
                # Percentage fallback when ATR unavailable
                if upnl < -self.sl_pct:
                    return 0 if pos > 0 else 2
                if upnl >  self.tp_pct:
                    return 0 if pos > 0 else 2

        return action

    def position_size(self, pi: dict, mkt: dict,
                      confidence: float) -> float:
        eq   = pi.get("current_equity", self.init_cap)
        base = eq * self.max_pos
        cs   = float(
            np.clip(confidence/(confidence+1), 0.5, 1))
        cv   = mkt.get("vol_20", 0.02)
        av   = mkt.get("vol_60", 0.02)
        vs   = float(
            np.clip(av/(cv+self.EPS), 0.5, 2))
        wr   = pi.get("recent_win_rate", 0.5)
        aw   = pi.get("avg_win_size",    0.01)
        al   = abs(pi.get("avg_loss_size", 0.01))
        kelly = 0.0
        if al > self.EPS and wr > 0:
            kelly = float(np.clip(
                wr - (1-wr)/(
                    aw/(al+self.EPS)+self.EPS),
                0, 0.25))
        ks  = max(kelly/(self.max_pos+self.EPS), 0.1)
        sz  = min(base*cs*vs*ks, eq*self.max_pos)
        return max(sz, 0)

    def record(self):
        self.cnt += 1


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 9 — ENSEMBLE DOUBLE-Q AGENT                      ║
# ╚══════════════════════════════════════════════════════════╝

class EnsembleDoubleQAgent:
    def __init__(self, cfg: SystemConfig):
        self.cfg       = cfg
        self.n_a       = cfg.N_ACTIONS
        self.n_m       = cfg.N_ENSEMBLE_MODELS
        self.gamma     = cfg.GAMMA
        self.epsilon   = cfg.INITIAL_EPSILON
        self.eps_min   = cfg.EPSILON_MIN
        self.eps_decay = cfg.EPSILON_DECAY
        self.conf_thr  = cfg.CONFIDENCE_THRESHOLD
        self.batch_sz  = cfg.BATCH_SIZE
        self.fitted    = False
        self.step      = 0
        self.EPS       = 1e-10

        self.buf = PrioritizedReplayBuffer(
            cfg.REPLAY_BUFFER_CAPACITY,
            cfg.PRIORITY_ALPHA,
            cfg.PRIORITY_BETA_START,
            cfg.PRIORITY_BETA_INCREMENT)

        self.q1: Dict[int, List[dict]] = {}
        self.q2: Dict[int, List[dict]] = {}
        self._init()
        self.ew = np.ones(self.n_m)/self.n_m

    def _make(self, i: int) -> XGBRegressor:
        p = {
            **self.cfg.XGB_BASE,
            **self.cfg.ENSEMBLE_CONFIGS[i]
        }
        return XGBRegressor(**p)

    def _init(self):
        for a in range(self.n_a):
            self.q1[a] = [
                {"model": self._make(i),
                 "fi": None, "perf": 0.}
                for i in range(self.n_m)]
            self.q2[a] = [
                {"model": self._make(i),
                 "fi": None, "perf": 0.}
                for i in range(self.n_m)]

    def _pred_ens(self, ens: List[dict],
                  s: np.ndarray) -> List[float]:
        out = []
        for m in ens:
            x = (s[:,m["fi"]] if m["fi"] is not None
                 else s)
            try:
                out.append(
                    float(m["model"].predict(x)[0]))
            except Exception:
                out.append(0.0)
        return out

    def _q(self, ens: Dict, s: np.ndarray,
           a: int) -> float:
        ps = self._pred_ens(
            ens[a], s.reshape(1,-1))
        return float(np.average(ps, weights=self.ew))

    def select(self, s: np.ndarray,
               training=True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_a-1)
        if not self.fitted:
            return random.randint(0, self.n_a-1)
        sv = s.reshape(1,-1)
        qv = np.zeros(self.n_a)
        cf = np.zeros(self.n_a)
        for a in range(self.n_a):
            ps = (self._pred_ens(self.q1[a], sv) +
                  self._pred_ens(self.q2[a], sv))
            qv[a] = np.mean(ps)
            cf[a] = 1/(np.std(ps)+self.EPS)
        best = int(np.argmax(qv))
        nc   = cf[best]/(cf.sum()+self.EPS)
        if nc < self.conf_thr:
            return 1
        return best

    def confidence(self, s: np.ndarray) -> float:
        if not self.fitted: return 0.5
        sv = s.reshape(1,-1)
        all_ps = []
        for a in range(self.n_a):
            all_ps.extend(
                self._pred_ens(self.q1[a], sv))
        return float(1/(np.std(all_ps)+self.EPS))

    def store(self, s, a, r, ns, done):
        td = None
        if self.fitted:
            cq = self._q(self.q1, s, a)
            if done:
                tgt = r
            else:
                nqs = [self._q(self.q1, ns, aa)
                       for aa in range(self.n_a)]
                tgt = r + self.gamma * max(nqs)
            td = abs(tgt - cq)
        self.buf.add(s, a, r, ns, done, td)

    def train(self) -> Optional[dict]:
        if self.buf.size < self.batch_sz:
            return None
        if self.buf.size < self.cfg.MIN_BUFFER_SIZE:
            return None
        self.step += 1
        b  = self.buf.sample(self.batch_sz)
        S, A, R, NS, D = (
            b["states"], b["actions"], b["rewards"],
            b["next_states"], b["dones"])
        W   = b["weights"]
        idx = b["indices"]
        tgts = np.zeros(self.batch_sz, np.float32)
        tds  = np.zeros(self.batch_sz, np.float32)

        for i in range(self.batch_sz):
            if D[i]:
                tgts[i] = R[i]
            else:
                if random.random() < 0.5:
                    ba  = int(np.argmax(
                        [self._q(self.q1, NS[i], aa)
                         for aa in range(self.n_a)]))
                    tgt = (R[i] + self.gamma *
                           self._q(self.q2, NS[i], ba))
                else:
                    ba  = int(np.argmax(
                        [self._q(self.q2, NS[i], aa)
                         for aa in range(self.n_a)]))
                    tgt = (R[i] + self.gamma *
                           self._q(self.q1, NS[i], ba))
                tgts[i] = tgt
            tds[i] = abs(
                tgts[i] -
                self._q(self.q1, S[i], int(A[i])))

        self.buf.update_priorities(idx, tds)
        self._retrain(S, A, tgts, W)
        self.epsilon = max(
            self.eps_min,
            self.epsilon * self.eps_decay)
        return {
            "td_err":  float(np.mean(tds)),
            "mean_r":  float(np.mean(R)),
            "epsilon": self.epsilon,
            "buf":     self.buf.size,
        }

    def _retrain(self, S, A, tgts, W):
        n_feat = S.shape[1]
        for a in range(self.n_a):
            mask = A == a
            if mask.sum() < 10: continue
            Xa = S[mask]; ya = tgts[mask]
            wa = W[mask]; n  = len(Xa)
            for i in range(self.n_m):
                bsz = max(10, int(0.8*n))
                bi  = np.random.choice(
                    n, bsz, replace=True)
                cf  = self.cfg.ENSEMBLE_CONFIGS[i][
                    "colsample_bytree"]
                nf  = max(2, int(cf*n_feat))
                fi  = np.sort(np.random.choice(
                    n_feat, nf, replace=False))
                self.q1[a][i]["fi"] = fi
                self.q2[a][i]["fi"] = fi

                Xt  = Xa[bi][:,fi]
                Xt  = Xt + np.random.normal(
                    0, self.cfg.NOISE_INJECTION_LEVEL,
                    Xt.shape)
                dm  = np.random.binomial(
                    1,
                    1-self.cfg.FEATURE_DROPOUT_RATE,
                    Xt.shape)
                Xt  = Xt * dm
                try:
                    self.q1[a][i]["model"].fit(
                        Xt, ya[bi],
                        sample_weight=wa[bi])
                except Exception as e:
                    logger.debug(
                        f"Q1 fit a={a} m={i}: {e}")

                bi2 = np.random.choice(
                    n, bsz, replace=True)
                Xt2 = Xa[bi2][:,fi]
                Xt2 = Xt2 + np.random.normal(
                    0, self.cfg.NOISE_INJECTION_LEVEL,
                    Xt2.shape)
                dm2 = np.random.binomial(
                    1,
                    1-self.cfg.FEATURE_DROPOUT_RATE,
                    Xt2.shape)
                Xt2 = Xt2 * dm2
                try:
                    self.q2[a][i]["model"].fit(
                        Xt2, ya[bi2],
                        sample_weight=wa[bi2])
                except Exception as e:
                    logger.debug(
                        f"Q2 fit a={a} m={i}: {e}")
        self.fitted = True

    def update_ew(self, val_S: np.ndarray,
                  val_tgts: np.ndarray):
        scores = np.zeros(self.n_m)
        for i in range(self.n_m):
            errs, cnt = 0.0, 0
            for a in range(self.n_a):
                fi = self.q1[a][i]["fi"]
                if fi is None: continue
                x  = val_S[:,fi]
                try:
                    p = self.q1[a][i]["model"].predict(x)
                    errs += mean_squared_error(
                        val_tgts, p)
                    cnt  += 1
                except Exception:
                    pass
            if cnt > 0:
                scores[i] = -(errs/cnt)
        ex = np.exp(scores - scores.max())
        self.ew = ex / ex.sum()


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 10 — WALK-FORWARD ENGINE                         ║
# ╚══════════════════════════════════════════════════════════╝

class WalkForward:
    def __init__(self, cfg: SystemConfig):
        self.tw   = cfg.TRAIN_WINDOW
        self.ri   = cfg.RETRAIN_INTERVAL
        self.mt   = cfg.MIN_TRAIN_SAMPLES
        self.rp   = cfg.REGIME_CHANGE_P_VALUE
        self.rl   = cfg.REGIME_LOOKBACK
        self.rd   = cfg.RECENCY_WEIGHT_DECAY
        self.last = 0; self.cnt = 0
        self.recent: deque = deque(maxlen=200)

    def record(self, r: float):
        self.recent.append(r)

    def should_retrain(self, t: int,
                       data: pd.DataFrame) -> bool:
        return (
            (t-self.last) >= self.ri or
            self.regime_change(data, t) or
            self._degraded())

    def regime_change(self, data: pd.DataFrame,
                      t: int) -> bool:
        if t < 2*self.rl: return False
        r   = data["returns"].fillna(0)
        rec = r.iloc[t-self.rl:t].values
        prv = r.iloc[t-2*self.rl:t-self.rl].values
        try:
            _, p = ks_2samp(rec, prv)
            if p < self.rp:
                logger.info(
                    f"Regime change at step "
                    f"{t} p={p:.5f}")
                return True
        except Exception:
            pass
        return False

    def _degraded(self) -> bool:
        if len(self.recent) < 100: return False
        arr = list(self.recent)
        r50 = np.mean(arr[-50:])
        p50 = np.mean(arr[-100:-50])
        if p50 != 0 and r50 < p50 * 0.5:
            logger.info(
                "Performance degradation detected")
            return True
        return False

    def window(self, data: pd.DataFrame, t: int):
        st    = max(0, t-self.tw)
        chunk = data.iloc[st:t].copy()
        n     = len(chunk)
        w     = np.exp(np.linspace(self.rd, 0, n))
        self.last = t; self.cnt += 1
        return chunk, w/w.sum()


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 11 — HYPERPARAMETER OPTIMIZER                    ║
# ╚══════════════════════════════════════════════════════════╝

class HPOptimizer:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def optimize(self, X: np.ndarray,
                 y: np.ndarray) -> dict:
        def objective(trial):
            params = {
                "n_estimators":
                    trial.suggest_int(
                        "n_estimators", 100, 1000),
                "max_depth":
                    trial.suggest_int(
                        "max_depth", 3, 10),
                "learning_rate":
                    trial.suggest_float(
                        "lr", 0.005, 0.3, log=True),
                "min_child_weight":
                    trial.suggest_int("mcw", 1, 20),
                "subsample":
                    trial.suggest_float(
                        "ss", 0.5, 1.0),
                "colsample_bytree":
                    trial.suggest_float(
                        "cs", 0.3, 1.0),
                "gamma":
                    trial.suggest_float("gm", 0, 10),
                "reg_alpha":
                    trial.suggest_float(
                        "ra", 1e-6, 100, log=True),
                "reg_lambda":
                    trial.suggest_float(
                        "rl", 1e-6, 100, log=True),
                "tree_method": "hist",
                "verbosity":   0,
                "random_state":42,
            }
            folds  = self._cv_folds(
                len(X),
                self.cfg.OPTUNA_CV_SPLITS,
                self.cfg.EMBARGO_PCT)
            scores = []
            for fi, (tri, tei) in enumerate(folds):
                m = XGBClassifier(**params)
                try:
                    m.fit(
                        X[tri], y[tri],
                        eval_set=[(X[tei], y[tei])],
                        verbose=False)
                    auc = roc_auc_score(
                        y[tei],
                        m.predict_proba(X[tei])[:,1])
                    scores.append(auc)
                except Exception:
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
            show_progress_bar=False)
        logger.info(
            f"Best AUC: {study.best_value:.4f}")
        return study.best_params

    @staticmethod
    def _cv_folds(n, ns, ep):
        emb  = int(n*ep); fsz = n//ns; folds = []
        for i in range(ns):
            ts = i*fsz
            te = (i+1)*fsz if i < ns-1 else n
            tr = np.concatenate([
                np.arange(0, max(0, ts-emb)),
                np.arange(min(n, te+emb), n)
            ]).astype(int)
            tv = np.arange(ts, te).astype(int)
            if len(tr) >= 10 and len(tv) >= 5:
                folds.append((tr, tv))
        return folds


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 12 — ONNX EXPORT                                 ║
# ╚══════════════════════════════════════════════════════════╝

class ONNXExporter:
    def __init__(self, cfg: SystemConfig):
        self.out = cfg.OUTPUT_DIR

    def _export_one(self, mdl, net_name: str,
                    a: int, mi: int,
                    in_dim: int) -> str:
        import onnx
        from onnx import shape_inference

        init_types = [
            ("input", FloatTensorType([1, in_dim]))
        ]
        try:
            onnx_model = convert_sklearn(
                mdl,
                initial_types=init_types,
                target_opset=15,
                options={
                    type(mdl): {"nocopy": True}
                })
        except Exception:
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
        logger.info(
            f"  ✔ {fname}: "
            f"in{in_sh} → out{out_sh} = {val:.6f}")
        return fpath

    def export_agent(self,
                     agent: "EnsembleDoubleQAgent",
                     selected: List[str],
                     scaler_params: dict,
                     state_dim: int,
                     report: dict):
        os.makedirs(self.out, exist_ok=True)
        exported = []

        for net_name, ens in [("q1", agent.q1),
                               ("q2", agent.q2)]:
            for a in range(agent.n_a):
                for mi in range(agent.n_m):
                    m      = ens[a][mi]
                    mdl    = m["model"]
                    fi     = m["fi"]
                    in_dim = (len(fi)
                              if fi is not None
                              else state_dim)
                    try:
                        self._export_one(
                            mdl, net_name,
                            a, mi, in_dim)
                        exported.append(
                            f"{net_name}_a{a}"
                            f"_m{mi}.onnx")
                    except Exception as e:
                        logger.warning(
                            f"ONNX export failed "
                            f"{net_name} a={a} "
                            f"m={mi}: {e}")
                        try:
                            mdl.save_model(
                                os.path.join(
                                    self.out,
                                    f"{net_name}_a{a}"
                                    f"_m{mi}.json"))
                        except Exception:
                            pass

        fi_map = {}
        for a in range(agent.n_a):
            for mi in range(agent.n_m):
                fi  = agent.q1[a][mi]["fi"]
                key = f"a{a}_m{mi}"
                fi_map[key] = (
                    fi.tolist()
                    if fi is not None
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
                "N_ACTIONS":
                    agent.n_a,
                "N_ENSEMBLE_MODELS":
                    agent.n_m,
                "GAMMA":
                    agent.gamma,
                "STATE_DIM":
                    state_dim,
                "N_MARKET_FEATURES":
                    len(selected),
                "N_PORTFOLIO_FEATURES": 9,
                "N_REGIME_FEATURES":    8,
                "SELECTED_FEATURES":
                    selected,
                "CONFIDENCE_THRESHOLD":
                    agent.conf_thr,
            }, f, indent=2)
        with open(os.path.join(
                self.out,
                "training_report.json"), "w") as f:
            json.dump(
                {k: float(v)
                 if isinstance(v, (np.floating, float))
                 else v
                 for k,v in report.items()},
                f, indent=2)

        logger.info(
            f"Exported {len(exported)} ONNX models "
            f"to {self.out}/")
        return exported


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 13 — PERFORMANCE MONITOR                         ║
# ╚══════════════════════════════════════════════════════════╝

class PerfMonitor:
    def __init__(self, cfg: SystemConfig):
        self.rf  = cfg.RISK_FREE_RATE
        self.EPS = 1e-10

    def evaluate(self, eq_curve: List[dict],
                 trade_log: List[dict]) -> dict:
        if len(eq_curve) < 2: return {}
        eq  = np.array(
            [e["equity"] for e in eq_curve])
        ret = np.diff(eq)/(eq[:-1]+self.EPS)
        n_d = max(len(ret)/24, 1)
        tr  = eq[-1]/eq[0]-1
        ar  = (1+tr)**(252/n_d)-1
        av  = np.std(ret)*np.sqrt(252*24)
        mx  = self._mdd(eq)
        dn  = ret[ret<0]
        dv  = (np.std(dn) if len(dn)>0
               else self.EPS)*np.sqrt(252*24)
        sh  = (ar-self.rf)/(av+self.EPS)
        so  = (ar-self.rf)/(dv+self.EPS)
        ca  = ar/(abs(mx)+self.EPS)
        tdf = (pd.DataFrame(trade_log)
               if trade_log
               else pd.DataFrame())
        nt  = len(tdf)
        wt  = (int((tdf["pnl"]>0).sum())
               if nt > 0 else 0)
        wr  = wt/(nt+self.EPS)
        gp  = (float(tdf[tdf["pnl"]>0]["pnl"].sum())
               if wt > 0 else 0)
        gl  = (float(
                   tdf[tdf["pnl"]<0]["pnl"]
                   .abs().sum())
               if nt-wt > 0 else 0)
        pf  = gp/(gl+self.EPS)
        m   = {
            "total_return":   tr,
            "ann_return":     ar,
            "ann_vol":        av,
            "max_dd":         mx,
            "sharpe":         sh,
            "sortino":        so,
            "calmar":         ca,
            "trades":         nt,
            "win_rate":       wr,
            "profit_factor":  pf,
        }
        self._print(m)
        return m

    def _mdd(self, eq):
        pk = eq[0]; mx = 0
        for e in eq:
            pk = max(pk, e)
            mx = max(mx, (pk-e)/(pk+self.EPS))
        return mx

    def _print(self, m):
        sep = "═"*50
        print(f"\n╔{sep}╗")
        print(f"║{'PERFORMANCE REPORT':^50}║")
        print(f"╠{sep}╣")
        print(
            f"║  Total Return:    "
            f"{m['total_return']:>10.2%}"
            f"{'':>28}║")
        print(
            f"║  Ann. Return:     "
            f"{m['ann_return']:>10.2%}"
            f"{'':>28}║")
        print(
            f"║  Ann. Vol:        "
            f"{m['ann_vol']:>10.2%}"
            f"{'':>28}║")
        print(
            f"║  Max Drawdown:    "
            f"{m['max_dd']:>10.2%}"
            f"{'':>28}║")
        print(
            f"║  Sharpe:          "
            f"{m['sharpe']:>10.3f}"
            f"{'':>28}║")
        print(
            f"║  Sortino:         "
            f"{m['sortino']:>10.3f}"
            f"{'':>28}║")
        print(
            f"║  Calmar:          "
            f"{m['calmar']:>10.3f}"
            f"{'':>28}║")
        print(
            f"║  Trades:          "
            f"{m['trades']:>10}"
            f"{'':>28}║")
        print(
            f"║  Win Rate:        "
            f"{m['win_rate']:>10.2%}"
            f"{'':>28}║")
        print(
            f"║  Profit Factor:   "
            f"{m['profit_factor']:>10.2f}"
            f"{'':>28}║")
        print(f"╚{sep}╝\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 14 — SCALER BUILDER                              ║
# ╚══════════════════════════════════════════════════════════╝

def build_scaler_params(feat_data: pd.DataFrame,
                        selected: List[str]) -> dict:
    params = {}
    for f in selected:
        if f not in feat_data.columns:
            params[f] = {"mean": 0.0, "std": 1.0}
            continue
        mu  = float(feat_data[f].mean())
        std = float(feat_data[f].std())
        if std < 1e-10: std = 1.0
        params[f] = {"mean": mu, "std": std}
    return params


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE  (mkt dict extended for FIX 4)            ║
# ╚══════════════════════════════════════════════════════════╝

def train_and_export(
    cfg:      SystemConfig = None,
    filepath: str          = None,
    source:   str          = "synthetic",
) -> dict:
    if cfg is None: cfg = SystemConfig()
    print("\n╔══════════════════════════════════════╗")
    print("║  XGBoost-RL  TRAINING PIPELINE       ║")
    print("╚══════════════════════════════════════╝\n")

    print("Phase 1: Data")
    data = DataIngestion.load(
        source, filepath, cfg.HISTORICAL_BARS)
    print(
        f"  {len(data):,} bars  "
        f"{data['timestamp'].iloc[0]} → "
        f"{data['timestamp'].iloc[-1]}\n")

    print("Phase 2: Features")
    fe        = FeatureEngine(cfg)
    feat_data = fe.build(data)
    print(
        f"  {feat_data.shape[1]} columns, "
        f"{len(feat_data):,} rows\n")

    print("Phase 3: Feature Selection")
    sel      = FeatureSelector(cfg)
    target   = (feat_data["returns"].shift(-1) > 0
                ).astype(int).iloc[:-1]
    feat_sub = feat_data.iloc[:-1].copy()
    selected = sel.select(
        feat_sub, target, cfg.MAX_FEATURES_SELECTED)

    cut = int(0.6 * len(feat_sub))
    prob, shift = sel.adversarial_check(
        feat_sub[selected].iloc[:cut],
        feat_sub[selected].iloc[cut:])
    if shift:
        selected = [
            f for f in selected if f not in prob]
        print(
            f"  Removed {len(prob)} shifted features"
            f" → {len(selected)} remain\n")
    else:
        print("  No distribution shift\n")

    print("Phase 4: Hyperparameter Optimization")
    Xopt = (feat_sub[selected].iloc[:cut]
            .fillna(0).values.astype(np.float32))
    yopt = target.iloc[:cut].values
    hp   = HPOptimizer(cfg)
    best = hp.optimize(Xopt, yopt)
    cfg.XGB_BASE.update(best)
    print(f"  Best params: {best}\n")

    print("Phase 5: Scaler Parameters")
    scaler_params = build_scaler_params(
        feat_sub[selected].iloc[:cut], selected)

    print("Phase 6: Initialize Agent & Environment")
    agent  = EnsembleDoubleQAgent(cfg)
    port   = PortfolioManager(cfg)
    risk   = RiskManager(cfg)
    reward = RewardFunction(cfg)
    wf     = WalkForward(cfg)
    sb     = StateBuilder()
    perf   = PerfMonitor(cfg)

    test_pi    = port.info(
        float(feat_data["close"].iloc[200]))
    test_state = sb.build(
        feat_data, 200, selected, test_pi)
    state_dim  = len(test_state)
    print(f"  State dim: {state_dim}\n")

    print("Phase 7: Walk-Forward Training")
    from tqdm import tqdm
    start_idx = max(cfg.MIN_TRAIN_SAMPLES, 200)
    end_idx   = len(feat_data) - 1
    selected  = [
        f for f in selected
        if f in feat_data.columns]

    all_results: List[dict] = []
    train_log:   List[dict] = []

    for t in tqdm(range(start_idx, end_idx),
                  desc="Training", unit="step"):

        if wf.should_retrain(t, feat_data):
            if wf.regime_change(feat_data, t):
                agent  = EnsembleDoubleQAgent(cfg)
                reward.reset()
            if len(all_results) > 100:
                rs = np.array(
                    [r["state"]
                     for r in all_results[-100:]])
                rt = np.array(
                    [r["reward"]
                     for r in all_results[-100:]])
                agent.update_ew(rs, rt)

        pi    = port.info(
            float(feat_data["close"].iloc[t]),
            feat_data["timestamp"].iloc[t])
        state = sb.build(
            feat_data, t, selected, pi)
        raw_a = agent.select(state, training=True)

        row = feat_data.iloc[t]

        # ── FIX 4: extend mkt with atr_14 and
        # current_price so RiskManager can compute
        # dynamic ATR stops; also pass entry_px from
        # PortfolioManager so the SL/TP is anchored
        # to the actual fill price. ─────────────────
        mkt = {
            "vol_20":        float(
                row.get("vol_20", 0.02)),
            "vol_60":        float(
                row.get("vol_60", 0.02)),
            "atr_14":        float(
                row.get("atr_14", 0.0)),
            "current_price": float(
                feat_data["close"].iloc[t]),
            "entry_px":      port.entry_px,
            "date":          str(
                feat_data["timestamp"].iloc[t])[:10],
        }

        val_a = risk.validate(raw_a, pi, mkt)
        conf  = agent.confidence(state)
        size  = risk.position_size(pi, mkt, conf)
        price = float(feat_data["close"].iloc[t])
        time  = feat_data["timestamp"].iloc[t]
        res   = port.execute(val_a, price, time, size)
        if res["executed"]:
            risk.record()

        # Pass vol_20 through port_info so the
        # inaction penalty in RewardFunction can
        # read it without a separate lookup.
        upi   = port.info(price, time)
        upi["vol_20"] = mkt["vol_20"]

        rwd   = reward.calc(
            val_a, res["port_ret"], upi)
        wf.record(rwd)

        if t < end_idx - 1:
            ns   = sb.build(
                feat_data, t+1, selected, upi)
            done = False
        else:
            ns = state; done = True

        agent.store(state, val_a, rwd, ns, done)
        tm = agent.train()
        if tm: train_log.append(tm)

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
            print(
                f"\nEmergency stop at step {t}: "
                f"equity=${res['equity']:,.2f}")
            break

    print("\nPhase 8: Evaluation")
    metrics = perf.evaluate(
        port.eq_curve, port.trade_log)

    print("Phase 9: ONNX Export")
    exporter = ONNXExporter(cfg)
    exported = exporter.export_agent(
        agent, selected, scaler_params,
        state_dim, metrics)
    print(f"  Exported {len(exported)} ONNX files\n")

    print("╔══════════════════════════════════════╗")
    print("║  TRAINING COMPLETE                   ║")
    print(
        f"║  Artifacts → {cfg.OUTPUT_DIR:<24}║")
    print("╚══════════════════════════════════════╝")
    return metrics
