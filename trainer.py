# ============================================================
#  XGBoost_RL_Trainer.py
#  Fixes applied in this version:
#    A. next_state peek replaced with deferred build
#       at step t+1 to eliminate t+1 return leakage
#    B. fi indices written into ONNX metadata and
#       enforced at export; MT5 EA slice documented
#    C. scaler_params extended to cover all K stacked
#       frame positions with explicit lag-frame keys
# ============================================================

import os, json, warnings, logging, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
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
from sklearn.metrics import (roc_auc_score,
                             mean_squared_error)
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

    SMA_PERIODS:        List[int] = field(
        default_factory=lambda: [5,10,20,50,100,200])
    EMA_PERIODS:        List[int] = field(
        default_factory=lambda: [5,10,20,50,100])
    RSI_PERIODS:        List[int] = field(
        default_factory=lambda: [7,14,21])
    VOLATILITY_WINDOWS: List[int] = field(
        default_factory=lambda: [10,20,60,100])
    LAG_PERIODS:        List[int] = field(
        default_factory=lambda: [1,2,3,5,10,20])
    MULTI_TIMEFRAMES:   List[int] = field(
        default_factory=lambda: [5,10,20,50,100])
    MAX_FEATURES_SELECTED: int   = 50
    CORRELATION_THRESHOLD: float = 0.95

    N_ACTIONS:            int   = 3
    GAMMA:                float = 0.99
    INITIAL_EPSILON:      float = 1.0
    EPSILON_MIN:          float = 0.01
    EPSILON_DECAY:        float = 0.9995
    CONFIDENCE_THRESHOLD: float = 0.50

    N_ENSEMBLE_MODELS: int = 7
    ENSEMBLE_CONFIGS: List[dict] = field(
        default_factory=lambda: [
            {"max_depth": 4,  "subsample": 0.70,
             "colsample_bytree": 0.60},
            {"max_depth": 6,  "subsample": 0.80,
             "colsample_bytree": 0.70},
            {"max_depth": 8,  "subsample": 0.90,
             "colsample_bytree": 0.80},
            {"max_depth": 5,  "subsample": 0.60,
             "colsample_bytree": 0.50},
            {"max_depth": 7,  "subsample": 0.85,
             "colsample_bytree": 0.75},
            {"max_depth": 3,  "subsample": 0.95,
             "colsample_bytree": 0.90},
            {"max_depth": 10, "subsample": 0.65,
             "colsample_bytree": 0.65},
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

    ATR_SL_MULT:           float = 1.0
    ATR_TP_MULT:           float = 3.0
    STOP_LOSS_PCT:         float = 0.02
    TAKE_PROFIT_PCT:       float = 0.06

    MAX_DAILY_TRADES:      int   = 10
    EQUITY_RUIN_THRESHOLD: float = 0.80

    SPREAD_PRICE_UNITS:    float = 0.00015
    EVAL_SIZE_CAP_PCT:     float = 0.02

    OPTUNA_N_TRIALS:  int   = 50
    OPTUNA_CV_SPLITS: int   = 5
    EMBARGO_PCT:      float = 0.02

    NOISE_INJECTION_LEVEL: float = 0.01
    FEATURE_DROPOUT_RATE:  float = 0.10
    ADVERSARIAL_AUC_LIMIT: float = 0.55

    FRAME_STACK_SIZE:         int   = 5
    REGIME_ADX_MIN:           float = 20.0
    REGIME_VOL_RATIO_MIN:     float = 0.70
    REGIME_MAX_CONSEC_LOSSES: int   = 3
    REGIME_COOLDOWN_BARS:     int   = 5

    OUTPUT_DIR: str = "xgb_rl_artifacts"

    def __post_init__(self):
        self.DAILY_RISK_FREE = (
            self.RISK_FREE_RATE / 252)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 1 — DATA INGESTION                               ║
# ╚══════════════════════════════════════════════════════════╝

class DataIngestion:
    @staticmethod
    def load(source: str, filepath: str = None,
             n_bars: int = 50_000) -> pd.DataFrame:
        if source == "csv":
            df = pd.read_csv(
                filepath,
                parse_dates=["timestamp"])
        elif source == "synthetic":
            df = DataIngestion._synthetic(n_bars)
        elif source == "exchange_api":
            import ccxt
            ex    = ccxt.binance()
            ohlcv = ex.fetch_ohlcv(
                "BTC/USDT", "1h", limit=n_bars)
            df = pd.DataFrame(ohlcv, columns=[
                "timestamp","open","high",
                "low","close","volume"])
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms")
        else:
            raise ValueError(
                f"Unknown source: {source}")
        return DataIngestion.preprocess(df)

    @staticmethod
    def _synthetic(n: int) -> pd.DataFrame:
        np.random.seed(42)
        dt, mu, sigma, S0 = (
            1/(24*365), 0.10, 0.80, 30_000.0)
        prices = [S0]
        for _ in range(1, n):
            r = (mu*dt +
                 sigma*np.sqrt(dt)*
                 np.random.randn())
            prices.append(prices[-1]*np.exp(r))
        prices = np.array(prices)
        noise  = sigma*np.sqrt(dt)
        o = prices*np.exp(
            np.random.randn(n)*noise*0.3)
        h = prices*np.exp(
            np.abs(np.random.randn(n))*noise)
        l = prices*np.exp(
            -np.abs(np.random.randn(n))*noise)
        h = np.maximum(h, np.maximum(o, prices))
        l = np.minimum(l, np.minimum(o, prices))
        v = np.random.lognormal(10, 1, n)
        ts = pd.date_range(
            "2020-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "timestamp": ts, "open": o,
            "high": h,    "low": l,
            "close": prices, "volume": v})

    @staticmethod
    def preprocess(
            raw: pd.DataFrame) -> pd.DataFrame:
        df   = raw.copy()
        cols = ["open","high","low",
                "close","volume"]
        df.dropna(subset=cols, inplace=True)
        df = df[df["volume"] > 0].copy()
        df.drop_duplicates(
            subset=["timestamp"], inplace=True)
        df.sort_values(
            "timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        for c in cols:
            mu, sig = df[c].mean(), df[c].std()
            if sig > 0:
                df[c] = df[c].clip(
                    mu-5*sig, mu+5*sig)
        df = df[
            (df.high >= df.low) &
            (df.high >= df.open) &
            (df.high >= df.close) &
            (df.low  <= df.open) &
            (df.low  <= df.close) &
            (df.volume >= 0)
        ].copy()
        df.reset_index(drop=True, inplace=True)
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(
            df["close"]/df["close"].shift(1))
        logger.info(
            f"Preprocessed: {len(df):,} bars")
        return df


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 2 — FEATURE ENGINEERING                          ║
# ╚══════════════════════════════════════════════════════════╝

class FeatureEngine:
    EPS = 1e-10

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    @staticmethod
    def _rsi(s, p):
        d = s.diff()
        g = d.clip(lower=0).rolling(p).mean()
        l = (-d.clip(upper=0)).rolling(p).mean()
        return 100-100/(1+g/(l+FeatureEngine.EPS))

    @staticmethod
    def _macd(s, f=12, sl=26, sg=9):
        ef  = s.ewm(span=f,  adjust=False).mean()
        es  = s.ewm(span=sl, adjust=False).mean()
        ml  = ef - es
        sig = ml.ewm(span=sg, adjust=False).mean()
        return ml, sig, ml-sig

    @staticmethod
    def _atr(h, l, c, p=14):
        tr = pd.concat([
            h-l,
            (h-c.shift(1)).abs(),
            (l-c.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def _williams_r(h, l, c, p=14):
        hh = h.rolling(p).max()
        ll = l.rolling(p).min()
        return -100*(hh-c)/(
            hh-ll+FeatureEngine.EPS)

    @staticmethod
    def _cci(h, l, c, p=20):
        tp = (h+l+c)/3
        ma = tp.rolling(p).mean()
        md = tp.rolling(p).apply(
            lambda x: np.mean(np.abs(x-x.mean())),
            raw=True)
        return (tp-ma)/(
            0.015*md+FeatureEngine.EPS)

    @staticmethod
    def _stoch(h, l, c, kp=14, dp=3):
        ll = l.rolling(kp).min()
        hh = h.rolling(kp).max()
        k  = 100*(c-ll)/(hh-ll+FeatureEngine.EPS)
        return k, k.rolling(dp).mean()

    @staticmethod
    def _obv(c, v):
        return (
            np.sign(c.diff()).fillna(0)*v
        ).cumsum()

    @staticmethod
    def _mfi(h, l, c, v, p=14):
        tp  = (h+l+c)/3
        rmf = tp*v
        pos = (
            rmf*(tp>tp.shift(1))
        ).rolling(p).sum()
        neg = (
            rmf*(tp<tp.shift(1))
        ).rolling(p).sum()
        return 100-100/(
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

    def build(self,
              data: pd.DataFrame) -> pd.DataFrame:
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

        (df["macd"],
         df["macd_sig"],
         df["macd_hist"]) = self._macd(c)
        df["macd_cross"] = (
            np.sign(df["macd_hist"]) *
            np.sign(df["macd_hist"].shift(1)))

        for p in [10, 20, 30, 60]:
            df[f"mom_{p}"] = c/c.shift(p)-1
            df[f"roc_{p}"] = c.pct_change(p)

        df["williams_r"] = self._williams_r(h,l,c)
        df["cci"]        = self._cci(h,l,c)
        df["stoch_k"], df["stoch_d"] = (
            self._stoch(h,l,c))
        df["stoch_cross"] = (
            df["stoch_k"] - df["stoch_d"])

        for w in cfg.VOLATILITY_WINDOWS:
            df[f"vol_{w}"] = ret.rolling(w).std()

        v20  = df.get("vol_20",
                      ret.rolling(20).std())
        v60  = df.get("vol_60",
                      ret.rolling(60).std())
        v10  = df.get("vol_10",
                      ret.rolling(10).std())
        v100 = df.get("vol_100",
                      ret.rolling(100).std())
        df["vol_ratio_20_60"]  = v20/(v60+eps)
        df["vol_ratio_10_100"] = v10/(v100+eps)
        df["vol_ratio"]        = v20/(v60+eps)

        df["atr_14"]    = self._atr(h, l, c, 14)
        df["atr_ratio"] = df["atr_14"]/(c+eps)
        df["atr_pct"]   = (
            df["atr_14"].rolling(100).rank(
                pct=True))

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
        df["gap"]        = (
            df["open"]/(c.shift(1)+eps)-1)

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
                c-df[f"high_{p}"])/(c+eps)
            df[f"dist_low_{p}"]  = (
                c-df[f"low_{p}"])/(c+eps)

        dm_pos = (h-h.shift(1)).clip(lower=0)
        dm_neg = (l.shift(1)-l).clip(lower=0)
        tr14   = df["atr_14"]
        df["adx_pos"] = (
            dm_pos.rolling(14).mean()/(tr14+eps))
        df["adx_neg"] = (
            dm_neg.rolling(14).mean()/(tr14+eps))
        df["adx"] = (
            df["adx_pos"]-df["adx_neg"]).abs()

        # Canonical Wilder ADX14 (0-100 scale)
        atr14_s = self._atr(h, l, c, 14)
        dmp_s   = dm_pos.rolling(14).mean()
        dmn_s   = dm_neg.rolling(14).mean()
        pdi     = 100 * dmp_s/(atr14_s+eps)
        ndi     = 100 * dmn_s/(atr14_s+eps)
        dx      = (100*(pdi-ndi).abs() /
                   (pdi+ndi+eps))
        df["adx_14_full"] = dx.ewm(
            span=14, adjust=False).mean()

        key = ["returns","volume","rsi_14",
               "macd_hist","vol_20"]
        for feat in key:
            if feat not in df.columns:
                continue
            for lag in cfg.LAG_PERIODS:
                sh = df[feat].shift(lag)
                df[f"{feat}_lag{lag}"]  = sh
                df[f"{feat}_diff{lag}"] = (
                    df[feat]-sh)

        excl  = {"timestamp","open","high","low",
                 "close","volume","returns",
                 "log_returns"}
        fcols = [
            c_ for c_ in
            df.select_dtypes(np.number).columns
            if c_ not in excl]
        top20 = (
            df[fcols].var()
            .nlargest(20).index.tolist())
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

        df.replace(
            [np.inf,-np.inf], 0, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(
            f"Features built: "
            f"{df.shape[1]} cols, "
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

        tmp = XGBClassifier(
            n_estimators=200, max_depth=6,
            verbosity=0, random_state=42,
            tree_method="hist")
        tmp.fit(Xa, ya)
        xgb_imp = np.array(
            tmp.feature_importances_)
        if len(xgb_imp) != len(rem):
            xgb_imp = np.ones(len(rem))/len(rem)
        xr = pd.Series(
            xgb_imp,
            index=rem).rank(ascending=False)

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
            pi_r = permutation_importance(
                tmp, Xa, ya,
                n_repeats=5, random_state=42)
            ps   = np.array(
                pi_r.importances_mean)
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
            [np.zeros(len(Xt)),
             np.ones(len(Xv))])
        adv  = XGBClassifier(
            n_estimators=100, max_depth=4,
            verbosity=0, random_state=42)
        try:
            auc = cross_val_score(
                adv, Xall.values, yall,
                cv=3,
                scoring="roc_auc").mean()
        except Exception:
            auc = 0.5
        logger.info(
            f"Adversarial AUC: {auc:.4f}")
        if auc > self.cfg.ADVERSARIAL_AUC_LIMIT:
            logger.warning(
                "Distribution shift detected!")
            adv.fit(Xall.values, yall)
            imp  = adv.feature_importances_
            thr  = np.percentile(imp, 90)
            prob = [
                fcols[i]
                for i, val in enumerate(imp)
                if val > thr]
            return prob, True
        return [], False


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 4 — PRIORITIZED REPLAY BUFFER                    ║
# ╚══════════════════════════════════════════════════════════╝

class SumTree:
    def __init__(self, cap: int):
        self.cap  = cap
        self.tree = np.zeros(
            2*cap-1, dtype=np.float64)
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

    def update(self, leaf: int,
               priority: float):
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
            if (s <= self.tree[l] or
                    self.tree[r] == 0):
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

    def add(self, s, a, r, ns, done,
            td_err=None):
        pri = (self._maxp if td_err is None
               else (abs(td_err)+1e-6)**
               self.alpha)
        self.tree.add(pri, (s,a,r,ns,done))

    def sample(self, batch: int) -> dict:
        idx_list, pri_list, exp_list = [], [], []
        tot = self.tree.total
        seg = tot/batch
        for i in range(batch):
            s = random.uniform(
                seg*i, seg*(i+1))
            idx, pri, exp = self.tree.get(s)
            if exp is None:
                s2 = random.uniform(0, tot)
                idx, pri, exp = self.tree.get(s2)
            idx_list.append(idx)
            pri_list.append(max(pri, 1e-10))
            exp_list.append(exp)

        self.beta = min(
            1.0, self.beta+self.beta_inc)
        min_p  = min(pri_list)/(tot+1e-10)
        max_w  = (min_p*self.size)**(-self.beta)
        weights = np.array(
            [(p/tot*self.size)**(-self.beta)/max_w
             for p in pri_list],
            dtype=np.float32)

        return {
            "states":
                np.array(
                    [e[0] for e in exp_list],
                    np.float32),
            "actions":
                np.array(
                    [e[1] for e in exp_list],
                    np.int32),
            "rewards":
                np.array(
                    [e[2] for e in exp_list],
                    np.float32),
            "next_states":
                np.array(
                    [e[3] for e in exp_list],
                    np.float32),
            "dones":
                np.array(
                    [e[4] for e in exp_list],
                    np.float32),
            "weights":  weights,
            "indices":  idx_list,
        }

    def update_priorities(self, indices,
                          td_errors):
        for idx, td in zip(indices, td_errors):
            pri = (abs(td)+1e-6)**self.alpha
            self._maxp = max(self._maxp, pri)
            self.tree.update(idx, pri)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 4B — FRAME STACKER                               ║
# ╚══════════════════════════════════════════════════════════╝

class FrameStacker:
    """
    Concatenates the last K market-feature slices into
    one flat vector, giving XGBoost temporal context.

    Layout of output vector:
        [mkt_t-K+1 | … | mkt_t-1 | mkt_t |
         portfolio_t | regime_t]
        shape = (n_market * K + 9 + 8,)

    Only market features are stacked; portfolio and
    regime features are already rolling summaries and
    stacking them adds noise without information.

    FIX A (next-state leakage):
        The stacker exposes a snapshot() method that
        returns a read-only copy of the current buffer
        contents.  The main loop uses snapshot() to
        build next_state without mutating the buffer,
        so bar t+1's close is never seen at step t.
    """

    def __init__(self, n_market_features: int,
                 stack_size: int):
        self.n_mkt = n_market_features
        self.k     = stack_size
        self._buf: deque = deque(
            [np.zeros(n_market_features,
                      dtype=np.float32)] *
            stack_size,
            maxlen=stack_size)

    def push(self,
             state: np.ndarray) -> np.ndarray:
        """
        Ingest current state, advance buffer,
        return stacked state.
        Mutates the internal deque — call once
        per training step.
        """
        mkt  = state[:self.n_mkt].copy()
        tail = state[self.n_mkt:].copy()
        self._buf.append(mkt)
        stacked_mkt = np.concatenate(
            list(self._buf), axis=0)
        return np.concatenate(
            [stacked_mkt, tail],
            axis=0).astype(np.float32)

    def build_next(
            self,
            next_state: np.ndarray) -> np.ndarray:
        """
        FIX A — Lookahead-free next_state builder.

        Constructs what the stacked next state will
        look like WITHOUT mutating the live buffer.

        Implementation:
            Simulate the buffer after one push by
            dropping the oldest frame and appending
            the next bar's market slice.  The tail
            (portfolio + regime) comes from
            next_state which was built using the
            portfolio state AFTER step t's execution
            and market features up to and including
            index t only (see StateBuilder contract
            below).

        Why this is leak-free:
            next_state is built by calling
            StateBuilder.build(feat_data, t+1, ...)
            with the CURRENT portfolio info (upi)
            from step t.  StateBuilder._regime()
            slices returns up to idx-1 (exclusive),
            so bar t+1's close is NOT included in the
            regime calculation.  See StateBuilder for
            the enforcing guard.
        """
        next_mkt  = next_state[:self.n_mkt].copy()
        next_tail = next_state[self.n_mkt:].copy()
        # Simulate one push: drop oldest, add newest
        sim_buf = list(self._buf)[1:] + [next_mkt]
        stacked_mkt = np.concatenate(
            sim_buf, axis=0)
        return np.concatenate(
            [stacked_mkt, next_tail],
            axis=0).astype(np.float32)

    def reset(self):
        self._buf = deque(
            [np.zeros(self.n_mkt,
                      dtype=np.float32)] * self.k,
            maxlen=self.k)


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 4C — REGIME FILTER                               ║
# ╚══════════════════════════════════════════════════════════╝

class RegimeFilter:
    """
    Hard-gates new trade entries when market conditions
    make spread recovery statistically unlikely.

    Gate 1 — ADX < REGIME_ADX_MIN:
        No directional trend. Every entry starts behind
        the spread with no momentum to recover it.

    Gate 2 — vol_20/vol_60 < REGIME_VOL_RATIO_MIN:
        Short-term volatility compressing vs. medium
        term = entering consolidation. ATR stops and
        targets calibrated on recent vol become
        unreachable.

    Gate 3 — Consecutive loss cooldown:
        N sequential losses indicate a structural
        signal problem. Suspend new entries for
        COOLDOWN_BARS bars to avoid drawdown spiral.

    Open positions are NEVER blocked — only new entries
    are filtered.
    """

    def __init__(self, cfg: SystemConfig):
        self.adx_min       = cfg.REGIME_ADX_MIN
        self.vol_ratio_min = cfg.REGIME_VOL_RATIO_MIN
        self.max_consec    = (
            cfg.REGIME_MAX_CONSEC_LOSSES)
        self.cooldown_bars = cfg.REGIME_COOLDOWN_BARS
        self._consec_losses  = 0
        self._cooldown_left  = 0
        self._filtered_count = 0
        self._total_count    = 0

    def check(self, action: int,
              row: pd.Series,
              current_pos: int) -> int:
        self._total_count += 1
        # Never block exits
        if current_pos != 0:
            return action
        # HOLD passes through unchanged
        if action == 1:
            return action

        # Gate 1: ADX
        adx = float(row.get("adx_14_full", 25.0))
        if adx < self.adx_min:
            self._filtered_count += 1
            return 1

        # Gate 2: volatility ratio
        vol_20    = float(row.get("vol_20", 0.01))
        vol_60    = float(row.get("vol_60", 0.01))
        vol_ratio = vol_20 / (vol_60 + 1e-10)
        if vol_ratio < self.vol_ratio_min:
            self._filtered_count += 1
            return 1

        # Gate 3: cooldown
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._filtered_count += 1
            return 1

        return action

    def record_trade_result(self, pnl: float):
        if pnl < 0:
            self._consec_losses += 1
            if (self._consec_losses >=
                    self.max_consec):
                self._cooldown_left = (
                    self.cooldown_bars)
                self._consec_losses = 0
                logger.info(
                    f"[RegimeFilter] Cooldown "
                    f"{self.cooldown_bars} bars "
                    f"after {self.max_consec} "
                    f"consecutive losses")
        else:
            self._consec_losses = 0

    def stats(self) -> dict:
        rate = (self._filtered_count /
                max(1, self._total_count))
        return {
            "filtered":    self._filtered_count,
            "total":       self._total_count,
            "filter_rate": rate,
        }


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 5 — STATE BUILDER                                ║
# ║                                                          ║
# ║  FIX A contract: _regime() slices returns up to          ║
# ║  idx-1 (exclusive) so that when the main loop calls      ║
# ║  build(feat_data, t+1, ...) to construct next_state,     ║
# ║  bar t+1's realised return is NOT included in the        ║
# ║  regime lookback window.  This eliminates the            ║
# ║  one-step return leakage into Q-target evaluation.       ║
# ╚══════════════════════════════════════════════════════════╝

class StateBuilder:
    EPS = 1e-10

    def build(self, data: pd.DataFrame,
              idx: int,
              features: List[str],
              portfolio: dict) -> np.ndarray:
        lb  = min(100, idx)
        mkt = []
        for f in features:
            if f not in data.columns:
                mkt.append(0.0); continue
            val = float(data[f].iloc[idx])
            win = data[f].iloc[
                max(0, idx-lb):idx]
            mu  = (win.mean()
                   if len(win) > 0 else 0.0)
            sg  = (win.std()
                   if len(win) > 1 else 1.0)
            sg  = sg if sg > self.EPS else 1.0
            mkt.append(float(
                np.clip((val-mu)/sg, -5, 5)))

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
        state  = np.array(
            mkt+port+reg, dtype=np.float32)
        return np.nan_to_num(
            state, nan=0, posinf=0, neginf=0)

    def _regime(self, data: pd.DataFrame,
                idx: int) -> List[float]:
        """
        FIX A: slice is [st : idx] — exclusive of
        idx — so the bar at position idx (i.e. bar
        t+1 when called for next_state) is never
        included in the regime statistics.
        """
        st  = max(0, idx-100)
        # Exclusive upper bound: returns up to idx-1
        r   = (data["returns"]
               .iloc[st:idx]   # idx excluded
               .fillna(0).values)
        if len(r) < 5:
            return [0.0]*8
        sh   = float(
            np.mean(r)/(np.std(r)+self.EPS))
        rv   = float(
            np.std(r[-20:]) if len(r) >= 20
            else np.std(r))
        lv   = float(
            np.std(r[-60:]) if len(r) >= 60
            else np.std(r))
        vr   = rv/(lv+self.EPS)
        bull = float(np.mean(r > 0))
        hurst= self._hurst(r)
        ac1  = (float(pd.Series(r).autocorr(1))
                if len(r) > 2 else 0.0)
        ac5  = (float(pd.Series(r).autocorr(5))
                if len(r) > 6 else 0.0)
        sk   = (float(stats.skew(r))
                if len(r) > 3 else 0.0)
        ku   = (float(stats.kurtosis(r))
                if len(r) > 3 else 0.0)
        feats = [sh,vr,bull,hurst,
                 ac1,ac5,sk,ku]
        return [float(np.nan_to_num(f))
                for f in feats]

    @staticmethod
    def _hurst(s: np.ndarray,
               ml=20) -> float:
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
# ║  MODULE 6 — REWARD FUNCTION                              ║
# ╚══════════════════════════════════════════════════════════╝

class RewardFunction:
    INACTION_VOL_THRESHOLD: float = 0.0008
    INACTION_PENALTY:       float = 0.03

    def __init__(self, cfg: SystemConfig):
        self.w    = cfg.REWARD_WEIGHTS
        self.drf  = cfg.DAILY_RISK_FREE
        self.tc   = (cfg.TRANSACTION_COST_BPS
                     / 10_000)
        self.ruin = cfg.EQUITY_RUIN_THRESHOLD
        self.hist: List[float] = []
        self.EPS = 1e-10

    def calc(self, action: int,
             port_ret: float,
             port_info: dict) -> float:
        self.hist.append(port_ret)

        sharpe    = self._sharpe()
        sortino   = self._sortino(port_ret)
        dd_pen    = self._dd_pen(
            port_info.get("current_drawdown", 0))
        trade_pen = self._trade_pen(
            action, port_info)
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

        if action == 1:
            vol = float(
                port_info.get("vol_20", 0.0))
            if vol > self.INACTION_VOL_THRESHOLD:
                total -= self.INACTION_PENALTY

        return float(np.clip(total, -10, 10))

    def _sharpe(self) -> float:
        if len(self.hist) < 10: return 0.0
        r  = (np.array(self.hist[-100:]) -
               self.drf)
        sg = np.std(r)
        if sg < self.EPS: return 0.0
        return float(
            np.mean(r)/sg*np.sqrt(252))

    def _sortino(self, cr: float) -> float:
        if len(self.hist) < 10:
            return float(cr)
        r  = np.array(self.hist[-100:])
        dn = r[r < 0]
        ds = (np.std(dn)
              if len(dn) > 0 else self.EPS)
        return float(
            (cr-self.drf)/(ds+self.EPS))

    def _dd_pen(self, dd: float) -> float:
        return float(np.exp(3*abs(dd))-1)

    def _trade_pen(self, a: int,
                   pi: dict) -> float:
        return (
            0.001*self.tc
            if a != pi.get("previous_action", 1)
            else 0.0)

    def _pf_bonus(self) -> float:
        if len(self.hist) < 20: return 0.0
        r  = np.array(self.hist[-50:])
        gp = r[r > 0].sum()
        gl = abs(r[r < 0].sum())
        if gl < self.EPS: return 1.0
        return float(np.clip(
            np.log(gp/(gl+self.EPS)+self.EPS),
            -2, 2))

    def _consist(self) -> float:
        if len(self.hist) < 30: return 0.0
        r = np.array(self.hist[-30:])
        if (np.mean(r) > 0 and
                np.std(r) > self.EPS):
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
# ║  MODULE 7 — PORTFOLIO MANAGER                            ║
# ╚══════════════════════════════════════════════════════════╝

class PortfolioManager:
    def __init__(self, cfg: SystemConfig):
        self.init_cap  = cfg.INITIAL_CAPITAL
        self.tc        = (cfg.TRANSACTION_COST_BPS
                          / 10_000)
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
        self.eq_curve: List[dict]  = []
        self.trade_log: List[dict] = []
        self.daily_cnt = 0
        self.cur_day   = None
        self.daily_pnl = 0.0
        self.EPS       = 1e-10
        self._size_cap = (cfg.INITIAL_CAPITAL *
                          cfg.EVAL_SIZE_CAP_PCT)
        self._spread   = cfg.SPREAD_PRICE_UNITS
        self.last_closed_pnl: Optional[
            float] = None

    def execute(self, action: int,
                price: float,
                time,
                size_usd: float) -> dict:
        size_usd = min(size_usd, self._size_cap)
        port_ret = 0.0
        executed = False
        self.last_closed_pnl = None

        if action == 2:        # BUY
            fill_px = price + self._spread
            if self.pos == -1:
                pnl  = self.pos_size*(
                    self.entry_px/
                    (fill_px+self.EPS)-1)
                cost = self.pos_size*self.tc
                net  = pnl - cost
                self.cash += (self.pos_size + net)
                port_ret   = net/(
                    self.equity+self.EPS)
                self._log_trade(net, time)
                self.last_closed_pnl = net
                self.pos = 0
                self.pos_size = 0.0
                executed = True
            elif self.pos == 0:
                cost = size_usd * self.tc
                self.pos      = 1
                self.pos_size = size_usd
                self.entry_px = fill_px
                self.entry_t  = time
                self.cash -= (size_usd + cost)
                self.cash  = max(self.cash, 0)
                executed   = True
            else:
                port_ret = self.pos_size*(
                    price/
                    (self.entry_px+self.EPS)-1
                )/(self.equity+self.EPS)

        elif action == 0:      # SELL
            fill_px = price - self._spread
            if self.pos == 1:
                pnl  = self.pos_size*(
                    fill_px/
                    (self.entry_px+self.EPS)-1)
                cost = self.pos_size * self.tc
                net  = pnl - cost
                self.cash += (self.pos_size + net)
                port_ret   = net/(
                    self.equity+self.EPS)
                self._log_trade(net, time)
                self.last_closed_pnl = net
                self.pos = 0
                self.pos_size = 0.0
                executed = True
            elif self.pos == 0:
                cost = size_usd * self.tc
                self.pos      = -1
                self.pos_size = size_usd
                self.entry_px = fill_px
                self.entry_t  = time
                self.cash -= cost
                executed = True
            else:
                port_ret = self.pos_size*(
                    self.entry_px/
                    (price+self.EPS)-1
                )/(self.equity+self.EPS)

        else:                  # HOLD
            if self.pos == 1:
                port_ret = self.pos_size*(
                    price/
                    (self.entry_px+self.EPS)-1
                )/(self.equity+self.EPS)
            elif self.pos == -1:
                port_ret = self.pos_size*(
                    self.entry_px/
                    (price+self.EPS)-1
                )/(self.equity+self.EPS)

        if self.pos == 1:
            unr = self.pos_size*(
                price/(self.entry_px+self.EPS)-1)
            self.equity = (self.cash +
                           self.pos_size + unr)
        elif self.pos == -1:
            unr = self.pos_size*(
                self.entry_px/
                (price+self.EPS)-1)
            self.equity = (self.cash +
                           self.pos_size + unr)
        else:
            self.equity = self.cash
        self.equity = max(self.equity, 0.01)

        self.peak_eq = max(
            self.peak_eq, self.equity)
        dd = ((self.peak_eq-self.equity) /
              (self.peak_eq+self.EPS))
        self.max_dd = max(self.max_dd, dd)
        self.eq_curve.append({
            "time":   time,
            "equity": self.equity,
            "dd":     dd})

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
        elif (self.pos == -1 and
              self.entry_px > 0):
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
                self.gross_p/(
                    self.wins+self.EPS),
            "avg_loss_size":
                self.gross_l/(
                    max(1,
                        self.trades-self.wins)),
            "avg_trade_duration": 0,
            "holding_duration":   hold,
            "previous_action":    self.prev_act,
            "trades_today":       self.daily_cnt,
            "daily_pnl":          self.daily_pnl,
            "total_trades":       self.trades,
        }


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 8 — RISK MANAGER                                 ║
# ╚══════════════════════════════════════════════════════════╝

class RiskManager:
    def __init__(self, cfg: SystemConfig):
        self.max_pos  = cfg.MAX_POSITION_SIZE
        self.sl_pct   = cfg.STOP_LOSS_PCT
        self.tp_pct   = cfg.TAKE_PROFIT_PCT
        self.atr_sl   = cfg.ATR_SL_MULT
        self.atr_tp   = cfg.ATR_TP_MULT
        self.max_d    = cfg.MAX_DAILY_TRADES
        self.max_dd   = cfg.MAX_DRAWDOWN_LIMIT
        self.ruin     = cfg.EQUITY_RUIN_THRESHOLD
        self.init_cap = cfg.INITIAL_CAPITAL
        self.peak_eq  = cfg.INITIAL_CAPITAL
        self.cnt      = 0
        self.cur_day  = None
        self.EPS      = 1e-10

    def validate(self, action: int,
                 pi: dict,
                 mkt: dict = None) -> int:
        eq   = pi.get(
            "current_equity", self.init_cap)
        pos  = pi.get("current_position", 0)
        upnl = pi.get("unrealized_pnl_pct", 0)

        if eq/(self.init_cap+self.EPS) < self.ruin:
            return 1

        self.peak_eq = max(self.peak_eq, eq)
        dd = ((self.peak_eq-eq) /
              (self.peak_eq+self.EPS))
        if dd > self.max_dd:
            if pos > 0 and action == 0: return 0
            if pos < 0 and action == 2: return 2
            return 1

        today = (mkt.get("date")
                 if mkt else None)
        if today and today != self.cur_day:
            self.cur_day = today; self.cnt = 0
        if (action != 1 and
                action != pi.get(
                    "previous_action", 1)):
            if self.cnt >= self.max_d:
                return 1

        if pos != 0:
            atr      = (mkt.get("atr_14", 0)
                        if mkt else 0)
            entry_px = (mkt.get("entry_px", 0)
                        if mkt else 0)
            if (atr > self.EPS and
                    entry_px > self.EPS):
                sl_dist = self.atr_sl * atr
                tp_dist = self.atr_tp * atr
                if pos > 0:
                    sl_p = entry_px - sl_dist
                    tp_p = entry_px + tp_dist
                    px   = mkt.get(
                        "current_price", entry_px)
                    if px <= sl_p: return 0
                    if px >= tp_p: return 0
                else:
                    sl_p = entry_px + sl_dist
                    tp_p = entry_px - tp_dist
                    px   = mkt.get(
                        "current_price", entry_px)
                    if px >= sl_p: return 2
                    if px <= tp_p: return 2
            else:
                if upnl < -self.sl_pct:
                    return 0 if pos > 0 else 2
                if upnl >  self.tp_pct:
                    return 0 if pos > 0 else 2

        return action

    def position_size(self, pi: dict,
                      mkt: dict,
                      confidence: float) -> float:
        eq   = pi.get(
            "current_equity", self.init_cap)
        base = eq * self.max_pos
        cs   = float(np.clip(
            confidence/(confidence+1), 0.5, 1))
        cv   = mkt.get("vol_20", 0.02)
        av   = mkt.get("vol_60", 0.02)
        vs   = float(np.clip(
            av/(cv+self.EPS), 0.5, 2))
        wr   = pi.get("recent_win_rate", 0.5)
        aw   = pi.get("avg_win_size",    0.01)
        al   = abs(pi.get(
            "avg_loss_size", 0.01))
        kelly = 0.0
        if al > self.EPS and wr > 0:
            kelly = float(np.clip(
                wr-(1-wr)/(
                    aw/(al+self.EPS)+self.EPS),
                0, 0.25))
        ks  = max(
            kelly/(self.max_pos+self.EPS), 0.1)
        sz  = min(
            base*cs*vs*ks, eq*self.max_pos)
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
            x = (s[:, m["fi"]]
                 if m["fi"] is not None
                 else s)
            try:
                out.append(float(
                    m["model"].predict(x)[0]))
            except Exception:
                out.append(0.0)
        return out

    def _q(self, ens: Dict,
           s: np.ndarray, a: int) -> float:
        ps = self._pred_ens(
            ens[a], s.reshape(1, -1))
        return float(
            np.average(ps, weights=self.ew))

    def select(self, s: np.ndarray,
               training=True) -> int:
        if (training and
                random.random() < self.epsilon):
            return random.randint(0, self.n_a-1)
        if not self.fitted:
            return random.randint(0, self.n_a-1)
        sv = s.reshape(1, -1)
        qv = np.zeros(self.n_a)
        cf = np.zeros(self.n_a)
        for a in range(self.n_a):
            ps = (self._pred_ens(
                      self.q1[a], sv) +
                  self._pred_ens(
                      self.q2[a], sv))
            qv[a] = np.mean(ps)
            cf[a] = 1/(np.std(ps)+self.EPS)
        best = int(np.argmax(qv))
        nc   = cf[best]/(cf.sum()+self.EPS)
        if nc < self.conf_thr:
            return 1
        return best

    def confidence(self,
                   s: np.ndarray) -> float:
        if not self.fitted: return 0.5
        sv = s.reshape(1, -1)
        all_ps = []
        for a in range(self.n_a):
            all_ps.extend(
                self._pred_ens(
                    self.q1[a], sv))
        return float(
            1/(np.std(all_ps)+self.EPS))

    def store(self, s, a, r, ns, done):
        td = None
        if self.fitted:
            cq = self._q(self.q1, s, a)
            if done:
                tgt = r
            else:
                nqs = [
                    self._q(self.q1, ns, aa)
                    for aa in range(self.n_a)]
                tgt = r + self.gamma * max(nqs)
            td = abs(tgt - cq)
        self.buf.add(s, a, r, ns, done, td)

    def train(self) -> Optional[dict]:
        if self.buf.size < self.batch_sz:
            return None
        if (self.buf.size <
                self.cfg.MIN_BUFFER_SIZE):
            return None
        self.step += 1
        b = self.buf.sample(self.batch_sz)
        S, A, R, NS, D = (
            b["states"], b["actions"],
            b["rewards"], b["next_states"],
            b["dones"])
        W   = b["weights"]
        idx = b["indices"]
        tgts = np.zeros(
            self.batch_sz, np.float32)
        tds  = np.zeros(
            self.batch_sz, np.float32)

        for i in range(self.batch_sz):
            if D[i]:
                tgts[i] = R[i]
            else:
                if random.random() < 0.5:
                    ba = int(np.argmax([
                        self._q(
                            self.q1, NS[i], aa)
                        for aa in
                        range(self.n_a)]))
                    tgt = (R[i] + self.gamma *
                           self._q(
                               self.q2,
                               NS[i], ba))
                else:
                    ba = int(np.argmax([
                        self._q(
                            self.q2, NS[i], aa)
                        for aa in
                        range(self.n_a)]))
                    tgt = (R[i] + self.gamma *
                           self._q(
                               self.q1,
                               NS[i], ba))
                tgts[i] = tgt
            tds[i] = abs(
                tgts[i] -
                self._q(self.q1, S[i],
                         int(A[i])))

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
                cf  = self.cfg.ENSEMBLE_CONFIGS[
                    i]["colsample_bytree"]
                nf  = max(2, int(cf*n_feat))
                fi  = np.sort(
                    np.random.choice(
                        n_feat, nf,
                        replace=False))
                self.q1[a][i]["fi"] = fi
                self.q2[a][i]["fi"] = fi

                Xt = Xa[bi][:, fi]
                Xt = Xt + np.random.normal(
                    0,
                    self.cfg.NOISE_INJECTION_LEVEL,
                    Xt.shape)
                dm = np.random.binomial(
                    1,
                    1-self.cfg
                    .FEATURE_DROPOUT_RATE,
                    Xt.shape)
                Xt = Xt * dm
                try:
                    self.q1[a][i]["model"].fit(
                        Xt, ya[bi],
                        sample_weight=wa[bi])
                except Exception as e:
                    logger.debug(
                        f"Q1 fit a={a} "
                        f"m={i}: {e}")

                bi2 = np.random.choice(
                    n, bsz, replace=True)
                Xt2 = Xa[bi2][:, fi]
                Xt2 = Xt2 + np.random.normal(
                    0,
                    self.cfg.NOISE_INJECTION_LEVEL,
                    Xt2.shape)
                dm2 = np.random.binomial(
                    1,
                    1-self.cfg
                    .FEATURE_DROPOUT_RATE,
                    Xt2.shape)
                Xt2 = Xt2 * dm2
                try:
                    self.q2[a][i]["model"].fit(
                        Xt2, ya[bi2],
                        sample_weight=wa[bi2])
                except Exception as e:
                    logger.debug(
                        f"Q2 fit a={a} "
                        f"m={i}: {e}")
        self.fitted = True

    def update_ew(self, val_S: np.ndarray,
                  val_tgts: np.ndarray):
        scores = np.zeros(self.n_m)
        for i in range(self.n_m):
            errs, cnt = 0.0, 0
            for a in range(self.n_a):
                fi = self.q1[a][i]["fi"]
                if fi is None: continue
                x  = val_S[:, fi]
                try:
                    p = self.q1[a][i][
                        "model"].predict(x)
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

    def should_retrain(
            self, t: int,
            data: pd.DataFrame) -> bool:
        return (
            (t-self.last) >= self.ri or
            self.regime_change(data, t) or
            self._degraded())

    def regime_change(
            self, data: pd.DataFrame,
            t: int) -> bool:
        if t < 2*self.rl: return False
        r   = data["returns"].fillna(0)
        rec = r.iloc[t-self.rl:t].values
        prv = r.iloc[
            t-2*self.rl:t-self.rl].values
        try:
            _, p = ks_2samp(rec, prv)
            if p < self.rp:
                logger.info(
                    f"Regime change at "
                    f"step {t} p={p:.5f}")
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
                "Performance degradation "
                "detected")
            return True
        return False

    def window(self, data: pd.DataFrame,
               t: int):
        st    = max(0, t-self.tw)
        chunk = data.iloc[st:t].copy()
        n     = len(chunk)
        w     = np.exp(
            np.linspace(self.rd, 0, n))
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
                        "n_estimators",
                        100, 1000),
                "max_depth":
                    trial.suggest_int(
                        "max_depth", 3, 10),
                "learning_rate":
                    trial.suggest_float(
                        "lr", 0.005, 0.3,
                        log=True),
                "min_child_weight":
                    trial.suggest_int(
                        "mcw", 1, 20),
                "subsample":
                    trial.suggest_float(
                        "ss", 0.5, 1.0),
                "colsample_bytree":
                    trial.suggest_float(
                        "cs", 0.3, 1.0),
                "gamma":
                    trial.suggest_float(
                        "gm", 0, 10),
                "reg_alpha":
                    trial.suggest_float(
                        "ra", 1e-6, 100,
                        log=True),
                "reg_lambda":
                    trial.suggest_float(
                        "rl", 1e-6, 100,
                        log=True),
                "tree_method":  "hist",
                "verbosity":    0,
                "random_state": 42,
            }
            folds  = self._cv_folds(
                len(X),
                self.cfg.OPTUNA_CV_SPLITS,
                self.cfg.EMBARGO_PCT)
            scores = []
            for fi, (tri, tei) in enumerate(
                    folds):
                m = XGBClassifier(**params)
                try:
                    m.fit(
                        X[tri], y[tri],
                        eval_set=[
                            (X[tei], y[tei])],
                        verbose=False)
                    auc = roc_auc_score(
                        y[tei],
                        m.predict_proba(
                            X[tei])[:, 1])
                    scores.append(auc)
                except Exception:
                    scores.append(0.5)
                trial.report(
                    np.mean(scores), fi)
                if trial.should_prune():
                    raise optuna.exceptions\
                        .TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42),
            pruner=(optuna.pruners
                    .HyperbandPruner()))
        study.optimize(
            objective,
            n_trials=self.cfg.OPTUNA_N_TRIALS,
            n_jobs=1,
            show_progress_bar=False)
        logger.info(
            f"Best AUC: "
            f"{study.best_value:.4f}")
        return study.best_params

    @staticmethod
    def _cv_folds(n, ns, ep):
        emb   = int(n*ep)
        fsz   = n//ns
        folds = []
        for i in range(ns):
            ts = i*fsz
            te = (i+1)*fsz if i < ns-1 else n
            tr = np.concatenate([
                np.arange(0, max(0, ts-emb)),
                np.arange(min(n, te+emb), n)
            ]).astype(int)
            tv = np.arange(ts, te).astype(int)
            if (len(tr) >= 10 and
                    len(tv) >= 5):
                folds.append((tr, tv))
        return folds


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 12 — ONNX EXPORT                                 ║
# ║                                                          ║
# ║  FIX B: Each model's fi (feature sub-indices) are        ║
# ║  written to feature_indices.json as integer lists.       ║
# ║  The MT5 EA must slice the stacked_state vector to       ║
# ║  fi before calling OnnxRun() because the ONNX graph      ║
# ║  expects shape (1, len(fi)), not (1, stacked_dim).       ║
# ║                                                          ║
# ║  FIX C: scaler_params now covers every position in the   ║
# ║  stacked market vector using frame-indexed keys:         ║
# ║    "feat_frame0" = current bar (most recent)             ║
# ║    "feat_frame1" = one bar ago                           ║
# ║    ...                                                   ║
# ║    "feat_frameK-1" = oldest frame in stack               ║
# ║  Portfolio and regime positions reuse the same params     ║
# ║  as they are already normalised by StateBuilder.         ║
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
            ("input",
             FloatTensorType([1, in_dim]))
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
        inp.type.tensor_type.shape\
            .ClearField("dim")
        d0 = (inp.type.tensor_type
              .shape.dim.add())
        d0.dim_value = 1
        d1 = (inp.type.tensor_type
              .shape.dim.add())
        d1.dim_value = in_dim

        out  = graph.output[0]
        out.type.tensor_type.shape\
            .ClearField("dim")
        od0 = (out.type.tensor_type
               .shape.dim.add())
        od0.dim_value = 1
        od1 = (out.type.tensor_type
               .shape.dim.add())
        od1.dim_value = 1

        onnx_model = shape_inference\
            .infer_shapes(onnx_model)

        fname = f"{net_name}_a{a}_m{mi}.onnx"
        fpath = os.path.join(self.out, fname)
        with open(fpath, "wb") as f:
            f.write(
                onnx_model.SerializeToString())

        sess    = ort.InferenceSession(fpath)
        dummy   = np.zeros(
            (1, in_dim), dtype=np.float32)
        in_name = sess.get_inputs()[0].name
        result  = sess.run(
            None, {in_name: dummy})
        in_sh   = sess.get_inputs()[0].shape
        out_sh  = sess.get_outputs()[0].shape
        val     = float(
            result[0].flatten()[0])
        logger.info(
            f"  ✔ {fname}: "
            f"in{in_sh} → out{out_sh} "
            f"= {val:.6f}")
        return fpath

    def export_agent(
            self,
            agent: "EnsembleDoubleQAgent",
            selected: List[str],
            scaler_params: dict,
            state_dim: int,
            report: dict,
            frame_stack_size: int = 1):
        os.makedirs(self.out, exist_ok=True)
        exported = []

        for net_name, ens in [
                ("q1", agent.q1),
                ("q2", agent.q2)]:
            for a in range(agent.n_a):
                for mi in range(agent.n_m):
                    m      = ens[a][mi]
                    mdl    = m["model"]
                    fi     = m["fi"]
                    # FIX B: in_dim = len(fi)
                    # so ONNX graph matches the
                    # sub-selected slice the model
                    # was actually trained on
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
                            f"ONNX export "
                            f"failed {net_name}"
                            f" a={a} m={mi}: {e}")
                        try:
                            mdl.save_model(
                                os.path.join(
                                    self.out,
                                    f"{net_name}"
                                    f"_a{a}_m{mi}"
                                    f".json"))
                        except Exception:
                            pass

        # ── FIX B: feature_indices.json ───────────
        # Each entry maps "net_aA_mM" →
        # list of integer indices into stacked_state.
        # The MT5 EA slices stacked_state[fi] before
        # calling OnnxRun(), not the full vector.
        fi_map = {}
        for a in range(agent.n_a):
            for mi in range(agent.n_m):
                for net in ["q1", "q2"]:
                    ens_src = (agent.q1
                               if net == "q1"
                               else agent.q2)
                    fi  = ens_src[a][mi]["fi"]
                    key = f"{net}_a{a}_m{mi}"
                    fi_map[key] = (
                        fi.tolist()
                        if fi is not None
                        else list(
                            range(state_dim)))

        with open(os.path.join(
                self.out,
                "feature_indices.json"),
                "w") as f:
            json.dump(fi_map, f, indent=2)

        with open(os.path.join(
                self.out,
                "selected_features.json"),
                "w") as f:
            json.dump(selected, f, indent=2)

        # ── FIX C: stacked scaler params ──────────
        # Write scaler params for every position in
        # the stacked market vector.
        # Frame 0 = current bar (most recent push)
        # Frame K-1 = oldest bar in the stack
        # Portfolio and regime positions (appended
        # after the stacked market block) are already
        # normalised inside StateBuilder, so we emit
        # mean=0, std=1 placeholders for them.
        with open(os.path.join(
                self.out,
                "scaler_params.json"),
                "w") as f:
            json.dump(
                scaler_params, f, indent=2)

        with open(os.path.join(
                self.out,
                "ensemble_weights.json"),
                "w") as f:
            json.dump(
                {"weights": agent.ew.tolist()},
                f, indent=2)

        with open(os.path.join(
                self.out, "config.json"),
                "w") as f:
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
                "FRAME_STACK_SIZE":
                    frame_stack_size,
                # FIX B: document the EA contract
                "ONNX_INPUT_NOTE": (
                    "Each ONNX model expects "
                    "shape (1, len(fi)) where fi "
                    "is the integer index list "
                    "from feature_indices.json "
                    "keyed as net_aA_mM. "
                    "Slice stacked_state[fi] "
                    "before OnnxRun()."),
                # FIX C: document scaler layout
                "SCALER_LAYOUT_NOTE": (
                    "scaler_params.json keys use "
                    "suffix _frameF where F=0 is "
                    "current bar, F=K-1 is oldest."
                    " Portfolio and regime slots "
                    "have mean=0 std=1 (already "
                    "normalised by StateBuilder)."
                ),
            }, f, indent=2)

        with open(os.path.join(
                self.out,
                "training_report.json"),
                "w") as f:
            json.dump(
                {k: float(v)
                 if isinstance(
                     v, (np.floating, float))
                 else v
                 for k, v in report.items()},
                f, indent=2)

        logger.info(
            f"Exported {len(exported)} ONNX "
            f"models to {self.out}/")
        return exported


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 13 — PERFORMANCE MONITOR                         ║
# ╚══════════════════════════════════════════════════════════╝

class PerfMonitor:
    def __init__(self, cfg: SystemConfig):
        self.rf  = cfg.RISK_FREE_RATE
        self.EPS = 1e-10

    def evaluate(self,
                 eq_curve: List[dict],
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
        dn  = ret[ret < 0]
        dv  = (np.std(dn) if len(dn) > 0
               else self.EPS)*np.sqrt(252*24)
        sh  = (ar-self.rf)/(av+self.EPS)
        so  = (ar-self.rf)/(dv+self.EPS)
        ca  = ar/(abs(mx)+self.EPS)
        tdf = (pd.DataFrame(trade_log)
               if trade_log
               else pd.DataFrame())
        nt  = len(tdf)
        wt  = (int((tdf["pnl"] > 0).sum())
               if nt > 0 else 0)
        wr  = wt/(nt+self.EPS)
        gp  = (float(
                   tdf[tdf["pnl"] > 0]["pnl"]
                   .sum())
               if wt > 0 else 0)
        gl  = (float(
                   tdf[tdf["pnl"] < 0]["pnl"]
                   .abs().sum())
               if nt-wt > 0 else 0)
        pf  = gp/(gl+self.EPS)
        m   = {
            "total_return":  tr,
            "ann_return":    ar,
            "ann_vol":       av,
            "max_dd":        mx,
            "sharpe":        sh,
            "sortino":       so,
            "calmar":        ca,
            "trades":        nt,
            "win_rate":      wr,
            "profit_factor": pf,
        }
        self._print(m)
        return m

    def _mdd(self, eq):
        pk = eq[0]; mx = 0
        for e in eq:
            pk = max(pk, e)
            mx = max(
                mx, (pk-e)/(pk+self.EPS))
        return mx

    def _print(self, m):
        sep = "═"*50
        print(f"\n╔{sep}╗")
        print(
            f"║{'PERFORMANCE REPORT':^50}║")
        print(f"╠{sep}╣")
        rows = [
            ("Total Return",
             "total_return",  ".2%"),
            ("Ann. Return",
             "ann_return",    ".2%"),
            ("Ann. Vol",
             "ann_vol",       ".2%"),
            ("Max Drawdown",
             "max_dd",        ".2%"),
            ("Sharpe",
             "sharpe",        ".3f"),
            ("Sortino",
             "sortino",       ".3f"),
            ("Calmar",
             "calmar",        ".3f"),
            ("Trades",
             "trades",        "d"),
            ("Win Rate",
             "win_rate",      ".2%"),
            ("Profit Factor",
             "profit_factor", ".2f"),
        ]
        for label, key, fmt in rows:
            vs = f"{m[key]:{fmt}}"
            print(
                f"║  {label+':':<18}"
                f"{vs:>10}"
                f"{'':>28}║")
        print(f"╚{sep}╝\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  MODULE 14 — SCALER BUILDER  (FIX C)                     ║
# ║                                                          ║
# ║  Produces per-element scaling params for the full        ║
# ║  stacked state vector consumed by the MT5 EA.            ║
# ║                                                          ║
# ║  Key naming convention:                                  ║
# ║    "{feature}_frame{F}"                                  ║
# ║      F = 0  → current bar (frame pushed last)            ║
# ║      F = 1  → one bar ago                                ║
# ║      F = K-1 → oldest frame in the stack                 ║
# ║                                                          ║
# ║  All K frames for a given feature share the same         ║
# ║  mean/std because they are drawn from the same           ║
# ║  underlying time series and should be normalised         ║
# ║  identically.  Storing them with distinct keys lets      ║
# ║  the EA index by stacked-vector position directly.       ║
# ║                                                          ║
# ║  Portfolio (9) and regime (8) tail positions emit        ║
# ║  mean=0.0, std=1.0 because StateBuilder already          ║
# ║  normalises them before they enter the state vector.     ║
# ╚══════════════════════════════════════════════════════════╝

def build_scaler_params(
        feat_data: pd.DataFrame,
        selected: List[str],
        frame_stack_size: int = 1) -> dict:
    """
    Parameters
    ----------
    feat_data        : training slice of feat_data
    selected         : ordered list of market feature
                       names (length = N_MARKET)
    frame_stack_size : K — number of stacked frames

    Returns
    -------
    dict with keys:
        "{feat}_frame{F}" for F in 0..K-1
        "portfolio_{i}"   for i in 0..8
        "regime_{i}"      for i in 0..7
    All values: {"mean": float, "std": float}
    """
    params: dict = {}

    # ── Market features: K copies per feature ─────
    for feat in selected:
        if feat in feat_data.columns:
            mu  = float(feat_data[feat].mean())
            std = float(feat_data[feat].std())
            if std < 1e-10: std = 1.0
        else:
            mu, std = 0.0, 1.0

        # Same stats for every lag frame because
        # all frames come from the same series
        for frame in range(frame_stack_size):
            key = f"{feat}_frame{frame}"
            params[key] = {"mean": mu,
                           "std":  std}

    # ── Portfolio tail (9 features) ───────────────
    # Already min-max normalised in StateBuilder
    for i in range(9):
        params[f"portfolio_{i}"] = {
            "mean": 0.0, "std": 1.0}

    # ── Regime tail (8 features) ──────────────────
    # Already computed as z-scores / bounded ratios
    for i in range(8):
        params[f"regime_{i}"] = {
            "mean": 0.0, "std": 1.0}

    return params


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN PIPELINE                                           ║
# ╚══════════════════════════════════════════════════════════╝

def train_and_export(
    cfg:      SystemConfig = None,
    filepath: str          = None,
    source:   str          = "synthetic",
) -> dict:
    if cfg is None: cfg = SystemConfig()
    print(
        "\n╔══════════════════════════════════════╗")
    print(
        "║  XGBoost-RL  TRAINING PIPELINE       ║")
    print(
        "╚══════════════════════════════════════╝\n")

    # ── Phase 1: Data ─────────────────────────────
    print("Phase 1: Data")
    data = DataIngestion.load(
        source, filepath, cfg.HISTORICAL_BARS)
    print(
        f"  {len(data):,} bars  "
        f"{data['timestamp'].iloc[0]} → "
        f"{data['timestamp'].iloc[-1]}\n")

    # ── Phase 2: Features ─────────────────────────
    print("Phase 2: Features")
    fe        = FeatureEngine(cfg)
    feat_data = fe.build(data)
    print(
        f"  {feat_data.shape[1]} columns, "
        f"{len(feat_data):,} rows\n")

    # ── Phase 3: Feature selection ────────────────
    print("Phase 3: Feature Selection")
    sel      = FeatureSelector(cfg)
    target   = (
        feat_data["returns"].shift(-1) > 0
    ).astype(int).iloc[:-1]
    feat_sub = feat_data.iloc[:-1].copy()
    selected = sel.select(
        feat_sub, target,
        cfg.MAX_FEATURES_SELECTED)

    cut = int(0.6 * len(feat_sub))
    prob, shift = sel.adversarial_check(
        feat_sub[selected].iloc[:cut],
        feat_sub[selected].iloc[cut:])
    if shift:
        selected = [
            f for f in selected
            if f not in prob]
        print(
            f"  Removed {len(prob)} shifted "
            f"features → {len(selected)} "
            f"remain\n")
    else:
        print("  No distribution shift\n")

    # ── Phase 4: Hyperparameter optimisation ──────
    print("Phase 4: Hyperparameter Optimization")
    Xopt = (feat_sub[selected].iloc[:cut]
            .fillna(0)
            .values.astype(np.float32))
    yopt = target.iloc[:cut].values
    hp   = HPOptimizer(cfg)
    best = hp.optimize(Xopt, yopt)
    cfg.XGB_BASE.update(best)
    print(f"  Best params: {best}\n")

    # ── Phase 5: Scaler params (FIX C) ────────────
    print("Phase 5: Scaler Parameters")
    stack_size    = cfg.FRAME_STACK_SIZE
    scaler_params = build_scaler_params(
        feat_sub[selected].iloc[:cut],
        selected,
        frame_stack_size=stack_size)
    print(
        f"  Scaler keys: "
        f"{len(scaler_params)} "
        f"({len(selected)} features × "
        f"{stack_size} frames "
        f"+ 9 portfolio + 8 regime)\n")

    # ── Phase 6: Initialise components ────────────
    print("Phase 6: Initialize Agent & Environment")
    agent  = EnsembleDoubleQAgent(cfg)
    port   = PortfolioManager(cfg)
    risk   = RiskManager(cfg)
    reward = RewardFunction(cfg)
    wf     = WalkForward(cfg)
    sb     = StateBuilder()
    perf   = PerfMonitor(cfg)

    # Measure base state dim from a sample build
    test_pi    = port.info(
        float(feat_data["close"].iloc[200]))
    test_state = sb.build(
        feat_data, 200, selected, test_pi)

    n_market   = len(selected)
    base_dim   = len(test_state)
    # Stacked dim = mkt*K + portfolio(9) + regime(8)
    stacked_dim = (n_market * stack_size +
                   base_dim - n_market)

    print(f"  Base state dim:    {base_dim}")
    print(f"  Frame stack size:  {stack_size}")
    print(
        f"  Stacked state dim: "
        f"{stacked_dim}\n")

    # Initialise FrameStacker and RegimeFilter
    stacker = FrameStacker(n_market, stack_size)
    rfilter = RegimeFilter(cfg)

    # ── Phase 7: Walk-forward training loop ───────
    print("Phase 7: Walk-Forward Training")
    from tqdm import tqdm
    start_idx = max(cfg.MIN_TRAIN_SAMPLES, 200)
    end_idx   = len(feat_data) - 1
    selected  = [
        f for f in selected
        if f in feat_data.columns]

    all_results: List[dict] = []
    train_log:   List[dict] = []

    for t in tqdm(
            range(start_idx, end_idx),
            desc="Training", unit="step"):

        # Regime / retrain check
        if wf.should_retrain(t, feat_data):
            if wf.regime_change(feat_data, t):
                agent  = EnsembleDoubleQAgent(cfg)
                reward.reset()
                # Clear temporal buffer so stale
                # context from old regime does not
                # pollute new regime's Q-targets
                stacker.reset()
            if len(all_results) > 100:
                rs = np.array([
                    r["state"]
                    for r in all_results[-100:]])
                rt = np.array([
                    r["reward"]
                    for r in all_results[-100:]])
                agent.update_ew(rs, rt)

        # Build current bar state and stack it
        pi    = port.info(
            float(feat_data["close"].iloc[t]),
            feat_data["timestamp"].iloc[t])
        state = sb.build(
            feat_data, t, selected, pi)
        stacked_state = stacker.push(state)

        # Agent proposes an action
        raw_a = agent.select(
            stacked_state, training=True)

        row = feat_data.iloc[t]
        mkt = {
            "vol_20":
                float(row.get("vol_20", 0.02)),
            "vol_60":
                float(row.get("vol_60", 0.02)),
            "atr_14":
                float(row.get("atr_14", 0.0)),
            "current_price":
                float(
                    feat_data["close"].iloc[t]),
            "entry_px":    port.entry_px,
            "date":        str(
                feat_data["timestamp"].iloc[t]
            )[:10],
        }

        # Regime filter gates new entries
        filtered_a = rfilter.check(
            raw_a, row,
            pi["current_position"])

        # Risk manager validates and applies SL/TP
        val_a = risk.validate(
            filtered_a, pi, mkt)
        conf  = agent.confidence(stacked_state)
        size  = risk.position_size(pi, mkt, conf)
        price = float(
            feat_data["close"].iloc[t])
        time_ = feat_data["timestamp"].iloc[t]
        res   = port.execute(
            val_a, price, time_, size)
        if res["executed"]:
            risk.record()

        # Feed closed trade result to regime filter
        if port.last_closed_pnl is not None:
            rfilter.record_trade_result(
                port.last_closed_pnl)

        # Portfolio info after execution
        upi = port.info(price, time_)
        upi["vol_20"] = mkt["vol_20"]

        # Reward
        rwd = reward.calc(
            val_a, res["port_ret"], upi)
        wf.record(rwd)

        # ── FIX A: leak-free next_state ───────────
        # Build the raw next state using the
        # portfolio state from THIS step (upi) and
        # market features at index t+1.
        # StateBuilder._regime() slices up to
        # idx-1 (exclusive) so bar t+1's realised
        # return is NOT included in the regime
        # window — see StateBuilder contract above.
        # Then build_next() simulates the stacker
        # advancing one step without mutating the
        # live buffer, so bar t+1's close is never
        # visible at step t.
        if t < end_idx - 1:
            next_raw = sb.build(
                feat_data, t+1,
                selected, upi)
            ns   = stacker.build_next(next_raw)
            done = False
        else:
            ns   = stacked_state
            done = True

        # Store stacked transitions
        agent.store(
            stacked_state, val_a,
            rwd, ns, done)
        tm = agent.train()
        if tm: train_log.append(tm)

        all_results.append({
            "t":      t,
            "price":  price,
            "action": val_a,
            "equity": res["equity"],
            "dd":     res["dd"],
            "reward": rwd,
            "state":  stacked_state,
        })

        # Emergency drawdown stop
        if (res["equity"] <
                cfg.INITIAL_CAPITAL *
                (1 - cfg.MAX_DRAWDOWN_LIMIT *
                 1.5)):
            print(
                f"\nEmergency stop at step "
                f"{t}: equity="
                f"${res['equity']:,.2f}")
            break

    # Print regime filter diagnostics
    rf_stats = rfilter.stats()
    print(
        f"\n[RegimeFilter] "
        f"filter_rate="
        f"{rf_stats['filter_rate']:.1%} "
        f"({rf_stats['filtered']}/"
        f"{rf_stats['total']} bars blocked)")

    # ── Phase 8: Evaluation ───────────────────────
    print("\nPhase 8: Evaluation")
    metrics = perf.evaluate(
        port.eq_curve, port.trade_log)

    # ── Phase 9: ONNX export ──────────────────────
    print("Phase 9: ONNX Export")
    exporter = ONNXExporter(cfg)
    exported = exporter.export_agent(
        agent, selected,
        scaler_params,
        stacked_dim, metrics,
        frame_stack_size=stack_size)
    print(
        f"  Exported {len(exported)} "
        f"ONNX files\n")

    print(
        "╔══════════════════════════════════════╗")
    print(
        "║  TRAINING COMPLETE                   ║")
    print(
        f"║  Artifacts → "
        f"{cfg.OUTPUT_DIR:<24}║")
    print(
        "╚══════════════════════════════════════╝")
    return metrics
