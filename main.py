import os
import time
import json
import socket
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, List, Tuple, Callable, Dict, Any

import requests
from requests.exceptions import RequestException
import numpy as np
import pandas as pd
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# ── 1. 环境与日志配置 ──────────────────────────────────────────────────────────
TZ_BJS       = pytz.timezone('Asia/Shanghai')
STATE_FILE   = 'pushed_state.json'
IS_MANUAL    = os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
IS_CI        = os.environ.get('GITHUB_ACTIONS') == 'true'
PUSH_EMPTY   = os.environ.get('PUSH_EMPTY_RESULT', 'false').lower() in ('true', '1', 'yes')

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


# ── 2. 数据契约与配置 (Schema & Config) ────────────────────────────────────────
@dataclass(frozen=True)
class Cols:
    """统一管理 A 股字段名，杜绝魔术字符串"""
    S_PRICE: str = '最新价'
    S_HIGH: str  = '最高'
    S_LOW: str   = '最低'
    S_OPEN: str  = '今开'
    S_PCT: str   = '涨跌幅'
    S_TURN: str  = '换手率'
    S_AMT: str   = '成交额'
    S_VOL: str   = '成交量'
    S_CODE: str  = '代码'
    S_NAME: str  = '名称'
    S_MCAP: str  = '流通市值'
    S_PE: str    = '市盈率-动态'
    S_PB: str    = '市净率'
    S_VR: str    = '量比'
    
    H_DATE: str  = '日期'
    H_OPEN: str  = '开盘'
    H_CLOSE: str = '收盘'
    H_HIGH: str  = '最高'
    H_LOW: str   = '最低'
    H_VOL: str   = '成交量'
    
    I_CLOSE: str = 'close'
    B_NAME: str  = '板块名称'
    B_PCT: str   = '涨跌幅'

C = Cols()

class EnvParser:
    """运行时配置解析，确保环境变量错误时系统不崩溃"""
    @staticmethod
    def get_float(key: str, default: float) -> float:
        val = os.environ.get(key)
        if not val: return default
        try:
            return float(val)
        except (ValueError, TypeError):
            log.warning(f"⚠️ 配置异常: {key}='{val}'，回退至默认值 {default}")
            return default

@dataclass(frozen=True)
class Config:
    MIN_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MIN_CAP', 30e8))
    MAX_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MAX_CAP', 300e8))
    MIN_PE: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PE', 0))
    MAX_PE: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PE', 60))
    MIN_PB: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PB', 0))
    MAX_PB: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PB', 5))
    MIN_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MIN_TURNOVER', 3.0))
    MIN_AMOUNT: float    = field(default_factory=lambda: EnvParser.get_float('MIN_AMOUNT', 1e8))
    MIN_PCT_CHG: float   = field(default_factory=lambda: EnvParser.get_float('MIN_PCT_CHG', 4.5))
    MIN_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MIN_VOL_RATIO', 1.2))
    MAX_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MAX_VOL_RATIO', 5.0))
    
    REQUIRED_COLS: tuple = (C.S_PRICE, C.S_OPEN, C.S_HIGH, C.S_LOW, C.S_VOL, C.S_AMT, 
                            C.S_PCT, C.S_TURN, C.S_CODE, C.S_NAME, C.S_MCAP, C.S_PE, C.S_PB)
    OPTIONAL_COLS: tuple = (C.S_VR,)
    HIST_COLS: tuple     = (C.H_DATE, C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL)


@dataclass
class Signal:
    code: str
    name: str
    price: float
    pct_chg: str
    score: int
    level: str
    position_advice: str
    trigger_time: str
    reasons: str
    stop_loss: float
    target1: float
    target2: float
    ma10: float
    sector: str = ""
    sector_pct: float = 0.0


# ── 3. 数据拉取模块 (Data Fetching) ───────────────────────────────────────────
import akshare as ak

def retry(times=3, delay=2, exceptions=(RequestException, ConnectionError, ValueError, pd.errors.EmptyDataError)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt < times - 1:
                        time.sleep(delay * (2 ** attempt))
                    else:
                        raise
        return wrapper
    return decorator

@retry(times=3, delay=2)
def fetch_hist(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust='qfq')
    if df is None or df.empty: raise ValueError('history_empty')
    c_conf = Config()
    missing = [c for c in c_conf.HIST_COLS if c not in df.columns]
    if missing: raise ValueError(f'missing_cols: {missing}')
    return df[list(c_conf.HIST_COLS)].copy()

@retry(times=3, delay=2)
def fetch_spot() -> pd.DataFrame:
    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty: raise ValueError('spot_empty')
    return df

def get_fund_flow_map() -> dict:
    """稳健获取全市场今日资金流向"""
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        if df is None or df.empty: return {}
        target_cols = ['主力净流入-净额', '今日主力净流入净额', '主力净流入净额']
        col = next((c for c in target_cols if c in df.columns), None)
        if not col:
            col = next((c for c in df.columns if '主力净流入' in c and '净额' in c), None)
        if not col: return {}
        return dict(zip(df[C.S_CODE], pd.to_numeric(df[col], errors='coerce').fillna(0.0)))
    except Exception:
        return {}


# ── 4. A 股指标计算引擎 (Indicator Engine) ─────────────────────────────────────
def tdx_dma(close: pd.Series, alpha: pd.Series) -> pd.Series:
    c, a = close.to_numpy(), alpha.to_numpy()
    out, last = np.empty_like(c), c[0]
    for i in range(len(c)):
        last = a[i] * c[i] + (1 - a[i]) * last if not np.isnan(a[i]) else last
        out[i] = last
    return pd.Series(out, index=close.index)

def calc_atr_adx(hist: pd.DataFrame, period: int = 14):
    high, low, close = hist[C.H_HIGH], hist[C.H_LOW], hist[C.H_CLOSE]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up, dn = high.diff(), -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=hist.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=hist.index).rolling(period).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom * 100)
    return atr, dx.rolling(period).mean()

class AShareTechnicals:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        close = self.df[C.H_CLOSE]
        high, low, vol = self.df[C.H_HIGH], self.df[C.H_LOW], self.df[C.H_VOL]
        
        # 1. 均线与成交量均线
        for span in (10, 20, 60, 250):
            self.df[f'MA{span}'] = close.rolling(span).mean()
        self.df['MA5_V'] = vol.rolling(5).mean()
        self.df['MA20_V'] = vol.rolling(20).mean()

        # 2. MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['DIF'] = ema12 - ema26
        self.df['DEA'] = self.df['DIF'].ewm(span=9, adjust=False).mean()

        # 3. 动态移动平均 (DMA)
        cc = (abs((2 * close + high + low) / 4 - self.df['MA20']) / self.df['MA20'])
        self.df['上'] = 1.07 * tdx_dma(close, cc)

        # 4. ATR/ADX/OBV/RSI
        self.df['ATR'], self.df['ADX'] = calc_atr_adx(self.df)
        self.df['REF_C'] = close.shift()
        self.df['REF_上'] = self.df['上'].shift()
        self.df['PCT_CHG'] = close.pct_change() * 100
        self.df['OBV'] = np.where(close > self.df['REF_C'], vol, np.where(close < self.df['REF_C'], -vol, 0)).cumsum()
        
        delta = close.diff()
        rs = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / (-delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean())
        self.df['RSI'] = 100 - (100 / (1 + rs))

        self.today = self.df.iloc[-1]
        self.yest = self.df.iloc[-2]
        self.two_days_ago = self.df.iloc[-3] if len(self.df) >= 3 else None

    def get_features(self) -> Optional[dict]:
        """全量特征提取，带极值防护"""
        df, today, yest = self.df, self.today, self.yest
        if pd.isna(today['ATR']) or today['ATR'] <= 1e-5 or pd.isna(today['MA250']): return None

        min_1y, max_1y = df[C.H_LOW].min(), df[C.H_HIGH].max()
        rng = max_1y - min_1y
        if rng <= 0: return None
        
        price_pct = (today[C.H_CLOSE] - min_1y) / rng
        body = today[C.H_CLOSE] - today[C.H_OPEN]
        
        # 核心突破判断
        bull_trend = (today[C.H_CLOSE] >= today['MA250']) and (today['MA20'] >= today['MA60']) and (today['ADX'] > 20)
        macd_bull = (today['DIF'] > 0) and (today['DEA'] > 0)
        is_cross = (today[C.H_CLOSE] > today['上']) and (today['REF_C'] <= today['REF_上']) and (today[C.H_VOL] > today['MA20_V'] * 1.4)
        
        if not (bull_trend and macd_bull and is_cross): return None

        # 筹码峰逻辑
        has_chip_break = False
        rec120 = df.iloc[-121:-1]
        if len(rec120) > 20 and rec120[C.H_VOL].sum() > 0:
            counts, edges = np.histogram(rec120[C.H_CLOSE].values, bins=20, weights=rec120[C.H_VOL].values)
            poc = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
            has_chip_break = bool((today['REF_C'] <= poc) and (today[C.H_CLOSE] > poc))

        zt_mask = df['PCT_CHG'].iloc[-61:-1] >= 9.5
        
        return {
            'price_pct': price_pct, 'max_1y': max_1y, 'adx': float(today['ADX']),
            'has_zt': bool(zt_mask.any()), 'has_consecutive_zt': bool((zt_mask.rolling(2).sum() >= 2).any()),
            'vcp_amp': (df[C.H_HIGH].iloc[-11:-1].max() - df[C.H_LOW].iloc[-11:-1].min()) / df[C.H_LOW].iloc[-11:-1].min() if df[C.H_LOW].iloc[-11:-1].min() > 0 else 0,
            'upper_shadow_pct': ((today[C.H_HIGH] - today[C.H_CLOSE]) / body) * 100 if body > 0 else 0.0,
            'lower_shadow_ratio': (min(today[C.H_OPEN], today[C.H_CLOSE]) - today[C.H_LOW]) / today[C.H_OPEN] if pd.notna(today[C.H_OPEN]) and today[C.H_OPEN] > 0 else 0.0,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'has_pullback': bool(self.two_days_ago is not None and self.two_days_ago[C.H_CLOSE] > self.two_days_ago['上'] and yest[C.H_CLOSE] <= yest['上'] and yest[C.H_VOL] < yest['MA5_V'] and yest[C.H_CLOSE] > yest['MA20']),
            'has_chip_break': has_chip_break,
            'ma10_val': float(today['MA10']), 'ma20_val': float(today['MA20']), 'atr_val': float(today['ATR']),
            'low_val': float(today[C.H_LOW]), 'recent_20_low': float(df[C.H_LOW].iloc[-20:].min()),
            'boll_lower': float(today['MA20'] - 2 * df[C.H_CLOSE].iloc[-20:].dropna().std()) if len(df[C.H_CLOSE].iloc[-20:].dropna()) >= 2 else np.nan,
            'close_60d_ago': float(df[C.H_CLOSE].iloc[-60]) if len(df) >= 60 else 0.0,
        }


# ── 5. 打分引擎 (Scoring Engine) ──────────────────────────────────────────────
@dataclass
class Factor:
    condition: Callable[[dict], bool]
    points: int
    weight: float = 1.0
    template: str = ""

def apply_scoring(data: dict) -> tuple[int, str, str]:
    adx = data['adx']
    tw, rw = (1.5, 0.8) if adx > 30 else (0.8, 1.5) if adx < 20 else (1.0, 1.0)
    meta = f"🧭 ADX={adx:.1f} {'【趋势增强】' if adx > 30 else '【震荡增强】' if adx < 20 else ''}"

    factors = [
        Factor(lambda d: d['price_pct'] < 0.15, 15, rw, "🟢 极度冰点(分位{price_pct_pct:.1f}%)支撑极强"),
        Factor(lambda d: 0.15 <= d['price_pct'] < 0.3, 10, rw, "🟢 底部区域(分位{price_pct_pct:.1f}%)蓄势待发"),
        Factor(lambda d: d['has_zt'], 15, 1.0, "🔥 历史涨停基因(股性活跃)"),
        Factor(lambda d: d['has_consecutive_zt'], 10, 1.0, "🔥🔥 连板妖股基因(辨识度极高)"),
        Factor(lambda d: d['vol_ratio'] >= 2.0, 10, 1.0, "🔵 完美倍量突破({vol_ratio:.1f}x)"),
        Factor(Factor(lambda d: d['has_chip_break'], 15, tw, "🏔️ 跨越半年筹码密集峰，空间打开").condition, 15, tw, "🏔️ 跨越半年筹码密集峰，空间打开"),
        Factor(lambda d: d['has_obv_break'], 10, tw, "💸 OBV量能先行创近期新高"),
        Factor(lambda d: d['has_pullback'], 15, 1.0, "🪃 老鸭头缩量回踩后反包确认"),
        Factor(lambda d: d['rs_rating'] > 20, 10, tw, "🏆 相对强度极强(超额收益 {rs_rating:.1f}%)"),
        Factor(lambda d: d['has_fund_inflow'], 10, tw, "💰 主力抢筹({fund_flow_w:.0f}万)态度坚决"),
    ]

    score, reasons = 40, [meta] if meta else []
    data['price_pct_pct'] = data['price_pct'] * 100
    
    for f in factors:
        if f.condition(data):
            score += int(f.points * f.weight)
            reasons.append(f.template.format(**data))

    score = min(score, 100)
    level = '⭐⭐⭐⭐⭐ [S级]' if score >= 85 else '⭐⭐⭐⭐ [A级]' if score >= 75 else '⭐⭐⭐ [B级]'
    return score, level, '\n'.join(reasons)


# ── 6. 核心分析流水线 (Main Pipeline) ──────────────────────────────────────────
def process_stock(row: pd.Series, raw_hist: pd.DataFrame, now: datetime, market_ok: bool, index_ret: float, flow: float) -> Optional[Signal]:
    if len(raw_hist) < 250: return None
    
    # 缝合实时 Bar
    hist = raw_hist.copy()
    if str(hist[C.H_DATE].iloc[-1]) != now.strftime('%Y-%m-%d') and is_trading_time(now):
        synthetic = pd.DataFrame([{
            C.H_DATE: now.strftime('%Y-%m-%d'), C.H_OPEN: float(row.get(C.S_OPEN, row[C.S_PRICE])),
            C.H_HIGH: float(row[C.S_HIGH]), C.H_LOW: float(row.get(C.S_LOW, row[C.S_PRICE])),
            C.H_CLOSE: float(row[C.S_PRICE]), C.H_VOL: float(row.get(C.S_VOL, 1.0))
        }])
        hist = pd.concat([hist, synthetic], ignore_index=True)

    # 预检
    if hist.iloc[-1][C.H_CLOSE] <= hist.iloc[-1][C.H_OPEN] or hist.iloc[-1][C.H_VOL] <= 0: return None
    
    engine = AShareTechnicals(hist)
    data = engine.get_features()
    if not data: return None

    # 特征富化
    data['vol_ratio'] = float(row.get(C.S_VR, 1.0))
    data['fund_flow'] = flow
    data['fund_flow_w'] = flow / 10000.0
    data['has_fund_inflow'] = flow > max(1e7, float(row.get(C.S_MCAP, 0)) * 0.001)
    data['rs_rating'] = ((row[C.S_PRICE] / data['close_60d_ago'] - 1) * 100 - index_ret) if data['close_60d_ago'] > 0 else 0
    
    # 评分与风控
    score, level, reas = apply_scoring(data)
    
    supports = [data['ma20_val'], data['recent_20_low'], data['boll_lower']]
    valid_supports = [s for s in supports if pd.notna(s) and s < row[C.S_PRICE]]
    stop = max(valid_supports + [row[C.S_PRICE] * 0.94]) * 0.99
    risk_pct = ((row[C.S_PRICE] - stop) / row[C.S_PRICE]) * 100
    
    if risk_pct > 10.0:
        log.info(f"🚫 风控拦截: {row[C.S_NAME]} ({row[C.S_CODE]}) 止损空间过深({risk_pct:.1f}%)")
        return None

    pos = (30 if score >= 85 else 20 if score >= 75 else 10)
    if not market_ok: pos //= 2

    return Signal(
        code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
        pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
        position_advice=f"⚖️ 建议仓位 {pos}% (风险系数 {risk_pct:.1f}%)",
        trigger_time=now.strftime('%H:%M'), reasons=reas,
        stop_loss=round(stop, 2), target1=round(row[C.S_PRICE]*(1+risk_pct*2.1/100), 2),
        target2=round(data['max_1y'], 2), ma10=round(data['ma10_val'], 2)
    )


# ── 7. 控制器 (Controller) ───────────────────────────────────────────────────
def get_signals() -> tuple[list[Signal], set, int]:
    now = datetime.now(TZ_BJS)
    log.info('🚀 A股量化打板监控系统启动...')
    if not IS_MANUAL and not is_trading_time(now): return [], set(), 0

    c_conf = Config()
    pushed = load_pushed_state()
    
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_spot = ex.submit(fetch_spot)
        f_flow = ex.submit(get_fund_flow_map)
        df_raw = f_spot.result(timeout=20)
        flow_map = f_flow.result(timeout=10)

    df_clean, market_ok, mkt_msg, index_ret = extract_market_context(df_raw, c_conf)
    log.info(f"📈 市场环境: {mkt_msg}")

    # 向量化初筛
    mask = (df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & \
           (df_clean[C.S_MCAP].between(c_conf.MIN_CAP, c_conf.MAX_CAP)) & \
           (df_clean[C.S_TURN] >= c_conf.MIN_TURNOVER)
    
    if C.S_VR in df_clean.columns:
        # A 股 U 型量比逻辑：早盘放宽至 8.0，午后收紧
        t_val = now.hour * 100 + now.minute
        vr_min = max(c_conf.MIN_VOL_RATIO, 1.5 if not market_ok else 0)
        vr_max = 8.0 if t_val <= 1030 else c_conf.MAX_VOL_RATIO
        mask &= df_clean[C.S_VR].between(vr_min, vr_max)

    pool = df_clean[mask].pipe(lambda d: d[~d[C.S_CODE].isin(pushed)]).copy()
    if pool.empty: return [], pushed, 0

    # 深度分析
    signals = []
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=400)).strftime('%Y%m%d')
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_hist, r[C.S_CODE], start_s, end_s): r for _, r in pool.iterrows()}
        for f in as_completed(futures):
            row = futures[f]
            try:
                hist = f.result(timeout=6)
                sig = process_stock(row, hist, now, market_ok, index_ret, flow_map.get(row[C.S_CODE], 0.0))
                if sig:
                    signals.append(sig)
                    pushed.add(row[C.S_CODE])
            except Exception as e:
                log.debug(f"跳过 {row[C.S_CODE]}: {e}")

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals, pushed, len(pool)


def send_dingtalk(signals: list[Signal], total: int) -> None:
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook: return
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    header = f"🤖 A股量化执行纪律单 {now_str}\n\n"
    if not signals:
        if not (IS_MANUAL and PUSH_EMPTY): return
        content = f"{header}✅ 深度体检 {total} 只标的，无极致突破信号，系统守候中。"
    else:
        parts = [f"【{s.name} ({s.code})】\n📊 评分：{s.score}分 {s.level}\n💰 现价：¥{s.price} ({s.pct_chg})\n--- 核心逻辑 ---\n{s.reasons}\n--- 资金管理 ---\n{s.position_advice}\n🛑 止损：¥{s.stop_loss}\n🥇 目标1：¥{s.target1}\n🔗 https://quote.eastmoney.com/unify/r/0.{s.code}\n" for s in signals]
        content = header + "\n".join(parts)

    try:
        requests.post(webhook, json={'msgtype': 'text', 'text': {'content': content}}, timeout=10)
        log.info(f"✅ 推送成功 ({len(signals)} 只)")
    except Exception as e:
        log.error(f"❌ 推送失败: {e}")

if __name__ == '__main__':
    try:
        sigs, pushed, total = get_signals()
        send_dingtalk(sigs, total)
        if sigs: save_pushed_state(pushed)
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
