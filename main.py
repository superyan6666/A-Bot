import os
import time
import json
import socket
import logging
from dataclasses import dataclass, field, asdict
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
TRADING_LOG  = 'trade_history.json' 
IS_MANUAL    = os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
IS_CI        = os.environ.get('GITHUB_ACTIONS') == 'true'
PUSH_EMPTY   = os.environ.get('PUSH_EMPTY_RESULT', 'false').lower() in ('true', '1', 'yes')

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── 状态管理函数 ──────────────────────────────────────────────────────────────
def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def load_pushed_state() -> set:
    today = _today_str()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            if state.get('date') == today:
                return set(state.get('pushed_codes', []))
        except Exception as e:
            log.warning(f"读取推送记录失败: {e}")
    return set()

def save_pushed_state(codes: set) -> None:
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump({'date': _today_str(), 'pushed_codes': list(codes)}, f)
    except Exception as e:
        log.error(f"保存推送记录失败: {e}")


# ── 2. 数据契约与配置 (Schema & Config) ────────────────────────────────────────
@dataclass(frozen=True)
class Cols:
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
    INFO_ITEM: str = 'item'
    INFO_VAL: str  = 'value'

C = Cols()

class EnvParser:
    @staticmethod
    def get_float(key: str, default: float) -> float:
        val = os.environ.get(key)
        if not val: return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

@dataclass(frozen=True)
class Config:
    MIN_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MIN_CAP', 20e8))
    MAX_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MAX_CAP', 500e8))
    MIN_PE: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PE', 0))
    MAX_PE: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PE', 100))
    MIN_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MIN_TURNOVER', 2.0))
    MIN_PCT_CHG: float   = field(default_factory=lambda: EnvParser.get_float('MIN_PCT_CHG', 3.5))
    MIN_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MIN_VOL_RATIO', 1.0))
    MAX_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MAX_VOL_RATIO', 10.0))
    
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


# ── 3. 数学与工具库 (MathUtils) ────────────────────────────────────────────────
class MathUtils:
    @staticmethod
    def tdx_dma(close: pd.Series, alpha: pd.Series) -> pd.Series:
        c, a = close.to_numpy(), alpha.to_numpy()
        out, last = np.empty_like(c), c[0]
        for i in range(len(c)):
            last = a[i] * c[i] + (1 - a[i]) * last if not np.isnan(a[i]) else last
            out[i] = last
        return pd.Series(out, index=close.index)

    @staticmethod
    def calc_atr_adx(hist: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
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


# ── 4. 数据拉取模块 (Robust Data Fetching) ──────────────────────────────────
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
    try:
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        if df is None or df.empty: return {}
        priority_targets = ['主力净流入-净额', '今日主力净流入净额', '主力净流入净额']
        col = next((c for c in priority_targets if c in df.columns), None)
        if not col:
            col = next((c for c in df.columns if '主力净流入' in c and '净额' in c), None)
        if not col: return {}
        return dict(zip(df[C.S_CODE], pd.to_numeric(df[col], errors='coerce').fillna(0.0)))
    except Exception:
        return {}


# ── 5. 指标与评分引擎 (Core Logic) ───────────────────────────────────────────
class AShareTechnicals:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        close = self.df[C.H_CLOSE]
        high, low, vol = self.df[C.H_HIGH], self.df[C.H_LOW], self.df[C.H_VOL]
        
        for span in (10, 20, 60, 250):
            self.df[f'MA{span}'] = close.rolling(span).mean()
        self.df['MA5_V'] = vol.rolling(5).mean()
        self.df['MA20_V'] = vol.rolling(20).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        self.df['DIF'] = ema12 - ema26
        self.df['DEA'] = self.df['DIF'].ewm(span=9, adjust=False).mean()

        cc = (abs((2 * close + high + low) / 4 - self.df['MA20']) / self.df['MA20'])
        self.df['上'] = 1.07 * MathUtils.tdx_dma(close, cc)
        self.df['ATR'], self.df['ADX'] = MathUtils.calc_atr_adx(self.df)
        self.df['REF_C'] = close.shift()
        self.df['REF_上'] = self.df['上'].shift()
        self.df['PCT_CHG'] = close.pct_change() * 100
        self.df['OBV'] = np.where(close > self.df['REF_C'], vol, np.where(close < self.df['REF_C'], -vol, 0)).cumsum()
        
        self.today = self.df.iloc[-1]
        self.yest = self.df.iloc[-2]
        self.two_days_ago = self.df.iloc[-3] if len(self.df) >= 3 else None

    def get_features(self) -> Optional[dict]:
        df, today, yest = self.df, self.today, self.yest
        if pd.isna(today['ATR']) or today['ATR'] <= 1e-5: return None

        min_1y, max_1y = df[C.H_LOW].min(), df[C.H_HIGH].max()
        rng = max_1y - min_1y
        if rng <= 0: return None
        
        price_pct = (today[C.H_CLOSE] - min_1y) / rng
        if price_pct > 0.85: return None

        # 核心多头趋势 + 通道突破
        if not (today[C.H_CLOSE] >= today['MA60'] and today['DIF'] > -0.15): return None
        if not (today[C.H_CLOSE] > today['上'] and today[C.H_VOL] > today['MA20_V'] * 1.2): return None

        # 筹码峰算法
        has_chip_break = False
        rec120 = df.iloc[-121:-1]
        if len(rec120) > 20 and rec120[C.H_VOL].sum() > 0:
            counts, edges = np.histogram(rec120[C.H_CLOSE].values, bins=20, weights=rec120[C.H_VOL].values)
            poc = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
            has_chip_break = bool((today['REF_C'] <= poc) and (today[C.H_CLOSE] > poc))

        zt_mask = df['PCT_CHG'].iloc[-61:-1] >= 9.5
        
        return {
            'price_pct': price_pct, 'max_1y': max_1y, 'adx': float(today['ADX']),
            'bull_rank': (today['MA20'] > today['MA60']),
            'has_zt': bool(zt_mask.any()), 'has_consecutive_zt': bool((zt_mask.rolling(2).sum() >= 2).any()),
            'vcp_amp': (df[C.H_HIGH].iloc[-11:-1].max() - df[C.H_LOW].iloc[-11:-1].min()) / df[C.H_LOW].iloc[-11:-1].min() if df[C.H_LOW].iloc[-11:-1].min() > 0 else 0.5,
            'upper_shadow_pct': ((today[C.H_HIGH] - today[C.H_CLOSE]) / (today[C.H_CLOSE]-today[C.H_OPEN])*100) if (today[C.H_CLOSE]-today[C.H_OPEN]) > 0 else 0.0,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'has_pullback': bool(self.two_days_ago is not None and self.two_days_ago[C.H_CLOSE] > self.two_days_ago['上'] and yest[C.H_CLOSE] <= yest['上'] and yest[C.H_VOL] < yest['MA5_V'] and yest[C.H_CLOSE] > yest['MA20']),
            'has_chip_break': has_chip_break,
            'ma10_val': float(today['MA10']), 'ma20_val': float(today['MA20']), 'atr_val': float(today['ATR']),
            'low_val': float(today[C.H_LOW]), 'recent_20_low': float(df[C.H_LOW].iloc[-20:].min()),
            'boll_lower': float(today['MA20'] - 2 * df[C.H_CLOSE].iloc[-20:].dropna().std()) if len(df[C.H_CLOSE].iloc[-20:].dropna()) >= 2 else np.nan,
            'close_60d_ago': float(df[C.H_CLOSE].iloc[-60]) if len(df) >= 60 else 0.0,
        }

@dataclass
class Factor:
    condition: Callable[[dict], bool]
    points: int
    weight: float = 1.0
    template: str = ""

def apply_scoring(data: dict) -> tuple[int, str, str]:
    adx = data['adx']
    tw, rw = (1.4, 0.7) if adx > 25 else (0.8, 1.4) if adx < 15 else (1.0, 1.0)
    meta = f"🧭 ADX={adx:.1f} {'【强趋势模式】' if adx > 25 else '【低位转强模式】' if adx < 15 else ''}"

    factors = [
        Factor(lambda d: d['price_pct'] < 0.25, 15, rw, "🟢 处于波段底部起涨区(分位{price_pct_pct:.1f}%)"),
        Factor(lambda d: 0.25 <= d['price_pct'] < 0.55, 10, 1.0, "🟢 处于主升拉升区(分位{price_pct_pct:.1f}%)"),
        Factor(lambda d: d['bull_rank'], 10, 1.0, "📈 均线确认多头强势趋势"),
        Factor(lambda d: d['has_zt'], 15, 1.0, "🔥 历史涨停基因活跃(股性佳)"),
        Factor(lambda d: d['has_consecutive_zt'], 12, 1.0, "🔥🔥 连板妖股潜质(市场高度辨识度)"),
        Factor(lambda d: d['vol_ratio'] >= 1.8, 10, 1.0, "🔵 显著放量突破({vol_ratio:.1f}x)"),
        Factor(lambda d: d['has_chip_break'], 15, tw, "🏔️ 强力跨越筹码密集峰(解套动能释放)"),
        Factor(lambda d: d['vcp_amp'] < 0.12, 8, 1.0, "🟣 VCP收缩形态(振幅{vcp_amp_pct:.1f}%)蓄势充分"),
        Factor(lambda d: d['has_obv_break'], 10, tw, "💸 OBV量能创近期新高"),
        Factor(lambda d: d['has_pullback'], 15, 1.0, "🪃 符合缩量回踩形态确认"),
        Factor(lambda d: d['rs_rating'] > 15, 10, tw, "🏆 相对强度优势(超额收益 {rs_rating:.1f}%)"),
        Factor(lambda d: d['has_fund_inflow'], 12, tw, "💰 主力抢筹({fund_flow_w:.0f}万)状态确认"),
    ]

    score, reasons = 40, [meta] if meta else []
    data['price_pct_pct'] = data['price_pct'] * 100
    data['vcp_amp_pct'] = data['vcp_amp'] * 100
    
    for f in factors:
        if f.condition(data):
            score += int(f.points * f.weight)
            reasons.append(f.template.format(**data))

    score = min(score, 100)
    level = '⭐⭐⭐⭐⭐ [S级极品]' if score >= 82 else '⭐⭐⭐⭐ [A级强势]' if score >= 72 else '⭐⭐⭐ [B级标准]'
    return score, level, '\n'.join(reasons)


# ── 6. 核心分析流水线 (Signal Pipeline) ───────────────────────────────────────
def is_trading_time(now: datetime) -> bool:
    t = now.hour * 100 + now.minute
    return (925 <= t <= 1135) or (1255 <= t <= 1505)

def process_stock(row: pd.Series, raw_hist: pd.DataFrame, now: datetime, market_ok: bool, index_ret: float, flow: float) -> Optional[Signal]:
    if len(raw_hist) < 250: return None
    
    hist = raw_hist.copy()
    if str(hist[C.H_DATE].iloc[-1]) != now.strftime('%Y-%m-%d') and is_trading_time(now):
        synthetic = pd.DataFrame([{
            C.H_DATE: now.strftime('%Y-%m-%d'), C.H_OPEN: float(row.get(C.S_OPEN, row[C.S_PRICE])),
            C.H_HIGH: float(row[C.S_HIGH]), C.H_LOW: float(row.get(C.S_LOW, row[C.S_PRICE])),
            C.H_CLOSE: float(row[C.S_PRICE]), C.H_VOL: float(row.get(C.S_VOL, 1.0))
        }])
        hist = pd.concat([hist, synthetic], ignore_index=True)

    if hist.iloc[-1][C.H_CLOSE] <= hist.iloc[-1][C.H_OPEN] or hist.iloc[-1][C.H_VOL] <= 0: return None
    
    engine = AShareTechnicals(hist)
    data = engine.get_features()
    if not data: return None

    # 特征富化
    data['vol_ratio'] = float(row.get(C.S_VR, 1.0))
    data['fund_flow_w'] = flow / 10000.0
    data['has_fund_inflow'] = flow > max(1e7, float(row.get(C.S_MCAP, 0)) * 0.001)
    data['rs_rating'] = ((row[C.S_PRICE] / data['close_60d_ago'] - 1) * 100 - index_ret) if data['close_60d_ago'] > 0 else 0
    
    score, level, reas = apply_scoring(data)
    if score < 60: return None 

    # 风险测算
    supports = [data['ma20_val'], data['recent_20_low'], data['boll_lower']]
    valid_supports = [s for s in supports if pd.notna(s) and s < row[C.S_PRICE]]
    stop = max(valid_supports + [row[C.S_PRICE] * 0.925]) * 0.992
    risk_pct = ((row[C.S_PRICE] - stop) / row[C.S_PRICE]) * 100 if row[C.S_PRICE] > 0 else 99
    
    if risk_pct > 8.5: return None

    pos = (30 if score >= 82 else 20 if score >= 72 else 10)
    if not market_ok: pos //= 2

    return Signal(
        code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
        pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
        position_advice=f"⚖️ 建议仓位 {pos}% (风险系数 {risk_pct:.1f}%)",
        trigger_time=now.strftime('%H:%M'), reasons=reas,
        stop_loss=round(stop, 2), target1=round(row[C.S_PRICE]*(1+risk_pct*2.2/100), 2),
        target2=round(data['max_1y'], 2), ma10=round(data['ma10_val'], 2)
    )


# ── 7. 控制器与板块增强 (Orchestrator & Sector Bridge) ──────────────────────────
def fetch_stock_sector(code: str) -> str:
    """【通路修复】：补充获取个股所属行业信息"""
    try:
        df = ak.stock_individual_info_em(symbol=code)
        v = df[df[C.INFO_ITEM] == '行业'][C.INFO_VAL].values[0]
        return str(v) if pd.notna(v) else ""
    except: return ""

def fetch_sector_strength() -> dict:
    """【通路修复】：获取全行业今日涨跌幅，用于强度背书"""
    try:
        df = ak.stock_board_industry_spot_em()
        return dict(zip(df[C.B_NAME], df[C.B_PCT]))
    except: return {}

def extract_market_context(df_raw: pd.DataFrame, c_conf: Config) -> tuple[pd.DataFrame, bool, str, float]:
    if len(df_raw) < 1000: return pd.DataFrame(), False, "API 响应异常", 0.0
    market_ok, market_msg, index_ret = True, "", 0.0
    try:
        idx_df = ak.stock_zh_index_daily_em(symbol='sh000001')
        cl = idx_df[C.I_CLOSE]
        ma20 = cl.rolling(20).mean().iloc[-1]
        market_ok = (cl.iloc[-1] > ma20)
        market_msg = f"上证 {cl.iloc[-1]:.2f} {'(做多窗口)' if market_ok else '(防守窗口)'}"
        index_ret = ((cl.iloc[-1] / cl.iloc[-60]) - 1) * 100 if len(cl) >= 60 else 0.0
    except: pass
    df = df_raw.dropna(subset=list(c_conf.REQUIRED_COLS))
    df = df[~df[C.S_NAME].str.contains('ST|退')]
    return df, market_ok, market_msg, index_ret

def get_signals() -> tuple[list[Signal], set, int]:
    now = datetime.now(TZ_BJS)
    log.info('🚀 A股弹性平衡量化引擎启动...')
    if not IS_MANUAL and not is_trading_time(now): return [], set(), 0

    c_conf, pushed = Config(), load_pushed_state()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_spot, f_flow = ex.submit(fetch_spot), ex.submit(get_fund_flow_map)
        df_raw, flow_map = f_spot.result(timeout=25), f_flow.result(timeout=15)

    df_clean, m_ok, m_msg, idx_ret = extract_market_context(df_raw, c_conf)
    log.info(f"📈 扫描中: {m_msg}")

    mask = (df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & \
           (df_clean[C.S_MCAP].between(c_conf.MIN_CAP, c_conf.MAX_CAP)) & \
           (df_clean[C.S_TURN] >= c_conf.MIN_TURNOVER)
    
    if C.S_VR in df_clean.columns:
        t_val = now.hour * 100 + now.minute
        vr_max = 9.5 if t_val <= 1030 else 6.5
        mask &= df_clean[C.S_VR].between(c_conf.MIN_VOL_RATIO, vr_max)

    pool = df_clean[mask].pipe(lambda d: d[~d[C.S_CODE].isin(pushed)]).copy()
    if pool.empty: return [], pushed, 0

    log.info(f"✅ 初选捕获 {len(pool)} 只个股，正在并行深度计算指标...")

    initial_signals = []
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=450)).strftime('%Y%m%d')
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_hist, r[C.S_CODE], start_s, end_s): r for _, r in pool.iterrows()}
        for f in as_completed(futures):
            row = futures[f]
            try:
                hist = f.result(timeout=8)
                sig = process_stock(row, hist, now, m_ok, idx_ret, flow_map.get(row[C.S_CODE], 0.0))
                if sig: initial_signals.append(sig)
            except Exception as e:
                log.debug(f"流水线抛弃 {row[C.S_CODE]}: {e}")

    # 【通路接龙】：后置填充板块强度，只针对通过筛选的极少数精英
    if initial_signals:
        sec_strengths = fetch_sector_strength()
        for s in initial_signals:
            s.sector = fetch_stock_sector(s.code)
            s.sector_pct = sec_strengths.get(s.sector, 0.0)
            pushed.add(s.code) # 正式加入已推送名单

    initial_signals.sort(key=lambda x: x.score, reverse=True)
    return initial_signals, pushed, len(pool)


def send_dingtalk(signals: list[Signal], total: int) -> None:
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook: return
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    header = f"🤖 AI 量化执行纪律单 (弹性增强版) {now_str}\n\n"
    if not signals:
        if not (IS_MANUAL and PUSH_EMPTY): return
        content = f"{header}✅ 深度体检 {total} 只标的，未发现极致信号，系统待机中。"
    else:
        parts = []
        for s in signals:
            sec_info = f"🏷️ 板块：{s.sector} (今日 {s.sector_pct:+.2f}%)\n" if s.sector else ""
            parts.append(
                f"【{s.name} ({s.code})】\n"
                f"{sec_info}"
                f"📊 评分：{s.score}分 {s.level}\n"
                f"💰 现价：¥{s.price} ({s.pct_chg})\n"
                f"--- 核心逻辑 ---\n{s.reasons}\n"
                f"--- 资金管理 ---\n{s.position_advice}\n"
                f"🛑 止损：¥{s.stop_loss}\n"
                f"🥇 目标1：¥{s.target1}\n"
                f"🔗 https://quote.eastmoney.com/unify/r/0.{s.code}\n"
            )
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
