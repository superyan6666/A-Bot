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
PUSH_EMPTY   = os.environ.get('PUSH_EMPTY_RESULT', 'true').lower() in ('true', '1', 'yes')

# 强制要求所有隐式网络请求最多 15 秒超时，避免后台线程卡死阻碍 Python 进程退出
socket.setdefaulttimeout(15.0)

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
    # --- 为小白定制的安全基线配置 ---
    MIN_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MIN_CAP', 30e8)) # 提高至30亿，规避微盘股流动性杀
    MAX_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MAX_CAP', 500e8))
    MIN_PE: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PE', 0))    # 必须盈利，拒绝亏损股
    MAX_PE: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PE', 100))  # 拒绝估值极度泡沫
    MIN_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MIN_TURNOVER', 1.8))
    MIN_PCT_CHG: float   = field(default_factory=lambda: EnvParser.get_float('MIN_PCT_CHG', 3.0))
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


# ── 3. 数学与工具库 ────────────────────────────────────────────────────────────
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


# ── 4. 数据拉取模块 ────────────────────────────────────────────────────────────
import akshare as ak

# 【修正】：将 Exception 纳入重试捕获范围，针对 akshare 库不确定的网络抛错进行全面拦截
def retry(times=4, delay=2, exceptions=(Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt < times - 1:
                        log.debug(f"抓取受挫，将在 {delay * (2 ** attempt)}s 后重试: {e}")
                        time.sleep(delay * (2 ** attempt))
                    else:
                        raise
        return wrapper
    return decorator

@retry(times=4, delay=2)
def fetch_index(symbol: str) -> pd.DataFrame:
    """带重试护甲的指数获取器"""
    df = ak.stock_zh_index_daily_em(symbol=symbol)
    if df is None or df.empty: raise ValueError(f'index_empty_{symbol}')
    return df

@retry(times=3, delay=2)
def fetch_hist(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust='qfq')
    if df is None or df.empty: raise ValueError('history_empty')
    return df[list(Config.HIST_COLS)].copy()

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


# ── 5. A 股指标引擎 (Indicator Engine) ─────────────────────────────────────────
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
        # 【防站岗机制】：强行压低水位线，近一年处在中高位（>65%）的股票直接一票否决
        if price_pct > 0.65: return None 

        if not (today[C.H_CLOSE] >= today['MA60'] and today['DIF'] > -0.25): return None
        if not (today[C.H_CLOSE] > today['上'] and today[C.H_VOL] >= today['MA20_V'] * 1.0): return None

        ma_convergence = (abs(today['MA10'] - today['MA20']) / today['MA20'] < 0.03) and \
                         (abs(today['MA20'] - today['MA60']) / today['MA60'] < 0.05)
        extreme_shrink_vol = yest[C.H_VOL] < today['MA20_V'] * 0.65

        rec120 = df.iloc[-121:-1]
        has_chip_break = False
        if len(rec120) > 20 and rec120[C.H_VOL].sum() > 0:
            counts, edges = np.histogram(rec120[C.H_CLOSE].values, bins=20, weights=rec120[C.H_VOL].values)
            poc = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
            has_chip_break = bool((today['REF_C'] <= poc) and (today[C.H_CLOSE] > poc))

        red_days = 0
        for i in range(1, 4):
            if df[C.H_CLOSE].iloc[-i] > df[C.H_OPEN].iloc[-i]: red_days += 1
            else: break

        return {
            'price_pct': price_pct, 'max_1y': max_1y, 'adx': float(today['ADX']),
            'bull_rank': (today['MA20'] > today['MA60']),
            'ma_convergence': ma_convergence,
            'extreme_shrink_vol': extreme_shrink_vol,
            'has_zt': bool((df['PCT_CHG'].iloc[-61:-1] >= 9.5).any()),
            'has_consecutive_zt': bool(((df['PCT_CHG'].iloc[-61:-1] >= 9.5).rolling(2).sum() >= 2).any()),
            'vcp_amp': (df[C.H_HIGH].iloc[-11:-1].max() - df[C.H_LOW].iloc[-11:-1].min()) / df[C.H_LOW].iloc[-11:-1].min() if df[C.H_LOW].iloc[-11:-1].min() > 0 else 0.5,
            'upper_shadow_pct': ((today[C.H_HIGH] - today[C.H_CLOSE]) / (today[C.H_CLOSE]-today[C.H_OPEN])*100) if (today[C.H_CLOSE]-today[C.H_OPEN]) > 0 else 0.0,
            'lower_shadow_ratio': (min(today[C.H_OPEN], today[C.H_CLOSE]) - today[C.H_LOW]) / today[C.H_OPEN] if pd.notna(today[C.H_OPEN]) and today[C.H_OPEN] > 0 else 0.0,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'has_pullback': bool(self.two_days_ago is not None and self.two_days_ago[C.H_CLOSE] > self.two_days_ago['上'] and yest[C.H_CLOSE] <= yest['上'] and yest[C.H_VOL] < yest['MA5_V'] and yest[C.H_CLOSE] > yest['MA20']),
            'has_chip_break': has_chip_break,
            'dist_ma20': (today[C.H_CLOSE] / today['MA20'] - 1) * 100,
            'red_days': red_days,
            'ma10_val': float(today['MA10']), 'ma20_val': float(today['MA20']), 'atr_val': float(today['ATR']),
            'low_val': float(today[C.H_LOW]), 'recent_20_low': float(df[C.H_LOW].iloc[-20:].min()),
            'boll_lower': float(today['MA20'] - 2 * df[C.H_CLOSE].iloc[-20:].dropna().std()) if len(df[C.H_CLOSE].iloc[-20:].dropna()) >= 2 else np.nan,
            'close_60d_ago': float(df[C.H_CLOSE].iloc[-60]) if len(df) >= 60 else 0.0,
        }


# ── 6. 打分引擎 (Enhanced Scoring Engine) ───────────────────────────────────
@dataclass
class Factor:
    condition: Callable[[dict], bool]
    points: int
    weight: float = 1.0
    template: str = ""

def apply_scoring(data: dict) -> tuple[int, str, str]:
    adx = data['adx']
    tw, rw = (1.4, 0.7) if adx > 25 else (0.8, 1.4) if adx < 15 else (1.0, 1.0)
    meta = f"🧭 趋势雷达: {'当前处于主升浪' if adx > 25 else '当前处于底部反转期' if adx < 15 else '震荡蓄势中'}"

    factors = [
        # --- 核心安全边际 (小白防呆白话文版) ---
        Factor(lambda d: d['price_pct'] < 0.25, 15, rw, "🟢 【绝对低位】股价在近一年极低位置，现在买入相当于抄底，长线拿着不慌"),
        Factor(lambda d: 0.25 <= d['price_pct'] <= 0.45, 10, 1.0, "🟢 【相对低位】刚刚从底部爬起来，输时间不输钱"),
        Factor(lambda d: d['pe'] > 0 and d['pe'] < 30, 8, 1.0, "🛡️ 【业绩护体】公司确实在赚钱(市盈率健康)，不是炒概念的空气股"),
        Factor(lambda d: d['pb'] > 0 and d['pb'] < 1.5, 10, 1.0, "🧱 【跌无可跌】买这只股的价格快接近它变卖资产的价格了，安全垫极厚"), # 【新增】破净/低估值护盾
        
        # --- 结构与趋势 ---
        Factor(lambda d: d['bull_rank'], 10, 1.0, "📈 【顺势而为】均线多头排列，跟着大部队的方向走"),
        Factor(lambda d: d['ma_convergence'], 12, 1.0, "🌪️ 【面临变盘】短期和长期成本均线交织在一起，随时准备向上爆发"), 
        
        # --- 资金与动能 ---
        Factor(lambda d: d['has_zt'], 10, 1.0, "🔥 【股性活跃】这只股历史上容易涨停，不会一潭死水"),
        Factor(lambda d: d['vol_ratio'] >= 1.8, 10, 1.0, "🔵 【放量确认】今天成交量明显放大，说明有大资金在干活"),
        Factor(lambda d: d['red_days'] >= 3, 8, 1.0, "🔴 【稳步推升】连续几天都在涨，主力在偷偷温和建仓"),
        
        # --- 筹码与量能 ---
        Factor(lambda d: d['has_chip_break'], 15, tw, "🏔️ 【抛压真空】上方的套牢盘已被消化，向上拉升阻力很小"),
        Factor(lambda d: d['vcp_amp'] < 0.12, 10, 1.0, "🟣 【蓄势待发】近期上下波动越来越小，主力控盘极稳，即将出方向"),
        Factor(lambda d: d['extreme_shrink_vol'], 10, 1.0, "🧊 【没人砸盘】爆发前夕成交量萎缩到极致，说明里面的散户该卖的都卖了"), 
        Factor(lambda d: d['has_obv_break'], 10, tw, "💸 【真金白银】通过模型测算，近期有真实的机构资金在持续流入"),
        Factor(lambda d: d['has_pullback'], 15, 1.0, "🪃 【上车机会】主力拉升后的缩量洗盘，正好给我们上车的机会"),
        Factor(lambda d: d['lower_shadow_ratio'] > 0.03, 8, 1.0, "📌 【强力护盘】盘中跌下去被大资金迅速买回，下方支撑极强"), 
        Factor(lambda d: d['has_fund_inflow'], 12, tw, f"💰 【大单抢筹】监控到今日主力净流入金额较大"),
        Factor(lambda d: d.get('sector_ok', False), 12, 1.0, "🌟 【热点风口】所属板块目前也是全市场最赚钱的板块之一"),
        
        # --- 【防接盘减分项】 ---
        Factor(lambda d: d['has_consecutive_zt'] and d['price_pct'] < 0.40, 10, 1.0, "🔥🔥 【低位连板】刚刚启动的龙头，辨识度高且相对安全"),
        Factor(lambda d: d['has_consecutive_zt'] and d['price_pct'] >= 0.40, -20, 1.0, "⚠️ 【高位接盘预警】股价已经被炒高，千万别追，容易站岗！"),
        Factor(lambda d: d['upper_shadow_pct'] > 18, -15, 1.0, "⚠️ 【诱多预警】冲高后大幅回落，上方抛压极重，小心主力骗炮"),
        Factor(lambda d: d['dist_ma20'] > 18, -20, 1.0, "🚫 【追高预警】目前涨得太急，离均线太远，随时可能大幅回调"),
    ]

    score, reasons = 40, [meta] if meta else []
    data['price_pct_pct'] = data['price_pct'] * 100
    data['vcp_amp_pct'] = data['vcp_amp'] * 100
    
    for f in factors:
        if f.condition(data):
            score += int(f.points * f.weight)
            reasons.append(f.template.format(**data))

    score = max(0, min(score, 100))
    level = '⭐⭐⭐⭐⭐ [S级·安心底仓]' if score >= 85 else '⭐⭐⭐⭐ [A级·稳健标的]' if score >= 75 else '⭐⭐⭐ [B级及格]'
    return score, level, '\n'.join(reasons)

def get_signals() -> tuple[list[Signal], set, int, str]:
    now = datetime.now(TZ_BJS)
    run_mode = os.environ.get('RUN_MODE', 'normal')
    
    log.info('🚀 防呆长线安全级·量化引擎启动...')
    if not IS_MANUAL and not is_trading_time(now): return [], set(), 0, ""

    c_conf, pushed = Config(), load_pushed_state()

    if run_mode == 'market_only':
        log.info("🤖 进入 [纯大盘体检模式] ，短路全量个股运算，开始多维指数提取...")
        market_msg = extract_pure_market_context()
        return [], pushed, 0, market_msg
    
    ex1 = ThreadPoolExecutor(max_workers=2)
    f_spot = ex1.submit(fetch_spot)
    f_flow = ex1.submit(get_fund_flow_map)
    df_raw = pd.DataFrame()
    flow_map = {}
    try:
        df_raw = f_spot.result(timeout=40) 
        flow_map = f_flow.result(timeout=20)
    except FuturesTimeoutError:
        log.error("❌ 行情或资金流请求严重超时(>40s)，触发系统防卡死降级处理。")
    except Exception as e:
        log.error(f"❌ 横截面数据获取失败: {e}")
    finally:
        ex1.shutdown(wait=False, cancel_futures=True) 

    df_clean, m_ok, m_msg, idx_ret = extract_market_context(df_raw, c_conf)
    log.info(f"📈 扫描全市场宏观:\n{m_msg}")
    
    if df_clean.empty:
        return [], pushed, 0, m_msg

    # 【新增白名单与物理隔离】：过滤科创板(688)和北交所(8/4/9)，过滤ST，过滤一字板陷阱
    mask = (df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & \
           (df_clean[C.S_MCAP].between(c_conf.MIN_CAP, c_conf.MAX_CAP)) & \
           (df_clean[C.S_TURN] >= c_conf.MIN_TURNOVER) & \
           (df_clean[C.S_PE] > c_conf.MIN_PE) & (df_clean[C.S_PE] <= c_conf.MAX_PE) & \
           (df_clean[C.S_PB] > 0) & (df_clean[C.S_PB] <= 10.0) & \
           (~df_clean[C.S_CODE].str.startswith(('688', '8', '4', '9'))) & \
           (df_clean[C.S_HIGH] > df_clean[C.S_LOW]) # 拒绝一字板
    
    if C.S_VR in df_clean.columns:
        t_val = now.hour * 100 + now.minute
        vr_max = 9.5 if t_val <= 1030 else 6.0
        mask &= df_clean[C.S_VR].between(c_conf.MIN_VOL_RATIO, vr_max)

    pool = df_clean[mask].pipe(lambda d: d[~d[C.S_CODE].isin(pushed)]).copy()
    if pool.empty: return [], pushed, 0, m_msg

    sec_strengths = fetch_sector_strength()
    candidate_data = []
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=450)).strftime('%Y%m%d')
    
    ex2 = ThreadPoolExecutor(max_workers=10)
    futures = {ex2.submit(fetch_hist, r[C.S_CODE], start_s, end_s): r for _, r in pool.iterrows()}
    
    try:
        for f in as_completed(futures, timeout=90):
            row = futures[f]
            try:
                hist = f.result(timeout=5)
                result = process_stock(row, hist, now, m_ok, idx_ret, flow_map.get(row[C.S_CODE], 0.0))
                if result:
                    data, stop, risk = result
                    try: sector = fetch_stock_sector(row[C.S_CODE])
                    except: sector = ""
                    s_pct = sec_strengths.get(sector, 0.0)
                    data.update({'sector': sector, 'sector_pct': s_pct, 'sector_ok': (s_pct > 1.0)})
                    
                    score, level, reas = apply_scoring(data)
                    if score >= 65: # 提升准入门槛，只推真正的底仓极品
                        pos_pos = (30 if score >= 85 else 20 if score >= 75 else 10)
                        if not m_ok: pos_pos //= 2
                        
                        candidate_data.append(Signal(
                            code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
                            pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
                            position_advice=f"⚖️ 建议仓位 {pos_pos}% (最大下行风险被锁死在 {risk:.1f}%)",
                            trigger_time=now.strftime('%H:%M'), reasons=reas,
                            stop_loss=round(stop, 2), target1=round(row[C.S_PRICE]*(1+risk*2.5/100), 2),
                            target2=round(data['max_1y'], 2), ma10=round(data['ma10_val'], 2),
                            sector=sector, sector_pct=s_pct
                        ))
                        pushed.add(row[C.S_CODE])
            except FuturesTimeoutError:
                log.debug(f"⚠️ 跳过异常标的(请求超时): {row[C.S_CODE]}")
            except Exception as e:
                pass
    except FuturesTimeoutError:
        log.warning("⏳ 总体强制熔断保护。")
    finally:
        ex2.shutdown(wait=False, cancel_futures=True)

    candidate_data.sort(key=lambda x: x.score, reverse=True)
    return candidate_data, pushed, len(pool), m_msg


# ── 9. 通知推送逻辑 ───────────────────────────────────────────────────────────
def send_dingtalk(signals: list[Signal], total: int, market_msg: str) -> None:
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook: return
    
    now_str = datetime.now(TZ_BJS).strftime('%Y-%m-%d %H:%M')
    run_mode = os.environ.get('RUN_MODE', 'normal')
    
    header = f"🤖 AI 量化安全底仓策略 {now_str}\n"
    if run_mode == 'market_only':
        header = f"🤖 AI 盘中大盘深度体检 {now_str}\n"
        
    if market_msg:
        header += f"\n{market_msg}\n\n"

    if run_mode == 'market_only':
        content = header + "✅ 大盘分析播报完毕，本次任务短路了全量个股运算。"
    elif not signals:
        if not PUSH_EMPTY: return
        content = f"{header}✅ 安全雷达体检 {total} 只异动标的，未发现完全符合安全边际的标的，系统防守中。"
    else:
        parts = []
        for s in signals:
            sec_info = f"🏷️ 所属板块：{s.sector} (板块今日 {s.sector_pct:+.2f}%)\n" if s.sector else ""
            # 给创业板增加高波动提示
            warn_msg = "⚡ 注意：该股为创业板(20%涨跌幅)，新手请务必缩减单笔买入金额！\n" if str(s.code).startswith('300') else ""
            
            parts.append(
                f"【{s.name} ({s.code})】\n"
                f"{sec_info}{warn_msg}"
                f"📊 综合评分：{s.score}分 {s.level}\n"
                f"💰 当前价格：¥{s.price} ({s.pct_chg})\n"
                f"--- 💡 为什么机器选出它？ ---\n{s.reasons}\n"
                f"--- 🛡️ 新手保姆级操作纪律 ---\n"
                f"{s.position_advice}\n"
                f"🎯 止盈纪律：不要贪心，如果赚钱后股价跌破 ¥{s.ma10}(10日均线) 请立刻卖出锁定利润。\n"
                f"🛑 认错纪律：如果买入后不幸跌破 ¥{s.stop_loss}，说明逻辑失效，请无条件割肉离场！\n"
                f"🔗 一键看盘：https://quote.eastmoney.com/unify/r/0.{s.code}\n"
            )
        content = header + "\n".join(parts)

    try:
        requests.post(webhook, json={'msgtype': 'text', 'text': {'content': content}}, timeout=10)
        log.info(f"✅ 推送成功 ({len(signals)} 只个股/大盘)")
    except Exception as e:
        log.error(f"❌ 推送失败: {e}")

if __name__ == '__main__':
    try:
        sigs, pushed, total, m_msg = get_signals()
        send_dingtalk(sigs, total, m_msg)
        if sigs: save_pushed_state(pushed)
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
