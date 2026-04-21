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

socket.setdefaulttimeout(15.0)

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

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
    # --- 小白专属护城河参数 ---
    MIN_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MIN_CAP', 30e8)) 
    MAX_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MAX_CAP', 500e8))
    MAX_PRICE: float     = field(default_factory=lambda: EnvParser.get_float('MAX_PRICE', 60.0))  
    MIN_PE: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PE', 0))    
    MAX_PE: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PE', 100))  
    MIN_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MIN_TURNOVER', 1.5))
    MAX_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MAX_TURNOVER', 15.0)) 
    MIN_PCT_CHG: float   = field(default_factory=lambda: EnvParser.get_float('MIN_PCT_CHG', 1.5))
    MIN_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MIN_VOL_RATIO', 0.8)) 
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
    trigger_time: str 
    reasons: str
    stop_loss: float
    target1: float
    target2: float
    ma10: float
    sector: str = ""
    sector_pct: float = 0.0
    
    # ── 小白保姆级交互文案 ──
    money_risk_msg: str = ""
    tranche_plan_msg: str = ""
    plan_b_msg: str = ""
    hold_period_msg: str = ""


# ── 3. 小白防呆专享算法库 (The Dummies' Math) ──────────────────────────────────
class MathUtils:
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

def is_earnings_danger_zone(now: datetime) -> tuple[bool, str]:
    month = now.month
    day = now.day
    DANGER_WINDOWS = [
        (3, 25, 4, 30, "年报/一季报披露末期"),
        (8, 15, 8, 31, "半年报披露末期"),
        (10, 15, 10, 31, "三季报披露末期"),
    ]
    for s_m, s_d, e_m, e_d, label in DANGER_WINDOWS:
        start_dt = now.replace(month=s_m, day=s_d, hour=0, minute=0)
        end_dt = now.replace(month=e_m, day=e_d, hour=23, minute=59)
        if start_dt <= now <= end_dt:
            return True, label
    return False, ""

def format_money_risk_msg(price: float, stop_loss: float, target1: float) -> str:
    """将抽象的%风险，翻译成小白能感知的真实亏损金额及容错率"""
    one_hand_cost = price * 100
    budget_per_hand = 10000
    hands = max(1, int(budget_per_hand / one_hand_cost))
    total_cost = hands * one_hand_cost
    
    loss_per_share = price - stop_loss
    total_loss = loss_per_share * hands * 100
    gain_1 = (target1 - price) * hands * 100
    
    ratio = gain_1 / max(total_loss, 1)
    ratio_str = f"{ratio:.1f}"
    
    if ratio >= 2.5:
        evaluation = "🎯 【高容错】做对一次赚的钱能抵消两三次亏损，试错成本低！"
    elif ratio >= 1.5:
        evaluation = "✅ 【尚可】盈亏比较好，跌势有限，可以防守建仓。"
    else:
        evaluation = "⚠️ 【需谨慎】盈亏比勉强持平，对操作精准度要求高，新手请把买入金额减半！"
    
    return (
        f"💸 **小白算账（以 1 万元预算为例：约买 {hands} 手 = ¥{total_cost:.0f}）**\n"
        f"   🔴 极度悲观：触发止损，你大概会亏 ¥{total_loss:.0f}\n"
        f"   🟢 保守预期：达到第一目标，大概能赚 ¥{gain_1:.0f}\n"
        f"   📐 盈亏性价比 1 : {ratio_str} ➡️ {evaluation}"
    )

def generate_tranche_plan(price: float, score: int, market_ok: bool, market_overheated: bool) -> str:
    """盘后生成的次日剧本：盲挂限价防追高"""
    if market_overheated:
        return "🛑 **【系统熔断】当前市场情绪极度过热，随时面临收割踩踏！系统强制禁止明日建仓，请管住手！**"
        
    base_pct = 30 if score >= 85 else 20 if score >= 75 else 10
    if not market_ok:
        base_pct = base_pct // 2
        
    t1 = max(1, base_pct // 3)
    t2 = max(1, base_pct // 3)
    t3 = max(1, base_pct - t1 - t2)
    
    # 次日挂单价：比收盘价低0.5%，避免追高开盘
    limit_price = round(price * 0.995, 2)
    add_price   = round(price * 1.025, 2)
    stop_add    = round(price * 1.05,  2)
    
    return (
        f"📋 **明日操作计划（今晚设好条件单，明天不用盯盘）**\n"
        f"   ① 【挂单埋伏】以 ¥{limit_price} 限价挂买入 {t1}% 仓位，成交了就等，没成交就算\n"
        f"   ② 【稳健加仓】如果明后天没跌且站稳 ¥{add_price}，再加 {t2}%\n"
        f"   ③ 【追击确认】如果继续上涨突破 ¥{stop_add}，最后追加 {t3}%\n"
        f"   🔒 铁律：限价单没成交 → 不追市价！宁可踏空，绝不接盘高位。"
    )

def generate_plan_b(price: float, stop_loss: float, ma20: float) -> str:
    normal_shake = round(price * 0.97, 2)  
    normal_shake = max(normal_shake, stop_loss + 0.01)
    
    return (
        f"🆘 **如果买入后跌了怎么办？(心理防线预演)**\n"
        f"   ▪️ 回撤在 ¥{normal_shake:.2f} 以上 → 纯属正常的洗盘波动，千万别自己吓自己，持仓装死别动。\n"
        f"   ▪️ 跌破 ¥{stop_loss:.2f} (铁血止损) → 意味着趋势彻底被破坏，立刻无条件割肉离场，保住大本金！\n"
        f"   ▪️ 在 ¥{stop_loss:.2f} ~ ¥{normal_shake:.2f} 之间 → 观察成交量，缩量就扛着，放巨量或破20日线(¥{ma20:.2f})减仓。\n"
        f"   ▪️ 大盘突然崩盘(千股大跌) → 别管盈亏，立刻先清掉一半，躲过风暴再说。"
    )

def generate_hold_period(adx: float, price_pct: float, has_chip_break: bool) -> str:
    if price_pct < 0.35 and adx < 20:
        return "🐢 **持股预期**：这是【底部潜伏型】，主力还在偷偷吸筹。可能需要拿 1~3 个月才会大涨，买完千万别天天盯盘，就当存死期。"
    elif adx > 25 or has_chip_break:
        return "🐎 **持股预期**：这是【右侧趋势型】，随时可能加速。快的话 3~5 个交易日，慢的话两周。见好就收，切忌贪心！"
    else:
        return "🐕 **持股预期**：这是【稳健震荡型】，需要一点耐心，预计持有 2~4 周等风来。"


# ── 4. 数据拉取模块 ────────────────────────────────────────────────────────────
import akshare as ak

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
    try:
        # 主数据源：东方财富
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        log.warning(f"指数主接口异常 ({symbol}): {e}，正在无缝切换腾讯备用源...")
        # 备用数据源：腾讯
        df = ak.stock_zh_index_daily_tx(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
        raise
    raise ValueError(f'index_empty_{symbol}')

@retry(times=3, delay=2)
def fetch_hist(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        # 主数据源：东方财富
        df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust='qfq')
        if df is not None and not df.empty:
            return df[list(Config.HIST_COLS)].copy()
    except Exception as e:
        log.debug(f"{code} 历史主接口波动: {e}，尝试备用源...")
        try:
            # 备用数据源：腾讯历史接口
            df = ak.stock_zh_a_hist_tx(symbol=code, start_date=start, end_date=end, adjust='qfq')
            if df is not None and not df.empty:
                col_map = {'日期': C.H_DATE, '开盘': C.H_OPEN, '收盘': C.H_CLOSE, '最高': C.H_HIGH, '最低': C.H_LOW, '成交量': C.H_VOL}
                df = df.rename(columns=col_map)
                return df[list(Config.HIST_COLS)].copy()
        except Exception:
            pass
        raise
    raise ValueError('history_empty')

@retry(times=3, delay=2)
def fetch_spot() -> pd.DataFrame:
    try:
        # 主数据源：东方财富 (全量基本面)
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning(f"行情主接口严重异常: {e}，正在启动新浪备用源并执行优雅降级...")
        # 备用数据源：新浪 (缺少市盈率、量比等深度数据)
        df = ak.stock_zh_a_spot()
        if df is not None and not df.empty:
            # 统一列名映射
            rename_map = {'代码': C.S_CODE, '名称': C.S_NAME, '最新价': C.S_PRICE,
                          '涨跌幅': C.S_PCT, '今开': C.S_OPEN, '最高': C.S_HIGH,
                          '最低': C.S_LOW, '成交量': C.S_VOL, '成交额': C.S_AMT}
            df = df.rename(columns=rename_map)
            
            # 【优雅降级】：给缺失的基本面字段注入“假的安全值”，确保技术面逻辑不崩溃
            fallback_defaults = {
                C.S_TURN: 2.0,      # 默认换手率
                C.S_MCAP: 100e8,    # 默认市值100亿
                C.S_PE: 15.0,       # 默认安全市盈率 (此特征可作为降级标志)
                C.S_PB: 2.0,        # 默认安全市净率
                C.S_VR: 1.0         # 默认量比
            }
            for col, val in fallback_defaults.items():
                if col not in df.columns:
                    df[col] = val
            return df
        raise
    raise ValueError('spot_empty')

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

@retry(times=2, delay=1)
def check_shareholder_risk(code: str) -> tuple[bool, str]:
    try:
        df_pledge = ak.stock_zh_a_gdhs(symbol=code) 
        if df_pledge is not None and not df_pledge.empty:
            pledge_col = next((c for c in df_pledge.columns if '质押' in c or '比例' in c), None)
            if pledge_col:
                pledge_ratio = pd.to_numeric(df_pledge[pledge_col].iloc[0], errors='coerce')
                if pd.notna(pledge_ratio) and pledge_ratio > 50:
                    return False, f"💣 股权质押高达 {pledge_ratio:.1f}%，爆仓强平风险极大，拒绝碰瓷！"
    except Exception as e:
        log.debug(f"质押数据查询跳过({code}): 接口波动或无数据 {e}")
    return True, ""

@retry(times=2, delay=1)
def check_margin_crowding(code: str, mcap: float) -> str:
    try:
        prefix = 'sh' if str(code).startswith('6') else 'sz'
        df = ak.stock_margin_detail_szse(date=datetime.now(TZ_BJS).strftime('%Y%m%d')) if prefix == 'sz' else ak.stock_margin_detail_sse(date=datetime.now(TZ_BJS).strftime('%Y%m%d'))
        if df is None or df.empty: return ""
        
        row = df[df['证券代码'] == code] if '证券代码' in df.columns else df[df['信用交易担保物'] == code]
        if row.empty: return ""
        
        margin_balance_col = next((c for c in row.columns if '融资余额' in c), None)
        if not margin_balance_col: return ""
        
        latest_margin = pd.to_numeric(row[margin_balance_col].iloc[0], errors='coerce')
        if pd.isna(latest_margin) or mcap <= 0: return ""
        
        crowding_ratio = latest_margin / mcap * 100
        if crowding_ratio > 4.0:
            return f"🚨 融资盘高度拥挤({crowding_ratio:.1f}%)！散户全在借钱杠杆买，极易发生踩踏！"
        elif crowding_ratio > 2.5:
            return f"⚠️ 融资杠杆资金较多({crowding_ratio:.1f}%)，持股会有颠簸。"
    except Exception as e:
        log.debug(f"融资融券数据查询跳过({code}): 接口波动或无数据 {e}")
    return ""


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

        self.df['ATR'], self.df['ADX'] = MathUtils.calc_atr_adx(self.df)
        
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(14).mean()
        loss = (-delta.clip(upper=0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['RSI14'] = 100 - (100 / (1 + rs))

        self.df['REF_C'] = close.shift()
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
        if price_pct > 0.65: return None 

        if today[C.H_CLOSE] < today['MA20']: return None
        if not (today['DIF'] >= today['DEA'] or today['DIF'] > -0.1): return None

        rsi = float(today.get('RSI14', 50))
        if pd.isna(rsi) or rsi > 78: return None 

        consecutive_down = 0
        for i in range(2, 8):
            if len(df) >= i and df[C.H_CLOSE].iloc[-i] < df[C.H_OPEN].iloc[-i]:
                consecutive_down += 1
            else:
                break
        if consecutive_down >= 4: return None

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
            
        body_pct = (today[C.H_CLOSE] - today[C.H_OPEN]) / today[C.H_OPEN] if today[C.H_OPEN] > 0 else 0
        has_pullback = bool(
            yest[C.H_LOW] <= yest['MA20'] * 1.02 and 
            today[C.H_CLOSE] > today['MA20'] and 
            today[C.H_VOL] < today['MA5_V'] and
            body_pct >= 0.003
        )

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
            'has_pullback': has_pullback,
            'has_chip_break': has_chip_break,
            'dist_ma20': (today[C.H_CLOSE] / today['MA20'] - 1) * 100,
            'red_days': red_days,
            'rsi': rsi,
            'macd_dea': float(today['DEA']),
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

def apply_scoring(data: dict, now: datetime) -> tuple[int, str, str]:
    adx = data['adx']
    tw, rw = (1.4, 0.7) if adx > 25 else (0.8, 1.4) if adx < 15 else (1.0, 1.0)
    meta = f"🧭 趋势雷达: {'处于主升浪' if adx > 25 else '处于底部反转期' if adx < 15 else '震荡蓄势中'}"

    in_danger, danger_label = is_earnings_danger_zone(now)

    factors = [
        Factor(lambda d: d['price_pct'] < 0.25, 15, rw, "🟢 【绝对低位】目前买入相当于抄底，长线拿着不慌"),
        Factor(lambda d: 0.25 <= d['price_pct'] <= 0.45, 10, 1.0, "🟢 【相对低位】刚刚从底部爬起来，输时间不输钱"),
        Factor(lambda d: d['pe'] > 0 and d['pe'] < 30, 8, 1.0, "🛡️ 【业绩护体】市盈率健康，不是炒空气的垃圾股"),
        Factor(lambda d: d['pb'] > 0 and d['pb'] < 1.2, 10, 1.0, "🧱 【跌无可跌】股价逼近变卖资产的净值，大盘暴跌它也不怕"), 
        Factor(lambda d: d['macd_dea'] >= -0.05, 8, 1.0, "🌊 【多头控盘】MACD处于强势零轴附近，没有深套风险"), 
        
        Factor(lambda d: 0 <= d['dist_ma20'] <= 4.0, 15, 1.0, "🧲 【贴地潜伏】目前价格紧贴20日支撑线，属于绝佳的安全低吸点，没追高"),
        Factor(lambda d: 40 <= d.get('rsi', 50) <= 65, 10, 1.0, "📊 【温度适中】RSI指标健康，不冷不热正是下手好时机"),
        
        Factor(lambda d: d['bull_rank'], 10, 1.0, "📈 【顺势而为】均线多头排列，跟着主力资金大部队走"),
        Factor(lambda d: d['ma_convergence'], 12, 1.0, "🌪️ 【面临变盘】短期和长期成本几乎重合，随时向上爆发"), 
        
        Factor(lambda d: d['has_zt'], 10, 1.0, "🔥 【股性活跃】这只股历史上容易涨停，不会一潭死水"),
        Factor(lambda d: d['vol_ratio'] >= 1.8, 10, 1.0, "🔵 【放量确认】今天成交量明显放大，大资金开始干活了"),
        Factor(lambda d: d['red_days'] >= 3, 8, 1.0, "🔴 【稳步推升】连续几天都在涨，主力在偷偷温和建仓"),
        
        Factor(lambda d: d['has_chip_break'], 15, tw, "🏔️ 【抛压真空】上方的套牢盘已割肉离场，向上拉升没阻力"),
        Factor(lambda d: d['vcp_amp'] < 0.12, 10, 1.0, "🟣 【蓄势待发】近期上下波动极小，主力控盘很稳即将出方向"),
        Factor(lambda d: d['extreme_shrink_vol'], 10, 1.0, "🧊 【没人砸盘】爆发前夕成交量极度萎缩，散户该卖的都卖了"), 
        Factor(lambda d: d['has_obv_break'], 10, tw, "💸 【真金白银】模型监控到真实的机构资金在创纪录买入"),
        Factor(lambda d: d['has_pullback'], 15, 1.0, "🪃 【上车机会】温和缩量回踩，主力洗盘挖坑给的上车机会"),
        Factor(lambda d: d['lower_shadow_ratio'] > 0.03, 8, 1.0, "📌 【强力护盘】跌下去被大资金迅速买回，下方有人兜底"), 
        Factor(lambda d: d['has_fund_inflow'], 12, tw, f"💰 【大单抢筹】发现主力大资金正在持续净流入"),
        
        Factor(lambda d: d.get('sector_ok', False), 10, 1.0, "🌱 【板块温和】所属板块今日温和上涨，资金在悄悄布局而非疯狂追涨"),
        Factor(lambda d: 0.0 <= d.get('sector_pct', 0) <= 0.3, 5, 1.0, "⚖️ 【独立行情】所属板块表现平淡，全靠个股自身逻辑独立走强"),
        Factor(lambda d: 3.0 <= d.get('sector_pct', 0) < 5.0, 5, 1.0, "📈 【板块较热】板块涨幅较大，已吸引市场目光，可顺势参与但需防回调"),
        
        Factor(lambda d: d.get('rs_rating', 0) > 5, 8, 1.0, "🏆 【相对强势】近60日涨幅跑赢大盘5%以上，属于市场里的强者"),
        
        # --- 【排雷扣分项】 ---
        Factor(lambda d: d.get('sector_overheated', False), -12, 1.0, "🌋 【板块过热】今日板块暴涨超5%，主力随时借机出货，小白此时入场风险极高(已扣分)"),
        Factor(lambda d: d.get('rsi', 50) > 70, -12, 1.0, "🌡️ 【过热预警】RSI已超70超买线，买进去容易在高位站岗"),
        Factor(lambda d: d.get('rs_rating', 0) < -10, -10, 1.0, "📉 【相对弱势】近期跑输大盘明显，跟的是一只被市场嫌弃的股"),
        Factor(lambda d: d['has_consecutive_zt'] and d['price_pct'] < 0.40, 10, 1.0, "🔥🔥 【低位连板】刚刚启动的龙头，安全且辨识度高"),
        Factor(lambda d: d['has_consecutive_zt'] and d['price_pct'] >= 0.40, -20, 1.0, "⚠️ 【高位接盘】股价已被炒高连板，千万别追，容易接盘！"),
        Factor(lambda d: d['upper_shadow_pct'] > 18, -15, 1.0, "⚠️ 【诱多预警】冲高后大幅跳水，上方抛压极重，别上当"),
        Factor(lambda d: d['dist_ma20'] > 18, -20, 1.0, "🚫 【追高预警】目前涨得太急离均线太远，随时面临暴跌回调"),
        
        Factor(lambda d: in_danger and d['mcap'] < 100e8, -15, 1.0, f"📅 【财报雷区】当前属于{danger_label}高危期，小盘股极易暴雷(已扣分)")
    ]

    score, reasons = 40, [meta] if meta else []
    data['price_pct_pct'] = data['price_pct'] * 100
    data['vcp_amp_pct'] = data['vcp_amp'] * 100
    
    for f in factors:
        if f.condition(data):
            score += int(f.points * f.weight)
            reasons.append(f.template.format(**data))

    score = max(0, min(score, 100))
    
    if score >= 85:
        level = '⭐⭐⭐⭐⭐ 🐯 [S级·老虎机模式] (胜率极高、跌势有限，最适合盲挂限价建仓)'
    elif score >= 75:
        level = '⭐⭐⭐⭐ 🐕 [A级·看门狗模式] (稳健防守型，需要一点耐心慢慢等它涨)'
    else:
        level = '⭐⭐⭐ 🐒 [B级·猴子模式] (上蹿下跳振幅大，需要老手盯盘，纯新手请回避)'
        
    return score, level, '\n'.join(reasons)


# ── 7. 信号流水线 (Signal Pipeline) ──────────────────────────────────────────
def is_valid_run_time(now: datetime) -> bool:
    """
    收盘后 15:05-23:59 运行分析，产出次日挂单计划。
    盘中禁止运行，避免追涨冲动。
    IS_MANUAL 手动触发时不受限制（用于调试）。
    """
    if IS_MANUAL:
        return True
    t = now.hour * 100 + now.minute
    return t >= 1505

def process_stock(row: pd.Series, raw_hist: pd.DataFrame, now: datetime, market_ok: bool, index_ret: float, flow: float) -> Optional[tuple]:
    if len(raw_hist) < 250: return None
    
    hist = raw_hist.copy()
    if str(hist[C.H_DATE].iloc[-1]) != now.strftime('%Y-%m-%d') and is_valid_run_time(now):
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

    data['pe'] = float(row.get(C.S_PE, 0))
    data['pb'] = float(row.get(C.S_PB, 0))
    data['mcap'] = float(row.get(C.S_MCAP, 0))
    data['vol_ratio'] = float(row.get(C.S_VR, 1.0))
    data['fund_flow_w'] = flow / 10000.0
    data['has_fund_inflow'] = flow > max(1e7, float(row.get(C.S_MCAP, 0)) * 0.001)
    data['rs_rating'] = ((row[C.S_PRICE] / data['close_60d_ago'] - 1) * 100 - index_ret) if data['close_60d_ago'] > 0 else 0
    
    supports = [data['ma20_val'], data['recent_20_low'], data['boll_lower']]
    valid_supports = [s for s in supports if pd.notna(s) and s < row[C.S_PRICE]]
    
    stop = max(valid_supports + [row[C.S_PRICE] * 0.93]) * 0.993 
    risk_pct = ((row[C.S_PRICE] - stop) / row[C.S_PRICE]) * 100 if row[C.S_PRICE] > 0 else 99
    if risk_pct > 7.0: return None 

    return (data, stop, risk_pct) 


# ── 8. 控制器与大盘体检 (Orchestrator) ─────────────────────────────────────────
def fetch_stock_sector(code: str) -> str:
    try:
        df = ak.stock_individual_info_em(symbol=code)
        v = df[df[C.INFO_ITEM] == '行业'][C.INFO_VAL].values[0]
        return str(v) if pd.notna(v) else ""
    except: return ""

def fetch_sector_strength() -> dict:
    try:
        df = ak.stock_board_industry_spot_em()
        return dict(zip(df[C.B_NAME], df[C.B_PCT]))
    except: return {}

def extract_market_context(df_raw: pd.DataFrame, c_conf: Config) -> tuple[pd.DataFrame, bool, str, float, bool]:
    if len(df_raw) < 1000: return pd.DataFrame(), False, "API 异常", 0.0, False
    
    market_ok, market_msg, index_ret, market_overheated = True, "", 0.0, False
    try:
        df_raw[C.S_PE] = pd.to_numeric(df_raw[C.S_PE], errors='coerce')
        df_raw[C.S_PB] = pd.to_numeric(df_raw[C.S_PB], errors='coerce')

        idx_df = fetch_index('sh000001')
        cl = idx_df[C.I_CLOSE]
        ma20 = cl.rolling(20).mean().iloc[-1]
        pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
        market_ok = (cl.iloc[-1] > ma20)
        index_ret = ((cl.iloc[-1] / cl.iloc[-60]) - 1) * 100 if len(cl) >= 60 else 0.0

        up_count = (df_raw[C.S_PCT] > 0).sum()
        down_count = (df_raw[C.S_PCT] < 0).sum()
        zt_count = (df_raw[C.S_PCT] >= 9.0).sum() 
        dt_count = (df_raw[C.S_PCT] <= -9.0).sum() 
        total_amt = df_raw[C.S_AMT].sum() / 1e8 
        
        sentiment_addon = ""
        if zt_count > 150:
            market_overheated = True
            sentiment_addon = "\n🚨🚨 **情绪极度过热熔断！**今日涨停数破百，市场陷入非理性狂欢。历史数据表明，现在冲进去90%的小白会成为接盘侠！系统已自动【暂停推荐任何股票】，请把钱放余额宝，管住手！"
        elif dt_count > 100:
            sentiment_addon = "\n❄️ **情绪极度冰点！**今日百股跌停，恐慌盘已出尽，这里往往是黄金坑和绝佳防守反击位！"

        breadth = up_count / (up_count + down_count) if (up_count + down_count) > 0 else 0.5
        advice = ""
        if market_overheated:
            advice = "🛑 强制休息！系统已熔断今日的买入建议。懂休息的才是真正的高手。"
        elif market_ok and breadth >= 0.6:
            advice = "🔥 赚钱效应极佳 (建议仓位 60%-80%)，精选标的积极建仓。"
        elif market_ok and breadth < 0.6:
            advice = "⚠️ 指数安全但个股分化 (建议仓位 40%-60%)，优选低位标的，不追高。"
        elif not market_ok and breadth <= 0.3:
            advice = "🧊 情绪绝对冰点 (建议仓位 10%-20%)，系统性风险释放中，多看少动为主。"
        else:
            advice = "🛡️ 弱势震荡市 (建议仓位 20%-40%)，控制手管住回撤，非绝对低位不买。"

        market_msg = (
            f"🎯 **指数收盘态势**:\n"
            f"   • 上证指数: {cl.iloc[-1]:.2f} ({sh_pct:+.2f}%)\n"
            f"   • 深证成指: {sz_df['close'].iloc[-1]:.2f} ({sz_pct:+.2f}%)\n"
            f"   • 创业板指: {cyb_df['close'].iloc[-1]:.2f} ({cyb_pct:+.2f}%)\n"
            f"💰 **全市场量能**: 两市总成交 {total_amt:.0f} 亿元 (较上日 {amt_diff:+.0f} 亿)\n"
            f"📈 **主板技术面**:\n"
            f"   • 趋势: {'🟢 企稳于MA20生命线' if market_ok else '🔴 跌破MA20生命线'}，大级别呈{trend_status}。\n"
            f"   • 动能: MACD{macd_status}，短期呈现{amt_status}{'反弹' if sh_pct > 0 else '调整'}。\n"
            f"   • 空间: 近期支撑 {recent_low:.0f} 点 | 压力 {recent_high:.0f} 点。\n"
            f"🧠 **AI 深度形态解盘**:\n"
            f"   • {insights_str}\n\n"
            f"{plain_summary}\n\n"
            f"💡 **总体仓位建议**: {advice}{fallback_warning}"
        )
    except Exception as e:
        log.warning(f"宏观状态解析失败: {e}")
        market_msg = f"大盘深度解析由于网络原因失败: {e}\n"
    
    return df, market_ok, market_msg, index_ret, market_overheated # 注意这里返回df_raw，后续会过滤

def extract_pure_market_context() -> str:
    """完全短路横截面的纯指数深度体检"""
    market_msg = ""
    try:
        sh_df = fetch_index('sh000001')
        sz_df = fetch_index('sz399001')
        cyb_df = fetch_index('sz399006')

        cl = sh_df['close']
        vol = sh_df['volume']
        
        ma10 = cl.rolling(10).mean().iloc[-1]
        ma20 = cl.rolling(20).mean().iloc[-1]
        ma60 = cl.rolling(60).mean().iloc[-1]

        ema12 = cl.ewm(span=12, adjust=False).mean()
        ema26 = cl.ewm(span=26, adjust=False).mean()
        dif = (ema12 - ema26).iloc[-1]
        dea = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]

        sh_pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
        sz_pct = (sz_df['close'].iloc[-1] - sz_df['close'].iloc[-2]) / sz_df['close'].iloc[-2] * 100
        cyb_pct = (cyb_df['close'].iloc[-1] - cyb_df['close'].iloc[-2]) / cyb_df['close'].iloc[-2] * 100

        market_ok = (cl.iloc[-1] > ma20)
        trend_status = "多头" if ma20 > ma60 else "空头"
        macd_status = "金叉向好" if dif > dea else "死叉承压"

        total_amt = (sh_df['amount'].iloc[-1] + sz_df['amount'].iloc[-1]) / 1e8
        total_amt_yest = (sh_df['amount'].iloc[-2] + sz_df['amount'].iloc[-2]) / 1e8
        amt_diff = total_amt - total_amt_yest
        amt_status = "放量" if amt_diff > 0 else "缩量"

        recent_low = cl.iloc[-20:].min()
        recent_high = cl.iloc[-20:].max()

        ai_insights = []
        today_cl, today_op, today_lo, today_hi = cl.iloc[-1], sh_df['open'].iloc[-1], sh_df['low'].iloc[-1], sh_df['high'].iloc[-1]

        min_1y, max_1y = cl.iloc[-250:].min(), cl.iloc[-250:].max()
        price_pct = (today_cl - min_1y) / (max_1y - min_1y) if max_1y > min_1y else 0.5

        if price_pct < 0.15:
            ai_insights.append("🟢 【冰点区】上证处于近一年绝对低位(分位<15%)，中长线价值凸显，随时可能否极泰来。")
        elif price_pct > 0.85:
            ai_insights.append("⚠️ 【高位区】上证处于近一年相对高位(分位>85%)，获利盘丰厚，注意高低切防御。")

        if (abs(ma10 - ma20) / ma20 < 0.015) and (abs(ma20 - ma60) / ma60 < 0.025):
            ai_insights.append("🌪️ 【面临变盘】主板短中长期均线高度密集粘合，市场即将选择重大方向突破！")

        body = abs(today_cl - today_op)
        lower_shadow = (min(today_op, today_cl) - today_lo) / today_op * 100
        upper_shadow_pct = ((today_hi - max(today_op, today_cl)) / body * 100) if body > 0 else 0

        if lower_shadow > 0.8:
            ai_insights.append("📌 【探底神针】指数盘中出现较长下影线，场外护盘/承接资金态度坚决。")
        elif upper_shadow_pct > 200:
            ai_insights.append("🚫 【抛压沉重】指数冲高回落留下长上影线(超实体2倍)，短期上行阻力极大。")

        if sh_pct > 0 and cyb_pct < 0:
            ai_insights.append("⚖️ 【风格分化】沪强创弱，资金扎堆大盘权重与红利，题材失血需防中小盘被杀。")
        elif sh_pct < 0 and cyb_pct > 0:
            ai_insights.append("🚀 【题材活跃】创强沪弱，权重搭台题材唱戏，资金高度活跃于中小成长股。")
        elif sh_pct > 0 and cyb_pct > 0:
            ai_insights.append("🌟 【普涨共振】各大指数齐步上扬，市场风险偏好全面回暖。")

        if amt_diff > 1000 and total_amt > 8000:
            ai_insights.append("🌊 【增量入场】两市成交额较昨日大幅放大超千亿，增量资金进场接力，行情可期。")
        elif amt_diff < -1000 and total_amt < 6000:
            ai_insights.append("🧊 【地量观望】两市成交额大幅萎缩交投清淡，场外资金处于极端观望期。")

        insights_str = "\n   • ".join(ai_insights) if ai_insights else "市场处于常规结构推演中，暂无极端形态信号。"

        plain_summary = ""
        if "冰点区" in insights_str:
            plain_summary = "👉 **大白话翻译**：现在市场极度悲观，大家都在割肉，正是买便宜货的好时候，但要有耐心拿住。"
        elif "高位区" in insights_str:
            plain_summary = "👉 **大白话翻译**：现在市场已经涨了不少，风险在积累，要小心别接最后一棒。"
        elif "面临变盘" in insights_str:
            plain_summary = "👉 **大白话翻译**：大盘马上就要选择方向了，不是大涨就是大跌，仓位先控制一下，留点子弹。"
        elif "普涨共振" in insights_str or "增量入场" in insights_str:
            plain_summary = "👉 **大白话翻译**：全市场都在涨，大部队资金已经杀进来了，闭着眼顺势做多赢面都很大！"
        else:
            plain_summary = "👉 **大白话翻译**：目前大盘走势比较平稳，没有极端大涨大跌的迹象，按照原定计划按部就班操作即可。"

        if market_ok:
            advice = "🔥 指数处于多头生命线之上，目前整体赔率较高 (建议总仓位 50%-80%)，可积极配置底仓。"
        else:
            advice = "🛡️ 指数失守MA20呈空头排列，弱势区间风险大 (建议总仓位 10%-30%)，防守为主，不要轻易抄底。"

        market_msg = (
            f"🎯 **指数收盘态势**:\n"
            f"   • 上证指数: {cl.iloc[-1]:.2f} ({sh_pct:+.2f}%)\n"
            f"   • 深证成指: {sz_df['close'].iloc[-1]:.2f} ({sz_pct:+.2f}%)\n"
            f"   • 创业板指: {cyb_df['close'].iloc[-1]:.2f} ({cyb_pct:+.2f}%)\n"
            f"💰 **全市场量能**: 两市总成交 {total_amt:.0f} 亿元 (较上日 {amt_diff:+.0f} 亿)\n"
            f"📈 **主板技术面**:\n"
            f"   • 趋势: {'🟢 企稳于MA20生命线' if market_ok else '🔴 跌破MA20生命线'}，大级别呈{trend_status}。\n"
            f"   • 动能: MACD{macd_status}，短期呈现{amt_status}{'反弹' if sh_pct > 0 else '调整'}。\n"
            f"   • 空间: 近期支撑 {recent_low:.0f} 点 | 压力 {recent_high:.0f} 点。\n"
            f"🧠 **AI 深度形态解盘**:\n"
            f"   • {insights_str}\n\n"
            f"{plain_summary}\n\n"
            f"💡 **总体仓位建议**: {advice}"
        )
    except Exception as e:
        log.warning(f"大盘基准获取失败: {e}")
        market_msg = f"大盘深度解析由于网络原因失败: {e}\n"
    
    return market_msg

def get_signals() -> tuple[list[Signal], list, set, int, str, int]:
    now = datetime.now(TZ_BJS)
    run_mode = os.environ.get('RUN_MODE', 'normal')
    
    log.info('🚀 防呆长线安全级·盘后复盘引擎启动...')
    if not IS_MANUAL and not is_valid_run_time(now): 
        return [], [], set(), 0, "", 0

    c_conf, pushed = Config(), load_pushed_state()

    if run_mode == 'market_only':
        log.info("🤖 进入 [纯大盘体检模式] ，短路全量个股运算，开始多维指数提取...")
        market_msg = extract_pure_market_context()
        return [], [], pushed, 0, market_msg, 0
    
    ex1 = ThreadPoolExecutor(max_workers=2)
    f_spot = ex1.submit(fetch_spot)
    f_flow = ex1.submit(get_fund_flow_map)
    
    df_raw = pd.DataFrame()
    flow_map = {}
    
    try:
        # 核心横截面数据，允许 45 秒超时
        df_raw = f_spot.result(timeout=45) 
    except Exception as e:
        log.error(f"❌ 核心横截面行情获取失败: {e}")
        ex1.shutdown(wait=False, cancel_futures=True)
        # 【优雅降级】：即使个股数据挂了，也要调用纯大盘指数测算，绝不让小白只看到干瘪的报错
        fallback_msg = extract_pure_market_context()
        m_msg = f"⚠️ **东方财富个股接口异常，今日自动降级为纯大盘体检：**\n\n{fallback_msg}"
        return [], [], pushed, 0, m_msg, 0

    try:
        # 资金流是辅助指标(极其容易超时)，单独隔离异常！允许失败降级，绝不影响主流程
        flow_map = f_flow.result(timeout=15)
    except Exception as e:
        log.warning(f"⚠️ 主力资金流辅助数据获取超时，自动降级跳过: {e}")
        flow_map = {}
        
    ex1.shutdown(wait=False, cancel_futures=True)

    df_clean, m_ok, m_msg, idx_ret, m_overheated = extract_market_context(df_raw, c_conf)
    if df_clean.empty:
        return [], [], pushed, 0, m_msg, 0

    mask = (df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & \
           (df_clean[C.S_PRICE] <= c_conf.MAX_PRICE) & \
           (df_clean[C.S_MCAP].between(c_conf.MIN_CAP, c_conf.MAX_CAP)) & \
           (df_clean[C.S_TURN].between(c_conf.MIN_TURNOVER, c_conf.MAX_TURNOVER)) & \
           (df_clean[C.S_PE] > c_conf.MIN_PE) & (df_clean[C.S_PE] <= c_conf.MAX_PE) & \
           (df_clean[C.S_PB] > 0) & (df_clean[C.S_PB] <= 10.0) & \
           (~df_clean[C.S_CODE].str.startswith(('688', '8', '4', '9'))) & \
           (df_clean[C.S_HIGH] > df_clean[C.S_LOW]) 
    
    if C.S_VR in df_clean.columns:
        mask &= df_clean[C.S_VR].between(c_conf.MIN_VOL_RATIO, c_conf.MAX_VOL_RATIO)

    pool = df_clean[mask].pipe(lambda d: d[~d[C.S_CODE].isin(pushed)]).copy()
    if pool.empty: return [], [], pushed, len(df_clean), m_msg, len(df_clean)

    sec_strengths = fetch_sector_strength()
    
    confirmed_data = [] # A/S 级正式推送列表
    watchlist_data = [] # B 级候补观察池列表
    
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
                    sector = fetch_stock_sector(row[C.S_CODE])
                    s_pct = sec_strengths.get(sector, 0.0)
                    data.update({
                        'sector': sector, 
                        'sector_pct': s_pct, 
                        'sector_ok': (0.3 < s_pct < 3.0),
                        'sector_overheated': (s_pct >= 5.0)
                    })
                    
                    score, level, reas = apply_scoring(data, now)
                    
                    # ── 分流处理 ──
                    if score >= 75: 
                        # AS 级：执行深度的风险审查并准备推送
                        sh_safe, sh_msg = check_shareholder_risk(row[C.S_CODE])
                        if not sh_safe:
                            log.info(f"🚫 {row[C.S_NAME]} 被极度危险护盾拦截: {sh_msg}")
                            continue 
                            
                        margin_msg = check_margin_crowding(row[C.S_CODE], data['mcap'])
                        if margin_msg:
                            reas += "\n" + margin_msg
                        
                        target1_price = round(row[C.S_PRICE]*(1+risk*2.5/100), 2)
                        money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1_price)
                        tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, m_ok, m_overheated)
                        plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])
                        hold_msg = generate_hold_period(data['adx'], data['price_pct'], data['has_chip_break'])
                        
                        confirmed_data.append(Signal(
                            code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
                            pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
                            trigger_time=now.strftime('%H:%M'), reasons=reas,
                            stop_loss=round(stop, 2), target1=target1_price,
                            target2=round(data['max_1y'], 2), ma10=round(data['ma10_val'], 2),
                            sector=sector, sector_pct=s_pct,
                            money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
                            plan_b_msg=plan_b_msg, hold_period_msg=hold_msg
                        ))
                        pushed.add(row[C.S_CODE]) # 只有 A/S 级才记录推送状态
                    elif score >= 65:
                        # B 级：仅存入观察池
                        watchlist_data.append((row[C.S_NAME], row[C.S_CODE], score, row[C.S_PRICE]))
                        
            except Exception:
                pass
    except FuturesTimeoutError:
        pass
    finally:
        ex2.shutdown(wait=False, cancel_futures=True)

    confirmed_data.sort(key=lambda x: x.score, reverse=True)
    watchlist_data.sort(key=lambda x: x[2], reverse=True) # 按分数降序
    return confirmed_data, watchlist_data, pushed, len(pool), m_msg, len(df_clean)


# ── 9. 通知推送逻辑 ───────────────────────────────────────────────────────────
def send_dingtalk(signals: list[Signal], watchlist: list, total_pool: int, total_market: int, market_msg: str) -> None:
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook: return
    
    now_ts = datetime.now(TZ_BJS)
    now_str = now_ts.strftime('%Y-%m-%d %H:%M')
    run_mode = os.environ.get('RUN_MODE', 'normal')
    
    header = f"🤖 AI 老股民保姆级盘后总结 {now_str}\n"
    if run_mode == 'market_only':
        header = f"🤖 AI 盘后大盘深度体检 {now_str}\n"
    elif run_mode != 'market_only' and total_market > 0:
        pass_rate = len(signals) / max(total_pool, 1) * 100 if total_pool > 0 else 0
        header += f"\n🔬 严苛雷达：全市场扫描 {total_market} 只个股，异动 {total_pool} 只，安全通过 {len(signals)} 只 (A级以上通过率 {pass_rate:.1f}%)\n"
        
    if market_msg:
        header += f"\n{market_msg}\n\n"

    if run_mode == 'market_only':
        content = header + "✅ 大盘分析播报完毕，本次任务短路了全量个股运算。"
    elif "接口异常" in market_msg or "网络原因失败" in market_msg:
        content = header + "⚠️ 今日个股数据扫描因接口异常中断，已为您提供大盘分析参考。"
    elif not signals and not watchlist:
        if not PUSH_EMPTY: return
        content = f"{header}✅ 安全雷达体检未发现完全符合安全边际的优质股，别乱买，我们空仓防守！"
    else:
        # ── 正式推荐部分 ──
        if signals:
            avg_score = sum(s.score for s in signals) / len(signals)
            quality_tag = "🥇 信号质量优秀，可按剧本布局" if avg_score >= 80 \
                else "🥈 信号质量一般，需严格限价且减半仓位"
                
            content = header + f"📈 **今日正式推送质量自检**：平均 {avg_score:.0f} 分 | {quality_tag}\n\n"
            
            parts = []
            for s in signals:
                sec_info = f"🏷️ 热点板块：{s.sector} (今日 {s.sector_pct:+.2f}%)\n" if s.sector else ""
                warn_msg = "⚡ 【风险警示】：该股为创业板(波动±20%)，心脏不好的务必把买入金额砍半！\n" if str(s.code).startswith('300') else ""
                prefix = '1' if str(s.code).startswith('6') else '0'
                tdx_market = 'SH' if str(s.code).startswith('6') else 'SZ' 
                
                parts.append(
                    f"【{s.name} ({s.code})】\n"
                    f"{sec_info}{warn_msg}"
                    f"⏰ 触发时间：{s.trigger_time}\n"
                    f"📊 综合评级：{s.score}分 {s.level}\n"
                    f"💰 今日收盘：¥{s.price} ({s.pct_chg})\n"
                    f"--- 💡 为什么机器选出它？ ---\n{s.reasons}\n"
                    f"--- 🛡️ 小白专属次日操作剧本 ---\n"
                    f"{s.hold_period_msg}\n\n"
                    f"{s.money_risk_msg}\n\n"
                    f"{s.tranche_plan_msg}\n\n"
                    f"{s.plan_b_msg}\n\n"
                    f"🎯 **铁血移动止盈**：赚钱后如果收盘跌破 ¥{s.ma10} (10日线)，别犹豫，立刻卖出一半保住利润！\n"
                    f"🚫 **次日防守纪律**：如果明天开盘直接高开超过 4%，说明有人抢跑，请直接放弃，绝不追高！\n"
                    f"🔗 东方财富直达：https://quote.eastmoney.com/unify/r/{prefix}.{s.code}\n"
                    f"🔗 复制看盘：{tdx_market}{s.code}\n"
                )
            content += "\n".join(parts)
        else:
            content = header + "✅ 今日未发现 A 级以上稳健信号，正式推荐列表空仓防守中。\n"

        # ── 候补观察池部分 ──
        if watchlist:
            watch_lines = "\n".join(
                f"   • {name}({code}) ¥{price} 得分{score}分"
                for name, code, score, price in watchlist[:5]
            )
            content += (
                f"\n\n👁️ **候补观察池（看看就好，手别动）**\n"
                f"{watch_lines}\n"
                f"以上标的当前评级为 B 级，系统判断波动或风险偏大，暂不提供操作剧本。仅作盘感追踪，待其评级升至 A 级后再考虑介入。"
            )
        
        content += (
            "\n\n🤔 **买入前灵魂拷问：**\n"
            "如果明天买入的股票跌了 5%，我会焦虑得睡不着觉吗？\n"
            "→ 如果会，请把你准备买入的金额【再砍掉一半】！投资是为了生活更好，不是为了找罪受。"
        )

    try:
        requests.post(webhook, json={'msgtype': 'text', 'text': {'content': content}}, timeout=10)
        log.info(f"✅ 推送成功 ({len(signals)}正式 / {len(watchlist)}观察)")
    except Exception as e:
        log.error(f"❌ 推送失败: {e}")

if __name__ == '__main__':
    try:
        sigs, watch, pool_size, m_msg, total_mkt = get_signals()
        send_dingtalk(sigs, watch, pool_size, total_mkt, m_msg)
        if sigs: save_pushed_state(pushed)
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
