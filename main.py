import os
import time
import json
import socket
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Tuple, Callable

import requests
import numpy as np
import pandas as pd
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

# ═════════════════════════════════════════════════════════════════════════════
# 1. 环境与核心配置 (Environment & Config)
# ═════════════════════════════════════════════════════════════════════════════
TZ_BJS       = pytz.timezone('Asia/Shanghai')
STATE_FILE   = 'pushed_state.json'
SPOT_CACHE   = 'spot_cache.pkl'
HIST_CACHE_DIR = 'hist_cache'
PAPER_TRADES_FILE = 'paper_trades.json' 
IS_MANUAL    = os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
PUSH_EMPTY   = os.environ.get('PUSH_EMPTY_RESULT', 'true').lower() in ('true', '1', 'yes')

socket.setdefaulttimeout(15.0)

logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

if not os.path.exists(HIST_CACHE_DIR):
    os.makedirs(HIST_CACHE_DIR, exist_ok=True)

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')


# ═════════════════════════════════════════════════════════════════════════════
# 2. 推送状态与模拟盘自进化系统 (State & AI Evolution)
# ═════════════════════════════════════════════════════════════════════════════
def load_pushed_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                if 'date' in data and 'pushed_codes' in data:
                    return {code: data['date'] for code in data.get('pushed_codes', [])}
                return data
        except Exception as e:
            log.warning(f"读取推送记录失败: {e}")
    return {}

def save_pushed_state(pushed_dict: dict) -> None:
    today_dt = datetime.now(TZ_BJS).date()
    clean_dict = {}
    for code, date_str in pushed_dict.items():
        try:
            expire_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if (today_dt - expire_date).days <= 7:
                clean_dict[code] = date_str
        except Exception:
            pass
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(clean_dict, f)
    except Exception:
        pass
        
    today_suffix = _today_str().replace('-', '')
    try:
        for fname in os.listdir(HIST_CACHE_DIR):
            if not fname.endswith(f"{today_suffix}.pkl"):
                os.remove(os.path.join(HIST_CACHE_DIR, fname))
    except Exception:
        pass

def is_recently_pushed(code: str, pushed: dict) -> bool:
    if code not in pushed:
        return False
    try:
        expire_date = datetime.strptime(pushed[code], '%Y-%m-%d').date()
        today_date = datetime.now(TZ_BJS).date()
        return today_date < expire_date
    except Exception:
        return False

def load_and_update_paper_trades(df_spot: pd.DataFrame) -> tuple[list, dict]:
    trades = []
    if os.path.exists(PAPER_TRADES_FILE):
        try:
            with open(PAPER_TRADES_FILE, 'r') as f:
                trades = json.load(f)
        except Exception: pass

    spot_dict = df_spot.set_index(Cols.S_CODE).to_dict('index')
    
    stats = {
        '85-100': {'win': 0, 'total': 0}, '80-85': {'win': 0, 'total': 0},
        '75-80': {'win': 0, 'total': 0}, '70-75': {'win': 0, 'total': 0}, '<70': {'win': 0, 'total': 0}
    }

    active_trades = []
    today_date = datetime.now(TZ_BJS).date()
    
    for t in trades:
        status = t.get('status', 'PENDING')
        code = t.get('code')
        
        if status == 'PENDING' and code in spot_dict:
            row = spot_dict[code]
            high = float(row.get(Cols.S_HIGH, 0))
            low = float(row.get(Cols.S_LOW, 0))
            close = float(row.get(Cols.S_PRICE, 0))
            buy_date = datetime.strptime(t['date'], '%Y-%m-%d').date()
            days_held = (today_date - buy_date).days

            if high >= t['target']:
                t['status'] = 'WIN'
            elif low <= t['stop']:
                t['status'] = 'LOSS'
            elif days_held > 10:  
                t['status'] = 'WIN' if close > t['buy_price'] else 'LOSS'

        if t['status'] in ('WIN', 'LOSS'):
            bucket = t.get('score_bucket', '<70')
            if bucket in stats:
                stats[bucket]['total'] += 1
                if t['status'] == 'WIN':
                    stats[bucket]['win'] += 1

        active_trades.append(t)

    return active_trades[-500:], stats

def save_paper_trades(trades: list):
    try:
        with open(PAPER_TRADES_FILE, 'w') as f:
            json.dump(trades, f)
    except Exception as e:
        log.error(f"保存模拟盘账本失败: {e}")

def get_score_bucket(score: float) -> str:
    if score >= 85: return '85-100'
    if score >= 80: return '80-85'
    if score >= 75: return '75-80'
    if score >= 70: return '70-75'
    return '<70'


# ═════════════════════════════════════════════════════════════════════════════
# 3. 数据契约模型 (Data Schema & Models)
# ═════════════════════════════════════════════════════════════════════════════
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
    MIN_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MIN_CAP', 30e8)) 
    MAX_CAP: float       = field(default_factory=lambda: EnvParser.get_float('MAX_CAP', 2000e8))
    MAX_PRICE: float     = field(default_factory=lambda: EnvParser.get_float('MAX_PRICE', 500.0))  
    MIN_PE: float        = field(default_factory=lambda: EnvParser.get_float('MIN_PE', 0))    
    MAX_PE: float        = field(default_factory=lambda: EnvParser.get_float('MAX_PE', 300))      
    MIN_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MIN_TURNOVER', 0.5))
    MAX_TURNOVER: float  = field(default_factory=lambda: EnvParser.get_float('MAX_TURNOVER', 40.0)) 
    MIN_PCT_CHG: float   = field(default_factory=lambda: EnvParser.get_float('MIN_PCT_CHG', -4.0))  
    MIN_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MIN_VOL_RATIO', 0.5))  
    MAX_VOL_RATIO: float = field(default_factory=lambda: EnvParser.get_float('MAX_VOL_RATIO', 15.0))
    
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
    ma10: float
    
    money_risk_msg: str = ""
    tranche_plan_msg: str = ""
    plan_b_msg: str = ""
    hold_period_msg: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# 4. 专业量化算法核心库 (Quant Algorithms)
# ═════════════════════════════════════════════════════════════════════════════
class MathUtils:
    @staticmethod
    def calc_vcp_quality(df: pd.DataFrame) -> Tuple[float, bool]:
        if len(df) < 31:
            return 0.5, False
            
        segments = []
        for i in [(-31, -21), (-21, -11), (-11, -1)]:
            seg = df.iloc[i[0]:i[1]]
            low = seg[C.H_LOW].min()
            if low > 0:
                amp = (seg[C.H_HIGH].max() - low) / low
                segments.append(amp)
                
        if len(segments) < 3:
            return segments[-1] if segments else 0.5, False
            
        is_vcp = segments[0] > segments[1] > segments[2]
        return segments[-1], is_vcp

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

def calc_target_price(price: float, stop: float, data: dict) -> float:
    risk_amt = price - stop
    if data.get('has_chip_break'):
        max_1y = data.get('max_1y', price * 1.20)
        chip_target = price + (max_1y - price) * 0.5
        min_target = price + risk_amt * 1.5
        return round(max(chip_target, min_target), 2)
    return round(price + risk_amt * 2.0, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 5. 排版与文案渲染器 (Markdown Renderers)
# ═════════════════════════════════════════════════════════════════════════════
def format_money_risk_msg(price: float, stop_loss: float, target1: float) -> str:
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
        evaluation = "🎯 **高容错**：做对一次抵消多次亏损！"
    elif ratio >= 1.5:
        evaluation = "✅ **尚可**：跌势有限，可防守建仓。"
    else:
        evaluation = "⚠️ **需谨慎**：操作要求高，务必**减半仓位**！"
    
    return (
        f"- 💸 **仓位测算**：买 {hands} 手约需 `¥{total_cost:.0f}` (按1万预算计)\n"
        f"- 🔴 **止损风险**：预估最大回撤约 `-¥{total_loss:.0f}`\n"
        f"- 🟢 **止盈目标**：第一波段预估盈利 `+¥{gain_1:.0f}`\n"
        f"- 📐 **盈亏比**：`1 : {ratio_str}` ➡️ {evaluation}"
    )

def generate_tranche_plan(price: float, score: int, market_ok: bool, market_overheated: bool) -> str:
    if market_overheated:
        return "🛑 **【系统熔断】当前市场情绪极度过热！随时面临收割踩踏，系统强制禁止明日建仓！**"
        
    base_pct = 30 if score >= 85 else 20 if score >= 70 else 10
    if not market_ok:
        base_pct = base_pct // 2
        
    t1 = max(1, base_pct // 3)
    t2 = max(1, base_pct // 3)
    t3 = max(1, base_pct - t1 - t2)
    
    lower_bound = round(price * 0.985, 2)
    upper_bound = round(price * 1.005, 2)
    add_price   = round(price * 1.025, 2)
    stop_add    = round(price * 1.05,  2)
    
    return (
        f"- **① 关注支撑**：次日重点观察 `¥{lower_bound} - ¥{upper_bound}` 区间，若缩量企稳可分批 **{t1}%** 试错。\n"
        f"- **② 稳健加仓**：若后续确认上攻站稳 `¥{add_price}`，可适当加仓 **{t2}%**。\n"
        f"- **③ 追击确认**：突破形态上沿 `¥{stop_add}`，最后追加确认仓位 **{t3}%**。"
    )

def generate_plan_b(price: float, stop_loss: float, ma20: float) -> str:
    normal_shake = round(price * 0.97, 2)  
    normal_shake = max(normal_shake, stop_loss + 0.01)
    
    return (
        f"- **📉 正常波动**：只要收盘未破 `¥{normal_shake:.2f}`，属于正常洗盘震荡。\n"
        f"- **🔪 铁血防线**：有效跌破 `¥{stop_loss:.2f}`，说明逻辑证伪，**必须无条件执行止损！**\n"
        f"- **💥 系统风险**：若遇大盘单日非理性暴跌，优先保住本金安全。"
    )

def generate_hold_period(adx: float, price_pct: float, has_chip_break: bool) -> str:
    if price_pct < 0.35 and adx < 20:
        return "- **⏳ 持股预期**：🐢 **【底部潜伏型】(1~3个月)**，存死期别盯盘。"
    elif adx > 25 or has_chip_break:
        return "- **⏳ 持股预期**：🐎 **【右侧趋势型】(3~10天)**，随时加速，切忌贪心。"
    else:
        return "- **⏳ 持股预期**：🐕 **【稳健震荡型】(2~4周)**，需要耐心等风来。"


# ═════════════════════════════════════════════════════════════════════════════
# 6. 统一数据代理与本地数据湖 (Data Proxy & Data Lake)
# ═════════════════════════════════════════════════════════════════════════════
import akshare as ak

try:
    import baostock as bs
except ImportError:
    bs = None

try:
    import efinance as ef
except ImportError:
    ef = None

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

class DataProxy:
    """数据获取多源路由层 (Fallback Waterfall)"""
    def __init__(self):
        self.bs_logged_in = False

    def __del__(self):
        if self.bs_logged_in and bs is not None:
            try: bs.logout()
            except: pass

    def _login_baostock(self):
        if bs is not None and not self.bs_logged_in:
            bs.login()
            self.bs_logged_in = True

    # ---- [1. Historical Data] ----
    def _fetch_hist_tushare(self, code, start, end):
        # TODO: 预留给 Tushare / QMT 实盘高级接口 (最高优)
        return None

    def _fetch_hist_baostock(self, code, start, end):
        if bs is None: return None
        self._login_baostock()
        try:
            prefix = 'sh.' if code.startswith('6') else 'sz.'
            # baostock 的日期格式要求 YYYY-MM-DD
            start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            rs = bs.query_history_k_data_plus(prefix + code,
                "date,open,close,high,low,volume",
                start_date=start_fmt, end_date=end_fmt,
                frequency="d", adjustflag="1") # 1 = 前复权
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            if not data_list: return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            df = df.rename(columns={'date': C.H_DATE, 'open': C.H_OPEN, 'close': C.H_CLOSE, 'high': C.H_HIGH, 'low': C.H_LOW, 'volume': C.H_VOL})
            for col in [C.H_OPEN, C.H_CLOSE, C.H_HIGH, C.H_LOW, C.H_VOL]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df[list(Config.HIST_COLS)]
        except Exception as e:
            log.debug(f"[Tier 2 BaoStock] 获取历史失败: {e}")
            return None

    @retry(times=3, delay=2)
    def _fetch_hist_akshare(self, code, start, end):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust='qfq')
            if df is not None and not df.empty:
                return df[list(Config.HIST_COLS)].copy()
        except Exception as e:
            try:
                df = ak.stock_zh_a_hist_tx(symbol=code, start_date=start, end_date=end, adjust='qfq')
                if df is not None and not df.empty:
                    col_map = {'日期': C.H_DATE, '开盘': C.H_OPEN, '收盘': C.H_CLOSE, '最高': C.H_HIGH, '最低': C.H_LOW, '成交量': C.H_VOL}
                    df = df.rename(columns=col_map)
                    return df[list(Config.HIST_COLS)].copy()
            except Exception:
                pass
            raise ValueError(f'akshare history empty for {code}')
            
    def get_hist(self, code, start, end) -> pd.DataFrame:
        df = self._fetch_hist_tushare(code, start, end)
        if df is not None: return df
        df = self._fetch_hist_baostock(code, start, end)
        if df is not None: return df
        return self._fetch_hist_akshare(code, start, end)

    # ---- [2. Spot Data (实时横截面)] ----
    def _fetch_spot_qmt(self):
        # TODO: 预留给 QMT / Tushare (最高优)
        return None

    def _fetch_spot_efinance(self):
        if ef is None: return None
        try:
            df = ef.stock.get_realtime_quotes()
            if df is not None and not df.empty:
                rename_map = {'代码': C.S_CODE, '名称': C.S_NAME, '最新价': C.S_PRICE,
                              '涨跌幅': C.S_PCT, '今开': C.S_OPEN, '最高': C.S_HIGH,
                              '最低': C.S_LOW, '成交量': C.S_VOL, '成交额': C.S_AMT,
                              '换手率': C.S_TURN, '市盈率-动态': C.S_PE, '市净率': C.S_PB, '量比': C.S_VR}
                df = df.rename(columns=rename_map)
                return df
        except Exception as e:
            log.debug(f"[Tier 2 efinance] 获取实时行情失败: {e}")
        return None

    @retry(times=3, delay=5)
    def _fetch_spot_akshare(self):
        try:
            time.sleep(random.uniform(1.0, 3.0))
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log.warning(f"行情主接口异常: {e}，正在启动新浪备用源执行优雅降级...")
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                rename_map = {'代码': C.S_CODE, '名称': C.S_NAME, '最新价': C.S_PRICE,
                              '涨跌幅': C.S_PCT, '今开': C.S_OPEN, '最高': C.S_HIGH,
                              '最低': C.S_LOW, '成交量': C.S_VOL, '成交额': C.S_AMT}
                df = df.rename(columns=rename_map)
                fallback_defaults = {C.S_TURN: 2.0, C.S_MCAP: 100e8, C.S_PE: -1.0, C.S_PB: 2.0, C.S_VR: 1.0}
                for col, val in fallback_defaults.items():
                    if col not in df.columns: df[col] = val
                return df
            raise ValueError('spot_empty')
            
    def get_spot(self) -> pd.DataFrame:
        df = self._fetch_spot_qmt()
        if df is not None: return df
        df = self._fetch_spot_efinance()
        if df is not None: return df
        return self._fetch_spot_akshare()

    # ---- [3. Index & Context] ----
    @retry(times=4, delay=2)
    def get_index(self, symbol: str) -> pd.DataFrame:
        try:
            df = ak.stock_zh_index_daily_tx(symbol=symbol)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as e:
            log.warning(f"腾讯指数接口波动 ({symbol}): {e}，切换东方财富源...")
        
        df = ak.stock_zh_index_daily_em(symbol=symbol)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
        raise ValueError(f'index_empty_{symbol}')

    @retry(times=3, delay=2)
    def get_core_pool(self) -> set:
        pool = set()
        try:
            for idx in ["000300", "000905", "000852", "399006"]:
                df = ak.index_stock_cons(symbol=idx)
                if df is not None and not df.empty:
                    col = next((c for c in df.columns if '代码' in c), None)
                    if col: pool.update(df[col].astype(str).str.zfill(6).tolist())
        except Exception as e:
            log.warning(f"获取核心成分股池失败，降级为全市场扫描: {e}")
        return pool

    @retry(times=3, delay=2)
    def get_hot_sectors(self) -> dict:
        hot_stocks = {}
        try:
            df = ak.stock_board_industry_name_em()
            if df is None or df.empty: return {}
            top_sectors = df.nlargest(5, '涨跌幅')['板块名称'].tolist()
            for sector in top_sectors:
                try:
                    time.sleep(0.5) 
                    cons = ak.stock_board_industry_cons_em(symbol=sector)
                    if cons is not None and not cons.empty:
                        col = next((c for c in cons.columns if '代码' in c), None)
                        if col:
                            for code in cons[col].astype(str).str.zfill(6).tolist():
                                hot_stocks[code] = sector
                except Exception: pass
        except Exception as e:
            log.warning(f"主线板块榜单数据获取失败: {e}")
        return hot_stocks

    @retry(times=2, delay=2)
    def get_northbound_flow(self) -> tuple[float, str]:
        try:
            df = ak.stock_em_hsgt_north_net_flow_in(indicator="沪深港通")
            if df is not None and not df.empty:
                col = 'value' if 'value' in df.columns else df.columns[-1]
                today_flow = float(df.iloc[-1][col]) / 1e8
                if today_flow > 30: return today_flow, f"\n- 🌊 **聪明钱流向**：北水大举流入 **+{today_flow:.0f}亿**"
                elif today_flow < -30: return today_flow, f"\n- ❄️ **聪明钱流向**：北水大幅流出 **{today_flow:.0f}亿**"
                else: return today_flow, f"\n- ⚖️ **聪明钱流向**：北向资金温和 (**{today_flow:+.0f}亿**)"
        except Exception: pass
        return 0.0, ""

class LocalDataLake:
    """本地数据湖缓存拦截层"""
    def __init__(self, proxy: DataProxy):
        self.proxy = proxy
        self.spot_cache = SPOT_CACHE
        self.hist_dir = HIST_CACHE_DIR

    def fetch_spot(self) -> pd.DataFrame:
        if os.path.exists(self.spot_cache):
            mtime = os.path.getmtime(self.spot_cache)
            if time.time() - mtime < 3600 * 2:  
                log.info("📦 检测到本地有效缓存，跳过全量接口请求，直接极速加载...")
                try: return pd.read_pickle(self.spot_cache)
                except Exception: pass
        df = self.proxy.get_spot()
        try: df.to_pickle(self.spot_cache)
        except Exception: pass
        return df

    def fetch_hist(self, code: str, start: str, end: str) -> Optional[pd.DataFrame]:
        cache_file = os.path.join(self.hist_dir, f"{code}_{end}.pkl")
        if os.path.exists(cache_file):
            try: return pd.read_pickle(cache_file)
            except Exception: pass
        df = self.proxy.get_hist(code, start, end)
        if df is not None:
            try: df.to_pickle(cache_file)
            except Exception: pass
        return df

    def fetch_index(self, symbol: str): return self.proxy.get_index(symbol)
    def fetch_core_pool(self): return self.proxy.get_core_pool()
    def fetch_hot_sectors(self): return self.proxy.get_hot_sectors()
    def fetch_northbound_flow(self): return self.proxy.get_northbound_flow()

# ── 实例化全局单例，保持对外接口完全兼容 ──
_DATA_PROXY = DataProxy()
_DATA_LAKE = LocalDataLake(_DATA_PROXY)

def fetch_spot(): return _DATA_LAKE.fetch_spot()
def fetch_hist(code, start, end): return _DATA_LAKE.fetch_hist(code, start, end)
def fetch_index(symbol): return _DATA_LAKE.fetch_index(symbol)
def fetch_core_pool(): return _DATA_LAKE.fetch_core_pool()
def fetch_hot_sectors(): return _DATA_LAKE.fetch_hot_sectors()
def fetch_northbound_flow(): return _DATA_LAKE.fetch_northbound_flow()


# ═════════════════════════════════════════════════════════════════════════════
# 7. 技术面特征抽取引擎 (Feature Extraction Engine)
# ═════════════════════════════════════════════════════════════════════════════
class AShareTechnicals:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        close = self.df[C.H_CLOSE]
        high, low, vol = self.df[C.H_HIGH], self.df[C.H_LOW], self.df[C.H_VOL]
        
        for span in (10, 20, 60): 
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
        
        if today[C.H_CLOSE] < today['MA20'] * 0.85: return None

        rsi = float(today.get('RSI14', 50))
        if pd.isna(rsi) or rsi > 85: return None 

        consecutive_down = 0
        for i in range(2, 8):
            if len(df) >= i and df[C.H_CLOSE].iloc[-i] < df[C.H_OPEN].iloc[-i]:
                consecutive_down += 1
            else:
                break

        extreme_shrink_vol = yest[C.H_VOL] < today['MA20_V'] * 0.75

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
            
        total_range = today[C.H_HIGH] - today[C.H_LOW]
        upper_shadow = today[C.H_HIGH] - max(today[C.H_OPEN], today[C.H_CLOSE])
        upper_shadow_pct = (upper_shadow / total_range * 100) if total_range > 1e-5 else 0.0

        last_hist_pct = float(df['PCT_CHG'].iloc[-2]) if len(df) >= 2 else 0.0
        has_pullback = bool(
            today[C.H_CLOSE] >= today['MA20'] * 0.97 and 
            today[C.H_VOL] < today['MA5_V'] * 1.2 and
            -6.0 <= last_hist_pct <= 3.5
        )
        
        surge_5d = (today[C.H_CLOSE] / df[C.H_CLOSE].iloc[-6] - 1) * 100 if len(df) >= 6 else 0.0
        
        vcp_amp, is_true_vcp = MathUtils.calc_vcp_quality(df)

        return {
            'price_pct': price_pct, 'max_1y': max_1y, 'adx': float(today['ADX']),
            'bull_rank': (today['MA20'] > today['MA60']),
            'extreme_shrink_vol': extreme_shrink_vol,
            'has_zt': bool((df['PCT_CHG'].iloc[-61:-1] >= 9.5).any()),
            'has_consecutive_zt': bool(((df['PCT_CHG'].iloc[-61:-1] >= 9.5).rolling(2).sum() >= 2).any()),
            'vcp_amp': vcp_amp,
            'is_true_vcp': is_true_vcp,
            'upper_shadow_pct': upper_shadow_pct,
            'lower_shadow_ratio': (min(today[C.H_OPEN], today[C.H_CLOSE]) - today[C.H_LOW]) / today[C.H_OPEN] if pd.notna(today[C.H_OPEN]) and today[C.H_OPEN] > 0 else 0.0,
            'has_obv_break': bool(df['OBV'].iloc[-1] > df['OBV'].iloc[-21:-1].max()),
            'has_pullback': has_pullback,
            'has_chip_break': has_chip_break,
            'dist_ma20': (today[C.H_CLOSE] / today['MA20'] - 1) * 100,
            'red_days': red_days,
            'rsi': rsi,
            'consecutive_down': consecutive_down,
            'surge_5d': surge_5d,
            'macd_dea': float(today['DEA']),
            'ma10_val': float(today['MA10']), 'ma20_val': float(today['MA20']), 'atr_val': float(today['ATR']),
            'close_val': float(today[C.H_CLOSE]),
            'low_val': float(today[C.H_LOW]), 'recent_20_low': float(df[C.H_LOW].iloc[-20:].min()),
            'boll_lower': float(today['MA20'] - 2 * df[C.H_CLOSE].iloc[-20:].dropna().std()) if len(df[C.H_CLOSE].iloc[-20:].dropna()) >= 2 else np.nan,
            'close_60d_ago': float(df[C.H_CLOSE].iloc[-60]) if len(df) >= 60 else 0.0,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 8. 打分与自适应演化引擎 (Scoring & Evolution Engine)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Factor:
    condition: Callable[[dict], bool]
    points: int
    weight: float = 1.0
    template: str = ""

def apply_scoring(data: dict, now: datetime, m_regime: str, vol_surge: bool, win_stats: dict) -> tuple[int, str, str]:
    adx = data['adx']
    tw, rw = (1.4, 0.7) if adx > 25 else (0.8, 1.4) if adx < 15 else (1.0, 1.0)
    
    f_val, f_mom, f_rev, f_risk = 1.0, 1.0, 1.0, 1.0
    regime_msg = ""
    if m_regime == 'BULL':
        f_mom, f_val, f_risk = 1.3, 0.8, 0.8  
        regime_msg = "🔥 **[多头加权]** 重动量突破，容忍高位波动"
    elif m_regime == 'BEAR':
        f_val, f_mom, f_rev, f_risk = 1.3, 0.6, 1.2, 1.5  
        regime_msg = "🐻 **[空头加权]** 重防守低估，严惩高位接盘"
    elif m_regime == 'PANIC':
        f_rev, f_mom, f_val, f_risk = 1.5, 0.3, 1.2, 1.5  
        regime_msg = "🧊 **[冰点加权]** 重超跌反转，规避连板接力"
    else:
        regime_msg = "⚖️ **[均衡加权]** 因子权重保持中立映射"

    if vol_surge:
        f_mom += 0.2
        regime_msg += " | 🌊 **[量能爆发]** 大盘放量，动量进一步加权"
        
    meta = (
        f"- 🧭 **趋势雷达**：{'处于强势主升浪中' if adx > 25 else '正处于底部反转期' if adx < 15 else '平稳震荡蓄势中'}\n"
        f"- ⚙️ **因子暴露**：{regime_msg}"
    )

    in_danger, danger_label = is_earnings_danger_zone(now)

    factors = [
        Factor(lambda d: d['mcap'] > 300e8 and 0 < d['pe'] < 25 and d['pb'] < 3, 10, f_val, "- 🏢 **价值蓝筹**：大市值低估值核心资产，防守属性极强"),
        Factor(lambda d: d['vol_ratio'] > 1.0 and d['rs_rating'] > 5, 10, f_mom, "- 🚀 **强势领涨**：近期显著强于大盘，资金接力意愿极强"),
        Factor(lambda d: d['price_pct'] < 0.3 and 0 < d['pb'] < 1.0, 8, f_val, "- ♻️ **困境反转**：股价严重破净且处于绝对低位，安全垫极厚"),
        
        Factor(lambda d: d.get('in_hot_sector', False), 12, f_mom, "- 🌡️ **身处主线**：所在板块【{hot_sector_name}】今日强势领涨，踏准市场节奏"),
        
        Factor(lambda d: d['price_pct'] < 0.25, 12, f_rev * rw, "- 🟢 **绝对低位**：目前买入相当于抄底，长线持有安全"),
        Factor(lambda d: 0.25 <= d['price_pct'] <= 0.45, 8, f_rev, "- 🟢 **相对低位**：刚刚从底部爬起来，输时间不输钱"),
        Factor(lambda d: d['price_pct'] > 0.45, 6, f_mom, "- 📈 **多头趋势**：股价已脱离底部，处于健康的主升浪区间"),
        Factor(lambda d: d['price_pct'] > 0.85, 8, f_mom, "- 🚀 **高位突破**：股价处于年度高位，强者恒强趋势极佳"), 
        
        Factor(lambda d: d['pe'] > 0 and d['pe'] < 40, 5, f_val, "- 🛡️ **业绩护体**：市盈率健康，不是炒空气的无基本面股"),
        Factor(lambda d: d['macd_dea'] >= -0.05, 5, 1.0, "- 🌊 **多头控盘**：大周期趋势仍强，没有被深套的风险"), 
        
        Factor(lambda d: -2.0 <= d['dist_ma20'] <= 6.0, 12, 1.0, "- 🧲 **贴地潜伏**：目前价格紧贴均线支撑，绝佳安全低吸点"),
        Factor(lambda d: 6.0 < d['dist_ma20'] <= 15.0, 6, f_mom, "- 🚀 **强势发力**：距离20日线有空间，依托短期均线强势上攻"),
        Factor(lambda d: d['dist_ma20'] < -2.0, -10, f_risk, "- ⚠️ **破位嫌疑**：当前已跌破20日线，需警惕趋势走坏 (扣分)"),
        
        Factor(lambda d: 30 <= d.get('rsi', 50) <= 72, 5, 1.0, "- 📊 **温度适中**：RSI处于健康买入区间，正是下手时机"),
        
        Factor(lambda d: d['bull_rank'], 8, f_mom, "- 📈 **顺势而为**：均线多头排列，跟着主力资金大部队走"),
        
        Factor(lambda d: d['has_zt'], 8, 1.0, "- 🔥 **股性活跃**：该股历史上容易涨停，不会一潭死水"),
        Factor(lambda d: d['vol_ratio'] >= 1.8, 8, 1.0, "- 🔵 **放量确认**：今天成交量明显放大，大资金开始干活了"),
        Factor(lambda d: d['red_days'] >= 2, 5, 1.0, "- 🔴 **稳步推升**：最近重心都在上移，主力在偷偷温和建仓"),
        
        Factor(lambda d: d['has_chip_break'], 12, tw * f_mom, "- 🏔️ **抛压真空**：上方的套牢盘已割肉离场，向上拉升没阻力"),
        Factor(lambda d: d.get('is_true_vcp', False), 12, 1.0, "- 🎯 **形态确认**：呈现经典 VCP (波动率收敛) 结构，洗盘极度充分"),
        Factor(lambda d: not d.get('is_true_vcp', False) and d['vcp_amp'] < 0.12, 6, 1.0, "- 🟣 **蓄势待发**：近期波动极小，面临短线方向选择"),
        Factor(lambda d: d['extreme_shrink_vol'], 8, 1.0, "- 🧊 **没人砸盘**：爆发前夕成交极度萎缩，散户该卖的都卖了"), 
        Factor(lambda d: d['has_obv_break'], 10, tw * f_mom, "- 💸 **真金白银**：模型监控到真实的资金在创纪录净流入"),
        Factor(lambda d: d['has_pullback'], 12, 1.0, "- 🪃 **黄金深坑**：出现温和缩量回踩，主力洗盘给出的上车良机"),
        Factor(lambda d: d['lower_shadow_ratio'] > 0.03, 5, 1.0, "- 📌 **强力护盘**：跌下去被大资金迅速买回，下方有人兜底"), 
        
        Factor(lambda d: d.get('rs_rating', 0) > 5,  8, f_mom, "- 🏆 **跑赢大盘**：近60日涨幅超越指数，有资金在持续运作"),
        
        # --- 【排雷扣分项】 ---
        Factor(lambda d: d.get('surge_5d', 0) > 28, -20, f_risk, "- 🚫 **短期暴涨**：近5日涨幅过大透支空间，极易高位站岗 (重度扣分)"),
        Factor(lambda d: d.get('consecutive_down', 0) >= 4, -15, f_risk, "- 🔪 **飞刀预警**：近期连续阴线急跌，左侧接飞刀风险大 (重度扣分)"),
        Factor(lambda d: d.get('rsi', 50) > 80, -10, f_risk, "- 🌡️ **短期过热**：RSI偏高短线超买，操作需要进一步缩减仓位"),
        Factor(lambda d: d.get('rs_rating', 0) < -10, -8, f_risk, "- 📉 **跑输大盘**：近期持续弱于大盘，跟的是被冷落的股票"),
        Factor(lambda d: d['has_consecutive_zt'] and d['price_pct'] < 0.40, 10, f_mom, "- 🔥 **低位连板**：刚刚启动的龙头，安全且市场辨识度极高"),
        Factor(lambda d: d['has_consecutive_zt'] and d['price_pct'] >= 0.90, -15, f_risk, "- ⚠️ **高位接盘**：股价已被炒高连板，千万别追容易接盘！"),
        Factor(lambda d: d['upper_shadow_pct'] > 35, -15, f_risk, "- ⚠️ **诱多预警**：冲高后大幅跳水，上方抛压极重别上当！"),
        Factor(lambda d: d['dist_ma20'] > 25, -15, f_risk, "- 🚫 **追高预警**：目前涨得太急离均线太远，随时面临暴跌回调"),
        
        Factor(lambda d: in_danger and d['mcap'] < 100e8, -8, f_risk, f"- 📅 **财报防雷**：当前属于{danger_label}，小盘股需防业绩变脸 (扣分)")
    ]

    raw_score, reasons = 45, [meta] if meta else []
    
    for f in factors:
        if f.condition(data):
            raw_score += int(f.points * f.weight)
            try:
                reasons.append(f.template.format(**data))
            except KeyError:
                reasons.append(f.template)

    raw_score = max(0, min(raw_score, 100))
    
    # ── 【AI 胜率自进化机制】 ──
    bucket = get_score_bucket(raw_score)
    b_stats = win_stats.get(bucket, {'win': 0, 'total': 0})
    if b_stats['total'] >= 5:  
        wr = b_stats['win'] / b_stats['total']
        multiplier = 0.8 + 0.4 * wr
        final_score = int(raw_score * multiplier)
        reasons.append(f"- 🧬 **AI自进化**：该分数段实盘历史胜率 `{wr*100:.1f}%`，系统执行动态调分：**{raw_score} ➡️ {final_score}**")
    else:
        final_score = raw_score
        reasons.append(f"- 🧬 **AI自进化**：该分数段暂无足够历史样本以供进化。")
    
    final_score = max(0, min(final_score, 100))
    
    if final_score >= 85:
        level = '⭐⭐⭐⭐⭐ 🐯 **[S级·老虎机]** (胜率极高，跌势有限)'
    elif final_score >= 75:
        level = '⭐⭐⭐⭐ 🐕 **[A级·看门狗]** (防守兼备，需耐心等涨)'
    elif final_score >= 70:
        level = '⭐⭐⭐ 🦊 **[B+级·小狐狸]** (次优机会，必须控制仓位)'
    else:
        level = '⭐⭐ 🐒 **[B级·小猕猴]** (上蹿下跳振幅大，新手回避)'
        
    return final_score, level, '\n'.join(reasons)

def prescreen_score(row: pd.Series) -> float:
    s = 50.0
    vr = float(row.get(C.S_VR, 1.0))
    pct = float(row.get(C.S_PCT, 0.0))
    if vr > 1.5 and pct > 0:
        s += 15
    elif vr < 0.7:
        s -= 10
        
    pe = float(row.get(C.S_PE, -1.0))
    pb = float(row.get(C.S_PB, 99.0))
    if 0 < pe < 40:
        s += 8
    if 0 < pb < 2:
        s += 5
        
    mcap = float(row.get(C.S_MCAP, 0.0))
    if 50e8 < mcap < 500e8:
        s += 8
        
    amt = float(row.get(C.S_AMT, 0.0))
    if amt > 1e8:  
        s += 5
        
    if 1.0 < pct < 7.0:
        s += 10
    elif pct > 9.0:
        s -= 15  
        
    return min(max(s, 0.0), 100.0)


# ═════════════════════════════════════════════════════════════════════════════
# 9. 核心流水线与主控 (Pipeline & Orchestrator)
# ═════════════════════════════════════════════════════════════════════════════
def is_valid_run_time(now: datetime) -> bool:
    if IS_MANUAL:
        return True
    t = now.hour * 100 + now.minute
    # 【重磅更新：尾盘法放行】将原本 15:05 的锁解除，提前至 14:45，支持在收盘前 15 分钟介入
    return t >= 1445

def process_stock(row: pd.Series, raw_hist: pd.DataFrame, now: datetime, market_ok: bool, index_ret: float, hot_sectors_map: dict) -> Optional[tuple]:
    if len(raw_hist) < 120: return None
    
    hist = raw_hist.copy()
    if str(hist[C.H_DATE].iloc[-1]) != now.strftime('%Y-%m-%d') and is_valid_run_time(now):
        synthetic = pd.DataFrame([{
            C.H_DATE: now.strftime('%Y-%m-%d'), C.H_OPEN: float(row.get(C.S_OPEN, row[C.S_PRICE])),
            C.H_HIGH: float(row[C.S_HIGH]), C.H_LOW: float(row.get(C.S_LOW, row[C.S_PRICE])),
            C.H_CLOSE: float(row[C.S_PRICE]), C.H_VOL: float(row.get(C.S_VOL, 1.0))
        }])
        hist = pd.concat([hist, synthetic], ignore_index=True)

    if hist.iloc[-1][C.H_VOL] <= 0: return None
    
    engine = AShareTechnicals(hist)
    data = engine.get_features()
    if not data: return None

    atr_pct = (data['atr_val'] / data['close_val']) * 100
    if atr_pct > 8.0:
        return None

    data['pe'] = float(row.get(C.S_PE, 0))
    data['pb'] = float(row.get(C.S_PB, 0))
    data['mcap'] = float(row.get(C.S_MCAP, 0))
    data['vol_ratio'] = float(row.get(C.S_VR, 1.0))
    data['rs_rating'] = ((row[C.S_PRICE] / data['close_60d_ago'] - 1) * 100 - index_ret) if data['close_60d_ago'] > 0 else 0
    data['code'] = str(row[C.S_CODE])
    
    data['in_hot_sector'] = data['code'] in hot_sectors_map
    data['hot_sector_name'] = hot_sectors_map.get(data['code'], "热门")
    
    atr_stop = data['close_val'] - 2.0 * data['atr_val']
    stop = max(atr_stop, row[C.S_PRICE] * 0.88)
    stop = round(stop, 2)
    
    risk_pct = ((row[C.S_PRICE] - stop) / row[C.S_PRICE]) * 100 if row[C.S_PRICE] > 0 else 99
    
    if risk_pct > 25.0: return None 

    return (data, stop, risk_pct) 

def extract_market_context(df_raw: pd.DataFrame, c_conf: Config) -> tuple[pd.DataFrame, bool, str, float, bool, str, bool]:
    market_ok, market_msg, index_ret, market_overheated = True, "", 0.0, False
    market_regime = "NEUTRAL"
    vol_surge = False
    
    if len(df_raw) < 1000: return pd.DataFrame(), False, "API 异常，横截面数据不足", 0.0, False, market_regime, vol_surge
    
    north_flow, north_msg = fetch_northbound_flow()
    
    try:
        df_raw[C.S_PE] = pd.to_numeric(df_raw[C.S_PE], errors='coerce')
        df_raw[C.S_PB] = pd.to_numeric(df_raw[C.S_PB], errors='coerce')

        idx_df = fetch_index('sh000001')
        cl = idx_df['close']
        ma20 = cl.rolling(20).mean().iloc[-1]
        pct = (cl.iloc[-1] - cl.iloc[-2]) / cl.iloc[-2] * 100
        
        vol_col = 'volume' if 'volume' in idx_df.columns else 'amount' if 'amount' in idx_df.columns else None
        if vol_col and len(idx_df) >= 6:
            today_vol = float(idx_df[vol_col].iloc[-1])
            ma5_vol = float(idx_df[vol_col].iloc[-6:-1].mean())
            if ma5_vol > 0 and today_vol > ma5_vol * 1.25:
                vol_surge = True
        
        market_trend_ok = cl.iloc[-1] > ma20
        up_count = (df_raw[C.S_PCT] > 0).sum()
        down_count = (df_raw[C.S_PCT] < 0).sum()
        total_count = up_count + down_count
        breadth = up_count / total_count if total_count > 0 else 0.5
        vix_proxy = cl.pct_change().abs().tail(5).mean() * 100
        index_ret = ((cl.iloc[-1] / cl.iloc[-60]) - 1) * 100 if len(cl) >= 60 else 0.0
        
        zt_count = (df_raw[C.S_PCT] >= 9.0).sum() 
        dt_count = (df_raw[C.S_PCT] <= -9.0).sum() 
        total_amt = df_raw[C.S_AMT].sum() / 1e8 
        
        sentiment_addon = ""
        if zt_count > 150:
            market_overheated = True
            sentiment_addon = "\n- 🚨 **情绪熔断**：今日涨停破百市场极度狂欢，系统禁止推荐个股防踩踏！"

        if breadth < 0.25 and vix_proxy > 1.5:
            market_regime = "PANIC"
            market_state = "🧊 **恐慌冰点 (PANIC)**"
            advice = "仓位 10%-20%。系统性风险急剧释放，多看少动，仅适合轻仓左侧防守试错。"
            market_ok = False
        elif market_trend_ok and breadth > 0.6:
            market_regime = "BULL"
            market_state = "🔥 **强势多头 (BULL)**"
            advice = "仓位 60%-80%。赚钱效应极佳，资金活跃，跟随主线积极做多。"
            market_ok = True
        elif not market_trend_ok and breadth <= 0.4:
            market_regime = "BEAR"
            market_state = "🐻 **弱势空头 (BEAR)**"
            advice = "仓位 20%-30%。均线压制且空头力量主导，控制手管住回撤。"
            market_ok = False
        else:
            market_regime = "NEUTRAL"
            market_state = "⚖️ **震荡均衡 (NEUTRAL)**"
            advice = "仓位 40%-60%。指数暂无大级别风险，重个股轻大盘，不盲目追高。"
            market_ok = True

        if north_flow <= -80.0:
            market_ok = False
            market_state += " ⚠️(外资砸盘)"
            advice = "🚨 **外资大举出逃，强行压制做多情绪，建议立即防守并缩减仓位！** " + advice
        elif north_flow >= 50.0 and not market_overheated:
            market_ok = True
            market_state += " 🚀(外资抢筹)"

        fallback_warning = ""
        if C.S_PE in df_raw.columns and (df_raw[C.S_PE] == -1.0).sum() > len(df_raw) * 0.9: 
            fallback_warning = "\n\n> ⚠️ **数据源降级警报**\n> 频繁测试触发东方财富接口临时限制，已切至新浪备用源。基本面过滤(市盈率/量比等)暂时失效，请自行排雷！"

        market_msg = (
            f"### 📊 大盘多维体检\n"
            f"- **上证指数**：`{cl.iloc[-1]:.2f}` (今日 **{pct:+.2f}%**)\n"
            f"- **综合判定**：{market_state}\n"
            f"- **市场广度**：红盘 `{up_count}` 家 / 绿盘 `{down_count}` 家 (涨停 `{zt_count}` / 跌停 `{dt_count}`)\n"
            f"- **两市量能**：约 `{total_amt:.0f}` 亿元{sentiment_addon}{north_msg}\n\n"
            f"**💡 仓位建议**：{advice}{fallback_warning}"
        )
    except Exception as e:
        log.warning(f"宏观状态解析失败: {e}")
        market_msg = f"大盘深度解析由于网络原因失败: {e}\n"
    
    df = df_raw.dropna(subset=list(c_conf.REQUIRED_COLS))
    df = df[~df[C.S_NAME].str.contains('ST|退')]
    return df, market_ok, market_msg, index_ret, market_overheated, market_regime, vol_surge

def get_signals() -> tuple[list[Signal], list, set, int, str, int]:
    now = datetime.now(TZ_BJS)
    run_mode = os.environ.get('RUN_MODE', 'normal')
    
    log.info('🚀 防呆长线安全级·盘后复盘引擎启动...')
    if not IS_MANUAL and not is_valid_run_time(now): 
        return [], [], set(), 0, "", 0

    pushed = load_pushed_state() 

    try:
        df_raw = fetch_spot()
    except Exception as e:
        log.error(f"❌ 核心横截面行情获取失败: {e}")
        return [], [], pushed, 0, f"⚠️ **行情接口异常，体检中断**: {e}", 0

    c_conf = Config()
    df_clean, m_ok, m_msg, idx_ret, m_overheated, m_regime, vol_surge = extract_market_context(df_raw, c_conf)

    if run_mode == 'market_only':
        log.info("🤖 [大盘体检模式] 完毕，退出个股运算。")
        return [], [], pushed, 0, m_msg, len(df_raw)

    paper_trades, win_stats = load_and_update_paper_trades(df_raw)
    hot_sectors_map = fetch_hot_sectors()

    if df_clean.empty:
        return [], [], pushed, 0, m_msg, 0

    core_pool = fetch_core_pool()
    if core_pool:
        df_clean = df_clean[df_clean[C.S_CODE].isin(core_pool)]
        log.info(f"💎 已开启【核心优质股池】模式，限定扫描 {len(core_pool)} 只国家队核心及高弹性成分股。")

    is_fallback = (df_clean[C.S_PE] == -1.0).sum() > len(df_clean) * 0.9
    pe_cond = (df_clean[C.S_PE] > c_conf.MIN_PE) | is_fallback
    
    mask = (df_clean[C.S_PCT] >= c_conf.MIN_PCT_CHG) & \
           (df_clean[C.S_PRICE] <= c_conf.MAX_PRICE) & \
           (df_clean[C.S_MCAP].between(c_conf.MIN_CAP, c_conf.MAX_CAP)) & \
           (df_clean[C.S_TURN].between(c_conf.MIN_TURNOVER, c_conf.MAX_TURNOVER)) & \
           pe_cond & (df_clean[C.S_PE] <= c_conf.MAX_PE) & \
           (df_clean[C.S_PB] > 0) & (df_clean[C.S_PB] <= 30.0) & \
           (~df_clean[C.S_CODE].str.startswith(('688', '8', '4', '9'))) & \
           (df_clean[C.S_HIGH] > df_clean[C.S_LOW]) 
    
    if C.S_VR in df_clean.columns and not is_fallback:
        mask &= df_clean[C.S_VR].between(c_conf.MIN_VOL_RATIO, c_conf.MAX_VOL_RATIO)

    recent_pushed_codes = {str(c) for c in df_clean[C.S_CODE] if is_recently_pushed(str(c), pushed)}
    pool = df_clean[mask].pipe(lambda d: d[~d[C.S_CODE].isin(recent_pushed_codes)]).copy()
    
    if pool.empty: return [], [], pushed, len(df_clean), m_msg, len(df_clean)
    
    if len(pool) > 400:
        log.info(f"💡 触发防爆流截断，基于 Spot 截面数据执行廉价预筛分，保留前 400 只高潜标的参与决选。")
        pool['_pre_score'] = pool.apply(prescreen_score, axis=1)
        pool = pool.sort_values(by='_pre_score', ascending=False).head(400)
        pool = pool.drop(columns=['_pre_score'])

    confirmed_data = [] 
    watchlist_data = [] 
    
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=450)).strftime('%Y%m%d')
    
    ex2 = ThreadPoolExecutor(max_workers=4)
    futures = {ex2.submit(fetch_hist, r[C.S_CODE], start_s, end_s): r for _, r in pool.iterrows()}
    
    try:
        for f in as_completed(futures, timeout=1200): 
            row = futures[f]
            try:
                hist = f.result()
                result = process_stock(row, hist, now, m_ok, idx_ret, hot_sectors_map)
                if result:
                    data, stop, risk = result
                    
                    score, level, reas = apply_scoring(data, now, m_regime, vol_surge, win_stats)
                    
                    if score >= 70: 
                        target1_price = calc_target_price(row[C.S_PRICE], stop, data)
                        
                        money_msg = format_money_risk_msg(row[C.S_PRICE], stop, target1_price)
                        tranche_msg = generate_tranche_plan(row[C.S_PRICE], score, m_ok, m_overheated)
                        plan_b_msg = generate_plan_b(row[C.S_PRICE], stop, data['ma20_val'])
                        hold_msg = generate_hold_period(data['adx'], data['price_pct'], data['has_chip_break'])
                        
                        confirmed_data.append(Signal(
                            code=row[C.S_CODE], name=row[C.S_NAME], price=row[C.S_PRICE],
                            pct_chg=f"{row[C.S_PCT]}%", score=score, level=level,
                            trigger_time=now.strftime('%H:%M'), reasons=reas,
                            stop_loss=round(stop, 2), target1=target1_price,
                            ma10=round(data['ma10_val'], 2),
                            money_risk_msg=money_msg, tranche_plan_msg=tranche_msg,
                            plan_b_msg=plan_b_msg, hold_period_msg=hold_msg
                        ))
                    elif score >= 60:  
                        watchlist_data.append((row[C.S_NAME], row[C.S_CODE], score, row[C.S_PRICE]))
                        
            except Exception as e:
                log.debug(f"计算个股 {row[C.S_CODE]} 时发生异常或被过滤: {e}")
                pass
    except FuturesTimeoutError:
        log.warning("⚠️ 后台运算达到极值，提前熔断保存已有成果。")
    finally:
        ex2.shutdown(wait=False, cancel_futures=True)

    confirmed_data.sort(key=lambda x: (x.score, x.code), reverse=True)
    
    # ── 【核心优化落地：简单组合控制 (Max 2 Per Sector)】 ──
    final_confirmed = []
    sector_counts = {}
    for s in confirmed_data:
        sector = hot_sectors_map.get(s.code)
        if sector:
            # 如果该板块已有 2 只上榜，直接拦截（防止同板块集中爆破）
            if sector_counts.get(sector, 0) >= 2:
                continue
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        final_confirmed.append(s)
        
    watchlist_data.sort(key=lambda x: (x[2], x[1]), reverse=True) 
    
    # 仅将决选且展出给用户的 Top 10 个股记录进状态锁与模拟盘账本
    for s in final_confirmed[:10]:
        cd_days = 1 if s.score >= 85 else 3
        expire_dt = now + timedelta(days=cd_days)
        pushed[s.code] = expire_dt.strftime('%Y-%m-%d')
        
        paper_trades.append({
            'date': _today_str(),
            'code': s.code,
            'score_bucket': get_score_bucket(s.score),
            'buy_price': s.price,
            'target': s.target1,
            'stop': s.stop_loss,
            'status': 'PENDING'
        })
    
    save_paper_trades(paper_trades)

    return final_confirmed, watchlist_data, pushed, len(pool), m_msg, len(df_clean)


# ═════════════════════════════════════════════════════════════════════════════
# 10. 钉钉网关与推送 (Webhook & Notification)
# ═════════════════════════════════════════════════════════════════════════════
def send_dingtalk(signals: list[Signal], watchlist: list, total_pool: int, total_market: int, market_msg: str) -> None:
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook:
        log.error("❌ 未配置 DINGTALK_WEBHOOK 环境变量，取消推送！")
        return
    
    now_ts = datetime.now(TZ_BJS)
    now_str = now_ts.strftime('%Y-%m-%d %H:%M')
    run_mode = os.environ.get('RUN_MODE', 'normal')
    
    header = (
        f"## 🤖 AI量化保姆级盘后总结\n"
        f"> **{now_str}**\n>\n"
        f"> ⚠️ **郑重声明**：本报告由量化模型自动生成，仅供技术交流与策略复盘，**绝不构成任何投资建议**。股市有风险，入市需谨慎，盈亏请自负。\n\n"
    )
    if run_mode == 'market_only':
        header = f"## 🤖 AI量化大盘深度体检\n> **{now_str}**\n\n"
    elif run_mode != 'market_only' and total_market > 0:
        pass_rate = len(signals) / max(total_pool, 1) * 100 if total_pool > 0 else 0
        header += f"**🔬 漏斗数据**：全市场白名单 `{total_market}` 只，异动提取 `{total_pool}` 只，完美过线 `{len(signals)}` 只 (B+级以上优选率 **{pass_rate:.1f}%**)\n\n"
        
    if market_msg:
        header += f"{market_msg}\n\n---\n\n"

    if run_mode == 'market_only':
        content = header + "✅ 大盘分析播报完毕，本次任务短路了全量个股运算。"
    elif "接口异常" in market_msg or "网络原因失败" in market_msg:
        content = header + "⚠️ 今日部分个股数据扫描因接口受限中断，已为您提供核心大盘分析参考。"
    elif not signals and not watchlist:
        if not PUSH_EMPTY: return
        content = f"{header}✅ **机器体检结果**：今日未发现形态完全符合安全边际的标的，别乱买，建议**空仓防守**！"
    else:
        if signals:
            MAX_DISPLAY = 5
            display_signals = signals[:MAX_DISPLAY]
            hidden_count = len(signals) - len(display_signals)
            
            avg_score = sum(s.score for s in display_signals) / len(display_signals)
            quality_tag = "🥇 **绝佳** (建议严格按剧本执行)" if avg_score >= 80 \
                else "🥈 **尚可** (建议严格限价，减半仓位)"
                
            content = header + f"### 📈 今日核心精选 (Top 5)\n**精选均分：{avg_score:.0f} 分** | {quality_tag}\n\n"
            
            cold_gate = (
                "> **🛑 买入前冷静自检（30秒）**\n"
                "> 1. 这笔闲钱 **3年内** 绝对不会急用？\n"
                "> 2. 就算不小心 **亏掉30%** 也不会睡不着？\n"
                "> 3. 能管住手，**绝不因为下跌反复盯盘**？\n"
                "> \n"
                "> *✅ 三项全对 ➡️ 允许按下方计划执行*\n"
                "> *❌ 有一项不对 ➡️ 请立即把买入预算砍掉一半！*\n\n"
                "---\n\n"
            )
            content += cold_gate
            
            parts = []
            for s in display_signals:
                warn_msg = "> ⚡ **【风险警示】** 该股为创业板(波动±20%)，心脏不好请务必**缩减仓位**！\n\n" if str(s.code).startswith('300') else ""
                prefix = '1' if str(s.code).startswith('6') else '0'
                tdx_market = 'SH' if str(s.code).startswith('6') else 'SZ' 
                
                sina_market = 'sh' if str(s.code).startswith('6') else 'sz'
                kline_url = f"http://image.sinajs.cn/newchart/weekly/n/{sina_market}{s.code}.gif"
                
                parts.append(
                    f"#### 🎯 {s.name} (`{s.code}`)\n"
                    f"{warn_msg}"
                    f"- **综合评级**：`{s.score}` 分 {s.level}\n"
                    f"- **今日收盘**：`¥{s.price}` ({s.pct_chg})\n\n"
                    f"![大周期周K线图]({kline_url})\n\n"
                    f"**💡 为什么机器选出它？**\n{s.reasons}\n\n"
                    f"**🛡️ 小白专属操作剧本**\n"
                    f"{s.hold_period_msg}\n"
                    f"{s.money_risk_msg}\n\n"
                    f"{s.tranche_plan_msg}\n\n"
                    f"{s.plan_b_msg}\n\n"
                    f"> **纪律红线**\n"
                    f"> 🎯 **止盈**：收盘跌破 `¥{s.ma10}` (10日线)，立刻卖出一半保住利润！\n"
                    f"> 🚫 **防守**：明日开盘直接高开 **> 4%** 说明资金抢跑，直接放弃，绝不追高！\n\n"
                    f"[🔗 点击跳转东方财富 App 查阅详情](https://quote.eastmoney.com/unify/r/{prefix}.{s.code})\n\n"
                    f"*📌 通达信看盘助手：复制代码 `{s.code}` 后打开通达信 App 即可弹出*"
                )
            content += "\n\n---\n\n".join(parts)
            
            if hidden_count > 0:
                hidden_names = "、".join([f"{s.name}(`{s.code}` **{s.score}分**)" for s in signals[MAX_DISPLAY:]])
                content += f"\n\n---\n*⚠️ 受限于篇幅，以下 **{hidden_count} 只** 达标个股被系统折叠（已按分数排序）：*\n> {hidden_names}"
                
        else:
            content = header + "✅ 今日未发现 B+ 级以上核心机会，正式推荐列表空仓防守中。\n"

        if watchlist:
            watch_lines = "\n".join(
                f"- `{code}` **{name}** (¥{price}) 得分: **{score}**"
                for name, code, score, price in watchlist[:5]
            )
            content += (
                f"\n\n---\n### 👁️ 候补观察池（只看不买）\n"
                f"{watch_lines}\n\n"
                f"*注：以上标的评级不足 70 分，系统判断波动或风险偏大，暂不提供操作剧本。待其评级升至发车线后再考虑介入。*"
            )
        
        content += (
            "\n\n---\n### 🤔 每日灵魂拷问\n"
            "如果明天买入的股票跌了 5%，我会焦虑得睡不着觉吗？\n\n"
            "> **如果会，请把你准备买入的金额【再砍掉一半】！投资是为了生活更好，不是花钱找罪受。**"
        )

    try:
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': '🤖 AI量化盘后提醒',
                'text': content
            }
        }
        res = requests.post(webhook, json=payload, timeout=10)
        res_dict = res.json()
        
        if res_dict.get('errcode', 0) != 0:
            log.error(f"❌ 钉钉接口拒绝推送，请检查「自定义关键词」是否匹配！返回信息: {res_dict}")
        else:
            log.info(f"✅ 推送成功 ({len(signals)}正式 / {len(watchlist)}观察)")
            
    except Exception as e:
        log.error(f"❌ 推送网络请求失败: {e}")

if __name__ == '__main__':
    try:
        sigs, watch, pushed, pool_size, m_msg, total_mkt = get_signals()
        send_dingtalk(sigs, watch, pool_size, total_mkt, m_msg)
        if sigs: save_pushed_state(pushed)
    except Exception as e:
        log.critical(f"系统崩溃: {e}", exc_info=True)
        webhook_url = os.environ.get('DINGTALK_WEBHOOK')
        if webhook_url:
            error_msg = f"🚨 **AI量化引擎崩溃告警**\n\n**时间**: {_today_str()}\n**环境**: GitHub Actions\n**异常信息**: {str(e)[:300]}..."
            try:
                requests.post(webhook_url, json={"msgtype": "markdown", "markdown": {"title": "系统崩溃告警", "text": error_msg}}, timeout=5)
            except:
                pass
