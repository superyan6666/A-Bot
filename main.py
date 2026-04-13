import akshare as ak
import pandas as pd
import numpy as np
import requests
import os
import time
import json
import socket
import logging
import gc
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
TZ_BJS = pytz.timezone('Asia/Shanghai') # 设置时区为北京时间
STATE_FILE = 'pushed_state.json'        # 盘中去重记忆文件
# 识别是否为 GitHub 手动点击触发
IS_MANUAL_RUN = os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
# 【优化点 8】：识别是否在 GitHub Actions 环境中运行
IS_GITHUB_ACTION = os.environ.get('GITHUB_ACTIONS') == 'true'
# 【优化点 11】：是否允许推送空结果（读取环境变量，默认为 False）
PUSH_EMPTY_RESULT = os.environ.get('PUSH_EMPTY_RESULT', 'false').lower() in ['true', '1', 'yes']

# 【优化点 6】：通过环境变量动态控制日志级别，防 Actions 日志过长
LOG_LEVEL_STR = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

# 【优化点 13】：配置标准 Logging 模块，告别 print，支持分级与时间戳
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 【优化点 12】：统一策略参数配置类，方便后期调整或外置为 JSON
class Config:
    MIN_MARKET_CAP = 30 * 10**8      # 流通市值下限（默认 30亿）
    MAX_MARKET_CAP = 300 * 10**8     # 流通市值上限（默认 300亿）
    MIN_PE = 0                       # 动态市盈率下限
    MAX_PE = 60                      # 动态市盈率上限
    MIN_PB = 0                       # 市净率下限
    MAX_PB = 5                       # 市净率上限
    MIN_TURN_OVER = 3.0              # 最低换手率阈值（%）
    MIN_AMOUNT = 100000000           # 最低成交额阈值（默认 1亿）
    MIN_PCT_CHG = 4.5                # 最低涨跌幅阈值（%）
    MIN_VOL_RATIO = 1.2              # 最低量比阈值
    MAX_VOL_RATIO = 5.0              # 最高量比阈值

# 设置全局 socket 超时时间(10秒)
socket.setdefaulttimeout(10.0)

# ================= 辅助函数 =================
def get_bjs_time():
    return datetime.now(TZ_BJS)

def is_trading_time(now):
    time_val = now.hour * 100 + now.minute
    if (925 <= time_val <= 1130) or (1300 <= time_val <= 1500):
        return True
    return False

def get_passed_trading_mins(now):
    if now.hour == 9 and now.minute >= 30:
        return now.minute - 30
    elif now.hour == 10:
        return 30 + now.minute
    elif now.hour == 11 and now.minute <= 30:
        return 90 + now.minute
    elif 13 <= now.hour < 15:
        return 120 + (now.hour - 13) * 60 + now.minute
    elif now.hour >= 15:
        return 240
    return 1 

def load_pushed_state():
    today_str = get_bjs_time().strftime("%Y-%m-%d")
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            if state.get('date') == today_str:
                return set(state.get('pushed_codes', []))
        except:
            pass
    return set()

def save_pushed_state(pushed_codes):
    today_str = get_bjs_time().strftime("%Y-%m-%d")
    state = {'date': today_str, 'pushed_codes': list(pushed_codes)}
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def calculate_tdx_dma(close_series, a_series):
    c_arr = close_series.to_numpy()
    a_arr = a_series.to_numpy()
    dma_arr = np.zeros_like(c_arr)
    last_dma = c_arr[0]
    for i in range(len(c_arr)):
        if np.isnan(a_arr[i]):
            dma_arr[i] = last_dma
        else:
            dma_arr[i] = a_arr[i] * c_arr[i] + (1 - a_arr[i]) * last_dma
            last_dma = dma_arr[i]
    return pd.Series(dma_arr, index=close_series.index)

def calculate_atr_adx(hist, period=14):
    """纯 Pandas 向量化计算真实波动率(ATR)与平均趋向指数(ADX)"""
    high = hist['最高']
    low = hist['最低']
    close = hist['收盘']
    
    # 1. 计算 True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # 2. 计算 +DM 和 -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # 3. 计算 +DI 和 -DI
    plus_di = 100 * (pd.Series(plus_dm, index=hist.index).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=hist.index).rolling(window=period).mean() / atr)
    
    # 4. 计算 DX 和 ADX (严重错误修复：防范除零异常)
    denom = plus_di + minus_di
    # replace 0 with NaN 避免运行时除零警告，计算后再 fillna 补回 0
    dx = (abs(plus_di - minus_di) / denom.replace(0, np.nan)).fillna(0) * 100
    adx = dx.rolling(window=period).mean()
    
    return atr, adx

def fetch_hist_data(code, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df is None or df.empty:
                raise ValueError("历史数据为空，触发退避重试")
                
            # 数据完整性校验，拦截脏数据或接口变异
            keep_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
            if not all(c in df.columns for c in keep_cols):
                raise ValueError(f"历史数据缺失核心列，触发退避")
                
            available_cols = [c for c in keep_cols if c in df.columns]
            return df[available_cols].copy()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

def fetch_single_stock_data(code, start_date_str, end_date_str):
    """用于线程池的单个股票数据获取包装函数"""
    hist = fetch_hist_data(code, start_date_str, end_date_str)
    return code, hist

def fetch_spot_with_retry(retries=3, delay=2):
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                raise ValueError("实时行情返回数据为空")
            return df
        except (requests.exceptions.RequestException, socket.error, TimeoutError, ValueError) as e:
            logger.warning(f"实时行情获取失败 [{type(e).__name__}] (尝试 {attempt+1}/{retries})，等待 {delay} 秒后重试...")
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            raise Exception(f"🚨 遇到不可恢复的代码逻辑异常 [{type(e).__name__}]: {e}")
            
    raise Exception("🚨 实时行情获取彻底失败，请检查网络或 Akshare 接口状态！")

def check_market_environment():
    """确保大盘处于安全做多区间"""
    try:
        # 【严重错误修复】：修正为可靠性更高的 _em 后缀接口
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        if df is None or len(df) < 20:
            return True, "大盘数据获取失败，默认放行"
        
        today_close = df['close'].iloc[-1]
        yesterday_close = df['close'].iloc[-2]
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        
        pct_chg = (today_close - yesterday_close) / yesterday_close * 100
        
        # 指数在20日线上，或者今天涨跌幅 > -0.5% (允许小跌，拒绝单边暴跌)
        is_ok = (today_close > ma20) or (pct_chg > -0.5)
        
        status_msg = f"上证指数 {today_close:.2f} (涨幅 {pct_chg:.2f}%)，20日线 {ma20:.2f}"
        if not is_ok:
            status_msg += " ⚠️ 系统性风险较高"
        return is_ok, status_msg
    except Exception as e:
        logger.warning(f"大盘环境检测异常: {e}")
        return True, "大盘状态未知(默认放行)"

def get_all_sector_strength():
    """获取全市场行业板块实时涨幅"""
    try:
        df = ak.stock_board_industry_spot_em()
        return dict(zip(df['板块名称'], df['涨跌幅']))
    except Exception as e:
        logger.warning(f"获取板块强度失败: {e}")
        return {}

def get_stock_sector(code):
    try:
        df = ak.stock_individual_info_em(symbol=code)
        industry = df[df['item'] == '行业']['value'].values[0]
        if pd.isna(industry) or industry == "--" or not str(industry).strip():
            return ""
        return industry
    except Exception:
        return ""

def calculate_target_points(current_price, today_low, ma20, max_250_high, atr):
    """结合个股真实波动率与均线支撑的双重智能止损"""
    # 1. 纯技术面支撑止损 (原逻辑)
    support_stop = min(today_low, ma20) * 0.99
    
    # 2. 动态波动率止损 (现价下移 1.5 倍 ATR)
    tech_stop = current_price - 1.5 * atr
    
    # 3. 强强联合：取两者中较高的一个（防守更紧凑，贴合支撑位）
    stop_loss = max(tech_stop, support_stop)
    
    # 兜底防护：防止异常数据导致止损价高于现价
    if stop_loss >= current_price:
        stop_loss = current_price * 0.95
        
    stop_loss = round(stop_loss, 2)
    
    risk_value = current_price - stop_loss
    risk_percent = (risk_value / current_price) * 100 
    
    target_1 = round(current_price + risk_value * 2.0, 2)
    target_2 = round(max(max_250_high, target_1 * 1.15), 2)
    return stop_loss, target_1, target_2, risk_percent

def get_position_advice(score, risk_percent, market_ok):
    """引入大盘环境风控，大盘弱势强制底仓"""
    if score >= 85:
        base_pos = 30
        tag = "重仓狙击"
    elif score >= 75:
        base_pos = 20
        tag = "标准配置"
    else:
        base_pos = 10
        tag = "轻仓试错"

    # 大盘环境惩罚
    market_tag = ""
    if not market_ok:
        base_pos = base_pos // 2
        market_tag = " [大盘弱势半仓]"

    if risk_percent > 8.0:
        final_pos = base_pos // 2
        return f"⚠️ 建议 {final_pos}% 仓位 (原定{tag}{market_tag}，但单笔止损风险高达 {risk_percent:.1f}%，触发双重减半！)"
    else:
        return f"⚖️ 建议 {base_pos}% 仓位 ({tag}{market_tag}，单笔止损风险控制在极佳的 {risk_percent:.1f}%)"

def generate_reason_and_score(price_percentile, vol_ratio, vcp_amplitude, upper_shadow_ratio, has_zt_gene, has_macd_div, has_rsi_oversold, has_chip_breakthrough, has_obv_breakout, has_60d_breakout, has_gap_up, has_pullback_confirm, adx):
    score = 40
    reasons = []

    # 因子得分动态权重化，大环境感知
    if adx > 30:
        trend_weight = 1.5
        reversal_weight = 0.8
        reasons.append(f"🧭 【顺势增强】当前ADX({adx:.1f})显示趋势强劲，系统已自动调高趋势类因子权重(1.5x)。")
    elif adx < 20:
        trend_weight = 0.8
        reversal_weight = 1.5
        reasons.append(f"🧭 【震荡增强】当前ADX({adx:.1f})显示处于震荡市，系统已自动调高反转/超卖类因子权重(1.5x)。")
    else:
        trend_weight = 1.0
        reversal_weight = 1.0

    perc = price_percentile * 100
    if perc < 15:
        score += int(15 * reversal_weight)
        reasons.append(f"🟢 【波段潜伏】处于近1年极度冰点(分位{perc:.1f}%)，下方支撑极强，性价比爆棚。")
    elif perc < 30:
        score += int(10 * reversal_weight)
        reasons.append(f"🟢 【波段底部】处于近1年底部区域(分位{perc:.1f}%)，趋势反转确立。")
    else:
        score += 5
        reasons.append(f"🟡 【中位起涨】处于近1年中低位(分位{perc:.1f}%)，属于波段加速期。")

    vcp = vcp_amplitude * 100
    if vcp < 10:
        score += 10
        reasons.append(f"🟣 【洗盘极致】突破前横盘振幅仅{vcp:.1f}%，筹码真空，拉升阻力极小。")
    elif vcp < 15:
        score += 5
        reasons.append(f"🟣 【温和洗盘】突破前蓄势充分(振幅{vcp:.1f}%)。")

    if has_zt_gene:
        score += 15
        reasons.append("🔥 【主力基因】近期有涨停记录，股性极为活跃，属于资金记忆抱团标的。")

    if 2.0 <= vol_ratio <= 3.0:
        score += 10
        reasons.append(f"🔵 【完美倍量】温和放量{vol_ratio:.1f}倍，换手极其健康。")
    elif 1.5 <= vol_ratio < 2.0 or 3.0 < vol_ratio <= 4.0:
        score += 5
        reasons.append(f"🔵 【显著放量】今日放量{vol_ratio:.1f}倍，资金态度坚决。")

    shadow = upper_shadow_ratio * 100
    if shadow < 5:
        score += 10
        reasons.append("🔴 【绝对强势】光头阳线突围，盘中抛压被通吃，明天必有溢价。")
    elif shadow < 15:
        score += 5
        reasons.append(f"🔴 【实体推升】上影线极短(比例{shadow:.1f}%)，多头控盘能力强。")

    if has_chip_breakthrough:
        score += int(15 * trend_weight)
        reasons.append("🏔️ 【跨越筹码峰】今日强势突破近半年核心筹码密集区，上方套牢盘抛压一扫而空，主升浪空间彻底打开！")
        
    if has_macd_div:
        score += int(10 * reversal_weight)
        reasons.append("💥 【底背离共振】近期呈现标准MACD底背离形态，杀跌动能枯竭，反转信号极其强烈。")
        
    if has_rsi_oversold:
        score += int(5 * reversal_weight)
        reasons.append("📉 【超卖反弹】前期RSI曾触极度超卖区，空头情绪宣泄完毕，底部夯实。")

    if has_60d_breakout:
        score += int(15 * trend_weight)
        reasons.append("🚀 【箱体突破】一举跃过近60日波段前高，呈现经典'N字型'主升浪突破形态，上攻意愿极其坚决！")
        
    if has_obv_breakout:
        score += int(10 * trend_weight)
        reasons.append("💸 【量能先行】OBV(能量潮)创出阶段新高，量在价先，揭示主力资金近期一直在暗中吃货吸筹！")
        
    if has_gap_up:
        score += int(5 * trend_weight)
        reasons.append("⚡ 【跳空抢筹】今日早盘呈现跳空高开缺口且未回补，集合竞价阶段多头已呈碾压逼空之势。")

    if has_pullback_confirm:
        score += 15
        reasons.append("🪃 【回踩确认】昨日极致缩量回踩均线(洗出浮筹)，今日放量反包突破，量价配合天衣无缝，买点极稳！")

    # 引入因子相关性惩罚，消除同源共线性冗余
    vol_signals_count = sum([1 if vol_ratio >= 2.0 else 0, has_obv_breakout, has_chip_breakthrough])
    if vol_signals_count >= 2:
        penalty = int(score * 0.1) 
        score -= penalty
        reasons.append(f"⚠️ 【风控折价】量能/资金面指标高度共振(触发{vol_signals_count}项)，触发防重叠计分机制(-{penalty}分)。")

    score = min(score, 100)

    if score >= 85:
        level = "⭐⭐⭐⭐⭐ [S级极品]"
    elif score >= 75:
        level = "⭐⭐⭐⭐ [A级强势]"
    else:
        level = "⭐⭐⭐ [B级标准]"

    return score, level, "\n".join(reasons)

# ================= 主筛选逻辑 =================
def get_signals():
    now = get_bjs_time()
    logger.info("🚀 开始启动量化扫描引擎...")
    
    if not IS_MANUAL_RUN:
        if not is_trading_time(now) and now.hour < 15:
            logger.info("当前时间不在交易时段，自动休眠。")
            return pd.DataFrame(), set(), 0
    else:
        logger.info("💡 检测到手动触发 (workflow_dispatch)，正在强制执行全市场扫描...")

    already_pushed = load_pushed_state()
    
    # 大盘环境定调
    market_ok, market_msg = check_market_environment()
    logger.info(f"📈 大盘宏观研判: {market_msg}")
    
    # 埋点测速
    t0 = time.time()
    df_spot = fetch_spot_with_retry()
    t1 = time.time()
    logger.info(f"⏱️ 节点耗时 -> 实时行情获取: {t1 - t0:.2f} 秒")
    
    base_required_cols = ['最新价', '最高', '涨跌幅', '换手率', '成交额', '代码', '名称', '流通市值', '市盈率-动态', '市净率']
    missing_cols = [col for col in base_required_cols if col not in df_spot.columns]
    
    if missing_cols:
        logger.error(f"严重警告：实时行情缺失核心基础列 {missing_cols}，无法继续筛选，请检查接口！")
        return pd.DataFrame(), already_pushed, 0
    
    subset_cols = base_required_cols.copy()
    if '量比' in df_spot.columns:
        subset_cols.append('量比')
    else:
        logger.warning("提示：行情中未检测到 '量比' 字段，将自动降级跳过量比过滤。")
        
    df_spot = df_spot.dropna(subset=subset_cols)
    df_spot = df_spot[~df_spot['名称'].str.contains('ST|退')]
    df_spot = df_spot[~df_spot['代码'].isin(already_pushed)]
    
    # 应用外部配置阈值
    market_cap_condition = (df_spot['流通市值'] >= Config.MIN_MARKET_CAP) & (df_spot['流通市值'] <= Config.MAX_MARKET_CAP)
    fundamental_condition = (df_spot['市盈率-动态'] > Config.MIN_PE) & (df_spot['市盈率-动态'] < Config.MAX_PE) & (df_spot['市净率'] > Config.MIN_PB) & (df_spot['市净率'] < Config.MAX_PB)
    turnover_condition = df_spot['换手率'] >= Config.MIN_TURN_OVER       
    amount_condition = df_spot['成交额'] >= Config.MIN_AMOUNT    
    rise_condition = df_spot['涨跌幅'] >= Config.MIN_PCT_CHG           
    
    pool_condition = rise_condition & market_cap_condition & fundamental_condition & turnover_condition & amount_condition
    
    if '量比' in df_spot.columns:
        # 量比阈值动态化，根据 A 股 U 型成交特征调整
        current_min_vr = Config.MIN_VOL_RATIO
        current_max_vr = Config.MAX_VOL_RATIO
        time_val = now.hour * 100 + now.minute
        
        if time_val <= 1030:
            current_max_vr = 8.0  
            logger.info("🌅 当前为早盘高波动期，系统自动将量比上限放宽至 8.0")
        elif 1100 <= time_val <= 1330:
            current_min_vr = 1.0  
            logger.info("☕ 当前为午间平淡期，系统自动将量比下限调低至 1.0")
            
        vr_condition = (df_spot['量比'] >= current_min_vr) & (df_spot['量比'] <= current_max_vr) 
        pool_condition = pool_condition & vr_condition
        
    target_pool = df_spot[pool_condition].copy()
    total_count = len(target_pool)
    logger.info(f"✅ 初筛完毕！发现 {total_count} 只符合【高活跃+强异动+基本面健康】的精选股票。")
    
    if total_count == 0:
        return pd.DataFrame(), already_pushed, 0

    end_date_str = now.strftime("%Y%m%d")
    start_date_str = (now - timedelta(days=400)).strftime("%Y%m%d")
    
    # --- 并发拉取历史数据 ---
    t2 = time.time()
    
    # 环境降级策略
    if IS_GITHUB_ACTION:
        workers_count = min(4, max(2, total_count // 4))
    else:
        workers_count = min(12, max(4, total_count // 2))
        
    logger.info(f"🔄 正在启动并发拉取 {total_count} 只股票的历史数据 (动态线程数: {workers_count})，请稍候...")
    codes = target_pool['代码'].tolist()
    hist_dict = {}
    
    with ThreadPoolExecutor(max_workers=workers_count) as executor:
        future_to_code = {
            executor.submit(fetch_single_stock_data, code, start_date_str, end_date_str): code
            for code in codes
        }
        for future in as_completed(future_to_code):
            try:
                code_res, hist = future.result(timeout=5.0)
                if hist is not None and len(hist) >= 250:
                    hist_dict[code_res] = hist
            except Exception:
                pass
                
    t3 = time.time()
    logger.info(f"⏱️ 节点耗时 -> 历史数据并发拉取: {t3 - t2:.2f} 秒 (成功获取 {len(hist_dict)} 只有效数据)")

    final_signals = []
    
    # --- 核心量价模型测算 ---
    t4 = time.time()
    for idx, (index, row) in enumerate(target_pool.iterrows(), 1):
        
        if total_count >= 10 and idx % max(1, total_count // 10) == 0:
            logger.debug(f"⚙️ 指标测算进度: {idx}/{total_count} ({(idx/total_count)*100:.0f}%)")
            
        code = row['代码']
        name = row['名称']
        
        if code not in hist_dict:
            continue
            
        try:
            hist = hist_dict[code].copy()
            
            hist['PCT_CHG'] = (hist['收盘'] - hist['收盘'].shift(1)) / hist['收盘'].shift(1) * 100
            hist['MA10'] = hist['收盘'].rolling(window=10).mean()
            hist['MA20'] = hist['收盘'].rolling(window=20).mean()
            hist['MA60'] = hist['收盘'].rolling(window=60).mean()
            hist['MA250'] = hist['收盘'].rolling(window=250).mean()
            hist['MA5_V'] = hist['成交量'].rolling(window=5).mean()
            hist['MA20_V'] = hist['成交量'].rolling(window=20).mean()
            
            hist['EMA12'] = hist['收盘'].ewm(span=12, adjust=False).mean()
            hist['EMA26'] = hist['收盘'].ewm(span=26, adjust=False).mean()
            hist['DIF'] = hist['EMA12'] - hist['EMA26']
            hist['DEA'] = hist['DIF'].ewm(span=9, adjust=False).mean()
            
            hist['CC'] = abs((2 * hist['收盘'] + hist['最高'] + hist['最低']) / 4 - hist['MA20']) / hist['MA20']
            hist['DD'] = calculate_tdx_dma(hist['收盘'], hist['CC'])
            hist['上'] = 1.07 * hist['DD']
            
            atr, adx = calculate_atr_adx(hist, period=14)
            hist['ATR'] = atr
            hist['ADX'] = adx
            
            hist['REF_C'] = hist['收盘'].shift(1)
            hist['REF_上'] = hist['上'].shift(1)
            
            today = hist.iloc[-1]
            
            if pd.isna(today['ATR']) or pd.isna(today['ADX']):
                continue
                
            min_1y = hist['最低'].min()
            max_1y = hist['最高'].max()
            price_range = max_1y - min_1y
            
            if price_range <= 0: continue
            
            price_percentile = (today['收盘'] - min_1y) / price_range
            recent_10_high = hist['最高'].iloc[-11:-1].max()
            recent_10_low = hist['最低'].iloc[-11:-1].min()
            vcp_amplitude = (recent_10_high - recent_10_low) / recent_10_low
            
            body_length = today['收盘'] - today['开盘']
            if body_length <= 0: continue
            upper_shadow_ratio = (today['最高'] - today['收盘']) / body_length
            
            vol_ratio = row.get('量比', 1.0)
            if pd.isna(vol_ratio): vol_ratio = 1.0
                
            has_zt_gene = (hist['PCT_CHG'].iloc[-61:-1] >= 9.5).any()
            
            current_adx = today['ADX']
            prev_adx = hist['ADX'].iloc[-2]
            prev2_adx = hist['ADX'].iloc[-3]
            is_strong_trend = (current_adx > 25) or ((current_adx > prev_adx) and (prev_adx > prev2_adx))
            
            is_bull_trend = (today['收盘'] >= today['MA250']) and (today['MA20'] >= today['MA60']) and is_strong_trend
            is_macd_bull = (today['DIF'] > 0) and (today['DEA'] > 0)
            
            breakout_strength = (today['收盘'] - today['上']) / today['ATR'] if today['ATR'] > 0 else 0
            is_strong_breakout = (today['收盘'] > today['开盘']) and (breakout_strength >= 0.3) and (today['成交量'] > today['MA20_V'] * 1.5)
            
            is_cross = (today['收盘'] > today['上']) and (today['REF_C'] <= today['REF_上']) and is_strong_breakout
            
            yesterday = hist.iloc[-2]
            has_pullback_confirm = (yesterday['成交量'] < yesterday['MA5_V']) and (abs(yesterday['PCT_CHG']) < 2.5) and (yesterday['收盘'] > yesterday['MA10'])
            
            if is_bull_trend and is_macd_bull and is_cross and (1.2 < vol_ratio < 5.0) and (price_percentile < 0.45):
                hist['OBV'] = np.where(hist['收盘'] > hist['REF_C'], hist['成交量'], 
                                       np.where(hist['收盘'] < hist['REF_C'], -hist['成交量'], 0)).cumsum()
                has_obv_breakout = hist['OBV'].iloc[-1] > hist['OBV'].iloc[-21:-1].max()

                recent_60_high = hist['最高'].iloc[-61:-1].max()
                has_60d_breakout = (today['收盘'] > recent_60_high) and (today['REF_C'] <= recent_60_high)
                yesterday_high = hist['最高'].iloc[-2]
                has_gap_up = today['开盘'] > yesterday_high

                delta = hist['收盘'].diff()
                gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
                rs = gain / loss
                hist['RSI'] = 100 - (100 / (1 + rs))
                has_rsi_oversold = hist['RSI'].iloc[-15:-1].min() < 30 
                
                recent_60_lows = hist['最低'].iloc[-60:-1]
                low_vals = recent_60_lows.values
                if len(low_vals) >= 5:
                    local_min_mask = (
                        (low_vals[2:-2] <= low_vals[1:-3]) & 
                        (low_vals[2:-2] <= low_vals[0:-4]) & 
                        (low_vals[2:-2] <= low_vals[3:-1]) & 
                        (low_vals[2:-2] <= low_vals[4:])
                    )
                    local_min_indices = recent_60_lows.index[2:-2][local_min_mask]
                    if len(local_min_indices) >= 2:
                        idx1, idx2 = local_min_indices[-2], local_min_indices[-1]
                        has_macd_div = (hist.loc[idx2, '最低'] < hist.loc[idx1, '最低']) and (hist.loc[idx2, 'DIF'] > hist.loc[idx1, 'DIF'])
                    else:
                        has_macd_div = False
                else:
                    has_macd_div = False
                    
                recent_120 = hist.iloc[-121:-1]
                if len(recent_120) > 20:
                    recent_close = recent_120['收盘'].values
                    recent_vol = recent_120['成交量'].values
                    
                    if recent_vol.sum() == 0:
                        has_chip_breakthrough = False
                    else:
                        hist_counts, bin_edges = np.histogram(recent_close, bins=20, weights=recent_vol)
                        max_bin_idx = np.argmax(hist_counts)
                        poc_price = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx+1]) / 2
                        has_chip_breakthrough = (today['REF_C'] <= poc_price) and (today['收盘'] > poc_price)
                else:
                    has_chip_breakthrough = False

                score, level, reason_text = generate_reason_and_score(
                    price_percentile, vol_ratio, vcp_amplitude, upper_shadow_ratio, has_zt_gene,
                    has_macd_div, has_rsi_oversold, has_chip_breakthrough, 
                    has_obv_breakout, has_60d_breakout, has_gap_up, has_pullback_confirm,
                    today['ADX']  
                )
                
                stop_loss, target_1, target_2, risk_percent = calculate_target_points(
                    current_price=row['最新价'], 
                    today_low=today['最低'], 
                    ma20=today['MA20'], 
                    max_250_high=max_1y,
                    atr=today['ATR']
                )
                
                position_advice = get_position_advice(score, risk_percent, market_ok)
                
                final_signals.append({
                    '代码': code,
                    '名称': name,
                    '现价': row['最新价'],
                    '涨幅': f"{row['涨跌幅']}%",
                    '评分': score,
                    '评级': level,
                    '仓位建议': position_advice,
                    '触发时间': now.strftime("%H:%M"),
                    '选股逻辑': reason_text,
                    '止损点': stop_loss,
                    '目标1': target_1,
                    '目标2': target_2,
                    'MA10': round(today['MA10'], 2)  
                })
                already_pushed.add(code)
                
        except Exception as e:
            pass
        finally:
            if 'hist' in locals():
                del hist
            if idx % 20 == 0:
                gc.collect()

    del hist_dict 
    t5 = time.time()
    logger.info(f"⏱️ 节点耗时 -> 核心指标运算: {t5 - t4:.2f} 秒")

    # 板块强度终极过滤
    if final_signals:
        sector_strength_dict = get_all_sector_strength()
        logger.info(f"🔍 正在为 {len(final_signals)} 只初步合格标的进行【板块主线验证】...")
        
        sector_map = {}
        with ThreadPoolExecutor(max_workers=min(4, len(final_signals))) as executor:
            future_to_code = {executor.submit(get_stock_sector, row['代码']): row['代码'] for row in final_signals}
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    sector_map[code] = future.result(timeout=3.0)
                except Exception:
                    sector_map[code] = ""

        valid_signals = []
        for row in final_signals:
            sector = sector_map.get(row['代码'], "")
            sector_pct = sector_strength_dict.get(sector, 0.0)
            
            if sector and sector_pct < 1.0:
                logger.info(f"🚫 剔除假突破: {row['名称']} ({row['代码']}) -> 所属[{sector}]板块今日涨幅仅 {sector_pct}%, 未达主线强度标准(>=1.0%)。")
                continue
            
            row['板块'] = sector
            row['板块涨幅'] = sector_pct
            valid_signals.append(row)
            
        final_signals = valid_signals

    result_df = pd.DataFrame(final_signals)
    if not result_df.empty:
        result_df = result_df.sort_values(by='评分', ascending=False)
        
    return result_df, already_pushed, total_count

# ================= 消息推送引擎 =================
def send_dingtalk_msg(data_df, total_count):
    webhook_url = os.environ.get('DINGTALK_WEBHOOK')
    now_str = get_bjs_time().strftime("%Y-%m-%d %H:%M")
    
    MAX_MSG_LEN = 15000 

    logger.info("="*40)
    logger.info("📡 正在进入钉钉推送环节...")

    if not webhook_url:
        logger.warning("🚨 警告：环境变量 DINGTALK_WEBHOOK 未读取到！")
    else:
        logger.info("✅ 检测到 Webhook URL，正在准备报文...")

    if data_df is None or data_df.empty:
        logger.info(f"📉 最终结果：深度体检了 {total_count} 只异动股，0 只合格。")
        if IS_MANUAL_RUN and PUSH_EMPTY_RESULT:
            message = f"🤖 AI 量化执行纪律单 (手动检测) {now_str}\n\n✅ 节点：GitHub Actions 运行正常\n✅ 数据：AkShare 接口连通正常\n✅ 结果：深度体检 {total_count} 只股票，合格率 0%。\n━━━━━━━━━━━━━━━━━━\n💡 备忘录：系统运转完美。无符合条件标的，宁缺毋滥。"
            messages_to_send = [message]
        else:
            logger.info("🛑 未开启空结果推送或处于自动运行模式，静默退出不打扰。")
            logger.info("="*40)
            return
    else:
        logger.info(f"🎉 成功锁定 {len(data_df)} 只高胜率牛股！开始生成组合报文...")
        title_tag = "手动检测" if IS_MANUAL_RUN else "盘中狙击"

        # 【严重错误修复：移除此处的重复散布收集并发逻辑】
        
        messages_to_send = []
        current_msg = f"🤖 AI 量化执行纪律单 ({title_tag}) {now_str}\n\n"
        
        for _, row in data_df.iterrows():
            # 直接使用在核心引擎过滤时已拿到的板块数据，避免缩进错误及运行开销
            sector = row.get('板块', "")
            sector_pct = row.get('板块涨幅', 0.0)
            
            stock_msg = f"【{row['名称']} ({row['代码']})】\n"
            if sector:
                stock_msg += f"🏷️ 所属板块：{sector} (今日涨幅: {sector_pct}%)\n"
            stock_msg += f"📊 综合评分：{row['评分']}分 {row['评级']}\n"
            stock_msg += f"💰 当前现价：¥{row['现价']} ({row['涨幅']})\n"
            stock_msg += "--- 📝 核心交易逻辑 ---\n"
            stock_msg += f"{row['选股逻辑']}\n"
            stock_msg += f"--- 💼 资金与点位管理 ---\n"
            stock_msg += f"{row['仓位建议']}\n"
            stock_msg += f"🛑 防守止损：¥{row['止损点']} (破今日底/20日线离场)\n"
            stock_msg += f"🥇 第一止盈：¥{row['目标1']} (盈亏比 1:2 减仓点)\n"
            stock_msg += f"🚀 终极目标：¥{row['目标2']} (大波段年内前高)\n"
            
            stock_msg += f"💡 动态止盈：到达目标1减仓1/3，剩余底仓以10日均线(约¥{row['MA10']})为移动跟踪止盈线，收盘跌破清仓。\n"
            
            market_prefix = "sh" if str(row['代码']).startswith("6") else "sz"
            stock_msg += f"🔗 快速查看: https://quote.eastmoney.com/{market_prefix}{row['代码'][:6]}.html\n"
            
            stock_msg += "━━━━━━━━━━━━━━━━━━\n"
            
            if len(current_msg) + len(stock_msg) > MAX_MSG_LEN:
                messages_to_send.append(current_msg)
                current_msg = f"🤖 AI 量化执行纪律单 ({title_tag}) {now_str} (续...)\n\n" + stock_msg
            else:
                current_msg += stock_msg
                
        if current_msg:
            messages_to_send.append(current_msg)

    if not webhook_url:
        logger.info(f"📄 因为没有配置 Webhook，模拟内容如下 (共 {len(messages_to_send)} 段):\n{messages_to_send[0]}")
        logger.info("="*40)
        return

    t6 = time.time()
    for idx, msg_chunk in enumerate(messages_to_send):
        payload = {"msgtype": "text", "text": {"content": msg_chunk}}
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"📥 钉钉回执 [{idx+1}/{len(messages_to_send)}]: {response.text}")
            if "keywords not in content" in response.text:
                logger.error("❌ 错误：钉钉安全拦截！请确认钉钉机器人的【自定义关键词】是否精确包含字母: AI")
            elif "ok" in response.text:
                logger.info(f"✅ 第 {idx+1} 段消息推送成功！")
            
            if len(messages_to_send) > 1:
                time.sleep(1)
        except Exception as e:
            logger.error(f"❌ 网络推送崩溃: {e}")
            
    t7 = time.time()
    logger.info(f"⏱️ 节点耗时 -> 最终消息推送: {t7 - t6:.2f} 秒")
    logger.info("="*40)

if __name__ == "__main__":
    try:
        signals_df, updated_pushed, total_count = get_signals()
        send_dingtalk_msg(signals_df, total_count)
        if not signals_df.empty:
            save_pushed_state(updated_pushed)
    except Exception as e:
        logger.critical(f"主程序崩溃: {e}")
