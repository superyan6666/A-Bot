import akshare as ak
import pandas as pd
import numpy as np
import requests
import os
import time
import json
from datetime import datetime, timedelta
import pytz

# ================= 配置区 =================
TZ_BJS = pytz.timezone('Asia/Shanghai') # 设置时区为北京时间
STATE_FILE = 'pushed_state.json'        # 盘中去重记忆文件
IS_MANUAL_RUN = os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch' # 识别是否为手动触发测试

# ================= 辅助函数 =================
def get_bjs_time():
    """获取当前北京时间"""
    return datetime.now(TZ_BJS)

def is_trading_time(now):
    """判断是否在 A股 交易时间内"""
    time_val = now.hour * 100 + now.minute
    if (925 <= time_val <= 1130) or (1300 <= time_val <= 1500):
        return True
    return False

def get_passed_trading_mins(now):
    """计算今天已经走过的交易分钟数，用于盘中量能外推计算虚拟成交量"""
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
    """加载今天已经推送过的股票代码，防止一天内被同一只股票重复轰炸"""
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
    """保存今天推送过的股票记录到本地"""
    today_str = get_bjs_time().strftime("%Y-%m-%d")
    state = {'date': today_str, 'pushed_codes': list(pushed_codes)}
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

# ================= 核心算法函数 =================
def calculate_tdx_dma(close_series, a_series):
    """[极速版] 实现通达信的 DMA(X, A) 动态移动平均函数"""
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

def fetch_hist_data(code, start_date, end_date, retries=3):
    """带重试机制的 K线数据拉取，增强网络鲁棒性"""
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if not df.empty:
                return df
        except Exception:
            time.sleep(0.5)
    return None

def calculate_target_points(current_price, today_low, ma20, max_250_high):
    """智能点位测算：计算止损位和目标止盈位"""
    # 止损点：破今天最低价 或者 破20日均线（两者取较低的，给予一定容错空间）
    stop_loss = min(today_low, ma20)
    stop_loss = round(stop_loss * 0.99, 2)  # 给1%的缓冲防毛刺
    
    if stop_loss >= current_price:
        stop_loss = round(current_price * 0.95, 2)
        
    risk_value = current_price - stop_loss
    
    # 第一目标：严格按照 1:2 盈亏比计算短线阻力
    target_1 = round(current_price + risk_value * 2.0, 2)
    # 终极目标：前期一年内的高点。如果距离太近，则定为第一目标的上方15%
    target_2 = round(max(max_250_high, target_1 * 1.15), 2)
    
    return stop_loss, target_1, target_2

def generate_reason_and_score(price_percentile, vol_ratio, vcp_amplitude, upper_shadow_ratio, has_zt_gene):
    """AI 打分与动态话术生成引擎 (满分100分)"""
    score = 40
    reasons = []

    perc = price_percentile * 100
    if perc < 15:
        score += 15
        reasons.append(f"🟢 【长线潜伏】处于3年极度冰点(分位{perc:.1f}%)，下方支撑极强，性价比爆棚。")
    elif perc < 30:
        score += 10
        reasons.append(f"🟢 【长线底部】处于3年底部区域(分位{perc:.1f}%)，趋势反转确立。")
    else:
        score += 5
        reasons.append(f"🟡 【中位起涨】处于历史中低位(分位{perc:.1f}%)，属于波段加速期。")

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
    
    # 如果是手动触发，强制放行；否则执行严格的交易时间拦截
    if not IS_MANUAL_RUN:
        if not is_trading_time(now) and now.hour < 15:
            print(f"[{now.strftime('%H:%M')}] 当前时间不在交易时段，脚本休眠。")
            return pd.DataFrame(), set()

    already_pushed = load_pushed_state()
    df_spot = ak.stock_zh_a_spot_em()
    df_spot = df_spot.dropna(subset=['最新价', '最高', '涨跌幅', '代码', '名称', '流通市值', '市盈率-动态', '市净率'])
    df_spot = df_spot[~df_spot['名称'].str.contains('ST|退')]
    df_spot = df_spot[~df_spot['代码'].isin(already_pushed)]
    
    # 基础安全底线过滤 (30-300亿盘子，有市盈率，无极大泡沫)
    market_cap_condition = (df_spot['流通市值'] >= 30 * 10**8) & (df_spot['流通市值'] <= 300 * 10**8)
    fundamental_condition = (df_spot['市盈率-动态'] > 0) & (df_spot['市盈率-动态'] < 60) & (df_spot['市净率'] > 0) & (df_spot['市净率'] < 5)
    
    # 涨幅初筛 (大于3.5%异动)
    pool_condition = (df_spot['涨跌幅'] >= 3.5) & market_cap_condition & fundamental_condition
    target_pool = df_spot[pool_condition].copy()
    
    end_date_str = now.strftime("%Y%m%d")
    start_date_str = (now - timedelta(days=1095)).strftime("%Y%m%d") # 提取3年数据
    
    passed_mins = get_passed_trading_mins(now)
    vol_scale_factor = 240 / passed_mins if passed_mins > 0 else 1.0
    
    final_signals = []
    
    for index, row in target_pool.iterrows():
        code = row['代码']
        name = row['名称']
        
        try:
            hist = fetch_hist_data(code, start_date_str, end_date_str)
            if hist is None or len(hist) < 250: 
                continue
                
            hist['PCT_CHG'] = (hist['收盘'] - hist['收盘'].shift(1)) / hist['收盘'].shift(1) * 100
            
            # 均线计算
            hist['MA20'] = hist['收盘'].rolling(window=20).mean()
            hist['MA60'] = hist['收盘'].rolling(window=60).mean()
            hist['MA250'] = hist['收盘'].rolling(window=250).mean()
            hist['MA5_V'] = hist['成交量'].rolling(window=5).mean()
            
            # MACD 逻辑 (补强：动能水上共振)
            hist['EMA12'] = hist['收盘'].ewm(span=12, adjust=False).mean()
            hist['EMA26'] = hist['收盘'].ewm(span=26, adjust=False).mean()
            hist['DIF'] = hist['EMA12'] - hist['EMA26']
            hist['DEA'] = hist['DIF'].ewm(span=9, adjust=False).mean()
            
            # --- 还原通达信公式 ---
            hist['CC'] = abs((2 * hist['收盘'] + hist['最高'] + hist['最低']) / 4 - hist['MA20']) / hist['MA20']
            hist['DD'] = calculate_tdx_dma(hist['收盘'], hist['CC'])
            hist['上'] = 1.07 * hist['DD']
            
            hist['REF_C'] = hist['收盘'].shift(1)
            hist['REF_上'] = hist['上'].shift(1)
            
            today = hist.iloc[-1]
            min_3y = hist['最低'].min()
            max_3y = hist['最高'].max()
            max_1y = hist['最高'].iloc[-250:].max() 
            price_range = max_3y - min_3y
            
            if price_range <= 0: continue
            
            # 核心数据测量
            price_percentile = (today['收盘'] - min_3y) / price_range
            recent_10_high = hist['最高'].iloc[-11:-1].max()
            recent_10_low = hist['最低'].iloc[-11:-1].min()
            vcp_amplitude = (recent_10_high - recent_10_low) / recent_10_low
            
            body_length = today['收盘'] - today['开盘']
            if body_length <= 0: continue
            upper_shadow_ratio = (today['最高'] - today['收盘']) / body_length
            
            virtual_vol = today['成交量'] * vol_scale_factor
            vol_ratio = virtual_vol / today['MA5_V'] if today['MA5_V'] > 0 else 1
            has_zt_gene = (hist['PCT_CHG'].iloc[-61:-1] >= 9.5).any()
            
            # --- 买入护城河判断 ---
            # 1. 均线共振：必须在年线之上，且中期趋势(MA20)必须大于(MA60)
            is_bull_trend = (today['收盘'] >= today['MA250']) and (today['MA20'] >= today['MA60'])
            
            # 2. MACD水上起步：确保趋势有底层动能支撑
            is_macd_bull = (today['DIF'] > 0) and (today['DEA'] > 0)
            
            # 3. 乖离上轨突破
            is_cross = (today['收盘'] > today['上']) and (today['REF_C'] <= today['REF_上']) 
            
            # 基础条件：均线多头、MACD多头、过上轨、今日温和放量(1.2~5.0倍量，防天量见天价)、3年低位区间(<45%分位)
            if is_bull_trend and is_macd_bull and is_cross and (1.2 < vol_ratio < 5.0) and (price_percentile < 0.45):
                score, level, reason_text = generate_reason_and_score(
                    price_percentile, vol_ratio, vcp_amplitude, upper_shadow_ratio, has_zt_gene
                )
                
                # 计算交易点位
                stop_loss, target_1, target_2 = calculate_target_points(
                    current_price=row['最新价'], 
                    today_low=today['最低'], 
                    ma20=today['MA20'], 
                    max_250_high=max_1y
                )
                
                final_signals.append({
                    '代码': code,
                    '名称': name,
                    '现价': row['最新价'],
                    '涨幅': f"{row['涨跌幅']}%",
                    '评分': score,
                    '评级': level,
                    '触发时间': now.strftime("%H:%M"),
                    '选股逻辑': reason_text,
                    '止损点': stop_loss,
                    '目标1': target_1,
                    '目标2': target_2
                })
                already_pushed.add(code)
            
            time.sleep(0.05)
            
        except Exception as e:
            continue

    result_df = pd.DataFrame(final_signals)
    if not result_df.empty:
        result_df = result_df.sort_values(by='评分', ascending=False)
        
    return result_df, already_pushed

# ================= 消息推送引擎 =================
def send_dingtalk_msg(data_df):
    webhook_url = os.environ.get('DINGTALK_WEBHOOK')
    now_str = get_bjs_time().strftime("%Y-%m-%d %H:%M")

    if data_df is None or data_df.empty:
        print("本时段无新增高分标的。")
        # 增加手动检测的空跑回执
        if IS_MANUAL_RUN:
            message = f"🤖 AI 量化执行纪律单 (手动检测) {now_str}\n\n"
            message += "✅ 运行环境：GitHub Actions 节点正常\n"
            message += "✅ 数据通道：AkShare 接口连通正常\n"
            message += "✅ 消息通道：DingTalk Webhook 触达正常\n"
            message += "━━━━━━━━━━━━━━━━━━\n"
            message += "📉 扫描结果：今日全市场扫描完毕，暂无符合【底部多维共振突破】的高胜率标的。\n\n"
            message += "💡 交易备忘录：系统运转完美。宁缺毋滥是盈利的前提，猎人请继续耐心等待绝佳击球点。"
            if webhook_url:
                requests.post(webhook_url, json={"msgtype": "text", "text": {"content": message}}, timeout=5)
        return

    # 【注意】：此处的 'AI' 是为了通过钉钉机器人关键词校验，勿删！
    title_tag = "手动检测" if IS_MANUAL_RUN else "盘中狙击"
    message = f"🤖 AI 量化执行纪律单 ({title_tag}) {now_str}\n\n"
    
    for _, row in data_df.iterrows():
        message += f"【{row['名称']} ({row['代码']})】\n"
        message += f"📊 评分：{row['评分']}分 {row['评级']}\n"
        message += f"💰 现价：¥{row['现价']} ({row['涨幅']})\n"
        message += "--- 📝 核心交易逻辑 ---\n"
        message += f"{row['选股逻辑']}\n"
        message += "--- 🎯 操作点位计划 ---\n"
        message += f"🛑 防守止损：¥{row['止损点']} (破今日底/20日线离场)\n"
        message += f"🥇 第一止盈：¥{row['目标1']} (盈亏比 1:2 减仓点)\n"
        message += f"🚀 终极目标：¥{row['目标2']} (大波段年内前高)\n"
        message += "━━━━━━━━━━━━━━━━━━\n"
    
    if not webhook_url:
        print("未配置 webhook 环境变量，在控制台打印测试结果:\n", message)
        return

    payload = {"msgtype": "text", "text": {"content": message}}
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        print(f"钉钉推送响应: {response.text}")
    except Exception as e:
        print(f"钉钉推送失败: {e}")

if __name__ == "__main__":
    try:
        signals_df, updated_pushed = get_signals()
        if not signals_df.empty:
            send_dingtalk_msg(signals_df)
            save_pushed_state(updated_pushed) # 保存记录防止群消息轰炸
    except Exception as e:
        print(f"主程序崩溃: {e}")
