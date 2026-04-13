import akshare as ak
import pandas as pd
import numpy as np
import requests
import os
import time
import json
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
TZ_BJS = pytz.timezone('Asia/Shanghai') # 设置时区为北京时间
STATE_FILE = 'pushed_state.json'        # 盘中去重记忆文件
# 识别是否为 GitHub 手动点击触发
IS_MANUAL_RUN = os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'

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

def fetch_hist_data(code, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if not df.empty:
                return df
        except Exception:
            time.sleep(0.5)
    return None

def fetch_single_stock_data(code, start_date_str, end_date_str):
    """用于线程池的单个股票数据获取包装函数"""
    hist = fetch_hist_data(code, start_date_str, end_date_str)
    return code, hist

def get_stock_sector(code):
    """【新增】获取个股所属板块/行业信息"""
    try:
        df = ak.stock_individual_info_em(symbol=code)
        industry = df[df['item'] == '行业']['value'].values[0]
        return industry if pd.notna(industry) else "核心概念"
    except Exception:
        return "热门题材"

def calculate_target_points(current_price, today_low, ma20, max_250_high):
    stop_loss = min(today_low, ma20)
    stop_loss = round(stop_loss * 0.99, 2)
    if stop_loss >= current_price:
        stop_loss = round(current_price * 0.95, 2)
        
    risk_value = current_price - stop_loss
    # 【新增资金管理维度】：计算现价到止损位的回撤风险百分比
    risk_percent = (risk_value / current_price) * 100 
    
    target_1 = round(current_price + risk_value * 2.0, 2)
    target_2 = round(max(max_250_high, target_1 * 1.15), 2)
    return stop_loss, target_1, target_2, risk_percent

def get_position_advice(score, risk_percent):
    """【新增】基于胜率(打分)与赔率(回撤风险)的动态仓位管理系统"""
    if score >= 85:
        base_pos = 30
        tag = "重仓狙击"
    elif score >= 75:
        base_pos = 20
        tag = "标准配置"
    else:
        base_pos = 10
        tag = "轻仓试错"

    # 风控惩罚机制：如果单笔止损风险过大(>8%)，强制仓位减半
    if risk_percent > 8.0:
        final_pos = base_pos // 2
        return f"⚠️ 建议 {final_pos}% 仓位 (原定{tag}，但单笔止损风险高达 {risk_percent:.1f}%，触发风控减半！)"
    else:
        return f"⚖️ 建议 {base_pos}% 仓位 ({tag}，单笔止损风险控制在极佳的 {risk_percent:.1f}%)"

def generate_reason_and_score(price_percentile, vol_ratio, vcp_amplitude, upper_shadow_ratio, has_zt_gene, has_macd_div, has_rsi_oversold, has_chip_breakthrough, has_obv_breakout, has_60d_breakout, has_gap_up):
    score = 40
    reasons = []

    perc = price_percentile * 100
    if perc < 15:
        score += 15
        reasons.append(f"🟢 【波段潜伏】处于近1年极度冰点(分位{perc:.1f}%)，下方支撑极强，性价比爆棚。")
    elif perc < 30:
        score += 10
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

    # 【新增】三大高阶指标加分项：核武级共振
    if has_chip_breakthrough:
        score += 15
        reasons.append("🏔️ 【跨越筹码峰】今日强势突破近半年核心筹码密集区，上方套牢盘抛压一扫而空，主升浪空间彻底打开！")
        
    if has_macd_div:
        score += 10
        reasons.append("💥 【底背离共振】近期呈现标准MACD底背离形态，杀跌动能枯竭，反转信号极其强烈。")
        
    if has_rsi_oversold:
        score += 5
        reasons.append("📉 【超卖反弹】前期RSI曾触及极度超卖区，空头情绪宣泄完毕，底部夯实。")

    # 【新增】三大Alpha胜率挖掘加分项：主力资金轨迹
    if has_60d_breakout:
        score += 15
        reasons.append("🚀 【箱体突破】一举跃过近60日波段前高，呈现经典'N字型'主升浪突破形态，上攻意愿极其坚决！")
        
    if has_obv_breakout:
        score += 10
        reasons.append("💸 【量能先行】OBV(能量潮)创出阶段新高，量在价先，揭示主力资金近期一直在暗中吃货吸筹！")
        
    if has_gap_up:
        score += 5
        reasons.append("⚡ 【跳空抢筹】今日早盘呈现跳空高开缺口且未回补，集合竞价阶段多头已呈碾压逼空之势。")

    # 分数封顶 100 分
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
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 开始启动量化扫描引擎...")
    
    if not IS_MANUAL_RUN:
        if not is_trading_time(now) and now.hour < 15:
            print(f"[{now.strftime('%H:%M')}] 当前时间不在交易时段，自动休眠。")
            return pd.DataFrame(), set(), 0
    else:
        print("💡 检测到手动触发 (workflow_dispatch)，正在强制执行全市场扫描...")

    already_pushed = load_pushed_state()
    df_spot = ak.stock_zh_a_spot_em()
    
    # 增加 '成交额' 和 '量比' 字段以进行流动性过滤
    df_spot = df_spot.dropna(subset=['最新价', '最高', '涨跌幅', '换手率', '成交额', '量比', '代码', '名称', '流通市值', '市盈率-动态', '市净率'])
    df_spot = df_spot[~df_spot['名称'].str.contains('ST|退')]
    df_spot = df_spot[~df_spot['代码'].isin(already_pushed)]
    
    market_cap_condition = (df_spot['流通市值'] >= 30 * 10**8) & (df_spot['流通市值'] <= 300 * 10**8)
    fundamental_condition = (df_spot['市盈率-动态'] > 0) & (df_spot['市盈率-动态'] < 60) & (df_spot['市净率'] > 0) & (df_spot['市净率'] < 5)
    
    # 【提速优化核心区】：收紧初筛漏斗，过滤跟风弱势股
    turnover_condition = df_spot['换手率'] >= 3.0       # 换手率从 2.0 提高到 3.0
    amount_condition = df_spot['成交额'] >= 100000000    # 成交额必须大于 1 亿元
    rise_condition = df_spot['涨跌幅'] >= 4.5           # 涨幅从 3.5 提高到 4.5
    vr_condition = (df_spot['量比'] >= 1.2) & (df_spot['量比'] <= 5.0) # 【新增】盘中量比预过滤，拦截无效计算
    
    pool_condition = rise_condition & market_cap_condition & fundamental_condition & turnover_condition & amount_condition & vr_condition
    target_pool = df_spot[pool_condition].copy()
    
    total_count = len(target_pool)
    print(f"✅ 初筛完毕！发现 {total_count} 只符合【高活跃+强异动+基本面健康】的精选股票。")
    
    if total_count == 0:
        return pd.DataFrame(), already_pushed, 0

    end_date_str = now.strftime("%Y%m%d")
    # 【优化点】：缩短回溯窗口至 400 天，减少数据冗余与计算开销
    start_date_str = (now - timedelta(days=400)).strftime("%Y%m%d")
    passed_mins = get_passed_trading_mins(now)
    vol_scale_factor = 240 / passed_mins if passed_mins > 0 else 1.0
    
    # --- 并发拉取历史数据 ---
    print(f"🔄 正在启动并发拉取 {total_count} 只股票的历史数据，请稍候...")
    codes = target_pool['代码'].tolist()
    hist_dict = {}
    
    # 使用 5 个线程，以避免瞬间高频请求触发源站限流
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_code = {
            executor.submit(fetch_single_stock_data, code, start_date_str, end_date_str): code
            for code in codes
        }
        for future in as_completed(future_to_code):
            try:
                code_res, hist = future.result()
                if hist is not None and len(hist) >= 250:
                    hist_dict[code_res] = hist
            except Exception as e:
                pass
                
    print(f"📥 历史数据拉取完成，成功获取 {len(hist_dict)} 只股票的有效数据。开始逐一测算...")

    final_signals = []
    
    # --- 核心量价模型测算 (纯 CPU 计算) ---
    for idx, (index, row) in enumerate(target_pool.iterrows(), 1):
        code = row['代码']
        name = row['名称']
        
        # 已经在多线程里拉取失败或数据不够长度的，直接跳过
        if code not in hist_dict:
            continue
            
        try:
            hist = hist_dict[code]
            
            hist['PCT_CHG'] = (hist['收盘'] - hist['收盘'].shift(1)) / hist['收盘'].shift(1) * 100
            hist['MA20'] = hist['收盘'].rolling(window=20).mean()
            hist['MA60'] = hist['收盘'].rolling(window=60).mean()
            hist['MA250'] = hist['收盘'].rolling(window=250).mean()
            hist['MA5_V'] = hist['成交量'].rolling(window=5).mean()
            
            hist['EMA12'] = hist['收盘'].ewm(span=12, adjust=False).mean()
            hist['EMA26'] = hist['收盘'].ewm(span=26, adjust=False).mean()
            hist['DIF'] = hist['EMA12'] - hist['EMA26']
            hist['DEA'] = hist['DIF'].ewm(span=9, adjust=False).mean()
            
            hist['CC'] = abs((2 * hist['收盘'] + hist['最高'] + hist['最低']) / 4 - hist['MA20']) / hist['MA20']
            hist['DD'] = calculate_tdx_dma(hist['收盘'], hist['CC'])
            hist['上'] = 1.07 * hist['DD']
            
            hist['REF_C'] = hist['收盘'].shift(1)
            hist['REF_上'] = hist['上'].shift(1)
            
            today = hist.iloc[-1]
            
            # 【新增 Alpha 1】：计算 OBV (能量潮) 资金潜伏
            hist['OBV'] = np.where(hist['收盘'] > hist['REF_C'], hist['成交量'], 
                                   np.where(hist['收盘'] < hist['REF_C'], -hist['成交量'], 0)).cumsum()
            has_obv_breakout = hist['OBV'].iloc[-1] > hist['OBV'].iloc[-21:-1].max()

            # 【新增 Alpha 2】：计算 60日N字箱体突破
            recent_60_high = hist['最高'].iloc[-61:-1].max()
            has_60d_breakout = (today['收盘'] > recent_60_high) and (today['REF_C'] <= recent_60_high)
            
            # 【新增 Alpha 3】：计算 集合竞价跳空缺口
            yesterday_high = hist['最高'].iloc[-2]
            has_gap_up = today['开盘'] > yesterday_high

            # 【新增】：1. 计算 RSI (14日)
            delta = hist['收盘'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            has_rsi_oversold = hist['RSI'].iloc[-15:-1].min() < 30 # 近15天内曾跌破30超卖线
            
            # 【新增】：2. 计算 MACD 底背离 (对比近40天内的两段低谷)
            window1 = hist.iloc[-40:-20]
            window2 = hist.iloc[-20:-1]
            if not window1.empty and not window2.empty:
                min_p1, min_p2 = window1['最低'].min(), window2['最低'].min()
                min_macd1, min_macd2 = window1['DIF'].min(), window2['DIF'].min()
                # 价格创新低，但MACD(DIF)未创新低，形成底背离
                has_macd_div = (min_p2 < min_p1) and (min_macd2 > min_macd1)
            else:
                has_macd_div = False
                
            # 【新增】：3. 计算 筹码峰突破 (近半年约120个交易日)
            recent_120 = hist.iloc[-121:-1]
            if len(recent_120) > 20:
                # 将价格切分为20个区间，统计每个区间的成交量，找出最大值即为“筹码密集峰(POC)”
                bins = pd.cut(recent_120['收盘'], bins=20)
                volume_by_price = recent_120.groupby(bins, observed=True)['成交量'].sum()
                poc_bin = volume_by_price.idxmax()
                poc_price = poc_bin.mid # 筹码峰核心价位
                # 昨天还在筹码峰下方，今天一举突破！
                has_chip_breakthrough = (today['REF_C'] <= poc_price) and (today['收盘'] > poc_price)
            else:
                has_chip_breakthrough = False
            
            # 【优化点】：配合 400 天窗口，将分位计算改为 1年内最高最低
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
            
            virtual_vol = today['成交量'] * vol_scale_factor
            vol_ratio = virtual_vol / today['MA5_V'] if today['MA5_V'] > 0 else 1
            has_zt_gene = (hist['PCT_CHG'].iloc[-61:-1] >= 9.5).any()
            
            is_bull_trend = (today['收盘'] >= today['MA250']) and (today['MA20'] >= today['MA60'])
            is_macd_bull = (today['DIF'] > 0) and (today['DEA'] > 0)
            is_cross = (today['收盘'] > today['上']) and (today['REF_C'] <= today['REF_上']) 
            
            if is_bull_trend and is_macd_bull and is_cross and (1.2 < vol_ratio < 5.0) and (price_percentile < 0.45):
                score, level, reason_text = generate_reason_and_score(
                    price_percentile, vol_ratio, vcp_amplitude, upper_shadow_ratio, has_zt_gene,
                    has_macd_div, has_rsi_oversold, has_chip_breakthrough, 
                    has_obv_breakout, has_60d_breakout, has_gap_up
                )
                
                # 获取点位与风险率
                stop_loss, target_1, target_2, risk_percent = calculate_target_points(
                    current_price=row['最新价'], 
                    today_low=today['最低'], 
                    ma20=today['MA20'], 
                    max_250_high=max_1y
                )
                
                # 生成仓位与风控建议
                position_advice = get_position_advice(score, risk_percent)
                
                sector = get_stock_sector(code)
                
                final_signals.append({
                    '代码': code,
                    '名称': name,
                    '现价': row['最新价'],
                    '涨幅': f"{row['涨跌幅']}%",
                    '板块': sector,
                    '评分': score,
                    '评级': level,
                    '仓位建议': position_advice,
                    '触发时间': now.strftime("%H:%M"),
                    '选股逻辑': reason_text,
                    '止损点': stop_loss,
                    '目标1': target_1,
                    '目标2': target_2
                })
                already_pushed.add(code)
                
        except Exception as e:
            continue

    print(f"📊 所有历史数据测算完成！")
    result_df = pd.DataFrame(final_signals)
    if not result_df.empty:
        result_df = result_df.sort_values(by='评分', ascending=False)
        
    return result_df, already_pushed, total_count

# ================= 消息推送引擎 =================
def send_dingtalk_msg(data_df, total_count):
    webhook_url = os.environ.get('DINGTALK_WEBHOOK')
    now_str = get_bjs_time().strftime("%Y-%m-%d %H:%M")

    print("\n" + "="*40)
    print("📡 正在进入钉钉推送环节...")

    if not webhook_url:
        print("🚨 警告：环境变量 DINGTALK_WEBHOOK 未读取到！")
        print("👉 请务必在 GitHub 的 Settings -> Secrets and variables -> Actions 中配置。")
    else:
        print("✅ 检测到 Webhook URL，正在打包报文...")

    if data_df is None or data_df.empty:
        print(f"📉 最终结果：深度体检了 {total_count} 只异动股，0 只合格。")
        if IS_MANUAL_RUN:
            message = f"🤖 AI 量化执行纪律单 (手动检测) {now_str}\n\n✅ 节点：GitHub Actions 运行正常\n✅ 数据：AkShare 接口连通正常\n✅ 结果：深度体检 {total_count} 只股票，合格率 0%。\n━━━━━━━━━━━━━━━━━━\n💡 备忘录：系统运转完美。策略已成功过滤所有诱多风险，宁缺毋滥是盈利的前提，猎人请耐心等待极品信号。"
        else:
            print("🛑 自动运行模式下，无高分标的，取消静默推送。")
            print("="*40 + "\n")
            return
    else:
        print(f"🎉 成功锁定 {len(data_df)} 只高胜率牛股！")
        title_tag = "手动检测" if IS_MANUAL_RUN else "盘中狙击"
        message = f"🤖 AI 量化执行纪律单 ({title_tag}) {now_str}\n\n"
        for _, row in data_df.iterrows():
            message += f"【{row['名称']} ({row['代码']})】\n"
            message += f"🏷️ 所属板块：{row['板块']} (若迎合今日主线，溢价极高)\n"
            message += f"📊 综合评分：{row['评分']}分 {row['评级']}\n"
            message += f"💰 当前现价：¥{row['现价']} ({row['涨幅']})\n"
            message += "--- 📝 核心交易逻辑 ---\n"
            message += f"{row['选股逻辑']}\n"
            message += "--- 💼 资金与点位管理 ---\n"
            message += f"{row['仓位建议']}\n"
            message += f"🛑 防守止损：¥{row['止损点']} (破今日底/20日线离场)\n"
            message += f"🥇 第一止盈：¥{row['目标1']} (盈亏比 1:2 减仓点)\n"
            message += f"🚀 终极目标：¥{row['目标2']} (大波段年内前高)\n"
            message += "━━━━━━━━━━━━━━━━━━\n"

    if not webhook_url:
        print("📄 因为没有配置 Webhook，模拟的话术内容如下：\n", message)
        print("="*40 + "\n")
        return

    payload = {"msgtype": "text", "text": {"content": message}}
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        print(f"📥 钉钉服务器回执: {response.text}")
        if "keywords not in content" in response.text:
            print("❌ 错误：钉钉安全拦截！请确认钉钉机器人的【自定义关键词】是否精确包含字母: AI")
        elif "ok" in response.text:
            print("✅ 消息推送大成功！请检查手机。")
    except Exception as e:
        print(f"❌ 网络推送崩溃: {e}")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    try:
        signals_df, updated_pushed, total_count = get_signals()
        send_dingtalk_msg(signals_df, total_count)
        if not signals_df.empty:
            save_pushed_state(updated_pushed)
    except Exception as e:
        print(f"主程序崩溃: {e}")
