from dataclasses import dataclass
from typing import Callable

@dataclass
class Factor:
    condition: Callable[[dict], bool]
    points: int
    weight: float = 1.0
    template: str = ""
    group: str = ""

def get_factors_config(f_val: float, f_mom: float, f_rev: float, f_risk: float, 
                       tw: float, rw: float, m_regime: str, 
                       in_danger: bool, danger_label: str) -> list[Factor]:
    """
    数据驱动的因子引擎配置。
    投研人员可在此处增删改查因子规则，核心引擎会自动加载并执行。
    """
    return [
        Factor(lambda d: d.get('macd_divergence', False), 25, 1.0, "- 🧲 **MACD底背离**：日线级别价格创新低但动能衰竭，极其罕见的左侧黄金坑 (触发强加权)"),
        Factor(lambda d: d.get('mcap', 0) > 300e8 and 0 < d.get('pe', -1) < 25 and d.get('pb', 10) < 3, 10, f_val, "- 🏢 **价值蓝筹**：大市值低估值核心资产，防守属性极强", "VAL"),
        Factor(lambda d: d.get('vol_ratio', 0) > 1.0 and d.get('rs_rating', 0) > 5, 10, f_mom, "- 🚀 **强势领涨**：近期显著强于大盘，资金接力意愿极强", "MOM"),
        Factor(lambda d: d.get('price_pct', 1.0) < 0.3 and 0 < d.get('pb', 10) < 1.0, 8, f_val, "- ♻️ **困境反转**：股价严重破净且处于绝对低位，安全垫极厚", "VAL"),
        
        Factor(lambda d: d.get('in_hot_sector', False), 12, f_mom, "- 🌡️ **身处主线**：所在板块【{hot_sector_name}】今日强势领涨，踏准市场节奏"),
        
        Factor(lambda d: d.get('price_pct', 1.0) < 0.25, 12, f_rev * rw, "- 🟢 **绝对低位**：目前买入相当于抄底，长线持有安全", "POS"),
        Factor(lambda d: 0.25 <= d.get('price_pct', 1.0) <= 0.45, 8, f_rev, "- 🟢 **相对低位**：刚刚从底部爬起来，输时间不输钱", "POS"),
        Factor(lambda d: d.get('price_pct', 0.0) > 0.45, 6, f_mom, "- 📈 **多头趋势**：股价已脱离底部，处于健康的主升浪区间", "POS"),
        Factor(lambda d: d.get('price_pct', 0.0) > 0.85, 8, f_mom, "- 🚀 **高位突破**：股价处于年度高位，强者恒强趋势极佳", "MOM"), 
        
        Factor(lambda d: d.get('pe', -1) > 0 and d.get('pe', 100) < 40, 5, f_val, "- 🛡️ **业绩护体**：市盈率健康，不是炒空气的无基本面股", "VAL"),
        Factor(lambda d: d.get('macd_dea', -1.0) >= -0.05, 5, 1.0, "- 🌊 **多头控盘**：大周期趋势仍强，没有被深套的风险"), 
        
        Factor(lambda d: -2.0 <= d.get('dist_ma20', 100) <= 6.0, 12, 1.0, "- 🧲 **贴地潜伏**：目前价格紧贴均线支撑，绝佳安全低吸点", "MA20"),
        Factor(lambda d: 6.0 < d.get('dist_ma20', 0) <= 15.0, 6, f_mom, "- 🚀 **强势发力**：距离20日线有空间，依托短期均线强势上攻", "MA20"),
        Factor(lambda d: d.get('dist_ma20', 0) < -2.0, -10, f_risk, "- ⚠️ **破位嫌疑**：当前已跌破20日线，需警惕趋势走坏 (扣分)"),
        
        Factor(lambda d: 30 <= d.get('rsi', 50) <= 72, 5, 1.0, "- 📊 **温度适中**：RSI处于健康买入区间，正是下手时机"),
        
        Factor(lambda d: d.get('bull_rank', False), 8, f_mom, "- 📈 **顺势而为**：均线多头排列，跟着主力资金大部队走"),
        
        Factor(lambda d: d.get('has_zt', False), 8, 1.0, "- 🔥 **股性活跃**：该股历史上容易涨停，不会一潭死水"),
        Factor(lambda d: d.get('vol_ratio', 0) >= 1.8, 8, 1.0, "- 🔵 **放量确认**：今天成交量明显放大，大资金开始干活了", "VOL"),
        Factor(lambda d: d.get('red_days', 0) >= 2, 5, 1.0, "- 🔴 **稳步推升**：最近重心都在上移，主力在偷偷温和建仓"),
        
        Factor(lambda d: d.get('has_chip_break', False), 12, tw * f_mom, "- 🏔️ **抛压真空**：上方的套牢盘已割肉离场，向上拉升没阻力", "VCP"),
        Factor(lambda d: d.get('is_true_vcp', False), 12, 1.0, "- 🎯 **形态确认**：呈现经典 VCP (波动率收敛) 结构，洗盘极度充分", "VCP"),
        Factor(lambda d: not d.get('is_true_vcp', False) and d.get('vcp_amp', 1.0) < 0.12, 6, 1.0, "- 🟣 **蓄势待发**：近期波动极小，面临短线方向选择", "VCP"),
        Factor(lambda d: d.get('extreme_shrink_vol', False), 8, 1.0, "- 🧊 **没人砸盘**：爆发前夕成交极度萎缩，散户该卖的都卖了", "VCP"), 
        Factor(lambda d: d.get('has_obv_break', False), 10, tw * f_mom, "- 💸 **真金白银**：模型监控到真实的资金在创纪录净流入", "VOL"),
        Factor(lambda d: d.get('has_pullback', False), 12, 1.0, "- 🪃 **黄金深坑**：出现温和缩量回踩，主力洗盘给出的上车良机", "VCP"),
        Factor(lambda d: d.get('lower_shadow_ratio', 0) > 0.03, 5, 1.0, "- 📌 **强力护盘**：跌下去被大资金迅速买回，下方有人兜底", "VCP"), 
        
        Factor(lambda d: d.get('rs_rating', 0) > 5,  8, f_mom, "- 🏆 **跑赢大盘**：近60日涨幅超越指数，有资金在持续运作"),
        
        # --- 【排雷扣分项】 ---
        Factor(lambda d: d.get('surge_5d', 0) > 28, -20, f_risk, "- 🚫 **短期暴涨**：近5日涨幅过大透支空间，极易高位站岗 (重度扣分)"),
        Factor(lambda d: d.get('consecutive_down', 0) >= 4, -15, f_risk, "- 🔪 **飞刀预警**：近期连续阴线急跌，左侧接飞刀风险大 (重度扣分)"),
        Factor(lambda d: d.get('rsi', 50) > 80, -10, f_risk, "- 🌡️ **短期过热**：RSI偏高短线超买，操作需要进一步缩减仓位"),
        Factor(lambda d: d.get('rs_rating', 0) < -10, -8, f_risk, "- 📉 **跑输大盘**：近期持续弱于大盘，跟的是被冷落的股票"),
        Factor(lambda d: d.get('has_consecutive_zt', False) and d.get('price_pct', 1.0) < 0.40, 10, f_mom, "- 🔥 **低位连板**：刚刚启动的龙头，安全且市场辨识度极高", "MOM"),
        Factor(lambda d: d.get('has_consecutive_zt', False) and d.get('price_pct', 0.0) >= 0.90 and not (d.get('is_first_dip', False) and m_regime != 'BEAR'), -15, f_risk, "- ⚠️ **高位接盘**：股价已被炒高连板，千万别追容易接盘！"),
        Factor(lambda d: d.get('is_first_dip', False) and m_regime != 'BEAR', 20, f_mom, "- 🐉 **龙头首阴**：连板龙头首次缩量温和回调，量价健康且未破5日线，游资经典接力点！"),
        Factor(lambda d: d.get('upper_shadow_pct', 0) > 35, -15, f_risk, "- ⚠️ **诱多预警**：冲高后大幅跳水，上方抛压极重别上当！"),
        Factor(lambda d: d.get('dist_ma20', 0) > 25, -15, f_risk, "- 🚫 **追高预警**：目前涨得太急离均线太远，随时面临暴跌回调"),
        
        Factor(lambda d: in_danger and d.get('mcap', 100e8) < 100e8, -8, f_risk, f"- 📅 **财报防雷**：当前属于{danger_label}，小盘股需防业绩变脸 (扣分)")
    ]
