import os
import json
import time
import requests
import logging
from datetime import datetime
import pandas as pd
import akshare as ak

from main import fetch_spot, fetch_hot_sectors, fetch_northbound_flow, Cols, TZ_BJS

log = logging.getLogger(__name__)

def _today_str() -> str:
    return datetime.now(TZ_BJS).strftime('%Y-%m-%d')

def format_dingtalk_pct(pct, is_us=False):
    if pct > 0:
        color = "#2fc25b" if is_us else "#F04864"
        emoji = "🟢" if is_us else "🔴"
        return f'<font color="{color}">{emoji} +{pct:.2f}%</font>'
    elif pct < 0:
        color = "#F04864" if is_us else "#2fc25b"
        emoji = "🔴" if is_us else "🟢"
        return f'<font color="{color}">{emoji} {pct:.2f}%</font>'
    else:
        return f'<font color="#8c8c8c"> 0.00%</font>'

class MacroBrain:
    @staticmethod
    def get_ashare_indices():
        indices_data = {}
        try:
            url = "https://hq.sinajs.cn/list=s_sh000001,s_sz399001,s_sz399006,s_sz399300"
            headers = {"Referer": "https://finance.sina.com.cn/"}
            res = requests.get(url, headers=headers, timeout=5)
            # 返回格式: var hq_str_s_sh000001="上证指数,3055.2013,1.2312,1.23,234123,234123";
            for line in res.text.strip().split('\n'):
                if '="' in line:
                    parts = line.split('="')[1].strip('";').split(',')
                    if len(parts) >= 4:
                        name = parts[0]
                        pct = float(parts[3])
                        # Normalize names
                        if name == "沪深300": target_name = "沪深300"
                        elif name == "创业板指": target_name = "创业板指"
                        elif name == "上证指数": target_name = "上证指数"
                        else: target_name = name
                        
                        indices_data[target_name] = {"pct": pct}
            log.info(f"新浪A股大盘获取成功: {indices_data}")
        except Exception as e:
            log.warning(f"获取新浪A股大盘失败: {e}")
        return indices_data

    @staticmethod
    def get_global_indices():
        indices_data = {}
        us_tech = {}
        try:
            url = "https://hq.sinajs.cn/list=int_dji,int_nasdaq,int_sp500,b_HSI"
            headers = {"Referer": "https://finance.sina.com.cn/"}
            res = requests.get(url, headers=headers, timeout=5)
            for line in res.text.strip().split('\n'):
                if '="' in line:
                    key = line.split('=')[0]
                    parts = line.split('="')[1].strip('";').split(',')
                    
                    if "int_" in key and len(parts) >= 4:
                        name = parts[0]
                        price = float(parts[1])
                        pct = float(parts[3])
                        indices_data[name] = {"pct": pct, "price": price}
                    elif "b_HSI" in key and len(parts) >= 6:
                        indices_data["恒生指数"] = {"pct": float(parts[6]), "price": float(parts[1])}
        except Exception as e:
            log.warning(f"新浪外盘数据失败: {e}")

        # 获取美股核心科技股 (Mag 7)
        try:
            mag7 = ["gb_nvda", "gb_aapl", "gb_msft", "gb_tsla", "gb_googl", "gb_amzn", "gb_meta"]
            url = f"https://hq.sinajs.cn/list={','.join(mag7)}"
            headers = {"Referer": "https://finance.sina.com.cn/"}
            res = requests.get(url, headers=headers, timeout=5)
            for line in res.text.strip().split('\n'):
                if '="' in line:
                    parts = line.split('="')[1].strip('";').split(',')
                    if len(parts) >= 3:
                        name = parts[0]
                        price = float(parts[1])
                        pct = float(parts[2])
                        us_tech[name] = {"pct": pct, "price": price}
        except Exception as e:
            log.warning(f"获取美股科技龙头失败: {e}")
            
        return indices_data, us_tech

class MacroJudgmentEngine:
    @staticmethod
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_judgments():
        judgments = []
        try:
            import yfinance as yf
            import pandas as pd
            
            # 使用 yfinance 静默下载高阶宏观数据
            tickers = yf.Tickers("^TNX ^VIX HG=F GC=F USDCNH=X ^GSPC 000300.SS")
            # 使用 history 一次性拉取过去1个月的数据以计算技术指标
            hist = tickers.history(period="1mo")
            close_df = hist['Close']

            # 1. 铜金比 (Copper/Gold Ratio)
            hg_close = close_df['HG=F'].dropna().iloc[-1]
            gc_close = close_df['GC=F'].dropna().iloc[-1]
            cgr = hg_close / gc_close
            cgr_5d_ago = close_df['HG=F'].dropna().iloc[-6] / close_df['GC=F'].dropna().iloc[-6]
            cgr_trend = "攀升 (Risk-On)" if cgr > cgr_5d_ago else "回落 (衰退预警)"

            # 2. VIX 和 TNX
            vix = close_df['^VIX'].dropna().iloc[-1]
            tnx = close_df['^TNX'].dropna().iloc[-1]

            # 3. 离岸人民币
            cnh = close_df['USDCNH=X'].dropna().iloc[-1]

            # 4. 技术面 (RSI & MTM)
            gspc = close_df['^GSPC'].dropna()
            csi300 = close_df['000300.SS'].dropna()

            gspc_rsi = MacroJudgmentEngine.calc_rsi(gspc).iloc[-1]
            gspc_mtm = gspc.iloc[-1] - gspc.iloc[-6] # 5-day momentum

            csi300_rsi = MacroJudgmentEngine.calc_rsi(csi300).iloc[-1]
            csi300_mtm = csi300.iloc[-1] - csi300.iloc[-6]

            # --- 全球宏观定调 ---
            macro_msg = f"**【聪明钱底牌】** 铜金比近5日**{cgr_trend}**；VIX 恐慌指数报 **{vix:.2f}** "
            if vix < 15: macro_msg += "<font color=\"#F04864\">(极度贪婪)</font>；"
            elif vix > 20: macro_msg += "<font color=\"#2fc25b\">(恐慌对冲)</font>；"
            else: macro_msg += "(情绪平稳)；"
            
            macro_msg += f"美债 10 年期收益率报 **{tnx:.3f}%** "
            if tnx > 4.2: macro_msg += "🔴 **<font color=\"#2fc25b\">(对全球科技股估值形成强压制)</font>**。"
            elif tnx < 3.8: macro_msg += "🟢 **<font color=\"#F04864\">(宽松预期发酵，利好成长股)</font>**。"
            else: macro_msg += "(处于中性震荡区间)。"
            
            judgments.append(macro_msg)

            # --- A股大盘定调 ---
            a_msg = f"**【A股动能诊断】** 沪深 300 5日动量(MTM)为 **{csi300_mtm:+.2f}**，RSI(14)报 **{csi300_rsi:.1f}**。离岸人民币汇率报 **{cnh:.4f}**。"
            if csi300_rsi < 30 and csi300_mtm > 0:
                a_msg += " 🟢 **<font color=\"#F04864\">(超卖且动量拐头，具备极强左侧博弈价值)</font>**。"
            elif csi300_rsi > 70:
                a_msg += " 🔴 **<font color=\"#2fc25b\">(进入超买区，建议规避高位追涨，做好防守)</font>**。"
            elif cnh > 7.25:
                a_msg += " 🔴 **<font color=\"#2fc25b\">(汇率承压，外资流出压力较大)</font>**。"
            else:
                a_msg += " (技术面呈中性，按现有模型执行)。"
                
            judgments.append(a_msg)

            # --- 美股大盘定调 ---
            us_msg = f"**【美股动能诊断】** 标普 500 5日动量(MTM)为 **{gspc_mtm:+.2f}**，RSI(14)报 **{gspc_rsi:.1f}**。"
            if gspc_rsi > 70:
                us_msg += " 🔴 **<font color=\"#2fc25b\">(动能极度过热，随时可能迎来技术性回调)</font>**。"
            elif gspc_rsi < 30:
                us_msg += " 🟢 **<font color=\"#F04864\">(极度恐慌超卖，聪明钱可能开始逢低建仓)</font>**。"
            else:
                us_msg += " (维持原有趋势顺势而为)。"
                
            judgments.append(us_msg)

        except ImportError:
            log.warning("yfinance 或 pandas 未安装，跳过高阶宏观研判")
        except Exception as e:
            log.warning(f"高阶研判引擎运行失败: {e}", exc_info=True)
            judgments.append("> <font color=\"#8c8c8c\">引擎数据抓取异常，研判熔断</font>")

        return judgments

class NewsDigest:
    @staticmethod
    def score_news(title):
        score = 0
        # 黑名单屏蔽：高管人事变动、负面情绪（做多只看利好）、例行会议、互动平台灌水
        anti_words = [
            "高管", "辞职", "离职", "聘任", "董事", "股东大会", "互动平台", 
            "早报", "必读", "提示性公告", "下降", "走低", "回落", "暴跌", 
            "跌停", "不及预期", "减持", "亏损", "下滑", "大跌", "收跌", 
            "低迷", "恶化", "立案", "退市", "违规"
        ]
        if any(w in title for w in anti_words): return -1
        
        # T1 宏观与顶层政策（最高权重，直接定调市场方向）
        t1_words = ["发改委", "工信部", "央行", "国务院", "新规", "印发", "降准", "降息", "证监会", "政治局", "重磅"]
        for w in t1_words:
            if w in title: score += 10
            
        # T2 行业前瞻与业绩指引（核心逻辑，决定个股与板块上限）
        t2_words = ["超预期", "指引", "订单", "需求爆发", "上调", "产能", "供不应求", "扭亏", "净利", "商业化", "突破", "暴增"]
        for w in t2_words:
            if w in title: score += 5
            
        return score

    @staticmethod
    def get_news(limit=5):
        news_list = []
        scored_news = []
        
        # 1. 首选: Tushare 机构级财联社电报
        try:
            token = os.environ.get("TUSHARE_TOKEN", "").strip()
            if token:
                import tushare as ts
                pro = ts.pro_api(token)
                df = pro.news(src='cls', limit=limit+30)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        time_str = row['datetime'][11:16]
                        title = row['title'] if row['title'] else row['content'][:50]+"..."
                        score = NewsDigest.score_news(title)
                        if score > 0:
                            # 加入时间戳辅助排序，同分的情况下越新的排越前面
                            scored_news.append((score, row['datetime'], time_str, title))
                            
                    if scored_news:
                        scored_news.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        for _, _, time_str, title in scored_news[:limit]:
                            news_list.append(f"> **[{time_str}]** {title}")
                        log.info(f"机构级财联社新闻智能提纯成功，获取 {len(news_list)} 条高价值资讯")
                        return news_list
        except Exception as e:
            log.warning(f"Tushare 机构新闻获取失败: {e}")

        # 2. 降级: 新浪 7x24 财经新闻智能提纯
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=50&page=1"
            res = requests.get(url, headers={"Referer": "https://finance.sina.com.cn/"}, timeout=5).json()
            data = res.get('result', {}).get('data', [])
            
            for doc in data:
                title = doc.get('title', '')
                score = NewsDigest.score_news(title)
                if score > 0:
                    dt = datetime.fromtimestamp(int(doc['ctime']))
                    scored_news.append((score, doc['ctime'], dt.strftime('%H:%M'), title))
                    
            if scored_news:
                scored_news.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for _, _, time_str, title in scored_news[:limit]:
                    news_list.append(f"> **[{time_str}]** {title}")
                
        except Exception as e:
            log.warning(f"获取新浪新闻兜底失败: {e}")
            
        return news_list

class BriefingRenderer:
    @staticmethod
    def render() -> str:
        date_str = _today_str()
        
        # 1. 宏观数据
        ashare_idx = MacroBrain.get_ashare_indices()
        global_idx, us_tech = MacroBrain.get_global_indices()
        flow_amt, flow_msg = fetch_northbound_flow()
        
        # 2. 宏观高阶研判
        judgments = MacroJudgmentEngine.get_judgments()
        
        # 3. 新闻摘要
        news = NewsDigest.get_news(limit=7)
        
        # 渲染 Markdown
        lines = []
        lines.append(f"## 🤖 AI 每日市场简报\n*{date_str}*\n")
        
        # --- 全球宏观 ---
        lines.append("---\n### 🌍 全球宏观雷达")
        global_strs = []
        for name, data in global_idx.items():
            pct = data['pct']
            is_us = name in ["纳斯达克", "标普500", "道琼斯"]
            global_strs.append(f"- **{name}**: {data['price']:.2f} {format_dingtalk_pct(pct, is_us)}")
        if global_strs:
            lines.extend(global_strs)
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无实时数据</font>")
            
        # 美股核心科技股 (US Tech Leaders)
        if us_tech:
            us_tech_strs = []
            for name, data in us_tech.items():
                us_tech_strs.append(f"{name} {format_dingtalk_pct(data['pct'], is_us=True)}")
            lines.append(f"> 🇺🇸 **美股核心科技**: " + " | ".join(us_tech_strs))
            
        # --- A股大盘 ---
        lines.append("\n### 🇨🇳 A股大盘体检")
        ashare_strs = []
        for name, data in ashare_idx.items():
            pct = data.get('pct', 0.0)
            close_val = f"{data['close']:.2f} " if 'close' in data else ""
            ashare_strs.append(f"- **{name}**: {close_val}{format_dingtalk_pct(pct)}")
        if ashare_strs:
            lines.extend(ashare_strs)
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无实时数据</font>")
            
        if flow_msg:
            lines.append(f"\n{flow_msg.strip()}")
            
        # --- 聪明钱与高阶研判 ---
        lines.append("\n### 🧠 高阶量化研判 (Smart Money)")
        if judgments:
            for j in judgments:
                lines.append(f"> {j}")
        else:
            lines.append("> <font color=\"#8c8c8c\">研判引擎暂无数据输出</font>")
            
        # --- 市场热点 ---
        lines.append("\n### 📰 核心投研资讯")
        if news:
            lines.extend(news)
        else:
            lines.append("> <font color=\"#8c8c8c\">暂无重大新闻</font>")
            
        lines.append("\n---\n*<font color=\"#8c8c8c\">Antigravity 机构级量化引擎自动生成</font>*")
        return "\n".join(lines)

def send_dingtalk(content: str):
    webhook = os.environ.get('DINGTALK_WEBHOOK')
    if not webhook:
        log.warning("未配置 DINGTALK_WEBHOOK，仅在控制台输出：\n" + content)
        return
        
    try:
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': '🤖 每日市场简报',
                'text': content
            }
        }
        res = requests.post(webhook, json=payload, timeout=10)
        res_dict = res.json()
        if res_dict.get('errcode', 0) != 0:
            log.error(f"❌ 钉钉推送失败: {res_dict}")
        else:
            log.info("✅ 简报推送成功")
    except Exception as e:
        log.error(f"❌ 推送网络请求失败: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 网络探测 THS MCP (作为验证手段，静默失败)
    try:
        log.info("正在探测 THS News MCP 连通性...")
        res = requests.get("https://mcp.xfyun.cn/mcp/ths-news", timeout=5)
        log.info(f"THS MCP 探测返回状态码: {res.status_code}")
    except Exception as e:
        log.warning(f"THS MCP 探测超时或失败，云端可能无法直连: {e}")

    try:
        report = BriefingRenderer.render()
        log.info(f"生成简报如下:\n{report}")
        send_dingtalk(report)
    except Exception as e:
        log.critical(f"简报生成崩溃: {e}", exc_info=True)
