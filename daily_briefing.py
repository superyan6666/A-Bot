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
            
        return indices_data

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
        result = {"macro": [], "us_tech": "", "cn_tech": ""}
        try:
            import yfinance as yf
            import pandas as pd
            
            # 引入 ^SKEW (黑天鹅指数) 与 CL=F (原油)
            tickers = yf.Tickers("^TNX ^VIX ^SKEW HG=F GC=F CL=F ^GSPC 000300.SS")
            hist = tickers.history(period="1mo")
            close_df = hist['Close']

            def get_last(ticker):
                s = close_df[ticker].dropna()
                return s.iloc[-1] if not s.empty else 0.0

            def get_mtm(ticker, days=5):
                s = close_df[ticker].dropna()
                if len(s) > days: return s.iloc[-1] - s.iloc[-days-1]
                return 0.0

            hg = get_last('HG=F')
            gc = get_last('GC=F')
            cl = get_last('CL=F')
            vix = get_last('^VIX')
            skew = get_last('^SKEW')
            tnx = get_last('^TNX')
            
            # 离岸人民币通过新浪获取 (修复缺失问题)
            cnh = 0.0
            try:
                res_cnh = requests.get("https://hq.sinajs.cn/list=fx_susdcny", headers={"Referer": "https://finance.sina.com.cn/"}, timeout=5)
                cnh = float(res_cnh.text.split('="')[1].split(',')[1])
            except Exception as e:
                log.warning(f"获取离岸人民币失败: {e}")
            
            cgr = (hg / gc) * 100 if gc else 0
            
            # --- 1. 大宗与汇率 (Commodities & FX) ---
            comm_msg = f"**1. 大宗与汇率**\n> 纽约期金 **{gc:.1f}** | WTI原油 **{cl:.1f}**"
            cnh_str = f"\n> 离岸人民币 **{cnh:.4f}**"
            if cnh > 7.25: 
                cnh_str += " 🔴 <font color=\"#2fc25b\">(离岸人民币贬值承压，外资被动流出风险加剧)</font>"
            
            cgr_str = f"\n> 💡 铜金比(宏观复苏先行器) **{cgr:.4f}**"
            comm_msg += cnh_str + cgr_str
            result["macro"].append(comm_msg)
            
            # --- 2. 期权与利率 (Options & Rates) ---
            opt_msg = "**2. 期权与黑天鹅指标**"
            
            tnx_str = f"\n> 美10年期国债 **{tnx:.3f}%**"
            if tnx > 4.2: tnx_str += " 🔴 <font color=\"#2fc25b\">(美债高企，强力压制全球科技股估值)</font>"
            elif tnx < 3.8: tnx_str += " 🟢 <font color=\"#F04864\">(美债回落，成长股迎流动性溢价)</font>"
            
            vix_str = f"\n> VIX恐慌指数 **{vix:.2f}**"
            if vix < 15: vix_str += " 🟢 <font color=\"#F04864\">(期权市场极度贪婪)</font>"
            elif vix > 20: vix_str += " 🔴 <font color=\"#2fc25b\">(华尔街大资金已启动对冲)</font>"
            
            skew_str = f"\n> SKEW黑天鹅指数 **{skew:.1f}**"
            if skew > 135: skew_str += " ⚠️ <font color=\"#2fc25b\">(尾部风险指标异动，需防范系统性黑天鹅！)</font>"
            
            opt_msg += tnx_str + vix_str + skew_str
            result["macro"].append(opt_msg)
            
            # --- 3. 大盘动能诊断 ---
            csi300 = close_df['000300.SS'].dropna()
            gspc = close_df['^GSPC'].dropna()
            
            csi300_rsi = MacroJudgmentEngine.calc_rsi(csi300).iloc[-1] if not csi300.empty else 50
            gspc_rsi = MacroJudgmentEngine.calc_rsi(gspc).iloc[-1] if not gspc.empty else 50
            csi300_mtm = get_mtm('000300.SS')
            gspc_mtm = get_mtm('^GSPC')

            # 美股技术趋势
            us_msg = f"> 📊 **技术趋势**: 标普500 5日动量(MTM) **{gspc_mtm:+.2f}** (RSI: {gspc_rsi:.1f})"
            if gspc_rsi > 70: us_msg += "\n> 🔴 <font color=\"#2fc25b\">美股动能极度过热，随时面临技术性回调</font>"
            elif gspc_rsi < 30: us_msg += "\n> 🟢 <font color=\"#F04864\">美股恐慌超卖，长线资金建仓点</font>"
            result["us_tech"] = us_msg
            
            # A股技术趋势
            a_msg = f"> 📊 **技术趋势**: 沪深300 5日动量(MTM) **{csi300_mtm:+.2f}** (RSI: {csi300_rsi:.1f})"
            if csi300_rsi < 30 and csi300_mtm > 0: a_msg += "\n> 🟢 <font color=\"#F04864\">A股严重超卖且动量拐头，具备左侧博弈价值</font>"
            elif csi300_rsi > 70: a_msg += "\n> 🔴 <font color=\"#2fc25b\">A股逼近超买区，建议规避追涨</font>"
            result["cn_tech"] = a_msg

        except ImportError:
            log.warning("yfinance 或 pandas 未安装，跳过高阶宏观研判")
        except Exception as e:
            log.warning(f"高阶研判引擎运行失败: {e}", exc_info=True)
            result["macro"].append("> <font color=\"#8c8c8c\">引擎数据抓取异常，研判熔断</font>")

        return result

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
        
        # 1. 宏观高阶研判 (置顶)
        judgments = MacroJudgmentEngine.get_judgments()
        
        # 2. 宏观数据
        ashare_idx = MacroBrain.get_ashare_indices()
        global_idx = MacroBrain.get_global_indices()
        flow_amt, flow_msg = fetch_northbound_flow()
        
        # 3. 新闻摘要
        news = NewsDigest.get_news(limit=7)
        
        # 渲染 Markdown
        lines = []
        lines.append(f"## 🤖 AI 每日市场简报\n*{date_str}*\n")
        
        # --- 聪明钱与高阶研判 (置顶呈现) ---
        lines.append("---\n### 🧠 机构级量化研判 (Smart Money)\n")
        if judgments.get("macro"):
            lines.append("\n\n".join(judgments["macro"]))
        else:
            lines.append("> <font color=\"#8c8c8c\">研判引擎暂无数据输出</font>")
            
        # --- 美股大盘 ---
        lines.append("\n---\n### 🇺🇸 美股大盘体检")
        global_strs = []
        for name, data in global_idx.items():
            if name == "恒生指数": continue # HSI belongs to China logically, but let's keep it here or separate?
            pct = data['pct']
            is_us = name in ["纳斯达克", "标普500", "道琼斯"]
            global_strs.append(f"- **{name}**: {data['price']:.2f} {format_dingtalk_pct(pct, is_us)}")
            
        if "恒生指数" in global_idx:
            pct = global_idx["恒生指数"]['pct']
            global_strs.append(f"- **恒生指数**: {global_idx['恒生指数']['price']:.2f} {format_dingtalk_pct(pct, False)}")
            
        if global_strs:
            lines.extend(global_strs)
        else:
            lines.append("- <font color=\"#8c8c8c\">暂无实时数据</font>")
            
        if judgments.get("us_tech"):
            lines.append(judgments["us_tech"])
            
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
            
        if judgments.get("cn_tech"):
            lines.append(judgments["cn_tech"])
            
        if flow_msg:
            lines.append(f"\n> 💰 **北向资金**: {flow_msg.strip().replace('北向资金: ', '')}")
            
        # --- 市场热点 ---
        lines.append("\n### 📰 核心投研资讯\n")
        if news:
            lines.append("\n\n".join(news))
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
