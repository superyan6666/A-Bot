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

class HotStockRadar:
    @staticmethod
    def get_hot_overview(limit=5):
        try:
            log.info("开始拉取 fetch_hot_sectors()")
            hot_stocks = fetch_hot_sectors()
            if hot_stocks:
                log.info(f"fetch_hot_sectors() 返回个股数量: {len(hot_stocks)}")
                sector_counts = {}
                for code, sector in hot_stocks.items():
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
                return [s[0] for s in sorted_sectors[:limit]]
            else:
                log.warning("fetch_hot_sectors() 为空，尝试直接解析新浪行业板块兜底...")
                url = "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
                headers = {"Referer": "https://finance.sina.com.cn/"}
                res = requests.get(url, headers=headers, timeout=5)
                res.encoding = 'gbk'
                text = res.text.split("=")[1].strip().strip(";")
                import json
                data = json.loads(text)
                
                sectors = []
                for k, v in data.items():
                    parts = v.split(',')
                    if len(parts) >= 6:
                        node_code = parts[0]
                        name = parts[1]
                        try:
                            pct = float(parts[5])
                            sectors.append((node_code, name, pct))
                        except: pass
                
                sectors.sort(key=lambda x: x[2], reverse=True)
                # 过滤涨幅极小的无意义板块
                sectors = [s for s in sectors if s[2] > 0.5]
                top_sectors = []
                
                for s in sectors[:limit]:
                    node_code, name, pct = s
                    top_3_stocks = []
                    try:
                        # 动态获取板块前3龙头
                        node_url = f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=4&sort=changepercent&asc=0&node={node_code}"
                        node_res = requests.get(node_url, timeout=3).json()
                        for st in node_res:
                            st_name = st.get('name', '')
                            # 过滤 ST 股
                            if "ST" in st_name.upper(): continue
                            st_pct = float(st.get('changepercent', 0.0))
                            top_3_stocks.append(f"{st_name} {format_dingtalk_pct(st_pct)}")
                            if len(top_3_stocks) >= 3: break
                    except Exception as e:
                        log.warning(f"获取板块 {name} 龙头失败: {e}")
                        
                    if top_3_stocks:
                        stocks_str = " | ".join(top_3_stocks)
                        top_sectors.append(f"**{name}** {format_dingtalk_pct(pct)} ➜ 👑 {stocks_str}")
                        
                return top_sectors

        except Exception as e:
            log.warning(f"获取热门板块全线失败: {e}", exc_info=True)
            return []

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
        
        # 2. 热门板块
        top_sectors = HotStockRadar.get_hot_overview()
        
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
            
        # --- 热门赛道 ---
        lines.append("\n### 🔥 今日资金主线")
        if top_sectors:
            for i, sector in enumerate(top_sectors, 1):
                lines.append(f"{i}. {sector}")
        else:
            lines.append("> <font color=\"#8c8c8c\">市场情绪低迷，未识别到明显主线</font>")
            
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
