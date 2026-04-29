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
                    if len(parts) >= 13:
                        name = parts[1]
                        try:
                            ldr_name = parts[12]
                            # 机构风控：如果板块龙头是 ST 股，说明该板块是垃圾股炒作，直接过滤该板块
                            if "ST" in ldr_name.upper():
                                continue
                                
                            pct = float(parts[5])
                            ldr_pct = float(parts[11])
                            sectors.append((name, pct, ldr_name, ldr_pct))
                        except: pass
                
                sectors.sort(key=lambda x: x[1], reverse=True)
                # 过滤涨幅极小的无意义板块
                sectors = [s for s in sectors if s[1] > 0.5]
                top_sectors = []
                for s in sectors[:limit]:
                    top_sectors.append(f"**{s[0]}** {format_dingtalk_pct(s[1])} ➜ 👑 龙头: **{s[2]}** {format_dingtalk_pct(s[3])}")
                return top_sectors

        except Exception as e:
            log.warning(f"获取热门板块全线失败: {e}", exc_info=True)
            return []

class NewsDigest:
    @staticmethod
    def get_news(limit=5):
        news_list = []
        
        # 1. 首选: Tushare 机构级财联社电报 (过滤水文)
        try:
            token = os.environ.get("TUSHARE_TOKEN", "").strip()
            if token:
                import tushare as ts
                pro = ts.pro_api(token)
                df = pro.news(src='cls', limit=limit+5)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        time_str = row['datetime'][11:16]
                        title = row['title'] if row['title'] else row['content'][:50]+"..."
                        # 过滤无意义简报
                        if len(title) > 8 and "盘前必读" not in title and "早报" not in title:
                            news_list.append(f"> **[{time_str}]** {title}")
                        if len(news_list) >= limit: break
                    if news_list:
                        log.info("机构级财联社新闻获取成功")
                        return news_list
        except Exception as e:
            log.warning(f"Tushare 机构新闻获取失败: {e}")

        # 2. 降级: 新浪 7x24 过滤式财经新闻
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=20&page=1"
            res = requests.get(url, headers={"Referer": "https://finance.sina.com.cn/"}, timeout=5).json()
            data = res.get('result', {}).get('data', [])
            
            # 高价值新闻关键词
            keywords = ["政策", "央行", "突发", "利好", "涨停", "大跌", "新规", "会议", "突破", "美联储", "外资"]
            
            for doc in data:
                title = doc.get('title', '')
                if any(k in title for k in keywords) or len(title) > 20:
                    dt = datetime.fromtimestamp(int(doc['ctime']))
                    news_list.append(f"> **[{dt.strftime('%H:%M')}]** {title}")
                if len(news_list) >= limit: break
                
        except Exception as e:
            log.warning(f"获取新浪新闻兜底失败: {e}")
            
        return news_list

class BriefingRenderer:
    @staticmethod
    def render() -> str:
        date_str = _today_str()
        
        # 1. 宏观数据
        ashare_idx = MacroBrain.get_ashare_indices()
        global_idx = MacroBrain.get_global_indices()
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
