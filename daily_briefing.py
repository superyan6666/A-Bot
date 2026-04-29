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
            # int_dji (道琼斯), int_nasdaq (纳斯达克), int_sp500 (标普), b_HSI (恒指)
            url = "https://hq.sinajs.cn/list=int_dji,int_nasdaq,int_sp500,b_HSI"
            headers = {"Referer": "https://finance.sina.com.cn/"}
            res = requests.get(url, headers=headers, timeout=5)
            # 返回格式: var hq_str_int_nasdaq="纳斯达克,15927.90,1.23,1.5, ..."; 
            # 港股格式可能不同，按逗号分隔，第4个通常是涨跌幅或第3个。
            for line in res.text.strip().split('\n'):
                if '="' in line:
                    key = line.split('=')[0]
                    parts = line.split('="')[1].strip('";').split(',')
                    
                    if "int_" in key and len(parts) >= 4:
                        name = parts[0]
                        pct = float(parts[3])
                        indices_data[name] = {"pct": pct}
                    elif "b_HSI" in key and len(parts) >= 6:
                        # 恒指: "恒生指数,16000.00,15000.00,16500.00,16200.00,200.00,1.25"
                        # 通常第6或第7是涨跌幅
                        indices_data["恒生指数"] = {"pct": float(parts[6])}
            log.info(f"新浪外盘获取成功: {indices_data}")
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
                # var S_KV = {"new_bljc":"new_bljc,玻璃建材,55,13.23,2.45, ..."}
                text = res.text.split("=")[1].strip().strip(";")
                import json
                data = json.loads(text)
                
                sectors = []
                for k, v in data.items():
                    parts = v.split(',')
                    if len(parts) >= 6:
                        name = parts[1]
                        try:
                            pct = float(parts[5])
                            sectors.append((name, pct))
                        except: pass
                
                sectors.sort(key=lambda x: x[1], reverse=True)
                top_sectors = [s[0] for s in sectors[:limit]]
                log.info(f"新浪兜底板块提取成功: {top_sectors}")
                return top_sectors

        except Exception as e:
            log.warning(f"获取热门板块全线失败: {e}", exc_info=True)
            return []

class NewsDigest:
    @staticmethod
    def get_news(limit=5):
        log.info("尝试拉取新浪 7x24 财经滚播新闻...")
        news_list = []
        try:
            # lid=2509 表示A股/财经相关滚播
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=10&page=1"
            headers = {"Referer": "https://finance.sina.com.cn/"}
            res = requests.get(url, headers=headers, timeout=5).json()
            
            data = res.get('result', {}).get('data', [])
            if data:
                for doc in data[:limit]:
                    # 时间戳转换
                    dt = datetime.fromtimestamp(int(doc['ctime']))
                    time_str = dt.strftime('%H:%M')
                    title = doc.get('title', '')
                    if title:
                        news_list.append(f"【{time_str}】{title}")
                log.info(f"新浪新闻获取成功，条数: {len(news_list)}")
            else:
                log.warning("新浪新闻接口返回数据为空")
        except Exception as e:
            log.warning(f"获取新闻失败: {e}", exc_info=True)
            
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
        lines.append(f"📊 **AI 每日市场简报 · {date_str}**\n")
        
        # --- 全球宏观 ---
        lines.append("━━━ 🌍 **全球宏观雷达** ━━━")
        global_strs = []
        for name, data in global_idx.items():
            pct = data['pct']
            sign = "+" if pct > 0 else ""
            global_strs.append(f"{name}: {sign}{pct}%")
        if global_strs:
            lines.append("- " + " | ".join(global_strs))
        else:
            lines.append("- 暂无全球指数实时数据")
            
        # --- A股大盘 ---
        lines.append("\n━━━ 🇨🇳 **A股大盘体检** ━━━")
        ashare_strs = []
        for name, data in ashare_idx.items():
            pct = data['pct']
            sign = "+" if pct > 0 else ""
            ashare_strs.append(f"{name}: {sign}{pct}%")
        if ashare_strs:
            lines.append("- " + " | ".join(ashare_strs))
        else:
            lines.append("- 暂无A股指数数据")
            
        if flow_msg:
            lines.append(flow_msg.strip())
            
        # --- 热门赛道 ---
        lines.append("\n━━━ 🔥 **今日热门主线** ━━━")
        if top_sectors:
            for i, sector in enumerate(top_sectors, 1):
                lines.append(f"{i}. {sector}")
        else:
            lines.append("- 市场情绪低迷，未识别到明显主线")
            
        # --- 市场热点 ---
        lines.append("\n━━━ 📰 **市场热点追踪** ━━━")
        if news:
            for i, n in enumerate(news, 1):
                lines.append(f"{i}. {n}")
        else:
            lines.append("- 暂无重大新闻")
            
        lines.append("\n---\n*🤖 由 Antigravity 量化引擎自动生成*")
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
