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
            df = ak.stock_zh_index_spot_em()
            if df is not None and not df.empty:
                for code, name in [("000300", "沪深300"), ("399006", "创业板指"), ("000001", "上证指数")]:
                    row = df[df["代码"] == code]
                    if not row.empty:
                        indices_data[name] = {
                            "pct": float(row.iloc[0]["涨跌幅"]),
                            "close": float(row.iloc[0]["最新价"])
                        }
        except Exception as e:
            log.warning(f"获取A股大盘指数失败: {e}")
        return indices_data

    @staticmethod
    def get_global_indices():
        indices_data = {}
        # 恒生指数
        try:
            df_hk = ak.stock_hk_spot_em()
            if df_hk is not None and not df_hk.empty:
                row = df_hk[df_hk["代码"] == "800000"] # 恒生指数代码可能变化，备选：直接用 yfinance 或放弃。使用akshare尽量兼容
                if not row.empty:
                    indices_data["恒生指数"] = {"pct": float(row.iloc[0]["涨跌幅"])}
        except Exception:
            pass

        # 美股指数
        try:
            df_us = ak.stock_us_spot_em()
            if df_us is not None and not df_us.empty:
                # 纳斯达克 100 或 标普 500
                for code, name in [("105.NDX", "纳斯达克"), ("105.SPX", "标普500")]:
                    row = df_us[df_us["代码"] == code]
                    if not row.empty:
                        indices_data[name] = {"pct": float(row.iloc[0]["涨跌幅"])}
        except Exception:
            pass
            
        return indices_data

class HotStockRadar:
    @staticmethod
    def get_hot_overview(limit=5):
        try:
            hot_stocks = fetch_hot_sectors()
            # hot_stocks 结构为: {code: sector_name}
            # 我们需要统计各个板块的热度
            sector_counts = {}
            for code, sector in hot_stocks.items():
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # 按成分股数量排序
            sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
            return [s[0] for s in sorted_sectors[:limit]]
        except Exception as e:
            log.warning(f"获取热门板块失败: {e}")
            return []

class NewsDigest:
    @staticmethod
    def get_news(limit=5):
        # 尝试使用 THS MCP (此处预留网络探测与调用逻辑)
        # 实际运行中由于没有官方 client SDK 暂用 akshare 财联社电报兜底
        log.info("尝试拉取最新市场新闻...")
        news_list = []
        try:
            # 降级：使用财联社电报
            df = ak.stock_telegraph_cls()
            if df is not None and not df.empty:
                # 过滤出有标题的新闻
                df = df[df["标题"].str.len() > 0]
                df = df.head(limit)
                for _, row in df.iterrows():
                    news_list.append(f"【{row['发布时间']}】{row['标题']}")
        except Exception as e:
            log.warning(f"获取新闻兜底失败: {e}")
            
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
