# 仅展示修改部分：优化内部超时控制
# ... (前部代码保持一致)

def get_signals() -> tuple[list[Signal], set, int]:
    now = datetime.now(TZ_BJS)
    log.info('🚀 弹性加固型量化引擎启动...')
    if not IS_MANUAL and not is_trading_time(now): return [], set(), 0

    c_conf, pushed = Config(), load_pushed_state()
    
    # 【优化：缩短网络等待上限】
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_spot, f_flow = ex.submit(fetch_spot), ex.submit(get_fund_flow_map)
        try:
            df_raw = f_spot.result(timeout=15) # 15秒必须拿到行情
            flow_map = f_flow.result(timeout=10)
        except FuturesTimeoutError:
            log.error("❌ 核心数据获取超时，可能网络波动")
            return [], pushed, 0

    df_clean, m_ok, m_msg, idx_ret = extract_market_context(df_raw, c_conf)
    log.info(f"📈 扫描全市场: {m_msg}")

    # ... (初选过滤器逻辑)

    log.info(f"✅ 初选捕获 {len(pool)} 只个股，执行深度特征工程...")

    sec_strengths = fetch_sector_strength()
    candidate_data = []
    end_s, start_s = now.strftime('%Y%m%d'), (now - timedelta(days=450)).strftime('%Y%m%d')
    
    # 【优化：提高并行效率，严格控制单股抓取时间】
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(fetch_hist, r[C.S_CODE], start_s, end_s): r for _, r in pool.iterrows()}
        for f in as_completed(futures):
            row = futures[f]
            try:
                # 每只股票的历史数据下载不能超过 5 秒，否则跳过
                hist = f.result(timeout=5) 
                result = process_stock(row, hist, now, m_ok, idx_ret, flow_map.get(row[C.S_CODE], 0.0))
                if result:
                    data, stop, risk = result
                    # 极速获取板块信息 (设置 2s 超时)
                    try:
                        sector = fetch_stock_sector(row[C.S_CODE])
                    except:
                        sector = ""
                    
                    s_pct = sec_strengths.get(sector, 0.0)
                    data.update({
                        'sector': sector, 'sector_pct': s_pct, 
                        'sector_ok': (s_pct > 1.0)
                    })
                    
                    score, level, reas = apply_scoring(data)
                    if score >= 60:
                        # ... (封装 Signal 逻辑)
                        candidate_data.append(Signal(...))
                        pushed.add(row[C.S_CODE])
            except Exception as e:
                log.debug(f"分析跳过 {row[C.S_CODE]}: {e}")

    candidate_data.sort(key=lambda x: x.score, reverse=True)
    return candidate_data, pushed, len(pool)

# ... (后续代码)
