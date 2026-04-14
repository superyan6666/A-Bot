🤖 QuantBot - GitHub Actions 顶级量化盯盘机器人

这是一个专为 GitHub Actions 设计的零成本、免维护量化多因子盯盘机器人。依托 GitHub 官方容器，7x24 小时监控全球市场异动，并将自带“止损建议”与“因子评级”的投资机会实时推送给您。

✨ 核心特性 (Features)

🆓 零成本永不停机：基于 GitHub Actions 定时触发（Cron），无服务器维护成本。

🛡️ 绝对降级护城河：摒弃脆弱的爬虫依赖。内置硬核 150 大流动性票池，配合 yfinance 原生免封禁底层，确保数年稳定运行。

🧠 智能“三合一”架构：

⚡ sentinel (高频哨兵)：盘中每半小时扫描全市场活跃资金池，秒级捕获 >3.5% 的分时脉冲与 >4% 的跳空缺口，附带日线 ATR 支撑位与实时新闻。

🎯 matrix (多因子矩阵)：盘尾深度扫描共振点。聚合了 OBV 能量潮底背离、米奈尔维尼 (Minervini) 主升浪模板、TTM Squeeze 动能点火 以及 机构资金 (CMF) 等十大顶级因子。

📅 backtest (自动回测与归因)：每周五自动对齐真实 K 线，生成 T+3 周期的胜率报表，并独立统计每个技术因子在当前市场的真实胜率。

⚖️ 大盘与板块防诱多风控：自动感知 VIX 恐慌指数与大盘趋势 (Regime)。一旦识别到单日“同板块高潮”，立即启动 CROWDING_PENALTY（板块拥挤度降权），拒绝接盘诱多。

🔔 交易的“最后一公里”：推送消息直接附带基于波动率计算的 ATR 科学止损价。支持 Telegram 与 钉钉/飞书 Webhook 实时接收。

🚀 快速开始 (Quick Start)

只需 3 步，即可在您的 GitHub 账户下免费运行此机器人。

1. Fork 本仓库

点击页面右上角的 Fork 按钮，将本项目复制到您自己的 GitHub 账号下。

2. 配置环境变量 (Secrets)

进入您 Fork 后的仓库，点击 Settings -> Secrets and variables -> Actions -> New repository secret，添加以下通知配置：

Secret 名称

描述

是否必填

WEBHOOK_URL

钉钉/飞书/企业微信 Webhook 地址（支持逗号分隔配置多个）

选填

TELEGRAM_BOT_TOKEN

Telegram 机器人的 Token (通过 @BotFather 获取)

选填

TELEGRAM_CHAT_ID

Telegram 接收消息的 Chat ID

选填

注意：通知渠道必须至少填写一种，否则程序无处报警。

3. 启用与运行

点击仓库顶部的 Actions 标签页，确认启用工作流。

在左侧列表中选择 Tickeron Pro AI Bot。

点击 Run workflow，您可以手动选择 matrix 模式进行一次全市场深度扫描，体验量化推送。

🕒 自动化运行时间表

默认运行逻辑（UTC 时间，仅工作日执行）：

Sentinel: 每半小时 (xx:00, xx:30)

Matrix: 每小时 (xx:15)

Backtest: 每周五 (23:00)

⚠️ 免责声明 (Disclaimer)

本项目仅供编程学习与量化策略研究使用。所有技术指标与自动化产生的信号不构成任何投资建议。市场有风险，投资需谨慎！使用者需对自身的投资行为负责。
