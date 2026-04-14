name: Intraday Right-Side Trading Bot

on:
  schedule:
    # A股交易日期间，每隔 30 分钟运行一次
    - cron: '*/30 1-3,5-7 * * 1-5'
  workflow_dispatch:
    inputs:
      push_empty:
        description: '是否在无信号时也推送状态报告？'
        required: false
        default: false
        type: boolean

permissions:
  contents: write

jobs:
  run-bot:
    runs-on: ubuntu-latest
    
    # 【严谨性建议】：盘中监控对时间要求极高。
    # 设置 4 分钟强行熔断，既能保证信号不过时，也能最大化节省 GitHub 免费时长。
    timeout-minutes: 4
    
    steps:
      # 【一致性修复 1】：升级至 v4，消除 Node 16 废弃警告，加快启动速度
      - name: 检出代码
        uses: actions/checkout@v4

      - name: 搭建 Python 环境
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      # 利用缓存跳过 C/C++ 库(如 numpy, pandas)的编译与下载，大幅提速
      - name: 缓存 pip 依赖
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-quant-v2
          restore-keys: |
            ${{ runner.os }}-pip-quant-

      # 【一致性修复 2】：必须追加 --upgrade 参数！
      # 核心原理：让大型库(pandas/numpy)命中缓存极速安装，同时强迫 pip 检查并拉取最新版的 akshare，防止接口变动导致系统瘫痪。
      - name: 安装依赖
        run: |
          python -m pip install --upgrade pip
          pip install --upgrade akshare pandas numpy requests pytz

      - name: 启动监控引擎
        env:
          DINGTALK_WEBHOOK: ${{ secrets.DINGTALK_WEBHOOK }}
          # 【一致性修复 3】：使用现代化的 inputs 上下文，逻辑运算更安全
          PUSH_EMPTY_RESULT: ${{ inputs.push_empty || 'false' }}
          PYTHONUNBUFFERED: "1"
        run: python main.py

      - name: 保存运行状态
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "[bot] sync state"
          file_pattern: 'pushed_state.json'
