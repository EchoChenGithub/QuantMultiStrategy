# 量化多策略研究与框架 (基于华泰人工智能系列研报)

## 项目目标

本项目旨在复现华泰证券GLM选股策略的基础上，搭建一个可扩展的量化回测框架，并在同一数据集上系统性地实现、回测和比较多种类型的选股策略（从简单因子到机器学习、深度学习），最终评估不同方法的有效性。

## 项目结构 (初步)

- data/: 存放数据
- src/: 核心代码
  - dataloader/
  - factors/
  - strategies/
  - backtester/
  - visualizer/
  - utils/
- notebooks/: 探索性分析
- results/: 回测结果
- scripts/: 运行脚本
- requirements.txt: 环境依赖
- README.md: 项目说明

## 数据说明 (Data Description)

本项目使用的数据包含两部分：`stocks.csv` (股票行情数据) 和 `factors.csv` (因子及收益数据)。

*   **数据来源:** 基于公开市场数据及对华泰证券2017年《人工智能选股之广义线性模型》研报中因子定义的模拟/计算。
*   **时间范围:** 2012年1月 至 2021年12月
*   **数据频率:** 月度 (每月最后一个交易日截面数据)
*   **主要内容:**
    *   `stocks.csv`: 包含股票代码 (code)、日期 (datetime)、开盘价 (open)、最高价 (high)、最低价 (low)、收盘价 (close)、成交量 (volume) 等基础行情信息。
    *   `factors.csv`: 包含股票代码 (code)、日期 (datetime)、多个因子暴露值 (如 pb_ratio, pe_ratio, roe 等)、股票所属行业代码 (industry_code)、以及**下期收益率** (`next_ret`，作为模型训练的目标变量)。
*   **样本量:** 共约 34,000+ 条月度样本记录。
*   **存储:** 数据文件存放于项目根目录下的 `/data/` 文件夹中，**根据 `.gitignore` 配置，数据文件本身不纳入 Git 版本控制**。

## 环境配置 (Environment Setup)

建议使用 Python 3.8 或更高版本。可以通过以下命令安装所需依赖库：

```bash
pip install -r requirements.txt