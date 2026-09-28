"""评估模块

对听书知识库的检索与生成质量做量化评估，并提供可视化界面驱动。

子模块::

    eval/
    ├── dataset/    金标集（jsonl）、反例集、说明文档
    ├── hooks/     rerank 后抽取 chunk_id 的钩子
    ├── metrics/   P/R/F1、Macro-F1、生成层指标、报表渲染
    ├── perf/      分阶段耗时埋点
    ├── reports/   评估结果落盘目录（自动创建）
    ├── runner.py  评估主流程
    └── reset.py   刷新（清理评估导入的数据）
"""

__all__ = ["runner", "reset", "hooks", "metrics", "perf"]
