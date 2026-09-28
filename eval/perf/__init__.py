"""性能埋点子包

提供节点级 / 阶段级耗时采集与报表生成。
"""

from .timer import (
    PerfTracer,
    get_tracer,
    register_tracer,
    unregister_tracer,
    render_perf_report,
)

__all__ = [
    "PerfTracer", "get_tracer", "register_tracer",
    "unregister_tracer", "render_perf_report",
]
