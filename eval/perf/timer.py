"""分阶段耗时埋点

设计要点
--------
1. **零侵入**：采集器按 ``task_id`` 注册到全局字典，节点基类只会做一次
   ``get_tracer(task_id)`` 查询；没有注册采集器时返回 ``None``，
   正常业务链路不产生任何额外开销。

2. **线程安全**：LangGraph 的四路召回是并行执行的，因此采集器内部用
   ``threading.Lock`` 保护共享结构，避免并发写导致数据错乱。

3. **两级采集**：
   - 节点级（node）：由 ``BaseNode.__call__`` 自动打点，无需各节点改动；
   - 阶段级（stage）：由评估/服务层手工圈定，例如"导入阶段""查询阶段"。

4. **分位数**：同一节点被多次调用（批量评估时会跑几十条 query），
   输出 P50 / P95 比只看均值更能暴露长尾问题。
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Dict, List, Optional, Any


# ------------------------------------------------------------------
# 全局采集器注册表（key: task_id）
# ------------------------------------------------------------------
_TRACERS: Dict[str, "PerfTracer"] = {}
_REGISTRY_LOCK = threading.Lock()


def register_tracer(task_id: str) -> "PerfTracer":
    """为指定任务注册并返回一个新的采集器（同 id 重复注册会覆盖）。"""
    tracer = PerfTracer(task_id)
    with _REGISTRY_LOCK:
        _TRACERS[task_id] = tracer
    return tracer


def get_tracer(task_id: Optional[str]) -> Optional["PerfTracer"]:
    """按 task_id 取采集器，不存在返回 None（调用方据此决定是否打点）。"""
    if not task_id:
        return None
    with _REGISTRY_LOCK:
        return _TRACERS.get(task_id)


def unregister_tracer(task_id: str) -> None:
    """任务结束后注销采集器，释放内存。"""
    with _REGISTRY_LOCK:
        _TRACERS.pop(task_id, None)


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _percentile(values: List[float], pct: float) -> float:
    """线性插值分位数，pct 取值 0~100。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


class _Recorder:
    """同名事件的耗时聚合器。"""

    def __init__(self, name: str):
        self.name = name
        self.durations: List[float] = []   # 单位：毫秒

    def add(self, duration_ms: float) -> None:
        self.durations.append(duration_ms)

    def to_dict(self) -> Dict[str, Any]:
        ds = self.durations
        return {
            "name": self.name,
            "calls": len(ds),
            "total_ms": round(sum(ds), 2),
            "avg_ms": round(statistics.fmean(ds), 2) if ds else 0.0,
            "min_ms": round(min(ds), 2) if ds else 0.0,
            "max_ms": round(max(ds), 2) if ds else 0.0,
            "p50_ms": round(_percentile(ds, 50), 2),
            "p95_ms": round(_percentile(ds, 95), 2),
        }


class PerfTracer:
    """一次任务的性能采集器。

    Usage::

        tracer = register_tracer(task_id)
        with tracer.stage("import"):
            ...
        tracer.count("chunks", 128)
        report = tracer.summary()
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._lock = threading.RLock()
        self._nodes: Dict[str, _Recorder] = {}
        self._stages: Dict[str, _Recorder] = {}
        self._counters: Dict[str, float] = {}
        self._running_nodes: Dict[int, tuple] = {}   # thread/调用栈 -> (name, t0)
        self._running_stages: Dict[int, tuple] = {}
        self._started_at = time.perf_counter()
        self._ended_at: Optional[float] = None

    # ---------------- 生命周期 ----------------
    def finish(self) -> None:
        """标记任务结束（不传则 summary 时取当前时间）。"""
        self._ended_at = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = self._ended_at if self._ended_at is not None else time.perf_counter()
        return (end - self._started_at) * 1000

    # ---------------- 节点级打点 ----------------
    def node_start(self, name: str) -> None:
        self._running_nodes[id(threading.current_thread())] = (name, time.perf_counter())

    def node_end(self, name: str) -> None:
        now = time.perf_counter()
        key = id(threading.current_thread())
        started = self._running_nodes.pop(key, None)
        if started is None:
            return
        start_name, t0 = started
        with self._lock:
            self._nodes.setdefault(start_name or name, _Recorder(start_name or name)).add((now - t0) * 1000)

    # ---------------- 阶段级打点 ----------------
    def stage_start(self, name: str) -> None:
        self._running_stages[id(threading.current_thread())] = (name, time.perf_counter())

    def stage_end(self, name: str) -> None:
        now = time.perf_counter()
        key = id(threading.current_thread())
        started = self._running_stages.pop(key, None)
        if started is None:
            return
        start_name, t0 = started
        with self._lock:
            self._stages.setdefault(start_name or name, _Recorder(start_name or name)).add((now - t0) * 1000)

    class _Stage:
        """``with tracer.stage("xxx"):`` 上下文管理器。"""

        def __init__(self, tracer: "PerfTracer", name: str):
            self._tracer = tracer
            self._name = name

        def __enter__(self):
            self._tracer.stage_start(self._name)
            return self._tracer

        def __exit__(self, exc_type, exc, tb):
            self._tracer.stage_end(self._name)
            return False

    def stage(self, name: str) -> "_Stage":
        return PerfTracer._Stage(self, name)

    # ---------------- 计数类指标 ----------------
    def count(self, name: str, value: float = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_count(self, name: str, value: float) -> None:
        with self._lock:
            self._counters[name] = value

    # ---------------- 输出 ----------------
    def summary(self) -> Dict[str, Any]:
        """汇总为可序列化字典。"""
        with self._lock:
            total = self.elapsed_ms
            nodes = [r.to_dict() for r in self._nodes.values()]
            stages = [r.to_dict() for r in self._stages.values()]

            for item in nodes:
                item["pct_of_total"] = round(item["total_ms"] / total * 100, 2) if total else 0.0
            for item in stages:
                item["pct_of_total"] = round(item["total_ms"] / total * 100, 2) if total else 0.0

            nodes.sort(key=lambda x: x["total_ms"], reverse=True)
            stages.sort(key=lambda x: x["total_ms"], reverse=True)

            return {
                "task_id": self.task_id,
                "total_ms": round(total, 2),
                "nodes": nodes,
                "stages": stages,
                "counters": dict(self._counters),
            }


# ------------------------------------------------------------------
# 报表渲染
# ------------------------------------------------------------------
def _fmt_table(title: str, rows: List[Dict[str, Any]], total_ms: float) -> List[str]:
    """渲染单张耗时表，返回文本行列表。"""
    lines: List[str] = []
    if not rows:
        return lines

    header = f"{title:<28}{'次数':>6}{'总耗时ms':>12}{'均值ms':>10}{'P50':>10}{'P95':>10}{'占比':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['name']:<28}{r['calls']:>6}{r['total_ms']:>12.1f}"
            f"{r['avg_ms']:>10.1f}{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}"
            f"{r['pct_of_total']:>7.1f}%"
        )
    return lines


def render_perf_report(summary: Dict[str, Any]) -> str:
    """把 summary 渲染成人类可读的文本报表。"""
    lines: List[str] = []
    lines.append("=" * 84)
    lines.append("性能埋点报表（分阶段耗时）")
    lines.append("=" * 84)
    lines.append(f"任务 ID : {summary.get('task_id', '')}")
    lines.append(f"总耗时  : {summary.get('total_ms', 0):.1f} ms "
                 f"({summary.get('total_ms', 0) / 1000:.2f} s)")

    counters = summary.get("counters") or {}
    if counters:
        lines.append("")
        lines.append("【计数指标】")
        for k, v in counters.items():
            lines.append(f"  - {k}: {v if isinstance(v, int) or v != int(v) else int(v)}")

    stages = summary.get("stages") or []
    if stages:
        lines.append("")
        lines.append("【阶段耗时】")
        lines.extend(_fmt_table("阶段", stages, summary.get("total_ms", 0)))

    nodes = summary.get("nodes") or []
    if nodes:
        lines.append("")
        lines.append("【节点耗时】（由 BaseNode 自动打点，按总耗时降序）")
        lines.extend(_fmt_table("节点", nodes, summary.get("total_ms", 0)))

    lines.append("=" * 84)
    return "\n".join(lines)
