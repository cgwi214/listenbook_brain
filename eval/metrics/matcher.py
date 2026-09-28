"""金标匹配器

chunk_id 是 Milvus 自动生成的主键，每次重新导入都会变，因此金标集不能写死 chunk_id，
改用**稳定定位符**（来源文件 / 书名 / 作者 / 内容类型 / 章节 / 关键词）。

匹配规则：定位符中所有**非空**字段必须同时满足，全部为空则不约束（视为恒真）。

字段语义
--------
===================  ============================================
定位符字段           含义
===================  ============================================
source_file          chunk 的 ``source_file_name`` 包含该串
book_name            chunk 的 ``book_name`` 包含该串
author_name          chunk 的 ``author_name`` 包含该串
content_type         chunk 的 ``content_type`` 相等（精确）
entry_name           chunk 的 ``entry_name`` 包含该串
keywords             chunk 的 ``content + title`` 中**全部命中**才算匹配
exclude_keywords     chunk 的 ``content + title`` 中**任一命中**即判不匹配
===================  ============================================
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# 允许出现在金标定位符里的字段
LOCATOR_FIELDS = (
    "source_file",
    "book_name",
    "author_name",
    "content_type",
    "entry_name",
    "keywords",
    "exclude_keywords",
)

# 定位符字段 -> chunk 字段 的映射
_FIELD_TO_CHUNK = {
    "source_file": "source_file_name",
    "book_name": "book_name",
    "author_name": "author_name",
    "content_type": "content_type",
    "entry_name": "entry_name",
}


def _norm(value: Any) -> str:
    """统一转小写字符串，便于做包含匹配。"""
    if value is None:
        return ""
    return str(value).strip().lower()


def _as_keywords(value: Any) -> List[str]:
    """把 keywords 字段规整成字符串列表（兼容 str 与 list 两种写法）。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def chunk_matches_locator(chunk: Dict[str, Any], locator: Dict[str, Any]) -> bool:
    """判断单个切片是否命中单个金标定位符。

    Args:
        chunk: 切片字典（至少含 content / book_name / source_file_name 等字段）。
        locator: 金标定位符字典。

    Returns:
        命中返回 True，否则 False。定位符全空时返回 True（不约束）。
    """
    if not isinstance(locator, dict) or not locator:
        return True

    constrained = False  # 是否至少有一个生效的约束条件

    # ---- 1. 字段包含匹配 ----
    for locator_field, chunk_field in _FIELD_TO_CHUNK.items():
        expected = _norm(locator.get(locator_field))
        if not expected:
            continue

        constrained = True
        actual = _norm(chunk.get(chunk_field))

        if locator_field == "content_type":
            # 内容类型是白名单枚举值，要求精确相等
            if actual != expected:
                return False
        else:
            if expected not in actual:
                return False

    # ---- 2. 关键词必须全部命中 ----
    haystack = f"{chunk.get('content') or ''} {chunk.get('title') or ''}"
    haystack = _norm(haystack)

    for kw in _as_keywords(locator.get("keywords")):
        constrained = True
        if _norm(kw) not in haystack:
            return False

    # ---- 3. 排除关键词：命中任一即判负 ----
    for kw in _as_keywords(locator.get("exclude_keywords")):
        if _norm(kw) in haystack:
            return False

    # 定位符里一个有效约束都没有（例如只写了 exclude_keywords），视为不约束
    return True


def chunk_matches_any(chunk: Dict[str, Any], locators: Sequence[Dict[str, Any]]) -> bool:
    """切片命中任意一个定位符即算相关。

    Args:
        chunk: 切片字典。
        locators: 金标定位符列表。

    Returns:
        命中任一返回 True；定位符列表为空返回 False（无金标时不算相关）。
    """
    if not locators:
        return False
    return any(chunk_matches_locator(chunk, loc) for loc in locators)


def matched_locator_indices(chunk: Dict[str, Any], locators: Sequence[Dict[str, Any]]) -> List[int]:
    """返回该切片命中的全部定位符下标（用于统计金标覆盖度）。"""
    return [i for i, loc in enumerate(locators or []) if chunk_matches_locator(chunk, loc)]


def find_hit_positions(
    retrieved: Sequence[Dict[str, Any]],
    locators: Sequence[Dict[str, Any]],
) -> List[int]:
    """返回召回列表中命中金标的位置下标（0-based，按召回顺序）。"""
    return [i for i, chunk in enumerate(retrieved or []) if chunk_matches_any(chunk, locators)]


def covered_locators(
    retrieved: Sequence[Dict[str, Any]],
    locators: Sequence[Dict[str, Any]],
) -> Set[int]:
    """返回被召回结果覆盖到的金标定位符下标集合（用于算召回率）。"""
    covered: Set[int] = set()
    for chunk in retrieved or []:
        covered.update(matched_locator_indices(chunk, locators))
    return covered
