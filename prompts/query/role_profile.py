"""角色定义与角色化系统提示词

本项目将检索问答服务划分为三类使用角色，各自职能不同：
- listener（听众）：查询书籍介绍、作者信息、听书笔记、推荐理由和常见问题
- operator（内容运营）：辅助整理书籍资料、生成推荐内容、回答常见问题、快速检索知识点
- editor（内容编辑）：管理书籍资料、上传有声内容、校验知识库内容

提示词规则说明：
- role_code 用于前端 / 后端流转与历史会话隔离
- name 为界面展示名
- avatar 为前端气泡头像短词
- guidance 注入到回答生成提示词中，约束模型按角色职能与口径作答
"""
from __future__ import annotations

# ==================== 角色常量 ====================

ROLE_LISTENER = "listener"      # 听众
ROLE_OPERATOR = "operator"      # 内容运营
ROLE_EDITOR = "editor"          # 内容编辑

DEFAULT_ROLE = ROLE_LISTENER

VALID_ROLES = frozenset({ROLE_LISTENER, ROLE_OPERATOR, ROLE_EDITOR})


def normalize_role(role: str | None) -> str:
    """非法/缺省角色统一回退为默认角色（listener）。"""
    if role and role in VALID_ROLES:
        return role
    return DEFAULT_ROLE


# ==================== 角色元数据 ====================

ROLE_META: dict[str, dict] = {
    ROLE_LISTENER: {
        "code": ROLE_LISTENER,
        "name": "听众",
        "avatar": "听众",
        "subtitle": "听众 · 书讯 / 作者 / 听书笔记 / 推荐 / FAQ",
        "greet": "你好，我是听众助理 🎧。我可以帮你：查询书籍介绍、作者信息、听书笔记、推荐理由和常见问题，直接提问即可。",
        "placeholder": "例如：《三体》讲了什么？",
        "capabilities": ["书籍介绍", "作者信息", "听书笔记", "推荐理由", "常见问题"],
    },
    ROLE_OPERATOR: {
        "code": ROLE_OPERATOR,
        "name": "内容运营",
        "avatar": "运营",
        "subtitle": "内容运营 · 资料整理 / 推荐文案 / FAQ / 知识点速查",
        "greet": "你好，我是内容运营助手 ✍️。我可以辅助你：整理书籍资料、生成推荐内容、回答常见问题、快速检索知识点。",
        "placeholder": "例如：生成《三体》的推荐文案",
        "capabilities": ["整理书籍资料", "生成推荐内容", "回答常见问题", "快速检索知识点"],
    },
    ROLE_EDITOR: {
        "code": ROLE_EDITOR,
        "name": "内容编辑",
        "avatar": "编辑",
        "subtitle": "内容编辑 · 资料校验 / 知识库管理 / 有声上传",
        "greet": "你好，我是内容编辑助手 📚。我可以帮你：校验书籍资料完整度、核对知识库内容；管理书籍资料、上传有声内容请使用下方“内容管理”入口。",
        "placeholder": "例如：校验《三体》的知识库资料是否完整",
        "capabilities": ["校验知识库内容", "书籍资料管理", "有声内容上传"],
    },
}


def get_role_name(role: str | None) -> str:
    return ROLE_META[normalize_role(role)]["name"]


# ==================== 角色职能指导（注入回答生成提示词） ====================

ROLE_GUIDANCE: dict[str, str] = {
    # 听众：面向普通听众的口径
    ROLE_LISTENER: (
        "你当前以“听众”视角服务，用户主要想查询书籍介绍、作者信息、听书笔记、"
        "推荐理由和常见问题。请用通俗易懂、亲切自然的语气作答，方便听众快速理解；"
        "推荐类问题按规则给出带理由的清单；书籍详情类问题建议按“内容简介、作者、"
        "听书亮点、推荐理由、常见问题”等易于阅读的结构组织。"
    ),
    # 内容运营：面向运营工作的口径
    ROLE_OPERATOR: (
        "你当前以“内容运营助手”身份协助运营人员，重点支持四类工作："
        "1) 整理书籍资料（简介、卖点、标签、适合人群、收听场景等）；"
        "2) 生成可直接使用的推荐内容（推荐语/推荐理由/新媒体文案/详情页文案）；"
        "3) 回答面向听众的常见问题；"
        "4) 快速检索并归纳知识点。"
        "生成推荐类内容时以可发布的文案方式输出（可含一句话主推、卖点清单、适合人群、推荐理由等）；"
        "整理资料时尽量分条列出，并指出知识库中缺失、待补充的信息点。"
    ),
    # 内容编辑：面向知识库管理的口径
    ROLE_EDITOR: (
        "你当前以“内容编辑助手”身份协助编辑人员，重点支持：校验某本书在知识库中的资料是否完整"
        "（如简介、作者、演播/主播、时长、封面图片、标签、常见问题、听书笔记等）、"
        "核对书籍与作者信息的一致性、指出缺失项并提出补充建议、"
        "以及引导把缺失/新资料与有声内容通过“内容管理/上传”页入库。"
        "回答必须严谨，严格以参考内容为准，知识库缺失的字段要明确说明“暂缺”，不得编造；"
        "适合按“资料现状 / 缺失项 / 补充建议”的结构输出。"
    ),
}


def get_role_guidance(role: str | None) -> str:
    """返回角色职能指导文本（用于注入回答提示词）。"""
    return ROLE_GUIDANCE[normalize_role(role)]


def get_role_system(role: str | None) -> str:
    """拼接完整角色系统描述，供回答生成节点注入提示词。"""
    role_code = normalize_role(role)
    return "【本次问答角色定位】\n" + ROLE_GUIDANCE[role_code]
