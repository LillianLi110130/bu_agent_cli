from __future__ import annotations
from datetime import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_if_missing(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing.strip():
            return "kept"
    path.write_text(content, encoding="utf-8")
    return "created"


def _ensure_dir(path: Path) -> str:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"路径已存在但不是目录: {path}")
        return "kept"
    path.mkdir(parents=True, exist_ok=True)
    return "created"


def _frontmatter(page_type: str, *, status: str = "active") -> str:
    return (
        "---\n"
        f"type: {page_type}\n"
        f"status: {status}\n"
        f"updated_at: {_now_iso()}\n"
        "---\n\n"
    )


def _wiki_agent_content() -> str:
    return """# Wiki 维护协议

当前工作区使用一个由 LLM 持续维护的知识 Wiki。你的职责不是临时总结，而是持续编译、更新和交叉链接这个 Wiki。

## 三层结构

1. 原始来源：`kb/raw/`
   - 是事实源头
   - 编译进 Wiki 的过程中只读
   - 绝不直接改写原始来源
2. Wiki 页面：`kb/wiki/`
   - 由 Agent 维护
   - 分为 `summaries/`、`concepts/`、`entities/`、`synthesis/`
   - 是长期沉淀层，不是一次性回答缓存
3. 维护协议：`kb/WIKI_AGENT.md`
   - 规定 Wiki 的结构、约定和执行方式

补充说明：

- `kb/state/` 用于保存编译状态等记录
- 当前主要使用 `kb/state/ingest-status.md` 记录 `kb/raw/` 中哪些文件已经编译完成

## 核心原则

- 优先更新已有页面，不要轻易创建近义重复页
- 当待编译来源路径与现有摘要页 frontmatter 中的 `source_path` 一致时，应视为同一来源的再次编译
- 同一来源重复编译时，优先更新已有摘要页、概念综合页和实体页，不新建重复页面
- 新来源进入后，要把它编译进现有认知，而不是只生成孤立摘要
- 页面之间必须主动建立链接，避免形成孤儿页
- 如果新旧结论冲突，要显式保留冲突和证据，不要静默覆盖
- 结论尽量绑定具体摘要页和原始来源路径
- `kb/wiki/index.md` 是内容导航入口，回答前优先读取
- `kb/wiki/log.md` 记录整个知识库的演化时间线

## 页面分类

### `summaries/`

- 单个来源的摘要页
- 关注：这份来源说了什么、带来了什么新信息、涉及哪些概念和实体
- 应尽量提炼高价值信息，而不是机械重写整份原始文档

### `concepts/`

- 跨来源概念综合页
- 关注：综合观点、核心争议、趋势、未解决问题

### `entities/`

- 命名实体页
- 适用于人物、组织、产品、地点、项目、论文、方法、框架等会被反复引用的对象

### `synthesis/`

- 查询后沉淀下来的综合页
- 适用于跨来源比较、专题分析、重要连接、阶段性综合结论

## 命名与落页规则

- 新页面文件名默认优先使用中文，保持简洁、稳定、可读，只有在专有名词本身以英文更常见、更稳定，或使用中文会明显造成歧义时，才使用英文文件名
- 同一对象尽量只维护一个主页面
- 如果来源中的命名实体是核心对象，必须创建或更新对应实体页
- 如果一个对象只是边缘性提及且无长期价值，可以不建实体页，但要能说明原因
- 页面内涉及 `相关概念`、`相关实体`、`关键来源` 等栏目时，只要本次编译已经确定目标页及其路径，就默认写入标准 Markdown 相对链接。

## 概念识别规则

- 概念是跨来源、可持续累积的知识单元，不是单篇来源标题的重复
- 如果一份来源提出了稳定的制度、概念、争议、比较维度、流程或分类法，应优先落到概念综合页
- 如果某个概念已经在来源摘要、实体页或日志中反复出现，就不应只停留在零散提及里，应创建或更新对应概念综合页
- 如果一份来源明显改变了某个已有综合结论，也应更新对应概念综合页
- 如果本次没有创建任何概念综合页，应能说明原因

## 实体识别规则

- 命名实体包括但不限于：人物、公司/组织、产品、项目、地点、书名、论文名、方法名、框架名、活动名，以及其他可能被后续重复引用的专有名词
- 如果某个命名实体是来源的核心对象，必须创建或更新对应实体页
- 如果某个命名实体在当前来源中被多次提及，或明显可能在后续资料中继续出现，也应创建或更新对应实体页
- 如果只是一次性、边缘性提及，且没有长期价值，可以不建实体页
- 如果本次没有创建任何实体页，应能说明原因

## 索引维护规则

每次将新来源编译进 Wiki，或进行了有意义的结构性维护后，都应更新 `kb/wiki/index.md`。

索引不是随意罗列文件，而是内容导向的总目录。回答问题前应优先读它，再决定读取哪些页面。

索引中的每个页面条目应尽量包含：

- 页面链接
- 一行摘要
- 必要时补充更新时间、来源数、状态

推荐条目格式：

`- [页面名](相对路径) - 一行摘要`

索引按分类组织：

- Overview
- Summaries
- Concepts
- Entities
- Synthesis

维护要求：

- 新增或显著更新页面后，必须同步更新索引
- 避免只写“暂无”而不补目录说明
- 避免把索引写成日志；索引应服务于导航和检索
- 如果某一类页面很多，优先保留高信号摘要而不是长段说明

## 编译记录规则

`kb/state/ingest-status.md` 用于记录 `kb/raw/` 中已经编译完成的来源。

维护要求：

- 这里记录已经编译过的来源
- 条目中的 `source` 使用纯路径文本，不使用 Markdown 链接
- 当编译时没有指定来源路径，应检查 `kb/raw/` 中哪些来源还没有出现在这个文件里
- 每当一个来源成功编译完成后，都应立即把它补充到这个文件里

建议使用统一条目格式：

`- source: kb/raw/example.md`
`  - first_compiled_at: 2026-04-20`
`  - last_compiled_at: 2026-04-20`

## 日志维护规则

`kb/wiki/log.md` 是追加式日志，不要重写历史。

建议使用统一标题格式：

`## [YYYY-MM-DD] 操作类型 | 标题`

其中操作类型建议使用可读中文，例如：

- `初始化`
- `编译`
- `查询`
- `巡检`
- `维护`

如果同一来源被再次编译，日志中应明确说明“本次是二次编译”或“本次是补充编译”，并简要写明主要补充或修正点。
如果日志中提到已经存在的摘要页、概念综合页、实体页或综合页，应使用标准 Markdown 相对链接，而不是只写纯文本名称。

## 操作约定

### 编译来源

编译前优先读取：

- `kb/wiki/index.md`
- `kb/state/ingest-status.md`

如果没有指定来源路径，就把 `kb/raw/` 中尚未登记为已编译的来源逐个编译。
每完成一个来源，都先完成一次最小闭环，再处理下一个来源。

### 查询

查询前优先读取：

- `kb/wiki/index.md`

回答时只读必要页面；如果 wiki 本身不足，再回退读取原始来源。
除非用户明确要求写回 wiki，否则默认只回答并给出沉淀建议，不直接修改页面。

### 巡检

巡检前优先读取：

- `kb/WIKI_AGENT.md`
- `kb/wiki/index.md`
- `kb/wiki/log.md` 的最近内容

如果用户指定了关注点，优先检查相关页面；否则抽样检查全库。

处理策略：

- 小而低风险的问题：可直接修复
- 中等问题：给出建议并说明影响范围
- 大问题：给出后续应把哪些来源编译进 Wiki，或需要怎样维护

## 页面结构建议

以下模板不是死板格式，但应尽量保持稳定。

### Summary Page

```markdown
---
type: summary
status: active
updated_at: 2026-04-13T10:00:00+08:00
source_path: kb/raw/example.md
source_kind: article
---

# 来源：示例来源

## 摘要
- 

## 关键信息
- 

## 涉及概念
- [示例概念](../concepts/示例概念.md)

## 涉及实体
- [示例实体](../entities/示例实体.md)

## 新增 / 修正 / 冲突
- 新增：
- 修正：
- 冲突：
```

### Concept Page

```markdown
---
type: concept
status: active
updated_at: 2026-04-13T10:00:00+08:00
---

# 概念：示例概念

## 当前综合结论
- 

## 核心争议或不确定点
- 

## 相关概念
- [示例概念二](./示例概念二.md)

## 相关实体
- [示例实体](../entities/示例实体.md)

## 关键来源
- [示例来源](../summaries/示例来源.md)

## 未解决问题
- 
```

### Entity Page

```markdown
---
type: entity
status: active
updated_at: 2026-04-13T10:00:00+08:00
entity_kind: organization
---

# 实体：示例实体

## 它是谁 / 是什么
- 

## 当前已知的重要结论
- 

## 相关概念
- [示例概念](../concepts/示例概念.md)

## 相关实体
- [示例实体](./示例实体.md)

## 关键来源
- [示例来源](../summaries/示例来源.md)

## 备注 / 争议 / 待确认
- 
```

### Synthesis Page

```markdown
---
type: synthesis
status: active
updated_at: 2026-04-13T10:00:00+08:00
---

# 综合：示例专题

## 问题
- 

## 当前综合结论
- 

## 相关概念
- [示例概念](../concepts/示例概念.md)

## 相关实体
- [示例实体](../entities/示例实体.md)

## 关键来源
- [示例来源](../summaries/示例来源.md)

## 后续问题
- 
```
"""


def _index_content() -> str:
    return """# Wiki 索引

这是当前工作区 Wiki 的总目录，也是回答问题时的首要入口。

使用方式：

- 先从这里定位相关页面
- 再下钻阅读少量最相关的 `summaries / concepts / entities / synthesis`
- 只有在 wiki 本身不足时，才回退读取 `kb/raw/`

条目格式约定：

- 每个条目尽量写成 `- [页面名](路径) - 一行摘要`
- 一行摘要优先说明“这页解决什么问题”或“记录了什么对象/概念”

## 总览

- [Overview](./Overview.md) - Wiki 总览页
- [维护规范](../WIKI_AGENT.md) - Wiki 结构与维护规则
- [操作日志](./log.md) - 编译、查询、巡检等活动记录

## Summaries

- 暂无摘要页。新增来源摘要后，请在这里追加条目。

## Concepts

- 暂无概念综合页。新增概念综合页后，请在这里追加条目。

## Entities

- 暂无实体页。新增实体页后，请在这里追加条目。

## Synthesis

- 暂无综合页。新增综合页后，请在这里追加条目。
"""


def _log_content() -> str:
    date = datetime.now().date().isoformat()
    return (
        "# Wiki 日志\n\n"
        "这里按时间顺序追加记录重要 Wiki 操作。\n\n"
        f"## [{date}] 初始化 | 初始化 Wiki 工作区\n"
        "- 创建了基础目录结构和初始文件。\n"
    )


def _ingest_status_content() -> str:
    return """# 编译记录

这里记录已经编译进 Wiki 的原始来源。
"""

def _home_content() -> str:
    return (
        _frontmatter("home")
        + "# Overview\n\n"
        + "这是当前工作区的知识 Wiki。\n\n"
        + "它的目标不是保存原始资料本身，而是把资料中的结论、概念、争议和关联逐步编译进一个可持续维护的知识层，减少每次查询都重新从原始资料出发。\n\n"
        + "## 导航\n\n"
        + "- [索引](./index.md)\n"
        + "- [维护规范](../WIKI_AGENT.md)\n"
        + "- [操作日志](./log.md)\n\n"
        + "## 页面类型\n\n"
        + "- [Summaries](./summaries/) - 单个来源的摘要与证据入口\n"
        + "- [Concepts](./concepts/) - 跨来源概念综合页、综合结论与比较框架\n"
        + "- [Entities](./entities/) - 人物、组织、产品、地点等命名实体\n"
        + "- [Synthesis](./synthesis/) - 查询后沉淀的专题综合、比较与分析页面\n\n"
        + "## 使用方式\n\n"
        + "- 将原始资料放入 `kb/raw/`\n"
        + "- 编译新的原始来源，持续补充摘要页、概念综合页和实体页\n"
        + "- 基于现有 Wiki 查询问题，必要时再回看原始来源\n"
        + "- 定期检查矛盾、缺口与孤儿页，并持续维护链接关系\n\n"
        + "## 维护原则\n\n"
        + "- 原始资料只读\n"
        + "- 正式知识沉淀在 `kb/wiki/`\n"
        + "- 页面之间应主动建立交叉链接\n"
    )

def main() -> int:
    working_dir = Path.cwd()
    wiki_root = working_dir / "kb"

    created_dirs: list[str] = []
    created_files: list[str] = []
    kept_files: list[str] = []

    for rel_dir in (
        "kb/raw",
        "kb/state",
        "kb/wiki",
        "kb/wiki/summaries",
        "kb/wiki/concepts",
        "kb/wiki/synthesis",
        "kb/wiki/entities",
    ):
        status = _ensure_dir(working_dir / rel_dir)
        if status == "created":
            created_dirs.append(rel_dir)

    files = {
        "kb/WIKI_AGENT.md": _wiki_agent_content(),
        "kb/wiki/index.md": _index_content(),
        "kb/state/ingest-status.md": _ingest_status_content(),
        "kb/wiki/log.md": _log_content(),
        "kb/wiki/Overview.md": _home_content(),
    }
    for rel_path, content in files.items():
        status = _write_if_missing(working_dir / rel_path, content)
        if status == "created":
            created_files.append(rel_path)
        else:
            kept_files.append(rel_path)

    print("LLM Wiki 已初始化。")
    print(f"根目录: {wiki_root}")
    if created_dirs:
        print("已创建目录：")
        for item in created_dirs:
            print(f"  - {item}")
    if created_files:
        print("已创建文件：")
        for item in created_files:
            print(f"  - {item}")
    if kept_files:
        print("保留已有文件：")
        for item in kept_files:
            print(f"  - {item}")
    print("下一步建议：")
    print("  1. 把资料放进 kb/raw/")
    print("  2. 用 llm-wiki skill 编译新的原始来源")
    print("  3. 用 llm-wiki skill 基于现有 Wiki 回答问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
