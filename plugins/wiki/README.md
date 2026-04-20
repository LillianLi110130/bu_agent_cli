# wiki

用于在当前工作区维护持续积累的 Markdown 知识 Wiki 的内置插件。

核心结构：

- `llm-wiki/raw/`：原始来源，只读
- `llm-wiki/wiki/`：正式 Wiki 页面
- `llm-wiki/WIKI_AGENT.md`：维护协议
- `llm-wiki/wiki/log.md`：知识库演化记录
- `llm-wiki/state/ingest-status.md`：已编译来源记录

Wiki 页面分为四类：

- `summaries/`：单个来源的摘要页
- `concepts/`：跨来源的概念综合页
- `entities/`：人物、组织、产品、地点等命名实体页
- `synthesis/`：查询后沉淀的专题综合、比较和分析页

包含资源：

- `/wiki:init`
- `/wiki:ingest <source-path>`：编译指定来源；不带参数时，会逐个处理 `llm-wiki/raw/` 下尚未记录为已编译的来源
- `/wiki:query <question>`
- `/wiki:lint [focus]`

推荐起手流程：

```text
/wiki:init
/wiki:ingest llm-wiki/raw/example.md
/wiki:query 目前这个 Wiki 里有哪些核心知识？
/wiki:lint
```
