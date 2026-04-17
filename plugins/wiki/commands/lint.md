---
name: lint
description: 对工作区 Wiki 做一次健康检查。
usage: /wiki:lint [focus]
category: Knowledge
examples:
  - /wiki:lint
  - /wiki:lint contradictions
  - /wiki:lint missing entity pages
---

# Wiki 巡检

请对当前工作区 Wiki 做一次轻量健康检查。

可选关注点：

`{{args}}`

巡检步骤：

1. 先阅读 `llm-wiki/WIKI_AGENT.md`、`llm-wiki/wiki/index.md` 和 `llm-wiki/wiki/log.md` 的最近内容
2. 如果用户指定了关注点，优先检查相关页面；否则抽样检查全库
3. 重点检查：
   - 页面之间是否存在矛盾
   - 是否有已经过时的结论
   - 是否有孤儿页或弱连接页面
   - 是否存在被多次提到但没有独立页面的重要概念或实体
   - 是否缺少交叉链接
   - 是否缺少引用或引用过弱
4. 对发现的问题进行分级：
   - 可直接修复的小问题
   - 需要人工确认的中等问题
   - 需要新增来源或大范围整理的问题
5. 如果是小而明确、低风险的修复，可以直接修并记录
6. 否则输出发现和下一步建议

规则：

- 不要修改原始来源
- 修改范围要小、要保守
- 如果做了修复，需要视情况同步更新 `index.md` 并追加 `log.md`
- 如果没有发现明显问题，也要明确说明，并指出仍然存在的盲区
- 如果发现数据缺口、弱证据或明显空白，要指出下一步建议把哪类来源编译进 Wiki
- 如果发现 `index.md` 中条目缺摘要、分类错位、遗漏重要页面，也应视为需要修复的问题
