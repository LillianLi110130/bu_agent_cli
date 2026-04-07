# Skill 使用指南

## 这份文档讲什么
这是一份只面向使用者的 skill 说明。重点回答三件事：

- skill 是怎么用的
- 当前有哪些层级的 skill
- 自定义 skill 应该放到哪里

## 一、怎么使用 skill
在 CLI 里，skill 的调用方式很简单：

```text
@<skill-name> 你的问题或任务
```

例如：

```text
@brainstorming 帮我先梳理这个需求的几个可行方案
@systematic-debugging 帮我定位这个报错的根因
@test-driven-development 先补测试，再修改实现
```

你也可以先输入：

```text
/skills
```

查看当前可用 skill 列表。输入 `@` 后再按 `Tab`，也可以补全 skill 名称。

## 二、当前有哪些层级的 skill
当前 CLI 里的 skill 可以分成 4 类来源。

### 1. 系统内置 skill
这是 CLI 自带的默认 skill，启动后会同步到：

```text
~/.tg_agent/skills/.builtin/
```

它们是默认可用能力，适合作为基础模板使用。这个目录不建议手动改，因为内置内容会在启动时重新同步。

### 2. 用户级自定义 skill
这是你自己的全局 skill，对所有项目都可用。放在：

```text
~/.tg_agent/skills/
```

适合放你长期反复使用的工作方式，比如“写周报”“做 SQL 审核”“按团队规范写测试”。

### 3. 项目级私有 skill
这是只对当前项目生效的 skill，放在：

```text
<workspace>/.tg_agent/skills/
```

适合放项目内部规则、个人临时调优版本，或者你不想直接放进仓库根目录的 skill。

### 4. 项目级共享 skill
这是当前项目里最直接、最适合团队共享的 skill，放在：

```text
<workspace>/skills/
```

如果你希望一个 skill 跟着仓库一起走、让团队成员都能直接用，优先放这里。

## 三、skill 的优先级
如果不同层级里有同名 skill，CLI 会按下面顺序覆盖：

```text
<workspace>/skills
> <workspace>/.tg_agent/skills
> ~/.tg_agent/skills
> ~/.tg_agent/skills/.builtin
```

也就是说：

- 项目里的 skill 优先级最高
- 用户自己的全局 skill 会覆盖系统内置 skill
- 真正决定是否“同名覆盖”的，是 skill 文件里的 `name`

不是看目录名，而是看 skill frontmatter 里的名字。

## 四、插件里的 skill
除了上面 4 类目录来源，插件也可以自带 skill。

这类 skill 一般带命名空间，调用方式类似：

```text
@review-kit:code-review 请审查当前改动的回归风险
```

插件 skill 和普通 skill 的区别是：它通常跟某个插件一起出现，名称里会带上 `插件名:skill名`，适合做更明确的专题能力。

## 五、自定义 skill 放哪里最合适
可以按使用范围来选。

### 想所有项目都能用
放到：

```text
~/.tg_agent/skills/
```

### 只想当前项目可用
放到：

```text
<workspace>/skills/
```

### 只想自己在当前项目里用，不想明显暴露在仓库根目录
放到：

```text
<workspace>/.tg_agent/skills/
```

### 不要放的地方
不要把自定义内容直接写进：

```text
~/.tg_agent/skills/.builtin/
```

这里属于系统内置同步目录，后续可能被覆盖。

## 六、自定义 skill 的最小结构
一个 skill 至少需要一个独立目录，里面放一个 `SKILL.md` 或 `skill.md`。

示例：

```text
skills/
└─ my-review/
   └─ SKILL.md
```

最小内容可以是：

```md
---
name: my-review
description: 用团队要求做代码审查
category: Review
---

# My Review

先关注正确性、回归风险和缺失测试。
输出时先给结论，再给建议。
```

保存后，在对应作用域里就可以这样使用：

```text
@my-review 请审查当前改动
```

## 七、给用户的实际建议

### 需要团队共享
优先放 `<workspace>/skills/`。

### 需要个人长期复用
优先放 `~/.tg_agent/skills/`。

### 只是临时试验
先放 `<workspace>/.tg_agent/skills/`，稳定后再决定是否移到仓库里的 `skills/`。

### 想覆盖默认 skill
使用相同的 `name` 即可，但建议只在明确知道自己要替换什么时这样做。

## 八、最常用的几个操作

查看 skill：

```text
/skills
```

调用 skill：

```text
@brainstorming 帮我拆思路
@my-review 帮我审查改动
@review-kit:code-review 帮我做聚焦式 review
```

补全 skill：

```text
输入 @ 后按 Tab
```

## 一句话总结
把 skill 理解成“可切换的工作模式”即可。全局常用的放 `~/.tg_agent/skills/`，项目共享的放 `<workspace>/skills/`，项目私有的放 `<workspace>/.tg_agent/skills/`，不要改 `.builtin`，需要时直接用 `@skill名` 调用。
