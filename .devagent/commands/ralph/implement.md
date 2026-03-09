<!--
 * @Author: 何东80296485 donnert@cmbchina.com
 * @Date: 2026-02-06 16:03:03
 * @LastEditors: 何东80296485 donnert@cmbchina.com
 * @LastEditTime: 2026-02-06 16:03:19
 * @FilePath: \ralph_for_agent\.devagent\commands\ralph\implement.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
---
描述：规划计划、完成功能实现
---

## 定义

### 路径定义
- <implement_dir>:{#implement_dir}
- <plan_dir>:{#plan_dir}
- <task_file>:{#task_file}

### 名词定义

- `task_name`:{#task_name}

## 实现

读取<task_file>文件中的任务描述，按以下步骤开始工作

### 步骤一 **依赖检查**

根据当前需要实现的任务内容，确认依赖任务是否已实现，并在<implement_dir>获取依赖任务的实现文档

### 步骤二 **规划步骤**

创建<implement_dir>/<task_name>_implement.md文件
将任务的实现步骤写入 <implement_dir>/<task_name>_implement.md 规划实现步骤部分

<task_name>_implement.md文档使用下述格式：

```markdown
# 功能实现：[TASK_NAME]

## 概述
[2-3句摘要描述]

## 规划实现步骤

- [ ] 步骤一
- [ ] 步骤二
- [ ] 步骤三
  
## 测试验证

- **测试案例一**：✅通过
  - **输入**
  - **预期输出**
  - **实际输出**
- **测试案例一**：❌失败
  - **输入**
  - **预期输出**
  - **实际输出**

```
╔══════════════════════════════════════════════════════════════╗
║                     Test Results                             ║
╠══════════════════════════════════════════════════════════════╣
║ Status:     ✅ ALL TESTS PASSED                              ║
║ Total:      [n] tests                                        ║
║ Passed:     [n](100%)                                        ║
║ Failed:     0                                                ║
║ Flaky:      0                                                ║
║ Duration:   9.1s                                             ║
╚══════════════════════════════════════════════════════════════╝

Artifacts:
📸 Screenshots: [n] files
```

## 代码审查清单

### Summary
- Total files reviewed: [number]
- Critical issues: [number]
- Major issues: [number]
- Minor suggestions: [number]

### File-by-File Analysis

#### [Filename]
**Changes:** [Brief description of what was modified]

**Issues Found:**
- **Critical**: [List critical issues or "None"]
- **Major**: [List major issues or "None"]
- **Minor**: [List minor suggestions or "None"]
- **Positive Notes**: [Acknowledge well-done aspects]

**Recommendations:**
- [Specific action items for improvement]

### Overall Assessment
[Overall quality rating and key takeaways]

## 总结
<!--实现结果总结-->
```

### 步骤三 **实现功能**

根据规划步骤完成功能实现，并对<task_name>_implement.md进行更新，
对于<task_file>中包含测试要求的，请使用**qa-validation-engineer** subagent完成测试工作并对<task_name>_implement.md进行更新。

### 步骤四 **功能检查** 

如果<plan_dir>/PLAN_CHECKLIST.md存在且内容不为空，请使用**code-reviewer** subagent完成对于<plan_dir>/PLAN_CHECKLIST.md清单要求的代码检视工作，并将检查报告更新至<implement_dir>/<task_name>_implement.md文档中。
