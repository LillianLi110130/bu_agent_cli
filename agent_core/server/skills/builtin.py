"""
Built-in skills for the server environment.

These skills are included by default and don't require file system access.
"""

# List of built-in skills as dictionaries (for ConfigSkillLoader)
# These are loaded by default when using ConfigSkillLoader
BUILTIN_SKILLS = [
    {
        "name": "calculator",
        "display_name": "Calculator",
        "description": "执行基本算术运算，包括加减乘除、小数处理、括号和运算优先级。当用户要求计算、求解数学表达式时使用，例如 '123 + 456 等于多少？'、'计算 15.5 * 3'、'(10 + 5) * 2 - 8' 等。",
        "content": """# Calculator

This skill provides guidance for performing accurate arithmetic calculations.

## Supported Operations

- **Addition**: `a + b`
- **Subtraction**: `a - b`
- **Multiplication**: `a * b`
- **Division**: `a / b`

## Calculation Rules

1. **Order of operations**: Follow standard PEMDAS (Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)
2. **Decimal precision**: Maintain appropriate precision for decimal calculations
3. **Division by zero**: Handle gracefully with an error message
4. **Negative numbers**: Support negative values in calculations

## Examples

```
123 + 456 = 579
15.5 * 3 = 46.5
(10 + 5) * 2 - 8 = 22
100 / 4 = 25
```

## Output Format

Present results clearly with:
- The original expression
- The calculated result
- Step-by-step breakdown for complex calculations (optional, when helpful)

Example:
```
Expression: (10 + 5) * 2 - 8
Step 1: 10 + 5 = 15
Step 2: 15 * 2 = 30
Step 3: 30 - 8 = 22
Result: 22
```
""",
        "category": "Math",
        "source": "config",
        "version": "1.0",
        "tags": ["math", "calculation", "arithmetic"],
    },
    {
        "name": "brainstorming",
        "display_name": "Brainstorming",
        "description": "在进行任何创造性工作之前必须使用 - 创建功能、构建组件、添加功能或修改行为。在实现之前探索用户意图、需求和设计。",
        "content": """# Brainstorming Ideas Into Designs

## Overview

Help turn ideas into fully formed designs and specs through natural collaborative dialogue.

Start by understanding the current project context, then ask questions one at a time to refine the idea. Once you understand what you're building, present the design in small sections (200-300 words), checking after each section whether it looks right so far.

## The Process

**Understanding the idea:**
- Check out the current project state first (files, docs, recent commits)
- Ask questions one at a time to refine the idea
- Prefer multiple choice questions when possible, but open-ended is fine too
- Only one question per message - if a topic needs more exploration, break it into multiple questions
- Focus on understanding: purpose, constraints, success criteria

**Exploring approaches:**
- Propose 2-3 different approaches with trade-offs
- Present options conversationally with your recommendation and reasoning
- Lead with your recommended option and explain why

**Presenting the design:**
- Once you believe you understand what you're building, present the design
- Break it into sections of 200-300 words
- Ask after each section whether it looks right so far
- Cover: architecture, components, data flow, error handling, testing
- Be ready to go back and clarify if something doesn't make sense

## After the Design

**Documentation:**
- Write the validated design to documentation
- Use clear and concise language
- Commit the design document to version control

**Implementation (if continuing):**
- Ask: "Ready to proceed with implementation?"
- Create detailed implementation plan
- Break down into smaller, implementable steps

## Key Principles

- **One question at a time** - Don't overwhelm with multiple questions
- **Multiple choice preferred** - Easier to answer than open-ended when possible
- **YAGNI ruthlessly** - Remove unnecessary features from all designs
- **Explore alternatives** - Always propose 2-3 approaches before settling
- **Incremental validation** - Present design in sections, validate each
- **Be flexible** - Go back and clarify when something doesn't make sense
""",
        "category": "Planning",
        "source": "config",
        "version": "1.0",
        "tags": ["planning", "design", "brainstorming"],
    },
    {
        "name": "code_reviewer",
        "display_name": "Code Reviewer",
        "description": "审查代码的最佳实践、潜在错误和改进建议。检查代码质量、安全性、性能和可维护性。",
        "content": """# Code Review Guidelines

## Overview

Review code for best practices, bugs, security issues, performance problems, and maintainability concerns.

## Review Checklist

**Correctness:**
- Does the code implement the requirements correctly?
- Are there edge cases that aren't handled?
- Is the error handling appropriate?

**Security:**
- Are there any SQL injection, XSS, or other vulnerabilities?
- Is user input properly validated and sanitized?
- Are sensitive data properly protected?

**Performance:**
- Are there any obvious performance bottlenecks?
- Is there proper caching where needed?
- Are database queries optimized?

**Readability:**
- Is the code easy to understand?
- Are variable and function names descriptive?
- Is there adequate documentation?

**Maintainability:**
- Is the code well-organized?
- Are there any code duplications?
- Is the code testable?

## Review Format

Structure your review as:

1. **Summary**: Brief overview of what the code does
2. **Issues**: List any problems found (critical, major, minor)
3. **Suggestions**: Improvement suggestions
4. **Positive aspects**: What's done well
5. **Conclusion**: Overall assessment

## Example Review

```
## Summary
This PR adds user authentication with JWT tokens.

## Issues
- [Critical] No rate limiting on login endpoint (DoS vulnerability)
- [Major] JWT secret is hardcoded in source
- [Minor] Missing input validation on email field

## Suggestions
- Consider using a library for JWT management
- Add unit tests for authentication logic
- Extract token generation to a separate service

## Positive Aspects
- Clean separation of concerns
- Good use of async/await
- Comprehensive error messages

## Conclusion
The implementation is well-structured but needs security improvements before merging.
```
""",
        "category": "Development",
        "source": "config",
        "version": "1.0",
        "tags": ["code", "review", "quality"],
    },
    {
        "name": "debugger",
        "display_name": "Systematic Debugger",
        "description": "系统化调试方法和工具。帮助定位、分析和修复代码问题。",
        "content": """# Systematic Debugging

## Overview

Use a systematic approach to identify, analyze, and fix bugs efficiently.

## Debugging Process

**1. Understand the Problem**
- Reproduce the issue consistently
- Identify the expected vs actual behavior
- Gather relevant context (logs, error messages, stack traces)

**2. Form Hypotheses**
- List possible causes for the issue
- Prioritize hypotheses by likelihood
- Plan tests to validate each hypothesis

**3. Test Hypotheses**
- Add logging/print statements strategically
- Use debugger breakpoints
- Create minimal reproductions

**4. Isolate the Root Cause**
- Binary search: comment out half the code
- Change one variable at a time
- Compare working vs non-working states

**5. Implement Fix**
- Make minimal changes to fix the issue
- Add regression tests
- Document the root cause and fix

**6. Verify**
- Test the fix with the original issue
- Check for unintended side effects
- Run existing test suite

## Common Patterns

**"It works on my machine"**
- Check environment differences (versions, config)
- Verify dependencies are installed correctly
- Look for platform-specific code

**"It used to work"**
- Check recent changes (git diff)
- Look for data migration issues
- Verify external API changes

**"Intermittent bug"**
- Race conditions? Check async code
- Resource leaks? Check for unclosed connections
- Timing issues? Add delays to reproduce

## Questions to Ask

1. What changed since it last worked?
2. What's different between working and non-working cases?
3. Can I create a minimal reproduction?
4. What do the logs show?
5. Is there a similar working example to compare?
""",
        "category": "Development",
        "source": "config",
        "version": "1.0",
        "tags": ["debugging", "troubleshooting"],
    },
    {
        "name": "writer",
        "display_name": "Writing Assistant",
        "description": "写作辅助技能，帮助改进文本的清晰度、结构和风格。",
        "content": """# Writing Assistant

## Overview

Help improve writing clarity, structure, and style for various types of content.

## Principles of Good Writing

**Clarity:**
- Use simple, direct language
- Avoid jargon unless necessary
- Define technical terms when used
- One idea per sentence

**Structure:**
- Start with a clear thesis or purpose
- Use paragraphs to organize related ideas
- Include transitions between sections
- End with a conclusion or call to action

**Style:**
- Use active voice
- Vary sentence length
- Remove unnecessary words
- Choose precise vocabulary

## Writing Types

**Technical Documentation:**
- Start with prerequisites
- Include code examples
- Use consistent terminology
- Provide troubleshooting section

**API Documentation:**
- Describe each endpoint clearly
- Include request/response examples
- Document error codes
- Show authentication method

**Tutorials/How-to Guides:**
- Break into numbered steps
- Include expected outputs
- Add screenshots where helpful
- Provide troubleshooting tips

**Communication/Email:**
- Clear subject line
- State purpose upfront
- Use bullet points for lists
- Include clear call to action

## Editing Checklist

- [ ] Main point is clear in first paragraph
- [ ] Each paragraph has one main idea
- [ ] Transitions connect ideas smoothly
- [ ] Active voice used where appropriate
- [ ] Jargon explained or avoided
- [ ] Examples clarify complex points
- [ ] Conclusion summarizes key points
- [ ] No unnecessary words or phrases

## Revision Process

1. **First pass**: Focus on structure and flow
2. **Second pass**: Improve clarity and word choice
3. **Final pass**: Check grammar and formatting
""",
        "category": "Writing",
        "source": "config",
        "version": "1.0",
        "tags": ["writing", "documentation", "communication"],
    },
]
