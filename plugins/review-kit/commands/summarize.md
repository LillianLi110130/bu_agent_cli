---
name: summarize
description: Summarize the current workspace or a specific target.
usage: /review-kit:summarize [target]
category: Review
examples:
  - /review-kit:summarize
  - /review-kit:summarize src/auth.py
mode: python
script: scripts/summarize.py
---

Generate a workspace summary prompt from the current project state.
