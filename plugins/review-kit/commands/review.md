---
name: review
description: Review the current code or a specific target.
usage: /review-kit:review [target]
category: Review
examples:
  - /review-kit:review
  - /review-kit:review auth.py
  - /review-kit:review payment flow
---

# Review Command

You are performing a focused code review.

Review priorities:

- correctness bugs
- regression risks
- missing or weak tests
- unsafe assumptions

Expected output:

- list findings first
- keep the summary brief
- if no findings are discovered, say so explicitly

Target or focus:
{{args}}
