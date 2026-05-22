# Screenshots

Screenshots are not part of the default browser-harness workflow. Use DOM/CDP
inspection and actions for normal navigation, page understanding, target
discovery, clicking, and verification.

Read this file only when the user explicitly asks for a screenshot artifact or
visual debugging evidence.

`capture_screenshot()` writes a PNG of the current viewport. By default it saves
to `~/.tg_agent/tmp/browser-harness/shot.png` unless `BH_TMP_DIR` is set. The
file is in device pixels, not CSS viewport pixels.

Do not use screenshot-derived coordinates for normal clicks. Prefer
getBoundingClientRect() and pass the rect center to click_at_xy().

If the user explicitly asks for visual debugging, remember that a saved PNG path
is not automatically visual input to the model. Only then may you call the
`analyze_image(path, prompt)` tool after taking the screenshot.

Use full-page screenshots (`full=True`) only when the requested artifact must
include content below the fold; they are much larger and slower than
viewport-only screenshots.
