# Screenshots

`capture_screenshot()` writes a PNG of the current viewport. By default it saves to `~/.tg_agent/tmp/browser-harness/shot.png` unless `BH_TMP_DIR` is set. The file is in **device pixels** — on a 2× display a 2296×1143 CSS viewport produces a 4592×2286 PNG.

A saved screenshot path is not automatically visual input to the model. When you need the agent to inspect the image, call the `analyze_image(path, prompt)` tool after taking the screenshot.

That matters for two reasons:

1. **Click coordinates are CSS pixels.** Don't read a target off the image and pass it to `click_at_xy()` directly without dividing by `devicePixelRatio`. The simplest workflow is to take the screenshot, look at it in a viewer that shows CSS coordinates, or measure relative positions and use `js("window.devicePixelRatio")` to convert.

2. **Some LLMs reject images > 2000 px per side.** Long sessions on 2× displays will eventually hit this. Pass `max_dim=1800` to downscale the file before it gets into the conversation:

```python
path = capture_screenshot(max_dim=1800)
print(path)
```

The downscale only happens when the image actually exceeds `max_dim`, so it's safe to leave on for every shot.

Then ask the vision tool to inspect it:

```text
analyze_image(path, "Identify visible controls, important text, dialogs, and target coordinates.")
```

Use full-page screenshots (`full=True`) only when you need to see content below the fold — they are much larger and slower than viewport-only.
