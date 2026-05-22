---
name: browser
description: Direct browser control via CDP. Use when the user wants to automate, scrape, test, or interact with web pages. Connects to the user's already-running Chrome.
---

# browser-harness

Direct browser control via CDP. For task-specific edits, use `agent-workspace/agent_helpers.py`. For setup, install, or connection problems, read install.md.

## Usage

```bash
browser-harness <<'PY'
new_tab("https://docs.browser-use.com")
wait_for_load()
print(page_info())
PY
```

- Invoke as browser-harness — it's on $PATH. No cd, no uv run.
- Use the heredoc form for every multi-line command. It prevents shell quote mangling inside Python strings and JavaScript snippets.
- First navigation is new_tab(url), not goto_url(url) — goto runs in the user's active tab and clobbers their work.

## Tool call shape

```bash
browser-harness <<'PY'
# any python. helpers pre-imported. daemon auto-starts.
PY
```

run.py calls ensure_daemon() before exec — you never start/stop manually unless you want to.

## Interaction skills

If you start struggling with a specific mechanic while navigating, look in interaction-skills/ for helpers. The available interaction skills with actionable guidance are:
- connection.md
- dialogs.md
- screenshots.md
- tabs.md
- text-targets.md

## What actually works

- Default workflow is DOM/CDP-only. Use page_info(), js(...), DOM queries, wait_for_element(), fill_input(), getBoundingClientRect(), wait_for_network_idle(), http_get(), and raw cdp(...) for normal browser inspection and actions.
- Do not use screenshot/image analysis for normal navigation, page understanding, target discovery, clicking, or verification. Screenshot tooling is outside the default workflow; use it only when the user explicitly asks for a screenshot artifact or visual debugging.
- Page understanding: inspect URL, document state, visible text, links, buttons, inputs, forms, roles, aria labels, hrefs, and bounding rects with js(...). If a page looks missing from DOM/CDP, first verify current_tab(), list_tabs(include_chrome=False), ensure_real_tab(), wait_for_load(), wait_for_element(), and document.body.innerText.
- Targeting: prefer stable text, aria-label, href, role, name, id, selector, or getBoundingClientRect(). For a target with a rect, click the center with click_at_xy(). Do not derive click coordinates from screenshots.
- Clicking: click_at_xy() takes CSS viewport coordinates. Get those coordinates from getBoundingClientRect(), not from screenshot pixels. Hit-testing happens in Chrome's browser process, so coordinate clicks go through iframes / shadow DOM / cross-origin when the CSS viewport coordinate is correct. For document lists, search results, recent items, table rows, cards, and external links, tab detection is part of the click: record list_tabs(include_chrome=False) before clicking, then inspect all tabs after clicking. Do not judge success only from the current tab URL.
- New tabs after clicks: for document lists, search results, recent items, table rows, cards, external links, and any target that might use window.open, always record list_tabs(include_chrome=False) before the click. After the click, inspect all tabs, not only newly-created tabs: the site may open a new tab, reuse an existing tab, or leave the source tab URL unchanged. Switch with switch_tab(target) only when the matching tab is the expected continuation of the task. If it is an auth wall, payment/approval flow, download page, or unrelated content, stop or ask the user before continuing.
- Bulk HTTP: http_get(url) + ThreadPoolExecutor. No browser for static pages (249 Netflix pages in 2.8s).
- After goto: wait_for_load().
- Wrong/stale tab: ensure_real_tab(). Use it when the current tab is stale or internal; the daemon also auto-recovers from stale sessions on the next call.
- Verification: print(page_info()) is the simplest "is this alive?" check. Prefer URL/title/selector/text/network-idle checks. Do not verify normal browser actions with screenshot/image analysis.
- page_info()["title"] is document.title, not the visible page heading. If it is empty, inspect visible headings or body text with js(...) before assuming the page is blank or unloaded.
- DOM reads: use js(...) for inspection and extraction when coordinates are the wrong tool or when the needed information is text/structure rather than visual state.
- Iframe sites (Azure blades, Salesforce): click_at_xy(x, y) passes through; only drop to iframe DOM work when coordinate clicks are the wrong tool.
- Auth wall: redirected to login → stop and ask the user. Don't type credentials unless the user explicitly provides them in text.
- Raw CDP for anything helpers don't cover: pass CDP params as keyword args,
  e.g. `cdp("Runtime.evaluate", expression="document.title", returnByValue=True)`.
  Do not pass a params dict as the second positional argument; that slot is
  `session_id`. Prefer `js(...)` for Runtime.evaluate.

After-action verification:

Do not decide success from a single generic signal unless that signal directly
proves the expected state. Verify the thing the action was supposed to change.

- After filling inputs: read the exact element value / checked / selected state
  with `js(...)`, not URL, title, or `document.body.innerText`.
- After clicking navigation, list, or link targets: inspect tabs, URL/title, and
  visible target text.
- After SPA buttons or form submits: use `wait_for_network_idle()`, then check
  success/error text, changed controls, disabled/loading state, or the expected
  selector.
- After opening menus, dialogs, or popovers: re-read visible controls, overlays,
  `aria-expanded`, `role=dialog/menu/listbox`, and bounding rects.
- After selecting items: verify `aria-selected`, `checked`, `value`, class, or
  nearby selected-state DOM, not only that the text exists somewhere on the page.
- If the expected state is ambiguous, print the relevant DOM state and continue
  from that evidence; do not immediately retry the same action.

DOM/CDP targeting order:

```text
1. Confirm the attached tab: current_tab(), list_tabs(include_chrome=False),
   ensure_real_tab(), page_info().
2. Wait for readiness: wait_for_load(), then wait_for_element(...) for SPA content.
3. Build a compact DOM inventory with js(...): visible text, links, buttons,
   inputs, roles, aria labels, hrefs, disabled states, and bounding rects.
4. Choose a deterministic target by text, aria-label, role, href, name, id,
   selector, or nearby DOM structure.
5. For inputs, use fill_input(); for buttons/links, click the rect center with
   click_at_xy(); for special cases, use raw cdp(...).
6. Verify with URL/title/text/selector/network-idle.
7. If DOM/CDP cannot proceed, report the exact blocker or ask the user. Do not
   switch to screenshot analysis unless the user explicitly asks for visual debugging.
```

Useful DOM inventory snippet:

```python
print(js("""
(() => {
  const pick = e => {
    const r = e.getBoundingClientRect();
    const text = (e.innerText || e.value || e.getAttribute('aria-label') || e.title || '').trim();
    return {
      tag: e.tagName,
      text: text.slice(0, 120),
      id: e.id || '',
      cls: String(e.className || '').slice(0, 120),
      role: e.getAttribute('role') || '',
      aria: e.getAttribute('aria-label') || '',
      href: e.href || '',
      disabled: !!e.disabled || e.getAttribute('aria-disabled') === 'true',
      rect: {x:r.x, y:r.y, w:r.width, h:r.height}
    };
  };
  return [...document.querySelectorAll('a,button,input,textarea,select,[role=button],[role=link],[tabindex]')]
    .filter(e => {
      const r = e.getBoundingClientRect();
      return r.width > 0 && r.height > 0 && r.bottom >= 0 && r.right >= 0 &&
             r.top <= innerHeight && r.left <= innerWidth;
    })
    .slice(0, 80)
    .map(pick);
})()
"""))
```

Text targets in list/table/card UIs:

For document lists, search results, recent files, tables, cards, workspace apps,
and other SPA rows, the visible target text is often a `span`/`div` with a
framework click handler, not an `a[href]`.
Do not stop after `querySelectorAll("a")` returns nothing, and do not assume the
center of a full row container is the real hit target. Search by visible text,
inspect several candidate elements, prefer the smallest visible rect that
contains the exact title, then click the rect center. After clicking, inspect
tabs as part of the click; a stable current-page URL does not prove failure.
Full snippet: `interaction-skills/text-targets.md`.

When a click may open a new tab, detect and decide explicitly. If the user asked
for a named document/page, prefer a tab whose title or URL matches that requested
target over blindly switching to the newest tab.

```python
before = {t["targetId"] for t in list_tabs(include_chrome=False)}
expected_text = "expected document or page title"

click_at_xy(x, y)
wait(1)

tabs = list_tabs(include_chrome=False)
new_tabs = [
    t for t in tabs
    if t["targetId"] not in before
]
print(new_tabs)

# Decide from title/url and the user's task. Switch only if it is the expected next page.
matching_tabs = [
    t for t in tabs
    if expected_text in ((t.get("title") or "") + " " + (t.get("url") or ""))
]
new_matching_tabs = [
    t for t in new_tabs
    if expected_text in ((t.get("title") or "") + " " + (t.get("url") or ""))
]
if new_matching_tabs:
    switch_tab(new_matching_tabs[-1])
    wait_for_load()
    print(page_info())
elif matching_tabs:
    switch_tab(matching_tabs[-1])
    wait_for_load()
    print(page_info())
elif new_tabs:
    target = new_tabs[-1]
    if "expected-host-or-title" in (target.get("url", "") + " " + target.get("title", "")):
        switch_tab(target)
        wait_for_load()
        print(page_info())
    else:
        print("New tab opened; not switching until it is confirmed relevant:", target)
```

## Common form/search workflow

For ordinary search boxes and form inputs, prefer the existing helpers before
inventing selectors, helper names, or raw CDP calls.

```python
new_tab("https://www.baidu.com")
wait_for_load()
wait_for_element("#kw", timeout=5, visible=True)
fill_input("#kw", "深圳天气")
press_key("Enter")
wait_for_load()
print(js("document.body.innerText"))
```

- Open or switch to the correct page first: use `new_tab(url)` for first navigation and `ensure_real_tab()` if the current tab may be stale, internal, or not the intended page.
- Wait before interacting: after navigation use `wait_for_load()`, and before touching dynamic fields use `wait_for_element(selector, timeout=5, visible=True)`.
- `getBoundingClientRect()` already returns viewport coordinates suitable for click_at_xy(). If it returns zero, do not assume a coordinate-system mismatch; first check current tab, selector correctness, load state, and element visibility.
- For normal inputs, use `fill_input(selector, text, timeout=5)`. It focuses, clears, inserts the full text with `Input.insertText`, and dispatches input/change events for framework-managed fields. This avoids duplicate non-ASCII input such as `你你好好`.
- `type_text(text)` only inserts into the currently focused element. Do not use it unless focus was just set intentionally by a coordinate click, `fill_input`, or explicit JS focus.
- Use exact helper names. Use `press_key("Enter")`; do not invent `press_enter()` or similar helpers. Use `press_key()` for keys such as Enter, Tab, Backspace, Escape, and arrows, not for general text input.
- Submit forms with `press_key("Enter")` or a visible `click_at_xy(x, y)` first. Use DOM fallbacks such as `form.submit()` only after helper-based interaction fails.
- Avoid raw `cdp("Input.dispatchKeyEvent", ...)` for keyboard input unless existing helpers are insufficient and the active session has been verified.

## Helper boundary

- Connect to the user's running Chrome. Don't launch your own browser.
- Put task-specific helpers in `agent-workspace/agent_helpers.py`; do not edit core helpers for one-off page logic.
- Prefer existing helpers and raw CDP calls before adding abstractions.

## Gotchas (field-tested)

- Omnibox popups are fake page targets. Filter chrome://omnibox-popup... and other internals when you need a real tab.
- CDP target order != Chrome's visible tab-strip order. Use UI automation when the user means "the first/second tab I can see"; Target.activateTarget only shows a known target.
- Default daemon sessions can go stale. ensure_real_tab() re-attaches to a real page.
- Use DOM/CDP to drive exploration. Build summaries of visible controls, text, forms, links, roles, aria labels, and rects with js(...).
- Prefer compositor-level actions for visible UI targets, but derive coordinates from getBoundingClientRect().
- Current URL unchanged after a click does not mean the click failed. Check tabs first, especially on document/workspace apps that open content in a new or reused tab.
- If a coordinate click misses, do not blindly retry. Re-read the target rect, viewport size, scroll position, visibility, disabled state, and overlay/modals from DOM/CDP. Do not switch to screenshot analysis unless the user explicitly asks for visual debugging.
- If you need framework-specific DOM tricks, check interaction-skills/ first. That is where dropdown, dialog, iframe, shadow DOM, and form-specific guidance belongs.
