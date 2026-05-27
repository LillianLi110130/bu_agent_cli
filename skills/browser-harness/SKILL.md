---
name: browser
description: Direct browser control via CDP. Use when the user wants to automate, scrape, test, or interact with web pages. Connects to the user's already-running Chrome.
---

# browser-harness

Direct browser control via CDP. Use it for browser automation, scraping,
testing, and page interaction when the task should run in the user's browser.

This skill is registered as `browser`; load it with `skill_view(name="browser")`.

For setup, installation, daemon, or connection problems, read `install.md`.
For task-specific helper code, use `agent-workspace/agent_helpers.py`.

## Invocation

Use the `browser_harness` tool on all platforms. Pass the complete Python
script as the `script` argument. The tool starts the underlying harness and
passes the script directly to stdin. Helpers are pre-imported and `run.py`
calls `ensure_daemon()` before executing the script.

```json
{
  "script": "new_tab(\"https://docs.browser-use.com\")\nwait_for_load()\nprint(page_info())\n"
}
```

Rules:

- Use `browser_harness` for every browser-harness script in this repository.
- First navigation is `new_tab(url)`, not `goto_url(url)`. `goto_url()` runs in
  the user's active tab and can clobber their work.
- Stop on auth walls. Do not type credentials unless the user explicitly
  provides them in text.

## Default Workflow

Use DOM/CDP first. Screenshots are not part of normal navigation,
understanding, clicking, or verification.

```text
1. Confirm the attached tab:
   current_tab(), list_tabs(include_chrome=False), ensure_real_tab(), page_info()
2. Navigate or attach:
   new_tab(url), switch_tab(target), wait_for_load()
3. Wait for dynamic content:
   wait_for_element(selector, timeout=5, visible=True) when needed
4. Inspect DOM state with js(...):
   visible text, links, buttons, inputs, roles, aria labels, hrefs, disabled
   states, selected values, and bounding rects
5. Act:
   fill_input() for normal inputs, press_key() for special keys, click_at_xy()
   for visible buttons/links/rows, raw cdp(...) only when helpers are not enough.
   Before clicking anything that may navigate, open content, or use window.open,
   record before_tabs = list_tabs(include_chrome=False)
6. Verify:
   Re-read list_tabs(include_chrome=False) before deciding a click had no
   effect. Then verify URL, title, visible text, selector state, network idle,
   dialogs, toasts, loading state, input values, and other state changed by the
   action
```

Minimal DOM inventory:

```python
print(js("""
(() => [...document.querySelectorAll('a,button,input,textarea,select,[role=button],[role=link],[tabindex]')]
  .filter(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom >= 0 && r.right >= 0 &&
           r.top <= innerHeight && r.left <= innerWidth;
  })
  .slice(0, 80)
  .map(e => {
    const r = e.getBoundingClientRect();
    return {
      tag: e.tagName,
      text: (e.innerText || e.value || e.getAttribute('aria-label') || e.title || '').trim().slice(0, 120),
      id: e.id || '',
      role: e.getAttribute('role') || '',
      aria: e.getAttribute('aria-label') || '',
      href: e.href || '',
      disabled: !!e.disabled || e.getAttribute('aria-disabled') === 'true',
      rect: {x:r.x, y:r.y, w:r.width, h:r.height}
    };
  }))()
"""))
```

## Interaction Skill Routing

Before handling a specialized browser mechanic, read the matching file:

- Connection, install, daemon, or Chrome remote debugging issues:
  `install.md` first, then `interaction-skills/connection.md` if needed.
- Native `alert`, `confirm`, `prompt`, or `beforeunload` dialogs:
  `interaction-skills/dialogs.md`.
- Custom selects, menus, dropdowns, comboboxes, or ARIA listboxes:
  `interaction-skills/dropdowns.md`.
- New or reused tabs, visible tab order, or tab switching:
  `interaction-skills/tabs.md`.
- Named documents, rows, cards, search results, workspace items, or table cells:
  `interaction-skills/text-targets.md`.
- Screenshot artifacts or visual debugging requested by the user:
  `interaction-skills/screenshots.md`.

Other files in `interaction-skills/` cover cookies, downloads, iframes,
cross-origin iframes, shadow DOM, scrolling, viewport, print-to-PDF, uploads,
network requests, drag-and-drop, and profile sync. Use them when the task hits
that mechanism.

## Tab-Aware Clicks

For links, search results, documents, rows, cards, recent items, external links,
and any target that may open or reuse another tab:

1. Record `before_tabs = list_tabs(include_chrome=False)` before clicking.
2. Click once.
3. Wait briefly.
4. Re-read `tabs = list_tabs(include_chrome=False)`.
5. Inspect both new and existing tabs by title and URL.
6. Switch only to a tab confirmed relevant to the user's requested target.
7. Do not report "no effect" until tab changes have been checked.

Many workspace apps leave the source tab URL and body unchanged while opening
or reusing another tab for the actual content.

## After-Action Scan

After every meaningful click, input, submit, navigation, or tab switch, inspect
the state class that may have changed before retrying the same action or
reporting failure.

At minimum after a "no visible effect" action, re-run:

- `page_info()`
- `list_tabs(include_chrome=False)`
- a compact DOM scan for dialogs, modals, drawers, popovers, menus, dropdowns,
  toasts, overlays, tooltips, loading indicators, disabled states, changed
  input/control values, and target text appearing or disappearing

Check these state classes as relevant:

- Navigation/tab: URL, title, route, new tabs, reused tabs.
- Native dialogs: `page_info()` may return `{"dialog": ...}` for `alert`,
  `confirm`, `prompt`, or `beforeunload`; read `interaction-skills/dialogs.md`
  before handling them.
- DOM surfaces: modals, drawers, popovers, menus, dropdowns, toasts, overlays,
  tooltips, confirmation panels, permission prompts rendered in the page.
- Element state: value, checked, selected, disabled, readonly, `aria-expanded`,
  `aria-selected`, class, loading/spinner state.
- Content/network: success/error text, counters, row/list updates,
  `wait_for_network_idle()`, pending loading indicators.
- Focus/frame/download: active element, selection, iframe-local content,
  download or file signals.

If the expected state is ambiguous, print the relevant DOM state and continue
from that evidence. Do not immediately retry the same action.

## Core Rules

- Page understanding: inspect URL, title, visible text, links, buttons, inputs,
  forms, roles, aria labels, hrefs, disabled states, and bounding rects with
  `js(...)`.
- `page_info()["title"]` is `document.title`, not the visible page heading. If
  it is empty, inspect headings or body text before assuming the page is blank.
- Clicking: `click_at_xy()` takes CSS viewport coordinates. Derive them from
  `getBoundingClientRect()`, not screenshot pixels.
- Coordinate clicks go through iframes, shadow DOM, and cross-origin content
  when the viewport coordinate is correct. Drop to frame-specific DOM work only
  when coordinates are the wrong tool.
- After clicks on document lists, search results, recent items, tables, cards,
  and external links, inspect `list_tabs(include_chrome=False)`. Sites may open
  a new tab, reuse an existing tab, or leave the source tab URL unchanged.
- If a click or input appears to do nothing, run the After-Action Scan before
  retrying or reporting failure.
- For raw CDP, pass params as keyword arguments:
  `cdp("Runtime.evaluate", expression="document.title", returnByValue=True)`.
  Do not pass a params dict as the second positional argument; that slot is
  `session_id`.
- Prefer `js(...)` over raw `Runtime.evaluate` for page JavaScript.
- Use `http_get(url)` for static pages and APIs. Wrap it in
  `ThreadPoolExecutor` for bulk fetches. Do not use a browser when HTTP is
  enough.

## Common Recipes

### Form Or Search

```python
new_tab("https://www.baidu.com")
wait_for_load()
wait_for_element("#kw", timeout=5, visible=True)
fill_input("#kw", "深圳天气")
press_key("Enter")
wait_for_load()
print(js("document.body.innerText"))
```

Rules:

- Use `fill_input(selector, text, timeout=5)` for normal inputs. It focuses,
  clears, inserts the full text with `Input.insertText`, and dispatches
  `input/change` events for framework-managed fields.
- Use `type_text(text)` only when focus was just set intentionally.
- Use `press_key("Enter")`, `press_key("Tab")`, `press_key("Escape")`,
  `press_key("Backspace")`, and arrow keys for special keys. Do not invent
  helper names such as `press_enter()`.
- Submit with `press_key("Enter")` or a visible `click_at_xy(x, y)` first. Use
  DOM fallbacks such as `form.submit()` only after helper-based interaction
  fails.
- For multiple similar controls, enumerate candidates first and choose by
  nearby labels, aria/name/id, current value, options, and form structure.

### Text Targets

For named documents, rows, cards, search results, and workspace items, the
visible target is often a `span` or `div` with delegated framework handlers,
not an `a[href]` or native button.

Use `interaction-skills/text-targets.md` before acting. The short version:

- Search by exact visible text.
- Inspect several candidate elements.
- Prefer the smallest visible rect containing the target text.
- Re-read the rect immediately before clicking.
- Record tabs before the click and inspect all tabs after the click.
- Switch only to a tab that is confirmed relevant to the user's requested
  target.

## Helper Boundary

- Connect to the user's running Chrome. Do not launch your own browser unless
  `install.md` explicitly calls for a dedicated automation Chrome during setup
  or troubleshooting.
- Core reusable helpers live in `src/browser_harness/helpers.py`.
- Task-specific or site-specific helpers belong in
  `agent-workspace/agent_helpers.py`.
- Prefer existing helpers and raw CDP calls before adding abstractions.
- Do not edit core helpers for one-off page logic.

## Gotchas

- Omnibox popups and other internal pages can appear as page targets. Use
  `list_tabs(include_chrome=False)` or `ensure_real_tab()` when you need a real
  user page.
- CDP target order is not Chrome's visible tab-strip order. Use platform UI
  automation when the user means "the first/second tab I can see".
- `switch_tab()` attaches the harness to a tab. Use
  `cdp("Target.activateTarget", targetId=tid)` when the tab also needs to become
  visibly active.
- Default daemon sessions can go stale. `ensure_real_tab()` and the daemon's
  stale-session recovery usually fix the next call.
- If `getBoundingClientRect()` returns zero, first check the current tab,
  selector correctness, load state, visibility, and overlays.
- Current URL, title, or body text unchanged after a click is not evidence of
  failure until `list_tabs(include_chrome=False)` has been checked. Many
  workspace apps open or reuse a different tab while leaving the source tab
  unchanged.
