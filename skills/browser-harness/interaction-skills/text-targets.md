# Text Targets

Use this when the user asks for a named document, row, card, search result, or
workspace item and ordinary `a[href]` / button discovery does not find the
target.

Many SPA list UIs render the visible target as a `span` or `div` with framework
or delegated click handling. `onclick: false` does not mean the element is not
clickable; React/Vue/AntD-style handlers often do not appear as DOM `onclick`
properties.

Search by visible text, inspect several candidates, choose the smallest visible
rect that contains the exact target text, re-read its rect immediately before
clicking, and then inspect tabs. Do not judge success only from the current tab
URL.

```python
import json

target_text = "requested document title"
print(js(f"""
(() => {{
  const needle = {json.dumps(target_text)};
  const selectors = [
    'a', 'button', 'span', 'div',
    '[role=button]', '[role=link]', '[tabindex]', '[onclick]',
    '[class*="name"]', '[class*="title"]'
  ].join(',');
  const pick = e => {{
    const r = e.getBoundingClientRect();
    const text = (e.innerText || e.textContent || '').trim();
    return {{
      tag: e.tagName,
      text: text.slice(0, 160),
      id: e.id || '',
      cls: String(e.className || '').slice(0, 160),
      role: e.getAttribute('role') || '',
      aria: e.getAttribute('aria-label') || '',
      href: e.href || '',
      onclick: !!e.onclick || e.hasAttribute('onclick'),
      area: Math.round(r.width * r.height),
      rect: {{x:r.x, y:r.y, w:r.width, h:r.height}}
    }};
  }};
  return [...document.querySelectorAll(selectors)]
    .filter(e => {{
      const text = (e.innerText || e.textContent || '').trim();
      if (!text.includes(needle)) return false;
      const r = e.getBoundingClientRect();
      return r.width > 0 && r.height > 0 && r.bottom >= 0 && r.right >= 0 &&
             r.top <= innerHeight && r.left <= innerWidth;
    }})
    .map(pick)
    .sort((a, b) => a.area - b.area || a.text.length - b.text.length)
    .slice(0, 12);
}})()
"""))
```

After choosing the best candidate, click its current rect center and inspect
tabs. Prefer a newly opened matching tab, then an existing matching tab, then a
new tab only if it is the expected continuation of the task.

```python
import json

before = {t["targetId"] for t in list_tabs(include_chrome=False)}

# Re-read the chosen element immediately before clicking; dynamic lists can move.
rect = js(f"""
(() => {{
  const needle = {json.dumps(target_text)};
  const candidates = [...document.querySelectorAll('span,div,a,button,[role=button],[role=link],[tabindex],[onclick]')]
    .filter(e => (e.innerText || e.textContent || '').trim().includes(needle))
    .map(e => {{
      const r = e.getBoundingClientRect();
      return {{x:r.x, y:r.y, w:r.width, h:r.height, area:r.width*r.height}};
    }})
    .filter(r => r.w > 0 && r.h > 0)
    .sort((a, b) => a.area - b.area)[0];
  return candidates || null;
}})()
""")
if not rect:
    raise RuntimeError(f"text target not visible: {target_text}")

click_at_xy(rect["x"] + rect["w"] / 2, rect["y"] + rect["h"] / 2)
wait(1)

tabs = list_tabs(include_chrome=False)
new_tabs = [t for t in tabs if t["targetId"] not in before]

def title_or_url(tab):
    return (tab.get("title") or "") + " " + (tab.get("url") or "")

new_matching_tabs = [t for t in new_tabs if target_text in title_or_url(t)]
matching_tabs = [t for t in tabs if target_text in title_or_url(t)]

if new_matching_tabs:
    switch_tab(new_matching_tabs[-1])
    wait_for_load()
elif matching_tabs:
    switch_tab(matching_tabs[-1])
    wait_for_load()
elif new_tabs:
    print("New tab opened but not confirmed:", new_tabs)
else:
    print(page_info())
```
