# Dropdowns

Use this when a form has custom selects, comboboxes, repeated placeholders, or
dropdowns that do not open after clicking.

## Rules

- Do not identify a dropdown only by placeholder text such as "请选择..." or
  "Select...".
- If multiple dropdowns exist, enumerate candidates before clicking.
- Verify identity with nearby labels, aria/name/id, current value, available
  options, disabled/readonly state, and neighboring form structure.
- If a click appears to do nothing, check disabled/readonly state before
  retrying.
- Some menus render outside the control, often under `document.body`; inspect
  visible listbox/menu/option nodes after opening.

## Candidate inventory

Use this before interacting when several dropdowns share the same placeholder or
when the control's label/position is ambiguous.

```python
print(js("""
(() => {
  const selector = [
    'select',
    '[role=combobox]',
    '[aria-haspopup="listbox"]',
    '.drop-list',
    '.ant-select',
    '.el-select',
    '.select'
  ].join(',');
  const visible = e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom >= 0 && r.right >= 0 &&
           r.top <= innerHeight && r.left <= innerWidth;
  };
  const labelText = e => {
    const id = e.id;
    if (id) {
      const label = document.querySelector(`label[for="${CSS.escape(id)}"]`);
      if (label) return label.innerText.trim();
    }
    const wrapper = e.closest('.form-item,.ant-form-item,.el-form-item,.field,.row,li,td,th,div');
    return wrapper ? (wrapper.innerText || '').trim().slice(0, 300) : '';
  };
  const optionsOf = e => [...e.querySelectorAll('option,[role=option],li,.option,.drop-item,.ant-select-item-option,.el-select-dropdown__item')]
    .map(o => (o.innerText || o.textContent || '').trim())
    .filter(Boolean)
    .slice(0, 20);
  return [...document.querySelectorAll(selector)]
    .filter(visible)
    .map((e, index) => {
      const r = e.getBoundingClientRect();
      return {
        index,
        tag: e.tagName,
        text: (e.innerText || e.textContent || e.value || '').trim().slice(0, 200),
        label: labelText(e),
        id: e.id || '',
        name: e.getAttribute('name') || '',
        role: e.getAttribute('role') || '',
        aria: e.getAttribute('aria-label') || '',
        disabled: !!e.disabled || e.getAttribute('aria-disabled') === 'true' ||
          /disabled|is-disabled/.test(String(e.className || '')),
        readonly: !!e.readOnly || e.getAttribute('aria-readonly') === 'true' ||
          e.getAttribute('readonly') !== null,
        options: optionsOf(e),
        rect: {x:r.x, y:r.y, w:r.width, h:r.height}
      };
    });
})()
"""))
```

`options` may be empty for remote-loaded or portal-rendered dropdowns. In that
case, use labels, aria/name/id, current value, disabled/readonly state, and
nearby form structure to choose a candidate, then open it only to inspect
visible options.

## Probe options safely

After choosing a candidate, open it and inspect visible menu options before
selecting anything.

```python
# x/y should be the center of the confirmed dropdown rect.
click_at_xy(x, y)
wait(0.5)
print(js("""
(() => {
  const optionSelector = [
    '[role=option]',
    '[role=menuitem]',
    '.option',
    '.drop-item',
    '.ant-select-item-option',
    '.el-select-dropdown__item',
    'li'
  ].join(',');
  const visible = e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom >= 0 && r.right >= 0 &&
           r.top <= innerHeight && r.left <= innerWidth;
  };
  return [...document.querySelectorAll(optionSelector)]
    .filter(visible)
    .map((e, index) => {
      const r = e.getBoundingClientRect();
      return {
        index,
        text: (e.innerText || e.textContent || '').trim(),
        disabled: !!e.disabled || e.getAttribute('aria-disabled') === 'true' ||
          /disabled|is-disabled/.test(String(e.className || '')),
        rect: {x:r.x, y:r.y, w:r.width, h:r.height}
      };
    })
    .filter(o => o.text);
})()
"""))
```

If no options appear after opening, do not blindly retry. Check whether the
candidate is disabled/readonly, whether a different dropdown is active, whether
an overlay blocks clicks, and whether the menu renders in a frame or portal.

## Select option

1. Confirm the dropdown identity from label/options/state.
2. Click the dropdown rect center.
3. Read visible options from the opened menu.
4. Click the target option rect center.
5. Verify the dropdown's current value, selected text, or form state with
   `js(...)`.

For destructive or high-impact changes, stop before confirming any follow-up
dialog unless the user explicitly asked to proceed.
