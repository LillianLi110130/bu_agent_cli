# Web Frontend

Desktop validation console for the BU Agent local worker flow.

- React
- Ant Design
- Less Modules
- Vite
- TypeScript

## Install

```bash
cd web
npm install
```

## Run Against `test_server.py`

1. Start the Python server from the repository root:

```bash
$env:TEST_SERVER_USER_HOME=".tmp_test_server_home"
conda run -n 314 python test_server.py
```

2. Start the frontend in another terminal:

```bash
cd web
npm run dev
```

3. Open the Vite URL shown in the terminal, usually `http://127.0.0.1:5173`.

The Vite dev server proxies `/web-console/*` requests to `http://127.0.0.1:8000`, so no extra CORS setup is needed for local validation.

`TEST_SERVER_USER_HOME` is optional but recommended on Windows. It isolates the test server's
`~/.tg_agent` runtime state and avoids local permission conflicts during validation.

## Optional Authorization Header

If your server path requires authentication, you can inject the header through Vite env vars:

```bash
$env:VITE_REMOTE_CONSOLE_AUTHORIZATION="Bearer <token>"
npm run dev
```

Or provide only the token value:

```bash
$env:VITE_REMOTE_CONSOLE_AUTH_TOKEN="<token>"
npm run dev
```

Standard request APIs use `axios`, while the SSE event stream uses authenticated `fetch` so the browser can send custom headers with `text/event-stream`.

## Optional Mock Mode

If you want to run the UI without the Python server, enable mock mode:

```bash
$env:VITE_REMOTE_CONSOLE_USE_MOCK="true"
npm run dev
```

By default, mock mode is disabled so the page connects to the local Python server.

## Build

```bash
npm run build
```
