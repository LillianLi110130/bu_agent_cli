Place your Windows shortcut icon here as `crab.ico`.

The portable build script will automatically copy this file into the bundle root
and the desktop shortcut created by `deploy.bat` / `win_deploy.ps1` will use it.

You can also bypass this default location and provide an explicit path with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\windows\build_windows_portable.ps1 -ShortcutIcon D:\path\to\your\crab.ico
```
