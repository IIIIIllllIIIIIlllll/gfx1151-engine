# attic/launcher — 旧 Windows 托盘启动器（备份，不再构建）

`start_win.exe` 的旧实现：原生 Win32 托盘程序 + 启动配置面板（launch_panel.inc）。
2026-10 起被取代：托盘能力移入 `gdec-api-win.exe`（`src/api/tray_win.cpp`），
配置面板由网页控制台（`#/engine` 页）取代，根目录 `start_win.exe` 变为最小
转发 stub（`src/start_stub_win.cpp`）。

此处仅为备份，不参与任何构建；代码依赖的 `svcconf` 等接口日后可能漂移，
不保证还能编译。
