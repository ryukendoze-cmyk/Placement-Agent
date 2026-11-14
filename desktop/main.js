const { app, BrowserWindow, Menu, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const http = require("http");
const https = require("https");
const { spawn } = require("child_process");

const DEFAULT_SERVER_URL = "http://127.0.0.1:8010";
const READY_TIMEOUT_MS = 30000;
const POLL_INTERVAL_MS = 500;

let mainWindow = null;
let settingsWindow = null;
let serverProcess = null;

function getSettingsPath() {
  return path.join(app.getPath("userData"), "careeros-settings.json");
}

function loadServerUrl() {
  try {
    const raw = fs.readFileSync(getSettingsPath(), "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.serverUrl === "string" && parsed.serverUrl.trim()) {
      return parsed.serverUrl.trim().replace(/\/+$/, "");
    }
  } catch {
    // Use default when settings file does not exist or is invalid
  }
  return DEFAULT_SERVER_URL;
}

function saveServerUrl(url) {
  const cleaned = String(url || DEFAULT_SERVER_URL).trim().replace(/\/+$/, "");
  fs.mkdirSync(path.dirname(getSettingsPath()), { recursive: true });
  fs.writeFileSync(getSettingsPath(), JSON.stringify({ serverUrl: cleaned }, null, 2), "utf8");
  return cleaned;
}

function isLocalUrl(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    return parsed.hostname === "127.0.0.1" || parsed.hostname === "localhost";
  } catch {
    return false;
  }
}

function probeUrl(targetUrl, timeoutMs = 2000) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (ok) => {
      if (!settled) {
        settled = true;
        resolve(ok);
      }
    };
    try {
      const parsed = new URL(targetUrl);
      const lib = parsed.protocol === "https:" ? https : http;
      const req = lib.get(
        parsed,
        { timeout: timeoutMs },
        (res) => {
          res.resume();
          finish(res.statusCode >= 200 && res.statusCode < 400);
        }
      );
      req.on("timeout", () => {
        req.destroy();
        finish(false);
      });
      req.on("error", () => finish(false));
    } catch {
      finish(false);
    }
  });
}

function sendLog(line) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("startup-log", line);
  }
}

function sendStatus(message, error = false) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("startup-status", { message, error });
  }
}

function killServerProcess() {
  if (!serverProcess) return;
  const pid = serverProcess.pid;
  serverProcess = null;
  if (!pid) return;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
    } else {
      process.kill(-pid, "SIGTERM");
    }
  } catch {
    // Process already exited
  }
}

function spawnLocalBackend(serverUrl) {
  if (serverProcess) return;
  let port = "8010";
  try {
    const parsed = new URL(serverUrl);
    if (parsed.port) port = parsed.port;
  } catch {
    port = "8010";
  }
  const repoRoot = path.resolve(__dirname, "..");
  sendLog(`Spawning local backend from ${repoRoot} on port ${port}...`);

  serverProcess = spawn(
    "uv",
    ["run", "uvicorn", "app.api.__main__:app", "--port", port],
    {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
      detached: process.platform !== "win32"
    }
  );

  serverProcess.stdout.on("data", (chunk) => {
    chunk
      .toString("utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .forEach((line) => sendLog(line));
  });

  serverProcess.stderr.on("data", (chunk) => {
    chunk
      .toString("utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .forEach((line) => sendLog(line));
  });

  serverProcess.on("exit", (code) => {
    sendLog(`Backend process exited (code ${code ?? "null"}).`);
    serverProcess = null;
  });
}

async function connectOrStart() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  const serverUrl = loadServerUrl();
  const healthUrl = `${serverUrl}/healthz`;
  const readyUrl = `${serverUrl}/readyz`;

  const alreadyUp = await probeUrl(healthUrl, 1500);
  if (alreadyUp) {
    await mainWindow.loadURL(serverUrl);
    return;
  }

  await mainWindow.loadFile(path.join(__dirname, "splash.html"));
  sendLog(`Probed ${healthUrl} — unreachable.`);

  if (!isLocalUrl(serverUrl)) {
    sendStatus(`Remote server ${serverUrl} is unreachable. Check Settings -> Server Connection.`, true);
    return;
  }

  sendStatus("Starting local CareerOS backend...");
  spawnLocalBackend(serverUrl);

  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    const ready = await probeUrl(readyUrl, 1500);
    if (ready) {
      sendLog(`Backend ready at ${readyUrl}. Opening CareerOS...`);
      await mainWindow.loadURL(serverUrl);
      return;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }

  sendStatus("Backend did not become ready within 30s. Check logs above.", true);
}

function openServerSettingsModal() {
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.focus();
    return;
  }
  settingsWindow = new BrowserWindow({
    width: 480,
    height: 250,
    parent: mainWindow || undefined,
    modal: true,
    resizable: false,
    title: "Server Connection",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  settingsWindow.setMenuBarVisibility(false);

  const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#F4F3EC; color:#1B2420; margin:0; padding:20px; }
  h2 { margin:0 0 8px; font-size:16px; color:#1D5C42; }
  p { margin:0 0 12px; font-size:12px; color:#5B6660; }
  input { width:100%; box-sizing:border-box; padding:9px 10px; border:1px solid #C8C6BA; border-radius:8px; font-size:13px; }
  .row { display:flex; justify-content:flex-end; gap:8px; margin-top:16px; }
  button { padding:8px 14px; border-radius:8px; border:none; cursor:pointer; font-size:13px; font-weight:600; }
  .cancel { background:#E3E1D6; color:#1B2420; }
  .save { background:#1D5C42; color:#fff; }
</style>
</head>
<body>
  <h2>Server Connection</h2>
  <p>Enter the JKLU CareerOS server URL (default: http://127.0.0.1:8010).</p>
  <input id="url" type="url" placeholder="http://127.0.0.1:8010" />
  <div class="row">
    <button class="cancel" onclick="window.close()">Cancel</button>
    <button class="save" id="save">Save &amp; Connect</button>
  </div>
  <script>
    const input = document.getElementById("url");
    window.careerosDesktop.getServerUrl().then((u) => { input.value = u; });
    document.getElementById("save").addEventListener("click", async () => {
      await window.careerosDesktop.saveServerUrl(input.value);
      window.close();
    });
  </script>
</body>
</html>`;

  settingsWindow.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html));
  settingsWindow.on("closed", () => {
    settingsWindow = null;
  });
}

function buildAppMenu() {
  const template = [
    {
      label: "File",
      submenu: [{ role: "reload" }, { role: "toggleDevTools" }, { type: "separator" }, { role: "quit" }]
    },
    {
      label: "Settings",
      submenu: [
        {
          label: "Server Connection",
          click: () => openServerSettingsModal()
        }
      ]
    }
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    title: "JKLU CareerOS",
    icon: path.resolve(__dirname, "..", "app", "static", "icons", "icon-512.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  connectOrStart();
}

ipcMain.handle("get-server-url", () => loadServerUrl());
ipcMain.handle("save-server-url", async (_event, url) => {
  const saved = saveServerUrl(url);
  await connectOrStart();
  return saved;
});
ipcMain.handle("retry-connection", async () => {
  await connectOrStart();
});

app.whenReady().then(() => {
  buildAppMenu();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  killServerProcess();
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  killServerProcess();
});
