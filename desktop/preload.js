const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("careerosDesktop", {
  onStartupLog: (cb) => {
    ipcRenderer.on("startup-log", (_event, line) => cb(line));
  },
  onStartupStatus: (cb) => {
    ipcRenderer.on("startup-status", (_event, status) => cb(status));
  },
  getServerUrl: () => ipcRenderer.invoke("get-server-url"),
  saveServerUrl: (url) => ipcRenderer.invoke("save-server-url", url),
  retryConnection: () => ipcRenderer.invoke("retry-connection")
});
