const CACHE_NAME = "jklu-careeros-v2";
const PRECACHE_ASSETS = [
  "/static/app.js",
  "/static/responsive.css",
  "/static/campus_drive.js",
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/vendor/htmx.min.js",
  "/static/vendor/tailwindcss.js",
  "/static/vendor/chart.umd.min.js",
  "/static/vendor/fonts.css"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      Promise.allSettled(
        PRECACHE_ASSETS.map((url) =>
          fetch(url, { cache: "no-cache" }).then((res) => {
            if (res.ok) return cache.put(url, res);
            return null;
          })
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Never cache API, auth, or streaming endpoints
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/auth/")) {
    return;
  }

  // Static & vendored assets: cache-first, populate cache on miss
  if (
    url.pathname.startsWith("/static/vendor/") ||
    url.pathname === "/static/app.js" ||
    url.pathname === "/static/responsive.css" ||
    url.pathname === "/static/campus_drive.js" ||
    url.pathname.startsWith("/static/icons/")
  ) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (response && response.ok) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return response;
        });
      })
    );
    return;
  }

  // Navigation requests: network-only (no-store) so authenticated role HTML is
  // never cached across role switches or sign-out, with offline fallback page.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request, { cache: "no-store" }).catch(() =>
        new Response(
          "<!doctype html><html><body style='font-family:sans-serif;background:#F4F3EC;color:#1D5C42;padding:2rem;'><h2>JKLU CareerOS Offline</h2><p>Unable to reach the CareerOS server. Check your connection and retry.</p></body></html>",
          { status: 503, headers: { "Content-Type": "text/html; charset=utf-8" } }
        )
      )
    );
  }
});
