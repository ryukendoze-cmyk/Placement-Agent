/* Theme controller for JKLU CareerOS (light / dark mode persistence + Chart.js sync). */
(function () {
  var STORAGE_KEY = "careeros-theme";

  function isDark() {
    return document.documentElement.classList.contains("dark");
  }

  function updateToggleButtons(dark) {
    var btns = document.querySelectorAll("#theme-toggle, [data-theme-toggle]");
    btns.forEach(function (btn) {
      btn.setAttribute("aria-pressed", dark ? "true" : "false");
      btn.setAttribute("title", dark ? "Switch to light mode" : "Switch to dark mode");
      btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
      var lbl = btn.querySelector(".theme-toggle-label");
      if (lbl) {
        lbl.textContent = dark ? "Light Mode" : "Dark Mode";
      }
    });
  }

  function syncChartTheme(dark) {
    if (typeof Chart === "undefined") return;
    var textColor = dark ? "#D0D7DE" : "#4A483F";
    var gridColor = dark ? "#252E3D" : "#EDEBE3";
    if (Chart.defaults) {
      Chart.defaults.color = textColor;
      Chart.defaults.borderColor = gridColor;
    }
    var instances = Chart.instances ? Object.values(Chart.instances) : [];
    instances.forEach(function (chart) {
      if (!chart || !chart.options) return;
      if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
        chart.options.plugins.legend.labels.color = textColor;
      }
      var scales = chart.options.scales || {};
      Object.keys(scales).forEach(function (axis) {
        var s = scales[axis];
        if (!s) return;
        if (s.ticks) s.ticks.color = textColor;
        if (s.grid && s.grid.display !== false) s.grid.color = gridColor;
      });
      if (chart.config && chart.config.type === "doughnut" && chart.data && Array.isArray(chart.data.datasets)) {
        chart.data.datasets.forEach(function (ds) {
          ds.borderColor = dark ? "#12171F" : "#FFFFFF";
        });
      }
      try {
        chart.update("none");
      } catch (_err) {
        /* ignore chart redraw race */
      }
    });
  }

  function applyTheme(theme, persist) {
    var dark = theme === "dark";
    var root = document.documentElement;
    root.classList.toggle("dark", dark);
    root.setAttribute("data-theme", dark ? "dark" : "light");

    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute("content", dark ? "#090C10" : "#1D5C42");
    }

    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, dark ? "dark" : "light");
      } catch (_err) {
        /* ignore storage quota errors */
      }
    }

    updateToggleButtons(dark);
    syncChartTheme(dark);
    try {
      window.dispatchEvent(new CustomEvent("careeros:themechange", { detail: { theme: dark ? "dark" : "light" } }));
    } catch (_err) {
      /* ignore */
    }
  }

  function initialTheme() {
    try {
      var saved = localStorage.getItem(STORAGE_KEY);
      if (saved === "dark" || saved === "light") return saved;
    } catch (_err) {
      /* ignore */
    }
    return "light";
  }

  function toggleTheme() {
    applyTheme(isDark() ? "light" : "dark", true);
  }

  window.CareerOSTheme = {
    get: function () { return isDark() ? "dark" : "light"; },
    set: function (mode) { applyTheme(mode, true); },
    toggle: toggleTheme,
    syncCharts: function () { syncChartTheme(isDark()); }
  };

  function bindAll() {
    applyTheme(initialTheme(), false);
    document.querySelectorAll("#theme-toggle, [data-theme-toggle]").forEach(function (btn) {
      if (btn.dataset.themeBound === "1") return;
      btn.dataset.themeBound = "1";
      btn.addEventListener("click", toggleTheme);
    });
    // Late pass after page inline chart scripts finish mounting
    setTimeout(function () { syncChartTheme(isDark()); }, 80);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindAll);
  } else {
    bindAll();
  }

  window.addEventListener("storage", function (e) {
    if (e.key === STORAGE_KEY && (e.newValue === "dark" || e.newValue === "light")) {
      applyTheme(e.newValue, false);
    }
  });
})();
