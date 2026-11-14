(function () {
  "use strict";

  var state = {
    runs: [],
    activeRunId: null,
    activeStudentId: null,
    streaming: false
  };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = String(text);
    return node;
  }

  function currentRole() {
    var bodyRole = (document.body && document.body.dataset && document.body.dataset.role) || "";
    if (bodyRole) return bodyRole;
    var badge = document.getElementById("role-badge") || document.getElementById("nav-role-badge");
    if (badge && badge.textContent) {
      var low = badge.textContent.toLowerCase();
      if (low.indexOf("tpo") !== -1) return "tpo_admin";
      if (low.indexOf("faculty") !== -1 || low.indexOf("mentor") !== -1) return "faculty_mentor";
    }
    return "student";
  }

  function resolveStudentId() {
    if (state.activeStudentId) return state.activeStudentId;
    if (window.__CURRENT_STUDENT_ID__) return window.__CURRENT_STUDENT_ID__;
    if (document.body && document.body.dataset && document.body.dataset.studentId) {
      return document.body.dataset.studentId;
    }
    var sel = document.getElementById("stu-select");
    if (sel && sel.value) return sel.value;
    var m = (window.location.pathname || "").match(/\/students\/(stu-[A-Za-z0-9]+)/);
    return m ? m[1] : null;
  }

  function ensureStudentForDemoGoal() {
    var existing = resolveStudentId();
    if (existing) return Promise.resolve(existing);
    if (currentRole() === "student") return Promise.resolve(null);
    return fetch("/api/students")
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (list) {
        if (Array.isArray(list) && list.length && list[0].id) {
          state.activeStudentId = list[0].id;
          return list[0].id;
        }
        return null;
      })
      .catch(function () { return null; });
  }

  function updateHeaderTelemetry(run) {
    var provEl = document.getElementById("ab-stat-provider");
    var grdEl = document.getElementById("ab-stat-grounding");
    var latEl = document.getElementById("ab-stat-latency");
    var tokEl = document.getElementById("ab-stat-tokens");
    var wfEl = document.getElementById("drawer-workflow-badge");
    var auditEl = document.getElementById("drawer-audit-link");
    var role = currentRole();

    if (run) {
      var tel = run.telemetry || {};
      var cit = run.citations || {};
      if (provEl) provEl.textContent = String(tel.provider || run.provider_used || "--");
      if (grdEl) {
        var gs = cit.grounding_score;
        grdEl.textContent = gs != null ? (gs + "%") : "--";
      }
      if (latEl) latEl.textContent = run.total_latency_ms != null ? (run.total_latency_ms + " ms") : "--";
      if (tokEl) {
        if (tel.tokens != null) {
          var cost = Number(tel.cost_usd || 0);
          tokEl.textContent = tel.tokens + "t / $" + cost.toFixed(4);
        } else {
          tokEl.textContent = "tracked";
        }
      }
      if (wfEl) {
        wfEl.classList.remove("hidden");
        wfEl.textContent = String(run.workflow_kind || "orchestrator");
      }
      if (auditEl) {
        if (role === "tpo_admin" && run.audit_trace_url) {
          auditEl.classList.remove("hidden");
          auditEl.href = run.audit_trace_url;
        } else {
          auditEl.classList.add("hidden");
        }
      }
    }

    fetch("/api/agent/telemetry")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (t) {
        if (!t || !t.llm) return;
        if (!run && provEl) provEl.textContent = String(t.llm.provider || "mock");
        if (!run && grdEl) grdEl.textContent = (t.avg_grounding_score != null ? t.avg_grounding_score + "%" : "--");
        if (!run && latEl) latEl.textContent = (t.avg_latency_ms != null ? Math.round(t.avg_latency_ms) + " ms" : "--");
        if (tokEl && (!run || !run.telemetry || run.telemetry.tokens == null)) {
          tokEl.textContent = (t.llm.tokens || 0) + "t / $" + Number(t.llm.cost_usd || 0).toFixed(4);
        }
      })
      .catch(function () {});
  }

  function renderGroundingBox(run) {
    var box = document.getElementById("drawer-grounding-box");
    var pill = document.getElementById("drawer-grounding-pill");
    var fbPill = document.getElementById("drawer-fallback-pill");
    var verWrap = document.getElementById("drawer-verified-pills");
    var unverWrap = document.getElementById("drawer-unverified-pills");
    if (!box || !run) {
      if (box) box.classList.add("hidden");
      return;
    }
    var cit = run.citations || {};
    var gs = cit.grounding_score;
    var verified = Array.isArray(cit.verified) ? cit.verified : [];
    var unverified = Array.isArray(cit.unverified) ? cit.unverified : [];

    box.classList.remove("hidden");
    if (pill) {
      pill.textContent = gs != null ? (gs + "% Grounded") : "N/A";
      pill.className = "px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold " +
        (unverified.length > 0
          ? "bg-[#FDE8E8] text-[#9E2A20]"
          : "bg-[#E3F2EA] text-[#164934]");
    }
    if (fbPill) {
      if (run.fallback_triggered) fbPill.classList.remove("hidden");
      else fbPill.classList.add("hidden");
    }
    if (verWrap) {
      verWrap.innerHTML = "";
      if (verified.length) {
        verified.forEach(function (v) {
          verWrap.appendChild(el("span", "px-2 py-0.5 rounded-full bg-[#EBF4EF] text-[#164934] text-[10px] font-mono font-semibold", "VERIFIED: " + v));
        });
      } else {
        verWrap.appendChild(el("span", "text-[11px] text-[#767F79]", "No external entity references required verification."));
      }
    }
    if (unverWrap) {
      unverWrap.innerHTML = "";
      if (unverified.length) {
        unverWrap.classList.remove("hidden");
        unverified.forEach(function (u) {
          unverWrap.appendChild(el("span", "px-2 py-0.5 rounded-full bg-[#FDE8E8] border border-[#F5B8B8] text-[#9E2A20] text-[10px] font-mono font-bold", "UNVERIFIED: " + u));
        });
      } else {
        unverWrap.classList.add("hidden");
      }
    }
  }

  function renderClarifyBox(run) {
    var box = document.getElementById("drawer-clarify-box");
    var qEl = document.getElementById("drawer-clarify-question");
    if (!box) return;
    if (run && run.status === "awaiting_input") {
      box.classList.remove("hidden");
      if (qEl) qEl.textContent = run.final_answer || "Please clarify your request to continue.";
    } else {
      box.classList.add("hidden");
    }
  }

  function appendAnnotatedAnswer(container, answerText) {
    var text = String(answerText || "");
    var parts = text.split(/(\[UNVERIFIED:[^\]]+\])/g);
    parts.forEach(function (part) {
      if (!part) return;
      if (part.indexOf("[UNVERIFIED:") === 0) {
        container.appendChild(el("span", "inline-block px-1.5 py-0.5 rounded bg-[#FDE8E8] text-[#9E2A20] font-mono text-[10px] font-bold mx-0.5", part));
      } else {
        container.appendChild(document.createTextNode(part));
      }
    });
  }

  function buildStepCard(step) {
    var card = el("div", "rounded-xl border border-[#E5E3DA] bg-white p-3 space-y-1.5");
    var head = el("div", "flex items-center justify-between gap-2");
    var left = el("div", "flex items-center gap-1.5 min-w-0");
    left.appendChild(el("span", "px-1.5 py-0.5 rounded bg-[#EFECE4] text-[#4A4943] text-[10px] font-mono font-bold shrink-0", "Step " + (step.step_order || "?")));
    left.appendChild(el("span", "text-[11px] font-mono font-semibold text-[#1A211D] truncate", step.tool_name || "tool"));
    head.appendChild(left);

    var badges = el("div", "flex items-center gap-1 shrink-0");
    if (step.error_recovered) {
      badges.appendChild(el("span", "px-1.5 py-0.5 rounded-full bg-[#FEF3C7] text-[#92400E] text-[9px] font-mono font-bold", "RECOVERED"));
    }
    var ok = !!step.verification_passed;
    badges.appendChild(el(
      "span",
      "px-1.5 py-0.5 rounded-full text-[9px] font-mono font-bold " +
      (ok ? "bg-[#E3F2EA] text-[#164934]" : "bg-[#FDE8E8] text-[#9E2A20]"),
      ok ? "VERIFIED" : "FAILED"
    ));
    if (step.latency_ms != null) {
      badges.appendChild(el("span", "text-[10px] font-mono text-[#767F79]", step.latency_ms + "ms"));
    }
    head.appendChild(badges);
    card.appendChild(head);

    if (step.thought) {
      card.appendChild(el("p", "text-xs text-[#3B423E] leading-relaxed", step.thought));
    }
    if (step.error_recovered) {
      card.appendChild(el("p", "text-[11px] font-mono text-[#92400E] bg-[#FFFBEB] rounded-lg px-2 py-1 border border-[#F2C97D]", "Recovery: " + step.error_recovered));
    }

    var details = el("details", "pt-1");
    var sum = el("summary", "text-[10px] font-mono text-[#57605A] cursor-pointer select-none", "Inspect Tool I/O");
    details.appendChild(sum);
    var ioPre = el("pre", "mt-1.5 p-2 rounded-lg bg-[#F9F8F4] border border-[#E5E3DA] text-[10px] font-mono text-[#1A211D] overflow-x-auto max-h-40");
    try {
      ioPre.textContent = JSON.stringify({ input: step.tool_input || {}, output: step.tool_output || {} }, null, 2);
    } catch (e) {
      ioPre.textContent = "[unserializable I/O]";
    }
    details.appendChild(ioPre);
    card.appendChild(details);
    return card;
  }

  function renderRun(run) {
    state.activeRunId = run ? run.id : null;
    var trace = document.getElementById("drawer-trace");
    updateHeaderTelemetry(run);
    renderGroundingBox(run);
    renderClarifyBox(run);
    if (!trace) return;
    trace.innerHTML = "";

    if (!run) {
      trace.appendChild(el("p", "text-xs text-[#767F79] italic", "No agent runs recorded yet. Choose a preset goal above or click Run Agent."));
      return;
    }

    var meta = el("div", "flex flex-wrap items-center justify-between gap-2 pb-1 border-b border-[#E5E3DA]");
    meta.appendChild(el("span", "text-[11px] font-semibold text-[#1A211D] truncate", run.goal || run.id));
    var statusColor = run.status === "failed"
      ? "bg-[#FDE8E8] text-[#9E2A20]"
      : (run.status === "awaiting_approval" || run.status === "awaiting_input")
        ? "bg-[#FEF3C7] text-[#92400E]"
        : "bg-[#E3F2EA] text-[#164934]";
    meta.appendChild(el("span", "px-2 py-0.5 rounded-full text-[10px] font-mono font-bold " + statusColor, String(run.status || "unknown")));
    trace.appendChild(meta);

    var steps = Array.isArray(run.steps) ? run.steps : [];
    if (steps.length) {
      steps.forEach(function (s) { trace.appendChild(buildStepCard(s)); });
    } else if (Array.isArray(run.plan) && run.plan.length) {
      run.plan.forEach(function (p, i) {
        trace.appendChild(buildStepCard({
          step_order: i + 1,
          tool_name: p.tool || "plan_step",
          thought: p.thought || "",
          tool_input: p.args || {},
          tool_output: {},
          verification_passed: true
        }));
      });
    }

    if (run.final_answer) {
      var ansBox = el("div", "rounded-2xl border border-[#DCE4DE] bg-[#F6F8F6] p-3 space-y-1");
      ansBox.appendChild(el("p", "text-[10px] font-mono uppercase tracking-wider text-[#1D5C42] font-bold", "Verified Synthesis"));
      var body = el("div", "text-xs text-[#1A211D] whitespace-pre-wrap leading-relaxed");
      appendAnnotatedAnswer(body, run.final_answer);
      ansBox.appendChild(body);
      trace.appendChild(ansBox);
    }
  }

  function populateRunSelect(runs) {
    var sel = document.getElementById("drawer-run-select");
    if (!sel) return;
    sel.innerHTML = "";
    if (!runs || !runs.length) {
      sel.appendChild(el("option", "", "No runs yet"));
      return;
    }
    runs.forEach(function (r) {
      var label = "[" + (r.workflow_kind || "agent") + "] " + String(r.goal || r.id).slice(0, 52) + " (" + r.status + ")";
      var opt = el("option", "", label);
      opt.value = r.id;
      if (state.activeRunId && state.activeRunId === r.id) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  function loadRuns(selectRunId) {
    return fetch("/api/agent/runs?limit=30")
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (runs) {
        state.runs = Array.isArray(runs) ? runs : [];
        if (selectRunId) state.activeRunId = selectRunId;
        else if (!state.activeRunId && state.runs.length) state.activeRunId = state.runs[0].id;
        populateRunSelect(state.runs);
        var chosen = state.runs.find(function (r) { return r.id === state.activeRunId; }) || state.runs[0] || null;
        renderRun(chosen);
        return state.runs;
      })
      .catch(function () {
        updateHeaderTelemetry(null);
        return [];
      });
  }

  function runPipelineStreaming(url, body, handlers) {
    handlers = handlers || {};
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (resp) {
      if (!resp.ok || !resp.body) {
        throw new Error("HTTP " + resp.status);
      }
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      function processChunk(block) {
        var evName = "message";
        var dataLines = [];
        block.split(/\r?\n/).forEach(function (line) {
          if (line.indexOf("event:") === 0) evName = line.slice(6).trim();
          else if (line.indexOf("data:") === 0) dataLines.push(line.slice(5).trim());
        });
        if (!dataLines.length) return;
        var parsed = {};
        try { parsed = JSON.parse(dataLines.join("\n")); } catch (e) { parsed = { raw: dataLines.join("\n") }; }
        if (handlers.onEvent) handlers.onEvent(evName, parsed);
      }

      function pump() {
        return reader.read().then(function (res) {
          if (res.done) {
            if (buffer.trim()) processChunk(buffer);
            if (handlers.onDone) handlers.onDone();
            return;
          }
          buffer += decoder.decode(res.value, { stream: true });
          var parts = buffer.split(/\r?\n\r?\n/);
          buffer = parts.pop() || "";
          parts.forEach(processChunk);
          return pump();
        });
      }
      return pump();
    });
  }

  function executeGoal(goalOverride, opts) {
    opts = opts || {};
    var goalInput = document.getElementById("drawer-goal");
    var goal = String(goalOverride != null ? goalOverride : (goalInput ? goalInput.value : "")).trim();
    if (!goal) return Promise.resolve(null);
    if (goalInput && goalOverride != null) goalInput.value = goal;

    var trace = document.getElementById("drawer-trace");
    var runBtn = document.getElementById("drawer-run");
    if (runBtn) runBtn.disabled = true;
    state.streaming = true;
    if (trace) {
      trace.innerHTML = "";
      trace.appendChild(el("p", "text-xs font-mono text-[#1D5C42]", "Planning multi-step execution..."));
    }

    var role = currentRole();
    var needsStudent = !opts.cohortOnly;
    var sidPromise = needsStudent ? ensureStudentForDemoGoal() : Promise.resolve(resolveStudentId());

    return sidPromise.then(function (sid) {
      var liveSteps = [];
      var liveRun = {
        id: null,
        goal: goal,
        actor_role: role,
        status: "running",
        workflow_kind: "agent_orchestrator",
        citations: {},
        steps: liveSteps
      };

      return runPipelineStreaming(
        "/api/agent/run/stream",
        {
          goal: goal,
          student_id: sid,
          actor_role: role,
          simulate_failure: !!opts.simulateFailure
        },
        {
          onEvent: function (ev, data) {
            if (ev === "run_started") {
              liveRun.id = data.run_id;
              renderRun(liveRun);
            } else if (ev === "plan_created") {
              liveRun.plan = data.plan || [];
              renderRun(liveRun);
            } else if (ev === "step_started") {
              liveSteps.push({
                step_order: data.step_order,
                tool_name: data.tool_name,
                thought: data.thought,
                tool_input: data.tool_input,
                tool_output: {},
                verification_passed: false
              });
              renderRun(liveRun);
            } else if (ev === "step_completed") {
              var idx = liveSteps.findIndex(function (s) { return s.step_order === data.step_order; });
              if (idx >= 0) liveSteps[idx] = data;
              else liveSteps.push(data);
              if (data.error_recovered) liveRun.fallback_triggered = true;
              renderRun(liveRun);
            } else if (ev === "verification_done") {
              liveRun.citations = data.citations || {};
              renderRun(liveRun);
            } else if (ev === "awaiting_input") {
              liveRun.status = "awaiting_input";
              liveRun.final_answer = data.question || "";
              renderRun(liveRun);
              loadRuns(data.run_id);
            } else if (ev === "complete") {
              liveRun = data;
              renderRun(data);
              loadRuns(data.id);
            } else if (ev === "error") {
              if (trace) trace.appendChild(el("p", "text-xs text-[#9E2A20] font-mono", "Error: " + (data.message || "unknown")));
            }
          }
        }
      );
    }).catch(function (err) {
      if (trace) trace.appendChild(el("p", "text-xs text-[#9E2A20] font-mono", "Agent run failed: " + err.message));
    }).finally(function () {
      state.streaming = false;
      if (runBtn) runBtn.disabled = false;
    });
  }

  function renderQuickGoals() {
    var wrap = document.getElementById("drawer-quick-goals");
    if (!wrap) return;
    wrap.innerHTML = "";
    var role = currentRole();
    var presets = role === "student"
      ? [
          { label: "Rank Jobs & Gaps", goal: "Rank my top matches, check SOP-04 readiness, and plan missing skills", cohortOnly: false },
          { label: "Check Drive Eligibility", goal: "Check my SOP eligibility for Dream drives and explain blockers", cohortOnly: false },
          { label: "Audit Resume Bullets", goal: "Audit my resume bullets and ATS readiness", cohortOnly: false }
        ]
      : [
          { label: "Cohort Bottlenecks", goal: "Analyze cohort bottlenecks and propose an intervention workshop", cohortOnly: true },
          { label: "Rank Student & Plan", goal: "Rank jobs and build an upskilling plan for this student", cohortOnly: false },
          { label: "Simulate Error -> Fallback Recovery", goal: "Rank jobs and explain SOP eligibility", simulateFailure: true, cohortOnly: false }
        ];
    presets.forEach(function (p) {
      var btn = el(
        "button",
        "px-2.5 py-1 rounded-full text-[10px] font-semibold border cursor-pointer transition-colors " +
        (p.simulateFailure
          ? "bg-[#FFFBEB] border-[#F2C97D] text-[#92400E] hover:bg-[#FEF3C7]"
          : "bg-white border-[#E0DCD0] text-[#1A211D] hover:bg-[#F5F3EC]"),
        p.label
      );
      btn.type = "button";
      btn.addEventListener("click", function () {
        executeGoal(p.goal, { simulateFailure: !!p.simulateFailure, cohortOnly: !!p.cohortOnly });
      });
      wrap.appendChild(btn);
    });
  }

  function replyClarification() {
    var input = document.getElementById("drawer-clarify-input");
    var btn = document.getElementById("drawer-clarify-send");
    var reply = input ? input.value.trim() : "";
    if (!reply || !state.activeRunId) return;
    if (btn) btn.disabled = true;
    fetch("/api/agent/runs/" + encodeURIComponent(state.activeRunId) + "/reply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reply: reply })
    })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (updated) {
        if (input) input.value = "";
        if (updated) {
          renderRun(updated);
          loadRuns(updated.id);
        }
      })
      .finally(function () { if (btn) btn.disabled = false; });
  }

  function bindUI() {
    renderQuickGoals();
    var runBtn = document.getElementById("drawer-run");
    var goalInput = document.getElementById("drawer-goal");
    var refreshBtn = document.getElementById("drawer-refresh");
    var runSel = document.getElementById("drawer-run-select");
    var clarifySend = document.getElementById("drawer-clarify-send");
    var openBtn = document.getElementById("open-drawer");

    if (runBtn && !runBtn.dataset.abBound) {
      runBtn.dataset.abBound = "1";
      runBtn.addEventListener("click", function () { executeGoal(null); });
    }
    if (goalInput && !goalInput.dataset.abBound) {
      goalInput.dataset.abBound = "1";
      goalInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          executeGoal(null);
        }
      });
    }
    if (refreshBtn && !refreshBtn.dataset.abBound) {
      refreshBtn.dataset.abBound = "1";
      refreshBtn.addEventListener("click", function () { loadRuns(state.activeRunId); });
    }
    if (runSel && !runSel.dataset.abBound) {
      runSel.dataset.abBound = "1";
      runSel.addEventListener("change", function () {
        var id = runSel.value;
        var chosen = state.runs.find(function (r) { return r.id === id; });
        if (chosen) renderRun(chosen);
      });
    }
    if (clarifySend && !clarifySend.dataset.abBound) {
      clarifySend.dataset.abBound = "1";
      clarifySend.addEventListener("click", replyClarification);
    }
    if (openBtn && !openBtn.dataset.abBound) {
      openBtn.dataset.abBound = "1";
      openBtn.addEventListener("click", function () {
        renderQuickGoals();
        loadRuns(null);
      });
    }
  }

  window.AgentBrain = {
    load: loadRuns,
    refresh: function () { return loadRuns(state.activeRunId); },
    run: executeGoal,
    setStudentId: function (sid) { state.activeStudentId = sid || null; },
    replyClarification: replyClarification
  };
  if (typeof window.loadAgentBrain !== "function") {
    window.loadAgentBrain = function () { return loadRuns(null); };
  }
  if (typeof window.runPipelineStreaming !== "function") {
    window.runPipelineStreaming = runPipelineStreaming;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindUI);
  } else {
    bindUI();
  }
})();
