(function () {
  "use strict";

  var FALLBACK_TAXONOMY = [
    { id: "python", name: "Python", category: "languages", aliases: ["python", "python3"] },
    { id: "javascript", name: "JavaScript", category: "languages", aliases: ["javascript", "js"] },
    { id: "typescript", name: "TypeScript", category: "languages", aliases: ["typescript", "ts"] },
    { id: "java", name: "Java", category: "languages", aliases: ["java"] },
    { id: "cplusplus", name: "C++", category: "languages", aliases: ["c++", "cpp"] },
    { id: "sql", name: "SQL", category: "languages", aliases: ["sql", "pl/sql"] },
    { id: "react", name: "React", category: "web_frontend", aliases: ["react", "reactjs"] },
    { id: "nextjs", name: "Next.js", category: "web_frontend", aliases: ["next.js", "nextjs"] },
    { id: "fastapi", name: "FastAPI", category: "web_backend", aliases: ["fastapi"] },
    { id: "django", name: "Django", category: "web_backend", aliases: ["django"] },
    { id: "nodejs", name: "Node.js", category: "web_backend", aliases: ["node.js", "nodejs"] },
    { id: "postgresql", name: "PostgreSQL", category: "databases", aliases: ["postgres", "postgresql"] },
    { id: "mongodb", name: "MongoDB", category: "databases", aliases: ["mongodb", "mongo"] },
    { id: "redis", name: "Redis", category: "databases", aliases: ["redis"] },
    { id: "docker", name: "Docker", category: "devops_cloud", aliases: ["docker"] },
    { id: "kubernetes", name: "Kubernetes", category: "devops_cloud", aliases: ["kubernetes", "k8s"] },
    { id: "aws", name: "AWS", category: "devops_cloud", aliases: ["aws", "amazon web services"] },
    { id: "git", name: "Git", category: "tools", aliases: ["git"] },
    { id: "machine_learning", name: "Machine Learning", category: "data_ml", aliases: ["machine learning", "ml"] },
    { id: "llms", name: "LLMs & Generative AI", category: "data_ml", aliases: ["llm", "genai", "rag"] }
  ];

  var SAMPLE_JD = [
    "Company: Razorpay Technologies",
    "Role: Software Development Engineer (Backend)",
    "Location: Bengaluru (Hybrid)",
    "CTC: INR 14 - 18 LPA",
    "Batch: 2027 B.Tech graduating batch (CSE, IT, ECE, AI&DS)",
    "Eligibility: Minimum CGPA >= 7.5, zero active backlogs.",
    "Must have: Python, FastAPI, PostgreSQL, Redis, Docker, REST APIs",
    "Preferred / Good to have: Kubernetes, AWS, Kafka, Blockchain-Quantum-Rust++"
  ].join("\n");

  var state = {
    mode: "create",           // "create" | "edit"
    editingDriveId: null,
    activeTab: "upload",      // "upload" | "paste"
    parserMode: "auto",       // "auto" | "static" | "llm"
    selectedFile: null,
    taxonomy: null,
    taxonomyMap: {},
    requiredSkills: [],
    preferredSkills: [],
    unmappedSkills: [],
    allowedBranches: [],
    lastParsedSchema: null,
    onSavedCallback: null
  };

  function $(id) {
    return document.getElementById(id);
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function notify(msg, kind) {
    if (typeof window.toast === "function") {
      window.toast(msg, kind || "info");
    }
  }

  function computeSop01Tier(ctcMax, expLevel, salaryDisclosed) {
    var exp = (expLevel || "Fresher").trim();
    if (exp === "Internship" || !salaryDisclosed) {
      return {
        tier: null,
        source: null,
        label: "Not classified (SOP-01 applies to disclosed CTC offers)",
        cls: "bg-[#EFECE4] text-[#4A483F] border-[#E2DEC9]"
      };
    }
    var val = Number(ctcMax);
    if (!ctcMax || isNaN(val) || val <= 0) {
      return {
        tier: null,
        source: null,
        label: "Not classified (SOP-01 applies to disclosed CTC offers)",
        cls: "bg-[#EFECE4] text-[#4A483F] border-[#E2DEC9]"
      };
    }
    if (val >= 15.0) {
      return {
        tier: "super_dream",
        source: "SOP-01",
        label: "Super Dream (SOP-01: >= 15.0 LPA)",
        cls: "bg-[#1D5C42] text-white border-[#154531]"
      };
    }
    if (val >= 8.0) {
      return {
        tier: "dream",
        source: "SOP-01",
        label: "Dream (SOP-01: 8.0 - 14.99 LPA)",
        cls: "bg-[#E3F2EA] text-[#1B5E43] border-[#C8E4D5]"
      };
    }
    return {
      tier: "standard",
      source: "SOP-01",
      label: "Standard (SOP-01: < 8.0 LPA)",
      cls: "bg-[#EEF5FB] text-[#16436E] border-[#C8D9E8]"
    };
  }

  function updateLiveTierBadge() {
    var ctcMaxEl = $("cd-ctc-max");
    var ctcMinEl = $("cd-ctc-min");
    var expEl = $("cd-experience-level");
    var discEl = $("cd-salary-disclosed");
    var discLbl = $("cd-salary-disclosed-label");
    var badge = $("cd-policy-tier-badge");
    if (!badge) return;

    var disclosed = discEl ? discEl.checked : true;
    if (discLbl) {
      discLbl.textContent = disclosed ? "Salary Disclosed" : "Undisclosed";
    }
    if (ctcMinEl) {
      ctcMinEl.disabled = !disclosed;
      ctcMinEl.classList.toggle("opacity-50", !disclosed);
    }
    if (ctcMaxEl) {
      ctcMaxEl.disabled = !disclosed;
      ctcMaxEl.classList.toggle("opacity-50", !disclosed);
    }

    var ctcMax = ctcMaxEl && ctcMaxEl.value !== "" ? parseFloat(ctcMaxEl.value) : null;
    var exp = expEl ? expEl.value : "Fresher";
    var info = computeSop01Tier(ctcMax, exp, disclosed);
    badge.className = "inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border " + info.cls;
    badge.textContent = info.label;
    return info;
  }

  function loadTaxonomy() {
    if (state.taxonomy) {
      return Promise.resolve(state.taxonomy);
    }
    return fetch("/api/campus-drives/taxonomy")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        var list = Array.isArray(data) ? data : (data && Array.isArray(data.skills) ? data.skills : FALLBACK_TAXONOMY);
        state.taxonomy = list;
        state.taxonomyMap = {};
        list.forEach(function (s) { state.taxonomyMap[s.id] = s; });
        return list;
      })
      .catch(function () {
        state.taxonomy = FALLBACK_TAXONOMY;
        state.taxonomyMap = {};
        FALLBACK_TAXONOMY.forEach(function (s) { state.taxonomyMap[s.id] = s; });
        return FALLBACK_TAXONOMY;
      });
  }

  function skillLabel(skillId) {
    var rec = state.taxonomyMap[skillId];
    return rec ? rec.name : skillId;
  }

  function renderSkillChips() {
    var reqBox = $("cd-required-skills");
    var prefBox = $("cd-preferred-skills");
    var unmWrap = $("cd-unmapped-wrap");
    var unmBox = $("cd-unmapped-skills");

    if (reqBox) {
      reqBox.innerHTML = "";
      state.requiredSkills.forEach(function (sid) {
        var chip = document.createElement("span");
        chip.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[#E3F2EA] text-[#1B5E43] border border-[#C8E4D5]";
        chip.innerHTML = '<span>' + esc(skillLabel(sid)) + '</span>' +
          '<span class="text-[10px] font-mono opacity-60">' + esc(sid) + '</span>' +
          '<button type="button" aria-label="Remove ' + esc(sid) + '" class="ml-0.5 text-[#1B5E43] hover:text-[#9E2A20] font-bold cursor-pointer">&times;</button>';
        chip.querySelector("button").addEventListener("click", function () {
          state.requiredSkills = state.requiredSkills.filter(function (x) { return x !== sid; });
          renderSkillChips();
        });
        reqBox.appendChild(chip);
      });
    }

    if (prefBox) {
      prefBox.innerHTML = "";
      state.preferredSkills.forEach(function (sid) {
        var chip = document.createElement("span");
        chip.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[#EFECE4] text-[#4A483F] border border-[#D5D2C6]";
        chip.innerHTML = '<span>' + esc(skillLabel(sid)) + '</span>' +
          '<span class="text-[10px] font-mono opacity-60">' + esc(sid) + '</span>' +
          '<button type="button" aria-label="Remove ' + esc(sid) + '" class="ml-0.5 text-[#4A483F] hover:text-[#9E2A20] font-bold cursor-pointer">&times;</button>';
        chip.querySelector("button").addEventListener("click", function () {
          state.preferredSkills = state.preferredSkills.filter(function (x) { return x !== sid; });
          renderSkillChips();
        });
        prefBox.appendChild(chip);
      });
    }

    if (unmWrap && unmBox) {
      if (state.unmappedSkills.length) {
        unmWrap.classList.remove("hidden");
        unmBox.innerHTML = "";
        state.unmappedSkills.forEach(function (raw) {
          var chip = document.createElement("span");
          chip.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-[#F4F3EC] text-[#7A776E] border border-dashed border-[#C7C4B8]";
          chip.textContent = raw;
          unmBox.appendChild(chip);
        });
      } else {
        unmWrap.classList.add("hidden");
        unmBox.innerHTML = "";
      }
    }
  }

  function wireSkillAutocomplete(inputId, suggestId, targetListName) {
    var inp = $(inputId);
    var box = $(suggestId);
    if (!inp || !box) return;

    function updateDropdown() {
      var q = (inp.value || "").trim().toLowerCase();
      if (!q) {
        box.classList.add("hidden");
        box.innerHTML = "";
        return;
      }
      loadTaxonomy().then(function (list) {
        var matches = list.filter(function (s) {
          if ((s.id || "").toLowerCase().indexOf(q) >= 0) return true;
          if ((s.name || "").toLowerCase().indexOf(q) >= 0) return true;
          return (s.aliases || []).some(function (a) { return String(a).toLowerCase().indexOf(q) >= 0; });
        }).slice(0, 8);

        if (!matches.length) {
          box.classList.add("hidden");
          box.innerHTML = "";
          return;
        }
        box.innerHTML = "";
        matches.forEach(function (s) {
          var item = document.createElement("button");
          item.type = "button";
          item.className = "w-full text-left px-3 py-2 hover:bg-[#E3F2EA] flex items-center justify-between gap-2 border-b border-[#F4F3EC] last:border-b-0 cursor-pointer";
          item.innerHTML = '<span class="font-medium text-[#1C1E1B]">' + esc(s.name) + '</span>' +
            '<span class="font-mono text-[10px] text-[#7A776E]">' + esc(s.id) + '</span>';
          item.addEventListener("click", function () {
            var arr = state[targetListName];
            if (arr.indexOf(s.id) === -1) {
              arr.push(s.id);
            }
            inp.value = "";
            box.classList.add("hidden");
            renderSkillChips();
          });
          box.appendChild(item);
        });
        box.classList.remove("hidden");
      });
    }

    inp.addEventListener("input", updateDropdown);
    inp.addEventListener("focus", updateDropdown);
    document.addEventListener("click", function (ev) {
      if (!box.contains(ev.target) && ev.target !== inp) {
        box.classList.add("hidden");
      }
    });
  }

  function renderBranchChips() {
    document.querySelectorAll("#cd-branches .cd-branch-chip").forEach(function (btn) {
      var br = btn.getAttribute("data-branch");
      var active = state.allowedBranches.indexOf(br) >= 0;
      if (active) {
        btn.className = "cd-branch-chip px-3 py-1 rounded-full text-xs font-semibold border border-[#1D5C42] bg-[#1D5C42] text-white transition-colors cursor-pointer";
      } else {
        btn.className = "cd-branch-chip px-3 py-1 rounded-full text-xs font-semibold border border-[#D5D2C6] bg-white text-[#4A483F] hover:border-[#1D5C42] transition-colors cursor-pointer";
      }
    });
  }

  function formatFallbackReason(reason) {
    if (!reason) return "";
    var map = {
      offline_mode: "Offline / Mock LLM mode active: deterministic static parser executed cleanly.",
      forced_static: "Static parser explicitly selected by operator.",
      llm_timeout: "LLM exceeded the 6s timeout budget; automatic recovery to deterministic static parser succeeded.",
      llm_invalid_json: "LLM returned malformed JSON; automatic recovery to deterministic static parser succeeded.",
      llm_schema_validation: "LLM output failed schema validation; automatic recovery to deterministic static parser succeeded.",
      llm_grounding_failed: "LLM output failed grounding check; automatic recovery to deterministic static parser succeeded.",
      circuit_open: "LLM circuit breaker open; deterministic static parser executed without waiting.",
      llm_error: "LLM provider error; automatic recovery to deterministic static parser succeeded.",
      client_fixture_404: "Backend parser endpoint unavailable (404); populated using built-in CampusDriveSchema fixture."
    };
    return map[reason] || ("Fallback triggered: " + reason);
  }

  function renderEvidenceTooltips(meta) {
    var evidence = (meta && meta.evidence) || {};
    var fieldConf = (meta && meta.field_confidence) || {};

    document.querySelectorAll("[data-evidence-field]").forEach(function (wrap) {
      var field = wrap.getAttribute("data-evidence-field");
      var snippet = evidence[field];
      var conf = fieldConf[field];
      var cap = wrap.querySelector("[data-evidence-caption='" + field + "']");

      var tipParts = [];
      if (snippet) tipParts.push("Evidence: " + snippet);
      if (conf != null) tipParts.push("Confidence: " + Math.round(Number(conf) * 100) + "%");
      var tip = tipParts.join(" | ");

      wrap.title = tip;
      var ctrl = wrap.querySelector("input, select");
      if (ctrl) ctrl.title = tip;

      if (cap) {
        if (snippet || conf != null) {
          var label = snippet ? ('"' + snippet + '"') : "Inferred";
          if (conf != null) label += " (" + Math.round(Number(conf) * 100) + "%)";
          cap.textContent = label;
          cap.title = tip;
          cap.classList.remove("hidden");
        } else {
          cap.textContent = "";
          cap.classList.add("hidden");
        }
      }
    });
  }

  function renderExplainabilityBanner(schema) {
    var banner = $("jd-explainability-banner");
    if (!banner) return;
    var meta = (schema && schema.parser_metadata) || null;
    if (!meta) {
      banner.classList.add("hidden");
      renderEvidenceTooltips(null);
      return;
    }

    banner.classList.remove("hidden");

    var engineBadge = $("jd-engine-badge");
    if (engineBadge) {
      var engMap = {
        llm_structured: "LLM Structured",
        static_fallback: "Static Deterministic",
        hybrid_merge: "Hybrid Merge"
      };
      var eng = meta.engine_used || "static_fallback";
      engineBadge.textContent = engMap[eng] || eng;
      if (eng === "llm_structured") {
        engineBadge.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#1D5C42] text-white";
      } else if (eng === "hybrid_merge") {
        engineBadge.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#16436E] text-white";
      } else {
        engineBadge.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#4A483F] text-white";
      }
    }

    var traceBadge = $("jd-trace-badge");
    if (traceBadge) {
      traceBadge.textContent = schema.trace_id ? ("trace: " + schema.trace_id) : "trace: pending";
    }

    var pct = Math.max(0, Math.min(100, Math.round(Number(meta.confidence_score || 0) * 100)));
    var bar = $("jd-confidence-bar");
    var pctEl = $("jd-confidence-pct");
    if (bar) bar.style.width = pct + "%";
    if (pctEl) pctEl.textContent = pct + "%";

    var fbEl = $("jd-fallback-reason");
    if (fbEl) {
      if (meta.fallback_reason) {
        fbEl.textContent = formatFallbackReason(meta.fallback_reason);
        fbEl.classList.remove("hidden");
      } else {
        fbEl.classList.add("hidden");
        fbEl.textContent = "";
      }
    }

    var attList = $("jd-attempts-list");
    if (attList) {
      attList.innerHTML = "";
      var attempts = meta.attempts || [];
      if (!attempts.length) {
        attList.innerHTML = '<span class="text-[#7A776E]">No attempt trace recorded.</span>';
      } else {
        attempts.forEach(function (a) {
          var row = document.createElement("div");
          var ok = a.status === "ok";
          row.className = "flex items-center justify-between gap-2 px-2 py-1 rounded-lg " + (ok ? "bg-[#E3F2EA]/70 text-[#1B5E43]" : "bg-[#FBECE9] text-[#9E2A20]");
          row.innerHTML = '<span class="font-semibold uppercase">' + esc(a.engine) + ' (' + esc(a.status) + ')</span>' +
            '<span>' + Number(a.latency_ms || 0) + ' ms</span>';
          if (a.error) row.title = String(a.error);
          attList.appendChild(row);
        });
      }
    }

    var defList = $("jd-defaults-list");
    if (defList) {
      defList.innerHTML = "";
      var defs = meta.defaults_applied || [];
      if (!defs.length) {
        defList.innerHTML = '<span class="text-[#7A776E]">No defaults applied.</span>';
      } else {
        defs.forEach(function (d) {
          var row = document.createElement("div");
          row.className = "flex items-center justify-between gap-2 px-2 py-1 rounded-lg bg-[#FAF9F5] border border-[#ECEAE2]";
          row.innerHTML = '<span class="font-mono">' + esc(d.field) + ' = <strong>' + esc(d.value) + '</strong></span>' +
            '<span class="px-1.5 py-0.5 rounded bg-[#EFECE4] text-[#4A483F] font-mono text-[10px]">' + esc(d.source || "config") + '</span>';
          defList.appendChild(row);
        });
      }
    }

    var missList = $("jd-missing-list");
    if (missList) {
      missList.innerHTML = "";
      var missing = meta.missing_fields_flagged || [];
      if (!missing.length) {
        missList.innerHTML = '<span class="text-[11px] text-[#7A776E]">None flagged.</span>';
      } else {
        missing.forEach(function (m) {
          var isConflict = String(m).indexOf("conflict:") === 0;
          var chip = document.createElement("span");
          chip.className = isConflict
            ? "inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-[#FBECE9] text-[#9E2A20] border border-[#F3D2CC]"
            : "inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-[#FEF6E7] text-[#9A5B13] border border-[#F3D9A4]";
          chip.textContent = m;
          missList.appendChild(chip);
        });
      }
    }

    renderEvidenceTooltips(meta);
  }

  function populateFormFromSchema(schema) {
    state.lastParsedSchema = schema || {};
    var elig = (schema && schema.eligibility) || {};

    if ($("cd-company")) $("cd-company").value = schema.company || "";
    if ($("cd-role-title")) $("cd-role-title").value = schema.role_title || "";
    if ($("cd-location")) $("cd-location").value = schema.location || "";
    if ($("cd-work-mode")) $("cd-work-mode").value = schema.work_mode || "On-site";
    if ($("cd-experience-level")) $("cd-experience-level").value = schema.experience_level || "Fresher";
    if ($("cd-salary-disclosed")) $("cd-salary-disclosed").checked = schema.salary_disclosed !== false;
    if ($("cd-ctc-min")) $("cd-ctc-min").value = schema.ctc_lpa_min != null ? schema.ctc_lpa_min : "";
    if ($("cd-ctc-max")) $("cd-ctc-max").value = schema.ctc_lpa_max != null ? schema.ctc_lpa_max : "";
    if ($("cd-stipend")) $("cd-stipend").value = schema.stipend_inr_per_month != null ? schema.stipend_inr_per_month : "";

    if ($("cd-min-cgpa")) $("cd-min-cgpa").value = elig.min_cgpa != null ? elig.min_cgpa : "";
    if ($("cd-max-backlogs")) $("cd-max-backlogs").value = elig.max_active_backlogs != null ? elig.max_active_backlogs : "";
    if ($("cd-graduation-year")) $("cd-graduation-year").value = elig.graduation_year != null ? elig.graduation_year : "";

    state.allowedBranches = Array.isArray(elig.allowed_branches) ? elig.allowed_branches.slice() : [];
    state.requiredSkills = Array.isArray(schema.required_skills) ? schema.required_skills.slice() : [];
    state.preferredSkills = Array.isArray(schema.preferred_skills) ? schema.preferred_skills.slice() : [];
    state.unmappedSkills = Array.isArray(schema.unmapped_skill_mentions) ? schema.unmapped_skill_mentions.slice() : [];

    if ($("jd-raw-text") && schema.raw_jd_text) {
      $("jd-raw-text").value = schema.raw_jd_text;
    }

    renderBranchChips();
    loadTaxonomy().then(renderSkillChips);
    updateLiveTierBadge();
    renderExplainabilityBanner(schema);
  }

  function buildSchemaPayload() {
    var disclosed = $("cd-salary-disclosed") ? $("cd-salary-disclosed").checked : true;
    var ctcMinVal = $("cd-ctc-min") && $("cd-ctc-min").value !== "" ? parseFloat($("cd-ctc-min").value) : null;
    var ctcMaxVal = $("cd-ctc-max") && $("cd-ctc-max").value !== "" ? parseFloat($("cd-ctc-max").value) : null;
    var stipendVal = $("cd-stipend") && $("cd-stipend").value !== "" ? parseInt($("cd-stipend").value, 10) : null;
    var minCgpaVal = $("cd-min-cgpa") && $("cd-min-cgpa").value !== "" ? parseFloat($("cd-min-cgpa").value) : null;
    var maxBacklogsVal = $("cd-max-backlogs") && $("cd-max-backlogs").value !== "" ? parseInt($("cd-max-backlogs").value, 10) : null;
    var gradYearVal = $("cd-graduation-year") && $("cd-graduation-year").value !== "" ? parseInt($("cd-graduation-year").value, 10) : null;
    var expLevel = $("cd-experience-level") ? $("cd-experience-level").value : "Fresher";
    var tierInfo = computeSop01Tier(ctcMaxVal, expLevel, disclosed);
    var base = state.lastParsedSchema || {};
    var rawText = ($("jd-raw-text") && $("jd-raw-text").value.trim()) || base.raw_jd_text || "";

    return {
      id: base.id || undefined,
      trace_id: base.trace_id || undefined,
      company: ($("cd-company") ? $("cd-company").value : "").trim(),
      role_title: ($("cd-role-title") ? $("cd-role-title").value : "").trim(),
      location: ($("cd-location") ? $("cd-location").value : "").trim() || "Jaipur / Remote",
      work_mode: $("cd-work-mode") ? $("cd-work-mode").value : "On-site",
      ctc_lpa_min: disclosed ? ctcMinVal : null,
      ctc_lpa_max: disclosed ? ctcMaxVal : null,
      salary_disclosed: disclosed,
      stipend_inr_per_month: stipendVal,
      policy_tier: tierInfo.tier,
      policy_tier_source: tierInfo.source,
      experience_level: expLevel,
      source_portal: "On-Campus (Local)",
      source_badge: "On-Campus \u00b7 JKLU",
      origin: base.origin || "tpo_smart_jd",
      eligibility: {
        min_cgpa: minCgpaVal,
        allowed_branches: state.allowedBranches.slice(),
        max_active_backlogs: maxBacklogsVal,
        graduation_year: gradYearVal
      },
      required_skills: state.requiredSkills.slice(),
      preferred_skills: state.preferredSkills.slice(),
      unmapped_skill_mentions: state.unmappedSkills.slice(),
      raw_jd_text: rawText,
      parser_metadata: base.parser_metadata || undefined
    };
  }

  function clientFixtureFallback(rawText, mode, simFail) {
    var fallbackReason = simFail === "timeout"
      ? "llm_timeout"
      : (simFail === "invalid_json" ? "llm_invalid_json" : "client_fixture_404");
    var attempts = [];
    if (mode !== "static") {
      attempts.push({
        engine: "llm",
        status: "failed",
        latency_ms: simFail === "timeout" ? 6005 : 240,
        error: simFail === "timeout" ? "LLM request timed out (>6s)" : "404 / parse fallback"
      });
    }
    attempts.push({ engine: "static", status: "ok", latency_ms: 14, error: null });

    return {
      id: "cd-demo0001",
      trace_id: "tr-demo0001",
      company: "Razorpay Technologies",
      role_title: "Software Development Engineer (Backend)",
      location: "Bengaluru",
      work_mode: "Hybrid",
      ctc_lpa_min: 14.0,
      ctc_lpa_max: 18.0,
      salary_disclosed: true,
      stipend_inr_per_month: null,
      policy_tier: "super_dream",
      policy_tier_source: "SOP-01",
      experience_level: "Fresher",
      source_portal: "On-Campus (Local)",
      source_badge: "On-Campus \u00b7 JKLU",
      origin: "tpo_smart_jd",
      eligibility: {
        min_cgpa: 7.5,
        allowed_branches: ["CSE", "IT", "ECE", "AI&DS"],
        max_active_backlogs: 0,
        graduation_year: 2027
      },
      required_skills: ["python", "fastapi", "postgresql", "redis", "docker", "rest_api"],
      preferred_skills: ["kubernetes", "aws", "kafka"],
      unmapped_skill_mentions: ["Blockchain-Quantum-Rust++"],
      raw_jd_text: rawText || SAMPLE_JD,
      parser_metadata: {
        engine_used: "static_fallback",
        fallback_reason: fallbackReason,
        attempts: attempts,
        confidence_score: 0.86,
        field_confidence: {
          company: 0.9,
          role_title: 0.9,
          ctc_lpa_max: 0.9,
          min_cgpa: 0.9,
          max_active_backlogs: 0.3,
          graduation_year: 0.9
        },
        evidence: {
          company: "Company: Razorpay Technologies",
          role_title: "Role: Software Development Engineer (Backend)",
          ctc_lpa_max: "CTC: INR 14 - 18 LPA",
          min_cgpa: "Minimum CGPA >= 7.5",
          max_active_backlogs: "zero active backlogs",
          graduation_year: "2027 B.Tech graduating batch"
        },
        missing_fields_flagged: [],
        defaults_applied: [
          { field: "max_active_backlogs", value: 0, source: "SOP-03" }
        ]
      }
    };
  }

  function runParseJd() {
    var btn = $("jd-parse-btn");
    var rawTextEl = $("jd-raw-text");
    var rawText = rawTextEl ? rawTextEl.value.trim() : "";
    var simEl = $("jd-simulate-failure");
    var simFail = simEl ? simEl.value : "none";

    if (state.activeTab === "upload" && !state.selectedFile && !rawText) {
      notify("Please select a JD file (.pdf/.docx/.txt) or paste JD text first.", "error");
      return;
    }
    if (state.activeTab === "paste" && !rawText) {
      notify("Please paste a JD or email text before parsing.", "error");
      return;
    }

    var fd = new FormData();
    if (state.activeTab === "upload" && state.selectedFile) {
      fd.append("file", state.selectedFile);
    } else {
      fd.append("text", rawText);
    }
    fd.append("force_parser", state.parserMode);
    fd.append("simulate_failure", simFail);

    if (btn) {
      btn.disabled = true;
      btn.classList.add("opacity-60", "cursor-wait");
    }

    fetch("/api/campus-drives/parse-jd", {
      method: "POST",
      body: fd
    })
      .then(function (r) {
        if (r.status === 404) {
          return { __fallback404: true };
        }
        if (!r.ok) {
          return r.json().then(function (j) {
            throw new Error((j && (j.detail || JSON.stringify(j))) || ("HTTP " + r.status));
          });
        }
        return r.json();
      })
      .then(function (data) {
        var schema = data && data.__fallback404
          ? clientFixtureFallback(rawText, state.parserMode, simFail)
          : data;
        populateFormFromSchema(schema);
        notify("JD parsed (" + ((schema.parser_metadata && schema.parser_metadata.engine_used) || "static") + "). Review fields below.", "success");
      })
      .catch(function (err) {
        notify("Could not parse JD: " + err.message, "error");
      })
      .finally(function () {
        if (btn) {
          btn.disabled = false;
          btn.classList.remove("opacity-60", "cursor-wait");
        }
      });
  }

  function renderCohortSummaryText(summary) {
    if (!summary) return "Cohort re-match completed.";
    var total = summary.students_total != null ? summary.students_total : "?";
    var elig = summary.eligible != null ? summary.eligible : 0;
    var gapsObj = summary.top_gaps || {};
    var gaps = [];
    if (Array.isArray(gapsObj)) {
      gaps = gapsObj.slice(0, 4).map(function (g) {
        return Array.isArray(g) ? (g[0] + " (" + g[1] + ")") : (g.skill || JSON.stringify(g));
      });
    } else {
      gaps = Object.keys(gapsObj).slice(0, 4).map(function (k) {
        return k + " (" + gapsObj[k] + ")";
      });
    }
    return "Eligible students: " + elig + " / " + total +
      (gaps.length ? (" | Top skill gaps: " + gaps.join(", ")) : " | No major skill gaps");
  }

  function pollCohortTask(taskId, initialSummary) {
    var banner = $("cd-rematch-banner");
    var titleEl = $("cd-rematch-title");
    var detailEl = $("cd-rematch-details");
    var spinnerEl = $("cd-rematch-spinner");
    var reloadBtn = $("cd-rematch-reload-btn");

    if (banner) banner.classList.remove("hidden");
    if (spinnerEl) spinnerEl.classList.remove("hidden");
    if (titleEl) titleEl.textContent = "Running cohort re-matching in background...";
    if (detailEl && initialSummary) detailEl.textContent = renderCohortSummaryText(initialSummary);

    if (!taskId) {
      if (spinnerEl) spinnerEl.classList.add("hidden");
      if (titleEl) titleEl.textContent = "Drive published and cohort re-matched.";
      if (reloadBtn) reloadBtn.classList.remove("hidden");
      notify(renderCohortSummaryText(initialSummary), "success");
      return;
    }

    var attempts = 0;
    function tick() {
      attempts += 1;
      fetch("/api/tasks/" + encodeURIComponent(taskId))
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (t) {
          if (t.status === "completed" || t.status === "succeeded" || t.status === "done") {
            if (spinnerEl) spinnerEl.classList.add("hidden");
            if (titleEl) titleEl.textContent = "Cohort re-match complete";
            var summary = (t.result && (t.result.cohort_summary || t.result)) || initialSummary;
            var msg = renderCohortSummaryText(summary);
            if (detailEl) detailEl.textContent = msg;
            if (reloadBtn) reloadBtn.classList.remove("hidden");
            notify("Cohort re-match complete: " + msg, "success");
            return;
          }
          if (t.status === "failed") {
            if (spinnerEl) spinnerEl.classList.add("hidden");
            if (titleEl) titleEl.textContent = "Cohort re-match failed";
            if (detailEl) detailEl.textContent = t.error || "Background task failed.";
            notify("Cohort re-match failed: " + (t.error || "unknown"), "error");
            return;
          }
          if (attempts < 40) {
            setTimeout(tick, 600);
          }
        })
        .catch(function () {
          if (attempts < 10) {
            setTimeout(tick, 800);
          } else if (spinnerEl) {
            spinnerEl.classList.add("hidden");
          }
        });
    }
    setTimeout(tick, 400);
  }

  function submitCampusDrive(publishNow) {
    var payload = buildSchemaPayload();
    if (!payload.company || !payload.role_title) {
      notify("Company and Role Title are required.", "error");
      return;
    }
    payload.publish_now = !!publishNow;
    payload.confirm = !!publishNow;

    var btn = publishNow ? $("cd-publish-now-btn") : $("cd-submit-queue-btn");
    if (btn) {
      btn.disabled = true;
      btn.classList.add("opacity-60", "cursor-wait");
    }

    fetch("/api/campus-drives", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) {
        return r.json().then(function (j) {
          if (!r.ok) throw new Error((j && (j.detail || JSON.stringify(j))) || ("HTTP " + r.status));
          return j;
        });
      })
      .then(function (res) {
        if (publishNow) {
          notify("Campus drive published to On-Campus Opportunities.", "success");
          pollCohortTask(res.task_id, res.cohort_summary);
        } else {
          notify("Submitted to TPO Approval Queue (" + (res.approval_id || res.id) + ").", "success");
          closeSmartJdModal();
        }
      })
      .catch(function (err) {
        notify("Could not save campus drive: " + err.message, "error");
      })
      .finally(function () {
        if (btn) {
          btn.disabled = false;
          btn.classList.remove("opacity-60", "cursor-wait");
        }
      });
  }

  function saveEditedDrive() {
    if (!state.editingDriveId) return;
    var payload = buildSchemaPayload();
    if (!payload.company || !payload.role_title) {
      notify("Company and Role Title are required.", "error");
      return;
    }
    var btn = $("cd-save-edit-btn");
    if (btn) {
      btn.disabled = true;
      btn.classList.add("opacity-60", "cursor-wait");
    }
    fetch("/api/campus-drives/" + encodeURIComponent(state.editingDriveId), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (r) {
        return r.json().then(function (j) {
          if (!r.ok) throw new Error((j && (j.detail || JSON.stringify(j))) || ("HTTP " + r.status));
          return j;
        });
      })
      .then(function (updated) {
        notify("Campus drive updated and re-validated.", "success");
        closeSmartJdModal();
        if (typeof state.onSavedCallback === "function") {
          state.onSavedCallback(updated);
        }
      })
      .catch(function (err) {
        notify("Could not update drive: " + err.message, "error");
      })
      .finally(function () {
        if (btn) {
          btn.disabled = false;
          btn.classList.remove("opacity-60", "cursor-wait");
        }
      });
  }

  function openSmartJdModal() {
    state.mode = "create";
    state.editingDriveId = null;
    state.onSavedCallback = null;
    if ($("smart-jd-modal-title")) $("smart-jd-modal-title").textContent = "Add Campus Drive (Smart JD)";
    if ($("jd-ingest-section")) $("jd-ingest-section").classList.remove("hidden");
    if ($("cd-create-actions")) $("cd-create-actions").classList.remove("hidden");
    if ($("cd-edit-actions")) $("cd-edit-actions").classList.add("hidden");
    if ($("cd-rematch-banner")) $("cd-rematch-banner").classList.add("hidden");
    if ($("cd-rematch-reload-btn")) $("cd-rematch-reload-btn").classList.add("hidden");

    populateFormFromSchema({
      company: "",
      role_title: "",
      location: "",
      work_mode: "On-site",
      experience_level: "Fresher",
      salary_disclosed: true,
      eligibility: { allowed_branches: [] },
      required_skills: [],
      preferred_skills: [],
      unmapped_skill_mentions: []
    });
    if ($("jd-explainability-banner")) $("jd-explainability-banner").classList.add("hidden");
    if ($("smart-jd-modal")) $("smart-jd-modal").classList.remove("hidden");
    loadTaxonomy();
  }

  function openSmartJdModalForEdit(driveId, fallbackSchema, onSaved) {
    state.mode = "edit";
    state.editingDriveId = driveId;
    state.onSavedCallback = onSaved || null;
    if ($("smart-jd-modal-title")) $("smart-jd-modal-title").textContent = "Edit Campus Drive (" + driveId + ")";
    if ($("jd-ingest-section")) $("jd-ingest-section").classList.add("hidden");
    if ($("cd-create-actions")) $("cd-create-actions").classList.add("hidden");
    if ($("cd-edit-actions")) $("cd-edit-actions").classList.remove("hidden");
    if ($("cd-rematch-banner")) $("cd-rematch-banner").classList.add("hidden");
    if ($("smart-jd-modal")) $("smart-jd-modal").classList.remove("hidden");

    loadTaxonomy();
    fetch("/api/campus-drives/" + encodeURIComponent(driveId))
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        populateFormFromSchema(data || fallbackSchema || {});
      })
      .catch(function () {
        populateFormFromSchema(fallbackSchema || {});
      });
  }

  function closeSmartJdModal() {
    if ($("smart-jd-modal")) $("smart-jd-modal").classList.add("hidden");
  }

  // -------------------------------------------------------------------------
  // Offline Pure-JS QR SVG Matrix Generator (Model 2 Byte Mode, EC Level L)
  // -------------------------------------------------------------------------
  var QR_EXP = new Array(256);
  var QR_LOG = new Array(256);
  (function initGf() {
    var x = 1;
    for (var i = 0; i < 255; i++) {
      QR_EXP[i] = x;
      QR_LOG[x] = i;
      x <<= 1;
      if (x & 256) x ^= 0x11d;
    }
    QR_EXP[255] = QR_EXP[0];
  })();

  function gfMul(a, b) {
    if (a === 0 || b === 0) return 0;
    return QR_EXP[(QR_LOG[a] + QR_LOG[b]) % 255];
  }

  function rsPoly(ecLen) {
    var g = [1];
    for (var i = 0; i < ecLen; i++) {
      var next = new Array(g.length + 1).fill(0);
      var root = QR_EXP[i];
      for (var j = 0; j < g.length; j++) {
        next[j] ^= g[j];
        next[j + 1] ^= gfMul(g[j], root);
      }
      g = next;
    }
    return g;
  }

  function rsEncode(data, ecLen) {
    var gen = rsPoly(ecLen);
    var res = data.slice().concat(new Array(ecLen).fill(0));
    for (var i = 0; i < data.length; i++) {
      var coef = res[i];
      if (coef !== 0) {
        for (var j = 0; j < gen.length; j++) {
          res[i + j] ^= gfMul(gen[j], coef);
        }
      }
    }
    return res.slice(data.length);
  }

  var QR_VERSIONS = [
    null,
    { size: 21, dataBytes: 19, ecBytes: 7, align: [] },
    { size: 25, dataBytes: 34, ecBytes: 10, align: [6, 18] },
    { size: 29, dataBytes: 55, ecBytes: 15, align: [6, 22] },
    { size: 33, dataBytes: 80, ecBytes: 20, align: [6, 26] },
    { size: 37, dataBytes: 108, ecBytes: 26, align: [6, 30] },
    { size: 41, dataBytes: 136, ecBytes: 18, blocks: 2, align: [6, 34] },
    { size: 45, dataBytes: 156, ecBytes: 20, blocks: 2, align: [6, 22, 38] },
    { size: 49, dataBytes: 194, ecBytes: 24, blocks: 2, align: [6, 24, 42] }
  ];

  function encodeQrMatrix(text) {
    var bytes = [];
    for (var i = 0; i < text.length; i++) {
      var c = text.charCodeAt(i);
      if (c < 128) bytes.push(c);
      else bytes.push(63);
    }
    var ver = 1;
    while (ver < QR_VERSIONS.length && (bytes.length + 2) > QR_VERSIONS[ver].dataBytes) {
      ver++;
    }
    if (ver >= QR_VERSIONS.length) return null;

    var spec = QR_VERSIONS[ver];
    var bits = [];
    function pushBits(val, len) {
      for (var b = len - 1; b >= 0; b--) bits.push((val >> b) & 1);
    }
    pushBits(4, 4); // byte mode
    pushBits(bytes.length, 8);
    bytes.forEach(function (b) { pushBits(b, 8); });
    pushBits(0, Math.min(4, spec.dataBytes * 8 - bits.length));
    while (bits.length % 8 !== 0) bits.push(0);
    var padBytes = [0xec, 0x11];
    var pi = 0;
    while (bits.length < spec.dataBytes * 8) {
      pushBits(padBytes[pi % 2], 8);
      pi++;
    }

    var codewords = [];
    for (var k = 0; k < bits.length; k += 8) {
      var v = 0;
      for (var b2 = 0; b2 < 8; b2++) v = (v << 1) | bits[k + b2];
      codewords.push(v);
    }

    var blocks = spec.blocks || 1;
    var allData = [];
    var allEc = [];
    var bSize = Math.floor(codewords.length / blocks);
    var extra = codewords.length % blocks;
    var offset = 0;
    for (var bi = 0; bi < blocks; bi++) {
      var len = bSize + (bi >= blocks - extra ? 1 : 0);
      var slice = codewords.slice(offset, offset + len);
      offset += len;
      allData.push(slice);
      allEc.push(rsEncode(slice, spec.ecBytes));
    }

    var finalBytes = [];
    var maxDataLen = Math.max.apply(null, allData.map(function (a) { return a.length; }));
    for (var di = 0; di < maxDataLen; di++) {
      for (var b3 = 0; b3 < blocks; b3++) {
        if (di < allData[b3].length) finalBytes.push(allData[b3][di]);
      }
    }
    for (var ei = 0; ei < spec.ecBytes; ei++) {
      for (var b4 = 0; b4 < blocks; b4++) {
        finalBytes.push(allEc[b4][ei]);
      }
    }

    var n = spec.size;
    var mod = Array.from({ length: n }, function () { return new Array(n).fill(null); });
    var fixed = Array.from({ length: n }, function () { return new Array(n).fill(false); });

    function setFixed(r, c, val) {
      if (r >= 0 && r < n && c >= 0 && c < n) {
        mod[r][c] = val;
        fixed[r][c] = true;
      }
    }

    function drawFinder(r0, c0) {
      for (var dr = -1; dr <= 7; dr++) {
        for (var dc = -1; dc <= 7; dc++) {
          var inBox = dr >= 0 && dr <= 6 && dc >= 0 && dc <= 6;
          var onBorder = dr === 0 || dr === 6 || dc === 0 || dc === 6;
          var inCore = dr >= 2 && dr <= 4 && dc >= 2 && dc <= 4;
          setFixed(r0 + dr, c0 + dc, inBox && (onBorder || inCore) ? 1 : 0);
        }
      }
    }

    drawFinder(0, 0);
    drawFinder(0, n - 7);
    drawFinder(n - 7, 0);

    for (var t = 8; t < n - 8; t++) {
      setFixed(6, t, t % 2 === 0 ? 1 : 0);
      setFixed(t, 6, t % 2 === 0 ? 1 : 0);
    }
    setFixed(n - 8, 8, 1);

    if (spec.align.length) {
      spec.align.forEach(function (ar) {
        spec.align.forEach(function (ac) {
          if (fixed[ar][ac]) return;
          for (var dr = -2; dr <= 2; dr++) {
            for (var dc = -2; dc <= 2; dc++) {
              var onB = Math.abs(dr) === 2 || Math.abs(dc) === 2 || (dr === 0 && dc === 0);
              setFixed(ar + dr, ac + dc, onB ? 1 : 0);
            }
          }
        });
      });
    }

    for (var f = 0; f <= 8; f++) {
      if (f !== 6) {
        if (!fixed[8][f]) setFixed(8, f, 0);
        if (!fixed[f][8]) setFixed(f, 8, 0);
      }
    }
    for (var f2 = n - 8; f2 < n; f2++) {
      if (!fixed[8][f2]) setFixed(8, f2, 0);
      if (!fixed[f2][8]) setFixed(f2, 8, 0);
    }

    var bitStream = [];
    finalBytes.forEach(function (b) {
      for (var s = 7; s >= 0; s--) bitStream.push((b >> s) & 1);
    });

    var bitIdx = 0;
    var dir = -1;
    for (var col = n - 1; col >= 1; col -= 2) {
      if (col === 6) col--;
      for (var step = 0; step < n; step++) {
        var row = dir === -1 ? (n - 1 - step) : step;
        for (var dc2 = 0; dc2 < 2; dc2++) {
          var cc = col - dc2;
          if (!fixed[row][cc]) {
            var bit = bitIdx < bitStream.length ? bitStream[bitIdx++] : 0;
            // Mask 0: (row + col) % 2 === 0
            if ((row + cc) % 2 === 0) bit ^= 1;
            mod[row][cc] = bit;
          }
        }
      }
      dir = -dir;
    }

    // Format info for EC Level L (01) + Mask 0 (000) -> 01000 -> 111011111000100
    var fmt = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0];
    var coordsA = [
      [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 7], [8, 8],
      [7, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8]
    ];
    var coordsB = [
      [n - 1, 8], [n - 2, 8], [n - 3, 8], [n - 4, 8], [n - 5, 8], [n - 6, 8], [n - 7, 8],
      [8, n - 8], [8, n - 7], [8, n - 6], [8, n - 5], [8, n - 4], [8, n - 3], [8, n - 2], [8, n - 1]
    ];
    for (var fi = 0; fi < 15; fi++) {
      mod[coordsA[fi][0]][coordsA[fi][1]] = fmt[fi];
      mod[coordsB[fi][0]][coordsB[fi][1]] = fmt[fi];
    }

    return mod;
  }

  function renderQrSvg(url) {
    var matrix = encodeQrMatrix(url);
    if (!matrix) {
      return '<div class="w-28 h-28 flex items-center justify-center text-[10px] font-mono text-[#7A776E] text-center p-2">URL ready to copy</div>';
    }
    var n = matrix.length;
    var rects = "";
    for (var r = 0; r < n; r++) {
      for (var c = 0; c < n; c++) {
        if (matrix[r][c] === 1) {
          rects += '<rect x="' + (c + 2) + '" y="' + (r + 2) + '" width="1" height="1" fill="#1C1E1B"/>';
        }
      }
    }
    var vb = n + 4;
    return '<svg width="116" height="116" viewBox="0 0 ' + vb + ' ' + vb + '" shape-rendering="crispEdges">' +
      '<rect width="' + vb + '" height="' + vb + '" fill="#FFFFFF"/>' + rects + '</svg>';
  }

  // -------------------------------------------------------------------------
  // JNF Link Drawer
  // -------------------------------------------------------------------------
  function generateJnfLink() {
    var expEl = $("jnf-expires-hours");
    var hours = expEl ? parseInt(expEl.value || "72", 10) : 72;
    var btn = $("jnf-generate-btn");
    if (btn) {
      btn.disabled = true;
      btn.classList.add("opacity-60", "cursor-wait");
    }

    fetch("/api/campus-drives/jnf-links", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expires_hours: hours })
    })
      .then(function (r) {
        if (r.status === 404) {
          return { token: "demo-jnf-token", url: "/jnf/demo-jnf-token", __fallback404: true };
        }
        return r.json().then(function (j) {
          if (!r.ok) throw new Error((j && (j.detail || JSON.stringify(j))) || ("HTTP " + r.status));
          return j;
        });
      })
      .then(function (res) {
        var fullUrl = window.location.origin + (res.url || ("/jnf/" + res.token));
        var resCard = $("jnf-result-card");
        var urlInp = $("jnf-full-url");
        var openLink = $("jnf-open-link");
        var qrEl = $("jnf-qr-svg");
        if (resCard) resCard.classList.remove("hidden");
        if (urlInp) urlInp.value = fullUrl;
        if (openLink) openLink.href = fullUrl;
        if (qrEl) qrEl.innerHTML = renderQrSvg(fullUrl);
        notify("Recruiter JNF link generated (valid for " + hours + "h).", "success");
      })
      .catch(function (err) {
        notify("Could not generate JNF link: " + err.message, "error");
      })
      .finally(function () {
        if (btn) {
          btn.disabled = false;
          btn.classList.remove("opacity-60", "cursor-wait");
        }
      });
  }

  function openJnfDrawer() {
    var modal = $("jnf-drawer-modal");
    if (modal) modal.classList.remove("hidden");
    generateJnfLink();
  }

  function closeJnfDrawer() {
    var modal = $("jnf-drawer-modal");
    if (modal) modal.classList.add("hidden");
  }

  // -------------------------------------------------------------------------
  // Event Wiring
  // -------------------------------------------------------------------------
  function init() {
    var openModalBtn = $("open-smart-jd-modal");
    var closeModalBtn = $("close-smart-jd-modal");
    var cancelModalBtn = $("cd-cancel-btn");
    if (openModalBtn && !openModalBtn.disabled) {
      openModalBtn.addEventListener("click", openSmartJdModal);
    }
    if (closeModalBtn) closeModalBtn.addEventListener("click", closeSmartJdModal);
    if (cancelModalBtn) cancelModalBtn.addEventListener("click", closeSmartJdModal);

    var openJnfBtn = $("open-jnf-drawer-btn");
    var closeJnfBtn = $("close-jnf-drawer");
    var genJnfBtn = $("jnf-generate-btn");
    var copyJnfBtn = $("jnf-copy-btn");
    if (openJnfBtn && !openJnfBtn.disabled) {
      openJnfBtn.addEventListener("click", openJnfDrawer);
    }
    if (closeJnfBtn) closeJnfBtn.addEventListener("click", closeJnfDrawer);
    if (genJnfBtn) genJnfBtn.addEventListener("click", generateJnfLink);
    if (copyJnfBtn) {
      copyJnfBtn.addEventListener("click", function () {
        var inp = $("jnf-full-url");
        if (!inp || !inp.value) return;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(inp.value).then(function () {
            notify("JNF URL copied to clipboard.", "success");
          }).catch(function () {
            inp.select();
            document.execCommand("copy");
            notify("JNF URL copied.", "success");
          });
        } else {
          inp.select();
          document.execCommand("copy");
          notify("JNF URL copied.", "success");
        }
      });
    }

    // Tabs: Upload vs Paste
    document.querySelectorAll(".jd-tab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.activeTab = btn.getAttribute("data-jd-tab");
        document.querySelectorAll(".jd-tab-btn").forEach(function (b) {
          var isAct = b.getAttribute("data-jd-tab") === state.activeTab;
          b.className = isAct
            ? "jd-tab-btn px-3.5 py-1.5 rounded-full bg-white text-[#1C1E1B] shadow-2xs border border-[#D5D2C6] transition-all cursor-pointer"
            : "jd-tab-btn px-3.5 py-1.5 rounded-full text-[#5C5A52] hover:text-[#1C1E1B] transition-all cursor-pointer";
        });
        if ($("jd-panel-upload")) $("jd-panel-upload").classList.toggle("hidden", state.activeTab !== "upload");
        if ($("jd-panel-paste")) $("jd-panel-paste").classList.toggle("hidden", state.activeTab !== "paste");
      });
    });

    // Parser mode segmented control
    document.querySelectorAll(".jd-mode-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.parserMode = btn.getAttribute("data-parser-mode");
        document.querySelectorAll(".jd-mode-btn").forEach(function (b) {
          var isAct = b.getAttribute("data-parser-mode") === state.parserMode;
          b.className = isAct
            ? "jd-mode-btn px-3 py-1 rounded-full bg-white text-[#1C1E1B] border border-[#D5D2C6] shadow-2xs cursor-pointer"
            : "jd-mode-btn px-3 py-1 rounded-full text-[#5C5A52] hover:text-[#1C1E1B] cursor-pointer";
        });
      });
    });

    // Sample JD button
    var sampleBtn = $("jd-load-sample-btn");
    if (sampleBtn) {
      sampleBtn.addEventListener("click", function () {
        var pasteTab = $("jd-tab-paste");
        if (pasteTab) pasteTab.click();
        if ($("jd-raw-text")) $("jd-raw-text").value = SAMPLE_JD;
      });
    }

    // Drag & drop + file input
    var fileInp = $("jd-file-input");
    var dropzone = $("jd-dropzone");
    var fileLbl = $("jd-selected-file");
    function setSelectedFile(f) {
      state.selectedFile = f || null;
      if (fileLbl) {
        if (f) {
          fileLbl.textContent = f.name + " (" + Math.max(1, Math.round(f.size / 1024)) + " KB)";
          fileLbl.classList.remove("hidden");
        } else {
          fileLbl.classList.add("hidden");
        }
      }
    }
    if (fileInp) {
      fileInp.addEventListener("change", function () {
        setSelectedFile(fileInp.files && fileInp.files[0]);
      });
    }
    if (dropzone) {
      dropzone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropzone.classList.add("border-[#1D5C42]", "bg-[#E3F2EA]/40");
      });
      dropzone.addEventListener("dragleave", function () {
        dropzone.classList.remove("border-[#1D5C42]", "bg-[#E3F2EA]/40");
      });
      dropzone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropzone.classList.remove("border-[#1D5C42]", "bg-[#E3F2EA]/40");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
          setSelectedFile(e.dataTransfer.files[0]);
        }
      });
    }

    if ($("jd-parse-btn")) $("jd-parse-btn").addEventListener("click", runParseJd);

    // Live SOP-01 tier updates
    ["cd-ctc-min", "cd-ctc-max", "cd-experience-level", "cd-salary-disclosed"].forEach(function (id) {
      var el = $(id);
      if (el) {
        el.addEventListener("input", updateLiveTierBadge);
        el.addEventListener("change", updateLiveTierBadge);
      }
    });

    // Branches multi-select
    document.querySelectorAll("#cd-branches .cd-branch-chip").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var br = btn.getAttribute("data-branch");
        if (state.allowedBranches.indexOf(br) >= 0) {
          state.allowedBranches = state.allowedBranches.filter(function (b) { return b !== br; });
        } else {
          state.allowedBranches.push(br);
        }
        renderBranchChips();
      });
    });
    if ($("cd-branches-clear")) {
      $("cd-branches-clear").addEventListener("click", function () {
        state.allowedBranches = [];
        renderBranchChips();
      });
    }

    // Skill autocompletes
    wireSkillAutocomplete("cd-req-skill-input", "cd-req-skill-suggest", "requiredSkills");
    wireSkillAutocomplete("cd-pref-skill-input", "cd-pref-skill-suggest", "preferredSkills");

    // Submit / Publish / Save
    if ($("cd-submit-queue-btn")) {
      $("cd-submit-queue-btn").addEventListener("click", function () {
        submitCampusDrive(false);
      });
    }
    if ($("cd-publish-now-btn")) {
      $("cd-publish-now-btn").addEventListener("click", function () {
        var msg = "Approve and publish this campus drive immediately? This will create a live On-Campus job and trigger cohort re-matching for all students.";
        if (typeof window.confirmAction === "function") {
          window.confirmAction(msg, function () {
            submitCampusDrive(true);
          });
        } else {
          submitCampusDrive(true);
        }
      });
    }
    if ($("cd-save-edit-btn")) {
      $("cd-save-edit-btn").addEventListener("click", saveEditedDrive);
    }
    if ($("cd-rematch-reload-btn")) {
      $("cd-rematch-reload-btn").addEventListener("click", function () {
        window.location.reload();
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.CampusDriveUI = {
    openSmartJdModal: openSmartJdModal,
    openSmartJdModalForEdit: openSmartJdModalForEdit,
    closeSmartJdModal: closeSmartJdModal,
    openJnfDrawer: openJnfDrawer,
    pollCohortTask: pollCohortTask,
    computeSop01Tier: computeSop01Tier,
    renderQrSvg: renderQrSvg
  };
})();
