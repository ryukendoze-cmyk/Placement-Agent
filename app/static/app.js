// Adaptive mock-interview voice UI.
// Voice out: Web Speech speechSynthesis (TTS). Voice in: webkitSpeechRecognition (STT).
// The text input + Send button are ALWAYS present as a fallback (Firefox has no SpeechRecognition).
let interviewId = "";
let recognition = null;
let recognizing = false;
let evaluating = false;
let turn = 1;
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

function $(id) { return document.getElementById(id); }

// Global toast system (replaces blocking browser dialogs). Safe to call on any page.
function toast(message, kind, ttlMs) {
  var root = document.getElementById("toast-root");
  if (!root) {
    root = document.createElement("div");
    root.id = "toast-root";
    root.setAttribute("role", "status");
    root.setAttribute("aria-live", "polite");
    document.body.appendChild(root);
  }
  var node = document.createElement("div");
  node.className = "toast " + (kind || "info");
  var text = document.createElement("span");
  text.style.flex = "1";
  text.textContent = String(message || "");
  var close = document.createElement("button");
  close.type = "button";
  close.setAttribute("aria-label", "Dismiss");
  close.textContent = "\u00d7";
  close.addEventListener("click", function () { if (node.parentNode) node.parentNode.removeChild(node); });
  node.appendChild(text);
  node.appendChild(close);
  root.appendChild(node);
  var ttl = typeof ttlMs === "number" ? ttlMs : 5000;
  setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, ttl);
  return node;
}
window.toast = toast;
function showToast(message, kind) { return toast(message, kind); }

const RUBRIC_KEYS = ["technical_accuracy", "depth_and_examples", "communication_clarity"];
const RUBRIC_LABELS = {
  technical_accuracy: "Technical accuracy",
  depth_and_examples: "Depth & examples",
  communication_clarity: "Communication clarity"
};
let timerHandle = null;
let timerStart = 0;

function speak(text) {
  try {
    if ("speechSynthesis" in window && text && text.trim()) {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
  } catch (e) { /* speech not available - ignore */ }
}

function renderStepper(done, current) {
  const el = $("stepper");
  if (!el) return;
  el.innerHTML = "";
  for (let i = 1; i <= 8; i++) {
    const dot = document.createElement("span");
    dot.className = "turn-dot" + (i <= done ? " done" : (i === current ? " current" : ""));
    if (i <= done) {
      dot.innerHTML = '<svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>';
    } else {
      dot.textContent = i;
    }
    dot.title = "Turn " + i;
    el.appendChild(dot);
  }
}

function setDifficultyBadge(d) {
  const b = $("difficulty-badge");
  if (!b) return;
  const names = ["Easy", "Medium", "Hard"];
  const v = (d === 0 || d === 1 || d === 2) ? d : 1;
  b.className = "badge badge-" + v;
  b.textContent = "Difficulty: " + names[v];
}

function renderRubric(rubric, matched, missed, ideal) {
  const box = $("rubric");
  if (!box) return;
  if (!rubric) { box.style.display = "none"; return; }
  box.style.display = "block";
  RUBRIC_KEYS.forEach(function (k) {
    const fill = $("rubric-" + k);
    const val = $("rubric-val-" + k);
    const v = Math.max(0, Math.min(100, Number(rubric[k]) || 0));
    if (fill) fill.style.width = v + "%";
    if (val) val.textContent = v;
  });
  const kw = $("keywords");
  if (kw) {
    const m = (matched || []).join(", ");
    const ms = (missed || []).join(", ");
    kw.textContent = (m ? "Matched: " + m : "") + (ms ? (m ? " · " : "") + "Missed: " + ms : "");
  }
  const ia = $("ideal-answer");
  if (ia) {
    ia.textContent = ideal || "A strong answer defines the concept, gives a concrete example, and names trade-offs.";
    ia.hidden = true;
  }
  const tg = $("toggle-ideal");
  if (tg) tg.textContent = "Show ideal answer";
}

// M18.3: display WPM / STAR / delivery / coaching telemetry pill for a turn.
function renderSpeechTelemetry(sa) {
  const box = $("speech-telemetry");
  if (!box) return;
  if (!sa) { box.hidden = true; box.textContent = ""; return; }
  const star = sa.star_score != null ? sa.star_score : 0;
  const wpm = sa.wpm || 0;
  const delivery = sa.delivery_score != null ? sa.delivery_score : 0;
  const fillers = (sa.filler_words || []).length;
  const pacing = wpm > 0 ? (wpm + " WPM") : "typed (no pacing)";
  box.hidden = false;
  box.className = "text-xs font-mono text-[#4A483F] bg-[#FAF9F5] border border-[#E5E3DC] rounded-xl px-3.5 py-2.5 mt-3";
  box.textContent = "Delivery " + delivery + "/100 | STAR " + star + "/100 | " +
    pacing + " | fillers " + fillers +
    (sa.coaching_tip ? " | " + sa.coaching_tip : "");
}

function startTimer() {
  stopTimer();
  timerStart = Date.now();
  timerHandle = setInterval(function () {
    const t = $("timer");
    if (!t) return;
    const s = Math.floor((Date.now() - timerStart) / 1000);
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    t.textContent = mm + ":" + ss;
  }, 1000);
}

function stopTimer() {
  if (timerHandle) { clearInterval(timerHandle); timerHandle = null; }
}

function setPulse(on) {
  const p = $("pulse");
  if (p) p.className = on ? "on" : "";
}

function showQuestion(question) {
  $("status").textContent = "Turn " + turn + " of 8";
  $("question").textContent = question.question;
  $("skill").textContent = question.skill_id;
  setDifficultyBadge(question.difficulty);
  renderStepper(turn - 1, turn);
  setPulse(true);
  startTimer();
  speak(question.question);
}

function setEvaluating(active) {
  evaluating = active;
  const sendBtn = $("send");
  const micBtn = $("mic");
  const answerBox = $("answer");
  if (active) {
    if (sendBtn) {
      sendBtn.disabled = true;
      sendBtn.textContent = "Evaluating with AI...";
      sendBtn.classList.add("opacity-60", "cursor-wait");
    }
    if (micBtn) micBtn.disabled = true;
    if (answerBox) answerBox.disabled = true;
    $("status").textContent = "Turn " + turn + " of 8 · Evaluating with AI...";
    const fb = $("feedback");
    if (fb) {
      fb.className = "text-sm font-medium text-[#1B5E43] bg-[#E3F2EA] border border-[#C8E4D5] rounded-2xl px-4 py-3 mb-4";
      fb.textContent = turn >= 8
        ? "AI is evaluating your final answer and generating overall interview feedback..."
        : "AI is evaluating your answer and preparing the next question...";
    }
  } else {
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send answer";
      sendBtn.classList.remove("opacity-60", "cursor-wait");
    }
    if (micBtn && SR) micBtn.disabled = false;
    if (answerBox) answerBox.disabled = false;
  }
}

function renderSummary(d) {
  $("status").textContent = "Interview complete";
  $("interview").style.display = "none";
  $("summary").style.display = "block";
  stopTimer();
  setPulse(false);
  renderStepper(8, 8);
  $("score").textContent = (d.score == null ? "?" : d.score) + "/100";
  $("summary-feedback").textContent = d.feedback || "";
  const bars = $("bars");
  bars.innerHTML = "";
  Object.keys(d.per_skill || {}).forEach(function (k) {
    const v = d.per_skill[k];
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = k;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = Math.max(0, Math.min(100, v)) + "%";
    track.appendChild(fill);
    const val = document.createElement("span");
    val.className = "bar-val";
    val.textContent = v;
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(val);
    bars.appendChild(row);
  });
  const ro = $("rubric-overall");
  if (ro) {
    ro.innerHTML = "";
    const rub = d.rubric_overall || {};
    RUBRIC_KEYS.forEach(function (k) {
      const row = document.createElement("div");
      row.className = "bar-row";
      const label = document.createElement("span");
      label.className = "bar-label";
      label.textContent = RUBRIC_LABELS[k] || k.replace(/_/g, " ");
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      const v = Math.max(0, Math.min(100, Number(rub[k]) || 0));
      fill.style.width = v + "%";
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "bar-val";
      val.textContent = rub[k] == null ? "—" : rub[k];
      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);
      ro.appendChild(row);
    });
  }
}

function printScorecard() {
  window.print();
}

async function loadInitial() {
  const r = await fetch("/api/interviews/" + interviewId);
  if (!r.ok) { $("question").textContent = "Interview not found."; return; }
  const d = await r.json();
  if (d.finished) {
    renderSummary({ score: d.score, per_skill: (d.feedback || {}).per_skill,
                    feedback: (d.feedback || {}).overall,
                    rubric_overall: (d.feedback || {}).rubric_overall });
    return;
  }
  turn = d.transcript.filter(function (t) { return t.score !== null; }).length + 1;
  const pending = (function () {
    for (let i = d.transcript.length - 1; i >= 0; i--) {
      if (d.transcript[i].score === null) return d.transcript[i];
    }
    return null;
  })();
  if (pending) showQuestion({ question: pending.q, skill_id: pending.skill_id, difficulty: pending.difficulty });
}

async function submitAnswer(text) {
  if (evaluating) return;
  text = (text || "").trim();
  if (!text) return;
  // M18.3: elapsed turn duration for acoustic telemetry (guard clock skew).
  const elapsedSec = (timerStart > 0)
    ? Math.max(1.0, (Date.now() - timerStart) / 1000)
    : null;
  setEvaluating(true);
  try {
    const r = await fetch("/api/interviews/" + interviewId + "/answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(elapsedSec
        ? { text: text, duration_seconds: elapsedSec }
        : { text: text })
    });
    const d = await r.json().catch(function () { return {}; });
    setEvaluating(false);
    if (!r.ok) {
      $("status").textContent = "Turn " + turn + " of 8";
      const fb = $("feedback");
      if (fb) {
        fb.className = "text-sm font-medium text-[#9E2A20] bg-[#FDF0ED] border border-[#F5D0C8] rounded-2xl px-4 py-3 mb-4";
        fb.textContent = "Could not evaluate answer: " + (d.detail || r.statusText || "Server error");
      }
      return;
    }
    if (d.finished) {
      renderSummary(d.summary);
      return;
    }
    stopTimer();
    setPulse(false);
    const fb = $("feedback");
    if (fb) {
      fb.className = "text-sm text-[#1C1E1B] bg-[#F2F7F4] border border-[#DCE8E1] rounded-2xl px-4 py-3 mb-4";
      fb.textContent = "Score " + d.score + "/100 — " + (d.feedback || "");
    }
    renderRubric(d.rubric, d.matched_keywords, d.missed_keywords, d.ideal_answer_hint);
    renderSpeechTelemetry(d.speech_analytics);
    if (d.followup_probe) {
      const probeBox = $("followup-probe");
      if (probeBox) {
        probeBox.hidden = false;
        probeBox.textContent = "Follow-up: " + d.followup_probe;
      }
    }
    turn += 1;
    $("answer").value = "";
    $("answer").focus();
    if (d.question) showQuestion(d.question);
  } catch (err) {
    setEvaluating(false);
    $("status").textContent = "Turn " + turn + " of 8";
    const fb = $("feedback");
    if (fb) {
      fb.className = "text-sm font-medium text-[#9E2A20] bg-[#FDF0ED] border border-[#F5D0C8] rounded-2xl px-4 py-3 mb-4";
      fb.textContent = "Network error while evaluating answer. Make sure the server is running and try again.";
    }
  }
}

function setMicRecording(isRecording) {
  recognizing = isRecording;
  const micBtn = $("mic");
  if (!micBtn) return;
  if (isRecording) {
    micBtn.className = "px-4 py-1.5 rounded-full bg-[#9E2A20] hover:bg-[#7F231B] text-white text-xs font-semibold inline-flex items-center justify-center gap-1.5 whitespace-nowrap cursor-pointer";
    micBtn.innerHTML = '<svg class="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg><span>Stop</span>';
  } else {
    micBtn.className = "px-4 py-1.5 rounded-full bg-[#1D5C42] hover:bg-[#154531] text-white text-xs font-semibold inline-flex items-center justify-center gap-1.5 whitespace-nowrap cursor-pointer";
    micBtn.innerHTML = '<svg class="w-3.5 h-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"/></svg><span>Record</span>';
  }
}

function startListening() {
  if (!SR || evaluating) return;
  if (recognizing) { if (recognition) recognition.stop(); setMicRecording(false); return; }
  recognition = new SR();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.onresult = function (e) {
    const spoken = e.results[0][0].transcript;
    if ($("answer")) $("answer").value = spoken;
    submitAnswer(spoken);
  };
  recognition.onend = function () { setMicRecording(false); };
  recognition.onerror = function () { setMicRecording(false); };
  setMicRecording(true);
  try { recognition.start(); } catch (e) { setMicRecording(false); }
}

function listenAgain() { speak($("question").textContent); }

document.addEventListener("DOMContentLoaded", function () {
  var iidEl = $("interview-id");
  var sendBtn = $("send");
  // interview controls only exist on /interviews/{id}; toast/shell pages skip this block
  if (!iidEl || !sendBtn || !$("answer")) return;
  interviewId = (iidEl.textContent || "").trim();
  if (!SR) { $("mic").disabled = true; $("mic").title = "Speech recognition not supported here - use the text box."; }
  sendBtn.addEventListener("click", function () { submitAnswer($("answer").value); });
  $("answer").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitAnswer($("answer").value);
    }
  });
  $("listen").addEventListener("click", listenAgain);
  $("mic").addEventListener("click", startListening);
  var ti = $("toggle-ideal");
  if (ti) ti.addEventListener("click", function () {
    const ia = $("ideal-answer");
    if (!ia) return;
    ia.hidden = !ia.hidden;
    ti.textContent = ia.hidden ? "Show ideal answer" : "Hide ideal answer";
  });
  const pb = $("print-btn");
  if (pb) pb.addEventListener("click", printScorecard);
  loadInitial();
});
