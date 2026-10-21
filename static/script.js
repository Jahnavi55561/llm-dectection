const $ = (id) => document.getElementById(id);
const input = $("input-text");
const button = $("analyze-btn");
const charCount = $("char-count");
const message = $("message");
const resultsContainer = $("results");
const historyContainer = $("history-list");
let lastResults = [];

// Character counter
input.addEventListener("input", () => {
  const len = input.value.length;
  charCount.textContent = `${len.toLocaleString()} / 10,000 characters`;
  if (len > 10000) {
    charCount.style.color = "var(--red)";
  } else {
    charCount.style.color = "var(--muted)";
  }
});

// HTML escaping to prevent XSS
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[ch]));
}

// URL sanitization (only allow http/https)
function sanitizeUrl(url) {
  if (!url) return "";
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch {
    // Relative or invalid URL
  }
  return "";
}

// Navigation active state handler
document.querySelectorAll("#sidebar-nav .nav-item").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll("#sidebar-nav .nav-item").forEach((l) => l.classList.remove("active"));
    link.classList.add("active");
  });
});

function renderResult(item) {
  const status = ["Supported", "Contradicted", "Unverified"].includes(item.status) ? item.status : "Unverified";
  const safeUrl = sanitizeUrl(item.source_url);
  const source = safeUrl
    ? `<a class="source" href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.source || "Open source")} ↗</a>`
    : (item.source ? `<div class="detail"><b>Source:</b> ${escapeHtml(item.source)}</div>` : `<div class="detail">No specific source citation available.</div>`);

  const confidenceBadge = (status !== "Unverified" && typeof item.confidence === "number" && item.confidence > 0)
    ? `<span class="confidence">Confidence: ${(item.confidence * 100).toFixed(0)}%</span>`
    : "";
  const evidence = item.evidence
    ? `<div class="evidence"><b>Retrieved Evidence:</b><br>${escapeHtml(item.evidence.slice(0, 1500))}${item.evidence.length > 1500 ? "…" : ""}</div>`
    : "";
  const correction = item.correction && item.correction_validated
    ? `<div class="correction"><b>Validated Correction:</b> ${escapeHtml(item.correction)}</div>`
    : "";

  return `
    <article class="result">
      <div class="result-head">
        <div class="claim">${item.id}. ${escapeHtml(item.sentence)}</div>
        <div class="badges">
          ${confidenceBadge}
          <span class="badge ${status}">${status}</span>
        </div>
      </div>
      <div class="detail">${escapeHtml(item.explanation || "")}</div>
      ${source}
      ${evidence}
      ${correction}
    </article>
  `;
}

// Analyze button event listener
button.addEventListener("click", async () => {
  const text = input.value.trim();
  if (!text) {
    message.textContent = "Please paste or type text to analyze.";
    input.focus();
    return;
  }
  if (text.length > 10000) {
    message.textContent = "Input text exceeds the 10,000 character limit.";
    return;
  }

  button.disabled = true;
  button.textContent = "Analyzing in Parallel…";
  message.textContent = "Retrieving evidence and running DeBERTa NLI verification…";

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    let data;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      const errText = await response.text();
      throw new Error(`Server returned HTTP ${response.status}: ${errText.slice(0, 120)}`);
    }

    if (!response.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    lastResults = data.results || [];
    resultsContainer.className = "results";
    resultsContainer.innerHTML = lastResults.length
      ? lastResults.map(renderResult).join("")
      : '<div class="empty">No sentences were detected in the input.</div>';

    const counts = data.counts || {};
    $("supported-count").textContent = counts.Supported || 0;
    $("contradicted-count").textContent = counts.Contradicted || 0;
    $("unverified-count").textContent = counts.Unverified || 0;
    $("total-count").textContent = data.total_sentences || lastResults.length;
    $("corrected-paragraph").value = data.corrected_paragraph || "";

    message.textContent = "Analysis complete. Review the evidence citations and predictions above.";
    loadHistory();
  } catch (err) {
    message.textContent = `Error: ${err.message || "Could not connect to the backend server."}`;
  } finally {
    button.disabled = false;
    button.textContent = "Analyze Text →";
  }
});

// Copy button with in-place feedback
const copyBtn = $("copy-btn");
copyBtn.addEventListener("click", async () => {
  const text = $("corrected-paragraph").value;
  if (!text) {
    copyBtn.textContent = "No text to copy";
    setTimeout(() => { copyBtn.textContent = "Copy Paragraph"; }, 1500);
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    $("corrected-paragraph").select();
    document.execCommand("copy");
  }
  copyBtn.textContent = "✓ Copied!";
  setTimeout(() => { copyBtn.textContent = "Copy Paragraph"; }, 1800);
});

// Quick claim / chat verification
const chatInput = $("chat-input");
const chatBtn = $("chat-btn");
const chatResult = $("chat-result");

async function handleQuickVerify() {
  const msg = (chatInput?.value || "").trim();
  if (!msg) {
    if (chatInput) chatInput.focus();
    return;
  }
  chatBtn.disabled = true;
  chatBtn.textContent = "Verifying…";
  chatResult.style.display = "block";
  chatResult.innerHTML = '<div class="detail">Querying knowledge sources and evaluating claim…</div>';

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });
    const data = await response.json();
    const status = ["Supported", "Contradicted", "Unverified"].includes(data.status) ? data.status : "Unverified";
    const confidenceBadge = (status !== "Unverified" && typeof data.confidence === "number" && data.confidence > 0)
      ? `<span class="confidence">Confidence: ${(data.confidence * 100).toFixed(0)}%</span>`
      : "";

    chatResult.innerHTML = `
      <div class="result-head">
        <div class="claim">${escapeHtml(msg)}</div>
        <div class="badges">
          ${confidenceBadge}
          <span class="badge ${status}">${status}</span>
        </div>
      </div>
      <div class="detail">${escapeHtml(data.explanation || data.reply || "")}</div>
      ${data.source ? `<div class="detail" style="margin-top:6px;"><b>Source:</b> ${escapeHtml(data.source)}</div>` : ""}
    `;
  } catch (err) {
    chatResult.innerHTML = `<div class="detail" style="color:var(--red);">Verification request failed: ${escapeHtml(err.message)}</div>`;
  } finally {
    chatBtn.disabled = false;
    chatBtn.textContent = "Verify Claim →";
  }
}

if (chatBtn && chatInput) {
  chatBtn.addEventListener("click", handleQuickVerify);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleQuickVerify();
    }
  });
}

// History loader
async function loadHistory() {
  try {
    const response = await fetch("/api/history");
    const data = await response.json();
    const history = data.history || [];

    if (!history.length) {
      historyContainer.className = "history-list empty";
      historyContainer.innerHTML = "No previous analyses recorded yet.";
      return;
    }

    historyContainer.className = "history-list";
    historyContainer.innerHTML = history.map((item) => `
      <div class="history-item">
        <div class="history-meta">
          <small>${new Date(item.timestamp).toLocaleTimeString()} · ${item.total_sentences} ${item.total_sentences === 1 ? "sentence" : "sentences"}</small>
          <span class="counts-badge">
            <span style="color:var(--green)">✓ ${item.counts?.Supported || 0}</span>
            <span style="color:var(--red)">✕ ${item.counts?.Contradicted || 0}</span>
            <span style="color:var(--amber)">? ${item.counts?.Unverified || 0}</span>
          </span>
        </div>
        <div class="history-preview">${escapeHtml(item.preview)}</div>
      </div>
    `).join("");
  } catch {
    historyContainer.innerHTML = '<div class="empty">History unavailable.</div>';
  }
}

// Clear history button with backend sync
$("clear-history-btn").addEventListener("click", async () => {
  try {
    await fetch("/api/history/clear", { method: "POST" });
  } catch {
    // Continue even if network error
  }
  historyContainer.className = "history-list empty";
  historyContainer.innerHTML = "History cleared.";
});

// Health check loader
async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const model = data.verification_model || {};
    let statusText = "will load on first analysis";
    if (model.available === true) statusText = "ready & loaded";
    if (model.available === false) statusText = "unavailable (see README)";

    $("health").textContent = `Backend: online · NLI Model: ${model.name || "not configured"} (${statusText}).`;
  } catch {
    $("health").textContent = "Backend health check failed. Ensure app.py is running.";
  }
}

loadHealth();
loadHistory();

