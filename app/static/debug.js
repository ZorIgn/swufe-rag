const state = { options: null, examples: [], exampleOffset: 0, lastPayload: null, controller: null };
const $ = (id) => document.getElementById(id);
const sourcePlaceholder = $("sourcePanel").innerHTML;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  window.setTimeout(() => node.classList.remove("show"), 2200);
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

function populateSelect(node, values, fallback) {
  node.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    return option;
  }));
  if (fallback && values.includes(fallback)) node.value = fallback;
}

function renderExamples() {
  const host = $("exampleList");
  const visible = [];
  for (let i = 0; i < Math.min(4, state.examples.length); i += 1) {
    visible.push(state.examples[(state.exampleOffset + i) % state.examples.length]);
  }
  host.replaceChildren(...visible.map((example) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example-item";
    button.textContent = `${example.category} / ${example.question}`;
    button.addEventListener("click", () => {
      $("question").value = example.question;
      $("college").value = example.college;
      $("cohort").value = example.cohort;
      $("question").focus();
    });
    return button;
  }));
}

async function boot() {
  try {
    const [options, examples] = await Promise.all([api("/api/debug/options"), api("/api/debug/examples")]);
    state.options = options;
    state.examples = examples;
    populateSelect($("college"), options.colleges, options.colleges[0]);
    populateSelect($("cohort"), options.cohorts, options.cohorts[0]);
    $("systemMode").textContent = `${options.mode} runtime`;
    $("systemMeta").textContent = `${options.chunk_count} chunks · scope filter ready`;
    renderExamples();
  } catch (error) {
    $("stateLamp").classList.add("offline");
    $("systemMode").textContent = "runtime offline";
    $("systemMeta").textContent = error.message;
  }
}

function requestBody() {
  return {
    question: $("question").value.trim(),
    college: $("college").value || null,
    cohort: $("cohort").value || null,
    top_k: Number($("topK").value)
  };
}

function answerHtml(markdown) {
  const safe = escapeHtml(markdown);
  return safe
    .replace(/\[(\d+)\]/g, '<button class="citation-marker" data-marker="$1" aria-label="查看引用 $1">$1</button>')
    .split(/\n{2,}/)
    .map((paragraph) => `<p>${paragraph.replace(/\n/g, "<br>")}</p>`)
    .join("");
}

function renderAnswer(payload) {
  $("answerState").hidden = true;
  const content = $("answerContent");
  content.hidden = false;
  content.classList.toggle("refused", Boolean(payload.refused));
  content.innerHTML = answerHtml(payload.answer_md);
  $("runSummary").textContent = `${payload.refused ? "拒答" : "已回答"} · ${payload.latency_ms ?? "–"} ms · ${payload.mode ?? "debug"}`;
  $("copyJson").disabled = false;
  content.querySelectorAll(".citation-marker").forEach((button) => {
    button.addEventListener("click", () => selectCitation(Number(button.dataset.marker)));
  });
}

function renderCitations(citations = []) {
  $("citationCount").textContent = `${citations.length} 条`;
  const host = $("citationList");
  host.replaceChildren(...citations.map((citation) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation-card";
    button.dataset.marker = citation.marker;
    button.innerHTML = `<strong>[${citation.marker}] ${escapeHtml(citation.doc_title)}</strong><span>${escapeHtml(citation.article)} · ${escapeHtml(citation.chunk_id)}</span><q>${escapeHtml(citation.quote)}</q>`;
    button.addEventListener("click", () => loadSource(citation.chunk_id));
    return button;
  }));
}

function renderRetrieval(retrieved = []) {
  const host = $("retrievalTable");
  if (!retrieved.length) {
    host.className = "retrieval-empty";
    host.textContent = "本次没有召回块";
    return;
  }
  host.className = "";
  host.replaceChildren(...retrieved.map((chunk, index) => {
    const row = document.createElement("div");
    row.className = "retrieval-row";
    const score = Number(chunk.score || 0);
    row.innerHTML = `<div class="retrieval-rank">${String(index + 1).padStart(2, "0")}</div><div class="retrieval-title"><button type="button">${escapeHtml(chunk.doc_title)}</button><span>${escapeHtml(chunk.article || "")}</span></div><div class="retrieval-scope">${escapeHtml(chunk.college || "")}<br>${escapeHtml(chunk.cohort || "")}</div><div><strong>${score.toFixed(3)}</strong><div class="score-bar"><i style="transform:scaleX(${Math.max(0, Math.min(1, score))})"></i></div></div><div class="retrieval-summary">${escapeHtml(chunk.summary || chunk.text || "")}</div>`;
    row.querySelector("button").addEventListener("click", () => loadSource(chunk.chunk_id));
    return row;
  }));
}

function resetSourcePanel() {
  const host = $("sourcePanel");
  host.className = "source-empty";
  host.innerHTML = sourcePlaceholder;
}

async function loadSource(chunkId) {
  const host = $("sourcePanel");
  host.className = "source-empty";
  host.textContent = "正在读取原文…";
  try {
    const source = await api(`/api/debug/source/${encodeURIComponent(chunkId)}`);
    host.className = "source-doc";
    host.innerHTML = `<h3>${escapeHtml(source.doc_title)}</h3><div class="meta">${escapeHtml(source.article)} · ${escapeHtml(source.college)} · ${escapeHtml(source.cohort)} · ${escapeHtml(source.status)}</div><pre>${escapeHtml(source.text)}</pre><div class="source-links"><a href="${escapeHtml(source.page_url)}" target="_blank" rel="noreferrer">通知页面 ↗</a><a href="${escapeHtml(source.file_url)}" target="_blank" rel="noreferrer">附件原文 ↗</a></div>`;
  } catch (error) {
    host.textContent = error.message;
  }
}

function selectCitation(marker) {
  const citation = state.lastPayload?.citations?.find((item) => Number(item.marker) === marker);
  if (citation) loadSource(citation.chunk_id);
}

async function run(mode) {
  const body = requestBody();
  if (!body.question) { toast("先输入一个问题"); $("question").focus(); return; }
  resetSourcePanel();
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  document.body.classList.add("is-loading");
  $("runSummary").textContent = mode === "ask" ? "正在检索并校验证据…" : "正在检索…";
  try {
    const payload = await api(`/api/debug/${mode}`, {
      method: "POST", body: JSON.stringify(body), signal: state.controller.signal
    });
    state.lastPayload = payload;
    if (mode === "ask") {
      renderAnswer(payload);
      renderCitations(payload.citations);
    } else {
      $("runSummary").textContent = `仅检索 · ${payload.retrieved.length} 条`;
      renderCitations([]);
    }
    renderRetrieval(payload.retrieved);
  } catch (error) {
    if (error.name !== "AbortError") toast(error.message);
  } finally {
    document.body.classList.remove("is-loading");
  }
}

$("queryForm").addEventListener("submit", (event) => { event.preventDefault(); run("ask"); });
$("retrieveButton").addEventListener("click", () => run("retrieve"));
$("topK").addEventListener("input", () => { $("topKValue").value = $("topK").value; });
$("shuffleExamples").addEventListener("click", () => { state.exampleOffset = (state.exampleOffset + 4) % state.examples.length; renderExamples(); });
$("copyJson").addEventListener("click", async () => { await navigator.clipboard.writeText(JSON.stringify(state.lastPayload, null, 2)); toast("已复制本次 JSON"); });
document.addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter") { event.preventDefault(); run("ask"); } });

boot();
