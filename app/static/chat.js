const $ = (id) => document.getElementById(id);
const sessionId = sessionStorage.getItem("swufe-session") || crypto.randomUUID();
sessionStorage.setItem("swufe-session", sessionId);

function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("show");
  window.setTimeout(() => $("toast").classList.remove("show"), 2200);
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function addMessage(role, text, mode = "", refused = false) {
  const article = document.createElement("article");
  article.className = `message ${role}${refused ? " refused" : ""}`;
  const meta = document.createElement("div");
  meta.className = "message-meta";
  const name = document.createElement("span");
  name.textContent = role === "user" ? "你" : "教务助手";
  const badge = document.createElement("b");
  badge.textContent = mode === "school_rag" ? (refused ? "证据不足" : "可信 RAG") : (mode === "general_chat" ? "普通对话" : "已发送");
  meta.append(name, badge);
  const body = document.createElement("p");
  body.textContent = text;
  article.append(meta, body);
  $("messages").append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
}

function sourceButton(citation) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "source-card";
  const title = document.createElement("strong");
  title.textContent = `[${citation.marker}] ${citation.doc_title}`;
  const meta = document.createElement("span");
  meta.textContent = `${citation.article} · ${citation.chunk_id}`;
  button.append(title, meta);
  button.addEventListener("click", () => loadSource(citation.chunk_id));
  return button;
}

function officialLink(link) {
  const anchor = document.createElement("a");
  anchor.className = "official-link";
  anchor.href = link.page_url;
  anchor.target = "_blank";
  anchor.rel = "noreferrer";
  anchor.textContent = `${link.title} ↗`;
  return anchor;
}

function renderSources(payload) {
  const host = $("sources");
  const items = [
    ...(payload.citations || []).map(sourceButton),
    ...(payload.official_links || []).map(officialLink),
  ];
  host.className = items.length ? "" : "source-empty";
  host.replaceChildren(...(items.length ? items : [document.createTextNode("本轮没有引用或查询入口。")]));
  $("sourceCount").textContent = String(items.length);
  $("sourceDetail").hidden = true;
}

async function loadSource(chunkId) {
  try {
    const source = await api(`/source/${encodeURIComponent(chunkId)}`);
    const detail = $("sourceDetail");
    detail.replaceChildren();
    const title = document.createElement("h3");
    title.textContent = source.doc_title;
    const meta = document.createElement("p");
    meta.textContent = `${source.article} · ${source.college} · ${source.cohort}`;
    const text = document.createElement("pre");
    text.textContent = source.text;
    const page = document.createElement("a");
    page.href = source.page_url; page.target = "_blank"; page.rel = "noreferrer"; page.textContent = "通知页 ↗";
    const file = document.createElement("a");
    file.href = source.file_url; file.target = "_blank"; file.rel = "noreferrer"; file.textContent = "附件原文 ↗";
    detail.append(title, meta, text, page, file);
    detail.hidden = false;
  } catch (error) { toast(error.message); }
}

async function boot() {
  try {
    const options = await api("/options");
    for (const value of options.colleges || []) $("college").append(new Option(value, value));
    for (const value of options.cohorts || []) $("cohort").append(new Option(value, value));
    $("status").textContent = `${options.chunk_count} 个知识块 · ${options.mode}`;
  } catch (error) { $("status").textContent = error.message; }
}

$("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  addMessage("user", question);
  $("question").value = "";
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  $("status").textContent = "正在判断问题类型…";
  try {
    const payload = await api("/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        college: $("college").value || null,
        cohort: $("cohort").value || null,
        session_id: sessionId,
      }),
    });
    addMessage("assistant", payload.answer_md, payload.mode, payload.refused);
    renderSources(payload);
    $("status").textContent = `${payload.mode === "school_rag" ? "学校事实" : "普通对话"} · ${payload.latency_ms} ms`;
  } catch (error) { toast(error.message); $("status").textContent = "请求失败"; }
  finally { button.disabled = false; $("question").focus(); }
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chatForm").requestSubmit();
  }
});

boot();
