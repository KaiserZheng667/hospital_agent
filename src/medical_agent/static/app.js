const state = { user: null, csrfToken: null, patients: [], threadId: null, sessions: [], busy: false };

const elements = {
  userName: document.querySelector("#user-name"), department: document.querySelector("#department"),
  adminLink: document.querySelector("#admin-link"),
  logout: document.querySelector("#logout-button"), loginScreen: document.querySelector("#login-screen"),
  loginForm: document.querySelector("#login-form"), loginButton: document.querySelector("#login-button"),
  loginError: document.querySelector("#login-error"), username: document.querySelector("#username"),
  password: document.querySelector("#password"), sessions: document.querySelector("#session-list"),
  messages: document.querySelector("#messages"), form: document.querySelector("#composer"),
  input: document.querySelector("#message-input"), send: document.querySelector("#send-button"),
  newSession: document.querySelector("#new-session"),
  title: document.querySelector("#conversation-title"), toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && state.csrfToken && path !== "/api/auth/login") {
    headers["X-CSRF-Token"] = state.csrfToken;
  }
  const response = await fetch(path, { credentials: "same-origin", ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || "请求失败"); error.status = response.status; throw error;
  }
  return payload;
}

function showError(message) {
  elements.toast.textContent = message; elements.toast.classList.add("show");
  setTimeout(() => elements.toast.classList.remove("show"), 3000);
}
function escapeHtml(value) { const node = document.createElement("div"); node.textContent = value; return node.innerHTML; }

function renderMessage(role, content, trace = []) {
  document.querySelector("#empty-state")?.remove();
  const article = document.createElement("article"); article.className = `message ${role}`;
  const traceHtml = trace.length
    ? `<details class="trace"><summary>查看执行轨迹 · ${trace.length} 步</summary><ol class="trace-list">${trace.map(step => `<li><strong>${escapeHtml(step.node)}</strong> · ${escapeHtml(step.detail)}</li>`).join("")}</ol></details>` : "";
  article.innerHTML = role === "assistant"
    ? `<p class="assistant-label">MEDAGENT</p><div class="bubble">${escapeHtml(content)}</div>${traceHtml}`
    : `<div class="bubble">${escapeHtml(content)}</div>`;
  elements.messages.append(article); elements.messages.scrollTop = elements.messages.scrollHeight;
  return article;
}

function renderSessions() {
  elements.sessions.innerHTML = state.sessions.map(session => `
    <button class="session-item ${session.thread_id === state.threadId ? "active" : ""}" data-thread="${session.thread_id}">${escapeHtml(session.title)}</button>`).join("");
  elements.sessions.querySelectorAll("[data-thread]").forEach(button => {
    button.addEventListener("click", () => openSession(button.dataset.thread));
  });
}
async function loadSessions(autoOpen = true) {
  const payload = await api("/api/sessions"); state.sessions = payload.sessions;
  if (autoOpen && !state.threadId && state.sessions.length) state.threadId = state.sessions[0].thread_id;
  renderSessions(); if (state.threadId) await loadMessages();
}
async function loadAuthorizedPatients() { state.patients = (await api("/api/patients")).patients; }
async function createSession() {
  const payload = await api("/api/sessions", { method: "POST", body: "{}" });
  state.threadId = payload.thread_id; elements.messages.innerHTML = ""; elements.title.textContent = payload.title;
  await loadSessions(false); restoreEmptyState(); elements.input.focus();
}

function restoreEmptyState() {
  elements.messages.innerHTML = `<div id="empty-state" class="empty-state">
    <div class="empty-icon">✦</div><p class="eyebrow">MEDICAL AI AGENT</p>
    <h2>你好，我是医疗数据助手</h2>
    <p>我可以帮助你查询有权限访问的合成骨质疏松患者访视指标和历史趋势，也可以在当前会话中继续追问或复述结果。请问有什么可以帮助你？</p>
    <div class="suggestions"><button data-prompt="查询患者 P10086 的最新骨密度">查询 P10086 最新骨密度</button>
    <button data-prompt="查询患者 P20001 的25-羟基维生素D">查询 P20001 维生素D</button>
    <button data-prompt="查询患者 P30005 最近几个月的骨密度变化">查看 P30005 骨密度趋势</button></div>
    <label class="patient-picker-label" for="patient-picker">选择患者后，再输入需要查询的内容</label>
    <select id="patient-picker" class="patient-picker"></select></div>`;
  bindSuggestions();
}
async function openSession(threadId) { if (!state.busy) { state.threadId = threadId; renderSessions(); await loadMessages(); } }
async function loadMessages() {
  const payload = await api(`/api/sessions/${encodeURIComponent(state.threadId)}/messages`);
  elements.messages.innerHTML = "";
  const session = state.sessions.find(item => item.thread_id === state.threadId);
  elements.title.textContent = session?.title || "患者数据咨询";
  if (!payload.messages.length) restoreEmptyState(); else payload.messages.forEach(message => renderMessage(message.role, message.content));
}

async function sendMessage(message) {
  if (!message.trim() || state.busy) return; if (!state.threadId) await createSession();
  state.busy = true; elements.send.disabled = true; renderMessage("user", message.trim());
  const pending = renderMessage("assistant", "正在读取会话状态并执行工作流……"); pending.querySelector(".bubble").classList.add("typing");
  try {
    const payload = await api("/api/chat", { method: "POST", body: JSON.stringify({ thread_id: state.threadId, message: message.trim() }) });
    pending.remove(); renderMessage("assistant", payload.answer, payload.trace); await loadSessions(false);
    elements.title.textContent = state.sessions.find(item => item.thread_id === state.threadId)?.title || "患者数据咨询";
  } catch (error) { pending.remove(); showError(error.message); }
  finally { state.busy = false; elements.send.disabled = false; elements.input.focus(); }
}

function bindSuggestions() {
  document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => sendMessage(button.dataset.prompt)));
  const picker = document.querySelector("#patient-picker");
  if (picker) {
    picker.innerHTML = `<option value="">选择患者（共 ${state.patients.length} 人）</option>` + state.patients.map(patient => `<option value="${patient.patient_id}">${patient.patient_id} · ${patient.display_name}</option>`).join("");
    picker.addEventListener("change", () => {
      if (!picker.value) return;
      elements.input.value = `患者 ${picker.value}：`;
      elements.input.focus();
    });
  }
}

async function initializeAuthenticatedApp() {
  const bootstrap = await api("/api/bootstrap"); state.user = bootstrap.user; state.csrfToken = bootstrap.csrf_token;
  if (!state.user.permissions.includes("agent:use") && state.user.permissions.includes("user:read")) {
    window.location.replace("/admin"); return;
  }
  elements.userName.textContent = state.user.name; elements.department.textContent = state.user.department;
  elements.adminLink.hidden = !state.user.permissions.includes("user:read");
  elements.loginScreen.classList.add("hidden"); await loadAuthorizedPatients();
  if (!bootstrap.qwen_available) throw new Error("千问 API 尚未配置，请先设置 DASHSCOPE_API_KEY");
  await loadSessions(); if (!state.threadId) await createSession(); bindSuggestions();
}
function showLogin() {
  state.user = null; state.csrfToken = null; state.threadId = null; state.sessions = [];
  elements.loginScreen.classList.remove("hidden"); elements.username.focus();
}

elements.loginForm.addEventListener("submit", async event => {
  event.preventDefault(); elements.loginButton.disabled = true; elements.loginError.textContent = "";
  try {
    const payload = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ username: elements.username.value, password: elements.password.value }) });
    state.csrfToken = payload.csrf_token; elements.password.value = ""; await initializeAuthenticatedApp();
  } catch (error) { elements.loginError.textContent = error.message; }
  finally { elements.loginButton.disabled = false; }
});
elements.logout.addEventListener("click", async () => { try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } finally { showLogin(); } });
elements.form.addEventListener("submit", event => { event.preventDefault(); const message = elements.input.value; elements.input.value = ""; sendMessage(message); });
elements.input.addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.form.requestSubmit(); } });
elements.newSession.addEventListener("click", createSession);

initializeAuthenticatedApp().catch(error => { if (error.status !== 401) showError(error.message); showLogin(); });
