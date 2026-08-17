const adminState = { user: null, csrfToken: null, users: [], patients: [], patientTarget: null };
const el = {
  name: document.querySelector("#admin-name"), agentLink: document.querySelector("#agent-link"),
  logout: document.querySelector("#admin-logout"), createForm: document.querySelector("#create-user-form"),
  userRows: document.querySelector("#user-rows"), auditRows: document.querySelector("#audit-rows"),
  refreshUsers: document.querySelector("#refresh-users"), refreshAudit: document.querySelector("#refresh-audit"),
  dialog: document.querySelector("#patient-dialog"), dialogTitle: document.querySelector("#patient-dialog-title"),
  patientOptions: document.querySelector("#patient-options"), savePatients: document.querySelector("#save-patients"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && adminState.csrfToken) headers["X-CSRF-Token"] = adminState.csrfToken;
  const response = await fetch(path, { credentials: "same-origin", ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401) { window.location.replace("/"); throw new Error("登录已失效"); }
  if (!response.ok) throw new Error(payload.detail || "请求失败");
  return payload;
}
function escapeHtml(value) { const node = document.createElement("div"); node.textContent = value ?? ""; return node.innerHTML; }
function toast(message, error = false) {
  el.toast.textContent = message; el.toast.style.background = error ? "#7f3131" : "#1f6b59"; el.toast.classList.add("show");
  setTimeout(() => el.toast.classList.remove("show"), 3000);
}
function roleLabel(role) { return role === "administrator" ? "管理员" : "医生"; }

async function loadUsers() {
  adminState.users = (await api("/api/admin/users")).users;
  el.userRows.innerHTML = adminState.users.map(user => `
    <tr data-user="${escapeHtml(user.user_id)}">
      <td class="account-name"><strong>${escapeHtml(user.username)}</strong><span>${escapeHtml(user.user_id)}</span></td>
      <td class="account-name"><strong>${escapeHtml(user.display_name)}</strong><span>${escapeHtml(user.department)}</span></td>
      <td><span class="${user.is_active ? "status-active" : "status-disabled"}">${user.is_active ? "已启用" : "已停用"}</span><br><button class="secondary status-button" data-active="${user.is_active}">${user.is_active ? "停用" : "启用"}</button></td>
      <td><div class="role-controls">
        <label><input type="checkbox" value="doctor" ${user.roles.includes("doctor") ? "checked" : ""}> 医生</label>
        <label><input type="checkbox" value="administrator" ${user.roles.includes("administrator") ? "checked" : ""}> 管理员</label>
        <button class="save-roles">保存角色</button>
      </div></td>
      <td>${user.roles.includes("doctor") ? '<button class="secondary patient-access">分配患者</button>' : '<span class="status-disabled">非医生角色</span>'}</td>
    </tr>`).join("");
  bindUserActions();
}

function bindUserActions() {
  el.userRows.querySelectorAll("tr[data-user]").forEach(row => {
    const userId = row.dataset.user;
    row.querySelector(".status-button").addEventListener("click", async event => {
      const isActive = event.currentTarget.dataset.active === "true";
      try {
        await api(`/api/admin/users/${encodeURIComponent(userId)}/status`, { method: "PATCH", body: JSON.stringify({ is_active: !isActive }) });
        toast("账号状态已更新"); await loadUsers(); await loadAudit();
      } catch (error) { toast(error.message, true); }
    });
    row.querySelector(".save-roles").addEventListener("click", async () => {
      const roles = [...row.querySelectorAll('.role-controls input:checked')].map(input => input.value);
      try {
        await api(`/api/admin/users/${encodeURIComponent(userId)}/roles`, { method: "PUT", body: JSON.stringify({ roles }) });
        toast("角色已更新，该账号旧会话已撤销"); await loadUsers(); await loadAudit();
      } catch (error) { toast(error.message, true); }
    });
    row.querySelector(".patient-access")?.addEventListener("click", () => openPatientAccess(userId));
  });
}

async function openPatientAccess(userId) {
  try {
    if (!adminState.patients.length) adminState.patients = (await api("/api/admin/patients")).patients;
    const selected = new Set((await api(`/api/admin/users/${encodeURIComponent(userId)}/patient-access`)).patient_ids);
    const user = adminState.users.find(item => item.user_id === userId);
    adminState.patientTarget = userId; el.dialogTitle.textContent = `${user.display_name}的患者范围`;
    el.patientOptions.innerHTML = adminState.patients.map(patient => `<label><input type="checkbox" value="${escapeHtml(patient.patient_id)}" ${selected.has(patient.patient_id) ? "checked" : ""}> ${escapeHtml(patient.patient_id)} · ${escapeHtml(patient.display_name)}</label>`).join("");
    el.dialog.showModal();
  } catch (error) { toast(error.message, true); }
}

async function loadAudit() {
  const events = (await api("/api/admin/audit?limit=100")).events;
  el.auditRows.innerHTML = events.map(event => `<tr><td>${escapeHtml(new Date(event.occurred_at).toLocaleString())}</td><td>${escapeHtml(event.actor_user_id || "未认证")}</td><td>${escapeHtml(event.event_type)}</td><td>${escapeHtml(event.outcome)}</td><td>${escapeHtml([event.target_type, event.target_id].filter(Boolean).join(": "))}</td></tr>`).join("");
}

el.createForm.addEventListener("submit", async event => {
  event.preventDefault();
  const payload = {
    username: document.querySelector("#new-username").value,
    display_name: document.querySelector("#new-display-name").value,
    department: document.querySelector("#new-department").value,
    password: document.querySelector("#new-password").value,
    roles: [document.querySelector("#new-role").value],
  };
  try {
    await api("/api/admin/users", { method: "POST", body: JSON.stringify(payload) });
    el.createForm.reset(); toast("账号已创建"); await loadUsers(); await loadAudit();
  } catch (error) { toast(error.message, true); }
});
el.savePatients.addEventListener("click", async () => {
  const patientIds = [...el.patientOptions.querySelectorAll("input:checked")].map(input => input.value);
  try {
    await api(`/api/admin/users/${encodeURIComponent(adminState.patientTarget)}/patient-access`, { method: "PUT", body: JSON.stringify({ patient_ids: patientIds }) });
    el.dialog.close(); toast("患者范围已更新"); await loadAudit();
  } catch (error) { toast(error.message, true); }
});
el.refreshUsers.addEventListener("click", () => loadUsers().catch(error => toast(error.message, true)));
el.refreshAudit.addEventListener("click", () => loadAudit().catch(error => toast(error.message, true)));
el.logout.addEventListener("click", async () => { try { await api("/api/auth/logout", { method: "POST", body: "{}" }); } finally { window.location.replace("/"); } });

async function initialize() {
  const bootstrap = await api("/api/bootstrap"); adminState.user = bootstrap.user; adminState.csrfToken = bootstrap.csrf_token;
  if (!adminState.user.permissions.includes("user:read")) { window.location.replace("/"); return; }
  el.name.textContent = `${adminState.user.name} · ${adminState.user.department}`;
  el.agentLink.hidden = !adminState.user.permissions.includes("agent:use");
  await Promise.all([loadUsers(), loadAudit()]);
}
initialize().catch(error => toast(error.message, true));
