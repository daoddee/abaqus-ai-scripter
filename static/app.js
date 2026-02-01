let pythonVersion = "py3";
let activeTab = "scripter";

const appShell = document.getElementById("appShell");
const sidebarToggleTop = document.getElementById("sidebarToggleTop");
const sidebarCollapseBtn = document.getElementById("sidebarCollapseBtn");

const newProjectBtn = document.getElementById("newProjectBtn");
const uploadBtn = document.getElementById("uploadBtn");
const settingsBtn = document.getElementById("settingsBtn");
const bellBtn = document.getElementById("bellBtn");

const projectList = document.getElementById("projectList");
const taskList = document.getElementById("taskList");
const emptyProjects = document.getElementById("emptyProjects");
const emptyTasks = document.getElementById("emptyTasks");

const pyDropdownBtn = document.getElementById("pyDropdownBtn");
const pyDropdownMenu = document.getElementById("pyDropdownMenu");
const pyDropdownLabel = document.getElementById("pyDropdownLabel");

const tabs = document.querySelectorAll(".tab");
const panelScripter = document.getElementById("panel-scripter");
const panelDebugger = document.getElementById("panel-debugger");

const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");

const debugPrompt = document.getElementById("debugPrompt");
const debugCode = document.getElementById("debugCode");
const debugBtn = document.getElementById("debugBtn");

const resultBox = document.getElementById("resultBox");
const copyBtn = document.getElementById("copyBtn");
const clearBtn = document.getElementById("clearBtn");

const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");

const ragToggle = document.getElementById("ragToggle");
const ragPeekBtn = document.getElementById("ragPeekBtn");

/* ---------- Modal ---------- */
const modal = document.getElementById("modal");
const modalTitle = document.getElementById("modalTitle");
const modalBody = document.getElementById("modalBody");
const modalClose = document.getElementById("modalClose");
const modalOk = document.getElementById("modalOk");
const modalCancel = document.getElementById("modalCancel");
const modalInputWrap = document.getElementById("modalInputWrap");
const modalText = document.getElementById("modalText");

let modalMode = "ok"; // "ok" | "prompt"
let modalResolve = null;

function openModal(title, body, opts = {}) {
  modalTitle.textContent = title;
  modalBody.textContent = body || "";

  modalMode = opts.mode || "ok";
  modalInputWrap.classList.toggle("hidden", modalMode !== "prompt");

  if (modalMode === "prompt") {
    modalText.value = opts.defaultValue || "";
    modalText.placeholder = opts.placeholder || "Type here...";
    setTimeout(() => modalText.focus(), 50);
  }

  modal.classList.remove("hidden");
  return new Promise((resolve) => { modalResolve = resolve; });
}

function closeModal(result = null) {
  modal.classList.add("hidden");
  const r = modalResolve;
  modalResolve = null;
  if (r) r(result);
}

modalClose.addEventListener("click", () => closeModal(null));
modalCancel.addEventListener("click", () => closeModal(null));
modalOk.addEventListener("click", () => {
  if (modalMode === "prompt") closeModal(modalText.value.trim());
  else closeModal(true);
});
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(null); });

/* ---------- Output placeholder styling ---------- */
function setResult(text, isPlaceholder = false) {
  resultBox.textContent = text;
  resultBox.classList.toggle("placeholder", !!isPlaceholder);
}

setResult("(results will appear here)", true);

/* ---------- Sidebar toggle ---------- */
function toggleSidebar() {
  appShell.classList.toggle("sidebar-collapsed");
  localStorage.setItem("sidebarCollapsed", appShell.classList.contains("sidebar-collapsed") ? "1" : "0");
}

sidebarToggleTop.addEventListener("click", toggleSidebar);
sidebarCollapseBtn.addEventListener("click", toggleSidebar);

// restore collapsed state
if (localStorage.getItem("sidebarCollapsed") === "1") {
  appShell.classList.add("sidebar-collapsed");
}

/* ---------- Local projects/tasks storage ---------- */
function loadJSON(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || "") ?? fallback; }
  catch { return fallback; }
}
function saveJSON(key, value) { localStorage.setItem(key, JSON.stringify(value)); }

function renderProjects() {
  const projects = loadJSON("projects", []);
  projectList.innerHTML = "";
  if (!projects.length) {
    emptyProjects.classList.remove("hidden");
    return;
  }
  emptyProjects.classList.add("hidden");
  projects.forEach((p) => {
    const btn = document.createElement("button");
    btn.className = "project-item";
    btn.innerHTML = `<span class="dot"></span><span class="label">${escapeHtml(p.name)}</span>`;
    btn.addEventListener("click", () => openModal("Project", `Selected: ${p.name}`));
    projectList.appendChild(btn);
  });
}

function renderTasks() {
  const tasks = loadJSON("tasks", []);
  taskList.innerHTML = "";
  if (!tasks.length) {
    emptyTasks.classList.remove("hidden");
    return;
  }
  emptyTasks.classList.add("hidden");
  tasks.slice().reverse().forEach((t) => {
    const btn = document.createElement("button");
    btn.className = "task-item";
    btn.innerHTML = `<span class="dot"></span><span class="label">${escapeHtml(t.title)}</span>`;
    btn.addEventListener("click", () => openModal("Thread", `Thread: ${t.title}`));
    taskList.appendChild(btn);
  });
}

function ensureTaskFromPrompt(prompt) {
  const tasks = loadJSON("tasks", []);
  if (tasks.length) return; // only create first thread when user starts a conversation
  const title = titleFromPrompt(prompt);
  tasks.push({ id: Date.now(), title });
  saveJSON("tasks", tasks);
  renderTasks();
}

function titleFromPrompt(prompt) {
  const p = (prompt || "").trim().replace(/\s+/g, " ");
  if (!p) return "New chat";
  const words = p.split(" ").slice(0, 6).join(" ");
  return words.length < p.length ? words + "…" : words;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#39;");
}

renderProjects();
renderTasks();

/* ---------- Sidebar buttons ---------- */
newProjectBtn.addEventListener("click", async () => {
  const name = await openModal("New project", "Enter a project name:", {
    mode: "prompt",
    placeholder: "e.g., Cantilever beam study"
  });
  if (!name) return;

  const projects = loadJSON("projects", []);
  projects.push({ id: Date.now(), name });
  saveJSON("projects", projects);
  renderProjects();
});

uploadBtn.addEventListener("click", () => {
  fileInput.value = "";
  fileInput.click();
});

settingsBtn.addEventListener("click", () => openModal("Settings", "Settings UI coming next."));
bellBtn.addEventListener("click", () => openModal("Notifications", "No notifications yet."));

/* ---------- Dropdown ---------- */
pyDropdownBtn.addEventListener("click", () => pyDropdownMenu.classList.toggle("open"));
document.addEventListener("click", (e) => {
  if (!pyDropdownMenu.contains(e.target) && !pyDropdownBtn.contains(e.target)) {
    pyDropdownMenu.classList.remove("open");
  }
});
pyDropdownMenu.querySelectorAll(".dropdown-item").forEach((item) => {
  item.addEventListener("click", () => {
    pythonVersion = item.getAttribute("data-py");
    pyDropdownLabel.textContent = (pythonVersion === "py3") ? "Python 3 (newer Abaqus)" : "Python 2.7 (older Abaqus)";
    pyDropdownMenu.classList.remove("open");
  });
});

/* ---------- Tabs ---------- */
tabs.forEach((t) => {
  t.addEventListener("click", () => {
    tabs.forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    activeTab = t.getAttribute("data-tab");
    if (activeTab === "scripter") {
      panelScripter.classList.add("active");
      panelDebugger.classList.remove("active");
    } else {
      panelDebugger.classList.add("active");
      panelScripter.classList.remove("active");
    }
  });
});

/* ---------- Upload handling ---------- */
attachBtn.addEventListener("click", () => {
  fileInput.value = "";
  fileInput.click();
});

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  setResult("Uploading...", false);

  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("chat_id", "ui");

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!data.ok) {
      setResult("(results will appear here)", true);
      await openModal("Upload failed", (data.error || "") + "\n\n" + (data.details || ""));
      return;
    }
    const lines = data.files.map(f => `• ${f.original_name} (${f.size_bytes} bytes)`).join("\n");
    setResult("(results will appear here)", true);
    await openModal("Uploaded", "Saved files:\n" + lines);
  } catch (e) {
    setResult("(results will appear here)", true);
    await openModal("Upload error", String(e));
  }
});

/* ---------- RAG preview ---------- */
function currentQuery() {
  if (activeTab === "debugger") {
    const p = (debugPrompt.value || "").trim();
    const c = (debugCode.value || "").trim();
    return (p + "\n" + c).trim();
  }
  return (promptInput.value || "").trim();
}

ragPeekBtn.addEventListener("click", async () => {
  const q = currentQuery();
  if (!q) {
    await openModal("RAG", "Type a request first, then click “Show retrieved snippets”.");
    return;
  }
  try {
    const res = await fetch("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, k: 6 })
    });
    const data = await res.json();
    if (!data.ok) {
      await openModal("RAG retrieve failed", (data.error || "") + "\n\n" + (data.details || ""));
      return;
    }

    const body = (data.hits || []).map((h, i) => {
      const t = (h.text || "").slice(0, 900);
      return `[${i+1}] score=${(h.score||0).toFixed(3)} source=${h.source} chunk=${h.chunk_index}\n${t}${(h.text||"").length>900 ? "\n...[truncated]" : ""}`;
    }).join("\n\n----------------------------------------\n\n");

    await openModal("Retrieved snippets", body || "(no hits)");
  } catch (e) {
    await openModal("RAG error", String(e));
  }
});

/* ---------- Generate ---------- */
async function callGenerate(payload) {
  setResult("Thinking...", false);
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) {
      setResult(`Error: ${data.error}\n\n${data.details || ""}`, false);
      return;
    }
    setResult(data.result, false);

    if (payload.use_rag && data.rag_hits && data.rag_hits.length) {
      const short = data.rag_hits.map((h, i) => {
        const t = (h.text || "").slice(0, 260).replace(/\s+/g, " ");
        return `[${i+1}] ${h.source}  score=${(h.score||0).toFixed(3)}\n${t}${(h.text||"").length>260 ? " ..." : ""}`;
      }).join("\n\n");
      await openModal("RAG used (top matches)", short);
    }
  } catch (err) {
    setResult(`Network error: ${String(err)}`, false);
  }
}

sendBtn.addEventListener("click", async () => {
  const prompt = (promptInput.value || "").trim();
  if (!prompt) { setResult("Please type a prompt first.", false); return; }

  ensureTaskFromPrompt(prompt);

  await callGenerate({
    mode: "scripter",
    python_version: pythonVersion,
    prompt,
    use_rag: !!ragToggle.checked,
    rag_k: 6
  });
});

debugBtn.addEventListener("click", async () => {
  const prompt = (debugPrompt.value || "").trim();
  const code = (debugCode.value || "").trim();
  if (!prompt && !code) { setResult("Paste an error description and/or code to analyze.", false); return; }

  ensureTaskFromPrompt(prompt || code);

  await callGenerate({
    mode: "debugger",
    python_version: pythonVersion,
    prompt: prompt || "(no description provided)",
    code,
    use_rag: !!ragToggle.checked,
    rag_k: 6
  });
});

/* ---------- Copy/Clear ---------- */
copyBtn.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(resultBox.textContent || ""); await openModal("Copied", "Output copied."); }
  catch { await openModal("Copy failed", "Clipboard blocked."); }
});
clearBtn.addEventListener("click", () => setResult("(results will appear here)", true));
