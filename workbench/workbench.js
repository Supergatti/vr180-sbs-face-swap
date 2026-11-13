"use strict";
const $ = id => document.getElementById(id);
const isTest = location.pathname === "/test";
const jobKey = isTest ? "vr180-test-job" : "vr180-job";
if (isTest) {
  document.title = "双目取景测试 · 视差";
  $("pageName").textContent = "测试设置";
  $("viewerTitle").textContent = "双目取景测试";
  $("pageLabel").textContent = "PIPELINE TEST";
  $("taskNav").removeAttribute("aria-current");
  $("testNav").setAttribute("aria-current", "page");
  $("formalActions").hidden = true;
  $("testActions").hidden = false;
}
const state = {video: "", info: null, frame: 0, faces: [[], []], selected: [null, null], manual: false, busy: false, job: null};
let drag, previewSeq = 0, sourceURL, testMediaKey;
const running = () => state.job?.state === "running";
const number = id => Number($(id).value);
const hasPair = () => state.selected.every(Boolean);
const bounds = () => ({start: number("start"), end: number("end")});
const videoReady = () => Boolean(state.info) && $("videoPath").value.trim() === state.video;
const testFrozen = () => isTest && !!state.job;
function selectionRequirement() {
  if (!videoReady()) return ["请先载入目标视频", $("videoPath").value.trim() ? "load" : "videoPath", "去载入"];
  const {start, end} = bounds();
  if (!["start", "end"].every(id => $(id).value !== "") || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end >= state.info.count) return ["片段起止帧无效", "start", "调整片段"];
  if (state.frame < start || state.frame > end) return ["参考帧不在所选片段内", "frame", "调整参考帧"];
  if (!hasPair()) {
    const missing = state.selected[0] ? "右眼" : state.selected[1] ? "左眼" : "左右眼";
    const detected = state.faces.some(faces => faces.length);
    const candidates = state.faces.some((faces, eye) => !state.selected[eye] && faces.length);
    return [missing + "目标未选择", candidates ? "stage" : detected ? "draw" : "detect", candidates ? "去框选" : detected ? "去画框" : "去检测"];
  }
  return null;
}
function renderRequirement() {
  if (running()) return ["当前任务处理中"];
  if (state.busy) return ["正在处理请求…"];
  if (videoReady() && !$("facePath").value.trim()) return ["请选择来源人脸", "facePath", "去选择"];
  return selectionRequirement();
}
function notice(text = "", error = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("error", error);
}
async function request(url, data, raw = false) {
  const response = await fetch(url, data === undefined ? {} : {
    method: "POST", headers: {"Content-Type": raw ? "application/octet-stream" : "application/json"},
    body: raw ? data : JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}
function controls() {
  const locked = state.busy || running();
  const frozen = testFrozen();
  const ready = videoReady() && !frozen;
  const requirement = renderRequirement();
  const enabled = {load: $("videoPath").value.trim(), detect: ready, draw: ready, clear: state.selected.some(Boolean),
    render: !requirement, seek: ready, frame: ready};
  for (const [id, allowed] of Object.entries(enabled)) $(id).disabled = locked || !allowed;
  $("load").disabled ||= frozen;
  $("clear").disabled ||= frozen;
  $("renderHint").textContent = requirement?.[0] || "固定双眼取景";
  $("renderHelp").hidden = !requirement?.[1];
  $("renderHelp").textContent = requirement?.[2] || "";
  $("perspTab").disabled = !hasPair() || locked;
  for (const id of ["videoPath", "videoFile", "padding", "start", "end"]) $(id).disabled = locked || frozen;
  for (const id of ["facePath", "faceFile", "provider", "enhance"]) $(id).disabled = locked;
  $("draw").setAttribute("aria-pressed", state.manual);
  $("overlay").classList.toggle("drawing", state.manual && !locked && !frozen);
  $("leftState").textContent = state.selected[0] ? "已选择" : "未选择";
  $("rightState").textContent = state.selected[1] ? "已选择" : "未选择";
  if (state.info) {
    const {start, end} = bounds();
    $("clipInfo").textContent = end >= start ? (end - start + 1) + " 帧 · " + ((end - start + 1) / state.info.fps).toFixed(2) + " 秒" : "无效片段";
  }
  for (const btn of $("boxes").querySelectorAll("button")) btn.disabled = locked || frozen;
  if (isTest) {
    const job = state.job;
    const issue = selectionRequirement();
    $("testRead").disabled = locked || frozen || !$("videoPath").value.trim();
    $("testExport").disabled = locked || frozen || !!issue;
    $("testSwap").disabled = locked || job?.state !== "paused" || job?.stage !== "swap" || !$("facePath").value.trim();
    $("testRefill").disabled = locked || job?.state !== "paused" || job?.stage !== "refill";
    $("resetTest").disabled = locked || !job;
    $("testHint").textContent = locked ? "当前步骤处理中" : !job ? issue?.[0] || "可以导出左右眼透视片段" :
      job.state === "paused" ? (job.stage === "swap" ? ($("facePath").value.trim() ? "透视导出完成 · 可以测试换脸" : "透视导出完成 · 请选择来源人脸") : "换脸完成 · 可以测试回填") :
      job.state === "error" ? "测试失败 · 可重新测试" : "全部测试完成";
  }
}
function show(tab) {
  $("erp").hidden = tab !== "erp" || !state.info;
  $("crops").hidden = tab !== "crops";
  $("empty").hidden = !!state.info;
  $("erpTab").setAttribute("aria-pressed", tab === "erp");
  $("perspTab").setAttribute("aria-pressed", tab === "crops");
}
async function busy(fn) {
  if (state.busy || running()) return;
  state.busy = true;
  controls();
  try { await fn(); }
  catch (error) { notice(error.message, true); }
  finally { state.busy = false; controls(); }
}
function addBox(item, eye, picked = false, draft = false) {
  const [x, y, w, h] = item.box;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "face-box" + (picked ? " selected" : "");
  Object.assign(button.style, {left: (eye + x) * 50 + "%", top: y * 100 + "%", width: w * 50 + "%", height: h * 100 + "%"});
  const text = item.score == null ? "自选" : Math.round(item.score * 100) + "%";
  button.setAttribute("aria-label", (eye ? "右眼" : "左眼") + "人脸 " + text);
  button.setAttribute("aria-pressed", String(picked));
  if (draft) button.style.pointerEvents = "none";
  const tag = document.createElement("span");
  tag.textContent = text;
  button.append(tag);
  button.addEventListener("click", () => {
    if (state.manual || state.busy || running()) return;
    state.selected[eye] = item;
    refreshBoxes();
    notice(hasPair() ? "已选双眼目标 · 可查看透视裁片" : "请选择另一只眼中的同一人物");
  });
  $("boxes").append(button);
}
function refreshBoxes(draft) {
  $("boxes").replaceChildren();
  state.faces.forEach((faces, eye) => {
    faces.filter(item => item !== state.selected[eye]).forEach(item => addBox(item, eye));
    if (state.selected[eye]) addBox(state.selected[eye], eye, true);
  });
  if (draft) addBox({box: draft.box}, draft.eye, true, true);
  controls();
}
function resetSelection() {
  previewSeq++;
  state.faces = [[], []];
  state.selected = [null, null];
  state.manual = false;
  refreshBoxes();
  show("erp");
}
async function inspect(path, frame, detect = false, resetClip = false) {
  const result = await request("/api/inspect", {video: path, frame, detect, provider: $("provider").value});
  const changed = path !== state.video || frame !== state.frame;
  state.video = path;
  state.info = result.info;
  state.frame = result.frame;
  if (changed || resetClip) resetSelection();
  $("frameImage").src = result.image;
  for (const id of ["seek", "frame", "start", "end"]) $(id).max = result.info.count - 1;
  $("seek").value = $("frame").value = result.frame;
  if (resetClip) {
    $("start").value = 0;
    $("end").value = result.info.count - 1;
  }
  $("videoInfo").textContent = result.info.width + " × " + result.info.height + " · " + result.info.fps.toFixed(2) + " FPS";
  if (detect) {
    state.selected = [null, null];
    state.faces = result.faces;
    state.manual = false;
    refreshBoxes();
    notice("左眼 " + result.faces[0].length + " 个 · 右眼 " + result.faces[1].length + " 个" +
      (result.faces.every(a => a.length) ? " · 请选择同一人物" : " · 可手动画框补选"));
  } else notice("");
  show("erp");
}
$("load").onclick = () => busy(() => inspect($("videoPath").value.trim(), 0, false, true));
$("videoPath").addEventListener("keydown", e => { if (e.key === "Enter") $("load").click(); });
$("detect").onclick = () => busy(async () => { notice("正在检测左右眼…"); await inspect(state.video, state.frame, true); });
$("clear").onclick = () => { state.selected = [null, null]; refreshBoxes(); show("erp"); notice(""); };
$("draw").onclick = () => { state.manual = !state.manual; show("erp"); controls(); };
$("erpTab").onclick = () => show("erp");
$("perspTab").onclick = () => busy(async () => {
  const token = ++previewSeq;
  const result = await request("/api/preview", {video: state.video, frame: state.frame, boxes: state.selected.map(a => a.box), padding: number("padding")});
  if (token !== previewSeq) return;
  $("cropL").src = result.images[0];
  $("cropR").src = result.images[1];
  show("crops");
  notice("固定视场 " + result.h_fov.toFixed(1) + "° × " + result.v_fov.toFixed(1) + "°");
});
$("seek").oninput = () => { $("frame").value = $("seek").value; };
for (const id of ["seek", "frame"]) $(id).onchange = () => busy(() => inspect(state.video, number(id)));
for (const id of ["videoPath", "facePath", "start", "end"]) $(id).addEventListener("input", controls);
$("padding").oninput = () => { $("padValue").textContent = number("padding").toFixed(1) + "×"; if (!$("crops").hidden) show("erp"); };

function point(event, eye) {
  const rect = $("overlay").getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width * 2 - eye;
  const y = (event.clientY - rect.top) / rect.height;
  return [Math.max(0, Math.min(x, 1)), Math.max(0, Math.min(y, 1))];
}
function rectangle(a, b) { return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])]; }
$("overlay").onpointerdown = event => {
  if (!state.manual || state.busy || running() || testFrozen() || event.button !== 0) return;
  const rect = $("overlay").getBoundingClientRect();
  const eye = event.clientX - rect.left < rect.width / 2 ? 0 : 1;
  drag = {eye, start: point(event, eye), pointer: event.pointerId};
  $("overlay").setPointerCapture(event.pointerId);
};
$("overlay").onpointermove = event => {
  if (drag?.pointer !== event.pointerId) return;
  refreshBoxes({eye: drag.eye, box: rectangle(drag.start, point(event, drag.eye))});
};
$("overlay").onpointerup = event => {
  if (drag?.pointer !== event.pointerId) return;
  const {eye, start} = drag;
  drag = null;
  const box = rectangle(start, point(event, eye));
  if (box[2] >= .008 && box[3] >= .008) {
    state.selected[eye] = {box};
    if (hasPair()) state.manual = false;
    notice(hasPair() ? "已选双眼目标 · 可查看透视裁片" : "请框选另一只眼中的同一人物");
  }
  refreshBoxes();
};
$("overlay").onpointercancel = () => { drag = null; refreshBoxes(); };

for (const kind of ["video", "face"]) $(kind + "File").onchange = e => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = "";
  busy(async () => {
    notice("正在导入 " + file.name + "…");
    const result = await request("/api/upload?kind=" + kind + "&name=" + encodeURIComponent(file.name), file, true);
    $(kind + "Path").value = result.path;
    if (kind === "video") await inspect(result.path, 0, false, true);
    else {
      if (sourceURL) URL.revokeObjectURL(sourceURL);
      sourceURL = URL.createObjectURL(file);
      $("sourceImage").src = sourceURL;
      $("sourceName").textContent = file.name;
      $("sourceThumb").hidden = false;
      notice("");
    }
  });
};
$("facePath").addEventListener("input", () => { $("sourceThumb").hidden = true; });
const labels = {export: "透视导出中", swap: "FaceFusion 处理中", refill: "球面回填中", done: "已完成"};
const artifactLabels = {"left_flat.mp4": "左眼透视", "right_flat.mp4": "右眼透视", "left_swapped.mp4": "左眼换脸", "right_swapped.mp4": "右眼换脸"};
const phases = {prepare: "正在准备", project: "透视导出", analyse: "检查输入视频", extract: "FaceFusion 抽帧", face_swapper: "换脸", face_enhancer: "人脸增强", merge: "FaceFusion 视频编码", audio: "FaceFusion 输出整理", composite: "球面回填", finalize: "等待编码完成"};
const eyeName = {left: "左眼（1/2）", right: "右眼（2/2）"};
const stageNames = {export: "透视导出", swap: "FaceFusion", refill: "球面回填"};
let previewRequest;
function updateDetails(job, index) {
  $("jobDetails").hidden = false;
  const p = job.progress || {};
  const bar = $("frameProgress");
  const label = job.state === "paused" ? "已完成，等待下一步" : job.state === "done" ? "全部完成" :
    (job.state === "error" ? "失败 · " : "步骤 " + (index + 1) + "/3 · ") + (eyeName[p.eye] ? eyeName[p.eye] + " · " : "") + (phases[p.phase] || stageNames[job.stage]);
  $("progressLabel").textContent = label;
  bar.hidden = job.state === "error" && !p.total;
  if (job.state === "done" || job.state === "paused") {
    bar.value = 100;
    $("frameCount").textContent = "100%";
  } else if (p.total) {
    bar.value = p.completed / p.total * 100;
    $("frameCount").textContent = p.completed + " / " + p.total + " 帧 · " + Math.floor(bar.value) + "%";
  } else {
    bar.removeAttribute("value");
    $("frameCount").textContent = "";
  }
  $("jobRange").textContent = "区间 " + (job.range?.join("–") || "—") + " · " + job.frames + " 帧";
  $("jobProvider").textContent = "推理设置 " + (job.provider || "—").toUpperCase();
  $("jobElapsed").textContent = "处理用时 " + (job.elapsed || 0).toFixed(1) + " 秒";
  const fps = job.frames && job.elapsed ? job.frames / job.elapsed : 0;
  $("jobSpeed").textContent = fps >= .5 ? "处理速度 " + fps.toFixed(1) + " 帧/秒" : fps > 0 ? "处理速度 " + (1 / fps).toFixed(2) + " 秒/帧" : "";
  const lines = job.logs || [];
  $("jobLog").hidden = !lines.length;
  const log = $("logLines");
  const text = lines.map(line => new Date(line.time * 1000).toLocaleTimeString() + "  " + line.message).join("\n");
  if (log.textContent !== text) {
    const follow = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
    log.textContent = text;
    if (follow) log.scrollTop = log.scrollHeight;
  }
  if (job.preview && job.preview.url !== previewRequest) {
    const {url, phase, stage, eye, frame} = job.preview;
    previewRequest = url;
    const next = new Image();
    next.onload = () => {
      if (state.job?.id !== job.id || previewRequest !== url) return;
      $("liveImage").src = url;
      $("liveCaption").textContent = ["第 " + frame + " 帧", phases[phase] || stageNames[stage], eyeName[eye] || (stage === "export" ? "双眼裁片" : "SBS")].join(" · ");
      $("livePreview").hidden = false;
    };
    next.onerror = () => { if (previewRequest === url) previewRequest = undefined; };
    next.src = url;
  }
}
function link(label, url, download = true) {
  const a = document.createElement("a");
  a.textContent = label;
  a.href = url + (download ? "?download=1" : "");
  if (download) a.download = "";
  $("links").append(a);
}
function updateJob(job) {
  if (state.job?.id !== job.id) {
    $("livePreview").hidden = true;
    previewRequest = undefined;
  }
  state.job = job;
  $("jobState").textContent = job.state === "error" ? "处理失败" : job.state === "paused" ? "等待下一步" : labels[job.stage];
  const index = ["export", "swap", "refill", "done"].indexOf(job.stage);
  updateDetails(job, index);
  [...$("steps").children].forEach((li, i) => {
    li.classList.toggle("active", i === index && job.state === "running");
    li.classList.toggle("done", i < index);
    const step = ["export", "swap", "refill"][i];
    const label = ["透视导出", job.enhance ? "FaceFusion 换脸与增强" : "FaceFusion 换脸", "球面回填"][i];
    li.textContent = job.times?.[step] != null ? label + " · " + job.times[step].toFixed(1) + " 秒" : label;
  });
  $("links").replaceChildren();
  $("results").hidden = false;
  $("output").hidden = !job.output;
  link("带框 PNG", job.monitor);
  if (job.output) {
    if ($("output").getAttribute("src") !== job.output) $("output").src = job.output;
    link("下载 SBS 视频", job.output);
    const elapsed = isTest ? Object.values(job.times).reduce((sum, time) => sum + time, 0) : job.finished - job.started;
    notice("已输出 " + job.frames + " 帧 · " + elapsed.toFixed(1) + " 秒");
  } else if (job.error) notice(job.error, true);
  else notice("");
  if (isTest) {
    const files = Object.entries(job.files || {}).filter(([name]) => name in artifactLabels);
    for (const [name, url] of files) link(artifactLabels[name], url);
    if (job.files?.["selection.json"]) link("框选记录", job.files["selection.json"]);
    const key = files.map(([, url]) => url).join("|");
    if (key !== testMediaKey) {
      testMediaKey = key;
      $("testMedia").replaceChildren(...files.map(([name, url]) => {
        const details = document.createElement("details");
        const title = document.createElement("summary");
        title.textContent = artifactLabels[name];
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "none";
        video.src = url;
        details.append(title, video);
        return details;
      }));
      $("testMedia").hidden = !files.length;
    }
  }
  controls();
}
async function poll(id) {
  if (state.job?.id !== id) return;
  try {
    const job = await request("/api/jobs/" + id);
    updateJob(job);
    if (job.state === "running") setTimeout(() => poll(id), 750);
  } catch (error) {
    notice(error.message + " · 正在重试", true);
    setTimeout(() => poll(id), 3000);
  }
}
function submitJob(url, data) {
  return busy(async () => {
  notice("正在创建任务…");
  const job = await request(url, data);
  updateJob(job);
  sessionStorage.setItem(jobKey, job.id);
  if (job.state === "running") setTimeout(() => poll(job.id), 1000);
  else if (job.state === "paused") notice("");
  });
}
const swapOptions = () => ({face: $("facePath").value.trim(), provider: $("provider").value, enhance: $("enhance").checked});
const jobOptions = () => ({video: state.video, frame: state.frame, ...bounds(), boxes: state.selected.map(a => a.box), padding: number("padding"), ...swapOptions()});
$("render").onclick = () => submitJob("/api/jobs", jobOptions());
$("renderHelp").onclick = () => {
  const target = renderRequirement()?.[1];
  if (!target) return;
  if (["stage", "detect", "draw", "frame"].includes(target)) show("erp");
  $(target).scrollIntoView({block: "center"});
  if (target === "stage") $("boxes").querySelector("button:not(.selected)")?.focus({preventScroll: true});
  else $(target).focus({preventScroll: true});
};
$("testRead").onclick = () => videoReady() ? busy(() => inspect(state.video, state.frame)) : $("load").click();
$("testExport").onclick = () => submitJob("/api/tests", jobOptions());
for (const stage of ["swap", "refill"]) $(stage === "swap" ? "testSwap" : "testRefill").onclick = () => {
  submitJob("/api/jobs/" + state.job.id, {stage, ...swapOptions()});
};
$("resetTest").onclick = () => {
  state.job = null;
  sessionStorage.removeItem(jobKey);
  $("results").hidden = true;
  $("jobDetails").hidden = true;
  $("livePreview").hidden = true;
  previewRequest = undefined;
  $("output").removeAttribute("src");
  $("output").load();
  $("testMedia").replaceChildren();
  $("testMedia").hidden = true;
  testMediaKey = undefined;
  $("jobState").textContent = "等待任务";
  [...$("steps").children].forEach((li, i) => { li.className = ""; li.textContent = ["透视导出", "FaceFusion 换脸", "球面回填"][i]; });
  notice("");
  controls();
};
request("/api/health").then(result => { $("health").textContent = result.model ? "本地模型就绪" : "检测权重未安装"; })
  .catch(() => { $("health").textContent = "服务未连接"; });
const lastJob = sessionStorage.getItem(jobKey);
if (lastJob) busy(async () => {
  try {
    const job = await request("/api/jobs/" + lastJob);
    updateJob(job);
    if (job.state === "running") poll(lastJob);
  } catch {
    sessionStorage.removeItem(jobKey);
  }
});
controls();
