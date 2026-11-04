'use strict';

/* ============================================================ helpers */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const COLORS = ['#4f46e5', '#0ea5e9', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#84cc16', '#f97316'];
const STAGES = [
  ['extract', '提取音轨', 'ffmpeg 解码视频音轨'],
  ['separate', '人声分离', 'Demucs htdemucs 去除背景音乐/音效'],
  ['diarize', '说话人日志', 'pyannote-3.0 分割 + CAM++ 声纹'],
  ['refine', '重叠剔除 / 声纹提纯', '去掉多人同时说话，滑窗声纹校验'],
  ['asr', '语音识别', 'SenseVoice 转写每个片段'],
  ['assets', '试听与缩略图', '生成试听片段、视频截图'],
];

function fmt(sec) {
  sec = Math.max(0, sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
  const mm = String(m).padStart(h ? 2 : 1, '0'), ss = String(s).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function fmtDur(sec) {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? `${h} 小时 ` : '') + (h || m ? `${m} 分 ` : '') + `${s} 秒`;
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* empty body */ }
  if (!r.ok) throw new Error((body && body.detail) || `${r.status} ${r.statusText}`);
  return body;
}
const postJSON = (path, data) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast ${kind}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add('hidden'), 4200);
}

// small per-viewer conveniences (custom names, selection, excluded segments)
const local = {
  get(key, def) { try { const v = JSON.parse(localStorage.getItem(key)); return v ?? def; } catch { return def; } },
  set(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* private mode */ } },
};

/* ============================================================ state */
const state = {
  jobs: [],
  jobId: null,
  job: null,
  result: null,
  selected: new Set(),
  names: {},
  excluded: new Set(),
  match: null, // {kind, label, scores:[{id, score, ...}]}
  pollTimer: null,
  exportTimer: null,
  exportKey: null,
  drawerSpk: null,
  drawerSort: 'time',
  ex: { denoise: true, denoise_db: 15, loudnorm: true, min_sim: 0, min_segment: 1.0, gap: 0.5, slices: false },
};
const main = () => $('#main');
const spkIndex = (id) => (state.result ? state.result.speakers.findIndex((s) => s.id === id) : 0);
const spkColor = (id) => COLORS[Math.max(0, spkIndex(id)) % COLORS.length];
const spkName = (id) => state.names[id] || (state.result?.speakers.find((s) => s.id === id)?.label ?? id);
const fileUrl = (path) => `/api/jobs/${encodeURIComponent(state.jobId)}/files/${path}`;
// analysis outputs keep their names across re-analysis: version them by finish time
const assetUrl = (path) => `${fileUrl(path)}?v=${Math.round(state.job?.finished || state.job?.created || 0)}`;
const chip = (id) => `<span class="chip" style="--c:${spkColor(id)}">${esc(spkName(id))}</span>`;
function saveLocal() {
  local.set(`av:names:${state.jobId}`, state.names);
  local.set(`av:sel:${state.jobId}`, [...state.selected]);
  local.set(`av:excl:${state.jobId}`, [...state.excluded]);
}

/* ============================================================ player */
const player = {
  el: null, end: null, tag: null, raf: 0,
  init() {
    this.el = $('#player');
    this.el.addEventListener('pause', () => this.markUI());
    // rAF is throttled in background tabs; timeupdate is the fallback stop check
    this.el.addEventListener('timeupdate', () => { if (this.end != null && this.el.currentTime >= this.end) this.stop(); });
    this.el.addEventListener('play', () => { pauseOthers(this.el); this.markUI(); });
  },
  toggleRange(src, start, end, tag) {
    if (this.tag === tag && !this.el.paused) { this.stop(); return; }
    const a = this.el;
    const go = () => { a.currentTime = start; a.play().catch(() => {}); this.loop(); };
    this.end = end; this.tag = tag;
    if (a.dataset.src !== src) {
      a.dataset.src = src;
      a.src = src;
      a.addEventListener('loadedmetadata', go, { once: true });
    } else go();
    this.markUI();
  },
  stop() { this.el.pause(); this.end = null; this.tag = null; this.markUI(); drawTimeline(); },
  loop() {
    cancelAnimationFrame(this.raf);
    const step = () => {
      if (this.end != null && this.el.currentTime >= this.end) { this.stop(); return; }
      drawTimeline();
      if (!this.el.paused) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  },
  markUI() {
    const tag = this.el.paused ? null : this.tag;
    $$('[data-play]').forEach((el) => el.classList.toggle('playing', el.dataset.play === tag));
    $$('[data-play] .playbtn').forEach((b) => { b.textContent = b.closest('[data-play]').dataset.play === tag ? '❚❚' : '▶'; });
  },
};
function pauseOthers(except) { $$('audio').forEach((a) => { if (a !== except && !a.paused) a.pause(); }); }
document.addEventListener('play', (e) => { if (e.target.tagName === 'AUDIO' && e.target !== player.el) { player.el.pause(); pauseOthers(e.target); } }, true);
function playSegment(seg) {
  player.toggleRange(assetUrl('vocals_browse.mp3'), Math.max(0, seg.start - 0.05), seg.end + 0.05, `seg-${seg.id}`);
}

/* ============================================================ routing */
function route() {
  clearTimeout(state.pollTimer);
  clearTimeout(state.exportTimer);
  closeDrawer();
  player.stop();
  const m = location.hash.match(/^#\/job\/(.+)$/);
  if (m) openJob(decodeURIComponent(m[1]));
  else renderNew();
  renderJobList();
}

async function refreshJobs() {
  try { state.jobs = await api('/api/jobs'); } catch { /* server down: keep old list */ }
  renderJobList();
}
function renderJobList() {
  const statusText = { queued: '排队中', running: '处理中', done: '已完成', error: '失败' };
  $('#job-list').innerHTML = state.jobs.length ? state.jobs.map((j) => `
    <li><a href="#/job/${encodeURIComponent(j.id)}" class="${j.id === state.jobId && location.hash.startsWith('#/job/') ? 'active' : ''}">
      <span class="jn" title="${esc(j.name)}">${esc(j.name)}</span>
      <span class="js"><i class="dotstat ${j.status}"></i>${statusText[j.status] || j.status}${j.status === 'running' ? ` · ${Math.round((j.progress || 0) * 100)}%` : ''}
      · ${new Date(j.created * 1000).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
    </a></li>`).join('') : '<li class="muted" style="padding:6px 10px">暂无任务</li>';
}

/* ============================================================ new job */
function renderNew() {
  state.jobId = null;
  $('#export-bar').classList.add('hidden');
  main().innerHTML = `
    <div class="hero">
      <h2>从多人视频中提取目标说话人的纯净语音</h2>
      <p>上传播客、访谈或会议视频。系统会分离人声、识别每位说话人、剔除重叠语音，
        你只需试听确认目标人物（或用台词 / 参考音频 / 照片自动匹配），即可一键导出适合 TTS / 声音克隆训练的 MP3。</p>
    </div>
    <section class="card">
      <label class="dropzone" id="dz">
        <span class="big">🎬</span>
        <strong>拖入视频或音频文件，或点击选择</strong>
        <span class="muted">支持 mp4 / mkv / mov / webm / mp3 / wav / m4a 等 ffmpeg 可解码的格式</span>
        <span class="file" id="dz-file"></span>
        <input type="file" id="file" accept="video/*,audio/*,.mkv,.flv,.ts" hidden>
      </label>
      <div class="opts-row">
        <label><input type="checkbox" id="opt-sep" checked> 人声分离（Demucs，去除背景音乐，约为视频时长的 0.5 倍耗时）</label>
        <label>说话人数
          <select id="opt-n">
            <option value="0">自动估计</option>
            ${[2, 3, 4, 5, 6, 7, 8].map((n) => `<option value="${n}">${n} 人</option>`).join('')}
          </select>
        </label>
        <span class="grow"></span>
        <button class="btn primary" id="btn-start" disabled>开始处理</button>
      </div>
      <div id="upload-prog" class="big-progress hidden"><div class="progress"><i></i></div><b>0%</b></div>
    </section>
    <section class="card">
      <h3>处理流程</h3>
      <div class="steps-explain">
        ${STAGES.map(([, t, d], i) => `<div><b>${i + 1}. ${t}</b><span>${d}</span></div>`).join('')}
        <div><b>7. 合并导出</b><span>DPDFNet 降噪 · EBU R128 响度 · 320 kbps MP3 + 字幕</span></div>
      </div>
    </section>`;
  const input = $('#file'), dz = $('#dz');
  let file = null;
  const pick = (f) => { file = f; $('#dz-file').textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : ''; $('#btn-start').disabled = !f; };
  input.addEventListener('change', () => pick(input.files[0]));
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('over'); if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });
  $('#btn-start').addEventListener('click', () => {
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    fd.append('separate', $('#opt-sep').checked);
    fd.append('num_speakers', $('#opt-n').value);
    const xhr = new XMLHttpRequest();
    const prog = $('#upload-prog');
    prog.classList.remove('hidden');
    $('#btn-start').disabled = true;
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const p = Math.round((e.loaded / e.total) * 100);
      $('i', prog).style.width = `${p}%`;
      $('b', prog).textContent = p < 100 ? `${p}%` : '校验中…';
    };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 300) { toast(body.detail || '上传失败', 'err'); $('#btn-start').disabled = false; return; }
      location.hash = `#/job/${encodeURIComponent(body.id)}`;
      refreshJobs();
    };
    xhr.onerror = () => { toast('上传失败：无法连接服务器', 'err'); $('#btn-start').disabled = false; };
    xhr.open('POST', '/api/jobs');
    xhr.send(fd);
  });
}

/* ============================================================ job */
async function openJob(id) {
  if (state.jobId !== id) {
    state.jobId = id;
    state.result = null;
    state.match = null;
    state.names = local.get(`av:names:${id}`, {});
    state.selected = new Set(local.get(`av:sel:${id}`, []));
    state.excluded = new Set(local.get(`av:excl:${id}`, []));
  }
  main().innerHTML = '<div class="empty">加载中…</div>';
  await refreshJob();
}

async function refreshJob() {
  const id = state.jobId;
  let job;
  try { job = await api(`/api/jobs/${encodeURIComponent(id)}`); } catch (e) {
    main().innerHTML = `<div class="card"><div class="errbox">${esc(e.message)}</div></div>`;
    $('#export-bar').classList.add('hidden');
    return;
  }
  if (id !== state.jobId) return;
  const wasDone = state.job && state.job.id === id && state.job.status === 'done' && state.result;
  state.job = job;
  if (!wasDone && job.names && Object.keys(job.names).length) state.names = { ...job.names };
  if (job.status === 'done') {
    if (!wasDone) {
      state.result = job.result;
      state.ex.min_sim = Math.max(state.ex.min_sim, job.result.options?.min_sim ?? 0);
      state.selected = new Set([...state.selected].filter((s) => job.result.speakers.some((x) => x.id === s)));
      renderResult();
    }
    renderExports();
    if (Object.values(job.exports || {}).some((e) => ['queued', 'running'].includes(e.status))) pollExports();
  } else if (job.status === 'error') {
    renderError(job);
  } else {
    renderProgress(job);
    state.pollTimer = setTimeout(refreshJob, 1500);
  }
  const idx = state.jobs.findIndex((j) => j.id === id);
  if (idx >= 0) { Object.assign(state.jobs[idx], job); renderJobList(); }
}

function renderProgress(job) {
  $('#export-bar').classList.add('hidden');
  const skip = job.options && job.options.separate === false;
  const order = STAGES.map((s) => s[0]);
  const cur = order.indexOf(job.stage);
  const elapsed = job.started ? (Date.now() / 1000 - job.started) : 0;
  main().innerHTML = `
    <section class="card">
      <h2 style="font-size:19px">${esc(job.name)}</h2>
      <p class="muted" style="margin:6px 0 0">${job.status === 'queued' ? '排队中，等待前一个任务完成…' : `处理中 · 已用时 ${fmtDur(elapsed)}`}
        · 处理完成后会自动显示结果，可以离开此页面。</p>
      <div class="big-progress"><div class="progress"><i style="width:${(job.progress || 0) * 100}%"></i></div><b>${Math.round((job.progress || 0) * 100)}%</b></div>
      <p style="margin:8px 0 0">${esc(job.message || '')}</p>
      <ul class="stages">
        ${STAGES.map(([k, t, d], i) => {
          const cls = skip && k === 'separate' ? 'skip' : i < cur ? 'done' : i === cur && job.status === 'running' ? 'active' : '';
          return `<li class="${cls}"><span class="ic">${cls === 'done' ? '✓' : ''}</span><b>${t}</b><span class="muted">${d}</span></li>`;
        }).join('')}
      </ul>
    </section>`;
}

function renderError(job) {
  $('#export-bar').classList.add('hidden');
  main().innerHTML = `
    <section class="card">
      <h2 style="font-size:19px">${esc(job.name)}</h2>
      <p class="muted">处理失败</p>
      <div class="errbox">${esc(job.error || '未知错误')}</div>
      <div style="margin-top:14px;display:flex;gap:8px">
        <button class="btn primary" id="btn-retry">重试</button>
        <button class="btn danger" id="btn-del">删除任务</button>
      </div>
    </section>`;
  $('#btn-retry').onclick = () => reanalyze(job.options?.num_speakers || 0);
  $('#btn-del').onclick = deleteJob;
}

async function reanalyze(n) {
  try {
    await postJSON(`/api/jobs/${encodeURIComponent(state.jobId)}/reanalyze`, { num_speakers: Number(n) || 0 });
    state.result = null;
    state.job = null;
    state.match = null;
    toast('已重新排队（会复用已分离的人声与说话人日志，速度较快）');
    refreshJob();
    refreshJobs();
  } catch (e) { toast(e.message, 'err'); }
}

async function deleteJob() {
  if (!confirm('确定删除该任务及其所有文件？此操作无法撤销。')) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(state.jobId)}`, { method: 'DELETE' });
    location.hash = '#/new';
    refreshJobs();
  } catch (e) { toast(e.message, 'err'); }
}

/* ============================================================ result */
function renderResult() {
  const { job, result } = state;
  const opts = result.options || {};
  main().innerHTML = `
    <section class="card">
      <div class="summary">
        <div>
          <h2>${esc(job.name)}</h2>
          <div class="meta">
            <span class="pill">时长 ${fmt(result.duration)}</span>
            <span class="pill">${result.speakers.length} 位说话人</span>
            <span class="pill">剔除重叠语音 ${fmtDur(result.overlap_seconds)}</span>
            ${result.fragments?.groups ? `<span class="pill" title="笑声、插话等不足以构成说话人的零散声音，已从所有说话人中排除">排除零散声音 ${fmtDur(result.fragments.seconds)}</span>` : ''}
            <span class="pill">${opts.separate ? 'Demucs 人声分离' : '未做人声分离'}</span>
            <span class="pill">${result.segments.length} 个纯净片段</span>
          </div>
        </div>
        <div class="actions">
          <button class="btn small" id="btn-re">重新分析…</button>
          <button class="btn small danger" id="btn-del">删除</button>
        </div>
        <div class="reanalyze hidden" id="re-box">
          <span>说话人数</span>
          <select id="re-n"><option value="0">自动估计</option>${[2, 3, 4, 5, 6, 7, 8].map((n) => `<option value="${n}" ${opts.num_speakers === n ? 'selected' : ''}>${n} 人</option>`).join('')}</select>
          <button class="btn small primary" id="re-go">开始</button>
          <span class="muted" style="font-size:12px">自动估计根据声纹谱聚类的特征值间隔判断人数；结果不符时可指定准确人数重新分析（复用已分离的人声，很快）。同一人被拆成多个说话人时，也可在下方多选后一起导出。</span>
        </div>
      </div>
    </section>

    <section class="card">
      <h3>说话人时间轴 <span class="sub">点击色块试听该片段（已去除重叠和低纯净度部分）</span></h3>
      <div class="timeline-wrap"><canvas id="timeline" height="10"></canvas></div>
      <div class="legend">${result.speakers.map((s) => `<span><i style="background:${spkColor(s.id)}"></i>${esc(spkName(s.id))}</span>`).join('')}</div>
    </section>

    <section class="card">
      <h3>识别目标人物 <span class="sub">三种方式任选其一，结果会标注在下方的说话人卡片上</span></h3>
      <div class="tabs" id="id-tabs">
        <button data-tab="text" class="active">台词搜索</button>
        <button data-tab="audio">参考音频（声纹）</button>
        <button data-tab="image">参考照片（人脸）</button>
      </div>
      <div data-pane="text">
        <form class="search-form" id="search-form">
          <input type="search" name="q" placeholder="输入目标人物说过的一句话，部分内容也能匹配" autocomplete="off">
          <button class="btn primary">搜索</button>
        </form>
        <div id="search-out" class="match-out"></div>
      </div>
      <div data-pane="audio" class="hidden">
        <label class="filepick"><span class="big">🎙️</span>
          <span><b>选择目标人物的参考音频</b><br><span class="hint">建议 5 秒以上的单人说话录音（任意音视频格式），系统会与每位说话人的声纹做余弦相似度比对</span></span>
          <input type="file" id="ref-audio" accept="audio/*,video/*">
        </label>
        <div id="audio-out" class="match-out"></div>
      </div>
      <div data-pane="image" class="hidden">
        <label class="filepick"><span class="big">🖼️</span>
          <span><b>选择目标人物照片</b><br><span class="hint">可直接用视频截图。系统会在每位说话人说话时的画面中查找该人脸（镜头通常对着正在说话的人）</span></span>
          <input type="file" id="ref-image" accept="image/*">
        </label>
        <div id="image-out" class="match-out"></div>
      </div>
    </section>

    <section class="card" id="exports-card" style="display:none">
      <h3>已导出的文件</h3>
      <ul class="exports-list" id="exports-list"></ul>
    </section>

    <section class="speakers" id="speakers"></section>`;

  $('#btn-re').onclick = () => $('#re-box').classList.toggle('hidden');
  $('#re-go').onclick = () => reanalyze($('#re-n').value);
  $('#btn-del').onclick = deleteJob;
  $$('#id-tabs button').forEach((b) => b.addEventListener('click', () => {
    $$('#id-tabs button').forEach((x) => x.classList.toggle('active', x === b));
    $$('[data-pane]').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== b.dataset.tab));
  }));
  $('#search-form').addEventListener('submit', (e) => { e.preventDefault(); doSearch(e.target.q.value); });
  $('#ref-audio').addEventListener('change', (e) => e.target.files[0] && doMatch('audio', e.target.files[0]));
  $('#ref-image').addEventListener('change', (e) => e.target.files[0] && doMatch('image', e.target.files[0]));
  const canvas = $('#timeline');
  canvas.addEventListener('click', onTimelineClick);
  canvas.addEventListener('mousemove', (e) => { canvas.title = timelineHit(e)?.seg?.text || ''; });
  renderSpeakers();
  drawTimeline();
  renderExportBar();
}

function renderSpeakers() {
  const { result } = state;
  const best = state.match?.scores?.[0]?.id;
  const scoreOf = (id) => state.match?.scores?.find((s) => s.id === id);
  $('#speakers').innerHTML = result.speakers.map((s, i) => {
    const segs = result.segments.filter((g) => g.spk === s.id);
    const snipIds = (s.preview_segments || []).slice(0, 3);
    const snips = (snipIds.length ? snipIds.map((id) => result.segments[id]) : segs.filter((g) => g.text).slice(0, 3)).filter(Boolean);
    const sc = scoreOf(s.id);
    const badge = sc ? `<span class="badge ${s.id === best ? '' : 'dim'}" title="${esc(state.match.label)}">${s.id === best ? '最佳匹配 · ' : ''}${esc(state.match.fmt(sc))}</span>` : '';
    const thumbs = (s.thumbs || []).length
      ? `<div class="thumbs">${s.thumbs.map((t) => `<button data-act="thumb" data-time="${t.time}" title="${fmt(t.time)} 处的画面，点击试听"><img loading="lazy" src="${assetUrl(`thumbs/${t.file}`)}" alt=""><span>${fmt(t.time)}</span></button>`).join('')}</div>`
      : '<div class="thumbs none">无视频画面</div>';
    const similar = (s.similar || []).length ? `<div class="similar">声纹与 ${s.similar.map((x) => esc(spkName(x.id))).join('、')} 较接近，可能是同一人被拆分</div>` : '';
    return `
      <article class="spk ${state.selected.has(s.id) ? 'selected' : ''}" data-spk="${s.id}" style="--c:${COLORS[i % COLORS.length]}">
        <header>
          <input type="checkbox" data-act="select" ${state.selected.has(s.id) ? 'checked' : ''} title="选择用于导出">
          <input class="name" type="text" data-act="rename" value="${esc(state.names[s.id] || '')}" placeholder="${esc(s.label)}（点击命名）">
          <span class="chip" style="--c:${COLORS[i % COLORS.length]}">${s.id}</span>
          ${badge}
        </header>
        ${thumbs}
        <div class="stats">
          <div><b>${fmt(s.clean_seconds)}</b><span>纯净语音</span></div>
          <div><b>${s.num_segments}</b><span>片段数</span></div>
          <div><b>${fmt(s.raw_seconds)}</b><span>发言总时长</span></div>
        </div>
        <audio controls preload="none" src="${assetUrl(`previews/${s.id}.mp3`)}"></audio>
        <ul class="snips">${snips.map((g) => `<li data-act="seg" data-seg="${g.id}" data-play="seg-${g.id}"><span class="t">${fmt(g.start)}</span><span class="x">${esc(g.text || '（无文字）')}</span></li>`).join('')}</ul>
        ${similar}
        <footer>
          <button class="btn small" data-act="segments">查看全部 ${segs.length} 段</button>
          <button class="btn small" data-act="only">只选此人</button>
        </footer>
      </article>`;
  }).join('');
  player.markUI();
}

document.addEventListener('click', (e) => {
  const el = e.target.closest('[data-act]');
  if (!el || !state.result) return;
  const card = el.closest('[data-spk]');
  const id = card?.dataset.spk;
  const act = el.dataset.act;
  if (act === 'thumb') {
    const t = Number(el.dataset.time);
    const seg = state.result.segments.find((g) => g.spk === id && g.start <= t && g.end >= t);
    if (seg) playSegment(seg);
  } else if (act === 'seg') {
    const seg = state.result.segments[Number(el.dataset.seg)];
    if (seg) playSegment(seg);
  } else if (act === 'segments') {
    openDrawer(id);
  } else if (act === 'only') {
    state.selected = new Set([id]);
    saveLocal();
    renderSpeakers();
    renderExportBar();
  } else if (act === 'pick-best') {
    state.selected = new Set([el.dataset.id]);
    saveLocal();
    renderSpeakers();
    renderExportBar();
    $(`[data-spk="${el.dataset.id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
});
document.addEventListener('change', (e) => {
  const el = e.target;
  if (!state.result) return;
  const id = el.closest('[data-spk]')?.dataset.spk;
  if (el.dataset.act === 'select') {
    el.checked ? state.selected.add(id) : state.selected.delete(id);
    el.closest('.spk').classList.toggle('selected', el.checked);
    saveLocal();
    renderExportBar();
  } else if (el.dataset.act === 'rename') {
    const v = el.value.trim();
    v ? (state.names[id] = v) : delete state.names[id];
    saveLocal();
    api(`/api/jobs/${encodeURIComponent(state.jobId)}/names`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state.names) }).catch(() => {});
    $$('.legend span').forEach((sp, i) => { sp.lastChild.textContent = spkName(state.result.speakers[i].id); });
    renderExportBar();
  }
});

/* ---------------- timeline ---------------- */
const TL = { lane: 16, gap: 5, axis: 20, left: 0 };
function drawTimeline() {
  const canvas = $('#timeline');
  if (!canvas || !state.result) return;
  const { speakers, segments, duration } = state.result;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = speakers.length * (TL.lane + TL.gap) + TL.axis;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = `${h}px`;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const x = (t) => (t / duration) * w;
  speakers.forEach((s, i) => {
    const y = i * (TL.lane + TL.gap);
    ctx.fillStyle = css.getPropertyValue('--panel-2');
    ctx.fillRect(0, y, w, TL.lane);
    ctx.fillStyle = spkColor(s.id);
    const sel = state.selected.size === 0 || state.selected.has(s.id);
    ctx.globalAlpha = sel ? 1 : 0.35;
    for (const g of segments) {
      if (g.spk !== s.id) continue;
      ctx.globalAlpha = (sel ? 1 : 0.35) * (state.excluded.has(g.id) ? 0.25 : 1);
      ctx.fillRect(x(g.start), y + 1, Math.max(1, x(g.end) - x(g.start)), TL.lane - 2);
    }
    ctx.globalAlpha = 1;
  });
  // axis
  const ay = speakers.length * (TL.lane + TL.gap) + 4;
  ctx.fillStyle = css.getPropertyValue('--muted');
  ctx.font = '11px system-ui, sans-serif';
  // keep at least ~56 px between tick labels
  const step = [15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600].find((s) => (s / duration) * w >= 56) || 7200;
  for (let t = 0; t <= duration; t += step) {
    ctx.fillRect(x(t), ay - 3, 1, 4);
    const label = fmt(t);
    const tx = Math.min(Math.max(0, x(t) - 12), w - 30);
    ctx.fillText(label, tx, ay + 11);
  }
  // playhead
  const a = player.el;
  if (a && !a.paused && a.dataset.src && a.dataset.src.includes('vocals_browse.mp3')) {
    ctx.fillStyle = css.getPropertyValue('--err');
    ctx.fillRect(x(a.currentTime), 0, 2, ay - 4);
  }
}
function timelineHit(e) {
  const canvas = $('#timeline');
  const r = canvas.getBoundingClientRect();
  const lane = Math.floor((e.clientY - r.top) / (TL.lane + TL.gap));
  const t = ((e.clientX - r.left) / r.width) * state.result.duration;
  const spk = state.result.speakers[lane];
  if (!spk) return null;
  const pad = state.result.duration / r.width * 3; // 3px tolerance
  const seg = state.result.segments.find((g) => g.spk === spk.id && g.start - pad <= t && g.end + pad >= t);
  return { spk, seg, t };
}
function onTimelineClick(e) {
  const hit = timelineHit(e);
  if (hit?.seg) playSegment(hit.seg);
}
window.addEventListener('resize', () => drawTimeline());

/* ---------------- identification ---------------- */
function highlight(text, q) {
  const norm = q.replace(/[\s\p{P}]/gu, '');
  if (!norm) return esc(text);
  const i = text.indexOf(norm);
  if (i < 0) return esc(text);
  return `${esc(text.slice(0, i))}<mark>${esc(norm)}</mark>${esc(text.slice(i + norm.length))}`;
}
function renderScores(el, scores, fmtScore, max = 1) {
  const best = scores[0];
  el.innerHTML = `
    ${best ? `<div class="verdict"><span>最可能是 ${chip(best.id)}</span><button class="btn small" data-act="pick-best" data-id="${best.id}">选中并准备导出</button></div>` : ''}
    ${scores.map((s) => `
      <div class="scorebar"><span class="who">${chip(s.id)}</span>
        <span class="bar"><i style="width:${Math.max(2, Math.min(100, (Math.max(0, s.score) / max) * 100))}%;background:${spkColor(s.id)}"></i></span>
        <b>${esc(fmtScore(s))}</b></div>`).join('')}`;
}
async function doSearch(q) {
  const out = $('#search-out');
  if (!q.trim()) return;
  out.innerHTML = '<span class="muted">搜索中…</span>';
  try {
    const r = await api(`/api/jobs/${encodeURIComponent(state.jobId)}/search?q=${encodeURIComponent(q)}`);
    if (!r.hits.length) { out.innerHTML = '<span class="muted">没有找到相近的台词，换一句试试（识别文本可能存在同音错字）。</span>'; return; }
    const total = r.votes.reduce((a, v) => a + v.score, 0);
    const scores = r.votes.map((v) => ({ id: v.id, score: v.score / total }));
    state.match = { kind: 'text', label: '台词匹配占比', scores, fmt: (s) => `台词 ${Math.round(s.score * 100)}%` };
    renderScores(out, scores, (s) => `${Math.round(s.score * 100)}%`);
    out.insertAdjacentHTML('beforeend', `<ul class="hits">${r.hits.map((h) => `
      <li data-act="seg" data-seg="${h.segment}" data-play="seg-${h.segment}">
        <button class="playbtn" type="button">▶</button>${chip(h.spk)}
        <span>${highlight(h.text, q)}</span><span class="muted">${fmt(h.start)}</span></li>`).join('')}</ul>`);
    renderSpeakers();
  } catch (e) { out.innerHTML = `<div class="errbox">${esc(e.message)}</div>`; }
}
async function doMatch(kind, file) {
  const out = $(`#${kind}-out`);
  out.innerHTML = `<span class="muted">${kind === 'audio' ? '提取声纹并比对中…' : '在各说话人的画面中查找人脸…（约 10–30 秒）'}</span>`;
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await api(`/api/jobs/${encodeURIComponent(state.jobId)}/match/${kind}`, { method: 'POST', body: fd });
    if (kind === 'audio') {
      state.match = { kind, label: '声纹余弦相似度', scores: r.scores, fmt: (s) => `声纹 ${s.score.toFixed(2)}` };
      renderScores(out, r.scores, (s) => s.score.toFixed(3));
    } else {
      state.match = { kind, label: '人脸相似度（括号内为命中帧比例）', scores: r.scores, fmt: (s) => `人脸 ${s.score.toFixed(2)}` };
      renderScores(out, r.scores, (s) => `${s.score.toFixed(2)} (${Math.round(s.match_rate * 100)}%)`);
    }
    renderSpeakers();
  } catch (e) { out.innerHTML = `<div class="errbox">${esc(e.message)}</div>`; }
}

/* ---------------- segment drawer ---------------- */
function openDrawer(id) {
  state.drawerSpk = id;
  renderDrawer();
  $('#drawer').classList.remove('hidden');
  $('#drawer-backdrop').classList.remove('hidden');
}
function closeDrawer() {
  $('#drawer').classList.add('hidden');
  $('#drawer-backdrop').classList.add('hidden');
  state.drawerSpk = null;
}
$('#drawer-backdrop').addEventListener('click', closeDrawer);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });
function renderDrawer() {
  const id = state.drawerSpk;
  const segs = state.result.segments.filter((g) => g.spk === id);
  if (state.drawerSort === 'purity') segs.sort((a, b) => a.sim + a.margin - (b.sim + b.margin));
  const kept = segs.filter((g) => !state.excluded.has(g.id));
  const keptDur = kept.reduce((a, g) => a + g.end - g.start, 0);
  $('#drawer').innerHTML = `
    <header><h3>${chip(id)} 的全部片段</h3><button class="btn small" id="dr-close">关闭</button></header>
    <div class="tools">
      <span>保留 ${kept.length} / ${segs.length} 段 · ${fmtDur(keptDur)}</span>
      <span class="grow" style="flex:1"></span>
      <label>排序 <select id="dr-sort"><option value="time">按时间</option><option value="purity" ${state.drawerSort === 'purity' ? 'selected' : ''}>纯净度从低到高</option></select></label>
      <button class="btn small" id="dr-all">全部保留</button>
    </div>
    <ul class="seglist">${segs.map((g) => {
      const purity = Math.max(0, Math.min(1, (g.sim - 0.4) / 0.5));
      return `<li data-play="seg-${g.id}" class="${state.excluded.has(g.id) ? 'off' : ''}">
        <button class="playbtn" data-seg-play="${g.id}">▶</button>
        <span class="t">${fmt(g.start)}</span><span class="d">${(g.end - g.start).toFixed(1)}s</span>
        <span class="purity" title="声纹相似度 ${g.sim} · 区分度 ${g.margin}"><i style="width:${purity * 100}%"></i></span>
        <span class="txt">${esc(g.text || '（无文字）')}</span>
        <input type="checkbox" data-incl="${g.id}" ${state.excluded.has(g.id) ? '' : 'checked'} title="导出时包含">
      </li>`;
    }).join('')}</ul>`;
  $('#dr-close').onclick = closeDrawer;
  $('#dr-sort').onchange = (e) => { state.drawerSort = e.target.value; renderDrawer(); };
  $('#dr-all').onclick = () => { segs.forEach((g) => state.excluded.delete(g.id)); saveLocal(); renderDrawer(); renderExportBar(); drawTimeline(); };
  $$('[data-seg-play]', $('#drawer')).forEach((b) => b.addEventListener('click', () => playSegment(state.result.segments[Number(b.dataset.segPlay)])));
  $$('[data-incl]', $('#drawer')).forEach((c) => c.addEventListener('change', () => {
    const sid = Number(c.dataset.incl);
    c.checked ? state.excluded.delete(sid) : state.excluded.add(sid);
    c.closest('li').classList.toggle('off', !c.checked);
    saveLocal();
    renderExportBar();
    drawTimeline();
  }));
  player.markUI();
}

/* ---------------- export ---------------- */
function exportSegments() {
  const ex = state.ex;
  return state.result.segments.filter((g) => state.selected.has(g.spk) && !state.excluded.has(g.id)
    && g.sim >= ex.min_sim && g.end - g.start >= ex.min_segment && (g.text || '').length > 0
    && (g.event || 'Speech') === 'Speech');
}
function defaultName() {
  return [...state.selected].map((id) => state.names[id] || id).join('_');
}
function renderExportBar() {
  const bar = $('#export-bar');
  if (!state.result || state.selected.size === 0) { bar.classList.add('hidden'); return; }
  const segs = exportSegments();
  const secs = segs.reduce((a, g) => a + g.end - g.start + 0.1, 0);
  const fileSecs = secs + Math.max(0, segs.length - 1) * state.ex.gap;
  const keepName = $('#ex-name')?.dataset.touched ? $('#ex-name').value : defaultName();
  const ex = state.ex;
  const running = state.exportKey && ['queued', 'running'].includes(state.job?.exports?.[state.exportKey]?.status);
  bar.classList.remove('hidden');
  bar.innerHTML = `
    <div class="row">
      <span>导出 ${[...state.selected].map(chip).join(' ')}</span>
      <span class="est">${segs.length} 段 · 语音 ${fmtDur(secs)}（文件约 ${fmt(fileSecs)}）</span>
      <span class="grow"></span>
      <label>文件名 <input type="text" id="ex-name" value="${esc(keepName)}" ${$('#ex-name')?.dataset.touched ? 'data-touched="1"' : ''}></label>
      <button class="btn primary" id="ex-go" ${running || !segs.length ? 'disabled' : ''}>合并并导出 MP3</button>
    </div>
    <div class="row">
      <label><input type="checkbox" id="ex-dn" ${ex.denoise ? 'checked' : ''}> DPDFNet 降噪</label>
      <label>强度 <select id="ex-dndb">${[[8, '轻'], [15, '中'], [30, '强']].map(([v, t]) => `<option value="${v}" ${ex.denoise_db === v ? 'selected' : ''}>${t}</option>`).join('')}</select></label>
      <label><input type="checkbox" id="ex-ln" ${ex.loudnorm ? 'checked' : ''}> 响度标准化 −18 LUFS</label>
      <label title="片段声纹与该说话人中心的最低相似度；越高越纯净、时长越少">纯净度 <input type="range" id="ex-sim" min="${state.result.options?.min_sim ?? 0.45}" max="0.85" step="0.01" value="${ex.min_sim}"><output>${ex.min_sim.toFixed(2)}</output></label>
      <label>最短片段 <select id="ex-min">${[1, 2, 3].map((v) => `<option value="${v}" ${ex.min_segment === v ? 'selected' : ''}>${v} 秒</option>`).join('')}</select></label>
      <label>片段间隔 <select id="ex-gap">${[0.3, 0.5, 1].map((v) => `<option value="${v}" ${ex.gap === v ? 'selected' : ''}>${v} 秒</option>`).join('')}</select></label>
      <label><input type="checkbox" id="ex-sl" ${ex.slices ? 'checked' : ''}> 附带切片包（WAV + 文本标注）</label>
    </div>
    <div class="row ex-status" id="ex-status"></div>`;
  const bind = (sel, key, conv) => $(sel).addEventListener('change', (e) => { state.ex[key] = conv(e.target); renderExportBar(); drawTimeline(); });
  bind('#ex-dn', 'denoise', (t) => t.checked);
  bind('#ex-dndb', 'denoise_db', (t) => Number(t.value));
  bind('#ex-ln', 'loudnorm', (t) => t.checked);
  bind('#ex-min', 'min_segment', (t) => Number(t.value));
  bind('#ex-gap', 'gap', (t) => Number(t.value));
  bind('#ex-sl', 'slices', (t) => t.checked);
  $('#ex-sim').addEventListener('input', (e) => { e.target.nextElementSibling.textContent = Number(e.target.value).toFixed(2); });
  bind('#ex-sim', 'min_sim', (t) => Number(t.value));
  $('#ex-name').addEventListener('input', (e) => { e.target.dataset.touched = '1'; });
  $('#ex-go').addEventListener('click', startExport);
  renderExportStatus();
}

async function startExport() {
  const ex = state.ex;
  const body = {
    speakers: [...state.selected],
    name: $('#ex-name').value.trim() || defaultName(),
    denoise: ex.denoise,
    denoise_db: ex.denoise_db,
    loudnorm: ex.loudnorm,
    min_sim: ex.min_sim,
    min_segment: ex.min_segment,
    gap: ex.gap,
    slices: ex.slices,
    exclude: [...state.excluded],
  };
  try {
    const r = await postJSON(`/api/jobs/${encodeURIComponent(state.jobId)}/export`, body);
    state.exportKey = r.key;
    state.job.exports = { ...(state.job.exports || {}), [r.key]: { status: 'queued', progress: 0, message: '排队中' } };
    renderExportBar();
    pollExports();
  } catch (e) { toast(e.message, 'err'); }
}

async function pollExports() {
  clearTimeout(state.exportTimer);
  const id = state.jobId;
  try {
    const exps = await api(`/api/jobs/${encodeURIComponent(id)}/exports`);
    if (id !== state.jobId) return;
    const before = state.job.exports || {};
    state.job.exports = exps;
    if (!state.exportKey) {
      const run = Object.entries(exps).find(([, e]) => ['queued', 'running'].includes(e.status));
      if (run) state.exportKey = run[0];
    }
    const cur = exps[state.exportKey];
    if (cur && cur.status === 'done' && before[state.exportKey]?.status !== 'done') toast('导出完成', 'ok');
    if (cur && cur.status === 'error' && before[state.exportKey]?.status !== 'error') toast(cur.message, 'err');
    renderExportStatus();
    renderExports();
    if (Object.values(exps).some((e) => ['queued', 'running'].includes(e.status))) state.exportTimer = setTimeout(pollExports, 1000);
    else if ($('#ex-go')) $('#ex-go').disabled = !exportSegments().length;
  } catch { state.exportTimer = setTimeout(pollExports, 3000); }
}

function downloadLinks(key, meta) {
  const f = meta.files || {};
  const link = (name, label) => `<a class="btn small" href="${fileUrl(`export/${encodeURIComponent(name)}`)}?download=1">${label}</a>`;
  return [f.mp3 && link(f.mp3, '⬇ MP3'), f.srt && link(f.srt, '⬇ 字幕 SRT'), f.zip && link(f.zip, '⬇ 切片包 ZIP')].filter(Boolean).join('');
}
function renderExportStatus() {
  const el = $('#ex-status');
  if (!el) return;
  const cur = state.exportKey && state.job?.exports?.[state.exportKey];
  if (!cur) { el.innerHTML = ''; return; }
  if (['queued', 'running'].includes(cur.status)) {
    el.innerHTML = `<div class="progress"><i style="width:${(cur.progress || 0) * 100}%"></i></div><span>${esc(cur.message || '')}</span>`;
    if ($('#ex-go')) $('#ex-go').disabled = true;
  } else if (cur.status === 'done' && cur.meta) {
    el.innerHTML = `<span class="ex-done"><b>✓ ${esc(state.exportKey)}.mp3</b> <span class="muted">${cur.meta.segments} 段 · 语音 ${fmtDur(cur.meta.speech_seconds)}</span>
      <audio controls preload="none" src="${fileUrl(`export/${encodeURIComponent(cur.meta.files.mp3)}`)}?v=${Math.round(cur.updated || 0)}"></audio>${downloadLinks(state.exportKey, cur.meta)}</span>`;
  } else if (cur.status === 'error') {
    el.innerHTML = `<span class="errbox" style="padding:6px 10px">${esc(cur.message)}</span>`;
  }
}
function renderExports() {
  const card = $('#exports-card');
  if (!card) return;
  const done = Object.entries(state.job?.exports || {}).filter(([, e]) => e.status === 'done' && e.meta);
  card.style.display = done.length ? '' : 'none';
  $('#exports-list').innerHTML = done.sort((a, b) => b[1].updated - a[1].updated).map(([key, e]) => `
    <li><span class="n">${esc(key)}.mp3</span>${(e.meta.speakers || []).map(chip).join('')}
      <span class="muted">${e.meta.segments} 段 · 语音 ${fmtDur(e.meta.speech_seconds)} · ${e.meta.options?.denoise ? '已降噪' : '未降噪'}</span>
      <audio controls preload="none" src="${fileUrl(`export/${encodeURIComponent(e.meta.files.mp3)}`)}?v=${Math.round(e.updated || 0)}"></audio>
      ${downloadLinks(key, e.meta)}</li>`).join('');
}

/* ============================================================ boot */
async function health() {
  const el = $('#health');
  try {
    const h = await api('/api/health');
    el.className = `pill ${h.models_ready ? 'ok' : 'warn'}`;
    el.textContent = h.models_ready ? `● 模型就绪 · ${h.threads} 线程 CPU` : '● 首次任务将自动下载模型';
  } catch {
    el.className = 'pill err';
    el.textContent = '● 后端未连接';
  }
}
player.init();
window.addEventListener('hashchange', route);
health();
refreshJobs().then(route);
setInterval(refreshJobs, 5000);
