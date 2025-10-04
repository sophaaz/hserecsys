// app.js — Two-Tower demo (TF.js) — fast train + 3 comparison tables + robust PCA
// -----------------------------------------------------------------------------
(async function App() {
  'use strict';

  // ------------------------------- Config ------------------------------------
  const CONFIG = {
    embDim: 32,
    userHidden: 64,
    itemHidden: 64,

    // Быстрее тренируем: BPR + меньше эпох + крупный батч
    learningRate: 0.003,
    l2: 1e-4,
    normalize: true,
    lossType: 'bpr',     // 'bpr' быстрее, чем in-batch softmax

    epochs: 6,
    batchSize: 2048,

    posThreshold: 4,
    topK: 10,
    genreCount: 19,
    files: {
      item: 'data/u.item',
      data: 'data/u.data'
    },

    // Графики / PCA
    lossDrawEvery: 5,
    pcaItems: 500,

    // Исторический топ – порог по количеству оценок
    minRatingsForHistoricalTop: 50
  };

  // ------------------------------- State -------------------------------------
  const ST = {
    items: new Map(),            // rawItemId -> { title, year, genres[19] }
    interactionsRaw: [],         // { userId, itemId, rating }

    userMap: new Map(),          // rawUserId -> u
    itemMap: new Map(),          // rawItemId -> i
    revUser: [],                 // u -> rawUserId
    revItem: [],                 // i -> rawItemId

    positives: [],               // {u,i} (rating>=posThreshold)
    userSeen: new Map(),         // u -> Set(i)

    itemGenresDense: null,       // [I,19] L2-норм
    userGenresDense: null,       // [U,19] L2-норм

    // агрегаты для исторического топа
    itemSum: null,               // Float32Array[I]
    itemCnt: null,               // Uint32Array[I]

    model: null,
    stats: { nUsers: 0, nItems: 0, nRatings: 0 },
    lossHistory: []
  };

  // ------------------------------- DOM ---------------------------------------
  const $ = s => document.querySelector(s);
  const btnLoad  = $('#loadData');
  const btnTrain = $('#train');
  const btnTest  = $('#test');
  const statusEl = $('#status');
  const lossCanvas = $('#lossChart');
  const pcaCanvas  = $('#embeddingChart');
  const resultsEl  = $('#results');

  // контейнер для трёх таблиц (если нет — создадим внизу)
  let tablesHost = $('#comparison-tables');
  if (!tablesHost) {
    tablesHost = document.createElement('div');
    tablesHost.id = 'comparison-tables';
    document.body.appendChild(tablesHost);
  }

  const GENRES = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];

  const setStatus = msg => (statusEl ? (statusEl.textContent = msg) : console.log('[status]', msg));
  const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  // ------------------------------ Data loading --------------------------------
  // u.item: movieId|title|...|g0..g18
  async function loadItems() {
    const res = await fetch(CONFIG.files.item);
    if (!res.ok) throw new Error(`Failed to fetch ${CONFIG.files.item}: ${res.status}`);
    const text = await res.text();
    const lines = text.split(/\r?\n/);
    const G = CONFIG.genreCount;

    ST.items.clear();
    for (let lineNo = 0; lineNo < lines.length; lineNo++) {
      const line = lines[lineNo].trim();
      if (!line) continue;
      const parts = line.split('|');
      if (parts.length < 5 + G) continue;

      const rawItemId = parseInt(parts[0], 10);
      const titleRaw = String(parts[1] ?? '');
      const yearMatch = /\((\d{4})\)\s*$/.exec(titleRaw);
      const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
      const title = titleRaw.replace(/\(\d{4}\)\s*$/, '').trim();

      const flags = new Array(G);
      for (let g = 0; g < G; g++) {
        const v = parts[5 + g];
        flags[g] = (v === '1' || v === 1) ? 1 : 0;
      }
      ST.items.set(rawItemId, { title, year, genres: flags });
    }
  }

  // u.data: userId\titemId\trating\ttimestamp
  async function loadRatings() {
    const res = await fetch(CONFIG.files.data);
    if (!res.ok) throw new Error(`Failed to fetch ${CONFIG.files.data}: ${res.status}`);
    const text = await res.text();
    const lines = text.split(/\r?\n/);

    ST.interactionsRaw.length = 0;
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      const [uS, iS, rS] = line.split('\t');
      if (!uS || !iS || !rS) continue;
      const userId = parseInt(uS, 10);
      const itemId = parseInt(iS, 10);
      const rating = parseFloat(rS);
      if (!Number.isFinite(userId) || !Number.isFinite(itemId) || !Number.isFinite(rating)) continue;
      if (!ST.items.has(itemId)) continue;
      ST.interactionsRaw.push({ userId, itemId, rating });
    }
  }

  function buildMappingsAndAggregates() {
    // users
    {
      const users = new Set(ST.interactionsRaw.map(x => x.userId));
      let idx = 0;
      for (const u of users) { ST.userMap.set(u, idx++); ST.revUser.push(u); }
    }
    // items
    {
      const items = Array.from(ST.items.keys()).sort((a,b)=>a-b);
      let idx = 0;
      for (const i of items) { ST.itemMap.set(i, idx++); ST.revItem.push(i); }
    }

    const I = ST.revItem.length;
    ST.itemSum = new Float32Array(I);
    ST.itemCnt = new Uint32Array(I);

    // seen + positives + агрегаты по фильмам
    ST.userSeen.clear(); ST.positives.length = 0;
    for (const { userId, itemId, rating } of ST.interactionsRaw) {
      const u = ST.userMap.get(userId);
      const i = ST.itemMap.get(itemId);
      if (u == null || i == null) continue;

      if (!ST.userSeen.has(u)) ST.userSeen.set(u, new Set());
      ST.userSeen.get(u).add(i);
      if (rating >= CONFIG.posThreshold) ST.positives.push({ u, i });

      ST.itemSum[i] += rating;
      ST.itemCnt[i] += 1;
    }

    ST.stats.nUsers = ST.revUser.length;
    ST.stats.nItems = ST.revItem.length;
    ST.stats.nRatings = ST.interactionsRaw.length;
  }

  function buildGenreMatrices() {
    const U = ST.stats.nUsers, I = ST.stats.nItems, G = CONFIG.genreCount;

    const itemGenres = Array.from({ length: I }, () => Array(G).fill(0));
    for (let i = 0; i < I; i++) {
      const rawItemId = ST.revItem[i];
      const meta = ST.items.get(rawItemId);
      itemGenres[i] = meta?.genres ? meta.genres.slice(0, G) : Array(G).fill(0);
    }

    const userGenres = Array.from({ length: U }, () => Array(G).fill(0));
    for (const { u, i } of ST.positives) {
      const gs = itemGenres[i];
      for (let g = 0; g < G; g++) userGenres[u][g] += gs[g];
    }

    const l2 = row => {
      let s = 0; for (const v of row) s += v*v;
      const d = Math.sqrt(s) || 1;
      return row.map(v => v / d);
    };
    ST.itemGenresDense = itemGenres.map(l2);
    ST.userGenresDense = userGenres.map(l2);
  }

  // ------------------------------ Model wiring --------------------------------
  function buildModel() {
    if (ST.model) { ST.model.dispose(); ST.model = null; }
    ST.model = new TwoTowerModel(
      ST.stats.nUsers, ST.stats.nItems, CONFIG.embDim,
      { lossType: CONFIG.lossType, lr: CONFIG.learningRate, userHidden: CONFIG.userHidden,
        itemHidden: CONFIG.itemHidden, l2: CONFIG.l2, normalize: CONFIG.normalize }
    );
    ST.model.setFeatures({ itemGenres: ST.itemGenresDense, userGenres: ST.userGenresDense });
  }

  function* batchIterator(pairs, batchSize) {
    const idx = pairs.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = (Math.random()*(i+1))|0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let start = 0; start < idx.length; start += batchSize) {
      const end = Math.min(idx.length, start + batchSize);
      const u = new Int32Array(end - start);
      const it = new Int32Array(end - start);
      for (let b = 0; b < end - start; b++) { const p = pairs[idx[start + b]]; u[b] = p.u; it[b] = p.i; }
      yield { users: u, items: it, size: (end - start), pct: end / idx.length };
    }
  }

  // ------------------------------ Training + charts ---------------------------
  function drawLoss() {
    if (!lossCanvas) return;
    // если канвас без размеров — зададим
    if (!lossCanvas.width || !lossCanvas.height) { lossCanvas.width = 640; lossCanvas.height = 220; }
    const ctx = lossCanvas.getContext('2d');
    const W = lossCanvas.width, H = lossCanvas.height;
    ctx.clearRect(0,0,W,H);

    if (!ST.lossHistory.length) return;
    const maxV = Math.max(...ST.lossHistory);
    const minV = Math.min(...ST.lossHistory);
    const range = (maxV - minV) || 1;

    // оси
    ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(40,10); ctx.lineTo(40,H-30); ctx.lineTo(W-10,H-30); ctx.stroke();

    // линия лосса
    ctx.strokeStyle = '#007acc'; ctx.lineWidth = 2; ctx.beginPath();
    ST.lossHistory.forEach((v, i) => {
      const x = 40 + (i/(ST.lossHistory.length-1))*(W-50);
      const y = (H-30) - ((v - minV)/range)*(H-40);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();

    // подписи
    ctx.fillStyle = '#000'; ctx.font = '12px Arial';
    ctx.fillText(`min ${minV.toFixed(4)}`, 45, 14);
    ctx.fillText(`max ${maxV.toFixed(4)}`, 120, 14);
    ctx.fillText('batches →', W-90, H-12);
  }

  async function train() {
    if (!ST.model) { setStatus('Model is not initialized'); return; }
    if (!ST.positives.length) { setStatus('No positive pairs to train on'); return; }
    await tf.ready();
    setStatus('Training…');

    ST.lossHistory.length = 0;
    let batchCount = 0;

    for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
      let lossSum = 0, count = 0;

      for (const batch of batchIterator(ST.positives, CONFIG.batchSize)) {
        const loss = await ST.model.trainStep(batch.users, batch.items);
        lossSum += loss * batch.size; count += batch.size;
        ST.lossHistory.push(loss);
        batchCount++;

        if (batchCount % CONFIG.lossDrawEvery === 0) {
          drawLoss();
          setStatus(`Epoch ${epoch}/${CONFIG.epochs} — loss ~ ${(lossSum/count).toFixed(4)}`);
          await tf.nextFrame();
        }
      }

      const meanLoss = lossSum / Math.max(1, count);
      console.log(`[epoch ${epoch}/${CONFIG.epochs}] mean loss = ${meanLoss.toFixed(5)}`);
    }

    // финальный рендер графика
    drawLoss();
    setStatus('Training complete ✅');
    btnTest && (btnTest.disabled = false);
    await drawItemPCA();
  }

  // ------------------------------ PCA (500 items, robust) ---------------------
  async function drawItemPCA() {
    if (!pcaCanvas || !ST.model) return;
    if (!pcaCanvas.width || !pcaCanvas.height) { pcaCanvas.width = 640; pcaCanvas.height = 420; }
    const ctx = pcaCanvas.getContext('2d'), W = pcaCanvas.width, H = pcaCanvas.height;
    ctx.clearRect(0,0,W,H);
    setStatus('Computing PCA (up to 500 items)…');

    // 1) Эмбеддинги айтемов + подвыборка
    const I = await ST.model.materializeItemEmbeddings();
    const total = I.shape[0];
    const N = Math.min(CONFIG.pcaItems, total);
    if (N < 2) return; // нечего рисовать
    const idxArr = new Int32Array(N);
    for (let i=0;i<N;i++) idxArr[i] = Math.floor(i * total / N);
    const idx = tf.tensor1d(idxArr, 'int32');
    const X = tf.gather(I, idx);                    // [N, D]

    // 2) Центрирование
    const mean = tf.mean(X, 0, true);               // [1,D]
    const Xc = X.sub(mean);                         // [N,D]

    // 3) PCA: пытаемся через SVD, при ошибке — фолбэк на первые 2 оси
    let pts;
    try {
      const svd = tf.svd ? tf.svd : null;
      if (!svd) throw new Error('svd not available');
      const { v } = tf.svd(Xc, true);               // v: [D,D]
      const V2 = v.slice([0,0],[v.shape[0],2]);     // [D,2]
      const proj = tf.matMul(Xc, V2);               // [N,2]
      pts = await proj.array();
      v.dispose(); V2.dispose(); proj.dispose();
    } catch (e) {
      // фолбэк: просто первые две координаты
      const D = Xc.shape[1];
      const take = Math.min(2, D);
      const proj = Xc.slice([0,0],[N,take]).arraySync();
      // если только одна ось — добавим нулевую вторую
      if (take === 1) pts = proj.map(row => [row[0], 0]);
      else pts = proj;
    }

    // 4) Нормируем к канвасу и рисуем
    const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xR = (xMax - xMin) || 1, yR = (yMax - yMin) || 1;

    ctx.fillStyle = 'rgba(0,122,204,0.65)';
    for (let i=0;i<N;i++){
      const x = ((pts[i][0]-xMin)/xR) * (W-40) + 20;
      const y = ((pts[i][1]-yMin)/yR) * (H-40) + 20;
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI*2); ctx.fill();
    }
    ctx.fillStyle = '#000'; ctx.font = '12px Arial';
    ctx.fillText(`Item Embeddings projection • ${N} items`, 10, 18);

    // 5) чистим временные тензоры
    idx.dispose(); X.dispose(); Xc.dispose(); mean.dispose();
  }

  // --------------------------- Historical Top 10 -------------------------------
  function getTop10Historical() {
    const out = [];
    for (let i = 0; i < ST.stats.nItems; i++) {
      const cnt = ST.itemCnt[i];
      if (cnt < CONFIG.minRatingsForHistoricalTop) continue;
      const avg = ST.itemSum[i] / cnt;
      const rawItemId = ST.revItem[i];
      const meta = ST.items.get(rawItemId);
      out.push({ i, rawItemId, title: meta?.title || `Movie ${rawItemId}`, year: meta?.year ?? '—', rating: avg, cnt });
    }
    out.sort((a,b) => b.rating - a.rating || b.cnt - a.cnt);
    return out.slice(0, 10);
  }

  // --------- Baseline (без DL): контент-бейз по жанрам (косинус) --------------
  function getTopKContentBaseline(uIdx, K = CONFIG.topK) {
    const uVec = ST.userGenresDense[uIdx]; // уже L2-норм
    const seen = ST.userSeen.get(uIdx) || new Set();
    const scores = [];
    for (let i = 0; i < ST.stats.nItems; i++) {
      if (seen.has(i)) continue;
      const gi = ST.itemGenresDense[i]; // L2-норм → dot == cosine
      let s = 0;
      for (let g = 0; g < CONFIG.genreCount; g++) s += (uVec[g] * gi[g]);
      if (s > 0) {
        const rawItemId = ST.revItem[i];
        const meta = ST.items.get(rawItemId);
        scores.push({ i, rawItemId, title: meta?.title || `Movie ${rawItemId}`, year: meta?.year ?? '—', rating: s });
      }
    }
    scores.sort((a,b) => b.rating - a.rating);
    return scores.slice(0, K);
  }

  // --------- DL (two-tower): top-K по dot(u, items) ---------------------------
  async function getTopKDeep(uIdx, K = CONFIG.topK) {
    const seen = ST.userSeen.get(uIdx) || new Set();
    const { indices, scores } = await ST.model.getTopKForUser(uIdx, Math.min(ST.stats.nItems, Math.max(K*5, 200)));
    const out = [];
    for (let k = 0; k < indices.length && out.length < K; k++) {
      const i = indices[k];
      if (seen.has(i)) continue;
      const rawItemId = ST.revItem[i];
      const meta = ST.items.get(rawItemId);
      out.push({ i, rawItemId, title: meta?.title || `Movie ${rawItemId}`, year: meta?.year ?? '—', rating: scores[k] });
    }
    return out;
  }

  // ----------------------------- Render helpers --------------------------------
  function renderRecommendations(list, rawUserId) {
    if (!resultsEl) return console.log('[results]', list);
    if (!list?.length) { resultsEl.innerHTML = '<p>No recommendations.</p>'; return; }

    const rows = list.map((r, idx) => `
      <tr>
        <td>${idx + 1}</td>
        <td>${escapeHtml(r.title)}</td>
        <td>${(r.genres || []).join(', ') || '—'}</td>
        <td>${Number.isFinite(r.rating) ? r.rating.toFixed(3) : '—'}</td>
      </tr>`).join('');

    resultsEl.innerHTML = `
      <h3>Top ${list.length} Recommendations for User ${rawUserId}</h3>
      <table>
        <thead><tr><th>#</th><th>Title</th><th>Genres</th><th>Score</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function renderComparisonTables({ historical, baseline, deep }) {
    const mkTable = (title, rows) => {
      const trs = rows.map((r, idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td>${escapeHtml(r.title)}</td>
          <td>${Number.isFinite(r.rating) ? r.rating.toFixed(3) : '—'}</td>
          <td>${r.year ?? '—'}</td>
        </tr>`).join('');
      return `
        <section class="comp-table">
          <h3>${title}</h3>
          <table>
            <thead><tr><th>rank</th><th>movie</th><th>rating</th><th>year</th></tr></thead>
            <tbody>${trs}</tbody>
          </table>
        </section>`;
    };

    tablesHost.innerHTML = [
      mkTable('Top 10 Rated Movies (Historical)', historical),
      mkTable('Top 10 Recommended Movies without deep learning', baseline),
      mkTable('Top 10 Recommended Movies with deep learning', deep),
    ].join('');
  }

  // ------------------------------- UI flow ------------------------------------
  if (btnLoad) btnLoad.onclick = async () => {
    try {
      setStatus('Loading data…');
      await tf.ready();
      await Promise.all([loadItems(), loadRatings()]);
      buildMappingsAndAggregates();
      buildGenreMatrices();
      setStatus(`Loaded. Users=${ST.stats.nUsers}, Items=${ST.stats.nItems}, Ratings=${ST.stats.nRatings}`);
      btnTrain && (btnTrain.disabled = false);

      // Сразу покажем Historical Top 10 (для пустого сравнения)
      const historical = getTop10Historical();
      renderComparisonTables({ historical, baseline: [], deep: [] });
    } catch (e) {
      console.error(e); setStatus(`Load error: ${e?.message || e}`);
    }
  };

  if (btnTrain) btnTrain.onclick = async () => {
    try {
      if (!ST.stats.nUsers || !ST.stats.nItems) { setStatus('Load data first'); return; }
      buildModel();
      await train();
    } catch (e) {
      console.error(e); setStatus(`Training error: ${e?.message || e}`);
    }
  };

  if (btnTest) btnTest.onclick = async () => {
    try {
      if (!ST.model) { setStatus('Train model first'); return; }
      // выберем юзера с приличной историей
      const rawUser = pickUserRaw(5);
      setStatus(`Scoring for user ${rawUser}…`);

      // DL рекомендации
      const uIdx = ST.userMap.get(rawUser);
      const deep = await getTopKDeep(uIdx, CONFIG.topK);

      // Baseline контент-бейз (без DL)
      const baseline = getTopKContentBaseline(uIdx, CONFIG.topK);

      // Historical Top (один и тот же для всех)
      const historical = getTop10Historical();

      // Отрисуем сравнение внизу
      renderComparisonTables({ historical, baseline, deep });

      // А в основном блоке покажем DL-рекомендации с жанрами
      const withGenres = deep.map(r => ({
        ...r,
        genres: (ST.items.get(ST.revItem[r.i])?.genres || []).map((v,g) => v ? GENRES[g] : null).filter(Boolean)
      }));
      renderRecommendations(withGenres, rawUser);

      setStatus(`Done. Shown top ${withGenres.length} for user ${rawUser}`);
    } catch (e) {
      console.error(e); setStatus(`Test error: ${e?.message || e}`);
    }
  };

  // -------------------------- Helpers: user picker ----------------------------
  function pickUserRaw(minPos = 5) {
    for (let u = 0; u < ST.revUser.length; u++) {
      if ((ST.userSeen.get(u)?.size || 0) >= minPos) return ST.revUser[u];
    }
    return ST.revUser[0];
  }

  // Экспорт для дебага
  window._tt = { ST, CONFIG };

})();
