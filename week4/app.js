// app.js — Two-Tower demo glue (TF.js) — совместим с index.html (loadData/train/test)
// -----------------------------------------------------------------------------
(async function App() {
  'use strict';

  // ------------------------------- Config ------------------------------------
  const CONFIG = {
    embDim: 32,
    userHidden: 64,
    itemHidden: 64,
    learningRate: 0.01,
    l2: 1e-4,
    normalize: true,
    epochs: 10,
    batchSize: 2048,
    posThreshold: 4,
    topK: 10,
    genreCount: 19,
    files: {
      item: 'data/u.item',   // ВАЖНО: файлы лежат в /data
      data: 'data/u.data'
    }
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
    itemGenresDense: null,       // [I,19]
    userGenresDense: null,       // [U,19]
    model: null,
    stats: { nUsers: 0, nItems: 0, nRatings: 0 }
  };

  // ------------------------------- DOM ---------------------------------------
  const $ = s => document.querySelector(s);
  const btnLoad  = $('#loadData');   // из index.html
  const btnTrain = $('#train');      // из index.html
  const btnTest  = $('#test');       // из index.html
  const statusEl = $('#status');     // из index.html
  const resultsEl = $('#results');   // из index.html

  const GENRES = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];

  function setStatus(msg, kind='info') {
    if (!statusEl) return console.log('[status]', msg);
    statusEl.textContent = msg;
  }

  function renderResults(list) {
    if (!resultsEl) return console.log('[results]', list);
    if (!list || !list.length) {
      resultsEl.innerHTML = '<p>No recommendations.</p>';
      return;
    }
    const rows = list.map((r, idx) => `
      <tr>
        <td>${idx + 1}</td>
        <td>${escapeHtml(r.title)}</td>
        <td>${r.genres.join(', ') || '—'}</td>
        <td>${r.score.toFixed(3)}</td>
      </tr>`).join('');
    resultsEl.innerHTML = `
      <h3>Recommendations</h3>
      <table>
        <thead><tr><th>#</th><th>Title</th><th>Genres</th><th>Score</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

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
      if (parts.length < 5 + G) {
        console.warn(`[u.item] skip malformed line ${lineNo + 1} (got ${parts.length})`);
        continue;
      }
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

  function buildMappings() {
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

    // seen + positives
    ST.userSeen.clear(); ST.positives.length = 0;
    for (const { userId, itemId, rating } of ST.interactionsRaw) {
      const u = ST.userMap.get(userId);
      const i = ST.itemMap.get(itemId);
      if (u == null || i == null) continue;

      if (!ST.userSeen.has(u)) ST.userSeen.set(u, new Set());
      ST.userSeen.get(u).add(i);
      if (rating >= CONFIG.posThreshold) ST.positives.push({ u, i });
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
      { lossType: 'softmax', lr: CONFIG.learningRate, userHidden: CONFIG.userHidden,
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
      yield { users: u, items: it, size: (end - start) };
    }
  }

  async function train() {
    if (!ST.model) { setStatus('Model is not initialized'); return; }
    if (!ST.positives.length) { setStatus('No positive pairs to train on'); return; }
    setStatus('Training…');
    let seen = 0;
    for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
      let lossSum = 0, cnt = 0;
      for (const batch of batchIterator(ST.positives, CONFIG.batchSize)) {
        const loss = await ST.model.trainStep(batch.users, batch.items);
        lossSum += loss * batch.size; cnt += batch.size; seen += batch.size;
        if (cnt % (CONFIG.batchSize * 2) === 0) setStatus(`Training… epoch ${epoch}/${CONFIG.epochs}, loss ~ ${ (lossSum/cnt).toFixed(4) }`);
        await tf.nextFrame();
      }
      setStatus(`Epoch ${epoch}/${CONFIG.epochs} done, mean loss = ${(lossSum / Math.max(1,cnt)).toFixed(5)}`);
    }
    setStatus('Training complete ✅');
  }

  function pickUserRaw(minPos = 5) {
    // выбираем юзера с достаточным числом оценённых фильмов
    for (let u = 0; u < ST.revUser.length; u++) {
      if ((ST.userSeen.get(u)?.size || 0) >= minPos) return ST.revUser[u];
    }
    return ST.revUser[0];
  }

  async function recommendForRawUser(rawUserId, K = CONFIG.topK) {
    if (!ST.model) { setStatus('Train a model first'); return []; }
    const u = ST.userMap.get(rawUserId);
    if (u == null) { setStatus('Unknown user id'); return []; }

    const seen = ST.userSeen.get(u) || new Set();
    const { indices, scores } = await ST.model.getTopKForUser(u, Math.min(ST.stats.nItems, Math.max(K * 5, 200)));

    const out = [];
    for (let k = 0; k < indices.length && out.length < K; k++) {
      const i = indices[k];
      if (seen.has(i)) continue;
      const rawItemId = ST.revItem[i];
      const meta = ST.items.get(rawItemId);
      out.push({
        itemIndex: i,
        rawItemId,
        title: meta?.title || `Movie ${rawItemId}`,
        genres: (meta?.genres || []).map((v, g) => v ? GENRES[g] : null).filter(Boolean),
        score: scores[k]
      });
    }
    return out;
  }

  // ------------------------------- UI events ----------------------------------
  if (btnLoad) btnLoad.onclick = async () => {
    try {
      setStatus('Loading data…');
      await tf.ready();
      await Promise.all([loadItems(), loadRatings()]);
      buildMappings();
      buildGenreMatrices();
      setStatus(`Loaded. Users=${ST.stats.nUsers}, Items=${ST.stats.nItems}, Ratings=${ST.stats.nRatings}`);
      if (btnTrain) btnTrain.disabled = false;
    } catch (e) {
      console.error(e); setStatus(`Load error: ${e?.message || e}`);
    }
  };

  if (btnTrain) btnTrain.onclick = async () => {
    try {
      if (!ST.stats.nUsers || !ST.stats.nItems) { setStatus('Load data first'); return; }
      buildModel();
      await train();
      if (btnTest) btnTest.disabled = false;
    } catch (e) {
      console.error(e); setStatus(`Training error: ${e?.message || e}`);
    }
  };

  if (btnTest) btnTest.onclick = async () => {
    try {
      if (!ST.model) { setStatus('Train model first'); return; }
      const rawUser = pickUserRaw(5);
      setStatus(`Scoring for user ${rawUser}…`);
      const recs = await recommendForRawUser(rawUser, CONFIG.topK);
      renderResults(recs);
      setStatus(`Done. Shown top ${recs.length} for user ${rawUser}`);
    } catch (e) {
      console.error(e); setStatus(`Test error: ${e?.message || e}`);
    }
  };

  // Экспорт для дебага
  window._tt = { ST, CONFIG, recommendForRawUser };

})();
