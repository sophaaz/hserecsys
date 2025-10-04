// app.js — Two-Tower TF.js demo (CPU, softmax), live loss chart, 3 tables, PCA(500)
// ---------------------------------------------------------------------------------
(async function App() {
  'use strict';

  // ------------------------------ DOM refs ------------------------------------
  const $ = s => document.querySelector(s);
  const btnLoad  = $('#loadData');
  const btnTrain = $('#train');
  const btnTest  = $('#test');
  const statusEl = $('#status');
  const lossCanvas = $('#lossChart');
  const pcaCanvas  = $('#embeddingChart');
  const resultsEl  = $('#results');

  // Контейнер для трёх сравнительных таблиц (создадим, если нет)
  let tablesHost = $('#comparison-tables');
  if (!tablesHost) {
    tablesHost = document.createElement('div');
    tablesHost.id = 'comparison-tables';
    document.body.appendChild(tablesHost);
  }

  // ------------------------------ Config --------------------------------------
  const CONFIG = {
    // Модель
    embDim: 32,
    userHidden: 64,
    itemHidden: 64,
    normalize: true,
    l2: 1e-4,
    learningRate: 0.002,
    lossType: 'softmax',     // важно: без BPR → никаких Neg-ядер

    // Тренинг (CPU-friendly)
    epochs: 4,
    batchSize: 512,
    lossDrawEvery: 2,        // обновлять график чаще

    // Данные
    files: { item: 'data/u.item', data: 'data/u.data' },
    genreCount: 19,
    posThreshold: 4,         // «позитив» — рейтинг ≥ 4 (fallback до 3 если мало)

    // Рекомендации / валидации
    topK: 10,
    minRatingsForHistoricalTop: 50,

    // PCA
    pcaItems: 500,
    pcaPowerIters: 40
  };

  const GENRES = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];

  // ------------------------------ State ---------------------------------------
  const ST = {
    items: new Map(),          // rawItemId -> {title, year, genres[19]}
    interactionsRaw: [],       // [{userId,itemId,rating}]
    // индексация
    userMap: new Map(), revUser: [],   // rawUserId->u, u->rawUserId
    itemMap: new Map(), revItem: [],   // rawItemId->i, i->rawItemId
    // обучающие пары и вспомогательное
    positives: [],             // [{u,i}] для рейтингов ≥ posThreshold
    userSeen: new Map(),       // u -> Set(i)
    itemGenresDense: null,     // Array[I][19] (L2-норм)
    userGenresDense: null,     // Array[U][19] (L2-норм)
    // агрегаты для historical top
    itemSum: null,             // Float32Array[I]
    itemCnt: null,             // Uint32Array[I]
    // модель/метрики
    model: null,
    lossHistory: [],
    stats: { nUsers: 0, nItems: 0, nRatings: 0 },
    backend: 'unknown'
  };

  // ------------------------------ Utils ---------------------------------------
  const setStatus = msg => { if (statusEl) statusEl.textContent = msg; console.log('[status]', msg); };
  const escapeHtml = s => String(s).replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  // Надёжная инициализация TF.js: форсируем CPU, чтобы исключить WebGL-краши
  async function ensureTF_CPU() {
    if (typeof tf === 'undefined') throw new Error('TensorFlow.js не загружен (нет <script src="tf.min.js">).');
    try { await tf.setBackend('cpu'); } catch {}
    await tf.ready();
    try { ST.backend = tf.getBackend(); } catch { ST.backend = 'cpu'; }
  }

  // ------------------------------ Data loading --------------------------------
  // u.item: movieId|title|...|19 жанров (0/1)
  async function loadItems() {
    const res = await fetch(CONFIG.files.item);
    if (!res.ok) throw new Error(`Failed to fetch ${CONFIG.files.item}: ${res.status}`);
    const text = await res.text();
    const lines = text.split(/\r?\n/);
    const G = CONFIG.genreCount;

    ST.items.clear();
    for (const line0 of lines) {
      const line = line0.trim();
      if (!line) continue;
      const parts = line.split('|');
      if (parts.length < 5 + G) continue;

      const rawItemId = parseInt(parts[0], 10);
      const titleRaw = String(parts[1] ?? '');
      const yearMatch = /\((\d{4})\)\s*$/.exec(titleRaw);
      const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
      const title = titleRaw.replace(/\(\d{4}\)\s*$/, '').trim();

      const flags = Array.from({length:G}, (_,g)=> (parts[5+g]==='1'||parts[5+g]===1)?1:0);
      ST.items.set(rawItemId, { title, year, genres: flags });
    }
  }

  // u.data: userId\titemId\trating\ttimestamp
  async function loadRatings() {
    const res = await fetch(CONFIG.files.data);
    if (!res.ok) throw new Error(`Failed to fetch ${CONFIG.files.data}: ${res.status}`);
    const text = await res.text();
    const lines = text.split(/\r?\\n/);

    ST.interactionsRaw.length = 0;
    for (const line0 of lines) {
      const line = (line0 || '').trim();
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
      let u=0; for (const raw of users){ ST.userMap.set(raw,u++); ST.revUser.push(raw); }
    }
    // items
    {
      const items = Array.from(ST.items.keys()).sort((a,b)=>a-b);
      let i=0; for (const raw of items){ ST.itemMap.set(raw,i++); ST.revItem.push(raw); }
    }

    const I = ST.revItem.length;
    ST.itemSum = new Float32Array(I);
    ST.itemCnt = new Uint32Array(I);

    ST.positives.length = 0;
    ST.userSeen.clear();

    for (const {userId,itemId,rating} of ST.interactionsRaw) {
      const u = ST.userMap.get(userId);
      const i = ST.itemMap.get(itemId);
      if (u==null || i==null) continue;

      if (!ST.userSeen.has(u)) ST.userSeen.set(u,new Set());
      ST.userSeen.get(u).add(i);

      if (rating >= CONFIG.posThreshold) ST.positives.push({u,i});

      ST.itemSum[i] += rating;
      ST.itemCnt[i] += 1;
    }

    ST.stats.nUsers = ST.revUser.length;
    ST.stats.nItems = ST.revItem.length;
    ST.stats.nRatings = ST.interactionsRaw.length;

    // fallback, если мало позитивов → опускаем порог до 3
    if (ST.positives.length < 1000) {
      ST.positives.length = 0;
      ST.userSeen.clear();
      for (const {userId,itemId,rating} of ST.interactionsRaw) {
        const u = ST.userMap.get(userId);
        const i = ST.itemMap.get(itemId);
        if (!ST.userSeen.has(u)) ST.userSeen.set(u,new Set());
        ST.userSeen.get(u).add(i);
        if (rating >= 3) ST.positives.push({u,i});
      }
    }
  }

  function buildGenreMatrices() {
    const U = ST.revUser.length;
    const I = ST.revItem.length;
    const G = CONFIG.genreCount;

    // itemGenresDense: [I,19]
    const itemGenres = Array.from({length:I}, ()=>Array(G).fill(0));
    for (let i=0;i<I;i++){
      const raw = ST.revItem[i];
      const meta = ST.items.get(raw);
      itemGenres[i] = meta?.genres ? meta.genres.slice(0,G) : Array(G).fill(0);
    }

    // userGenresDense: [U,19] — сумма жанров по позитивам
    const userGenres = Array.from({length:U}, ()=>Array(G).fill(0));
    for (const {u,i} of ST.positives) {
      const g = itemGenres[i];
      for (let k=0;k<G;k++) userGenres[u][k] += g[k];
    }

    const l2row = row => {
      let s=0; for (const v of row) s+=v*v;
      const d = Math.sqrt(s) || 1;
      return row.map(v=>v/d);
    };

    ST.itemGenresDense = itemGenres.map(l2row);
    ST.userGenresDense = userGenres.map(l2row);
  }

  // ------------------------------ Model wiring --------------------------------
  function buildModel() {
    if (ST.model) { ST.model.dispose(); ST.model = null; }
    ST.model = new TwoTowerModel(
      ST.revUser.length,
      ST.revItem.length,
      CONFIG.embDim,
      {
        lossType: CONFIG.lossType,
        lr: CONFIG.learningRate,
        userHidden: CONFIG.userHidden,
        itemHidden: CONFIG.itemHidden,
        l2: CONFIG.l2,
        normalize: CONFIG.normalize
      }
    );
    ST.model.setFeatures({ itemGenres: ST.itemGenresDense, userGenres: ST.userGenresDense });
  }

  function* batchIterator(pairs, batch) {
    // перетасовка
    const idx = pairs.map((_,i)=>i);
    for (let i=idx.length-1;i>0;i--){ const j=(Math.random()*(i+1))|0; [idx[i],idx[j]]=[idx[j],idx[i]]; }
    for (let s=0;s<idx.length;s+=batch) {
      const e = Math.min(idx.length, s+batch);
      const u = new Int32Array(e-s);
      const it= new Int32Array(e-s);
      for (let b=0;b<e-s;b++){ const p = pairs[idx[s+b]]; u[b]=p.u; it[b]=p.i; }
      yield { users: u, items: it, size: e-s };
    }
  }

  // ------------------------------ Live loss chart -----------------------------
  function drawLoss() {
    if (!lossCanvas) return;
    if (!lossCanvas.width || !lossCanvas.height) { lossCanvas.width = 680; lossCanvas.height = 240; }
    const ctx = lossCanvas.getContext('2d');
    const W = lossCanvas.width, H = lossCanvas.height;
    ctx.clearRect(0,0,W,H);

    if (!ST.lossHistory.length) return;

    const mn = Math.min(...ST.lossHistory);
    const mx = Math.max(...ST.lossHistory);
    const R  = (mx - mn) || 1;

    // оси
    ctx.strokeStyle = '#d0d0d0'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(40,10); ctx.lineTo(40,H-28); ctx.lineTo(W-10,H-28); ctx.stroke();

    // линия лосса
    ctx.strokeStyle = '#2d7ef7'; ctx.lineWidth = 2;
    ctx.beginPath();
    ST.lossHistory.forEach((v,i)=>{
      const x = 40 + (i/(ST.lossHistory.length-1))*(W-50);
      const y = (H-28) - ((v - mn)/R)*(H-40);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();

    // подписи
    ctx.fillStyle = '#333'; ctx.font = '12px system-ui, -apple-system, Arial';
    ctx.fillText(`min ${mn.toFixed(4)}`, 45, 14);
    ctx.fillText(`max ${mx.toFixed(4)}`, 120, 14);
    ctx.fillText('batches →', W-88, H-10);
  }

  // ------------------------------ Training ------------------------------------
  async function train() {
    await ensureTF_CPU();
    if (!ST.model) { setStatus('Model is not initialized'); return; }
    if (!ST.positives.length) { setStatus('No positive pairs to train on'); return; }

    // блокируем кнопки
    btnLoad && (btnLoad.disabled = true);
    btnTrain && (btnTrain.disabled = true);
    btnTest && (btnTest.disabled = true);

    const totalBatches = Math.ceil(ST.positives.length / CONFIG.batchSize) * CONFIG.epochs;
    let doneBatches = 0;
    ST.lossHistory.length = 0;

    for (let ep=1; ep<=CONFIG.epochs; ep++) {
      let sum=0, cnt=0, iBatch=0;

      for (const bt of batchIterator(ST.positives, CONFIG.batchSize)) {
        const loss = await ST.model.trainStep(bt.users, bt.items);  // softmax loss внутри модели
        sum += loss * bt.size; cnt += bt.size;
        ST.lossHistory.push(loss);
        doneBatches++; iBatch++;

        if (iBatch % CONFIG.lossDrawEvery === 0) {
          drawLoss();
          const pct = Math.round((doneBatches/totalBatches)*100);
          setStatus(`Epoch ${ep}/${CONFIG.epochs} — loss ~ ${(sum/cnt).toFixed(4)} — ${pct}% (backend: ${ST.backend})`);
          await tf.nextFrame();               // даём браузеру отрисовать
        }
      }

      // микро-пауза между эпохами, чтоб UI не залип
      await new Promise(r=>setTimeout(r,0));
    }

    drawLoss();
    setStatus('Training complete ✅');

    // разблокируем кнопки
    btnLoad && (btnLoad.disabled = false);
    btnTrain && (btnTrain.disabled = false);
    btnTest && (btnTest.disabled = false);
  }

  // ------------------------------ PCA (500 items) -----------------------------
  async function drawItemPCA() {
    await ensureTF_CPU();
    if (!pcaCanvas || !ST.model) return;
    if (!pcaCanvas.width || !pcaCanvas.height) { pcaCanvas.width = 680; pcaCanvas.height = 420; }
    const ctx = pcaCanvas.getContext('2d');
    const W = pcaCanvas.width, H = pcaCanvas.height;
    ctx.clearRect(0,0,W,H);

    setStatus(`Computing PCA (500 items)… (backend: ${ST.backend})`);

    // 1) Эмбеддинги айтемов + равномерная подвыборка N
    const I = await ST.model.materializeItemEmbeddings(); // [M, D]
    const total = I.shape[0];
    const N = Math.min(CONFIG.pcaItems, total);
    if (N < 2) return;

    const idxArr = new Int32Array(N);
    for (let i=0;i<N;i++) idxArr[i] = Math.floor(i*total/N);
    const idx = tf.tensor1d(idxArr,'int32');
    const X = tf.gather(I, idx);                  // [N, D]

    // 2) Центрирование
    const mean = tf.mean(X, 0, true);             // [1, D]
    const Xc   = X.sub(mean);                     // [N, D]

    // 3) Ковариация и степенной метод => 2 главные компоненты
    const Cov  = tf.matMul(Xc, Xc, true, false);  // [D, D]

    const v1 = await powerIteration(Cov, CONFIG.pcaPowerIters);
    const Cov_v1 = tf.matMul(Cov, v1);
    const l1 = tf.sum(tf.mul(v1, Cov_v1));

    const v1T = v1.transpose();
    const outer1 = tf.matMul(v1, v1T);
    const Cov2 = tf.sub(Cov, outer1.mul(l1));

    const v2 = await powerIteration(Cov2, CONFIG.pcaPowerIters);

    const V2 = tf.concat([v1, v2], 1);            // [D,2]
    const proj = tf.matMul(Xc, V2);               // [N,2]
    const pts = await proj.array();

    // 4) Нормировка и отрисовка
    const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xR = (xMax-xMin)||1, yR=(yMax-yMin)||1;

    ctx.fillStyle = 'rgba(0,122,204,0.65)';
    for (let i=0;i<N;i++){
      const x = ((pts[i][0]-xMin)/xR) * (W-40) + 20;
      const y = ((pts[i][1]-yMin)/yR) * (H-40) + 20;
      ctx.beginPath(); ctx.arc(x,y,2.4,0,Math.PI*2); ctx.fill();
    }
    ctx.fillStyle = '#333'; ctx.font = '12px system-ui, -apple-system, Arial';
    ctx.fillText(`Item Embeddings projection • ${N} items (PCA)`, 10, 18);

    // очистка временных тензоров
    idx.dispose(); X.dispose(); Xc.dispose(); mean.dispose();
    Cov.dispose(); v1.dispose(); Cov_v1.dispose(); l1.dispose(); v1T.dispose(); outer1.dispose(); Cov2.dispose(); v2.dispose(); V2.dispose(); proj.dispose();
  }

  function powerIteration(C, iters=40){
    return tf.tidy(()=> {
      let v = tf.randomNormal([C.shape[0],1]);
      for (let t=0;t<iters;t++){
        v = tf.matMul(C, v);
        v = v.div(tf.norm(v).add(1e-8));
      }
      return v;
    });
  }

  // ------------------------------ Historical Top ------------------------------
  function getTop10Historical() {
    const out = [];
    for (let i=0;i<ST.revItem.length;i++){
      const cnt = ST.itemCnt[i];
      if (cnt < CONFIG.minRatingsForHistoricalTop) continue;
      const avg = ST.itemSum[i] / cnt;
      const raw = ST.revItem[i];
      const meta = ST.items.get(raw);
      out.push({ i, rawItemId: raw, title: meta?.title || `Movie ${raw}`, year: meta?.year ?? '—', rating: avg, cnt });
    }
    out.sort((a,b)=> b.rating - a.rating || b.cnt - a.cnt);
    return out.slice(0,10);
  }

  // ------------------------------ Baseline (без DL) ---------------------------
  function getTopKContentBaseline(uIdx, K=CONFIG.topK) {
    const seen = ST.userSeen.get(uIdx) || new Set();
    const uVec = ST.userGenresDense[uIdx]; // L2-норм
    const scores = [];
    for (let i=0;i<ST.revItem.length;i++){
      if (seen.has(i)) continue;
      const gi = ST.itemGenresDense[i];
      let s=0; for (let g=0; g<CONFIG.genreCount; g++) s += uVec[g]*gi[g]; // косинус на L2-норм. векторах
      if (s>0){
        const raw = ST.revItem[i]; const meta = ST.items.get(raw);
        scores.push({ i, rawItemId: raw, title: meta?.title || `Movie ${raw}`, year: meta?.year ?? '—', rating: s });
      }
    }
    scores.sort((a,b)=> b.rating - a.rating);
    return scores.slice(0, K);
  }

  // ------------------------------ Deep (Two-Tower) ----------------------------
  async function getTopKDeep(uIdx, K=CONFIG.topK){
    await ensureTF_CPU();
    const seen = ST.userSeen.get(uIdx) || new Set();
    const { indices, scores } = await ST.model.getTopKForUser(uIdx, Math.min(ST.revItem.length, Math.max(K*5, 200)));
    const out = [];
    for (let k=0;k<indices.length && out.length<K;k++){
      const i = indices[k]; if (seen.has(i)) continue;
      const raw = ST.revItem[i]; const meta = ST.items.get(raw);
      out.push({ i, rawItemId: raw, title: meta?.title || `Movie ${raw}`, year: meta?.year ?? '—', rating: scores[k] });
    }
    return out;
  }

  // ------------------------------ Rendering -----------------------------------
  function renderComparisonTables({historical, baseline, deep}) {
    const mk = (title, rows) => `
      <section class="comp-table">
        <h3>${escapeHtml(title)}</h3>
        <table>
          <thead><tr><th>rank</th><th>movie</th><th>rating</th><th>year</th></tr></thead>
          <tbody>
            ${rows.map((r,idx)=>`
              <tr>
                <td>${idx+1}</td>
                <td>${escapeHtml(r.title)}</td>
                <td>${Number.isFinite(r.rating)?r.rating.toFixed(3):'—'}</td>
                <td>${r.year ?? '—'}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </section>`;
    tablesHost.innerHTML = [
      mk('Top 10 Rated Movies (Historical)', historical),
      mk('Top 10 Recommended Movies without deep learning', baseline),
      mk('Top 10 Recommended Movies with deep learning', deep)
    ].join('');
  }

  function renderRecommendations(list, rawUserId) {
    if (!resultsEl) return;
    if (!list?.length) { resultsEl.innerHTML = '<p>No recommendations.</p>'; return; }
    const rows = list.map((r,idx)=>`
      <tr>
        <td>${idx+1}</td>
        <td>${escapeHtml(r.title)}</td>
        <td>${Number.isFinite(r.rating)?r.rating.toFixed(3):'—'}</td>
        <td>${r.year ?? '—'}</td>
      </tr>`).join('');
    resultsEl.innerHTML = `
      <h3>Top ${list.length} Recommendations for User ${rawUserId}</h3>
      <table>
        <thead><tr><th>#</th><th>Title</th><th>Score</th><th>Year</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  // ------------------------------ UI handlers ---------------------------------
  btnLoad && (btnLoad.onclick = async () => {
    try{
      setStatus('Loading data…');
      await ensureTF_CPU();
      await Promise.all([loadItems(), loadRatings()]);
      buildMappingsAndAggregates();
      buildGenreMatrices();
      setStatus(`Loaded. Users=${ST.revUser.length}, Items=${ST.revItem.length}, Ratings=${ST.stats.nRatings} (backend: ${ST.backend})`);
      btnTrain && (btnTrain.disabled = false);

      // сразу отрисуем historical top
      const historical = getTop10Historical();
      renderComparisonTables({ historical, baseline: [], deep: [] });
    }catch(e){ console.error(e); setStatus(`Load error: ${e?.message||e}`); }
  });

  btnTrain && (btnTrain.onclick = async () => {
    try{
      if (!ST.revUser.length || !ST.revItem.length){ setStatus('Load data first'); return; }
      buildModel();
      await train();
    }catch(e){ console.error(e); setStatus(`Training error: ${e?.message||e}`); }
  });

  btnTest && (btnTest.onclick = async () => {
    try{
      if (!ST.model){ setStatus('Train model first'); return; }
      await ensureTF_CPU();

      const rawUser = pickUserRaw(5);
      const uIdx = ST.userMap.get(rawUser);

      const deep = await getTopKDeep(uIdx, CONFIG.topK);
      const baseline = getTopKContentBaseline(uIdx, CONFIG.topK);
      const historical = getTop10Historical();

      renderComparisonTables({ historical, baseline, deep });
      renderRecommendations(deep, rawUser);

      await drawItemPCA();
      setStatus(`Done (backend: ${ST.backend})`);
    }catch(e){ console.error(e); setStatus(`Test error: ${e?.message||e}`); }
  });

  function pickUserRaw(minPos=5){
    for (let u=0; u<ST.revUser.length; u++){
      if ((ST.userSeen.get(u)?.size || 0) >= minPos) return ST.revUser[u];
    }
    return ST.revUser[0];
  }

  // ------------------------------ Global error log ----------------------------
  window.addEventListener('error', e=>{
    console.error('[GlobalError]', e.message, e.error?.stack||'');
  });

})();
