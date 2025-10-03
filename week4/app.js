// app.js
// -----------------------------------------------------------------------------
// App glue-code for Two-Tower Recommender (TF.js) running fully in-browser.
// - Загружает MovieLens файлы (u.item, u.data)
// - Собирает плотные индексы пользователей/фильмов
// - Строит жанровые матрицы features (itemGenres, userGenres)
// - Инициализирует TwoTowerModel, тренирует по in-batch softmax
// - Делает top-K рекомендации для выбранного пользователя
//
// Файл не зависит жёстко от разметки: если элементы UI есть — подключаемся,
// если нет — всё работает через console/log.
// Требуемый порядок подключений в index.html:
//   <script src="tf.min.js"></script>
//   <script src="two-tower.js"></script>
//   <script src="app.js"></script>
// -----------------------------------------------------------------------------

(async function AppBootstrap() {
  'use strict';

  // ------------------------------- Config ------------------------------------
  const CONFIG = {
    embDim: 32,            // выходная размерность башен
    userHidden: 64,        // ширина скрытого слоя в user-tower
    itemHidden: 64,        // ширина скрытого слоя в item-tower
    learningRate: 0.01,    // Adam lr
    l2: 1e-4,              // L2-регуляризация
    normalize: true,       // L2-нормировка выходов башен (косинус-подобный скор)
    epochs: 10,
    batchSize: 2048,       // для in-batch softmax батч — это и позитивы, и негативы
    posThreshold: 4,       // считаем позитивом rating >= 4
    topK: 10,
    genreCount: 19,        // в u.item 19 жанровых флагов (колонки 5..23)
    files: {
      item: './u.item',
      data: './u.data'
    }
  };

  // ------------------------------- State -------------------------------------
  const ST = {
    // Сырые словари
    items: new Map(),            // rawItemId -> { title, year, genres[19] }
    interactionsRaw: [],         // { userId, itemId, rating }

    // Плотные индексы
    userMap: new Map(),          // rawUserId -> u ∈ [0..U-1]
    itemMap: new Map(),          // rawItemId -> i ∈ [0..I-1]
    revUser: [],                 // u -> rawUserId
    revItem: [],                 // i -> rawItemId

    // Позитивные пары (для обучения two-tower)
    positives: [],               // {u, i}

    // Для фильтрации просмотренного при рекомендациях
    userSeen: new Map(),         // u -> Set(i)

    // Жанровые фичи
    itemGenresDense: null,       // [I, 19] массив массивов (или Tensor2D в setFeatures)
    userGenresDense: null,       // [U, 19] агрегат по историям

    // Модель
    model: null,                 // instance of TwoTowerModel

    // Статистика
    stats: { nUsers: 0, nItems: 0, nRatings: 0 }
  };

  // ------------------------------ DOM helpers --------------------------------
  const $ = s => document.querySelector(s);
  const on = (el, ev, fn) => el && el.addEventListener(ev, fn);

  function setStatus(text, ok = null) {
    const el = $('#status');
    if (!el) { console.log('[status]', text); return; }
    el.textContent = text;
    el.className = 'muted';
    if (ok === true) el.classList.add('status-ok');
    if (ok === false) el.classList.add('status-err');
  }

  function setTrainUI({ epoch, epochs, loss }) {
    const pe = $('#train-epoch');
    const tr = $('#train-trainrmse'); // переиспользуем поле под loss (для простоты)
    const vr = $('#train-valrmse');
    if (pe) pe.textContent = `epoch ${epoch}/${epochs}`;
    if (tr) tr.textContent = `loss: ${Number.isFinite(loss) ? loss.toFixed(4) : '–'}`;
    if (vr) vr.textContent = `val RMSE: –`; // в two-tower сейчас не считаем RMSE
  }

  function setProgress(pct) {
    const bar = $('#train-progress .bar');
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  }

  function populateUsersSelect() {
    const sel = $('#user-select');
    if (!sel) return;
    sel.innerHTML = '<option value="">— select user —</option>';
    for (const rawId of ST.revUser.map((raw, u) => raw)) {
      const opt = document.createElement('option');
      opt.value = String(rawId);
      opt.textContent = String(rawId);
      sel.appendChild(opt);
    }
  }

  function renderTopK(rawUserId, top) {
    const tbody = $('#results-table tbody');
    if (!tbody) { console.log('[topK]', top); return; }
    tbody.innerHTML = '';
    if (!top || top.length === 0) {
      const tr = document.createElement('tr'); const td = document.createElement('td');
      td.colSpan = 5; td.textContent = 'No recommendations.'; tr.appendChild(td); tbody.appendChild(tr); return;
    }
    top.forEach((r, idx) => {
      const tr = document.createElement('tr');

      const tdRank = document.createElement('td'); tdRank.textContent = String(idx + 1); tr.appendChild(tdRank);
      const tdTitle = document.createElement('td'); tdTitle.textContent = r.title; tr.appendChild(tdTitle);
      const tdGenres = document.createElement('td'); tdGenres.textContent = r.genres.join(', ') || '—'; tr.appendChild(tdGenres);
      const tdPred = document.createElement('td'); tdPred.textContent = r.score.toFixed(3); tr.appendChild(tdPred);
      const tdExplain = document.createElement('td'); tdExplain.textContent = 'two-tower dot score'; tr.appendChild(tdExplain);

      tbody.appendChild(tr);
    });
  }

  // ------------------------------ Data loading --------------------------------
  // u.item: movieId|title|releaseDate|videoReleaseDate|imdbURL|g0..g18 (19 жанров)
async function loadItems() {
  const res = await fetch(CONFIG.files.item);
  if (!res.ok) throw new Error(`Failed to fetch ${CONFIG.files.item}: ${res.status}`);
  const text = await res.text();
  const lines = text.split(/\r?\n/);
  const G = CONFIG.genreCount;

  ST.items.clear();

  for (let lineNo = 0; lineNo < lines.length; lineNo++) {
    const raw = lines[lineNo];
    const line = raw.trim();
    if (!line) continue;

    const parts = line.split('|');
    // строгая валидация: в ML-100K должно быть минимум 5 + 19 полей
    if (parts.length < 5 + G) {
      // можно залогировать и пропустить «битую» строку, чтобы не падать
      console.warn(`[u.item] skip malformed line ${lineNo + 1}: expected >= ${5 + G} fields, got ${parts.length}`);
      continue;
    }

    const rawItemId = parseInt(parts[0], 10);
    // title всегда приводим к строке, даже если поле пустое
    const titleRaw = String(parts[1] ?? '');
    const yearMatch = /\((\d{4})\)\s*$/.exec(titleRaw);    // безопасный exec вместо .match на undefined
    const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
    const title = titleRaw.replace(/\(\d{4}\)\s*$/, '').trim();

    // жанры: колонки 5..(5+G-1)
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
    const lines = text.split('\n');

    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      const [uS, iS, rS] = line.split('\t');
      if (!uS || !iS || !rS) continue;
      const userId = parseInt(uS, 10);
      const itemId = parseInt(iS, 10);
      const rating = parseFloat(rS);
      if (!Number.isFinite(userId) || !Number.isFinite(itemId) || !Number.isFinite(rating)) continue;
      if (!ST.items.has(itemId)) continue; // фильтрация фильмов без метаданных
      ST.interactionsRaw.push({ userId, itemId, rating });
    }
  }

  // Плотные индексы raw→dense и обратно
  function buildMappings() {
    // Пользователи
    {
      const users = new Set(ST.interactionsRaw.map(x => x.userId));
      let idx = 0;
      for (const u of users) { ST.userMap.set(u, idx++); ST.revUser.push(u); }
    }
    // Фильмы
    {
      const items = Array.from(ST.items.keys());
      items.sort((a, b) => a - b);
      let idx = 0;
      for (const i of items) { ST.itemMap.set(i, idx++); ST.revItem.push(i); }
    }

    // Заполняем userSeen и positives
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

  // Матрица жанров фильмов [I, 19] + агрегат жанров пользователей [U, 19]
  function buildGenreMatrices() {
    const U = ST.stats.nUsers;
    const I = ST.stats.nItems;
    const G = CONFIG.genreCount;

    // itemGenres: dense-порядок индексов
    const itemGenres = Array.from({ length: I }, () => Array(G).fill(0));
    for (let i = 0; i < I; i++) {
      const rawItemId = ST.revItem[i];
      const meta = ST.items.get(rawItemId);
      itemGenres[i] = meta?.genres ? meta.genres.slice(0, G) : Array(G).fill(0);
    }

    // userGenres: суммируем жанры положительных фильмов (rating>=posThreshold)
    const userGenres = Array.from({ length: U }, () => Array(G).fill(0));
    for (const { u, i } of ST.positives) {
      const gs = itemGenres[i];
      for (let g = 0; g < G; g++) userGenres[u][g] += gs[g];
    }

    // L2-нормировка строк для стабильности
    const l2 = row => {
      let s = 0; for (const v of row) s += v * v;
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
      ST.stats.nUsers,
      ST.stats.nItems,
      CONFIG.embDim,
      {
        lossType: 'softmax',
        lr: CONFIG.learningRate,
        userHidden: CONFIG.userHidden,
        itemHidden: CONFIG.itemHidden,
        l2: CONFIG.l2,
        normalize: CONFIG.normalize
      }
    );
    ST.model.setFeatures({
      itemGenres: ST.itemGenresDense,
      userGenres: ST.userGenresDense
    });
  }

  // ------------------------------- Training -----------------------------------
  function* batchIterator(pairs, batchSize) {
    // Перемешиваем индексы и отдаём батчи {users[], items[]}
    const idx = pairs.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let start = 0; start < idx.length; start += batchSize) {
      const end = Math.min(idx.length, start + batchSize);
      const u = new Int32Array(end - start);
      const it = new Int32Array(end - start);
      for (let b = 0; b < end - start; b++) {
        const p = pairs[idx[start + b]];
        u[b] = p.u; it[b] = p.i;
      }
      yield { users: u, items: it, size: (end - start), pct: end / idx.length };
    }
  }

  async function train() {
    if (!ST.model) { setStatus('Model is not initialized', false); return; }
    if (ST.positives.length === 0) { setStatus('No positive pairs to train on', false); return; }

    setStatus('Training…');
    setProgress(0);
    for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
      let lossSum = 0, count = 0;

      for (const batch of batchIterator(ST.positives, CONFIG.batchSize)) {
        const loss = await ST.model.trainStep(batch.users, batch.items);
        lossSum += loss * batch.size; count += batch.size;
        setTrainUI({ epoch, epochs: CONFIG.epochs, loss });
        setProgress(((epoch - 1) / CONFIG.epochs + batch.pct / CONFIG.epochs) * 100);
        // подышим, чтобы UI не зависал
        await tf.nextFrame();
      }

      const meanLoss = lossSum / Math.max(1, count);
      console.log(`[epoch ${epoch}/${CONFIG.epochs}] loss=${meanLoss.toFixed(5)}`);
    }
    setProgress(100);
    setStatus('Training complete', true);
  }

  // ----------------------------- Recommendations ------------------------------
  async function recommendForRawUser(rawUserId, K = CONFIG.topK) {
    if (!ST.model) { setStatus('Train a model first', false); return []; }
    const u = ST.userMap.get(rawUserId);
    if (u == null) { setStatus('Unknown user id', false); return []; }

    // Получаем top-K по dot(u, all items), затем выбрасываем уже просмотренные
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

  // GENRES справочник (для печати жанров в таблице)
  const GENRES = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];

  // ------------------------------- UI binding ---------------------------------
  on(window, 'load', async () => {
    // Вешаем обработчики, если элементы существуют
    on($('#btn-load'), 'click', async () => {
      try {
        setStatus('Loading u.item and u.data…');
        await tf.ready();
        await Promise.all([loadItems(), loadRatings()]);
        buildMappings();
        buildGenreMatrices();
        populateUsersSelect();
        setStatus(`Loaded. Users=${ST.stats.nUsers}, Items=${ST.stats.nItems}, Ratings=${ST.stats.nRatings}`, true);
        // Отрисуем счётчики, если такая панель есть
        const su = $('#stat-users'), si = $('#stat-items'), sr = $('#stat-ratings');
        if (su) su.textContent = String(ST.stats.nUsers);
        if (si) si.textContent = String(ST.stats.nItems);
        if (sr) sr.textContent = String(ST.stats.nRatings);
      } catch (e) {
        console.error(e); setStatus(e.message || 'Load error', false);
      }
    });

    on($('#btn-train'), 'click', async () => {
      try {
        if (!ST.stats.nUsers || !ST.stats.nItems) { setStatus('Load data first', false); return; }
        // Обновим конфиг из инпутов, если они есть
        const getNum = (sel, dflt) => {
          const el = $(sel); if (!el) return dflt;
          const v = parseFloat(String(el.value).replace(',', '.'));
          return Number.isFinite(v) ? v : dflt;
        };
        CONFIG.embDim       = getNum('#param-k', CONFIG.embDim);
        CONFIG.epochs       = getNum('#param-epochs', CONFIG.epochs);
        CONFIG.batchSize    = getNum('#param-batch', CONFIG.batchSize);
        CONFIG.learningRate = getNum('#param-lr', CONFIG.learningRate);
        CONFIG.l2           = getNum('#param-lambda', CONFIG.l2);

        buildModel();
        await train();
      } catch (e) {
        console.error(e); setStatus(e.message || 'Training error', false);
      }
    });

    on($('#btn-recommend'), 'click', async () => {
      try {
        const sel = $('#user-select');
        const raw = sel ? parseInt(sel.value, 10) : NaN;
        if (!Number.isFinite(raw)) { setStatus('Select a user', false); return; }
        const topNEl = $('#topn');
        const K = topNEl ? Math.max(1, Math.min(50, parseInt(topNEl.value, 10) || CONFIG.topK)) : CONFIG.topK;
        setStatus('Scoring…');
        const recs = await recommendForRawUser(raw, K);
        renderTopK(raw, recs);
        setStatus('Done', true);
      } catch (e) {
        console.error(e); setStatus(e.message || 'Recommend error', false);
      }
    });
  });

  // ------------------------------- Expose for debug ---------------------------
  window._tt = { ST, CONFIG, recommendForRawUser, buildModel, train };

})();

