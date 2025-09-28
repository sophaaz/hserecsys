// script.js
// Matrix Factorization (biased) in-browser with TensorFlow.js
// r_hat(u,i) = mu + bu[u] + bi[i] + dot(P[u], Q[i])
// Полный файл с правками: корректный парсинг чисел с запятой,
// явный await tf.ready(), статусы, проверки сплита, аккуратная утилизация памяти.

// ---------------- App State ----------------
const state = {
  loaded: false,
  trained: false,
  stopRequested: false,

  // Data dimensions
  U: 0,
  I: 0,

  // Hyperparams (defaults; редактируются в UI)
  k: 16,
  epochs: 15,
  batchSize: 2048,
  lr: 0.01,
  lambda: 1e-4,

  // Train/val split indices (Int32Array of indices into ratingsTriples)
  split: { trainIdx: null, valIdx: null },

  // TF Variables for MF
  P: null,   // [U, k]
  Q: null,   // [I, k]
  bu: null,  // [U]
  bi: null,  // [I]
  mu: null,  // scalar

  optimizer: null
};

// ---------------- UI Helpers ----------------
const $ = sel => document.querySelector(sel);

function setStatus(text, ok = null) {
  const el = $('#status');
  el.textContent = text;
  el.className = 'muted';
  if (ok === true) el.classList.add('status-ok');
  if (ok === false) el.classList.add('status-err');
}

function updateStatsUI() {
  $('#stat-users').textContent = STATS.nUsers.toString();
  $('#stat-items').textContent = STATS.nItems.toString();
  $('#stat-ratings').textContent = STATS.nRatings.toString();
  $('#stat-mu').textContent = STATS.mean.toFixed(3);
}

function setRecommendControlsEnabled(on) {
  $('#user-select').disabled = !on;
  $('#btn-recommend').disabled = !on;
}

function setProgress(percent) {
  const bar = $('#train-progress .bar');
  bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function setTrainInfo(epoch, epochs, trainRMSE, valRMSE) {
  $('#train-epoch').textContent = `epoch ${epoch}/${epochs}`;
  $('#train-trainrmse').textContent = `train RMSE: ${Number.isFinite(trainRMSE) ? trainRMSE.toFixed(4) : '–'}`;
  $('#train-valrmse').textContent = `val RMSE: ${Number.isFinite(valRMSE) ? valRMSE.toFixed(4) : '–'}`;
}

// ---------------- Initialization ----------------
window.addEventListener('load', () => {
  // Wire buttons
  $('#btn-load').addEventListener('click', onLoadData);
  $('#btn-train').addEventListener('click', onTrain);
  $('#btn-cancel').addEventListener('click', onCancel);
  $('#btn-recommend').addEventListener('click', onRecommend);

  // Defaults → UI
  $('#param-k').value = state.k;
  $('#param-epochs').value = state.epochs;
  $('#param-batch').value = state.batchSize;
  $('#param-lr').value = state.lr;
  $('#param-lambda').value = state.lambda;

  setStatus('Waiting to load MovieLens files (u.item, u.data)...');
});

// ---------------- Load Data ----------------
async function onLoadData() {
  try {
    setStatus('Loading files (u.item, u.data)...');
    await loadData();
    state.loaded = true;
    state.U = userIndexByRawId.size;
    state.I = movieIndexByRawId.size;
    updateStatsUI();
    setStatus('Data loaded. Configure MF and train.', true);

    // Populate user select with raw IDs (sorted)
    const sel = $('#user-select');
    sel.innerHTML = '<option value="">— select user —</option>';
    const rawUsers = Array.from(userIndexByRawId.keys()).sort((a, b) => a - b);
    for (const rawId of rawUsers) {
      const opt = document.createElement('option');
      opt.value = String(rawId);
      opt.textContent = String(rawId);
      sel.appendChild(opt);
    }

    // Enable training
    $('#btn-train').disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`Error while loading data: ${err.message}`, false);
  }
}

// ---------------- Train ----------------
async function onTrain() {
  if (!state.loaded) {
    setStatus('Load data first.', false);
    return;
  }

  // Ждём инициализации TF.js (важно для некоторых браузеров)
  await tf.ready();
  console.log('TF backend:', tf.getBackend());
  setStatus('Starting training…');

  // Read hyperparams (поддержка запятой как десятичного разделителя)
  state.k = clampInt($('#param-k').value, 2, 128, 16);
  state.epochs = clampInt($('#param-epochs').value, 1, 100, 15);
  state.batchSize = clampInt($('#param-batch').value, 64, 4096, 2048);
  state.lr = clampFloat($('#param-lr').value, 1e-5, 0.5, 0.01);
  state.lambda = clampFloat($('#param-lambda').value, 0, 0.1, 1e-4);

  // Build / rebuild model
  disposeModel();
  buildModel();

  // Make a split 90/10
  makeTrainValSplit(0.9);
  if (!state.split.trainIdx?.length || !state.split.valIdx?.length) {
    setStatus('Train/val split is empty — check that ratings were parsed.', false);
    $('#btn-train').disabled = false; $('#btn-cancel').disabled = true;
    return;
  }

  // UI state
  state.stopRequested = false;
  $('#btn-cancel').disabled = false;
  $('#btn-train').disabled = true;
  setRecommendControlsEnabled(false);
  setProgress(0);
  setTrainInfo(0, state.epochs, NaN, NaN);

  try {
    await trainLoop();
    if (!state.stopRequested) {
      state.trained = true;
      setStatus('Training complete. You can now get recommendations.', true);
      setRecommendControlsEnabled(true);
      $('#btn-cancel').disabled = true;
    } else {
      setStatus('Training canceled.', false);
      $('#btn-train').disabled = false;
      $('#btn-cancel').disabled = true;
    }
  } catch (err) {
    console.error(err);
    setStatus(`Training error: ${err.message}`, false);
    $('#btn-train').disabled = false;
    $('#btn-cancel').disabled = true;
  }
}

function onCancel() {
  state.stopRequested = true;
}

// Create TF variables and optimizer
function buildModel() {
  state.P  = tf.variable(tf.randomNormal([state.U, state.k], 0, 0.01, 'float32'), true, 'P');
  state.Q  = tf.variable(tf.randomNormal([state.I, state.k], 0, 0.01, 'float32'), true, 'Q');
  state.bu = tf.variable(tf.zeros([state.U], 'float32'), true, 'bu');
  state.bi = tf.variable(tf.zeros([state.I], 'float32'), true, 'bi');
  state.mu = tf.scalar(STATS.mean, 'float32');

  state.optimizer = tf.train.adam(state.lr);
}

// Free all TF resources
function disposeModel() {
  try {
    if (state.P) state.P.dispose();
    if (state.Q) state.Q.dispose();
    if (state.bu) state.bu.dispose();
    if (state.bi) state.bi.dispose();
    if (state.mu) state.mu.dispose();
    if (state.optimizer && state.optimizer.dispose) state.optimizer.dispose();
  } catch {}
  state.P = state.Q = state.bu = state.bi = state.mu = null;
  state.optimizer = null;
  state.trained = false;
}

// Prepare train/val split as index arrays into ratingsTriples
function makeTrainValSplit(trainFrac = 0.9) {
  const n = ratingsTriples.length;
  const idx = new Int32Array(n);
  for (let t = 0; t < n; t++) idx[t] = t;

  // Fisher–Yates shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }

  const nTrain = Math.floor(n * trainFrac);
  state.split.trainIdx = idx.slice(0, nTrain);
  state.split.valIdx = idx.slice(nTrain);
}

// Main training loop: batch SGD with Adam, report RMSEs
async function trainLoop() {
  const nTrain = state.split.trainIdx.length;
  const stepsPerEpoch = Math.ceil(nTrain / state.batchSize);

  for (let epoch = 1; epoch <= state.epochs; epoch++) {
    if (state.stopRequested) break;

    shuffleInt32(state.split.trainIdx);

    let mseSum = 0;
    let count = 0;

    for (let step = 0; step < stepsPerEpoch; step++) {
      if (state.stopRequested) break;

      const start = step * state.batchSize;
      const end = Math.min(nTrain, start + state.batchSize);

      // Build batch tensors
      const { uBatch, iBatch, rBatch } = buildBatch(state.split.trainIdx, start, end);

      // Minimize loss on this batch
      const batchMSE = state.optimizer.minimize(() => {
        return tf.tidy(() => {
          const pred = predictBatch(uBatch, iBatch); // [B]
          const err = tf.sub(pred, rBatch);          // [B]
          const mse = tf.mean(tf.mul(err, err));     // scalar

          // L2 regularization on gathered rows
          const Pu = tf.gather(state.P, uBatch);
          const Qi = tf.gather(state.Q, iBatch);
          const bu = tf.gather(state.bu, uBatch);
          const bi = tf.gather(state.bi, iBatch);

          const reg =
            tf.mul(state.lambda, tf.addN([
              tf.sum(tf.mul(Pu, Pu)),
              tf.sum(tf.mul(Qi, Qi)),
              tf.sum(tf.mul(bu, bu)),
              tf.sum(tf.mul(bi, bi))
            ]));

          return tf.add(mse, reg);
        });
      }, /* returnCost */ true);

      // Read MSE value (Train RMSE estimate)
      const mseVal = (await batchMSE.data())[0];
      batchMSE.dispose();
      mseSum += mseVal * (end - start);
      count += (end - start);

      // Progress bar & cooperative yield
      const progress = ((epoch - 1) / state.epochs + (step + 1) / state.epochs / stepsPerEpoch) * 100;
      setProgress(progress);
      if ((step & 1) === 1) await tf.nextFrame();

      // Dispose batch tensors
      uBatch.dispose(); iBatch.dispose(); rBatch.dispose();
    }

    const trainRMSE = Math.sqrt(mseSum / Math.max(1, count));
    const valRMSE = await computeSplitRMSE(state.split.valIdx, 4096);

    setTrainInfo(epoch, state.epochs, trainRMSE, valRMSE);
  }
}

// Build batch tensors from split indices range
function buildBatch(splitIdx, start, end) {
  const B = end - start;
  const uArr = new Int32Array(B);
  const iArr = new Int32Array(B);
  const rArr = new Float32Array(B);

  for (let b = 0; b < B; b++) {
    const t = splitIdx[start + b];
    const tri = ratingsTriples[t];
    uArr[b] = tri.u;
    iArr[b] = tri.i;
    rArr[b] = tri.r;
  }

  const uBatch = tf.tensor1d(uArr, 'int32');
  const iBatch = tf.tensor1d(iArr, 'int32');
  const rBatch = tf.tensor1d(rArr, 'float32');
  return { uBatch, iBatch, rBatch };
}

// Predict batch ratings for pairs (uBatch[i], iBatch[i]) → [B]
function predictBatch(uBatch, iBatch) {
  return tf.tidy(() => {
    const Pu = tf.gather(state.P, uBatch);   // [B,k]
    const Qi = tf.gather(state.Q, iBatch);   // [B,k]
    const bu = tf.gather(state.bu, uBatch);  // [B]
    const bi = tf.gather(state.bi, iBatch);  // [B]

    const dot = tf.sum(tf.mul(Pu, Qi), 1);   // [B]
    const base = tf.addN([dot, bu, bi]);     // [B]
    const withMu = tf.add(base, state.mu);   // [B]
    return withMu; // не клиппим во время тренировки
  });
}

// Compute RMSE over a split of triples (indices), chunked to limit memory
async function computeSplitRMSE(splitIdx, chunk = 8192) {
  if (!splitIdx || splitIdx.length === 0) return NaN;

  let sse = 0;
  let n = 0;

  for (let offset = 0; offset < splitIdx.length; offset += chunk) {
    const end = Math.min(splitIdx.length, offset + chunk);
    const { uBatch, iBatch, rBatch } = buildBatch(splitIdx, offset, end);
    const pred = predictBatch(uBatch, iBatch); // [B]
    const err = tf.sub(pred, rBatch);
    const se = await tf.sum(tf.mul(err, err)).data();
    sse += se[0];
    n += (end - offset);

    uBatch.dispose(); iBatch.dispose(); rBatch.dispose();
    pred.dispose(); err.dispose();
    await tf.nextFrame();
  }

  return Math.sqrt(sse / Math.max(1, n));
}

// ---------------- Recommend ----------------
function onRecommend() {
  if (!state.trained) {
    setStatus('Train the model first.', false);
    return;
  }
  const rawIdStr = $('#user-select').value;
  if (!rawIdStr) {
    setStatus('Select a user first.', false);
    return;
  }
  const rawU = Number(rawIdStr);
  const u = userIndexByRawId.get(rawU);
  if (u === undefined) {
    setStatus('Invalid user.', false);
    return;
  }
  const topN = clampInt($('#topn').value, 1, 50, 10);
  const recs = recommendForUser(u, topN);
  renderRecommendations(recs);
}

// Compute predictions for user u for all items; filter already-rated; return topN with explanation parts
function recommendForUser(u, topN = 10) {
  const seen = userRatedItems.get(u) || new Set();

  // Vectorized prediction for all items
  const { preds, dots, buScalar, biArray } = tf.tidy(() => {
    const pu = tf.gather(state.P, tf.tensor1d([u], 'int32')).reshape([state.k]); // [k]
const bu = state.bu.gather(tf.tensor1d([u], 'int32')).reshape([]);           // scalar

const dotVec = tf.matMul(state.Q, pu.reshape([state.k, 1])).reshape([state.I]); // [I]

// Было:
// const base = tf.addN([dotVec, state.bi, bu, state.mu]);

// Стало (правильно, с broadcasting скаляров):
let base = tf.add(dotVec, state.bi); // [I]
base = tf.add(base, bu);             // [I] + [] -> [I]
base = tf.add(base, state.mu);       // [I] + [] -> [I]

const clipped = tf.clipByValue(base, 1, 5);
                                  // [I]

    return {
      preds: clipped, // [I]
      dots: dotVec,   // [I]
      buScalar: bu,   // scalar
      biArray: state.bi // [I] (variable; данные снимем отдельно)
    };
  });

  // Считываем на CPU одним заходом
  const predArr = Array.from(preds.dataSync()); // [I]
  const dotArr  = Array.from(dots.dataSync());  // [I]
  const buVal   = (awaitScalar(buScalar));
  const biArr   = Array.from(biArray.dataSync());

  preds.dispose(); dots.dispose(); // buScalar и biArray — variables, не освобождаем

  const mu = STATS.mean;
  const candidates = [];

  for (let i = 0; i < state.I; i++) {
    if (seen.has(i)) continue;

    const movie = movies[i];
    candidates.push({
      rank: 0,
      u, i,
      rawUserId: getRawUserIdFromDense(u),
      rawItemId: movie.rawId,
      title: movie.title,
      genres: movie.genres,
      pred: predArr[i],
      parts: { mu, bu: buVal, bi: biArr[i], dot: dotArr[i] }
    });
  }

  candidates.sort((a, b) => b.pred - a.pred);
  for (let r = 0; r < candidates.length; r++) candidates[r].rank = r + 1;
  return candidates.slice(0, topN);
}

function renderRecommendations(recs) {
  const tbody = $('#results-table tbody');
  tbody.innerHTML = '';
  if (!recs || recs.length === 0) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 5;
    td.textContent = 'No recommendations (check that the user has enough ratings and the model is trained).';
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const r of recs) {
    const tr = document.createElement('tr');

    const tdRank = document.createElement('td');
    tdRank.textContent = String(r.rank);
    tr.appendChild(tdRank);

    const tdTitle = document.createElement('td');
    tdTitle.textContent = r.title;
    tr.appendChild(tdTitle);

    const tdGenres = document.createElement('td');
    tdGenres.textContent = r.genres.join(', ') || '—';
    tr.appendChild(tdGenres);

    const tdPred = document.createElement('td');
    tdPred.textContent = r.pred.toFixed(3);
    tr.appendChild(tdPred);

    const tdExplain = document.createElement('td');
    tdExplain.textContent =
      `${r.parts.mu.toFixed(3)} | ${r.parts.bu.toFixed(3)} | ` +
      `${r.parts.bi.toFixed(3)} | ${r.parts.dot.toFixed(3)}`;
    tr.appendChild(tdExplain);

    tbody.appendChild(tr);
  }
}

// ---------------- Utilities ----------------
function clampInt(v, min, max, dflt) {
  const n = parseInt(String(v).replace(',', '.'), 10);
  if (!Number.isFinite(n)) return dflt;
  return Math.max(min, Math.min(max, n));
}
function clampFloat(v, min, max, dflt) {
  const n = parseFloat(String(v).replace(',', '.'));
  if (!Number.isFinite(n)) return dflt;
  return Math.max(min, Math.min(max, n));
}
function shuffleInt32(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
}
function getRawUserIdFromDense(uDense) {
  if (!getRawUserIdFromDense.cache) {
    const map = new Map(); // dense -> raw
    for (const [raw, dense] of userIndexByRawId.entries()) map.set(dense, raw);
    getRawUserIdFromDense.cache = map;
  }
  return getRawUserIdFromDense.cache.get(uDense);
}
getRawUserIdFromDense.cache = null;

function awaitScalar(scalarTensor) {
  // Быстрое чтение одного скаляра
  return scalarTensor.dataSync()[0];
}

// Clean TF resources on page unload
window.addEventListener('beforeunload', disposeModel);

