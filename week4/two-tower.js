// two-tower.js
// -----------------------------------------------------------------------------
// Two-Tower Recommender with Deep Towers (TF.js)
// -----------------------------------------------------------------------------
// Идея two-towers: отдельно кодируем пользователя и фильм в одно пространство
// признаков. Скоринг = dot(u, i). Это позволяет быстрое retrieval: посчитали
// эмбеддинги всех фильмов один раз, дальше один matmul на пользователя.
//
// Дип-часть: после ID-embedding мы добавляем MLP (>=1 hidden layer) и подмешиваем
// контент-фичи жанров (из u.item). Так мы объединяем коллаборативный сигнал (ID)
// и контент (жанры), а нелинейность MLP учит более выразительные представления.
//
// In-batch negatives (softmax): для батча (u, i+) логиты = U @ I+^T.
// Для каждой строки таргет — диагональ. Остальные столбцы в строке — «негативы».
// Это стабильно и экономно: не нужно явно сэмплировать кучу отрицательных примеров.
//
// Альтернатива — BPR: оптимизируем разность s(u,i+) − s(u,i−) через логистическую
// функцию, сэмплируя по одному негативу на пользователя.
// -----------------------------------------------------------------------------

(function initTwoTower(global) {
  'use strict';

  class TwoTowerModel {
    /**
     * @param {number} numUsers
     * @param {number} numItems
     * @param {number} embDim
     * @param {{lossType?:'softmax'|'bpr', lr?:number, userHidden?:number, itemHidden?:number, l2?:number, normalize?:boolean}} opts
     */
    constructor(numUsers, numItems, embDim, opts = {}) {
      if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js not loaded. Include <script src="tf.min.js"></script> before two-tower.js');
      }
      this.numUsers = numUsers | 0;
      this.numItems = numItems | 0;
      this.embDim   = embDim  | 0;

      // Параметры обучения/регуляризации
      this.lossType  = opts.lossType  || 'softmax';
      this.lr        = Number.isFinite(opts.lr) ? opts.lr : 0.01;
      this.userHidden = (opts.userHidden | 0) || (this.embDim + 19 + 16);
      this.itemHidden = (opts.itemHidden | 0) || (this.embDim + 19 + 16);
      this.l2        = Number.isFinite(opts.l2) ? opts.l2 : 1e-4;
      this.normalize = opts.normalize !== false; // по умолчанию нормируем (косинус-подобный скор)

      // --- ID-Embedding таблицы (это НЕ дип-модуль; просто обучаемые lookup-векторы)
      this.userEmbedding = tf.variable(
        tf.randomNormal([this.numUsers, this.embDim], 0, 0.05, 'float32'),
        true, 'userEmbedding'
      );
      this.itemEmbedding = tf.variable(
        tf.randomNormal([this.numItems, this.embDim], 0, 0.05, 'float32'),
        true, 'itemEmbedding'
      );

      // --- Жанровые признаки (задать через setFeatures)
      this.itemGenreMat = null; // Tensor2D [numItems, 19]
      this.userGenreMat = null; // Tensor2D [numUsers, 19] (опционально)
      this.numGenres = 0;

      // --- Слои MLP в башнях (создадим лениво при первом проходе, когда известен inDim)
      this._userDense1 = null; this._userOut = null;
      this._itemDense1 = null; this._itemOut = null;

      // --- Оптимизатор
      this.optimizer = tf.train.adam(this.lr);

      // --- Кеш предвычисленных эмбеддингов айтемов (для быстрого top-K)
      this._cachedItemEmb = null; this._cacheDirty = true;
    }

    // -------------------------------------------------------------------------
    // Признаки (жанры) — подмешиваем контент к ID
    // -------------------------------------------------------------------------
    /**
     * @param {{itemGenres: tf.Tensor2D|number[][]|Float32Array, userGenres?: tf.Tensor2D|number[][]|Float32Array, normalizeRows?: boolean}} p
     */
    setFeatures(p = {}) {
      const { itemGenres, userGenres, normalizeRows = true } = p;

      const to2D = (data, rows) => {
        if (data == null) return null;
        if (data instanceof tf.Tensor) return /** @type {tf.Tensor2D} */(data);
        if (Array.isArray(data)) return tf.tensor2d(data);
        if (data instanceof Float32Array) {
          if (rows <= 0 || data.length % rows !== 0) throw new Error('Invalid flat feature shape');
          const cols = data.length / rows;
          return tf.tensor2d(data, [rows, cols]);
        }
        throw new Error('Unsupported feature type');
      };

      if (itemGenres) {
        this.itemGenreMat?.dispose();
        this.itemGenreMat = to2D(itemGenres, this.numItems);
        this.numGenres = this.itemGenreMat.shape[1] | 0;
        if (normalizeRows) {
          this.itemGenreMat = tf.tidy(() => tf.linalg.l2Normalize(this.itemGenreMat, 1));
        }
      }
      if (userGenres) {
        this.userGenreMat?.dispose();
        this.userGenreMat = to2D(userGenres, this.numUsers);
        if (normalizeRows) {
          this.userGenreMat = tf.tidy(() => tf.linalg.l2Normalize(this.userGenreMat, 1));
        }
        if (!this.numGenres) this.numGenres = this.userGenreMat.shape[1] | 0;
      }

      this._invalidateItemCache();
    }

    // -------------------------------------------------------------------------
    // USER-tower: ID (+жанры, если есть) → MLP → эмбеддинг пользователя
    // Почему MLP: нелинейная проекция смешанных фич в общее пространство
    // -------------------------------------------------------------------------
    /**
     * @param {tf.Tensor1D} userIdxInt32  [B] int32
     * @returns {tf.Tensor2D} [B, embDim]
     */
    userForward(userIdxInt32) {
      return tf.tidy(() => {
        const idEmb = tf.gather(this.userEmbedding, userIdxInt32); // [B, D]
        let feats = idEmb;
        if (this.userGenreMat) {
          const uGenres = tf.gather(this.userGenreMat, userIdxInt32); // [B, G]
          feats = tf.concat([idEmb, uGenres], 1); // [B, D+G]
        }

        if (!this._userDense1) {
          const inDim = feats.shape[1];
          this._userDense1 = tf.layers.dense({
            units: this.userHidden, activation: 'relu',
            kernelInitializer: 'glorotUniform', biasInitializer: 'zeros',
            name: 'user_dense_1'
          });
          this._userOut = tf.layers.dense({
            units: this.embDim, activation: null,
            kernelInitializer: 'glorotUniform', biasInitializer: 'zeros',
            name: 'user_out'
          });
          // Build weights (прогон нулей) чтобы слои инициализировали веса
          this._userOut.apply(this._userDense1.apply(tf.zeros([1, inDim])));
        }

        const h = this._userDense1.apply(feats);
        let out = /** @type {tf.Tensor2D} */ (this._userOut.apply(h));
        if (this.normalize) out = tf.linalg.l2Normalize(out, 1); // L2-норма → косинус-подобный скор
        return out;
      });
    }

    // -------------------------------------------------------------------------
    // ITEM-tower: ID + жанры → MLP → эмбеддинг фильма
    // -------------------------------------------------------------------------
    /**
     * @param {tf.Tensor1D} itemIdxInt32  [B] int32
     * @returns {tf.Tensor2D} [B, embDim]
     */
    itemForward(itemIdxInt32) {
      return tf.tidy(() => {
        const idEmb = tf.gather(this.itemEmbedding, itemIdxInt32); // [B, D]
        let feats = idEmb;
        if (this.itemGenreMat) {
          const iGenres = tf.gather(this.itemGenreMat, itemIdxInt32); // [B, G]
          feats = tf.concat([idEmb, iGenres], 1); // [B, D+G]
        }

        if (!this._itemDense1) {
          const inDim = feats.shape[1];
          this._itemDense1 = tf.layers.dense({
            units: this.itemHidden, activation: 'relu',
            kernelInitializer: 'glorotUniform', biasInitializer: 'zeros',
            name: 'item_dense_1'
          });
          this._itemOut = tf.layers.dense({
            units: this.embDim, activation: null,
            kernelInitializer: 'glorotUniform', biasInitializer: 'zeros',
            name: 'item_out'
          });
          this._itemOut.apply(this._itemDense1.apply(tf.zeros([1, inDim])));
        }

        const h = this._itemDense1.apply(feats);
        let out = /** @type {tf.Tensor2D} */ (this._itemOut.apply(h));
        if (this.normalize) out = tf.linalg.l2Normalize(out, 1);
        return out;
      });
    }

    // -------------------------------------------------------------------------
    // Dot-product: простая и быстрая мера близости в общем пространстве
    // -------------------------------------------------------------------------
    /**
     * @param {tf.Tensor2D} uEmb [B,D]
     * @param {tf.Tensor2D} iEmb [B,D]
     * @returns {tf.Tensor1D} [B]
     */
    score(uEmb, iEmb) {
      return tf.tidy(() => tf.sum(tf.mul(uEmb, iEmb), 1));
    }

    // -------------------------------------------------------------------------
    // In-batch softmax loss: logits = U @ I+^T, таргет — диагональ
    // Численно стабильно через logSumExp
    // -------------------------------------------------------------------------
    /**
     * @param {tf.Tensor2D} U [B,D]
     * @param {tf.Tensor2D} Ipos [B,D]
     * @returns {tf.Tensor} scalar
     */
    _softmaxInBatchLoss(U, Ipos) {
      return tf.tidy(() => {
        const logits = tf.matMul(U, Ipos, false, true);  // [B,B]
        const rowLSE = tf.logSumExp(logits, 1);          // [B]
        const B = logits.shape[0] | 0;
        const diag = tf.sum(tf.mul(logits, tf.eye(B)), 1); // [B]
        const logProb = tf.sub(diag, rowLSE);            // [B]
        return tf.neg(tf.mean(logProb));                 // −mean(log p(diag))
      });
    }

    // -------------------------------------------------------------------------
    // BPR: −log σ(s_pos − s_neg) с одним сэмплингом негатива на пользователя
    // -------------------------------------------------------------------------
    /**
     * @param {tf.Tensor2D} U [B,D]
     * @param {tf.Tensor2D} Ipos [B,D]
     * @param {tf.Tensor1D} negIdxInt32 [B]
     * @returns {tf.Tensor} scalar
     */
    _bprLoss(U, Ipos, negIdxInt32) {
      return tf.tidy(() => {
        const Ineg = this.itemForward(negIdxInt32);     // [B,D]
        const sPos = this.score(U, Ipos);               // [B]
        const sNeg = this.score(U, Ineg);               // [B]
        const diff = tf.sub(sPos, sNeg);                // [B]
        const loss = tf.neg(tf.mean(tf.logSigmoid(diff)));
        Ineg.dispose();
        return loss;
      });
    }

    // -------------------------------------------------------------------------
    // Один шаг обучения: Adam, tape/minimize, возврат scalar-лосса для UI
    // -------------------------------------------------------------------------
    /**
     * @param {Int32Array|number[]} userIdxs length B
     * @param {Int32Array|number[]} posItemIdxs length B
     * @returns {Promise<number>}
     */
    async trainStep(userIdxs, posItemIdxs) {
      const uIdx = tf.tensor1d(userIdxs, 'int32');
      const iPos = tf.tensor1d(posItemIdxs, 'int32');
      let iNeg = null;
      if (this.lossType === 'bpr') {
        iNeg = tf.tensor1d(this._sampleNegatives(posItemIdxs), 'int32');
      }

      const lossScalar = await this.optimizer.minimize(() => {
        return tf.tidy(() => {
          const U    = this.userForward(uIdx);   // [B,D]
          const Ipos = this.itemForward(iPos);   // [B,D]

          let loss = (this.lossType === 'bpr')
            ? this._bprLoss(U, Ipos, /** @type {tf.Tensor1D} */(iNeg))
            : this._softmaxInBatchLoss(U, Ipos);

          // L2-регуляризация: стабилизируем обучение и ограничиваем нормы
          if (this.l2 > 0) {
            const uRows = tf.gather(this.userEmbedding, uIdx);
            const iRows = tf.gather(this.itemEmbedding, iPos);
            const reg = tf.mul(this.l2, tf.addN([
              tf.sum(tf.mul(U, U)),
              tf.sum(tf.mul(Ipos, Ipos)),
              tf.sum(tf.mul(uRows, uRows)),
              tf.sum(tf.mul(iRows, iRows))
            ]));
            loss = tf.add(loss, reg);
          }
          return loss;
        });
      }, true);

      uIdx.dispose(); iPos.dispose();
      if (iNeg) iNeg.dispose();

      this._invalidateItemCache(); // веса поменялись → кеш айтемов невалиден

      const val = (await lossScalar.data())[0];
      lossScalar.dispose();
      return val;
    }

    // -------------------------------------------------------------------------
    // Инференс / Retrieval: эмбеддинг пользователя, матрица эмбеддингов айтемов,
    // top-K по одному матричному умножению
    // -------------------------------------------------------------------------
    /**
     * @param {number} uIdx
     * @returns {tf.Tensor1D} [embDim]
     */
    getUserEmbedding(uIdx) {
      const u = tf.tensor1d([uIdx | 0], 'int32');
      const U = this.userForward(u);         // [1,D]
      const out = U.reshape([this.embDim]);  // [D]
      u.dispose(); U.dispose();
      return out;
    }

    /**
     * @param {number} [batch=4096]
     * @returns {Promise<tf.Tensor2D>} [numItems, embDim]
     */
    async materializeItemEmbeddings(batch = 4096) {
      if (!this._cacheDirty && this._cachedItemEmb) return this._cachedItemEmb;

      const chunks = [];
      for (let start = 0; start < this.numItems; start += batch) {
        const end = Math.min(this.numItems, start + batch);
        const idx = tf.tensor1d(Array.from({ length: end - start }, (_, j) => start + j), 'int32');
        const emb = this.itemForward(idx); // [chunk, D]
        chunks.push(emb);
        idx.dispose();
        await tf.nextFrame();
      }
      const all = tf.concat(chunks, 0);
      chunks.forEach(t => t.dispose());
      this._cachedItemEmb?.dispose();
      this._cachedItemEmb = all;
      this._cacheDirty = false;
      return this._cachedItemEmb;
    }

    /**
     * @param {number} uIdx
     * @param {number} K
     * @returns {Promise<{indices:Int32Array, scores:Float32Array}>}
     */
    async getTopKForUser(uIdx, K = 10) {
      const uEmb = this.getUserEmbedding(uIdx);               // [D]
      const I = await this.materializeItemEmbeddings();       // [I,D]
      const scores2D = tf.matMul(I, uEmb.reshape([this.embDim, 1])); // [I,1]
      const scores = scores2D.reshape([this.numItems]);       // [I]
      const { values, indices } = tf.topk(scores, Math.min(K, this.numItems), true);
      const vals = values.dataSync(); const idxs = indices.dataSync();
      uEmb.dispose(); scores2D.dispose(); scores.dispose(); values.dispose(); indices.dispose();
      return { indices: new Int32Array(idxs), scores: new Float32Array(vals) };
    }

    // -------------------------------------------------------------------------
    // Внутренние утилиты: сэмплинг негатива, инвалидация кеша, освобождение
    // -------------------------------------------------------------------------
    _invalidateItemCache() {
      this._cacheDirty = true;
      if (this._cachedItemEmb) { this._cachedItemEmb.dispose(); this._cachedItemEmb = null; }
    }

    _sampleNegatives(posItemIdxs) {
      const B = posItemIdxs.length | 0;
      const out = new Int32Array(B);
      for (let k = 0; k < B; k++) {
        const pos = posItemIdxs[k] | 0;
        let neg = pos;
        // простая отбраковка совпадений; для продакшена можно заменить на smart sampler
        while (neg === pos) neg = (Math.random() * this.numItems) | 0;
        out[k] = neg;
      }
      return out;
    }

    dispose() {
      this.userEmbedding?.dispose();
      this.itemEmbedding?.dispose();
      this.itemGenreMat?.dispose();
      this.userGenreMat?.dispose();
      this._cachedItemEmb?.dispose();
      for (const l of [this._userDense1, this._userOut, this._itemDense1, this._itemOut]) {
        if (l && Array.isArray(l.trainableWeights)) {
          for (const v of l.trainableWeights) v.val?.dispose?.();
        }
      }
      if (this.optimizer?.dispose) this.optimizer.dispose();
      this._userDense1 = this._userOut = this._itemDense1 = this._itemOut = null;
    }
  }

  // UMD-экспорт
  global.TwoTowerModel = TwoTowerModel;
})(typeof window !== 'undefined' ? window : globalThis);
