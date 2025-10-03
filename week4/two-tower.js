// two-tower.js
// -----------------------------------------------------------------------------
// Two-Tower Recommender with Deep Towers (TF.js)
// -----------------------------------------------------------------------------

(function initTwoTower(global) {
  'use strict';

  // Универсальная L2-нормализация по строкам (совместима с любыми версиями TF.js)
  function l2NormalizeRows(x, eps = 1e-6) {
    return tf.tidy(() => {
      const norm = tf.norm(x, 'euclidean', 1, true); // [rows,1]
      return x.div(norm.add(eps));
    });
  }

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

      this.lossType  = opts.lossType  || 'softmax';
      this.lr        = Number.isFinite(opts.lr) ? opts.lr : 0.01;
      this.userHidden = (opts.userHidden | 0) || (this.embDim + 19 + 16);
      this.itemHidden = (opts.itemHidden | 0) || (this.embDim + 19 + 16);
      this.l2        = Number.isFinite(opts.l2) ? opts.l2 : 1e-4;
      this.normalize = opts.normalize !== false;

      // ID-таблицы (lookup)
      this.userEmbedding = tf.variable(
        tf.randomNormal([this.numUsers, this.embDim], 0, 0.05, 'float32'),
        true, 'userEmbedding'
      );
      this.itemEmbedding = tf.variable(
        tf.randomNormal([this.numItems, this.embDim], 0, 0.05, 'float32'),
        true, 'itemEmbedding'
      );

      // Жанры
      this.itemGenreMat = null; // Tensor2D [numItems, 19]
      this.userGenreMat = null; // Tensor2D [numUsers, 19] (опц.)
      this.numGenres = 0;

      // MLP-слои (создаются лениво)
      this._userDense1 = null; this._userOut = null;
      this._itemDense1 = null; this._itemOut = null;

      this.optimizer = tf.train.adam(this.lr);
      this._cachedItemEmb = null; this._cacheDirty = true;
    }

    // ------------------------------ Features ----------------------------------
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
        if (normalizeRows) this.itemGenreMat = l2NormalizeRows(this.itemGenreMat);
      }
      if (userGenres) {
        this.userGenreMat?.dispose();
        this.userGenreMat = to2D(userGenres, this.numUsers);
        if (normalizeRows) this.userGenreMat = l2NormalizeRows(this.userGenreMat);
        if (!this.numGenres) this.numGenres = this.userGenreMat.shape[1] | 0;
      }

      this._invalidateItemCache();
    }

    // ------------------------------ USER tower --------------------------------
    userForward(userIdxInt32) {
      return tf.tidy(() => {
        const idEmb = tf.gather(this.userEmbedding, userIdxInt32); // [B,D]
        let feats = idEmb;
        if (this.userGenreMat) {
          const uGenres = tf.gather(this.userGenreMat, userIdxInt32); // [B,G]
          feats = tf.concat([idEmb, uGenres], 1); // [B,D+G]
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
          this._userOut.apply(this._userDense1.apply(tf.zeros([1, inDim])));
        }

        const h = this._userDense1.apply(feats);
        let out = /** @type {tf.Tensor2D} */ (this._userOut.apply(h));
        if (this.normalize) out = l2NormalizeRows(out);
        return out;
      });
    }

    // ------------------------------ ITEM tower --------------------------------
    itemForward(itemIdxInt32) {
      return tf.tidy(() => {
        const idEmb = tf.gather(this.itemEmbedding, itemIdxInt32); // [B,D]
        let feats = idEmb;
        if (this.itemGenreMat) {
          const iGenres = tf.gather(this.itemGenreMat, itemIdxInt32); // [B,G]
          feats = tf.concat([idEmb, iGenres], 1); // [B,D+G]
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
        if (this.normalize) out = l2NormalizeRows(out);
        return out;
      });
    }

    // ------------------------------- Scoring ----------------------------------
    score(uEmb, iEmb) {
      return tf.tidy(() => tf.sum(tf.mul(uEmb, iEmb), 1));
    }

    // ------------------------------- Losses -----------------------------------
    _softmaxInBatchLoss(U, Ipos) {
      return tf.tidy(() => {
        const logits = tf.matMul(U, Ipos, false, true);  // [B,B]
        const rowLSE = tf.logSumExp(logits, 1);          // [B]
        const B = logits.shape[0] | 0;
        const diag = tf.sum(tf.mul(logits, tf.eye(B)), 1); // [B]
        const logProb = tf.sub(diag, rowLSE);            // [B]
        return tf.neg(tf.mean(logProb));
      });
    }

    _bprLoss(U, Ipos, negIdxInt32) {
      return tf.tidy(() => {
        const Ineg = this.itemForward(negIdxInt32);     // [B,D]
        const sPos = this.score(U, Ipos);               // [B]
        const sNeg = this.score(U, Ineg);               // [B]
        const diff = tf.sub(sPos, sNeg);
        const loss = tf.neg(tf.mean(tf.logSigmoid(diff)));
        Ineg.dispose();
        return loss;
      });
    }

    // ------------------------------- Train ------------------------------------
    async trainStep(userIdxs, posItemIdxs) {
      const uIdx = tf.tensor1d(userIdxs, 'int32');
      const iPos = tf.tensor1d(posItemIdxs, 'int32');
      let iNeg = null;
      if (this.lossType === 'bpr') iNeg = tf.tensor1d(this._sampleNegatives(posItemIdxs), 'int32');

      const lossScalar = await this.optimizer.minimize(() => {
        return tf.tidy(() => {
          const U    = this.userForward(uIdx);
          const Ipos = this.itemForward(iPos);

          let loss = (this.lossType === 'bpr')
            ? this._bprLoss(U, Ipos, /** @type {tf.Tensor1D} */(iNeg))
            : this._softmaxInBatchLoss(U, Ipos);

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

      this._invalidateItemCache();

      const val = (await lossScalar.data())[0];
      lossScalar.dispose();
      return val;
    }

    // ------------------------------ Inference ---------------------------------
    getUserEmbedding(uIdx) {
      const u = tf.tensor1d([uIdx | 0], 'int32');
      const U = this.userForward(u);         // [1,D]
      const out = U.reshape([this.embDim]);  // [D]
      u.dispose(); U.dispose();
      return out;
    }

    async materializeItemEmbeddings(batch = 4096) {
      if (!this._cacheDirty && this._cachedItemEmb) return this._cachedItemEmb;
      const chunks = [];
      for (let start = 0; start < this.numItems; start += batch) {
        const end = Math.min(this.numItems, start + batch);
        const idx = tf.tensor1d(Array.from({ length: end - start }, (_, j) => start + j), 'int32');
        const emb = this.itemForward(idx);
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

    // ------------------------------- Utils ------------------------------------
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

  global.TwoTowerModel = TwoTowerModel;
})(typeof window !== 'undefined' ? window : globalThis);
