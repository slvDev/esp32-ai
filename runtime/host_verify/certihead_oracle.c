/* CertiHead oracle: how few output-head rows can we score and still be EXACT?
 *
 * The output head is 14.6 ms of a 25.0 ms token -- 58% -- and at 86.3 MiB/s it
 * is bandwidth bound on 1.26 MiB. Reading it faster is close to done. The only
 * remaining lever is reading less of it.
 *
 * Approximate retrieval (ANN/HNSW over output embeddings) is prior art and is
 * approximate: it validates that quality "stays acceptable". This project's
 * whole evidence discipline is byte-identical output, so an approximate head
 * would forfeit the one claim that makes the speed numbers mean anything.
 *
 * The alternative is a certificate. Partition rows into contiguous tiles and
 * precompute, per tile, a quantity U_j(x) that provably upper-bounds
 * max_{i in tile j} <x, w_i>. Score a small hot set first to get an incumbent
 * best logit, then skip every tile whose bound cannot beat it. Rows skipped
 * this way cannot have contained the argmax -- not "probably did not", cannot.
 * The result is identical to the full scan by construction, so CE, perplexity
 * and the 200-token digest are all preserved without needing to be re-measured.
 *
 * This is a HOST oracle. It reports how many rows a device would have to read;
 * it does not itself run on the device and claims no tok/s. The promotion gate
 * is stated in temphelp.md: <25% mean rows scored at 100% argmax identity.
 *
 * Two bounds are implemented because they cost the same and it is not obvious
 * which wins on real hidden states:
 *
 *   centroid   U = <x, c_j> + ||x|| * max_i ||w_i - c_j||     (Cauchy-Schwarz
 *              on the residual; tight when a tile's rows point the same way)
 *   box        U = sum_d max(x_d * lo_jd, x_d * hi_jd)        (per-dimension
 *              interval arithmetic; tight when rows vary per coordinate)
 *
 * Both are sound. Neither can ever return a bound below the tile's true max, so
 * a bug in them shows up as an argmax mismatch, which is checked on every
 * position rather than sampled.
 *
 * Build (MSVC):
 *   cl /O2 /std:c17 /DLLM_KV_INT8 /I runtime runtime\host_verify\certihead_oracle.c
 * Usage:
 *   certihead_oracle <model.bin> <tokens.bin> [windows] [tile] [bound] [hot]
 *     bound: 0=centroid 1=box 2=both (min of the two, still sound)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../llm.h"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

/* ---- dequantized head, built once ------------------------------------- */

static float *W;          /* [rows][cols], dequantized head                */
static float *Wnorm;      /* [rows]                                        */
static int    Vrows, Dcols;

/* Dequantize exactly as matvec_q_range accumulates: per group, code-8 scaled by
 * that group's fp16 scale. Same values, so a full scan here reproduces the
 * device's logits up to fp summation order -- and the oracle's ground truth is
 * its OWN full scan, so pruning correctness is isolated from quantization. */
static void build_dense_head(const QT *t) {
  Vrows = t->rows; Dcols = t->cols;
  W = (float *)malloc((size_t)Vrows * Dcols * sizeof(float));
  Wnorm = (float *)malloc((size_t)Vrows * sizeof(float));
  for (int r = 0; r < Vrows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float *dst = W + (size_t)r * Dcols;
    double nn = 0.0;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group, end = begin + t->group;
      if (end > t->cols) end = t->cols;
      float scale = half2float(sc[gi]);
      for (int j = begin; j < end; j++) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        float v = (float)(code - 8) * scale;
        dst[j] = v; nn += (double)v * v;
      }
    }
    Wnorm[r] = (float)sqrt(nn);
  }
}

/* ---- dump for the cross-checking harness -------------------------------
 *
 * research/certihead_gemma/ reimplements the certified scan in torch, so the
 * same analysis can be run on heads this C harness cannot load (Gemma-3-270M is
 * 262144 x 640 fp32 safetensors, not our quantized format).
 *
 * A reimplementation is worth nothing until it reproduces THIS harness's
 * numbers on THIS model, so both sides need the same dequantized weights and
 * the same hidden states. Rows are dumped in their ORIGINAL, pre-permutation
 * order and the hidden states exactly as the head received them: the torch side
 * re-derives the norm permutation, the tiling and the bounds itself. If it
 * agreed only because it had been handed our intermediate results, the control
 * would prove nothing. */
static FILE *dump_x = NULL;

static void dump_open(const char *prefix) {
  char p[512];
  snprintf(p, sizeof p, "%s_W.f32", prefix);
  FILE *f = fopen(p, "wb");
  if (!f) { perror(p); exit(1); }
  fwrite(W, sizeof(float), (size_t)Vrows * Dcols, f);
  fclose(f);
  snprintf(p, sizeof p, "%s_meta.txt", prefix);
  f = fopen(p, "w");
  fprintf(f, "rows %d\ncols %d\n", Vrows, Dcols);
  fclose(f);
  snprintf(p, sizeof p, "%s_x.f32", prefix);
  dump_x = fopen(p, "wb");
  if (!dump_x) { perror(p); exit(1); }
  fprintf(stderr, "dump: W %d x %d, streaming x -> %s_x.f32\n",
          Vrows, Dcols, prefix);
}

/* ---- offline row reordering -------------------------------------------
 *
 * The first sweep found every bound sound but none close to useful: 64-100% of
 * rows still had to be scored. The reason is structural rather than numerical.
 * Tiles are contiguous in TOKEN-ID order, and token ids are assigned by the
 * tokenizer's merge order, which has nothing to do with where a row points in
 * embedding space. A tile of 32 arbitrary directions has a residual radius
 * nearly as large as the whole matrix, so its bound is nearly the global
 * maximum and certifies nothing.
 *
 * Tiles must be built from rows that resemble each other. Since row order in
 * the head is arbitrary, it can simply be changed: permute the rows offline,
 * store the permutation, and emit logits back at their original indices. The
 * model is untouched -- this is a physical layout change, exactly the "model
 * binary format for streaming inference" idea, and it keeps tiles contiguous so
 * a device still reads whole 64-byte lines.
 *
 *   1  norm   sort by ||w||. The cheapest possible signal: a tile of small-norm
 *             rows has a small bound and dies immediately against a high
 *             incumbent. Ignores direction entirely.
 *   2  kmeans spherical k-means on unit-normalised rows, one cluster per tile.
 *             Groups rows by direction, which is what the residual bound
 *             actually measures.
 */
static int *perm;      /* new (physical) index -> original token id */
static int *inv_perm;  /* original token id -> new physical index   */

static float dot_f(const float *a, const float *b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

static int cmp_norm(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  float nx = Wnorm[x], ny = Wnorm[y];
  return (nx > ny) - (nx < ny);
}

static void permute_rows(int mode, int tile) {
  perm = (int *)malloc((size_t)Vrows * sizeof(int));
  for (int r = 0; r < Vrows; r++) perm[r] = r;
  if (mode == 0) return;

  if (mode == 1) {
    qsort(perm, (size_t)Vrows, sizeof(int), cmp_norm);
  } else {
    int K = (Vrows + tile - 1) / tile;
    float *U = (float *)malloc((size_t)Vrows * Dcols * sizeof(float));
    for (int r = 0; r < Vrows; r++) {           /* unit-normalise for cosine */
      const float *w = W + (size_t)r * Dcols;
      float inv = Wnorm[r] > 1e-12f ? 1.f / Wnorm[r] : 0.f;
      for (int d = 0; d < Dcols; d++) U[(size_t)r * Dcols + d] = w[d] * inv;
    }
    float *C = (float *)malloc((size_t)K * Dcols * sizeof(float));
    int *asg = (int *)malloc((size_t)Vrows * sizeof(int));
    for (int k = 0; k < K; k++)                 /* seed: evenly spaced rows */
      memcpy(C + (size_t)k * Dcols,
             U + (size_t)((long long)k * Vrows / K) * Dcols,
             (size_t)Dcols * sizeof(float));
    for (int it = 0; it < 8; it++) {
      for (int r = 0; r < Vrows; r++) {
        const float *u = U + (size_t)r * Dcols;
        int bk = 0; float bs = -1e30f;
        for (int k = 0; k < K; k++) {
          float s = dot_f(u, C + (size_t)k * Dcols, Dcols);
          if (s > bs) { bs = s; bk = k; }
        }
        asg[r] = bk;
      }
      memset(C, 0, (size_t)K * Dcols * sizeof(float));
      int *cnt = (int *)calloc((size_t)K, sizeof(int));
      for (int r = 0; r < Vrows; r++) {
        float *c = C + (size_t)asg[r] * Dcols;
        const float *u = U + (size_t)r * Dcols;
        for (int d = 0; d < Dcols; d++) c[d] += u[d];
        cnt[asg[r]]++;
      }
      for (int k = 0; k < K; k++) {             /* renormalise to the sphere */
        float *c = C + (size_t)k * Dcols;
        if (!cnt[k]) { memcpy(c, U + (size_t)(k % Vrows) * Dcols,
                              (size_t)Dcols * sizeof(float)); continue; }
        float n = 0.f;
        for (int d = 0; d < Dcols; d++) n += c[d] * c[d];
        n = n > 1e-20f ? 1.f / sqrtf(n) : 0.f;
        for (int d = 0; d < Dcols; d++) c[d] *= n;
      }
      free(cnt);
    }
    /* counting sort by cluster: rows of one cluster become contiguous */
    int *head = (int *)calloc((size_t)K + 1, sizeof(int));
    for (int r = 0; r < Vrows; r++) head[asg[r] + 1]++;
    for (int k = 0; k < K; k++) head[k + 1] += head[k];
    for (int r = 0; r < Vrows; r++) perm[head[asg[r]]++] = r;
    free(U); free(C); free(asg); free(head);
  }

  /* Physically reorder W and Wnorm so tiles are contiguous in memory, which is
   * the whole point -- a device must still stream whole cache lines. */
  float *W2 = (float *)malloc((size_t)Vrows * Dcols * sizeof(float));
  float *N2 = (float *)malloc((size_t)Vrows * sizeof(float));
  for (int r = 0; r < Vrows; r++) {
    memcpy(W2 + (size_t)r * Dcols, W + (size_t)perm[r] * Dcols,
           (size_t)Dcols * sizeof(float));
    N2[r] = Wnorm[perm[r]];
  }
  free(W); free(Wnorm); W = W2; Wnorm = N2;
  inv_perm = (int *)malloc((size_t)Vrows * sizeof(int));
  for (int r = 0; r < Vrows; r++) inv_perm[perm[r]] = r;
}

/* ---- tile summaries ---------------------------------------------------- */

static int    NT, TILE;
static float *Cen;      /* [NT][D] centroid                                 */
static float *MaxRes;   /* [NT]    max ||w_i - c_j||                        */
static float *Lo, *Hi;  /* [NT][D] per-dimension interval                   */

static void build_tiles(int tile) {
  TILE = tile;
  NT = (Vrows + tile - 1) / tile;
  Cen = (float *)calloc((size_t)NT * Dcols, sizeof(float));
  MaxRes = (float *)calloc((size_t)NT, sizeof(float));
  Lo = (float *)malloc((size_t)NT * Dcols * sizeof(float));
  Hi = (float *)malloc((size_t)NT * Dcols * sizeof(float));
  for (int j = 0; j < NT; j++) {
    int b = j * tile, e = b + tile; if (e > Vrows) e = Vrows;
    float *c = Cen + (size_t)j * Dcols;
    float *lo = Lo + (size_t)j * Dcols, *hi = Hi + (size_t)j * Dcols;
    for (int d = 0; d < Dcols; d++) { lo[d] = 1e30f; hi[d] = -1e30f; }
    for (int r = b; r < e; r++) {
      const float *w = W + (size_t)r * Dcols;
      for (int d = 0; d < Dcols; d++) {
        c[d] += w[d];
        if (w[d] < lo[d]) lo[d] = w[d];
        if (w[d] > hi[d]) hi[d] = w[d];
      }
    }
    float inv = 1.f / (float)(e - b);
    for (int d = 0; d < Dcols; d++) c[d] *= inv;
    float mx = 0.f;
    for (int r = b; r < e; r++) {
      const float *w = W + (size_t)r * Dcols;
      double s = 0.0;
      for (int d = 0; d < Dcols; d++) { double t = w[d] - c[d]; s += t * t; }
      float n = (float)sqrt(s); if (n > mx) mx = n;
    }
    MaxRes[j] = mx;
  }
}

/* ---- statistics -------------------------------------------------------- */

static long long positions, mismatches;
static double sum_rows, sum_equiv;
static int *rows_hist;          /* rows scored per position, for percentiles */
static long long hist_n;
static int hot_prev[8];         /* previous argmax and runners-up            */
static int hot_n;
static int BOUND, HOTK;

static int cmp_int(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

static float dot(const float *a, const float *b, int n) {
  float s = 0.f;
  for (int i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

/* The instrumented head. Runs the honest full scan to produce logits (so the
 * model keeps decoding normally and the run stays a real trajectory), then
 * replays the certified search over the same hidden state and checks it. */
static void oracle_head(const QT *t, const float *x, float *y) {
  (void)t;
  if (dump_x) fwrite(x, sizeof(float), (size_t)Dcols, dump_x);
  /* Ground truth: full scan, identical arithmetic to the pruned path. Indices
   * here are PHYSICAL (post-permutation); logits go back to y at the original
   * token id so the model decodes exactly as before. */
  int best = 0; float bestv = -1e30f;
  for (int r = 0; r < Vrows; r++) {
    float v = dot(x, W + (size_t)r * Dcols, Dcols);
    y[perm[r]] = v;
    if (v > bestv) { bestv = v; best = r; }
  }

  /* ---- certified search ---- */
  int scored = 0;
  float inc = -1e30f; int inc_i = -1;

  /* Hot set: last position's winner and runners-up. Costs HOTK rows and buys a
   * high incumbent immediately, which is what makes the bounds bite. Under
   * autoregressive decoding the next winner is very often near the last one. */
  for (int k = 0; k < hot_n && k < HOTK; k++) {
    int o = hot_prev[k];
    if (o < 0 || o >= Vrows) continue;
    int r = inv_perm[o];                 /* hot set is in token ids */
    float v = dot(x, W + (size_t)r * Dcols, Dcols);
    scored++;
    if (v > inc) { inc = v; inc_i = r; }
  }

  float xn = 0.f;
  for (int d = 0; d < Dcols; d++) xn += x[d] * x[d];
  xn = sqrtf(xn);

  /* Tile bounds. Cost is charged below in row-equivalents, not hidden. */
  static float *U = NULL; static int *ord = NULL;
  if (!U) { U = (float *)malloc((size_t)NT * sizeof(float));
            ord = (int *)malloc((size_t)NT * sizeof(int)); }
  for (int j = 0; j < NT; j++) {
    float u;
    if (BOUND == 0) {
      u = dot(x, Cen + (size_t)j * Dcols, Dcols) + xn * MaxRes[j];
    } else {
      const float *lo = Lo + (size_t)j * Dcols, *hi = Hi + (size_t)j * Dcols;
      float s = 0.f;
      for (int d = 0; d < Dcols; d++) {
        float a = x[d] * lo[d], b = x[d] * hi[d];
        s += a > b ? a : b;
      }
      u = s;
      if (BOUND == 2) {
        float u2 = dot(x, Cen + (size_t)j * Dcols, Dcols) + xn * MaxRes[j];
        if (u2 < u) u = u2;   /* min of two valid upper bounds is still valid */
      }
    }
    U[j] = u; ord[j] = j;
  }

  /* Visit tiles most-promising first: a high incumbent early prunes the rest.
   * Insertion by selection would be O(NT^2); a simple descending sort on the
   * bound is what a device would approximate with a few buckets. */
  for (int a = 0; a < NT - 1; a++) {          /* partial selection sort:      */
    int m = a;                                 /* only until bounds fall below */
    for (int b = a + 1; b < NT; b++) if (U[ord[b]] > U[ord[m]]) m = b;
    int tmp = ord[a]; ord[a] = ord[m]; ord[m] = tmp;
    if (U[ord[a]] <= inc) break;               /* everything after is skippable */
  }

  for (int k = 0; k < NT; k++) {
    int j = ord[k];
    if (U[j] <= inc) continue;                 /* certificate: cannot contain max */
    int b = j * TILE, e = b + TILE; if (e > Vrows) e = Vrows;
    for (int r = b; r < e; r++) {
      float v = dot(x, W + (size_t)r * Dcols, Dcols);
      scored++;
      if (v > inc) { inc = v; inc_i = r; }
    }
  }

  if (inc_i != best) mismatches++;

  /* Honest cost: rows actually dotted, PLUS the bound computation expressed in
   * row-equivalents. One centroid bound is D mults (1 row). One box bound is D
   * mults plus D compares (~1 row). Charging zero for bounds would be the
   * "isolated microbenchmark" error this ledger has already been burned by. */
  double bound_rows = (double)NT * (BOUND == 2 ? 2.0 : 1.0);
  sum_rows += scored;
  sum_equiv += scored + bound_rows;
  if (hist_n < 4000000) rows_hist[hist_n++] = scored;
  positions++;

  /* Refresh the hot set from this step's top few, for the next position. */
  /* Carry this step's top-K forward as the next step's hot set. Autoregressive
   * decoding is locally repetitive -- the ledger already measured this model's
   * emitted stream as ~12x more repetitive than the corpus -- so last step's
   * leaders are a cheap, high incumbent. A device would keep this same list;
   * the K rows it costs are charged in `scored` like any others. */
  int K = HOTK < 8 ? HOTK : 8;
  float kv[8]; int ki[8];
  for (int i = 0; i < K; i++) { kv[i] = -1e30f; ki[i] = -1; }
  for (int o = 0; o < Vrows; o++) {
    float v = y[o];
    if (v <= kv[K - 1]) continue;
    int i = K - 1;
    while (i > 0 && kv[i - 1] < v) { kv[i] = kv[i - 1]; ki[i] = ki[i - 1]; i--; }
    kv[i] = v; ki[i] = o;
  }
  for (int i = 0; i < K; i++) hot_prev[i] = ki[i];
  hot_n = K;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: %s <model.bin> <tokens.bin> [windows] [tile] [bound] [hot]\n"
            "  bound: 0=centroid 1=box 2=both\n", argv[0]);
    return 2;
  }
  int windows = argc > 3 ? atoi(argv[3]) : 32;
  int tile = argc > 4 ? atoi(argv[4]) : 32;
  BOUND = argc > 5 ? atoi(argv[5]) : 1;
  HOTK = argc > 6 ? atoi(argv[6]) : 3;

  size_t n; uint8_t *buf = read_file(argv[1], &n);
  Model m; if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }
  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn;
  int V = m.out_vocab, S = m.c.seq_len;

  int pmode = argc > 7 ? atoi(argv[7]) : 0;
  build_dense_head(&m.out_head);
  if (argc > 8) dump_open(argv[8]);
  permute_rows(pmode, tile);
  build_tiles(tile);
  printf("permute %d (%s)\n", pmode,
         pmode == 0 ? "none" : pmode == 1 ? "norm-sorted" : "spherical k-means");

  Scratch s;
  s.x = (float *)malloc(D * 4); s.h = (float *)malloc((F > D ? F : D) * 4);
  s.qkv = (float *)malloc(3 * D * 4); s.att = (float *)malloc(D * 4);
  s.g1 = (float *)malloc(F * 4); s.g2 = (float *)malloc((P > F ? P : F) * 4);
  s.ple = (float *)malloc((size_t)L * P * 4);
  s.tmpP = (float *)malloc((size_t)L * P * 4);
  s.trow = (float *)malloc((size_t)L * P * 4);
  s.logits = (float *)malloc((size_t)V * 4);
  s.scores = (float *)malloc((size_t)m.c.n_heads * S * 4);
  s.kcache8 = (int8_t *)malloc((size_t)L * S * llm_krow(&m.c));
  s.vcache8 = (int8_t *)malloc((size_t)L * S * D);
  s.kscale = (float *)malloc((size_t)L * S * 4);
  s.vscale = (float *)malloc((size_t)L * S * 4);
  s.wq = (int8_t *)malloc((size_t)m.c.n_heads * S);
  s.wscale = (float *)malloc((size_t)m.c.n_heads * 4);
  s.acc = (int32_t *)malloc((size_t)2 * D * 4);
  s.qq = (int8_t *)malloc(llm_krow(&m.c));

  rows_hist = (int *)malloc(sizeof(int) * 4000000);
  m.head_matvec = oracle_head;

  size_t vn; uint16_t *val = (uint16_t *)read_file(argv[2], &vn);
  size_t ntok = vn / 2;
  size_t win = (size_t)S;
  size_t avail = ntok / win;
  if ((size_t)windows > avail) windows = (int)avail;

  for (int w = 0; w < windows; w++) {
    const uint16_t *seq = val + (size_t)w * win;
    for (int pos = 0; pos + 1 < (int)win; pos++)
      llm_forward(&m, seq[pos], pos, &s);
    hot_n = 0;   /* windows are independent; do not carry locality across them */
  }

  qsort(rows_hist, (size_t)hist_n, sizeof(int), cmp_int);
  int p50 = rows_hist[hist_n / 2];
  int p95 = rows_hist[(size_t)(hist_n * 0.95)];
  int p99 = rows_hist[(size_t)(hist_n * 0.99)];
  double mean = sum_rows / (double)positions;
  double meq = sum_equiv / (double)positions;

  printf("tile %3d  bound %d  hot %d   rows %d\n", tile, BOUND, HOTK, Vrows);
  printf("  positions        : %lld\n", positions);
  printf("  argmax mismatches: %lld  %s\n", mismatches,
         mismatches ? "*** UNSOUND ***" : "(exact)");
  printf("  rows scored mean : %.1f  (%.2f%% of head)\n", mean,
         100.0 * mean / Vrows);
  printf("  + bound overhead : %.1f  (%.2f%% of head)  <- honest cost\n", meq,
         100.0 * meq / Vrows);
  printf("  p50 / p95 / p99  : %d / %d / %d  (%.1f%% / %.1f%% / %.1f%%)\n",
         p50, p95, p99, 100.0 * p50 / Vrows, 100.0 * p95 / Vrows,
         100.0 * p99 / Vrows);
  /* Bytes, not rows. The head is bandwidth bound, so the only figure that can
   * predict a device gain is PSRAM traffic per token -- and the index is
   * traffic too. Reporting "30% of rows" while silently reading a megabyte of
   * bounds would be exactly the isolated-microbenchmark error this project has
   * already been burned by twice.
   *
   * Row cost is the device's own figure: 1.26 MiB / 25,353 rows.
   * Index is priced INT8: lo/hi quantised per dimension, rounded outward (lo
   * down, hi up) so the bound stays a valid upper bound. Rounding outward only
   * loosens it, never breaks soundness. */
  const double row_bytes = 1.26 * 1048576.0 / (double)Vrows;
  double idx_box_i8 = (double)NT * Dcols * 2;            /* lo,hi int8        */
  double idx_cen_i8 = (double)NT * (Dcols + 4);          /* centroid + radius */
  double idx = (BOUND == 0) ? idx_cen_i8
             : (BOUND == 1) ? idx_box_i8 : idx_box_i8 + idx_cen_i8;
  double rows_bytes = mean * row_bytes;
  double base = (double)Vrows * row_bytes;
  printf("  index (int8)     : %.0f B  %s\n", idx,
         idx <= 65536 ? "<- fits internal SRAM comfortably"
                      : idx <= 262144 ? "<- large for SRAM, may need PSRAM"
                                      : "<- must live in PSRAM, counts as traffic");
  printf("  PSRAM per token  : rows %.0f B + index %.0f B = %.0f B\n",
         rows_bytes, idx, rows_bytes + idx);
  printf("  vs dense %.0f B  : %.2fx less traffic (index in PSRAM)\n",
         base, base / (rows_bytes + idx));
  printf("  if index in SRAM : %.2fx less traffic\n", base / rows_bytes);
  return mismatches ? 1 : 0;
}
