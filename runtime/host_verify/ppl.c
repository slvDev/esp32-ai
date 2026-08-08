// Host perplexity harness: run the C inference over real val tokens and report
// cross-entropy. Compile with and without -DLLM_INT8_ACT to measure the exact
// quality cost of int8-activation quantization (the SIMD numerics change) before
// committing it to the device kernel.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "../llm.h"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

#ifdef LLM_HOST_SPLIT
/* Same halving the device uses: worker takes [0, n/2), this core takes
 * [n/2, n). Run sequentially so the split boundary is exercised with no
 * concurrency in play. */
static void host_split_par_for(void (*fn)(void *, int, int), void *ctx, int n) {
  if (n < 2) { fn(ctx, 0, n); return; }
  fn(ctx, 0, n / 2);
  fn(ctx, n / 2, n);
}
#endif

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model.bin> <token-data.bin> [windows]\n", argv[0]);
    return 2;
  }
  const char *bin = argv[1];
  const char *valp = argv[2];
  int windows = argc > 3 ? atoi(argv[3]) : 8;

  size_t n; uint8_t *buf = read_file(bin, &n);
  Model m; if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }
#ifdef LLM_HOST_SPLIT
  /* Exercise every par_for decomposition WITHOUT threads.
   *
   * The device splits attention's score, softmax and V-accumulation passes
   * across two cores. Each split is claimed exact "by construction" -- the
   * halves are supposed to share no state and reorder no float reduction. That
   * claim has never been testable on the host, because the host leaves par_for
   * NULL and always runs the whole range in one call.
   *
   * This runs the two halves as two sequential calls on the same thread. It
   * cannot detect a data race, but it detects the thing that actually goes
   * wrong in this codebase: a range function that reads state its own half did
   * not produce, or a reduction whose result depends on being done in one pass.
   * If CE moves against the unsplit build, a split is not exact.
   *
   * Built as a separate binary rather than a runtime flag so the comparison is
   * between two whole programs, which is the same discipline the board
   * captures use. */
  m.par_for = host_split_par_for;
#endif
  // V is the OUTPUT vocabulary: the softmax is over the logits the model
  // produces, not over padded embedding rows it never scores.
  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn, V = m.out_vocab, S = m.c.seq_len;

  Scratch s;
  s.x = malloc(D*4); s.h = malloc((F>D?F:D)*4); s.qkv = malloc(3*D*4); s.att = malloc(D*4);
  s.g1 = malloc(F*4); s.g2 = malloc((P>F?P:F)*4);
  s.ple = malloc(L*P*4); s.tmpP = malloc(L*P*4); s.trow = malloc(L*P*4);
  s.logits = malloc(V*4);
#ifdef LLM_KV_INT8
  /* The int8 KV path scores every head in one traversal, so all heads' scores
     are live at once; and the cache itself is int8 with per-row scales. */
  s.scores = malloc((size_t)m.c.n_heads*S*4);
  /* K uses the padded per-head layout; see llm_krow() in llm.h. Computing
     L*S*D here would under-allocate and corrupt attention silently. */
  s.kcache8 = malloc((size_t)L*S*llm_krow(&m.c)); s.vcache8 = malloc((size_t)L*S*D);
  s.kscale = malloc((size_t)L*S*4);  s.vscale = malloc((size_t)L*S*4);
  s.wq = malloc((size_t)m.c.n_heads*S); s.wscale = malloc((size_t)m.c.n_heads*4);
  /* Two V-accumulation banks plus the quantized query, moved out of
     llm_forward's frame; see the Scratch declaration in llm.h. */
  s.acc = malloc((size_t)2*D*4); s.qq = malloc(llm_krow(&m.c));
#else
  s.scores = malloc(S*4);
  s.kcache = malloc((size_t)L*S*D*4); s.vcache = malloc((size_t)L*S*D*4);
#endif

  size_t vn; uint16_t *val = (uint16_t *)read_file(valp, &vn);
  size_t n_tok = vn / 2;

  double nll = 0.0; long count = 0;
  for (int w = 0; w < windows; w++) {
    size_t base = (size_t)w * S;
    if (base + S + 1 > n_tok) break;
    for (int pos = 0; pos < S; pos++) {
      int tok = val[base + pos];
      llm_forward(&m, tok, pos, &s);
      int target = val[base + pos + 1];
      // cross-entropy at this position
      float mx = -1e30f;
      for (int v = 0; v < V; v++) if (s.logits[v] > mx) mx = s.logits[v];
      double sum = 0.0;
      for (int v = 0; v < V; v++) sum += exp((double)s.logits[v] - mx);
      double ce = log(sum) - ((double)s.logits[target] - mx);
      nll += ce; count++;
    }
  }
  double mean = nll / count;
#ifdef LLM_INT8_ACT
#  ifdef LLM_KV_INT8
  const char *mode = "int8-act +int8KV";
#  else
  const char *mode = "int8-act fp32KV";
#  endif
#else
  const char *mode = "fp32-activations";
#endif
  printf("%-18s  val CE %.4f  ppl %.2f   (%ld predictions)\n",
         mode, mean, exp(mean), count);
  return 0;
}
