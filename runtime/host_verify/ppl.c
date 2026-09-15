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
  // V is the OUTPUT vocabulary: the softmax is over the logits the model
  // produces, not over padded embedding rows it never scores.
  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn, V = m.out_vocab, S = m.c.seq_len;

  Scratch s;
  s.x = malloc(D*4); s.h = malloc((F>D?F:D)*4); s.qkv = malloc(3*D*4); s.att = malloc(D*4);
  s.g1 = malloc(F*4); s.g2 = malloc((P>F?P:F)*4);
  s.ple = malloc(L*P*4); s.tmpP = malloc(L*P*4); s.trow = malloc(L*P*4);
  s.logits = malloc(V*4); s.scores = malloc(S*4);
  s.kcache = malloc((size_t)L*S*D*4); s.vcache = malloc((size_t)L*S*D*4);
#ifdef LLM_KV_QUANT
  llm_kv_quant_bind(&m, &s, malloc(llm_kv_quant_bytes(&m)), NULL, NULL);
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
  const char *mode = "int8-activations";
#else
  const char *mode = "fp32-activations";
#endif
#ifdef LLM_KV_QUANT
  mode = "int8-act + quant-KV";
#endif
  printf("%-18s  val CE %.4f  ppl %.2f   (%ld predictions)\n",
         mode, mean, exp(mean), count);
  return 0;
}
