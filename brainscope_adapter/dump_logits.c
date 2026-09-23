// Print the C runtime's last-position logits for a prompt given as token ids.
// The brainscope twin (build_hf.py) must reproduce these numbers; verify_vs_c.py
// runs both sides and diffs them. Reuses runtime/llm.h unmodified - the same
// portable inference verify.c gates before anything touches the board.
//
//   cc -O3 -Wall -Wextra -I runtime -o dump_logits
//     brainscope_adapter/dump_logits.c -lm
//   ./dump_logits artifacts/tinystories/model.bin 1 500 1000 200 42 777 13 99
#include <stdio.h>
#include <stdlib.h>
#include "llm.h"

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
    fprintf(stderr, "usage: %s <model.bin> <token id> [id ...]\n", argv[0]);
    return 2;
  }
  size_t n;
  uint8_t *buf = read_file(argv[1], &n);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }

  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn,
      V = m.out_vocab, S = m.c.seq_len;
  Scratch s;
  s.x = malloc(D * 4); s.h = malloc((F > D ? F : D) * 4);
  s.qkv = malloc(3 * D * 4); s.att = malloc(D * 4);
  s.g1 = malloc(F * 4); s.g2 = malloc((P > F ? P : F) * 4);
  s.ple = malloc(L * P * 4); s.tmpP = malloc(L * P * 4); s.trow = malloc(L * P * 4);
  s.logits = malloc(V * 4);
  s.scores = malloc(S * 4);
  s.kcache = malloc((size_t)L * S * D * 4);
  s.vcache = malloc((size_t)L * S * D * 4);

  int plen = argc - 2;
  for (int i = 0; i < plen; i++) {
    int id = atoi(argv[2 + i]);
    if (id < 0 || id >= m.c.vocab) { fprintf(stderr, "id %d out of range\n", id); return 1; }
    llm_forward(&m, id, i, &s);
  }
  for (int i = 0; i < V; i++) printf("%.6f\n", s.logits[i]);
  return 0;
}
