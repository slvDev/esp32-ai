// Export the deployed C runtime's own next-token decisions for cheap bypass tests.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../llm.h"

static uint8_t *read_file(const char *path, size_t *size) {
  FILE *stream = fopen(path, "rb");
  if (!stream) { perror(path); exit(1); }
  fseek(stream, 0, SEEK_END);
  *size = (size_t)ftell(stream);
  fseek(stream, 0, SEEK_SET);
  uint8_t *data = (uint8_t *)malloc(*size);
  if (!data || fread(data, 1, *size, stream) != *size) {
    fprintf(stderr, "failed to read %s\n", path);
    exit(1);
  }
  fclose(stream);
  return data;
}

static void write_u32(FILE *stream, uint32_t value) {
  if (fwrite(&value, sizeof(value), 1, stream) != 1) exit(1);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s model.bin tokens.bin windows output.bin\n", argv[0]);
    return 2;
  }
  int windows = atoi(argv[3]);
  if (windows <= 0) return 2;

  size_t model_bytes;
  uint8_t *model_data = read_file(argv[1], &model_bytes);
  Model model;
  if (llm_load(model_data, &model)) {
    fprintf(stderr, "model load failed\n");
    return 1;
  }

  const int D = model.c.dim, L = model.c.n_layers, P = model.c.ple_dim;
  const int F = model.c.ffn, V = model.out_vocab, S = model.c.seq_len;
  Scratch scratch;
  scratch.x = (float *)malloc(D * 4);
  scratch.h = (float *)malloc((F > D ? F : D) * 4);
  scratch.qkv = (float *)malloc(3 * D * 4);
  scratch.att = (float *)malloc(D * 4);
  scratch.g1 = (float *)malloc(F * 4);
  scratch.g2 = (float *)malloc((P > F ? P : F) * 4);
  scratch.ple = (float *)malloc(L * P * 4);
  scratch.tmpP = (float *)malloc(L * P * 4);
  scratch.trow = (float *)malloc(L * P * 4);
  scratch.logits = (float *)malloc(V * 4);
#ifdef LLM_KV_INT8
  scratch.scores = (float *)malloc((size_t)model.c.n_heads * S * 4);
  scratch.kcache8 = (int8_t *)malloc((size_t)L * S * llm_krow(&model.c));
  scratch.vcache8 = (int8_t *)malloc((size_t)L * S * D);
  scratch.kscale = (float *)malloc((size_t)L * S * 4);
  scratch.vscale = (float *)malloc((size_t)L * S * 4);
  scratch.wq = (int8_t *)malloc((size_t)model.c.n_heads * S);
  scratch.wscale = (float *)malloc((size_t)model.c.n_heads * 4);
  scratch.acc = (int32_t *)malloc((size_t)2 * D * sizeof(int32_t));
  scratch.qq = (int8_t *)malloc(llm_krow(&model.c));
#else
  scratch.scores = (float *)malloc(S * 4);
  scratch.kcache = (float *)malloc((size_t)L * S * D * 4);
  scratch.vcache = (float *)malloc((size_t)L * S * D * 4);
#endif

  size_t token_bytes;
  uint16_t *tokens = (uint16_t *)read_file(argv[2], &token_bytes);
  size_t token_count = token_bytes / sizeof(uint16_t);
  if ((size_t)windows * S > token_count) {
    fprintf(stderr, "requested %d windows but token file holds only %zu\n",
            windows, token_count / S);
    return 2;
  }

  FILE *output = fopen(argv[4], "wb");
  if (!output) { perror(argv[4]); return 1; }
  fwrite("TARG", 1, 4, output);
  write_u32(output, 1);
  write_u32(output, (uint32_t)windows);
  write_u32(output, (uint32_t)S);

  uint16_t *decisions = (uint16_t *)malloc((size_t)S * sizeof(uint16_t));
  for (int window = 0; window < windows; ++window) {
    uint16_t *input = tokens + (size_t)window * S;
    for (int pos = 0; pos < S; ++pos) {
      llm_forward(&model, input[pos], pos, &scratch);
      int best = 0;
      for (int token = 1; token < V; ++token) {
        if (scratch.logits[token] > scratch.logits[best]) best = token;
      }
      decisions[pos] = (uint16_t)best;
    }
    fwrite(input, sizeof(uint16_t), (size_t)S, output);
    fwrite(decisions, sizeof(uint16_t), (size_t)S, output);
    fprintf(stderr, "\rteacher windows %d/%d", window + 1, windows);
  }
  fprintf(stderr, "\n");
  fclose(output);
  return 0;
}
