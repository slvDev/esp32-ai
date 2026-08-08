/* Capture the real deployed TinyStories residual stream for Cycle 05.
 * Compile with the same LLM_INT8_ACT/LLM_KV_INT8 flags as the board runtime.
 * The trace hook is a no-op in every build that does not define it. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static FILE *trace_file;
static void trace_event(int stage, int layer, const float *values, int count);
#define LLM_TRACE_EVENT(stage, layer, values, count) \
  trace_event((stage), (layer), (values), (count))
#include "../llm.h"

static void trace_event(int stage, int layer, const float *values, int count) {
  (void)stage; (void)layer;
  if (trace_file && fwrite(values, sizeof(float), (size_t)count, trace_file) !=
                        (size_t)count) {
    fprintf(stderr, "trace write failed\n");
    exit(1);
  }
}

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
  memset(&scratch, 0, sizeof scratch);
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
  if ((size_t)windows * S + 1 > token_count) {
    fprintf(stderr, "requested %d windows but token file holds only %zu\n",
            windows, (token_count - 1) / S);
    return 2;
  }

  trace_file = fopen(argv[4], "wb");
  if (!trace_file) { perror(argv[4]); return 1; }
  fwrite("EVDT", 1, 4, trace_file);
  write_u32(trace_file, 1);
  write_u32(trace_file, (uint32_t)windows);
  write_u32(trace_file, (uint32_t)S);
  write_u32(trace_file, (uint32_t)D);
  write_u32(trace_file, (uint32_t)L);
  write_u32(trace_file, (uint32_t)V);
  write_u32(trace_file, (uint32_t)(3 * L + 3));

  for (int window = 0; window < windows; ++window) {
    size_t base = (size_t)window * S;
    for (int pos = 0; pos < S; ++pos) {
      uint16_t token = tokens[base + pos];
      uint16_t target = tokens[base + pos + 1];
      if (fwrite(&token, 2, 1, trace_file) != 1 ||
          fwrite(&target, 2, 1, trace_file) != 1) return 1;
      llm_forward(&model, token, pos, &scratch);
      uint32_t top1 = 0;
      float max_logit = scratch.logits[0];
      for (int v = 1; v < V; ++v) {
        if (scratch.logits[v] > max_logit) {
          max_logit = scratch.logits[v];
          top1 = (uint32_t)v;
        }
      }
      float second = -1e30f;
      double sum = 0.0;
      for (int v = 0; v < V; ++v) {
        if ((uint32_t)v != top1 && scratch.logits[v] > second)
          second = scratch.logits[v];
        sum += exp((double)scratch.logits[v] - max_logit);
      }
      float ce = (float)(log(sum) - ((double)scratch.logits[target] - max_logit));
      float margin = max_logit - second;
      fwrite(&top1, 4, 1, trace_file);
      fwrite(&ce, 4, 1, trace_file);
      fwrite(&margin, 4, 1, trace_file);
    }
    fflush(trace_file);
    fprintf(stderr, "heartbeat window=%d/%d tokens=%d\n", window + 1, windows,
            (window + 1) * S);
    fflush(stderr);
  }
  fclose(trace_file);
  trace_file = NULL;
  return 0;
}
