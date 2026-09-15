// PLE TinyLM inference on the ESP32-S3, prompted over serial.
//
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped. Placement follows reads-per-token rather than what happens to
// fit:
//
//   flash   PLE table + token embedding   one row of the 25.2M-parameter
//                                         table per token
//   PSRAM   staged int8 core + head, KV   read once per position
//   SRAM    scratch + norm vectors        touched many times per token
//
// The logits array stays in PSRAM: 25,353 floats is 99 KiB, and the sampler
// reads it once per token.
//
// Type a prompt (ASCII, up to 512 bytes) and press return. It is encoded on
// the device with the same ByteLevel BPE the model was trained with, then the
// story is sampled one token at a time and streamed back.
//
// Same llm.h that is verified against PyTorch on the host; only the platform
// hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_random.h"

// int8 activations, required by the staged int8 kernel. Not bit-exact against
// the fp32 golden; verify.c must be built without this flag. Validation CE cost
// (runtime/host_verify/ppl.c, 32,768 predictions): 2.4793 -> 2.4796, ppl 11.93 / 11.94.
#define LLM_INT8_ACT 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
// On the ESP32-P4 the staged int8 rows are padded to 16 bytes so the PIE
// vector kernel reads whole vectors; every other target keeps rows packed.
#if CONFIG_IDF_TARGET_ESP32P4
#define LLM_STAGE_ALIGN 16
// Attention on the quantized KV cache, with both passes as PIE vector dots.
#define LLM_KV_QUANT 1
#include "../../runtime/llm_pie_dot.h"
#define LLM_DOT_S8V  llm_pie_dot_s8v
#define LLM_DOT_S16V llm_pie_dot_s16v
#endif
#include "../../runtime/llm.h"
#if CONFIG_IDF_TARGET_ESP32P4
// The head is bound by PSRAM latency, not compute: ask the L1 data cache's
// preload engine for the next block of rows while the current one computes.
// One engine serves both cores, so requests are serialised with a spinlock
// and a core waits for the engine to be idle before handing it a block.
#include "esp32p4/rom/cache.h"
// HEAD_PRELOAD: 0 none, 1 the L1 data cache engine (64 KB cache), 2 the L2
// engine (256 KB cache). HEAD_PRELOAD_BYTES is the block requested ahead.
#ifndef HEAD_PRELOAD
#define HEAD_PRELOAD 1
#endif
#ifndef HEAD_PRELOAD_BYTES
#define HEAD_PRELOAD_BYTES 8192
#endif
#if HEAD_PRELOAD == 1
#define PRELOAD_DONE()          Cache_L1_DCache_Preload_Done()
#define PRELOAD_START(a, n)     Cache_Start_L1_DCache_Preload((a), (n), 0)
#define PRELOAD_END(auto_)      Cache_End_L1_DCache_Preload(auto_)
#elif HEAD_PRELOAD == 2
#define PRELOAD_DONE()          Cache_L2_Cache_Preload_Done()
#define PRELOAD_START(a, n)     Cache_Start_L2_Cache_Preload((a), (n), 0)
#define PRELOAD_END(auto_)      Cache_End_L2_Cache_Preload(auto_)
#endif
#if HEAD_PRELOAD
static portMUX_TYPE preload_mux = portMUX_INITIALIZER_UNLOCKED;
static uint32_t preload_autoload = 1;
static inline void head_preload_start(const void *addr, size_t bytes) {
  for (;;) {
    portENTER_CRITICAL(&preload_mux);
    if (PRELOAD_DONE()) {
      preload_autoload = PRELOAD_START((uint32_t)(uintptr_t)addr, bytes);
      portEXIT_CRITICAL(&preload_mux);
      return;
    }
    portEXIT_CRITICAL(&preload_mux);
  }
}
static inline void head_preload_end(void) {
  while (!PRELOAD_DONE()) { }
  portENTER_CRITICAL(&preload_mux);
  PRELOAD_END(preload_autoload);
  portEXIT_CRITICAL(&preload_mux);
}
#define LLM_PRELOAD_BYTES HEAD_PRELOAD_BYTES
#define LLM_PRELOAD_START(addr, bytes) head_preload_start((addr), (bytes))
#define LLM_PRELOAD_END() head_preload_end()
#endif
#endif
#include "../../runtime/llm_pie.h"
#include "../../runtime/bpe_tokenizer.h"
#include "generated/vocab.h"
#include "generated/tokenizer_encoder.h"

// Set to 1 once a display is wired up - see display.h.
// Leave 0 to run serial-only (no panel needed).
#define USE_DISPLAY 0
#if USE_DISPLAY
#include "display.h"
#endif

// Tokens generated after the prompt, unless the story ends or the context fills.
static const int N_GENERATE = 200;
// Positions kept for the story, so a long prompt cannot leave nothing to write.
#define STORY_ROOM 64
// <|endoftext|> in the TinyStories tokenizer: the model emits it when a story ends.
#define EOT_ID 0
// Sampling, the same scheme as TinyLM.generate() in src/model.py. TEMPERATURE 0
// selects greedy decoding, which reproduces the original fixed-prompt demo.
static const float TEMPERATURE = 0.8f;
static const int TOP_K = 40;

static Model model;
static Scratch s;
static BpeTokenizer tokenizer;
static bool ready = false;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

static void *ps(size_t n) {
  // 16-byte aligned: staged weight rows must start on vector boundaries.
  void *p = heap_caps_aligned_alloc(16, n, MALLOC_CAP_SPIRAM);
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    Serial.printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    while (1) delay(1000);
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core. On the ESP32-P4 the rows
// go through the PIE vector kernel; elsewhere through the scalar one.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;
// Attention job: the worker takes heads [0, job_h) of layer job_l at job_pos.
static int job_l, job_pos, job_h;
enum { JOB_MATVEC, JOB_ATTN };
static int job_kind = JOB_MATVEC;

#if LLM_HAVE_PIE
// The head's activations in the layout the int4 kernel reads, built once per
// matvec by quantize_shared (both cores read them).
static int8_t xperm_shared[LLM_Q8_MAX_INPUT] __attribute__((aligned(16)));
static int32_t xsum8_shared[LLM_Q8_MAX_INPUT / 32 + 1];
#endif

// Ranged int8 matvec for one staged tensor on the calling core.
static void matvec_rows(const QT *t, const int8_t *xq, float xs, float *y,
                        int row_begin, int row_end) {
#if LLM_HAVE_PIE
  if (llm_pie4_ok(t)) { matvec_pie4_range(t, xperm_shared, xsum8_shared, xs, y, row_begin, row_end); return; }
  if (llm_pie_ok(t)) { matvec_pie_range(t, xq, xs, y, row_begin, row_end); return; }
#endif
  matvec_i8_range(t, xq, xs, y, row_begin, row_end);
}

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
#ifdef LLM_KV_QUANT
    if (job_kind == JOB_ATTN)
      llm_attend_heads(&model, &s, job_l, job_pos, 0, job_h, s.q8b, s.w16b, s.scoresb);
    else
#endif
      matvec_rows(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

#ifdef LLM_KV_QUANT
// Heads are independent: the worker core takes the first half with its own
// temporaries while this core takes the rest, each writing its own slice of
// s.att.
static void attn_par(Model *m, Scratch *sc, int l, int pos) {
  int H = m->c.n_heads;
  job_kind = JOB_ATTN; job_l = l; job_pos = pos; job_h = H / 2;
  xTaskNotifyGive(worker_h);
  llm_attend_heads(m, sc, l, pos, H / 2, H, sc->q8, sc->w16, sc->scores);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  job_kind = JOB_MATVEC;
}
#endif

// Quantize x into the shared activation buffer. The vector kernel reads the
// row's padded width, so the bytes past cols are cleared every call: the
// previous tensor may have been wider and left its values there.
static int8_t xq_shared[LLM_Q8_MAX_INPUT] __attribute__((aligned(16)));
static float quantize_shared(const QT *t, const float *x) {
  float xs;
  quantize_act(x, t->cols, xq_shared, &xs);
  for (int j = t->cols; j < t->stride8; j++) xq_shared[j] = 0;
#if LLM_HAVE_PIE
  if (t->w4) llm_pie4_prepare(xq_shared, t->cols, t->group, xperm_shared, xsum8_shared);
#endif
  return xs;
}

static void matvec_par(const QT *t, const float *x, float *y) {
  if (t->w8 == NULL && t->w4 == NULL) { MATVEC(t, x, y); return; }
  float xs = quantize_shared(t, x);      // once; both cores read the result
  if (t->rows < 128) { matvec_rows(t, xq_shared, xs, y, 0, t->rows); return; }
  job_t = t; job_xq = xq_shared; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_rows(t, xq_shared, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

#if LLM_HAVE_PIE
// The vector kernel must reproduce the scalar kernel exactly: same integer
// sums, same float operations in the same order. Check one tensor of each
// width the model has, plus a slice of the head, before trusting it.
static bool pie_self_check() {
  const QT *cases[] = { &model.ple_model_proj, &model.qkv[0], &model.down[0],
                        &model.ple_proj[0], &model.out_head };
  static float ref[1024], got[1024], x[LLM_Q8_MAX_INPUT];
  double worst = 0;
  for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    const QT *t = cases[c];
    for (int j = 0; j < t->cols; j++) x[j] = sinf(0.37f * j + c) * 1.7f;
    float xs = quantize_shared(t, x);
    int rows = t->rows < 1024 ? t->rows : 1024;
    if (llm_pie4_ok(t)) {
      // int4 staging: the reference is the scalar walk of the same nibbles.
      matvec_q8_range(t, xq_shared, xs, ref, 0, rows);
      matvec_pie4_range(t, xperm_shared, xsum8_shared, xs, got, 0, rows);
    } else if (llm_pie_ok(t)) {
      matvec_i8_range(t, xq_shared, xs, ref, 0, rows);
      matvec_pie_range(t, xq_shared, xs, got, 0, rows);
    } else { Serial.printf("pie: tensor %u not eligible\n", c); return false; }
    for (int r = 0; r < rows; r++) {
      double d = fabs((double)ref[r] - got[r]);
      if (d > worst) worst = d;
    }
  }
  // The attention dots, including the 40-bit read of the int16 accumulator:
  // extreme values first, then a pseudo-random fill, against the scalar sums.
  static int8_t a8[256] __attribute__((aligned(16))), b8[256] __attribute__((aligned(16)));
  static int16_t a16[256] __attribute__((aligned(16))), b16[256] __attribute__((aligned(16)));
  int dot_bad = 0;
  for (int pass = 0; pass < 2; pass++) {
    uint32_t r = 0x9E3779B9u ^ pass;
    for (int i = 0; i < 256; i++) {
      r ^= r << 13; r ^= r >> 17; r ^= r << 5;
      a8[i] = pass ? (int8_t)(r % 255) - 127 : (i & 1 ? 127 : -127);
      b8[i] = pass ? (int8_t)((r >> 8) % 255) - 127 : 127;
      a16[i] = pass ? (int16_t)(r % 65535) - 32767 : (i & 1 ? 32767 : -32767);
      b16[i] = pass ? (int16_t)((r >> 16) % 65535) - 32767 : (i & 1 ? 32767 : 32767);
    }
    for (int nvec = 1; nvec <= 16; nvec++) {
      if (llm_pie_dot_s8v(a8, b8, nvec) != llm_dot_s8v_scalar(a8, b8, nvec)) dot_bad++;
      if (llm_pie_dot_s16v(a16, b16, nvec) != llm_dot_s16v_scalar(a16, b16, nvec)) dot_bad++;
    }
  }
  Serial.printf("pie self-check: matvec max |vector - scalar| = %g, dot mismatches %d -> %s\n",
                worst, dot_bad, (worst == 0 && dot_bad == 0) ? "ok" : "FAIL");
  return worst == 0 && dot_bad == 0;
}
#endif

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  Serial.printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
#ifdef LLM_KV_QUANT
  // Quantized KV cache: keys, values and scales in PSRAM (the caches cover
  // them well: keeping the keys in SRAM measured no gain, and the SRAM is
  // better spent on the head's scale table), the per-head temporaries in SRAM.
  llm_kv_quant_bind(&model, &s,
                    ps_or_die(llm_kv_main_bytes(&model) + llm_kv_key_bytes(&model), "kv cache"),
                    NULL,
                    sram_or_die(llm_kv_hot_bytes(&model), "kv temporaries"));
  s.kcache = s.vcache = NULL;
#else
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
#endif
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

// Emit one token to every active output (serial always; panel when enabled).
static void emit(int tok) {
  if (tok >= VOCAB_N) return;
  const unsigned char *bytes = VOCAB_BLOB + VOCAB_OFF[tok];
  int len = VOCAB_OFF[tok + 1] - VOCAB_OFF[tok];
  // Non-blocking: when no host is draining the USB-CDC buffer (running as a
  // standalone gadget on the display), skip the write instead of stalling the
  // whole generation once the TX buffer fills.
  if ((int)Serial.availableForWrite() >= len) Serial.write(bytes, len);
#if USE_DISPLAY
  display_puts(bytes, len);
#endif
}

// ---- sampling ---------------------------------------------------------------
// Next token from the logits: greedy at TEMPERATURE 0, otherwise a draw from
// the TOP_K most likely tokens at TEMPERATURE. One pass over the logits keeps
// a small list sorted by score; the softmax then runs over that list only.
#define TOP_K_MAX 64
static int pick_token(const float *logits, int n) {
  if (TEMPERATURE <= 0.f || TOP_K <= 0) {
    int best = 0;
    for (int v = 1; v < n; v++) if (logits[v] > logits[best]) best = v;
    return best;
  }
  static int idx[TOP_K_MAX];
  static float sc[TOP_K_MAX];
  int k = TOP_K > TOP_K_MAX ? TOP_K_MAX : TOP_K, cnt = 0;
  for (int v = 0; v < n; v++) {
    float l = logits[v];
    if (cnt == k && l <= sc[k - 1]) continue;
    int i = (cnt < k) ? cnt++ : k - 1;
    while (i > 0 && sc[i - 1] < l) { sc[i] = sc[i - 1]; idx[i] = idx[i - 1]; i--; }
    sc[i] = l; idx[i] = v;
  }
  float mx = sc[0], sum = 0.f;
  for (int i = 0; i < cnt; i++) { sc[i] = expf((sc[i] - mx) / TEMPERATURE); sum += sc[i]; }
  float r = (float)esp_random() / 4294967296.0f * sum;
  for (int i = 0; i < cnt; i++) { r -= sc[i]; if (r <= 0.f) return idx[i]; }
  return idx[cnt - 1];
}

// ---- generation -------------------------------------------------------------
static void tell_story(const char *prompt) {
  uint16_t ids[BTK_MAX_INPUT_BYTES];
  int n = bpe_encode_ascii(&tokenizer, prompt, ids, BTK_MAX_INPUT_BYTES);
  if (n == BTK_ERR_NOT_ASCII) { Serial.println("(ascii only)"); return; }
  int max_prompt = model.c.seq_len - STORY_ROOM;
  if (n <= 0 || n > max_prompt) {
    Serial.printf("(prompt too long: %d tokens, max %d)\n", n, max_prompt);
    return;
  }

  // Prime with the prompt. The KV cache is simply overwritten from position 0.
  Serial.print(">>> ");
  int pos = 0, tok = 0;
  for (int i = 0; i < n; i++) {
    tok = ids[i];
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);
  int64_t t_start = esp_timer_get_time(), decode_us = 0;
  int decoded = 0;
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
    tok = pick_token(s.logits, model.out_vocab);
    if (tok == EOT_ID) break;
    emit(tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens
  }
  int64_t total_us = esp_timer_get_time() - t_start;
  blink(0);

  Serial.printf("\n\n--- %d prompt + %d generated tokens in %.2f s ---\n",
                n, decoded, total_us / 1e6);
  if (decoded) {
    Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                  decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  }
  if (s.profile.calls) {
    float k = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / k, s.profile.attn_us / k,
                  s.profile.ffn_us / k, s.profile.ple_us / k,
                  s.profile.head_us / k);
  }
#if USE_DISPLAY
  if (decoded) display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
}

// The line buffer holds BTK_MAX_INPUT_BYTES, but the CDC receive queue defaults
// to 256, so a longer paste would overrun it before loop() drains it.
#define SERIAL_RX_BYTES (2 * BTK_MAX_INPUT_BYTES)

void setup() {
  // Must be requested before begin(); a failed request falls back to 256.
  size_t rx_bytes = Serial.setRxBufferSize(SERIAL_RX_BYTES);
  Serial.begin(115200);
  delay(1500);
  Serial.printf("\n=== %s PLE TinyLM ===\n", CONFIG_IDF_TARGET);
  if (rx_bytes != SERIAL_RX_BYTES) {
    Serial.println("serial RX buffer allocation failed");
    return;
  }

  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE,
                         &tokenizer)) {
    Serial.println("tokenizer load failed");
    return;
  }

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("model partition not found"); return; }
  const void *base;
  esp_partition_mmap_handle_t h;
  esp_err_t err = esp_partition_mmap(part, 0, part->size,
                                     ESP_PARTITION_MMAP_DATA, &base, &h);
  if (err != ESP_OK) { Serial.printf("mmap failed: %d\n", err); return; }

  if (llm_load((const uint8_t *)base, &model)) { Serial.println("bad model magic"); return; }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads,
                c->ffn, c->ple_dim, part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#endif

  // The model header states how many logits it produces; vocab.h carries the
  // decode table and the encoder asset the merge table. All three come from
  // the same tokenizer, so any disagreement means a stale header.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }
  if ((int)tokenizer.active_vocab != VOCAB_N) {
    Serial.printf("FATAL: encoder/decoder mismatch: encoder %u ids, vocab.h %d\n",
                  (unsigned)tokenizer.active_vocab, VOCAB_N);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = llm_stage_core_int8_alloc(&model, ps);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it too: it is 85%% of the dense MACs. With the vector
  // unit it stays int4 and is unpacked in registers: the head is bound by
  // PSRAM bandwidth, so half the bytes is the win.
#if LLM_HAVE_PIE
  if (model.out_head.cols % 32 == 0 && model.out_head.group % 32 == 0) {
    // Codes in PSRAM; the per-row scales (100 KB) in SRAM, read once per row.
    size_t code_bytes = (size_t)model.out_head.rows * model.out_head.row_bytes;
    llm_stage_int4_split(&model.out_head, ps_or_die(code_bytes, "staged head int4"),
                         sram_or_die(llm_stage_int4_scale_bytes(&model.out_head), "head scales"));
    ++staged;
  } else
#endif
  {
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
    llm_stage_int8(&model.out_head, b);
    ++staged;
  }
  Serial.printf("weights-> PSRAM  %d tensors int8, %.2f MB allocated\n",
                staged, psram_used / 1048576.0);

#if LLM_HAVE_PIE
  if (!pie_self_check()) { Serial.println("FATAL: PIE kernel disagrees with scalar"); return; }
#endif

  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = matvec_par;
    model.head_matvec  = matvec_par;
#ifdef LLM_KV_QUANT
    model.attn_heads = attn_par;
#endif
  } else {
    Serial.println("dual-core worker failed; running single core");
  }

  // FNV-1a over the mapped image. scripts/deploy.sh prints the same value for
  // the file it flashed; the two must agree.
  {
    const uint8_t *img = (const uint8_t *)base;
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < model.image_bytes; i++) { fp ^= img[i]; fp *= 16777619u; }
    Serial.printf("build: bytes=%u fp=%08x sram=%uB psram=%.2fMB\n",
                  (unsigned)model.image_bytes, fp,
                  (unsigned)(sram_used + STATIC_SRAM_BYTES),
                  psram_used / 1048576.0);
  }
#ifdef CPU_MHZ
  // Requested explicitly, over the board file's F_CPU. The HAL reports whether
  // it accepted the value; the line below shows what the chip actually runs.
  Serial.printf("cpu: request %d MHz -> %s\n", CPU_MHZ, setCpuFrequencyMhz(CPU_MHZ) ? "accepted" : "rejected");
#endif
  Serial.printf("cpu: %u MHz, %d cores\n", (unsigned)getCpuFrequencyMhz(), (int)portNUM_PROCESSORS);
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);
#ifdef LLM_KV_QUANT
  const char *kv_mode = "int8 keys / int16 values, vector dots";
#else
  const char *kv_mode = "fp32";
#endif
  Serial.printf("sampling: temperature %.2f, top-k %d | matvec: %s | head: %s | kv cache: %s | attention: %s\n\n",
                TEMPERATURE, TOP_K, LLM_HAVE_PIE ? "PIE vector" : "scalar",
#ifdef LLM_PRELOAD_START
                model.out_head.w4 ? "int4, vector unpack, cache preload" : "int8", kv_mode,
#else
                model.out_head.w4 ? "int4, vector unpack" : "int8", kv_mode,
#endif
                model.attn_heads ? "both cores" : "one core");
  Serial.println("type a story prompt and press return.");
  // A UART bridge can deliver a glitch byte around reset. Anything received
  // before this point is not a prompt, so drop it rather than let it poison
  // the first line typed.
  while (Serial.available()) Serial.read();
  ready = true;
  Serial.println("READY>");
}

void loop() {
  static char line[BTK_MAX_INPUT_BYTES];
  static int len = 0;
  static bool overflowed = false;
  if (!ready) { delay(1000); return; }
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      line[len] = '\0';
      if (overflowed) Serial.println("(prompt too long)");
      else if (len) tell_story(line);
      len = 0; overflowed = false;
      Serial.println("READY>");
    } else if (len == 0 && ((uint8_t)ch < 0x20 || (uint8_t)ch >= 0x80)) {
      // Leading control or non-ASCII byte: line noise, not the prompt.
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = ch;
    } else {
      // Past the buffer. Keep consuming to the newline so the tail of an
      // oversized line cannot be read as the beginning of the next one.
      overflowed = true;
    }
  }
  delay(5);
}
