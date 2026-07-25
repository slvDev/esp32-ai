// PLE TinyLM inference on the ESP32-S3.
// The 28.9M-param model (14.9MB, 4-bit) lives in a flash 'model' partition,
// memory-mapped so the 25M table is read a row at a time from flash; the hot
// tied head plus scratch and KV cache sit in PSRAM. Same llm.h that was verified
// against PyTorch on the host -- only the platform hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../common/llm.h"
#include "vocab.h"

// Set to 1 once a GMT020-02-7P (2.0" 240x320 ST7789) is wired up — see display.h.
// Leave 0 to run serial-only (no panel needed).
#define USE_DISPLAY 0
#if USE_DISPLAY
#include "display.h"
#endif

static const int PROMPT_IDS[] = {433, 447, 259, 405}; // "Once upon a time"
static const int N_GENERATE = 200;
static const int DEFAULT_MORE = 32;
static const int MAX_PROMPT_IDS = 64;
static const int MAX_PROMPT_BANK = 8;
static const int SAMPLE_TOP_K = 32;
static const float SAMPLE_TEMP = 0.95f;
static const float REPEAT_PENALTY = 0.28f;
static const int RECENT_WINDOW = 96;

static const int PROMPT_IDS_0[] = {433, 447, 259, 405};
static const int PROMPT_IDS_1[] = {2322, 259, 1759, 1814};
static const int PROMPT_IDS_2[] = {10938, 557, 317, 263, 1078};
static const int PROMPT_IDS_3[] = {345, 259, 1122, 1529};
static const int PROMPT_IDS_4[] = {2614, 282, 259, 918, 1148};
static const int PROMPT_IDS_5[] = {1570, 263, 3051, 348, 263, 1944};
static const int PROMPT_IDS_6[] = {44, 537, 5725, 12, 317, 259, 2397};
static const int PROMPT_IDS_7[] = {427, 7361, 3699};

// Emit one token to every active output (serial always; TFT when enabled).
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

Model model;
Scratch s;
static bool model_ready = false;
static int cur_pos = 0;
static int cur_tok = 0;
static int decoded_total = 0;
static int prompt_bank[MAX_PROMPT_BANK][MAX_PROMPT_IDS];
static int prompt_bank_lens[MAX_PROMPT_BANK];
static int prompt_bank_count = 0;
static int prompt_bank_idx = 0;
static uint32_t prng_state = 1;
static int64_t window_start_us = 0;
static int64_t window_decode_us = 0;
static int window_tokens = 0;
static int recent_ids[RECENT_WINDOW];
static int recent_len = 0;
static int recent_head = 0;
static uint8_t recent_counts[VOCAB_N];

// ---- int8 output head (SIMD-friendly) --------------------------------------
// The head is scanned in full every token and dominates runtime. We stage it as
// int8 in PSRAM at boot (int4 nibbles unpacked ONCE), so per token there is no
// nibble unpacking and no float conversion of weights -- just int8 x int8 ->
// int32 dot per row. Its input dim (D=96) is a single group, so one scale per
// row. int8-activation quality was validated on host (val perplexity delta ~0,
// see firmware/host_verify/ppl.c). Output rows split across both LX7 cores.
static int8_t *head_w8 = NULL;      // [rows * cols] unpacked int8 weights (-7..7)
static float  *head_scale8 = NULL;  // [rows] per-row dequant scale
static int head_rows, head_cols;

static int8_t head_actq[128];       // quantized activation, shared by both cores
static float  head_acts;            // its scale

// int8 dot -> int32. Tight and branch-free so the S3 int SIMD / -O3 unrolls it.
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

static void head_rows_range(float *y, int r0, int r1) {
  for (int r = r0; r < r1; r++)
    y[r] = (float)dot_i8(head_actq, head_w8 + (size_t)r * head_cols, head_cols)
           * head_scale8[r] * head_acts;
}

// dual-core plumbing (worker does the first half of the rows on core 0)
static TaskHandle_t head_worker;
static TaskHandle_t inference_task;
static float *volatile head_job_y;
static volatile int head_job_split;

static void head_worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    head_rows_range(head_job_y, 0, head_job_split);
    xTaskNotifyGive(inference_task);
  }
}

// Matches Model.head_matvec (QT*, float*, float*); QT unused (weights staged).
static void head_matvec_int8(const QT *t, const float *x, float *y) {
  (void)t;
  quantize_act(x, head_cols, head_actq, &head_acts);  // once; both cores read it
  head_job_y = y;
  head_job_split = head_rows / 2;
  xTaskNotifyGive(head_worker);
  head_rows_range(y, head_job_split, head_rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static void *ps(size_t n) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  if (!p) { Serial.printf("PSRAM alloc failed (%u bytes)\n", (unsigned)n); while (1) delay(1000); }
  return p;
}

// Unpack the (row-capped) head from int4 to int8 in PSRAM, once at boot.
static void stage_head_int8(QT *t) {
  head_rows = t->rows; head_cols = t->cols;
  head_w8 = (int8_t *)ps((size_t)head_rows * head_cols);
  head_scale8 = (float *)ps((size_t)head_rows * sizeof(float));
  for (int r = 0; r < head_rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = head_w8 + (size_t)r * head_cols;
    for (int j = 0; j < head_cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    head_scale8[r] = half2float(t->scales[(size_t)r * t->n_groups]);  // n_groups==1
  }
  Serial.printf("head staged int8: %.2f MB\n",
                ((size_t)head_rows * head_cols + (size_t)head_rows * 4) / 1e6);
}

static void blink(uint8_t g) {
#ifdef RGB_BUILTIN
  rgbLedWrite(RGB_BUILTIN, 0, g, g / 3);
#endif
}

static uint32_t prng_next() {
  prng_state = prng_state * 1664525u + 1013904223u;
  return prng_state;
}

static float prng_unit() {
  // 24-bit mantissa-style fraction in [0,1).
  return (float)((prng_next() >> 8) & 0x00FFFFFFu) / 16777216.0f;
}

static void recent_clear() {
  memset(recent_counts, 0, sizeof(recent_counts));
  recent_len = 0;
  recent_head = 0;
}

static void recent_note(int tok) {
  if (tok < 0 || tok >= VOCAB_N) return;

  if (recent_len == RECENT_WINDOW) {
    int old = recent_ids[recent_head];
    if (old >= 0 && old < VOCAB_N && recent_counts[old] > 0) recent_counts[old]--;
    recent_ids[recent_head] = tok;
    recent_head = (recent_head + 1) % RECENT_WINDOW;
  } else {
    int idx = (recent_head + recent_len) % RECENT_WINDOW;
    recent_ids[idx] = tok;
    recent_len++;
  }

  if (recent_counts[tok] < 255) recent_counts[tok]++;
}

static int sample_topk_token(const float *logits, int vocab, int k) {
  if (k < 1) k = 1;
  if (k > 16) k = 16;

  int top_ids[16] = {0};
  float top_vals[16];
  for (int i = 0; i < 16; i++) top_vals[i] = -1e30f;

  int n = 0;
  for (int v = 0; v < vocab; v++) {
    float val = logits[v];
    if (val <= top_vals[k - 1]) continue;

    int pos = k - 1;
    while (pos > 0 && val > top_vals[pos - 1]) {
      top_vals[pos] = top_vals[pos - 1];
      top_ids[pos] = top_ids[pos - 1];
      pos--;
    }
    top_vals[pos] = val;
    top_ids[pos] = v;
    if (n < k) n++;
  }

  if (n <= 0) return 0;
  return top_ids[prng_next() % (uint32_t)n];
}

static int sample_next_token(const float *logits, int vocab, int k, float temp) {
  if (k < 1) k = 1;
  if (k > 64) k = 64;
  if (temp < 0.1f) temp = 0.1f;

  int top_ids[64] = {0};
  float top_vals[64];
  for (int i = 0; i < 64; i++) top_vals[i] = -1e30f;

  int n = 0;
  for (int v = 0; v < vocab; v++) {
    float val = logits[v] - REPEAT_PENALTY * (float)recent_counts[v];
    if (val <= top_vals[k - 1]) continue;

    int pos = k - 1;
    while (pos > 0 && val > top_vals[pos - 1]) {
      top_vals[pos] = top_vals[pos - 1];
      top_ids[pos] = top_ids[pos - 1];
      pos--;
    }
    top_vals[pos] = val;
    top_ids[pos] = v;
    if (n < k) n++;
  }

  if (n <= 0) return 0;
  if (n == 1) return top_ids[0];

  float maxv = top_vals[0];
  float probs[64];
  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    float z = (top_vals[i] - maxv) / temp;
    float p = expf(z);
    probs[i] = p;
    s += p;
  }

  float r = prng_unit() * s;
  float c = 0.0f;
  for (int i = 0; i < n; i++) {
    c += probs[i];
    if (r <= c) return top_ids[i];
  }
  return top_ids[n - 1];
}

static int next_token_greedy() {
  int best = 0;
  float bv = -1e30f;
  for (int v = 0; v < VOCAB_N; v++) {
    if (s.logits[v] > bv) {
      bv = s.logits[v];
      best = v;
    }
  }
  return best;
}

// Greedy decode for n tokens, continuing from current context.
static void generate_more(int n) {
  if (!model_ready) {
    Serial.println("model not ready");
    return;
  }
  if (cur_pos >= model.c.seq_len) {
    Serial.printf("context full (%d/%d). Use 'reset' or 'prompt_ids ...'\n", cur_pos, model.c.seq_len);
    return;
  }
  if (n <= 0) n = DEFAULT_MORE;

  int can = model.c.seq_len - cur_pos;
  if (n > can) n = can;

  int64_t t0 = esp_timer_get_time();
  int64_t decode_us = 0;
  int decoded = 0;

  for (int step = 0; step < n; step++) {
    int best = 0;
    float bv = -1e30f;
    for (int v = 0; v < VOCAB_N; v++) {
      if (s.logits[v] > bv) {
        bv = s.logits[v];
        best = v;
      }
    }

    cur_tok = best;
    emit(cur_tok);
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, cur_tok, cur_pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    decoded_total++;
    if ((step & 7) == 0) delay(0);
  }

  int64_t total_us = esp_timer_get_time() - t0;
  Serial.printf("\n\n--- +%d tokens (total %d, pos %d/%d) in %.2f s ---\n",
                decoded, decoded_total, cur_pos, model.c.seq_len, total_us / 1e6);
  if (decoded > 0) {
    Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                  decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  }
}

static void clear_runtime_state() {
  recent_clear();
  memset(s.kcache, 0, (size_t)model.c.n_layers * model.c.seq_len * model.c.dim * sizeof(float));
  memset(s.vcache, 0, (size_t)model.c.n_layers * model.c.seq_len * model.c.dim * sizeof(float));
  memset(s.ple, 0, (size_t)model.c.n_layers * model.c.ple_dim * sizeof(float));
  memset(s.tmpP, 0, (size_t)model.c.n_layers * model.c.ple_dim * sizeof(float));
  memset(s.trow, 0, (size_t)model.c.n_layers * model.c.ple_dim * sizeof(float));
  cur_pos = 0;
  cur_tok = 0;
  decoded_total = 0;
}

static void clear_cache_state() {
  memset(s.kcache, 0, (size_t)model.c.n_layers * model.c.seq_len * model.c.dim * sizeof(float));
  memset(s.vcache, 0, (size_t)model.c.n_layers * model.c.seq_len * model.c.dim * sizeof(float));
  memset(s.ple, 0, (size_t)model.c.n_layers * model.c.ple_dim * sizeof(float));
  memset(s.tmpP, 0, (size_t)model.c.n_layers * model.c.ple_dim * sizeof(float));
  memset(s.trow, 0, (size_t)model.c.n_layers * model.c.ple_dim * sizeof(float));
  cur_pos = 0;
}

static void set_prompt_slot(int slot, const int *ids, int n_ids) {
  if (slot < 0 || slot >= MAX_PROMPT_BANK) return;
  if (n_ids < 1) return;
  if (n_ids > MAX_PROMPT_IDS) n_ids = MAX_PROMPT_IDS;
  for (int i = 0; i < n_ids; i++) {
    prompt_bank[slot][i] = ids[i];
  }
  prompt_bank_lens[slot] = n_ids;
}

static void init_prompt_bank() {
  set_prompt_slot(0, PROMPT_IDS_0, sizeof(PROMPT_IDS_0) / sizeof(int));
  set_prompt_slot(1, PROMPT_IDS_1, sizeof(PROMPT_IDS_1) / sizeof(int));
  set_prompt_slot(2, PROMPT_IDS_2, sizeof(PROMPT_IDS_2) / sizeof(int));
  set_prompt_slot(3, PROMPT_IDS_3, sizeof(PROMPT_IDS_3) / sizeof(int));
  set_prompt_slot(4, PROMPT_IDS_4, sizeof(PROMPT_IDS_4) / sizeof(int));
  set_prompt_slot(5, PROMPT_IDS_5, sizeof(PROMPT_IDS_5) / sizeof(int));
  set_prompt_slot(6, PROMPT_IDS_6, sizeof(PROMPT_IDS_6) / sizeof(int));
  set_prompt_slot(7, PROMPT_IDS_7, sizeof(PROMPT_IDS_7) / sizeof(int));
  prompt_bank_count = 8;
  prompt_bank_idx = 0;
}

static void start_prompt_ids(const int *ids, int n_ids) {
  if (!model_ready) {
    Serial.println("model not ready");
    return;
  }
  if (n_ids <= 0) {
    Serial.println("no prompt ids provided");
    return;
  }

  clear_runtime_state();
  Serial.print("\n>>> ");
  for (int i = 0; i < n_ids && cur_pos < model.c.seq_len; i++) {
    int id = ids[i];
    if (id < 0 || id >= VOCAB_N) {
      Serial.printf("\ninvalid token id %d (skipped)\n", id);
      continue;
    }
    cur_tok = id;
    emit(cur_tok);
    llm_forward(&model, cur_tok, cur_pos++, &s);
    recent_note(cur_tok);
  }
}

static void start_active_prompt() {
  if (prompt_bank_count <= 0) {
    init_prompt_bank();
  }
  start_prompt_ids(prompt_bank[prompt_bank_idx], prompt_bank_lens[prompt_bank_idx]);
}

static void print_prompt_list() {
  Serial.printf("prompt bank: %d/%d entries, active=%d\n",
                prompt_bank_count, MAX_PROMPT_BANK, prompt_bank_idx);
  for (int i = 0; i < prompt_bank_count; i++) {
    Serial.printf("  [%d]%s len=%d\n", i, (i == prompt_bank_idx ? "*" : " "), prompt_bank_lens[i]);
  }
}

static void start_default_prompt() {
  prompt_bank_idx = 0;
  start_active_prompt();
}

static void prime_story_seed() {
  if (prompt_bank_count > 0) {
    prompt_bank_idx = (int)(prng_next() % (uint32_t)prompt_bank_count);
  }
  start_active_prompt();
  int extra = 1 + (int)(prng_next() % 4u);
  for (int i = 0; i < extra && cur_pos < model.c.seq_len; i++) {
    int tok = sample_next_token(s.logits, VOCAB_N, 24, 1.05f);
    cur_tok = tok;
    emit(cur_tok);
    llm_forward(&model, cur_tok, cur_pos++, &s);
    recent_note(cur_tok);
  }
}

static void print_serial_help() {
  Serial.println("\nserial commands:");
  Serial.println("  <enter>          generate 32 more tokens (auto-resets if context is full)");
  Serial.println("  more [N]         generate N more tokens");
  Serial.println("  reset            reset to default prompt");
  Serial.println("  prompt_ids a,b,c reset and use token-id prompt");
  Serial.println("  prompt_add a,b,c save a prompt slot");
  Serial.println("  next_prompt      rotate to next saved prompt and generate");
  Serial.println("  prompt_list      show saved prompt slots");
  Serial.println("  status           show token/context position");
  Serial.println("  help             show this help");
}

static void print_cli_prompt() {
  Serial.print("cmd> ");
}

static char ascii_lower(char c) {
  if (c >= 'A' && c <= 'Z') return (char)(c - 'A' + 'a');
  return c;
}

static bool equals_ci(const char *a, const char *b) {
  while (*a && *b) {
    if (ascii_lower(*a) != ascii_lower(*b)) return false;
    a++; b++;
  }
  return *a == '\0' && *b == '\0';
}

static bool starts_with_ci(const char *s, const char *prefix) {
  while (*prefix) {
    if (*s == '\0') return false;
    if (ascii_lower(*s) != ascii_lower(*prefix)) return false;
    s++; prefix++;
  }
  return true;
}

static void handle_serial_command(const char *line_in) {
  char line[192];
  strncpy(line, line_in, sizeof(line) - 1);
  line[sizeof(line) - 1] = '\0';

  // Trim leading spaces.
  char *p = line;
  while (*p == ' ' || *p == '\t' || *p == '\r') p++;

  // Trim trailing spaces.
  size_t plen = strlen(p);
  while (plen > 0 && (p[plen - 1] == ' ' || p[plen - 1] == '\t' || p[plen - 1] == '\r')) {
    p[--plen] = '\0';
  }

  if (*p == '\0') {
    if (cur_pos >= model.c.seq_len) {
      Serial.println("context full; restarting from default prompt");
      start_default_prompt();
    }
    generate_more(DEFAULT_MORE);
    return;
  }

  if (equals_ci(p, "help")) {
    print_serial_help();
    return;
  }

  if (equals_ci(p, "reset")) {
    start_active_prompt();
    generate_more(DEFAULT_MORE);
    return;
  }

  if (equals_ci(p, "status")) {
    Serial.printf("status: pos %d/%d, generated %d tokens\n", cur_pos, model.c.seq_len, decoded_total);
    return;
  }

  if (starts_with_ci(p, "more")) {
    int n = DEFAULT_MORE;
    if (p[4] != '\0') n = atoi(p + 4);
    if (cur_pos >= model.c.seq_len) {
      Serial.println("context full; restarting from default prompt");
      start_default_prompt();
    }
    generate_more(n);
    return;
  }

  if (starts_with_ci(p, "prompt_ids")) {
    char *args = p + 10;
    while (*args == ' ' || *args == '\t') args++;
    if (*args == '\0') {
      Serial.println("usage: prompt_ids 433,447,259,405");
      return;
    }
    int ids[MAX_PROMPT_IDS];
    int n_ids = 0;

    char *tok = strtok(args, ", ");
    while (tok && n_ids < MAX_PROMPT_IDS) {
      ids[n_ids++] = atoi(tok);
      tok = strtok(NULL, ", ");
    }
    set_prompt_slot(prompt_bank_idx, ids, n_ids);
    start_prompt_ids(ids, n_ids);
    generate_more(DEFAULT_MORE);
    return;
  }

  if (starts_with_ci(p, "prompt_add")) {
    char *args = p + 10;
    while (*args == ' ' || *args == '\t') args++;
    if (*args == '\0') {
      Serial.println("usage: prompt_add 433,447,259,405");
      return;
    }
    if (prompt_bank_count >= MAX_PROMPT_BANK) {
      Serial.printf("prompt bank full (%d). overwrite with prompt_ids or reboot.\n", MAX_PROMPT_BANK);
      return;
    }
    int ids[MAX_PROMPT_IDS];
    int n_ids = 0;
    char *tok = strtok(args, ", ");
    while (tok && n_ids < MAX_PROMPT_IDS) {
      ids[n_ids++] = atoi(tok);
      tok = strtok(NULL, ", ");
    }
    set_prompt_slot(prompt_bank_count, ids, n_ids);
    prompt_bank_idx = prompt_bank_count;
    prompt_bank_count++;
    Serial.printf("saved prompt slot %d\n", prompt_bank_idx);
    start_active_prompt();
    generate_more(DEFAULT_MORE);
    return;
  }

  if (equals_ci(p, "next_prompt")) {
    if (prompt_bank_count <= 0) {
      init_prompt_bank();
    }
    prompt_bank_idx = (prompt_bank_idx + 1) % prompt_bank_count;
    Serial.printf("switched to prompt slot %d\n", prompt_bank_idx);
    start_active_prompt();
    generate_more(DEFAULT_MORE);
    return;
  }

  if (equals_ci(p, "prompt_list")) {
    print_prompt_list();
    return;
  }

  if (strchr(p, ' ') || strchr(p, '"') || strchr(p, '\'')) {
    Serial.println("plain text prompts are not supported on-device; use 'prompt_ids ...' or 'reset'.");
  } else {
    Serial.println("unknown command. type 'help'");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM ===");

  // Map the model partition.
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
  Serial.printf("model: V=%d D=%d L=%d H=%d F=%d P=%d  (mapped %.1f MB)\n",
                c->vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim,
                part->size / 1e6);

#if USE_DISPLAY
  display_begin();
#endif

  // Cap head rows to the trained vocab BEFORE staging: the tokenizer learned
  // 25,353 entries; the padded rows above that can never be emitted (and have no
  // decode entry), so we neither stage nor score them.
  model.tok_emb.rows = VOCAB_N;
  stage_head_int8(&model.tok_emb);  // int8-staged head; input embedding still uses mmap
  inference_task = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(head_worker_main, "head", 4096, NULL, 2,
                             &head_worker, 0) != pdPASS) {
    Serial.println("head worker creation failed");
    return;
  }
  model.head_matvec = head_matvec_int8;

  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, V = c->vocab, S = c->seq_len;
  s.x = (float *)ps(D * 4);
  s.h = (float *)ps((F > D ? F : D) * 4);
  s.qkv = (float *)ps(3 * D * 4);
  s.att = (float *)ps(D * 4);
  s.g1 = (float *)ps(F * 4);
  s.g2 = (float *)ps((P > F ? P : F) * 4);
  s.ple = (float *)ps(L * P * 4);
  s.tmpP = (float *)ps(L * P * 4);
  s.trow = (float *)ps(L * P * 4);
  s.logits = (float *)ps(V * 4);
  s.scores = (float *)ps(S * 4);
  s.kcache = (float *)ps((size_t)L * S * D * 4);
  s.vcache = (float *)ps((size_t)L * S * D * 4);
  Serial.printf("PSRAM free after alloc: %u KB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

  model_ready = true;
  init_prompt_bank();
  prng_state = esp_random();
  llm_profile_reset(&s);
  prime_story_seed();

  window_start_us = esp_timer_get_time();
  window_decode_us = 0;
  window_tokens = 0;

  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#if USE_DISPLAY
  // Closing card: compute-only tok/s (the model's own speed) + ms/token.
  if (decoded_total > 0) {
    display_stats(0, 0);
  }
#endif
  blink(0);
}

void loop() {
  if (!model_ready) {
    delay(50);
    return;
  }

  if (cur_pos >= model.c.seq_len) {
    clear_cache_state();
    llm_forward(&model, cur_tok, cur_pos++, &s);
    recent_note(cur_tok);
  }

  cur_tok = sample_next_token(s.logits, VOCAB_N, SAMPLE_TOP_K, SAMPLE_TEMP);
  // Prevent pathological immediate repeats such as "slide slide slide".
  if (cur_pos > 0 && recent_len > 0) {
    int last_idx = (recent_head + recent_len - 1) % RECENT_WINDOW;
    int last_tok = recent_ids[last_idx];
    if (cur_tok == last_tok) {
      cur_tok = sample_next_token(s.logits, VOCAB_N, SAMPLE_TOP_K, SAMPLE_TEMP + 0.1f);
    }
  }
  emit(cur_tok);
  blink((decoded_total & 1) ? 40 : 8);

  int64_t d0 = esp_timer_get_time();
  llm_forward(&model, cur_tok, cur_pos++, &s);
  recent_note(cur_tok);
  window_decode_us += esp_timer_get_time() - d0;
  decoded_total++;
  window_tokens++;

  if (window_tokens >= 32) {
    window_start_us = esp_timer_get_time();
    window_decode_us = 0;
    window_tokens = 0;
  }

  if ((decoded_total & 7) == 0) delay(0);
}
