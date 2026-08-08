// PLE TinyLM inference on the ESP32-S3.
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
// The logits array stays in PSRAM: 25,353 floats is 99 KiB, and the argmax
// reads it once per token.
//
// Same llm.h that is verified against PyTorch on the host; only the platform
// hooks differ here.

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

// int8 activations, required by the staged int8 kernel. Not bit-exact against
// the fp32 golden; verify.c must be built without this flag. Validation CE cost
// (runtime/host_verify/ppl.c, 32,768 predictions): 2.4793 -> 2.4796, ppl 11.93 / 11.94.
#define LLM_INT8_ACT 1
// LOCAL ADDITION: route the int8 dot product through the ESP32-S3 PIE vector
// unit where the operands allow it. Arithmetic is identical -- same int8 values,
// same accumulation order within a group -- so this is a pure speed change with
// no effect on the emitted tokens. Verified by generating from the same prompt
// and diffing the output text against the scalar build.
#define LLM_SIMD_S8 1
// LOCAL ADDITION: int8 KV cache + position-major traversal. See llm.h.
// Disabled for the FREE build: int8 KV and the int8-q/fast-expf path that
// depends on it are the only two changes that alter output. Everything else
// (SIMD, int4 head, SRAM tiering, int4 core) is bit-identical.
#define LLM_KV_INT8 1
#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() esp_timer_get_time()
#include "../../runtime/llm.h"
#include "generated/vocab.h"

// Set to 1 once a display is wired up - see display.h.
// Leave 0 to run serial-only (no panel needed).
//
// LOCAL CHANGE (ESP32-Gemma3-270M reproduction): forced to 0. With no panel
// wired, display_puts() emits a failed i2c_master_transmit per glyph, each of
// which logs an ESP_ERR_INVALID_STATE line over the same USB-CDC link the
// tokens go out on. That cost lands inside total_us and nowhere else, which is
// why the board reported 5.47 tok/s wall against 94.3 ms/token of compute.
// The compute figure was never wrong; the wall figure was measuring the error
// flood. Serial-only is also the configuration the upstream 9.88 tok/s number
// was taken in, so this is the comparable one.
#define USE_DISPLAY 0
#if USE_DISPLAY
#include "display.h"
#endif

// Fold the argmax into the output-head kernel so the logits are never written.
//
// Greedy decoding consumes exactly one number from the head -- which row won --
// yet the runtime materializes all out_vocab logits in PSRAM and then reads
// them back to find it. On this model that round trip is 198 KiB/token on a bus
// measured at 60.7 MB/s. The head is already 64.5% of the token and pinned at
// 86% of that bus ceiling, so bytes removed here convert to time at ~1:1.
//
// Exact, not approximate: the same float is computed in the same order and
// compared instead of stored, with the flat scan's lowest-index-wins tie-break
// preserved across the dual-core split. Set to 0 to A/B against the storing
// path without touching anything else.
#define FUSE_ARGMAX 1
static bool head_fused = false;

/* CertiHead: skip output-head tiles that provably cannot contain the argmax.
 * Exact -- the emitted token is identical to the dense scan by construction, so
 * this is a pure traffic change and the 200-token digest must not move. Set to
 * 0 to A/B against the dense fused-argmax path with nothing else altered. */
#define CERT_HEAD 1

/* Two ablation switches for design choices this project has never tested
 * against their own absence.
 *
 * CERT_PERM -- rows are permuted by ||w|| at staging. EXP-113 ranked that above
 * angular clustering and concluded "the row order matters more than the bound",
 * but every arm of that comparison HAD an ordering; none was compared against
 * leaving the rows in token-id order. A host study (EXP-123) then found that
 * removing norm structure entirely from this head IMPROVES pruning on the host
 * (37.79% -> 28.17%), which the ordering comparison could not have revealed.
 *
 * CERT_SEED -- two tiles are scanned before any pruning decision, to raise the
 * incumbent. On the host, bound-ordered scanning alone already reaches the
 * perfect-incumbent bound, so the seed looks free to remove. But the host scans
 * sequentially and the device must fix its survivor set before splitting across
 * cores, and EXP-116 recorded a flat split LOSING to one core for exactly that
 * reason. So this one is genuinely uncertain and has to be measured here.
 *
 * CERT_PERM defaults to the shipped behaviour and stayed there: the board
 * ablation vindicated it decisively (33.2% -> 72.7% of rows and 61.5 -> 45.3
 * tok/s without it). CERT_SEED's default was flipped to 0 by the same
 * experiment; see below. */
#ifndef CERT_PERM
#define CERT_PERM 1   /* 1 = norm-sort rows at staging, 0 = token-id order */
#endif
#ifndef CERT_SEED
/* DEFAULT FLIPPED TO 0 by EXP-124, on the board, 4 arms x 2 captures.
 * The seed changed rows scanned by exactly zero -- 8427 in both arms, to the
 * row -- and throughput by less than run-to-run noise (61.96/61.53 without it
 * against 61.48/61.89 with it). Its two tiles were already among the
 * highest-bound tiles, so bound-ordered scanning reaches them regardless.
 *
 * Kept behind the flag rather than deleted. The seed was introduced to fix a
 * real defect (EXP-116: a flat dual-core split lost to a single core because it
 * had to fix the survivor set before either half could improve the incumbent).
 * The wave scheduler evidently subsumes that, but if the scheduler changes the
 * reason can come back. Set to 1 to restore. */
#define CERT_SEED 0   /* 1 = two-tile incumbent seed, 0 = pure bound order */
#endif

static const int PROMPT_IDS[] = {433, 447, 259, 405}; // "Once upon a time"
static const int N_GENERATE = 200;

Model model;
Scratch s;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)

// Staging allocator that prefers INTERNAL SRAM over PSRAM.
//
// Measured on this board: PSRAM streams at 60.7 MB/s, but a working set inside
// the 32 KB data cache runs at 232 MB/s scalar and 947 MB/s through the vector
// unit. Internal SRAM is not merely "some more RAM" -- it is a different
// bandwidth tier, and the core is small enough to reach it.
//
// The core stages to ~556 KB int8 against ~293 KB of free internal SRAM, so it
// does not all fit. Tensors are offered SRAM in the order the stager walks
// them and fall back to PSRAM once it is exhausted, which puts roughly half the
// core in the fast tier. A reserve is held back because the scratch buffers,
// FreeRTOS stacks and the WiFi/USB stacks also live in internal SRAM, and
// starving them turns a speedup into a boot failure.
#define SRAM_RESERVE_BYTES (72 * 1024)
static size_t sram_staged = 0, psram_staged = 0;

static void *stage_alloc(size_t n) {
  size_t freei = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (freei > n + SRAM_RESERVE_BYTES) {
    void *p = heap_caps_aligned_alloc(16, (n + 15) & ~(size_t)15,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (p) { sram_staged += n; return p; }
  }
  void *p = heap_caps_aligned_alloc(16, (n + 15) & ~(size_t)15, MALLOC_CAP_SPIRAM);
  if (p) { psram_staged += n; psram_used += n; }
  return p;
}

static void *ps(size_t n) {
  // 16-byte aligned: staged weight rows are contiguous at stride `cols`, so a
  // 16-aligned base plus a cols that is a multiple of 16 makes every row
  // 16-aligned, which is what the PIE vector load requires. Without this the
  // SIMD path silently falls back to scalar on every row and the whole change
  // measures as a no-op.
  void *p = heap_caps_aligned_alloc(16, (n + 15) & ~(size_t)15, MALLOC_CAP_SPIRAM);
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

// ---- int4 head resident in PSRAM -------------------------------------------
// The head is stored int4 in flash and staged to int8 in PSRAM, which DOUBLES
// the bytes read per token on the one tensor measured to be bandwidth-bound:
// 25353 x 96 = 2.43 MB at 55.8 MB/s = 43.6 ms, 59% of the token. Keeping it
// int4 in PSRAM halves that to 1.16 MB.
//
// Bit-identical, not approximate: int4 codes are 0..15 mapping to values -8..7,
// which the int8 staging represents exactly. The same int32 dot product comes
// out either way.
//
// Two things make an int4 vector kernel practical:
//
// 1. REPACKED LAYOUT. Nibbles are stored interleaved (byte j>>1 holds column j
//    in the low nibble and j+1 in the high), so a 16-byte vector load yields
//    even columns in one half and odd in the other -- the wrong order, needing
//    a shuffle the PIE unit does not have. Since we build this copy ourselves
//    at boot, we repack: byte k of a 32-column block holds column k in the low
//    nibble and column k+16 in the high. One load then unpacks into lanes
//    0..15 and 16..31 in natural order, matching two consecutive loads of xq.
//
// 2. NO PER-LANE ZERO POINT. sum((code-8)*x) == sum(code*x) - 8*sum(x), and
//    sum(x) depends only on the activation, so it is computed once per matvec
//    rather than per row. That removes a vector subtract from every one of the
//    25,353 rows. Requires n_groups == 1 (true here: cols 96 < group 128);
//    with more groups the correction would need per-group sums.
//
// Extracting the high nibble uses a 32-bit shift then a byte mask: shifting the
// word right by 4 pulls the neighbouring byte's low nibble into the high half
// of each byte, and the mask discards it, leaving each byte's original high
// nibble in place.
static uint8_t *head4 = NULL;        // rows x (cols/2), repacked
static float   *head4_scale = NULL;  // one fp32 group scale per row
static int      head4_rowb = 0;
static const QT *head4_src = NULL;

static const uint8_t NIB_MASK[16] __attribute__((aligned(16))) = {
  0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
  0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

/* Physical row -> token id. Identity unless CertiHead reorders the head.
 *
 * Row order in an output head is arbitrary -- it comes from the tokenizer's
 * merge order and says nothing about where a row points -- so it can be chosen.
 * The host oracle measured that sorting rows by ||w|| takes the fraction of
 * rows a certificate must score from 67% to 30%, beating directional
 * clustering, because the per-dimension bound is dominated by magnitude spread
 * rather than angular spread.
 *
 * Permuting HERE rather than in the model file is deliberate: the flashed
 * weights are byte-for-byte the ones every previous cycle used, so any speed
 * change cannot be a different model. It costs one extra pass at boot. */
static int32_t *head4_tok = NULL;

static float *cert_norm_tmp = NULL;   /* boot-only, freed after sorting */

static int cert_cmp_norm(const void *a, const void *b) {
  int x = *(const int32_t *)a, y = *(const int32_t *)b;
  float nx = cert_norm_tmp[x], ny = cert_norm_tmp[y];
  return (nx > ny) - (nx < ny);
}

static bool stage_head_int4(const QT *t) {
  if (t->n_groups != 1 || (t->cols % 32) != 0) return false;
  head4_rowb = t->cols / 2;
  size_t bytes = (size_t)t->rows * head4_rowb;
  head4 = (uint8_t *)heap_caps_aligned_alloc(16, (bytes + 15) & ~(size_t)15,
                                             MALLOC_CAP_SPIRAM);
  head4_scale = (float *)heap_caps_aligned_alloc(16, (size_t)t->rows * 4,
                                                 MALLOC_CAP_SPIRAM);
  head4_tok = (int32_t *)heap_caps_aligned_alloc(16, (size_t)t->rows * 4,
                                                 MALLOC_CAP_SPIRAM);
  if (!head4 || !head4_scale || !head4_tok) return false;

  for (int r = 0; r < t->rows; r++) head4_tok[r] = r;
#if CERT_HEAD && CERT_PERM
  /* ||w_r|| = scale_r * sqrt(sum (code-8)^2). One pass over the source. */
  cert_norm_tmp = (float *)heap_caps_malloc((size_t)t->rows * 4,
                                            MALLOC_CAP_SPIRAM);
  if (!cert_norm_tmp) return false;
  for (int r = 0; r < t->rows; r++) {
    const uint8_t *src = t->codes + (size_t)r * t->row_bytes;
    int32_t nn = 0;
    for (int j = 0; j < t->cols; j++) {
      int c = (j & 1) ? (src[j >> 1] >> 4) : (src[j >> 1] & 0xF);
      int u = c - 8;
      nn += u * u;
    }
    cert_norm_tmp[r] = half2float(t->scales[r]) * sqrtf((float)nn);
  }
  qsort(head4_tok, (size_t)t->rows, sizeof(int32_t), cert_cmp_norm);
  heap_caps_free(cert_norm_tmp);
  cert_norm_tmp = NULL;
#endif

  for (int r = 0; r < t->rows; r++) {
    int srow = head4_tok[r];              /* which token id lands at row r */
    const uint8_t *src = t->codes + (size_t)srow * t->row_bytes;
    uint8_t *dst = head4 + (size_t)r * head4_rowb;
    for (int blk = 0; blk * 32 < t->cols; blk++) {
      for (int k = 0; k < 16; k++) {
        int jl = blk * 32 + k, jh = jl + 16;
        int cl = (jl & 1) ? (src[jl >> 1] >> 4) : (src[jl >> 1] & 0xF);
        int ch = (jh & 1) ? (src[jh >> 1] >> 4) : (src[jh >> 1] & 0xF);
        dst[blk * 16 + k] = (uint8_t)(cl | (ch << 4));
      }
    }
    head4_scale[r] = half2float(t->scales[srow]);
  }
  head4_src = t;
  psram_used += bytes + (size_t)t->rows * 8;
  return true;
}

// y[row_begin:row_end] from the repacked int4 head. `corr` is 8*sum(xq).
static void head_i4_range(const int8_t *xq, float x_scale, int32_t corr,
                          float *y, int row_begin, int row_end) {
  int nblk = head4_rowb / 16;
  asm volatile("ssai 4");                      // SAR = 4 for ee.vsr.32
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *w = head4 + (size_t)r * head4_rowb;
    const int8_t *xp = xq;
    asm volatile("ee.zero.accx");
    asm volatile("ee.vld.128.ip q7, %0, 0" :: "r"(NIB_MASK));   // mask, kept live
    for (int b = 0; b < nblk; b++) {
      asm volatile(
          "ee.vld.128.ip  q0, %0, 16   \n"   // 16 bytes = 32 packed weights
          "ee.andq        q1, q0, q7   \n"   // low nibbles  -> cols k..k+15
          "ee.vsr.32      q2, q0       \n"   // >>4 (32-bit)
          "ee.andq        q2, q2, q7   \n"   // high nibbles -> cols k+16..k+31
          "ee.vld.128.ip  q3, %1, 16   \n"   // xq[k .. k+15]
          "ee.vld.128.ip  q4, %1, 16   \n"   // xq[k+16 .. k+31]
          "ee.vmulas.s8.accx q1, q3    \n"
          "ee.vmulas.s8.accx q2, q4    \n"
          : "+r"(w), "+r"(xp) :: "memory");
    }
    uint32_t lo;
    asm volatile("rur.accx_0 %0" : "=r"(lo));
    // (sum(code*x) - 8*sum(x)) is exactly the int8 path's int32 accumulator.
    y[r] = (float)((int32_t)lo - corr) * head4_scale[r] * x_scale;
  }
}

// Same kernel, but each logit is compared and discarded instead of stored.
//
// Under greedy decoding the logits array is write-only: llm_forward fills
// out_vocab floats and the caller reads them back once to take an argmax. Here
// that is 25,353 x 4 B = 99 KiB written to PSRAM and 99 KiB read back --
// 198 KiB of bus traffic per token to produce one integer. At the measured
// 60.7 MB/s that is ~3.3 ms of a 38.68 ms token spent on a value nothing else
// consumes. Keeping the running max in a register deletes both directions.
//
// Tie-breaking must reproduce the flat scan exactly. The original walks
// v = 0..V-1 taking strictly-greater, so the LOWEST index wins a tie. Each half
// keeps that rule internally and the combine prefers the worker half (rows
// [0, split)), which is where the lower indices are. Both halves start at
// -1e30f like the original, so an all-below-threshold token still returns 0.
static float hm_val[2];
static int   hm_idx[2];

static void head_i4_argmax_range(const int8_t *xq, float x_scale, int32_t corr,
                                 int slot, int row_begin, int row_end) {
  int nblk = head4_rowb / 16;
  float best = -1e30f;
  int   bidx = row_begin;
  asm volatile("ssai 4");
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *w = head4 + (size_t)r * head4_rowb;
    const int8_t *xp = xq;
    asm volatile("ee.zero.accx");
    asm volatile("ee.vld.128.ip q7, %0, 0" :: "r"(NIB_MASK));
    for (int b = 0; b < nblk; b++) {
      asm volatile(
          "ee.vld.128.ip  q0, %0, 16   \n"
          "ee.andq        q1, q0, q7   \n"
          "ee.vsr.32      q2, q0       \n"
          "ee.andq        q2, q2, q7   \n"
          "ee.vld.128.ip  q3, %1, 16   \n"
          "ee.vld.128.ip  q4, %1, 16   \n"
          "ee.vmulas.s8.accx q1, q3    \n"
          "ee.vmulas.s8.accx q2, q4    \n"
          : "+r"(w), "+r"(xp) :: "memory");
    }
    uint32_t lo;
    asm volatile("rur.accx_0 %0" : "=r"(lo));
    // Identical expression and evaluation order to the storing kernel above, so
    // the compared value is the same float that would have been written.
    float v = (float)((int32_t)lo - corr) * head4_scale[r] * x_scale;
    if (v > best) { best = v; bidx = r; }
  }
  hm_val[slot] = best;
  hm_idx[slot] = bidx;
}

// ---- generic int4 staging for the core -------------------------------------
// The head proved int4-in-PSRAM halves traffic on a bandwidth-bound tensor with
// bit-identical output. The core is the same situation: staged to int8 it is
// ~566 KB, of which only 225 KB fits in internal SRAM. At int4 it is ~300 KB and
// nearly all of it fits -- so this change buys twice: half the bytes, and those
// bytes move from the 60.7 MB/s tier to the 232/947 MB/s tier.
//
// Generalizing the head kernel needs one thing it did not: cols that are not a
// multiple of 32 (ffn_down has cols=66). Handled by padding the COLUMN count up
// and zeroing the activation tail. A padded term contributes code*0 = 0 to
// sum(code*x) and 0 to sum(x), so it cancels out of both the product and the
// zero-point correction whatever weight code sits there. Padding the activation
// rather than the weights is what makes that true.
//
// All core tensors have cols <= group (128), so n_groups == 1 throughout and the
// single hoisted 8*sum(x) correction stays valid.
#define MAX_I4_TENSORS 64
// Tagged so arduino-cli's auto-generated prototypes, which it inserts above
// this point, can name `struct I4T` as an incomplete type.
typedef struct I4T { const QT *src; uint8_t *w4; float *scale; int rowb, cpad; } I4T;
static I4T i4tab[MAX_I4_TENSORS];
static int n_i4 = 0;
static size_t i4_sram = 0, i4_psram = 0;

static struct I4T *i4_find(const QT *t) {
  for (int i = 0; i < n_i4; i++) if (i4tab[i].src == t) return &i4tab[i];
  return NULL;
}

static bool stage_tensor_int4(const QT *t) {
  if (t->n_groups != 1 || n_i4 >= MAX_I4_TENSORS) return false;
  int cpad = (t->cols + 31) & ~31;
  int rowb = cpad / 2;
  size_t bytes = (size_t)t->rows * rowb;
  uint8_t *w4 = (uint8_t *)stage_alloc(bytes);
  float *sc = (float *)stage_alloc((size_t)t->rows * 4);
  if (!w4 || !sc) return false;
  memset(w4, 0x88, bytes);                 // code 8 == value 0 in padded lanes
  for (int r = 0; r < t->rows; r++) {
    const uint8_t *src = t->codes + (size_t)r * t->row_bytes;
    uint8_t *dst = w4 + (size_t)r * rowb;
    for (int blk = 0; blk * 32 < cpad; blk++) {
      for (int k = 0; k < 16; k++) {
        int jl = blk * 32 + k, jh = jl + 16;
        int cl = (jl < t->cols) ? ((jl & 1) ? (src[jl >> 1] >> 4) : (src[jl >> 1] & 0xF)) : 8;
        int ch = (jh < t->cols) ? ((jh & 1) ? (src[jh >> 1] >> 4) : (src[jh >> 1] & 0xF)) : 8;
        dst[blk * 16 + k] = (uint8_t)(cl | (ch << 4));
      }
    }
    sc[r] = half2float(t->scales[r]);
  }
  i4tab[n_i4++] = (I4T){t, w4, sc, rowb, cpad};
  return true;
}

// Shared int4 GEMV: same kernel as the head, parameterized by the staged record.
static void i4_range(const struct I4T *T, const int8_t *xq, float x_scale, int32_t corr,
                     float *y, int row_begin, int row_end) {
  int nblk = T->rowb / 16;
  asm volatile("ssai 4");
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *w = T->w4 + (size_t)r * T->rowb;
    const int8_t *xp = xq;
    asm volatile("ee.zero.accx");
    asm volatile("ee.vld.128.ip q7, %0, 0" :: "r"(NIB_MASK));
    for (int b = 0; b < nblk; b++) {
      asm volatile(
          "ee.vld.128.ip  q0, %0, 16   \n"
          "ee.andq        q1, q0, q7   \n"
          "ee.vsr.32      q2, q0       \n"
          "ee.andq        q2, q2, q7   \n"
          "ee.vld.128.ip  q3, %1, 16   \n"
          "ee.vld.128.ip  q4, %1, 16   \n"
          "ee.vmulas.s8.accx q1, q3    \n"
          "ee.vmulas.s8.accx q2, q4    \n"
          : "+r"(w), "+r"(xp) :: "memory");
    }
    uint32_t lo;
    asm volatile("rur.accx_0 %0" : "=r"(lo));
    y[r] = (float)((int32_t)lo - corr) * T->scale[r] * x_scale;
  }
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;
static int job_begin;

// 0 = staged int8, 1 = int4 head, 2 = int4 core, 3 = int4 head + fused argmax,
// 4 = generic range split (Model.par_for)
static int job_kind;
static int32_t job_corr;      // 8*sum(xq), int4 paths only
static const struct I4T *job_i4;
static void (*job_fn)(void *, int, int);
static void *job_ctx;

#if CERT_HEAD
static void cert_scan_span(const int8_t *xq, int32_t corr, int begin, int end,
                           int slot);
#endif

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
#if CERT_HEAD
    if (job_kind == 5)
      cert_scan_span(job_xq, job_corr, job_begin, job_split, 0);
    else
#endif
    if (job_kind == 4)
      job_fn(job_ctx, 0, job_split);
    else if (job_kind == 3)
      head_i4_argmax_range(job_xq, job_xs, job_corr, 0, 0, job_split);
    else if (job_kind == 2)
      i4_range(job_i4, job_xq, job_xs, job_corr, job_y, 0, job_split);
    else if (job_kind == 1)
      head_i4_range(job_xq, job_xs, job_corr, job_y, 0, job_split);
    else
      matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

// Stage every core tensor as int4, in the order llm_stage_core_int8_alloc walks
// them so that the SRAM-first allocator fills the fast tier with the same set.
static int stage_core_int4(Model *m) {
  int n = 0;
  if (stage_tensor_int4(&m->ple_model_proj)) n++;
  for (int l = 0; l < m->c.n_layers; l++) {
    QT *ts[7] = {&m->qkv[l], &m->attn_proj[l], &m->gate[l], &m->up[l],
                 &m->down[l], &m->ple_gate[l], &m->ple_proj[l]};
    for (int i = 0; i < 7; i++) if (stage_tensor_int4(ts[i])) n++;
  }
  return n;
}

// Layer hook. Falls back to the int4-from-flash path for anything not staged,
// which is correct but slow -- it should never fire, and the boot line reports
// the staged count so a silent fallback is visible rather than mysterious.
static void layer_matvec_i4(const QT *t, const float *x, float *y) {
  const struct I4T *T = i4_find(t);
  if (!T) { matvec_q8(t, x, y); return; }
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  // Zero the padded tail: these lanes must contribute nothing to either the
  // product or the zero-point correction.
  for (int j = t->cols; j < T->cpad; j++) xq[j] = 0;
  int32_t sumx = 0;
  for (int j = 0; j < t->cols; j++) sumx += xq[j];
  int32_t corr = 8 * sumx;
  if (!worker_h || t->rows < 128) { i4_range(T, xq, xs, corr, y, 0, t->rows); return; }
  job_kind = 2; job_i4 = T; job_xq = xq; job_xs = xs; job_corr = corr; job_y = y;
  job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  i4_range(T, xq, xs, corr, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  job_kind = 0;
}

// Head hook: same split as matvec_par, over the int4 head.
static void head_matvec_i4(const QT *t, const float *x, float *y) {
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  int32_t sumx = 0;
  for (int j = 0; j < t->cols; j++) sumx += xq[j];
  int32_t corr = 8 * sumx;              // once per token, not once per row
  if (!worker_h) { head_i4_range(xq, xs, corr, y, 0, t->rows); return; }
  job_kind = 1; job_xq = xq; job_xs = xs; job_corr = corr; job_y = y;
  job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  head_i4_range(xq, xs, corr, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  job_kind = 0;
}

// Greedy head hook: identical arithmetic, no logits written. `y` is ignored --
// llm_forward passes s->logits and reads nothing back, so leaving the buffer
// stale is safe on this path and is the entire point. The chosen token is left
// in head_argmax for the decode loop.
//
// Guarded so it is only ever installed when nothing downstream needs a full
// logit vector. The host verifier (runtime/host_verify/ppl.c) never sets
// head_matvec at all, so it keeps the storing path and CE stays measurable.
static int head_argmax;

static void head_matvec_i4_am(const QT *t, const float *x, float *y) {
  (void)y;
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  int32_t sumx = 0;
  for (int j = 0; j < t->cols; j++) sumx += xq[j];
  int32_t corr = 8 * sumx;
  if (!worker_h) {
    head_i4_argmax_range(xq, xs, corr, 1, 0, t->rows);
    head_argmax = hm_idx[1];
    return;
  }
  job_kind = 3; job_xq = xq; job_xs = xs; job_corr = corr;
  job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  head_i4_argmax_range(xq, xs, corr, 1, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  job_kind = 0;
  // >=, not >: on a tie the flat scan keeps the lower index, which is slot 0.
  int win = (hm_val[0] >= hm_val[1]) ? hm_idx[0] : hm_idx[1];
  /* head4_tok is identity unless CertiHead permuted the head at staging. It is
   * mapped unconditionally because this path is also the FALLBACK when the
   * certified index cannot be allocated -- and rows are permuted before that
   * allocation is attempted. Without this the fallback emits physical row
   * numbers as token ids: measured, on the tile=64 arm, whose 76 KB index does
   * not fit internal SRAM. The board ran at a plausible 39.17 tok/s and
   * produced a completely different story. */
  head_argmax = head4_tok ? head4_tok[win] : win;
}

#if CERT_HEAD
/* ---------------- CertiHead: exact certified output projection -------------
 *
 * The head is 14.6 ms of a 25.0 ms token and is bandwidth bound at 86.3 MiB/s
 * on 1.26 MiB. Reading it faster is nearly exhausted; the lever is reading less.
 *
 * For each tile of CERT_TILE physically-adjacent rows, store the per-dimension
 * interval [lo_j, hi_j] that every row in the tile falls inside. Then for a
 * quantized activation xq,
 *
 *     max_{r in tile} sum_j w_rj * xq_j  <=  sum_j max(xq_j*lo_j, xq_j*hi_j)
 *
 * because each term is independently maximised. A tile whose bound is below the
 * best logit already found CANNOT contain the argmax -- not "probably does
 * not". The emitted token is identical to the dense scan by construction, so
 * the CE, perplexity and 200-token digest carry over unchanged.
 *
 * Costs 199 * 96 * 2 = 38 KB of internal SRAM. The host oracle showed the
 * cheaper-looking 16-row tiling scores fewer rows but needs a 304 KB index that
 * cannot live in SRAM, so it loses once index traffic is charged; and the board
 * bandwidth probe showed that skipping is free above ~4 KB chunks but costs 48%
 * at one cache line, which independently picks the same large tile.
 */
#ifndef CERT_TILE
#define CERT_TILE 128
#endif
/* Tiles per parallel wave. Large enough to amortise the ~34 us task-notify
 * round trip, small enough that the incumbent still tightens several times per
 * token. Swept on the board. */
#ifndef CERT_WAVE
#define CERT_WAVE 32
#endif

/* Bound family. 0 = per-dimension interval box (38 KB index, 199*96 MACs per
 * token). 1 = Cauchy-Schwarz on the row norm, |<w,xq>| <= ||w|| ||xq||, which
 * needs ONE float per tile (796 B) and one compare per tile.
 *
 * Motivated by measurement, not elegance: the board scanned 8,427 rows per
 * token at tile 64, 128, 256 AND 512 -- identical to the row across an 8x range
 * of geometry. With rows sorted by ||w||, the survivor boundary lands at the
 * same NORM regardless of where the tile edges fall, which says the box bound
 * is behaving as a norm threshold and its 96 dimensions of detail are not
 * buying anything. If mode 1 scans the same rows, 38 KB of internal SRAM and
 * 19,000 MACs per token are free. */
#ifndef CERT_BOUND
#define CERT_BOUND 0
#endif

static int8_t  *cert_lo = NULL, *cert_hi = NULL;  /* [NT][cols], internal SRAM */
static float   *cert_qs = NULL;                   /* [NT] index quant scale    */
static float   *cert_mn = NULL;                   /* [NT] max ||w|| in tile    */
static int      cert_nt = 0, cert_cols = 0, cert_rows_total = 0;
static bool     cert_on = false;

/* Per-token scratch. */
static float   *cert_u = NULL;    /* [NT] bound, in units of x_scale */
static int16_t *cert_ord = NULL;  /* [NT] tile ids, descending by bound */
static uint32_t cert_scanned = 0, cert_tokens = 0;
/* The head is 45% of the token and its internals have never been split. Same
 * discipline that found `prep`: attribute before optimising. Three phases --
 * the per-tile bound arithmetic, the sort that orders tiles by bound, and the
 * certified scan itself -- have very different fixes if they dominate. */
static uint64_t cert_bound_us = 0, cert_sort_us = 0, cert_scan_us = 0;

static bool build_cert_index(int rows, int cols) {
  cert_cols = cols;
  cert_rows_total = rows;
  cert_nt = (rows + CERT_TILE - 1) / CERT_TILE;
  size_t n = (size_t)cert_nt * cols;
#if CERT_BOUND == 0
  cert_lo = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
  cert_hi = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
  if (!cert_lo || !cert_hi) return false;
#endif
  cert_qs = (float *)heap_caps_malloc((size_t)cert_nt * 4, MALLOC_CAP_INTERNAL);
  cert_u = (float *)heap_caps_malloc((size_t)cert_nt * 4, MALLOC_CAP_INTERNAL);
  cert_ord = (int16_t *)heap_caps_malloc((size_t)cert_nt * 2, MALLOC_CAP_INTERNAL);
  cert_mn = (float *)heap_caps_malloc((size_t)cert_nt * 4, MALLOC_CAP_INTERNAL);
  if (!cert_qs || !cert_u || !cert_ord || !cert_mn) return false;

  float *flo = (float *)heap_caps_malloc((size_t)cols * 4, MALLOC_CAP_INTERNAL);
  float *fhi = (float *)heap_caps_malloc((size_t)cols * 4, MALLOC_CAP_INTERNAL);
  if (!flo || !fhi) return false;

  for (int tl = 0; tl < cert_nt; tl++) {
    int b = tl * CERT_TILE, e = b + CERT_TILE; if (e > rows) e = rows;
    /* max ||w|| over the tile, for the Cauchy-Schwarz bound. Computed ALWAYS:
     * it lived inside the CERT_BOUND==0 block for one build, which left
     * cert_mn uninitialised in norm mode. The bounds were then garbage, three
     * tiles survived instead of sixty-six, and the board reported 85.74 tok/s
     * with a DIFFERENT output digest. Fast and wrong. The byte check is the
     * only reason that did not become a result. */
    float mnorm = 0.f;
    for (int r = b; r < e; r++) {
      const uint8_t *w = head4 + (size_t)r * head4_rowb;
      float sc = head4_scale[r];
      int32_t nn = 0;
      for (int blk = 0; blk < head4_rowb / 16; blk++)
        for (int kk = 0; kk < 16; kk++) {
          uint8_t by = w[blk * 16 + kk];
          int ul = (int)(by & 0xF) - 8, uh = (int)(by >> 4) - 8;
          nn += ul * ul + uh * uh;
        }
      float nrm = sc * sqrtf((float)nn);
      if (nrm > mnorm) mnorm = nrm;
    }
    cert_mn[tl] = mnorm;

#if CERT_BOUND == 0
    for (int j = 0; j < cols; j++) { flo[j] = 1e30f; fhi[j] = -1e30f; }
    for (int r = b; r < e; r++) {
      /* Read back the STAGED layout, so the index describes exactly the bytes
       * the kernel will read -- not the source tensor it came from. */
      const uint8_t *w = head4 + (size_t)r * head4_rowb;
      float sc = head4_scale[r];
      int nblk = head4_rowb / 16;
      for (int blk = 0; blk < nblk; blk++) {
        for (int k = 0; k < 16; k++) {
          uint8_t byte = w[blk * 16 + k];
          int jl = blk * 32 + k, jh = jl + 16;
          float vl = (float)((int)(byte & 0xF) - 8) * sc;
          float vh = (float)((int)(byte >> 4) - 8) * sc;
          if (vl < flo[jl]) flo[jl] = vl;
          if (vl > fhi[jl]) fhi[jl] = vl;
          if (vh < flo[jh]) flo[jh] = vh;
          if (vh > fhi[jh]) fhi[jh] = vh;
        }
      }
    }
    float mx = 0.f;
    for (int j = 0; j < cols; j++) {
      float a = fabsf(flo[j]), c = fabsf(fhi[j]);
      if (a > mx) mx = a;
      if (c > mx) mx = c;
    }
    float qs = mx > 0.f ? mx / 127.f : 1.f;
    cert_qs[tl] = qs;
    /* Stored as MIDPOINT and HALF-RANGE rather than lo/hi, so the per-token
     * bound becomes two plain dot products instead of a max-of-two-products:
     *
     *     max(x*lo, x*hi)  ==  x*mid + |x|*half
     *
     * which is what makes it PIE-able -- ee.vmulas.s8.accx multiplies and
     * accumulates but cannot take a max, which is why the bound loop was the
     * only scalar kernel left in the head (1.074 ms, EXP-134).
     *
     * Soundness, for either sign of x, needs
     *     mid + half >= hi     and     mid - half <= lo
     * so half is taken as max(hi-mid, mid-lo) AFTER rounding, rather than as
     * (hi-lo)/2 which would be a half-ULP short whenever hi-lo is odd. Both
     * stay inside int8: hi-lo <= 254, so half <= 127.
     *
     * lo/hi themselves are still rounded outward first, for the same reason as
     * before: quantisation may only widen the interval, never narrow it. */
    int8_t *mid = cert_lo + (size_t)tl * cols, *half = cert_hi + (size_t)tl * cols;
    for (int j = 0; j < cols; j++) {
      float ql = floorf(flo[j] / qs), qh = ceilf(fhi[j] / qs);
      if (ql < -128.f) ql = -128.f;
      if (qh > 127.f) qh = 127.f;
      int il = (int)ql, ih = (int)qh;
      int im = (il + ih) >> 1;                  /* floor midpoint */
      int ihalf = (ih - im) > (im - il) ? (ih - im) : (im - il);
      if (im < -127) im = -127;
      if (ihalf > 127) ihalf = 127;
      mid[j] = (int8_t)im; half[j] = (int8_t)ihalf;
    }
#endif  /* CERT_BOUND == 0 */
  }
  heap_caps_free(flo); heap_caps_free(fhi);
#if CERT_BOUND == 0
  sram_used += 2 * n + (size_t)cert_nt * 10;
#else
  /* 796 B of max-norms replaces 38,208 B of per-dimension intervals, having
   * measured that the two prune the identical 8,427 rows. */
  sram_used += (size_t)cert_nt * 14;
#endif
  return true;
}

static int cert_cmp_desc(const void *a, const void *b) {
  float ua = cert_u[*(const int16_t *)a], ub = cert_u[*(const int16_t *)b];
  return (ua < ub) - (ua > ub);
}

/* Scan one tile with the existing PIE kernel, keeping the best. Values are in
 * units of x_scale (it is a positive common factor, so comparisons are
 * unaffected and one multiply per row is saved). */
static void cert_scan_tile(const int8_t *xq, int32_t corr, int tl,
                           int rows, float *best, int *best_row) {
  int b = tl * CERT_TILE, e = b + CERT_TILE; if (e > rows) e = rows;
  int nblk = head4_rowb / 16;
  asm volatile("ssai 4");
  for (int r = b; r < e; r++) {
    const uint8_t *w = head4 + (size_t)r * head4_rowb;
    const int8_t *xp = xq;
    asm volatile("ee.zero.accx");
    asm volatile("ee.vld.128.ip q7, %0, 0" :: "r"(NIB_MASK));
    for (int blk = 0; blk < nblk; blk++) {
      asm volatile(
          "ee.vld.128.ip  q0, %0, 16   \n"
          "ee.andq        q1, q0, q7   \n"
          "ee.vsr.32      q2, q0       \n"
          "ee.andq        q2, q2, q7   \n"
          "ee.vld.128.ip  q3, %1, 16   \n"
          "ee.vld.128.ip  q4, %1, 16   \n"
          "ee.vmulas.s8.accx q1, q3    \n"
          "ee.vmulas.s8.accx q2, q4    \n"
          : "+r"(w), "+r"(xp) :: "memory");
    }
    uint32_t acc;
    asm volatile("rur.accx_0 %0" : "=r"(acc));
    float v = (float)((int32_t)acc - corr) * head4_scale[r];
    /* Ties go to the lower TOKEN id, matching the dense scan. After permutation
     * a lower physical row is not a lower token, so compare head4_tok. */
    if (v > *best || (v == *best && *best_row >= 0 &&
                      head4_tok[r] < head4_tok[*best_row])) {
      *best = v; *best_row = r;
    }
  }
}

/* Scan cert_ord[begin..end) into slot's accumulator. Each core owns a slot, so
 * neither ever reads the other's incumbent -- the halves are independent and
 * the merge afterwards is exact. */
static float cert_best[2];
static int   cert_best_row[2];
static int   cert_rows_done[2];

static void cert_scan_span(const int8_t *xq, int32_t corr, int begin, int end,
                           int slot) {
  float best = -1e30f;
  int row = -1, rows = cert_rows_total;
  for (int k = begin; k < end; k++)
    cert_scan_tile(xq, corr, cert_ord[k], rows, &best, &row);
  cert_best[slot] = best;
  cert_best_row[slot] = row;
  cert_rows_done[slot] = (end - begin) * CERT_TILE;
}

static void head_matvec_i4_cert(const QT *t, const float *x, float *y) {
  (void)y;
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  int32_t sumx = 0;
  for (int j = 0; j < t->cols; j++) sumx += xq[j];
  int32_t corr = 8 * sumx;
  const int cols = t->cols, rows = t->rows;
  uint64_t ct0 = (uint64_t)esp_timer_get_time();

  /* Bounds for every tile: 199 * 96 int8 MACs against an SRAM-resident index. */
#if CERT_BOUND == 1
  /* ||xq||, once per token. The kernel's value is <w_r, xq> in units of
   * x_scale, so Cauchy-Schwarz bounds it directly. */
  int32_t xn2 = 0;
  for (int j = 0; j < cols; j++) xn2 += (int32_t)xq[j] * xq[j];
  float xnorm = sqrtf((float)xn2);
  for (int tl = 0; tl < cert_nt; tl++) {
    float u = xnorm * cert_mn[tl];
    cert_u[tl] = u + fabsf(u) * 1e-6f + 1e-30f;
    cert_ord[tl] = (int16_t)tl;
  }
#else
  /* |xq| once per token, not once per tile: the second dot product needs it for
   * all cert_nt tiles. 96 B of SRAM. */
  LLM_ALIGN16 static int8_t absx[LLM_Q8_MAX_INPUT];
  for (int j = 0; j < cols; j++) {
    int v = xq[j];
    absx[j] = (int8_t)(v < 0 ? -v : v);       /* xq is clamped to +-127, safe */
  }
  for (int tl = 0; tl < cert_nt; tl++) {
    const int8_t *mid = cert_lo + (size_t)tl * cols;
    const int8_t *half = cert_hi + (size_t)tl * cols;
    int32_t s;
#if defined(__XTENSA__)
    /* Two PIE dot-product chains: x.mid and |x|.half. Exact rewrite of the
     * scalar max-of-products via  max(x*lo, x*hi) == x*mid + |x|*half. */
    {
      const int8_t *mp = mid, *hp = half, *xp = xq, *ap = absx;
      asm volatile("ee.zero.accx");
      for (int b = 0; b < cols / 16; b++) {
        asm volatile(
            "ee.vld.128.ip     q0, %0, 16 \n"
            "ee.vld.128.ip     q1, %1, 16 \n"
            "ee.vmulas.s8.accx q0, q1     \n"
            "ee.vld.128.ip     q2, %2, 16 \n"
            "ee.vld.128.ip     q3, %3, 16 \n"
            "ee.vmulas.s8.accx q2, q3     \n"
            : "+r"(mp), "+r"(xp), "+r"(hp), "+r"(ap) :: "memory");
      }
      uint32_t lo32;
      asm volatile("rur.accx_0 %0" : "=r"(lo32));
      s = (int32_t)lo32;
    }
#else
    s = 0;
    for (int j = 0; j < cols; j++)
      s += (int32_t)xq[j] * mid[j] + (int32_t)absx[j] * half[j];
#endif
    /* 1e-6 relative slack absorbs the difference in float rounding between this
     * bound and the kernel's accumulation. It only loosens the certificate. */
    float u = (float)s * cert_qs[tl];
    cert_u[tl] = u + fabsf(u) * 1e-6f + 1e-30f;
    cert_ord[tl] = (int16_t)tl;
  }
#endif
  uint64_t ct1 = (uint64_t)esp_timer_get_time();
  cert_bound_us += ct1 - ct0;
  /* LSD radix sort on the bounds, descending, instead of qsort.
   *
   * Measured 0.330 ms/token for 199 int16 keys (EXP-134) -- ~1500 comparisons at
   * ~52 cycles each, which is the indirect comparator call, not the algorithm.
   * Once the bound loop dropped to 0.099 ms this became the head's second
   * largest term after the scan.
   *
   * IEEE-754 floats compare correctly as unsigned integers after a monotone
   * transform: flip all bits when negative, set the sign bit when positive.
   * Inverting gives descending order directly. Two 8-bit passes over the top 16
   * bits are NOT enough -- ties in the high half would keep qsort's ordering
   * only if the sort were stable at full width -- so all four passes run. It is
   * still ~1800 operations against ~1500 mispredicted comparator calls.
   *
   * Exactness: the ordering is total and identical to a correct descending sort,
   * so the certificate visits the same tiles in the same sequence. Ties between
   * equal bounds may be ordered differently than qsort ordered them; that is
   * safe because the scan's decision is `cert_u[tl] < best`, which does not
   * depend on tie order, and the argmax tie-break is by token id inside
   * cert_scan_tile. The digest is the gate. */
  {
    static uint32_t rk[LLM_MAX_HEADS * 64];   /* keys, >= cert_nt */
    static int16_t rtmp[LLM_MAX_HEADS * 64];
    for (int i = 0; i < cert_nt; i++) {
      union { float f; uint32_t u; } k;
      k.f = cert_u[i];
      k.u = (k.u & 0x80000000u) ? ~k.u : (k.u | 0x80000000u);
      rk[i] = ~k.u;                            /* invert -> descending */
      cert_ord[i] = (int16_t)i;
    }
    for (int sh = 0; sh < 32; sh += 8) {
      int cnt[256] = {0};
      for (int i = 0; i < cert_nt; i++) cnt[(rk[cert_ord[i]] >> sh) & 0xFF]++;
      int sum = 0;
      for (int b = 0; b < 256; b++) { int c = cnt[b]; cnt[b] = sum; sum += c; }
      for (int i = 0; i < cert_nt; i++)
        rtmp[cnt[(rk[cert_ord[i]] >> sh) & 0xFF]++] = cert_ord[i];
      memcpy(cert_ord, rtmp, (size_t)cert_nt * sizeof(int16_t));
    }
  }
  uint64_t ct2 = (uint64_t)esp_timer_get_time();
  cert_sort_us += ct2 - ct1;

  float best = -1e30f;
  int best_row = -1;
  int scanned_tiles = 0;

  /* Seed the incumbent before any pruning decision. Two cheap tiles:
   *   - the previous token's winning tile. Autoregressive output is locally
   *     repetitive, so last step's winner is usually near this step's.
   *   - the highest-bound tile, which is where the winner most often is.
   * A strong incumbent is what makes the certificate bite, and unlike the
   * sequential version this matters more here: the parallel split has to fix
   * the survivor set BEFORE the halves start improving it. */
  static int prev_row = -1;
#if CERT_SEED
  int seed_a = prev_row >= 0 ? prev_row / CERT_TILE : -1;
  int seed_b = cert_ord[0];
  if (seed_a >= 0) { cert_scan_tile(xq, corr, seed_a, rows, &best, &best_row);
                     scanned_tiles++; }
  if (seed_b != seed_a) { cert_scan_tile(xq, corr, seed_b, rows, &best, &best_row);
                          scanned_tiles++; }
#else
  /* No seed: the incumbent starts at -inf, so the first tile in bound order is
   * scanned unconditionally and every later decision uses whatever it found.
   * The two sentinels stay defined so the compaction loop below is unchanged;
   * -1 matches no tile index. */
  const int seed_a = -1, seed_b = -1;
#endif

  /* Compact the survivors in place. w <= k always, so this is safe.
   * Strict <: a tile whose bound EQUALS the incumbent may hold a tie with a
   * lower token id, and the dense scan would have found it. */
  int w = 0;
  for (int k = 0; k < cert_nt; k++) {
    int tl = cert_ord[k];
    if (cert_u[tl] < best) break;              /* sorted: the rest are too */
    if (tl == seed_a || tl == seed_b) continue;
    cert_ord[w++] = (int16_t)tl;
  }

  /* Scan in WAVES rather than one split.
   *
   * A single split measured WORSE than one core (53.91 vs 55.87 tok/s) because
   * it has to fix the survivor set before either half can improve the
   * incumbent: rows scanned went 32.7% -> 44.9%. The certificate's power comes
   * from evaluating tiles in descending-bound order and tightening the
   * incumbent as it goes, which is inherently sequential, and a flat split
   * destroys exactly that.
   *
   * Waves keep both. Each wave splits CERT_WAVE tiles across the two cores,
   * then the tail is re-pruned against the improved incumbent before the next
   * wave. The incumbent updates every CERT_WAVE tiles instead of every tile, so
   * most of the pruning survives, and the ~34 us notify round trip is amortised
   * over CERT_WAVE tiles instead of paid per tile. */
  int k = 0;
  while (k < w) {
    if (cert_u[cert_ord[k]] < best) break;   /* sorted: the tail is gone too */
    int end = k;
    while (end < w && cert_u[cert_ord[end]] >= best) end++;
    int take = end - k;
    if (take > CERT_WAVE) take = CERT_WAVE;

    if (take >= 2 && worker_h) {
      int half = take / 2;
      job_kind = 5; job_xq = xq; job_corr = corr;
      job_begin = k; job_split = k + half;
      xTaskNotifyGive(worker_h);
      cert_scan_span(xq, corr, k + half, k + take, 1);
      ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
      job_kind = 0;
      for (int sl = 0; sl < 2; sl++) {
        if (cert_best_row[sl] < 0) continue;
        if (cert_best[sl] > best ||
            (cert_best[sl] == best && best_row >= 0 &&
             head4_tok[cert_best_row[sl]] < head4_tok[best_row])) {
          best = cert_best[sl]; best_row = cert_best_row[sl];
        }
      }
    } else {
      for (int i = k; i < k + take; i++)
        cert_scan_tile(xq, corr, cert_ord[i], rows, &best, &best_row);
    }
    scanned_tiles += take;
    k += take;
  }

  cert_scan_us += (uint64_t)esp_timer_get_time() - ct2;
  cert_scanned += (uint32_t)scanned_tiles * CERT_TILE;
  cert_tokens++;

  prev_row = best_row;
  head_argmax = head4_tok[best_row];
}
#endif  /* CERT_HEAD */

// Generic range splitter behind Model.par_for. Worker takes [0, split), this
// core takes [split, n). Halves are equal because attention's per-position cost
// is flat -- unlike the head, where nothing varies either.
static void par_for_impl(void (*fn)(void *, int, int), void *ctx, int n) {
  if (!worker_h || n < 2) { fn(ctx, 0, n); return; }
  job_kind = 4; job_fn = fn; job_ctx = ctx; job_split = n / 2;
  xTaskNotifyGive(worker_h);
  fn(ctx, job_split, n);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  job_kind = 0;
}

static void matvec_par(const QT *t, const float *x, float *y) {
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

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
#ifdef LLM_KV_INT8
  // All heads are scored in one traversal of the cache, so every head's scores
  // must be live at once instead of reusing a single [S] buffer.
  s.scores = (float *)sram_or_die((size_t)c->n_heads * S * 4, "scores");
#else
  s.scores = (float *)sram_or_die(S * 4, "scores");
#endif
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
#ifdef LLM_KV_INT8
  // 1.18 MB of fp32 KV becomes 295 KB of int8 plus 12 KB of row scales. That
  // is the whole point: the slope of attn(pos) is this allocation's width.
  // K uses the per-head padded layout so the score kernel can use aligned PIE
  // loads; llm_krow() is the single source of truth and every host verifier
  // calls the same function. V is unpadded -- only the score pass needs it.
  // 16-byte aligned: the PIE score kernel issues ee.vld.128.ip against these
  // rows and an unaligned base would fault or silently misread. heap_caps_malloc
  // alone does not promise 16, so ask for it rather than hope.
  s.kcache8 = (int8_t *)heap_caps_aligned_alloc(16, (size_t)L * S * llm_krow(c),
                                                MALLOC_CAP_SPIRAM);
  if (!s.kcache8) { Serial.println("FATAL: kcache8 aligned alloc failed"); while (1) delay(1000); }
  s.vcache8 = (int8_t *)ps_or_die((size_t)L * S * D, "vcache8");
  s.kscale  = (float *)ps_or_die((size_t)L * S * 4, "kscale");
  s.vscale  = (float *)ps_or_die((size_t)L * S * 4, "vscale");
  // Small and touched every position: internal SRAM, not PSRAM.
  s.wq     = (int8_t *)sram_or_die((size_t)c->n_heads * S, "wq");
  s.wscale = (float *)sram_or_die((size_t)c->n_heads * 4, "wscale");
  // The two V-accumulation banks and the quantized query. These were locals in
  // llm_forward, sized by LLM_MAX_HEADS*64 and LLM_Q8_MAX_INPUT rather than by
  // this model, which put 20 KB of compile-time worst case on a task stack that
  // has 8 KB. Sized from the loaded config they come to 768 + 96 bytes.
  s.acc = (int32_t *)sram_or_die((size_t)2 * D * 4, "acc");  // 2 banks of H*Dh == D
  s.qq  = (int8_t *)heap_caps_aligned_alloc(16, llm_krow(c),
                                            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!s.qq) { Serial.println("FATAL: qq aligned alloc failed"); while (1) delay(1000); }
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

/* Included here, not at the top: the grammar head reuses head4/head4_scale/
 * head4_tok/NIB_MASK and the emit() helper, all of which are file-scope statics
 * defined above. Placing it at the top would need forward declarations for
 * every one of them, which is exactly the kind of duplicated interface that
 * lets two definitions drift apart. Compiles to nothing when ACTION_BENCH=0. */
#include "action_bench.h"

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== ESP32-S3 PLE TinyLM ===");

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
  // decode table. If they disagree, every emitted token would be decoded
  // against the wrong table.
  if (VOCAB_N != model.out_vocab) {
    Serial.printf("FATAL: tokenizer/model mismatch: vocab.h %d, model %d\n",
                  VOCAB_N, model.out_vocab);
    return;
  }

  alloc_scratch();
  copy_norms_to_sram();
  Serial.printf("hot set-> SRAM   %u B dynamic + %u B static = %u B managed\n",
                (unsigned)sram_used, (unsigned)STATIC_SRAM_BYTES,
                (unsigned)(sram_used + STATIC_SRAM_BYTES));

  // Stage every per-position tensor to int8 in PSRAM.
  int want = llm_core_stage_count(&model);
  int staged = stage_core_int4(&model);
  if (staged != want) {
    Serial.printf("FATAL: staged %d/%d core tensors\n", staged, want);
    while (1) delay(1000);
  }
  // The tied head is read per token, not per position, so the core helper does
  // not walk it. Stage it separately -- as int4 where possible, since it is the
  // bandwidth-bound tensor and int8 staging would double its traffic. Falls
  // back to int8 staging if the geometry does not suit the int4 kernel.
  bool head_is_i4 = stage_head_int4(&model.out_head);
  if (!head_is_i4) {
    void *b = ps_or_die(llm_stage_int8_bytes(&model.out_head), "staged head");
    llm_stage_int8(&model.out_head, b);
  }
  ++staged;
  Serial.printf("weights-> %d tensors int8; core SRAM %u B / PSRAM %u B "
                "(%.0f%% in fast tier)\n", staged,
                (unsigned)sram_staged, (unsigned)psram_staged,
                100.0 * sram_staged / (sram_staged + psram_staged + 1));

  // Whether the SIMD path actually engages, rather than assuming it. A
  // misaligned base or a cols not divisible by 16 makes every row fall back to
  // scalar, which would make the whole change measure as a no-op and be read as
  // "SIMD does not help here". Report the head explicitly: it is 63% of the
  // token and the only tensor whose speed decides the headline number.
  {
    const QT *hd = &model.out_head;
    size_t i8 = (size_t)hd->rows * hd->cols;
    size_t i4 = (size_t)hd->rows * head4_rowb + (size_t)hd->rows * 4;
    Serial.printf("head: %dx%d  int8 would read %.2f MB/token\n",
                  hd->rows, hd->cols, i8 / 1048576.0);
    if (head_is_i4)
      Serial.printf("head: INT4 in PSRAM, %.2f MB/token (%.2fx less), "
                    "base %p aligned %s\n", i4 / 1048576.0, (double)i8 / i4,
                    (void *)head4, ((uintptr_t)head4 & 15) ? "NO" : "yes");
    else
      Serial.printf("head: int8 fallback, base %p\n", (void *)hd->w8);
#ifndef LLM_HAVE_SIMD_S8
    Serial.println("head: built WITHOUT LLM_HAVE_SIMD_S8 (scalar kernel)");
#endif
  }

  main_h = xTaskGetCurrentTaskHandle();
  // 8 KB, not 4: the worker originally ran only matvec ranges, and now also
  // runs llm_score_range and llm_vacc_range. Under -O3 those inline their inner
  // loops and carry larger frames. A worker stack overflow does not fault
  // cleanly -- it smashes whatever the allocator put next to it, and surfaces
  // much later as a crash inside malloc (observed on the ESP-IDF port: printf
  // of a float -> _dtoa_r -> _Balloc -> calloc -> fault in heap_caps).
  if (xTaskCreatePinnedToCore(worker_main, "mv", 8192, NULL, 2, &worker_h, 0) == pdPASS) {
    // After the worker exists: matvec_par notifies worker_h.
    model.layer_matvec = layer_matvec_i4;
    model.head_matvec  = head_is_i4 ? head_matvec_i4 : matvec_par;
    model.par_for      = par_for_impl;
  } else {
    Serial.println("dual-core worker failed; running single core");
    model.layer_matvec = layer_matvec_i4;
    if (head_is_i4) model.head_matvec = head_matvec_i4;
  }

  // Fused argmax replaces the head hook entirely, so it needs the int4 head.
  // Reported rather than assumed: a silent fallback to the storing path would
  // look like "the optimization did nothing" instead of "it never ran".
#if CERT_HEAD
  if (head_is_i4 && build_cert_index(model.out_head.rows, model.out_head.cols)) {
    model.head_matvec = head_matvec_i4_cert;
    head_fused = true;
    cert_on = true;
    Serial.printf("argmax: FUSED + CERTIFIED  %d tiles x %d rows, index %u B SRAM\n",
                  cert_nt, CERT_TILE, (unsigned)(2 * cert_nt * model.out_head.cols));
    Serial.println("head rows: permuted by ||w|| at staging (weights unchanged)");
  } else
#endif
  if (FUSE_ARGMAX && head_is_i4) {
    model.head_matvec = head_matvec_i4_am;
    head_fused = true;
    Serial.println("argmax: FUSED into head (logits never materialized)");
#if CERT_HEAD
    /* Reached only when build_cert_index failed. Say so: a silent fallback
     * looks like "CertiHead did nothing" instead of "CertiHead never ran", and
     * the index is the largest internal-SRAM request in the firmware. */
    Serial.printf("CERTIHEAD DISABLED: index alloc failed (needed %u B internal "
                  "SRAM for %d rows / %d per tile)\n",
                  (unsigned)(2 * ((model.out_head.rows + CERT_TILE - 1) / CERT_TILE)
                             * model.out_head.cols),
                  model.out_head.rows, CERT_TILE);
#endif
  } else {
    Serial.printf("argmax: separate scan over %d logits%s\n", model.out_vocab,
                  FUSE_ARGMAX ? " (fusion wanted but head is not int4)" : "");
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
  Serial.printf("free: sram %.0f KB | psram %.2f MB\n\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);

  /* The action arm REPLACES the free-text run rather than following it. Sharing
   * a KV cache would put the action phase at positions 200+ against the
   * control's 0..199, and attention cost grows with position -- the grammar
   * would then appear to cost something it does not. */
#if ACTION_BENCH
  action_bench(ACTION_DECISIONS);
  blink(0);
  return;
#endif

  // ---- generate ----
  Serial.print(">>> ");
  int n_prompt = sizeof(PROMPT_IDS) / sizeof(int);
  int pos = 0, tok = 0;
  int64_t decode_us = 0;
  int decoded = 0;

  for (int i = 0; i < n_prompt; i++) {  // prime with the prompt
    tok = PROMPT_IDS[i];
    emit(tok);
    llm_forward(&model, tok, pos++, &s);
  }

  llm_profile_reset(&s);

  /* Perception starts HERE, not at boot, so the frame rate reported is the rate
   * sustained *during* the timed decode loop rather than an average diluted by
   * the seconds of staging and flash-fingerprinting that precede it. Compiled
   * out entirely in the Arduino build (no dfr1154_sensors.h on that include
   * path), which is what keeps the two build systems comparable. */
#ifdef DFR1154_SENSORS_H
  dfr_capture_start();
#endif

  int64_t argmax_us = 0, emit_us = 0;
  int64_t t_start = esp_timer_get_time();
  for (int step = 0; step < N_GENERATE && pos < model.c.seq_len; step++) {
    // The argmax is outside llm_forward and so absent from the profile, but it
    // is not free: out_vocab floats read from PSRAM every token. At 25,353 that
    // is 99 KiB per token over a bus measured at 60.7 MB/s. Timed separately
    // because the head is exactly what this project is trying to replace, and
    // the replacement has to beat head_us + argmax_us, not head_us alone.
    //
    // When fused, this line reads a register the head already filled and the
    // cost moves INTO head_us -- so compare the wall figure across builds, not
    // the head row, which now does strictly more work for strictly less time.
    int64_t a0 = esp_timer_get_time();
    int best;
    if (head_fused) {
      best = head_argmax;
    } else {
      best = 0; float bv = -1e30f;
      for (int v = 0; v < model.out_vocab; v++)
        if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    }
    argmax_us += esp_timer_get_time() - a0;
    tok = best;
    int64_t e0 = esp_timer_get_time();
    emit(tok);
    emit_us += esp_timer_get_time() - e0;
    blink((step & 1) ? 40 : 8);

    int64_t d0 = esp_timer_get_time();
    llm_forward(&model, tok, pos++, &s);
    decode_us += esp_timer_get_time() - d0;
    decoded++;
    if ((step & 7) == 0) delay(0);  // feed the task WDT ~every 8 tokens
  }
  int64_t total_us = esp_timer_get_time() - t_start;

  Serial.printf("\n\n--- %d tokens in %.2f s ---\n", decoded, total_us / 1e6);
  Serial.printf("throughput: %.2f tok/s   (%.1f ms/token)\n",
                decoded * 1e6 / total_us, decode_us / 1000.0 / decoded);
  // The two figures above measure different things and have been confused
  // before: the first is wall clock over the whole loop, the second is
  // llm_forward alone. Print the residual so the gap is never inferred again.
  Serial.printf("accounting ms/token: forward %.2f | argmax %.2f | emit %.2f "
                "| unaccounted %.2f | wall %.2f\n",
                decode_us / 1000.0 / decoded, argmax_us / 1000.0 / decoded,
                emit_us / 1000.0 / decoded,
                (total_us - decode_us - argmax_us - emit_us) / 1000.0 / decoded,
                total_us / 1000.0 / decoded);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("profile ms/token: input %.1f | attn %.1f | ffn %.1f | ple %.1f | head %.1f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
    // Attention split. `other` is the QKV/output projections and the k/v
    // quantize -- i.e. the part that is still weights and memory. Printed as a
    // residual so the four terms are forced to add up to attn and no cost can
    // hide between them.
    {
      float sc = s.profile.attn_score_us / n, so = s.profile.attn_soft_us / n,
            va = s.profile.attn_vacc_us / n, at = s.profile.attn_us / n;
      float pr = s.profile.attn_proj_us / n, op = s.profile.attn_outp_us / n,
            pp = s.profile.attn_prep_us / n;
      Serial.printf("  attn split ms: qkv %.2f | prep %.2f | score %.2f "
                    "| softmax %.2f | vacc %.2f | outproj %.2f | other %.2f "
                    " (sum %.2f of %.2f)\n",
                    pr, pp, sc, so, va, op, at - pr - pp - sc - so - va - op,
                    pr + pp + sc + so + va + op, at);
      Serial.printf("  prep split ms: rope %.3f | kv quantize+store %.3f "
                    "| q quantize + init %.3f\n",
                    s.profile.attn_rope_us / n, s.profile.attn_kvq_us / n,
                    pp - s.profile.attn_rope_us / n - s.profile.attn_kvq_us / n);
    }
  }
#if CERT_HEAD
  /* The device reports its own pruning rate. A certificate that silently
   * degenerated to a full scan would otherwise look like a null result rather
   * than a broken one. */
  if (cert_on && cert_tokens) {
    float rows = (float)cert_scanned / (float)cert_tokens;
    Serial.printf("  head split ms: bound %.3f | sort %.3f | scan %.3f\n",
                  cert_bound_us / (float)cert_tokens / 1000.f,
                  cert_sort_us / (float)cert_tokens / 1000.f,
                  cert_scan_us / (float)cert_tokens / 1000.f);
    Serial.printf("  certihead: %.0f of %d rows/token (%.1f%%), %.2fx less head "
                  "traffic\n", rows, model.out_head.rows,
                  100.f * rows / model.out_head.rows,
                  (float)model.out_head.rows / rows);
  }
#endif
#if VACC_STATS
  /* Effective V traffic. Dh int8 are read per LIVE (position,head) pair; a pair
   * whose quantised weight is zero touches no V bytes at all. Comparing that
   * against the nominal 6 x pos x D says whether vacc's 56 MiB/s is a kernel
   * that is leaving bus on the table or a kernel that is already reading the
   * minimum the algorithm allows. */
  {
    /* Sum the two per-core slots. Summing here rather than in the kernel is
     * what makes these counts exact -- see the retraction note in llm.h. */
    uint64_t live = llm_vacc_pairs_live[0] + llm_vacc_pairs_live[1];
    uint64_t seen = llm_vacc_pairs_seen[0] + llm_vacc_pairs_seen[1];
    uint64_t dead = llm_vacc_rows_dead[0] + llm_vacc_rows_dead[1];
    uint64_t rows = llm_vacc_rows_seen[0] + llm_vacc_rows_seen[1];
    double toks = (double)decoded;
    double Dh = model.c.dim / model.c.n_heads;
    double live_B = (double)live * Dh, nom_B = (double)seen * Dh;
    double ms = s.profile.attn_vacc_us / 1000.0 / toks;
    /* Self-check: seen/token must equal L * H * mean(pos+1). If it does not,
     * the counters are racing again and every byte figure below is void. */
    Serial.printf("vacc stats: %.0f pairs/token seen (expect L*H*meanpos), "
                  "%.1f%% live, %.1f%% of positions fully dead\n",
                  seen / toks, 100.0 * live / seen, 100.0 * dead / rows);
    Serial.printf("vacc traffic: nominal %.0f B/token, actual %.0f B/token "
                  "-> %.1f MiB/s effective over %.2f ms (bus peak 85.3)\n",
                  nom_B / toks, live_B / toks,
                  live_B / toks / ms * 1000.0 / 1048576.0, ms);
  }
#endif

  /* Perception result, printed next to the inference result rather than in a
   * separate run: the whole point is that these two numbers were produced by
   * the same silicon in the same seconds. Reported after the decode loop so a
   * capture task that died would show as a frame count that stopped advancing,
   * not as a silently absent workload. */
#ifdef DFR1154_SENSORS_H
  dfr_sensors_report();
  Serial.printf("free after decode: sram %.0f KB | psram %.2f MB\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);
#endif

#if USE_DISPLAY
  display_stats(decoded * 1e6f / decode_us, decode_us / 1000.0f / decoded);
#endif
  blink(0);
}

void loop() { delay(10000); }
