// ESP32-P4 vector kernel for the staged int8 matvec, using the PIE / Xespv
// extension (TRM chapter 4). Same arithmetic as matvec_i8_range in llm.h:
// per row, per group, an exact integer dot product scaled by the group's
// float scale, then by the activation scale. Results are bit-identical to the
// scalar kernel, which the firmware checks once at boot.
//
// ESP.VMULAS.S8.XACC multiplies 16 int8 pairs and adds all 16 products into
// the 40-bit XACC accumulator in one instruction; the .LD.IP form also loads
// the next 16 weights. A group of 128 is therefore 8 such steps.
//
// Requirements, all arranged by the caller:
//   - llm.h included with LLM_STAGE_ALIGN 16, so t->stride8 is a multiple of
//     16 and padding bytes are zero;
//   - t->w8 16-byte aligned (heap_caps_aligned_alloc), so every row is;
//   - xq 16-byte aligned and zero from cols up to stride8;
//   - t->group a multiple of 16.
// The PIE unit runs in force-alignment mode by default: a misaligned address
// silently reads the aligned data below it, so these are not optional.
#ifndef LLM_PIE_H
#define LLM_PIE_H
#include <stdint.h>
#include "llm.h"

#include "llm_pie_dot.h"

#if LLM_HAVE_PIE
#define llm_pie_dot llm_pie_dot_s8v

// Drop-in for matvec_i8_range on a staged tensor that meets the requirements.
static void matvec_pie_range(const QT *t, const int8_t *xq, float x_scale,
                             float *y, int row_begin, int row_end) {
  int g = t->group, ng = t->n_groups, stride = t->stride8;
  int gvec = g / 16;                       // vectors per full group
  for (int r = row_begin; r < row_end; r++) {
    const int8_t *w = t->w8 + (size_t)r * stride;
    const float *sc = t->scale8 + (size_t)r * ng;
    float acc = 0.f;
    for (int gi = 0; gi < ng; gi++) {
      int begin = gi * g;
      int span = stride - begin;           // padded bytes left in the row
      int nvec = span < g ? span / 16 : gvec;
      acc += (float)llm_pie_dot(w + begin, xq + begin, nvec) * sc[gi];
    }
    y[r] = acc * x_scale;
  }
}

// ---- int4 weights, unpacked in registers ------------------------------------
// For a tensor staged with llm_stage_int4 whose cols and group are multiples
// of 32. Reads half the bytes of the int8 kernel; the head is bandwidth-bound,
// so that is where the time goes.
static const uint8_t llm_pie4_mask[16] __attribute__((aligned(16))) = {
  0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
  0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F };

// Lay out the int8 activations for llm_pie_dot4 and compute 8 * sum(x) per
// group, the correction that turns code * x into (code - 8) * x.
static inline void llm_pie4_prepare(const int8_t *xq, int cols, int group,
                                    int8_t *xperm, int32_t *xsum8) {
  for (int c = 0; c < cols / 32; c++) {
    for (int i = 0; i < 16; i++) {
      xperm[c * 32 + i]      = xq[c * 32 + 2 * i];
      xperm[c * 32 + 16 + i] = xq[c * 32 + 2 * i + 1];
    }
  }
  int ng = (cols + group - 1) / group;
  for (int gi = 0; gi < ng; gi++) {
    int begin = gi * group, end = begin + group;
    if (end > cols) end = cols;
    int32_t sum = 0;
    for (int j = begin; j < end; j++) sum += xq[j];
    xsum8[gi] = 8 * sum;
  }
}

static inline void llm_pie4_row(const QT *t, const int8_t *xperm, const int32_t *xsum8,
                                float x_scale, float *y, int r) {
  int g = t->group, ng = t->n_groups, cols = t->cols;
  const uint8_t *w = t->w4 + (size_t)r * t->stride4;
  const float *sc = t->scale8 + (size_t)r * ng;
  float acc = 0.f;
  for (int gi = 0; gi < ng; gi++) {
    int begin = gi * g, span = cols - begin;
    if (span > g) span = g;
    int32_t d = llm_pie_dot4(w + begin / 2, xperm + begin, span / 32, llm_pie4_mask)
                - xsum8[gi];
    acc += (float)d * sc[gi];
  }
  y[r] = acc * x_scale;
}

// Same result as matvec_q8_range on the flash codes, and as matvec_i8_range.
//
// The compute per row is a few dozen instructions; the time is the cache
// misses on the packed rows in PSRAM. A platform with a cache preload engine
// defines LLM_PRELOAD_START(addr, bytes) and LLM_PRELOAD_END() before
// including this file: the rows are then taken in chunks of LLM_PRELOAD_BYTES
// and each chunk's successor is requested before the chunk is computed, so
// the engine streams the weights while the vector unit works on the previous
// chunk. START may block until the engine is free.
static void matvec_pie4_range(const QT *t, const int8_t *xperm, const int32_t *xsum8,
                              float x_scale, float *y, int row_begin, int row_end) {
#ifdef LLM_PRELOAD_START
  int stride = t->stride4;
  int chunk = LLM_PRELOAD_BYTES / stride;
  if (chunk < 1) chunk = 1;
  if (row_begin < row_end) {
    int r1 = row_begin + chunk;
    if (r1 > row_end) r1 = row_end;
    LLM_PRELOAD_START(t->w4 + (size_t)row_begin * stride, (size_t)(r1 - row_begin) * stride);
  }
  for (int r0 = row_begin; r0 < row_end; r0 += chunk) {
    int r1 = r0 + chunk;
    if (r1 > row_end) r1 = row_end;
    if (r1 < row_end) {
      int r2 = r1 + chunk;
      if (r2 > row_end) r2 = row_end;
      LLM_PRELOAD_START(t->w4 + (size_t)r1 * stride, (size_t)(r2 - r1) * stride);
    }
    for (int r = r0; r < r1; r++) llm_pie4_row(t, xperm, xsum8, x_scale, y, r);
  }
  LLM_PRELOAD_END();
#else
  for (int r = row_begin; r < row_end; r++) llm_pie4_row(t, xperm, xsum8, x_scale, y, r);
#endif
}

static inline int llm_pie4_ok(const QT *t) {
  return t->w4 != NULL && (t->cols % 32) == 0 && (t->group % 32) == 0 &&
         (((uintptr_t)t->w4) & 15) == 0 && (t->stride4 % 16) == 0;
}

// Whether a staged tensor can go through the vector kernel.
static inline int llm_pie_ok(const QT *t) {
  return t->w8 != NULL && (t->stride8 % 16) == 0 && (t->group % 16) == 0 &&
         (((uintptr_t)t->w8) & 15) == 0;
}
#endif

#endif
