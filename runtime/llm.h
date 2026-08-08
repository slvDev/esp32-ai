// Portable single-header inference for the PLE TinyLM. Same code runs on the
// host (verify against PyTorch golden) and on the ESP32-S3 (mmap'd flash). No
// dynamic arch: dims come from the model.bin header. Weights are bound in place
// from `base` by default; platforms may relocate hot tensors and override the
// output-head matvec.
//
// Matches src/model.py op-for-op: split-half RoPE, erf-GELU, SiLU-SwiGLU,
// RMSNorm(weight * x * rsqrt(mean(x^2)+eps)), tied input/output embedding, and
// the PLE input  (RMSNorm(proj(x)/sqrt(D)) + table[tok]*sqrt(P)) / sqrt(2)
// gated per layer as  x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l)).
#ifndef LLM_H
#define LLM_H
#include <stdint.h>
#include <math.h>
#include <string.h>
#define LLM_MAGIC 0x00454C50u        /* "PLE\0" */
#define LLM_FORMAT_VERSION 1u
#define LLM_FLAG_TIED_HEAD 1u        /* head is the token embedding */
#define LLM_FLAGS_KNOWN LLM_FLAG_TIED_HEAD
#define LLM_HEADER_BYTES 56u         /* version 1 */
#define LLM_HEADER_MAX 4096u         /* sanity bound on a future header */
#define LLM_MAX_LAYERS 32            /* fixed per-layer arrays below */
#define RMS_EPS 1e-6f
#define LLM_Q8_MAX_INPUT 4096
/* Portable 16-byte alignment: the device needs it for PIE vector loads, and the
   host verifier (MSVC) has to compile the same header. */
#if defined(_MSC_VER)
#  define LLM_ALIGN16 __declspec(align(16))
#elif defined(__GNUC__)
#  define LLM_ALIGN16 __attribute__((aligned(16)))
#else
#  define LLM_ALIGN16
#endif
// Upper bound on n_heads, for the per-head softmax state the LLM_KV_INT8 path
// keeps on the stack while all heads share one traversal of the KV cache.
#define LLM_MAX_HEADS 32

/* Optional host-side activation instrumentation.  Production builds leave
 * this as a no-op.  A diagnostic harness may define LLM_TRACE_EVENT before
 * including this header to capture the real residual stream without forking
 * the inference implementation.  stage: 0=input, 1=attention residual,
 * 2=FFN residual, 3=PLE residual, 4=pre-output-normalization residual,
 * 5=final normalized state. */
#ifndef LLM_TRACE_EVENT
#define LLM_TRACE_EVENT(stage, layer, values, count) ((void)0)
#endif


typedef struct {
  int vocab, dim, n_layers, n_heads, ffn, ple_dim, seq_len, group;
  float rope_theta;
} Cfg;

// A group-wise int4 tensor viewed in place: ragged packed nibbles (row-aligned
// to a byte) + fp16 group scales. Per-tensor group. No padding.
typedef struct {
  const uint8_t  *codes;   // rows*row_bytes, nibble = value+8, row_bytes=ceil(cols/2)
  const uint16_t *scales;  // rows*n_groups fp16
  int rows, cols, group, n_groups, row_bytes;
  /* Optional int8 staging: nibbles unpacked and group scales converted once, so
   * each matvec skips both. Numerics are unchanged - same codes, same scales,
   * same group sums. NULL selects the int4 path. */
  const int8_t *w8;        // rows*cols, values -7..7
  const float  *scale8;    // rows*n_groups, half2float'd once
} QT;

// IEEE half -> float.
static inline float half2float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, f;
  if (exp == 0) {
    if (man == 0) f = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      man &= 0x3FF; f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; memcpy(&out, &f, 4); return out;
}

typedef struct {
  Cfg c;
  QT tok_emb;             // [V, D]  input embedding
  QT out_head;            // [Vout, D]; first Vout rows of tok_emb when tied
  int out_vocab;          // logits produced per step
  QT ple_model_proj;      // [L*P, D]
  const float *ple_proj_norm; // [P]
  QT ple_table;           // [V, L*P]
  const float *attn_norm[LLM_MAX_LAYERS]; // [D]
  QT qkv[LLM_MAX_LAYERS];             // [3D, D]
  QT attn_proj[LLM_MAX_LAYERS];       // [D, D]
  const float *ffn_norm[LLM_MAX_LAYERS];  // [D]
  QT gate[LLM_MAX_LAYERS], up[LLM_MAX_LAYERS], down[LLM_MAX_LAYERS];
  QT ple_gate[LLM_MAX_LAYERS];        // [P, D]
  QT ple_proj[LLM_MAX_LAYERS];        // [D, P]
  const float *ple_norm[LLM_MAX_LAYERS];  // [D]
  const float *out_norm;      // [D]
  size_t image_bytes;     // bytes consumed by the image, which is smaller than
                          // the flash partition holding it

  // Optional platform overrides; NULL uses MATVEC. Separate hooks because a
  // tied 32,768-row head and a 66-row FFN matrix have different cost profiles.
  void (*head_matvec)(const QT *, const float *, float *);
  void (*layer_matvec)(const QT *, const float *, float *);

  /* Optional: run fn(ctx, begin, end) over [0, n) split across cores, returning
   * once both halves are done. NULL runs the whole range inline.
   *
   * A general range-splitter rather than another bespoke hook: attention is no
   * longer one hot loop but four terms of 1.5-3.0 ms each (measured), so the
   * useful primitive is "parallelise an arbitrary position range", not
   * "parallelise the matvec". The caller owns any per-half reduction state and
   * keys it off `begin`. */
  void (*par_for)(void (*fn)(void *, int, int), void *ctx, int n);
} Model;

// Advance a cursor over the file, binding one quant tensor. Reads the per-tensor
// group prefix, then ragged codes + fp16 scales.
static const uint8_t *bind_q(const uint8_t *p, QT *t, int rows, int cols) {
  int32_t group; memcpy(&group, p, 4); p += 4;
  t->rows = rows; t->cols = cols; t->group = group;
  t->n_groups = (cols + group - 1) / group;
  t->row_bytes = (cols + 1) / 2;
  t->codes = p;  p += (size_t)rows * t->row_bytes;
  t->scales = (const uint16_t *)p;  p += (size_t)rows * t->n_groups * 2;
  t->w8 = NULL; t->scale8 = NULL;   // int4 path until staged
  return p;
}
static const uint8_t *bind_f(const uint8_t *p, const float **t, int n) {
  *t = (const float *)p;  return p + (size_t)n * sizeof(float);
}

// Dequantize row r of a quant tensor into out[cols].
static inline void deq_row(const QT *t, int r, float *out) {
  const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
  const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
  for (int gi = 0; gi < t->n_groups; gi++) {
    int begin = gi * t->group;
    int end = begin + t->group;
    if (end > t->cols) end = t->cols;
    float scale = half2float(sc[gi]);
    int j = begin;
    if ((j & 1) && j < end) {
      out[j] = (float)((row[j >> 1] >> 4) - 8) * scale;
      j++;
    }
    for (; j + 1 < end; j += 2) {
      uint8_t byte = row[j >> 1];
      out[j] = (float)((byte & 0xF) - 8) * scale;
      out[j + 1] = (float)((byte >> 4) - 8) * scale;
    }
    if (j < end) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      out[j] = (float)(code - 8) * scale;
    }
  }
}

// y[row_begin:row_end] = W * x for a quant tensor W. Keeping the row range
// explicit lets platforms parallelize the large output head without changing
// any individual dot product.
static inline void matvec_q_range(const QT *t, const float *x, float *y,
                                  int row_begin, int row_end) {
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float acc = 0.f;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group;
      int end = begin + t->group;
      if (end > t->cols) end = t->cols;
      float scale = half2float(sc[gi]);
      float group_acc = 0.f;
      int j = begin;
      if ((j & 1) && j < end) {
        group_acc += (float)((row[j >> 1] >> 4) - 8) * x[j];
        j++;
      }
      for (; j + 1 < end; j += 2) {
        uint8_t byte = row[j >> 1];
        group_acc += (float)((byte & 0xF) - 8) * x[j];
        group_acc += (float)((byte >> 4) - 8) * x[j + 1];
      }
      if (j < end) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        group_acc += (float)(code - 8) * x[j];
      }
      acc += group_acc * scale;
    }
    y[r] = acc;
  }
}

// y[rows] = W * x[cols], W a quant tensor [rows, cols].
static inline void matvec_q(const QT *t, const float *x, float *y) {
  matvec_q_range(t, x, y, 0, t->rows);
}

// ---- int8-activation path ---------------------------------------------------
// Quantize x to int8 once per call, then per output row do int8*int8 -> int32
// group dot products, scaled by (x_scale * group_scale). The only approximation
// against matvec_q is the int8 activations. Split into quantize + ranged-dot so
// a parallel caller can quantize once and let both cores read the result.
/* Activation quantization. Measured at 0.61 ms/token across its 18+ calls
 * (EXP-129) -- ~85 cycles per element for what should be a few. It is the third
 * largest attention term despite doing almost no arithmetic, and it was
 * invisible until attention was fully attributed.
 *
 * QACT selects the implementation so the arms can be A/B'd on the board:
 *   0  reference. Already uses a reciprocal multiply, so there is no per-element
 *      divide to remove -- the cost is elsewhere.
 *   1  lrintf -> add-half-and-truncate. lrintf honours the current rounding mode
 *      and on this toolchain is a call, not an instruction. NOT bit-identical in
 *      principle: lrintf rounds halves to even, add-half rounds them away from
 *      zero, so exact .5 products differ. Rare, but it must be checked rather
 *      than assumed -- the digest is the gate.
 *   2  arm 1 plus a 4-way unrolled max reduction, which breaks the serial
 *      dependency through the running maximum. Rounding identical to arm 1.
 */
/* K-cache row layout.
 *
 * The score pass is 1.52 ms, 12x above its own MAC count, and EXP-132 showed the
 * gap is not a dependency stall (breaking the dependency made it worse). The
 * remaining candidate is issue rate: the head kernel uses PIE
 * (`ee.vmulas.s8.accx`, 16 int8 MACs per instruction) and the score kernel does
 * not, because each head's slice of a K row is Dh=24 bytes and nothing lands on
 * a 16-byte boundary.
 *
 * KPAD=1 pads each head's slice up to a 16-byte multiple, so per position the
 * cache holds H * 32 = 128 bytes instead of D = 96. That is 33% more KV traffic
 * bought against a 4x MAC rate -- a trade worth testing precisely because
 * attention was measured compute-bound (7.9 ms arithmetic behind 6.3 ms memory).
 *
 * BIT-EXACT: the pad bytes are zeroed and contribute zero to every dot product.
 *
 * llm_krow() is the single source of truth for the allocation size and is used
 * by the firmware and by every host verifier, so the two cannot drift. A
 * mismatch here would silently corrupt attention rather than fail loudly, which
 * is why no site is allowed to compute `L*S*D` for the K cache any more. */
#ifndef KPAD
#define KPAD 1
#endif

static inline int llm_khs(const Cfg *c) {   /* per-head stride, bytes */
  int Dh = c->dim / c->n_heads;
#if KPAD
  Dh = (Dh + 15) & ~15;
#endif
  return Dh;
}
static inline size_t llm_krow(const Cfg *c) {  /* K bytes per position */
  return (size_t)llm_khs(c) * c->n_heads;
}

#ifndef QACT
/* DEFAULT 1 as of EXP-130. Board A/B, 2 captures per arm:
 *     QACT=0  61.90 / 61.90 tok/s   prep 0.66   digest 43184d1c
 *     QACT=1  67.14 / 66.64 tok/s   prep 0.29   digest 43184d1c
 *     QACT=2  66.39 / 66.39 tok/s   prep 0.28   digest 43184d1c
 * lrintf was the entire cost. The 4-way unrolled max reduction (arm 2) buys
 * nothing on top, so the reduction was never the bottleneck and arm 2 is not
 * worth its code.
 *
 * The gain, 1.27 ms/token, EXCEEDS the 0.6 ms ceiling predicted from prep --
 * because that prediction counted only the attention call sites. quantize_act
 * also runs once per matvec (LLM_LMV), i.e. in FFN, PLE and both projections.
 * The prediction was wrong by under-counting callers, not by mis-measuring. */
#define QACT 1
#endif

static inline void quantize_act(const float *x, int n, int8_t *xq, float *x_scale) {
#if QACT == 2
  float m0 = 1e-8f, m1 = 1e-8f, m2 = 1e-8f, m3 = 1e-8f;
  int j = 0;
  for (; j + 3 < n; j += 4) {
    float a0 = fabsf(x[j]), a1 = fabsf(x[j + 1]);
    float a2 = fabsf(x[j + 2]), a3 = fabsf(x[j + 3]);
    if (a0 > m0) m0 = a0;
    if (a1 > m1) m1 = a1;
    if (a2 > m2) m2 = a2;
    if (a3 > m3) m3 = a3;
  }
  for (; j < n; j++) { float a = fabsf(x[j]); if (a > m0) m0 = a; }
  if (m1 > m0) m0 = m1;
  if (m3 > m2) m2 = m3;
  float xmax = m2 > m0 ? m2 : m0;
#else
  float xmax = 1e-8f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > xmax) xmax = a; }
#endif
  float inv = 127.f / xmax;
#if QACT >= 1
  for (int j = 0; j < n; j++) {
    float f = x[j] * inv;
    int q = (int)(f + (f >= 0.f ? 0.5f : -0.5f));   /* truncate, no libm call */
    xq[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
#else
  for (int j = 0; j < n; j++) {
    int q = (int)lrintf(x[j] * inv);
    xq[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
#endif
  *x_scale = xmax / 127.f;
}

/* Quantize a D-vector into the per-head padded K layout.
 *
 * One scale over the whole vector, exactly as quantize_act computes it -- the
 * scale must not become per-head or the cached values change. Only the
 * DESTINATION differs: each head's Dh values go to its own `stride`-byte slot
 * and the remaining bytes are zeroed, so they contribute nothing to any dot
 * product and the layout change is bit-exact. */
static inline void quantize_act_kpad(const float *x, int H, int Dh, int stride,
                                     int8_t *dst, float *x_scale) {
  const int n = H * Dh;
  float xmax = 1e-8f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > xmax) xmax = a; }
  const float inv = 127.f / xmax;
  for (int hh = 0; hh < H; hh++) {
    const float *src = x + hh * Dh;
    int8_t *d = dst + (size_t)hh * stride;
    for (int i = 0; i < Dh; i++) {
      float f = src[i] * inv;
      int q = (int)(f + (f >= 0.f ? 0.5f : -0.5f));
      d[i] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
    for (int i = Dh; i < stride; i++) d[i] = 0;   /* pad contributes zero */
  }
  *x_scale = xmax / 127.f;
}

static inline void matvec_q8_range(const QT *t, const int8_t *xq, float x_scale,
                            float *y, int row_begin, int row_end) {
  for (int r = row_begin; r < row_end; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    float acc = 0.f;
    for (int gi = 0; gi < t->n_groups; gi++) {
      int begin = gi * t->group, end = begin + t->group;
      if (end > t->cols) end = t->cols;
      int32_t g = 0;                       // group accumulator
      for (int j = begin; j < end; j++) {
        uint8_t byte = row[j >> 1];
        int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
        g += (code - 8) * (int)xq[j];
      }
      acc += (float)g * half2float(sc[gi]);
    }
    y[r] = acc * x_scale;
  }
}

/* Scalar int8 dot product. */
static inline int32_t llm_dot_i8(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  for (int i = 0; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

/* LOCAL ADDITION (ESP32-Gemma3-270M): the same dot product on the ESP32-S3 PIE
 * unit. EE.VMULAS.S8.ACCX consumes 16 int8 lanes per 128-bit Q register per
 * issue against one multiply-add per issue above. Measured on this board at
 * 56.8 Mops/s vs 24.8 scalar (2.29x) with the weights PSRAM-resident.
 *
 * Requires n % 16 == 0 and both pointers 16-byte aligned; EE.VLD.128.IP faults
 * otherwise. The caller checks and falls back rather than this routine handling
 * a tail: every tensor that matters is already a multiple of 16 (d_model 96,
 * ple_dim 128), and a tail path would add a branch to the inner loop for the
 * one tensor that cannot use SIMD anyway (ffn_hidden 66).
 *
 * ACCX is 40-bit and the low word suffices: n=96 of int8 x int8 peaks at
 * 96*127*127 = 1,548,384, well inside int32.
 *
 * Two lessons already paid for in student_simd_bench and preserved here: q
 * registers must NOT appear in the clobber list, and the accumulator is read
 * with rur.accx_0, never ee.srs.accx. */
#if defined(LLM_SIMD_S8) && defined(__XTENSA__)
#define LLM_HAVE_SIMD_S8 1
static inline int32_t llm_dot_i8_simd(const int8_t *a, const int8_t *b, int n) {
  asm volatile("ee.zero.accx");
  for (int i = 0; i < n; i += 16) {
    asm volatile("ee.vld.128.ip  q0, %0, 16  \n"
                 "ee.vld.128.ip  q1, %1, 16  \n"
                 "ee.vmulas.s8.accx q0, q1   \n"
                 : "+r"(a), "+r"(b) :: "memory");
  }
  uint32_t lo;
  asm volatile("rur.accx_0 %0" : "=r"(lo));
  return (int32_t)lo;
}
#define LLM_SIMD_OK(w, x, n) \
  ((((n) & 15) == 0) && ((((uintptr_t)(w)) | ((uintptr_t)(x))) & 15) == 0)
#endif

/* Same arithmetic as matvec_q8_range, reading pre-unpacked int8 weights.
 *
 * Keep out of line on ESP32-S3: with Arduino ESP32 3.3.10 at -O3, inlining
 * regressed inference from 95.0 to 155.2 ms/token. */
#if defined(__GNUC__)
__attribute__((noinline))
#endif
static void matvec_i8_range(const QT *t, const int8_t *xq, float x_scale,
                            float *y, int row_begin, int row_end) {
  int g = t->group, ng = t->n_groups, cols = t->cols;
  for (int r = row_begin; r < row_end; r++) {
    const int8_t *w = t->w8 + (size_t)r * cols;
    const float *sc = t->scale8 + (size_t)r * ng;
    float acc = 0.f;
    for (int gi = 0; gi < ng; gi++) {
      int begin = gi * g, end = begin + g;
      if (end > cols) end = cols;
      int32_t d;
#ifdef LLM_HAVE_SIMD_S8
      /* Checked per group rather than hoisted: the last group of a tensor whose
       * cols is not a multiple of the group size is short, so the answer can
       * differ between groups of the same tensor. */
      if (LLM_SIMD_OK(w + begin, xq + begin, end - begin))
        d = llm_dot_i8_simd(w + begin, xq + begin, end - begin);
      else
#endif
        d = llm_dot_i8(w + begin, xq + begin, end - begin);
      acc += (float)d * sc[gi];
    }
    y[r] = acc * x_scale;
  }
}

static inline void matvec_q8(const QT *t, const float *x, float *y) {
  /* 16-byte aligned so the SIMD path's operand check can pass; the vector load
   * faults on a misaligned address. */
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  if (t->w8 != NULL) { matvec_i8_range(t, xq, xs, y, 0, t->rows); return; }
  matvec_q8_range(t, xq, xs, y, 0, t->rows);
}

// llm_forward dispatches through MATVEC so one flag flips the whole model.
#ifdef LLM_INT8_ACT
#define MATVEC matvec_q8
#else
#define MATVEC matvec_q
#endif

// w is bound straight into the model image, where fp32 tensors follow
// byte-packed quantized ones and so need not be 4-byte aligned. x and out are
// caller scratch and always are.
static inline void rmsnorm(const float *x, const float *w, int n, float *out) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  float inv = 1.f / sqrtf(ss / n + RMS_EPS);
  if (((uintptr_t)w & (sizeof(float) - 1)) == 0) {
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
    return;
  }
  const uint8_t *wb = (const uint8_t *)w;
  for (int i = 0; i < n; i++) {
    float wi;
    memcpy(&wi, wb + (size_t)i * sizeof(float), sizeof wi);
    out[i] = wi * x[i] * inv;
  }
}
/* exp() for softmax weights, where the argument is always (score - max) <= 0.
 *
 * libm expf costs roughly 200 cycles and attention calls it n_heads*pos*L times
 * -- about 2,520 per token at position 104, ~2 ms. Softmax weights are then
 * divided by their own sum, so a relative error of ~1e-3 cancels almost exactly
 * and cannot move an argmax that is not already a coin flip.
 *
 * Computes 2^(x*log2e) by splitting into integer and fractional parts, building
 * the integer part directly in the IEEE-754 exponent field and approximating
 * the fractional part with a degree-3 minimax polynomial on [0,1).
 * Saturates to 0 below -87, where the float result underflows anyway. */
static inline float llm_fast_expf(float x) {
  if (x < -87.f) return 0.f;
  float y = x * 1.44269504088896f;            /* log2(e) */
  /* floorf is a libm CALL on this toolchain, and this function runs H*pos times
   * per layer inside the softmax pass. Two other libm calls (lrintf, twice) were
   * worth 1.4 ms/token today, so the same substitution is applied here.
   *
   * Exact for the range that reaches this point: x >= -87 is guaranteed by the
   * guard above and x <= 0 by the caller (scores minus their max), so
   * y in [-126, 0] and the int conversion cannot overflow. Truncation equals
   * floor for non-negative y and is one too high for negative non-integers,
   * which the correction restores. */
  float fi = (float)(int)y;
  if (fi > y) fi -= 1.f;
  float f = y - fi;                            /* in [0,1) */
  /* 2^f, degree-3 minimax; max relative error ~1.5e-4 over the interval */
  float p = 1.f + f * (0.69617f + f * (0.22588f + f * 0.07945f));
  union { float f; uint32_t u; } e;
  int32_t ei = (int32_t)fi + 127;
  if (ei <= 0) return 0.f;
  e.u = (uint32_t)ei << 23;                    /* 2^floor(y) */
  return p * e.f;
}

/* FASTACT: the same libm audit that found lrintf (EXP-130), applied to the two
 * activations. Both were calling into libm per element while llm_fast_expf sat
 * unused directly above them:
 *
 *   silu  F=66 elements x 6 layers  = 396 expf calls per token
 *   gelu  P=128 elements x 6 layers = 768 ERFF calls per token, and erff is far
 *         more expensive than expf. The PLE stage measures 1.5 ms.
 *
 * Unlike the lrintf change, this one alters real numerics: llm_fast_expf carries
 * ~1.5e-4 relative error, and gelu additionally moves from the exact erf form to
 * the tanh form. That is a quality change and is gated on held-out CE, not on
 * the 200-token digest.
 */
#ifndef FASTACT
#define FASTACT 0
#endif

#if FASTACT
static inline float llm_fast_tanhf(float z) {
  /* tanh(z) = 1 - 2/(e^{2z}+1); the fast exp's relative error passes through
   * roughly halved because the result is bounded by 1. */
  float e = llm_fast_expf(2.f * z);
  return 1.f - 2.f / (e + 1.f);
}
static inline float gelu(float x) {
  /* tanh form, the same approximation trained models normally use under the
   * name gelu_pytorch_tanh. Max absolute deviation from the erf form ~1e-3. */
  float t = 0.7978845608f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.f + llm_fast_tanhf(t));
}
static inline float silu(float x) { return x / (1.f + llm_fast_expf(-x)); }
#else
static inline float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.f + expf(-x)); }
#endif

// Parse header + bind all tensors. Returns 0 on ok, -1 on bad magic.
static int llm_load(const uint8_t *base, Model *m) {
  const uint8_t *p = base;
  uint32_t h[4]; memcpy(h, p, 16); p += 16;   /* magic, version, header_bytes, flags */
  if (h[0] != LLM_MAGIC) return -1;
  /* Version 0 is not a version. A newer one may have changed tensor order, so
   * refuse rather than mis-bind. An unknown flag bit means the writer asked for
   * behaviour this reader does not implement, which is equally unsafe. */
  if (h[1] == 0 || h[1] > LLM_FORMAT_VERSION) return -3;
  if (h[2] < LLM_HEADER_BYTES || h[2] > LLM_HEADER_MAX) return -2;
  if (h[3] & ~(uint32_t)LLM_FLAGS_KNOWN) return -3;
  uint32_t flags = h[3];
  uint32_t vio[2]; memcpy(vio, p, 8); p += 8; /* input_vocab, output_vocab */
  m->c.vocab = (int)vio[0];
  m->out_vocab = (int)vio[1];
  int32_t hv[7]; memcpy(hv, p, 28); p += 28;
  m->c.dim = hv[0]; m->c.n_layers = hv[1]; m->c.n_heads = hv[2];
  m->c.ffn = hv[3]; m->c.ple_dim = hv[4]; m->c.seq_len = hv[5]; m->c.group = hv[6];
  memcpy(&m->c.rope_theta, p, 4); p += 4;
  /* Skip any fields a later version appended. */
  p = base + h[2];

  /* Every dimension below indexes a fixed array or sizes an allocation, so a
   * malformed header must be rejected before any tensor is bound. */
  if (m->c.vocab <= 0 || m->out_vocab <= 0 ||
      m->c.dim <= 0 || m->c.ffn <= 0 || m->c.ple_dim <= 0 ||
      m->c.seq_len <= 0 || m->c.group <= 0 ||
      m->c.n_layers <= 0 || m->c.n_layers > LLM_MAX_LAYERS ||
      m->c.n_heads <= 0 || m->c.dim % m->c.n_heads != 0 ||
      (m->c.dim / m->c.n_heads) % 2 != 0)
    return -2;
  /* A tied head is a view of the first out_vocab embedding rows, so it cannot
   * be longer than the embedding. */
  if ((flags & LLM_FLAG_TIED_HEAD) && m->out_vocab > m->c.vocab) return -2;
#ifdef LLM_INT8_ACT
  /* quantize_act writes into a fixed LLM_Q8_MAX_INPUT buffer, sized by the
   * widest matvec input: dim, ffn or ple_dim. */
  if (m->c.dim > LLM_Q8_MAX_INPUT || m->c.ffn > LLM_Q8_MAX_INPUT ||
      m->c.ple_dim > LLM_Q8_MAX_INPUT)
    return -2;
#endif

  m->head_matvec = NULL;
  m->par_for = NULL;
  m->layer_matvec = NULL;
  int D = m->c.dim, L = m->c.n_layers, P = m->c.ple_dim, F = m->c.ffn, V = m->c.vocab;

  p = bind_q(p, &m->tok_emb, V, D);
  p = bind_q(p, &m->ple_model_proj, L * P, D);
  p = bind_f(p, &m->ple_proj_norm, P);
  p = bind_q(p, &m->ple_table, V, L * P);
  for (int i = 0; i < L; i++) {
    p = bind_f(p, &m->attn_norm[i], D);
    p = bind_q(p, &m->qkv[i], 3 * D, D);
    p = bind_q(p, &m->attn_proj[i], D, D);
    p = bind_f(p, &m->ffn_norm[i], D);
    p = bind_q(p, &m->gate[i], F, D);
    p = bind_q(p, &m->up[i], F, D);
    p = bind_q(p, &m->down[i], D, F);
    p = bind_q(p, &m->ple_gate[i], P, D);
    p = bind_q(p, &m->ple_proj[i], D, P);
    p = bind_f(p, &m->ple_norm[i], D);
  }
  p = bind_f(p, &m->out_norm, D);
  /* Tied: the head is the FIRST out_vocab rows of the token embedding. The
   * embedding may store more rows than the model can ever emit (padding above
   * the tokenizer size), so tying does not imply equal row counts.
   * Untied: the head is appended as the final tensor. */
  if (flags & LLM_FLAG_TIED_HEAD) {
    m->out_head = m->tok_emb;
    m->out_head.rows = m->out_vocab;
  } else {
    p = bind_q(p, &m->out_head, m->out_vocab, D);
  }
  m->image_bytes = (size_t)(p - base);
  return 0;
}

/* ---- int8 staging -------------------------------------------------------
 * Costs 2x the bytes of the int4 form. The caller supplies the buffer, so
 * placement is the platform's choice. */
/* Byte offset of the float scale array in a staged buffer. rows*cols need not
 * be a multiple of 4, so the offset is rounded up to keep the float* aligned. */
static inline size_t llm_stage_scale_offset(const QT *t) {
  size_t w_bytes = (size_t)t->rows * t->cols * sizeof(int8_t);
  return (w_bytes + sizeof(float) - 1) & ~(size_t)(sizeof(float) - 1);
}

static inline size_t llm_stage_int8_bytes(const QT *t) {
  return llm_stage_scale_offset(t)
       + (size_t)t->rows * t->n_groups * sizeof(float);
}

static inline void llm_stage_int8(QT *t, void *buffer) {
  int8_t *w = (int8_t *)buffer;
  float *sc = (float *)((uint8_t *)buffer + llm_stage_scale_offset(t));
  for (int r = 0; r < t->rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = w + (size_t)r * t->cols;
    for (int j = 0; j < t->cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    for (int gi = 0; gi < t->n_groups; gi++)
      sc[(size_t)r * t->n_groups + gi] =
          half2float(t->scales[(size_t)r * t->n_groups + gi]);
  }
  t->w8 = w;
  t->scale8 = sc;
}

/* Stage every per-position tensor, one allocation per tensor: a multi-megabyte
 * contiguous request fails on a fragmented heap even when total free is larger.
 * Returns the number staged; a tensor whose allocation fails keeps the int4
 * path. Callers requiring a guaranteed placement compare this against
 * llm_core_stage_count(). */
static inline int llm_stage_core_int8_alloc(Model *m, void *(*alloc)(size_t)) {
  int staged = 0;
  QT *tensors[7];
  void *buf = alloc(llm_stage_int8_bytes(&m->ple_model_proj));
  if (buf) { llm_stage_int8(&m->ple_model_proj, buf); ++staged; }
  for (int l = 0; l < m->c.n_layers; l++) {
    tensors[0] = &m->qkv[l];       tensors[1] = &m->attn_proj[l];
    tensors[2] = &m->gate[l];      tensors[3] = &m->up[l];
    tensors[4] = &m->down[l];      tensors[5] = &m->ple_gate[l];
    tensors[6] = &m->ple_proj[l];
    for (int i = 0; i < 7; i++) {
      void *b = alloc(llm_stage_int8_bytes(tensors[i]));
      if (b) { llm_stage_int8(tensors[i], b); ++staged; }
    }
  }
  return staged;
}

/* Number of tensors llm_stage_core_int8_alloc attempts. */
static inline int llm_core_stage_count(const Model *m) {
  return 1 + 7 * m->c.n_layers;
}

// Scratch buffers, caller-allocated (host: malloc; device: PSRAM).
typedef struct {
  float *x, *h, *qkv, *att, *g1, *g2, *ple, *tmpP, *trow, *logits;
  // fp32 path: [seq_len], reused by each head. LLM_KV_INT8 path scores all
  // heads in one traversal, so it needs [n_heads * seq_len] live at once.
  float *scores;
  float *kcache, *vcache; // [L * seq_len * D]
  int8_t *kcache8, *vcache8; // [L * seq_len * D], LLM_KV_INT8
  float *kscale, *vscale;    // [L * seq_len],     LLM_KV_INT8
  int8_t *wq;                // [n_heads * seq_len] quantized softmax weights
  float *wscale;             // [n_heads]
  /* Attention working buffers that used to be llm_forward locals.
   *
   * -fstack-usage measured llm_forward's frame at 21,312 bytes, because
   * `int32_t bank[2][LLM_MAX_HEADS*64]` (16 KB) and `int8_t qq[
   * LLM_Q8_MAX_INPUT]` (4 KB) were sized by compile-time maxima -- 32 heads of
   * 64 -- rather than by the model actually loaded. This model is H=4, Dh=24,
   * D=96, so it needed 768 and 96 bytes and reserved 20,480.
   *
   * The frame did not fit either task it ran in: ESP-IDF's main task had 16,384
   * bytes and Arduino's loop task has 8,192. Sizing these from the real config
   * and putting them where the other per-token buffers already live removes
   * ~20 KB from the frame and returns that internal SRAM to weight staging,
   * which cycle 18 measured is worth roughly 0.8% per 32 KB.
   *
   * acc is two contiguous banks of [H*Dh] -- one per half of the dual-core V
   * split, summed afterwards. Internal SRAM: touched every position. */
  int32_t *acc;              // [2 * n_heads * head_dim]
  int8_t *qq;                // [dim] quantized query, 16-byte aligned
#ifdef LLM_PROFILE
  struct {
    uint64_t input_us, attn_us, ffn_us, ple_us, head_us;
    /* Attention sub-passes. Three consecutive attention estimates in this
     * project were too optimistic because each costed the loop about to be
     * changed and ignored the rest; the byte model predicts memory well and
     * predicts nothing about compute, and attention stopped being a memory
     * structure once the KV cache went int8. Split it before optimising it. */
    uint64_t attn_proj_us;   /* QKV + output projections (weights, memory)   */
    uint64_t attn_score_us;  /* pass 1: q.k over the K cache                 */
    uint64_t attn_soft_us;   /* pass 2a: expf, scale fold, weight quantize   */
    uint64_t attn_vacc_us;   /* pass 2b: integer V accumulation              */
    /* The attention OUTPUT projection. Split out from attn_proj_us, which
     * despite its comment only ever covered the input-side rmsnorm + QKV: the
     * output matvec at the bottom of the layer was inside no counter at all and
     * was therefore the bulk of the 1.12 ms "other" the profile print reported.
     * An unattributed term that large is not a rounding residue, and this
     * project has twice optimised a stage it had not finished measuring. */
    uint64_t attn_outp_us;
    /* RoPE, the k/v cache quantize-and-store, the q quantize, and the per-head
     * accumulator init -- everything between the QKV projection and the score
     * pass. Split out after outp_us accounted for only 0.45 of the 1.12 ms
     * residual, i.e. the guess was half wrong and the rest had to be measured
     * rather than reasoned about. */
    uint64_t attn_prep_us;
    /* prep, split to discriminate its two candidate causes: partial-cache-line
     * PSRAM writes into the KV cache, versus scalar quantize_act. rope_us is
     * pure SRAM arithmetic; kvq_us is the quantize-and-store that touches
     * PSRAM. If kvq dominates, the fix is staging the row and writing one
     * aligned burst; if they are comparable, the fix is a PIE quantize kernel. */
    uint64_t attn_rope_us, attn_kvq_us;
    uint32_t calls;
  } profile;
#endif
} Scratch;

#ifdef LLM_PROFILE
static void llm_profile_reset(Scratch *s) {
  memset(&s->profile, 0, sizeof(s->profile));
}
#endif

// One decode step: token at position pos -> logits[V]. KV cache persists across calls.
/* Per-layer matvec dispatch: platform hook if set, otherwise MATVEC. */
#define LLM_LMV(m, t, x, y) \
  do { if ((m)->layer_matvec) (m)->layer_matvec((t), (x), (y)); \
       else MATVEC((t), (x), (y)); } while (0)

/* Attention pass 1 (q.k over the K cache), as a range so it can be split.
 *
 * Measured at 3.01 ms of attention's 8.76 -- the largest of its four terms, and
 * not the one this project had planned to optimise. Positions are independent:
 * each scores a disjoint slice of s->scores and keeps its own per-head max, so
 * the only reduction is a max over two slots afterwards. Slot is keyed off
 * `begin` because the callee does not know which core it landed on. */
typedef struct {
  const int8_t *kc, *qq;
  const float *kcs;
  float *scores, (*maxs)[LLM_MAX_HEADS];   /* [2][H], one row per half */
  float scale, q_scale;
  int D, H, Dh, S;
  int khs;        /* per-head stride in the K cache; == Dh when KPAD=0 */
  size_t krow;    /* K bytes per position; == D when KPAD=0 */
} ScoreJob;

static void llm_score_range(void *v, int begin, int end) {
  ScoreJob *j = (ScoreJob *)v;
  const int D = j->D, H = j->H, Dh = j->Dh, S = j->S;
  float *mx = j->maxs[begin == 0 ? 0 : 1];
  for (int hh = 0; hh < H; hh++) mx[hh] = -1e30f;
  const int khs = j->khs;
  for (int t = begin; t < end; t++) {
    const int8_t *kt = j->kc + (size_t)t * j->krow;
    float ks = j->kcs[t] * j->scale * j->q_scale;
    for (int hh = 0; hh < H; hh++) {
      /* With KPAD both slices start on a 16-byte boundary, which is what the
       * PIE load requires and what the unpadded Dh=24 layout could never give. */
      const int8_t *qh = j->qq + hh * khs;
      const int8_t *kh = kt + hh * khs;
      int32_t acc;
#if defined(__XTENSA__) && KPAD
      /* PIE: 16 int8 MACs per instruction, the same primitive the output head
       * already uses. Only reachable with KPAD, which is what puts both operands
       * on a 16-byte boundary -- the unpadded Dh=24 layout never could.
       *
       * The pad bytes are zero, so accumulating over khs instead of Dh adds
       * nothing: BIT-EXACT, verified on host as CE 2.4559 either way.
       *
       * ACCX is 40-bit; the sum is bounded by Dh*127*127 = 387k, so the low word
       * is the whole value. */
      {
        const int8_t *qp = qh, *kp = kh;
        asm volatile("ee.zero.accx");
        for (int b = 0; b < khs / 16; b++) {
          asm volatile(
              "ee.vld.128.ip     q0, %0, 16 \n"
              "ee.vld.128.ip     q1, %1, 16 \n"
              "ee.vmulas.s8.accx q0, q1     \n"
              : "+r"(qp), "+r"(kp) :: "memory");
        }
        uint32_t lo;
        asm volatile("rur.accx_0 %0" : "=r"(lo));
        acc = (int32_t)lo;
      }
#else
      /* DO NOT "optimise" this into multiple accumulators. Measured, EXP-132:
       * a 4-accumulator unroll to break the dependency on `acc` made it WORSE,
       * 1.53 -> 1.89 ms and 67.52 -> 66.30 tok/s, bit-exact but slower. The
       * simple form lets the compiler pipeline it; the unrolled form raised
       * register pressure and blocked that. The 12x gap to the MAC count is real
       * but it is not a dependency stall. */
      acc = 0;
      for (int i = 0; i < Dh; i++) acc += (int32_t)qh[i] * (int32_t)kh[i];
#endif
      float dot = (float)acc * ks;
      j->scores[(size_t)hh * S + t] = dot;
      if (dot > mx[hh]) mx[hh] = dot;
    }
  }
}

/* Attention pass 2b (integer V accumulation), as a splittable range.
 *
 * Each half owns a private accumulator bank and the two are summed afterwards,
 * so the split is exact -- integer addition is associative, unlike the fp32
 * version this replaced. The `!w8` skip stays inside the range, which means the
 * halves are not perfectly balanced once the softmax sharpens; measured to be
 * worth less than the notify round trip it would cost to rebalance. */
typedef struct {
  const int8_t *vc, *wq;
  /* Two contiguous banks of [H*Dh]; bank 1 starts at acc + H*Dh. Was a
   * `[2][LLM_MAX_HEADS*64]` array on llm_forward's stack, which reserved 16 KB
   * for the 768 bytes this model uses. */
  int32_t *acc;
  int D, H, Dh, S;
} VAccJob;

/* Diagnostic arm. Answers ONE question before any kernel work is attempted on
 * vacc: is it at its bandwidth floor, as the head scan turned out to be, or does
 * it have real headroom?
 *
 * The naive reading says headroom. 6 layers x ~104 positions x 96 B = 59,904 B
 * per token in 1.02 ms is 56 MiB/s against a bus measured at 85.3 -- a third
 * unused. But that arithmetic assumes every V row is read, and the `!w8` skip
 * means a row whose weight quantised to zero in EVERY head is never touched at
 * all. If the softmax is sharp, the bytes actually moved are far fewer than
 * 59,904 and the effective rate is far higher -- in which case vacc is already
 * finished and the next kernel cycle belongs somewhere else.
 *
 * Counted, not modelled. The same reasoning applied to the head scan only ended
 * when the scan was measured against a measured bus, and the modelled answer
 * had been wrong before that.
 *
 * Off by default: these are two increments in the innermost loop and would
 * perturb the very number they exist to explain. Separate build, not an
 * instrumented production one. */
#ifndef VACC_STATS
#define VACC_STATS 0
#endif
#if VACC_STATS
/* PER-SLOT, then summed by the reader.
 *
 * The first version used four plain globals incremented from both cores.
 * `pairs_seen` came back at 1,291 per token against an expected 2,496 -- almost
 * exactly half -- because a non-atomic read-modify-write from two cores loses
 * roughly one update in two. The RATIO survived that (both counters race
 * identically) but every absolute byte figure derived from it was wrong, and it
 * produced a "15.7 MiB/s effective" that was pure artefact. Retracted.
 *
 * vacc splits by POSITION and llm_vacc_range already derives its private
 * accumulator bank from `begin == 0`. Reuse exactly that discriminator: each
 * core owns a slot, nothing is shared, the sum afterwards is exact. */
static uint64_t llm_vacc_pairs_live[2];  /* (position,head) pairs with w8 != 0 */
static uint64_t llm_vacc_pairs_seen[2];  /* (position,head) pairs visited */
static uint64_t llm_vacc_rows_dead[2];   /* positions where ALL heads skipped */
static uint64_t llm_vacc_rows_seen[2];   /* positions visited */
#endif

static void llm_vacc_range(void *v, int begin, int end) {
  VAccJob *j = (VAccJob *)v;
  const int D = j->D, H = j->H, Dh = j->Dh, S = j->S;
  const int slot = (begin == 0) ? 0 : 1;
  int32_t *bank = j->acc + (slot ? H * Dh : 0);
  for (int i = 0; i < H * Dh; i++) bank[i] = 0;
  for (int t = begin; t < end; t++) {
    const int8_t *vt = j->vc + (size_t)t * D;
#if VACC_STATS
    int live_here = 0;
    llm_vacc_rows_seen[slot]++;
#endif
    for (int hh = 0; hh < H; hh++) {
      int32_t w8 = j->wq[(size_t)hh * S + t];
#if VACC_STATS
      llm_vacc_pairs_seen[slot]++;
      if (w8) { llm_vacc_pairs_live[slot]++; live_here = 1; }
#endif
      if (!w8) continue;
      int32_t *ah = bank + hh * Dh;
      const int8_t *vh = vt + hh * Dh;
      for (int i = 0; i < Dh; i++) ah[i] += w8 * (int32_t)vh[i];
    }
#if VACC_STATS
    if (!live_here) llm_vacc_rows_dead[slot]++;
#endif
  }
}

/* Attention pass 2a (softmax, V-scale fold, weight quantisation) as a range
 * over HEADS. Each head is fully independent -- its own denominator, its own
 * max, its own quantisation scale, its own row of wq -- so a head split shares
 * no state and reorders no float reduction. */
typedef struct {
  float *scores;
  int8_t *wq;
  const float *vcs;
  float *maxs, *denom, *wscale;
  int S, pos;
} SoftJob;

static void llm_soft_range(void *v, int begin, int end) {
  SoftJob *j = (SoftJob *)v;
  const int S = j->S, pos = j->pos;
  for (int hh = begin; hh < end; hh++) {
    float *sc = j->scores + (size_t)hh * S;
    float wmax = 0.f, den = 0.f;
    for (int t = 0; t <= pos; t++) {
      float w = llm_fast_expf(sc[t] - j->maxs[hh]);
      den += w;
      float we = w * j->vcs[t];
      sc[t] = we;
      float a = we < 0.f ? -we : we;
      if (a > wmax) wmax = a;
    }
    j->denom[hh] += den;
    float ws = wmax > 0.f ? wmax / 127.f : 1.f;
    j->wscale[hh] = ws;
    float inv = 1.f / ws;
    int8_t *wq = j->wq + (size_t)hh * S;
    for (int t = 0; t <= pos; t++) {
      /* Same lrintf trap as quantize_act (EXP-130), in a second function. Found by
     * auditing for the call rather than by profiling, since the fix there was
     * worth 1.27 ms/token and nothing guaranteed it appeared only once. This
     * site runs n_heads * pos times per layer. */
    float fq = sc[t] * inv;
    int q = (int)(fq + (fq >= 0.f ? 0.5f : -0.5f));
      wq[t] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
    }
  }
}

static void llm_forward(Model *m, int token, int pos, Scratch *s) {
  int D = m->c.dim, L = m->c.n_layers, P = m->c.ple_dim, F = m->c.ffn;
  int H = m->c.n_heads, Dh = D / H, S = m->c.seq_len;
#ifdef LLM_PROFILE
  uint64_t profile_t0 = (uint64_t)LLM_PROFILE_NOW();
#endif

  deq_row(&m->tok_emb, token, s->x);           // embedding

  // ---- per-layer input: (RMSNorm(proj(x)/sqrt(D)) + table[tok]*sqrt(P)) / sqrt(2)
  LLM_LMV(m, &m->ple_model_proj, s->x, s->tmpP); // [L*P]
  float dscale = 1.f / sqrtf((float)D);
  for (int i = 0; i < L * P; i++) s->tmpP[i] *= dscale;
  for (int l = 0; l < L; l++)
    rmsnorm(s->tmpP + l * P, m->ple_proj_norm, P, s->tmpP + l * P);
  deq_row(&m->ple_table, token, s->trow);      // [L*P]
  float sp = sqrtf((float)P), inv2 = 0.70710678f;
  for (int i = 0; i < L * P; i++)
    s->ple[i] = (s->tmpP[i] + s->trow[i] * sp) * inv2;
  LLM_TRACE_EVENT(0, -1, s->x, D);

  // RoPE frequencies are identical across every head and layer at a position.
  // Reuse trow (dead after constructing s->ple) instead of recomputing the same
  // pow/cos/sin values L*H times.
  float *rope_c = s->trow, *rope_s = s->trow + Dh / 2;
  for (int i = 0; i < Dh / 2; i++) {
    float freq = powf(m->c.rope_theta, -2.f * i / Dh);
    rope_c[i] = cosf(pos * freq);
    rope_s[i] = sinf(pos * freq);
  }
#ifdef LLM_PROFILE
  uint64_t profile_t1 = (uint64_t)LLM_PROFILE_NOW();
  s->profile.input_us += profile_t1 - profile_t0;
#endif

  for (int l = 0; l < L; l++) {
    // ---- attention
    /* The QKV projection is the last unattributed term in attention. With the
     * head down to 6.6 ms, attention's 2.0 ms residual is 12% of the whole
     * token -- the largest unexplained quantity in the profile -- and this
     * project has already had one cycle aim at attention's smallest term
     * because it costed the loop it planned to change instead of measuring. */
#ifdef LLM_PROFILE
    uint64_t pp0 = (uint64_t)LLM_PROFILE_NOW();
#endif
    rmsnorm(s->x, m->attn_norm[l], D, s->h);
    LLM_LMV(m, &m->qkv[l], s->h, s->qkv);        // [3D]
#ifdef LLM_PROFILE
    s->profile.attn_proj_us += (uint64_t)LLM_PROFILE_NOW() - pp0;
#endif
    float *q = s->qkv, *k = s->qkv + D, *v = s->qkv + 2 * D;
#ifdef LLM_PROFILE
    uint64_t pq0 = (uint64_t)LLM_PROFILE_NOW();
#endif
    // split-half RoPE at position pos, per head
    for (int hh = 0; hh < H; hh++) {
      float *qh = q + hh * Dh, *kh = k + hh * Dh;
      for (int i = 0; i < Dh / 2; i++) {
        float c = rope_c[i], sn = rope_s[i];
        float q1 = qh[i], q2 = qh[i + Dh / 2];
        qh[i] = q1 * c - q2 * sn; qh[i + Dh / 2] = q2 * c + q1 * sn;
        float k1 = kh[i], k2 = kh[i + Dh / 2];
        kh[i] = k1 * c - k2 * sn; kh[i + Dh / 2] = k2 * c + k1 * sn;
      }
    }
    float scale = 1.f / sqrtf((float)Dh);
#ifdef LLM_PROFILE
    uint64_t pr1 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.attn_rope_us += pr1 - pq0;
#endif
#ifdef LLM_KV_INT8
    /* int8 KV cache, head-major traversal.
     *
     * Two independent defects in the fp32 version, both bandwidth:
     *
     * 1. PRECISION. k and v are cached fp32, so each position costs
     *    n_layers * D * 4 * 2 bytes. Measured slope was 0.132 ms/position,
     *    i.e. 4,608 B/position. int8 quarters that.
     *
     * 2. TRAVERSAL. The fp32 loop is head-outer/position-inner, so it walks
     *    the whole cache once per head per pass -- 2*H = 8 traversals, each
     *    reading Dh*4 = 96 contiguous bytes out of every D*4 = 384. Strided,
     *    so the octal bus pays an address phase per 96-byte chunk and
     *    delivers 34.9 MB/s against the 60.7 MB/s it reaches when streaming.
     *    Here position is the OUTER loop and all H heads are served from each
     *    row while it is in registers, so the cache is walked exactly twice,
     *    sequentially.
     *
     * Not bit-identical -- this is a quantization, unlike the int4 head where
     * int4 was already the stored precision. Scales are per (layer, position)
     * row, which is the finest granularity that costs nothing to store.
     */
    /* K uses the padded row layout (llm_krow); V does not -- only the score pass
     * needs aligned per-head slices, and padding V would cost the same 33% for
     * nothing. */
    const int khs = llm_khs(&m->c);
    const size_t krow = llm_krow(&m->c);
    int8_t *kc = s->kcache8 + (size_t)l * S * krow;
    int8_t *vc = s->vcache8 + (size_t)l * S * D;
    float *kcs = s->kscale + (size_t)l * S, *vcs = s->vscale + (size_t)l * S;
    quantize_act_kpad(k, H, Dh, khs, kc + (size_t)pos * krow, &kcs[pos]);
    quantize_act(v, D, vc + (size_t)pos * D, &vcs[pos]);
#ifdef LLM_PROFILE
    s->profile.attn_kvq_us += (uint64_t)LLM_PROFILE_NOW() - pr1;
#endif

    float maxs[LLM_MAX_HEADS], denom[LLM_MAX_HEADS];
    for (int hh = 0; hh < H; hh++) { maxs[hh] = -1e30f; denom[hh] = 0.f; }
    for (int i = 0; i < D; i++) s->att[i] = 0.f;

    /* Quantizing q makes the score an INTEGER dot product. Cutting the KV cache
     * to int8 pushed attention out of the bandwidth-bound regime and into a
     * compute-bound one (measured: 7.9 ms of arithmetic behind 6.3 ms of
     * memory), and most of that arithmetic was converting each int8 back to
     * float to multiply it by an fp32 q. Keeping both sides integer removes
     * 96*pos*L conversions per token and replaces the fp32 FMA with an int MAC,
     * which this core does faster. The result is rescaled once per (head,
     * position) instead of per element. */
    /* [llm_krow] -- padded to match the K layout so both operands of the score
     * dot start on a 16-byte boundary. Was [D]; every allocation site was
     * updated in the same change. */
    int8_t *qq = s->qq;
    float q_scale;
    quantize_act_kpad(q, H, Dh, khs, qq, &q_scale);
#ifdef LLM_PROFILE
    s->profile.attn_prep_us += (uint64_t)LLM_PROFILE_NOW() - pq0;
#endif

#ifdef LLM_PROFILE
    uint64_t pa0 = (uint64_t)LLM_PROFILE_NOW();
#endif
    /* pass 1: one sequential walk, all heads scored per row */
    {
      float hmax[2][LLM_MAX_HEADS];
      ScoreJob sj = {kc, qq, kcs, s->scores, hmax, scale, q_scale,
                     D, H, Dh, S, khs, krow};
      /* Split only when there is enough work to cover the notify round trip.
       * At short contexts the second core costs more than it saves, which is
       * the same threshold logic the matvec split already uses. */
      if (m->par_for && pos >= 32) {
        m->par_for(llm_score_range, &sj, pos + 1);
        for (int hh = 0; hh < H; hh++)
          maxs[hh] = hmax[0][hh] > hmax[1][hh] ? hmax[0][hh] : hmax[1][hh];
      } else {
        llm_score_range(&sj, 0, pos + 1);
        for (int hh = 0; hh < H; hh++) maxs[hh] = hmax[0][hh];
      }
    }
    /* pass 2a: softmax weights, folded with each row's V scale, quantized to
     * int8 so the accumulation below is pure integer.
     *
     * The V accumulation was the last fp32 inner loop in the model and, after
     * the KV cache went int8, the largest single term in attention: D*pos*L
     * int8->float conversions plus an fp32 FMA each, ~60,000 per token. Folding
     * vs into w first means one scalar multiplies the whole row, and quantizing
     * that scalar makes the row update int8 x int8 -> int32.
     *
     * Weights are non-negative and sum to 1 before normalization, so a single
     * per-head scale over all positions is well conditioned: the largest weight
     * sets the scale and the ones that quantize to zero are those already
     * contributing far below the output's own int8 resolution. Cost measured on
     * held-out data rather than assumed -- see the ppl harness. */
#ifdef LLM_PROFILE
    uint64_t pa1 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.attn_score_us += pa1 - pa0;
#endif
    {
      SoftJob fj = {s->scores, s->wq, vcs, maxs, denom, s->wscale, S, pos};
      /* Split by HEAD, not by position.
       *
       * An earlier cycle rejected parallelising this pass on the grounds that
       * the `denom` reduction is a non-associative float sum, so splitting it
       * would forfeit bit-identity. That is true of a POSITION split, which was
       * the decomposition considered. It is not true of a head split: every
       * head owns its own denom, wmax, wscale and wq row, and touches no other
       * head's state. There is no reduction to reorder, so the result is
       * identical by construction rather than by luck.
       *
       * With softmax now the largest single term in the whole model (1.93 ms of
       * a 16.8 ms token, ahead of even the certified head) the decomposition was
       * worth revisiting rather than inheriting. H is 4, so the halves are
       * exactly balanced. */
      if (m->par_for && pos >= 32) m->par_for(llm_soft_range, &fj, H);
      else llm_soft_range(&fj, 0, H);
    }
#ifdef LLM_PROFILE
    uint64_t pa2 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.attn_soft_us += pa2 - pa1;
#endif
    /* pass 2b: one sequential walk of V, integer accumulate */
    {
      int32_t *acc = s->acc;             /* bank 0 */
      int32_t *bank1 = s->acc + H * Dh;  /* bank 1, the worker's half */
      VAccJob vj = {vc, s->wq, s->acc, D, H, Dh, S};
      if (m->par_for && pos >= 32) {
        m->par_for(llm_vacc_range, &vj, pos + 1);
        for (int i = 0; i < H * Dh; i++) acc[i] += bank1[i];
      } else {
        llm_vacc_range(&vj, 0, pos + 1);
      }
      for (int hh = 0; hh < H; hh++) {
        float k = s->wscale[hh] / denom[hh];
        float *ao = s->att + hh * Dh;
        const int32_t *ah = acc + hh * Dh;
        for (int i = 0; i < Dh; i++) ao[i] = (float)ah[i] * k;
      }
    }
#ifdef LLM_PROFILE
    s->profile.attn_vacc_us += (uint64_t)LLM_PROFILE_NOW() - pa2;
#endif
#else
    float *kc = s->kcache + (size_t)l * S * D, *vc = s->vcache + (size_t)l * S * D;
    memcpy(kc + (size_t)pos * D, k, D * sizeof(float));
    memcpy(vc + (size_t)pos * D, v, D * sizeof(float));
    // causal attention over 0..pos
    for (int hh = 0; hh < H; hh++) {
      float *qh = q + hh * Dh;
      float *ao = s->att + hh * Dh;
      for (int i = 0; i < Dh; i++) ao[i] = 0.f;
      float maxs = -1e30f;
      // pass 1: max for stable softmax
      for (int t = 0; t <= pos; t++) {
        float *kt = kc + (size_t)t * D + hh * Dh, dot = 0.f;
        for (int i = 0; i < Dh; i++) dot += qh[i] * kt[i];
        dot *= scale;
        s->scores[t] = dot;
        if (dot > maxs) maxs = dot;
      }
      float denom = 0.f;
      for (int t = 0; t <= pos; t++) {
        float w = expf(s->scores[t] - maxs); denom += w;
        float *vt = vc + (size_t)t * D + hh * Dh;
        for (int i = 0; i < Dh; i++) ao[i] += w * vt[i];
      }
      for (int i = 0; i < Dh; i++) ao[i] /= denom;
    }
#endif
#ifdef LLM_PROFILE
    uint64_t po0 = (uint64_t)LLM_PROFILE_NOW();
#endif
    LLM_LMV(m, &m->attn_proj[l], s->att, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
#ifdef LLM_PROFILE
    s->profile.attn_outp_us += (uint64_t)LLM_PROFILE_NOW() - po0;
#endif
    LLM_TRACE_EVENT(1, l, s->x, D);
#ifdef LLM_PROFILE
    uint64_t profile_t2 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.attn_us += profile_t2 - profile_t1;
#endif

    // ---- SwiGLU FFN
    rmsnorm(s->x, m->ffn_norm[l], D, s->h);
    LLM_LMV(m, &m->gate[l], s->h, s->g1);
    LLM_LMV(m, &m->up[l], s->h, s->g2);
    for (int i = 0; i < F; i++) s->g1[i] = silu(s->g1[i]) * s->g2[i];
    LLM_LMV(m, &m->down[l], s->g1, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
    LLM_TRACE_EVENT(2, l, s->x, D);
#ifdef LLM_PROFILE
    uint64_t profile_t3 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.ffn_us += profile_t3 - profile_t2;
#endif

    // ---- PLE gate: x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l))
    LLM_LMV(m, &m->ple_gate[l], s->x, s->g2);    // [P]
    for (int i = 0; i < P; i++) s->g2[i] = gelu(s->g2[i]) * s->ple[l * P + i];
    LLM_LMV(m, &m->ple_proj[l], s->g2, s->h);    // [D]
    rmsnorm(s->h, m->ple_norm[l], D, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
    LLM_TRACE_EVENT(3, l, s->x, D);
#ifdef LLM_PROFILE
    profile_t1 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.ple_us += profile_t1 - profile_t3;
#endif
  }

  LLM_TRACE_EVENT(4, -1, s->x, D);
  rmsnorm(s->x, m->out_norm, D, s->x);
  LLM_TRACE_EVENT(5, -1, s->x, D);
  // output head: logits[v] = dot(out_head_row[v], x). out_head is tok_emb when
  // the model ties them, and a separate tensor when it does not.
  if (m->head_matvec) m->head_matvec(&m->out_head, s->x, s->logits);
  else MATVEC(&m->out_head, s->x, s->logits);
#ifdef LLM_PROFILE
  s->profile.head_us += (uint64_t)LLM_PROFILE_NOW() - profile_t1;
  s->profile.calls++;
#endif
}

#endif
