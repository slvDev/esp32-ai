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
  const int8_t *w8;        // rows*stride8, values -7..7; padding bytes are 0
  const float  *scale8;    // rows*n_groups, half2float'd once
  int stride8;             // bytes per staged row: cols rounded up to
                           // LLM_STAGE_ALIGN, so a SIMD kernel can read whole
                           // vectors without a tail case
  /* Optional int4 staging: the packed nibbles copied as they are, with the
   * group scales converted once. Half the bytes of int8 staging, for a
   * kernel that unpacks nibbles in registers. NULL unless staged this way. */
  const uint8_t *w4;       // rows*stride4 packed nibbles, same layout as codes
  int stride4;             // bytes per staged row, = row_bytes
} QT;

/* Staged rows are padded to this many bytes. 1 keeps rows contiguous, which
 * is what the host verifier and the scalar kernel need nothing more than; a
 * platform with 16-byte vector loads defines 16 before including this file
 * and allocates staging buffers with matching alignment. */
#ifndef LLM_STAGE_ALIGN
#define LLM_STAGE_ALIGN 1
#endif
static inline int llm_stage_stride(int cols) {
  return (cols + LLM_STAGE_ALIGN - 1) / LLM_STAGE_ALIGN * LLM_STAGE_ALIGN;
}

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

struct Scratch;
typedef struct Model Model;

struct Model {
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
  /* Attention for one layer at one position over all heads, on the quantized
   * KV path only. NULL runs the heads in sequence on the calling core; a
   * platform can split them across cores with llm_attend_heads and the
   * second scratch set (q8b, w16b, scoresb). */
  void (*attn_heads)(Model *, struct Scratch *, int layer, int pos);
};

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
  t->stride8 = 0;
  t->w4 = NULL; t->stride4 = 0;
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
static inline void quantize_act(const float *x, int n, int8_t *xq, float *x_scale) {
  float xmax = 1e-8f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > xmax) xmax = a; }
  float inv = 127.f / xmax;
  for (int j = 0; j < n; j++) {
    int q = (int)lrintf(x[j] * inv);
    xq[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
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

/* ---- vector dot primitives --------------------------------------------------
 * Dot products over whole 16-byte vectors: nvec*16 int8 pairs, or nvec*8 int16
 * pairs, exact integer sums. These scalar versions are the reference; a
 * platform with a vector unit defines LLM_DOT_S8V / LLM_DOT_S16V to its own
 * before including this file. The int16 sum needs more than 32 bits. */
static inline int32_t llm_dot_s8v_scalar(const int8_t *a, const int8_t *b, int nvec) {
  int32_t acc = 0;
  for (int i = 0; i < nvec * 16; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}
static inline int64_t llm_dot_s16v_scalar(const int16_t *a, const int16_t *b, int nvec) {
  int64_t acc = 0;
  for (int i = 0; i < nvec * 8; i++) acc += (int64_t)a[i] * (int64_t)b[i];
  return acc;
}
#ifndef LLM_DOT_S8V
#define LLM_DOT_S8V llm_dot_s8v_scalar
#endif
#ifndef LLM_DOT_S16V
#define LLM_DOT_S16V llm_dot_s16v_scalar
#endif

/* ---- quantized KV cache ------------------------------------------------------
 * Optional attention path, selected by LLM_KV_QUANT. Keys are int8 with one
 * scale per position and head, each head padded to LLM_KPAD bytes so a key is
 * whole vectors. Values are int16 with one scale per position and head, stored
 * TRANSPOSED: per layer, head and feature, one row over positions. Both
 * attention passes then become the same integer dot products as the matvec:
 *   score_t = q8 . k8_t              (one dot per position)
 *   out_i   = v16t_i . w16           (one dot per feature over positions)
 * The softmax weight of each position is multiplied by that position's value
 * scale before it is quantized to int16, which is what lets the value scale
 * vary per position while the dot product still sees a single scale.
 * Bytes read per token drop about 3x against the fp32 cache and the MACs move
 * to the vector unit. Quality is measured on the host with ppl.c, since the
 * arithmetic here is exactly what the device runs. */
#ifdef LLM_KV_QUANT
#define LLM_KPAD 32     /* bytes per head in the key cache; head_dim <= 32 */
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
  int g = t->group, ng = t->n_groups, cols = t->cols, stride = t->stride8;
  for (int r = row_begin; r < row_end; r++) {
    const int8_t *w = t->w8 + (size_t)r * stride;
    const float *sc = t->scale8 + (size_t)r * ng;
    float acc = 0.f;
    for (int gi = 0; gi < ng; gi++) {
      int begin = gi * g, end = begin + g;
      if (end > cols) end = cols;
      acc += (float)llm_dot_i8(w + begin, xq + begin, end - begin) * sc[gi];
    }
    y[r] = acc * x_scale;
  }
}

static inline void matvec_q8(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
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
static inline float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.f + expf(-x)); }

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
#ifdef LLM_KV_QUANT
  if (m->c.dim / m->c.n_heads > LLM_KPAD) return -2;
#endif
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
  m->layer_matvec = NULL;
  m->attn_heads = NULL;
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
  size_t w_bytes = (size_t)t->rows * llm_stage_stride(t->cols) * sizeof(int8_t);
  return (w_bytes + sizeof(float) - 1) & ~(size_t)(sizeof(float) - 1);
}

static inline size_t llm_stage_int8_bytes(const QT *t) {
  return llm_stage_scale_offset(t)
       + (size_t)t->rows * t->n_groups * sizeof(float);
}

static inline void llm_stage_int8(QT *t, void *buffer) {
  int8_t *w = (int8_t *)buffer;
  float *sc = (float *)((uint8_t *)buffer + llm_stage_scale_offset(t));
  int stride = llm_stage_stride(t->cols);
  for (int r = 0; r < t->rows; r++) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    int8_t *dst = w + (size_t)r * stride;
    for (int j = 0; j < t->cols; j++) {
      uint8_t byte = row[j >> 1];
      int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
      dst[j] = (int8_t)(code - 8);
    }
    for (int j = t->cols; j < stride; j++) dst[j] = 0;
    for (int gi = 0; gi < t->n_groups; gi++)
      sc[(size_t)r * t->n_groups + gi] =
          half2float(t->scales[(size_t)r * t->n_groups + gi]);
  }
  t->w8 = w;
  t->scale8 = sc;
  t->stride8 = stride;
}

/* ---- int4 staging -----------------------------------------------------
 * A straight copy of the packed codes out of flash into faster memory, plus
 * float scales. The bytes are the same nibbles matvec_q8_range reads, so that
 * function stays the scalar reference for any kernel using this staging. */
static inline size_t llm_stage_int4_bytes(const QT *t) {
  size_t w_bytes = (size_t)t->rows * t->row_bytes;
  return ((w_bytes + sizeof(float) - 1) & ~(size_t)(sizeof(float) - 1))
       + (size_t)t->rows * t->n_groups * sizeof(float);
}
/* The scales alone, so they can be placed apart from the codes (they are
 * read once per row, and are small enough for fast memory). */
static inline size_t llm_stage_int4_scale_bytes(const QT *t) {
  return (size_t)t->rows * t->n_groups * sizeof(float);
}
static inline void llm_stage_int4_split(QT *t, void *codes_buf, void *scales_buf) {
  size_t w_bytes = (size_t)t->rows * t->row_bytes;
  uint8_t *w = (uint8_t *)codes_buf;
  float *sc = (float *)scales_buf;
  memcpy(w, t->codes, w_bytes);
  for (size_t i = 0; i < (size_t)t->rows * t->n_groups; i++)
    sc[i] = half2float(t->scales[i]);
  t->w4 = w;
  t->stride4 = t->row_bytes;
  t->scale8 = sc;
}
static inline void llm_stage_int4(QT *t, void *buffer) {
  size_t w_bytes = (size_t)t->rows * t->row_bytes;
  uint8_t *w = (uint8_t *)buffer;
  float *sc = (float *)((uint8_t *)buffer
                        + ((w_bytes + sizeof(float) - 1) & ~(size_t)(sizeof(float) - 1)));
  memcpy(w, t->codes, w_bytes);
  for (size_t i = 0; i < (size_t)t->rows * t->n_groups; i++)
    sc[i] = half2float(t->scales[i]);
  t->w4 = w;
  t->stride4 = t->row_bytes;
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
typedef struct Scratch {
  float *x, *h, *qkv, *att, *g1, *g2, *ple, *tmpP, *trow, *logits;
  float *scores; // [seq_len], reused by each attention head
  float *kcache, *vcache; // [L * seq_len * D]; unused under LLM_KV_QUANT
#ifdef LLM_KV_QUANT
  int8_t  *k8;    // [L][S][H][LLM_KPAD] int8 keys, zero padded
  float   *ks;    // [L][S][H] key scales
  int16_t *v16t;  // [L][H][Dh][S] int16 values, one row per feature
  float   *vs;    // [L][S][H] value scales
  int8_t  *q8;    // [LLM_KPAD] this head's query, 16-byte aligned
  int16_t *w16;   // [S] this head's softmax weights, 16-byte aligned
  int8_t  *q8b;   // second set of the three, for a second core
  int16_t *w16b;
  float   *scoresb; // [S]
#endif
#ifdef LLM_PROFILE
  struct {
    uint64_t input_us, attn_us, ffn_us, ple_us, head_us;
    uint32_t calls;
  } profile;
#endif
} Scratch;

#ifdef LLM_PROFILE
static void llm_profile_reset(Scratch *s) {
  memset(&s->profile, 0, sizeof(s->profile));
}
#endif

#ifdef LLM_KV_QUANT
/* The quantized cache is three placements, because they are read at very
 * different rates and a device can afford fast memory for the small ones:
 *   key   the int8 keys, one 32-byte row per position and head: read for
 *         every position at every step, the hottest part of the cache;
 *   hot   the per-head temporaries (two sets, for two cores);
 *   main  the transposed int16 values and both scale arrays.
 * llm_kv_quant_bind takes a buffer for each; a NULL key or hot buffer is
 * carved out of main instead, in which case main must be sized with
 * llm_kv_quant_bytes (the sum of all three). Sizes include alignment slack. */
static inline size_t llm_kv_key_bytes(const Model *m) {
  return (size_t)m->c.n_layers * m->c.seq_len * m->c.n_heads * LLM_KPAD + 16;
}
static inline size_t llm_kv_hot_bytes(const Model *m) {
  size_t S = m->c.seq_len;
  return 2 * (LLM_KPAD + S * sizeof(int16_t)) + S * sizeof(float) + 48;
}
static inline size_t llm_kv_main_bytes(const Model *m) {
  size_t L = m->c.n_layers, S = m->c.seq_len, H = m->c.n_heads, Dh = m->c.dim / m->c.n_heads;
  return L * H * Dh * S * sizeof(int16_t) + 2 * L * S * H * sizeof(float) + 16;
}
static inline size_t llm_kv_quant_bytes(const Model *m) {
  return llm_kv_main_bytes(m) + llm_kv_key_bytes(m) + llm_kv_hot_bytes(m);
}
static inline uint8_t *llm_align16(void *p) {
  return (uint8_t *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
}
static inline void llm_kv_quant_bind(const Model *m, Scratch *s, void *main_buf,
                                     void *key_buf, void *hot_buf) {
  size_t L = m->c.n_layers, S = m->c.seq_len, H = m->c.n_heads, Dh = m->c.dim / m->c.n_heads;
  uint8_t *p = llm_align16(main_buf);
  s->v16t = (int16_t *)p; p += L * H * Dh * S * sizeof(int16_t); /* S*2-byte rows */
  s->ks = (float *)p;     p += L * S * H * sizeof(float);
  s->vs = (float *)p;     p += L * S * H * sizeof(float);
  uint8_t *k = llm_align16(key_buf ? key_buf : p);
  s->k8 = (int8_t *)k;    k += L * S * H * LLM_KPAD;              /* 32-byte rows */
  if (!key_buf) p = k;
  uint8_t *h = llm_align16(hot_buf ? hot_buf : p);
  s->q8 = (int8_t *)h;    h += LLM_KPAD;
  s->w16 = (int16_t *)h;  h += S * sizeof(int16_t);
  h = llm_align16(h);
  s->q8b = (int8_t *)h;   h += LLM_KPAD;
  s->w16b = (int16_t *)h; h += S * sizeof(int16_t);
  s->scoresb = (float *)h;
}

/* Attention over heads h0..h1-1 of one layer at one position, on the
 * quantized cache, with the caller's temporaries. Reads k8/v16t written for
 * every position up to pos, writes only this range's slice of s->att, so two
 * ranges can run at once on two cores with separate temporaries. */
static void llm_attend_heads(Model *m, Scratch *s, int l, int pos, int h0, int h1,
                             int8_t *q8, int16_t *w16, float *scores) {
  int D = m->c.dim, H = m->c.n_heads, Dh = D / H, S = m->c.seq_len;
  float scale = 1.f / sqrtf((float)Dh);
  int nk = LLM_KPAD / 16, n16 = (pos + 1 + 7) / 8;
  const float *q = s->qkv;
  for (int hh = h0; hh < h1; hh++) {
    const float *qh = q + hh * Dh;
    float *ao = s->att + hh * Dh;
    float qmax = 1e-8f;
    for (int i = 0; i < Dh; i++) { float a = fabsf(qh[i]); if (a > qmax) qmax = a; }
    float qinv = 127.f / qmax;
    for (int i = 0; i < Dh; i++) {
      int v8 = (int)lrintf(qh[i] * qinv);
      q8[i] = (int8_t)(v8 > 127 ? 127 : (v8 < -127 ? -127 : v8));
    }
    for (int i = Dh; i < LLM_KPAD; i++) q8[i] = 0;
    float qs = qmax / 127.f * scale;
    const int8_t *kb = s->k8 + ((size_t)l * S * H + hh) * LLM_KPAD;
    const float *ksb = s->ks + (size_t)l * S * H + hh;
    const float *vsb = s->vs + (size_t)l * S * H + hh;
    float maxs = -1e30f;
    for (int t = 0; t <= pos; t++) {
      float dot = (float)LLM_DOT_S8V(q8, kb + (size_t)t * H * LLM_KPAD, nk)
                  * qs * ksb[(size_t)t * H];
      scores[t] = dot;
      if (dot > maxs) maxs = dot;
    }
    // softmax weights, each multiplied by its position's value scale, to int16
    float denom = 0.f, wmax = 0.f;
    for (int t = 0; t <= pos; t++) {
      float w = expf(scores[t] - maxs);
      denom += w;
      float wv = w * vsb[(size_t)t * H];
      scores[t] = wv;
      if (wv > wmax) wmax = wv;
    }
    float winv = 32767.f / wmax;
    for (int t = 0; t <= pos; t++) w16[t] = (int16_t)lrintf(scores[t] * winv);
    for (int t = pos + 1; t < n16 * 8; t++) w16[t] = 0;
    float os = wmax / 32767.f / denom;
    const int16_t *vt = s->v16t + ((size_t)l * H + hh) * Dh * S;
    for (int i = 0; i < Dh; i++)
      ao[i] = (float)LLM_DOT_S16V(vt + (size_t)i * S, w16, n16) * os;
  }
}
#endif

// One decode step: token at position pos -> logits[V]. KV cache persists across calls.
/* Per-layer matvec dispatch: platform hook if set, otherwise MATVEC. */
#define LLM_LMV(m, t, x, y) \
  do { if ((m)->layer_matvec) (m)->layer_matvec((t), (x), (y)); \
       else MATVEC((t), (x), (y)); } while (0)

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
    rmsnorm(s->x, m->attn_norm[l], D, s->h);
    LLM_LMV(m, &m->qkv[l], s->h, s->qkv);        // [3D]
    float *q = s->qkv, *k = s->qkv + D, *v = s->qkv + 2 * D;
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
#ifdef LLM_KV_QUANT
    // ---- quantized cache: store this position's k (int8) and v (int16)
    for (int hh = 0; hh < H; hh++) {
      const float *kh = k + hh * Dh, *vh = v + hh * Dh;
      float kmax = 1e-8f, vmax = 1e-8f;
      for (int i = 0; i < Dh; i++) {
        float a = fabsf(kh[i]); if (a > kmax) kmax = a;
        a = fabsf(vh[i]);       if (a > vmax) vmax = a;
      }
      size_t slot = ((size_t)l * S + pos) * H + hh;
      int8_t *kd = s->k8 + slot * LLM_KPAD;
      float kinv = 127.f / kmax;
      for (int i = 0; i < Dh; i++) {
        int q8 = (int)lrintf(kh[i] * kinv);
        kd[i] = (int8_t)(q8 > 127 ? 127 : (q8 < -127 ? -127 : q8));
      }
      for (int i = Dh; i < LLM_KPAD; i++) kd[i] = 0;
      s->ks[slot] = kmax / 127.f;
      int16_t *vt = s->v16t + ((size_t)l * H + hh) * Dh * S;
      float vinv = 32767.f / vmax;
      for (int i = 0; i < Dh; i++) {
        int q16 = (int)lrintf(vh[i] * vinv);
        vt[(size_t)i * S + pos] = (int16_t)(q16 > 32767 ? 32767 : (q16 < -32767 ? -32767 : q16));
      }
      s->vs[slot] = vmax / 32767.f;
    }
    // ---- causal attention over 0..pos, both passes as vector dots
    if (m->attn_heads) m->attn_heads(m, s, l, pos);
    else llm_attend_heads(m, s, l, pos, 0, H, s->q8, s->w16, s->scores);
#else
    float *kc = s->kcache + (size_t)l * S * D, *vc = s->vcache + (size_t)l * S * D;
    memcpy(kc + (size_t)pos * D, k, D * sizeof(float));
    memcpy(vc + (size_t)pos * D, v, D * sizeof(float));
    // causal attention over 0..pos
    float scale = 1.f / sqrtf((float)Dh);
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
    LLM_LMV(m, &m->attn_proj[l], s->att, s->h);
    for (int i = 0; i < D; i++) s->x[i] += s->h[i];
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
#ifdef LLM_PROFILE
    profile_t1 = (uint64_t)LLM_PROFILE_NOW();
    s->profile.ple_us += profile_t1 - profile_t3;
#endif
  }

  rmsnorm(s->x, m->out_norm, D, s->x);
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
