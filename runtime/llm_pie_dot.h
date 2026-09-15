// ESP32-P4 PIE / Xespv vector dot products (TRM chapter 4). No dependency on
// llm.h, so a platform can define LLM_DOT_S8V / LLM_DOT_S16V to these before
// including it.
//
// ESP.VMULAS.S8.XACC multiplies 16 int8 pairs and adds all products into the
// 40-bit XACC accumulator; the S16 form does 8 int16 pairs. The .LD.IP forms
// also load the next vector. Results are exact integers, identical to the
// scalar reference in llm.h, which the firmware checks at boot.
//
// The PIE instructions encode only registers x8-x15 and x24-x31, so operands
// are pinned to a2-a5. Loads must be 16-byte aligned: the unit runs in
// force-alignment mode and silently reads the aligned address below.
#ifndef LLM_PIE_DOT_H
#define LLM_PIE_DOT_H
#include <stdint.h>

#if defined(__riscv) && defined(CONFIG_IDF_TARGET_ESP32P4)
#define LLM_HAVE_PIE 1

// nvec*16 int8 pairs; nvec >= 1.
static inline int32_t llm_pie_dot_s8v(const int8_t *a_in, const int8_t *b_in, int nvec) {
  register const int8_t *a __asm__("a2") = a_in;
  register const int8_t *b __asm__("a3") = b_in;
  register int n __asm__("a4") = nvec - 1;
  register int32_t acc __asm__("a5");
  __asm__ volatile(
    "esp.zero.xacc\n"
    "esp.vld.128.ip q0, %[a], 16\n"
    "esp.vld.128.ip q1, %[b], 16\n"
    "beqz %[n], 2f\n"
    "1:\n"
    "esp.vmulas.s8.xacc.ld.ip q0, %[a], 16, q0, q1\n"
    "esp.vld.128.ip q1, %[b], 16\n"
    "addi %[n], %[n], -1\n"
    "bnez %[n], 1b\n"
    "2:\n"
    "esp.vmulas.s8.xacc q0, q1\n"
    "esp.movx.r.xacc.l %[acc]\n"
    : [acc] "=&r"(acc), [a] "+&r"(a), [b] "+&r"(b), [n] "+&r"(n) :: "memory");
  return acc;
}

// nvec*8 int16 pairs; nvec >= 1. The sum can exceed 32 bits, so both halves of
// XACC are read: the high instruction returns bits 39:32 zero-extended.
static inline int64_t llm_pie_dot_s16v(const int16_t *a_in, const int16_t *b_in, int nvec) {
  register const int16_t *a __asm__("a2") = a_in;
  register const int16_t *b __asm__("a3") = b_in;
  register int n __asm__("a4") = nvec - 1;
  register uint32_t lo __asm__("a5");
  register uint32_t hi __asm__("a0");
  __asm__ volatile(
    "esp.zero.xacc\n"
    "esp.vld.128.ip q0, %[a], 16\n"
    "esp.vld.128.ip q1, %[b], 16\n"
    "beqz %[n], 2f\n"
    "1:\n"
    "esp.vmulas.s16.xacc.ld.ip q0, %[a], 16, q0, q1\n"
    "esp.vld.128.ip q1, %[b], 16\n"
    "addi %[n], %[n], -1\n"
    "bnez %[n], 1b\n"
    "2:\n"
    "esp.vmulas.s16.xacc q0, q1\n"
    "esp.movx.r.xacc.l %[lo]\n"
    "esp.movx.r.xacc.h %[hi]\n"
    : [lo] "=&r"(lo), [hi] "=&r"(hi), [a] "+&r"(a), [b] "+&r"(b), [n] "+&r"(n) :: "memory");
  return ((int64_t)(int8_t)(hi & 0xFF) << 32) | (int64_t)lo;
}
// Dot product of nchunk chunks of 32 packed int4 weights against int8
// activations laid out to match the unpacking: for each chunk, the 16
// activations at even indices then the 16 at odd indices (llm_pie4_prepare).
// A 16-byte load of packed bytes gives the even weights in the low nibbles
// and the odd weights in the high nibbles; AND with 0x0F and a 4-bit shift of
// each 32-bit lane followed by the same AND separate them. The codes are
// used as they are (0..15): the caller subtracts 8 * sum(x), which is what
// makes the result identical to (code - 8) * x. nchunk >= 1.
static inline int32_t llm_pie_dot4(const uint8_t *w_in, const int8_t *xp_in,
                                   int nchunk, const uint8_t *mask16) {
  register const uint8_t *w __asm__("a2") = w_in;
  register const int8_t *x __asm__("a3") = xp_in;
  register int n __asm__("a4") = nchunk;
  register int32_t acc __asm__("a5");
  register const uint8_t *mk __asm__("a0") = mask16;
  register int sar __asm__("a1") = 4;
  __asm__ volatile(
    "esp.movx.w.sar %[sar]\n"
    "esp.vld.128.ip q6, %[mk], 0\n"
    "esp.zero.xacc\n"
    "1:\n"
    "esp.vld.128.ip q0, %[w], 16\n"
    "esp.vld.128.ip q4, %[x], 16\n"
    "esp.vld.128.ip q5, %[x], 16\n"
    "esp.andq q2, q0, q6\n"
    "esp.vsr.u32 q3, q0\n"
    "esp.andq q3, q3, q6\n"
    "esp.vmulas.s8.xacc q2, q4\n"
    "esp.vmulas.s8.xacc q3, q5\n"
    "addi %[n], %[n], -1\n"
    "bnez %[n], 1b\n"
    "esp.movx.r.xacc.l %[acc]\n"
    : [acc] "=&r"(acc), [w] "+&r"(w), [x] "+&r"(x), [n] "+&r"(n), [mk] "+&r"(mk)
    : [sar] "r"(sar)
    : "memory");
  return acc;
}
#else
#define LLM_HAVE_PIE 0
#endif

#endif
