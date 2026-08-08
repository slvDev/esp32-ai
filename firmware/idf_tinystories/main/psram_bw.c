/* PSRAM effective bandwidth as a function of visited fraction and chunk size.
 *
 * This exists to gate CertiHead. The host oracle showed that an exact certified
 * head need only read ~36% of the output rows, which divides out to 2.76x less
 * PSRAM traffic. That figure silently assumes bytes cost the same whether they
 * are read in one sweep or in scattered chunks -- and this project has already
 * been burned twice by exactly that assumption:
 *
 *   cycle 08  priced token time in bytes and missed cache line size
 *   cycle 21  priced internal SRAM in bytes and missed access frequency
 *
 * The dense head is a single 1.26 MiB sequential sweep, which is the friendliest
 * possible pattern for a burst-oriented memory. A certified head skips, so the
 * question is not "how many bytes" but "how fast are those bytes when you stop
 * reading the ones in between". If efficiency collapses below ~60% at the tile
 * sizes CertiHead wants, then reading 36% of the rows saves nothing and the
 * firmware work must not start.
 *
 * Method: one PSRAM buffer far larger than the 32 KB data cache, so every read
 * is a compulsory miss exactly as the head's are. Walk it in chunks of C bytes,
 * visiting one chunk in every `stride`, and time the walk. Two orders are
 * measured because they can differ:
 *
 *   strided  regular, every stride-th chunk. Easy for any prefetcher to follow,
 *            so it is the optimistic bound.
 *   scattered pseudo-random subset in ascending address order, which is what a
 *            data-dependent certificate actually produces. This is the honest
 *            one; the gap between the two is the prefetcher's contribution.
 *
 * Reported as effective MiB/s over BYTES ACTUALLY READ, not bytes spanned.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"

#define BW_BUF_BYTES (4 * 1024 * 1024)

/* Volatile sink: without it -O3 deletes the entire read loop and the benchmark
 * measures an empty for-statement, which is the classic way these numbers come
 * out impossibly high. */
static volatile uint32_t bw_sink;

static uint32_t read_chunk(const uint32_t *p, int words) {
  uint32_t a = 0;
  for (int i = 0; i < words; i++) a += p[i];
  return a;
}

/* One pass. Returns effective MiB/s over bytes actually touched. */
static float bw_pass(const uint8_t *buf, size_t bytes, int chunk, int stride,
                     int scattered) {
  int nchunks = (int)(bytes / (size_t)chunk);
  int words = chunk / 4;
  uint32_t acc = 0;
  size_t touched = 0;

  int64_t t0 = esp_timer_get_time();
  if (!scattered) {
    for (int c = 0; c < nchunks; c += stride) {
      acc += read_chunk((const uint32_t *)(buf + (size_t)c * chunk), words);
      touched += (size_t)chunk;
    }
  } else {
    /* Ascending addresses, irregular gaps: a cheap LCG decides whether each
     * chunk is visited, targeting the same 1/stride density. Addresses stay
     * monotonic because a certified head also scans tiles in order -- what
     * varies is which ones, not the direction. */
    uint32_t rng = 0x12345678u;
    uint32_t thresh = 0xFFFFFFFFu / (uint32_t)stride;
    for (int c = 0; c < nchunks; c++) {
      rng = rng * 1664525u + 1013904223u;
      if (rng > thresh) continue;
      acc += read_chunk((const uint32_t *)(buf + (size_t)c * chunk), words);
      touched += (size_t)chunk;
    }
  }
  int64_t us = esp_timer_get_time() - t0;
  bw_sink = acc;
  if (us <= 0 || touched == 0) return 0.f;
  return (float)touched / (float)us * (1000000.0f / 1048576.0f);
}

void psram_bw_probe(void) {
  uint8_t *buf = (uint8_t *)heap_caps_malloc(BW_BUF_BYTES, MALLOC_CAP_SPIRAM);
  if (!buf) { printf("bw: no PSRAM buffer\n"); return; }
  for (size_t i = 0; i < BW_BUF_BYTES; i += 4)
    *(uint32_t *)(buf + i) = (uint32_t)i;

  static const int chunks[] = {64, 256, 1024, 4096, 6656, 16384};
  /* 6656 B = 128 head rows x 52 B, the tile CertiHead's byte accounting chose. */
  static const int strides[] = {1, 2, 3, 4, 10};
  const int nc = sizeof(chunks) / sizeof(chunks[0]);
  const int ns = sizeof(strides) / sizeof(strides[0]);

  printf("\n=== PSRAM effective bandwidth: MiB/s over bytes READ ===\n");
  printf("chunk    100%%    50%%    33%%    25%%    10%%   | scattered 33%%\n");
  for (int ci = 0; ci < nc; ci++) {
    printf("%5dB", chunks[ci]);
    for (int si = 0; si < ns; si++) {
      float best = 0.f;
      for (int rep = 0; rep < 3; rep++) {
        float v = bw_pass(buf, BW_BUF_BYTES, chunks[ci], strides[si], 0);
        if (v > best) best = v;
      }
      printf(" %6.1f", best);
    }
    float sc = 0.f;
    for (int rep = 0; rep < 3; rep++) {
      float v = bw_pass(buf, BW_BUF_BYTES, chunks[ci], 3, 1);
      if (v > sc) sc = v;
    }
    printf("   |  %6.1f\n", sc);
  }
  printf("=== end bandwidth probe ===\n\n");
  heap_caps_free(buf);
}
