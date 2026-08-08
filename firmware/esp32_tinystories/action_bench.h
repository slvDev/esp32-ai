/* Structured physical-action decoding: the application-layer cost model.
 *
 * WHY THIS EXISTS
 * ---------------
 * Every number this project has published measures free-text generation:
 * 13.3 ms/token, 75 tok/s, a TinyStories paragraph. That is the engine on a
 * dyno. The thing the programme is actually building is an agent that emits
 * physical actions -- LOOK, IR_ON, REPORT_PRESENT -- and the relevant unit of
 * work there is not a token, it is a DECISION.
 *
 * A decision has a latency, and that latency is the number that belongs next to
 * the published detector figures on this same silicon:
 *
 *     YOLO26n      512x512   ~2062 ms/frame   (esp-dl, INT8, ESP32-S3)
 *     YOLOv11n     320x320   ~11 s/frame
 *     ESPDet-Pico  224x224    <140 ms/frame   (0.36M params)
 *
 * Those are perception latencies. Ours is a decision latency. They are not the
 * same quantity and this file does not pretend they are -- but they are the two
 * halves of the same control loop, and if a generative decision costs more than
 * a detection then the agent architecture is dead on arrival regardless of how
 * good the runtime is. So it has to be measured, not assumed.
 *
 * WHAT A GRAMMAR DOES TO THE HEAD
 * -------------------------------
 * The output head is 41% of a token (5.4 of 13.3 ms) and 94% of that is a
 * linear scan over 25,353 rows running at 84 MiB/s against an 85.3 MiB/s bus.
 * CertiHead cut it to 34.5% of rows and rho says nothing further is available
 * for this head without changing the model.
 *
 * A grammar cuts it a different way. When the next token must continue a valid
 * action string, only FAN[state] rows are admissible -- typically ONE, at most
 * eight at the choice point. The scan reads FAN*52 bytes instead of 8749*52.
 * The head should very nearly vanish. This measures whether it does, and
 * whether anything else then becomes the bottleneck.
 *
 * WHAT THIS DELIBERATELY DOES NOT CLAIM
 * -------------------------------------
 * The TinyStories checkpoint was never trained on this schema. WHICH action it
 * selects is therefore meaningless and no accuracy number appears here. What is
 * meaningful, and is what this measures, is that the runtime emits a
 * SCHEMA-VALID action every time regardless of model competence -- validity is
 * a property of the decoder, not of the weights -- and what that costs. The
 * cost model is Part 1. The policy that makes the choice good is Part 2A. Those
 * are separate claims and conflating them would be the easiest way to produce a
 * result that does not survive contact with a trained model.
 */
#ifndef ACTION_BENCH_H
#define ACTION_BENCH_H

/* ---- Arms. Set by #define, never by -D (see main/CMakeLists.txt). ----
 *   0  off; the firmware is byte-identical to the free-text benchmark build
 *   1  TERSE schema    "READ_LUX" / "REPORT_PRESENT"
 *   2  NATURAL schema  " read the light" / " it is there"
 * The two schemas denote identical physical acts. Any latency difference
 * between them is a property of the tokenizer alone, which is the entire
 * argument for treating action vocabulary as a design surface (Part 2A). */
#ifndef ACTION_BENCH
#define ACTION_BENCH 0
#endif

/* 50 decisions, not 200 tokens. The control's 200 is a token budget; this is a
 * decision budget, and the two schemas need different token counts to reach the
 * same number of decisions -- which is the point. */
#ifndef ACTION_DECISIONS
#define ACTION_DECISIONS 50
#endif

#if ACTION_BENCH
#include "generated/action_grammar.h"

#if ACTION_BENCH == 1
#define AG_(x) AG_TERSE_##x
static const char *AG_NAME = "TERSE";
#else
#define AG_(x) AG_NAT_##x
static const char *AG_NAME = "NATURAL";
#endif

/* token id -> physical head row, for the DFA's alphabet only.
 *
 * The full inverse of head4_tok would be 25,353 int16 = 50 KB of internal SRAM,
 * which is more than the CertiHead index and would not fit alongside it. The
 * grammar's alphabet is a few dozen ids, so resolve just those with one linear
 * pass over the permuted row table at init. O(V) once, O(1) per step. */
#define AG_MAXTOK (AG_(STATES) * AG_(MAXFAN))
static int32_t ag_tok_id[AG_MAXTOK];
static int32_t ag_tok_row[AG_MAXTOK];
static int     ag_ntok = 0;

/* Set by the decode loop, read by the installed head hook. */
static int ag_state = 0;
static uint64_t ag_head_us = 0;
static uint32_t ag_rows_read = 0, ag_head_calls = 0;

static int ag_row_of(int32_t tid) {
  for (int i = 0; i < ag_ntok; i++)
    if (ag_tok_id[i] == tid) return ag_tok_row[i];
  return -1;
}

static bool ag_init(int rows) {
  ag_ntok = 0;
  for (int s = 0; s < AG_(STATES); s++)
    for (int k = 0; k < AG_(FAN)[s]; k++) {
      int32_t t = AG_(TOK)[s][k];
      bool seen = false;
      for (int i = 0; i < ag_ntok; i++) if (ag_tok_id[i] == t) { seen = true; break; }
      if (!seen) { ag_tok_id[ag_ntok] = t; ag_tok_row[ag_ntok] = -1; ag_ntok++; }
    }
  /* head4_tok maps physical row -> token id. It is the identity unless
   * CertiHead permuted the head by ||w|| at staging, which it does by default.
   * Resolving through it rather than assuming identity is the same trap that
   * once made a fallback path emit physical row numbers as token ids and
   * produce a completely different, entirely plausible-looking story. */
  for (int r = 0; r < rows; r++) {
    int32_t t = head4_tok ? head4_tok[r] : r;
    for (int i = 0; i < ag_ntok; i++)
      if (ag_tok_id[i] == t) { ag_tok_row[i] = r; break; }
  }
  int missing = 0;
  for (int i = 0; i < ag_ntok; i++) if (ag_tok_row[i] < 0) missing++;
  if (missing) {
    Serial.printf("grammar: %d of %d alphabet ids have no head row -- ABORT\n",
                  missing, ag_ntok);
    return false;
  }
  return true;
}

/* Head hook. Scores ONLY the rows the grammar admits at ag_state.
 *
 * Single core on purpose: at FAN<=8 rows the fork/join notification to the
 * second core costs more than the work it would offload. That is not a
 * concession, it is the finding -- the head has stopped being a parallel
 * bandwidth problem and become a latency problem, and the correct kernel for
 * eight rows is not a smaller version of the correct kernel for 8,749. */
static void head_matvec_grammar(const QT *t, const float *x, float *y) {
  (void)y;
  uint64_t t0 = (uint64_t)esp_timer_get_time();
  LLM_ALIGN16 static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  quantize_act(x, t->cols, xq, &xs);
  int32_t sumx = 0;
  for (int j = 0; j < t->cols; j++) sumx += xq[j];
  const int32_t corr = 8 * sumx;

  const int fan = AG_(FAN)[ag_state];
  const int nblk = head4_rowb / 16;
  float best = -1e30f;
  int bestk = 0;
  asm volatile("ssai 4");
  for (int k = 0; k < fan; k++) {
    int r = ag_row_of(AG_(TOK)[ag_state][k]);
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
    if (v > best) { best = v; bestk = k; }
  }
  /* The chosen ARC, not just the token: the DFA advances on the arc, and
   * recovering the arc from the token id would be ambiguous if two arcs out of
   * one state ever carried the same id. They cannot today, but a grammar
   * compiled from a richer schema could, and this costs nothing. */
  head_argmax = AG_(TOK)[ag_state][bestk];
  ag_state = AG_(NEXT)[ag_state][bestk];
  ag_rows_read += fan;
  ag_head_calls++;
  ag_head_us += (uint64_t)esp_timer_get_time() - t0;
}

/* Decode `n_decisions` complete actions and report the cost model.
 *
 * Self-contained -- it primes its own KV cache from position 0 rather than
 * continuing the free-text run. Sharing a cache would mean the action phase
 * decodes at positions 200+ while the free-text control decodes at 0..199, and
 * attention cost grows with position, so the two arms' ms/token would differ
 * for a reason that has nothing to do with the grammar. Same positions, same
 * prompt, one variable. */
static int action_bench(int n_decisions) {
  const int rows = model.out_head.rows;
  int pos = 0;
  Serial.printf("\n=== ACTION GRAMMAR: %s schema, %d actions, %d DFA states, "
                "fan<=%d ===\n", AG_NAME, AG_(ACTIONS), AG_(STATES), AG_(MAXFAN));
  if (!head_fused) {
    Serial.println("grammar: needs the int4 fused head; not installed");
    return pos;
  }
  if (!ag_init(rows)) return pos;
  Serial.printf("grammar: alphabet %d ids resolved to head rows; "
                "%.1f tokens/action from the generator\n",
                ag_ntok, (double)AG_(MEAN_TOKENS));

  /* Prime on the unconstrained head, exactly as the free-text control does, so
   * the prompt occupies the same cache positions with the same values. Only the
   * last prompt token is held back: llm_forward CONSUMES a token and leaves the
   * choice for the next one in head_argmax, so the first grammar-constrained
   * decision has to be produced by a forward that runs with the hook already
   * installed. */
  const int n_prompt = (int)(sizeof(PROMPT_IDS) / sizeof(int));
  for (int i = 0; i < n_prompt - 1; i++)
    llm_forward(&model, PROMPT_IDS[i], pos++, &s);
  int tok = PROMPT_IDS[n_prompt - 1];

  model.head_matvec = head_matvec_grammar;
  ag_state = 0; ag_head_us = 0; ag_rows_read = 0; ag_head_calls = 0;
  llm_profile_reset(&s);

  /* Perception starts with the decision loop, not at boot, so the frame rate
   * reported is the rate sustained while decisions are being made. ACTION_BENCH
   * combined with SENSOR_MODE=2 is the only configuration that produces an
   * actual application operating point: decision latency with the sensor live.
   * Everything else in this file is still a bench. */
#ifdef DFR1154_SENSORS_H
  dfr_capture_start();
#endif

  int tokens = 0, decisions = 0;
  int64_t t0 = esp_timer_get_time();
  int64_t dec_t0 = t0;
  int64_t worst = 0;
  Serial.print(">>> ");
  while (decisions < n_decisions && pos < model.c.seq_len) {
    /* The hook scores only the admissible rows during llm_forward, leaves the
     * chosen token in head_argmax and has already advanced ag_state. */
    llm_forward(&model, tok, pos++, &s);
    tok = head_argmax;
    emit(tok);
    tokens++;
    if (AG_(ACCEPT)[ag_state]) {
      int64_t now = esp_timer_get_time();
      if (now - dec_t0 > worst) worst = now - dec_t0;
      dec_t0 = now;
      decisions++;
      ag_state = 0;                   /* next decision starts at the choice point */
      Serial.print(" | ");
    }
    if ((tokens & 7) == 0) delay(0);  /* feed the task WDT, as the control does */
  }
  int64_t total = esp_timer_get_time() - t0;

  Serial.printf("\n\n--- %d decisions, %d tokens in %.3f s ---\n",
                decisions, tokens, total / 1e6);
  Serial.printf("action: %.2f ms/decision (P100 %.2f) | %.2f tokens/decision "
                "| %.2f ms/token\n",
                total / 1000.0 / decisions, worst / 1000.0,
                (double)tokens / decisions, total / 1000.0 / tokens);
  Serial.printf("action: head %.3f ms/token over %u calls, %.1f rows/token "
                "of %d (%.3f%%)\n",
                ag_head_us / 1000.0 / ag_head_calls, (unsigned)ag_head_calls,
                (double)ag_rows_read / ag_head_calls, rows,
                100.0 * ag_rows_read / ag_head_calls / rows);
  if (s.profile.calls) {
    float n = (float)s.profile.calls * 1000.f;
    Serial.printf("action profile ms/token: input %.2f | attn %.2f | ffn %.2f "
                  "| ple %.2f | head %.2f\n",
                  s.profile.input_us / n, s.profile.attn_us / n,
                  s.profile.ffn_us / n, s.profile.ple_us / n,
                  s.profile.head_us / n);
  }
#ifdef DFR1154_SENSORS_H
  dfr_sensors_report();
#endif
  Serial.println("=== END_OF_RUN ===");
  return pos;
}

#endif /* ACTION_BENCH */
#endif /* ACTION_BENCH_H */
