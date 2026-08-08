"""Compile a DFR1154 physical-action schema into a token-id DFA for the device.

WHY
---
The runtime's output head is 41% of a token (5.4 of 13.3 ms) and 94% of that is
a linear scan over 25,353 rows at the PSRAM bus ceiling. CertiHead already
proved the only remaining lever is *reading fewer rows*, and that rho says the
row count cannot be cut further for this head without changing the model.

A grammar is the one thing that cuts rows without touching the model. When the
next token is constrained to a physical action schema, the admissible row set
is a handful of ids, and the scan has nothing left to scan. This generator
produces that row set, as a DFA, from the schema.

It also answers a question the Part-2A "physical vocabulary layout" campaign
needs an answer to before it can start: how much does the *surface form* of an
action cost? "READ_LUX" and "read the light" denote the same act but decompose
into very different numbers of BPE tokens against a vocabulary trained on
children's stories, and every extra token is a full forward pass. The generator
emits both spellings so the device can measure the difference rather than have
it asserted.

INPUT   generated/vocab.h    output-side token id -> UTF-8 bytes (25,353 rows)
OUTPUT  generated/action_grammar.h

The output vocabulary, not the input vocabulary, is the right table here: the
DFA constrains rows of the output head, and the model's input side is a
different, larger (32,768) vocabulary. Using the input table would produce ids
that index the wrong matrix -- silently, since both are in range.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOCAB_H = HERE.parent / "generated" / "vocab.h"
OUT_H = HERE.parent / "generated" / "action_grammar.h"


def load_vocab() -> list[bytes]:
    """Parse VOCAB_BLOB / VOCAB_OFF back out of the generated C header.

    Parsing the header rather than re-deriving from the tokenizer on purpose:
    the header is what the firmware actually compiles against, so a drift
    between the generator's idea of the vocabulary and the device's decode
    table is impossible by construction. That drift is exactly the class of bug
    that produces a plausible-looking run with the wrong tokens.
    """
    src = VOCAB_H.read_text(encoding="utf-8", errors="replace")

    n = int(re.search(r"#define VOCAB_N (\d+)", src).group(1))

    def array(name: str) -> list[int]:
        m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{(.*?)\}};", src, re.S)
        if not m:
            raise SystemExit(f"{name} not found in {VOCAB_H}")
        return [int(x) for x in re.findall(r"-?\d+", m.group(1))]

    blob = array("VOCAB_BLOB")
    off = array("VOCAB_OFF")
    if len(off) != n + 1:
        raise SystemExit(f"VOCAB_OFF has {len(off)} entries, expected {n + 1}")
    return [bytes(blob[off[i]:off[i + 1]]) for i in range(n)]


def build_index(vocab: list[bytes]) -> dict[bytes, int]:
    """First id wins on duplicates.

    Duplicates exist because the blob is a raw byte table and several ids can
    decode to the same bytes. Which one we pick does not matter for cost (every
    row is 52 bytes) but it must be deterministic, or two runs of the generator
    would produce different DFAs and the digest comparison across builds would
    be meaningless.
    """
    idx: dict[bytes, int] = {}
    for i, b in enumerate(vocab):
        if b and b not in idx:
            idx[b] = i
    return idx


def encode(text: str, idx: dict[bytes, int], maxlen: int) -> list[int] | None:
    """Greedy longest-match over the output vocabulary.

    Greedy, not true BPE-merge order: this is a *constraint* table, not a
    reproduction of the tokenizer's own segmentation. The DFA only needs some
    valid decomposition of the surface string into emittable rows. Returns None
    if any byte of the string is unreachable, which is a real possibility for
    uppercase-with-underscore spellings against a children's-story vocabulary
    and is precisely one of the things worth measuring.
    """
    data = text.encode("utf-8")
    out: list[int] = []
    pos = 0
    while pos < len(data):
        for ln in range(min(maxlen, len(data) - pos), 0, -1):
            tid = idx.get(data[pos:pos + ln])
            if tid is not None:
                out.append(tid)
                pos += ln
                break
        else:
            return None
    return out


# The schema from the sensor/tool programme, in two surface forms.
#
# TERSE is the machine-style spelling an API designer would reach for.
# NATURAL is the same act spelled in words the TinyStories corpus actually
# contains. They are semantically identical and physically identical -- the same
# GPIO toggles, the same OV3660 grab -- so any latency difference between them
# is purely a property of how the vocabulary was built, which is the whole
# argument for treating action vocabulary as a design surface rather than an
# inherited constant.
TERSE = [
    "READ_LUX", "IR_ON", "IR_OFF", "LOOK", "LISTEN",
    "WAIT", "STORE", "RECALL", "REPORT_PRESENT", "REPORT_ABSENT",
]
NATURAL = [
    " read the light", " light on", " light off", " look", " listen",
    " wait", " store", " recall", " it is there", " it is not there",
]


def emit(f, name: str, phrases: list[str], idx: dict[bytes, int],
         maxlen: int) -> tuple[int, int]:
    """Write one schema as a flat DFA and return (states, max fanout)."""
    seqs = []
    for p in phrases:
        ids = encode(p, idx, maxlen)
        if ids is None:
            print(f"  {name}: {p!r} NOT REPRESENTABLE, dropped")
            continue
        seqs.append((p, ids))

    # State 0 is the choice point: every action's first token is admissible.
    # After that each action is a deterministic chain, so the admissible set
    # collapses to one row and the head scan reads 52 bytes. The interesting
    # number is therefore the fanout at state 0 -- that is the widest the scan
    # ever gets under this grammar.
    trans: dict[int, list[int]] = {}
    nxt: dict[int, int] = {}
    state_of: dict[tuple, int] = {(): 0}
    n_states = 1
    for _p, ids in seqs:
        prefix: tuple = ()
        for t in ids:
            s = state_of[prefix]
            trans.setdefault(s, [])
            if t not in trans[s]:
                trans[s].append(t)
            prefix = prefix + (t,)
            if prefix not in state_of:
                state_of[prefix] = n_states
                n_states += 1
            nxt[(s, t)] = state_of[prefix]

    accept = {state_of[tuple(ids)] for _p, ids in seqs}

    f.write(f"\n/* ---- {name} ---- */\n")
    for p, ids in seqs:
        f.write(f"/* {p!r:24} -> {len(ids):2d} tokens {ids} */\n")
    tot = sum(len(i) for _p, i in seqs)
    f.write(f"/* {len(seqs)} actions, {tot} tokens total, "
            f"{tot / max(len(seqs), 1):.1f} tokens/action mean */\n")

    f.write(f"#define {name}_STATES {n_states}\n")
    f.write(f"#define {name}_ACTIONS {len(seqs)}\n")
    f.write(f"#define {name}_MEAN_TOKENS {tot / max(len(seqs), 1):.4f}f\n")

    maxfan = max((len(v) for v in trans.values()), default=0)
    f.write(f"#define {name}_MAXFAN {maxfan}\n")
    f.write(f"static const int16_t {name}_FAN[{name}_STATES] = {{")
    f.write(",".join(str(len(trans.get(s, []))) for s in range(n_states)))
    f.write("};\n")
    f.write(f"static const int32_t {name}_TOK[{name}_STATES][{maxfan}] = {{\n")
    for s in range(n_states):
        row = trans.get(s, [])
        row = row + [-1] * (maxfan - len(row))
        f.write("  {" + ",".join(str(t) for t in row) + "},\n")
    f.write("};\n")
    f.write(f"static const int16_t {name}_NEXT[{name}_STATES][{maxfan}] = {{\n")
    for s in range(n_states):
        row = trans.get(s, [])
        vals = [nxt[(s, t)] for t in row] + [-1] * (maxfan - len(row))
        f.write("  {" + ",".join(str(v) for v in vals) + "},\n")
    f.write("};\n")
    f.write(f"static const uint8_t {name}_ACCEPT[{name}_STATES] = {{")
    f.write(",".join("1" if s in accept else "0" for s in range(n_states)))
    f.write("};\n")
    return n_states, maxfan


def main() -> int:
    vocab = load_vocab()
    idx = build_index(vocab)
    maxlen = max(len(b) for b in vocab)
    print(f"vocab: {len(vocab)} output ids, {len(idx)} distinct byte strings, "
          f"longest {maxlen} B")

    with OUT_H.open("w", encoding="utf-8", newline="\n") as f:
        f.write("/* GENERATED by tools/generate_action_grammar.py -- "
                "do not edit.\n"
                " *\n"
                " * A DFA over OUTPUT-head row ids. At each state only FAN[s]\n"
                " * rows are admissible, so the head scan reads FAN[s]*52 bytes\n"
                " * instead of walking 25,353 rows. State 0 is the action choice\n"
                " * point and is the widest the scan ever gets.\n"
                " */\n"
                "#ifndef ACTION_GRAMMAR_H\n#define ACTION_GRAMMAR_H\n"
                "#include <stdint.h>\n")
        print("terse schema:")
        emit(f, "AG_TERSE", TERSE, idx, maxlen)
        print("natural schema:")
        emit(f, "AG_NAT", NATURAL, idx, maxlen)
        f.write("\n#endif /* ACTION_GRAMMAR_H */\n")

    print(f"wrote {OUT_H}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
