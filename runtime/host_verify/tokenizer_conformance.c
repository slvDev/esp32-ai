// Host receipt for the BTK1 ASCII ByteLevel-BPE encoder: encodes each argument
// and prints the ids, so a driver can compare them against Hugging Face.
//
// Strings arrive as command-line arguments rather than on stdin, which keeps
// tabs and newlines inside a probe intact. A delimited stream cannot carry
// them, and those are exactly the characters where pre-tokenization is easiest
// to get wrong.
//
// One line per argument: either the ids separated by spaces, an empty line if
// the input encodes to nothing, or ERROR:<code> for a BTK_ERR_* rejection.
//
//   cc -O2 -std=c11 -Wall -Wextra -Werror -o /tmp/tokconf
//      runtime/host_verify/tokenizer_conformance.c
//   /tmp/tokconf tokenizer.btk "hello world" "i'm fine"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../bpe_tokenizer.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s tokenizer.btk string [string ...]\n", argv[0]);
    return 2;
  }

  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 3;
  }
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 3; }
  long size = ftell(f);
  if (size <= 0) {
    fprintf(stderr, "%s is empty or unseekable\n", argv[1]);
    fclose(f);
    return 3;
  }
  rewind(f);
  uint8_t *asset = malloc((size_t)size);
  if (!asset) { fclose(f); return 4; }
  if (fread(asset, 1, (size_t)size, f) != (size_t)size) {
    fprintf(stderr, "short read on %s\n", argv[1]);
    free(asset);
    fclose(f);
    return 4;
  }
  fclose(f);

  // The file length is the asset length: the loader needs it to know the merge
  // table is actually present rather than merely claimed.
  BpeTokenizer tok;
  if (bpe_tokenizer_load(asset, (size_t)size, &tok)) {
    fprintf(stderr, "%s is not a loadable BTK1 asset\n", argv[1]);
    free(asset);
    return 5;
  }

  for (int arg = 2; arg < argc; arg++) {
    uint16_t ids[BTK_MAX_INPUT_BYTES];
    int n = bpe_encode_ascii(&tok, argv[arg], ids, BTK_MAX_INPUT_BYTES);
    if (n < 0) {
      printf("ERROR:%d\n", n);
      continue;
    }
    for (int i = 0; i < n; i++) printf("%s%u", i ? " " : "", ids[i]);
    printf("\n");
  }

  free(asset);
  return 0;
}
