/* ESP-IDF entry point. The algorithm is the Arduino sketch, included verbatim.
 *
 * Including the .ino rather than copying it is deliberate: this project's
 * central evidence is that a build emits a bit-identical token stream, and two
 * hand-maintained copies of the same kernel would eventually diverge in a way
 * that silently invalidates every such comparison. One source, two build
 * systems.
 */
#include "arduino_shim.h"

/* Included BEFORE the sketch so its include guard, SENSOR_MODE and the
 * dfr_* prototypes are visible inside it. The sketch's sensor hooks are
 * guarded on `#ifdef DFR1154_SENSORS_H`, so the Arduino build -- which has no
 * such header on its include path -- compiles them out and stays byte-identical
 * to the reference. One source, two build systems, and only one of them has a
 * camera. */
#include "dfr1154_sensors.h"

/* The sketch relies on arduino-cli's auto-generated forward declarations. As a
 * plain translation unit it needs them written out; anything used before its
 * definition goes here. */
static void blink(uint8_t g);
static void emit(int tok);

#include "../../esp32_tinystories/esp32_tinystories.ino"

/* Memory calibration, run before the model is staged so it measures the bus
 * rather than competing with a warm working set. Gated because it costs ~1 s of
 * boot and perturbs nothing else; the generated text is extracted between ">>>"
 * and the token count, so boot output cannot affect the equality check. */
#if BW_PROBE
extern "C" void psram_bw_probe(void);
#endif

extern "C" void app_main(void) {
#if BW_PROBE
  psram_bw_probe();
#endif

  /* ORDER IS LOAD-BEARING: sensors first, model second.
   *
   * The weight stager offers each tensor internal SRAM until free memory would
   * drop below SRAM_RESERVE_BYTES, so if it runs first it consumes every
   * internal byte down to the reserve. The esp32-camera driver then asks for
   * DMA-capable internal SRAM in contiguous blocks and fails. Booting in that
   * order would make camera init a function of the LM's appetite, and the
   * LM+camera arm would quietly degrade into the LM-only arm while still
   * printing a plausible tok/s -- the exact failure mode this harness exists to
   * rule out. Sensors claim first; the stager adapts to what is left. */
  dfr_sensors_begin();

#if SENSOR_HAS_LM
  setup();
#else
  /* Camera-only arm: the LM is not merely idle, it is absent. A staged-but-idle
   * model would still hold its PSRAM and its SRAM hot set, so the perception
   * ceiling it produced would not be a ceiling at all. */
  printf("\n=== camera-only arm: LM not loaded ===\n");
  dfr_capture_start();
  vTaskDelay(pdMS_TO_TICKS(10000));
  dfr_sensors_report();
  printf("free: sram %.0f KB | psram %.2f MB\n",
         heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0,
         heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0);
  /* This arm emits no token stream, so capture_run.py has no "accounting
   * ms/token" line to terminate on. Without a sentinel it would burn the full
   * --seconds and then fall through to the esptool hard-reset recovery path on
   * a run that actually succeeded. */
  printf("=== END_OF_RUN ===\n");
#endif

  for (;;) {
    loop();
  }
}
