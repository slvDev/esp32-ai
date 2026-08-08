/* DFR1154 physical I/O â€” pin map and coexistence harness.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * Every throughput number this project has produced so far was measured with
 * the LM as the only consumer of the board. That is a dyno number. The DFR1154
 * is an OV3660 + PDM mic + LTR-308 + IR illuminator carrier, and the research
 * programme (sensor-as-tool, active perception, long-horizon agency) needs the
 * LM to run *while* those are live. Whether that is even possible had never
 * been measured -- it was assumed, and the assumption was load-bearing.
 *
 * The single question this harness answers:
 *
 *     Does the exactness contract survive concurrent perception?
 *
 * Digest 43184d1c is the project's correctness proof. If it changes when the
 * camera is streaming, then either the runtime has a latent data race or the
 * memory pressure has silently altered a staging decision, and every previous
 * result is a lab artefact. If it holds, the contract is a property of the
 * system rather than of a quiet board.
 *
 * PIN MAP PROVENANCE
 * ------------------
 * Transcribed from DFRobot's own examples, not from a datasheet reading:
 *   camera   DFR1154_Examples/5.3 CameraWebServer/source code/5_3/5_3.ino
 *   mic/spkr DFR1154_Examples/5.2 Recording & Playback/source code/5_2/5_2.ino
 * The OV3660 has no PWDN and no RESET line brought out on this carrier; both
 * are -1, so the sensor cannot be power-cycled in software. That matters for
 * the agent programme: LOOK cannot be made free by powering the sensor down
 * between observations, so acquisition cost is a real cost.
 */
#ifndef DFR1154_SENSORS_H
#define DFR1154_SENSORS_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/* ---- Coexistence arm. Set by #define, never by -D. --------------------------
 *
 * The -D route was removed project-wide after it produced two retracted ledger
 * entries: a stale CMakeCache made four "sweep arms" flash one identical
 * binary and agree to the row. tools/sweep_sensors.ps1 edits the line below and
 * restores it in a finally block, and the firmware prints the arm it actually
 * built so a mismatch is visible in the capture rather than inferred.
 *
 *   0  LM only                     control; must reproduce 43184d1c bit-exact
 *   1  camera only                 perception ceiling with no LM competing
 *   2  LM + camera concurrent      the arm the whole programme depends on
 */
#ifndef SENSOR_MODE
#define SENSOR_MODE 0
#endif

#define SENSOR_HAS_CAMERA (SENSOR_MODE == 1 || SENSOR_MODE == 2)
#define SENSOR_HAS_LM     (SENSOR_MODE == 0 || SENSOR_MODE == 2)

/* ---- OV3660 (DMVP-3660, 940 nm sensitive) ---- */
#define DFR_PWDN_GPIO   -1   /* not brought out on this carrier */
#define DFR_RESET_GPIO  -1   /* not brought out on this carrier */
#define DFR_XCLK_GPIO    5
#define DFR_SIOD_GPIO    8   /* SCCB SDA -- shared with the LTR-308 */
#define DFR_SIOC_GPIO    9   /* SCCB SCL -- shared with the LTR-308 */
#define DFR_Y9_GPIO      4
#define DFR_Y8_GPIO      6
#define DFR_Y7_GPIO      7
#define DFR_Y6_GPIO     14
#define DFR_Y5_GPIO     17
#define DFR_Y4_GPIO     21
#define DFR_Y3_GPIO     18
#define DFR_Y2_GPIO     16
#define DFR_VSYNC_GPIO   1
#define DFR_HREF_GPIO    2
#define DFR_PCLK_GPIO   15

/* IR illuminator. DFRobot's camera example drives this as the "LED flash" via
 * LEDC; combined with the OV3660's 940 nm response it is the actuator that
 * makes active perception physical rather than simulated -- the agent can
 * change its own observation conditions. */
#define DFR_IRLED_GPIO  47

/* ---- PDM microphone / I2S speaker (not yet brought up; recorded here so the
 * pin map lives in one place) ---- */
#define DFR_MIC_CLK_GPIO   38
#define DFR_MIC_DATA_GPIO  39
#define DFR_SPK_BCLK_GPIO  45
#define DFR_SPK_LRCK_GPIO  46
#define DFR_SPK_DOUT_GPIO  42

/* ---- Capture geometry ------------------------------------------------------
 *
 * 240x240 RGB565, NOT the UXGA JPEG of the stock example.
 *
 * Two reasons, both deliberate. (a) The consumer here is a perception expert
 * feeding typed observations to the LM, not a browser: ESPDet-Pico runs at
 * 224x224 and YOLO26n at 512x512, so a 240x240 raw frame is the regime an
 * on-device detector actually wants. (b) JPEG would put the OV3660's internal
 * compressor between us and the memory system and make the FPS number a
 * statement about image entropy rather than about bus contention with the LM.
 * Raw RGB565 gives a fixed 115,200 B/frame, so frame rate times frame size is
 * an honest bandwidth figure that can be compared against the 85.3 MiB/s the
 * LM's head scan is already consuming.
 */
#define DFR_FRAME_BYTES (240 * 240 * 2)

/* Frame buffers live in PSRAM -- the same 8 MB the staged weights use. Two of
 * them so the driver can fill one while the consumer holds the other; with
 * fb_count=1 a slow consumer stalls the sensor and the FPS number would
 * measure our consumer instead of the pipeline. */
#define DFR_FB_COUNT 2

#ifdef __cplusplus
extern "C" {
#endif

/* Brings up the camera if the arm calls for it. MUST be called before the LM
 * stages its weights: the camera driver needs DMA-capable internal SRAM in
 * contiguous blocks, and the stager will happily consume every internal byte
 * down to SRAM_RESERVE_BYTES if it goes first. Ordering the other way makes
 * camera init failure a function of the LM's appetite, which would silently
 * turn the LM+camera arm into the LM-only arm.
 *
 * Returns true if the arm's sensors are all live. */
bool dfr_sensors_begin(void);

/* Starts the free-running capture task. Separate from begin() so the LM's
 * staging and the boot-time bandwidth probe are not competing with capture --
 * we want steady-state contention during decode, not a transient during setup. */
void dfr_capture_start(void);

/* Frames completed and bytes moved since dfr_capture_start(). Read after the
 * decode loop so the FPS reported is FPS *during* inference. */
void dfr_capture_stats(uint32_t *frames, uint64_t *bytes, uint32_t *fails);

/* One line of provenance for the capture, printed into the run log so an
 * arm can never be misattributed. */
void dfr_sensors_report(void);

#ifdef __cplusplus
}
#endif
#endif /* DFR1154_SENSORS_H */
