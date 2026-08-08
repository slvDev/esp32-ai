# esp_partition is a separate component from spi_flash in IDF v5.x; the sketch
# memory-maps the model out of the custom 'model' partition, so it is required
# rather than incidental.
idf_component_register(SRCS "main.cpp" "psram_bw.c" "dfr1154_sensors.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp_partition spi_flash esp_timer driver
                                esp_psram esp_mm)

# Memory calibration probe, off by default so benchmark builds are unchanged.
# Enable with: idf.py -DBW_PROBE=1 build
if(NOT DEFINED BW_PROBE)
  set(BW_PROBE 0)
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE BW_PROBE=${BW_PROBE})

# -O3 to match the Arduino reference build, which was compiled with
# `--build-property compiler.optimization_flags=-O3`. ESP-IDF's own
# COMPILER_OPTIMIZATION_PERF is only -O2, so comparing the port against the
# 30.82 tok/s reference without this would measure the optimiser, not the port.
target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

# -fstack-usage emits a .su file per object giving each function's exact frame
# size. It changes no generated code -- it only makes the compiler report what
# it already decided.
#
# Kept permanently rather than used once. llm_forward holds `int32_t
# bank[2][LLM_MAX_HEADS*64]` (16 KB) and `int8_t qq[LLM_Q8_MAX_INPUT]` (4 KB) as
# plain locals, sized by compile-time maxima rather than by this model's actual
# H=4/Dh=24, and a stack overflow here already cost this project a debugging
# cycle by surfacing as a bogus queue.c assertion. Frame sizes are load-bearing
# in this firmware and should be visible on every build, not rediscovered.
target_compile_options(${COMPONENT_LIB} PRIVATE -fstack-usage)

# CertiHead geometry is set by #define in the sketch, NOT by -D.
#
# The -D route was removed after it produced two retracted ledger entries. A
# failed sweep wrote the literal string '$t' into CMakeCache.txt; deleting the
# cache did not clear it from the generated ninja command, and idf.py -DCERT_TILE=64
# did not overwrite it either. Four "tile sweep" arms flashed one binary and
# agreed to the row, which was then written up as a structural finding.
#
# tools/sweep_tile.ps1 edits the #define and restores it in a finally block.
# Combined with OBJECT_DEPENDS below, that guarantees a real rebuild, and every
# arm asserts the firmware's boot line reports the geometry that was requested.


# main.cpp #includes the Arduino sketch, which #includes runtime/llm.h. CMake
# does not treat a .ino as a header, and a stale object silently reran an old
# kernel during a sweep -- the profile print did not change when the source did.
# Declare the dependency explicitly rather than trusting depfile scanning.
set_property(SOURCE main.cpp APPEND PROPERTY OBJECT_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../esp32_tinystories/esp32_tinystories.ino"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/llm.h")
