cmake_minimum_required(VERSION 3.16)

# ESP-IDF port of firmware/esp32_tinystories.
#
# Reason for existing: the Arduino FQBN does not expose the settings that
# actually bind this workload. The output head is bandwidth-bound at 99.8% of a
# measured 60.7 MB/s PSRAM ceiling, so no kernel work reaches it -- only raising
# the wall does. ESP-IDF exposes three knobs Arduino does not:
#
#   CONFIG_SPIRAM_SPEED_120M                 80 -> 120 MHz octal (datasheet max)
#   CONFIG_ESP32S3_DATA_CACHE_64KB           Arduino ships 32 KB of the 64 KB
#   CONFIG_ESP32S3_DATA_CACHE_LINE_64B       longer PSRAM bursts per miss
#
# The algorithm source is shared with the Arduino build rather than copied:
# main.cpp includes ../../esp32_tinystories/esp32_tinystories.ino directly, so
# the two branches can never silently diverge. The Arduino branch stays the
# frozen reference; this one is for profiling and production.

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(idf_tinystories)
