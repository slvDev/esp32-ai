cmake_minimum_required(VERSION 3.16)

idf_component_register(SRCS "app_main.cpp"
                       PRIV_REQUIRES nvs_flash esp-dl yolo26)

# =========================================================================
# 🛠️ USER CONFIGURATION: MODEL SELECTION
# =========================================================================
# Set the filename of the model you want to use.
# It searches in:
# 1. Component Zoo: esp-dl/models/yolo26/models/[p4|s3]/
#    like:
#         yolo26n_512_s8_p4.espdl
#         yolo26n_640_s8_p4.espdl
#         yolo26n_512_s8_s3.espdl
#         yolo26n_640_s8_s3.espdl
#
# 2. Local Project: main/models/[p4|s3]/
#    like
#         yolo26n_lego_512_s8_p4.espdl   # Custom Lego Detection (28 Classes)
# =========================================================================

# --- Internal Logic: Path and Symbol Resolution ---
idf_build_get_property(target IDF_TARGET)

if(${target} STREQUAL "esp32p4")
    set(MODEL_FILENAME "yolo26n_512_s8_p4.espdl")
elseif(${target} STREQUAL "esp32s3")
    set(MODEL_FILENAME "yolo26n_512_s8_s3.espdl")
else()
    message(FATAL_ERROR "YOLO26: Unsupported target ${target}")
endif()

# FIX: Robust Component Finding
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
else()
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

get_filename_component(espdl_root "${espdl_dir}/.." ABSOLUTE)

# Identify correct folder (p4 or s3)
if(${target} STREQUAL "esp32p4")
    set(PLATFORM "p4")
else()
    set(PLATFORM "s3")
endif()

# Define Paths
set(ZOO_DIR "${espdl_root}/models/yolo26/models/${PLATFORM}")
set(LOCAL_DIR "${CMAKE_CURRENT_LIST_DIR}/models/${PLATFORM}")

# Resolve the actual file path (Zoo priority, then Local)
set(ZOO_PATH "${ZOO_DIR}/${MODEL_FILENAME}")
set(LOCAL_PATH "${LOCAL_DIR}/${MODEL_FILENAME}")

if(EXISTS "${ZOO_PATH}")
    set(FINAL_PATH "${ZOO_PATH}")
    set(INCLUDE_DIR "${ZOO_DIR}") # Capture for header include
    message(STATUS "YOLO26: Using Stock Model: ${MODEL_FILENAME}")
elseif(EXISTS "${LOCAL_PATH}")
    set(FINAL_PATH "${LOCAL_PATH}")
    set(INCLUDE_DIR "${LOCAL_DIR}") # Capture for header include
    message(STATUS "YOLO26: Using Local Custom Model: ${MODEL_FILENAME}")
else()
    message(FATAL_ERROR "YOLO26: Model ${MODEL_FILENAME} not found in Zoo or Local dir!")
endif()

# FIX: Allow C++ to find the .hpp file
target_include_directories(${COMPONENT_LIB} PRIVATE "${INCLUDE_DIR}")

# Auto-detect model type from filename and define USE_LEGO_MODEL for C++
if(MODEL_FILENAME MATCHES "lego")
    add_definitions(-DUSE_LEGO_MODEL)
endif()

# Generate the Symbol name (Sanitized)
string(MAKE_C_IDENTIFIER ${MODEL_FILENAME} SAFE_NAME)
set(RAW_SYMBOL "_binary_${SAFE_NAME}_start")

# Inject the symbol as a quoted string definition for app_main.cpp
add_definitions(-DMODEL_SYMBOL_STR="${RAW_SYMBOL}")

# FIX: Ninja error fix by setting cmake_dir explicitly
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Embed Binary with 16-byte alignment
target_add_aligned_binary_data(${COMPONENT_LIB} "${FINAL_PATH}" BINARY)

# Embed test images (matched to model type)
if(MODEL_FILENAME MATCHES "lego")
    target_add_aligned_binary_data(${COMPONENT_LIB} "images/lego.jpg" BINARY)
else()
    target_add_aligned_binary_data(${COMPONENT_LIB} "images/bus.jpg" BINARY)
    target_add_aligned_binary_data(${COMPONENT_LIB} "images/person.jpg" BINARY)
endif()
