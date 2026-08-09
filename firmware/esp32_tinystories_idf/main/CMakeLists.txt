set(TINYSTORIES_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../runtime")
set(TINYSTORIES_VOCAB_DIR "${CMAKE_CURRENT_LIST_DIR}/../generated")
set(TINYSTORIES_VOCAB_HEADER "${TINYSTORIES_VOCAB_DIR}/vocab.h")
set(
    TINYSTORIES_ARTIFACT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../artifacts/tinystories"
    CACHE PATH
    "Directory containing the verified TinyStories release artifacts"
)
set(TINYSTORIES_MODEL_BIN "${TINYSTORIES_ARTIFACT_DIR}/model.bin")

if(NOT EXISTS "${TINYSTORIES_RUNTIME_DIR}/llm.h")
    message(FATAL_ERROR "Missing shared runtime: ${TINYSTORIES_RUNTIME_DIR}/llm.h")
endif()

if(NOT EXISTS "${TINYSTORIES_VOCAB_HEADER}")
    message(FATAL_ERROR "Missing generated vocabulary: ${TINYSTORIES_VOCAB_HEADER}. Follow the Prepare the released model section in firmware/esp32_tinystories_idf/README.md.")
endif()

idf_component_register(
    SRCS "esp32_tinystories_idf.cpp" "board_display.cpp"
    INCLUDE_DIRS "." "${TINYSTORIES_RUNTIME_DIR}" "${TINYSTORIES_VOCAB_DIR}"
    REQUIRES esp_board_manager esp_lcd esp_partition esp_timer driver heap
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

# A full `idf.py flash` always includes the released model. This intentionally
# makes flashing fail before touching the board when the artifact is absent.
# `app-flash` remains available after the model has been installed once.
set(TINYSTORIES_MODEL_PARTITION_SIZE 15007744) # 0xE50000
if(EXISTS "${TINYSTORIES_MODEL_BIN}")
    file(SIZE "${TINYSTORIES_MODEL_BIN}" TINYSTORIES_MODEL_BIN_SIZE)
    if(TINYSTORIES_MODEL_BIN_SIZE GREATER TINYSTORIES_MODEL_PARTITION_SIZE)
        message(FATAL_ERROR
            "model.bin (${TINYSTORIES_MODEL_BIN_SIZE} bytes) exceeds the model partition "
            "(${TINYSTORIES_MODEL_PARTITION_SIZE} bytes)"
        )
    endif()
else()
    message(WARNING "artifacts/tinystories/model.bin is absent. Run scripts/fetch_model.sh tinystories before flashing.")
endif()
esptool_py_flash_to_partition(flash "model" "${TINYSTORIES_MODEL_BIN}")
