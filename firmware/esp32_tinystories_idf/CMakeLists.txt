cmake_minimum_required(VERSION 3.16)

# `idf.py bmgr` generates board-specific defaults here. Load them first so the
# project's target-specific performance defaults can override conservative
# board memory settings.
set(BMGR_DEFAULTS "${CMAKE_CURRENT_LIST_DIR}/components/gen_bmgr_codes/board_manager.defaults")
if(EXISTS "${BMGR_DEFAULTS}")
    list(FIND SDKCONFIG_DEFAULTS "${BMGR_DEFAULTS}" BMGR_DEFAULTS_INDEX)
    if(BMGR_DEFAULTS_INDEX EQUAL -1)
        if(SDKCONFIG_DEFAULTS)
            list(PREPEND SDKCONFIG_DEFAULTS "${BMGR_DEFAULTS}")
        else()
            set(SDKCONFIG_DEFAULTS
                "${BMGR_DEFAULTS};${CMAKE_CURRENT_LIST_DIR}/sdkconfig.defaults"
            )
        endif()
    endif()
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(esp32_tinystories_idf)
