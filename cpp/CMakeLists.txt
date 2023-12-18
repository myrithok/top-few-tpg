cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME top-few-tpg)

include(FetchContent)

project(${PROJECT_NAME})

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CURL_LIBRARY "-lcurl")
find_package(CURL REQUIRED)

include_directories(.)
include_directories(include)
include_directories(include/gym)

set(CURL_LIBRARY "-lcurl")
find_package(CURL REQUIRED)

find_package(GEGELATI)
if (NOT GEGELATI_FOUND)
    FetchContent_Declare(
            GEGELATI
            GIT_REPOSITORY ttps://github.com/myrithok/top-few-tpg-gegelati.git
    )
    FetchContent_MakeAvailable(GEGELATI)
endif ()

file(GLOB_RECURSE
        gym_wrapper_files
        include/gym/gym.h
        json-forwards.h
        json.h
        jsoncpp.cpp
        random_agent.cpp

        src/wrapper/GymWrapper.h
        src/wrapper/GymWrapper.cpp
        src/wrapper/gym_binding.cpp
        src/main.cpp
)


include_directories(${CURL_INCLUDE_DIR})
include_directories(${GEGELATI_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${gym_wrapper_files})
target_link_libraries(${PROJECT_NAME} ${CURL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GEGELATI_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${PROJECT_NAME} PRIVATE TOP_FEW=true)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER=false)
target_compile_definitions(${PROJECT_NAME} PRIVATE EVALUATE=false)

add_executable(single-tpg ${gym_wrapper_files})
target_link_libraries(single-tpg ${CURL_LIBRARIES})
target_link_libraries(single-tpg ${GEGELATI_LIBRARIES})
target_compile_definitions(single-tpg PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(single-tpg PRIVATE TOP_FEW=false)
target_compile_definitions(single-tpg PRIVATE RENDER=false)
target_compile_definitions(single-tpg PRIVATE EVALUATE=false)

add_executable(${PROJECT_NAME}-render ${gym_wrapper_files})
target_link_libraries(${PROJECT_NAME}-render ${CURL_LIBRARIES})
target_link_libraries(${PROJECT_NAME}-render ${GEGELATI_LIBRARIES})
target_compile_definitions(${PROJECT_NAME}-render PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(${PROJECT_NAME}-render PRIVATE TOP_FEW=true)
target_compile_definitions(${PROJECT_NAME}-render PRIVATE RENDER=true)
target_compile_definitions(${PROJECT_NAME}-render PRIVATE EVALUATE=true)

add_executable(single-tpg-render ${gym_wrapper_files})
target_link_libraries(single-tpg-render ${CURL_LIBRARIES})
target_link_libraries(single-tpg-render ${GEGELATI_LIBRARIES})
target_compile_definitions(single-tpg-render PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(single-tpg-render PRIVATE TOP_FEW=false)
target_compile_definitions(single-tpg-render PRIVATE RENDER=true)
target_compile_definitions(single-tpg-render PRIVATE EVALUATE=true)

include_directories(${CURL_INCLUDE_DIR})
include_directories(${GEGELATI_INCLUDE_DIRS})
add_executable(evaluate ${gym_wrapper_files})
target_link_libraries(evaluate ${CURL_LIBRARIES})
target_link_libraries(evaluate ${GEGELATI_LIBRARIES})
target_compile_definitions(evaluate PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(evaluate PRIVATE TOP_FEW=false)
target_compile_definitions(evaluate PRIVATE RENDER=false)
target_compile_definitions(evaluate PRIVATE EVALUATE=true)

include_directories(${CURL_INCLUDE_DIR})
include_directories(${GEGELATI_INCLUDE_DIRS})
add_executable(evaluate-render ${gym_wrapper_files})
target_link_libraries(evaluate-render ${CURL_LIBRARIES})
target_link_libraries(evaluate-render ${GEGELATI_LIBRARIES})
target_compile_definitions(evaluate-render PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(evaluate-render PRIVATE TOP_FEW=false)
target_compile_definitions(evaluate-render PRIVATE RENDER=true)
target_compile_definitions(evaluate-render PRIVATE EVALUATE=true)

set(SEED 42)
target_compile_definitions(${PROJECT_NAME} PRIVATE SEED=${SEED})
target_compile_definitions(single-tpg PRIVATE SEED=${SEED})
target_compile_definitions(${PROJECT_NAME}-render PRIVATE SEED=${SEED})
target_compile_definitions(single-tpg-render PRIVATE SEED=${SEED})
target_compile_definitions(evaluate PRIVATE SEED=${SEED})
target_compile_definitions(evaluate-render PRIVATE SEED=${SEED})

