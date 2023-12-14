cmake_minimum_required(VERSION 3.8)

set (PROJECT_NAME top-few-tpg)

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
if(NOT GEGELATI_FOUND)
        FetchContent_Declare(
                GEGELATI
                GIT_REPOSITORY ttps://github.com/myrithok/top-few-tpg-gegelati.git
        )
        FetchContent_MakeAvailable(GEGELATI)
endif()

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

add_executable(single-tpg ${gym_wrapper_files})
target_link_libraries(single-tpg ${CURL_LIBRARIES})
target_link_libraries(single-tpg ${GEGELATI_LIBRARIES})
target_compile_definitions(single-tpg PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(single-tpg PRIVATE TOP_FEW=false)
