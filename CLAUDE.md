# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AtmoSwing (Analog Techniques for Statistical Weather forecastING) is a C++17 application suite implementing Analog Methods for meteorological forecasting and downscaling. It consists of four standalone tools: **Forecaster**, **Optimizer**, **Downscaler**, and **Viewer**.

## Build System

The project uses **CMake 3.18+** with **vcpkg** for dependency management. The `VCPKG_ROOT` environment variable must be defined before running CMake (it is enforced via `FATAL_ERROR`).

```bash
# Configure (all targets, with tests)
cmake -B build -DBUILD_FORECASTER=ON -DBUILD_VIEWER=ON -DBUILD_OPTIMIZER=ON \
      -DBUILD_DOWNSCALER=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# Configure headless server version (no viewer/GUI)
cmake -B build -DBUILD_FORECASTER=ON -DBUILD_VIEWER=OFF -DBUILD_OPTIMIZER=ON \
      -DBUILD_DOWNSCALER=ON -DBUILD_TESTS=ON -DUSE_GUI=OFF -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Create installer (Windows: ZIP + WIX; Linux: TGZ + DEB)
cmake --build build --target package
```

**vcpkg manifest features** are automatically derived from `BUILD_*` options. If you change a `BUILD_*` option, run CMake twice — the first run updates the cache, the second triggers vcpkg with the new feature set.

**CI pipelines** in `.github/workflows/` still use Conan v1 (migration to vcpkg in progress on `dev` branch). The local build requires vcpkg.

## Running Tests

Tests are built into `atmoswing-tests` (in `build/tests/` on Linux or `build/` on Windows).

```bash
# Run all tests
./build/tests/atmoswing-tests

# Run a specific test suite
./build/tests/atmoswing-tests --gtest_filter=TestSuiteName.*

# Run a single test
./build/tests/atmoswing-tests --gtest_filter=TestSuiteName.TestName

# GUI tests require xvfb (Linux)
xvfb-run --server-args="-screen 0, 1280x720x24" -a ./build/tests/atmoswing-tests

# Skip long-running calibration tests
cmake -B build -DTEST_CALIBRATION=OFF ...
```

Required environment variables for tests:
- `ECCODES_DEFINITION_PATH` — path to deployed eccodes definitions
- `PROJ_DATA` / `PROJ_LIB` — path to PROJ data (Viewer/GUI builds)

## Code Architecture

### Source Tree

```
src/
├── shared_base/           # Foundation library (required by all targets)
│   ├── core/              # File I/O (NetCDF/HDF5/GRIB), area/coordinate handling,
│   │                      # time series, parameter parsing, atmospheric data
│   └── gui/               # Base wxWidgets components
├── shared_processing/     # Algorithm library (used by Optimizer & Downscaler)
│   └── core/              # Analog criteria: RMSE, MD, S0, S1, S1G, DSD, …
├── app_forecaster/        # Operational forecasting (schedules & runs analog forecasts)
├── app_optimizer/         # Calibration: classic, single-step, genetic algorithms
├── app_downscaler/        # Climate downscaling (CMIP5/CORDEX support)
└── app_viewer/            # GIS visualization (wxWidgets + vroomgis)
```

**Dependency chain:** every app depends on `shared_base`; Optimizer and Downscaler also depend on `shared_processing`; Viewer adds vroomgis and GDAL/GEOS.

### Key External Dependencies

| Library | Purpose |
|---------|---------|
| wxWidgets 3.3.x | GUI (fetched via FetchContent) |
| eccodes 2.46 | GRIB format reading (fetched, patched) |
| vroomgis | GIS rendering in Viewer (fetched) |
| NetCDF-C + HDF5 | Scientific data format I/O |
| PROJ | Coordinate transformations |
| Eigen3 | Linear algebra |
| GDAL / GEOS | Geospatial data (Viewer only) |
| GTest | Unit testing |

### Build Options Reference

| Option | Default | Notes |
|--------|---------|-------|
| `BUILD_FORECASTER` | ON | |
| `BUILD_VIEWER` | ON | Forces `USE_GUI=ON` |
| `BUILD_OPTIMIZER` | ON | |
| `BUILD_DOWNSCALER` | ON | |
| `BUILD_TESTS` | ON* | *ON when any non-Viewer target is built |
| `BUILD_BENCHMARK` | OFF | Requires `BUILD_OPTIMIZER=ON` |
| `USE_GUI` | OFF | Set automatically by `BUILD_VIEWER`; can be ON without Viewer |
| `TEST_CALIBRATION` | ON | Disabling skips long optimizer calibration tests |
| `TEST_GUI` | OFF | Requires `BUILD_VIEWER=ON` and `USE_GUI=ON` |
| `USE_CPPCHECK` | OFF | Static analysis |
| `USE_CODECOV` | OFF | lcov coverage (GCC only) |
| `CREATE_INSTALLER` | OFF | CPack packaging |

## Code Style

Follow the Google C++ Style Guide with these project-specific conventions:
- **Types and methods:** `CamelCase`
- **Variables:** `camelCase`
- **Member variables:** `m_` prefix
- **Global variables:** `g_` prefix

Formatting is enforced by `.clang-format` (Google base, 4-space indent, 120-column limit, `PointerAlignment: Left`). Run clang-format before committing.
