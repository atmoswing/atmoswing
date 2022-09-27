from conans import ConanFile, CMake
import os


class AmtoSwing(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    requires = [
        "proj/9.0.1",
        "libcurl/7.84.0",
        "libtiff/4.3.0",
        "sqlite3/3.39.2",
        "eigen/3.4.0",
        "netcdf/4.8.1",
        "eccodes/2.27.0@terranum-conan+eccodes/stable",
    ]

    options = {
        "enable_tests": [True, False],
        "enable_benchmark": [True, False],
        "code_coverage": [True, False],
        "with_gui": [True, False],
        "build_forecaster": [True, False],
        "build_viewer": [True, False],
        "build_optimizer": [True, False],
        "build_downscaler": [True, False],
    }
    default_options = {
        "enable_tests": True,
        "enable_benchmark": False,
        "code_coverage": False,
        "with_gui": True,
        "build_forecaster": True,
        "build_viewer": True,
        "build_optimizer": True,
        "build_downscaler": True,
    }

    generators = "cmake", "gcc"

    def requirements(self):
        if self.options.enable_tests or self.options.code_coverage:
            self.requires("gtest/1.11.0")
        if self.options.enable_benchmark:
            self.requires("benchmark/1.6.2")
            self.requires("gtest/1.11.0")
        if self.options.with_gui:
            self.requires("wxwidgets/3.2.0@terranum-conan+wxwidgets/stable")
        else:
            self.requires("wxbase/3.2.1@terranum-conan+wxbase/stable")
        if self.options.build_viewer:
            self.requires("gdal/3.5.1@terranum-conan+gdal/stable")

    def configure(self):
        if self.options.code_coverage:
            self.options.enable_tests = True
        self.options["gdal"].with_curl = True # for xml support
        self.options["gdal"].shared = True
        if self.settings.os == "Linux":
            self.options["wxwidgets"].webview = False  # webview control isn't available on linux.
        if not self.options.with_gui:
            self.options["wxbase"].xml = True
            self.options["wxbase"].sockets = True

    def imports(self):
        # Copy libraries
        if self.settings.os == "Windows":
            self.copy("*.dll", dst="bin", src="@bindirs")
        if self.settings.os == "Macos":
            self.copy("*.dylib", dst="bin", src="@libdirs")
        if self.settings.os == "Linux":
            self.copy("*.so*", dst="bin", src="@libdirs")

        # Copy proj library data
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            self.copy("*", dst="share/proj", src="res", root_package="proj")
        if self.settings.os == "Macos":
            self.copy("*", dst="bin/AtmoSwing.app/Contents/share/proj", src="res", root_package="proj")
            if self.options.enable_tests:
                self.copy("*", dst="bin", src="res", root_package="proj")

        # Copy gdal library data
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            self.copy("*", dst="share", src="res", root_package="gdal")
        if self.settings.os == "Macos":
            self.copy("*", dst="bin/AtmoSwing.app/Contents/share/proj", src="res", root_package="gdal")
            if self.options.enable_tests:
                self.copy("*", dst="bin", src="res", root_package="gdal")

    def build(self):
        cmake = CMake(self)
        cmake.definitions["BUILD_TESTS"] = self.options.enable_tests
        cmake.definitions["BUILD_BENCHMARK"] = self.options.enable_benchmark
        cmake.definitions["USE_CODECOV"] = self.options.code_coverage
        cmake.definitions["USE_GUI"] = self.options.with_gui
        cmake.definitions["BUILD_FORECASTER"] = self.options.build_forecaster
        cmake.definitions["BUILD_VIEWER"] = self.options.build_viewer
        cmake.definitions["BUILD_OPTIMIZER"] = self.options.build_optimizer
        cmake.definitions["BUILD_DOWNSCALER"] = self.options.build_downscaler

        if self.options.code_coverage:
            cmake.definitions["BUILD_TESTS"] = "ON"
        if self.options.build_viewer:
            cmake.definitions["USE_GUI"] = "ON"

        cmake.configure()
        cmake.build()
        if self.settings.os == "Macos":
            cmake.install()
