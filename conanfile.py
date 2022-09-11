from conans import ConanFile, CMake
import os


class AmtoSwing(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    requires = [
        "proj/9.0.1",
        "libcurl/7.84.0",
        "libdeflate/1.10",
        "libtiff/4.3.0",
        "zlib/1.2.12",
        "eigen/3.4.0",
        "sqlite3/3.39.2",
        "jasper/2.0.33",
        "netcdf/4.8.1",
        "benchmark/1.6.2"
    ]

    options = {
        "tests": [True, False],
        "code_coverage": [True, False],
        "with_gui": [True, False]
    }
    default_options = {
        "tests": True,
        "code_coverage": False,
        "with_gui": True
    }

    generators = "cmake", "gcc"

    def requirements(self):
        if self.options.tests or self.options.code_coverage:
            self.requires("gtest/1.11.0")
        if self.options.with_gui:
            self.requires("wxwidgets/3.2.0@terranum-conan+wxwidgets/stable")
        else:
            self.requires("wxbase/3.1.6@pascalhorton+wxbase/stable")
        if self.settings.os == "Windows":
            self.requires("gdal/3.5.1@terranum-conan+gdal/stable")
        else:
            self.requires("gdal/3.4.1@terranum-conan+gdal/stable")

    def configure(self):
        if self.options.code_coverage:
            self.options.tests = True
        self.options["gdal"].with_curl = True # for xml support
        self.options["gdal"].shared = True
        if self.settings.os == "Linux":
            self.options["wxwidgets"].webview = False  # webview control isn't available on linux.

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
            if self.options.tests:
                self.copy("*", dst="bin", src="res", root_package="proj")

        # Copy gdal library data
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            self.copy("*", dst="share", src="res", root_package="gdal")
        if self.settings.os == "Macos":
            self.copy("*", dst="bin/AtmoSwing.app/Contents/share/proj", src="res", root_package="gdal")
            if self.options.tests:
                self.copy("*", dst="bin", src="res", root_package="gdal")

    def build(self):
        cmake = CMake(self)
        if self.options.tests:
            cmake.definitions["BUILD_TESTS"] = "ON"
        if self.options.code_coverage:
            cmake.definitions["BUILD_TESTS"] = "ON"
            cmake.definitions["USE_CODECOV"] = "ON"
        if self.options.with_gui:
            cmake.definitions["USE_GUI"] = "ON"
        else:
            cmake.definitions["USE_GUI"] = "OFF"
        cmake.configure()
        cmake.build()
        if self.settings.os == "Macos":
            cmake.install()
