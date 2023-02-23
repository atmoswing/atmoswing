from conan import ConanFile
from conan.tools.files import copy
from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake, cmake_layout


class AmtoSwing(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "enable_tests": [True, False],
        "enable_benchmark": [True, False],
        "code_coverage": [True, False],
        "with_gui": [True, False],
        "use_cuda": [True, False],
        "build_forecaster": [True, False],
        "build_viewer": [True, False],
        "build_optimizer": [True, False],
        "build_downscaler": [True, False],
        "create_installer": [True, False],
    }
    default_options = {
        "enable_tests": True,
        "enable_benchmark": False,
        "code_coverage": False,
        "with_gui": True,
        "use_cuda": False,
        "build_forecaster": True,
        "build_viewer": True,
        "build_optimizer": True,
        "build_downscaler": True,
        "create_installer": False,
    }

    generators = "cmake", "gcc"

    def requirements(self):
        self.requires("proj/9.1.1")
        self.requires("libcurl/7.87.0")
        #self.requires("libtiff/4.4.0")
        #self.requires("sqlite3/3.40.1")
        self.requires("eigen/3.4.0")
        self.requires("netcdf/4.8.1")
        #self.requires("libdeflate/1.17")
        #self.requires("libjpeg/9e")
        #self.requires("zlib/1.2.13")
        #self.requires("libpng/1.6.39")
        self.requires("eccodes/2.27.0@terranum-conan+eccodes/stable")
        if self.options.enable_tests or self.options.code_coverage:
            self.test_requires("gtest/1.13.0")
        if self.options.enable_benchmark:
            self.test_requires("benchmark/1.7.1")
            self.test_requires("gtest/1.13.0")
        if self.options.with_gui:
            self.requires("wxwidgets/3.2.1@terranum-conan+wxwidgets/stable")
        else:
            self.requires("wxbase/3.2.1@terranum-conan+wxbase/stable")
        if self.options.build_viewer:
            self.requires("gdal/3.5.2")
            #self.requires("gdal/3.5.1@terranum-conan+gdal/stable")

    def configure(self):
        if self.options.code_coverage:
            self.options.enable_tests = True
        self.options["gdal"].with_curl = True # for xml support
        self.options["gdal"].shared = True
        if not self.options.with_gui:
            self.options["wxbase"].xml = True
            self.options["wxbase"].sockets = True

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.definitions["BUILD_TESTS"] = self.options.enable_tests
        tc.definitions["BUILD_BENCHMARK"] = self.options.enable_benchmark
        tc.definitions["USE_CODECOV"] = self.options.code_coverage
        tc.definitions["USE_GUI"] = self.options.with_gui
        tc.definitions["USE_CUDA"] = self.options.use_cuda
        tc.definitions["BUILD_FORECASTER"] = self.options.build_forecaster
        tc.definitions["BUILD_VIEWER"] = self.options.build_viewer
        tc.definitions["BUILD_OPTIMIZER"] = self.options.build_optimizer
        tc.definitions["BUILD_DOWNSCALER"] = self.options.build_downscaler
        tc.definitions["CREATE_INSTALLER"] = self.options.create_installer

        if self.options.code_coverage:
            tc.definitions["BUILD_TESTS"] = "ON"
        if self.options.build_viewer:
            self.options.with_gui = True
            tc.definitions["USE_GUI"] = "ON"
        if self.settings.os == "Windows":
            if self.options.with_gui:
                tc.definitions["wxWidgets_CONFIGURATION"] = "mswu"
            else:
                tc.definitions["wxWidgets_CONFIGURATION"] = "baseu"

        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)

        #cmake.configure()
        cmake.build()
        if self.settings.os == "Macos":
            cmake.install()

    def imports(self):
        # Copy libraries
        if self.settings.os == "Windows":
            copy(self, "*.dll", dst="bin", src="@bindirs")
        if self.settings.os == "Macos":
            copy(self, "*.dylib", dst="bin", src="@libdirs")
        if self.settings.os == "Linux":
            copy(self, "*.so*", dst="bin", src="@libdirs")

        # Copy proj library data
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            copy(self, "*", dst="share/proj", src="res", root_package="proj")
        if self.settings.os == "Macos":
            copy(self, "*", dst="bin/AtmoSwing.app/Contents/share/proj", src="res", root_package="proj")
            if self.options.enable_tests:
                copy(self, "*", dst="bin", src="res", root_package="proj")

        # Copy eccodes library data
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            copy(self, "*", dst="share/eccodes", src="share/eccodes", root_package="eccodes")
        if self.settings.os == "Macos":
            copy(self, "*", dst="bin/AtmoSwing.app/Contents/share/eccodes", src="share/eccodes", root_package="eccodes")
            if self.options.enable_tests:
                copy(self, "*", dst="bin", src="share/eccodes", root_package="eccodes")

