from conans import ConanFile, CMake
import os


class AmtoSwing(ConanFile):

    def configure(self):
        if not self.options.with_gui:
            self.options.test_gui = False
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
            self.copy("*", dst="bin/AtmoSwing.app/Contents/share/proj", src="res",
                      root_package="proj")
            if self.options.enable_tests:
                self.copy("*", dst="bin", src="res", root_package="proj")

        # Copy eccodes library data
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            self.copy("*", dst="share/eccodes", src="share/eccodes",
                      root_package="eccodes")
        if self.settings.os == "Macos":
            self.copy("*", dst="bin/AtmoSwing.app/Contents/share/eccodes",
                      src="share/eccodes", root_package="eccodes")
            if self.options.enable_tests:
                self.copy("*", dst="bin", src="share/eccodes", root_package="eccodes")

        # Copy data files
        _source_folder = os.path.join(os.getcwd(), "..")
        if self.settings.os == "Windows" or self.settings.os == "Linux":
            self.copy("*", dst="share/atmoswing",
                      src=os.path.join(_source_folder, "data"))
        if self.settings.os == "Macos":
            self.copy("*", dst="bin/AtmoSwing.app/Contents/share/atmoswing",
                      src=os.path.join(_source_folder, "data"))

        # Copy translation files
        _source_folder = os.path.join(os.getcwd(), "..")
        if self.settings.os == "Windows":
            self.copy("*.mo", dst="bin/fr",
                      src=os.path.join(_source_folder, "locales/fr"))
        if self.settings.os == "Linux":
            self.copy("*.mo", dst="bin/share/locale/fr/LC_MESSAGES/",
                      src=os.path.join(_source_folder, "locales/fr"))
        if self.settings.os == "Macos":
            self.copy("*.mo", dst="bin/AtmoSwing.app/Contents/Resources/fr.lproj",
                      src=os.path.join(_source_folder, "locales/fr"))
