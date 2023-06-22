Use Poedit for translations.

1. Extract text using xgettext (also shipped with Poedit)
   - In a prompt, go to AtmoSwing's root directory
   - Run, for example under Windows:
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_viewer/core/*.cpp --keyword="_" --output="locales/app_viewer_core.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_viewer/gui/*.cpp --keyword="_" --output="locales/app_viewer_gui.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_downscaler/core/*.cpp --keyword="_" --output="locales/app_downscaler_core.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_downscaler/gui/*.cpp --keyword="_" --output="locales/app_downscaler_gui.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_forecaster/core/*.cpp --keyword="_" --output="locales/app_forecaster_core.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_forecaster/gui/*.cpp --keyword="_" --output="locales/app_forecaster_gui.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_optimizer/core/*.cpp --keyword="_" --output="locales/app_optimizer_core.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/app_optimizer/gui/*.cpp --keyword="_" --output="locales/app_optimizer_gui.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/shared_base/core/*.cpp --keyword="_" --output="locales/shared_base_core.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/shared_base/gui/*.cpp --keyword="_" --output="locales/shared_base_gui.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" src/shared_processing/core/*.cpp --keyword="_" --output="locales/shared_processing_core.pot" --c++
   &"C:\Program Files (x86)\Poedit\GettextTools\bin\xgettext.exe" cmake-build-debug/_deps/vroomgis-src/vroomgis/src/*.cpp --keyword="_" --output="locales/lib_vroomgis.pot" --c++
