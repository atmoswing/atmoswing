echo Running cmake...
mkdir bin
cd bin
cmake -G "Visual Studio 15 2017 Win64" -DBUILD_FORECASTER=ON -DBUILD_OPTIMIZER=ON -DBUILD_DOWNSCALER=ON -DBUILD_VIEWER=OFF -DCREATE_INSTALLER=ON -DUSE_GUI=OFF -DCMAKE_PREFIX_PATH="C:\projects\libs" -DwxWidgets_CONFIGURATION=baseu ..
cmake --build . --config release