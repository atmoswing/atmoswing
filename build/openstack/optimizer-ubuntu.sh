# Base on Ubuntu 22.07
sudo apt-get install -y cmake

conan profile new default --detect
conan profile update settings.compiler.libcxx=libstdc++11 default
conan remote add gitlab https://gitlab.com/api/v4/packages/conan

git clone https://github.com/atmoswing/atmoswing.git

cd atmoswing
mkdir bin
cd bin

conan install .. -s build_type=Release --build=missing --build=openjpeg -o enable_tests=True -o with_gui=False  \
        -o build_forecaster=False -o build_viewer=False -o build_optimizer=True -o build_downscaler=False

conan build ..
