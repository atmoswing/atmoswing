
# Project name
project(g2clib)


##### SOURCE FILES

# Grib2c library
file(GLOB_RECURSE src_lib_g2clib_h src/*.h)
set(src_lib_g2clib_c
        src/gridtemplates.c
        src/drstemplates.c
        src/pdstemplates.c
        src/gbits.c
        src/g2_unpack1.c
        src/g2_unpack2.c
        src/g2_unpack3.c
        src/g2_unpack4.c
        src/g2_unpack5.c
        src/g2_unpack6.c
        src/g2_unpack7.c
        src/g2_free.c
        src/g2_info.c
        src/g2_getfld.c
        src/simunpack.c
        src/comunpack.c
        src/pack_gp.c
        src/reduce.c
        src/specpack.c
        src/specunpack.c
        src/rdieee.c
        src/mkieee.c
        src/int_power.c
        src/simpack.c
        src/compack.c
        src/cmplxpack.c
        src/misspack.c
        src/jpcpack.c
        src/jpcunpack.c
        src/pngpack.c
        src/pngunpack.c
        src/dec_jpeg2000.c
        src/enc_jpeg2000.c
        src/dec_png.c
        src/enc_png.c
        src/g2_create.c
        src/g2_addlocal.c
        src/g2_addgrid.c
        src/g2_addfield.c
        src/g2_gribend.c
        src/getdim.c
        src/g2_miss.c
        src/getpoly.c
        src/seekgb.c
        )
list(APPEND src_lib_g2clib ${src_lib_g2clib_h} ${src_lib_g2clib_c})


##### LIBRARIES

add_library(g2clib STATIC ${src_lib_g2clib})
include_directories("src")


##### DEFINITIONS

# Specific definitions
set_target_properties(g2clib PROPERTIES COMPILE_DEFINITIONS "USE_JPEG2000; USE_PNG")

remove_definitions(-std=c++11)
