
# Project name
project(g2clib)


##### SOURCE FILES

# Grib2c library
file(GLOB_RECURSE src_lib_g2clib_h libs/g2clib/*.h)
set(src_lib_g2clib_c
        libs/g2clib/gridtemplates.c
        libs/g2clib/drstemplates.c
        libs/g2clib/pdstemplates.c
        libs/g2clib/gbits.c
        libs/g2clib/g2_unpack1.c
        libs/g2clib/g2_unpack2.c
        libs/g2clib/g2_unpack3.c
        libs/g2clib/g2_unpack4.c
        libs/g2clib/g2_unpack5.c
        libs/g2clib/g2_unpack6.c
        libs/g2clib/g2_unpack7.c
        libs/g2clib/g2_free.c
        libs/g2clib/g2_info.c
        libs/g2clib/g2_getfld.c
        libs/g2clib/simunpack.c
        libs/g2clib/comunpack.c
        libs/g2clib/pack_gp.c
        libs/g2clib/reduce.c
        libs/g2clib/specpack.c
        libs/g2clib/specunpack.c
        libs/g2clib/rdieee.c
        libs/g2clib/mkieee.c
        libs/g2clib/int_power.c
        libs/g2clib/simpack.c
        libs/g2clib/compack.c
        libs/g2clib/cmplxpack.c
        libs/g2clib/misspack.c
        libs/g2clib/jpcpack.c
        libs/g2clib/jpcunpack.c
        libs/g2clib/pngpack.c
        libs/g2clib/pngunpack.c
        libs/g2clib/dec_jpeg2000.c
        libs/g2clib/enc_jpeg2000.c
        libs/g2clib/dec_png.c
        libs/g2clib/enc_png.c
        libs/g2clib/g2_create.c
        libs/g2clib/g2_addlocal.c
        libs/g2clib/g2_addgrid.c
        libs/g2clib/g2_addfield.c
        libs/g2clib/g2_gribend.c
        libs/g2clib/getdim.c
        libs/g2clib/g2_miss.c
        libs/g2clib/getpoly.c
        libs/g2clib/seekgb.c
        )
list(APPEND src_lib_g2clib ${src_lib_g2clib_h} ${src_lib_g2clib_c})


##### LIBRARIES

add_library(g2clib STATIC ${src_lib_g2clib})


##### DEFINITIONS

# Specific definitions
set_target_properties(g2clib PROPERTIES COMPILE_DEFINITIONS "USE_JPEG2000; USE_PNG")

remove_definitions(-std=c++11)
