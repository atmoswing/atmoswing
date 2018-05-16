
# Project name
project(g2clib)


##### SOURCE FILES

# Grib2c library
file(GLOB_RECURSE src_lib_g2clib_h g2clib/*.h)
set(src_lib_g2clib_c
        g2clib/gridtemplates.c
        g2clib/drstemplates.c
        g2clib/pdstemplates.c
        g2clib/gbits.c
        g2clib/g2_unpack1.c
        g2clib/g2_unpack2.c
        g2clib/g2_unpack3.c
        g2clib/g2_unpack4.c
        g2clib/g2_unpack5.c
        g2clib/g2_unpack6.c
        g2clib/g2_unpack7.c
        g2clib/g2_free.c
        g2clib/g2_info.c
        g2clib/g2_getfld.c
        g2clib/simunpack.c
        g2clib/comunpack.c
        g2clib/pack_gp.c
        g2clib/reduce.c
        g2clib/specpack.c
        g2clib/specunpack.c
        g2clib/rdieee.c
        g2clib/mkieee.c
        g2clib/int_power.c
        g2clib/simpack.c
        g2clib/compack.c
        g2clib/cmplxpack.c
        g2clib/misspack.c
        g2clib/jpcpack.c
        g2clib/jpcunpack.c
        g2clib/pngpack.c
        g2clib/pngunpack.c
        g2clib/dec_jpeg2000.c
        g2clib/enc_jpeg2000.c
        g2clib/dec_png.c
        g2clib/enc_png.c
        g2clib/g2_create.c
        g2clib/g2_addlocal.c
        g2clib/g2_addgrid.c
        g2clib/g2_addfield.c
        g2clib/g2_gribend.c
        g2clib/getdim.c
        g2clib/g2_miss.c
        g2clib/getpoly.c
        g2clib/seekgb.c
        )
list(APPEND src_lib_g2clib ${src_lib_g2clib_h} ${src_lib_g2clib_c})


##### LIBRARIES

add_library(g2clib STATIC ${src_lib_g2clib})


##### DEFINITIONS

# Specific definitions
set_target_properties(g2clib PROPERTIES COMPILE_DEFINITIONS "USE_JPEG2000; USE_PNG")

remove_definitions(-std=c++11)
