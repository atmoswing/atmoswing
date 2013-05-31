#
# Find NetCDF include directories and libraries
#
# NetCDF_INCLUDE_DIRECTORIES - where to find netcdf.h
# NetCDF_LIBRARIES           - list of libraries to link against when using NetCDF
# NetCDF_FOUND               - Do not attempt to use NetCDF if "no", "0", or undefined.


find_path( NetCDF_INCLUDE_DIRECTORIES netcdf.h
  /usr/local/include
  /usr/include
)

find_library( NetCDF_C_LIBRARY
  NAMES netcdf libnetcdf
  /usr/local/lib64
  /usr/lib64
  /usr/lib64/netcdf-3
  /usr/local/lib
  /usr/lib
  /usr/lib/netcdf-3
)

set( NetCDF_LIBRARIES
  ${NetCDF_C_LIBRARY}
)

if ( NetCDF_INCLUDE_DIRECTORIES AND NetCDF_LIBRARIES )
  set( NetCDF_FOUND 1 )
else ( NetCDF_INCLUDE_DIRECTORIES AND NetCDF_LIBRARIES )
  set( NetCDF_FOUND 0 )
  
  if (NetCDF_INCLUDE_DIRECTORIES)
    MESSAGE("NetCDF_INCLUDE_DIRECTORIES found")
  else (NetCDF_INCLUDE_DIRECTORIES)
	MESSAGE("NetCDF_INCLUDE_DIRECTORIES not found")
  endif (NetCDF_INCLUDE_DIRECTORIES)
  
  if (NetCDF_LIBRARIES)
    MESSAGE("NetCDF_LIBRARIES found")
  else (NetCDF_LIBRARIES)
	MESSAGE("NetCDF_LIBRARIES not found")
  endif (NetCDF_LIBRARIES)
  
endif ( NetCDF_INCLUDE_DIRECTORIES AND NetCDF_LIBRARIES )

mark_as_advanced(
  NetCDF_PREFIX
)

