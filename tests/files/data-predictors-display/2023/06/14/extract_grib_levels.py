import pygrib

# Open the original GRIB2 file
original_grbs = pygrib.open('2023061400.NWS_GFS.hgt.024.grib2')

# Find the specific pressure levels you want to extract
target_pressure_levels = [1000, 500]  # Example: 1000 hPa and 500 hPa

# Create a new GRIB2 file to save the extracted data
output_file = 'extracted_data.grib2'
output_grbs = open(output_file,'wb')

# Loop through the GRIB2 messages in the original file
for original_grb in original_grbs:
    print(original_grb.level)
    if original_grb.level in target_pressure_levels:
        # Copy the GRIB2 message to the new file
        output_grbs.write(original_grb.tostring())

# Close the GRIB2 files
original_grbs.close()
output_grbs.close()
