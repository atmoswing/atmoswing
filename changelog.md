# Changelog AtmoSwing

## not released

### Fixed

*   Time units were missing in the predictand db.


## v2.1.7 - NOT RELEASED

### Added

*   Adding path to ecCodes definitions in the preferences.

### Fixed

*   Fixing CUDA options issues.
*   Docker: adding ecCodes definitions path to the environment variables.


## v2.1.6 - 22 Nov 2022

### Added

*   A mini-batch approach has been implemented for Genetic Algorithms.
*   Adding a warning if the preload option if not enabled for calibration.
*   Adding local GFS dataset class to avoid downloading.

### Changed

*   Improvement of time units description in generated nc files.
*   Removing usage of temporary storage in GAs optimization.
*   Curl: disable certificate checks on Windows as not supported.
*   Code formatting.

### Removed

*   Removing the history approach in GAs optimisations (not efficient). 


## v2.1.5 - 11 Oct 2022

### Changed

*   Moved the dependencies management to conan
*   Simplified the areas management by removing the composite approach.
*   Some code clean up and code formatting.
*   Moved all CI workflows to GitHub actions.

### Fixed

*   The weights computed by the optimizer cannot take negative values.
*   Addition of S0 and S1 with normalization by the reference value.
*   GFS urls on nomads have been fixed.
*   Optimizer: fixed convergence check when previous results are loaded.
*   Optimizer: relaxed criteria for convergence in GAs (using tolerance).
*   Optimizer: fixed an issue with latitude values > 90° that were not corrected.


## v2.1.4 - 09 Oct 2020

### Changed

*   Refactored time axis management in predictors loading.

### Fixed

*   Fixed an issue with missing dates in the FVG dataset.
*   Fixed an issue with NaNs in the standardization.


## v2.1.3 - 13 Jul 2020

### Added

*   Addition of Dockerfiles for the creation of Docker images.
*   Addition of the mean and the sd in the parameters for standardisation.
*   Standardisation for operational forecasting.
*   Handling a single 'number of analogs' for operational forecasting.
*   Handling forecasts without reference axis.
*   Addition of the mean of the analogs to the aggregated forecast xml.
*   Addition of the lead time hour to the aggregated results.
*   Addition of synthetic text (csv) export of the forecast.
*   Allowing a percentage of missing predictor data.

### Changed

*   Simplification of resulting parameters storage.
*   Reduction of optimization memory footprint.
*   Reduction of padding in structures to save memory.
*   Disabling using GAs history by default.
*   Changes in dependencies management.
*   Refactoring predictor files download.
*   Enabling optional downloading of predictor files.
*   Standardisation after data loading.
*   Changing output extension of forecast files to nc.
*   Changing the specification of the forecast synthetic exports.
*   Allowing NaNs to propagate in the forecast.
*   Updating to Proj 7.
*   Changing indentation back to 4.

### Fixed

*   Fixing GFS urls.
*   Avoid crash and display an error when desired latitude was not found.
*   Addition of the standardisation specification in resulting parameters.
*   Fixing issue when the reference axis is NaN.
*   Fixing lead times in plots.


## v2.1.2 - 02 Dec 2019

### Added

*   Efficient GPU processing with CUDA.
*   Ability to fully resume optimizations with GAs operators values.
*   Parameters that were already assessed during the optimization are not assessed again.
*   Parameters that are close to other parameters with poor skills are not assessed again.
*   Addition of the Google benchmark framework for testing CUDA implementation.

### Changed

*   The optimization workflow has been simplified.
*   Check for previous optimization convergence before loading data.
*   Transitioning to Google code style.
*   The dataset "NCEP_Reanalysis_v1" has been renamed to "NCEP_R1".
*   The dataset "NCEP_Reanalysis_v2" has been renamed to "NCEP_R2".
*   Some redundant tests have been removed.
*   Addition of tests for the datasets.
*   Changes in some libraries versions.

### Fixed

*   Error with weights in optimization initialization (when all weights = 0).
*   Optimization resuming might have started from an older file.
*   Some (rarely used) variables definition in some reanalyses were wrong.
*   Fixed an issue with latitudes axis when resampling.


## v2.1.1 - 17 Jul 2019

### Added

*   Addition of predictor data dumping to binary files (to reduce RAM usage).
*   Allow loading from dumped predictor data (and keep in RAM).
*   Option for replacing NaNs by -9999 (save processing time).
*   Addition of a bash script to install libraries on a server.

### Changed

*   Refactoring of the CUDA implementation.
*   Updating GFS urls.
*   Improving Eigen usage.
*   Testing downscaling parameters.
*   Speeding up Grib files loading.
*   Adding information to error messages.

### Fixed

*   Fixing Viewer issues to find the forecast files.
*   Fixing missing node (on_mean) when parsing xml files for the calibrator.
*   Fixing a log path issue.
*   Fixing a memory leak due to ecCodes index not deleted.
*   Fixing a bug when interval days are not used.


## v2.1.0 - 23 May 2019

### Added

*   Support of GRIB1 files with ecCodes.
*   Adding a generic NetCDF predictor class.
*   Addition of real gradients processing.
*   Addition of S1 variants: S0 and S2.
*   Addition of other nonspatial criteria.
*   Support of IFS outputs.
*   Addition of the ERA5 dataset.
*   Addition of custom period definition (selection of months).
*   Adding analog dates-only extraction method.
*   Get preloaded data from another level if not available.
*   Adding options for seasons definition.
*   Addition of 2D Gauss function for preditor weighting.
*   Implementing time dimension for grib files.
*   Addition of lightnings data normalization.
*   Adding on-the-fly standardization.
*   Support non trivial time arrays for data loading and missing files.
*   Supporting more complex predictor hours.

### Changed

*   Migrating from g2clib to ecCodes.
*   Improving support for NaNs.
*   Handling resulting files with many stations.
*   Allow Optimizer to not have a validation period.
*   Allow for negative MJD values.
*   Allow for retention of more analogs than the defined number.
*   Adding common variables definitions between predictors.
*   Addition of new reanalyses variables.
*   Allowing different file structures for ERA-interim.
*   Using more C++11 features.
*   Improving GRIB parsing efficiency.
*   Heavy refactoring of the time arrays.
*   Adding command-line logo.
*   Updating the predictand DB tool GUI.
*   Better management of missing files.
*   Getting rid of the pseudo-normalized criteria.
*   Refactoring the management of the time reference.
*   Removing custom exception class.
*   Logs refactoring.
*   Removing call to Forecaster from the Viewer.
*   Improving use of config.
*   Auto generate the dependencies for Debian packages.

### Fixed

*   Fixing minimum domain size for S1 and S2.
*   Fixing time issue with the 6-hrly time step.
*   Fix an issue related to GAs crossover on the criteria.
*   Fixing issue with a Google layer projection.
*   Fix a bug in multithreaded downloads.
*   Fix command line usage of builds with GUIs.


## v2.0.1 - 12 Dec 2018

### Added

*   Adding definition of a continuous validation period.

### Changed

*   Using https on nomads.
*   Setting the installer creation as optional.

### Fixed

*   Fixing About panel size and Ubuntu dependencies.
*   Fixing CMake issues.


## v2.0.0 - 19 Nov 2018

### Added

*   Addition (merge) of the code of the optimization with genetic algorithms to the main repository.
*   Creation of the Downscaler.
*   Addition of the NOAA 20CR-v2c ensemble dataset.
*   Addition of the CERA-20C dataset.
*   Addition of the CMIP5 dataset.
*   Addition of CORDEX data
*   Transforming geopotential into geopotential height.
*   Adding other MTW time steps.
*   Adding an option to specify different time steps for the calibration / archive periods.
*   Adding a time properties to take into account temporal shift in the predictand.
*   Handling of both 3h and 6h ERA-20C dataset.
*   Specification of the number of members in the parameters file.
*   Adding an option to remove duplicate date from members.
*   GFS urls are now configurables.
*   Getting predictor time step from files.
*   Getting the spatial resolution from file.
*   Adding capacity to read some unregistered predictor variables.
*   Adding GAs presets.

### Changed

*   Code moved to GitHub.
*   Adding continuous integration (Travis CI and AppVeyor).
*   Adding code coverage of the tests.
*   New MSI installer with WiX.
*   Getting some libraries through external projects.
*   Simplification of the CRPS calculation.
*   Speeding up data loading.
*   Adding possibility to skip data normalization.
*   Removing the slow coefficient approach in criteria calculation.
*   Removing the slower processing version.
*   Heavy refactoring to simplify class names.
*   Refactoring parameters files.
*   Refactoring processor code.
*   Reduce time for assessing the number of analogues.
*   Improving parameters file parsing.
*   Fix a bug when transforming Geopotential variable.
*   Better acceptance of NaNs in the predictand values.
*   Using initialization lists everywhere.
*   CMake files are now organized by folder.
*   Improving Forecaster messages.
*   Changing the predictor files listing approach.
*   New predictor area implementation.
*   Improving and simplifying GUIs.
*   The predictand DB build tool is accessible from anywhere.
*   Stopping the calculations when there is not enough potential analogs.
*   Limit the relevance map extension.
*   Allowing the duplicate dates by default.
*   Defaulting to 1 member.
*   Saving results from Monte Carlo analysis.

### Fixed

*   Fix archive length check with ensemble datasets.
*   Fixing an issue of grid resolution when loading data.
*   Fix issues with VS.
*   Fixing link issues with cURL on Linux.
*   Fixing new GFS files parsing.
*   Fix compiler warnings under Windows.
*   Correctly closing grib files.
*   Fixing screen resolution issue on Linux.
*   Adding missing CL help entries.
*   Force unlock weights when sum > 1.
*   Fixing Monte Carlo analysis.
*   Fixing background color.


## v1.5.0

### Added

*   Addition of the CFSR v2 dataset.
*   Addition of the MERRA2 dataset.
*   Addition of the JRA-55 subset data.
*   Addition of the JRA-55C subset.
*   Addition of the 20CR v2c dataset.
*   Addition of the ERA-20C dataset.
*   Allow for both relative and absolute paths for predictors.
*   Addition of the possibility to define the station id as parameter.
*   Addition of the addition preprocessing.
*   Addition of the average preprocessing.
*   Addition of the Monte-Carlo approach from the Optimizer.

### Changed

*   Refactoring predictor data classes.
*   Addition of support for the T382 truncature.
*   Renaming level type to product.
*   Split up of the big CMake file in smaller files.
*   Allowing preload of humidity index data.
*   Testing and improving preprocessing methods.
*   Improving preprocessing parameters handling.
*   Refactoring parameters loading.
*   Addition of a tolerance in the area matching.
*   Refactoring Classic Calibration.
*   Refactoring saving and loading results.
*   Addition of compression to optimizer results.
*   Improving handling of Gaussian grids in the classic calibration.
*   Saving both results details of calibration and validation.
*   Predictor file paths can now contain wildcards!
*   Refactoring logging.
*   Improvement of the predictor files lookup.
*   Changes in the "Classic +" method.
*   Better handling of intermediate resulting files.
*   Improving predictor datasets reading.

### Fixed

*   Fix of a bug when the area is 2 points wide.
*   Fix of a bug for regular and irregular grids.
*   Fix of a minor memory leak.
*   Fix some issues related to new predictors.
*   Fix loading of previous runs in the Optimizer.
*   Fix of an issue of precision when looking for time values in an array.


## v1.4.3

### Added

*   The new NCEP R1 archive format is now supported.
*   Preloading of multiple data IDs.
*   Addition of predictor data loading with threads.
*   Handling null pointers in the preloaded data.
*   Adding normalized criteria.
*   Sharing data pointers across analogy steps.
*   Addition of ERA-interim.
*   Improving notifictations when loading failed.
*   NCEP R2 tested.

### Changed

*   Renaming Calibrator into Optimizer.
*   Parsing NaNs as string to handle non-numerical cases for predictands.
*   Migrating from UnitTest++ to Google Test.
*   Skip gradients preprocessing when there are multiple criteria.
*   Using pointers to the parameter sets in order to keep changes in level selection.
*   Replacing ERA40 by ERA-interim.
*   Changes in the reanalysis datasets management.
*   Simplification of the meteorological parameter types.
*   Significant changes in netcdf files loading.
*   Addition of a functionality in the composite areas in order to handle the row lon = 360 = 0 degrees.
*   Addition of a method to remove duplicate row on multiple composites.
*   New management of predictor data for realtime series.
*   Using Grib2c instead of GDAL for Grib files, and data loading refactoring.

### Fixed

*   Fixed unit test issues.
*   Applying code inspection recommendations.
*   Fix of a segmentation fault in the optimizer.


## v1.4.2

### Added

*   Addition of the 300hPa level for GFS.
*   Highlight the optimal method for the station in the distribution plot and the analogs list.

### Changed

*   Newly created workspace now opens automatically.
*   Do not load already loaded forecasts.
*   Do not display the station height when null.
*   Handle file opening when double-clicking.
*   Improving CL usage.
*   Reload forecasts previously processed if an export is required.

### Fixed

*   Removal of a forecast from the GUI now works as expected.
*   Removing Projection specification from WMS files.
*   Past forecasts do load again.
*   Fix of a bug in data preloading.


## v1.4.1

### Added

*   Addition of the export of a synthetic xml file.
*   Addition of a tree control for the forecasts in the viewer.
*   Addition of an automatic methods aggregation in the viewer.
*   Creation of methods IDs.
*   Specification of the station IDs for specific parameters files.
*   New xml format for most files.
*   Addition of the export options to the command line configuration.
*   Addition of an overall progress display.

### Changed

*   Update to the new GFS URLs and format.
*   Adding a message in order to specify the selected models.
*   Removal of the coordinate system specification for the predictors.
*   No need to reload forecasts after processing.
*   Improving the display of former forecast files.
*   TreeCtrl images of different resolutions.
*   Change of every image/icon for a custom design.
*   Full support implemented for high resolution screens.
*   Updating the command line interface.
*   Forcing UTF-8 in the netCDF files.
*   Changing file version specification into major.minor
*   Removing TiCPP in order to use the native xml handling from wxWidgets.
*   Merging asCatalog and asCatalogPredictands.

### Fixed

*   Cleaning up processing and use of quantiles.
*   No need to reload forecasts after processing.
*   Debugging accents issue under Linux.
*   Removing « using namespace std » in order to keep a clean namespace resolution.
*   Removing asFrameXmlEditor.
*   Fix of a crash when no forecast is opened.
*   Replacing printf with wxPrintf.
*   Removing unnecessary .c_str() conversion on strings.
*   Fix of a corruption in the wxFormbuilder project.
*   Debugging netcdf issues under Linux.
*   Fixing namespace issues.


## v1.3.3

### Added

*   Addition of buttons in the viewer frame to go backward and forward in time.
*   Workspaces can now be saved to an xml file.
*   Addition of a wizard to create the workspace.
*   Addition of a control on the changes of the workspace to save before closing.
*   Addition of a configuration option in the forecaster.

### Changed

*   Separation of the preferences.
*   Definition of the preferences in the workspace.
*   Change of the configuration option by using a given batch file.
*   The loading of predictor data has significantly changed.
*   Better handles user errors in the parameters files.
*   Hide the elevation information when not available.
*   Changing the name of the U/V axis into X/Y to help users.

### Fixed

*   Cleanup of the forecaster config options.
*   Cleanup of the calibrator config options.
*   Correction of the path to the WMS layers.
*   Bug fix of unspecified directories for the archive predictors.
*   Limiting the number of parallel downloads.
*   Fix of the cURL hang with parallel downloading.
*   Removal of the definition of the analogs number on the forecast score.
*   Fix of an issue with the colors storage in the workspace.
*   Now keeps the same model selection when opening new forecasts.
*   Now keeps the same lead time when opening new forecasts.


## v1.3.2

### Added

*   Introduction of workspaces for the viewer.
*   Addition of WMS basemaps layers.
*   Merging the two viewer frames into one with a new lead time switcher.
*   Addition of the ability to optimize on multiple time series together.
*   Addition of the CRPS reliability skill score and removal of F0 loading methods.

### Changed

*   Improvement of the rank histogram with bootstraping.
*   Increase of boostraping to 10’000 for the rank histrogram.
*   Reduction in time for the assessment of all scores.
*   Improving performance by reducing reallocation.
*   Changing the MergeCouplesAndMultiply method into FormerHumidityIndex.

### Fixed

*   Fix of the paths for CUDA files.
*   Fix of a linking issue with the viewer.
*   Fix of a bug related to gradient preprocessing in validation.
*   Minor bug fix on the evaluation of all forecasting scores.
*   Removing of the S1 weighting method.
*   Bug fix in the preloading option for the classic calibration parameters.
*   Fix of a bug on the single instance checker.
*   Limitation of the zoom level to avoid the memory issue related to GDAL caching mechanism.


## v1.3.1

### Changed

*   Merge of the various CMake files into one project.

### Fixed

*   Debugging the new build process under Linux.


## v1.3.0

### Added

*   Implementation of GPU processing
*   Addition of a predictand pattern file.
*   Addition of compression to the forecast files.
*   Addition of CPack files.
*   Addition of a unit test on gradients preprocessing.

### Changed

*   The archive and calibration periods can now start in the middle of a year.
*   Better check the requested time limits when loading predictor data.

### Fixed

*   Removing a memory leak when aborting the app initialization.
*   Correction of the data ordering in the forecast results.
*   Bug fix in the time arrays intervals construction.
*   Fix of a bug in the validation processing with a partial final year.
*   Correction of the rank histogram.
*   Reduced cURL outputs and fix of the multithreaded downloads.
*   Adding a missing MSVC dll in the installation package.


## v1.2.0

### Added

*   The predictand DB is now generalized to data other than precipitation.
*   The Forecaster is now working with various predictands.
*   Addition of the Calibrator source code.
*   Addition of the option to build the Forecaster in CL without GUI.
*   Addition of the rank histogram (Talagrand diagram)
*   Addition of CRPS decomposition after Hersbach (2000).
*   Addition of the generation of xml parameters files after the calibration.

### Changed

*   Improvement of the CMake build process.
*   Better management of the NaNs during processing.
*   Significant changes in order to generalize the predictand DB class.
*   The catalogs were removed for the predictors classes and new specific data classes were generated.
*   Removing predictand database properties from parameters for calibration.
*   Changing predictors file names.
*   Changes in unit test filenames for more clarity.
*   Better initialization of the scrolled window.
*   Check fields in the parameters file of the forecaster and the calibrator.
*   Change of the version message in CL.

### Fixed

*   Fix of a change in GDAL regarding GRIB2 files origin.
*   Changing the order of includes in the asFileNetcdf class.
*   Unwanted slashes in paths under Linux were removed.
*   The viewer is now building again.
*   Fix of some bugs in unit tests.
*   Fix of format errors in the GFS urls.
*   Fix of an issue related to preprocessed predictors.
*   Logging of the url was discarded due to formatting issues leading to crashes.
*   Correction of bugs related to unit tests from the calibrator.
*   Fix of errors related to Eigen vectors.
*   Minor memory leaks were removed.
*   Removal of compilation warnings.
*   Casing fixed in the netCDF files.
*   The logging in unit tests was improved.
*   Fix of display issues in the sidebar.
*   Simplification of the time management.
*   Fix of errors related to optional parameters.
*   Removal of false warnings.
*   Resolving some unit tests failures.
*   The precipitation predictand class has been secured for RowMajor and Colmajor.
*   Removing the exhaustive calibration.
*   Removal of intermediate results printing.


## v1.0.3

### Added

*   Export of forecast text files from the time series plot.
*   Possibility to cancel the current forecast processing.
*   Better control of the log targets in the command-line mode.
*   Addition of data preloading functionality and data pointer sharing
*   Preprocessing of the humidity flux and other variables combination.
*   Addition of multithreading in the 2nd and following levels of analogy.
*   Addition of functionalities to the threads manager.
*   Handling of the NCEP reanalysis 2 dataset.
*   Handling of the NOAA OI-SST dataset and addition of adapted criteria.
*   Addition of the possibility to account for an axis shift in the predictor dataset.
*   Addition of the others predictand and creation of a generic instance function.
*   Addition of an option to stop calculation when there is NaN in data.
*   Addition of bad allocation catching.

### Changed

*   Faster check of previously existing forecast files: load predictand DB only when needed.
*   Change from q30 to q20 in the precipitation distribution
*   Display of the considered quantile and return period for the alarms panel
*   Better frame size restoration with maximization detection.
*   Data import from netCDF files is less sensitive to the data type.
*   Much faster import of forecast files.
*   Some clean-up of unused code.
*   Simplification of the file names of intermediate results.
*   Better management of the threads.
*   Improvement of the multithreading option management.
*   Better clean-up after processing.
*   Addition of typedefs.
*   Creation of 2 separate log files for the viewer and the forecaster.
*   Improvement of the CMake files.
*   Small improvements to the time series plots.
*   Insertion of many supplementary assertions.
*   Clean-up of config paths default values.

### Fixed

*   An error in the proxy port was fixed.
*   Preference « /Plot/PastDaysNb » was sometimes 3 or 5. Set 3 everywhere.
*   Do not load the same past forecasts twice in time series plots.
*   The forecasting launch from the viewer has been fixed.
*   Removal of the message box in the CL forecast.
*   Addition of a critical section on the config pointer.
*   Addition of critical sections for TiCPP.
*   Addition of critical sections for NetCDF.
*   Coordinates automatic fix was bugged in the parameters class.
*   Fix of a bug when trying to sort array with size of 1.
*   Bug fix in temporary file names creation.
*   Bug fixed in the enumeration of units
*   NetCDF file class may have badly estimated the array size.
*   Fix of memory filling by logging in the time array class.
