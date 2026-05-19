/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 * Portions Copyright 2017-2019 Pascal Horton, University of Bern.
 */

#include "asPredictor.h"

#include <wx/dir.h>
#include <wx/ffile.h>

#include "asAreaGridGeneric.h"
#include "asAreaGridRegular.h"
#include "asPredictorCustomLtheNR1.h"
#include "asPredictorCustomMFvgMeso.h"
#include "asPredictorCustomMFvgMesoPacked.h"
#include "asPredictorCustomMFvgSynop.h"
#include "asPredictorCustomMFvgSynopPacked.h"
#include "asPredictorCustomUnilNR1.h"
#include "asPredictorCustomUnilOisst2.h"
#include "asPredictorEcmwfCera20C.h"
#include "asPredictorEcmwfEra20C.h"
#include "asPredictorEcmwfEra5.h"
#include "asPredictorEcmwfEraInterim.h"
#include "asPredictorEcmwfIfs.h"
#include "asPredictorGeneric.h"
#include "asPredictorJmaJra55CSubset.h"
#include "asPredictorJmaJra55Subset.h"
#include "asPredictorNasaMerra2.h"
#include "asPredictorNasaMerra2Subset.h"
#include "asPredictorNcepCfsr.h"
#include "asPredictorNcepCfsrSubset.h"
#include "asPredictorNcepR1.h"
#include "asPredictorNcepR2.h"
#include "asPredictorNoaa20Cr2c.h"
#include "asPredictorNoaa20Cr2cEnsemble.h"
#include "asPredictorNoaaOisst2.h"
#include "asTimeArray.h"

asPredictor::asPredictor(const wxString& dataId)
    : _fileType(asFile::Netcdf),
      _initialized(false),
      _standardized(false),
      _axesChecked(false),
      _wasDumped(false),
      _dataId(dataId),
      _parameter(ParameterUndefined),
      _gribCode({asNOT_FOUND, asNOT_FOUND, asNOT_FOUND, asNOT_FOUND}),
      _unit(UnitUndefined),
      _strideAllowed(false),
      _level(0),
      _membersNb(1),
      _latPtsnb(0),
      _lonPtsnb(0),
      _isLatLon(true),
      _isPreprocessed(false),
      _isEnsemble(false),
      _canBeClipped(true),
      _parseTimeReference(false),
      _warnMissingFiles(true),
      _warnMissingLevels(true),
      _percentMissingAllowed(5) {
    _fStr.hasLevelDim = true;
    _fStr.singleLevel = false;
    _fStr.singleTimeStep = false;
    _fStr.timeStep = 0;
    _fInd.memberStart = 0;
    _fInd.memberCount = 1;
    _fInd.latStep = 0;
    _fInd.lonStep = 0;
    _fInd.level = 0;
    _fInd.timeStartFile = 0;
    _fInd.timeStartStorage = 0;
    _fInd.timeCountFile = 0;
    _fInd.timeCountStorage = 0;
    _fInd.timeConsistent = true;
    _fInd.timeStep = 0;

    if (dataId.Contains('/')) {
        wxString levelType = dataId.BeforeLast('/');
        _product = levelType;
        _dataId = dataId.AfterLast('/');
    } else {
        wxLogVerbose(_("The data ID (%s) does not contain the level type"), dataId);
    }
}

asPredictor* asPredictor::GetInstance(const wxString& datasetId, const wxString& dataId, const wxString& directory) {
    asPredictor* predictor = nullptr;

    if (datasetId.StartsWith("Generic") || datasetId.StartsWith("generic")) {
        predictor = new asPredictorGeneric(dataId);
        predictor->SetDatasetId(datasetId);
    } else if (datasetId.IsSameAs("NCEP_R1", false)) {
        predictor = new asPredictorNcepR1(dataId);
    } else if (datasetId.IsSameAs("NCEP_R2", false)) {
        predictor = new asPredictorNcepR2(dataId);
    } else if (datasetId.IsSameAs("NCEP_CFSR", false)) {
        predictor = new asPredictorNcepCfsr(dataId);
    } else if (datasetId.IsSameAs("NCEP_CFSR_subset", false)) {
        predictor = new asPredictorNcepCfsrSubset(dataId);
    } else if (datasetId.IsSameAs("ECMWF_ERA_interim", false)) {
        predictor = new asPredictorEcmwfEraInterim(dataId);
    } else if (datasetId.IsSameAs("ECMWF_ERA5", false)) {
        predictor = new asPredictorEcmwfEra5(dataId);
    } else if (datasetId.IsSameAs("ECMWF_ERA_20C", false)) {
        predictor = new asPredictorEcmwfEra20C(dataId);
    } else if (datasetId.IsSameAs("ECMWF_CERA_20C", false)) {
        predictor = new asPredictorEcmwfCera20C(dataId);
    } else if (datasetId.IsSameAs("ECMWF_IFS", false)) {
        predictor = new asPredictorEcmwfIfs(dataId);
    } else if (datasetId.IsSameAs("NASA_MERRA_2", false)) {
        predictor = new asPredictorNasaMerra2(dataId);
    } else if (datasetId.IsSameAs("NASA_MERRA_2_subset", false)) {
        predictor = new asPredictorNasaMerra2Subset(dataId);
    } else if (datasetId.IsSameAs("JMA_JRA_55_subset", false)) {
        predictor = new asPredictorJmaJra55Subset(dataId);
    } else if (datasetId.IsSameAs("JMA_JRA_55C_subset", false)) {
        predictor = new asPredictorJmaJra55CSubset(dataId);
    } else if (datasetId.IsSameAs("NOAA_20CR_v2c", false)) {
        predictor = new asPredictorNoaa20Cr2c(dataId);
    } else if (datasetId.IsSameAs("NOAA_20CR_v2c_ens", false)) {
        predictor = new asPredictorNoaa20Cr2cEnsemble(dataId);
    } else if (datasetId.IsSameAs("NOAA_OISST_v2", false)) {
        predictor = new asPredictorNoaaOisst2(dataId);
    } else if (datasetId.IsSameAs("Custom_Unil_NR1", false)) {
        predictor = new asPredictorCustomUnilNR1(dataId);
    } else if (datasetId.IsSameAs("Custom_Unil_OISST_v2", false)) {
        predictor = new asPredictorCustomUnilOisst2(dataId);
    } else if (datasetId.IsSameAs("Custom_LTHE_NR1", false)) {
        predictor = new asPredictorCustomLtheNR1(dataId);
    } else if (datasetId.IsSameAs("Custom_MeteoFVG_synop", false)) {
        predictor = new asPredictorCustomMFvgSynop(dataId);
    } else if (datasetId.IsSameAs("Custom_MeteoFVG_meso", false)) {
        predictor = new asPredictorCustomMFvgMeso(dataId);
    } else if (datasetId.IsSameAs("Custom_MeteoFVG_synop_packed", false)) {
        predictor = new asPredictorCustomMFvgSynopPacked(dataId);
    } else if (datasetId.IsSameAs("Custom_MeteoFVG_meso_packed", false)) {
        predictor = new asPredictorCustomMFvgMesoPacked(dataId);
    } else {
        wxLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return nullptr;
    }

    if (!directory.IsEmpty()) {
        predictor->SetDirectoryPath(directory);
    }

    if (!predictor->Init()) {
        wxLogError(_("The predictor did not initialize correctly."));
        return nullptr;
    }

    return predictor;
}

bool asPredictor::Init() {
    return false;
}

bool asPredictor::SetData(vva2f& val) {
    wxASSERT(_time.size() > 0);
    wxASSERT((int)_time.size() == (int)val.size());

    _latPtsnb = (int)val[0][0].rows();
    _lonPtsnb = (int)val[0][0].cols();
    _membersNb = (int)val[0].size();
    _data.clear();
    _data.reserve(_time.size() * val[0].size() * _latPtsnb * _lonPtsnb);
    _data = val;

    return true;
}

void asPredictor::DumpData() {
    _wasDumped = true;
    _data.clear();
}

bool asPredictor::SaveDumpFile() {
    wxASSERT(_time.size() > 0);
    wxASSERT(!_data.empty());

    wxString filePath = GetDumpFileName();

    wxFFile file(filePath, "wb");

    if (!file.IsOpened()) {
        wxLogError(_("Failed creating the file %s"), filePath);
        return false;
    }

    file.Write(&_latPtsnb, sizeof(int));
    file.Write(&_lonPtsnb, sizeof(int));

    int nLats = _axisLat.size();
    int nLons = _axisLon.size();

    file.Write(&nLats, sizeof(int));
    file.Write(&nLons, sizeof(int));

    file.Write(&_axisLat[0], nLats * sizeof(double));
    file.Write(&_axisLon[0], nLons * sizeof(double));

    size_t size = _time.size() * _membersNb * _latPtsnb * _lonPtsnb * sizeof(float);

    a2f data(_time.size() * _membersNb * _latPtsnb, _lonPtsnb);

    for (int t = 0; t < _data.size(); ++t) {
        for (int m = 0; m < _membersNb; ++m) {
            int l = t * _membersNb * _latPtsnb + m * _latPtsnb;
            data.block(l, 0, _latPtsnb, _lonPtsnb) = _data[t][m];
        }
    }

    if (file.Write(&data(0, 0), size) != size) {
        wxLogError(_("Failed writing the file %s"), filePath);
        return false;
    }

    if (!file.Close()) {
        wxLogError(_("Failed closing the file %s"), filePath);
        return false;
    }

    return true;
}

bool asPredictor::LoadDumpedData() {
    wxASSERT(_time.size() > 0);
    wxASSERT(_data.empty());

    wxString filePath = GetDumpFileName();

    wxFFile file(filePath, "rb");

    if (!file.IsOpened()) {
        wxLogError(_("Failed opening the file %s"), filePath);
        return false;
    }

    file.Read(&_latPtsnb, sizeof(int));
    file.Read(&_lonPtsnb, sizeof(int));

    int nLats, nLons;
    file.Read(&nLats, sizeof(int));
    file.Read(&nLons, sizeof(int));

    _axisLat.resize(nLats);
    _axisLon.resize(nLons);

    file.Read(&_axisLat[0], nLats * sizeof(double));
    file.Read(&_axisLon[0], nLons * sizeof(double));

    _data.resize(_time.size(), vector<a2f, Eigen::aligned_allocator<a2f>>(_membersNb, a2f(_latPtsnb, _lonPtsnb)));
    size_t size = _time.size() * _membersNb * _latPtsnb * _lonPtsnb * sizeof(float);

    a2f data(_time.size() * _membersNb * _latPtsnb, _lonPtsnb);
    file.Read(&data(0, 0), size);

    for (int t = 0; t < _data.size(); ++t) {
        for (int m = 0; m < _membersNb; ++m) {
            int l = t * _membersNb * _latPtsnb + m * _latPtsnb;
            _data[t][m] = data.block(l, 0, _latPtsnb, _lonPtsnb);
        }
    }

    if (!file.Close()) {
        wxLogError(_("Failed closing the file %s"), filePath);
        return false;
    }

    wxASSERT(!_data.empty());

    _wasDumped = false;

    return true;
}

bool asPredictor::DumpFileExists() const {
    return wxFileExists(GetDumpFileName());
}

wxString asPredictor::GetDumpFileName() const {
    wxString fileName(_datasetId + '-' + _dataId + '-');
    fileName << CreateHash();
    fileName << ".tmp";

    wxString dir = asConfig::GetUserDataDir() + "Temp";

    wxString filePath = dir + DS + fileName;
    if (!wxDir::Exists(dir)) {
        wxDir::Make(dir, wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);
    }

    return filePath;
}

size_t asPredictor::CreateHash() const {
    wxString hash;
    hash << _standardized;
    hash << _product;
    hash << _strideAllowed;
    hash << _level;
    hash << _time[0];
    hash << _time[_time.size() - 1];
    hash << _time.size();
    hash << _membersNb;
    hash << _isLatLon;
    hash << _isPreprocessed;
    hash << _isEnsemble;
    hash << _canBeClipped;
    hash << _preprocessMethod;

    std::size_t h = std::hash<std::string>{}(std::string(hash.mb_str()));

    return h;
}

bool asPredictor::CheckFilesPresence() {
    if (_files.empty()) {
        wxLogError(_("Empty files list for %s (%s)."), _dataId, _datasetName);
        return false;
    }

    int nbDirsToRemove = 0;
    int countMissing = 0;

    for (int i = 0; i < _files.size(); i++) {
        if (i > 0 && nbDirsToRemove > 0) {
            wxFileName fileName(_files[i]);
            for (int j = 0; j < nbDirsToRemove; ++j) {
                fileName.RemoveLastDir();
            }
            _files[i] = fileName.GetFullPath();
        }

        if (!wxFile::Exists(_files[i])) {
            // Search recursively in the parent directory
            wxFileName fileName(_files[i]);
            while (true) {
                // Check for wildcards
                if (wxIsWild(fileName.GetPath())) {
                    wxLogError(_("No wildcard is yet authorized in the path (%s)"), fileName.GetPath());
                    return false;
                } else if (wxIsWild(fileName.GetFullName())) {
                    wxArrayString files;
                    size_t nb = wxDir::GetAllFiles(fileName.GetPath(), &files, fileName.GetFullName());
                    if (nb == 1) {
                        _files[i] = files[0];
                        break;
                    } else if (nb > 1) {
                        wxLogError(_("Multiple files were found matching the name %s:"), fileName.GetFullName());
                        for (int j = 0; j < nb; ++j) {
                            wxLogError(files[j]);
                        }
                        return false;
                    }
                }

                if (i == 0) {
                    if (fileName.GetDirCount() < 2) {
                        wxLogError(_("File not found: %s"), _files[i]);
                        return false;
                    }

                    fileName.RemoveLastDir();
                    nbDirsToRemove++;
                    if (fileName.Exists()) {
                        _files[i] = fileName.GetFullPath();
                        break;
                    }
                } else {
                    if (_warnMissingFiles) {
                        wxLogWarning(_("File not found: %s"), _files[i]);
                    } else {
                        wxLogVerbose(_("File not found: %s"), _files[i]);
                    }
                    _files[i] = wxEmptyString;
                    countMissing++;
                    break;
                }
            }
        }
    }

    float percentMissing = 100.0 * float(countMissing) / float(_files.size());
    if (percentMissing > _percentMissingAllowed) {
        wxLogError(_("%.2f percent of the files are missing (%s, %s)."), percentMissing, _datasetId, _dataId);
        return false;
    }

    return true;
}

bool asPredictor::Load(asAreaGrid* desiredArea, asTimeArray& timeArray, float level) {
    _level = level;

    if (!_initialized) {
        if (!Init()) {
            wxLogError(_("Error at initialization of the predictor dataset %s."), _datasetName);
            return false;
        }
    }

    try {
        // List files and check availability
        ListFiles(timeArray);
        if (!CheckFilesPresence()) {
            wxLogError(_("Files not found for %s (%s)."), _dataId, _datasetName);
            return false;
        }
        wxLogVerbose(_("Predictor files found."));

        // Get file axes
        if (!EnquireFileStructure(timeArray)) {
            wxLogError(_("Failing to get the file structure."));
            return false;
        }
        wxLogVerbose(_("File structure parsed."));

        // Check the level availability
        if (!HasDesiredLevel(_warnMissingLevels)) {
            if (_warnMissingLevels) {
                wxLogError(_("Failing to get the desired level."));
            } else {
                wxLogVerbose(_("Failing to get the desired level."));
            }
            return false;
        }

        // Check the time array
        if (!CheckTimeArray(timeArray)) {
            wxLogError(_("The time array is not valid to load data."));
            return false;
        }

        // Create a new area matching the dataset
        asAreaGrid* dataArea = CreateMatchingArea(desiredArea);

        // Store time array
        _time = timeArray.GetTimeArray();
        if (_fStr.timeStep == 0) {
            _fInd.timeStep = 1;
        } else {
            _fInd.timeStep = wxMax(timeArray.GetTimeStepHours() / _fStr.timeStep, 1);
        }

        // Extract data from files
        wxLogVerbose(_("Extracting from files."));
        if (!ExtractFromFiles(dataArea, timeArray)) {
            if (_warnMissingFiles && _warnMissingLevels) {
                wxLogWarning(_("Extracting data from files failed."));
            } else {
                wxLogVerbose(_("Extracting data from files failed."));
            }
            wxDELETE(dataArea);
            return false;
        }

        // Transform data
        wxLogVerbose(_("Transforming data"));
        if (!TransformData()) {
            wxLogError(_("Data transformation has failed."));
            wxFAIL;
            return false;
        }

        // Interpolate the loaded data on the desired grid
        wxLogVerbose(_("Interpolating predictor grid."));
        if (desiredArea && desiredArea->IsRegular() && !InterpolateOnGrid(dataArea, desiredArea)) {
            wxLogError(_("Interpolation failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Check the data container length
        wxLogVerbose(_("Loading forecast data (predictorRealtime->Load)."));
        if (_time.size() > _data.size()) {
            wxLogError(_("The date and the data array lengths do not match (time = %d and data = %d)."),
                       (int)_time.size(), (int)_data.size());
            wxLogError(_("Time array starts on %s and ends on %s."), asTime::GetStringTime(_time[0], ISOdateTime),
                       asTime::GetStringTime(_time[_time.size() - 1], ISOdateTime));
            wxDELETE(dataArea);
            return false;
        }

        wxDELETE(dataArea);
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation (%s) caught when loading data %s (%s)."), msg, _dataId, _datasetName);
        return false;
    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        wxLogError(_("Failed to load data (exception)."));
        return false;
    }

    _membersNb = (int)_data[0].size();

    return true;
}

bool asPredictor::Load(asAreaGrid& desiredArea, asTimeArray& timeArray, float level) {
    return Load(&desiredArea, timeArray, level);
}

bool asPredictor::Load(asAreaGrid& desiredArea, double date, float level) {
    asTimeArray timeArray(date);
    timeArray.Init();

    return Load(&desiredArea, timeArray, level);
}

bool asPredictor::Load(asAreaGrid* desiredArea, double date, float level) {
    asTimeArray timeArray(date);
    timeArray.Init();

    return Load(desiredArea, timeArray, level);
}

void asPredictor::ListFiles(asTimeArray& timeArray) {
    _files = vwxs();
}

bool asPredictor::CheckTimeArray(asTimeArray& timeArray) {
    // Check the time steps
    if ((timeArray.GetTimeStepDays() > 0) && (_fStr.timeStep / 24.0 > timeArray.GetTimeStepDays())) {
        wxLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }

    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays() / (_fStr.timeStep / 24.0), &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }

    fractpart = modf((timeArray.GetStartingHour() - _fStr.firstHour) / _fStr.timeStep, &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired startDate (%gh) is not coherent with the data properties (fractpart = %g)."),
                   timeArray.GetStartingHour(), fractpart);
        return false;
    }

    return true;
}

void asPredictor::ConvertToMjd(a1d& time, double refValue) const {
    wxFAIL;
}

double asPredictor::FixTimeValue(double time) const {
    return time;
}

bool asPredictor::EnquireFileStructure(asTimeArray& timeArray) {
    wxASSERT(_files.size() > 0);

    switch (_fileType) {
        case (asFile::Netcdf): {
            return EnquireNetcdfFileStructure();
        }
        case (asFile::Grib): {
            return EnquireGribFileStructure(timeArray);
        }
        default: {
            wxLogError(_("Predictor file type not correctly defined."));
        }
    }

    return false;
}

bool asPredictor::ExtractFromFiles(asAreaGrid*& dataArea, asTimeArray& timeArray) {
    switch (_fileType) {
        case (asFile::Netcdf): {
            for (const auto& fileName : _files) {
                if (!ExtractFromNetcdfFile(fileName, dataArea, timeArray)) {
                    return false;
                }
            }
            break;
        }
        case (asFile::Grib): {
            for (const auto& fileName : _files) {
                if (!ExtractFromGribFile(fileName, dataArea, timeArray)) {
                    return false;
                }
            }
            break;
        }
        default: {
            wxLogError(_("Predictor file type not correctly defined."));
            return false;
        }
    }

    return true;
}

bool asPredictor::EnquireNetcdfFileStructure() {
    // Open the NetCDF file
    ThreadsManager().CritSectionNetCDF().Enter();
    asFileNetcdf ncFile(_files[0], asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(ncFile)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Close the nc file
    ncFile.Close();
    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asPredictor::ExtractFromNetcdfFile(const wxString& fileName, asAreaGrid*& dataArea, asTimeArray& timeArray) {
    // Open the NetCDF file
    ThreadsManager().CritSectionNetCDF().Enter();
    asFileNetcdf ncFile(fileName, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(ncFile)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Get indexes
    if (!GetAxesIndexes(dataArea, timeArray)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Load data
    if (!GetDataFromFile(ncFile)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Close the nc file
    ncFile.Close();
    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asPredictor::EnquireGribFileStructure(asTimeArray& timeArray) {
    wxASSERT(_files.size() > 0);

    a1d times = timeArray.GetTimeArray();

    // Open Grib files
    ThreadsManager().CritSectionGrib().Enter();
    asFileGrib gbFile0(_files[0], asFileGrib::ReadOnly);

    wxLogVerbose(_("Opening grib file to enquire the structure."));
    if (!gbFile0.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Set index position
    wxLogVerbose(_("Setting index position in the grib file."));
    if (!gbFile0.SetIndexPositionAnyLevel(_gribCode)) {
        gbFile0.Close();
        ThreadsManager().CritSectionGrib().Leave();
        return false;
    }

    // Parse file structure
    if (_fStr.singleTimeStep && _files.size() > 1) {
        wxASSERT(times.size() >= 1);

        wxLogVerbose(_("Creating an instance of the grib object to enquire the structure (2nd file)."));
        asFileGrib gbFile1 = asFileGrib(_files[1], asFileGrib::ReadOnly);

        wxLogVerbose(_("Opening grib file to enquire the structure (2nd file)."));
        if (!gbFile1.Open()) {
            gbFile0.Close();
            ThreadsManager().CritSectionGrib().Leave();
            wxFAIL;
            return false;
        }

        wxLogVerbose(_("Setting index position in the grib file (2nd file)."));
        if (!gbFile1.SetIndexPositionAnyLevel(_gribCode)) {
            gbFile0.Close();
            gbFile1.Close();
            ThreadsManager().CritSectionGrib().Leave();
            wxFAIL;
            return false;
        }

        wxLogVerbose(_("Parsing the grib structure."));
        if (!ParseFileStructure(&gbFile0, &gbFile1)) {
            gbFile0.Close();
            gbFile1.Close();
            ThreadsManager().CritSectionGrib().Leave();
            wxFAIL;
            return false;
        }

        gbFile1.Close();

    } else {
        wxLogVerbose(_("Parsing the grib structure (single file)."));
        if (!ParseFileStructure(&gbFile0)) {
            gbFile0.Close();
            ThreadsManager().CritSectionGrib().Leave();
            wxFAIL;
            return false;
        }
    }

    // Close the nc file
    gbFile0.Close();
    ThreadsManager().CritSectionGrib().Leave();

    return true;
}

bool asPredictor::ExtractFromGribFile(const wxString& fileName, asAreaGrid*& dataArea, asTimeArray& timeArray) {
    // Handle missing file
    if (fileName.IsEmpty()) {
        if (FillWithNaNs()) {
            return true;
        }
        wxFAIL;
        return false;
    }

    // Open the Grib file
    wxLogVerbose(_("Opening the grib file."));
    ThreadsManager().CritSectionGrib().Enter();
    asFileGrib gbFile(fileName, asFileGrib::ReadOnly);
    if (!gbFile.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Set index position
    wxLogVerbose(_("Setting index position in grib file."));
    if (!gbFile.SetIndexPosition(_gribCode, _level, _warnMissingLevels)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        return false;
    }

    // Parse file structure
    wxLogVerbose(_("Parsing grib file structure."));
    if (!ParseFileStructure(&gbFile)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Get indexes
    if (!GetAxesIndexes(dataArea, timeArray)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Load data
    if (!GetDataFromFile(gbFile)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Close the nc file
    gbFile.Close();
    ThreadsManager().CritSectionGrib().Leave();

    return true;
}

bool asPredictor::FillWithNaNs() {
    // Check that it's not the first file
    if (_data.empty()) {
        wxLogError(_("The first file cannot be missing."));
        return false;
    }

    // Check that it's 1 file per time step
    if (_fInd.timeCountFile > 1) {
        wxLogError(_("Missing files are handled only when there is 1 file per time step."));
        return false;
    }

    // Fill with NaNs
    va2f memLatLonData;
    for (int iMem = 0; iMem < _fInd.memberCount; iMem++) {
        a2f latLonData = NAN * a2f::Ones(_data[0][iMem].rows(), _data[0][iMem].cols());
        memLatLonData.push_back(latLonData);
    }
    _data.push_back(memLatLonData);

    return true;
}

bool asPredictor::ParseFileStructure(asFileNetcdf& ncFile) {
    if (!ExtractSpatialAxes(ncFile)) return false;
    if (!ExtractLevelAxis(ncFile)) return false;
    if (!ExtractTimeAxis(ncFile)) return false;

    return CheckFileStructure();
}

bool asPredictor::ExtractTimeAxis(asFileNetcdf& ncFile) {
    _fStr.time = a1d(ncFile.GetVarLength(_fStr.dimTimeName));

    switch (ncFile.GetVarType(_fStr.dimTimeName)) {
        case NC_DOUBLE:
            ncFile.GetVar(_fStr.dimTimeName, &_fStr.time[0]);
            break;
        case NC_FLOAT: {
            a1f axisTimeFloat(ncFile.GetVarLength(_fStr.dimTimeName));
            ncFile.GetVar(_fStr.dimTimeName, &axisTimeFloat[0]);
            for (int i = 0; i < axisTimeFloat.size(); ++i) {
                _fStr.time[i] = (double)axisTimeFloat[i];
            }
        } break;
        case NC_INT: {
            a1i axisTimeInt(ncFile.GetVarLength(_fStr.dimTimeName));
            ncFile.GetVar(_fStr.dimTimeName, &axisTimeInt[0]);
            for (int i = 0; i < axisTimeInt.size(); ++i) {
                _fStr.time[i] = (double)axisTimeInt[i];
            }
        } break;
        default:
            wxLogError(_("Variable type not supported yet for the time dimension."));
            return false;
    }

    double refValue = NAN;
    if (_parseTimeReference) {
        wxString refValueStr = ncFile.GetAttString("units", _fStr.dimTimeName);
        int start = refValueStr.Find("since");
        if (start != wxNOT_FOUND) {
            refValueStr = refValueStr.Remove(0, (size_t)start + 6);
            int end = refValueStr.Find(" ");
            if (end != wxNOT_FOUND) {
                refValueStr = refValueStr.Remove((size_t)end, refValueStr.Length() - end);
            }
            refValue = asTime::GetTimeFromString(refValueStr);
        } else {
            wxLogError(_("Time reference could not be extracted."));
            return false;
        }
    }

    ConvertToMjd(_fStr.time, refValue);

    _fStr.timeStep = 24.0 * (_fStr.time[wxMin(1, _fStr.time.size())] - _fStr.time[0]);
    _fStr.firstHour = 24 * fmod(_fStr.time[0], 1);

    return true;
}

bool asPredictor::ExtractLevelAxis(asFileNetcdf& ncFile) {
    if (_fStr.hasLevelDim) {
        _fStr.levels = a1d(ncFile.GetVarLength(_fStr.dimLevelName));

        nc_type ncTypeLevel = ncFile.GetVarType(_fStr.dimLevelName);
        switch (ncTypeLevel) {
            case NC_FLOAT: {
                a1f axisLevelFloat(ncFile.GetVarLength(_fStr.dimLevelName));
                ncFile.GetVar(_fStr.dimLevelName, &axisLevelFloat[0]);
                for (int i = 0; i < axisLevelFloat.size(); ++i) {
                    _fStr.levels[i] = (double)axisLevelFloat[i];
                }
            } break;
            case NC_INT: {
                a1i axisLevelInt(ncFile.GetVarLength(_fStr.dimLevelName));
                ncFile.GetVar(_fStr.dimLevelName, &axisLevelInt[0]);
                for (int i = 0; i < axisLevelInt.size(); ++i) {
                    _fStr.levels[i] = (double)axisLevelInt[i];
                }
            } break;
            case NC_DOUBLE: {
                ncFile.GetVar(_fStr.dimLevelName, &_fStr.levels[0]);
            } break;
            default:
                wxLogError(_("Variable type not supported yet for the level dimension."));
                return false;
        }

        // Check unit
        wxString unit = ncFile.GetAttString("units", _fStr.dimLevelName);
        if (unit.IsSameAs("millibars", false) || unit.IsSameAs("millibar", false) || unit.IsSameAs("hPa", false) ||
            unit.IsSameAs("mbar", false) || unit.IsSameAs("m", false) || unit.IsEmpty()) {
            // Nothing to do.
        } else if (unit.IsSameAs("Pa", false)) {
            for (int i = 0; i < _fStr.levels.size(); ++i) {
                _fStr.levels[i] /= 100;
            }
        } else {
            wxLogError(_("Unknown unit for the level dimension: %s."), unit);
            return false;
        }
    }

    return true;
}

bool asPredictor::ExtractSpatialAxes(asFileNetcdf& ncFile) {
    if (!ncFile.HasVariable(_fStr.dimLonName)) {
        if (ncFile.HasVariable("x")) {
            _fStr.dimLonName = "x";
        } else if (ncFile.HasVariable("lon")) {
            _fStr.dimLonName = "lon";
        } else if (ncFile.HasVariable("longitude")) {
            _fStr.dimLonName = "longitude";
        } else {
            wxLogError(_("X/longitude axis not found."));
            return false;
        }
    }

    if (!ncFile.HasVariable(_fStr.dimLatName)) {
        if (ncFile.HasVariable("y")) {
            _fStr.dimLatName = "y";
        } else if (ncFile.HasVariable("lat")) {
            _fStr.dimLonName = "lat";
        } else if (ncFile.HasVariable("latitude")) {
            _fStr.dimLonName = "latitude";
        } else {
            wxLogError(_("Y/latitude axis not found."));
            return false;
        }
    }

    _fStr.lons = a1d(ncFile.GetVarLength(_fStr.dimLonName));
    _fStr.lats = a1d(ncFile.GetVarLength(_fStr.dimLatName));

    wxASSERT(ncFile.GetVarType(_fStr.dimLonName) == ncFile.GetVarType(_fStr.dimLatName));
    nc_type ncTypeAxes = ncFile.GetVarType(_fStr.dimLonName);
    switch (ncTypeAxes) {
        case NC_FLOAT: {
            a1f axisLonFloat(ncFile.GetVarLength(_fStr.dimLonName));
            a1f axisLatFloat(ncFile.GetVarLength(_fStr.dimLatName));
            ncFile.GetVar(_fStr.dimLonName, &axisLonFloat[0]);
            ncFile.GetVar(_fStr.dimLatName, &axisLatFloat[0]);
            for (int i = 0; i < axisLonFloat.size(); ++i) {
                _fStr.lons[i] = (double)axisLonFloat[i];
            }
            for (int i = 0; i < axisLatFloat.size(); ++i) {
                _fStr.lats[i] = (double)axisLatFloat[i];
            }
            break;
        }
        case NC_DOUBLE: {
            ncFile.GetVar(_fStr.dimLonName, &_fStr.lons[0]);
            ncFile.GetVar(_fStr.dimLatName, &_fStr.lats[0]);
            break;
        }
        default:
            wxLogError(_("Variable type not supported yet for the level dimension."));
            return false;
    }

    return true;
}

bool asPredictor::ParseFileStructure(asFileGrib* gbFile0) {
    // Get full axes from the file
    gbFile0->GetXaxis(_fStr.lons);
    gbFile0->GetYaxis(_fStr.lats);
    gbFile0->GetLevels(_fStr.levels);

    // Time properties
    vd timeArray = gbFile0->GetRealTimeArray();
    _fStr.time.resize(timeArray.size());
    for (int i = 0; i < timeArray.size(); ++i) {
        _fStr.time[i] = FixTimeValue(timeArray[i]);
    }

    if (timeArray.size() > 1) {
        _fStr.timeStep = gbFile0->GetTimeStepHours();
        _fStr.firstHour = 24 * fmod(_fStr.time[0], 1);
    }

    return CheckFileStructure();
}

bool asPredictor::ParseFileStructure(asFileGrib* gbFile0, asFileGrib* gbFile1) {
    // Get full axes from the file
    gbFile0->GetXaxis(_fStr.lons);
    gbFile0->GetYaxis(_fStr.lats);
    gbFile0->GetLevels(_fStr.levels);

    // Time properties
    vd timeArray = gbFile0->GetRealTimeArray();
    _fStr.time.resize(timeArray.size());
    for (int i = 0; i < timeArray.size(); ++i) {
        _fStr.time[i] = timeArray[i];
    }

    _fStr.timeStep = asRound(24 * (gbFile1->GetTimeStart() - gbFile0->GetTimeStart()));
    _fStr.firstHour = 24 * fmod(_fStr.time[0], 1);

    return CheckFileStructure();
}

bool asPredictor::CheckFileStructure() {
    // Check for breaks in the longitude axis.
    if (_fStr.lons.size() > 1) {
        if (_fStr.lons[_fStr.lons.size() - 1] < _fStr.lons[0]) {
            int iBreak = 0;
            for (int i = 1; i < _fStr.lons.size(); ++i) {
                if (_fStr.lons[i] < _fStr.lons[i - 1]) {
                    if (iBreak != 0) {
                        wxLogError(_("Longitude axis seems not consistent (multiple breaks)."));
                        return false;
                    }
                    iBreak = i;
                }
            }
            for (int i = iBreak; i < _fStr.lons.size(); ++i) {
                _fStr.lons[i] += 360;
            }
        }
    }

    return true;
}

bool asPredictor::HasDesiredLevel(bool useWarnings) {
    if (_fStr.levels.size() == 0 && _level == 0) {
        return true;
    }

    for (int i = 0; i < _fStr.levels.size(); ++i) {
        if (_fStr.levels[i] == _level) {
            return true;
        }
    }

    if (_fStr.levels.size() == 1 && _level == 0) {
        wxLogWarning(_("Level %f was requested and %f was found in file (single level)"), _level, _fStr.levels[0]);
        return true;
    }

    if (useWarnings) {
        wxLogWarning(_("Cannot find level %f"), _level);
    } else {
        wxLogVerbose(_("Cannot find level %f"), _level);
    }

    return false;
}

asAreaGrid* asPredictor::CreateMatchingArea(asAreaGrid* desiredArea) {
    wxASSERT(_fStr.lons.size() > 0);
    wxASSERT(_fStr.lats.size() > 0);

    if (!desiredArea) {
        return nullptr;
    }

    bool strideAllowed = _fileType == asFile::Netcdf;

    if (desiredArea->IsFull()) {
        double xMin = _fStr.lons.minCoeff();
        int xPtsNb = (int)_fStr.lons.size();
        double yMin = _fStr.lats.minCoeff();
        int yPtsNb = (int)_fStr.lats.size();

        auto dataArea = new asAreaGridRegular(xMin, xPtsNb, yMin, yPtsNb, true, desiredArea->FlatsAllowed());
        if (!dataArea->InitializeAxes(_fStr.lons, _fStr.lats, strideAllowed)) {
            throw runtime_error(_("Failed at initializing the axes."));
        }

        _fInd.lonStep = 1;
        _fInd.latStep = 1;

        _lonPtsnb = dataArea->GetXptsNb();
        _latPtsnb = dataArea->GetYptsNb();
        _axisLon = dataArea->GetXaxis();
        _axisLat = dataArea->GetYaxis();

        // Order latitude axis (as data will also be ordered)
        asSortArray(&_axisLat[0], &_axisLat[_axisLat.size() - 1], Desc);

        return dataArea;
    }

    if (!desiredArea->InitializeAxes(_fStr.lons, _fStr.lats, true)) {
        throw runtime_error(_("Failed at initializing the axes."));
    }

    if (desiredArea->IsRegular()) {
        auto desiredAreaReg = dynamic_cast<asAreaGridRegular*>(desiredArea);

        if (!strideAllowed) {
            _fInd.lonStep = 1;
            _fInd.latStep = 1;
        } else {
            _fInd.lonStep = desiredAreaReg->GetXstepStride();
            _fInd.latStep = desiredAreaReg->GetYstepStride();
        }

        auto dataArea = new asAreaGridRegular(*desiredAreaReg);
        if (!dataArea->InitializeAxes(_fStr.lons, _fStr.lats, strideAllowed)) {
            throw runtime_error(_("Failed at initializing the axes."));
        }

        dataArea->CorrectCornersWithAxes();

        if (!strideAllowed) {
            dataArea->SetSameStepAsData();
        }

        _lonPtsnb = dataArea->GetXptsNb();
        _latPtsnb = dataArea->GetYptsNb();
        _axisLon = desiredArea->GetXaxis();
        _axisLat = desiredArea->GetYaxis();

        // Order latitude axis (as data will also be ordered)
        asSortArray(&_axisLat[0], &_axisLat[_axisLat.size() - 1], Desc);

        return dataArea;

    } else {
        auto desiredAreaGen = dynamic_cast<asAreaGridGeneric*>(desiredArea);
        _fInd.lonStep = 1;
        _fInd.latStep = 1;
        auto dataArea = new asAreaGridGeneric(*desiredAreaGen);
        if (!dataArea->InitializeAxes(_fStr.lons, _fStr.lats, strideAllowed)) {
            throw runtime_error(_("Failed at initializing the axes."));
        }

        _lonPtsnb = dataArea->GetXptsNb();
        _latPtsnb = dataArea->GetYptsNb();
        _axisLon = desiredArea->GetXaxis();
        _axisLat = desiredArea->GetYaxis();

        // Order latitude axis (as data will also be ordered)
        asSortArray(&_axisLat[0], &_axisLat[_axisLat.size() - 1], Desc);

        return dataArea;
    }
}

bool asPredictor::GetAxesIndexes(asAreaGrid*& dataArea, asTimeArray& timeArray) {
    int iStartTimeArray = timeArray.GetIndexFirstAfter(_fStr.time[0], _fStr.timeStep);
    int iEndTimeArray = timeArray.GetIndexFirstBefore(_fStr.time[_fStr.time.size() - 1], _fStr.timeStep);

    if (iStartTimeArray == asOUT_OF_RANGE || iEndTimeArray == asOUT_OF_RANGE) {
        _fInd.timeCountFile = 0;
        return true;
    }

    _fInd.timeCountStorage = iEndTimeArray - iStartTimeArray + 1;
    _fInd.timeStartStorage = iStartTimeArray;

    if (_fStr.time.size() > 1) {
        int iStartTimeFile = asFindClosest(&_fStr.time[0], &_fStr.time[_fStr.time.size() - 1],
                                           timeArray[iStartTimeArray]);
        int iEndTimeFile = asFindClosest(&_fStr.time[0], &_fStr.time[_fStr.time.size() - 1], timeArray[iEndTimeArray]);

        if (iStartTimeFile == asOUT_OF_RANGE || iEndTimeFile == asOUT_OF_RANGE) {
            return false;
        }

        _fInd.timeCountFile = (iEndTimeFile - iStartTimeFile) / _fInd.timeStep + 1;
        _fInd.timeStartFile = iStartTimeFile;

    } else {
        _fInd.timeCountFile = 1;
        _fInd.timeStartFile = 0;
    }

    if (_fInd.timeCountFile != _fInd.timeCountStorage) {
        _fInd.timeConsistent = false;
    } else {
        for (int i = 0; i < _fInd.timeCountFile; ++i) {
            if (_fStr.time[_fInd.timeStartFile + i * _fInd.timeStep] != timeArray[_fInd.timeStartStorage + i]) {
                _fInd.timeConsistent = false;
                break;
            }
        }
    }

    wxASSERT(_fInd.timeCountFile > 0);

    if (dataArea) {
        // Get the spatial extent
        auto lonMin = (float)dataArea->GetXaxisStart();
        auto latMinStart = (float)dataArea->GetYaxisStart();
        auto latMinEnd = (float)dataArea->GetYaxisEnd();

        // The dimensions lengths
        _fInd.area.lonCount = dataArea->GetXaxisPtsnb();
        _fInd.area.latCount = dataArea->GetYaxisPtsnb();

        // Get the spatial indices of the desired data
        _fInd.area.lonStart = asFind(&_fStr.lons[0], &_fStr.lons[_fStr.lons.size() - 1], lonMin, 0.01f,
                                     asHIDE_WARNINGS);
        if (_fInd.area.lonStart == asOUT_OF_RANGE) {
            // If not found, try with negative angles
            _fInd.area.lonStart = asFind(&_fStr.lons[0], &_fStr.lons[_fStr.lons.size() - 1], lonMin - 360, 0.01f,
                                         asHIDE_WARNINGS);
        }
        if (_fInd.area.lonStart == asOUT_OF_RANGE) {
            // If not found, try with angles above 360 degrees
            _fInd.area.lonStart = asFind(&_fStr.lons[0], &_fStr.lons[_fStr.lons.size() - 1], lonMin + 360, 0.01f,
                                         asHIDE_WARNINGS);
        }
        if (_fInd.area.lonStart < 0) {
            wxLogError(_("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f)"), lonMin, _fStr.lons[0],
                       (int)_fStr.lons.size(), _fStr.lons[_fStr.lons.size() - 1]);
            return false;
        }
        wxASSERT_MSG(_fInd.area.lonStart >= 0,
                     asStrF("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", _fStr.lons[0],
                            (int)_fStr.lons.size(), _fStr.lons[_fStr.lons.size() - 1], lonMin));

        int indexStartLat1 = asFind(&_fStr.lats[0], &_fStr.lats[_fStr.lats.size() - 1], latMinStart, 0.01f);
        int indexStartLat2 = asFind(&_fStr.lats[0], &_fStr.lats[_fStr.lats.size() - 1], latMinEnd, 0.01f);
        wxASSERT_MSG(indexStartLat1 >= 0, asStrF("Looking for %g in %g to %g", latMinStart, _fStr.lats[0],
                                                 _fStr.lats[_fStr.lats.size() - 1]));
        wxASSERT_MSG(indexStartLat2 >= 0,
                     asStrF("Looking for %g in %g to %g", latMinEnd, _fStr.lats[0], _fStr.lats[_fStr.lats.size() - 1]));
        _fInd.area.latStart = wxMin(indexStartLat1, indexStartLat2);
    } else {
        _fInd.area.lonStart = 0;
        _fInd.area.latStart = 0;
        _fInd.area.lonCount = _lonPtsnb;
        _fInd.area.latCount = _latPtsnb;
    }

    if (_fStr.hasLevelDim && !_fStr.singleLevel) {
        wxASSERT(_fStr.levels.size() > 0);
        _fInd.level = asFind(&_fStr.levels[0], &_fStr.levels[_fStr.levels.size() - 1], _level, 0.01f);
        if (_fInd.level < 0) {
            wxLogWarning(_("The desired level (%g) does not exist for %s"), _level, _fileVarName);
            return false;
        }
    } else if (_fStr.hasLevelDim && _fStr.singleLevel) {
        _fInd.level = 0;
    } else {
        if (_level > 0) {
            wxLogWarning(_("The desired level (%g) does not exist for %s"), _level, _fileVarName);
            return false;
        }
    }

    return true;
}

size_t* asPredictor::GetIndexesStartNcdf() const {
    if (!_isEnsemble) {
        if (_fStr.hasLevelDim) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)_fInd.timeStartFile;
            array[1] = (size_t)_fInd.level;
            array[2] = (size_t)_fInd.area.latStart;
            array[3] = (size_t)_fInd.area.lonStart;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t)_fInd.timeStartFile;
            array[1] = (size_t)_fInd.area.latStart;
            array[2] = (size_t)_fInd.area.lonStart;

            return array;
        }
    } else {
        if (_fStr.hasLevelDim) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t)_fInd.timeStartFile;
            array[1] = (size_t)_fInd.memberStart;
            array[2] = (size_t)_fInd.level;
            array[3] = (size_t)_fInd.area.latStart;
            array[4] = (size_t)_fInd.area.lonStart;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)_fInd.timeStartFile;
            array[1] = (size_t)_fInd.memberStart;
            array[2] = (size_t)_fInd.area.latStart;
            array[3] = (size_t)_fInd.area.lonStart;

            return array;
        }
    }
}

size_t* asPredictor::GetIndexesCountNcdf() const {
    if (!_isEnsemble) {
        if (_fStr.hasLevelDim) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)_fInd.timeCountFile;
            array[1] = 1;
            array[2] = (size_t)_fInd.area.latCount;
            array[3] = (size_t)_fInd.area.lonCount;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t)_fInd.timeCountFile;
            array[1] = (size_t)_fInd.area.latCount;
            array[2] = (size_t)_fInd.area.lonCount;

            return array;
        }
    } else {
        if (_fStr.hasLevelDim) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t)_fInd.timeCountFile;
            array[1] = (size_t)_fInd.memberCount;
            array[2] = 1;
            array[3] = (size_t)_fInd.area.latCount;
            array[4] = (size_t)_fInd.area.lonCount;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)_fInd.timeCountFile;
            array[1] = (size_t)_fInd.memberCount;
            array[2] = (size_t)_fInd.area.latCount;
            array[3] = (size_t)_fInd.area.lonCount;

            return array;
        }
    }
}

ptrdiff_t* asPredictor::GetIndexesStrideNcdf() const {
    if (!_isEnsemble) {
        if (_fStr.hasLevelDim) {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t)_fInd.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t)_fInd.latStep;
            array[3] = (ptrdiff_t)_fInd.lonStep;

            return array;
        } else {
            static ptrdiff_t array[3] = {0, 0, 0};
            array[0] = (ptrdiff_t)_fInd.timeStep;
            array[1] = (ptrdiff_t)_fInd.latStep;
            array[2] = (ptrdiff_t)_fInd.lonStep;

            return array;
        }
    } else {
        if (_fStr.hasLevelDim) {
            static ptrdiff_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (ptrdiff_t)_fInd.timeStep;
            array[1] = 1;
            array[2] = 1;
            array[3] = (ptrdiff_t)_fInd.latStep;
            array[4] = (ptrdiff_t)_fInd.lonStep;

            return array;
        } else {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t)_fInd.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t)_fInd.latStep;
            array[3] = (ptrdiff_t)_fInd.lonStep;

            return array;
        }
    }
}

int* asPredictor::GetIndexesStartGrib() const {
    static int array[3] = {0, 0, 0};
    array[0] = _fInd.timeStartFile;
    array[1] = _fInd.area.lonStart;
    array[2] = _fInd.area.latStart;

    return array;
}

int* asPredictor::GetIndexesCountGrib() const {
    static int array[3] = {0, 0, 0};
    array[0] = _fInd.timeCountFile;
    array[1] = _fInd.area.lonCount;
    array[2] = _fInd.area.latCount;

    return array;
}

bool asPredictor::GetDataFromFile(asFileNetcdf& ncFile) {
    // Check if loading data is relevant
    if (_fInd.timeCountFile == 0) {
        return true;
    }

    // Check if scaling is needed
    bool scalingNeeded = true;
    float dataAddOffset = 0, dataScaleFactor = 1;
    if (ncFile.HasAttribute("add_offset", _fileVarName)) {
        dataAddOffset = ncFile.GetAttFloat("add_offset", _fileVarName);
    }
    if (ncFile.HasAttribute("scale_factor", _fileVarName)) {
        dataScaleFactor = ncFile.GetAttFloat("scale_factor", _fileVarName);
    }
    if (dataAddOffset == 0 && dataScaleFactor == 1) scalingNeeded = false;

    // Create the arrays to receive the data
    vf dataF;

    // Resize the arrays to store the new data
    int totLength = _fInd.memberCount * _fInd.timeCountFile * _fInd.area.latCount * _fInd.area.lonCount;
    wxASSERT(totLength > 0);
    dataF.resize(totLength);

    // Get data from netCDF file.
    ncFile.GetVarSample(_fileVarName, GetIndexesStartNcdf(), GetIndexesCountNcdf(), GetIndexesStrideNcdf(), &dataF[0]);

    // Allocate space into compositeData if not already done
    if (_data.capacity() == 0) {
        int totSize = _fInd.memberCount * _time.size() * _fInd.area.latCount *
                      (_fInd.area.lonCount + 1);  // +1 in case of a border
        _data.reserve(totSize);
    }

    // Fill with NaN if data are missing before the file starts
    while (_fInd.timeStartStorage > _data.size()) {
        va2f memLatLonData(_fInd.memberCount, a2f::Ones(_fInd.area.latCount, _fInd.area.lonCount) * NAN);
        _data.push_back(memLatLonData);
    }

    // Loop to extract the data from the array
    int ind = 0;
    int iTimeStorage = _fInd.timeStartStorage;
    int iTimeFile = _fInd.timeStartFile;
    int iTimeData = 0;
    while (iTimeStorage < _fInd.timeStartStorage + _fInd.timeCountStorage) {
        if (!_fInd.timeConsistent) {
            if (iTimeFile > _fInd.timeStartFile + _fInd.timeCountFile - 1) {
                // Fill with NaN if data are missing after the data
                va2f memLatLonData(_fInd.memberCount, a2f::Ones(_fInd.area.latCount, _fInd.area.lonCount) * NAN);
                _data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (_time[iTimeStorage] < _fStr.time[iTimeFile]) {
                // Fill in missing data
                va2f memLatLonData(_fInd.memberCount, a2f::Ones(_fInd.area.latCount, _fInd.area.lonCount) * NAN);
                _data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (_time[iTimeStorage] > _fStr.time[iTimeFile]) {
                // If data contains dates we don't want to keep
                iTimeFile++;
                iTimeData++;
                continue;
            }
        }

        // Extract data
        va2f memLatLonData;
        for (int iMem = 0; iMem < _fInd.memberCount; iMem++) {
            a2f latLonData(_fInd.area.latCount, _fInd.area.lonCount);

            for (int iLat = 0; iLat < _fInd.area.latCount; iLat++) {
                for (int iLon = 0; iLon < _fInd.area.lonCount; iLon++) {
                    ind = iLon + iLat * _fInd.area.lonCount + iMem * _fInd.area.lonCount * _fInd.area.latCount +
                          iTimeData * _fInd.memberCount * _fInd.area.lonCount * _fInd.area.latCount;
                    if (_fStr.lats.size() > 0 && _fStr.lats[1] > _fStr.lats[0]) {
                        int latRevIndex = _fInd.area.latCount - 1 - iLat;
                        ind = iLon + latRevIndex * _fInd.area.lonCount +
                              iMem * _fInd.area.lonCount * _fInd.area.latCount +
                              iTimeData * _fInd.memberCount * _fInd.area.lonCount * _fInd.area.latCount;
                    }

                    latLonData(iLat, iLon) = dataF[ind];

                    // Check if not NaN
                    bool notNan = true;
                    for (double nanValue : _nanValues) {
                        if (dataF[ind] == nanValue || latLonData(iLat, iLon) == nanValue) {
                            notNan = false;
                        }
                    }
                    if (isnan(dataF[ind]) || isnan(latLonData(iLat, iLon))) {
                        notNan = false;
                    }
                    if (!notNan) {
                        latLonData(iLat, iLon) = NAN;
                    }
                }
            }

            if (scalingNeeded) {
                latLonData = latLonData * dataScaleFactor + dataAddOffset;
            }
            memLatLonData.push_back(latLonData);
        }
        _data.push_back(memLatLonData);

        iTimeStorage++;
        iTimeFile += _fInd.timeStep;
        iTimeData++;
    }

    return true;
}

bool asPredictor::GetDataFromFile(asFileGrib& gbFile) {
    // Check if loading data is relevant
    if (_fInd.timeCountFile == 0 || _fInd.timeCountStorage == 0) {
        return true;
    }

    // Grib files do not handle stride
    if (_fInd.lonStep != 1 || _fInd.latStep != 1) {
        wxLogError(_("Grib files do not handle stride."));
        return false;
    }

    // Create the arrays to receive the data
    vf dataF;

    // Resize the arrays to store the new data
    int totLength = _fInd.memberCount * _fInd.timeCountFile * _fInd.area.latCount * _fInd.area.lonCount;
    wxASSERT(totLength > 0);
    dataF.resize(totLength);

    // Extract data
    if (!gbFile.GetVarArray(GetIndexesStartGrib(), GetIndexesCountGrib(), &dataF[0])) {
        return false;
    }

    // Allocate space into compositeData if not already done
    if (_data.capacity() == 0) {
        int totSize = _fInd.memberCount * _time.size() * _fInd.area.latCount *
                      (_fInd.area.lonCount + 1);  // +1 in case of a border
        _data.reserve(totSize);
    }

    // Fill with NaN if data are missing before the file starts
    while (_fInd.timeStartStorage > _data.size()) {
        va2f memLatLonData(_fInd.memberCount, a2f::Ones(_fInd.area.latCount, _fInd.area.lonCount) * NAN);
        _data.push_back(memLatLonData);
    }

    // Loop to extract the data from the array
    int ind = 0;
    int iTimeStorage = _fInd.timeStartStorage;
    int iTimeFile = _fInd.timeStartFile;
    int iTimeData = 0;
    while (iTimeStorage < _fInd.timeStartStorage + _fInd.timeCountStorage) {
        if (!_fInd.timeConsistent) {
            if (iTimeFile > _fInd.timeStartFile + _fInd.timeCountFile - 1) {
                // Fill with NaN if data are missing after the data
                va2f memLatLonData(_fInd.memberCount, a2f::Ones(_fInd.area.latCount, _fInd.area.lonCount) * NAN);
                _data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (_time[iTimeStorage] < _fStr.time[iTimeFile]) {
                // Fill in missing data
                va2f memLatLonData(_fInd.memberCount, a2f::Ones(_fInd.area.latCount, _fInd.area.lonCount) * NAN);
                _data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (_time[iTimeStorage] > _fStr.time[iTimeFile]) {
                // If data contains dates we don't want to keep
                iTimeFile++;
                iTimeData++;
                continue;
            }
        }

        // Extract data
        va2f memLatLonData;
        for (int iMem = 0; iMem < _fInd.memberCount; iMem++) {
            a2f latLonData(_fInd.area.latCount, _fInd.area.lonCount);

            for (int iLat = 0; iLat < _fInd.area.latCount; iLat++) {
                for (int iLon = 0; iLon < _fInd.area.lonCount; iLon++) {
                    int latRevIndex = _fInd.area.latCount - 1 - iLat;
                    ind = iLon + latRevIndex * _fInd.area.lonCount + iMem * _fInd.area.lonCount * _fInd.area.latCount +
                          iTimeData * _fInd.memberCount * _fInd.area.lonCount * _fInd.area.latCount;

                    latLonData(iLat, iLon) = dataF[ind];

                    // Check if not NaN
                    bool notNan = true;
                    for (double nanValue : _nanValues) {
                        if (dataF[ind] == nanValue || latLonData(iLat, iLon) == nanValue) {
                            notNan = false;
                        }
                    }
                    if (isnan(dataF[ind]) || isnan(latLonData(iLat, iLon))) {
                        notNan = false;
                    }
                    if (!notNan) {
                        latLonData(iLat, iLon) = NAN;
                    }
                }
            }
            memLatLonData.push_back(latLonData);
        }
        _data.push_back(memLatLonData);

        iTimeStorage++;
        iTimeFile += _fInd.timeStep;
        iTimeData++;
    }

    return true;
}

bool asPredictor::TransformData() {
    if (wxFileConfig::Get()->ReadBool("/General/ReplaceNans", false)) {
        for (int iTime = 0; iTime < _data.size(); iTime++) {
            for (int iMem = 0; iMem < _data[0].size(); iMem++) {
                if (_data[iTime][iMem].hasNaN()) {
                    _data[iTime][iMem] = (!_data[iTime][iMem].isNaN()).select(_data[iTime][iMem], -9999);
                }
            }
        }
    }

    // See
    // http://www.ecmwf.int/en/faq/geopotential-defined-units-m2/s2-both-pressure-levels-and-surface-orography-how-can-height
    if (_parameter == Geopotential) {
        for (int iTime = 0; iTime < _data.size(); iTime++) {
            for (int iMem = 0; iMem < _data[0].size(); iMem++) {
                _data[iTime][iMem] = _data[iTime][iMem] / 9.80665;
            }
        }
        _parameter = GeopotentialHeight;
        _parameterName = "Geopotential height";
        _unit = m;
    }

    return true;
}

bool asPredictor::StandardizeData(double mean, double sd) {
    bool nansReplaced = wxFileConfig::Get()->ReadBool("/General/ReplaceNans", false);

    if (_data[0].size() > 1) {
        wxLogError(_("The standardization of ensemble datasets is not yet supported."));
        return false;
    }

    if (isnan(mean) || isnan(sd)) {
        // Get the mean
        double sum = 0;
        int count = 0;

        for (auto& datTime : _data) {
            for (auto& datMem : datTime) {
                if (!nansReplaced) {
                    sum += datMem.isNaN().select(0, datMem).sum();
                    count += datMem.size() - datMem.isNaN().count();
                } else {
                    sum += datMem.isNaN().select(0, (datMem == -9999).select(0, datMem)).sum();
                    count += datMem.size() - datMem.isNaN().count() - (datMem == -9999).count();
                }
            }
        }

        if (count == 0) {
            mean = 0;
        } else {
            mean = sum / (double)count;
        }

        // Get the standard deviation
        sd = 0;

        for (auto& datTime : _data) {
            for (auto& datMem : datTime) {
                if (!nansReplaced) {
                    sd += (datMem - mean).isNaN().select(0, datMem - mean).cwiseAbs2().sum();
                } else {
                    sd += datMem.isNaN().select(0, (datMem == -9999).select(0, datMem - mean)).cwiseAbs2().sum();
                }
            }
        }

        if (count <= 1) {
            sd = 1;
        } else {
            sd = std::sqrt(sd / (double)(count - 1));
        }
    }

    // Standardize
    for (auto& datTime : _data) {
        for (auto& datMem : datTime) {
            datMem = (datMem - mean) / sd;
        }
    }

    _standardized = true;

    return true;
}

bool asPredictor::ClipToArea(asAreaGrid* desiredArea) {
    double xMin = desiredArea->GetXmin();
    double xMax = desiredArea->GetXmax();
    if (xMin > xMax) {
        xMin -= 360;
    }

    wxASSERT(_axisLon.size() > 0);
    double toleranceLon = 0.1;
    if (_axisLon.size() > 1) {
        toleranceLon = std::abs(_axisLon[1] - _axisLon[0]) / 20;
    }
    int xStartIndex = asFind(&_axisLon[0], &_axisLon[_axisLon.size() - 1], xMin, toleranceLon, asHIDE_WARNINGS);
    int xEndIndex = asFind(&_axisLon[0], &_axisLon[_axisLon.size() - 1], xMax, toleranceLon, asHIDE_WARNINGS);
    if (xStartIndex < 0) {
        xStartIndex = asFind(&_axisLon[0], &_axisLon[_axisLon.size() - 1], xMin + 360, toleranceLon, asHIDE_WARNINGS);
        xEndIndex = asFind(&_axisLon[0], &_axisLon[_axisLon.size() - 1], xMax + 360, toleranceLon, asHIDE_WARNINGS);
        if (xStartIndex < 0 || xEndIndex < 0) {
            xStartIndex = asFind(&_axisLon[0], &_axisLon[_axisLon.size() - 1], xMin - 360, toleranceLon);
            xEndIndex = asFind(&_axisLon[0], &_axisLon[_axisLon.size() - 1], xMax - 360, toleranceLon);
            if (xStartIndex < 0 || xEndIndex < 0) {
                wxLogError(_("An error occurred while trying to clip data to another area (extended axis)."));
                wxLogError(_("Looking for lon %.2f and %.2f in between %.2f to %.2f."), xMin, xMax, _axisLon[0],
                           _axisLon[_axisLon.size() - 1]);
                return false;
            }
        }
    }
    if (xStartIndex < 0 || xEndIndex < 0) {
        wxLogError(_("An error occurred while trying to clip data to another area."));
        wxLogError(_("Looking for lon %.2f and %.2f in between %.2f to %.2f."), xMin, xMax, _axisLon[0],
                   _axisLon[_axisLon.size() - 1]);
        return false;
    }
    int xLength = xEndIndex - xStartIndex + 1;

    double yMin = desiredArea->GetYmin();
    double yMax = desiredArea->GetYmax();
    wxASSERT(_axisLat.size() > 0);
    double toleranceLat = 0.1;
    if (_axisLat.size() > 1) {
        toleranceLat = std::abs(_axisLat[1] - _axisLat[0]) / 20;
    }
    int yStartIndex = asFind(&_axisLat[0], &_axisLat[_axisLat.size() - 1], yMin, toleranceLat, asHIDE_WARNINGS);
    int yEndIndex = asFind(&_axisLat[0], &_axisLat[_axisLat.size() - 1], yMax, toleranceLat, asHIDE_WARNINGS);
    if (yStartIndex < 0 || yEndIndex < 0) {
        wxLogError(_("An error occurred while trying to clip data to another area."));
        wxLogError(_("Looking for lat %.2f and %.2f in between %.2f to %.2f."), yMin, yMax, _axisLat[0],
                   _axisLat[_axisLat.size() - 1]);
        return false;
    }

    int yStartIndexReal = wxMin(yStartIndex, yEndIndex);
    int yLength = std::abs(yEndIndex - yStartIndex) + 1;

    // Check if already the correct size
    if (yStartIndexReal == 0 && xStartIndex == 0 && yLength == _axisLat.size() && xLength == _axisLon.size()) {
        if (IsPreprocessed()) {
            if (_data[0][0].cols() == _axisLon.size() && _data[0][0].rows() == 2 * _axisLat.size()) {
                // Nothing to do
                return true;
            } else {
                // Clear axes
                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NAN;
                }
                _axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NAN;
                }
                _axisLat = newAxisLat;

                _latPtsnb = _axisLat.size();
                _lonPtsnb = _axisLon.size();
            }
        } else {
            // Nothing to do
            return true;
        }
    } else {
        if (!CanBeClipped()) {
            wxLogError(_("The preprocessed area cannot be clipped to another area."));
            return false;
        }

        if (IsPreprocessed()) {
            wxString method = GetPreprocessMethod();
            if (method.IsSameAs("Gradients") || method.IsSameAs("SimpleGradients") ||
                method.IsSameAs("RealGradients") || method.IsSameAs("SimpleGradientsWithGaussianWeights") ||
                method.IsSameAs("RealGradientsWithGaussianWeights")) {
                vva2f originalData = _data;

                if (originalData[0][0].cols() != _axisLon.size() || originalData[0][0].rows() != 2 * _axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError(
                        "originalData[0].cols() = %d, _axisLon.size() = %d, originalData[0].rows() = %d, "
                        "_axisLat.size() = %d",
                        (int)originalData[0][0].cols(), (int)_axisLon.size(), (int)originalData[0][0].rows(),
                        (int)_axisLat.size());
                    return false;
                }

                /*
                Illustration of the data arrangement
                    x = data
                    o = 0

                    xxxxxxxxxxx
                    xxxxxxxxxxx
                    xxxxxxxxxxx
                    ooooooooooo____
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                    xxxxxxxxxxo
                */

                for (int i = 0; i < originalData.size(); i++) {
                    for (int j = 0; j < originalData[i].size(); j++) {
                        a2f dat1 = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength - 1, xLength);
                        a2f dat2 = originalData[i][j].block(yStartIndexReal + _axisLat.size(), xStartIndex, yLength,
                                                            xLength - 1);
                        // Needs to be 0-filled for further simplification.
                        a2f datMerged = a2f::Zero(2 * yLength, xLength);
                        datMerged.block(0, 0, yLength - 1, xLength) = dat1;
                        datMerged.block(yLength, 0, yLength, xLength - 1) = dat2;
                        _data[i][j] = datMerged;
                    }
                }

                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NAN;
                }
                _axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NAN;
                }
                _axisLat = newAxisLat;

                _latPtsnb = _axisLat.size();
                _lonPtsnb = _axisLon.size();

                return true;

            } else if (method.IsSameAs("FormerHumidityIndex")) {
                vva2f originalData = _data;

                if (originalData[0][0].cols() != _axisLon.size() || originalData[0][0].rows() != 2 * _axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError(
                        "originalData[0].cols() = %d, _axisLon.size() = %d, originalData[0].rows() = %d, "
                        "_axisLat.size() = %d",
                        (int)originalData[0][0].cols(), (int)_axisLon.size(), (int)originalData[0][0].rows(),
                        (int)_axisLat.size());
                    return false;
                }

                for (int i = 0; i < originalData.size(); i++) {
                    for (int j = 0; j < originalData[i].size(); j++) {
                        a2f dat1 = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
                        a2f dat2 = originalData[i][j].block(yStartIndexReal + _axisLat.size(), xStartIndex, yLength,
                                                            xLength);
                        a2f datMerged(2 * yLength, xLength);
                        datMerged.block(0, 0, yLength, xLength) = dat1;
                        datMerged.block(yLength, 0, yLength, xLength) = dat2;
                        _data[i][j] = datMerged;
                    }
                }

                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NAN;
                }
                _axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NAN;
                }
                _axisLat = newAxisLat;

                _latPtsnb = _axisLat.size();
                _lonPtsnb = _axisLon.size();

                return true;

            } else if (method.IsSameAs("Multiply") || method.IsSameAs("Multiplication") ||
                       method.IsSameAs("HumidityFlux") || method.IsSameAs("HumidityIndex") ||
                       method.IsSameAs("Addition") || method.IsSameAs("Average")) {
                vva2f originalData = _data;

                if (originalData[0][0].cols() != _axisLon.size() || originalData[0][0].rows() != _axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError(
                        "originalData[0].cols() = %d, _axisLon.size() = %d, originalData[0].rows() = %d, "
                        "_axisLat.size() = %d",
                        (int)originalData[0][0].cols(), (int)_axisLon.size(), (int)originalData[0][0].rows(),
                        (int)_axisLat.size());
                    return false;
                }

                for (int i = 0; i < originalData.size(); i++) {
                    for (int j = 0; j < originalData[i].size(); j++) {
                        _data[i][j] = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
                    }
                }

                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NAN;
                }
                _axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NAN;
                }
                _axisLat = newAxisLat;

                _latPtsnb = _axisLat.size();
                _lonPtsnb = _axisLon.size();

                return true;

            } else {
                wxLogError(_("Wrong preprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    vva2f originalData = _data;
    for (int i = 0; i < originalData.size(); i++) {
        for (int j = 0; j < originalData[i].size(); j++) {
            _data[i][j] = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
        }
    }

    a1d newAxisLon(xLength);
    for (int i = 0; i < xLength; i++) {
        newAxisLon[i] = _axisLon[xStartIndex + i];
    }
    _axisLon = newAxisLon;

    a1d newAxisLat(yLength);
    for (int i = 0; i < yLength; i++) {
        newAxisLat[i] = _axisLat[yStartIndexReal + i];
    }
    _axisLat = newAxisLat;

    _latPtsnb = _axisLat.size();
    _lonPtsnb = _axisLon.size();

    return true;
}

bool asPredictor::Inline() {
    // Already inlined
    if (_lonPtsnb == 1 || _latPtsnb == 1) {
        return true;
    }

    wxASSERT(!_data.empty());

    int timeSize = _data.size();
    int membersNb = _data[0].size();
    int cols = _data[0][0].cols();
    int rows = _data[0][0].rows();

    a2f inlineData = a2f::Zero(1, cols * rows);

    vva2f newData;
    newData.reserve((membersNb * _time.size() * _lonPtsnb * _latPtsnb));
    newData.resize(timeSize);

    for (int iTime = 0; iTime < timeSize; iTime++) {
        for (int iMem = 0; iMem < membersNb; iMem++) {
            for (int iRow = 0; iRow < rows; iRow++) {
                inlineData.block(0, iRow * cols, 1, cols) = _data[iTime][iMem].row(iRow);
            }
            newData[iTime].push_back(inlineData);
        }
    }

    _data = newData;

    _latPtsnb = (int)_data[0][0].rows();
    _lonPtsnb = (int)_data[0][0].cols();
    a1d emptyAxis(1);
    emptyAxis[0] = NAN;
    _axisLat = emptyAxis;
    _axisLon = emptyAxis;

    return true;
}

bool asPredictor::InterpolateOnGrid(asAreaGrid* dataArea, asAreaGrid* desiredArea) {
    wxASSERT(dataArea);
    wxASSERT(desiredArea);
    bool changeXstart = false, changeXsteps = false, changeYstart = false, changeYsteps = false;

    // Check beginning on longitudes
    if (dataArea->GetXmin() != desiredArea->GetXmin()) {
        if (dataArea->GetXmin() + 360 != desiredArea->GetXmin() &&
            dataArea->GetXmin() - 360 != desiredArea->GetXmin()) {
            changeXstart = true;
        }
    }

    // Check beginning on latitudes
    if (dataArea->GetYmin() != desiredArea->GetYmin()) {
        changeYstart = true;
    }

    // Check the cells size on longitudes
    if (dataArea->IsRegular() && !dataArea->GridsOverlay(desiredArea)) {
        changeXsteps = true;
        changeYsteps = true;
    }

    // Proceed to the interpolation
    if (changeXstart || changeYstart || changeXsteps || changeYsteps) {
        if (!dataArea->IsRegular()) {
            wxLogError(_("Interpolation not allowed on irregular grids."));
            return false;
        }

        // Containers for results
        int finalLengthLon = desiredArea->GetXptsNb();
        int finalLengthLat = desiredArea->GetYptsNb();
        vva2f latlonTimeData(_data.size(), va2f(_data[0].size(), a2f(finalLengthLat, finalLengthLon)));

        // Creation of the axes
        a1f axisDataLon;
        if (dataArea->GetXptsNb() > 1) {
            auto xMin = (float)dataArea->GetXmin();
            auto xMax = (float)dataArea->GetXmax();
            if (dataArea->IsLatLon() && xMax < xMin) {
                xMax += 360;
            }
            axisDataLon = a1f::LinSpaced(dataArea->GetXptsNb(), xMin, xMax);
        } else {
            axisDataLon.resize(1);
            axisDataLon << dataArea->GetXmin();
        }

        a1f axisDataLat;
        if (dataArea->GetYptsNb() > 1) {
            axisDataLat = a1f::LinSpaced(dataArea->GetYptsNb(), dataArea->GetYmax(),
                                         dataArea->GetYmin());  // From top to bottom
        } else {
            axisDataLat.resize(1);
            axisDataLat << dataArea->GetYmax();
        }

        a1f axisFinalLon;
        if (desiredArea->GetXptsNb() > 1) {
            auto xMin = (float)desiredArea->GetXmin();
            auto xMax = (float)desiredArea->GetXmax();
            if (desiredArea->IsLatLon() && xMax < xMin) {
                xMax += 360;
            }
            axisFinalLon = a1f::LinSpaced(desiredArea->GetXptsNb(), xMin, xMax);
        } else {
            axisFinalLon.resize(1);
            axisFinalLon << desiredArea->GetXmin();
        }

        a1f axisFinalLat;
        if (desiredArea->GetYptsNb() > 1) {
            axisFinalLat = a1f::LinSpaced(desiredArea->GetYptsNb(), desiredArea->GetYmax(),
                                          desiredArea->GetYmin());  // From top to bottom
        } else {
            axisFinalLat.resize(1);
            axisFinalLat << desiredArea->GetYmax();
        }

        // Indices
        int indexXfloor, indexXceil;
        int indexYfloor, indexYceil;
        int axisDataLonEnd = axisDataLon.size() - 1;
        int axisDataLatEnd = axisDataLat.size() - 1;

        // Pointer to last used element
        int indexLastLon = 0, indexLastLat = 0;

        // Variables
        double dX, dY;
        float valLLcorner, valULcorner, valLRcorner, valURcorner;

        // The interpolation loop
        for (int iTime = 0; iTime < _data.size(); iTime++) {
            for (int iMem = 0; iMem < _data[0].size(); iMem++) {
                // Loop to extract the data from the array
                for (int iLat = 0; iLat < finalLengthLat; iLat++) {
                    // Try the 2 next latitudes (from the top)
                    if (axisDataLat.size() > indexLastLat + 1 && axisDataLat[indexLastLat + 1] == axisFinalLat[iLat]) {
                        indexYfloor = indexLastLat + 1;
                        indexYceil = indexLastLat + 1;
                    } else if (axisDataLat.size() > indexLastLat + 2 &&
                               axisDataLat[indexLastLat + 2] == axisFinalLat[iLat]) {
                        indexYfloor = indexLastLat + 2;
                        indexYceil = indexLastLat + 2;
                    } else {
                        // Search for floor and ceil
                        indexYfloor = indexLastLat + asFindFloor(&axisDataLat[indexLastLat],
                                                                 &axisDataLat[axisDataLatEnd], axisFinalLat[iLat]);
                        indexYceil = indexLastLat + asFindCeil(&axisDataLat[indexLastLat], &axisDataLat[axisDataLatEnd],
                                                               axisFinalLat[iLat]);
                    }

                    if (indexYfloor == asOUT_OF_RANGE || indexYfloor == asNOT_FOUND || indexYceil == asOUT_OF_RANGE ||
                        indexYceil == asNOT_FOUND) {
                        wxLogError(_("The desired point is not available in the data for interpolation. Latitude %f "
                                     "was not found in between %f (index %d) to %f (index %d) (size = %d)."),
                                   axisFinalLat[iLat], axisDataLat[indexLastLat], indexLastLat,
                                   axisDataLat[axisDataLatEnd], axisDataLatEnd, (int)axisDataLat.size());
                        return false;
                    }
                    wxASSERT_MSG(indexYfloor >= 0, asStrF("%f in %f to %f", axisFinalLat[iLat],
                                                          axisDataLat[indexLastLat], axisDataLat[axisDataLatEnd]));
                    wxASSERT(indexYceil >= 0);

                    // Save last index
                    indexLastLat = indexYfloor;

                    for (int iLon = 0; iLon < finalLengthLon; iLon++) {
                        // Try the 2 next longitudes
                        if (axisDataLon.size() > indexLastLon + 1 &&
                            axisDataLon[indexLastLon + 1] == axisFinalLon[iLon]) {
                            indexXfloor = indexLastLon + 1;
                            indexXceil = indexLastLon + 1;
                        } else if (axisDataLon.size() > indexLastLon + 2 &&
                                   axisDataLon[indexLastLon + 2] == axisFinalLon[iLon]) {
                            indexXfloor = indexLastLon + 2;
                            indexXceil = indexLastLon + 2;
                        } else {
                            // Search for floor and ceil
                            indexXfloor = indexLastLon + asFindFloor(&axisDataLon[indexLastLon],
                                                                     &axisDataLon[axisDataLonEnd], axisFinalLon[iLon]);
                            indexXceil = indexLastLon + asFindCeil(&axisDataLon[indexLastLon],
                                                                   &axisDataLon[axisDataLonEnd], axisFinalLon[iLon]);
                        }

                        if (indexXfloor == asOUT_OF_RANGE || indexXfloor == asNOT_FOUND ||
                            indexXceil == asOUT_OF_RANGE || indexXceil == asNOT_FOUND) {
                            wxLogError(_("The desired point is not available in the data for interpolation. Longitude "
                                         "%f was not found in between %f to %f."),
                                       axisFinalLon[iLon], axisDataLon[indexLastLon], axisDataLon[axisDataLonEnd]);
                            return false;
                        }

                        wxASSERT(indexXfloor >= 0);
                        wxASSERT(indexXceil >= 0);

                        // Save last index
                        indexLastLon = indexXfloor;

                        // Proceed to the interpolation
                        if (indexXceil == indexXfloor) {
                            dX = 0;
                        } else {
                            dX = (axisFinalLon[iLon] - axisDataLon[indexXfloor]) /
                                 (axisDataLon[indexXceil] - axisDataLon[indexXfloor]);
                        }
                        if (indexYceil == indexYfloor) {
                            dY = 0;
                        } else {
                            dY = (axisFinalLat[iLat] - axisDataLat[indexYfloor]) /
                                 (axisDataLat[indexYceil] - axisDataLat[indexYfloor]);
                        }

                        if (dX == 0 && dY == 0) {
                            latlonTimeData[iTime][iMem](iLat, iLon) = _data[iTime][iMem](indexYfloor, indexXfloor);
                        } else if (dX == 0) {
                            valLLcorner = _data[iTime][iMem](indexYfloor, indexXfloor);
                            valULcorner = _data[iTime][iMem](indexYceil, indexXfloor);

                            latlonTimeData[iTime][iMem](iLat, iLon) = (1 - dX) * (1 - dY) * valLLcorner +
                                                                      (1 - dX) * (dY)*valULcorner;
                        } else if (dY == 0) {
                            valLLcorner = _data[iTime][iMem](indexYfloor, indexXfloor);
                            valLRcorner = _data[iTime][iMem](indexYfloor, indexXceil);

                            latlonTimeData[iTime][iMem](iLat, iLon) = (1 - dX) * (1 - dY) * valLLcorner +
                                                                      (dX) * (1 - dY) * valLRcorner;
                        } else {
                            valLLcorner = _data[iTime][iMem](indexYfloor, indexXfloor);
                            valULcorner = _data[iTime][iMem](indexYceil, indexXfloor);
                            valLRcorner = _data[iTime][iMem](indexYfloor, indexXceil);
                            valURcorner = _data[iTime][iMem](indexYceil, indexXceil);

                            latlonTimeData[iTime][iMem](
                                iLat, iLon) = (1 - dX) * (1 - dY) * valLLcorner + (1 - dX) * (dY)*valULcorner +
                                              (dX) * (1 - dY) * valLRcorner + (dX) * (dY)*valURcorner;
                        }
                    }

                    indexLastLon = 0;
                }

                indexLastLat = 0;
            }
        }

        _data = latlonTimeData;
        _latPtsnb = finalLengthLat;
        _lonPtsnb = finalLengthLon;
    }

    return true;
}

float asPredictor::GetMinValue() const {
    float minValue = _data[0][0](0, 0);
    float tmpValue;

    for (const auto& dat : _data) {
        for (const auto& v : dat) {
            tmpValue = v.minCoeff();
            if (tmpValue < minValue) {
                minValue = tmpValue;
            }
        }
    }

    return minValue;
}

float asPredictor::GetMaxValue() const {
    float maxValue = _data[0][0](0, 0);
    float tmpValue;

    for (const auto& dat : _data) {
        for (const auto& v : dat) {
            tmpValue = v.maxCoeff();
            if (tmpValue > maxValue) {
                maxValue = tmpValue;
            }
        }
    }

    return maxValue;
}

bool asPredictor::HasNaN() const {
    for (const auto& dat : _data) {
        for (const auto& v : dat) {
            if (v.hasNaN()) {
                return true;
            }
        }
    }

    return false;
}

bool asPredictor::IsLatLon(const wxString& datasetId) {
    if (datasetId.IsSameAs("CORDEX", false)) {
        return false;
    }

    return true;
}

void asPredictor::CheckLevelTypeIsDefined() {
    if (_product.IsEmpty()) {
        throw runtime_error(
            _("The type of product must be defined for this dataset (prefix to the variable name. Ex: press/hgt)."));
    }
}

bool asPredictor::IsPressureLevel() const {
    return _product.IsSameAs("pressure_level", false) || _product.IsSameAs("pressure_levels", false) ||
           _product.IsSameAs("pressure", false) || _product.IsSameAs("press", false) ||
           _product.IsSameAs("isobaric", false) || _product.IsSameAs("pl", false) || _product.IsSameAs("pgbh", false) ||
           _product.IsSameAs("pgbhnl", false) || _product.IsSameAs("pgb", false);
}

bool asPredictor::IsIsentropicLevel() const {
    return _product.IsSameAs("isentropic_level", false) || _product.IsSameAs("isentropic", false) ||
           _product.IsSameAs("potential_temperature", false) || _product.IsSameAs("pt", false) ||
           _product.IsSameAs("ipvh", false) || _product.IsSameAs("ipv", false);
}

bool asPredictor::IsSurfaceLevel() const {
    return _product.IsSameAs("surface", false) || _product.IsSameAs("surf", false) ||
           _product.IsSameAs("ground", false) || _product.IsSameAs("sfc", false) || _product.IsSameAs("sf", false);
}

bool asPredictor::IsSurfaceFluxesLevel() const {
    return _product.IsSameAs("surface_fluxes", false) || _product.IsSameAs("fluxes", false) ||
           _product.IsSameAs("flux", false) || _product.IsSameAs("flxf06", false) || _product.IsSameAs("flx", false);
}

bool asPredictor::IsTotalColumnLevel() const {
    return _product.IsSameAs("total_column", false) || _product.IsSameAs("column", false) ||
           _product.IsSameAs("tc", false) || _product.IsSameAs("entire_atmosphere", false) ||
           _product.IsSameAs("ea", false);
}

bool asPredictor::IsPVLevel() const {
    return _product.IsSameAs("potential_vorticity", false) || _product.IsSameAs("pv", false) ||
           _product.IsSameAs("pv_surface", false) || _product.IsSameAs("epv", false);
}

bool asPredictor::IsGeopotential() const {
    return _dataId.IsSameAs("z", false) || _dataId.IsSameAs("h", false) || _dataId.IsSameAs("zg", false);
}

bool asPredictor::IsGeopotentialHeight() const {
    return _dataId.IsSameAs("z", false) || _dataId.IsSameAs("h", false) || _dataId.IsSameAs("zg", false) ||
           _dataId.IsSameAs("hgt", false);
}

bool asPredictor::IsAirTemperature() const {
    return _dataId.IsSameAs("t", false) || _dataId.IsSameAs("temp", false) || _dataId.IsSameAs("tmp", false) ||
           _dataId.IsSameAs("ta", false) || _dataId.IsSameAs("air", false);
}

bool asPredictor::IsRelativeHumidity() const {
    return _dataId.IsSameAs("rh", false) || _dataId.IsSameAs("rhum", false) || _dataId.IsSameAs("hur", false) ||
           _dataId.IsSameAs("r", false);
}

bool asPredictor::IsSpecificHumidity() const {
    return _dataId.IsSameAs("sh", false) || _dataId.IsSameAs("shum", false) || _dataId.IsSameAs("hus", false) ||
           _dataId.IsSameAs("q", false) || _dataId.IsSameAs("qv", false);
}

bool asPredictor::IsVerticalVelocity() const {
    return _dataId.IsSameAs("w", false) || _dataId.IsSameAs("vvel", false) || _dataId.IsSameAs("vv", false) ||
           _dataId.IsSameAs("wap", false) || _dataId.IsSameAs("omega", false);
}

bool asPredictor::IsTotalColumnWater() const {
    return _dataId.IsSameAs("tcw", false);
}

bool asPredictor::IsTotalColumnWaterVapour() const {
    return _dataId.IsSameAs("tcwv", false);
}

bool asPredictor::IsPrecipitableWater() const {
    return _dataId.IsSameAs("pwat", false) || _dataId.IsSameAs("p_wat", false) || _dataId.IsSameAs("pr_wtr", false) ||
           _dataId.IsSameAs("prwtr", false);
}

bool asPredictor::IsPressure() const {
    return _dataId.IsSameAs("pressure", false) || _dataId.IsSameAs("press", false) || _dataId.IsSameAs("pres", false);
}

bool asPredictor::IsSeaLevelPressure() const {
    return _dataId.IsSameAs("slp", false) || _dataId.IsSameAs("mslp", false) || _dataId.IsSameAs("psl", false) ||
           _dataId.IsSameAs("prmsl", false) || _dataId.IsSameAs("msl", false);
}

bool asPredictor::IsUwindComponent() const {
    return _dataId.IsSameAs("u", false) || _dataId.IsSameAs("ua", false) || _dataId.IsSameAs("ugrd", false) ||
           _dataId.IsSameAs("u_grd", false) || _dataId.IsSameAs("uwnd", false);
}

bool asPredictor::IsVwindComponent() const {
    return _dataId.IsSameAs("v", false) || _dataId.IsSameAs("va", false) || _dataId.IsSameAs("vgrd", false) ||
           _dataId.IsSameAs("v_grd", false) || _dataId.IsSameAs("vwnd", false);
}

bool asPredictor::IsPotentialVorticity() const {
    return _dataId.IsSameAs("pv", false) || _dataId.IsSameAs("pvort", false) || _dataId.IsSameAs("epv", false);
}

bool asPredictor::IsTotalPrecipitation() const {
    return _dataId.IsSameAs("tp", false) || _dataId.IsSameAs("prectot", false);
}

bool asPredictor::IsPrecipitationRate() const {
    return _dataId.IsSameAs("prate", false);
}