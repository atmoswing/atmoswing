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

#include "asAreaGenGrid.h"
#include "asAreaRegGrid.h"
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
#include "asPredictorGenericNetcdf.h"
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
    : m_fileType(asFile::Netcdf),
      m_initialized(false),
      m_standardized(false),
      m_axesChecked(false),
      m_wasDumped(false),
      m_dataId(dataId),
      m_parameter(ParameterUndefined),
      m_gribCode({asNOT_FOUND, asNOT_FOUND, asNOT_FOUND, asNOT_FOUND}),
      m_unit(UnitUndefined),
      m_strideAllowed(false),
      m_level(0),
      m_membersNb(1),
      m_latPtsnb(0),
      m_lonPtsnb(0),
      m_isLatLon(true),
      m_isPreprocessed(false),
      m_isEnsemble(false),
      m_canBeClipped(true),
      m_parseTimeReference(false),
      m_warnMissingFiles(true),
      m_warnMissingLevels(true),
      m_percentMissingAllowed(5) {
    m_fStr.hasLevelDim = true;
    m_fStr.singleLevel = false;
    m_fStr.singleTimeStep = false;
    m_fStr.timeStep = 0;
    m_fInd.memberStart = 0;
    m_fInd.memberCount = 1;
    m_fInd.latStep = 0;
    m_fInd.lonStep = 0;
    m_fInd.level = 0;
    m_fInd.timeStartFile = 0;
    m_fInd.timeStartStorage = 0;
    m_fInd.timeCountFile = 0;
    m_fInd.timeCountStorage = 0;
    m_fInd.timeConsistent = true;
    m_fInd.timeStep = 0;

    if (dataId.Contains('/')) {
        wxString levelType = dataId.BeforeLast('/');
        m_product = levelType;
        m_dataId = dataId.AfterLast('/');
    } else {
        wxLogVerbose(_("The data ID (%s) does not contain the level type"), dataId);
    }
}

asPredictor* asPredictor::GetInstance(const wxString& datasetId, const wxString& dataId, const wxString& directory) {
    asPredictor* predictor = nullptr;

    if (datasetId.IsSameAs("GenericNetcdf", false)) {
        predictor = new asPredictorGenericNetcdf(dataId);
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
    } else if (datasetId.IsSameAs("ECMWF_IFS_GRIB", false)) {
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
    wxASSERT(m_time.size() > 0);
    wxASSERT((int)m_time.size() == (int)val.size());

    m_latPtsnb = (int)val[0][0].rows();
    m_lonPtsnb = (int)val[0][0].cols();
    m_membersNb = (int)val[0].size();
    m_data.clear();
    m_data.reserve(m_time.size() * val[0].size() * m_latPtsnb * m_lonPtsnb);
    m_data = val;

    return true;
}

void asPredictor::DumpData() {
    m_wasDumped = true;
    m_data.clear();
}

bool asPredictor::SaveDumpFile() {
    wxASSERT(m_time.size() > 0);
    wxASSERT(!m_data.empty());

    wxString filePath = GetDumpFileName();

    wxFFile file(filePath, "wb");

    if (!file.IsOpened()) {
        wxLogError(_("Failed creating the file %s"), filePath);
        return false;
    }

    file.Write(&m_latPtsnb, sizeof(int));
    file.Write(&m_lonPtsnb, sizeof(int));

    int nLats = m_axisLat.size();
    int nLons = m_axisLon.size();

    file.Write(&nLats, sizeof(int));
    file.Write(&nLons, sizeof(int));

    file.Write(&m_axisLat[0], nLats * sizeof(double));
    file.Write(&m_axisLon[0], nLons * sizeof(double));

    size_t size = m_time.size() * m_membersNb * m_latPtsnb * m_lonPtsnb * sizeof(float);

    a2f data(m_time.size() * m_membersNb * m_latPtsnb, m_lonPtsnb);

    for (int t = 0; t < m_data.size(); ++t) {
        for (int m = 0; m < m_membersNb; ++m) {
            int l = t * m_membersNb * m_latPtsnb + m * m_latPtsnb;
            data.block(l, 0, m_latPtsnb, m_lonPtsnb) = m_data[t][m];
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
    wxASSERT(m_time.size() > 0);
    wxASSERT(m_data.empty());

    wxString filePath = GetDumpFileName();

    wxFFile file(filePath, "rb");

    if (!file.IsOpened()) {
        wxLogError(_("Failed opening the file %s"), filePath);
        return false;
    }

    file.Read(&m_latPtsnb, sizeof(int));
    file.Read(&m_lonPtsnb, sizeof(int));

    int nLats, nLons;
    file.Read(&nLats, sizeof(int));
    file.Read(&nLons, sizeof(int));

    m_axisLat.resize(nLats);
    m_axisLon.resize(nLons);

    file.Read(&m_axisLat[0], nLats * sizeof(double));
    file.Read(&m_axisLon[0], nLons * sizeof(double));

    m_data.resize(m_time.size(),
                  std::vector<a2f, Eigen::aligned_allocator<a2f>>(m_membersNb, a2f(m_latPtsnb, m_lonPtsnb)));
    size_t size = m_time.size() * m_membersNb * m_latPtsnb * m_lonPtsnb * sizeof(float);

    a2f data(m_time.size() * m_membersNb * m_latPtsnb, m_lonPtsnb);
    file.Read(&data(0, 0), size);

    for (int t = 0; t < m_data.size(); ++t) {
        for (int m = 0; m < m_membersNb; ++m) {
            int l = t * m_membersNb * m_latPtsnb + m * m_latPtsnb;
            m_data[t][m] = data.block(l, 0, m_latPtsnb, m_lonPtsnb);
        }
    }

    if (!file.Close()) {
        wxLogError(_("Failed closing the file %s"), filePath);
        return false;
    }

    wxASSERT(!m_data.empty());

    m_wasDumped = false;

    return true;
}

bool asPredictor::DumpFileExists() const {
    return wxFileExists(GetDumpFileName());
}

wxString asPredictor::GetDumpFileName() const {
    wxString fileName(m_datasetId + '-' + m_dataId + '-');
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
    hash << m_standardized;
    hash << m_parameter;
    hash << m_product;
    hash << m_unit;
    hash << m_strideAllowed;
    hash << m_level;
    hash << m_time[0];
    hash << m_time[m_time.size() - 1];
    hash << m_time.size();
    hash << m_membersNb;
    hash << m_isLatLon;
    hash << m_isPreprocessed;
    hash << m_isEnsemble;
    hash << m_canBeClipped;
    hash << m_preprocessMethod;

    std::size_t h = std::hash<std::string>{}(std::string(hash.mb_str()));

    return h;
}

bool asPredictor::CheckFilesPresence() {
    if (m_files.empty()) {
        wxLogError(_("Empty files list for %s (%s)."), m_dataId, m_datasetName);
        return false;
    }

    int nbDirsToRemove = 0;
    int countMissing = 0;

    for (int i = 0; i < m_files.size(); i++) {
        if (i > 0 && nbDirsToRemove > 0) {
            wxFileName fileName(m_files[i]);
            for (int j = 0; j < nbDirsToRemove; ++j) {
                fileName.RemoveLastDir();
            }
            m_files[i] = fileName.GetFullPath();
        }

        if (!wxFile::Exists(m_files[i])) {
            // Search recursively in the parent directory
            wxFileName fileName(m_files[i]);
            while (true) {
                // Check for wildcards
                if (wxIsWild(fileName.GetPath())) {
                    wxLogError(_("No wildcard is yet authorized in the path (%s)"), fileName.GetPath());
                    return false;
                } else if (wxIsWild(fileName.GetFullName())) {
                    wxArrayString files;
                    size_t nb = wxDir::GetAllFiles(fileName.GetPath(), &files, fileName.GetFullName());
                    if (nb == 1) {
                        m_files[i] = files[0];
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
                        wxLogError(_("File not found: %s"), m_files[i]);
                        return false;
                    }

                    fileName.RemoveLastDir();
                    nbDirsToRemove++;
                    if (fileName.Exists()) {
                        m_files[i] = fileName.GetFullPath();
                        break;
                    }
                } else {
                    if (m_warnMissingFiles) {
                        wxLogWarning(_("File not found: %s"), m_files[i]);
                    } else {
                        wxLogVerbose(_("File not found: %s"), m_files[i]);
                    }
                    m_files[i] = wxEmptyString;
                    countMissing++;
                    break;
                }
            }
        }
    }

    float percentMissing = 100.0 * float(countMissing) / float(m_files.size());
    if (percentMissing > m_percentMissingAllowed) {
        wxLogError(_("%.2f percent of the files are missing (%s, %s)."), percentMissing, m_datasetId, m_dataId);
        return false;
    }

    return true;
}

bool asPredictor::Load(asAreaGrid* desiredArea, asTimeArray& timeArray, float level) {
    m_level = level;

    if (!m_initialized) {
        if (!Init()) {
            wxLogError(_("Error at initialization of the predictor dataset %s."), m_datasetName);
            return false;
        }
    }

    try {
        // List files and check availability
        ListFiles(timeArray);
        if (!CheckFilesPresence()) {
            wxLogError(_("Files not found for %s (%s)."), m_dataId, m_datasetName);
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
        if (!HasDesiredLevel(m_warnMissingLevels)) {
            if (m_warnMissingLevels) {
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
        m_time = timeArray.GetTimeArray();
        if (m_fStr.timeStep == 0) {
            m_fInd.timeStep = 1;
        } else {
            m_fInd.timeStep = wxMax(timeArray.GetTimeStepHours() / m_fStr.timeStep, 1);
        }

        // Extract composite data from files
        wxLogVerbose(_("Extracting from files."));
        if (!ExtractFromFiles(dataArea, timeArray)) {
            if (m_warnMissingFiles && m_warnMissingLevels) {
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
        if (m_time.size() > m_data.size()) {
            wxLogError(_("The date and the data array lengths do not match (time = %d and data = %d)."),
                       (int)m_time.size(), (int)m_data.size());
            wxLogError(_("Time array starts on %s and ends on %s."), asTime::GetStringTime(m_time[0], ISOdateTime),
                       asTime::GetStringTime(m_time[m_time.size() - 1], ISOdateTime));
            wxDELETE(dataArea);
            return false;
        }

        wxDELETE(dataArea);
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation (%s) caught when loading data %s (%s)."), msg, m_dataId, m_datasetName);
        return false;
    } catch (std::exception& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        wxLogError(_("Failed to load data (exception)."));
        return false;
    }

    m_membersNb = (int)m_data[0].size();

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
    m_files = vwxs();
}

bool asPredictor::CheckTimeArray(asTimeArray& timeArray) {
    // Check the time steps
    if ((timeArray.GetTimeStepDays() > 0) && (m_fStr.timeStep / 24.0 > timeArray.GetTimeStepDays())) {
        wxLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }

    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays() / (m_fStr.timeStep / 24.0), &intpart);
    if (fractpart > 0.0001 && fractpart < 0.9999) {
        wxLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }

    fractpart = modf((timeArray.GetStartingHour() - m_fStr.firstHour) / m_fStr.timeStep, &intpart);
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
    wxASSERT(m_files.size() > 0);

    switch (m_fileType) {
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
    switch (m_fileType) {
        case (asFile::Netcdf): {
            for (const auto& fileName : m_files) {
                if (!ExtractFromNetcdfFile(fileName, dataArea, timeArray)) {
                    return false;
                }
            }
            break;
        }
        case (asFile::Grib): {
            for (const auto& fileName : m_files) {
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
    asFileNetcdf ncFile(m_files[0], asFileNetcdf::ReadOnly);
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
    wxASSERT(m_files.size() > 0);

    a1d times = timeArray.GetTimeArray();

    // Open Grib files
    ThreadsManager().CritSectionGrib().Enter();
    asFileGrib gbFile0(m_files[0], asFileGrib::ReadOnly);

    wxLogVerbose(_("Opening grib file to enquire the structure."));
    if (!gbFile0.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Set index position
    wxLogVerbose(_("Setting index position in the grib file."));
    if (!gbFile0.SetIndexPositionAnyLevel(m_gribCode)) {
        gbFile0.Close();
        ThreadsManager().CritSectionGrib().Leave();
        return false;
    }

    // Parse file structure
    if (m_fStr.singleTimeStep && m_files.size() > 1) {
        wxASSERT(times.size() > 1);

        wxLogVerbose(_("Creating an instance of the grib object to enquire the structure (2nd file)."));
        asFileGrib gbFile1 = asFileGrib(m_files[1], asFileGrib::ReadOnly);

        wxLogVerbose(_("Opening grib file to enquire the structure (2nd file)."));
        if (!gbFile1.Open()) {
            gbFile0.Close();
            ThreadsManager().CritSectionGrib().Leave();
            wxFAIL;
            return false;
        }

        wxLogVerbose(_("Setting index position in the grib file (2nd file)."));
        if (!gbFile1.SetIndexPositionAnyLevel(m_gribCode)) {
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
    if (!gbFile.SetIndexPosition(m_gribCode, m_level, m_warnMissingLevels)) {
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
    if (m_data.empty()) {
        wxLogError(_("The first file cannot be missing."));
        return false;
    }

    // Check that it's 1 file per time step
    if (m_fInd.timeCountFile > 1) {
        wxLogError(_("Missing files are handled only when there is 1 file per time step."));
        return false;
    }

    // Fill with NaNs
    va2f memLatLonData;
    for (int iMem = 0; iMem < m_fInd.memberCount; iMem++) {
        a2f latLonData = NaNf * a2f::Ones(m_data[0][iMem].rows(), m_data[0][iMem].cols());
        memLatLonData.push_back(latLonData);
    }
    m_data.push_back(memLatLonData);

    return true;
}

bool asPredictor::ParseFileStructure(asFileNetcdf& ncFile) {
    if (!ExtractSpatialAxes(ncFile)) return false;
    if (!ExtractLevelAxis(ncFile)) return false;
    if (!ExtractTimeAxis(ncFile)) return false;

    return CheckFileStructure();
}

bool asPredictor::ExtractTimeAxis(asFileNetcdf& ncFile) {
    m_fStr.time = a1d(ncFile.GetVarLength(m_fStr.dimTimeName));

    switch (ncFile.GetVarType(m_fStr.dimTimeName)) {
        case NC_DOUBLE:
            ncFile.GetVar(m_fStr.dimTimeName, &m_fStr.time[0]);
            break;
        case NC_FLOAT: {
            a1f axisTimeFloat(ncFile.GetVarLength(m_fStr.dimTimeName));
            ncFile.GetVar(m_fStr.dimTimeName, &axisTimeFloat[0]);
            for (int i = 0; i < axisTimeFloat.size(); ++i) {
                m_fStr.time[i] = (double)axisTimeFloat[i];
            }
        } break;
        case NC_INT: {
            a1i axisTimeInt(ncFile.GetVarLength(m_fStr.dimTimeName));
            ncFile.GetVar(m_fStr.dimTimeName, &axisTimeInt[0]);
            for (int i = 0; i < axisTimeInt.size(); ++i) {
                m_fStr.time[i] = (double)axisTimeInt[i];
            }
        } break;
        default:
            wxLogError(_("Variable type not supported yet for the time dimension."));
            return false;
    }

    double refValue = NaNd;
    if (m_parseTimeReference) {
        wxString refValueStr = ncFile.GetAttString("units", m_fStr.dimTimeName);
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

    ConvertToMjd(m_fStr.time, refValue);

    m_fStr.timeStep = 24.0 * (m_fStr.time[wxMin(1, m_fStr.time.size())] - m_fStr.time[0]);
    m_fStr.firstHour = 24 * fmod(m_fStr.time[0], 1);

    return true;
}

bool asPredictor::ExtractLevelAxis(asFileNetcdf& ncFile) {
    if (m_fStr.hasLevelDim) {
        m_fStr.levels = a1d(ncFile.GetVarLength(m_fStr.dimLevelName));

        nc_type ncTypeLevel = ncFile.GetVarType(m_fStr.dimLevelName);
        switch (ncTypeLevel) {
            case NC_FLOAT: {
                a1f axisLevelFloat(ncFile.GetVarLength(m_fStr.dimLevelName));
                ncFile.GetVar(m_fStr.dimLevelName, &axisLevelFloat[0]);
                for (int i = 0; i < axisLevelFloat.size(); ++i) {
                    m_fStr.levels[i] = (double)axisLevelFloat[i];
                }
            } break;
            case NC_INT: {
                a1i axisLevelInt(ncFile.GetVarLength(m_fStr.dimLevelName));
                ncFile.GetVar(m_fStr.dimLevelName, &axisLevelInt[0]);
                for (int i = 0; i < axisLevelInt.size(); ++i) {
                    m_fStr.levels[i] = (double)axisLevelInt[i];
                }
            } break;
            case NC_DOUBLE: {
                ncFile.GetVar(m_fStr.dimLevelName, &m_fStr.levels[0]);
            } break;
            default:
                wxLogError(_("Variable type not supported yet for the level dimension."));
                return false;
        }

        // Check unit
        wxString unit = ncFile.GetAttString("units", m_fStr.dimLevelName);
        if (unit.IsSameAs("millibars", false) || unit.IsSameAs("millibar", false) || unit.IsSameAs("hPa", false) ||
            unit.IsSameAs("mbar", false) || unit.IsSameAs("m", false) || unit.IsEmpty()) {
            // Nothing to do.
        } else if (unit.IsSameAs("Pa", false)) {
            for (int i = 0; i < m_fStr.levels.size(); ++i) {
                m_fStr.levels[i] /= 100;
            }
        } else {
            wxLogError(_("Unknown unit for the level dimension: %s."), unit);
            return false;
        }
    }

    return true;
}

bool asPredictor::ExtractSpatialAxes(asFileNetcdf& ncFile) {
    if (!ncFile.HasVariable(m_fStr.dimLonName)) {
        if (ncFile.HasVariable("x")) {
            m_fStr.dimLonName = "x";
        } else if (ncFile.HasVariable("lon")) {
            m_fStr.dimLonName = "lon";
        } else if (ncFile.HasVariable("longitude")) {
            m_fStr.dimLonName = "longitude";
        } else {
            wxLogError(_("X/longitude axis not found."));
            return false;
        }
    }

    if (!ncFile.HasVariable(m_fStr.dimLatName)) {
        if (ncFile.HasVariable("y")) {
            m_fStr.dimLatName = "y";
        } else if (ncFile.HasVariable("lat")) {
            m_fStr.dimLonName = "lat";
        } else if (ncFile.HasVariable("latitude")) {
            m_fStr.dimLonName = "latitude";
        } else {
            wxLogError(_("Y/latitude axis not found."));
            return false;
        }
    }

    m_fStr.lons = a1d(ncFile.GetVarLength(m_fStr.dimLonName));
    m_fStr.lats = a1d(ncFile.GetVarLength(m_fStr.dimLatName));

    wxASSERT(ncFile.GetVarType(m_fStr.dimLonName) == ncFile.GetVarType(m_fStr.dimLatName));
    nc_type ncTypeAxes = ncFile.GetVarType(m_fStr.dimLonName);
    switch (ncTypeAxes) {
        case NC_FLOAT: {
            a1f axisLonFloat(ncFile.GetVarLength(m_fStr.dimLonName));
            a1f axisLatFloat(ncFile.GetVarLength(m_fStr.dimLatName));
            ncFile.GetVar(m_fStr.dimLonName, &axisLonFloat[0]);
            ncFile.GetVar(m_fStr.dimLatName, &axisLatFloat[0]);
            for (int i = 0; i < axisLonFloat.size(); ++i) {
                m_fStr.lons[i] = (double)axisLonFloat[i];
            }
            for (int i = 0; i < axisLatFloat.size(); ++i) {
                m_fStr.lats[i] = (double)axisLatFloat[i];
            }
            break;
        }
        case NC_DOUBLE: {
            ncFile.GetVar(m_fStr.dimLonName, &m_fStr.lons[0]);
            ncFile.GetVar(m_fStr.dimLatName, &m_fStr.lats[0]);
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
    gbFile0->GetXaxis(m_fStr.lons);
    gbFile0->GetYaxis(m_fStr.lats);
    gbFile0->GetLevels(m_fStr.levels);

    // Time properties
    vd timeArray = gbFile0->GetRealTimeArray();
    m_fStr.time.resize(timeArray.size());
    for (int i = 0; i < timeArray.size(); ++i) {
        m_fStr.time[i] = FixTimeValue(timeArray[i]);
    }

    if (timeArray.size() > 1) {
        m_fStr.timeStep = gbFile0->GetTimeStepHours();
        m_fStr.firstHour = 24 * fmod(m_fStr.time[0], 1);
    }

    return CheckFileStructure();
}

bool asPredictor::ParseFileStructure(asFileGrib* gbFile0, asFileGrib* gbFile1) {
    // Get full axes from the file
    gbFile0->GetXaxis(m_fStr.lons);
    gbFile0->GetYaxis(m_fStr.lats);
    gbFile0->GetLevels(m_fStr.levels);

    // Time properties
    vd timeArray = gbFile0->GetRealTimeArray();
    m_fStr.time.resize(timeArray.size());
    for (int i = 0; i < timeArray.size(); ++i) {
        m_fStr.time[i] = timeArray[i];
    }

    m_fStr.timeStep = asRound(24 * (gbFile1->GetTimeStart() - gbFile0->GetTimeStart()));
    m_fStr.firstHour = 24 * fmod(m_fStr.time[0], 1);

    return CheckFileStructure();
}

bool asPredictor::CheckFileStructure() {
    // Check for breaks in the longitude axis.
    if (m_fStr.lons.size() > 1) {
        if (m_fStr.lons[m_fStr.lons.size() - 1] < m_fStr.lons[0]) {
            int iBreak = 0;
            for (int i = 1; i < m_fStr.lons.size(); ++i) {
                if (m_fStr.lons[i] < m_fStr.lons[i - 1]) {
                    if (iBreak != 0) {
                        wxLogError(_("Longitude axis seems not consistent (multiple breaks)."));
                        return false;
                    }
                    iBreak = i;
                }
            }
            for (int i = iBreak; i < m_fStr.lons.size(); ++i) {
                m_fStr.lons[i] += 360;
            }
        }
    }

    return true;
}

bool asPredictor::HasDesiredLevel(bool useWarnings) {
    if (m_fStr.levels.size() == 0 && m_level == 0) {
        return true;
    }

    for (int i = 0; i < m_fStr.levels.size(); ++i) {
        if (m_fStr.levels[i] == m_level) {
            return true;
        }
    }

    if (m_fStr.levels.size() == 1 && m_level == 0) {
        wxLogWarning(_("Level %f was requested and %f was found in file (single level)"), m_level, m_fStr.levels[0]);
        return true;
    }

    if (useWarnings) {
        wxLogWarning(_("Cannot find level %f"), m_level);
    } else {
        wxLogVerbose(_("Cannot find level %f"), m_level);
    }

    return false;
}

asAreaGrid* asPredictor::CreateMatchingArea(asAreaGrid* desiredArea) {
    wxASSERT(m_fStr.lons.size() > 0);
    wxASSERT(m_fStr.lats.size() > 0);

    if (desiredArea) {
        bool strideAllowed = m_fileType == asFile::Netcdf;

        if (!desiredArea->InitializeAxes(m_fStr.lons, m_fStr.lats, true)) {
            asThrowException(_("Failed at initializing the axes."));
        }

        if (desiredArea->IsRegular()) {
            auto desiredAreaReg = dynamic_cast<asAreaRegGrid*>(desiredArea);

            if (!strideAllowed) {
                m_fInd.lonStep = 1;
                m_fInd.latStep = 1;
            } else {
                m_fInd.lonStep = desiredAreaReg->GetXstepStride();
                m_fInd.latStep = desiredAreaReg->GetYstepStride();
            }

            auto dataArea = new asAreaRegGrid(*desiredAreaReg);
            if (!dataArea->InitializeAxes(m_fStr.lons, m_fStr.lats, strideAllowed)) {
                asThrowException(_("Failed at initializing the axes."));
            }

            dataArea->CorrectCornersWithAxes();

            if (!strideAllowed) {
                dataArea->SetSameStepAsData();
            }

            m_lonPtsnb = dataArea->GetXptsNb();
            m_latPtsnb = dataArea->GetYptsNb();
            m_axisLon = desiredArea->GetXaxis();
            m_axisLat = desiredArea->GetYaxis();

            // Order latitude axis (as data will also be ordered)
            asSortArray(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], Desc);

            return dataArea;

        } else {
            auto desiredAreaGen = dynamic_cast<asAreaGenGrid*>(desiredArea);
            m_fInd.lonStep = 1;
            m_fInd.latStep = 1;
            auto dataArea = new asAreaGenGrid(*desiredAreaGen);
            if (!dataArea->InitializeAxes(m_fStr.lons, m_fStr.lats, strideAllowed)) {
                asThrowException(_("Failed at initializing the axes."));
            }

            m_lonPtsnb = dataArea->GetXptsNb();
            m_latPtsnb = dataArea->GetYptsNb();
            m_axisLon = desiredArea->GetXaxis();
            m_axisLat = desiredArea->GetYaxis();

            // Order latitude axis (as data will also be ordered)
            asSortArray(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], Desc);

            return dataArea;
        }
    }

    return nullptr;
}

bool asPredictor::GetAxesIndexes(asAreaGrid*& dataArea, asTimeArray& timeArray) {
    int iStartTimeArray = timeArray.GetIndexFirstAfter(m_fStr.time[0], m_fStr.timeStep);
    int iEndTimeArray = timeArray.GetIndexFirstBefore(m_fStr.time[m_fStr.time.size() - 1], m_fStr.timeStep);

    if (iStartTimeArray == asOUT_OF_RANGE || iEndTimeArray == asOUT_OF_RANGE) {
        m_fInd.timeCountFile = 0;
        return true;
    }

    m_fInd.timeCountStorage = iEndTimeArray - iStartTimeArray + 1;
    m_fInd.timeStartStorage = iStartTimeArray;

    if (m_fStr.time.size() > 1) {
        int iStartTimeFile = asFindClosest(&m_fStr.time[0], &m_fStr.time[m_fStr.time.size() - 1],
                                           timeArray[iStartTimeArray]);
        int iEndTimeFile = asFindClosest(&m_fStr.time[0], &m_fStr.time[m_fStr.time.size() - 1],
                                         timeArray[iEndTimeArray]);

        if (iStartTimeFile == asOUT_OF_RANGE || iEndTimeFile == asOUT_OF_RANGE) {
            return false;
        }

        m_fInd.timeCountFile = (iEndTimeFile - iStartTimeFile) / m_fInd.timeStep + 1;
        m_fInd.timeStartFile = iStartTimeFile;

    } else {
        m_fInd.timeCountFile = 1;
        m_fInd.timeStartFile = 0;
    }

    if (m_fInd.timeCountFile != m_fInd.timeCountStorage) {
        m_fInd.timeConsistent = false;
    } else {
        for (int i = 0; i < m_fInd.timeCountFile; ++i) {
            if (m_fStr.time[m_fInd.timeStartFile + i * m_fInd.timeStep] != timeArray[m_fInd.timeStartStorage + i]) {
                m_fInd.timeConsistent = false;
                break;
            }
        }
    }

    wxASSERT(m_fInd.timeCountFile > 0);

    if (dataArea) {
        // Get the spatial extent
        auto lonMin = (float)dataArea->GetXaxisStart();
        auto latMinStart = (float)dataArea->GetYaxisStart();
        auto latMinEnd = (float)dataArea->GetYaxisEnd();

        // The dimensions lengths
        m_fInd.area.lonCount = dataArea->GetXaxisPtsnb();
        m_fInd.area.latCount = dataArea->GetYaxisPtsnb();

        // Get the spatial indices of the desired data
        m_fInd.area.lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1], lonMin, 0.01f,
                                      asHIDE_WARNINGS);
        if (m_fInd.area.lonStart == asOUT_OF_RANGE) {
            // If not found, try with negative angles
            m_fInd.area.lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1], lonMin - 360, 0.01f,
                                          asHIDE_WARNINGS);
        }
        if (m_fInd.area.lonStart == asOUT_OF_RANGE) {
            // If not found, try with angles above 360 degrees
            m_fInd.area.lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1], lonMin + 360, 0.01f,
                                          asHIDE_WARNINGS);
        }
        if (m_fInd.area.lonStart < 0) {
            wxLogError("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin, m_fStr.lons[0],
                       (int)m_fStr.lons.size(), m_fStr.lons[m_fStr.lons.size() - 1]);
            return false;
        }
        wxASSERT_MSG(m_fInd.area.lonStart >= 0,
                     asStrF("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", m_fStr.lons[0],
                            (int)m_fStr.lons.size(), m_fStr.lons[m_fStr.lons.size() - 1], lonMin));

        int indexStartLat1 = asFind(&m_fStr.lats[0], &m_fStr.lats[m_fStr.lats.size() - 1], latMinStart, 0.01f);
        int indexStartLat2 = asFind(&m_fStr.lats[0], &m_fStr.lats[m_fStr.lats.size() - 1], latMinEnd, 0.01f);
        wxASSERT_MSG(indexStartLat1 >= 0, asStrF("Looking for %g in %g to %g", latMinStart, m_fStr.lats[0],
                                                 m_fStr.lats[m_fStr.lats.size() - 1]));
        wxASSERT_MSG(indexStartLat2 >= 0, asStrF("Looking for %g in %g to %g", latMinEnd, m_fStr.lats[0],
                                                 m_fStr.lats[m_fStr.lats.size() - 1]));
        m_fInd.area.latStart = wxMin(indexStartLat1, indexStartLat2);
    } else {
        m_fInd.area.lonStart = 0;
        m_fInd.area.latStart = 0;
        m_fInd.area.lonCount = m_lonPtsnb;
        m_fInd.area.latCount = m_latPtsnb;
    }

    if (m_fStr.hasLevelDim && !m_fStr.singleLevel) {
        wxASSERT(m_fStr.levels.size() > 0);
        m_fInd.level = asFind(&m_fStr.levels[0], &m_fStr.levels[m_fStr.levels.size() - 1], m_level, 0.01f);
        if (m_fInd.level < 0) {
            wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVarName);
            return false;
        }
    } else if (m_fStr.hasLevelDim && m_fStr.singleLevel) {
        m_fInd.level = 0;
    } else {
        if (m_level > 0) {
            wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVarName);
            return false;
        }
    }

    return true;
}

size_t* asPredictor::GetIndexesStartNcdf() const {
    if (!m_isEnsemble) {
        if (m_fStr.hasLevelDim) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)m_fInd.timeStartFile;
            array[1] = (size_t)m_fInd.level;
            array[2] = (size_t)m_fInd.area.latStart;
            array[3] = (size_t)m_fInd.area.lonStart;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t)m_fInd.timeStartFile;
            array[1] = (size_t)m_fInd.area.latStart;
            array[2] = (size_t)m_fInd.area.lonStart;

            return array;
        }
    } else {
        if (m_fStr.hasLevelDim) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t)m_fInd.timeStartFile;
            array[1] = (size_t)m_fInd.memberStart;
            array[2] = (size_t)m_fInd.level;
            array[3] = (size_t)m_fInd.area.latStart;
            array[4] = (size_t)m_fInd.area.lonStart;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)m_fInd.timeStartFile;
            array[1] = (size_t)m_fInd.memberStart;
            array[2] = (size_t)m_fInd.area.latStart;
            array[3] = (size_t)m_fInd.area.lonStart;

            return array;
        }
    }
}

size_t* asPredictor::GetIndexesCountNcdf() const {
    if (!m_isEnsemble) {
        if (m_fStr.hasLevelDim) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)m_fInd.timeCountFile;
            array[1] = 1;
            array[2] = (size_t)m_fInd.area.latCount;
            array[3] = (size_t)m_fInd.area.lonCount;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t)m_fInd.timeCountFile;
            array[1] = (size_t)m_fInd.area.latCount;
            array[2] = (size_t)m_fInd.area.lonCount;

            return array;
        }
    } else {
        if (m_fStr.hasLevelDim) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t)m_fInd.timeCountFile;
            array[1] = (size_t)m_fInd.memberCount;
            array[2] = 1;
            array[3] = (size_t)m_fInd.area.latCount;
            array[4] = (size_t)m_fInd.area.lonCount;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t)m_fInd.timeCountFile;
            array[1] = (size_t)m_fInd.memberCount;
            array[2] = (size_t)m_fInd.area.latCount;
            array[3] = (size_t)m_fInd.area.lonCount;

            return array;
        }
    }
}

ptrdiff_t* asPredictor::GetIndexesStrideNcdf() const {
    if (!m_isEnsemble) {
        if (m_fStr.hasLevelDim) {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t)m_fInd.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t)m_fInd.latStep;
            array[3] = (ptrdiff_t)m_fInd.lonStep;

            return array;
        } else {
            static ptrdiff_t array[3] = {0, 0, 0};
            array[0] = (ptrdiff_t)m_fInd.timeStep;
            array[1] = (ptrdiff_t)m_fInd.latStep;
            array[2] = (ptrdiff_t)m_fInd.lonStep;

            return array;
        }
    } else {
        if (m_fStr.hasLevelDim) {
            static ptrdiff_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (ptrdiff_t)m_fInd.timeStep;
            array[1] = 1;
            array[2] = 1;
            array[3] = (ptrdiff_t)m_fInd.latStep;
            array[4] = (ptrdiff_t)m_fInd.lonStep;

            return array;
        } else {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t)m_fInd.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t)m_fInd.latStep;
            array[3] = (ptrdiff_t)m_fInd.lonStep;

            return array;
        }
    }
}

int* asPredictor::GetIndexesStartGrib() const {
    static int array[3] = {0, 0, 0};
    array[0] = m_fInd.timeStartFile;
    array[1] = m_fInd.area.lonStart;
    array[2] = m_fInd.area.latStart;

    return array;
}

int* asPredictor::GetIndexesCountGrib() const {
    static int array[3] = {0, 0, 0};
    array[0] = m_fInd.timeCountFile;
    array[1] = m_fInd.area.lonCount;
    array[2] = m_fInd.area.latCount;

    return array;
}

bool asPredictor::GetDataFromFile(asFileNetcdf& ncFile) {
    // Check if loading data is relevant
    if (m_fInd.timeCountFile == 0) {
        return true;
    }

    // Check if scaling is needed
    bool scalingNeeded = true;
    float dataAddOffset = 0, dataScaleFactor = 1;
    if (ncFile.HasAttribute("add_offset", m_fileVarName)) {
        dataAddOffset = ncFile.GetAttFloat("add_offset", m_fileVarName);
    }
    if (ncFile.HasAttribute("scale_factor", m_fileVarName)) {
        dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_fileVarName);
    }
    if (dataAddOffset == 0 && dataScaleFactor == 1) scalingNeeded = false;

    // Create the arrays to receive the data
    vf dataF;

    // Resize the arrays to store the new data
    int totLength = m_fInd.memberCount * m_fInd.timeCountFile * m_fInd.area.latCount * m_fInd.area.lonCount;
    wxASSERT(totLength > 0);
    dataF.resize(totLength);

    // Get data from netCDF file.
    ncFile.GetVarSample(m_fileVarName, GetIndexesStartNcdf(), GetIndexesCountNcdf(), GetIndexesStrideNcdf(), &dataF[0]);

    // Allocate space into compositeData if not already done
    if (m_data.capacity() == 0) {
        int totSize = m_fInd.memberCount * m_time.size() * m_fInd.area.latCount *
                      (m_fInd.area.lonCount + 1);  // +1 in case of a border
        m_data.reserve(totSize);
    }

    // Fill with NaN if data are missing before the file starts
    while (m_fInd.timeStartStorage > m_data.size()) {
        va2f memLatLonData(m_fInd.memberCount, a2f::Ones(m_fInd.area.latCount, m_fInd.area.lonCount) * NaNf);
        m_data.push_back(memLatLonData);
    }

    // Loop to extract the data from the array
    int ind = 0;
    int iTimeStorage = m_fInd.timeStartStorage;
    int iTimeFile = m_fInd.timeStartFile;
    int iTimeData = 0;
    while (iTimeStorage < m_fInd.timeStartStorage + m_fInd.timeCountStorage) {
        if (!m_fInd.timeConsistent) {
            if (iTimeFile > m_fInd.timeStartFile + m_fInd.timeCountFile - 1) {
                // Fill with NaN if data are missing after the data
                va2f memLatLonData(m_fInd.memberCount, a2f::Ones(m_fInd.area.latCount, m_fInd.area.lonCount) * NaNf);
                m_data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (m_time[iTimeStorage] < m_fStr.time[iTimeFile]) {
                // Fill in missing data
                va2f memLatLonData(m_fInd.memberCount, a2f::Ones(m_fInd.area.latCount, m_fInd.area.lonCount) * NaNf);
                m_data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (m_time[iTimeStorage] > m_fStr.time[iTimeFile]) {
                // If data contains dates we don't want to keep
                iTimeFile++;
                iTimeData++;
                continue;
            }
        }

        // Extract data
        va2f memLatLonData;
        for (int iMem = 0; iMem < m_fInd.memberCount; iMem++) {
            a2f latLonData(m_fInd.area.latCount, m_fInd.area.lonCount);

            for (int iLat = 0; iLat < m_fInd.area.latCount; iLat++) {
                for (int iLon = 0; iLon < m_fInd.area.lonCount; iLon++) {
                    ind = iLon + iLat * m_fInd.area.lonCount + iMem * m_fInd.area.lonCount * m_fInd.area.latCount +
                          iTimeData * m_fInd.memberCount * m_fInd.area.lonCount * m_fInd.area.latCount;
                    if (m_fStr.lats.size() > 0 && m_fStr.lats[1] > m_fStr.lats[0]) {
                        int latRevIndex = m_fInd.area.latCount - 1 - iLat;
                        ind = iLon + latRevIndex * m_fInd.area.lonCount +
                              iMem * m_fInd.area.lonCount * m_fInd.area.latCount +
                              iTimeData * m_fInd.memberCount * m_fInd.area.lonCount * m_fInd.area.latCount;
                    }

                    latLonData(iLat, iLon) = dataF[ind];

                    // Check if not NaN
                    bool notNan = true;
                    for (double nanValue : m_nanValues) {
                        if (dataF[ind] == nanValue || latLonData(iLat, iLon) == nanValue) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latLonData(iLat, iLon) = NaNf;
                    }
                }
            }

            if (scalingNeeded) {
                latLonData = latLonData * dataScaleFactor + dataAddOffset;
            }
            memLatLonData.push_back(latLonData);
        }
        m_data.push_back(memLatLonData);

        iTimeStorage++;
        iTimeFile += m_fInd.timeStep;
        iTimeData++;
    }

    return true;
}

bool asPredictor::GetDataFromFile(asFileGrib& gbFile) {
    // Check if loading data is relevant
    if (m_fInd.timeCountFile == 0 || m_fInd.timeCountStorage == 0) {
        return true;
    }

    // Grib files do not handle stride
    if (m_fInd.lonStep != 1 || m_fInd.latStep != 1) {
        wxLogError(_("Grib files do not handle stride."));
        return false;
    }

    // Create the arrays to receive the data
    vf dataF;

    // Resize the arrays to store the new data
    int totLength = m_fInd.memberCount * m_fInd.timeCountFile * m_fInd.area.latCount * m_fInd.area.lonCount;
    wxASSERT(totLength > 0);
    dataF.resize(totLength);

    // Extract data
    if (!gbFile.GetVarArray(GetIndexesStartGrib(), GetIndexesCountGrib(), &dataF[0])) {
        return false;
    }

    // Allocate space into compositeData if not already done
    if (m_data.capacity() == 0) {
        int totSize = m_fInd.memberCount * m_time.size() * m_fInd.area.latCount *
                      (m_fInd.area.lonCount + 1);  // +1 in case of a border
        m_data.reserve(totSize);
    }

    // Fill with NaN if data are missing before the file starts
    while (m_fInd.timeStartStorage > m_data.size()) {
        va2f memLatLonData(m_fInd.memberCount, a2f::Ones(m_fInd.area.latCount, m_fInd.area.lonCount) * NaNf);
        m_data.push_back(memLatLonData);
    }

    // Loop to extract the data from the array
    int ind = 0;
    int iTimeStorage = m_fInd.timeStartStorage;
    int iTimeFile = m_fInd.timeStartFile;
    int iTimeData = 0;
    while (iTimeStorage < m_fInd.timeStartStorage + m_fInd.timeCountStorage) {
        if (!m_fInd.timeConsistent) {
            if (iTimeFile > m_fInd.timeStartFile + m_fInd.timeCountFile - 1) {
                // Fill with NaN if data are missing after the data
                va2f memLatLonData(m_fInd.memberCount, a2f::Ones(m_fInd.area.latCount, m_fInd.area.lonCount) * NaNf);
                m_data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (m_time[iTimeStorage] < m_fStr.time[iTimeFile]) {
                // Fill in missing data
                va2f memLatLonData(m_fInd.memberCount, a2f::Ones(m_fInd.area.latCount, m_fInd.area.lonCount) * NaNf);
                m_data.push_back(memLatLonData);
                iTimeStorage++;
                continue;
            } else if (m_time[iTimeStorage] > m_fStr.time[iTimeFile]) {
                // If data contains dates we don't want to keep
                iTimeFile++;
                iTimeData++;
                continue;
            }
        }

        // Extract data
        va2f memLatLonData;
        for (int iMem = 0; iMem < m_fInd.memberCount; iMem++) {
            a2f latLonData(m_fInd.area.latCount, m_fInd.area.lonCount);

            for (int iLat = 0; iLat < m_fInd.area.latCount; iLat++) {
                for (int iLon = 0; iLon < m_fInd.area.lonCount; iLon++) {
                    int latRevIndex = m_fInd.area.latCount - 1 - iLat;
                    ind = iLon + latRevIndex * m_fInd.area.lonCount +
                          iMem * m_fInd.area.lonCount * m_fInd.area.latCount +
                          iTimeData * m_fInd.memberCount * m_fInd.area.lonCount * m_fInd.area.latCount;

                    latLonData(iLat, iLon) = dataF[ind];

                    // Check if not NaN
                    bool notNan = true;
                    for (double nanValue : m_nanValues) {
                        if (dataF[ind] == nanValue || latLonData(iLat, iLon) == nanValue) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latLonData(iLat, iLon) = NaNf;
                    }
                }
            }
            memLatLonData.push_back(latLonData);
        }
        m_data.push_back(memLatLonData);

        iTimeStorage++;
        iTimeFile += m_fInd.timeStep;
        iTimeData++;
    }

    return true;
}

bool asPredictor::TransformData() {
    if (wxFileConfig::Get()->ReadBool("/General/ReplaceNans", false)) {
        for (int iTime = 0; iTime < m_data.size(); iTime++) {
            for (int iMem = 0; iMem < m_data[0].size(); iMem++) {
                if (m_data[iTime][iMem].hasNaN()) {
                    m_data[iTime][iMem] = (!m_data[iTime][iMem].isNaN()).select(m_data[iTime][iMem], -9999);
                }
            }
        }
    }

    // See
    // http://www.ecmwf.int/en/faq/geopotential-defined-units-m2/s2-both-pressure-levels-and-surface-orography-how-can-height
    if (m_parameter == Geopotential) {
        for (int iTime = 0; iTime < m_data.size(); iTime++) {
            for (int iMem = 0; iMem < m_data[0].size(); iMem++) {
                m_data[iTime][iMem] = m_data[iTime][iMem] / 9.80665;
            }
        }
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_unit = m;
    }

    return true;
}

bool asPredictor::StandardizeData(double mean, double sd) {
    bool nansReplaced = wxFileConfig::Get()->ReadBool("/General/ReplaceNans", false);

    if (m_data[0].size() > 1) {
        wxLogError(_("The standardization of ensemble datasets is not yet supported."));
        return false;
    }

    if (asIsNaN(mean) || asIsNaN(sd)) {
        // Get the mean
        double sum = 0;
        int count = 0;

        for (auto& datTime : m_data) {
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

        for (auto& datTime : m_data) {
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
    for (auto& datTime : m_data) {
        for (auto& datMem : datTime) {
            datMem = (datMem - mean) / sd;
        }
    }

    m_standardized = true;

    return true;
}

bool asPredictor::ClipToArea(asAreaGrid* desiredArea) {
    double xMin = desiredArea->GetXmin();
    double xMax = desiredArea->GetXmax();
    if (xMin > xMax) {
        xMin -= 360;
    }

    wxASSERT(m_axisLon.size() > 0);
    double toleranceLon = 0.1;
    if (m_axisLon.size() > 1) {
        toleranceLon = std::abs(m_axisLon[1] - m_axisLon[0]) / 20;
    }
    int xStartIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMin, toleranceLon, asHIDE_WARNINGS);
    int xEndIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMax, toleranceLon, asHIDE_WARNINGS);
    if (xStartIndex < 0) {
        xStartIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMin + 360, toleranceLon,
                             asHIDE_WARNINGS);
        xEndIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMax + 360, toleranceLon, asHIDE_WARNINGS);
        if (xStartIndex < 0 || xEndIndex < 0) {
            xStartIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMin - 360, toleranceLon);
            xEndIndex = asFind(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], xMax - 360, toleranceLon);
            if (xStartIndex < 0 || xEndIndex < 0) {
                wxLogError(_("An error occurred while trying to clip data to another area (extended axis)."));
                wxLogError(_("Looking for lon %.2f and %.2f in between %.2f to %.2f."), xMin, xMax, m_axisLon[0],
                           m_axisLon[m_axisLon.size() - 1]);
                return false;
            }
        }
    }
    if (xStartIndex < 0 || xEndIndex < 0) {
        wxLogError(_("An error occurred while trying to clip data to another area."));
        wxLogError(_("Looking for lon %.2f and %.2f in between %.2f to %.2f."), xMin, xMax, m_axisLon[0],
                   m_axisLon[m_axisLon.size() - 1]);
        return false;
    }
    int xLength = xEndIndex - xStartIndex + 1;

    double yMin = desiredArea->GetYmin();
    double yMax = desiredArea->GetYmax();
    wxASSERT(m_axisLat.size() > 0);
    double toleranceLat = 0.1;
    if (m_axisLat.size() > 1) {
        toleranceLat = std::abs(m_axisLat[1] - m_axisLat[0]) / 20;
    }
    int yStartIndex = asFind(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], yMin, toleranceLat, asHIDE_WARNINGS);
    int yEndIndex = asFind(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], yMax, toleranceLat, asHIDE_WARNINGS);
    if (yStartIndex < 0 || yEndIndex < 0) {
        wxLogError(_("An error occurred while trying to clip data to another area."));
        wxLogError(_("Looking for lat %.2f and %.2f in between %.2f to %.2f."), yMin, yMax, m_axisLat[0],
                   m_axisLat[m_axisLat.size() - 1]);
        return false;
    }

    int yStartIndexReal = wxMin(yStartIndex, yEndIndex);
    int yLength = std::abs(yEndIndex - yStartIndex) + 1;

    // Check if already the correct size
    if (yStartIndexReal == 0 && xStartIndex == 0 && yLength == m_axisLat.size() && xLength == m_axisLon.size()) {
        if (IsPreprocessed()) {
            if (m_data[0][0].cols() == m_axisLon.size() && m_data[0][0].rows() == 2 * m_axisLat.size()) {
                // Nothing to do
                return true;
            } else {
                // Clear axes
                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNd;
                }
                m_axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NaNd;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();
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
                vva2f originalData = m_data;

                if (originalData[0][0].cols() != m_axisLon.size() ||
                    originalData[0][0].rows() != 2 * m_axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError(
                        "originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, "
                        "m_axisLat.size() = %d",
                        (int)originalData[0][0].cols(), (int)m_axisLon.size(), (int)originalData[0][0].rows(),
                        (int)m_axisLat.size());
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
                        a2f dat2 = originalData[i][j].block(yStartIndexReal + m_axisLat.size(), xStartIndex, yLength,
                                                            xLength - 1);
                        // Needs to be 0-filled for further simplification.
                        a2f datMerged = a2f::Zero(2 * yLength, xLength);
                        datMerged.block(0, 0, yLength - 1, xLength) = dat1;
                        datMerged.block(yLength, 0, yLength, xLength - 1) = dat2;
                        m_data[i][j] = datMerged;
                    }
                }

                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNd;
                }
                m_axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NaNd;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else if (method.IsSameAs("FormerHumidityIndex")) {
                vva2f originalData = m_data;

                if (originalData[0][0].cols() != m_axisLon.size() ||
                    originalData[0][0].rows() != 2 * m_axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError(
                        "originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, "
                        "m_axisLat.size() = %d",
                        (int)originalData[0][0].cols(), (int)m_axisLon.size(), (int)originalData[0][0].rows(),
                        (int)m_axisLat.size());
                    return false;
                }

                for (int i = 0; i < originalData.size(); i++) {
                    for (int j = 0; j < originalData[i].size(); j++) {
                        a2f dat1 = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
                        a2f dat2 = originalData[i][j].block(yStartIndexReal + m_axisLat.size(), xStartIndex, yLength,
                                                            xLength);
                        a2f datMerged(2 * yLength, xLength);
                        datMerged.block(0, 0, yLength, xLength) = dat1;
                        datMerged.block(yLength, 0, yLength, xLength) = dat2;
                        m_data[i][j] = datMerged;
                    }
                }

                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNd;
                }
                m_axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NaNd;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else if (method.IsSameAs("Multiply") || method.IsSameAs("Multiplication") ||
                       method.IsSameAs("HumidityFlux") || method.IsSameAs("HumidityIndex") ||
                       method.IsSameAs("Addition") || method.IsSameAs("Average")) {
                vva2f originalData = m_data;

                if (originalData[0][0].cols() != m_axisLon.size() || originalData[0][0].rows() != m_axisLat.size()) {
                    wxLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    wxLogError(
                        "originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, "
                        "m_axisLat.size() = %d",
                        (int)originalData[0][0].cols(), (int)m_axisLon.size(), (int)originalData[0][0].rows(),
                        (int)m_axisLat.size());
                    return false;
                }

                for (int i = 0; i < originalData.size(); i++) {
                    for (int j = 0; j < originalData[i].size(); j++) {
                        m_data[i][j] = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
                    }
                }

                a1d newAxisLon(xLength);
                for (int i = 0; i < xLength; i++) {
                    newAxisLon[i] = NaNd;
                }
                m_axisLon = newAxisLon;

                a1d newAxisLat(2 * yLength);
                for (int i = 0; i < 2 * yLength; i++) {
                    newAxisLat[i] = NaNd;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else {
                wxLogError(_("Wrong preprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    vva2f originalData = m_data;
    for (int i = 0; i < originalData.size(); i++) {
        for (int j = 0; j < originalData[i].size(); j++) {
            m_data[i][j] = originalData[i][j].block(yStartIndexReal, xStartIndex, yLength, xLength);
        }
    }

    a1d newAxisLon(xLength);
    for (int i = 0; i < xLength; i++) {
        newAxisLon[i] = m_axisLon[xStartIndex + i];
    }
    m_axisLon = newAxisLon;

    a1d newAxisLat(yLength);
    for (int i = 0; i < yLength; i++) {
        newAxisLat[i] = m_axisLat[yStartIndexReal + i];
    }
    m_axisLat = newAxisLat;

    m_latPtsnb = m_axisLat.size();
    m_lonPtsnb = m_axisLon.size();

    return true;
}

bool asPredictor::Inline() {
    // Already inlined
    if (m_lonPtsnb == 1 || m_latPtsnb == 1) {
        return true;
    }

    wxASSERT(!m_data.empty());

    int timeSize = m_data.size();
    int membersNb = m_data[0].size();
    int cols = m_data[0][0].cols();
    int rows = m_data[0][0].rows();

    a2f inlineData = a2f::Zero(1, cols * rows);

    vva2f newData;
    newData.reserve((membersNb * m_time.size() * m_lonPtsnb * m_latPtsnb));
    newData.resize(timeSize);

    for (int iTime = 0; iTime < timeSize; iTime++) {
        for (int iMem = 0; iMem < membersNb; iMem++) {
            for (int iRow = 0; iRow < rows; iRow++) {
                inlineData.block(0, iRow * cols, 1, cols) = m_data[iTime][iMem].row(iRow);
            }
            newData[iTime].push_back(inlineData);
        }
    }

    m_data = newData;

    m_latPtsnb = (int)m_data[0][0].rows();
    m_lonPtsnb = (int)m_data[0][0].cols();
    a1d emptyAxis(1);
    emptyAxis[0] = NaNd;
    m_axisLat = emptyAxis;
    m_axisLon = emptyAxis;

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
        vva2f latlonTimeData(m_data.size(), va2f(m_data[0].size(), a2f(finalLengthLat, finalLengthLon)));

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
        for (int iTime = 0; iTime < m_data.size(); iTime++) {
            for (int iMem = 0; iMem < m_data[0].size(); iMem++) {
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
                            latlonTimeData[iTime][iMem](iLat, iLon) = m_data[iTime][iMem](indexYfloor, indexXfloor);
                        } else if (dX == 0) {
                            valLLcorner = m_data[iTime][iMem](indexYfloor, indexXfloor);
                            valULcorner = m_data[iTime][iMem](indexYceil, indexXfloor);

                            latlonTimeData[iTime][iMem](iLat, iLon) = (1 - dX) * (1 - dY) * valLLcorner +
                                                                      (1 - dX) * (dY)*valULcorner;
                        } else if (dY == 0) {
                            valLLcorner = m_data[iTime][iMem](indexYfloor, indexXfloor);
                            valLRcorner = m_data[iTime][iMem](indexYfloor, indexXceil);

                            latlonTimeData[iTime][iMem](iLat, iLon) = (1 - dX) * (1 - dY) * valLLcorner +
                                                                      (dX) * (1 - dY) * valLRcorner;
                        } else {
                            valLLcorner = m_data[iTime][iMem](indexYfloor, indexXfloor);
                            valULcorner = m_data[iTime][iMem](indexYceil, indexXfloor);
                            valLRcorner = m_data[iTime][iMem](indexYfloor, indexXceil);
                            valURcorner = m_data[iTime][iMem](indexYceil, indexXceil);

                            latlonTimeData[iTime][iMem](iLat, iLon) =
                                (1 - dX) * (1 - dY) * valLLcorner + (1 - dX) * (dY)*valULcorner +
                                (dX) * (1 - dY) * valLRcorner + (dX) * (dY)*valURcorner;
                        }
                    }

                    indexLastLon = 0;
                }

                indexLastLat = 0;
            }
        }

        m_data = latlonTimeData;
        m_latPtsnb = finalLengthLat;
        m_lonPtsnb = finalLengthLon;
    }

    return true;
}

float asPredictor::GetMinValue() const {
    float minValue = m_data[0][0](0, 0);
    float tmpValue;

    for (const auto& dat : m_data) {
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
    float maxValue = m_data[0][0](0, 0);
    float tmpValue;

    for (const auto& dat : m_data) {
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
    for (const auto& dat : m_data) {
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
    if (m_product.IsEmpty()) {
        asThrowException(
            _("The type of product must be defined for this dataset (prefix to the variable name. Ex: press/hgt)."));
    }
}

bool asPredictor::IsPressureLevel() const {
    return m_product.IsSameAs("pressure_level", false) || m_product.IsSameAs("pressure_levels", false) ||
           m_product.IsSameAs("pressure", false) || m_product.IsSameAs("press", false) ||
           m_product.IsSameAs("isobaric", false) || m_product.IsSameAs("pl", false) ||
           m_product.IsSameAs("pgbh", false) || m_product.IsSameAs("pgbhnl", false) || m_product.IsSameAs("pgb", false);
}

bool asPredictor::IsIsentropicLevel() const {
    return m_product.IsSameAs("isentropic_level", false) || m_product.IsSameAs("isentropic", false) ||
           m_product.IsSameAs("potential_temperature", false) || m_product.IsSameAs("pt", false) ||
           m_product.IsSameAs("ipvh", false) || m_product.IsSameAs("ipv", false);
}

bool asPredictor::IsSurfaceLevel() const {
    return m_product.IsSameAs("surface", false) || m_product.IsSameAs("surf", false) ||
           m_product.IsSameAs("ground", false) || m_product.IsSameAs("sfc", false) || m_product.IsSameAs("sf", false);
}

bool asPredictor::IsSurfaceFluxesLevel() const {
    return m_product.IsSameAs("surface_fluxes", false) || m_product.IsSameAs("fluxes", false) ||
           m_product.IsSameAs("flux", false) || m_product.IsSameAs("flxf06", false) || m_product.IsSameAs("flx", false);
}

bool asPredictor::IsTotalColumnLevel() const {
    return m_product.IsSameAs("total_column", false) || m_product.IsSameAs("column", false) ||
           m_product.IsSameAs("tc", false) || m_product.IsSameAs("entire_atmosphere", false) ||
           m_product.IsSameAs("ea", false);
}

bool asPredictor::IsPVLevel() const {
    return m_product.IsSameAs("potential_vorticity", false) || m_product.IsSameAs("pv", false) ||
           m_product.IsSameAs("pv_surface", false) || m_product.IsSameAs("epv", false);
}

bool asPredictor::IsGeopotential() const {
    return m_dataId.IsSameAs("z", false) || m_dataId.IsSameAs("h", false) || m_dataId.IsSameAs("zg", false);
}

bool asPredictor::IsGeopotentialHeight() const {
    return m_dataId.IsSameAs("z", false) || m_dataId.IsSameAs("h", false) || m_dataId.IsSameAs("zg", false) ||
           m_dataId.IsSameAs("hgt", false);
}

bool asPredictor::IsAirTemperature() const {
    return m_dataId.IsSameAs("t", false) || m_dataId.IsSameAs("temp", false) || m_dataId.IsSameAs("tmp", false) ||
           m_dataId.IsSameAs("ta", false) || m_dataId.IsSameAs("air", false);
}

bool asPredictor::IsRelativeHumidity() const {
    return m_dataId.IsSameAs("rh", false) || m_dataId.IsSameAs("rhum", false) || m_dataId.IsSameAs("hur", false) ||
           m_dataId.IsSameAs("r", false);
}

bool asPredictor::IsSpecificHumidity() const {
    return m_dataId.IsSameAs("sh", false) || m_dataId.IsSameAs("shum", false) || m_dataId.IsSameAs("hus", false) ||
           m_dataId.IsSameAs("q", false) || m_dataId.IsSameAs("qv", false);
}

bool asPredictor::IsVerticalVelocity() const {
    return m_dataId.IsSameAs("w", false) || m_dataId.IsSameAs("vvel", false) || m_dataId.IsSameAs("wap", false) ||
           m_dataId.IsSameAs("omega", false);
}

bool asPredictor::IsTotalColumnWater() const {
    return m_dataId.IsSameAs("tcw", false);
}

bool asPredictor::IsTotalColumnWaterVapour() const {
    return m_dataId.IsSameAs("tcwv", false);
}

bool asPredictor::IsPrecipitableWater() const {
    return m_dataId.IsSameAs("pwat", false) || m_dataId.IsSameAs("p_wat", false) ||
           m_dataId.IsSameAs("pr_wtr", false) || m_dataId.IsSameAs("prwtr", false);
}

bool asPredictor::IsPressure() const {
    return m_dataId.IsSameAs("pressure", false) || m_dataId.IsSameAs("press", false) ||
           m_dataId.IsSameAs("pres", false);
}

bool asPredictor::IsSeaLevelPressure() const {
    return m_dataId.IsSameAs("slp", false) || m_dataId.IsSameAs("mslp", false) || m_dataId.IsSameAs("psl", false) ||
           m_dataId.IsSameAs("prmsl", false) || m_dataId.IsSameAs("msl", false);
}

bool asPredictor::IsUwindComponent() const {
    return m_dataId.IsSameAs("u", false) || m_dataId.IsSameAs("ua", false) || m_dataId.IsSameAs("ugrd", false) ||
           m_dataId.IsSameAs("u_grd", false) || m_dataId.IsSameAs("uwnd", false);
}

bool asPredictor::IsVwindComponent() const {
    return m_dataId.IsSameAs("v", false) || m_dataId.IsSameAs("va", false) || m_dataId.IsSameAs("vgrd", false) ||
           m_dataId.IsSameAs("v_grd", false) || m_dataId.IsSameAs("vwnd", false);
}

bool asPredictor::IsPotentialVorticity() const {
    return m_dataId.IsSameAs("pv", false) || m_dataId.IsSameAs("pvort", false) || m_dataId.IsSameAs("epv", false);
}

bool asPredictor::IsTotalPrecipitation() const {
    return m_dataId.IsSameAs("tp", false) || m_dataId.IsSameAs("prectot", false);
}

bool asPredictor::IsPrecipitationRate() const {
    return m_dataId.IsSameAs("prate", false);
}