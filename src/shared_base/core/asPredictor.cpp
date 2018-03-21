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
 */

#include "asPredictor.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <wx/dir.h>


asPredictor::asPredictor(const wxString &dataId)
        : m_initialized(false),
          m_axesChecked(false),
          m_dataId(dataId),
          m_parameter(ParameterUndefined),
          m_unit(UnitUndefined),
          m_strideAllowed(false),
          m_level(0),
          m_latPtsnb(0),
          m_lonPtsnb(0),
          m_isPreprocessed(false),
          m_isEnsemble(false),
          m_canBeClipped(true)
{

    m_fStr.hasLevelDim = true;
    m_fStr.singleLevel = false;
    m_fStr.timeStep = 0;
    m_fStr.timeStart = 0;
    m_fStr.timeEnd = 0;
    m_fStr.timeLength = 0;
    m_fInd.memberStart = 0;
    m_fInd.memberCount = 1;
    m_fInd.cutEnd = 0;
    m_fInd.cutStart = 0;
    m_fInd.latStep = 0;
    m_fInd.lonStep = 0;
    m_fInd.level = 0;
    m_fInd.timeArrayCount = 0;
    m_fInd.timeCount = 0;
    m_fInd.timeStart = 0;
    m_fInd.timeStep = 0;

    int arr[] = {asNOT_FOUND, asNOT_FOUND, asNOT_FOUND, asNOT_FOUND};
    AssignGribCode(arr);

    if (dataId.Contains('/')) {
        wxString levelType = dataId.BeforeFirst('/');
        m_product = levelType;
        m_dataId = dataId.AfterFirst('/');
    } else {
        wxLogVerbose(_("The data ID (%s) does not contain the level type"), dataId);
    }

}

bool asPredictor::SetData(vva2f &val)
{
    wxASSERT(m_time.size()> 0);
    wxASSERT((int) m_time.size() == (int) val.size());

    m_latPtsnb = (int) val[0][0].rows();
    m_lonPtsnb = (int) val[0][0].cols();
    m_data.clear();
    m_data.reserve(m_time.size() * val[0].size() * m_latPtsnb * m_lonPtsnb);
    m_data = val;

    return true;
}

bool asPredictor::CheckFilesPresence()
{
    if (m_files.empty()) {
        wxLogError(_("Empty files list."));
        return false;
    }

    int nbDirsToRemove = 0;

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
                }
            }
        }
    }

    return true;
}

bool asPredictor::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
{
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
            return false;
        }

        // The desired level
        if (desiredArea) {
            m_level = desiredArea->GetComposite(0).GetLevel();
        }

        // Get file axes
        if (!EnquireFileStructure()) {
            return false;
        }

        // Check the time array
        if (!CheckTimeArray(timeArray)) {
            wxLogError(_("The time array is not valid to load data."));
            return false;
        }

        // Create a new area matching the dataset
        asGeoAreaCompositeGrid *dataArea = CreateMatchingArea(desiredArea);

        // Store time array
        m_time = timeArray.GetTimeArray();
        m_fInd.timeStep = wxMax(timeArray.GetTimeStepHours() / m_fStr.timeStep, 1);

        // Number of composites
        int compositesNb = 1;
        if (dataArea) {
            compositesNb = dataArea->GetNbComposites();
            wxASSERT(compositesNb > 0);
        }

        // Extract composite data from files
        vvva2f compositeData((unsigned long) compositesNb);
        if (!ExtractFromFiles(dataArea, timeArray, compositeData)) {
            wxLogWarning(_("Extracting data from files failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Transform data
        if (!TransformData(compositeData)) {
            wxFAIL;
            return false;
        }

        // Merge the composites into m_data
        if (!MergeComposites(compositeData, dataArea)) {
            wxLogError(_("Merging the composites failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Interpolate the loaded data on the desired grid
        if (desiredArea && !InterpolateOnGrid(dataArea, desiredArea)) {
            wxLogError(_("Interpolation failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Check the data container length
        if ((unsigned) m_time.size() > m_data.size()) {
            wxLogError(_("The date and the data array lengths do not match (time = %d and data = %d)."), (int)m_time.size(), (int)m_data.size());
            wxDELETE(dataArea);
            return false;
        }

        wxDELETE(dataArea);
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught when loading data: %s"), msg);
        return false;
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to load data."));
        return false;
    }

    return true;
}

bool asPredictor::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray)
{
    return Load(&desiredArea, timeArray);
}

bool asPredictor::Load(asGeoAreaCompositeGrid &desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(&desiredArea, timeArray);
}

bool asPredictor::Load(asGeoAreaCompositeGrid *desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(desiredArea, timeArray);
}

bool asPredictor::EnquireFileStructure()
{
    wxASSERT(m_files.size() > 0);

    switch (m_fileType) {
        case (asFile::Netcdf) : {
            if (!EnquireNetcdfFileStructure()) {
                return false;
            }
            break;
        }
        case (asFile::Grib2) : {
            if (!EnquireGribFileStructure()) {
                return false;
            }
            break;
        }
        default: {
            wxLogError(_("Predictor file type not correctly defined."));
        }
    }

    return true;
}

bool asPredictor::ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData)
{
    switch (m_fileType) {
        case (asFile::Netcdf) : {
            for (const auto &fileName : m_files) {
                if (!ExtractFromNetcdfFile(fileName, dataArea, timeArray, compositeData)) {
                    return false;
                }
            }
            break;
        }
        case (asFile::Grib2) : {
            for (const auto &fileName : m_files) {
                if (!ExtractFromGribFile(fileName, dataArea, timeArray, compositeData)) {
                    return false;
                }
            }
            break;
        }
        default: {
            wxLogError(_("Predictor file type not correctly defined."));
        }
    }

    return true;
}

bool asPredictor::EnquireNetcdfFileStructure()
{
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

bool asPredictor::ExtractFromNetcdfFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                        asTimeArray &timeArray, vvva2f &compositeData)
{
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

    // Adjust axes if necessary
    dataArea = AdjustAxes(dataArea, compositeData);
    if (dataArea) {
        wxASSERT(dataArea->GetNbComposites() > 0);
    }

    // Get indexes
    if (!GetAxesIndexes(dataArea, timeArray, compositeData)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Load data
    if (!GetDataFromFile(ncFile, compositeData)) {
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

bool asPredictor::EnquireGribFileStructure()
{
    wxASSERT(m_files.size() > 1);

    // Open 2 Grib files
    ThreadsManager().CritSectionGrib().Enter();
    asFileGrib2 gbFile0(m_files[0], asFileGrib2::ReadOnly);
    if (!gbFile0.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }
    asFileGrib2 gbFile1(m_files[1], asFileGrib2::ReadOnly);
    if (!gbFile1.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Set index position
    if (!gbFile0.SetIndexPosition(m_gribCode, m_level)) {
        gbFile0.Close();
        gbFile1.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }
    if (!gbFile1.SetIndexPosition(m_gribCode, m_level)) {
        gbFile0.Close();
        gbFile1.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(&gbFile0, &gbFile1)) {
        gbFile0.Close();
        gbFile1.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Close the nc file
    gbFile0.Close();
    gbFile1.Close();
    ThreadsManager().CritSectionGrib().Leave();

    return true;
}

bool asPredictor::ExtractFromGribFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                      asTimeArray &timeArray, vvva2f &compositeData)
{
    // Open the Grib file
    ThreadsManager().CritSectionGrib().Enter();
    asFileGrib2 gbFile(fileName, asFileGrib2::ReadOnly);
    if (!gbFile.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Set index position
    if (!gbFile.SetIndexPosition(m_gribCode, m_level)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(&gbFile)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Adjust axes if necessary
    dataArea = AdjustAxes(dataArea, compositeData);
    if (dataArea) {
        wxASSERT(dataArea->GetNbComposites() > 0);
    }

    // Get indexes
    if (!GetAxesIndexes(dataArea, timeArray, compositeData)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Load data
    if (!GetDataFromFile(gbFile, compositeData)) {
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

bool asPredictor::ParseFileStructure(asFileNetcdf &ncFile)
{
    if (!ExtractSpatialAxes(ncFile)) return false;
    if (!ExtractLevelAxis(ncFile)) return false;
    if (!ExtractTimeAxis(ncFile)) return false;

    return CheckFileStructure();
}

bool asPredictor::ExtractTimeAxis(asFileNetcdf &ncFile)
{
    // Time dimension takes ages to load !! Avoid and get the first value.
    m_fStr.timeLength = ncFile.GetVarLength(m_fStr.dimTimeName);

    double timeFirstVal, timeLastVal;
    nc_type ncTypeTime = ncFile.GetVarType(m_fStr.dimTimeName);
    switch (ncTypeTime) {
        case NC_DOUBLE:
            timeFirstVal = ncFile.GetVarOneDouble(m_fStr.dimTimeName, 0);
            timeLastVal = ncFile.GetVarOneDouble(m_fStr.dimTimeName, m_fStr.timeLength - 1);
            break;
        case NC_FLOAT:
            timeFirstVal = (double) ncFile.GetVarOneFloat(m_fStr.dimTimeName, 0);
            timeLastVal = (double) ncFile.GetVarOneFloat(m_fStr.dimTimeName, m_fStr.timeLength - 1);
            break;
        case NC_INT:
            timeFirstVal = (double) ncFile.GetVarOneInt(m_fStr.dimTimeName, 0);
            timeLastVal = (double) ncFile.GetVarOneInt(m_fStr.dimTimeName, m_fStr.timeLength - 1);
            break;
        default:
            wxLogError(_("Variable type not supported yet for the time dimension."));
            return false;
    }

    double refValue = NaNd;
    if (m_datasetId.IsSameAs("NASA_MERRA_2", false) || m_datasetId.IsSameAs("NASA_MERRA_2_subset", false) ||
        m_datasetId.IsSameAs("NCEP_CFSR_subset", false) || m_datasetId.IsSameAs("CMIP5", false)) {

        wxString refValueStr = ncFile.GetAttString("units", "time");
        int start = refValueStr.Find("since");
        if (start != wxNOT_FOUND) {
            refValueStr = refValueStr.Remove(0, (size_t) start + 6);
            int end = refValueStr.Find(" ");
            if (end != wxNOT_FOUND) {
                refValueStr = refValueStr.Remove((size_t) end, refValueStr.Length() - end);
            }
            refValue = asTime::GetTimeFromString(refValueStr);
        } else {
            wxLogError(_("Time reference could not be extracted."));
            return false;
        }
    }

    m_fStr.timeStart = ConvertToMjd(timeFirstVal, refValue);
    m_fStr.timeEnd = ConvertToMjd(timeLastVal, refValue);
    m_fStr.timeStep = asRound(24 * (m_fStr.timeEnd - m_fStr.timeStart) / (m_fStr.timeLength - 1));
    m_fStr.firstHour = fmod(24 * m_fStr.timeStart, m_fStr.timeStep);

    return true;
}

bool asPredictor::ExtractLevelAxis(asFileNetcdf &ncFile)
{
    if (m_fStr.hasLevelDim) {
        m_fStr.levels = a1f(ncFile.GetVarLength(m_fStr.dimLevelName));

        nc_type ncTypeLevel = ncFile.GetVarType(m_fStr.dimLevelName);
        switch (ncTypeLevel) {
            case NC_FLOAT:
                ncFile.GetVar(m_fStr.dimLevelName, &m_fStr.levels[0]);
                break;
            case NC_INT: {
                a1i axisLevelInt(ncFile.GetVarLength(m_fStr.dimLevelName));
                ncFile.GetVar(m_fStr.dimLevelName, &axisLevelInt[0]);
                for (int i = 0; i < axisLevelInt.size(); ++i) {
                    m_fStr.levels[i] = (float) axisLevelInt[i];
                }
            }
                break;
            case NC_DOUBLE: {
                a1d axisLevelDouble(ncFile.GetVarLength(m_fStr.dimLevelName));
                ncFile.GetVar(m_fStr.dimLevelName, &axisLevelDouble[0]);
                for (int i = 0; i < axisLevelDouble.size(); ++i) {
                    m_fStr.levels[i] = (float) axisLevelDouble[i];
                }
            }
                break;
            default:
                wxLogError(_("Variable type not supported yet for the level dimension."));
                return false;
        }
    }

    return true;
}

bool asPredictor::ExtractSpatialAxes(asFileNetcdf &ncFile)
{
    m_fStr.lons = a1f(ncFile.GetVarLength(m_fStr.dimLonName));
    m_fStr.lats = a1f(ncFile.GetVarLength(m_fStr.dimLatName));

    wxASSERT(ncFile.GetVarType(m_fStr.dimLonName) == ncFile.GetVarType(m_fStr.dimLatName));
    nc_type ncTypeAxes = ncFile.GetVarType(m_fStr.dimLonName);
    switch (ncTypeAxes) {
        case NC_FLOAT:
            ncFile.GetVar(m_fStr.dimLonName, &m_fStr.lons[0]);
            ncFile.GetVar(m_fStr.dimLatName, &m_fStr.lats[0]);
            break;
        case NC_DOUBLE: {
            a1d axisLonDouble(ncFile.GetVarLength(m_fStr.dimLonName));
            a1d axisLatDouble(ncFile.GetVarLength(m_fStr.dimLatName));
            ncFile.GetVar(m_fStr.dimLonName, &axisLonDouble[0]);
            ncFile.GetVar(m_fStr.dimLatName, &axisLatDouble[0]);
            for (int i = 0; i < axisLonDouble.size(); ++i) {
                m_fStr.lons[i] = (float) axisLonDouble[i];
            }
            for (int i = 0; i < axisLatDouble.size(); ++i) {
                m_fStr.lats[i] = (float) axisLatDouble[i];
            }
        }
            break;
        default:
            wxLogError(_("Variable type not supported yet for the level dimension."));
            return false;
    }

    return true;
}

bool asPredictor::ParseFileStructure(asFileGrib2 *gbFile0, asFileGrib2 *gbFile1)
{
    // Get full axes from the file
    gbFile0->GetXaxis(m_fStr.lons);
    gbFile0->GetYaxis(m_fStr.lats);

    if (m_fStr.hasLevelDim && !m_fStr.singleLevel) {
        wxLogError(_("The level dimension is not yet implemented for Grib files."));
        return false;
    }

    // Yet handle a unique time value per file.
    m_fStr.timeLength = 1;
    m_fStr.timeStart = gbFile0->GetTime();
    m_fStr.timeEnd = gbFile0->GetTime();

    if(gbFile1 != nullptr) {
        double secondFileTime = gbFile1->GetTime();
        m_fStr.timeStep = asRound(24 * (secondFileTime - m_fStr.timeStart));
        m_fStr.firstHour = fmod(24 * m_fStr.timeStart, m_fStr.timeStep);
    }

    return CheckFileStructure();
}

bool asPredictor::CheckFileStructure()
{
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

asGeoAreaCompositeGrid *asPredictor::CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea)
{
    wxASSERT(m_fStr.lons.size() > 0);
    wxASSERT(m_fStr.lats.size() > 0);

    if (desiredArea) {
        m_fInd.lonStep = 1;
        m_fInd.latStep = 1;

        double dataXmin, dataYmin, dataXstep, dataYstep;
        int dataXptsnb, dataYptsnb;
        double xAxisStep = abs(m_fStr.lons[1]-m_fStr.lons[0]);
        double yAxisStep = abs(m_fStr.lats[1]-m_fStr.lats[0]);
        wxString gridType = desiredArea->GetGridTypeString();
        if (gridType.IsSameAs("Regular", false)) {
            double xAxisShift = fmod(m_fStr.lons[0], xAxisStep);
            double yAxisShift = fmod(m_fStr.lats[0], yAxisStep);
            dataXmin = floor((desiredArea->GetAbsoluteXmin() - xAxisShift) / xAxisStep) * xAxisStep + xAxisShift;
            dataYmin = floor((desiredArea->GetAbsoluteYmin() - yAxisShift) / yAxisStep) * yAxisStep + yAxisShift;
            double dataXmax = ceil((desiredArea->GetAbsoluteXmax() - xAxisShift) / xAxisStep) * xAxisStep + xAxisShift;
            double dataYmax = ceil((desiredArea->GetAbsoluteYmax() - yAxisShift) / yAxisStep) * yAxisStep + yAxisShift;
            if (m_strideAllowed && fmod(desiredArea->GetXstep(), xAxisStep) == 0 && fmod(desiredArea->GetYstep(), yAxisStep) == 0 ) {
                // If the desired step is a multiple of the data resolution
                dataXstep = desiredArea->GetXstep();
                dataYstep = desiredArea->GetYstep();
                m_fInd.lonStep = wxRound(desiredArea->GetXstep() / xAxisStep);
                m_fInd.latStep = wxRound(desiredArea->GetYstep() / yAxisStep);
            } else {
                dataXstep = xAxisStep;
                dataYstep = yAxisStep;
            }
            dataXptsnb = wxRound((dataXmax - dataXmin) / dataXstep + 1);
            dataYptsnb = wxRound((dataYmax - dataYmin) / dataYstep + 1);
        } else {
            dataXmin = desiredArea->GetAbsoluteXmin();
            dataYmin = desiredArea->GetAbsoluteYmin();
            dataXstep = desiredArea->GetXstep();
            dataYstep = desiredArea->GetYstep();
            dataXptsnb = desiredArea->GetXaxisPtsnb();
            dataYptsnb = desiredArea->GetYaxisPtsnb();
            if (abs(dataXstep - xAxisStep) > 0.1 * dataXstep || abs(dataYstep - yAxisStep) > 0.1 * dataYstep) {
                wxLogError(_("Interpolation is not allowed on irregular grids."));
                return nullptr;
            }
        }

        asGeoAreaCompositeGrid *dataArea = asGeoAreaCompositeGrid::GetInstance(gridType, dataXmin, dataXptsnb,
                                                                               dataXstep, dataYmin, dataYptsnb,
                                                                               dataYstep, desiredArea->GetLevel(),
                                                                               asNONE, asFLAT_ALLOWED);

        // Get axes length for preallocation
        m_lonPtsnb = dataArea->GetXaxisPtsnb();
        m_latPtsnb = dataArea->GetYaxisPtsnb();

        return dataArea;
    }

    return nullptr;
}

asGeoAreaCompositeGrid *asPredictor::AdjustAxes(asGeoAreaCompositeGrid *dataArea, vvva2f &compositeData)
{
    wxASSERT(m_fStr.lons.size()> 0);
    wxASSERT(m_fStr.lats.size()> 0);

    if (!m_axesChecked) {
        if (dataArea == nullptr) {
            // Get axes length for preallocation
            m_lonPtsnb = int(m_fStr.lons.size());
            m_latPtsnb = int(m_fStr.lats.size());
            m_axisLon = m_fStr.lons;
            m_axisLat = m_fStr.lats;
        } else {
            // Check that requested data do not overtake the file
            for (int iComp = 0; iComp < dataArea->GetNbComposites(); iComp++) {
                a1d axisLonComp = dataArea->GetXaxisComposite(iComp);

                wxASSERT(m_fStr.lons[m_fStr.lons.size() - 1] > m_fStr.lons[0]);
                wxASSERT(axisLonComp[axisLonComp.size() - 1] >= axisLonComp[0]);

                // Condition for change: The composite must not be fully outside (considered as handled).
                if (axisLonComp[axisLonComp.size() - 1] > m_fStr.lons[m_fStr.lons.size() - 1] &&
                    axisLonComp[0] <= m_fStr.lons[m_fStr.lons.size() - 1]) {
                    // If the last value corresponds to the maximum value of the reference system, create a new composite
                    if (axisLonComp[axisLonComp.size() - 1] == dataArea->GetAxisXmax() && dataArea->GetNbComposites() == 1) {
                        dataArea->SetLastRowAsNewComposite();
                        compositeData = vvva2f((unsigned long) dataArea->GetNbComposites());
					} else if (axisLonComp[axisLonComp.size() - 1] == dataArea->GetAxisXmax() && dataArea->GetNbComposites() > 1) {
                        dataArea->RemoveLastRowOnComposite(iComp);
                    } else if (axisLonComp[axisLonComp.size() - 1] != dataArea->GetAxisXmax()) {
                        wxLogVerbose(_("Correcting the longitude extent according to the file limits."));
                        double xWidth = m_fStr.lons[m_fStr.lons.size() - 1] - dataArea->GetAbsoluteXmin();
                        wxASSERT(xWidth >= 0);
                        int xPtsNb = 1 + xWidth / dataArea->GetXstep();
                        wxLogDebug(_("xPtsNb = %d."), xPtsNb);
                        asGeoAreaCompositeGrid *newdataArea = asGeoAreaCompositeGrid::GetInstance(
                                dataArea->GetGridTypeString(), dataArea->GetAbsoluteXmin(), xPtsNb,
                                dataArea->GetXstep(), dataArea->GetAbsoluteYmin(), dataArea->GetYaxisPtsnb(),
                                dataArea->GetYstep(), dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            a1d axisLon = dataArea->GetXaxis();
            m_axisLon.resize(axisLon.size());
            for (int i = 0; i < axisLon.size(); i++) {
                m_axisLon[i] = (float) axisLon[i];
            }
            m_lonPtsnb = dataArea->GetXaxisPtsnb();
            wxASSERT_MSG(m_axisLon.size() == m_lonPtsnb,
                         wxString::Format("m_axisLon.size()=%d, m_lonPtsnb=%d", (int) m_axisLon.size(), m_lonPtsnb));

            // Check that requested data do not overtake the file
            for (int iComp = 0; iComp < dataArea->GetNbComposites(); iComp++) {
                a1d axisLatComp = dataArea->GetYaxisComposite(iComp);

                if (m_fStr.lats[m_fStr.lats.size() - 1] > m_fStr.lats[0]) {
                    wxASSERT(axisLatComp[axisLatComp.size() - 1] >= axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size() - 1] > m_fStr.lats[m_fStr.lats.size() - 1] &&
                        axisLatComp[0] < m_fStr.lats[m_fStr.lats.size() - 1]) {
                        wxLogVerbose(_("Correcting the latitude extent according to the file limits."));
                        double yWidth = m_fStr.lats[m_fStr.lats.size() - 1] - dataArea->GetAbsoluteYmin();
                        wxASSERT(yWidth >= 0);
                        int yPtsNb = 1 + yWidth / dataArea->GetYstep();
                        wxLogDebug(_("yPtsNb = %d."), yPtsNb);
                        asGeoAreaCompositeGrid *newdataArea = asGeoAreaCompositeGrid::GetInstance(
                                dataArea->GetGridTypeString(), dataArea->GetAbsoluteXmin(), dataArea->GetXaxisPtsnb(),
                                dataArea->GetXstep(), dataArea->GetAbsoluteYmin(), yPtsNb, dataArea->GetYstep(),
                                dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }

                } else {
                    wxASSERT(axisLatComp[axisLatComp.size() - 1] >= axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size() - 1] > m_fStr.lats[0] && axisLatComp[0] < m_fStr.lats[0]) {
                        wxLogVerbose(_("Correcting the latitude extent according to the file limits."));
                        double yWidth = m_fStr.lats[0] - dataArea->GetAbsoluteYmin();
                        wxASSERT(yWidth >= 0);
                        int yPtsNb = 1 + yWidth / dataArea->GetYstep();
                        wxLogDebug(_("yPtsNb = %d."), yPtsNb);
                        asGeoAreaCompositeGrid *newdataArea = asGeoAreaCompositeGrid::GetInstance(
                                dataArea->GetGridTypeString(), dataArea->GetAbsoluteXmin(), dataArea->GetXaxisPtsnb(),
                                dataArea->GetXstep(), dataArea->GetAbsoluteYmin(), yPtsNb, dataArea->GetYstep(),
                                dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            a1d axisLat = dataArea->GetYaxis();
            m_axisLat.resize(axisLat.size());
            for (int i = 0; i < axisLat.size(); i++) {
                // Latitude axis in reverse order
                m_axisLat[i] = (float) axisLat[axisLat.size() - 1 - i];
            }
            m_latPtsnb = dataArea->GetYaxisPtsnb();
            wxASSERT_MSG(m_axisLat.size() == m_latPtsnb,
                         wxString::Format("m_axisLat.size()=%d, m_latPtsnb=%d", (int) m_axisLat.size(), m_latPtsnb));

            compositeData = vvva2f((unsigned long) dataArea->GetNbComposites());
        }

        m_axesChecked = true;
    }

    return dataArea;
}

void asPredictor::AssignGribCode(const int arr[])
{
    m_gribCode.clear();
    for (int i = 0; i < 4; ++i) {
        m_gribCode.push_back(arr[i]);
    }
}

size_t *asPredictor::GetIndexesStartNcdf(int iArea) const
{
    if (!m_isEnsemble) {
        if (m_fStr.hasLevelDim) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fInd.timeStart;
            array[1] = (size_t) m_fInd.level;
            array[2] = (size_t) m_fInd.areas[iArea].latStart;
            array[3] = (size_t) m_fInd.areas[iArea].lonStart;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t) m_fInd.timeStart;
            array[1] = (size_t) m_fInd.areas[iArea].latStart;
            array[2] = (size_t) m_fInd.areas[iArea].lonStart;

            return array;
        }
    } else {
        if (m_fStr.hasLevelDim) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t) m_fInd.timeStart;
            array[1] = (size_t) m_fInd.memberStart;
            array[2] = (size_t) m_fInd.level;
            array[3] = (size_t) m_fInd.areas[iArea].latStart;
            array[4] = (size_t) m_fInd.areas[iArea].lonStart;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fInd.timeStart;
            array[1] = (size_t) m_fInd.memberStart;
            array[2] = (size_t) m_fInd.areas[iArea].latStart;
            array[3] = (size_t) m_fInd.areas[iArea].lonStart;

            return array;
        }
    }

    return nullptr;
}

size_t *asPredictor::GetIndexesCountNcdf(int iArea) const
{
    if (!m_isEnsemble) {
        if (m_fStr.hasLevelDim) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fInd.timeCount;
            array[1] = 1;
            array[2] = (size_t) m_fInd.areas[iArea].latCount;
            array[3] = (size_t) m_fInd.areas[iArea].lonCount;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t) m_fInd.timeCount;
            array[1] = (size_t) m_fInd.areas[iArea].latCount;
            array[2] = (size_t) m_fInd.areas[iArea].lonCount;

            return array;
        }
    } else {
        if (m_fStr.hasLevelDim) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t) m_fInd.timeCount;
            array[1] = (size_t) m_fInd.memberCount;
            array[2] = 1;
            array[3] = (size_t) m_fInd.areas[iArea].latCount;
            array[4] = (size_t) m_fInd.areas[iArea].lonCount;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fInd.timeCount;
            array[1] = (size_t) m_fInd.memberCount;
            array[2] = (size_t) m_fInd.areas[iArea].latCount;
            array[3] = (size_t) m_fInd.areas[iArea].lonCount;

            return array;
        }
    }

    return nullptr;
}

ptrdiff_t *asPredictor::GetIndexesStrideNcdf() const
{
    if (!m_isEnsemble) {
        if (m_fStr.hasLevelDim) {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t) m_fInd.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t) m_fInd.latStep;
            array[3] = (ptrdiff_t) m_fInd.lonStep;

            return array;
        } else {
            static ptrdiff_t array[3] = {0, 0, 0};
            array[0] = (ptrdiff_t) m_fInd.timeStep;
            array[1] = (ptrdiff_t) m_fInd.latStep;
            array[2] = (ptrdiff_t) m_fInd.lonStep;

            return array;
        }
    } else {
        if (m_fStr.hasLevelDim) {
            static ptrdiff_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (ptrdiff_t) m_fInd.timeStep;
            array[1] = 1;
            array[2] = 1;
            array[3] = (ptrdiff_t) m_fInd.latStep;
            array[4] = (ptrdiff_t) m_fInd.lonStep;

            return array;
        } else {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t) m_fInd.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t) m_fInd.latStep;
            array[3] = (ptrdiff_t) m_fInd.lonStep;

            return array;
        }
    }

    return nullptr;
}

int *asPredictor::GetIndexesStartGrib(int iArea) const
{
    static int array[2] = {0, 0};
    array[0] = m_fInd.areas[iArea].lonStart;
    array[1] = m_fInd.areas[iArea].latStart;

    return array;
}

int *asPredictor::GetIndexesCountGrib(int iArea) const
{
    static int array[2] = {0, 0};
    array[0] = m_fInd.areas[iArea].lonCount;
    array[1] = m_fInd.areas[iArea].latCount;

    return array;
}

bool asPredictor::GetDataFromFile(asFileNetcdf &ncFile, vvva2f &compositeData)
{
    bool isShort = (ncFile.GetVarType(m_fileVarName) == NC_SHORT);
    bool isFloat = (ncFile.GetVarType(m_fileVarName) == NC_FLOAT);

    if (!isShort && !isFloat) {
        wxLogError(_("Loading data other than short or float is not implemented yet."));
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
    if (dataAddOffset == 0 && dataScaleFactor == 1)
        scalingNeeded = false;

    vvf vectData;

    for (int iArea = 0; iArea < compositeData.size(); iArea++) {

        // Create the arrays to receive the data
        vf dataF;
        vs dataS;

        // Resize the arrays to store the new data
        unsigned int totLength = (unsigned int) m_fInd.memberCount * m_fInd.timeArrayCount *
                                 m_fInd.areas[iArea].latCount * m_fInd.areas[iArea].lonCount;
        wxASSERT(totLength > 0);
        dataF.resize(totLength);
        if (isShort) {
            dataS.resize(totLength);
        }

        // Fill empty beginning with NaNs
        int indexBegining = 0;
        if (m_fInd.cutStart > 0) {
            int latlonlength = m_fInd.memberCount * m_fInd.areas[iArea].latCount *
                               m_fInd.areas[iArea].lonCount;
            for (int iEmpty = 0; iEmpty < m_fInd.cutStart; iEmpty++) {
                for (int iEmptylatlon = 0; iEmptylatlon < latlonlength; iEmptylatlon++) {
                    dataF[indexBegining] = NaNf;
                    indexBegining++;
                }
            }
        }

        // Fill empty end with NaNs
        int indexEnd = m_fInd.memberCount * m_fInd.timeCount * m_fInd.areas[iArea].latCount *
                       m_fInd.areas[iArea].lonCount - 1;
        if (m_fInd.cutEnd > 0) {
            int latlonlength = m_fInd.memberCount * m_fInd.areas[iArea].latCount *
                               m_fInd.areas[iArea].lonCount;
            for (int iEmpty = 0; iEmpty < m_fInd.cutEnd; iEmpty++) {
                for (int iEmptylatlon = 0; iEmptylatlon < latlonlength; iEmptylatlon++) {
                    indexEnd++;
                    dataF[indexEnd] = NaNf;
                }
            }
        }

        // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
        if (isFloat) {
            ncFile.GetVarSample(m_fileVarName, GetIndexesStartNcdf(iArea), GetIndexesCountNcdf(iArea),
                                GetIndexesStrideNcdf(), &dataF[indexBegining]);
        } else if (isShort) {
            ncFile.GetVarSample(m_fileVarName, GetIndexesStartNcdf(iArea), GetIndexesCountNcdf(iArea),
                                GetIndexesStrideNcdf(), &dataS[indexBegining]);
            dataF = vf(dataS.begin(), dataS.end());
        }

        // Keep data for later treatment
        vectData.push_back(dataF);
    }

    // Allocate space into compositeData if not already done
    if (compositeData[0].capacity() == 0) {
        unsigned int totSize = 0;
        for (int iArea = 0; iArea < compositeData.size(); iArea++) {
            totSize += m_fInd.memberCount * m_time.size() * m_fInd.areas[iArea].latCount *
                       (m_fInd.areas[iArea].lonCount + 1); // +1 in case of a border
        }
        compositeData.reserve(totSize);
    }

    // Transfer data
    for (int iArea = 0; iArea < compositeData.size(); iArea++) {
        // Extract data
        vf data = vectData[iArea];

        // Loop to extract the data from the array
        int ind = 0;
        for (int iTime = 0; iTime < m_fInd.timeArrayCount; iTime++) {
            va2f memlatlonData;
            for (int iMem = 0; iMem < m_fInd.memberCount; iMem++) {
                a2f latlonData(m_fInd.areas[iArea].latCount, m_fInd.areas[iArea].lonCount);

                for (int iLat = 0; iLat < m_fInd.areas[iArea].latCount; iLat++) {
                    for (int iLon = 0; iLon < m_fInd.areas[iArea].lonCount; iLon++) {
                        ind = iLon + iLat * m_fInd.areas[iArea].lonCount +
                              iMem * m_fInd.areas[iArea].lonCount * m_fInd.areas[iArea].latCount +
                              iTime * m_fInd.memberCount * m_fInd.areas[iArea].lonCount * m_fInd.areas[iArea].latCount;
                        if (m_fStr.lats.size() > 0 && m_fStr.lats[1] > m_fStr.lats[0]) {
                            int latRevIndex = m_fInd.areas[iArea].latCount - 1 - iLat;
                            ind = iLon + latRevIndex * m_fInd.areas[iArea].lonCount +
                                  iMem * m_fInd.areas[iArea].lonCount * m_fInd.areas[iArea].latCount +
                                  iTime * m_fInd.memberCount * m_fInd.areas[iArea].lonCount * m_fInd.areas[iArea].latCount;
                        }

                        latlonData(iLat, iLon) = data[ind];

                        // Check if not NaN
                        bool notNan = true;
                        for (size_t iNan = 0; iNan < m_nanValues.size(); iNan++) {
                            if (data[ind] == m_nanValues[iNan] || latlonData(iLat, iLon) == m_nanValues[iNan]) {
                                notNan = false;
                            }
                        }
                        if (!notNan) {
                            latlonData(iLat, iLon) = NaNf;
                        }
                    }
                }

                if (scalingNeeded) {
                    latlonData = latlonData * dataScaleFactor + dataAddOffset;
                }
                memlatlonData.push_back(latlonData);
            }
            compositeData[iArea].push_back(memlatlonData);
        }
        data.clear();
    }

    return true;
}

bool asPredictor::GetDataFromFile(asFileGrib2 &gbFile, vvva2f &compositeData)
{
    // Grib files do not handle stride
    if (m_fInd.lonStep != 1 || m_fInd.latStep != 1) {
        wxLogError(_("Grib files do not handle stride."));
        return false;
    }

    vvf vectData;

    for (int iArea = 0; iArea < compositeData.size(); iArea++) {

        // Create the arrays to receive the data
        vf dataF;

        // Resize the arrays to store the new data
        unsigned int totLength = (unsigned int) m_fInd.memberCount * m_fInd.timeArrayCount *
                                 m_fInd.areas[iArea].latCount * m_fInd.areas[iArea].lonCount;
        wxASSERT(totLength > 0);
        dataF.resize(totLength);

        // Extract data
        gbFile.GetVarArray(GetIndexesStartGrib(iArea), GetIndexesCountGrib(iArea), &dataF[0]);

        // Keep data for later treatment
        vectData.push_back(dataF);
    }

    // Allocate space into compositeData if not already done
    if (compositeData[0].capacity() == 0) {
        unsigned int totSize = 0;
        for (int iArea = 0; iArea < compositeData.size(); iArea++) {
            totSize += m_fInd.memberCount * m_time.size() * m_fInd.areas[iArea].latCount *
                       (m_fInd.areas[iArea].lonCount + 1); // +1 in case of a border
        }
        compositeData.reserve(totSize);
    }

    // Transfer data
    for (int iArea = 0; iArea < compositeData.size(); iArea++) {
        // Extract data
        vf data = vectData[iArea];
        va2f memlatlonData;

        for (int iMem = 0; iMem < m_fInd.memberCount; iMem++) {

            // Loop to extract the data from the array
            int ind = 0;
            a2f latlonData(m_fInd.areas[iArea].latCount, m_fInd.areas[iArea].lonCount);

            for (int iLat = 0; iLat < m_fInd.areas[iArea].latCount; iLat++) {
                for (int iLon = 0; iLon < m_fInd.areas[iArea].lonCount; iLon++) {
                    int latRevIndex = m_fInd.areas[iArea].latCount - 1 - iLat; // Index reversed in Grib files
                    ind = iLon + latRevIndex * m_fInd.areas[iArea].lonCount;
                    latlonData(iLat, iLon) = data[ind];

                    // Check if not NaN
                    bool notNan = true;
                    for (double nanValue : m_nanValues) {
                        if (data[ind] == nanValue || latlonData(iLat, iLon) == nanValue) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latlonData(iLat, iLon) = NaNf;
                    }
                }
            }
            memlatlonData.push_back(latlonData);
        }
        compositeData[iArea].push_back(memlatlonData);

        data.clear();
    }

    return true;
}

bool asPredictor::TransformData(vvva2f &compositeData)
{
    // See http://www.ecmwf.int/en/faq/geopotential-defined-units-m2/s2-both-pressure-levels-and-surface-orography-how-can-height
    if (m_parameter == Geopotential) {
        for (auto &area : compositeData) {
            for (int iTime = 0; iTime < area.size(); iTime++) {
                for (int iMem = 0; iMem < area[0].size(); iMem++) {
                    area[iTime][iMem] = area[iTime][iMem] / 9.80665;
                }
            }
        }
        m_parameter = GeopotentialHeight;
        m_parameterName = "Geopotential height";
        m_unit = m;
    }

    return true;
}

bool asPredictor::Inline()
{
    // Already inlined
    if (m_lonPtsnb == 1 || m_latPtsnb == 1) {
        return true;
    }

    wxASSERT(!m_data.empty());

    auto timeSize = (unsigned int) m_data.size();
    auto membersNb = (unsigned int) m_data[0].size();
    auto cols = (unsigned int) m_data[0][0].cols();
    auto rows = (unsigned int) m_data[0][0].rows();

    a2f inlineData = a2f::Zero(1, cols * rows);

    vva2f newData;
    newData.reserve((unsigned int) (membersNb * m_time.size() * m_lonPtsnb * m_latPtsnb));
    newData.resize(timeSize);

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iRow = 0; iRow < rows; iRow++) {
                inlineData.block(0, iRow * cols, 1, cols) = m_data[iTime][iMem].row(iRow);
            }
            newData[iTime].push_back(inlineData);
        }
    }

    m_data = newData;

    m_latPtsnb = (int) m_data[0][0].rows();
    m_lonPtsnb = (int) m_data[0][0].cols();
    a1f emptyAxis(1);
    emptyAxis[0] = NaNf;
    m_axisLat = emptyAxis;
    m_axisLon = emptyAxis;

    return true;
}

bool asPredictor::MergeComposites(vvva2f &compositeData, asGeoAreaCompositeGrid *area)
{
    if (area) {
        // Get a container with the final size
        unsigned long sizeTime = compositeData[0].size();
        unsigned long membersNb = compositeData[0][0].size();
        m_data = vva2f(sizeTime, va2f(membersNb, a2f(m_latPtsnb, m_lonPtsnb)));

        a2f blockUL, blockLL, blockUR, blockLR;
        int isblockUL = asNONE, isblockLL = asNONE, isblockUR = asNONE, isblockLR = asNONE;

        // Resize containers for composite areas
        for (int iArea = 0; iArea < area->GetNbComposites(); iArea++) {
            if ((area->GetComposite(iArea).GetXmax() == area->GetXmax()) &
                (area->GetComposite(iArea).GetYmin() == area->GetYmin())) {
                blockUL.resize(compositeData[iArea][0][0].rows(), compositeData[iArea][0][0].cols());
                isblockUL = iArea;
            } else if ((area->GetComposite(iArea).GetXmin() == area->GetXmin()) &
                       (area->GetComposite(iArea).GetYmin() == area->GetYmin())) {
                blockUR.resize(compositeData[iArea][0][0].rows(), compositeData[iArea][0][0].cols());
                isblockUR = iArea;
            } else if ((area->GetComposite(iArea).GetXmax() == area->GetXmax()) &
                       (area->GetComposite(iArea).GetYmax() == area->GetYmax())) {
                blockLL.resize(compositeData[iArea][0][0].rows(), compositeData[iArea][0][0].cols());
                isblockLL = iArea;
            } else if ((area->GetComposite(iArea).GetXmin() == area->GetXmin()) &
                       (area->GetComposite(iArea).GetYmax() == area->GetYmax())) {
                blockLR.resize(compositeData[iArea][0][0].rows(), compositeData[iArea][0][0].cols());
                isblockLR = iArea;
            } else {
                wxLogError(_("The data composite was not identified."));
                return false;
            }
        }

        // Merge the composite data together
        for (unsigned int iTime = 0; iTime < sizeTime; iTime++) {
            for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
                // Append the composite areas
                for (int iArea = 0; iArea < area->GetNbComposites(); iArea++) {
                    if (iArea == isblockUL) {
                        blockUL = compositeData[iArea][iTime][iMem];
                        m_data[iTime][iMem].topLeftCorner(blockUL.rows(), blockUL.cols()) = blockUL;
                    } else if (iArea == isblockUR) {
                        blockUR = compositeData[iArea][iTime][iMem];
                        m_data[iTime][iMem].block(0, m_lonPtsnb - blockUR.cols(), blockUR.rows(), blockUR.cols()) = blockUR;
                    } else if (iArea == isblockLL) {
                        blockLL = compositeData[iArea][iTime][iMem];
                        wxLogError(_("Not yet implemented."));
                        return false;
                    } else if (iArea == isblockLR) {
                        blockLR = compositeData[iArea][iTime][iMem];
                        wxLogError(_("Not yet implemented."));
                        return false;
                    } else {
                        wxLogError(_("The data composite cannot be build."));
                        return false;
                    }
                }
            }
        }
    } else {
        m_data = compositeData[0];
    }

    return true;
}

bool asPredictor::InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea)
{
    wxASSERT(dataArea);
    wxASSERT(dataArea->GetNbComposites() > 0);
    wxASSERT(desiredArea);
    wxASSERT(desiredArea->GetNbComposites() > 0);
    bool changeXstart = false, changeXsteps = false, changeYstart = false, changeYsteps = false;

    // Check beginning on longitudes
    if (dataArea->GetAbsoluteXmin() != desiredArea->GetAbsoluteXmin()) {
        changeXstart = true;
    }

    // Check beginning on latitudes
    if (dataArea->GetAbsoluteYmin() != desiredArea->GetAbsoluteYmin()) {
        changeYstart = true;
    }

    // Check the cells size on longitudes
    if (!dataArea->GridsOverlay(desiredArea)) {
        changeXsteps = true;
        changeYsteps = true;
    }

    // Proceed to the interpolation
    if (changeXstart || changeYstart || changeXsteps || changeYsteps) {
        // Containers for results
        int finalLengthLon = desiredArea->GetXaxisPtsnb();
        int finalLengthLat = desiredArea->GetYaxisPtsnb();
        vva2f latlonTimeData(m_data.size(), va2f(m_data[0].size(), a2f(finalLengthLat, finalLengthLon)));

        // Creation of the axes
        a1f axisDataLon;
        if (dataArea->GetXaxisPtsnb() > 1) {
            axisDataLon = a1f::LinSpaced(Eigen::Sequential, dataArea->GetXaxisPtsnb(), dataArea->GetAbsoluteXmin(),
                                         dataArea->GetAbsoluteXmax());
        } else {
            axisDataLon.resize(1);
            axisDataLon << dataArea->GetAbsoluteXmin();
        }

        a1f axisDataLat;
        if (dataArea->GetYaxisPtsnb() > 1) {
            axisDataLat = a1f::LinSpaced(Eigen::Sequential, dataArea->GetYaxisPtsnb(), dataArea->GetAbsoluteYmax(),
                                         dataArea->GetAbsoluteYmin()); // From top to bottom
        } else {
            axisDataLat.resize(1);
            axisDataLat << dataArea->GetAbsoluteYmax();
        }

        a1f axisFinalLon;
        if (desiredArea->GetXaxisPtsnb() > 1) {
            axisFinalLon = a1f::LinSpaced(Eigen::Sequential, desiredArea->GetXaxisPtsnb(),
                                          desiredArea->GetAbsoluteXmin(), desiredArea->GetAbsoluteXmax());
        } else {
            axisFinalLon.resize(1);
            axisFinalLon << desiredArea->GetAbsoluteXmin();
        }

        a1f axisFinalLat;
        if (desiredArea->GetYaxisPtsnb() > 1) {
            axisFinalLat = a1f::LinSpaced(Eigen::Sequential, desiredArea->GetYaxisPtsnb(),
                                          desiredArea->GetAbsoluteYmax(),
                                          desiredArea->GetAbsoluteYmin()); // From top to bottom
        } else {
            axisFinalLat.resize(1);
            axisFinalLat << desiredArea->GetAbsoluteYmax();
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
        for (unsigned int iTime = 0; iTime < m_data.size(); iTime++) {
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
                                                                 &axisDataLat[axisDataLatEnd],
                                                                 axisFinalLat[iLat]);
                        indexYceil = indexLastLat +
                                asFindCeil(&axisDataLat[indexLastLat], &axisDataLat[axisDataLatEnd], axisFinalLat[iLat]);
                    }

                    if (indexYfloor == asOUT_OF_RANGE || indexYfloor == asNOT_FOUND ||
                        indexYceil == asOUT_OF_RANGE || indexYceil == asNOT_FOUND) {
                        wxLogError(_("The desired point is not available in the data for interpolation. Latitude %f was not found inbetween %f (index %d) to %f (index %d) (size = %d)."),
                                   axisFinalLat[iLat], axisDataLat[indexLastLat], indexLastLat,
                                   axisDataLat[axisDataLatEnd], axisDataLatEnd, (int) axisDataLat.size());
                        return false;
                    }
                    wxASSERT_MSG(indexYfloor >= 0, wxString::Format("%f in %f to %f",
                                                                    axisFinalLat[iLat],
                                                                    axisDataLat[indexLastLat],
                                                                    axisDataLat[axisDataLatEnd]));
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
                                                                     &axisDataLon[axisDataLonEnd],
                                                                     axisFinalLon[iLon]);
                            indexXceil = indexLastLon +
                                    asFindCeil(&axisDataLon[indexLastLon], &axisDataLon[axisDataLonEnd],
                                               axisFinalLon[iLon]);
                        }

                        if (indexXfloor == asOUT_OF_RANGE || indexXfloor == asNOT_FOUND ||
                            indexXceil == asOUT_OF_RANGE || indexXceil == asNOT_FOUND) {
                            wxLogError(_("The desired point is not available in the data for interpolation. Longitude %f was not found inbetween %f to %f."),
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

                            latlonTimeData[iTime][iMem](iLat, iLon) =
                                    (1 - dX) * (1 - dY) * valLLcorner + (1 - dX) * (dY) * valULcorner;
                        } else if (dY == 0) {
                            valLLcorner = m_data[iTime][iMem](indexYfloor, indexXfloor);
                            valLRcorner = m_data[iTime][iMem](indexYfloor, indexXceil);

                            latlonTimeData[iTime][iMem](iLat, iLon) =
                                    (1 - dX) * (1 - dY) * valLLcorner + (dX) * (1 - dY) * valLRcorner;
                        } else {
                            valLLcorner = m_data[iTime][iMem](indexYfloor, indexXfloor);
                            valULcorner = m_data[iTime][iMem](indexYceil, indexXfloor);
                            valLRcorner = m_data[iTime][iMem](indexYfloor, indexXceil);
                            valURcorner = m_data[iTime][iMem](indexYceil, indexXceil);

                            latlonTimeData[iTime][iMem](iLat, iLon) =
                                    (1 - dX) * (1 - dY) * valLLcorner + (1 - dX) * (dY) * valULcorner +
                                    (dX) * (1 - dY) * valLRcorner + (dX) * (dY) * valURcorner;
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

float asPredictor::GetMinValue() const
{
    float minValue = m_data[0][0](0, 0);
    float tmpValue;

    for (const auto &dat : m_data) {
        for (const auto &v : dat) {
            tmpValue = v.minCoeff();
            if (tmpValue < minValue) {
                minValue = tmpValue;
            }
        }
    }

    return minValue;
}

float asPredictor::GetMaxValue() const
{
    float maxValue = m_data[0][0](0, 0);
    float tmpValue;

    for (const auto &dat : m_data) {
        for (const auto &v : dat) {
            tmpValue = v.maxCoeff();
            if (tmpValue > maxValue) {
                maxValue = tmpValue;
            }
        }
    }

    return maxValue;
}

void asPredictor::CheckLevelTypeIsDefined()
{
    if(m_product.IsEmpty()) {
        asThrowException(_("The type of product must be defined for this dataset (prefix to the variable name. Ex: press/hgt)."));
    }
}


