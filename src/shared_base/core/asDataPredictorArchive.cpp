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

#include "asDataPredictorArchive.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>
#include <asDataPredictorArchiveNcepReanalysis1.h>
#include <asDataPredictorArchiveNcepReanalysis1Subset.h>
#include <asDataPredictorArchiveNcepReanalysis1Lthe.h>
#include <asDataPredictorArchiveNcepReanalysis2.h>
#include <asDataPredictorArchiveNoaaOisst2.h>
#include <asDataPredictorArchiveNoaaOisst2Subset.h>


asDataPredictorArchive::asDataPredictorArchive(const wxString &dataId)
        : asDataPredictor(dataId)
{
    m_originalProviderStart = 0.0;
    m_originalProviderEnd = 0.0;
}

asDataPredictorArchive::~asDataPredictorArchive()
{

}

asDataPredictorArchive *asDataPredictorArchive::GetInstance(const wxString &datasetId, const wxString &dataId,
                                                            const wxString &directory)
{
    asDataPredictorArchive *predictor = NULL;

    if (datasetId.IsSameAs("NCEP_Reanalysis_v1", false)) {
        predictor = new asDataPredictorArchiveNcepReanalysis1(dataId);
    } else if (datasetId.IsSameAs("NCEP_Reanalysis_v1_subset", false)) {
        predictor = new asDataPredictorArchiveNcepReanalysis1Subset(dataId);
    } else if (datasetId.IsSameAs("NCEP_Reanalysis_v1_lthe", false)) {
        predictor = new asDataPredictorArchiveNcepReanalysis1Lthe(dataId);
    } else if (datasetId.IsSameAs("NCEP_Reanalysis_v2", false)) {
        predictor = new asDataPredictorArchiveNcepReanalysis2(dataId);
    } else if (datasetId.IsSameAs("NOAA_OISST_v2", false)) {
        predictor = new asDataPredictorArchiveNoaaOisst2(dataId);
    } else if (datasetId.IsSameAs("NOAA_OISST_v2_subset", false)) {
        predictor = new asDataPredictorArchiveNoaaOisst2Subset(dataId);
    } else {
        asLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return NULL;
    }

    if (!directory.IsEmpty()) {
        predictor->SetDirectoryPath(directory);
    }

    if (!predictor->Init()) {
        asLogError(_("The predictor did not initialize correctly."));
        return NULL;
    }

    return predictor;
}

bool asDataPredictorArchive::Init()
{
    return false;
}

bool asDataPredictorArchive::ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                              VVArray2DFloat &compositeData)
{
    VectorString filesList = GetListOfFiles(timeArray);

    if(!CheckFilesPresence(filesList)) {
        return false;
    }

#if wxUSE_GUI
    asDialogProgressBar progressBar(_("Loading data from files.\n"), int(filesList.size()));
#endif

    for (int i = 0; i < filesList.size(); i++) {
        wxString fileName = filesList[i];

#if wxUSE_GUI
        // Update the progress bar
        if (!progressBar.Update(i, wxString::Format(_("File: %s"), fileName))) {
            asLogWarning(_("The process has been canceled by the user."));
            return false;
        }
#endif

        if (!ExtractFromFile(fileName, dataArea, timeArray, compositeData)) {
            return false;
        }
    }

    return true;
}

bool asDataPredictorArchive::ExtractFromNetcdfFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                   asTimeArray &timeArray, VVArray2DFloat &compositeData)
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
    if (!ParseFileStructure(ncFile, dataArea, timeArray, compositeData)) {
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
    if (!GetAxesIndexes(ncFile, dataArea, timeArray, compositeData)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
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

bool asDataPredictorArchive::ParseFileStructure(asFileNetcdf &ncFile, asGeoAreaCompositeGrid *&dataArea,
                                                asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Get full axes from the netcdf file
    m_fileStructure.axisLon = Array1DFloat(ncFile.GetVarLength(m_fileStructure.dimLonName));
    ncFile.GetVar(m_fileStructure.dimLonName, &m_fileStructure.axisLon[0]);
    m_fileStructure.axisLat = Array1DFloat(ncFile.GetVarLength(m_fileStructure.dimLatName));
    ncFile.GetVar(m_fileStructure.dimLatName, &m_fileStructure.axisLat[0]);

    if (m_fileStructure.hasLevelDimension) {
        m_fileStructure.axisLevel = Array1DFloat(ncFile.GetVarLength(m_fileStructure.dimLevelName));
        ncFile.GetVar(m_fileStructure.dimLevelName, &m_fileStructure.axisLevel[0]);
    }

    // Time dimension takes ages to load !! Avoid and get the first value.
    m_fileStructure.axisTimeLength = ncFile.GetVarLength(m_fileStructure.dimTimeName);
    m_fileStructure.axisTimeFirstValue = ConvertToMjd(ncFile.GetVarOneDouble(m_fileStructure.dimTimeName, 0));
    m_fileStructure.axisTimeLastValue = ConvertToMjd(ncFile.GetVarOneDouble(m_fileStructure.dimTimeName, ncFile.GetVarLength(m_fileStructure.dimTimeName) - 1));

    return true;
}

size_t *asDataPredictorArchive::GetIndexesStartNcdf(int i_area) const
{
    if (m_fileStructure.hasLevelDimension) {
        static size_t array[4] = {0, 0, 0, 0};
        array[0] = (size_t) m_fileIndexes.timeStart;
        array[1] = (size_t) m_fileIndexes.level;
        array[2] = (size_t) m_fileIndexes.areas[i_area].latStart;
        array[3] = (size_t) m_fileIndexes.areas[i_area].lonStart;

        return array;
    } else {
        static size_t array[3] = {0, 0, 0};
        array[0] = (size_t) m_fileIndexes.timeStart;
        array[1] = (size_t) m_fileIndexes.areas[i_area].latStart;
        array[2] = (size_t) m_fileIndexes.areas[i_area].lonStart;

        return array;
    }

    return NULL;
}

size_t *asDataPredictorArchive::GetIndexesCountNcdf(int i_area) const
{
    if (m_fileStructure.hasLevelDimension) {
        static size_t array[4] = {0, 0, 0, 0};
        array[0] = (size_t) m_fileIndexes.timeCount;
        array[1] = 1;
        array[2] = (size_t) m_fileIndexes.areas[i_area].latCount;
        array[3] = (size_t) m_fileIndexes.areas[i_area].lonCount;

        return array;
    } else {
        static size_t array[3] = {0, 0, 0};
        array[0] = (size_t) m_fileIndexes.timeCount;
        array[1] = (size_t) m_fileIndexes.areas[i_area].latCount;
        array[2] = (size_t) m_fileIndexes.areas[i_area].lonCount;

        return array;
    }

    return NULL;
}

ptrdiff_t *asDataPredictorArchive::GetIndexesStrideNcdf(int i_area) const
{
    if (m_fileStructure.hasLevelDimension) {
        static ptrdiff_t array[4] = {0, 0, 0, 0};
        array[0] = (ptrdiff_t) m_fileIndexes.timeStep;
        array[1] = 1;
        array[2] = (ptrdiff_t) m_fileIndexes.latStep;
        array[3] = (ptrdiff_t) m_fileIndexes.lonStep;

        return array;
    } else {
        static ptrdiff_t array[3] = {0, 0, 0};
        array[0] = (ptrdiff_t) m_fileIndexes.timeStep;
        array[1] = (ptrdiff_t) m_fileIndexes.latStep;
        array[2] = (ptrdiff_t) m_fileIndexes.lonStep;

        return array;
    }

    return NULL;
}

bool asDataPredictorArchive::GetAxesIndexes(asFileNetcdf &ncFile, asGeoAreaCompositeGrid *&dataArea,
                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    m_fileIndexes.areas.clear();

    // Get the time length
    double timeArrayIndexStart = timeArray.GetIndexFirstAfter(m_fileStructure.axisTimeFirstValue);
    double timeArrayIndexEnd = timeArray.GetIndexFirstBefore(m_fileStructure.axisTimeLastValue);
    m_fileIndexes.timeArrayCount = int(timeArrayIndexEnd - timeArrayIndexStart + 1);
    m_fileIndexes.timeCount = int(timeArrayIndexEnd - timeArrayIndexStart + 1);

    // Correct the time start and end
    double valFirstTime = m_fileStructure.axisTimeFirstValue;
    m_fileIndexes.timeStart = 0;
    m_fileIndexes.cutStart = 0;
    bool firstFile = (compositeData[0].size() == 0);
    if (firstFile) {
        m_fileIndexes.cutStart = int(timeArrayIndexStart);
    }
    m_fileIndexes.cutEnd = 0;
    while (valFirstTime < timeArray[timeArrayIndexStart]) {
        valFirstTime += m_timeStepHours / 24.0;
        m_fileIndexes.timeStart++;
    }
    if (m_fileIndexes.timeStart + m_fileIndexes.timeCount > m_fileStructure.axisTimeLength) {
        m_fileIndexes.timeCount--;
        m_fileIndexes.cutEnd++;
    }

    // Go through every area
    m_fileIndexes.areas.resize(compositeData.size());
    for (int i_area = 0; i_area < compositeData.size(); i_area++) {

        if (dataArea) {
            // Get the spatial extent
            float lonMin = (float)dataArea->GetXaxisCompositeStart(i_area);
            float latMinStart = (float)dataArea->GetYaxisCompositeStart(i_area);
            float latMinEnd = (float)dataArea->GetYaxisCompositeEnd(i_area);

            // The dimensions lengths
            m_fileIndexes.areas[i_area].lonCount = dataArea->GetXaxisCompositePtsnb(i_area);
            m_fileIndexes.areas[i_area].latCount = dataArea->GetYaxisCompositePtsnb(i_area);

            // Get the spatial indices of the desired data
            m_fileIndexes.areas[i_area].lonStart = asTools::SortedArraySearch(&m_fileStructure.axisLon[0],
                                                                    &m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1],
                                                                    lonMin, 0.01f);
            if (m_fileIndexes.areas[i_area].lonStart == asOUT_OF_RANGE) {
                // If not found, try with negative angles
                m_fileIndexes.areas[i_area].lonStart = asTools::SortedArraySearch(&m_fileStructure.axisLon[0],
                                                                        &m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1],
                                                                        lonMin - 360, 0.01f);
            }
            if (m_fileIndexes.areas[i_area].lonStart == asOUT_OF_RANGE) {
                // If not found, try with angles above 360 degrees
                m_fileIndexes.areas[i_area].lonStart = asTools::SortedArraySearch(&m_fileStructure.axisLon[0],
                                                                        &m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1],
                                                                        lonMin + 360, 0.01f);
            }
            if (m_fileIndexes.areas[i_area].lonStart < 0) {
                asLogError(wxString::Format("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ",
                                            lonMin, m_fileStructure.axisLon[0], (int) m_fileStructure.axisLon.size(),
                                            m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1]));
                return false;
            }
            wxASSERT_MSG(m_fileIndexes.areas[i_area].lonStart >= 0,
                         wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f",
                                          m_fileStructure.axisLon[0], (int) m_fileStructure.axisLon.size(),
                                          m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1], lonMin));

            int indexStartLat1 = asTools::SortedArraySearch(&m_fileStructure.axisLat[0],
                                                            &m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1],
                                                            latMinStart, 0.01f);
            int indexStartLat2 = asTools::SortedArraySearch(&m_fileStructure.axisLat[0],
                                                            &m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1],
                                                            latMinEnd, 0.01f);
            wxASSERT_MSG(indexStartLat1 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinStart, m_fileStructure.axisLat[0],
                                          m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1]));
            wxASSERT_MSG(indexStartLat2 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinEnd, m_fileStructure.axisLat[0],
                                          m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1]));
            m_fileIndexes.areas[i_area].latStart = wxMin(indexStartLat1, indexStartLat2);
        } else {
            m_fileIndexes.areas[i_area].lonStart = 0;
            m_fileIndexes.areas[i_area].latStart = 0;
            m_fileIndexes.areas[i_area].lonCount = m_lonPtsnb;
            m_fileIndexes.areas[i_area].latCount = m_latPtsnb;
        }

        if (m_fileStructure.hasLevelDimension && !m_fileStructure.singleLevel) {
            m_fileIndexes.level = asTools::SortedArraySearch(&m_fileStructure.axisLevel[0], &m_fileStructure.axisLevel[
                    m_fileStructure.axisLevel.size() - 1], m_level, 0.01f);
            if (m_fileIndexes.level < 0) {
                asLogWarning(wxString::Format(_("The desired level (%g) does not exist for %s"), m_level,
                                              m_fileVariableName));
                return false;
            }
        } else if (m_fileStructure.hasLevelDimension && m_fileStructure.singleLevel) {
            m_fileIndexes.level = 0;
        } else {
            if (m_level > 0) {
                asLogWarning(wxString::Format(_("The desired level (%g) does not exist for %s"), m_level,
                                              m_fileVariableName));
                return false;
            }
        }
    }

    return true;
}

bool asDataPredictorArchive::GetDataFromFile(asFileNetcdf &ncFile, VVArray2DFloat &compositeData)
{
    bool isShort = (ncFile.GetVarType(m_fileVariableName) == NC_SHORT);
    bool isFloat = (ncFile.GetVarType(m_fileVariableName) == NC_FLOAT);

    if(!isShort && !isFloat) {
        asLogError(_("Loading data other than short or float is not implemented yet."));
    }

    // Check if scaling is needed
    bool scalingNeeded = true;
    float dataAddOffset = ncFile.GetAttFloat("add_offset", m_fileVariableName);
    if (asTools::IsNaN(dataAddOffset))
        dataAddOffset = 0;
    float dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_fileVariableName);
    if (asTools::IsNaN(dataScaleFactor))
        dataScaleFactor = 1;
    if (dataAddOffset == 0 && dataScaleFactor == 1)
        scalingNeeded = false;

    VVectorFloat vectData;

    for (int i_area = 0; i_area < compositeData.size(); i_area++) {

        // Create the arrays to receive the data
        VectorFloat dataF;
        VectorShort dataS;

        // Resize the arrays to store the new data
        int totLength = m_fileIndexes.timeArrayCount * m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
        wxASSERT(totLength > 0);
        dataF.resize(totLength);
        if (isShort) {
            dataS.resize(totLength);
        }

        // Fill empty beginning with NaNs
        int indexBegining = 0;
        if (m_fileIndexes.cutStart > 0) {
            int latlonlength = m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
            for (int i_empty = 0; i_empty < m_fileIndexes.cutStart; i_empty++) {
                for (int i_emptylatlon = 0; i_emptylatlon < latlonlength; i_emptylatlon++) {
                    dataF[indexBegining] = NaNFloat;
                    indexBegining++;
                }
            }
        }

        // Fill empty end with NaNs
        int indexEnd = m_fileIndexes.timeCount * m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount - 1;
        if (m_fileIndexes.cutEnd > 0) {
            int latlonlength = m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
            for (int i_empty = 0; i_empty < m_fileIndexes.cutEnd; i_empty++) {
                for (int i_emptylatlon = 0; i_emptylatlon < latlonlength; i_emptylatlon++) {
                    indexEnd++;
                    dataF[indexEnd] = NaNFloat;
                }
            }
        }

        // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
        if (isFloat) {
            ncFile.GetVarSample(m_fileVariableName, GetIndexesStartNcdf(i_area), GetIndexesCountNcdf(i_area),
                                GetIndexesStrideNcdf(i_area), &dataF[indexBegining]);
        } else if (isShort) {
            ncFile.GetVarSample(m_fileVariableName, GetIndexesStartNcdf(i_area), GetIndexesCountNcdf(i_area),
                                GetIndexesStrideNcdf(i_area), &dataS[indexBegining]);
            for (int i = 0; i < dataS.size(); i++) {
                dataF[i] = (float) dataS[i];
            }
        }

        // Keep data for later treatment
        vectData.push_back(dataF);
    }

    // Allocate space into compositeData if not already done
    if (compositeData[0].capacity() == 0) {
        int totSize = 0;
        for (int i_area = 0; i_area < compositeData.size(); i_area++) {
            totSize += m_time.size() * m_fileIndexes.areas[i_area].latCount * (m_fileIndexes.areas[i_area].lonCount + 1); // +1 in case of a border
        }
        compositeData.reserve(totSize);
    }

    // Transfer data
    for (int i_area = 0; i_area < compositeData.size(); i_area++) {
        // Extract data
        VectorFloat data = vectData[i_area];

        // Loop to extract the data from the array
        int ind = 0;
        for (int i_time = 0; i_time < m_fileIndexes.timeArrayCount; i_time++) {
            Array2DFloat latlonData = Array2DFloat(m_fileIndexes.areas[i_area].latCount, m_fileIndexes.areas[i_area].lonCount);

            for (int i_lat = 0; i_lat < m_fileIndexes.areas[i_area].latCount; i_lat++) {
                for (int i_lon = 0; i_lon < m_fileIndexes.areas[i_area].lonCount; i_lon++) {
                    ind = i_lon + i_lat * m_fileIndexes.areas[i_area].lonCount + i_time * m_fileIndexes.areas[i_area].lonCount * m_fileIndexes.areas[i_area].latCount;

                    if (scalingNeeded) {
                        latlonData(i_lat, i_lon) = data[ind] * dataScaleFactor + dataAddOffset;
                    } else {
                        latlonData(i_lat, i_lon) = data[ind];
                    }

                    // Check if not NaN
                    bool notNan = true;
                    for (size_t i_nan = 0; i_nan < m_nanValues.size(); i_nan++) {
                        if (data[ind] == m_nanValues[i_nan] || latlonData(i_lat, i_lon) == m_nanValues[i_nan]) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latlonData(i_lat, i_lon) = NaNFloat;
                    }
                }
            }
            compositeData[i_area].push_back(latlonData);
        }
        data.clear();
    }

    return true;
}

bool asDataPredictorArchive::ClipToArea(asGeoAreaCompositeGrid *desiredArea)
{
    double Xmin = desiredArea->GetAbsoluteXmin();
    double Xmax = desiredArea->GetAbsoluteXmax();
    wxASSERT(m_axisLon.size() > 0);
    int XstartIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], Xmin, 0.0,
                                                 asHIDE_WARNINGS);
    int XendIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1], Xmax, 0.0,
                                               asHIDE_WARNINGS);
    if (XstartIndex < 0) {
        XstartIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1],
                                                 Xmin + desiredArea->GetAxisXmax());
        XendIndex = asTools::SortedArraySearch(&m_axisLon[0], &m_axisLon[m_axisLon.size() - 1],
                                               Xmax + desiredArea->GetAxisXmax());
        if (XstartIndex < 0 || XendIndex < 0) {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."),
                                        Xmin + desiredArea->GetAxisXmax(), Xmax + desiredArea->GetAxisXmax(),
                                        m_axisLon[0], m_axisLon[m_axisLon.size() - 1]));
            return false;
        }
    }
    if (XstartIndex < 0 || XendIndex < 0) {

        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(
                wxString::Format(_("Looking for lon %.2f and %.2f inbetween %.2f to %.2f."), Xmin, Xmax, m_axisLon[0],
                                 m_axisLon[m_axisLon.size() - 1]));
        return false;
    }
    int Xlength = XendIndex - XstartIndex + 1;

    double Ymin = desiredArea->GetAbsoluteYmin();
    double Ymax = desiredArea->GetAbsoluteYmax();
    wxASSERT(m_axisLat.size() > 0);
    int YstartIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], Ymin, 0.0,
                                                 asHIDE_WARNINGS);
    int YendIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1], Ymax, 0.0,
                                               asHIDE_WARNINGS);
    if (XstartIndex < 0) {
        YstartIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1],
                                                 Ymin + desiredArea->GetAxisYmax());
        YendIndex = asTools::SortedArraySearch(&m_axisLat[0], &m_axisLat[m_axisLat.size() - 1],
                                               Ymax + desiredArea->GetAxisYmax());
        if (YstartIndex < 0 || YendIndex < 0) {
            asLogError(_("An error occured while trying to clip data to another area (extended axis)."));
            asLogError(wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."),
                                        Ymin + desiredArea->GetAxisYmax(), Ymax + desiredArea->GetAxisYmax(),
                                        m_axisLat[0], m_axisLat[m_axisLat.size() - 1]));
            return false;
        }
    }
    if (YstartIndex < 0 || YendIndex < 0) {
        asLogError(_("An error occured while trying to clip data to another area."));
        asLogError(
                wxString::Format(_("Looking for lat %.2f and %.2f inbetween %.2f to %.2f."), Ymin, Ymax, m_axisLat[0],
                                 m_axisLat[m_axisLat.size() - 1]));
        return false;
    }

    int YstartIndexReal = wxMin(YstartIndex, YendIndex);
    int Ylength = std::abs(YendIndex - YstartIndex) + 1;

    // Check if already the correct size
    if (YstartIndexReal == 0 && XstartIndex == 0 && Ylength == m_axisLat.size() && Xlength == m_axisLon.size()) {
        if (IsPreprocessed()) {
            if (m_data[0].cols() == m_axisLon.size() && m_data[0].rows() == 2 * m_axisLat.size()) {
                // Nothing to do
                return true;
            } else {
                // Clear axes
                Array1DFloat newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNFloat;
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
            asLogError(_("The preprocessed area cannot be clipped to another area."));
            return false;
        }

        if (IsPreprocessed()) {
            wxString method = GetPreprocessMethod();
            if (method.IsSameAs("Gradients")) {
                VArray2DFloat originalData = m_data;

                if (originalData[0].cols() != m_axisLon.size() || originalData[0].rows() != 2 * m_axisLat.size()) {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format(
                            "originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d",
                            (int) originalData[0].cols(), (int) m_axisLon.size(), (int) originalData[0].rows(),
                            (int) m_axisLat.size()));
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

                for (unsigned int i = 0; i < originalData.size(); i++) {
                    Array2DFloat dat1 = originalData[i].block(YstartIndexReal, XstartIndex, Ylength - 1, Xlength);
                    Array2DFloat dat2 = originalData[i].block(YstartIndexReal + m_axisLat.size(), XstartIndex, Ylength, Xlength - 1);
                    Array2DFloat datMerged = Array2DFloat::Zero(2 * Ylength, Xlength); // Needs to be 0-filled for further simplification.
                    datMerged.block(0, 0, Ylength - 1, Xlength) = dat1;
                    datMerged.block(Ylength, 0, Ylength, Xlength - 1) = dat2;
                    m_data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else if (method.IsSameAs("FormerHumidityIndex")) {
                VArray2DFloat originalData = m_data;

                if (originalData[0].cols() != m_axisLon.size() || originalData[0].rows() != 2 * m_axisLat.size()) {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format(
                            "originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d",
                            (int) originalData[0].cols(), (int) m_axisLon.size(), (int) originalData[0].rows(),
                            (int) m_axisLat.size()));
                    return false;
                }

                for (unsigned int i = 0; i < originalData.size(); i++) {
                    Array2DFloat dat1 = originalData[i].block(YstartIndexReal, XstartIndex, Ylength, Xlength);
                    Array2DFloat dat2 = originalData[i].block(YstartIndexReal + m_axisLat.size(), XstartIndex, Ylength, Xlength);
                    Array2DFloat datMerged(2 * Ylength, Xlength);
                    datMerged.block(0, 0, Ylength, Xlength) = dat1;
                    datMerged.block(Ylength, 0, Ylength, Xlength) = dat2;
                    m_data[i] = datMerged;
                }

                Array1DFloat newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else if (method.IsSameAs("Multiply") || method.IsSameAs("Multiplication") ||
                       method.IsSameAs("HumidityFlux")) {
                VArray2DFloat originalData = m_data;

                if (originalData[0].cols() != m_axisLon.size() || originalData[0].rows() != m_axisLat.size()) {
                    asLogError(_("Wrong axes lengths (cannot be clipped to another area)."));
                    asLogError(wxString::Format(
                            "originalData[0].cols() = %d, m_axisLon.size() = %d, originalData[0].rows() = %d, m_axisLat.size() = %d",
                            (int) originalData[0].cols(), (int) m_axisLon.size(), (int) originalData[0].rows(),
                            (int) m_axisLat.size()));
                    return false;
                }

                for (unsigned int i = 0; i < originalData.size(); i++) {
                    m_data[i] = originalData[i].block(YstartIndexReal, XstartIndex, Ylength, Xlength);
                }

                Array1DFloat newAxisLon(Xlength);
                for (int i = 0; i < Xlength; i++) {
                    newAxisLon[i] = NaNFloat;
                }
                m_axisLon = newAxisLon;

                Array1DFloat newAxisLat(2 * Ylength);
                for (int i = 0; i < 2 * Ylength; i++) {
                    newAxisLat[i] = NaNFloat;
                }
                m_axisLat = newAxisLat;

                m_latPtsnb = m_axisLat.size();
                m_lonPtsnb = m_axisLon.size();

                return true;

            } else {
                asLogError(_("Wrong proprocessing definition (cannot be clipped to another area)."));
                return false;
            }
        }
    }

    VArray2DFloat originalData = m_data;
    for (unsigned int i = 0; i < originalData.size(); i++) {
        m_data[i] = originalData[i].block(YstartIndexReal, XstartIndex, Ylength, Xlength);
    }

    Array1DFloat newAxisLon(Xlength);
    for (int i = 0; i < Xlength; i++) {
        newAxisLon[i] = m_axisLon[XstartIndex + i];
    }
    m_axisLon = newAxisLon;

    Array1DFloat newAxisLat(Ylength);
    for (int i = 0; i < Ylength; i++) {
        newAxisLat[i] = m_axisLat[YstartIndexReal + i];
    }
    m_axisLat = newAxisLat;

    m_latPtsnb = m_axisLat.size();
    m_lonPtsnb = m_axisLon.size();

    return true;
}

bool asDataPredictorArchive::CheckTimeArray(asTimeArray &timeArray) const
{
    if (!timeArray.IsSimpleMode()) {
        asLogError(_("The data loading only accepts time arrays in simple mode."));
        return false;
    }

    // Check against original dataset
    if (timeArray.GetFirst() < m_originalProviderStart) {
        asLogError(wxString::Format(
                _("The requested date (%s) is anterior to the beginning of the original dataset (%s)."),
                asTime::GetStringTime(timeArray.GetFirst(), YYYYMMDD),
                asTime::GetStringTime(m_originalProviderStart, YYYYMMDD)));
        return false;
    }
    if (!asTools::IsNaN(m_originalProviderEnd)) {
        if (timeArray.GetLast() > m_originalProviderEnd) {
            asLogError(
                    wxString::Format(_("The requested date (%s) is posterior to the end of the original dataset (%s)."),
                                     asTime::GetStringTime(timeArray.GetLast(), YYYYMMDD),
                                     asTime::GetStringTime(m_originalProviderEnd, YYYYMMDD)));
            return false;
        }
    }

    // Check the time steps
    if ((timeArray.GetTimeStepDays() > 0) && (m_timeStepHours / 24.0 > timeArray.GetTimeStepDays())) {
        asLogError(_("The desired timestep is smaller than the data timestep."));
        return false;
    }
    double intpart, fractpart;
    fractpart = modf(timeArray.GetTimeStepDays() / (m_timeStepHours / 24.0), &intpart);
    if (fractpart > 0.0000001) {
        asLogError(_("The desired timestep is not a multiple of the data timestep."));
        return false;
    }
    fractpart = modf((timeArray.GetFirstDayHour() - m_firstTimeStepHours) / m_timeStepHours, &intpart);
    if (fractpart > 0.0000001) {
        asLogError(wxString::Format(_("The desired startDate (%gh) is not coherent with the data properties."),
                                    timeArray.GetFirstDayHour()));
        return false;
    }

    return true;
}

VectorString asDataPredictorArchive::GetListOfFiles(asTimeArray &timeArray) const
{
    return VectorString();
}

bool asDataPredictorArchive::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                            asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return false;
}

double asDataPredictorArchive::ConvertToMjd(double timeValue) const
{
    return NaNDouble;
}