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

#include "asDataPredictorRealtime.h"
#include "asFileGrib2.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <asFileNetcdf.h>
#include <asDataPredictorRealtimeGfsForecast.h>
#include <asInternet.h>


asDataPredictorRealtime::asDataPredictorRealtime(const wxString &dataId)
        : asDataPredictor(dataId)
{
    m_runDateInUse = 0.0;
    m_restrictDownloads = false;
    m_restrictTimeHours = 0.0;
    m_restrictTimeStepHours = 0.0;
    m_forecastLeadTimeStart = 0.0;
    m_forecastLeadTimeEnd = 0.0;
    m_forecastLeadTimeStep = 0.0;
    m_runHourStart = 0.0;
    m_runUpdate = 0.0;
}

asDataPredictorRealtime::~asDataPredictorRealtime()
{

}

asDataPredictorRealtime *asDataPredictorRealtime::GetInstance(const wxString &datasetId, const wxString &dataId)
{
    asDataPredictorRealtime *predictor = NULL;

    if (datasetId.IsSameAs("NWS_GFS_Forecast", false)) {
        predictor = new asDataPredictorRealtimeGfsForecast(dataId);
    } else {
        asLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return NULL;
    }

    if (!predictor->Init()) {
        asLogError(_("The predictor did not initialize correctly."));
    }

    return predictor;
}

bool asDataPredictorRealtime::Init()
{
    return false;
}

int asDataPredictorRealtime::Download()
{
    wxASSERT(!m_predictorsRealtimeDirectory.IsEmpty());

    // Internet (cURL)
    asInternet internet;

    return internet.Download(GetUrls(), GetFileNames(), m_predictorsRealtimeDirectory);
}

bool asDataPredictorRealtime::CheckTimeArray(asTimeArray &timeArray) const
{
    return true;
}

double asDataPredictorRealtime::UpdateRunDateInUse()
{
    m_fileNames.clear();
    m_urls.clear();

    // TODO (Pascal#1#): Fix the use of m_timeZoneHours

    // Round time to the last available data
    double runHourStart = (double) m_runHourStart;
    double runUpdate = (double) m_runUpdate;
    double hourNow = (m_runDateInUse - floor(m_runDateInUse)) * 24;
    if (runUpdate > 0) {
        double factorUpdate = floor((hourNow - runHourStart) / runUpdate);
        m_runDateInUse = floor(m_runDateInUse) + (factorUpdate * runUpdate) / (double) 24;
    } else {
        m_runDateInUse = floor(m_runDateInUse) + runHourStart / (double) 24;
    }

    return m_runDateInUse;
}

double asDataPredictorRealtime::SetRunDateInUse(double val)
{
    // Get date and time
    if (val == 0) {
        val = asTime::NowMJD(asUTM);
    }

    m_runDateInUse = val;
    UpdateRunDateInUse();

    return m_runDateInUse;
}

double asDataPredictorRealtime::DecrementRunDateInUse()
{
    m_fileNames.clear();
    m_urls.clear();
    m_runDateInUse -= (double) m_runUpdate / (double) 24;

    return m_runDateInUse;
}

void asDataPredictorRealtime::RestrictTimeArray(double restrictTimeHours, double restrictTimeStepHours)
{
    m_restrictDownloads = true;
    m_restrictTimeHours = restrictTimeHours;
    m_restrictTimeStepHours = restrictTimeStepHours;
    wxASSERT(m_restrictTimeStepHours > 0);
    wxASSERT(m_restrictTimeHours > -100);
    wxASSERT(m_restrictTimeHours < 100);
}

bool asDataPredictorRealtime::BuildFilenamesUrls()
{
    m_dataDates.clear();
    m_fileNames.clear();
    m_urls.clear();

    wxString thisCommand = m_commandDownload;

    // Replace time in the command
    while (thisCommand.Find("CURRENTDATE") != wxNOT_FOUND) {
        int posStart = thisCommand.Find("CURRENTDATE");
        posStart--;
        thisCommand.Remove(posStart, 13); // Removes '[CURRENTDATE-'
        // Find end
        int posEnd = thisCommand.find("]", posStart);

        if (posEnd != wxNOT_FOUND && posEnd > posStart) {
            thisCommand.Remove(posEnd, 1); // Removes ']'
            wxString dateFormat = thisCommand.SubString(posStart, posEnd);
            wxString date = asTime::GetStringTime(m_runDateInUse, dateFormat);
            thisCommand.replace(posStart, date.Length(), date);
        }
    }

    // Restrict the downloads to used data
    if (m_restrictDownloads) {
        // Get the real lead time
        double dayRun = floor(m_runDateInUse);
        double desiredTime = dayRun + m_restrictTimeHours / 24.0;
        double diff = desiredTime - m_runDateInUse;
        m_forecastLeadTimeStart = (int) (diff * 24.0);
        m_forecastLeadTimeStep = m_restrictTimeStepHours;
        m_forecastLeadTimeEnd = floor(((double) m_forecastLeadTimeEnd - (double) m_forecastLeadTimeStart) /
                                      (double) m_forecastLeadTimeStep) * (double) m_forecastLeadTimeStep +
                                m_forecastLeadTimeStart;
    }

    wxASSERT(m_forecastLeadTimeStep > 0);
    wxASSERT(m_forecastLeadTimeEnd >= m_forecastLeadTimeStart);

    // Change the leadtimes
    for (int leadtime = m_forecastLeadTimeStart;
         leadtime <= m_forecastLeadTimeEnd; leadtime += m_forecastLeadTimeStep) {
        int currentLeadtime = leadtime;
        double runDateInUse = m_runDateInUse;

        // Manage if ledtime if negative -> get previous download
        while (currentLeadtime < 0) {
            currentLeadtime += m_runUpdate;
            runDateInUse -= (double) m_runUpdate / 24.0;
        }

        wxString thisCommandLeadTime = thisCommand;

        wxString timeStr = wxString::Format("%d", currentLeadtime);
        wxString timeStrFileName = wxEmptyString;

        thisCommandLeadTime.Replace("[LEADTIME-H]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-h]", timeStr);
        if (timeStr.Length() < 2)
            timeStr = "0" + timeStr;
        thisCommandLeadTime.Replace("[LEADTIME-HH]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-hh]", timeStr);
        if (timeStr.Length() < 3)
            timeStr = "0" + timeStr;
        timeStrFileName = timeStr;
        thisCommandLeadTime.Replace("[LEADTIME-HHH]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-hhh]", timeStr);
        if (timeStr.Length() < 4)
            timeStr = "0" + timeStr;
        thisCommandLeadTime.Replace("[LEADTIME-HHHH]", timeStr);
        thisCommandLeadTime.Replace("[LEADTIME-hhhh]", timeStr);

        // Filename
        wxString dirstructure = "YYYY";
        dirstructure.Append(DS);
        dirstructure.Append("MM");
        dirstructure.Append(DS);
        dirstructure.Append("DD");
        wxString directory = asTime::GetStringTime(runDateInUse, dirstructure);
        wxString nowstr = asTime::GetStringTime(runDateInUse, "YYYYMMDDhh");
        wxString leadtimestr = timeStrFileName;
        wxString ext = m_fileExtension;

        wxString filename = wxString::Format("%s.%s.%s.%s.%s", nowstr, m_datasetId, m_dataId, leadtimestr, ext);
        wxString filenameres = directory + DS + filename;

        double dataDate = runDateInUse + currentLeadtime / 24.0;

        // Save resulting strings
        m_urls.push_back(thisCommandLeadTime);
        m_fileNames.push_back(filenameres);
        m_dataDates.push_back(dataDate);
    }

    wxASSERT(m_dataDates.size() == m_urls.size());
    wxASSERT(m_dataDates.size() == m_fileNames.size());

    return true;
}

bool asDataPredictorRealtime::ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                              VVArray2DFloat &compositeData)
{
    VectorString filesList = GetListOfFiles(timeArray);

    if(!CheckFilesPresence(filesList)) {
        return false;
    }

    for (int i = 0; i < filesList.size(); i++) {
        wxString fileName = filesList[i];

        if (!ExtractFromFile(fileName, dataArea, timeArray, compositeData)) {
            return false;
        }
    }

    return true;
}

VectorString asDataPredictorRealtime::GetListOfFiles(asTimeArray &timeArray) const
{
    VectorString filesList;

    for (int i_file = 0; i_file < m_fileNames.size(); i_file++) {
        wxString filePath = wxEmptyString;

        // Check if the volume is present
        wxFileName fileName(m_fileNames[i_file]);
        if (!fileName.HasVolume() && !m_predictorsRealtimeDirectory.IsEmpty()) {
            filePath = m_predictorsRealtimeDirectory;
            filePath.Append(DS);
        }
        filePath.Append(m_fileNames[i_file]);

        filesList.push_back(filePath);
    }

    return filesList;
}

bool asDataPredictorRealtime::ExtractFromGribFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                  asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Open the NetCDF file
    asFileGrib2 gbFile(fileName, asFileGrib2::ReadOnly);
    if (!gbFile.Open()) {
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(gbFile, dataArea, timeArray, compositeData)) {
        gbFile.Close();
        wxFAIL;
        return false;
    }

    // Adjust axes if necessary
    dataArea = AdjustAxes(dataArea, compositeData);
    if (dataArea) {
        wxASSERT(dataArea->GetNbComposites() > 0);
    }

    // Get indexes
    if (!GetAxesIndexes(gbFile, dataArea, timeArray, compositeData)) {
        gbFile.Close();
        wxFAIL;
        return false;
    }

    // Load data
    if (!GetDataFromFile(gbFile, compositeData)) {
        gbFile.Close();
        wxFAIL;
        return false;
    }

    // Close the nc file
    gbFile.Close();

    return true;
}

bool asDataPredictorRealtime::ParseFileStructure(asFileGrib2 &gbFile, asGeoAreaCompositeGrid *&dataArea,
                                                 asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Get full axes from the netcdf file
    gbFile.GetXaxis(m_fileStructure.axisLon);
    gbFile.GetYaxis(m_fileStructure.axisLat);

    if (m_fileStructure.hasLevelDimension) {
        asLogError(_("The level dimension is not yet implemented for realtime predictors."));
        return false;
    }

    return true;
}

int *asDataPredictorRealtime::GetIndexesStartGrib(int i_area) const
{
    if (m_fileStructure.hasLevelDimension) {
        static int array[3] = {0, 0, 0};
        array[0] = m_fileIndexes.level;
        array[1] = m_fileIndexes.areas[i_area].lonStart;
        array[2] = m_fileIndexes.areas[i_area].latStart;

        return array;
    } else {
        static int array[2] = {0, 0};
        array[0] = m_fileIndexes.areas[i_area].lonStart;
        array[1] = m_fileIndexes.areas[i_area].latStart;

        return array;
    }

    return NULL;
}

int *asDataPredictorRealtime::GetIndexesCountGrib(int i_area) const
{
    if (m_fileStructure.hasLevelDimension) {
        static int array[3] = {0, 0, 0};
        array[0] = 1;
        array[1] = m_fileIndexes.areas[i_area].lonCount;
        array[2] = m_fileIndexes.areas[i_area].latCount;

        return array;
    } else {
        static int array[2] = {0, 0};
        array[0] = m_fileIndexes.areas[i_area].lonCount;
        array[1] = m_fileIndexes.areas[i_area].latCount;

        return array;
    }

    return NULL;
}

bool asDataPredictorRealtime::GetAxesIndexes(asFileGrib2 &gbFile, asGeoAreaCompositeGrid *&dataArea,
                                             asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    m_fileIndexes.areas.clear();

    // Get the time length
    m_fileIndexes.timeArrayCount = 1;
    m_fileIndexes.timeCount = 1;

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

        if (m_fileStructure.hasLevelDimension) {
            m_fileIndexes.level = asTools::SortedArraySearch(&m_fileStructure.axisLevel[0],
                                                             &m_fileStructure.axisLevel[m_fileStructure.axisLevel.size() - 1],
                                                             m_level, 0.01f);
            if (m_fileIndexes.level < 0) {
                asLogWarning(wxString::Format(_("The desired level (%g) does not exist for %s"), m_level,
                                              m_fileVariableName));
                return false;
            }
        }
    }

    return true;
}


bool asDataPredictorRealtime::GetDataFromFile(asFileGrib2 &gbFile, VVArray2DFloat &compositeData)
{
    // Check if scaling is needed
    bool scalingNeeded = true;
    float dataAddOffset = gbFile.GetOffset();
    if (asTools::IsNaN(dataAddOffset))
        dataAddOffset = 0;
    float dataScaleFactor = gbFile.GetScale();
    if (asTools::IsNaN(dataScaleFactor))
        dataScaleFactor = 1;
    if (dataAddOffset == 0 && dataScaleFactor == 1)
        scalingNeeded = false;

    VVectorFloat vectData;

    for (int i_area = 0; i_area < compositeData.size(); i_area++) {

        // Create the arrays to receive the data
        VectorFloat dataF;

        // Resize the arrays to store the new data
        int totLength = m_fileIndexes.timeArrayCount * m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
        wxASSERT(totLength > 0);
        dataF.resize(totLength);

        // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
        gbFile.GetVarArray(m_fileVariableName, GetIndexesStartGrib(i_area), GetIndexesCountGrib(i_area), m_level,
                           &dataF[0]);

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
        Array2DFloat latlonData = Array2DFloat(m_fileIndexes.areas[i_area].latCount,
                                               m_fileIndexes.areas[i_area].lonCount);

        for (int i_lat = 0; i_lat < m_fileIndexes.areas[i_area].latCount; i_lat++) {
            for (int i_lon = 0; i_lon < m_fileIndexes.areas[i_area].lonCount; i_lon++) {
                ind = i_lon + i_lat * m_fileIndexes.areas[i_area].lonCount;

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

        data.clear();
    }

    return true;
}

bool asDataPredictorRealtime::ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                             asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    return false;
}

double asDataPredictorRealtime::ConvertToMjd(double timeValue) const
{
    return NaNDouble;
}