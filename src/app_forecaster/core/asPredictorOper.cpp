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

#include "asPredictorOper.h"

#include <asTimeArray.h>
#include <asAreaCompGrid.h>
#include "asPredictorOperGfsForecast.h"
#include <asInternet.h>


asPredictorOper::asPredictorOper(const wxString &dataId)
        : asPredictor(dataId),
          m_leadTimeStart(0.0),
          m_leadTimeEnd(0.0),
          m_leadTimeStep(0.0),
          m_runHourStart(0.0),
          m_runUpdate(0.0),
          m_runDateInUse(0.0),
          m_commandDownload(),
          m_restrictDownloads(false),
          m_restrictTimeHours(0.0),
          m_restrictTimeStepHours(0.0)
{

}

void asPredictorOper::SetDefaultPredictorsUrls()
{
    wxConfigBase *pConfig = wxFileConfig::Get();

    wxString url;

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_HGT=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/hgt", url);
    pConfig->Write("/PredictorsUrl/GFS/hgt", url);

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_TMP=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/temp", url);
    pConfig->Write("/PredictorsUrl/GFS/temp", url);

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_VVEL=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/vvel", url);
    pConfig->Write("/PredictorsUrl/GFS/vvel", url);

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_RH=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/rh", url);
    pConfig->Write("/PredictorsUrl/GFS/rh", url);

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_UGRD=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/uwnd", url);
    pConfig->Write("/PredictorsUrl/GFS/uwnd", url);

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_VGRD=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/vwnd", url);
    pConfig->Write("/PredictorsUrl/GFS/vwnd", url);

    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_entire_atmosphere_%5C%28considered_as_a_single_layer%5C%29=on&var_PWAT=on&subregion=&leftlon=-20&rightlon=30&toplat=70&bottomlat=30&dir=%2Fgfs.[CURRENTDATE-YYYYMMDDhh]";
    url = pConfig->Read("/PredictorsUrl/GFS/pwat", url);
    pConfig->Write("/PredictorsUrl/GFS/pwat", url);

    pConfig->Flush();

}

asPredictorOper *asPredictorOper::GetInstance(const wxString &datasetId, const wxString &dataId)
{
    asPredictorOper *predictor = nullptr;

    if (datasetId.IsSameAs("NWS_GFS_Forecast", false)) {
        predictor = new asPredictorOperGfsForecast(dataId);
    } else {
        wxLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return nullptr;
    }

    if (!predictor->Init()) {
        wxLogError(_("The predictor did not initialize correctly."));
    }

    return predictor;
}

bool asPredictorOper::Init()
{
    return false;
}

int asPredictorOper::Download()
{
    wxASSERT(!m_predictorsRealtimeDir.IsEmpty());

    // Internet (cURL)
    asInternet internet;

    return internet.Download(GetUrls(), GetFileNames(), m_predictorsRealtimeDir);
}

bool asPredictorOper::CheckTimeArray(asTimeArray &timeArray) const
{
    return true;
}

double asPredictorOper::UpdateRunDateInUse()
{
    m_fileNames.clear();
    m_urls.clear();

    // Round time to the last available data
    double runHourStart = m_runHourStart;
    double runUpdate = m_runUpdate;
    double hourNow = (m_runDateInUse - floor(m_runDateInUse)) * 24;
    if (runUpdate > 0) {
        double factorUpdate = floor((hourNow - runHourStart) / runUpdate);
        m_runDateInUse = floor(m_runDateInUse) + (factorUpdate * runUpdate) / (double) 24;
    } else {
        m_runDateInUse = floor(m_runDateInUse) + runHourStart / (double) 24;
    }

    return m_runDateInUse;
}

double asPredictorOper::SetRunDateInUse(double val)
{
    // Get date and time
    if (val == 0) {
        val = asTime::NowMJD(asUTM);
    }

    m_runDateInUse = val;
    UpdateRunDateInUse();

    return m_runDateInUse;
}

double asPredictorOper::DecrementRunDateInUse()
{
    m_fileNames.clear();
    m_urls.clear();
    m_runDateInUse -= m_runUpdate / (double) 24;

    return m_runDateInUse;
}

void asPredictorOper::RestrictTimeArray(double restrictTimeHours, double restrictTimeStepHours, int leadTimeNb)
{
    m_restrictDownloads = true;
    m_restrictTimeHours = restrictTimeHours;
    m_restrictTimeStepHours = restrictTimeStepHours;
    m_leadTimeEnd = 24 * (leadTimeNb + floor(restrictTimeHours / restrictTimeStepHours));
    wxASSERT(m_restrictTimeStepHours > 0);
    wxASSERT(m_restrictTimeHours > -100);
    wxASSERT(m_restrictTimeHours < 100);
}

bool asPredictorOper::BuildFilenamesUrls()
{
    m_dataDates.clear();
    m_fileNames.clear();
    m_urls.clear();

    wxString thisCommand = m_commandDownload;

    // Replace time in the command
    while (thisCommand.Find("CURRENTDATE") != wxNOT_FOUND) {
        int posStart = thisCommand.Find("CURRENTDATE");
        if (posStart == wxNOT_FOUND) {
            break;
        }
        posStart--;
        auto posStartSt = (size_t) posStart;
        thisCommand.Remove(posStartSt, 13); // Removes '[CURRENTDATE-'
        // Find end
        int posEnd = thisCommand.find("]", posStartSt);

        if (posEnd != wxNOT_FOUND && posEnd > posStartSt) {
            auto posEndSt = (size_t) posEnd;
            thisCommand.Remove(posEndSt, 1); // Removes ']'
            wxString dateFormat = thisCommand.SubString(posStartSt, posEndSt);
            wxString date = asTime::GetStringTime(m_runDateInUse, dateFormat);
            thisCommand.replace(posStartSt, date.Length(), date);
        }
    }

    // Restrict the downloads to used data
    if (m_restrictDownloads) {
        // Get the real lead time
        double dayRun = floor(m_runDateInUse);
        double desiredTime = dayRun + m_restrictTimeHours / 24.0;
        double diff = desiredTime - m_runDateInUse;
        m_leadTimeStart = (int) (diff * 24.0);
        m_leadTimeStep = m_restrictTimeStepHours;
        m_leadTimeEnd = floor((m_leadTimeEnd - m_leadTimeStart) / m_leadTimeStep) *
                                m_leadTimeStep + m_leadTimeStart;
    }

    wxASSERT(m_leadTimeStep > 0);
    wxASSERT(m_leadTimeEnd >= m_leadTimeStart);

    // Change the leadtimes
    for (int leadtime = m_leadTimeStart;
         leadtime <= m_leadTimeEnd; leadtime += m_leadTimeStep) {
        int currentLeadtime = leadtime;
        double runDateInUse = m_runDateInUse;

        // Manage if ledtime if negative -> get previous download
        while (currentLeadtime < 0) {
            currentLeadtime += m_runUpdate;
            runDateInUse -= m_runUpdate / 24.0;
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

void asPredictorOper::ListFiles(asTimeArray &timeArray)
{
    for (const auto &currfileName : m_fileNames) {
        wxString filePath = wxEmptyString;

        // Check if the volume is present
        wxFileName fileName(currfileName);
        if (!fileName.HasVolume() && !m_predictorsRealtimeDir.IsEmpty()) {
            filePath = m_predictorsRealtimeDir;
            filePath.Append(DS);
        }
        filePath.Append(currfileName);

        m_files.push_back(filePath);
    }
}

bool asPredictorOper::GetAxesIndexes(asAreaCompGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData)
{
    m_fInd.areas.clear();

    // Get the time length
    m_fInd.timeArrayCount = 1;
    m_fInd.timeCount = 1;

    // Go through every area
    m_fInd.areas.resize(compositeData.size());
    for (int iArea = 0; iArea < compositeData.size(); iArea++) {

        if (dataArea) {
            // Get the spatial extent
            auto lonMin = (float) dataArea->GetXaxisCompositeStart(iArea);
            auto latMinStart = (float) dataArea->GetYaxisCompositeStart(iArea);
            auto latMinEnd = (float) dataArea->GetYaxisCompositeEnd(iArea);

            // The dimensions lengths
            m_fInd.areas[iArea].lonCount = dataArea->GetXaxisCompositePtsnb(iArea);
            m_fInd.areas[iArea].latCount = dataArea->GetYaxisCompositePtsnb(iArea);

            // Get the spatial indices of the desired data
            m_fInd.areas[iArea].lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1], lonMin, 0.01f,
                                                  asHIDE_WARNINGS);
            if (m_fInd.areas[iArea].lonStart == asOUT_OF_RANGE) {
                // If not found, try with negative angles
                m_fInd.areas[iArea].lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1],
                                                      lonMin - 360, 0.01f, asHIDE_WARNINGS);
            }
            if (m_fInd.areas[iArea].lonStart == asOUT_OF_RANGE) {
                // If not found, try with angles above 360 degrees
                m_fInd.areas[iArea].lonStart = asFind(&m_fStr.lons[0], &m_fStr.lons[m_fStr.lons.size() - 1],
                                                      lonMin + 360, 0.01f, asHIDE_WARNINGS);
            }
            if (m_fInd.areas[iArea].lonStart < 0) {
                wxLogError("Cannot find lonMin (%f) in the array axisDataLon ([0]=%f -> [%d]=%f) ", lonMin,
                           m_fStr.lons[0], (int) m_fStr.lons.size(), m_fStr.lons[m_fStr.lons.size() - 1]);
                return false;
            }
            wxASSERT_MSG(m_fInd.areas[iArea].lonStart >= 0,
                         wxString::Format("axisDataLon[0] = %f, &axisDataLon[%d] = %f & lonMin = %f", m_fStr.lons[0],
                                          (int) m_fStr.lons.size(), m_fStr.lons[m_fStr.lons.size() - 1], lonMin));

            int indexStartLat1 = asFind(&m_fStr.lats[0], &m_fStr.lats[m_fStr.lats.size() - 1], latMinStart, 0.01f);
            int indexStartLat2 = asFind(&m_fStr.lats[0], &m_fStr.lats[m_fStr.lats.size() - 1], latMinEnd, 0.01f);
            wxASSERT_MSG(indexStartLat1 >= 0,
                         wxString::Format("Looking for %g in %g to %g", latMinStart, m_fStr.lats[0],
                                          m_fStr.lats[m_fStr.lats.size() - 1]));
            wxASSERT_MSG(indexStartLat2 >= 0, wxString::Format("Looking for %g in %g to %g", latMinEnd, m_fStr.lats[0],
                                                               m_fStr.lats[m_fStr.lats.size() - 1]));
            m_fInd.areas[iArea].latStart = wxMin(indexStartLat1, indexStartLat2);
        } else {
            m_fInd.areas[iArea].lonStart = 0;
            m_fInd.areas[iArea].latStart = 0;
            m_fInd.areas[iArea].lonCount = m_lonPtsnb;
            m_fInd.areas[iArea].latCount = m_latPtsnb;
        }

        if (m_fStr.hasLevelDim && !m_fStr.singleLevel) {
            m_fInd.level = asFind(&m_fStr.levels[0], &m_fStr.levels[m_fStr.levels.size() - 1], m_level, 0.01f);
            if (m_fInd.level < 0) {
                wxLogWarning(_("The desired level (%g) does not exist for %s"), m_level, m_fileVarName);
                return false;
            }
        } else if (m_fStr.hasLevelDim && m_fStr.singleLevel) {
            m_fInd.level = 0;
        }
    }

    return true;
}

double asPredictorOper::ConvertToMjd(double timeValue, double refValue) const
{
    return NaNd;
}
