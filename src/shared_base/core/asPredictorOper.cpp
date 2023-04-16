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

#include "asAreaGrid.h"
#include "asInternet.h"
#include "asPredictorOperCustomFvg.h"
#include "asPredictorOperCustomVigicruesIfs.h"
#include "asPredictorOperEcmwfIfs.h"
#include "asPredictorOperMfArpege.h"
#include "asPredictorOperNwsGfs.h"
#include "asPredictorOperNwsGfsLocal.h"
#include "asTimeArray.h"

asPredictorOper::asPredictorOper(const wxString& dataId)
    : asPredictor(dataId),
      m_leadTimeStart(0),
      m_leadTimeStep(0),
      m_runHourStart(0),
      m_runUpdate(0),
      m_runDateInUse(0.0),
      m_commandDownload(),
      m_shouldDownload(false) {}

void asPredictorOper::SetDefaultPredictorsUrls() {
    wxConfigBase* pConfig = wxFileConfig::Get();

    wxString url;

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_"
        "500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_HGT=on&subregion=&"
        "leftlon=-32&rightlon=42&toplat=72&bottomlat=20&dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/hgt", pConfig->Read("/PredictorsUrl/GFS/hgt", url));

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_"
        "500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_TMP=on&subregion=&"
        "leftlon=-32&rightlon=42&toplat=72&bottomlat=20&dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/temp", pConfig->Read("/PredictorsUrl/GFS/temp", url));

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_"
        "500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_VVEL=on&subregion=&"
        "leftlon=-32&rightlon=42&toplat=72&bottomlat=20&dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/vvel", pConfig->Read("/PredictorsUrl/GFS/vvel", url));

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_"
        "500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_RH=on&subregion=&leftlon="
        "-32&rightlon=42&toplat=72&bottomlat=20&dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/rh", pConfig->Read("/PredictorsUrl/GFS/rh", url));

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_"
        "500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_UGRD=on&subregion=&"
        "leftlon=-32&rightlon=42&toplat=72&bottomlat=20&dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/uwnd", pConfig->Read("/PredictorsUrl/GFS/uwnd", url));

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_300_mb=on&lev_400_mb=on&lev_"
        "500_mb=on&lev_600_mb=on&lev_700_mb=on&lev_850_mb=on&lev_925_mb=on&lev_1000_mb=on&var_VGRD=on&subregion=&"
        "leftlon=-32&rightlon=42&toplat=72&bottomlat=20&dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/vwnd", pConfig->Read("/PredictorsUrl/GFS/vwnd", url));

    url =
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p50.pl?file=gfs.t[CURRENTDATE-hh]z.pgrb2full.0p50.f[LEADTIME-hhh]&lev_entire_atmosphere_%5C%"
        "28considered_as_a_single_layer%5C%29=on&var_PWAT=on&subregion=&leftlon=-32&rightlon=42&toplat=72&bottomlat=20&"
        "dir=%2Fgfs.[CURRENTDATE-YYYYMMDD]%2F[CURRENTDATE-hh]%2Fatmos";
    pConfig->Write("/PredictorsUrl/GFS/pwat", pConfig->Read("/PredictorsUrl/GFS/pwat", url));

    pConfig->Flush();
}

asPredictorOper* asPredictorOper::GetInstance(const wxString& datasetId, const wxString& dataId) {
    asPredictorOper* predictor = nullptr;

    if (datasetId.IsSameAs("NWS_GFS", false)) {
        predictor = new asPredictorOperNwsGfs(dataId);
    } else if (datasetId.IsSameAs("NWS_GFS_Local", false)) {
        predictor = new asPredictorOperNwsGfsLocal(dataId);
    } else if (datasetId.IsSameAs("ECMWF_IFS", false)) {
        predictor = new asPredictorOperEcmwfIfs(dataId);
    } else if (datasetId.IsSameAs("MF_ARPEGE", false)) {
        predictor = new asPredictorOperMfArpege(dataId);
    } else if (datasetId.IsSameAs("Custom_MeteoFVG", false)) {
        predictor = new asPredictorOperCustomFvg(dataId);
    } else if (datasetId.IsSameAs("Custom_Vigicrues_IFS", false)) {
        predictor = new asPredictorOperCustomVigicruesIfs(dataId);
    } else {
        wxLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return nullptr;
    }

    if (!predictor->Init()) {
        wxLogError(_("The predictor did not initialize correctly."));
    }

    return predictor;
}

int asPredictorOper::Download() {
    wxASSERT(!m_predictorsRealtimeDir.IsEmpty());

    return asInternet::Download(GetUrls(), GetFileNames(), m_predictorsRealtimeDir);
}

bool asPredictorOper::CheckTimeArray(asTimeArray& timeArray) {
    return true;
}

double asPredictorOper::UpdateRunDateInUse() {
    m_fileNames.clear();
    m_urls.clear();

    // Round time to the last available data
    double runHourStart = m_runHourStart;
    double runUpdate = m_runUpdate;
    double hourNow = (m_runDateInUse - floor(m_runDateInUse)) * 24;
    if (runUpdate > 0) {
        double factorUpdate = floor((hourNow - runHourStart) / runUpdate);
        m_runDateInUse = floor(m_runDateInUse) + (factorUpdate * runUpdate) / (double)24;
    } else {
        m_runDateInUse = floor(m_runDateInUse) + runHourStart / (double)24;
    }

    return m_runDateInUse;
}

double asPredictorOper::SetRunDateInUse(double val) {
    // Get date and time
    if (val == 0) {
        val = asTime::NowMJD(asUTM);
    }

    m_runDateInUse = val;
    UpdateRunDateInUse();

    return m_runDateInUse;
}

double asPredictorOper::DecrementRunDateInUse() {
    m_fileNames.clear();
    m_urls.clear();
    m_runDateInUse -= m_runUpdate / (double)24;

    return m_runDateInUse;
}

bool asPredictorOper::BuildFilenamesAndUrls(double predictorHour, double forecastTimeStepHours, int leadTimeNb) {
    m_dataDates.clear();
    m_fileNames.clear();
    m_urls.clear();

    // Restrict to used data
    if (forecastTimeStepHours >= 24) {
        // Get the real lead time
        double dayRun = floor(m_runDateInUse);
        double desiredTime = dayRun + predictorHour / 24.0;
        double diff = desiredTime - m_runDateInUse;
        m_leadTimeStart = (int)(diff * 24.0);
        m_leadTimeStep = forecastTimeStepHours;
    } else {
        m_leadTimeStart = (int)predictorHour;
    }

    wxASSERT(m_leadTimeStep > 0);

    // Change the lead times
    for (int iLeadTime = 0; iLeadTime < leadTimeNb; iLeadTime++) {
        int currentLeadtime = m_leadTimeStart + iLeadTime * m_leadTimeStep;
        double runDateInUse = m_runDateInUse;

        // Manage if lead time if negative -> get previous download
        while (currentLeadtime < 0) {
            currentLeadtime += m_runUpdate;
            runDateInUse -= m_runUpdate / 24.0;
        }

        wxString thisCommand = m_commandDownload;

        // Replace time in the command
        while (thisCommand.Find("CURRENTDATE") != wxNOT_FOUND) {
            int posStart = thisCommand.Find("CURRENTDATE");
            if (posStart == wxNOT_FOUND) {
                break;
            }
            posStart--;
            auto posStartSt = (size_t)posStart;
            thisCommand.Remove(posStartSt, 13);  // Removes '[CURRENTDATE-'
            // Find end
            int posEnd = thisCommand.find("]", posStartSt);

            if (posEnd != wxNOT_FOUND && posEnd > posStartSt) {
                auto posEndSt = (size_t)posEnd;
                thisCommand.Remove(posEndSt, 1);  // Removes ']'
                wxString dateFormat = thisCommand.SubString(posStartSt, posEndSt);
                wxString date = asTime::GetStringTime(runDateInUse, dateFormat);
                thisCommand.replace(posStartSt, date.Length(), date);
            }
        }

        wxString timeStr = asStrF("%d", currentLeadtime);

        thisCommand.Replace("[LEADTIME-H]", timeStr);
        thisCommand.Replace("[LEADTIME-h]", timeStr);
        if (timeStr.Length() < 2) timeStr = "0" + timeStr;
        thisCommand.Replace("[LEADTIME-HH]", timeStr);
        thisCommand.Replace("[LEADTIME-hh]", timeStr);
        if (timeStr.Length() < 3) timeStr = "0" + timeStr;
        thisCommand.Replace("[LEADTIME-HHH]", timeStr);
        thisCommand.Replace("[LEADTIME-hhh]", timeStr);
        if (timeStr.Length() < 4) timeStr = "0" + timeStr;
        thisCommand.Replace("[LEADTIME-HHHH]", timeStr);
        thisCommand.Replace("[LEADTIME-hhhh]", timeStr);

        // Filename
        wxString filePath = GetDirStructure(runDateInUse) + DS + GetFileName(runDateInUse, currentLeadtime);

        double dataDate = runDateInUse + currentLeadtime / 24.0;

        // Save resulting strings
        m_urls.push_back(thisCommand);
        m_fileNames.push_back(filePath);
        m_dataDates.push_back(dataDate);
    }

    wxASSERT(m_dataDates.size() == m_urls.size());
    wxASSERT(m_dataDates.size() == m_fileNames.size());

    return true;
}

void asPredictorOper::ListFiles(asTimeArray& timeArray) {
    for (const auto& currfileName : m_fileNames) {
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

bool asPredictorOper::ExtractFromFiles(asAreaGrid*& dataArea, asTimeArray& timeArray) {
    wxASSERT(m_files.size() == timeArray.GetSize());

    int idx = 0;
    while (true) {
        wxString fileName;
        vd dates;
        for (int i = idx; i < m_files.size(); i++) {
            if (!fileName.IsEmpty() && !fileName.IsSameAs(m_files[i])) {
                break;
            }
            fileName = m_files[i];
            dates.push_back(timeArray[i]);
            idx++;
        }
        asTimeArray newTimeArray(dates);
        newTimeArray.Init();

        switch (m_fileType) {
            case (asFile::Netcdf): {
                if (!ExtractFromNetcdfFile(fileName, dataArea, newTimeArray)) {
                    return false;
                }
                break;
            }
            case (asFile::Grib): {
                if (!ExtractFromGribFile(fileName, dataArea, newTimeArray)) {
                    return false;
                }
                break;
            }
            default: {
                wxLogError(_("Predictor file type not correctly defined."));
                return false;
            }
        }

        if (idx >= m_files.size() - 1) {
            return true;
        }
    }

    return true;
}

wxString asPredictorOper::GetDirStructure(const double date) {
    wxString dirStructure = "YYYY";
    dirStructure.Append(DS);
    dirStructure.Append("MM");
    dirStructure.Append(DS);
    dirStructure.Append("DD");

    return asTime::GetStringTime(date, dirStructure);
}

wxString asPredictorOper::GetFileName(const double date, const int leadTime) {
    wxString timeStr = asStrF("%d", leadTime);
    if (timeStr.Length() < 2) timeStr = "0" + timeStr;
    if (timeStr.Length() < 3) timeStr = "0" + timeStr;

    wxString dateStr = asTime::GetStringTime(date, "YYYYMMDDhh");

    return asStrF("%s.%s.%s.%s.%s", dateStr, m_datasetId, m_dataId, timeStr, m_fileExtension);
}