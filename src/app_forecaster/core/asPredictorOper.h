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

#ifndef ASPREDICTOROPER_H
#define ASPREDICTOROPER_H

#include <asIncludes.h>
#include <asPredictor.h>

class asPredictorOper
        : public asPredictor
{
public:
    asPredictorOper(const wxString &dataId);

    virtual ~asPredictorOper();

    static void SetDefaultPredictorsUrls();

    static asPredictorOper *GetInstance(const wxString &datasetId, const wxString &dataId);

    virtual bool Init();

    int Download();

    double UpdateRunDateInUse();

    double SetRunDateInUse(double val = 0);

    double DecrementRunDateInUse();

    bool BuildFilenamesUrls();

    void RestrictTimeArray(double restrictTimeHours, double restrictTimeStepHours, int leadTimeNb);

    void SetRestrictDownloads(bool val)
    {
        m_restrictDownloads = val;
    }

    double GetForecastLeadTimeStart() const
    {
        return m_leadTimeStart;
    }

    void SetForecastLeadTimeStart(int val)
    {
        m_leadTimeStart = val;
    }

    double GetForecastLeadTimeEnd() const
    {
        return m_leadTimeEnd;
    }

    void SetForecastLeadTimeEnd(int val)
    {
        m_leadTimeEnd = val;
    }

    double GetForecastLeadTimeStep() const
    {
        return m_leadTimeStep;
    }

    void SetForecastLeadTimeStep(int val)
    {
        m_leadTimeStep = val;
    }

    double GetRunHourStart() const
    {
        return m_runHourStart;
    }

    void SetRunHourStart(int val)
    {
        m_runHourStart = val;
    }

    double GetRunUpdate() const
    {
        return m_runUpdate;
    }

    void SetRunUpdate(double val)
    {
        m_runUpdate = val;
    }

    double GetRunDateInUse() const
    {
        return m_runDateInUse;
    }

    wxString GetCommandDownload() const
    {
        return m_commandDownload;
    }

    void SetCommandDownload(const wxString &val)
    {
        m_commandDownload = val;
    }

    int GetUlrsNb() const
    {
        int urlsNb = (int) m_urls.size();
        return urlsNb;
    }

    vwxs GetUrls() const
    {
        return m_urls;
    }

    wxString GetUrl(int i) const
    {
        wxASSERT(m_fileNames.size() == m_urls.size());
        if ((unsigned) i >= m_urls.size())
            return wxEmptyString;
        return m_urls[i];
    }

    void SetUrls(const vwxs &val)
    {
        m_urls = val;
    }

    vwxs GetFileNames() const
    {
        return m_fileNames;
    }

    void SetFileNames(const vwxs &val)
    {
        m_fileNames = val;
    }

    wxString GetFileName(int i) const
    {
        wxASSERT(m_fileNames.size() == m_urls.size());
        if ((unsigned) i >= m_fileNames.size())
            return wxEmptyString;
        return m_fileNames[i];
    }

    int GetDatesNb() const
    {
        wxASSERT(m_dataDates.size() == m_urls.size());
        wxASSERT(m_dataDates.size() == m_fileNames.size());
        int datesNb = m_dataDates.size();
        return datesNb;
    }

    vd GetDataDates() const
    {
        return m_dataDates;
    }

    double GetDataDate(int i) const
    {
        wxASSERT(m_dataDates.size() == m_urls.size());
        wxASSERT(m_dataDates.size() == m_fileNames.size());
        if ((unsigned) i >= m_dataDates.size())
            return asNOT_VALID;
        return m_dataDates[i];
    }

    double GetLastDataDate() const
    {
        wxASSERT(m_dataDates.size() == m_urls.size());
        wxASSERT(m_dataDates.size() == m_fileNames.size());
        return m_dataDates[m_dataDates.size() - 1];
    }

    wxString GetPredictorsRealtimeDirectory() const
    {
        return m_predictorsRealtimeDir;
    }

    void SetPredictorsRealtimeDirectory(const wxString &dir)
    {
        m_predictorsRealtimeDir = dir;
    }

protected:
    wxString m_predictorsRealtimeDir;
    double m_leadTimeStart;
    double m_leadTimeEnd;
    double m_leadTimeStep;
    double m_runHourStart;
    double m_runUpdate;
    double m_runDateInUse;
    wxString m_commandDownload;
    bool m_restrictDownloads;
    double m_restrictTimeHours;
    double m_restrictTimeStepHours;
    vwxs m_fileNames;
    vwxs m_urls;
    vd m_dataDates;

    virtual void ListFiles(asTimeArray &timeArray);

    int *GetIndexesStartGrib(int iArea) const;

    int *GetIndexesCountGrib(int iArea) const;

    virtual bool GetAxesIndexes(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData);

    virtual double ConvertToMjd(double timeValue, double refValue = NaNd) const;

    virtual bool CheckTimeArray(asTimeArray &timeArray) const;

};

#endif // ASPREDICTOROPER_H
