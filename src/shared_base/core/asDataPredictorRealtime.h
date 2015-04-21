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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#ifndef ASDATAPREDICTORREALTIME_H
#define ASDATAPREDICTORREALTIME_H

#include <asIncludes.h>
#include <asDataPredictor.h>

class asDataPredictorRealtime: public asDataPredictor
{
public:

    /** Default constructor
     * \param dataId The predictor data id
     */
    asDataPredictorRealtime(const wxString &dataId);

    /** Default destructor */
    virtual ~asDataPredictorRealtime();

    static asDataPredictorRealtime* GetInstance(const wxString &datasetId, const wxString &dataId);

    virtual bool Init();

    int Download();

    /** Update the RunDateInUse to the most recent hour or to the given date
     * \param now The desired date
     */
    double UpdateRunDateInUse();

    /** Set m_runDateInUse
     * \param val New value to set
     */
    double SetRunDateInUse(double val = 0);

    /** Decrement RunDateInUse to get older data */
    double DecrementRunDateInUse();

    /** Method to build the command (url) to download the file */
    bool BuildFilenamesUrls();

    /** Restrict the downloads to used data */
    void RestrictTimeArray(double restrictTimeHours, double restrictTimeStepHours);

    void SetRestrictDownloads(bool val)
    {
        m_restrictDownloads = val;
    }

    /** Access m_forecastLeadTimeStart
     * \return The current value of m_forecastLeadTimeStart
     */
    int GetForecastLeadTimeStart()
    {
        return m_forecastLeadTimeStart;
    }

    /** Set m_forecastLeadTimeStart
     * \param val New value to set
     */
    void SetForecastLeadTimeStart(int val)
    {
        m_forecastLeadTimeStart = val;
    }

    /** Access m_forecastLeadTimeEnd
     * \return The current value of m_forecastLeadTimeEnd
     */
    int GetForecastLeadTimeEnd()
    {
        return m_forecastLeadTimeEnd;
    }

    /** Set m_forecastLeadTimeEnd
     * \param val New value to set
     */
    void SetForecastLeadTimeEnd(int val)
    {
        m_forecastLeadTimeEnd = val;
    }

    /** Access m_forecastLeadTimeStep
     * \return The current value of m_forecastLeadTimeStep
     */
    int GetForecastLeadTimeStep()
    {
        return m_forecastLeadTimeStep;
    }

    /** Set m_forecastLeadTimeStep
     * \param val New value to set
     */
    void SetForecastLeadTimeStep(int val)
    {
        m_forecastLeadTimeStep = val;
    }

    /** Access m_runHourStart
     * \return The current value of m_runHourStart
     */
    int GetRunHourStart()
    {
        return m_runHourStart;
    }

    /** Set m_runHourStart
     * \param val New value to set
     */
    void SetRunHourStart(int val)
    {
        m_runHourStart = val;
    }

    /** Access m_runUpdate
     * \return The current value of m_runUpdate
     */
    int GetRunUpdate()
    {
        return m_runUpdate;
    }

    /** Set m_runUpdate
     * \param val New value to set
     */
    void SetRunUpdate(int val)
    {
        m_runUpdate = val;
    }

    /** Access m_runDateInUse
     * \return The current value of m_runDateInUse
     */
    double GetRunDateInUse()
    {
        return m_runDateInUse;
    }

    /** Access m_commandDownload
     * \return The current value of m_commandDownload
     */
    wxString GetCommandDownload()
    {
        return m_commandDownload;
    }

    /** Set m_commandDownload
     * \param val New value to set
     */
    void SetCommandDownload(const wxString &val)
    {
        m_commandDownload = val;
    }

    /** Access
     * \return The number of urls to download
     */
    int GetUlrsNb()
    {
        int urlsNb = m_urls.size();
        return urlsNb;
    }

    /** Access
     * \return The urls to download
     */
    VectorString GetUrls()
    {
        return m_urls;
    }

    /** Access
     * \return The url to download
     */
    wxString GetUrl(int i)
    {
        wxASSERT(m_fileNames.size()==m_urls.size());
        if ((unsigned)i>=m_urls.size()) return wxEmptyString;
        return m_urls[i];
    }

    /** Access
     * \param val New value to set
     */
    void SetUrls(const VectorString &val)
    {
        m_urls = val;
    }

    /** Access
     * \return The file names of the downloaded files
     */
    VectorString GetFileNames()
    {
        return m_fileNames;
    }

    /** Access
     * \param val New value to set
     */
    void SetFileNames(const VectorString &val)
    {
        m_fileNames = val;
    }

    /** Access
     * \return The file name of the downloaded file
     */
    wxString GetFileName(int i)
    {
        wxASSERT(m_fileNames.size()==m_urls.size());
        if ((unsigned)i>=m_fileNames.size()) return wxEmptyString;
        return m_fileNames[i];
    }

    /** Access
     * \return The number of data dates
     */
    int GetDatesNb()
    {
        wxASSERT(m_dataDates.size()==m_urls.size());
        wxASSERT(m_dataDates.size()==m_fileNames.size());
        int datesNb = m_dataDates.size();
        return datesNb;
    }

    /** Access
     * \return The dates of the downloaded files
     */
    VectorDouble GetDataDates()
    {
        return m_dataDates;
    }

    /** Access
     * \return The date of the downloaded file
     */
    double GetDataDate(int i)
    {
        wxASSERT(m_dataDates.size()==m_urls.size());
        wxASSERT(m_dataDates.size()==m_fileNames.size());
        if ((unsigned)i>=m_dataDates.size()) return asNOT_VALID;
        return m_dataDates[i];
    }

    /** Access
     * \return The lst date of the downloaded file
     */
    double GetLastDataDate()
    {
        wxASSERT(m_dataDates.size()==m_urls.size());
        wxASSERT(m_dataDates.size()==m_fileNames.size());
        return m_dataDates[m_dataDates.size()-1];
    }

    wxString GetPredictorsRealtimeDirectory()
    {
        return m_predictorsRealtimeDirectory;
    }
    
    void SetPredictorsRealtimeDirectory(const wxString &dir)
    {
        m_predictorsRealtimeDirectory = dir;
    }


protected:
    wxString m_predictorsRealtimeDirectory;
    double m_forecastLeadTimeStart;
    double m_forecastLeadTimeEnd;
    double m_forecastLeadTimeStep;
    double m_runHourStart;
    double m_runUpdate;
    double m_runDateInUse;
    wxString m_commandDownload;
    bool m_restrictDownloads;
    double m_restrictTimeHours;
    double m_restrictTimeStepHours;
    VectorString m_fileNames;
    VectorString m_urls;
    VectorDouble m_dataDates;

    /** Method to check the time array compatibility with the data
     * \param timeArray The time array to check
     * \return True if compatible with the data
     */
    virtual bool CheckTimeArray(asTimeArray &timeArray);

private:

};

#endif // ASDATAPREDICTORREALTIME_H
