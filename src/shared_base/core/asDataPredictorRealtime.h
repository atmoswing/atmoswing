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

    bool LoadFullArea(double date, float level);

    bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray);

    virtual bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray);

    /** Update the RunDateInUse to the most recent hour or to the given date
     * \param now The desired date
     */
    double UpdateRunDateInUse();

    /** Set m_RunDateInUse
     * \param val New value to set
     */
    double SetRunDateInUse(double val = 0);

    /** Decrement RunDateInUse to get older data */
    double DecrementRunDateInUse();

    /** Method to build the command (url) to download the file */
    bool BuildFilenamesUrls();

    /** Restrict the downloads to used data */
    void RestrictTimeArray(double restrictDTimeHours, double restrictTimeStepHours);

    void SetRestrictDownloads(bool val)
    {
        m_RestrictDownloads = val;
    }

    /** Access m_ForecastLeadTimeStart
     * \return The current value of m_ForecastLeadTimeStart
     */
    int GetForecastLeadTimeStart()
    {
        return m_ForecastLeadTimeStart;
    }

    /** Set m_ForecastLeadTimeStart
     * \param val New value to set
     */
    void SetForecastLeadTimeStart(int val)
    {
        m_ForecastLeadTimeStart = val;
    }

    /** Access m_ForecastLeadTimeEnd
     * \return The current value of m_ForecastLeadTimeEnd
     */
    int GetForecastLeadTimeEnd()
    {
        return m_ForecastLeadTimeEnd;
    }

    /** Set m_ForecastLeadTimeEnd
     * \param val New value to set
     */
    void SetForecastLeadTimeEnd(int val)
    {
        m_ForecastLeadTimeEnd = val;
    }

    /** Access m_ForecastLeadTimeStep
     * \return The current value of m_ForecastLeadTimeStep
     */
    int GetForecastLeadTimeStep()
    {
        return m_ForecastLeadTimeStep;
    }

    /** Set m_ForecastLeadTimeStep
     * \param val New value to set
     */
    void SetForecastLeadTimeStep(int val)
    {
        m_ForecastLeadTimeStep = val;
    }

    /** Access m_RunHourStart
     * \return The current value of m_RunHourStart
     */
    int GetRunHourStart()
    {
        return m_RunHourStart;
    }

    /** Set m_RunHourStart
     * \param val New value to set
     */
    void SetRunHourStart(int val)
    {
        m_RunHourStart = val;
    }

    /** Access m_RunUpdate
     * \return The current value of m_RunUpdate
     */
    int GetRunUpdate()
    {
        return m_RunUpdate;
    }

    /** Set m_RunUpdate
     * \param val New value to set
     */
    void SetRunUpdate(int val)
    {
        m_RunUpdate = val;
    }

    /** Access m_RunDateInUse
     * \return The current value of m_RunDateInUse
     */
    double GetRunDateInUse()
    {
        return m_RunDateInUse;
    }

    /** Access m_CommandDownload
     * \return The current value of m_CommandDownload
     */
    wxString GetCommandDownload()
    {
        return m_CommandDownload;
    }

    /** Set m_CommandDownload
     * \param val New value to set
     */
    void SetCommandDownload(const wxString &val)
    {
        m_CommandDownload = val;
    }

    /** Access
     * \return The number of urls to download
     */
    int GetUlrsNb()
    {
        int urlsNb = m_Urls.size();
        return urlsNb;
    }

    /** Access
     * \return The urls to download
     */
    VectorString GetUrls()
    {
        return m_Urls;
    }

    /** Access
     * \return The url to download
     */
    wxString GetUrl(int i)
    {
        wxASSERT(m_FileNames.size()==m_Urls.size());
        if ((unsigned)i>=m_Urls.size()) return wxEmptyString;
        return m_Urls[i];
    }

    /** Access
     * \param val New value to set
     */
    void SetUrls(const VectorString &val)
    {
        m_Urls = val;
    }

    /** Access
     * \return The file names of the downloaded files
     */
    VectorString GetFileNames()
    {
        return m_FileNames;
    }

    /** Access
     * \param val New value to set
     */
    void SetFileNames(const VectorString &val)
    {
        m_FileNames = val;
    }

    /** Access
     * \return The file name of the downloaded file
     */
    wxString GetFileName(int i)
    {
        wxASSERT(m_FileNames.size()==m_Urls.size());
        if ((unsigned)i>=m_FileNames.size()) return wxEmptyString;
        return m_FileNames[i];
    }

    /** Access
     * \return The number of data dates
     */
    int GetDatesNb()
    {
        wxASSERT(m_DataDates.size()==m_Urls.size());
        wxASSERT(m_DataDates.size()==m_FileNames.size());
        int datesNb = m_DataDates.size();
        return datesNb;
    }

    /** Access
     * \return The dates of the downloaded files
     */
    VectorDouble GetDataDates()
    {
        return m_DataDates;
    }

    /** Access
     * \return The date of the downloaded file
     */
    double GetDataDate(int i)
    {
        wxASSERT(m_DataDates.size()==m_Urls.size());
        wxASSERT(m_DataDates.size()==m_FileNames.size());
        if ((unsigned)i>=m_DataDates.size()) return asNOT_VALID;
        return m_DataDates[i];
    }

    /** Access
     * \return The lst date of the downloaded file
     */
    double GetLastDataDate()
    {
        wxASSERT(m_DataDates.size()==m_Urls.size());
        wxASSERT(m_DataDates.size()==m_FileNames.size());
        return m_DataDates[m_DataDates.size()-1];
    }


protected:
    double m_ForecastLeadTimeStart;
    double m_ForecastLeadTimeEnd;
    double m_ForecastLeadTimeStep;
    double m_RunHourStart;
    double m_RunUpdate;
    double m_RunDateInUse;
    wxString m_CommandDownload;
    bool m_RestrictDownloads;
    double m_RestrictDTimeHours;
    double m_RestrictTimeStepHours;
    VectorString m_FileNames;
    VectorString m_Urls;
    VectorDouble m_DataDates;
    FileFormat m_FileFormat;

    /** Method to check the time array compatibility with the data
     * \param timeArray The time array to check
     * \return True if compatible with the data
     */
    bool CheckTimeArray(asTimeArray &timeArray);

private:

};

#endif // ASDATAPREDICTORREALTIME_H
