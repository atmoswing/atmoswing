/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASCATALOGPREDICTORREALTIME_H
#define ASCATALOGPREDICTORREALTIME_H

#include <asIncludes.h>
#include <asCatalogPredictors.h>


class asCatalogPredictorsRealtime: public asCatalogPredictors
{
public:

    /** Default constructor
     * \param DataSetId The dataset ID
     * \param DataId The data ID. If not set, load the whole database information
     */
    asCatalogPredictorsRealtime(const wxString &AlternateFilePath = wxEmptyString);

    /** Default destructor */
    virtual ~asCatalogPredictorsRealtime();

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

     /** Method that get data information from file
     * \param DataSetId The data Set ID
     * \param DataId The data ID
     */
    bool Load(const wxString &DataSetId, const wxString &DataId = wxEmptyString);

    /** Method that get dataset information from file
     * \param DataSetId The data Set ID
     */
    bool LoadDatasetProp(const wxString &DataSetId);

    /** Method that get data information from file
     * \param DataSetId The data Set ID
     * \param DataId The data ID
     */
    bool LoadDataProp(const wxString &DataSetId, const wxString &DataId);

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
     * \return The file names of the downloaded files
     */
    VectorString GetFileNames()
    {
        return m_FileNames;
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
    int m_ForecastLeadTimeStart;
    int m_ForecastLeadTimeEnd;
    int m_ForecastLeadTimeStep;
    int m_RunHourStart;
    int m_RunUpdate;
    double m_RunDateInUse;
    wxString m_CommandDownload;
    VectorString m_FileNames;
    VectorString m_Urls;
    VectorDouble m_DataDates;
    bool m_RestrictDownloads;
    double m_RestrictDTimeHours;
    double m_RestrictTimeStepHours;

private:

};

#endif // ASCATALOGPREDICTORREALTIME_H
