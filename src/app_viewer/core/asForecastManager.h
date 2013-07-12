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
 
#ifndef ASFORECASTMANAGER_H
#define ASFORECASTMANAGER_H

#include <asIncludes.h>


class asResultsAnalogsForecast;

class asForecastManager
{
public:
    /** Default constructor */
    asForecastManager(wxWindow* parent);

    /** Default destructor */
    virtual ~asForecastManager();

    void ClearArrays();

    bool Open(const wxString &filePath, bool doRefresh = true);

    bool OpenPastForecast(const wxString &filePath, int forecastSelection);

    void LoadPastForecast(int forecastSelection);

    void UpdateAlarms();

    wxString GetModelName(int i_fcst);

    VectorString GetModelsNames();

    wxArrayString GetModelsNamesWxArray();

    VectorString GetFilePaths();

    wxArrayString GetFilePathsWxArray();

    Array1DFloat GetFullTargetDatesVector();

    wxArrayString GetStationNames(int i_fcst);

    wxString GetStationName(int i_fcst, int i_stat);

    wxArrayString GetStationNamesWithHeights(int i_fcst);

    wxString GetStationNameWithHeight(int i_fcst, int i_stat);

    int GetLeadTimeLength(int i_fcst);

    wxArrayString GetLeadTimes(int i_fcst);

    void AddDirectoryPastForecasts(const wxString &dir);

    /** Get the number of models
     * \return The number of current forecasts
     */
    int GetModelsNb()
    {
        return m_CurrentForecasts.size();
    }

    /** Get the number of current forecasts
     * \return The number of current forecasts
     */
    int GetCurrentForecastsNb()
    {
        return m_CurrentForecasts.size();
    }

    /** Access m_CurrentForecasts
     * \return The current value of m_CurrentForecasts
     */
    std::vector <asResultsAnalogsForecast*> GetCurrentForecasts()
    {
        return m_CurrentForecasts;
    }

    /** Access m_CurrentForecast[i]
     * \return The ith element in m_CurrentForecasts
     */
    asResultsAnalogsForecast* GetCurrentForecast(int i)
    {
        return m_CurrentForecasts[i];
    }

    /** Set m_CurrentForecasts
     * \param val New value to set
     */
    void SetCurrentForecasts(std::vector <asResultsAnalogsForecast*> val)
    {
        m_CurrentForecasts = val;
    }

    /** Access m_PastForecasts
     * \return The current value of m_PastForecasts
     */
    std::vector <asResultsAnalogsForecast*> GetPastForecasts(int forecast)
    {
        return m_PastForecasts[forecast];
    }

    /** Access m_PastForecasts
     * \return The current value of m_PastForecasts
     */
    asResultsAnalogsForecast* GetPastForecast(int forecast, int time)
    {
        return m_PastForecasts[forecast][time];
    }

    /** Access m_PastForecasts
     * \return The current value of m_PastForecasts
     */
    int GetPastForecastsNb(int i)
    {
        return m_PastForecasts[i].size();
    }

    /** Access m_LeadTimeOrigin
     * \return The current value of m_LeadTimeOrigin
     */
    double GetLeadTimeOrigin()
    {
        return m_LeadTimeOrigin;
    }

    /** Set m_LeadTimeOrigin
     * \param val New value to set
     */
    void SetLeadTimeOrigin(double val)
    {
        m_LeadTimeOrigin = val;
    }

protected:
private:
    wxWindow* m_Parent;
    std::vector <asResultsAnalogsForecast*> m_CurrentForecasts; //!< Member variable "m_CurrentForecasts"
    std::vector <std::vector <asResultsAnalogsForecast*> > m_PastForecasts; //!< Member variable "m_PastForecasts"
    double m_LeadTimeOrigin; //!< Member variable "m_LeadTimeOrigin"
    wxArrayString m_DirectoriesPastForecasts;

};

#endif // ASFORECASTMANAGER_H
