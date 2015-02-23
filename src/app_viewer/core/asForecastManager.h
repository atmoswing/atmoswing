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
 
#ifndef ASFORECASTMANAGER_H
#define ASFORECASTMANAGER_H

#include <asIncludes.h>
#include <asResultsAnalogsForecastAggregator.h>
#include "asWorkspace.h"

class asForecastManager
{
public:
    /** Default constructor */
    asForecastManager(wxWindow* parent, asWorkspace* workspace);

    /** Default destructor */
    virtual ~asForecastManager();

    void ClearArrays();

    void ClearForecasts();

    bool Open(const wxString &filePath, bool doRefresh = true);

    bool OpenPastForecast(int methodRow, int forecastRow, const wxString &filePath);

    void LoadPastForecast(int methodRow, int forecastRow);

    void LoadPastForecast(int methodRow);

    void UpdateAlarms();

    void AddDirectoryPastForecasts(const wxString &dir);

    asResultsAnalogsForecastAggregator* GetAggregator()
    {
        return m_Aggregator;
    }

    int GetMethodsNb()
    {
        return m_Aggregator->GetMethodsNb();
    }

    int GetForecastsNb(int methodRow)
    {
        return m_Aggregator->GetForecastsNb(methodRow);
    }

    int GetPastMethodsNb()
    {
        return m_Aggregator->GetPastMethodsNb();
    }

    int GetPastForecastsNb(int methodRow)
    {
        return m_Aggregator->GetPastForecastsNb(methodRow);
    }
    
    int GetPastForecastsNb(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetPastForecastsNb(methodRow, forecastRow);
    }

    asResultsAnalogsForecast* GetForecast(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetForecast(methodRow, forecastRow);
    }

    asResultsAnalogsForecast* GetPastForecast(int methodRow, int forecastRow, int leadtimeRow)
    {
        return m_Aggregator->GetPastForecast(methodRow, forecastRow, leadtimeRow);
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

    wxString GetForecastName(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetForecastName(methodRow, forecastRow);
    }

    wxString GetMethodName(int methodRow)
    {
        return m_Aggregator->GetMethodName(methodRow);
    }

    VectorString GetAllMethodNames()
    {
        return m_Aggregator->GetAllMethodNames();
    }

    VectorString GetAllForecastNames()
    {
        return m_Aggregator->GetAllForecastNames();
    }

    wxArrayString GetAllForecastNamesWxArray()
    {
        return m_Aggregator->GetAllForecastNamesWxArray();
    }

    VectorString GetFilePaths()
    {
        return m_Aggregator->GetFilePaths();
    }

    wxString GetFilePath(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetFilePath(methodRow, forecastRow);
    }

    wxArrayString GetFilePathsWxArray()
    {
        return m_Aggregator->GetFilePathsWxArray();
    }

    Array1DFloat GetTargetDates(int methodRow)
    {
        return m_Aggregator->GetTargetDates(methodRow);
    }

    Array1DFloat GetTargetDates(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetTargetDates(methodRow, forecastRow);
    }

    Array1DFloat GetFullTargetDates()
    {
        return m_Aggregator->GetFullTargetDates();
    }

    int GetForecastRowSpecificForStation(int methodRow, int stationRow)
    {
        return m_Aggregator->GetForecastRowSpecificForStation(methodRow, stationRow);
    }

    wxArrayString GetStationNames(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetStationNames(methodRow, forecastRow);
    }

    wxString GetStationName(int methodRow, int forecastRow, int stationRow)
    {
        return m_Aggregator->GetStationName(methodRow, forecastRow, stationRow);
    }

    wxArrayString GetStationNamesWithHeights(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetStationNamesWithHeights(methodRow, forecastRow);
    }

    wxString GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow)
    {
        return m_Aggregator->GetStationNameWithHeight(methodRow, forecastRow, stationRow);
    }

    int GetLeadTimeLength(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetLeadTimeLength(methodRow, forecastRow);
    }

    int GetLeadTimeLengthMax()
    {
        return m_Aggregator->GetLeadTimeLengthMax();
    }

    wxArrayString GetLeadTimes(int methodRow, int forecastRow)
    {
        return m_Aggregator->GetLeadTimes(methodRow, forecastRow);
    } 

protected:
private:
    wxWindow* m_Parent;
    asWorkspace* m_Workspace;
    asResultsAnalogsForecastAggregator* m_Aggregator;
    double m_LeadTimeOrigin; //!< Member variable "m_LeadTimeOrigin"
    wxArrayString m_DirectoriesPastForecasts;

};

#endif // ASFORECASTMANAGER_H
