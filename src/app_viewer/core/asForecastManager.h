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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#ifndef ASFORECASTMANAGER_H
#define ASFORECASTMANAGER_H

#include <asIncludes.h>
#include <asResultsAnalogsForecastAggregator.h>
#include "asWorkspace.h"

class asForecastManager
{
public:
    asForecastManager(wxWindow *parent, asWorkspace *workspace);

    virtual ~asForecastManager();

    bool HasForecasts();

    void ClearArrays();

    void ClearForecasts();

    bool Open(const wxString &filePath, bool doRefresh = true);

    bool OpenPastForecast(int methodRow, int forecastRow, const wxString &filePath);

    void LoadPastForecast(int methodRow, int forecastRow);

    void LoadPastForecast(int methodRow);

    void UpdateAlarms();

    void AddDirectoryPastForecasts(const wxString &dir);

    int GetLinearIndex(int methodRow, int forecastRow);

    int GetMethodRowFromLinearIndex(int linearIndex);

    int GetForecastRowFromLinearIndex(int linearIndex);

    asResultsAnalogsForecastAggregator *GetAggregator()
    {
        return m_aggregator;
    }

    int GetMethodsNb()
    {
        return m_aggregator->GetMethodsNb();
    }

    int GetForecastsNb(int methodRow)
    {
        return m_aggregator->GetForecastsNb(methodRow);
    }

    int GetPastMethodsNb()
    {
        return m_aggregator->GetPastMethodsNb();
    }

    int GetPastForecastsNb(int methodRow)
    {
        return m_aggregator->GetPastForecastsNb(methodRow);
    }

    int GetPastForecastsNb(int methodRow, int forecastRow)
    {
        return m_aggregator->GetPastForecastsNb(methodRow, forecastRow);
    }

    asResultsAnalogsForecast *GetForecast(int methodRow, int forecastRow)
    {
        return m_aggregator->GetForecast(methodRow, forecastRow);
    }

    asResultsAnalogsForecast *GetPastForecast(int methodRow, int forecastRow, int leadtimeRow)
    {
        return m_aggregator->GetPastForecast(methodRow, forecastRow, leadtimeRow);
    }

    double GetLeadTimeOrigin()
    {
        return m_leadTimeOrigin;
    }

    void SetLeadTimeOrigin(double val)
    {
        m_leadTimeOrigin = val;
    }

    wxString GetForecastName(int methodRow, int forecastRow)
    {
        return m_aggregator->GetForecastName(methodRow, forecastRow);
    }

    wxString GetMethodName(int methodRow)
    {
        return m_aggregator->GetMethodName(methodRow);
    }

    VectorString GetAllMethodNames()
    {
        return m_aggregator->GetAllMethodNames();
    }

    VectorString GetAllForecastNames()
    {
        return m_aggregator->GetAllForecastNames();
    }

    wxArrayString GetAllForecastNamesWxArray()
    {
        return m_aggregator->GetAllForecastNamesWxArray();
    }

    VectorString GetFilePaths()
    {
        return m_aggregator->GetFilePaths();
    }

    wxString GetFilePath(int methodRow, int forecastRow)
    {
        return m_aggregator->GetFilePath(methodRow, forecastRow);
    }

    wxArrayString GetFilePathsWxArray()
    {
        return m_aggregator->GetFilePathsWxArray();
    }

    Array1DFloat GetTargetDates(int methodRow)
    {
        return m_aggregator->GetTargetDates(methodRow);
    }

    Array1DFloat GetTargetDates(int methodRow, int forecastRow)
    {
        return m_aggregator->GetTargetDates(methodRow, forecastRow);
    }

    Array1DFloat GetFullTargetDates()
    {
        return m_aggregator->GetFullTargetDates();
    }

    int GetForecastRowSpecificForStationId(int methodRow, int stationId)
    {
        return m_aggregator->GetForecastRowSpecificForStationId(methodRow, stationId);
    }

    int GetForecastRowSpecificForStationRow(int methodRow, int stationRow)
    {
        return m_aggregator->GetForecastRowSpecificForStationRow(methodRow, stationRow);
    }

    wxArrayString GetStationNames(int methodRow, int forecastRow)
    {
        return m_aggregator->GetStationNames(methodRow, forecastRow);
    }

    wxString GetStationName(int methodRow, int forecastRow, int stationRow)
    {
        return m_aggregator->GetStationName(methodRow, forecastRow, stationRow);
    }

    wxArrayString GetStationNamesWithHeights(int methodRow, int forecastRow)
    {
        return m_aggregator->GetStationNamesWithHeights(methodRow, forecastRow);
    }

    wxString GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow)
    {
        return m_aggregator->GetStationNameWithHeight(methodRow, forecastRow, stationRow);
    }

    int GetLeadTimeLength(int methodRow, int forecastRow)
    {
        return m_aggregator->GetLeadTimeLength(methodRow, forecastRow);
    }

    int GetLeadTimeLengthMax()
    {
        return m_aggregator->GetLeadTimeLengthMax();
    }

    wxArrayString GetLeadTimes(int methodRow, int forecastRow)
    {
        return m_aggregator->GetLeadTimes(methodRow, forecastRow);
    }

protected:

private:
    wxWindow *m_parent;
    asWorkspace *m_workspace;
    asResultsAnalogsForecastAggregator *m_aggregator;
    double m_leadTimeOrigin;
    wxArrayString m_directoriesPastForecasts;

};

#endif // ASFORECASTMANAGER_H
