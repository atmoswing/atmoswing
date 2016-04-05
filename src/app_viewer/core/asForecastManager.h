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

    bool HasForecasts() const;

    void ClearArrays();

    void ClearForecasts();

    bool Open(const wxString &filePath, bool doRefresh = true);

    bool OpenPastForecast(int methodRow, int forecastRow, const wxString &filePath);

    void LoadPastForecast(int methodRow, int forecastRow);

    void LoadPastForecast(int methodRow);

    void AddDirectoryPastForecasts(const wxString &dir);

    int GetLinearIndex(int methodRow, int forecastRow) const;

    int GetMethodRowFromLinearIndex(int linearIndex) const;

    int GetForecastRowFromLinearIndex(int linearIndex) const;

    asResultsAnalogsForecastAggregator *GetAggregator() const
    {
        return m_aggregator;
    }

    int GetMethodsNb() const
    {
        return m_aggregator->GetMethodsNb();
    }

    int GetForecastsNb(int methodRow) const
    {
        return m_aggregator->GetForecastsNb(methodRow);
    }

    int GetPastMethodsNb() const
    {
        return m_aggregator->GetPastMethodsNb();
    }

    int GetPastForecastsNb(int methodRow) const
    {
        return m_aggregator->GetPastForecastsNb(methodRow);
    }

    int GetPastForecastsNb(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetPastForecastsNb(methodRow, forecastRow);
    }

    asResultsAnalogsForecast *GetForecast(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetForecast(methodRow, forecastRow);
    }

    asResultsAnalogsForecast *GetPastForecast(int methodRow, int forecastRow, int leadtimeRow) const
    {
        return m_aggregator->GetPastForecast(methodRow, forecastRow, leadtimeRow);
    }

    double GetLeadTimeOrigin() const
    {
        return m_leadTimeOrigin;
    }

    void SetLeadTimeOrigin(double val)
    {
        m_leadTimeOrigin = val;
    }

    wxString GetForecastName(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetForecastName(methodRow, forecastRow);
    }

    wxString GetMethodName(int methodRow) const
    {
        return m_aggregator->GetMethodName(methodRow);
    }

    VectorString GetAllMethodNames() const
    {
        return m_aggregator->GetAllMethodNames();
    }

    VectorString GetAllForecastNames() const
    {
        return m_aggregator->GetAllForecastNames();
    }

    wxArrayString GetAllForecastNamesWxArray() const
    {
        return m_aggregator->GetAllForecastNamesWxArray();
    }

    VectorString GetFilePaths() const
    {
        return m_aggregator->GetFilePaths();
    }

    wxString GetFilePath(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetFilePath(methodRow, forecastRow);
    }

    wxArrayString GetFilePathsWxArray() const
    {
        return m_aggregator->GetFilePathsWxArray();
    }

    Array1DFloat GetTargetDates(int methodRow) const
    {
        return m_aggregator->GetTargetDates(methodRow);
    }

    Array1DFloat GetTargetDates(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetTargetDates(methodRow, forecastRow);
    }

    Array1DFloat GetFullTargetDates() const
    {
        return m_aggregator->GetFullTargetDates();
    }

    int GetForecastRowSpecificForStationId(int methodRow, int stationId) const
    {
        return m_aggregator->GetForecastRowSpecificForStationId(methodRow, stationId);
    }

    int GetForecastRowSpecificForStationRow(int methodRow, int stationRow) const
    {
        return m_aggregator->GetForecastRowSpecificForStationRow(methodRow, stationRow);
    }

    wxArrayString GetStationNames(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetStationNames(methodRow, forecastRow);
    }

    wxString GetStationName(int methodRow, int forecastRow, int stationRow) const
    {
        return m_aggregator->GetStationName(methodRow, forecastRow, stationRow);
    }

    wxArrayString GetStationNamesWithHeights(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetStationNamesWithHeights(methodRow, forecastRow);
    }

    wxString GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow) const
    {
        return m_aggregator->GetStationNameWithHeight(methodRow, forecastRow, stationRow);
    }

    int GetLeadTimeLength(int methodRow, int forecastRow) const
    {
        return m_aggregator->GetLeadTimeLength(methodRow, forecastRow);
    }

    int GetLeadTimeLengthMax() const
    {
        return m_aggregator->GetLeadTimeLengthMax();
    }

    wxArrayString GetLeadTimes(int methodRow, int forecastRow) const
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
