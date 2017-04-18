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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#ifndef ASRESULTSANALOGSFORECASTAGGREGATOR_H
#define ASRESULTSANALOGSFORECASTAGGREGATOR_H

#include <asIncludes.h>
#include <asResultsAnalogsForecast.h>

class asResultsAnalogsForecastAggregator
        : public wxObject
{
public:
    asResultsAnalogsForecastAggregator();

    virtual ~asResultsAnalogsForecastAggregator();

    bool Add(asResultsAnalogsForecast *forecast);

    bool AddPastForecast(int methodRow, int forecastRow, asResultsAnalogsForecast *forecast);

    void ClearArrays();

    int GetMethodsNb() const;

    int GetForecastsNb(int methodRow) const;

    int GetPastMethodsNb() const;

    int GetPastForecastsNb(int methodRow) const;

    int GetPastForecastsNb(int methodRow, int forecastRow) const;

    asResultsAnalogsForecast *GetForecast(int methodRow, int forecastRow) const;

    asResultsAnalogsForecast *GetPastForecast(int methodRow, int forecastRow, int leadtimeRow) const;

    wxString GetForecastName(int methodRow, int forecastRow) const;

    wxString GetMethodName(int methodRow) const;

    vwxs GetAllMethodIds() const;

    vwxs GetAllMethodNames() const;

    vwxs GetAllForecastNames() const;

    wxArrayString GetAllForecastNamesWxArray() const;

    vwxs GetFilePaths() const;

    wxString GetFilePath(int methodRow, int forecastRow) const;

    wxArrayString GetFilePathsWxArray() const;

    a1f GetTargetDates(int methodRow) const;

    a1f GetTargetDates(int methodRow, int forecastRow) const;

    a1f GetFullTargetDates() const;

    int GetForecastRowSpecificForStationId(int methodRow, int stationId) const;

    int GetForecastRowSpecificForStationRow(int methodRow, int stationRow) const;

    wxArrayString GetStationNames(int methodRow, int forecastRow) const;

    wxString GetStationName(int methodRow, int forecastRow, int stationRow) const;

    wxArrayString GetStationNamesWithHeights(int methodRow, int forecastRow) const;

    wxString GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow) const;

    int GetLeadTimeLength(int methodRow, int forecastRow) const;

    int GetLeadTimeLengthMax() const;

    wxArrayString GetLeadTimes(int methodRow, int forecastRow) const;

    a1f GetMethodMaxValues(a1f &dates, int methodRow, int returnPeriodRef, float quantileThreshold) const;

    a1f GetOverallMaxValues(a1f &dates, int returnPeriodRef, float quantileThreshold) const;

    bool ExportSyntheticXml(const wxString &dirPath) const;

protected:

private:
    std::vector<std::vector<asResultsAnalogsForecast *> > m_forecasts;
    std::vector<std::vector<std::vector<asResultsAnalogsForecast *> > > m_pastForecasts;

};

#endif // ASRESULTSANALOGSFORECASTAGGREGATOR_H
