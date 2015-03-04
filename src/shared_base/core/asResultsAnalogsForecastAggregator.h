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
 * Portions Copyright 2015 Pascal Horton, Terr@num.
 */

#ifndef ASRESULTSANALOGSFORECASTAGGREGATOR_H
#define ASRESULTSANALOGSFORECASTAGGREGATOR_H

#include <asIncludes.h>
#include <asResultsAnalogsForecast.h>

class asResultsAnalogsForecastAggregator: public wxObject
{
public:

    /** Default constructor */
    asResultsAnalogsForecastAggregator();

    /** Default destructor */
    virtual ~asResultsAnalogsForecastAggregator();

    void Add(asResultsAnalogsForecast* forecast);

    void AddPastForecast(int methodRow, int forecastRow, asResultsAnalogsForecast* forecast);

    void ClearArrays();

    int GetMethodsNb();

    int GetForecastsNb(int methodRow);

    int GetPastMethodsNb();

    int GetPastForecastsNb(int methodRow);
    
    int GetPastForecastsNb(int methodRow, int forecastRow);

    asResultsAnalogsForecast* GetForecast(int methodRow, int forecastRow);

    asResultsAnalogsForecast* GetPastForecast(int methodRow, int forecastRow, int leadtimeRow);

    wxString GetForecastName(int methodRow, int forecastRow);

    wxString GetMethodName(int methodRow);
    
    VectorString GetAllMethodIds();

    VectorString GetAllMethodNames();

    VectorString GetAllForecastNames();

    wxArrayString GetAllForecastNamesWxArray();

    VectorString GetFilePaths();

    wxString GetFilePath(int methodRow, int forecastRow);

    wxArrayString GetFilePathsWxArray();
    
    Array1DFloat GetTargetDates(int methodRow);

    Array1DFloat GetTargetDates(int methodRow, int forecastRow);

    Array1DFloat GetFullTargetDates();
    
    int GetForecastRowSpecificForStationId(int methodRow, int stationId);
    
    int GetForecastRowSpecificForStationRow(int methodRow, int stationRow);

    wxArrayString GetStationNames(int methodRow, int forecastRow);

    wxString GetStationName(int methodRow, int forecastRow, int stationRow);

    wxArrayString GetStationNamesWithHeights(int methodRow, int forecastRow);

    wxString GetStationNameWithHeight(int methodRow, int forecastRow, int stationRow);

    int GetLeadTimeLength(int methodRow, int forecastRow);

    int GetLeadTimeLengthMax();

    wxArrayString GetLeadTimes(int methodRow, int forecastRow);

    Array1DFloat GetMethodMaxValues(Array1DFloat &dates, int methodRow, int returnPeriodRef, float quantileThreshold);

    Array1DFloat GetOverallMaxValues(Array1DFloat &dates, int returnPeriodRef, float quantileThreshold);

    bool ExportSyntheticXml(const wxString &dirPath);

protected:

private:
    std::vector <std::vector <asResultsAnalogsForecast*> > m_forecasts;
    std::vector <std::vector <std::vector <asResultsAnalogsForecast*> > > m_pastForecasts;

};

#endif // ASRESULTSANALOGSFORECASTAGGREGATOR_H
