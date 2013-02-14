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
 
#ifndef ASMETHODFORECASTING_H
#define ASMETHODFORECASTING_H

#include <asIncludes.h>
#include <asMethodStandard.h>
#include <asParametersForecast.h>

class asCatalogPredictorsRealtime;
class asResultsAnalogsDates;
class asResultsAnalogsForecast;

class asMethodForecasting: public asMethodStandard
{
public:
    asMethodForecasting(wxWindow* parent = NULL);
    virtual ~asMethodForecasting();

    virtual bool Manager();

    bool Forecast(asParametersForecast &params);

    /** Access m_ForecastDate
     * \return The current value of m_ForecastDate
     */
    double GetForecastDate()
    {
        return m_ForecastDate;
    }

    /** Set m_ForecastDate
     * \param val New value to set
     */
    void SetForecastDate(double val)
    {
        m_ForecastDate = val;
    }

    /** Get m_ResultsFilePaths
     * \return The current value of m_ResultsFilePaths
     */
    VectorString GetResultsFilePaths()
    {
        return m_ResultsFilePaths;
    }

protected:
    bool DownloadRealtimePredictors(asParametersForecast &params, int i_step, bool &forecastDateChanged);
    bool GetAnalogsDates(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step);
    bool GetAnalogsSubDates(asResultsAnalogsForecast &results, asParametersForecast &params, asResultsAnalogsForecast &resultsPrev, int i_step);
    bool GetAnalogsValues(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step);


private:
    double m_ForecastDate;
    wxString m_ModelName;
    wxString m_PredictorsArchiveDir;
    VectorString m_ResultsFilePaths;
    wxWindow* m_Parent;
};

#endif // ASMETHODFORECASTING_H
