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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */
 
#ifndef ASMETHODFORECASTING_H
#define ASMETHODFORECASTING_H

#include <asIncludes.h>
#include <asMethodStandard.h>
#include <asParametersForecast.h>
#include <asDataPredictorRealtime.h>
#include <asDataPredictorArchive.h>
#include <asPredictorCriteria.h>
#include <asBatchForecasts.h>
#include <asResultsAnalogsForecastAggregator.h>

class asResultsAnalogsDates;
class asResultsAnalogsForecast;

class asMethodForecasting: public asMethodStandard
{
public:
    asMethodForecasting(asBatchForecasts* batchForecasts, wxWindow* parent = NULL);
    virtual ~asMethodForecasting();

    void ClearForecasts();

    virtual bool Manager();

    bool Forecast(asParametersForecast &params);

    /** Access m_forecastDate
     * \return The current value of m_forecastDate
     */
    double GetForecastDate()
    {
        return m_forecastDate;
    }

    /** Set m_forecastDate
     * \param val New value to set
     */
    void SetForecastDate(double val)
    {
        m_forecastDate = val;
    }

    /** Get m_resultsFilePaths
     * \return The current value of m_resultsFilePaths
     */
    VectorString GetResultsFilePaths()
    {
        return m_resultsFilePaths;
    }

protected:
    bool DownloadRealtimePredictors(asParametersForecast &params, int i_step, bool &forecastDateChanged);
    bool GetAnalogsDates(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step);
    bool GetAnalogsSubDates(asResultsAnalogsForecast &results, asParametersForecast &params, asResultsAnalogsForecast &resultsPrev, int i_step);
    bool GetAnalogsValues(asResultsAnalogsForecast &results, asParametersForecast &params, int i_step);
    
    void DeletePreprocessData();
    void Cleanup();


private:
    asBatchForecasts* m_batchForecasts;
    double m_forecastDate;
    asResultsAnalogsForecastAggregator m_aggregator;
    VectorString m_resultsFilePaths;
    wxWindow* m_parent;
    std::vector < asDataPredictorArchive* > m_storagePredictorsArchivePreprocess;
	std::vector < asDataPredictorRealtime* > m_storagePredictorsRealtimePreprocess;
    std::vector < asDataPredictor* > m_storagePredictorsArchive;
	std::vector < asDataPredictor* > m_storagePredictorsRealtime;
    std::vector < asPredictorCriteria* > m_storageCriteria;
};

#endif // ASMETHODFORECASTING_H
