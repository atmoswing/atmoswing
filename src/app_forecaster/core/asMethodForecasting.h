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
 
#ifndef ASMETHODFORECASTING_H
#define ASMETHODFORECASTING_H

#include <asIncludes.h>
#include <asMethodStandard.h>
#include <asParametersForecast.h>
#include <asDataPredictorRealtime.h>
#include <asDataPredictorArchive.h>
#include <asPredictorCriteria.h>

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
    
    void DeletePreprocessData();
    void Cleanup();


private:
    double m_ForecastDate;
    wxString m_ModelName;
    wxString m_PredictorsArchiveDir;
    VectorString m_ResultsFilePaths;
    wxWindow* m_Parent;
    std::vector < asDataPredictorArchive* > m_StoragePredictorsArchivePreprocess;
	std::vector < asDataPredictorRealtime* > m_StoragePredictorsRealtimePreprocess;
    std::vector < asDataPredictor* > m_StoragePredictorsArchive;
	std::vector < asDataPredictor* > m_StoragePredictorsRealtime;
    std::vector < asPredictorCriteria* > m_StorageCriteria;
};

#endif // ASMETHODFORECASTING_H
