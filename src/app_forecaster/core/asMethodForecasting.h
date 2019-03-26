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
#include <asPredictorOper.h>
#include <asPredictor.h>
#include <asCriteria.h>
#include <asBatchForecasts.h>
#include <asResultsForecastAggregator.h>

class asResultsDates;

class asResultsForecast;

class asMethodForecasting
        : public asMethodStandard
{
public:
    explicit asMethodForecasting(asBatchForecasts *batchForecasts, wxWindow *parent = nullptr);

    ~asMethodForecasting() override;

    void ClearForecasts();

    bool Manager() override;

    bool Forecast(asParametersForecast &params);

    double GetForecastDate() const
    {
        return m_forecastDate;
    }

    void SetForecastDate(double val)
    {
        m_forecastDate = val;
    }

    vwxs GetResultsFilePaths() const
    {
        return m_resultsFilePaths;
    }

protected:
    bool DownloadRealtimePredictors(asParametersForecast &params, int iStep, bool &forecastDateChanged);

    bool PreprocessRealtimePredictors(std::vector<asPredictorOper *> predictors, const wxString &method,
                                      asPredictor *result);

    bool GetAnalogsDates(asResultsForecast &results, asParametersForecast &params, int iStep);

    bool GetAnalogsSubDates(asResultsForecast &results, asParametersForecast &params, asResultsForecast &resultsPrev,
                            int iStep);

    bool GetAnalogsValues(asResultsForecast &results, asParametersForecast &params, int iStep);

    void DeletePreprocessData();

    double GetEffectiveArchiveDataStart(asParameters *params) const override;

    double GetEffectiveArchiveDataEnd(asParameters *params) const override;

    void Cleanup();

private:
    asBatchForecasts *m_batchForecasts;
    double m_forecastDate;
    asResultsForecastAggregator m_aggregator;
    vwxs m_resultsFilePaths;
    wxWindow *m_parent;
    std::vector<asPredictor *> m_storagePredictorsArchivePreprocess;
    std::vector<asPredictorOper *> m_storagePredictorsRealtimePreprocess;
    std::vector<asPredictor *> m_storagePredictorsArchive;
    std::vector<asPredictor *> m_storagePredictorsRealtime;
    std::vector<asCriteria *> m_storageCriteria;
};

#endif // ASMETHODFORECASTING_H
