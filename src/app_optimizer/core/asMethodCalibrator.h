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

#ifndef ASMETHODCALIBRATOR_H
#define ASMETHODCALIBRATOR_H

#include <asIncludes.h>
#include <asMethodStandard.h>
#include <asParametersCalibration.h>
#include <asResultsAnalogsDates.h>
#include <asDataPredictorArchive.h>
#include <asResultsParametersArray.h>
#include <asResultsAnalogsDates.h>
#include <asResultsAnalogsValues.h>
#include <asResultsAnalogsForecastScores.h>
#include <asResultsAnalogsForecastScoreFinal.h>
#include <asResultsAnalogsScoresMap.h>
#include <asParametersCalibration.h>
#include <asParametersOptimization.h>
#include <asPredictorCriteria.h>
#include <asGeoAreaCompositeGrid.h>
#include <asTimeArray.h>
#include <asProcessor.h>
#include <asProcessorForecastScore.h>
#include <asPreprocessor.h>
#include <asForecastScore.h>


class asMethodCalibrator
        : public asMethodStandard
{
public:
    asMethodCalibrator();

    virtual ~asMethodCalibrator();

    bool GetAnalogsDates(asResultsAnalogsDates &results, asParametersScoring &params, int i_step, bool &containsNaNs);

    bool GetAnalogsSubDates(asResultsAnalogsDates &results, asParametersScoring &params,
                            asResultsAnalogsDates &anaDates, int i_step, bool &containsNaNs);

    bool GetAnalogsValues(asResultsAnalogsValues &results, asParametersScoring &params, asResultsAnalogsDates &anaDates,
                          int i_step);

    bool GetAnalogsForecastScores(asResultsAnalogsForecastScores &results, asParametersScoring &params,
                                  asResultsAnalogsValues &anaValues, int i_step);

    bool GetAnalogsForecastScoreFinal(asResultsAnalogsForecastScoreFinal &results, asParametersScoring &params,
                                      asResultsAnalogsForecastScores &anaScores, int i_step);

    bool SubProcessAnalogsNumber(asParametersCalibration &params, asResultsAnalogsDates &anaDatesPrevious,
                                 int i_step = 0);

    bool PreloadDataWithoutPreprocessing(asParametersScoring &params, int i_step, int i_ptor, int i_dat);

    bool PreloadDataWithPreprocessing(asParametersScoring &params, int i_step, int i_ptor);

    void Cleanup(std::vector<asDataPredictorArchive *> predictorsPreprocess);

    void Cleanup(std::vector<asDataPredictor *> predictors);

    void Cleanup(std::vector<asPredictorCriteria *> criteria);

    void DeletePreloadedData();

    void ClearAll();

    void ClearTemp();

    void PushBackBestTemp();

    void RemoveNaNsInTemp();

    void KeepBestTemp();

    void PushBackFirstTemp();

    void KeepFirstTemp();

    void SortScoresAndParametersTemp();

    bool PushBackInTempIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal);

    bool KeepIfBetter(asParametersCalibration &params, asResultsAnalogsForecastScoreFinal &scoreFinal);

    bool SetSelectedParameters(asResultsParametersArray &results);

    bool SetBestParameters(asResultsParametersArray &results);

    wxString GetPredictandStationIdsList(VectorInt &stationIds) const;

    bool Manager();

    virtual bool Validate(const int bestscorerow = 0);

    void SetScore(float valCalib)
    {
        m_scoresCalibTemp.push_back(valCalib);
    }

    void SetScoreOrder(Order val)
    {
        m_scoreOrder = val;
    }

    VectorFloat GetScoreClimatology() const
    {
        return m_scoreClimatology;
    }

    void SetScoreClimatology(VectorFloat val)
    {
        m_scoreClimatology = val;
    }

    bool IsPointerCopy(int i_step, int i_ptor, int i_dat) const
    {
        return m_preloadedArchivePointerCopy[i_step][i_ptor][i_dat];
    }

    void SetPredictandStationIds(VectorInt val)
    {
        m_predictandStationIds = val;
    }

protected:
    VectorInt m_predictandStationIds;
    VectorFloat m_scoresCalib;
    VectorFloat m_scoresCalibTemp;
    Order m_scoreOrder;
    float m_scoreValid;
    VectorFloat m_scoreClimatology;
    std::vector<asParametersCalibration> m_parameters;
    std::vector<asParametersCalibration> m_parametersTemp;
    asParametersCalibration m_originalParams;
    bool m_preloaded;
    bool m_validationMode;

    virtual bool Calibrate(asParametersCalibration &params) = 0;

    bool PreloadData(asParametersScoring &params);

    bool LoadData(std::vector<asDataPredictor *> &predictors, asParametersScoring &params, int i_step,
                  double timeStartData, double timeEndData);

    VArray1DFloat GetClimatologyData(asParametersScoring &params);

private:
    std::vector<std::vector<std::vector<std::vector<std::vector<asDataPredictorArchive *> > > > > m_preloadedArchive;
    std::vector<VVectorBool> m_preloadedArchivePointerCopy;

    bool PointersShared(asParametersScoring &params, int i_step, int i_ptor, int i_dat);

    double GetTimeStartCalibration(asParametersScoring &params) const;

    double GetTimeStartArchive(asParametersScoring &params) const;

    double GetTimeEndCalibration(asParametersScoring &params) const;

    double GetTimeEndArchive(asParametersScoring &params) const;

    void InitializePreloadedDataContainer(asParametersScoring &params);

    void LoadScoreOrder(asParametersCalibration &params);

    bool ExtractPreloadedData(std::vector<asDataPredictor *> &predictors, asParametersScoring &params, int i_step,
                              int i_ptor);

    bool ExtractDataWithoutPreprocessing(std::vector<asDataPredictor *> &predictors, asParametersScoring &params,
                                         int i_step, int i_ptor, double timeStartData, double timeEndData);

    bool ExtractDataWithPreprocessing(std::vector<asDataPredictor *> &predictors, asParametersScoring &params,
                                      int i_step, int i_ptor, double timeStartData, double timeEndData);

    bool HasPreloadedData(int i_step, int i_ptor) const;

    bool HasPreloadedData(int i_step, int i_ptor, int i_dat) const;

    bool GetRandomValidData(asParametersScoring &params, int i_step, int i_ptor, int i_dat);

    bool CheckDataIsPreloaded(const asParametersScoring &params) const;

    bool ProceedToDataPreloading(asParametersScoring &params);
};

#endif // ASMETHODCALIBRATOR_H
