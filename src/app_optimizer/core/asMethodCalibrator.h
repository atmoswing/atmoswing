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

#ifndef AS_METHOD_CALIBRATOR_H
#define AS_METHOD_CALIBRATOR_H

#include <utility>

#include "asAreaGrid.h"
#include "asCriteria.h"
#include "asIncludes.h"
#include "asMethodStandard.h"
#include "asParametersCalibration.h"
#include "asParametersOptimization.h"
#include "asPredictor.h"
#include "asPreprocessor.h"
#include "asProcessor.h"
#include "asProcessorScore.h"
#include "asResultsDates.h"
#include "asResultsParametersArray.h"
#include "asResultsScores.h"
#include "asResultsScoresMap.h"
#include "asResultsTotalScore.h"
#include "asResultsValues.h"
#include "asScore.h"
#include "asTimeArray.h"

class asMethodCalibrator : public asMethodStandard {
  public:
    asMethodCalibrator();

    ~asMethodCalibrator() override;

    bool PreloadDataOnly(asParametersScoring* params);

    bool GetAnalogsDates(asResultsDates& results, asParametersScoring* params, int iStep, bool& containsNaNs);

    bool GetAnalogsSubDates(asResultsDates& results, asParametersScoring* params, asResultsDates& anaDates, int iStep,
                            bool& containsNaNs);

    bool GetAnalogsValues(asResultsValues& results, asParametersScoring* params, asResultsDates& anaDates, int iStep);

    bool GetAnalogsScores(asResultsScores& results, asParametersScoring* params, asResultsValues& anaValues, int iStep);

    bool GetAnalogsTotalScore(asResultsTotalScore& results, asParametersScoring* params, asResultsScores& anaScores,
                              int iStep);

    bool SubProcessAnalogsNumber(asParametersCalibration& params, asResultsDates& anaDatesPrevious, int iStep = 0);

    virtual void ClearAll();

    virtual void ClearTemp();

    bool PushBackBestTemp();

    void RemoveNaNsInTemp();

    void KeepBestTemp();

    void PushBackFirstTemp();

    void KeepFirstTemp();

    virtual bool SortScoresAndParametersTemp();

    bool PushBackInTempIfBetter(asParametersCalibration& params, asResultsTotalScore& scoreFinal);

    bool KeepIfBetter(asParametersCalibration& params, asResultsTotalScore& scoreFinal);

    virtual bool SetBestParameters(asResultsParametersArray& results);

    wxString GetStationIdsList(vi& stationIds) const;

    bool Manager() override;

    bool SaveDetails(asParametersCalibration* params);

    virtual bool Validate(asParametersCalibration* params);

    void SetScoreOrder(Order val) {
        m_scoreOrder = val;
    }

    vf GetScoreClimatology() const {
        return m_scoreClimatology;
    }

    void SetScoreClimatology(vf val) {
        m_scoreClimatology = std::move(val);
    }

    void SetPredictandStationIds(vi val) {
        m_predictandStationIds = std::move(val);
    }

  protected:
    struct ParamExploration {
        double xMinStart;
        double xMinEnd;
        int xPtsNbStart;
        int xPtsNbEnd;
        int xPtsNbIter;
        double yMinStart;
        double yMinEnd;
        int yPtsNbIter;
        int yPtsNbStart;
        int yPtsNbEnd;
    };
    vi m_predictandStationIds;
    vf m_scoresCalib;
    vf m_scoresCalibTemp;
    Order m_scoreOrder;
    float m_scoreValid;
    vf m_scoreClimatology;
    vector<asParametersCalibration> m_parameters;
    vector<asParametersCalibration> m_parametersTemp;
    asParametersCalibration m_originalParams;
    bool m_validationMode;
    bool m_useMiniBatches;
    int m_miniBatchStart;
    int m_miniBatchEnd;

    virtual bool Calibrate(asParametersCalibration& params) = 0;

    va1f GetClimatologyData(asParametersScoring* params);

    double GetEffectiveArchiveDataStart(asParameters* params) const override;

    double GetEffectiveArchiveDataEnd(asParameters* params) const override;

    double GetTimeStartCalibration(asParametersScoring* params) const;

    double GetTimeEndCalibration(asParametersScoring* params) const;

  private:
    void LoadScoreOrder(asParametersCalibration& params);
};

#endif
