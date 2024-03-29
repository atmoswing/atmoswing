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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#ifndef AS_METHOD_DOWNSCALER_H
#define AS_METHOD_DOWNSCALER_H

#include <utility>

#include "asMethodStandard.h"

class asResultsDates;
class asParametersDownscaling;
class asResultsValues;
class asPredictorProj;
class asCriteria;

class asMethodDownscaler : public asMethodStandard {
  public:
    asMethodDownscaler();

    ~asMethodDownscaler() override;

    bool GetAnalogsDates(asResultsDates& results, asParametersDownscaling* params, int iStep, bool& containsNaNs);

    bool GetAnalogsSubDates(asResultsDates& results, asParametersDownscaling* params, asResultsDates& anaDates,
                            int iStep, bool& containsNaNs);

    bool GetAnalogsValues(asResultsValues& results, asParametersDownscaling* params, asResultsDates& anaDates,
                          int iStep);

    void ClearAll();

    bool Manager() override;

    bool IsArchivePointerCopy(int iStep, int iPtor, int iPre) const {
        return m_preloadedArchivePointerCopy[iStep][iPtor][iPre];
    }

    bool IsProjectionPointerCopy(int iStep, int iPtor, int iPre) const {
        return m_preloadedProjectionPointerCopy[iStep][iPtor][iPre];
    }

    void SetPredictandStationIds(const vi& val) {
        m_predictandStationIds = val;
    }

    void SetPredictorProjectionDataDir(const wxString& val) {
        m_predictorProjectionDataDir = val;
    }

  protected:
    wxString m_predictorProjectionDataDir;
    vi m_predictandStationIds;
    vector<asParametersDownscaling> m_parameters;

    virtual bool Downscale(asParametersDownscaling& params) = 0;

    bool LoadProjectionData(vector<asPredictor*>& predictors, asParametersDownscaling* params, int iStep,
                            double timeStartData, double timeEndData);

    bool ExtractProjectionDataWithoutPreprocessing(vector<asPredictor*>& predictors, asParametersDownscaling* params,
                                                   int iStep, int iPtor, double timeStartData, double timeEndData);

    bool ExtractProjectionDataWithPreprocessing(vector<asPredictor*>& predictors, asParametersDownscaling* params,
                                                int iStep, int iPtor, double timeStartData, double timeEndData);

    bool Preprocess(vector<asPredictorProj*> predictors, const wxString& method, asPredictor* result);

    bool SaveDetails(asParametersDownscaling* params);

    void Cleanup(vector<asPredictorProj*> predictors);

    void Cleanup(vector<asPredictor*> predictors) override;

    void Cleanup(vector<asCriteria*> criteria) override;

  private:
    vector<vector<vector<vector<vector<asPredictor*> > > > > m_preloadedArchive;
    vector<vector<vector<vector<vector<asPredictorProj*> > > > > m_preloadedProjection;
    vector<vvb> m_preloadedArchivePointerCopy;
    vector<vvb> m_preloadedProjectionPointerCopy;

    double GetTimeStartDownscaling(asParametersDownscaling* params) const;

    double GetTimeEndDownscaling(asParametersDownscaling* params) const;

    double GetEffectiveArchiveDataStart(asParameters* params) const override;

    double GetEffectiveArchiveDataEnd(asParameters* params) const override;
};

#endif
