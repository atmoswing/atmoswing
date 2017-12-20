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

#ifndef ASMETHODDOWNSCALER_H
#define ASMETHODDOWNSCALER_H

#include <asMethodStandard.h>


class asResultsDates;
class asParametersDownscaling;
class asResultsValues;
class asPredictorScenario;
class asCriteria;


class asMethodDownscaler
        : public asMethodStandard
{
public:
    asMethodDownscaler();

    virtual ~asMethodDownscaler();

    bool GetAnalogsDates(asResultsDates &results, asParametersDownscaling *params, int iStep, bool &containsNaNs);

    bool GetAnalogsSubDates(asResultsDates &results, asParametersDownscaling *params, asResultsDates &anaDates, int iStep,
                            bool &containsNaNs);

    bool GetAnalogsValues(asResultsValues &results, asParametersDownscaling *params, asResultsDates &anaDates, int iStep);

    void ClearAll();

    bool Manager();

    bool IsArchivePointerCopy(int iStep, int iPtor, int iPre) const
    {
        return m_preloadedArchivePointerCopy[iStep][iPtor][iPre];
    }

    bool IsScenarioPointerCopy(int iStep, int iPtor, int iPre) const
    {
        return m_preloadedScenarioPointerCopy[iStep][iPtor][iPre];
    }

    void SetPredictandStationIds(vi val)
    {
        m_predictandStationIds = val;
    }

protected:
    wxString m_predictorScenarioDataDir;
    vi m_predictandStationIds;
    std::vector<asParametersDownscaling> m_parameters;

    virtual bool Downscale(asParametersDownscaling &params) = 0;

    bool LoadScenarioulationData(std::vector<asPredictor *> &predictors, asParametersDownscaling *params, int iStep,
                                 double timeStartData, double timeEndData);

    bool ExtractScenarioulationDataWithoutPreprocessing(std::vector<asPredictor *> &predictors,
                                                        asParametersDownscaling *params, int iStep, int iPtor,
                                                        double timeStartData, double timeEndData);

    bool ExtractScenarioulationDataWithPreprocessing(std::vector<asPredictor *> &predictors,
                                                     asParametersDownscaling *params, int iStep, int iPtor,
                                                     double timeStartData, double timeEndData);

    bool Preprocess(std::vector<asPredictorScenario *> predictors, const wxString &method, asPredictor *result);

    bool SaveDetails(asParametersDownscaling *params);

    void Cleanup(std::vector<asPredictorScenario *> predictorsPreprocess);

private:
    std::vector<std::vector<std::vector<std::vector<std::vector<asPredictorArch *> > > > > m_preloadedArchive;
    std::vector<std::vector<std::vector<std::vector<std::vector<asPredictorScenario *> > > > > m_preloadedScenario;
    std::vector<vvb> m_preloadedArchivePointerCopy;
    std::vector<vvb> m_preloadedScenarioPointerCopy;

    double GetTimeStartDownscaling(asParametersDownscaling *params) const;

    double GetTimeEndDownscaling(asParametersDownscaling *params) const;
};

#endif // ASMETHODDOWNSCALER_H
