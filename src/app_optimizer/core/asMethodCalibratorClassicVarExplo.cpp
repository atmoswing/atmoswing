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

#include "asMethodCalibratorClassicVarExplo.h"

asMethodCalibratorClassicVarExplo::asMethodCalibratorClassicVarExplo()
    : asMethodCalibratorClassic() {}

asMethodCalibratorClassicVarExplo::~asMethodCalibratorClassicVarExplo() = default;

bool asMethodCalibratorClassicVarExplo::Calibrate(asParametersCalibration& params) {
    int iStep;
    wxFileConfig::Get()->Read("/VariablesExplo/Step", &iStep, params.GetStepsNb() - 1);

    wxLogMessage(_("Processing variables exploration for step %d"), iStep);
    wxLogMessage(
        _("Processing %d variables, %d hours, %d levels, %d criteria."),
        (int)params.GetPredictorDataIdVector(iStep, 0).size(), (int)params.GetPredictorHourVector(iStep, 0).size(),
        (int)params.GetPredictorLevelVector(iStep, 0).size(), (int)params.GetPredictorCriteriaVector(iStep, 0).size());

    if (iStep >= params.GetStepsNb()) {
        wxLogError(_("The given step number for variables exploration is above available steps."));
        return false;
    }

    for (int iPtor = 0; iPtor < params.GetPredictorsNb(iStep); iPtor++) {
        if (params.NeedsPreprocessing(iStep, iPtor)) {
            wxLogError(_("Calibration method not implemented to work with preprocessed data."));
            return false;
        } else {
            vwxs vPredictorDataId = params.GetPredictorDataIdVector(iStep, iPtor);

            for (const auto& dataId : vPredictorDataId) {
                params.SetPredictorDataId(iStep, iPtor, dataId);

                vd vPredictorHours = params.GetPredictorHourVector(iStep, iPtor);

                for (double hour : vPredictorHours) {
                    params.SetPredictorHour(iStep, iPtor, hour);

                    vf vPredictorLevels = params.GetPredictorLevelVector(iStep, iPtor);

                    for (float level : vPredictorLevels) {
                        params.SetPredictorLevel(iStep, iPtor, level);

                        vwxs vPredictorCriteria = params.GetPredictorCriteriaVector(iStep, iPtor);

                        for (const auto& criteria : vPredictorCriteria) {
                            params.SetPredictorCriteria(iStep, iPtor, criteria);

                            vf slctPredictorLevels;
                            slctPredictorLevels.push_back(level);
                            params.SetPreloadLevels(iStep, iPtor, slctPredictorLevels);

                            vd slctPreloadHours;
                            slctPreloadHours.push_back(hour);
                            params.SetPreloadHours(iStep, iPtor, slctPreloadHours);

                            m_originalParams = params;

                            if (!asMethodCalibratorClassic::Calibrate(params)) return false;

                            params = m_originalParams;

                            ClearAll();
                        }
                    }
                }
            }
        }
    }

    return true;
}
