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
        : asMethodCalibratorClassic()
{

}

asMethodCalibratorClassicVarExplo::~asMethodCalibratorClassicVarExplo()
{

}

bool asMethodCalibratorClassicVarExplo::Calibrate(asParametersCalibration &params)
{

    int i_step;
    wxFileConfig::Get()->Read("/Optimizer/VariablesExplo/Step", &i_step, params.GetStepsNb() - 1);

    wxLogMessage(_("Processing variables exploration for step %d"), i_step);
    wxLogMessage(_("Processing %d variables, %d hours, %d levels, %d criteria."),
                 (int) params.GetPredictorDataIdVector(i_step, 0).size(),
                 (int) params.GetPredictorTimeHoursVector(i_step, 0).size(),
                 (int) params.GetPredictorLevelVector(i_step, 0).size(),
                 (int) params.GetPredictorCriteriaVector(i_step, 0).size());

    if (i_step >= params.GetStepsNb()) {
        wxLogError(_("The given step number for variables exploration is above available steps."));
        return false;
    }

    for (int i_ptor = 0; i_ptor < params.GetPredictorsNb(i_step); i_ptor++) {
        if (params.NeedsPreprocessing(i_step, i_ptor)) {

            wxLogError(_("Calibration method not implemented to work with preprocessed data."));
            return false;
        } else {
            VectorString vPredictorDataId = params.GetPredictorDataIdVector(i_step, i_ptor);

            for (unsigned int i_predictordata = 0; i_predictordata < vPredictorDataId.size(); i_predictordata++) {
                params.SetPredictorDataId(i_step, i_ptor, vPredictorDataId[i_predictordata]);

                VectorDouble vPredictorTimeHours = params.GetPredictorTimeHoursVector(i_step, i_ptor);

                for (unsigned int i_predictortime = 0;
                     i_predictortime < vPredictorTimeHours.size(); i_predictortime++) {
                    params.SetPredictorTimeHours(i_step, i_ptor, vPredictorTimeHours[i_predictortime]);

                    VectorFloat vPredictorLevels = params.GetPredictorLevelVector(i_step, i_ptor);

                    for (unsigned int i_predictorlevel = 0;
                         i_predictorlevel < vPredictorLevels.size(); i_predictorlevel++) {
                        params.SetPredictorLevel(i_step, i_ptor, vPredictorLevels[i_predictorlevel]);

                        VectorString vPredictorCriteria = params.GetPredictorCriteriaVector(i_step, i_ptor);

                        for (unsigned int i_criteria = 0; i_criteria < vPredictorCriteria.size(); i_criteria++) {
                            params.SetPredictorCriteria(i_step, i_ptor, vPredictorCriteria[i_criteria]);

                            VectorFloat slctPredictorLevels;
                            slctPredictorLevels.push_back(vPredictorLevels[i_predictorlevel]);
                            params.SetPreloadLevels(i_step, i_ptor, slctPredictorLevels);

                            VectorDouble slctPreloadTimeHours;
                            slctPreloadTimeHours.push_back(vPredictorTimeHours[i_predictortime]);
                            params.SetPreloadTimeHours(i_step, i_ptor, slctPreloadTimeHours);

                            m_originalParams = params;

                            if (!asMethodCalibratorClassic::Calibrate(params))
                                return false;

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
