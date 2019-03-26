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

#include "asPredictorProj.h"

#include "asTimeArray.h"
#include "asAreaCompGrid.h"
#include "asPredictorProjCmip5.h"
#include "asPredictorProjCordex.h"


asPredictorProj::asPredictorProj(const wxString &dataId, const wxString &model, const wxString &scenario)
        : asPredictor(dataId),
          m_model(model),
          m_scenario(scenario)
{

}

asPredictorProj *asPredictorProj::GetInstance(const wxString &datasetId, const wxString &model, const wxString &scenario,
                                              const wxString &dataId, const wxString &directory)
{
    asPredictorProj *predictor = nullptr;

    if (datasetId.IsSameAs("CMIP5", false)) {
        predictor = new asPredictorProjCmip5(dataId, model, scenario);
    } else if (datasetId.IsSameAs("CORDEX", false)) {
        predictor = new asPredictorProjCordex(dataId, model, scenario);
    } else {
        wxLogError(_("The requested dataset does not exist. Please correct the dataset Id."));
        return nullptr;
    }

    if (!directory.IsEmpty()) {
        predictor->SetDirectoryPath(directory);
    }

    if (!predictor->Init()) {
        wxLogError(_("The predictor did not initialize correctly."));
        return nullptr;
    }

    return predictor;
}