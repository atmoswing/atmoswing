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

#include "asMethodDownscalerClassic.h"

#include "asParametersDownscaling.h"
#include "asResultsDates.h"
#include "asResultsValues.h"

asMethodDownscalerClassic::asMethodDownscalerClassic() : asMethodDownscaler() {}

asMethodDownscalerClassic::~asMethodDownscalerClassic() {}

bool asMethodDownscalerClassic::Downscale(asParametersDownscaling &params) {
    // Extract the stations IDs
    vvi stationsId = params.GetPredictandStationIdsVector();

    // Create a analogsdate object to save previous analogs dates selection.
    asResultsDates anaDatesPrevious;

    for (const auto &stationId : stationsId) {
        ClearAll();

        // Create results objects
        asResultsDates anaDates;
        asResultsValues anaValues;

        // Set the next station ID
        params.SetPredictandStationIds(stationId);

        // Process every step one after the other
        int stepsNb = params.GetStepsNb();
        for (int iStep = 0; iStep < stepsNb; iStep++) {
            bool containsNaNs = false;
            if (iStep == 0) {
                if (!GetAnalogsDates(anaDates, &params, iStep, containsNaNs)) return false;
            } else {
                if (!GetAnalogsSubDates(anaDates, &params, anaDatesPrevious, iStep, containsNaNs)) return false;
            }
            if (containsNaNs) {
                wxLogError(_("The dates selection contains NaNs"));
                return false;
            }
            if (!GetAnalogsValues(anaValues, &params, anaDates, iStep)) return false;

            // Keep the analogs dates of the best parameters set
            anaDatesPrevious = anaDates;
        }

        // Save
        SaveDetails(&params);
    }

    return true;
}
