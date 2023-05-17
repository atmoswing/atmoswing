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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include "asPredictorOperCustomVigicruesIfs.h"

#include "asAreaGrid.h"
#include "asTimeArray.h"

asPredictorOperCustomVigicruesIfs::asPredictorOperCustomVigicruesIfs(const wxString& dataId)
    : asPredictorOperEcmwfIfs(dataId) {
    // Set the basic properties.
    m_datasetId = "Custom_Vigicrues_IFS";
    m_datasetName = "Integrated Forecasting System (IFS) grib files for the Vigicrues network";
    m_fStr.hasLevelDim = true;
    m_leadTimeStep = 6;
    m_runHourStart = 0;
    m_runUpdate = 12;
    m_percentMissingAllowed = 70;
    m_fileExtension = "grb";
}

wxString asPredictorOperCustomVigicruesIfs::GetFileName(const double date, const int) {
    wxString dateStr = asTime::GetStringTime(date, "YYYYMMDDhhmm");

    return asStrF("CEP_%s_%s.%s", m_dataId.Upper(), dateStr, m_fileExtension);
}