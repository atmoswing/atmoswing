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

#include "asScoreContingencyTable.h"

asScoreContingencyTable::asScoreContingencyTable()
        : asScore()
{
    m_score = asScore::ContingencyTable;
    m_name = _("Contingency table");
    m_fullName = _("Contingency table class");
    m_scaleBest = NaNf;
    m_scaleWorst = NaNf;
}

asScoreContingencyTable::~asScoreContingencyTable()
{
    //dtor
}

float asScoreContingencyTable::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asTools::IsNaN(m_threshold));
    wxASSERT(!asTools::IsNaN(m_quantile));
    wxASSERT(m_quantile > 0);
    wxASSERT(m_quantile < 1);

    // Check inputs
    if (!CheckObservedValue(ObservedVal)) {
        return NaNf;
    }
    if (!CheckVectorLength( ForcastVals, nbElements)) {
        wxLogWarning(_("Problems in a vector length."));
        return NaNf;
    }

    // Create the container to sort the data
    a1f x(nbElements);

    // Remove the NaNs and copy content
    int nbPredict = CleanNans(ForcastVals, x, nbElements);
    if (nbPredict == asNOT_FOUND) {
        wxLogWarning(_("Only NaNs as inputs in the Contingency table processing function."));
        return NaNf;
    } else if (nbPredict <= 2) {
        wxLogWarning(_("Not enough elements to process the Contingency table."));
        return NaNf;
    }

    a1f cleanValues = x.head(nbPredict);
    float score = NaNf;

    // Get value for quantile
    float xQuantile = asTools::GetValueForQuantile(cleanValues, m_quantile);

    // Predicted and observed
    if (xQuantile >= m_threshold && ObservedVal >= m_threshold) {
        score = 1;
    }
        // Predicted but not observed
    else if (xQuantile >= m_threshold && ObservedVal < m_threshold) {
        score = 2;
    }
        // Not predicted but observed
    else if (xQuantile < m_threshold && ObservedVal >= m_threshold) {
        score = 3;
    }
        // Not predicted and not observed
    else if (xQuantile < m_threshold && ObservedVal < m_threshold) {
        score = 4;
    }

    return score;
}

bool asScoreContingencyTable::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    return true;
}
