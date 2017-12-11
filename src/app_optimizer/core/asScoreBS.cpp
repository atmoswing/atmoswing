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

#include "asScoreBS.h"

asScoreBS::asScoreBS()
        : asScore(asScore::BS, _("Brier score"), _("Brier score"), Asc, 0, NaNf)
{
}

asScoreBS::~asScoreBS()
{
    //dtor
}

float asScoreBS::Assess(float ObservedVal, const a1f &ForcastVals, int nbElements) const
{
    wxASSERT(ForcastVals.size() > 1);
    wxASSERT(nbElements > 0);
    wxASSERT(!asTools::IsNaN(m_threshold));

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
        wxLogWarning(_("Only NaNs as inputs in the Brier score processing function."));
        return NaNf;
    } else if (nbPredict <= 2) {
        wxLogWarning(_("Not enough elements to process the Brier score."));
        return NaNf;
    }

    // Sort the forcast array
    asTools::SortArray(&x[0], &x[nbPredict - 1], Asc);

    // Cumulative frequency
    a1f F = asTools::GetCumulativeFrequency(nbPredict);

    // Search probability
    float probaOccurrence;
    if (m_threshold < x[0]) {
        probaOccurrence = 1;
    } else if (m_threshold > x[nbPredict - 1]) {
        probaOccurrence = 0;
    } else {
        int ind = asTools::SortedArraySearchFloor(&x[0], &x[nbPredict - 1], m_threshold);
        if (ind < 0) {
            wxLogError(_("Error processing BS score."));
            return NaNf;
        }
        while (x[ind] <= m_threshold) {
            ind++;
        }

        if (m_threshold > x[ind - 1]) {
            probaOccurrence = F(ind - 1) + (F(ind) - F(ind - 1)) * (m_threshold - x(ind - 1)) / (x(ind) - x(ind - 1));
        } else {
            probaOccurrence = F[ind - 1];
        }
    }

    float probaObservedVal = 0;
    if (ObservedVal >= m_threshold) {
        probaObservedVal = 1;
    }

    return (probaOccurrence - probaObservedVal) * (probaOccurrence - probaObservedVal);
}

bool asScoreBS::ProcessScoreClimatology(const a1f &refVals, const a1f &climatologyData)
{
    return true;
}