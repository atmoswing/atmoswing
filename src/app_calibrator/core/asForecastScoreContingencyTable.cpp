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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#include "asForecastScoreContingencyTable.h"

asForecastScoreContingencyTable::asForecastScoreContingencyTable()
:
asForecastScore()
{
    m_Score = asForecastScore::ContingencyTable;
    m_Name = _("Contingency table");
    m_FullName = _("Contingency table class");
    m_Order = NoOrder;
    m_ScaleBest = NaNFloat;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreContingencyTable::~asForecastScoreContingencyTable()
{
    //dtor
}

float asForecastScoreContingencyTable::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);
    wxASSERT(!asTools::IsNaN(m_Threshold));
    wxASSERT(!asTools::IsNaN(m_Quantile));
    wxASSERT(m_Quantile>0);
    wxASSERT(m_Quantile<1);

    // Create the container to sort the data
    Array1DFloat x(nbElements);

    // Remove the NaNs and copy content
    int nbForecasts = CleanNans(ForcastVals, x, nbElements);
    if(nbForecasts==asNOT_FOUND)
    {
        asLogWarning(_("Only NaNs as inputs in the Contingency table processing function."));
        return NaNFloat;
    }
    else if(nbForecasts<=2)
    {
        asLogWarning(_("Not enough elements to process the Contingency table."));
        return NaNFloat;
    }

    Array1DFloat cleanValues = x.head(nbForecasts);
    float score = NaNFloat;

    // Get value for quantile
    float xQuantile = asTools::GetValueForQuantile(cleanValues, m_Quantile);

	// Forecasted and observed
	if (xQuantile>=m_Threshold && ObservedVal>=m_Threshold)
    {
        score = 1;
	}
	// Forecasted but not observed
	else if (xQuantile>=m_Threshold && ObservedVal<m_Threshold)
    {
        score = 2;
	}
	// Not forecasted but observed
	else if (xQuantile<m_Threshold && ObservedVal>=m_Threshold)
    {
        score = 3;
	}
	// Not forecasted and not observed
	else if (xQuantile<m_Threshold && ObservedVal<m_Threshold)
    {
        score = 4;
	}

    return score;
}

bool asForecastScoreContingencyTable::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
