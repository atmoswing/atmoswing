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
 
#include "asForecastScoreCRPSaccuracyAR.h"
#include "asForecastScoreCRPSAR.h"
#include "asForecastScoreCRPSsharpnessAR.h"

asForecastScoreCRPSaccuracyAR::asForecastScoreCRPSaccuracyAR()
:
asForecastScore()
{
    m_Score = asForecastScore::CRPSaccuracyAR;
    m_Name = _("CRPS Accuracy Approx Rectangle");
    m_FullName = _("Continuous Ranked Probability Score Accuracy approximation with the rectangle method");
    m_Order = Asc;
    m_ScaleBest = 0;
    m_ScaleWorst = NaNFloat;
}

asForecastScoreCRPSaccuracyAR::~asForecastScoreCRPSaccuracyAR()
{
    //dtor
}

float asForecastScoreCRPSaccuracyAR::Assess(float ObservedVal, const Array1DFloat &ForcastVals, int nbElements)
{
    wxASSERT(ForcastVals.size()>1);
    wxASSERT(nbElements>0);

    asForecastScoreCRPSAR scoreCRPSAR = asForecastScoreCRPSAR();
    float CRPS = scoreCRPSAR.Assess(ObservedVal, ForcastVals, nbElements);
    asForecastScoreCRPSsharpnessAR scoreCRPSsharpnessAR = asForecastScoreCRPSsharpnessAR();
    float CRPSsharpness = scoreCRPSsharpnessAR.Assess(ObservedVal, ForcastVals, nbElements);

    return CRPS-CRPSsharpness;
}

bool asForecastScoreCRPSaccuracyAR::ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData)
{
    return true;
}
