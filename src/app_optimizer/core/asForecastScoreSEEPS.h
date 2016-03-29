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

#ifndef ASFORECASTSCORESEEPS_H
#define ASFORECASTSCORESEEPS_H

#include <asIncludes.h>
#include "asForecastScore.h"

class asForecastScoreSEEPS
        : public asForecastScore
{
public:
    asForecastScoreSEEPS();

    ~asForecastScoreSEEPS();

    float Assess(float ObservedVal, const Array1DFloat &ForcastVals, int NbElements);

    bool ProcessScoreClimatology(const Array1DFloat &refVals, const Array1DFloat &climatologyData);

    void SetP1(float val)
    {
        m_p1 = val;
    }

    void SetP3(float val)
    {
        m_p3 = val;
    }

    void SetThresNull(float val)
    {
        m_thresNull = val;
    }

    void SetThresHigh(float val)
    {
        m_thresHigh = val;
    }

protected:

private:
    float m_p1;
    float m_p3;
    float m_thresNull;
    float m_thresHigh;

};

#endif
