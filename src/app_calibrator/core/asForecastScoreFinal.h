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
 
#ifndef ASFORECASTSCOREFINAL_H
#define ASFORECASTSCOREFINAL_H

#include <asIncludes.h>

#include <asTimeArray.h>

class asForecastScoreFinal: public wxObject
{
public:

    enum Period //!< Enumaration of forcast score combinations
    {
        Total, // total mean
        SpecificPeriod, // partial mean
        Summer, // partial mean on summer only
        Automn, // partial mean on fall only
        Winter, // partial mean on winter only
        Spring, // partial mean on spring only
    };

    asForecastScoreFinal(Period period);

    asForecastScoreFinal(const wxString& periodString);

    virtual ~asForecastScoreFinal();

    static asForecastScoreFinal* GetInstance(const wxString& scoreString, const wxString& periodString);

    virtual float Assess(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray) = 0;
    
    virtual float Assess(Array1DFloat &targetDates, Array2DFloat &forecastScores, asTimeArray &timeArray);

    virtual Array1DFloat AssessOnArray(Array1DFloat &targetDates, Array1DFloat &forecastScores, asTimeArray &timeArray);

    Period GetPeriod()
    {
        return m_period;
    }

    bool SingleValue()
    {
        return m_singleValue;
    }

    bool Has2DArrayArgument()
    {
        return m_has2DArrayArgument;
    }

    void SetRanksNb(int val)
    {
        m_ranksNb = val;
    }

protected:
    Period m_period;
    bool m_singleValue;
    bool m_has2DArrayArgument;
    int m_ranksNb;

private:


};

#endif // ASFORECASTSCOREFINAL_H
