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

#ifndef ASRESULTSANALOGSFORECASTSCOREFINAL_H
#define ASRESULTSANALOGSFORECASTSCOREFINAL_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsAnalogsForecastScoreFinal
        : public asResults
{
public:
    asResultsAnalogsForecastScoreFinal();

    virtual ~asResultsAnalogsForecastScoreFinal();

    void Init(asParametersScoring &params);

    float GetForecastScore()
    {
        return m_forecastScore;
    }

    void SetForecastScore(float val)
    {
        m_forecastScore = val;
    }

    Array1DFloat GetForecastScoreArray()
    {
        return m_forecastScoreArray;
    }

    void SetForecastScore(Array1DFloat val)
    {
        m_forecastScoreArray = val;
        m_hasSingleValue = false;
    }

    void SetForecastScoreArray(Array1DFloat val)
    {
        m_forecastScoreArray = val;
        m_hasSingleValue = false;
    }

    bool Save(const wxString &AlternateFilePath = wxEmptyString);

    bool Load(const wxString &AlternateFilePath = wxEmptyString);

protected:

    void BuildFileName(asParametersScoring &params);

private:
    bool m_hasSingleValue;
    float m_forecastScore;
    Array1DFloat m_forecastScoreArray;
};

#endif // ASRESULTSANALOGSFORECASTSCOREFINAL_H
