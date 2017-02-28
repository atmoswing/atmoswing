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

#ifndef ASRESULTSANALOGSFORECASTSCORES_H
#define ASRESULTSANALOGSFORECASTSCORES_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsAnalogsForecastScores
        : public asResults
{
public:
    asResultsAnalogsForecastScores();

    virtual ~asResultsAnalogsForecastScores();

    void Init(asParametersScoring &params);

    Array1DFloat GetTargetDates()
    {
        return m_targetDates;
    }

    void SetTargetDates(Array1DDouble &refDates)
    {
        m_targetDates.resize(refDates.rows());
        for (int i = 0; i < refDates.size(); i++) {
            m_targetDates[i] = (float) refDates[i];
            wxASSERT_MSG(m_targetDates[i] > 1, _("The target time array has unconsistent values"));
        }
    }

    void SetTargetDates(Array1DFloat &refDates)
    {
        m_targetDates.resize(refDates.rows());
        m_targetDates = refDates;
    }

    Array1DFloat GetForecastScores()
    {
        return m_forecastScores;
    }

    Array2DFloat GetForecastScores2DArray()
    {
        return m_forecastScores2DArray;
    }

    void SetForecastScores(Array1DDouble &forecastScores)
    {
        m_forecastScores.resize(forecastScores.rows());
        for (int i = 0; i < forecastScores.size(); i++) {
            m_forecastScores[i] = (float) forecastScores[i];
        }
    }

    void SetForecastScores(Array1DFloat &forecastScores)
    {
        m_forecastScores.resize(forecastScores.rows());
        m_forecastScores = forecastScores;
    }

    void SetForecastScores2DArray(Array2DFloat &forecastScores)
    {
        m_forecastScores2DArray.resize(forecastScores.rows(), forecastScores.cols());
        m_forecastScores2DArray = forecastScores;
    }

    bool Save();

    bool Load();

protected:
    void BuildFileName();

private:
    Array1DFloat m_targetDates;
    Array1DFloat m_forecastScores;
    Array2DFloat m_forecastScores2DArray;
};

#endif // ASRESULTSANALOGSFORECASTSCORES_H
