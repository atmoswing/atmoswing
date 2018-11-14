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

#ifndef ASRESULTSSCORES_H
#define ASRESULTSSCORES_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsScores
        : public asResults
{
public:
    asResultsScores();

    virtual ~asResultsScores();

    void Init(asParametersScoring *params);

    a1f &GetTargetDates()
    {
        return m_targetDates;
    }

    void SetTargetDates(a1d &refDates)
    {
        m_targetDates.resize(refDates.rows());
        for (int i = 0; i < refDates.size(); i++) {
            m_targetDates[i] = (float) refDates[i];
            wxASSERT_MSG(m_targetDates[i] > 1, _("The target time array has unconsistent values"));
        }
    }

    void SetTargetDates(a1f &refDates)
    {
        m_targetDates.resize(refDates.rows());
        m_targetDates = refDates;
    }

    a1f &GetScores()
    {
        return m_scores;
    }

    a2f &GetScores2DArray()
    {
        return m_scores2DArray;
    }

    void SetScores(a1d &scores)
    {
        m_scores.resize(scores.rows());
        for (int i = 0; i < scores.size(); i++) {
            m_scores[i] = (float) scores[i];
        }
    }

    void SetScores(a1f &scores)
    {
        m_scores.resize(scores.rows());
        m_scores = scores;
    }

    void SetScores2DArray(a2f &scores)
    {
        m_scores2DArray.resize(scores.rows(), scores.cols());
        m_scores2DArray = scores;
    }

    bool Save();

    bool Load();

protected:
    void BuildFileName();

private:
    a1f m_targetDates;
    a1f m_scores;
    a2f m_scores2DArray;
};

#endif // ASRESULTSSCORES_H
