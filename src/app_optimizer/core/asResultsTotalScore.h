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

#ifndef AS_RESULTS_SCORE_FINAL_H
#define AS_RESULTS_SCORE_FINAL_H

#include <asIncludes.h>
#include <asResults.h>

class asParametersScoring;

class asResultsTotalScore
        : public asResults
{
public:
    asResultsTotalScore();

    virtual ~asResultsTotalScore();

    void Init();

    float GetScore() const
    {
        return m_score;
    }

    void SetScore(float val)
    {
        m_score = val;
    }

    a1f GetScoreArray() const
    {
        return m_scoreArray;
    }

    void SetScore(a1f val)
    {
        m_scoreArray = val;
        m_hasSingleValue = false;
    }

    bool Save();

    bool Load();

protected:

    void BuildFileName();

private:
    bool m_hasSingleValue;
    float m_score;
    a1f m_scoreArray;
};

#endif
