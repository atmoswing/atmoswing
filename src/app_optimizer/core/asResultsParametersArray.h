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
 * Portions Copyright 2013-2014 Pascal Horton, Terranum.
 */

#ifndef AS_RESULTS_PARAMETERS_ARRAY_H
#define AS_RESULTS_PARAMETERS_ARRAY_H

#include "asIncludes.h"
#include "asParametersScoring.h"
#include "asResults.h"

class asResultsParametersArray : public asResults {
  public:
    asResultsParametersArray();

    ~asResultsParametersArray() override;

    void Init(const wxString& fileTag);

    void Clear();

    void StoreValues(asParametersScoring& params);

    void AddWithoutProcessingMedian(asParametersScoring& params, float scoreCalib);

    void Add(asParametersScoring& params, float scoreCalib, float scoreValid = NaNf);

    void Add(asParametersScoring& params, const a1f& scoreCalib, const a1f& scoreValid);

    void ProcessMedianScores();

    bool HasBeenAssessed(asParametersScoring& params, float& score);

    bool HasCloseOneBeenAssessed(asParametersScoring& params, float& score);

    bool Print(int fromIndex = 0) const;

    wxString PrintParams(int iParam) const;

    int GetCount() const {
        return int(m_parameters.size());
    }

    float GetMedianScore() const {
        return m_medianScore;
    }

  protected:
    void BuildFileName(const wxString& fileTag);

  private:
    std::vector<asParametersScoring::VectorParamsStep> m_parameters;
    asParametersScoring::ParamsScore m_scores;
    vvi m_predictandStationIds;
    vi m_analogsIntervalDays;
    int m_analogsExcludeDays;
    vf m_scoresCalib;
    vf m_scoresValid;
    va1f m_scoresCalibForScoreOnArray;
    va1f m_scoresValidForScoreOnArray;
    float m_medianScore;
};

#endif
