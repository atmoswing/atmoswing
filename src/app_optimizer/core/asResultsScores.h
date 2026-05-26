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

#ifndef AS_RESULTS_SCORES_H
#define AS_RESULTS_SCORES_H

#include "asHeadersBase.h"
#include "asResults.h"

class asParametersScoring;

class asResultsScores : public asResults {
  public:
    asResultsScores();

    virtual ~asResultsScores();

    void Init(asParametersScoring* params);

    a1f& GetTargetDates() {
        return _targetDates;
    }

    void SetTargetDates(a1d& refDates) {
        _targetDates.resize(refDates.rows());
        for (int i = 0; i < refDates.size(); i++) {
            _targetDates[i] = (float)refDates[i];
            wxASSERT_MSG(_targetDates[i] > 1, _("The target time array has unconsistent values"));
        }
    }

    void SetTargetDates(a1f& refDates) {
        _targetDates.resize(refDates.rows());
        _targetDates = refDates;
    }

    a1f& GetScores() {
        return _scores;
    }

    a2f& GetScores2DArray() {
        return _scores2DArray;
    }

    void SetScores(a1d& scores) {
        _scores.resize(scores.rows());
        for (int i = 0; i < scores.size(); i++) {
            _scores[i] = (float)scores[i];
        }
    }

    void SetScores(a1f& scores) {
        _scores.resize(scores.rows());
        _scores = scores;
    }

    void SetScores2DArray(a2f& scores) {
        _scores2DArray.resize(scores.rows(), scores.cols());
        _scores2DArray = scores;
    }

    bool Save();

    bool Load();

  protected:
    void BuildFileName();

  private:
    a1f _targetDates;
    a1f _scores;
    a2f _scores2DArray;
};

#endif
