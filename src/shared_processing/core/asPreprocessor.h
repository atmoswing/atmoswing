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

#ifndef AS_PREPROCESSOR_H
#define AS_PREPROCESSOR_H

#include "asIncludes.h"

class asPredictor;

class asPreprocessor : public wxObject {
  public:
    static bool Preprocess(const vector<asPredictor*>& predictors, const wxString& method, asPredictor* result);

    static bool PreprocessSimpleGradients(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessRealGradients(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessSimpleGradientsWithGaussianWeights(const vector<asPredictor*>& predictors,
                                                             asPredictor* result);

    static bool PreprocessRealGradientsWithGaussianWeights(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessSimpleCurvature(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessRealCurvature(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessSimpleCurvatureWithGaussianWeights(const vector<asPredictor*>& predictors,
                                                             asPredictor* result);

    static bool PreprocessRealCurvatureWithGaussianWeights(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessAddition(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessAverage(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessDifference(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessMultiplication(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessFormerHumidityIndex(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessMergeByHalfAndMultiply(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessHumidityFlux(const vector<asPredictor*>& predictors, asPredictor* result);

    static bool PreprocessWindSpeed(const vector<asPredictor*>& predictors, asPredictor* result);

    static void GetHorizontalDistances(const a1d& lonAxis, const a1d& latAxis, a2f& distXs, a2f& distYs);

  protected:
  private:
};

#endif
