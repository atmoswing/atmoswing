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
 */

#ifndef ASPREPROCESSOR_H
#define ASPREPROCESSOR_H

#include <asIncludes.h>

class asDataPredictor;
class asDataPredictorArchive;
class asDataPredictorRealtime;


class asPreprocessor: public wxObject
{
public:

    static bool Preprocess(std::vector < asDataPredictorArchive* > predictors, const wxString& method, asDataPredictor *result);
    static bool Preprocess(std::vector < asDataPredictorRealtime* > predictors, const wxString& method, asDataPredictor *result);
    static bool Preprocess(std::vector < asDataPredictor* > predictors, const wxString& method, asDataPredictor *result);

    /** Preprocess gradients
    * \param predictor The 3D matrix of data with the corresponding dates
    * \return True in case of success
    */
    static bool PreprocessGradients(std::vector < asDataPredictor* > predictors, asDataPredictor *result);

    static bool PreprocessDifference(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessMultiplication(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessFormerHumidityIndex(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessMergeByHalfAndMultiply(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessHumidityFlux(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessWindSpeed(std::vector < asDataPredictor* > predictors, asDataPredictor *result);

protected:
private:
};

#endif // ASPREPROCESSOR_H
