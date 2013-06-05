/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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

    enum Preprocesses //!< Enumaration of integrated preprocesses
    {
        Gradients, // for S1 (Teweles-Wobus)
        Multiplication // Multiplication
    };

    static bool Preprocess(std::vector < asDataPredictorArchive > predictors, const wxString& method, asDataPredictor *result);
    static bool Preprocess(std::vector < asDataPredictorRealtime > predictors, const wxString& method, asDataPredictor *result);
    static bool Preprocess(std::vector < asDataPredictor* > predictors, const wxString& method, asDataPredictor *result);

    /** Preprocess gradients
    * \param predictor The 3D matrix of data with the corresponding dates
    * \return True in case of success
    */
    static bool PreprocessGradients(std::vector < asDataPredictor* > predictors, asDataPredictor *result);

    static bool PreprocessDifference(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessMultiplication(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessMergeCouplesAndMultiply(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessMergeByHalfAndMultiply(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessHumidityFlux(std::vector < asDataPredictor* > predictors, asDataPredictor *result);
    static bool PreprocessWindSpeed(std::vector < asDataPredictor* > predictors, asDataPredictor *result);

protected:
private:
};

#endif // ASPREPROCESSOR_H
