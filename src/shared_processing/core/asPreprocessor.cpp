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

#include "asPreprocessor.h"

#include <asDataPredictor.h>
#include <asDataPredictorArchive.h>

#ifndef MINIMAL_LINKS

#include <asDataPredictorRealtime.h>

#endif

#include <asThreadPreprocessorGradients.h>


bool asPreprocessor::Preprocess(std::vector<asDataPredictorArchive *> predictors, const wxString &method,
                                asDataPredictor *result)
{
    std::vector<asDataPredictor *> ptorsPredictors(predictors.size());

    for (unsigned int i = 0; i < predictors.size(); i++) {
        ptorsPredictors[i] = predictors[i];
    }

    return Preprocess(ptorsPredictors, method, result);
}

#ifndef MINIMAL_LINKS

bool asPreprocessor::Preprocess(std::vector<asDataPredictorRealtime *> predictors, const wxString &method,
                                asDataPredictor *result)
{
    std::vector<asDataPredictor *> ptorsPredictors(predictors.size());

    for (unsigned int i = 0; i < predictors.size(); i++) {
        ptorsPredictors[i] = predictors[i];
    }

    return Preprocess(ptorsPredictors, method, result);
}

#endif

bool asPreprocessor::Preprocess(std::vector<asDataPredictor *> predictors, const wxString &method,
                                asDataPredictor *result)
{
    wxASSERT(result);

    result->SetPreprocessMethod(method);

    if (method.IsSameAs("Gradients")) {
        return PreprocessGradients(predictors, result);
    } else if (method.IsSameAs("Difference")) {
        return PreprocessDifference(predictors, result);
    } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply")) {
        return PreprocessMultiplication(predictors, result);
    } else if (method.IsSameAs("HumidityIndex")) {
        return PreprocessMultiplication(predictors, result);
    } else if (method.IsSameAs("HumidityFlux")) {
        return PreprocessHumidityFlux(predictors, result);
    } else if (method.IsSameAs("FormerHumidityIndex")) {
        return PreprocessFormerHumidityIndex(predictors, result);
    } else if (method.IsSameAs("MergeByHalfAndMultiply")) {
        return PreprocessMergeByHalfAndMultiply(predictors, result);
    } else if (method.IsSameAs("WindSpeed")) {
        return PreprocessWindSpeed(predictors, result);
    } else {
        asLogError(_("The preprocessing method was not correctly defined."));
        return false;
    }
}

bool asPreprocessor::PreprocessGradients(std::vector<asDataPredictor *> predictors, asDataPredictor *result)
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool allowMultithreading;
    pConfig->Read("/Processing/AllowMultithreading", &allowMultithreading, true);
    ThreadsManager().CritSectionConfig().Leave();

    // Only one predictor
    wxASSERT(predictors.size() > 0);
    wxASSERT(predictors.size() == 1);
    if (predictors.size() != 1) {
        asLogError(_("The number of predictors must be equal to 1 in asPreprocessor::PreprocessGradients"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    int rowsNb = predictors[0]->GetLatPtsnb();
    int colsNb = predictors[0]->GetLonPtsnb();
    int timeSize = predictors[0]->GetTimeSize();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);

    // Create container
    VArray2DFloat gradients(timeSize);
    gradients.reserve(timeSize * 2 * rowsNb * colsNb);

    Array2DFloat tmpgrad = Array2DFloat::Zero(2 * rowsNb, colsNb); // Needs to be 0-filled for further simplification.

    /*
    Illustration of the data arrangement
        x = data
        o = 0

        xxxxxxxxxxx
        xxxxxxxxxxx
        xxxxxxxxxxx
        ooooooooooo____
        xxxxxxxxxxo
        xxxxxxxxxxo
        xxxxxxxxxxo
        xxxxxxxxxxo
    */

    for (int i_time = 0; i_time < timeSize; i_time++) {
        // Vertical gradients
        tmpgrad.block(0, 0, rowsNb - 1, colsNb) = predictors[0]->GetData()[i_time].block(1, 0, rowsNb - 1, colsNb) -
                                                  predictors[0]->GetData()[i_time].block(0, 0, rowsNb - 1, colsNb);

        // Horizontal gradients
        tmpgrad.block(rowsNb, 0, rowsNb, colsNb - 1) =
                predictors[0]->GetData()[i_time].block(0, 1, rowsNb, colsNb - 1) -
                predictors[0]->GetData()[i_time].block(0, 0, rowsNb, colsNb - 1);

        if (asTools::HasNaN(tmpgrad)) {
            // std::cout << tmpgrad << std::endl;
            // std::cout << "\n" << std::endl;
            // std::cout << predictors[0]->GetData()[i_time] << std::endl;

            asLogError(_("NaN found during gradients preprocessing !"));
            return false;
        }

        gradients[i_time] = tmpgrad;
    }

    // Overwrite the data in the predictor object
    result->SetData(gradients);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessDifference(std::vector<asDataPredictor *> predictors, asDataPredictor *result)
{
    // More than one predictor
    if (predictors.size() != 2) {
        asLogError(_("The number of predictors must be equal to 2 in asPreprocessor::PreprocessDifference"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    wxASSERT(predictors[1]);
    int rowsNb = predictors[0]->GetLatPtsnb();
    int colsNb = predictors[0]->GetLonPtsnb();
    int timeSize = predictors[0]->GetTimeSize();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);

    // Create container
    VArray2DFloat resdiff(timeSize, Array2DFloat::Constant(rowsNb, colsNb, 1));

    for (int i_time = 0; i_time < timeSize; i_time++) {
        resdiff[i_time].fill(1);

        for (int i_row = 0; i_row < rowsNb; i_row++) {
            for (int i_col = 0; i_col < colsNb; i_col++) {
                resdiff[i_time](i_row, i_col) =
                        predictors[0]->GetData()[i_time](i_row, i_col) - predictors[1]->GetData()[i_time](i_row, i_col);
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(resdiff);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessMultiplication(std::vector<asDataPredictor *> predictors, asDataPredictor *result)
{
    // More than one predictor
    if (predictors.size() < 2) {
        asLogError(_("The number of predictors must be superior to 1 in asPreprocessor::PreprocessMultiplication"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    int rowsNb = predictors[0]->GetLatPtsnb();
    int colsNb = predictors[0]->GetLonPtsnb();
    int timeSize = predictors[0]->GetTimeSize();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);

    // Create container
    VArray2DFloat multi(timeSize, Array2DFloat::Constant(rowsNb, colsNb, 1));

    for (int i_time = 0; i_time < timeSize; i_time++) {
        multi[i_time].fill(1);

        for (int i_row = 0; i_row < rowsNb; i_row++) {
            for (int i_col = 0; i_col < colsNb; i_col++) {
                for (unsigned int i_dat = 0; i_dat < predictors.size(); i_dat++) {
                    wxASSERT(predictors[i_dat]);
                    multi[i_time](i_row, i_col) *= predictors[i_dat]->GetData()[i_time](i_row, i_col);
                }
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessFormerHumidityIndex(std::vector<asDataPredictor *> predictors, asDataPredictor *result)
{
    // More than one predictor
    int inputSize = predictors.size();
    if (inputSize != 4) {
        asLogError(_("The number of predictors must be equal to 4 in asPreprocessor::PreprocessFormerHumidityIndex"));
        return false;
    }

    // Merge
    wxASSERT(predictors[0]);
    VVArray2DFloat copyData = VVArray2DFloat(inputSize / 2);
    copyData.reserve(
            2 * predictors[0]->GetLatPtsnb() * predictors[0]->GetLonPtsnb() * predictors[0]->GetTimeSize() * inputSize);
    int counter = 0;
#ifdef _DEBUG
    int prevTimeSize = 0;
#endif // _DEBUG

    for (unsigned int i_dat = 0; i_dat < predictors.size(); i_dat += 2) {
        wxASSERT(predictors[i_dat]);
        wxASSERT(predictors[i_dat + 1]);

        // Get sizes
        int rowsNb1 = predictors[i_dat]->GetLatPtsnb();
        int colsNb1 = predictors[i_dat]->GetLonPtsnb();
        int rowsNb2 = predictors[i_dat + 1]->GetLatPtsnb();
        int colsNb2 = predictors[i_dat + 1]->GetLonPtsnb();
        int timeSize = predictors[i_dat]->GetTimeSize();

#ifdef _DEBUG
        if (i_dat > 0) {
            wxASSERT(prevTimeSize == timeSize);
        }
        prevTimeSize = timeSize;
#endif // _DEBUG

        wxASSERT(rowsNb1 > 0);
        wxASSERT(colsNb1 > 0);
        wxASSERT(rowsNb2 > 0);
        wxASSERT(colsNb2 > 0);
        wxASSERT(timeSize > 0);

        bool putBelow;
        int rowsNew, colsNew;
        if (colsNb1 == colsNb2) {
            colsNew = colsNb1;
            rowsNew = rowsNb1 + rowsNb2;
            putBelow = true;
        } else if (rowsNb1 == rowsNb2) {
            rowsNew = rowsNb1;
            colsNew = colsNb1 + colsNb2;
            putBelow = false;
        } else {
            asThrowException(_("The predictors sizes make them impossible to merge."));
        }

        Array2DFloat tmp(rowsNew, colsNew);

        for (int i_time = 0; i_time < timeSize; i_time++) {
            tmp.topLeftCorner(rowsNb1, colsNb1) = predictors[i_dat]->GetData()[i_time];

            if (putBelow) {
                tmp.block(rowsNb1, 0, rowsNb2, colsNb2) = predictors[i_dat + 1]->GetData()[i_time];
            } else {
                tmp.block(0, colsNb1, rowsNb2, colsNb2) = predictors[i_dat + 1]->GetData()[i_time];
            }

            copyData[counter].push_back(tmp);
        }

        counter++;
    }

    // Get sizes
    int rowsNb = copyData[0][0].rows();
    int colsNb = copyData[0][0].cols();
    int timeSize = copyData[0].size();

    wxASSERT(rowsNb > 0);
    wxASSERT(colsNb > 0);
    wxASSERT(timeSize > 0);

    // Create container
    VArray2DFloat multi(timeSize, Array2DFloat::Constant(rowsNb, colsNb, 1));

    for (int i_time = 0; i_time < timeSize; i_time++) {
        multi[i_time].fill(1);

        for (int i_row = 0; i_row < rowsNb; i_row++) {
            for (int i_col = 0; i_col < colsNb; i_col++) {
                for (unsigned int i_dat = 0; i_dat < copyData.size(); i_dat++) {
                    multi[i_time](i_row, i_col) *= copyData[i_dat][i_time](i_row, i_col);
                }
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessMergeByHalfAndMultiply(std::vector<asDataPredictor *> predictors,
                                                      asDataPredictor *result)
{
    // More than one predictor
    int inputSize = predictors.size();
    int factorSize = inputSize / 2;
    if (inputSize < 2) {
        asLogError(
                _("The number of predictors must be superior to 2 in asPreprocessor::PreprocessMergeByHalfAndMultiply"));
        return false;
    }
    if (inputSize % 2 != 0) {
        asLogError(
                _("The number of predictors must be dividable by 2 in asPreprocessor::PreprocessMergeByHalfAndMultiply"));
        return false;
    }

    // Handle sizes
    wxASSERT(predictors[0]);
    int originalRowsNb = predictors[0]->GetLatPtsnb();
    int originalColsNb = predictors[0]->GetLonPtsnb();
    int timeSize = predictors[0]->GetTimeSize();
    wxASSERT(originalRowsNb > 0);
    wxASSERT(originalColsNb > 0);
    wxASSERT(timeSize > 0);

    int newRowsNb = originalRowsNb * factorSize;
    int newColsNb = originalColsNb;

    // Initialize
    wxASSERT(predictors[0]);
    VVArray2DFloat copyData = VVArray2DFloat(2, VArray2DFloat(timeSize, Array2DFloat(newRowsNb, newColsNb)));

    // Merge
    for (unsigned int i_half = 0; i_half < 2; i_half++) {
        for (int i_time = 0; i_time < timeSize; i_time++) {
            for (int i_dat = 0; i_dat < inputSize / 2; i_dat++) {
                int i_curr = i_half * inputSize / 2 + i_dat;
                wxASSERT(predictors[i_curr]);
                wxASSERT(predictors[i_curr]->GetLatPtsnb() == originalRowsNb);
                wxASSERT(predictors[i_curr]->GetLonPtsnb() == originalColsNb);
                wxASSERT(predictors[i_curr]->GetTimeSize() == timeSize);

                copyData[i_half][i_time].block(i_dat * originalRowsNb, 0, originalRowsNb,
                                               originalColsNb) = predictors[i_curr]->GetData()[i_time];
            }
        }
    }

    // Create container
    VArray2DFloat multi(timeSize, Array2DFloat::Zero(newRowsNb, newColsNb));

    for (int i_time = 0; i_time < timeSize; i_time++) {
        for (int i_row = 0; i_row < newRowsNb; i_row++) {
            for (int i_col = 0; i_col < newColsNb; i_col++) {
                multi[i_time](i_row, i_col) = copyData[0][i_time](i_row, i_col) * copyData[1][i_time](i_row, i_col);
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(false);

    return true;
}

bool asPreprocessor::PreprocessHumidityFlux(std::vector<asDataPredictor *> predictors, asDataPredictor *result)
{
    // More than one predictor
    int inputSize = predictors.size();
    if (inputSize != 4) {
        asLogError(_("The number of predictors must be equal to 4 in asPreprocessor::PreprocessHumidityFlux"));
        return false;
    }
    wxASSERT(predictors[0]);

#ifdef _DEBUG
    for (unsigned int i_dat = 0; i_dat < predictors.size() - 1; i_dat++) {
        wxASSERT(predictors[i_dat]->GetData()[0].rows() == predictors[i_dat + 1]->GetData()[0].rows());
        wxASSERT(predictors[i_dat]->GetData()[0].cols() == predictors[i_dat + 1]->GetData()[0].cols());
        wxASSERT(predictors[i_dat]->GetData().size() == predictors[i_dat + 1]->GetData().size());
    }
#endif

    // Get sizes
    int rowsNb = predictors[0]->GetData()[0].rows();
    int colsNb = predictors[0]->GetData()[0].cols();
    int timeSize = predictors[0]->GetData().size();

    wxASSERT(rowsNb > 0);
    wxASSERT(colsNb > 0);
    wxASSERT(timeSize > 0);

    // Create container
    VArray2DFloat multi(timeSize, Array2DFloat::Constant(rowsNb, colsNb, 1));

    float wind;

    for (int i_time = 0; i_time < timeSize; i_time++) {
        multi[i_time].fill(1);

        for (int i_row = 0; i_row < rowsNb; i_row++) {
            for (int i_col = 0; i_col < colsNb; i_col++) {
                wind = sqrt(predictors[0]->GetData()[i_time](i_row, i_col) *
                            predictors[0]->GetData()[i_time](i_row, i_col) +
                            predictors[1]->GetData()[i_time](i_row, i_col) *
                            predictors[1]->GetData()[i_time](i_row, i_col));
                multi[i_time](i_row, i_col) = wind * predictors[2]->GetData()[i_time](i_row, i_col) *
                                              predictors[3]->GetData()[i_time](i_row, i_col);
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessWindSpeed(std::vector<asDataPredictor *> predictors, asDataPredictor *result)
{
    // More than one predictor
    int inputSize = predictors.size();
    if (inputSize != 2) {
        asLogError(_("The number of predictors must be equal to 2 in asPreprocessor::PreprocessWindSpeed"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    wxASSERT(predictors[1]);
    int rowsNb = predictors[0]->GetLatPtsnb();
    int colsNb = predictors[0]->GetLonPtsnb();
    int timeSize = predictors[0]->GetTimeSize();
    wxASSERT(rowsNb > 0);
    wxASSERT(colsNb > 0);
    wxASSERT(timeSize > 0);

    // Create container
    VArray2DFloat multi(timeSize, Array2DFloat::Constant(rowsNb, colsNb, 1));

    float wind;

    for (int i_time = 0; i_time < timeSize; i_time++) {
        multi[i_time].fill(1);

        for (int i_row = 0; i_row < rowsNb; i_row++) {
            for (int i_col = 0; i_col < colsNb; i_col++) {
                // Get wind value
                wind = sqrt(predictors[0]->GetData()[i_time](i_row, i_col) *
                            predictors[0]->GetData()[i_time](i_row, i_col) +
                            predictors[1]->GetData()[i_time](i_row, i_col) *
                            predictors[1]->GetData()[i_time](i_row, i_col));
                multi[i_time](i_row, i_col) = wind;
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}
