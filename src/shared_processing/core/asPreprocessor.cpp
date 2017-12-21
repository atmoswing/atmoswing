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

#include <asPredictor.h>

#include <asThreadPreprocessGradients.h>


bool asPreprocessor::Preprocess(std::vector<asPredictor *> predictors, const wxString &method, asPredictor *result)
{
    wxASSERT(result);

    result->SetPreprocessMethod(method);

    if (method.IsSameAs("Gradients")) {
        return PreprocessGradients(predictors, result);
    } else if (method.IsSameAs("Addition")) {
        return PreprocessAddition(predictors, result);
    } else if (method.IsSameAs("Average")) {
        return PreprocessAverage(predictors, result);
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
        wxLogError(_("The preprocessing method was not correctly defined."));
        return false;
    }
}

bool asPreprocessor::PreprocessGradients(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // Get the processing method
    ThreadsManager().CritSectionConfig().Enter();
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool allowMultithreading;
    pConfig->Read("/Processing/AllowMultithreading", &allowMultithreading, true);
    ThreadsManager().CritSectionConfig().Leave();

    // Only one predictor
    wxASSERT(!predictors.empty());
    wxASSERT(predictors.size() == 1);
    if (predictors.size() != 1) {
        wxLogError(_("The number of predictors must be equal to 1 in asPreprocessor::PreprocessGradients"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f gradients(timeSize);
    gradients.reserve(membersNb * timeSize * 2 * rowsNb * colsNb);

    a2f tmpgrad = a2f::Zero(2 * rowsNb, colsNb); // Needs to be 0-filled for further simplification.

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

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            // Vertical gradients
            tmpgrad.block(0, 0, rowsNb - 1, colsNb) = predictors[0]->GetData()[iTime][iMem].block(1, 0, rowsNb - 1, colsNb) -
                                                      predictors[0]->GetData()[iTime][iMem].block(0, 0, rowsNb - 1, colsNb);

            // Horizontal gradients
            tmpgrad.block(rowsNb, 0, rowsNb, colsNb - 1) =
                    predictors[0]->GetData()[iTime][iMem].block(0, 1, rowsNb, colsNb - 1) -
                    predictors[0]->GetData()[iTime][iMem].block(0, 0, rowsNb, colsNb - 1);

            if (asTools::HasNaN(tmpgrad)) {
                // std::cout << tmpgrad << std::endl;
                // std::cout << "\n" << std::endl;
                // std::cout << predictors[0]->GetData()[iTime] << std::endl;

                wxLogError(_("NaN found during gradients preprocessing !"));
                return false;
            }

            gradients[iTime].push_back(tmpgrad);
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(gradients);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessAddition(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    if (predictors.size() < 2) {
        wxLogError(_("The number of predictors must be superior to 1 in asPreprocessor::PreprocessAddition"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f addition(timeSize, va2f(membersNb, a2f::Zero(rowsNb, colsNb)));

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iPre = 0; iPre < predictors.size(); iPre++) {
                wxASSERT(predictors[iPre]);
                addition[iTime][iMem] += predictors[iPre]->GetData()[iTime][iMem];
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(addition);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessAverage(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    if (predictors.size() < 2) {
        wxLogError(_("The number of predictors must be superior to 1 in asPreprocessor::PreprocessAddition"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f average(timeSize, va2f(membersNb, a2f::Zero(rowsNb, colsNb)));

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iPre = 0; iPre < predictors.size(); iPre++) {
                wxASSERT(predictors[iPre]);
                average[iTime][iMem] += predictors[iPre]->GetData()[iTime][iMem];
            }
            average[iTime][iMem] /= float(predictors.size());
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(average);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessDifference(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    if (predictors.size() != 2) {
        wxLogError(_("The number of predictors must be equal to 2 in asPreprocessor::PreprocessDifference"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    wxASSERT(predictors[1]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f resdiff(timeSize, va2f(membersNb, a2f::Zero(rowsNb, colsNb)));

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            resdiff[iTime][iMem] = predictors[0]->GetData()[iTime][iMem] - predictors[1]->GetData()[iTime][iMem];
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(resdiff);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessMultiplication(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    if (predictors.size() < 2) {
        wxLogError(_("The number of predictors must be superior to 1 in asPreprocessor::PreprocessMultiplication"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 1);
    wxASSERT(colsNb > 1);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f multi(timeSize, va2f(membersNb, a2f::Constant(rowsNb, colsNb, 1)));

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iPre = 0; iPre < predictors.size(); iPre++) {
                wxASSERT(predictors[iPre]);
                multi[iTime][iMem] *= predictors[iPre]->GetData()[iTime][iMem];
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(true);

    return true;
}

bool asPreprocessor::PreprocessFormerHumidityIndex(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    unsigned int inputSize = (unsigned int) predictors.size();
    if (inputSize != 4) {
        wxLogError(_("The number of predictors must be equal to 4 in asPreprocessor::PreprocessFormerHumidityIndex"));
        return false;
    }

    // Merge
    wxASSERT(predictors[0]);
    vvva2f copyData = vvva2f(inputSize / 2);
    copyData.reserve(2 * predictors[0]->GetLatPtsnb() * predictors[0]->GetLonPtsnb() * predictors[0]->GetTimeSize() *
                     predictors[0]->GetMembersNb() * inputSize);
    int counter = 0;

#ifdef _DEBUG
    int prevTimeSize = 0;
#endif // _DEBUG

    for (unsigned int iPre = 0; iPre < predictors.size(); iPre += 2) {
        wxASSERT(predictors[iPre]);
        wxASSERT(predictors[iPre + 1]);

        // Get sizes
        int rowsNb1 = predictors[iPre]->GetLatPtsnb();
        int colsNb1 = predictors[iPre]->GetLonPtsnb();
        int rowsNb2 = predictors[iPre + 1]->GetLatPtsnb();
        int colsNb2 = predictors[iPre + 1]->GetLonPtsnb();
        int timeSize = predictors[iPre]->GetTimeSize();
        int membersNb = predictors[iPre]->GetMembersNb();

#ifdef _DEBUG
        if (iPre > 0) {
            wxASSERT(prevTimeSize == timeSize);
        }
        prevTimeSize = timeSize;
#endif // _DEBUG

        wxASSERT(rowsNb1 > 0);
        wxASSERT(colsNb1 > 0);
        wxASSERT(rowsNb2 > 0);
        wxASSERT(colsNb2 > 0);
        wxASSERT(timeSize > 0);
        wxASSERT(membersNb > 0);

        bool putBelow = false;
        int rowsNew = 0, colsNew = 0;
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

        va2f tmp((unsigned long) membersNb, a2f(rowsNew, colsNew));

        for (int iTime = 0; iTime < timeSize; iTime++) {
            for (int iMem = 0; iMem < membersNb; iMem++) {
                tmp[iMem].topLeftCorner(rowsNb1, colsNb1) = predictors[iPre]->GetData()[iTime][iMem];

                if (putBelow) {
                    tmp[iMem].block(rowsNb1, 0, rowsNb2, colsNb2) = predictors[iPre + 1]->GetData()[iTime][iMem];
                } else {
                    tmp[iMem].block(0, colsNb1, rowsNb2, colsNb2) = predictors[iPre + 1]->GetData()[iTime][iMem];
                }

                copyData[counter].push_back(tmp);
            }
        }

        counter++;
    }

    // Get sizes
    unsigned int rowsNb = (unsigned int) copyData[0][0][0].rows();
    unsigned int colsNb = (unsigned int) copyData[0][0][0].cols();
    unsigned int timeSize = (unsigned int) copyData[0].size();
    unsigned int membersNb = (unsigned int) copyData[0][0].size();

    wxASSERT(rowsNb > 0);
    wxASSERT(colsNb > 0);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f multi(timeSize, va2f(membersNb, a2f::Constant(rowsNb, colsNb, 1)));

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iRow = 0; iRow < rowsNb; iRow++) {
                for (unsigned int iCol = 0; iCol < colsNb; iCol++) {
                    for (unsigned int iPre = 0; iPre < copyData.size(); iPre++) {
                        multi[iTime][iMem](iRow, iCol) *= copyData[iPre][iTime][iMem](iRow, iCol);
                    }
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

bool asPreprocessor::PreprocessMergeByHalfAndMultiply(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    int inputSize = (int) predictors.size();
    int factorSize = inputSize / 2;
    if (inputSize < 2) {
        wxLogError(_("The number of predictors must be superior to 2 in asPreprocessor::PreprocessMergeByHalfAndMultiply"));
        return false;
    }
    if (inputSize % 2 != 0) {
        wxLogError(_("The number of predictors must be dividable by 2 in asPreprocessor::PreprocessMergeByHalfAndMultiply"));
        return false;
    }

    // Handle sizes
    wxASSERT(predictors[0]);
    unsigned int originalRowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int originalColsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(originalRowsNb > 0);
    wxASSERT(originalColsNb > 0);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    int newRowsNb = originalRowsNb * factorSize;
    int newColsNb = originalColsNb;

    // Initialize
    wxASSERT(predictors[0]);
    vvva2f copyData = vvva2f(2, vva2f(timeSize, va2f(membersNb, a2f::Zero(newRowsNb, newColsNb))));

    // Merge
    for (unsigned int iHalf = 0; iHalf < 2; iHalf++) {
        for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
            for (unsigned int iMem = 0; iMem < membersNb; iMem++) {

                for (int iPre = 0; iPre < inputSize / 2; iPre++) {
                    int iCurr = iHalf * inputSize / 2 + iPre;
                    wxASSERT(predictors[iCurr]);
                    wxASSERT(predictors[iCurr]->GetLatPtsnb() == originalRowsNb);
                    wxASSERT(predictors[iCurr]->GetLonPtsnb() == originalColsNb);
                    wxASSERT(predictors[iCurr]->GetTimeSize() == timeSize);
                    wxASSERT(predictors[iCurr]->GetMembersNb() == membersNb);

                    copyData[iHalf][iTime][iMem].block(iPre * originalRowsNb, 0, originalRowsNb,
                                                       originalColsNb) = predictors[iCurr]->GetData()[iTime][iMem];
                }
            }
        }
    }

    // Create container
    vva2f multi(timeSize, va2f(membersNb, a2f::Zero(newRowsNb, newColsNb)));

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (int iRow = 0; iRow < newRowsNb; iRow++) {
                for (int iCol = 0; iCol < newColsNb; iCol++) {
                    multi[iTime][iMem](iRow, iCol) = copyData[0][iTime][iMem](iRow, iCol) * copyData[1][iTime][iMem](iRow, iCol);
                }
            }
        }
    }

    // Overwrite the data in the predictor object
    result->SetData(multi);
    result->SetIsPreprocessed(true);
    result->SetCanBeClipped(false);

    return true;
}

bool asPreprocessor::PreprocessHumidityFlux(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    int inputSize = (int) predictors.size();
    if (inputSize != 4) {
        wxLogError(_("The number of predictors must be equal to 4 in asPreprocessor::PreprocessHumidityFlux"));
        return false;
    }
    wxASSERT(predictors[0]);

#ifdef _DEBUG
    for (unsigned int iPre = 0; iPre < predictors.size() - 1; iPre++) {
        wxASSERT(predictors[iPre]->GetData()[0][0].rows() == predictors[iPre + 1]->GetData()[0][0].rows());
        wxASSERT(predictors[iPre]->GetData()[0][0].cols() == predictors[iPre + 1]->GetData()[0][0].cols());
        wxASSERT(predictors[iPre]->GetData().size() == predictors[iPre + 1]->GetData().size());
    }
#endif

    // Get sizes
    wxASSERT(predictors[0]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 0);
    wxASSERT(colsNb > 0);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f multi(timeSize, va2f(membersNb, a2f::Zero(rowsNb, colsNb)));

    float wind;

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iRow = 0; iRow < rowsNb; iRow++) {
                for (unsigned int iCol = 0; iCol < colsNb; iCol++) {
                    wind = (float) sqrt(predictors[0]->GetData()[iTime][iMem](iRow, iCol) *
                                        predictors[0]->GetData()[iTime][iMem](iRow, iCol) +
                                        predictors[1]->GetData()[iTime][iMem](iRow, iCol) *
                                        predictors[1]->GetData()[iTime][iMem](iRow, iCol));
                    multi[iTime][iMem](iRow, iCol) = wind * predictors[2]->GetData()[iTime][iMem](iRow, iCol) *
                                                     predictors[3]->GetData()[iTime][iMem](iRow, iCol);
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

bool asPreprocessor::PreprocessWindSpeed(std::vector<asPredictor *> predictors, asPredictor *result)
{
    // More than one predictor
    int inputSize = (int) predictors.size();
    if (inputSize != 2) {
        wxLogError(_("The number of predictors must be equal to 2 in asPreprocessor::PreprocessWindSpeed"));
        return false;
    }

    // Get sizes
    wxASSERT(predictors[0]);
    wxASSERT(predictors[1]);
    unsigned int rowsNb = (unsigned int) predictors[0]->GetLatPtsnb();
    unsigned int colsNb = (unsigned int) predictors[0]->GetLonPtsnb();
    unsigned int timeSize = (unsigned int) predictors[0]->GetTimeSize();
    unsigned int membersNb = (unsigned int) predictors[0]->GetMembersNb();

    wxASSERT(rowsNb > 0);
    wxASSERT(colsNb > 0);
    wxASSERT(timeSize > 0);
    wxASSERT(membersNb > 0);

    // Create container
    vva2f multi(timeSize, va2f(membersNb, a2f::Zero(rowsNb, colsNb)));

    float wind;

    for (unsigned int iTime = 0; iTime < timeSize; iTime++) {
        for (unsigned int iMem = 0; iMem < membersNb; iMem++) {
            for (unsigned int iRow = 0; iRow < rowsNb; iRow++) {
                for (unsigned int iCol = 0; iCol < colsNb; iCol++) {
                    // Get wind value
                    wind = (float) sqrt(predictors[0]->GetData()[iTime][iMem](iRow, iCol) *
                                        predictors[0]->GetData()[iTime][iMem](iRow, iCol) +
                                        predictors[1]->GetData()[iTime][iMem](iRow, iCol) *
                                        predictors[1]->GetData()[iTime][iMem](iRow, iCol));
                    multi[iTime][iMem](iRow, iCol) = wind;
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
