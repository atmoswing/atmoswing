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
 */

/*
#include <asParametersScoring.h>


a1f asPostprocessor::Postprocess(const a1f &analogsValues, const a1f &analogsCriteria, asParametersScoring &params)
{
    wxString method(params.GetScorePostprocessMethod());

    if (method.IsSameAs("DuplicationOnCriteria"))
    {
        return PostprocessDuplicationOnCriteria(analogsValues, analogsCriteria);
    }
    else if (method.IsSameAs("DuplicationOnCriteriaExponent"))
    {
        return PostprocessDuplicationOnCriteriaExponent(analogsValues, analogsCriteria, params);
    }
    else
    {
        asThrowException(_("The postprocessing method was not correctly defined."));
    }
}

a1f asPostprocessor::PostprocessDuplicationOnCriteria(const a1f &analogsValues, const a1f &analogsCriteria)
{
    float sum=0, range=0;
    a1f analogsWeight(analogsValues.rows());

    // Process ranges
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        range = std::abs(analogsCriteria[analogsCriteria.rows()-1] - analogsCriteria[iDay]);
        sum += range;
        analogsWeight[iDay] = range;
    }

    // Process weights
    int nbtot=0;
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        analogsWeight[iDay] *= (float)1000/sum;
        analogsWeight[iDay] = wxMax(analogsWeight[iDay], (float)1); // Set the min to 1 to avoid a division by 0
        nbtot += asRound(analogsWeight[iDay]);
    }

    // Duplicate analogs based on the weights
    int counter=0;
    a1f analogsValuesModified(nbtot);
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        int number = asRound(analogsWeight[iDay]);

        for (int iNb=0; iNb<number; iNb++)
        {
            analogsValuesModified[counter] = analogsValues[iDay];
            counter++;
        }
    }

    return analogsValuesModified;
}

a1f asPostprocessor::PostprocessDuplicationOnCriteriaExponent(const a1f &analogsValues, const a1f &analogsCriteria,
asParametersScoring &params)
{
    float sum1=0, sum2=0, range=0;
    a1f analogsWeight(analogsValues.rows());

    // Process ranges
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        range = std::abs(analogsCriteria[analogsCriteria.rows()-1] - analogsCriteria[iDay]);
        sum1 += range;
        analogsWeight[iDay] = range;
    }

    // Process the exponent
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        analogsWeight[iDay] /= sum1;
        analogsWeight[iDay] = pow(analogsWeight[iDay], params.GetScorePostprocessDupliExp());
        sum2 += analogsWeight[iDay];
    }

    // Process weights
    int nbtot=0;
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        analogsWeight[iDay] *= (float)1000/sum2;
        analogsWeight[iDay] = wxMax(analogsWeight[iDay], (float)1); // Set the min to 1 to avoid a division by 0
        nbtot += asRound(analogsWeight[iDay]);
    }

    // Duplicate analogs based on the weights
    int counter=0;
    a1f analogsValuesModified(nbtot);
    for (int iDay=0; iDay<analogsValues.rows(); iDay++)
    {
        int number = asRound(analogsWeight[iDay]);

        for (int iNb=0; iNb<number; iNb++)
        {
            analogsValuesModified[counter] = analogsValues[iDay];
            counter++;
        }
    }

    return analogsValuesModified;
}
*/
