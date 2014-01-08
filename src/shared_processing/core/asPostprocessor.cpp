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

#include "asPostprocessor.h"
/*
#include <asParametersScoring.h>


Array1DFloat asPostprocessor::Postprocess(const Array1DFloat &analogsValues, const Array1DFloat &analogsCriteria, asParametersScoring &params)
{
    wxString method(params.GetForecastScorePostprocessMethod());

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

Array1DFloat asPostprocessor::PostprocessDuplicationOnCriteria(const Array1DFloat &analogsValues, const Array1DFloat &analogsCriteria)
{
    float sum=0, range=0;
    Array1DFloat analogsWeight(analogsValues.rows());

    // Process ranges
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        range = abs(analogsCriteria[analogsCriteria.rows()-1] - analogsCriteria[i_day]);
        sum += range;
        analogsWeight[i_day] = range;
    }

    // Process weights
    int nbtot=0;
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        analogsWeight[i_day] *= (float)1000/sum;
        analogsWeight[i_day] = wxMax(analogsWeight[i_day], (float)1); // Set the min to 1 to avoid a division by 0
        nbtot += asTools::Round(analogsWeight[i_day]);
    }

    // Duplicate analogs based on the weights
    int counter=0;
    Array1DFloat analogsValuesModified(nbtot);
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        int number = asTools::Round(analogsWeight[i_day]);

        for (int i_nb=0; i_nb<number; i_nb++)
        {
            analogsValuesModified[counter] = analogsValues[i_day];
            counter++;
        }
    }

    return analogsValuesModified;
}

Array1DFloat asPostprocessor::PostprocessDuplicationOnCriteriaExponent(const Array1DFloat &analogsValues, const Array1DFloat &analogsCriteria, asParametersScoring &params)
{
    float sum1=0, sum2=0, range=0;
    Array1DFloat analogsWeight(analogsValues.rows());

    // Process ranges
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        range = abs(analogsCriteria[analogsCriteria.rows()-1] - analogsCriteria[i_day]);
        sum1 += range;
        analogsWeight[i_day] = range;
    }

    // Process the exponent
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        analogsWeight[i_day] /= sum1;
        analogsWeight[i_day] = pow(analogsWeight[i_day], params.GetForecastScorePostprocessDupliExp());
        sum2 += analogsWeight[i_day];
    }

    // Process weights
    int nbtot=0;
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        analogsWeight[i_day] *= (float)1000/sum2;
        analogsWeight[i_day] = wxMax(analogsWeight[i_day], (float)1); // Set the min to 1 to avoid a division by 0
        nbtot += asTools::Round(analogsWeight[i_day]);
    }

    // Duplicate analogs based on the weights
    int counter=0;
    Array1DFloat analogsValuesModified(nbtot);
    for (int i_day=0; i_day<analogsValues.rows(); i_day++)
    {
        int number = asTools::Round(analogsWeight[i_day]);

        for (int i_nb=0; i_nb<number; i_nb++)
        {
            analogsValuesModified[counter] = analogsValues[i_day];
            counter++;
        }
    }

    return analogsValuesModified;
}
*/
