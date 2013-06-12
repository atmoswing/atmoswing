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
