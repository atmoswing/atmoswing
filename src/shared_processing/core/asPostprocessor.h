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

#ifndef ASPOSTPROCESSOR_H
#define ASPOSTPROCESSOR_H

#include <asIncludes.h>

class asParametersScoring;


class asPostprocessor: public wxObject
{
public:

    enum Postprocesses //!< Enumaration of integrated postprocesses
    {
        DuplicationOnCriteria, // Duplication of the analog days based on the criteria
        DuplicationOnCriteriaExponent // Duplication of the analog days based on the criteria with an exponent
    };
/*
    static Array1DFloat Postprocess(const Array1DFloat &analogsValues, const Array1DFloat &analogsCriteria, asParametersScoring &params);


    static Array1DFloat PostprocessDuplicationOnCriteria(const Array1DFloat &analogsValues, const Array1DFloat &analogsCriteria);
    static Array1DFloat PostprocessDuplicationOnCriteriaExponent(const Array1DFloat &analogsValues, const Array1DFloat &analogsCriteria, asParametersScoring &params);
*/

protected:
private:
};

#endif // ASPOSTPROCESSOR_H
