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
 
#ifndef ASPROCESSOR_H
#define ASPROCESSOR_H

#include <asIncludes.h>

class asTimeArray;
class asParameters;
class asResultsAnalogsDates;
class asPredictorCriteria;
class asResultsAnalogsValues;
class asDataPredictor;
class asDataPredictand;


class asProcessor: public wxObject
{
public:

    /** Analogs method processing
    * \param predictors The vector of predictors
    * \param timeArrayProcessor The time array of target for the predictors
    * \param criteria The vector of criteria
    * \param params The parameters
    * \param step The level of analogy (must be 0)
    * \param results The result object to store outputs in
    * \return true if succeeded
    */
    static bool GetAnalogsDates(std::vector < asDataPredictor > &predictorsArchive, std::vector < asDataPredictor > &predictorsTarget, asTimeArray &timeArrayArchiveData, asTimeArray &timeArrayArchiveSelection, asTimeArray &timeArrayTargetData, asTimeArray &timeArrayTargetSelection, std::vector < asPredictorCriteria* > criteria, asParameters &params, int step, asResultsAnalogsDates &results, bool &containsNaNs);

    /** Analogs method processing at a second level
    * \param predictors The vector of predictors
    * \param timeArrayProcessor The time array of target for the predictors
    * \param anaDates The previous analogs dates results
    * \param criteria The vector of criteria
    * \param params The parameters
    * \param step The level of analogy (must be 0)
    * \param results The result object to store outputs in
    * \return true if succeeded
    */
    static bool GetAnalogsSubDates(std::vector < asDataPredictor > &predictorsArchive, std::vector < asDataPredictor > &predictorsTarget, asTimeArray &timeArrayArchiveData, asTimeArray &timeArrayTargetData, asResultsAnalogsDates &anaDates, std::vector < asPredictorCriteria* > criteria, asParameters &params, int step, asResultsAnalogsDates &results, bool &containsNaNs);

    /** Analogs predictands values attribution
    * \param predictand The predictand data
    * \param anaDates The ResAnalogsDates structure
    * \param StationId The station ID
    * \return The ResAnalogsValues structure
    */
    static bool GetAnalogsValues(asDataPredictand &predictand, asResultsAnalogsDates &anaDates, asParameters &params, asResultsAnalogsValues &results);

protected:
private:
};

#endif
