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
