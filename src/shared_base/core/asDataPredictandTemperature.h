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
 
#ifndef ASDATAPREDICTANDTEMPERATURE_H
#define ASDATAPREDICTANDTEMPERATURE_H

#include <asIncludes.h>
#include <asDataPredictand.h>


class asDataPredictandTemperature: public asDataPredictand
{
public:
    asDataPredictandTemperature(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataSpatialAggregation);
    virtual ~asDataPredictandTemperature();

    virtual bool Load(const wxString &filePath);

    virtual bool Save(const wxString &AlternateDestinationDir = wxEmptyString);

    virtual bool BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString, const wxString &AlternateDestinationDir = wxEmptyString);


protected:

private:
    
    /** Initialize the containers
     * \return True on success
     */
    bool InitContainers();

};

#endif // ASDATAPREDICTANDTEMPERATURE_H
