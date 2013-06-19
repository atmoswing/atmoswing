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
