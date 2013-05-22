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
 
#ifndef ASFILEFORECASTINGMODELS_H
#define ASFILEFORECASTINGMODELS_H

#include <asIncludes.h>
#include <asFileXml.h>


class asFileForecastingModels : public asFileXml
{
public:
    /** Default constructor */
    asFileForecastingModels(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileForecastingModels();

    virtual bool InsertRootElement();

    virtual bool GoToRootElement();


protected:

private:

};

#endif // ASFILEFORECASTINGMODELS_H
