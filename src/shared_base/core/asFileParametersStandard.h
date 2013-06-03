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
 
#ifndef ASFILEPARAMETERSSTANDARD_H
#define ASFILEPARAMETERSSTANDARD_H

#include <asIncludes.h>
#include <asFileParameters.h>

class asFileParametersStandard : public asFileParameters
{
public:
    /** Default constructor */
    asFileParametersStandard(const wxString &FileName, const ListFileMode &FileMode = asFile::Replace);
    /** Default destructor */
    virtual ~asFileParametersStandard();

    bool InsertRootElement();
    bool GoToRootElement();

protected:
private:
};

#endif // ASFILEPARAMETERSSTANDARD_H
