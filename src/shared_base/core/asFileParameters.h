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
 
#ifndef ASFILEPARAMETERS_H
#define ASFILEPARAMETERS_H

#include <asIncludes.h>
#include <asFileXml.h>

class asFileParameters : public asFileXml
{
public:
    /** Default constructor */
    asFileParameters(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileParameters();

    virtual bool InsertRootElement() = 0;
    virtual bool GoToRootElement() = 0;

protected:
private:

};

#endif // ASFILEPARAMETERS_H
