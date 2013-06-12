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
 
#ifndef ASVERSION_H
#define ASVERSION_H

#include "wx/string.h"

const int ATMOSWING_MAJOR_VERSION = 1;
const int ATMOSWING_MINOR_VERSION = 0;
const int ATMOSWING_PATCH_VERSION = 3;
const extern wxString g_Version;

class asVersion
{
public:
    asVersion();
    virtual ~asVersion();
    static wxString GetFullString();


protected:
private:
};

#endif // ASVERSION_H
