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
 
#ifndef ASPANELSMANAGER_H
#define ASPANELSMANAGER_H

#include "asIncludes.h"

class asPanelsManager : public wxObject
{
public:
    asPanelsManager();
    virtual ~asPanelsManager();

    void LayoutFrame(wxWindow* element);
    wxWindow* GetTopFrame(wxWindow* element);

protected:

private:


};

#endif // ASPANELSMANAGER_H
