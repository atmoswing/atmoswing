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
 
#ifndef ASLOGWINDOW_H
#define ASLOGWINDOW_H

#include <asIncludes.h>

#include "wx/log.h"

class asLogWindow: public wxLogWindow
{
public:
    asLogWindow(wxFrame *parent, const wxString& title = _("Atmoswing log window"), bool show = true, bool passToOld = false);
    virtual ~asLogWindow();

    virtual void DoShow(bool bShow = true);

protected:

private:
    virtual bool OnFrameClose (wxFrame *frame = NULL);

};

#endif // ASLOGWINDOW_H
