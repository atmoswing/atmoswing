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
 
#ifndef ASDIALOGPROGRESSBAR_H
#define ASDIALOGPROGRESSBAR_H

#include <asIncludes.h>

#include "wx/progdlg.h"

class asDialogProgressBar: public wxObject
{
public:
    asDialogProgressBar(const wxString &DialogMessage, int ValueMax);
    virtual ~asDialogProgressBar();

    void Destroy();
    bool Update(int Value, const wxString &Message = wxEmptyString);

protected:

private:
    wxProgressDialog* m_ProgressBar;
    bool m_Initiated;
    int m_Steps;
    int m_DelayUpdate;
    int m_ValueMax;
    VectorInt m_VectorSteps;
    int m_CurrentStepIndex;

};

#endif // ASDIALOGPROGRESSBAR_H
