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
 
#include "asDialogProgressBar.h"

#include <wx/app.h>

asDialogProgressBar::asDialogProgressBar(const wxString &DialogMessage, int ValueMax)
{
    m_ProgressBar = NULL;
    m_Initiated = false;
    m_DelayUpdate = false;
    m_Steps = 100;
    m_ValueMax = ValueMax;
    m_CurrentStepIndex = 0;

    if (!g_SilentMode)
    {
        if (ValueMax>2*m_Steps)
        {
            m_DelayUpdate = true;
            m_VectorSteps.resize(m_Steps+1);
            for (int i=0; i<=m_Steps; i++)
            {
                m_VectorSteps[i] = i*ValueMax/m_Steps;
            }
        }

        if (ValueMax>10)
        {
            m_ProgressBar = new wxProgressDialog(_("Please wait"), DialogMessage, ValueMax, NULL, wxPD_AUTO_HIDE | wxPD_CAN_ABORT | wxPD_REMAINING_TIME | wxPD_ELAPSED_TIME | wxPD_SMOOTH); // wxPD_APP_MODAL |
            m_Initiated = true;
        }
    }

}

asDialogProgressBar::~asDialogProgressBar()
{
    if (m_Initiated)
    {
        m_ProgressBar->Update(m_ValueMax);
        m_ProgressBar->Destroy();
        m_Initiated = false;
        wxWakeUpIdle();
    }
}

void asDialogProgressBar::Destroy()
{
    if (m_Initiated)
    {
        m_ProgressBar->Update(m_ValueMax);
        m_ProgressBar->Destroy();
        m_Initiated = false;
        wxWakeUpIdle();
    }
}

bool asDialogProgressBar::Update(int Value, const wxString &Message)
{
    wxString NewMessage = Message;

    if (m_Initiated)
    {
        if (m_DelayUpdate)
        {
            if(Value>=m_VectorSteps[m_CurrentStepIndex])
            {
                m_CurrentStepIndex++;
                if(g_VerboseMode)
                {
                    if (!Message.IsEmpty())
                    {
                        wxString NewMessage = Message + wxString::Format("(%d/%d)", Value, m_ValueMax);
                    }
                }
                return m_ProgressBar->Update(Value, NewMessage);
            }
        }
        else
        {
            if(g_VerboseMode)
            {
                if (!Message.IsEmpty())
                {
                    wxString NewMessage = Message + wxString::Format("(%d/%d)", Value, m_ValueMax);
                }
            }
            return m_ProgressBar->Update(Value, NewMessage);
        }
    }
    return true;
}
