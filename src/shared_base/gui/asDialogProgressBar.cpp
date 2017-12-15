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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 */

#include "asDialogProgressBar.h"


asDialogProgressBar::asDialogProgressBar(const wxString &dialogMessage, int valueMax)
        : m_progressBar(nullptr),
          m_initiated(false),
          m_steps(100),
          m_delayUpdate(false),
          m_valueMax(valueMax),
          m_currentStepIndex(0)
{
    if (!g_silentMode) {
        if (valueMax > 2 * m_steps) {
            m_delayUpdate = true;
            m_vectorSteps.resize((unsigned long) (m_steps + 1));
            for (int i = 0; i <= m_steps; i++) {
                m_vectorSteps[i] = i * valueMax / m_steps;
            }
        }

        if (valueMax > 10) {
            m_progressBar = new wxProgressDialog(_("Please wait"), dialogMessage, valueMax, NULL,
                                                 wxPD_AUTO_HIDE | wxPD_CAN_ABORT | wxPD_REMAINING_TIME |
                                                 wxPD_ELAPSED_TIME | wxPD_SMOOTH); // wxPD_APP_MODAL |
            m_initiated = true;
        }
    }

}

asDialogProgressBar::~asDialogProgressBar()
{
    if (m_initiated) {
        m_progressBar->Update(m_valueMax);
        m_progressBar->Destroy();
        m_initiated = false;
        wxWakeUpIdle();
    }
}

void asDialogProgressBar::Destroy()
{
    if (m_initiated) {
        m_progressBar->Update(m_valueMax);
        m_progressBar->Destroy();
        m_initiated = false;
        wxWakeUpIdle();
    }
}

bool asDialogProgressBar::Update(int value, const wxString &message)
{
    wxString newMessage = message;

    if (m_initiated) {
        if (m_delayUpdate) {
            if (value >= m_vectorSteps[m_currentStepIndex]) {
                m_currentStepIndex++;
                if (g_verboseMode) {
                    if (!message.IsEmpty()) {
                        newMessage = message + wxString::Format("(%d/%d)", value, m_valueMax);
                    }
                }
                return m_progressBar->Update(value, newMessage);
            }
        } else {
            if (g_verboseMode) {
                if (!message.IsEmpty()) {
                    newMessage = message + wxString::Format("(%d/%d)", value, m_valueMax);
                }
            }
            return m_progressBar->Update(value, newMessage);
        }
    }
    return true;
}
