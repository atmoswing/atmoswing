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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#include "asWizardBatchForecasts.h"
#include "asFramePreferencesForecaster.h"

#include <wx/statline.h>
#include <wx/stdpaths.h>

wxDEFINE_EVENT(asEVT_ACTION_OPEN_BATCHFORECASTS, wxCommandEvent);

asWizardBatchForecasts::asWizardBatchForecasts( wxWindow* parent, asBatchForecasts* batchForecasts, wxWindowID id )
:
asWizardBatchForecastsVirtual( parent, id )
{
    m_BatchForecasts = batchForecasts;
}

asWizardBatchForecasts::~asWizardBatchForecasts()
{

}

void asWizardBatchForecasts::OnWizardFinished( wxWizardEvent& event )
{
    wxString filePath = m_FilePickerBatchFile->GetPath();
    m_BatchForecasts->SetFilePath(filePath);
    m_BatchForecasts->Save();
    
    if (!filePath.IsEmpty())
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", filePath);
    }

    // Open the preferences frame
    asFramePreferencesForecaster* frame = new asFramePreferencesForecaster(NULL, m_BatchForecasts);
    frame->Fit();
    frame->Show();
}

void asWizardBatchForecasts::OnLoadExistingBatchForecasts( wxCommandEvent& event )
{
    wxCommandEvent eventOpen (asEVT_ACTION_OPEN_BATCHFORECASTS);
    GetParent()->ProcessWindowEvent(eventOpen);
    Close();
}