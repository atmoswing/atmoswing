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
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asMethodStandard.h"


wxDEFINE_EVENT(asEVT_STATUS_STARTING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_RUNNING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_FAILED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_SUCCESS, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_DOWNLOADING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_DOWNLOADED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_LOADING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_LOADED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_SAVING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_SAVED, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_PROCESSING, wxCommandEvent);
wxDEFINE_EVENT(asEVT_STATUS_PROCESSED, wxCommandEvent);


asMethodStandard::asMethodStandard()
{
    m_ParamsFilePath = wxEmptyString;
    m_PredictandDBFilePath = wxEmptyString;
    m_PredictorDataDir = wxEmptyString;
    m_PredictandDB = NULL;
    m_Cancel = false;
}

asMethodStandard::~asMethodStandard()
{
    wxDELETE(m_PredictandDB);
}

bool asMethodStandard::Manager()
{
    return false;
}

bool asMethodStandard::LoadPredictandDB(const wxString &predictandDBFilePath)
{
    wxDELETE(m_PredictandDB);

    if (predictandDBFilePath.IsEmpty())
    {
        if (m_PredictandDBFilePath.IsEmpty())
        {
            asLogError(_("There is no predictand database file selected."));
            return false;
        }

        m_PredictandDB = asDataPredictand::GetInstance(m_PredictandDBFilePath);

        if(!m_PredictandDB->Load(m_PredictandDBFilePath))
        {
            asLogError(_("Couldn't load the predictand database."));
            return false;
        }
    }
    else
    {
        m_PredictandDB = asDataPredictand::GetInstance(predictandDBFilePath);

        if(!m_PredictandDB->Load(predictandDBFilePath))
        {
            asLogError(_("Couldn't load the predictand database."));
            return false;
        }
    }

    if (!m_PredictandDB) return false;
    wxASSERT(m_PredictandDB);

    return true;
}

void asMethodStandard::Cancel()
{
    m_Cancel = true;
}
