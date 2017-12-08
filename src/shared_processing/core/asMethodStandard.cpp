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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
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
        : m_cancel(false),
          m_predictandDB(nullptr)
{

}

asMethodStandard::~asMethodStandard()
{
    wxDELETE(m_predictandDB);
}

bool asMethodStandard::Manager()
{
    return false;
}

bool asMethodStandard::LoadPredictandDB(const wxString &predictandDBFilePath)
{
    wxDELETE(m_predictandDB);

    if (predictandDBFilePath.IsEmpty()) {
        if (m_predictandDBFilePath.IsEmpty()) {
            wxLogError(_("There is no predictand database file selected."));
            return false;
        }

        m_predictandDB = asPredictand::GetInstance(m_predictandDBFilePath);
        if (!m_predictandDB)
            return false;

        if (!m_predictandDB->Load(m_predictandDBFilePath)) {
            wxLogError(_("Couldn't load the predictand database."));
            return false;
        }
    } else {
        m_predictandDB = asPredictand::GetInstance(predictandDBFilePath);
        if (!m_predictandDB)
            return false;

        if (!m_predictandDB->Load(predictandDBFilePath)) {
            wxLogError(_("Couldn't load the predictand database."));
            return false;
        }
    }

    if (!m_predictandDB)
        return false;
    wxASSERT(m_predictandDB);

    return true;
}

void asMethodStandard::Cancel()
{
    m_cancel = true;
}
