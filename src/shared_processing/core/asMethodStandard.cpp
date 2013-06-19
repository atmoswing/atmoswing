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

}

void asMethodStandard::Cleanup()
{
    wxDELETE(m_PredictandDB);
}

bool asMethodStandard::Manager()
{
    return false;
}

bool asMethodStandard::LoadPredictandDB(const wxString &predictandDBType, const wxString &predictandDBFilePath)
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
