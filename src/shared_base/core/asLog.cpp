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

#include "asLog.h"

#include <wx/ffile.h>


asLog::asLog()
{
    m_logFile = NULL;
    m_logChain = NULL;
}

asLog::~asLog()
{
    delete wxLog::SetActiveTarget(NULL); // Instead of deleting m_logChain
    if (m_logFile) {
        m_logFile->Close();
        m_logFile->Detach();
        wxDELETE(m_logFile);
    }
}

bool asLog::CreateFile(const wxString &fileName)
{
    // Create the log file
    wxDELETE(m_logFile);
    wxString logpath = asConfig::GetLogDir();
    logpath.Append(DS);
    logpath.Append(fileName);
    m_logFile = new wxFFile(logpath, "w");
    wxLogStderr *pLogFile = new wxLogStderr(m_logFile->fp());
    m_logChain = new wxLogChain(pLogFile);

    return true;
}

bool asLog::CreateFileOnly(const wxString &fileName)
{
    // Create the log file
    wxDELETE(m_logFile);
    wxString logpath = asConfig::GetLogDir();
    logpath.Append(DS);
    logpath.Append(fileName);
    m_logFile = new wxFFile(logpath, "w");
    wxLogStderr *pLogFile = new wxLogStderr(m_logFile->fp());
    wxLog::SetActiveTarget(pLogFile);

    return true;
}

bool asLog::CreateFileOnlyAtPath(const wxString &fullPath)
{
    // Create the log file
    wxDELETE(m_logFile);
    m_logFile = new wxFFile(fullPath, "w");
    wxLogStderr *pLogFile = new wxLogStderr(m_logFile->fp());
    wxLog::SetActiveTarget(pLogFile);

    return true;
}

void asLog::SetLevel(int val)
{
    switch (val) {
        case 1 :
            wxLog::SetLogLevel(wxLOG_Error);
            break;
        case 2 :
            wxLog::SetLogLevel(wxLOG_Message);
            break;
        case 3 :
            wxLog::SetVerbose();
            wxLog::SetLogLevel(wxLOG_Info);
            break;
        case 4 :
            wxLog::SetVerbose();
            wxLog::SetLogLevel(wxLOG_Progress);
            break;
        default:
            wxLog::SetLogLevel(wxLOG_Message);
    }
}
