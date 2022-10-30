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

// Global log functions
asLog* g_pLog = new asLog();

asLog* Log() {
    return g_pLog;
}

void DeleteLog() {
    if (g_pLog) {
        wxLog::FlushActive();
    }
    wxDELETE(g_pLog);
    delete wxLog::SetActiveTarget(nullptr);
}

asLog::asLog()
    : m_logFile(nullptr),
      m_logChain(nullptr) {}

asLog::~asLog() {
    delete wxLog::SetActiveTarget(nullptr);  // Instead of deleting m_logChain
    ClearCurrentTarget();
}

void asLog::ClearCurrentTarget() {
    if (m_logFile) {
        m_logFile->Close();
        m_logFile->Detach();
        wxDELETE(m_logFile);
    }
}

void asLog::CreateFile(const wxString& fileName) {
    // Create the log file
    ClearCurrentTarget();
    wxString logpath = asConfig::GetLogDir();
    logpath.Append(wxFileName::GetPathSeparator());
    logpath.Append(fileName);
    m_logFile = new wxFFile(logpath, "w");

    if (m_logChain) {
        m_logChain->SetLog(new wxLogStderr(m_logFile->fp()));
    } else {
        m_logChain = new wxLogChain(new wxLogStderr(m_logFile->fp()));
    }
}

void asLog::CreateFileAtPath(const wxString& fullPath) {
    // Create the log file
    ClearCurrentTarget();
    m_logFile = new wxFFile(fullPath, "w");

    if (m_logChain) {
        m_logChain->SetLog(new wxLogStderr(m_logFile->fp()));
    } else {
        m_logChain = new wxLogChain(new wxLogStderr(m_logFile->fp()));
    }
}

void asLog::CreateFileOnly(const wxString& fileName) {
    // Create the log file
    ClearCurrentTarget();
    wxString logPath = asConfig::GetLogDir();
    logPath.Append(wxFileName::GetPathSeparator());
    logPath.Append(fileName);
    m_logFile = new wxFFile(logPath, "w");
    delete wxLog::SetActiveTarget(new wxLogStderr(m_logFile->fp()));
}

void asLog::CreateFileOnlyAtPath(const wxString& fullPath) {
    // Create the log file
    ClearCurrentTarget();
    m_logFile = new wxFFile(fullPath, "w");
    delete wxLog::SetActiveTarget(new wxLogStderr(m_logFile->fp()));
}

void asLog::SetLevel(int val) {
    switch (val) {
        case 1:
            wxLog::SetLogLevel(wxLOG_Error);
            break;
        case 2:
            wxLog::SetLogLevel(wxLOG_Message);
            break;
        case 3:
            wxLog::SetVerbose();
            wxLog::SetLogLevel(wxLOG_Info);
            break;
        default:
            wxLog::SetLogLevel(wxLOG_Message);
    }
}

void asLog::PrintToConsole(const wxString& msg) {
#ifndef __WIN32__
    if (!g_guiMode) {
        wxMessageOutput* msgOut = wxMessageOutput::Get();
        if (msgOut) {
            msgOut->Printf(msg);
        }
    }
#endif
}

#if USE_GUI
void asLogGui::DoLogRecord(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) {
    if (level <= wxLOG_Error) {
        wxLogGui::DoLogRecord(level, msg, info);
    }
}
#endif