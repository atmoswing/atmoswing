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
 
#include "asLogGlobalFunctions.h"

// Global log functions
asLog* g_pLog = new asLog();
asLog& Log()
{
    return *g_pLog;
}

void asLogError(const wxString &msg)
{
    g_pLog->Error(msg);
}

void asLogError(const wxChar* msg)
{
    g_pLog->Error(msg);
}

void asLogWarning(const wxString &msg)
{
    g_pLog->Warning(msg);
}

void asLogWarning(const wxChar* msg)
{
    g_pLog->Warning(msg);
}

void asLogMessage(const wxString &msg)
{
    g_pLog->Message(msg);
}

void asLogMessage(const wxChar* msg)
{
    g_pLog->Message(msg);
}

void asLogMessageImportant(const wxString &msg)
{
    g_pLog->Message(msg, true);
}

void asLogMessageImportant(const wxChar* msg)
{
    g_pLog->Message(msg, true);
}

void asLogState(const wxString &msg)
{
    g_pLog->State(msg);
}

void asLogState(const wxChar* msg)
{
    g_pLog->State(msg);
}

wxString asGetState()
{
    return g_pLog->GetState();
}

void DeleteLog()
{
    wxDELETE(g_pLog);
    delete wxLog::SetActiveTarget(NULL);
}
