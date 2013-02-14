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
