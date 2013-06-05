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
 
#include "asLog.h"

#include <wx/ffile.h>


asLog::asLog()
{
    m_LogFile = NULL;
    m_LogChain = NULL;
    m_Level = 2;
    m_Active = true;
    m_RemoveDuplicates = false;
    m_SignalDuplicates = false;
    m_MessageBoxOnError = false;
    #if wxUSE_GUI
        m_MessageBoxOnError = true;
    #endif
    m_Buffer = wxEmptyString;
    m_State = wxEmptyString;
}

asLog::~asLog()
{
    delete wxLog::SetActiveTarget(NULL); // Instead of deleting m_LogChain
    if(m_LogFile)
    {
        m_LogFile->Close();
        m_LogFile->Detach();
        wxDELETE(m_LogFile);
    }
}

bool asLog::CreateFile(const wxString &fileName)
{
    // Create the log file
    wxString logpath = asConfig::GetLogDir();
    logpath.Append(DS);
    logpath.Append(fileName);
    m_LogFile = new wxFFile(logpath, "w");
    wxLogStderr* pLogFile = new wxLogStderr(m_LogFile->fp());
    m_LogChain = new wxLogChain(pLogFile);

    return true;
}

bool asLog::CreateFileOnly(const wxString &fileName)
{
    // Create the log file
    wxString logpath = asConfig::GetLogDir();
    logpath.Append(DS);
    logpath.Append(fileName);
    m_LogFile = new wxFFile(logpath, "w");
    wxLogStderr* pLogFile = new wxLogStderr(m_LogFile->fp());
    wxLog::SetActiveTarget(pLogFile);

    return true;
}

bool asLog::CreateFileOnlyAtPath(const wxString &fullPath)
{
    // Create the log file
    m_LogFile = new wxFFile(fullPath, "w");
    wxLogStderr* pLogFile = new wxLogStderr(m_LogFile->fp());
    wxLog::SetActiveTarget(pLogFile);

    return true;
}

void asLog::SetLogNull()
{
    wxLogNull log;
}

void asLog::Suspend()
{
    wxLog::Suspend();
    wxLogMessage(_("Log suspended."));
}

void asLog::Resume()
{
    wxLog::Resume();
    wxLogMessage(_("Log resumed."));
}

void asLog::Error(const wxString &msg)
{
    if(m_Level>=1)
    {
        if(g_GuiMode && m_Active)
        {
            if(m_RemoveDuplicates)
            {
                if(m_Buffer.IsSameAs(msg))
                {
                    if(m_SignalDuplicates)
                    {
                        m_CritSectionLog.Enter();
                        wxLogError(_("Previous error occured multiple times."));
                        m_SignalDuplicates = false;
                        m_CritSectionLog.Leave();
                    }
                }
                else
                {
                    m_CritSectionLog.Enter();
                    wxLogError(msg);
                    m_SignalDuplicates = true;
                    m_Buffer = msg;
                    if (m_MessageBoxOnError)
                    {
                        #ifndef UNIT_TESTING
                            #if wxUSE_GUI
                                wxMessageBox(msg, _("An error occured"));
                            #endif
                        #endif
                    }
                    m_CritSectionLog.Leave();
                }
            }
            else
            {
                m_CritSectionLog.Enter();
                wxLogError(msg);
                if (m_MessageBoxOnError)
                {
                    #ifndef UNIT_TESTING
                        #if wxUSE_GUI
                            wxMessageBox(msg, _("An error occured"));
                        #endif
                    #endif
                }
                m_CritSectionLog.Leave();
            }
        }
        else
        {
            if (m_Target==asLog::File || m_Target==asLog::Both)
            {
                wxLogError(msg);
            }

            // To the command prompt
            if (m_Target==asLog::Screen || m_Target==asLog::Both)
            {
                wxMessageOutput* msgOut = wxMessageOutput::Get();
                if ( msgOut )
                {
                    wxString msgNew = _("Error: ") + msg;
                    msgOut->Printf( msgNew );
                }
                else
                {
                    wxFAIL_MSG( _("No wxMessageOutput object?") );
                }
            }
        }
    }
}

void asLog::Warning(const wxString &msg)
{
    if(m_Level>=2)
    {
        if (g_GuiMode && m_Active)
        {
            if(m_RemoveDuplicates)
            {
                if(m_Buffer.IsSameAs(msg))
                {
                    if(m_SignalDuplicates)
                    {
                        m_CritSectionLog.Enter();
                        wxLogWarning(_("Previous warning occured multiple times."));
                        m_SignalDuplicates = false;
                        m_CritSectionLog.Leave();
                    }
                }
                else
                {
                    m_CritSectionLog.Enter();
                    wxLogWarning(msg);
                    m_SignalDuplicates = true;
                    m_Buffer = msg;
                    m_CritSectionLog.Leave();
                }
            }
            else
            {
                m_CritSectionLog.Enter();
                wxLogWarning(msg);
                m_CritSectionLog.Leave();
            }
        }
        else
        {
            if (m_Target==asLog::File || m_Target==asLog::Both)
            {
                wxLogWarning(msg);
            }

            // To the command prompt
            if (m_Target==asLog::Screen || m_Target==asLog::Both)
            {
                wxMessageOutput* msgOut = wxMessageOutput::Get();
                if ( msgOut )
                {
                    wxString msgNew = _("Warning: ") + msg;
                    msgOut->Printf( msgNew );
                }
                else
                {
                    wxFAIL_MSG( _("No wxMessageOutput object?") );
                }
            }
        }
    }
}

void asLog::Message(const wxString &msg, bool force)
{
    if( (m_Level>0 && force) || m_Level>=3)
    {
        if (g_GuiMode && m_Active)
        {
            if(m_RemoveDuplicates)
            {
                if(m_Buffer.IsSameAs(msg))
                {
                    if(m_SignalDuplicates)
                    {
                        m_CritSectionLog.Enter();
                        wxLogMessage(_("Previous message occured multiple times."));
                        m_SignalDuplicates = false;
                        m_CritSectionLog.Leave();
                    }
                }
                else
                {
                    m_CritSectionLog.Enter();
                    wxLogMessage(msg);
                    m_SignalDuplicates = true;
                    m_Buffer = msg;
                    m_CritSectionLog.Leave();
                }
            }
            else
            {
                m_CritSectionLog.Enter();
                wxLogMessage(msg);
                m_CritSectionLog.Leave();
            }
        }
        else
        {
            if (m_Target==asLog::File || m_Target==asLog::Both)
            {
                wxLogMessage(msg);
            }

            // To the command prompt
            if (m_Target==asLog::Screen || m_Target==asLog::Both)
            {
                wxMessageOutput* msgOut = wxMessageOutput::Get();
                if ( msgOut )
                {
                    msgOut->Printf( msg );
                }
                else
                {
                    wxFAIL_MSG( _("No wxMessageOutput object?") );
                }
            }
        }
    }
}

void asLog::State(const wxString &msg)
{
    m_CritSectionLog.Enter();
    m_State = msg;
    m_CritSectionLog.Leave();
    Message(msg); // Also log it
}
