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
 
#include "asLog.h"

#include <wx/ffile.h>


asLog::asLog()
{
    m_LogFile = NULL;
    m_LogChain = NULL;
    m_Level = 2;
    m_Target = asLog::File;
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
    wxDELETE(m_LogFile);
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
    wxDELETE(m_LogFile);
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
    wxDELETE(m_LogFile);
    m_LogFile = new wxFFile(fullPath, "w");
    wxLogStderr* pLogFile = new wxLogStderr(m_LogFile->fp());
    wxLog::SetActiveTarget(pLogFile);

    return true;
}

bool asLog::IsVerbose()
{
    if (m_Level>=3) {
        return true;
    }
    else {
        return false;
    }
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
                        #if wxUSE_GUI
                            wxMessageBox(msg, _("An error occured"));
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
                    #if wxUSE_GUI
                        wxMessageBox(msg, _("An error occured"));
                    #endif
                }
                m_CritSectionLog.Leave();
            }
        }
        else
        {
			bool processed = false;

            if (m_Target==asLog::File || m_Target==asLog::Both)
            {
                wxLogError(msg);
				processed = true;
            }

            // To the command prompt
            if (m_Target==asLog::Screen)
            {
                wxMessageOutput* msgOut = wxMessageOutput::Get();
                if ( msgOut )
                {
                    wxString msgNew = _("Error: ") + msg;
                    msgOut->Printf( msgNew );
					processed = true;
                }
                else
                {
                    wxFAIL_MSG( _("No wxMessageOutput object?") );
                }
            }

			if(!processed)
			{
				printf("Error: %s\n", msg.mb_str(wxConvUTF8).data());
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
			bool processed = false;

            if (m_Target==asLog::File || m_Target==asLog::Both)
            {
                wxLogWarning(msg);
				processed = true;
            }

            // To the command prompt
            if (m_Target==asLog::Screen)
            {
                wxMessageOutput* msgOut = wxMessageOutput::Get();
                if ( msgOut )
                {
                    wxString msgNew = _("Warning: ") + msg;
                    msgOut->Printf( msgNew );
					processed = true;
                }
                else
                {
                    wxFAIL_MSG( _("No wxMessageOutput object?") );
                }
            }

			if(!processed)
			{
				printf("Warning: %s\n", msg.mb_str(wxConvUTF8).data());
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
			bool processed = false;

            if (m_Target==asLog::File || m_Target==asLog::Both)
            {
                wxLogMessage(msg);
				processed = true;
            }

            // To the command prompt
            if (m_Target==asLog::Screen)
            {
                wxMessageOutput* msgOut = wxMessageOutput::Get();
                if ( msgOut )
                {
                    msgOut->Printf( msg );
					processed = true;
                }
                else
                {
                    wxFAIL_MSG( _("No wxMessageOutput object?") );
                }
            }

			if(!processed)
			{
				printf("%s\n", msg.mb_str(wxConvUTF8).data());
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
