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
 
#include "asException.h"

asException::asException()
:
std::exception()
{
    m_Message = wxEmptyString;
    m_FileName = wxEmptyString;
    m_LineNum = 0;
    m_HasChild = false;
    asLogError(_("An exception occured."));
}

asException::asException(const wxString &message, const char *filename, unsigned int line)
{
    wxString wxfilename(filename, wxConvUTF8);
    m_Message = message;
    m_FileName = wxfilename;
    m_LineNum = line;
    m_HasChild = false;
    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_Message.c_str(), m_FileName.c_str(), m_LineNum);
    asLogError(logMessage);
}

asException::asException(const std::string &message, const char *filename, unsigned int line)
{
    wxString wxmessage(message.c_str(), wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_Message = wxmessage;
    m_FileName = wxfilename;
    m_FileName = wxString::FromAscii(filename);
    m_LineNum = line;
    m_HasChild = false;
    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_Message.c_str(), m_FileName.c_str(), m_LineNum);
    asLogError(logMessage);
}

asException::asException(const char *message, const char *filename, unsigned int line)
{
    wxString wxmessage(message, wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_Message = wxmessage;
    m_FileName = wxfilename;
    m_LineNum = line;
    m_HasChild = false;
    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_Message.c_str(), m_FileName.c_str(), m_LineNum);
    asLogError(logMessage);
}

asException::asException(const wxString &message, const char *filename, unsigned int line, asException prevexception)
{
    wxString wxfilename(filename, wxConvUTF8);
    m_Message = message;
    m_FileName = wxfilename;
    m_LineNum = line;
    m_HasChild = true;
    m_Previous = prevexception.m_Previous;

    PrevExceptions Previous;
    Previous.Message = prevexception.m_Message;
    Previous.FileName = prevexception.m_FileName;
    Previous.LineNum = prevexception.m_LineNum;
    m_Previous.push_back(&Previous);

    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_Message.c_str(), m_FileName.c_str(), m_LineNum);
    asLogError(logMessage);
}

asException::asException(const std::string &message, const char *filename, unsigned int line, asException prevexception)
{
    wxString wxmessage(message.c_str(), wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_Message = wxmessage;
    m_FileName = wxfilename;
    m_LineNum = line;
    m_HasChild = true;
    m_Previous = prevexception.m_Previous;

    PrevExceptions Previous;
    Previous.Message = prevexception.m_Message;
    Previous.FileName = prevexception.m_FileName;
    Previous.LineNum = prevexception.m_LineNum;
    m_Previous.push_back(&Previous);

    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_Message.c_str(), m_FileName.c_str(), m_LineNum);
    asLogError(logMessage);
}

asException::asException(const char *message, const char *filename, unsigned int line, asException prevexception)
{
    wxString wxmessage(message, wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_Message = wxmessage;
    m_FileName = wxfilename;
    m_LineNum = line;
    m_HasChild = true;
    m_Previous = prevexception.m_Previous;

    PrevExceptions Previous;
    Previous.Message = prevexception.m_Message;
    Previous.FileName = prevexception.m_FileName;
    Previous.LineNum = prevexception.m_LineNum;
    m_Previous.push_back(&Previous);

    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_Message.c_str(), m_FileName.c_str(), m_LineNum);
    asLogError(logMessage);
}
// TODO (phorton#5#): Is it alright ?
asException::~asException() throw ()
{
    //dtor
}

wxString asException::GetFullMessage()
{
    wxString fullmessage;

    if (GetHasChild())
    {
        int prevnb = m_Previous.size();

        for (int i=0; i<prevnb; i++)
        {
            int prevlinenum;
            wxString prevmessage;
            wxString prevfilename;

            prevlinenum = m_Previous[i]->LineNum;
            prevmessage = m_Previous[i]->Message;
            prevfilename = m_Previous[i]->FileName;
            prevmessage.Replace("\n"," // ");
            prevfilename = prevfilename.AfterLast('/');
            prevfilename = prevfilename.AfterLast('\\');

            if(!prevmessage.IsEmpty() && !prevfilename.IsEmpty())
            {
                fullmessage.Append( wxString::Format(_("%s\n    File: %s\n    Line: %d\n\n"), prevmessage.c_str(), prevfilename.c_str(), prevlinenum) );
            }
        }
    }

    int currlinenum = m_LineNum;
    wxString currmessage = m_Message;
    wxString currfilename = m_FileName;
    currmessage.Replace("\n"," // ");
    currfilename = currfilename.AfterLast('/');
    currfilename = currfilename.AfterLast('\\');

    fullmessage.Append( wxString::Format(_("%s\n    File: %s\n    Line: %d\n\n"), currmessage.c_str(), currfilename.c_str(), currlinenum) );

    return fullmessage;
}
