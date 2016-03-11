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
 
#include "asException.h"

asException::asException()
:
std::exception()
{
    m_message = wxEmptyString;
    m_fileName = wxEmptyString;
    m_lineNum = 0;
    m_hasChild = false;
    asLogError(_("An exception occured."));
}

asException::asException(const wxString &message, const char *filename, unsigned int line)
{
    wxString wxfilename(filename, wxConvUTF8);
    m_message = message;
    m_fileName = wxfilename;
    m_lineNum = line;
    m_hasChild = false;
    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_message, m_fileName, m_lineNum);
    asLogError(logMessage);
}

asException::asException(const std::string &message, const char *filename, unsigned int line)
{
    wxString wxmessage(message.c_str(), wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_message = wxmessage;
    m_fileName = wxfilename;
    m_fileName = wxString::FromAscii(filename);
    m_lineNum = line;
    m_hasChild = false;
    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_message, m_fileName, m_lineNum);
    asLogError(logMessage);
}

asException::asException(const char *message, const char *filename, unsigned int line)
{
    wxString wxmessage(message, wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_message = wxmessage;
    m_fileName = wxfilename;
    m_lineNum = line;
    m_hasChild = false;
    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_message, m_fileName, m_lineNum);
    asLogError(logMessage);
}

asException::asException(const wxString &message, const char *filename, unsigned int line, asException prevexception)
{
    wxString wxfilename(filename, wxConvUTF8);
    m_message = message;
    m_fileName = wxfilename;
    m_lineNum = line;
    m_hasChild = true;
    m_previous = prevexception.m_previous;

    PrevExceptions Previous;
    Previous.Message = prevexception.m_message;
    Previous.FileName = prevexception.m_fileName;
    Previous.LineNum = prevexception.m_lineNum;
    m_previous.push_back(&Previous);

    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_message, m_fileName, m_lineNum);
    asLogError(logMessage);
}

asException::asException(const std::string &message, const char *filename, unsigned int line, asException prevexception)
{
    wxString wxmessage(message.c_str(), wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_message = wxmessage;
    m_fileName = wxfilename;
    m_lineNum = line;
    m_hasChild = true;
    m_previous = prevexception.m_previous;

    PrevExceptions Previous;
    Previous.Message = prevexception.m_message;
    Previous.FileName = prevexception.m_fileName;
    Previous.LineNum = prevexception.m_lineNum;
    m_previous.push_back(&Previous);

    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_message, m_fileName, m_lineNum);
    asLogError(logMessage);
}

asException::asException(const char *message, const char *filename, unsigned int line, asException prevexception)
{
    wxString wxmessage(message, wxConvUTF8);
    wxString wxfilename(filename, wxConvUTF8);
    m_message = wxmessage;
    m_fileName = wxfilename;
    m_lineNum = line;
    m_hasChild = true;
    m_previous = prevexception.m_previous;

    PrevExceptions Previous;
    Previous.Message = prevexception.m_message;
    Previous.FileName = prevexception.m_fileName;
    Previous.LineNum = prevexception.m_lineNum;
    m_previous.push_back(&Previous);

    wxString logMessage;
    logMessage = wxString::Format(_("An exception occured: %s. File: %s (%d)"), m_message, m_fileName, m_lineNum);
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
        int prevnb = m_previous.size();

        for (int i=0; i<prevnb; i++)
        {
            int prevlinenum;
            wxString prevmessage;
            wxString prevfilename;

            prevlinenum = m_previous[i]->LineNum;
            prevmessage = m_previous[i]->Message;
            prevfilename = m_previous[i]->FileName;
            prevmessage.Replace("\n"," // ");
            prevfilename = prevfilename.AfterLast('/');
            prevfilename = prevfilename.AfterLast('\\');

            if(!prevmessage.IsEmpty() && !prevfilename.IsEmpty())
            {
                fullmessage.Append( wxString::Format(_("%s\n    File: %s\n    Line: %d\n\n"), prevmessage, prevfilename, prevlinenum) );
            }
        }
    }

    int currlinenum = m_lineNum;
    wxString currmessage = m_message;
    wxString currfilename = m_fileName;
    currmessage.Replace("\n"," // ");
    currfilename = currfilename.AfterLast('/');
    currfilename = currfilename.AfterLast('\\');

    fullmessage.Append( wxString::Format(_("%s\n    File: %s\n    Line: %d\n\n"), currmessage, currfilename, currlinenum) );

    return fullmessage;
}
