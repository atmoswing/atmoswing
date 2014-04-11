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
 
#include "asFileAscii.h"

asFileAscii::asFileAscii(const wxString &FileName, const ListFileMode &FileMode)
:
asFile(FileName, FileMode)
{

}

asFileAscii::~asFileAscii()
{
    //dtor
}

bool asFileAscii::Open()
{
    if (!Find()) return false;

    switch (m_FileMode)
    {
        case (ReadOnly):
            m_File.open(m_FileName.GetFullPath().mb_str(), fstream::in);
            break;

        case (Write):
            m_File.open(m_FileName.GetFullPath().mb_str(), fstream::out);
            break;

        case (Replace):
            m_File.open(m_FileName.GetFullPath().mb_str(), fstream::trunc | fstream::out);
            break;

        case (New):
            m_File.open(m_FileName.GetFullPath().mb_str(), fstream::out);
            break;

        case (Append):
            m_File.open(m_FileName.GetFullPath().mb_str(), fstream::app | fstream::out);
            break;
    }

    if (!m_File.is_open()) return false;

    m_Opened = true;

    return true;
}

bool asFileAscii::Close()
{
    wxASSERT(m_Opened);

    m_File.close();
    return true;
}

void asFileAscii::AddLineContent(const wxString &LineContent)
{
    wxASSERT(m_Opened);

    wxString LineContentCopy = LineContent;
    LineContentCopy.Append("\n");

    m_File <<  LineContentCopy.mb_str();

    // Check the state flags
    if (m_File.fail()) asThrowException(wxString::Format(_("An error occured while trying to write in file %s"), m_FileName.GetFullPath().c_str()));
}

const wxString asFileAscii::GetLineContent()
{
    wxASSERT(m_Opened);

    std::string tmpLineContent;

    if(!m_File.eof())
    {
         getline (m_File, tmpLineContent);

         // Check the state flags
         if ((!m_File.eof()) && (m_File.fail())) asThrowException(wxString::Format(_("An error occured while trying to write in file %s"), m_FileName.GetFullPath().c_str()));
    } else {
        asThrowException(wxString::Format(_("You are trying to read a line after the end of the file %s"), m_FileName.GetFullPath().c_str()));
    }

    wxString LineContent = wxString(tmpLineContent.c_str(), wxConvUTF8);

    return LineContent;
}

const wxString asFileAscii::GetFullContent()
{
    wxASSERT(m_Opened);

    std::string tmpContent;
    std::string apptmpContent;

    while (! m_File.eof() )
    {
        getline (m_File, tmpContent);
        apptmpContent.append(tmpContent);

        // Check the state flags
        if ((!m_File.eof()) && (m_File.fail())) asThrowException(wxString::Format(_("An error occured while trying to read in file %s"), m_FileName.GetFullPath().c_str()));
    }

    wxString Content(apptmpContent.c_str(), wxConvUTF8);

    return Content;
}

const wxString asFileAscii::GetFullContentWhithoutReturns()
{
    wxASSERT(m_Opened);

    std::string tmpContent;
    std::string apptmpContent;

    while (! m_File.eof() )
    {
        getline (m_File, tmpContent);
        apptmpContent.append(tmpContent);
    }

    // Check the state flags
    if ((!m_File.eof()) && (m_File.fail())) asThrowException(wxString::Format(_("An error occured while trying to read in file %s"), m_FileName.GetFullPath().c_str()));

    wxString Content(apptmpContent.c_str(), wxConvUTF8);

    return Content;
}

int asFileAscii::GetInt()
{
    wxASSERT(m_Opened);

    int tmp;
    m_File >> tmp;
    return tmp;
}

float asFileAscii::GetFloat()
{
    wxASSERT(m_Opened);

    float tmp;
    m_File >> tmp;
    return tmp;
}

double asFileAscii::GetDouble()
{
    wxASSERT(m_Opened);

    double tmp;
    m_File >> tmp;
    return tmp;
}

bool asFileAscii::SkipLines(int linesNb)
{
    wxASSERT(m_Opened);

    for(int i_line=0; i_line<linesNb; i_line++)
    {
        if (! m_File.eof() )
        {
            GetLineContent();
        }
        else
        {
            asLogError(_("Reached the end of the file while skipping lines."));
            return false;
        }
    }

    return true;
}

bool asFileAscii::SkipElements(int elementNb)
{
    wxASSERT(m_Opened);

    float tmp;

    for(int i_el=0; i_el<elementNb; i_el++)
    {
        if (! m_File.eof() )
        {
            m_File >> tmp;
        }
        else
        {
            asLogError(_("Reached the end of the file while skipping lines."));
            return false;
        }
    }

    return true;
}

bool asFileAscii::EndOfFile()
{
    wxASSERT(m_Opened);

    return m_File.eof();
}
