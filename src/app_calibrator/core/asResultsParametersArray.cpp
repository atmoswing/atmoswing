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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#include "asResultsParametersArray.h"

#include "wx/fileconf.h"

#include <asFileAscii.h>


asResultsParametersArray::asResultsParametersArray()
:
asResults()
{

}

asResultsParametersArray::~asResultsParametersArray()
{
    //dtor
}

void asResultsParametersArray::Init(const wxString &fileTag)
{
    BuildFileName(fileTag);

    // Resize to 0 to avoid keeping old results
    m_Parameters.resize(0);
    m_ScoresCalib.resize(0);
    m_ScoresValid.resize(0);
}

void asResultsParametersArray::BuildFileName(const wxString &fileTag)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_FilePath = wxFileConfig::Get()->Read("/Paths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
    m_FilePath.Append(wxString::Format("/Calibration/%s_%s.txt", time.c_str(), fileTag.c_str()));
}

void asResultsParametersArray::Add(asParametersScoring params, float scoreCalib)
{
    m_Parameters.push_back(params);
    m_ScoresCalib.push_back(scoreCalib);
    m_ScoresValid.push_back(NaNFloat);
}

void asResultsParametersArray::Add(asParametersScoring params, float scoreCalib, float scoreValid)
{
    m_Parameters.push_back(params);
    m_ScoresCalib.push_back(scoreCalib);
    m_ScoresValid.push_back(scoreValid);
}

void asResultsParametersArray::Add(asParametersScoring params, Array1DFloat scoreCalib, Array1DFloat scoreValid)
{
    m_ParametersForScoreOnArray.push_back(params);
    m_ScoresCalibForScoreOnArray.push_back(scoreCalib);
    m_ScoresValidForScoreOnArray.push_back(scoreValid);
}

void asResultsParametersArray::Clear()
{
    // Resize to 0 to avoid keeping old results
    m_Parameters.resize(0);
    m_ScoresCalib.resize(0);
    m_ScoresValid.resize(0);
}

bool asResultsParametersArray::Print()
{
    // Create a file
    asFileAscii fileRes(m_FilePath, asFileAscii::Replace);
    if(!fileRes.Open()) return false;

    wxString header;
    header = _("Optimization processed ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (unsigned int i_param=0; i_param<m_Parameters.size(); i_param++)
    {
        content.Append(m_Parameters[i_param].Print());
        content.Append(wxString::Format("Calib\t%e\t", m_ScoresCalib[i_param]));
        content.Append(wxString::Format("Valid\t%e", m_ScoresValid[i_param]));
        content.Append("\n");
    }
    
    // Write every parameter for scores on array one after the other
    for (unsigned int i_param=0; i_param<m_ParametersForScoreOnArray.size(); i_param++)
    {
        content.Append(m_ParametersForScoreOnArray[i_param].Print());
        content.Append("Calib\t");
        for (unsigned int i_row=0; i_row<m_ScoresCalibForScoreOnArray[i_param].size(); i_row++)
        {
            content.Append(wxString::Format("%e\t", m_ScoresCalibForScoreOnArray[i_param][i_row]));
        }
        content.Append("Valid\t");
        for (unsigned int i_row=0; i_row<m_ScoresValidForScoreOnArray[i_param].size(); i_row++)
        {
            content.Append(wxString::Format("%e\t", m_ScoresValidForScoreOnArray[i_param][i_row]));
        }
        content.Append("\n");
    }

    fileRes.AddLineContent(content);

    fileRes.Close();

    return true;
}

void asResultsParametersArray::CreateFile()
{
    // Create a file
    asFileAscii fileRes(m_FilePath, asFileAscii::Replace);
    fileRes.Open();

    wxString header;
    header = _("Optimization processed ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);

    fileRes.Close();
}

bool asResultsParametersArray::AppendContent()
{
    // Create a file
    asFileAscii fileRes(m_FilePath, asFileAscii::Append);
    if(!fileRes.Open()) return false;

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (unsigned int i_param=0; i_param<m_Parameters.size(); i_param++)
    {
        content.Append(wxString::Format("Param(%d)\t", i_param));
        content.Append(m_Parameters[i_param].Print());
        content.Append(wxString::Format("Score calib\t%e\t", m_ScoresCalib[i_param]));
        content.Append(wxString::Format("valid\t%e", m_ScoresValid[i_param]));
        content.Append("\n");
    }

    fileRes.AddLineContent(content);

    fileRes.Close();

    return true;
}
