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
    m_parameters.resize(0);
    m_scoresCalib.resize(0);
    m_scoresValid.resize(0);
}

void asResultsParametersArray::BuildFileName(const wxString &fileTag)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/OptimizerResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), concentrate);
    m_filePath.Append(wxString::Format("/Optimizer/%s_%s.txt", time, fileTag));
}

void asResultsParametersArray::Add(asParametersScoring params, float scoreCalib)
{
    m_parameters.push_back(params);
    m_scoresCalib.push_back(scoreCalib);
    m_scoresValid.push_back(NaNFloat);
}

void asResultsParametersArray::Add(asParametersScoring params, float scoreCalib, float scoreValid)
{
    m_parameters.push_back(params);
    m_scoresCalib.push_back(scoreCalib);
    m_scoresValid.push_back(scoreValid);
}

void asResultsParametersArray::Add(asParametersScoring params, Array1DFloat scoreCalib, Array1DFloat scoreValid)
{
    m_parametersForScoreOnArray.push_back(params);
    m_scoresCalibForScoreOnArray.push_back(scoreCalib);
    m_scoresValidForScoreOnArray.push_back(scoreValid);
}

void asResultsParametersArray::Clear()
{
    // Resize to 0 to avoid keeping old results
    m_parameters.resize(0);
    m_scoresCalib.resize(0);
    m_scoresValid.resize(0);
}

bool asResultsParametersArray::Print()
{
    // Create a file
    asFileAscii fileRes(m_filePath, asFileAscii::Replace);
    if(!fileRes.Open()) return false;

    wxString header;
    header = _("Optimization processed ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (unsigned int i_param=0; i_param<m_parameters.size(); i_param++)
    {
        content.Append(m_parameters[i_param].Print());
        content.Append(wxString::Format("Calib\t%e\t", m_scoresCalib[i_param]));
        content.Append(wxString::Format("Valid\t%e", m_scoresValid[i_param]));
        content.Append("\n");
    }
    
    // Write every parameter for scores on array one after the other
    for (unsigned int i_param=0; i_param<m_parametersForScoreOnArray.size(); i_param++)
    {
        content.Append(m_parametersForScoreOnArray[i_param].Print());
        content.Append("Calib\t");
        for (unsigned int i_row=0; i_row<m_scoresCalibForScoreOnArray[i_param].size(); i_row++)
        {
            content.Append(wxString::Format("%e\t", m_scoresCalibForScoreOnArray[i_param][i_row]));
        }
        content.Append("Valid\t");
        for (unsigned int i_row=0; i_row<m_scoresValidForScoreOnArray[i_param].size(); i_row++)
        {
            content.Append(wxString::Format("%e\t", m_scoresValidForScoreOnArray[i_param][i_row]));
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
    asFileAscii fileRes(m_filePath, asFileAscii::Replace);
    fileRes.Open();

    wxString header;
    header = _("Optimization processed ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);

    fileRes.Close();
}

bool asResultsParametersArray::AppendContent()
{
    // Create a file
    asFileAscii fileRes(m_filePath, asFileAscii::Append);
    if(!fileRes.Open()) return false;

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (unsigned int i_param=0; i_param<m_parameters.size(); i_param++)
    {
        content.Append(wxString::Format("Param(%d)\t", i_param));
        content.Append(m_parameters[i_param].Print());
        content.Append(wxString::Format("Score calib\t%e\t", m_scoresCalib[i_param]));
        content.Append(wxString::Format("valid\t%e", m_scoresValid[i_param]));
        content.Append("\n");
    }

    fileRes.AddLineContent(content);

    fileRes.Close();

    return true;
}
