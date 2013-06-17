#include "asResultsParametersArray.h"

#include "wx/fileconf.h"

#include <asFileAscii.h>
#include <asParametersScoring.h>


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
    m_FilePath = wxFileConfig::Get()->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDefaultUserWorkingDir());
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
