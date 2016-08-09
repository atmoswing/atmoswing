#include "asFileParametersOptimization.h"

asFileParametersOptimization::asFileParametersOptimization(const wxString &FileName, const ListFileMode &FileMode)
        : asFileParameters(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileParametersOptimization::~asFileParametersOptimization()
{
    //dtor
}

bool asFileParametersOptimization::EditRootElement()
{
    if (!GetRoot())
        return false;
    GetRoot()->AddAttribute("target", "calibrator");
    return true;
}

bool asFileParametersOptimization::CheckRootElement() const
{
    if (!GetRoot())
        return false;
    if (!IsAnAtmoSwingFile())
        return false;
    if (!FileVersionIsOrAbove(1.0))
        return false;

    if (!GetRoot()->GetAttribute("target").IsSameAs("calibrator", false)) {
        asLogError(wxString::Format(_("The file %s is not a parameters file for the Calibrator."),
                                    m_fileName.GetFullName()));
        return false;
    }
    return true;
}
