#include "asFileParametersOptimization.h"

asFileParametersOptimization::asFileParametersOptimization(const wxString &FileName, const ListFileMode &FileMode)
:
asFileParameters(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileParametersOptimization::~asFileParametersOptimization()
{
    //dtor
}

bool asFileParametersOptimization::InsertRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile")) return false;
    if(!InsertElement(wxEmptyString, "OptimizationSet")) return false;
    if(!GoToFirstNodeWithPath("OptimizationSet")) return false;
    return true;
}

bool asFileParametersOptimization::GoToRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile.OptimizationSet"))
    {
        asLogError(wxString::Format(_("The file %s is not an Atmoswing optimization parameters file."), m_FileName.GetFullName()));
        return false;
    }
    return true;
}
