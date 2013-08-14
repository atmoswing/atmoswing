#include "asFileParametersCalibration.h"

asFileParametersCalibration::asFileParametersCalibration(const wxString &FileName, const ListFileMode &FileMode)
:
asFileParameters(FileName, FileMode)
{
    // FindAndOpen() processed by asFileXml
}

asFileParametersCalibration::~asFileParametersCalibration()
{
    //dtor
}

bool asFileParametersCalibration::InsertRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile")) return false;
    if(!InsertElement(wxEmptyString, "CalibrationSet")) return false;
    if(!GoToFirstNodeWithPath("CalibrationSet")) return false;
    return true;
}

bool asFileParametersCalibration::GoToRootElement()
{
    if(!GoToFirstNodeWithPath("AtmoswingFile.CalibrationSet"))
    {
        asLogError(wxString::Format(_("The file %s is not an Atmoswing calibration parameters file."), m_FileName.GetFullName()));
        return false;
    }
    return true;
}
