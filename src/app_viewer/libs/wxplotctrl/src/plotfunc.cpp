/////////////////////////////////////////////////////////////////////////////
// Name:        plotfunc.cpp
// Purpose:     wxPlotFunction curve for wxPlotCtrl
// Author:      John Labenski
// Modified by:
// Created:     12/01/2000
// Copyright:   (c) John Labenski
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#if defined(__GNUG__) && !defined(NO_GCC_PRAGMA)
#pragma implementation "plotfunc.h"
#endif

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/bitmap.h"
#include "wx/dataobj.h"
#include "wx/dcmemory.h"
#include "wx/msgdlg.h"
#include "wx/textdlg.h"

#endif  // WX_PRECOMP

#include <math.h>

#include "wx/plotctrl/fparser.h"
#include "wx/plotctrl/plotfunc.h"

const wxPlotFunction wxNullPlotFunction;

//----------------------------------------------------------------------------
// wxPlotFuncRefData
//----------------------------------------------------------------------------

class wxPlotFuncRefData : public wxPlotCurveRefData {
  public:
    wxPlotFuncRefData()
        : wxPlotCurveRefData() {}

    wxPlotFuncRefData(const wxPlotFuncRefData& data);

    wxFunctionParser _parser;
};

wxPlotFuncRefData::wxPlotFuncRefData(const wxPlotFuncRefData& data)
    : wxPlotCurveRefData() {
    wxPlotCurveRefData::Copy(data);
    _parser = data._parser;
}

#define M_PLOTFUNCDATA ((wxPlotFuncRefData*)_refData)

//-----------------------------------------------------------------------------
// wxPlotFunction
//-----------------------------------------------------------------------------
IMPLEMENT_DYNAMIC_CLASS(wxPlotFunction, wxPlotCurve);

wxObjectRefData* wxPlotFunction::CreateRefData() const {
    return new wxPlotFuncRefData;
}

wxObjectRefData* wxPlotFunction::CloneRefData(const wxObjectRefData* data) const {
    return new wxPlotFuncRefData(*(const wxPlotFuncRefData*)data);
}

bool wxPlotFunction::Create(const wxPlotFunction& curve) {
    wxCHECK_MSG(curve.Ok(), false, wxT("invalid plot function"));
    UnRef();
    Ref(curve);
    return true;
}

void wxPlotFunction::Destroy() {
    UnRef();
}

bool wxPlotFunction::Ok() const {
    return _refData && M_PLOTFUNCDATA->_parser.Ok();
}

int wxPlotFunction::Create(const wxString& function, const wxString& vars, bool useDegrees) {
    UnRef();

    _refData = new wxPlotFuncRefData();
    wxCHECK_MSG(_refData, 0, wxT("can't allocate memory"));

    int i = M_PLOTFUNCDATA->_parser.Parse(function, vars, useDegrees);

    if (!M_PLOTFUNCDATA->_parser.ErrorMsg().IsEmpty()) return i;

    return -1;
}

int wxPlotFunction::Parse(const wxString& function, const wxString& vars, bool useDegrees) {
    wxCHECK_MSG(_refData, 0, wxT("Invalid plotfunction"));

    int i = M_PLOTFUNCDATA->_parser.Parse(function, vars, useDegrees);

    if (!M_PLOTFUNCDATA->_parser.ErrorMsg().IsEmpty()) return i;

    return -1;
}

wxString wxPlotFunction::GetFunctionString() const {
    wxCHECK_MSG(Ok(), wxEmptyString, wxT("invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.GetFunctionString();
}

wxString wxPlotFunction::GetVariableString() const {
    wxCHECK_MSG(Ok(), wxEmptyString, wxT("invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.GetVariableString();
}

wxString wxPlotFunction::GetVariableName(size_t n) const {
    wxCHECK_MSG(Ok(), wxEmptyString, wxT("invalid plotfunction"));
    wxCHECK_MSG((int(n) < GetNumberVariables()), wxEmptyString, wxT("invalid variable index"));
    return M_PLOTFUNCDATA->_parser.GetVariableName(n);
}

int wxPlotFunction::GetNumberVariables() const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.GetNumberVariables();
}

bool wxPlotFunction::GetUseDegrees() const {
    wxCHECK_MSG(_refData, false, wxT("Invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.GetUseDegrees();
}

wxString wxPlotFunction::GetErrorMsg() const {
    wxCHECK_MSG(_refData, wxEmptyString, wxT("Invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.ErrorMsg();
}

double wxPlotFunction::GetY(double x) const {
    wxCHECK_MSG(Ok(), 0.0, wxT("invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.Eval(&x);
}

double wxPlotFunction::GetValue(double* x) const {
    wxCHECK_MSG(Ok(), 0.0, wxT("invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.Eval(x);
}

bool wxPlotFunction::AddConstant(const wxString& name, double value) {
    wxCHECK_MSG(Ok(), false, wxT("invalid plotfunction"));
    return M_PLOTFUNCDATA->_parser.AddConstant(name, value);
}

//-----------------------------------------------------------------------------
// wxClipboardGet/SetPlotFunction
//-----------------------------------------------------------------------------

#include "wx/clipbrd.h"

#if wxUSE_DATAOBJ && wxUSE_CLIPBOARD

wxPlotFunction wxClipboardGetPlotFunction() {
    bool is_opened = wxTheClipboard->IsOpened();
    wxPlotFunction plotFunc;

    if (is_opened || wxTheClipboard->Open()) {
        wxTextDataObject textDataObject;
        if (wxTheClipboard->IsSupported(wxDataFormat(wxDF_TEXT)) && wxTheClipboard->GetData(textDataObject)) {
            wxString str = textDataObject.GetText();
            plotFunc.Create(str.BeforeLast(wxT(';')), str.AfterLast(wxT(';')));
        }

        if (!is_opened) wxTheClipboard->Close();
    }

    return plotFunc;
}

bool wxClipboardSetPlotFunction(const wxPlotFunction& plotFunc) {
    wxCHECK_MSG(plotFunc.Ok(), false, wxT("Invalid wxPlotFunction to copy to clipboard"));
    bool is_opened = wxTheClipboard->IsOpened();

    if (is_opened || wxTheClipboard->Open()) {
        wxString str = plotFunc.GetFunctionString() + wxT(";") + plotFunc.GetVariableString();
        wxTextDataObject* textDataObject = new wxTextDataObject(str);
        wxTheClipboard->SetData(textDataObject);

        if (!is_opened) wxTheClipboard->Close();

        return true;
    }

    return false;
}

#endif  // wxUSE_DATAOBJ && wxUSE_CLIPBOARD
