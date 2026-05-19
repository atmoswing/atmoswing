/////////////////////////////////////////////////////////////////////////////
// Name:        plotcurv.cpp
// Purpose:     wxPlotCurve for wxPlotCtrl
// Author:      John Labenski
// Modified by:
// Created:     12/01/2000
// Copyright:   (c) John Labenski
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#if defined(__GNUG__) && !defined(NO_GCC_PRAGMA)
#pragma implementation "plotcurv.h"
#endif

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/bitmap.h"
#include "wx/dcmemory.h"

#endif  // WX_PRECOMP

#include "wx/plotctrl/plotcurv.h"

const wxRect2DDouble wxNullPlotBounds(0, 0, 0, 0);

const wxPlotCurve wxNullPlotCurve;

//----------------------------------------------------------------------------
// Interpolate
//----------------------------------------------------------------------------
double LinearInterpolateX(double x0, double y0, double x1, double y1, double y) {
    // wxCHECK_MSG( (y1 - y0) != 0.0, 0.0, wxT("Divide by zero, LinearInterpolateX()") );
    return ((y - y0) * (x1 - x0) / (y1 - y0) + x0);
}

double LinearInterpolateY(double x0, double y0, double x1, double y1, double x) {
    // wxCHECK_MSG( (x1 - x0) != 0.0, 0.0, wxT("Divide by zero, LinearInterpolateY()") );
    double m = (y1 - y0) / (x1 - x0);
    return (m * x + (y0 - m * x0));
}

//----------------------------------------------------------------------------
// wxPlotCurveRefData
//----------------------------------------------------------------------------

wxArrayGenericPen wxPlotCurveRefData::sm_defaultPens;

void InitPlotCurveDefaultPens() {
    static bool s_init_default_pens = false;
    if (!s_init_default_pens) {
        s_init_default_pens = true;
        wxPlotCurveRefData::sm_defaultPens.Add(wxGenericPen(wxGenericColour(0, 0, 0), 1, wxPENSTYLE_SOLID));
        wxPlotCurveRefData::sm_defaultPens.Add(wxGenericPen(wxGenericColour(0, 0, 255), 1, wxPENSTYLE_SOLID));
        wxPlotCurveRefData::sm_defaultPens.Add(wxGenericPen(wxGenericColour(255, 0, 0), 1, wxPENSTYLE_SOLID));
    }
}

wxPlotCurveRefData::wxPlotCurveRefData()
    : wxObjectRefData() {
    InitPlotCurveDefaultPens();
    _pens = sm_defaultPens;
}

wxPlotCurveRefData::wxPlotCurveRefData(const wxPlotCurveRefData& data)
    : wxObjectRefData() {
    Copy(data);
}

void wxPlotCurveRefData::Copy(const wxPlotCurveRefData& source) {
    _boundingRect = source._boundingRect;
    _pens = source._pens;
    _optionNames = source._optionNames;
    _optionValues = source._optionValues;
}

#define M_PLOTCURVEDATA ((wxPlotCurveRefData*)_refData)

//-----------------------------------------------------------------------------
// wxPlotCurve
//-----------------------------------------------------------------------------
IMPLEMENT_DYNAMIC_CLASS(wxPlotCurve, wxObject);

wxObjectRefData* wxPlotCurve::CreateRefData() const {
    return new wxPlotCurveRefData;
}

wxObjectRefData* wxPlotCurve::CloneRefData(const wxObjectRefData* data) const {
    return new wxPlotCurveRefData(*(const wxPlotCurveRefData*)data);
}

wxPlotCurve::wxPlotCurve()
    : wxObject() {
    // Note: You must do this in your constructor in order to use the the curve
    // _refData = new wxPlotCurveRefData (or wxMySubclassedPlotCurveRefData)
}

bool wxPlotCurve::Ok() const {
    return (M_PLOTCURVEDATA != NULL);
}

wxRect2DDouble wxPlotCurve::GetBoundingRect() const {
    wxCHECK_MSG(Ok(), wxNullPlotBounds, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_boundingRect;
}

void wxPlotCurve::SetBoundingRect(const wxRect2DDouble& rect) {
    wxCHECK_RET(Ok(), wxT("invalid plotcurve"));
    M_PLOTCURVEDATA->_boundingRect = rect;
}

//----------------------------------------------------------------------------
// Get/Set Pen
//----------------------------------------------------------------------------

wxGenericPen wxPlotCurve::GetPen(wxPlotPen_Type colour_type) const {
    wxCHECK_MSG(Ok(), wxGenericPen(), wxT("invalid plotcurve"));
    wxCHECK_MSG((colour_type >= 0) && (colour_type < (int)M_PLOTCURVEDATA->_pens.GetCount()), wxGenericPen(),
                wxT("invalid plot colour"));

    return M_PLOTCURVEDATA->_pens[colour_type];
}

void wxPlotCurve::SetPen(wxPlotPen_Type colour_type, const wxGenericPen& pen) {
    wxCHECK_RET(Ok(), wxT("invalid plotcurve"));
    wxCHECK_RET((colour_type >= 0) && (colour_type < (int)M_PLOTCURVEDATA->_pens.GetCount()),
                wxT("invalid plot colour"));

    M_PLOTCURVEDATA->_pens[colour_type] = pen;
}

wxGenericPen wxPlotCurve::GetDefaultPen(wxPlotPen_Type colour_type) {
    InitPlotCurveDefaultPens();
    wxCHECK_MSG((colour_type >= 0) && (colour_type < int(wxPlotCurveRefData::sm_defaultPens.GetCount())),
                wxGenericPen(), wxT("invalid plot colour"));
    return wxPlotCurveRefData::sm_defaultPens[colour_type];
}

void wxPlotCurve::SetDefaultPen(wxPlotPen_Type colour_type, const wxGenericPen& pen) {
    InitPlotCurveDefaultPens();
    wxCHECK_RET((colour_type >= 0) && (colour_type < int(wxPlotCurveRefData::sm_defaultPens.GetCount())),
                wxT("invalid plot colour"));
    wxPlotCurveRefData::sm_defaultPens[colour_type] = pen;
}

wxBrush wxPlotCurve::GetBrush() const {
    return M_PLOTCURVEDATA->_brush;
}

void wxPlotCurve::SetBrush(const wxBrush& brush) {
    M_PLOTCURVEDATA->_brush = brush;
}

// ----------------------------------------------------------------------------
// Get/Set Option names/values
// ----------------------------------------------------------------------------

size_t wxPlotCurve::GetOptionCount() const {
    wxCHECK_MSG(M_PLOTCURVEDATA, 0, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_optionNames.GetCount();
}

int wxPlotCurve::HasOption(const wxString& name) const {
    wxCHECK_MSG(M_PLOTCURVEDATA, wxNOT_FOUND, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_optionNames.Index(name);
}

wxString wxPlotCurve::GetOptionName(size_t i) const {
    wxCHECK_MSG(M_PLOTCURVEDATA && (i < GetOptionCount()), wxEmptyString, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_optionNames[i];
}

wxString wxPlotCurve::GetOptionValue(size_t i) const {
    wxCHECK_MSG(M_PLOTCURVEDATA && (i < GetOptionCount()), wxEmptyString, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_optionValues[i];
}

int wxPlotCurve::SetOption(const wxString& name, const wxString& value, bool update) {
    wxCHECK_MSG(M_PLOTCURVEDATA, -1, wxT("invalid plotcurve"));
    int n = M_PLOTCURVEDATA->_optionNames.Index(name);
    if (n == wxNOT_FOUND) {
        n = M_PLOTCURVEDATA->_optionNames.Add(name);
        M_PLOTCURVEDATA->_optionValues.Insert(value, n);
    } else if (update) {
        M_PLOTCURVEDATA->_optionNames[n] = name;
        M_PLOTCURVEDATA->_optionValues[n] = value;
    }
    return n;
}

int wxPlotCurve::SetOption(const wxString& name, int option, bool update) {
    return SetOption(name, wxString::Format(wxT("%d"), option), update);
}

wxString wxPlotCurve::GetOption(const wxString& name) const {
    wxCHECK_MSG(M_PLOTCURVEDATA, wxEmptyString, wxT("invalid plotcurve"));
    int n = M_PLOTCURVEDATA->_optionNames.Index(name);

    if (n == wxNOT_FOUND) return wxEmptyString;

    return M_PLOTCURVEDATA->_optionValues[n];
}

int wxPlotCurve::GetOption(const wxString& name, wxString& value) const {
    wxCHECK_MSG(M_PLOTCURVEDATA, wxNOT_FOUND, wxT("invalid plotcurve"));

    int n = M_PLOTCURVEDATA->_optionNames.Index(name);

    if (n == wxNOT_FOUND) return wxNOT_FOUND;

    value = M_PLOTCURVEDATA->_optionValues[n];
    return n;
}

int wxPlotCurve::GetOptionInt(const wxString& name) const {
    wxCHECK_MSG(M_PLOTCURVEDATA, 0, wxT("invalid plotcurve"));
    return wxAtoi(GetOption(name));
}

wxArrayString wxPlotCurve::GetOptionNames() const {
    wxCHECK_MSG(M_PLOTCURVEDATA, wxArrayString(), wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_optionNames;
}

wxArrayString wxPlotCurve::GetOptionValues() const {
    wxCHECK_MSG(M_PLOTCURVEDATA, wxArrayString(), wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->_optionValues;
}

//-------------------------------------------------------------------------

void wxPlotCurve::SetClientObject(wxClientData* data) {
    wxCHECK_RET(M_PLOTCURVEDATA, wxT("invalid plotcurve"));
    M_PLOTCURVEDATA->SetClientObject(data);
}

wxClientData* wxPlotCurve::GetClientObject() const {
    wxCHECK_MSG(M_PLOTCURVEDATA, NULL, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->GetClientObject();
}

void wxPlotCurve::SetClientData(void* data) {
    wxCHECK_RET(M_PLOTCURVEDATA, wxT("invalid plotcurve"));
    M_PLOTCURVEDATA->SetClientData(data);
}

void* wxPlotCurve::GetClientData() const {
    wxCHECK_MSG(M_PLOTCURVEDATA, NULL, wxT("invalid plotcurve"));
    return M_PLOTCURVEDATA->GetClientData();
}
