/////////////////////////////////////////////////////////////////////////////
// Name:        genergdi.cpp
// Purpose:     Generic gdi pen and colour
// Author:      John Labenski
// Modified by:
// Created:     12/01/2000
// Copyright:   (c) John Labenski
// Licence:     wxWidgets license
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#include "wx/bitmap.h"
#include "wx/things/genergdi.h"
#include "wx/tokenzr.h"

const wxGenericColour wxNullGenericColour;
const wxGenericPen wxNullGenericPen;
const wxGenericBrush wxNullGenericBrush;

#include "wx/arrimpl.cpp"
WX_DEFINE_OBJARRAY(wxArrayGenericColour)

WX_DEFINE_OBJARRAY(wxArrayGenericPen)

WX_DEFINE_OBJARRAY(wxArrayGenericBrush)

//----------------------------------------------------------------------------
// wxGenericColour
//----------------------------------------------------------------------------
IMPLEMENT_DYNAMIC_CLASS(wxGenericColour, wxObject)

class wxGenericColourRefData : public wxObjectRefData {
  public:
    wxGenericColourRefData(unsigned char r = 0, unsigned char g = 0, unsigned char b = 0, unsigned char a = 255)
        : wxObjectRefData(),
          _r(r),
          _g(g),
          _b(b),
          _a(a) {}

    wxGenericColourRefData(const wxGenericColourRefData& data)
        : wxObjectRefData(),
          _r(data._r),
          _g(data._g),
          _b(data._b),
          _a(data._a) {}

    unsigned char _r, _g, _b, _a;
};

#define M_GCOLOURDATA ((wxGenericColourRefData*)_refData)

//----------------------------------------------------------------------------
wxObjectRefData* wxGenericColour::CreateRefData() const {
    return new wxGenericColourRefData;
}

wxObjectRefData* wxGenericColour::CloneRefData(const wxObjectRefData* data) const {
    return new wxGenericColourRefData(*(const wxGenericColourRefData*)data);
}

void wxGenericColour::Create(const wxGenericColour& c) {
    Ref(c);
}

void wxGenericColour::Create(const wxColour& c) {
    UnRef();
    _refData = new wxGenericColourRefData;
    Set(c);
}

void wxGenericColour::Create(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha) {
    UnRef();
    _refData = new wxGenericColourRefData(red, green, blue, alpha);
}

void wxGenericColour::CreateABGR(unsigned long colABGR) {
    UnRef();
    _refData = new wxGenericColourRefData;
    SetABGR(colABGR);
}

void wxGenericColour::CreateARGB(unsigned long colARGB) {
    UnRef();
    _refData = new wxGenericColourRefData;
    SetARGB(colARGB);
}

void wxGenericColour::Create(const wxString& colourName) {
    UnRef();
    _refData = new wxGenericColourRefData;
    Set(colourName);
}

void wxGenericColour::Set(const wxGenericColour& c) {
    wxCHECK_RET(Ok() && c.Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_r = c.GetRed();
    M_GCOLOURDATA->_g = c.GetGreen();
    M_GCOLOURDATA->_b = c.GetBlue();
    M_GCOLOURDATA->_a = c.GetAlpha();
}

void wxGenericColour::Set(const wxColour& c) {
    wxCHECK_RET(Ok() && c.Ok(), wxT("Invalid colour"));
    M_GCOLOURDATA->_r = c.Red();
    M_GCOLOURDATA->_g = c.Green();
    M_GCOLOURDATA->_b = c.Blue();
}

void wxGenericColour::Set(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_r = red;
    M_GCOLOURDATA->_g = green;
    M_GCOLOURDATA->_b = blue;
    M_GCOLOURDATA->_a = alpha;
}

void wxGenericColour::SetABGR(unsigned long colABGR) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_r = (unsigned char)(0xFF & colABGR);
    M_GCOLOURDATA->_g = (unsigned char)(0xFF & (colABGR >> 8));
    M_GCOLOURDATA->_b = (unsigned char)(0xFF & (colABGR >> 16));
    M_GCOLOURDATA->_a = (unsigned char)(0xFF & (colABGR >> 24));
}

void wxGenericColour::SetARGB(unsigned long colARGB) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_b = (unsigned char)(0xFF & colARGB);
    M_GCOLOURDATA->_g = (unsigned char)(0xFF & (colARGB >> 8));
    M_GCOLOURDATA->_r = (unsigned char)(0xFF & (colARGB >> 16));
    M_GCOLOURDATA->_a = (unsigned char)(0xFF & (colARGB >> 24));
}

void wxGenericColour::Set(const wxString& colourName) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    Set(wxColour(colourName));
}

void wxGenericColour::SetRed(unsigned char r) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_r = r;
}

void wxGenericColour::SetGreen(unsigned char g) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_g = g;
}

void wxGenericColour::SetBlue(unsigned char b) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_b = b;
}

void wxGenericColour::SetAlpha(unsigned char a) {
    wxCHECK_RET(Ok(), wxT("Invalid generic colour"));
    M_GCOLOURDATA->_a = a;
}

unsigned char wxGenericColour::GetRed() const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid generic colour"));
    return M_GCOLOURDATA->_r;
}

unsigned char wxGenericColour::GetGreen() const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid generic colour"));
    return M_GCOLOURDATA->_g;
}

unsigned char wxGenericColour::GetBlue() const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid generic colour"));
    return M_GCOLOURDATA->_b;
}

unsigned char wxGenericColour::GetAlpha() const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid generic colour"));
    return M_GCOLOURDATA->_a;
}

bool wxGenericColour::IsSameAs(const wxGenericColour& c) const {
    wxCHECK_MSG(Ok() && c.Ok(), false, wxT("Invalid generic colour"));
    wxGenericColourRefData* cData = (wxGenericColourRefData*)c.GetRefData();
    return (M_GCOLOURDATA->_r == cData->_r) && (M_GCOLOURDATA->_g == cData->_g) &&
           (M_GCOLOURDATA->_b == cData->_b) && (M_GCOLOURDATA->_a == cData->_a);
}

bool wxGenericColour::IsSameAs(const wxColour& c) const {
    wxCHECK_MSG(Ok() && c.Ok(), false, wxT("Invalid colour"));
    return (M_GCOLOURDATA->_r == c.Red()) && (M_GCOLOURDATA->_g == c.Green()) && (M_GCOLOURDATA->_b == c.Blue());
}

// This code is assumed to be public domain, originally from Paul Bourke, July 1996
// http://astronomy.swin.edu.au/~pbourke/colour/colourramp/source1.c

wxGenericColour wxGenericColour::GetHotColdColour(double v) const {
    wxGenericColour c(255, 255, 255);
    const double vmin = 0.0, vmax = 255.0, dv = vmax - vmin;

    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;

    if (v < (vmin + 0.25 * dv)) {
        c.SetRed(0);
        c.SetGreen(int(vmax * (4.0 * (v - vmin) / dv) + 0.5));
    } else if (v < (vmin + 0.5 * dv)) {
        c.SetRed(0);
        c.SetBlue(int(vmax * (1.0 + 4.0 * (vmin + 0.25 * dv - v) / dv) + 0.5));
    } else if (v < (vmin + 0.75 * dv)) {
        c.SetRed(int(vmax * (4.0 * (v - vmin - 0.5 * dv) / dv) + 0.5));
        c.SetBlue(0);
    } else {
        c.SetGreen(int(vmax * (1.0 + 4.0 * (vmin + 0.75 * dv - v) / dv) + 0.5));
        c.SetBlue(0);
    }

    return c;
}


//----------------------------------------------------------------------------
// wxGenericPen
//----------------------------------------------------------------------------
IMPLEMENT_DYNAMIC_CLASS(wxGenericPen, wxObject)

class wxGenericPenRefData : public wxObjectRefData {
  public:
    wxGenericPenRefData(int width = 1, wxPenStyle style = wxPENSTYLE_SOLID, wxPenCap cap = wxCAP_ROUND,
                        wxPenJoin join = wxJOIN_ROUND)
        : wxObjectRefData(),
          _width(width),
          _style(style),
          _cap(cap),
          _join(join),
          _dash_count(0),
          _dash(NULL) {}

    wxGenericPenRefData(const wxGenericPenRefData& data)
        : wxObjectRefData(),
          _colour(data._colour),
          _width(data._width),
          _style(data._style),
          _cap(data._cap),
          _join(data._join),
          _dash_count(data._dash_count),
          _dash(NULL) {
        if (data._dash) {
            _dash = (wxDash*)malloc(_dash_count * sizeof(wxDash));
            memcpy(_dash, data._dash, _dash_count * sizeof(wxDash));
        }
    }

    ~wxGenericPenRefData() {
        if (_dash) free(_dash);
    }

    wxGenericColour _colour;
    int _width;
    wxPenStyle _style;
    wxPenCap _cap;
    wxPenJoin _join;

    int _dash_count;  // don't arbitrarily adjust these!
    wxDash* _dash;
};

#define M_GPENDATA ((wxGenericPenRefData*)_refData)

//----------------------------------------------------------------------------
wxObjectRefData* wxGenericPen::CreateRefData() const {
    return new wxGenericPenRefData;
}

wxObjectRefData* wxGenericPen::CloneRefData(const wxObjectRefData* data) const {
    return new wxGenericPenRefData(*(const wxGenericPenRefData*)data);
}

void wxGenericPen::Create(const wxGenericPen& pen) {
    Ref(pen);
}

void wxGenericPen::Create(const wxPen& pen) {
    UnRef();
    _refData = new wxGenericPenRefData;
    Set(pen);
}

void wxGenericPen::Create(const wxGenericColour& colour, int width, wxPenStyle style, wxPenCap cap, wxPenJoin join) {
    UnRef();
    _refData = new wxGenericPenRefData(width, style, cap, join);
    M_GPENDATA->_colour = colour;
}

void wxGenericPen::Create(const wxColour& colour, int width, wxPenStyle style, wxPenCap cap, wxPenJoin join) {
    Create(wxGenericColour(colour), width, style, cap, join);
}

void wxGenericPen::Set(const wxGenericPen& pen) {
    wxCHECK_RET(Ok() && pen.Ok(), wxT("Invalid generic pen"));
    SetColour(pen.GetColour());
    M_GPENDATA->_width = pen.GetWidth();
    M_GPENDATA->_style = pen.GetStyle();
    M_GPENDATA->_cap = pen.GetCap();
    M_GPENDATA->_join = pen.GetJoin();

    wxDash* dash;
    int n_dashes = pen.GetDashes(&dash);
    SetDashes(n_dashes, dash);
}

void wxGenericPen::Set(const wxPen& pen) {
    wxCHECK_RET(Ok() && pen.Ok(), wxT("Invalid generic pen"));
    SetColour(pen.GetColour());
    M_GPENDATA->_width = pen.GetWidth();
    M_GPENDATA->_style = pen.GetStyle();
    M_GPENDATA->_cap = pen.GetCap();
    M_GPENDATA->_join = pen.GetJoin();

    wxDash* dash;
    int n_dashes = pen.GetDashes(&dash);
    SetDashes(n_dashes, dash);

    // or SetDashes(pen.GetDashCount(), pen.GetDash()); not in msw 2.4
}

void wxGenericPen::SetColour(const wxGenericColour& colour) {
    wxCHECK_RET(Ok() && colour.Ok(), wxT("Invalid generic pen or colour"));
    M_GPENDATA->_colour = colour;
}

void wxGenericPen::SetColour(const wxColour& colour) {
    SetColour(wxGenericColour(colour));
}

void wxGenericPen::SetColour(int red, int green, int blue, int alpha) {
    SetColour(wxGenericColour(red, green, blue, alpha));
}

void wxGenericPen::SetCap(wxPenCap capStyle) {
    wxCHECK_RET(Ok(), wxT("Invalid generic pen"));
    M_GPENDATA->_cap = capStyle;
}

void wxGenericPen::SetJoin(wxPenJoin joinStyle) {
    wxCHECK_RET(Ok(), wxT("Invalid generic pen"));
    M_GPENDATA->_join = joinStyle;
}

void wxGenericPen::SetStyle(wxPenStyle style) {
    wxCHECK_RET(Ok(), wxT("Invalid generic pen"));
    M_GPENDATA->_style = style;
}

void wxGenericPen::SetWidth(int width) {
    wxCHECK_RET(Ok(), wxT("Invalid generic pen"));
    M_GPENDATA->_width = width;
}

void wxGenericPen::SetDashes(int number_of_dashes, const wxDash* dash) {
    wxCHECK_RET(Ok(), wxT("Invalid generic pen"));
    wxCHECK_RET(((number_of_dashes == 0) && !dash) || ((number_of_dashes > 0) && dash), wxT("Invalid dashes for pen"));

    // internal double check to see if somebody's messed with this
    // wxCHECK_RET(((M_GPENDATA->_dash_count == 0) && !M_GPENDATA->_dash) ||
    //            ((M_GPENDATA->_dash_count != 0) &&  M_GPENDATA->_dash), wxT("Invalid internal dashes for pen"));

    if (M_GPENDATA->_dash) {
        free(M_GPENDATA->_dash);
        M_GPENDATA->_dash = NULL;
        M_GPENDATA->_dash_count = 0;
    }

    if (!dash) return;

    M_GPENDATA->_dash_count = number_of_dashes;
    M_GPENDATA->_dash = (wxDash*)malloc(number_of_dashes * sizeof(wxDash));
    memcpy(M_GPENDATA->_dash, dash, number_of_dashes * sizeof(wxDash));
}

wxPen wxGenericPen::GetPen() const {
    wxCHECK_MSG(Ok(), wxNullPen, wxT("Invalid generic pen"));
    wxPen pen(M_GPENDATA->_colour.GetColour(), M_GPENDATA->_width, M_GPENDATA->_style);
    pen.SetCap(M_GPENDATA->_cap);
    pen.SetJoin(M_GPENDATA->_join);
    if (M_GPENDATA->_dash_count > 0) pen.SetDashes(M_GPENDATA->_dash_count, M_GPENDATA->_dash);

    return pen;
}

wxGenericColour wxGenericPen::GetGenericColour() const {
    wxCHECK_MSG(Ok(), wxNullGenericColour, wxT("Invalid generic pen"));
    return M_GPENDATA->_colour;
}

wxColour wxGenericPen::GetColour() const {
    wxCHECK_MSG(Ok(), wxNullColour, wxT("Invalid generic pen"));
    return M_GPENDATA->_colour.GetColour();
}

int wxGenericPen::GetWidth() const {
    wxCHECK_MSG(Ok(), 1, wxT("Invalid generic pen"));
    return M_GPENDATA->_width;
}

wxPenStyle wxGenericPen::GetStyle() const {
    wxCHECK_MSG(Ok(), wxPENSTYLE_SOLID, wxT("Invalid generic pen"));
    return M_GPENDATA->_style;
}

wxPenCap wxGenericPen::GetCap() const {
    wxCHECK_MSG(Ok(), wxCAP_ROUND, wxT("Invalid generic pen"));
    return M_GPENDATA->_cap;
}

wxPenJoin wxGenericPen::GetJoin() const {
    wxCHECK_MSG(Ok(), wxJOIN_ROUND, wxT("Invalid generic pen"));
    return M_GPENDATA->_join;
}

int wxGenericPen::GetDashes(wxDash** ptr) const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid generic pen"));
    *ptr = (wxDash*)M_GPENDATA->_dash;
    return M_GPENDATA->_dash_count;
}

int wxGenericPen::GetDashCount() const {
    wxCHECK_MSG(Ok(), 0, wxT("Invalid generic pen"));
    return M_GPENDATA->_dash_count;
}

wxDash* wxGenericPen::GetDash() const {
    wxCHECK_MSG(Ok(), NULL, wxT("Invalid generic pen"));
    return M_GPENDATA->_dash;
}

bool wxGenericPen::IsSameAs(const wxGenericPen& pen) const {
    wxCHECK_MSG(Ok() && pen.Ok(), false, wxT("Invalid generic pen"));
    auto pData = (wxGenericPenRefData*)pen.GetRefData();

    if ((M_GPENDATA->_colour != pData->_colour) || (M_GPENDATA->_width != pData->_width) ||
        (M_GPENDATA->_style != pData->_style) || (M_GPENDATA->_cap != pData->_cap) ||
        (M_GPENDATA->_join != pData->_join) || (M_GPENDATA->_dash_count != pen.GetDashCount()))
        return false;

    if (M_GPENDATA->_dash_count > 0)
        return memcmp(M_GPENDATA->_dash, pen.GetDash(), M_GPENDATA->_dash_count * sizeof(wxDash)) == 0;

    return true;
}

bool wxGenericPen::IsSameAs(const wxPen& pen) const {
    wxCHECK_MSG(Ok() && pen.Ok(), false, wxT("Invalid generic pen"));
    wxGenericPen gp(pen);
    gp.GetGenericColour().SetAlpha(M_GPENDATA->_colour.GetAlpha());
    return IsSameAs(gp);
}


//----------------------------------------------------------------------------
// wxGenericBrush
//----------------------------------------------------------------------------
IMPLEMENT_DYNAMIC_CLASS(wxGenericBrush, wxObject)

class wxGenericBrushRefData : public wxObjectRefData {
  public:
    wxGenericBrushRefData(const wxGenericColour& c = wxNullGenericColour, wxBrushStyle style = wxBRUSHSTYLE_SOLID)
        : wxObjectRefData(),
          _colour(c),
          _style(style) {}

    wxGenericBrushRefData(const wxGenericBrushRefData& data)
        : wxObjectRefData(),
          _colour(data._colour),
          _style(data._style),
          _stipple(data._stipple) {}

    ~wxGenericBrushRefData() {}

    wxGenericColour _colour;
    wxBrushStyle _style;
    wxBitmap _stipple;
};

#define M_GBRUSHDATA ((wxGenericBrushRefData*)_refData)

//----------------------------------------------------------------------------
wxObjectRefData* wxGenericBrush::CreateRefData() const {
    return new wxGenericBrushRefData;
}

wxObjectRefData* wxGenericBrush::CloneRefData(const wxObjectRefData* data) const {
    return new wxGenericBrushRefData(*(const wxGenericBrushRefData*)data);
}

void wxGenericBrush::Create(const wxGenericBrush& brush) {
    Ref(brush);
}

void wxGenericBrush::Create(const wxBrush& brush) {
    UnRef();
    _refData = new wxGenericBrushRefData;
    Set(brush);
}

void wxGenericBrush::Create(const wxGenericColour& colour, wxBrushStyle style) {
    UnRef();
    _refData = new wxGenericBrushRefData(colour, style);
}

void wxGenericBrush::Create(const wxColour& colour, wxBrushStyle style) {
    Create(wxGenericColour(colour), style);
}

void wxGenericBrush::Create(const wxBitmap& stipple) {
    UnRef();
    wxCHECK_RET(stipple.Ok(), wxT("Invalid bitmap in wxGenericBrush::Create"));

    wxBrushStyle style = stipple.GetMask() ? wxBRUSHSTYLE_STIPPLE_MASK_OPAQUE : wxBRUSHSTYLE_STIPPLE;
    _refData = new wxGenericBrushRefData(wxNullGenericColour, style);
    M_GBRUSHDATA->_stipple = stipple;
}

void wxGenericBrush::Set(const wxGenericBrush& brush) {
    wxCHECK_RET(Ok() && brush.Ok(), wxT("Invalid generic brush"));
    SetColour(brush.GetColour());
    M_GBRUSHDATA->_style = brush.GetStyle();
    wxBitmap* stipple = brush.GetStipple();
    if (stipple && stipple->Ok()) M_GBRUSHDATA->_stipple = *stipple;
}

void wxGenericBrush::Set(const wxBrush& brush) {
    wxCHECK_RET(Ok() && brush.Ok(), wxT("Invalid generic brush"));
    SetColour(brush.GetColour());
    M_GBRUSHDATA->_style = brush.GetStyle();
    wxBitmap* stipple = brush.GetStipple();
    if (stipple && stipple->Ok()) M_GBRUSHDATA->_stipple = *stipple;
}

void wxGenericBrush::SetColour(const wxGenericColour& colour) {
    wxCHECK_RET(Ok() && colour.Ok(), wxT("Invalid generic brush or colour"));
    M_GBRUSHDATA->_colour = colour;
}

void wxGenericBrush::SetColour(const wxColour& colour) {
    SetColour(wxGenericColour(colour));
}

void wxGenericBrush::SetColour(int red, int green, int blue, int alpha) {
    SetColour(wxGenericColour(red, green, blue, alpha));
}

void wxGenericBrush::SetStyle(wxBrushStyle style) {
    wxCHECK_RET(Ok(), wxT("Invalid generic brush"));
    M_GBRUSHDATA->_style = style;
}

void wxGenericBrush::SetStipple(const wxBitmap& stipple) {
    wxCHECK_RET(Ok(), wxT("Invalid generic brush"));
    M_GBRUSHDATA->_stipple = stipple;
    M_GBRUSHDATA->_style = stipple.GetMask() ? wxBRUSHSTYLE_STIPPLE_MASK_OPAQUE : wxBRUSHSTYLE_STIPPLE;
}

wxBrush wxGenericBrush::GetBrush() const {
    wxCHECK_MSG(Ok(), wxNullBrush, wxT("Invalid generic brush"));
    if (M_GBRUSHDATA->_stipple.Ok()) return wxBrush(M_GBRUSHDATA->_stipple);

    return wxBrush(M_GBRUSHDATA->_colour.GetColour(), M_GBRUSHDATA->_style);
}

wxGenericColour wxGenericBrush::GetGenericColour() const {
    wxCHECK_MSG(Ok(), wxNullGenericColour, wxT("Invalid generic brush"));
    return M_GBRUSHDATA->_colour;
}

wxColour wxGenericBrush::GetColour() const {
    wxCHECK_MSG(Ok(), wxNullColour, wxT("Invalid generic brush"));
    return M_GBRUSHDATA->_colour.GetColour();
}

wxBrushStyle wxGenericBrush::GetStyle() const {
    wxCHECK_MSG(Ok(), wxBRUSHSTYLE_SOLID, wxT("Invalid generic brush"));
    return M_GBRUSHDATA->_style;
}

wxBitmap* wxGenericBrush::GetStipple() const {
    wxCHECK_MSG(Ok(), NULL, wxT("Invalid generic brush"));
    return &M_GBRUSHDATA->_stipple;
}

bool wxGenericBrush::IsSameAs(const wxGenericBrush& brush) const {
    wxCHECK_MSG(Ok() && brush.Ok(), 1, wxT("Invalid generic brush"));
    wxGenericBrushRefData* bData = (wxGenericBrushRefData*)brush.GetRefData();
    return (M_GBRUSHDATA->_colour == bData->_colour) && (M_GBRUSHDATA->_style == bData->_style) &&
#if wxCHECK_VERSION(2, 7, 2)
           (M_GBRUSHDATA->_stipple.IsSameAs(bData->_stipple));
#else
           (M_GBRUSHDATA->_stipple == bData->_stipple);
#endif  // wxCHECK_VERSION(2,7,2)
}

bool wxGenericBrush::IsSameAs(const wxBrush& brush) const {
    wxCHECK_MSG(Ok() && brush.Ok(), 1, wxT("Invalid generic brush"));
    wxGenericBrush gB(brush);
    gB.GetGenericColour().SetAlpha(M_GBRUSHDATA->_colour.GetAlpha());
    return IsSameAs(gB);
}
