/////////////////////////////////////////////////////////////////////////////
// Name:        plotmark.cpp
// Purpose:     wxPlotMarker
// Author:      John Labenski
// Modified by:
// Created:     8/27/2002
// Copyright:   (c) John Labenski
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#if defined(__GNUG__) && !defined(NO_GCC_PRAGMA)
#pragma implementation "plotmark.h"
#endif

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/wx.h"

#endif  // WX_PRECOMP

#include "wx/arrimpl.cpp"
#include "wx/image.h"
#include "wx/plotctrl/plotctrl.h"
#include "wx/plotctrl/plotmark.h"
WX_DEFINE_OBJARRAY(wxArrayPlotMarker);

//-----------------------------------------------------------------------------
// wxPlotMarkerRefData
//-----------------------------------------------------------------------------

class wxPlotMarkerRefData : public wxObjectRefData {
  public:
    wxPlotMarkerRefData(int type = 0, const wxRect2DDouble& rect = wxRect2DDouble())
        : wxObjectRefData(),
          _markerType(type),
          _rect(rect),
          _size(wxSize(-1, -1)) {}

    wxPlotMarkerRefData(const wxPlotMarkerRefData& data)
        : wxObjectRefData(),
          _markerType(data._markerType),
          _rect(data._rect),
          _size(data._size),
          _bitmap(data._bitmap),
          _pen(data._pen),
          _brush(data._brush) {}

    int _markerType;
    wxRect2DDouble _rect;
    wxSize _size;
    wxBitmap _bitmap;
    wxGenericPen _pen;
    wxGenericBrush _brush;
};

#define M_PMARKERDATA ((wxPlotMarkerRefData*)_refData)

//-----------------------------------------------------------------------------
// wxPlotMarker
//-----------------------------------------------------------------------------

IMPLEMENT_DYNAMIC_CLASS(wxPlotMarker, wxObject);

wxObjectRefData* wxPlotMarker::CreateRefData() const {
    return new wxPlotMarkerRefData;
}

wxObjectRefData* wxPlotMarker::CloneRefData(const wxObjectRefData* data) const {
    return new wxPlotMarkerRefData(*(const wxPlotMarkerRefData*)data);
}

void wxPlotMarker::Create(int marker_type, const wxRect2DDouble& rect, const wxSize& size, const wxGenericPen& pen,
                          const wxGenericBrush& brush, const wxBitmap& bitmap) {
    UnRef();
    _refData = new wxPlotMarkerRefData(marker_type, rect);
    M_PMARKERDATA->_size = size;
    M_PMARKERDATA->_pen = pen;
    M_PMARKERDATA->_brush = brush;
    M_PMARKERDATA->_bitmap = bitmap;
}

int wxPlotMarker::GetMarkerType() const {
    wxCHECK_MSG(Ok(), wxPLOTMARKER_NONE, wxT("Invalid plot marker"));
    return M_PMARKERDATA->_markerType;
}

void wxPlotMarker::SetMarkerType(int type) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_markerType = type;
}

wxRect2DDouble wxPlotMarker::GetPlotRect() const {
    wxCHECK_MSG(Ok(), wxRect2DDouble(), wxT("Invalid plot marker"));
    return M_PMARKERDATA->_rect;
}

wxRect2DDouble& wxPlotMarker::GetPlotRect() {
    static wxRect2DDouble s_rect;
    wxCHECK_MSG(Ok(), s_rect, wxT("Invalid plot marker"));
    return M_PMARKERDATA->_rect;
}

void wxPlotMarker::SetPlotRect(const wxRect2DDouble& rect) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_rect = rect;
}

wxPoint2DDouble wxPlotMarker::GetPlotPosition() const {
    wxCHECK_MSG(Ok(), wxPoint2DDouble(), wxT("Invalid plot marker"));
    return M_PMARKERDATA->_rect.GetLeftTop();
}

void wxPlotMarker::SetPlotPosition(const wxPoint2DDouble& pos) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_rect._x = pos._x;
    M_PMARKERDATA->_rect._y = pos._y;
}

wxSize wxPlotMarker::GetSize() const {
    wxCHECK_MSG(Ok(), wxSize(-1, -1), wxT("Invalid plot marker"));
    return M_PMARKERDATA->_size;
}

void wxPlotMarker::SetSize(const wxSize& size) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_size = size;
}

wxGenericPen wxPlotMarker::GetPen() const {
    wxCHECK_MSG(Ok(), wxNullGenericPen, wxT("Invalid plot marker"));
    return M_PMARKERDATA->_pen;
}

void wxPlotMarker::SetPen(const wxGenericPen& pen) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_pen = pen;
}

wxGenericBrush wxPlotMarker::GetBrush() const {
    wxCHECK_MSG(Ok(), wxNullGenericBrush, wxT("Invalid plot marker"));
    return M_PMARKERDATA->_brush;
}

void wxPlotMarker::SetBrush(const wxGenericBrush& brush) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_brush = brush;
}

wxBitmap wxPlotMarker::GetBitmap() const {
    wxCHECK_MSG(Ok(), wxNullBitmap, wxT("Invalid plot marker"));
    return M_PMARKERDATA->_bitmap;
}

void wxPlotMarker::SetBitmap(const wxBitmap& bitmap) {
    wxCHECK_RET(Ok(), wxT("Invalid plot marker"));
    M_PMARKERDATA->_bitmap = bitmap;
}
