/////////////////////////////////////////////////////////////////////////////
// Name:        plotctrl.cpp
// Purpose:     wxPlotCtrl
// Author:      John Labenski, Robert Roebling
// Modified by:
// Created:     8/27/2002
// Copyright:   (c) John Labenski, Robert Roebling
// Licence:     wxWindows license
/////////////////////////////////////////////////////////////////////////////

#if defined(__GNUG__) && !defined(NO_GCC_PRAGMA)
#pragma implementation "plotctrl.h"
#endif

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/dcclient.h"
#include "wx/dcmemory.h"
#include "wx/dcscreen.h"
#include "wx/event.h"
#include "wx/geometry.h"
#include "wx/msgdlg.h"
#include "wx/panel.h"
#include "wx/scrolbar.h"
#include "wx/sizer.h"
#include "wx/textctrl.h"
#include "wx/timer.h"

#endif  // WX_PRECOMP

#include <float.h>
#include <limits.h>
#include <math.h>

#include "wx/graphics.h"
#include "wx/image.h"
#include "wx/math.h"
#include "wx/plotctrl/plotctrl.h"
#include "wx/plotctrl/plotdraw.h"
#include "wx/splitter.h"

// MSVC hogs global namespace with these min/max macros - remove them
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#ifdef GetYValue  // Visual Studio 7 defines this
#undef GetYValue
#endif

//-----------------------------------------------------------------------------
// Consts
//-----------------------------------------------------------------------------

#define MAX_PLOT_ZOOMS 5
#define TIC_STEPS 3

std::numeric_limits<wxDouble> wxDouble_limits;
const wxDouble wxPlotCtrl_MIN_DBL = wxDouble_limits.min() * 10;
const wxDouble wxPlotCtrl_MAX_DBL = wxDouble_limits.max() / 10;
const wxDouble wxPlotCtrl_MAX_RANGE = wxDouble_limits.max() / 5;

// Draw borders around the axes, title, and labels for sizing testing
// #define DRAW_BORDERS

#include "wx/arrimpl.cpp"
WX_DEFINE_OBJARRAY(wxArrayPoint2DDouble);
WX_DEFINE_OBJARRAY(wxArrayRect2DDouble);
WX_DEFINE_OBJARRAY(wxArrayPlotCurve);

#include "grab.xpm"
#include "hand.xpm"

static wxCursor s_handCursor;
static wxCursor s_grabCursor;

// same as wxPlotRect2DDouble::Contains, but doesn't convert to wxPoint2DDouble
inline bool wxPlotRect2DDoubleContains(double x, double y, const wxRect2DDouble& rect) {
    return ((x >= rect._x) && (y >= rect._y) && (x <= rect.GetRight()) && (y <= rect.GetBottom()));
}

//----------------------------------------------------------------------------
// Event types
//----------------------------------------------------------------------------

// wxPlotCtrlEvent
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_ADD_CURVE)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_DELETING_CURVE)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_DELETED_CURVE)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_CURVE_SEL_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_CURVE_SEL_CHANGED)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_MOUSE_MOTION)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_CLICKED)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_DOUBLECLICKED)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_POINT_CLICKED)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_POINT_DOUBLECLICKED)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_SEL_CREATING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_SEL_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_SEL_CREATED)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_VIEW_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_VIEW_CHANGED)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_CURSOR_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_CURSOR_CHANGED)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_ERROR)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_BEGIN_TITLE_EDIT)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_END_TITLE_EDIT)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_BEGIN_X_LABEL_EDIT)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_END_X_LABEL_EDIT)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_BEGIN_Y_LABEL_EDIT)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_END_Y_LABEL_EDIT)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGED)

// wxPlotCtrlSelEvent
// DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CREATING)
// DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CREATED)
// DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CHANGED)

/*
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_VALUE_SEL_CREATING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_VALUE_SEL_CREATED)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_VALUE_SEL_CHANGING)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_VALUE_SEL_CHANGED)
DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_SEL_CHANGED)

DEFINE_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_CREATE)
*/

// The code below translates the event.GetEventType to a string name for debugging
#define aDEFINE_LOCAL_EVENT_TYPE(t) \
    if (eventType == t) return wxString(wxT(#t));

wxString wxPlotCtrl_GetEventName(wxEventType eventType) {
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_ADD_CURVE) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_DELETING_CURVE) aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_DELETED_CURVE)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_CURVE_SEL_CHANGING) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_CURVE_SEL_CHANGED)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_MOUSE_MOTION) aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_CLICKED)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_DOUBLECLICKED) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_POINT_CLICKED) aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_POINT_DOUBLECLICKED)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_SEL_CREATING) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_AREA_SEL_CHANGING) aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_AREA_SEL_CREATED)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_VIEW_CHANGING) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_VIEW_CHANGED)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_CURSOR_CHANGING) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_CURSOR_CHANGED)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_ERROR)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_BEGIN_TITLE_EDIT) aDEFINE_LOCAL_EVENT_TYPE(
        wxEVT_PLOTCTRL_END_TITLE_EDIT)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_BEGIN_X_LABEL_EDIT)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_END_X_LABEL_EDIT)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_BEGIN_Y_LABEL_EDIT)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_END_Y_LABEL_EDIT)

    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGING)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGED)

    // wxPlotCtrlSelEvent
    // DEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CREATING)
    // DEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CREATED)
    // DEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CHANGING)
    aDEFINE_LOCAL_EVENT_TYPE(wxEVT_PLOTCTRL_RANGE_SEL_CHANGED)

        return wxT("Unknown Event Type");
}

//-----------------------------------------------------------------------------
// wxPlotCtrlEvent
//-----------------------------------------------------------------------------
IMPLEMENT_ABSTRACT_CLASS(wxPlotCtrlEvent, wxNotifyEvent)

wxPlotCtrlEvent::wxPlotCtrlEvent(wxEventType commandType, wxWindowID id, wxPlotCtrl* window)
    : wxNotifyEvent(commandType, id),
      _curve(NULL),
      _curve_index(-1),
      _curve_dataindex(-1),
      _mouse_func(wxPLOTCTRL_MOUSE_NOTHING),
      _x(0),
      _y(0) {
    SetEventObject((wxObject*)window);
}

//-----------------------------------------------------------------------------
// wxPlotCtrlSelEvent
//-----------------------------------------------------------------------------
IMPLEMENT_ABSTRACT_CLASS(wxPlotCtrlSelEvent, wxPlotCtrlEvent)

wxPlotCtrlSelEvent::wxPlotCtrlSelEvent(wxEventType commandType, wxWindowID id, wxPlotCtrl* window)
    : wxPlotCtrlEvent(commandType, id, window),
      _selecting(false) {}

//-----------------------------------------------------------------------------
// wxPlotCtrlArea
//-----------------------------------------------------------------------------
IMPLEMENT_ABSTRACT_CLASS(wxPlotCtrlArea, wxWindow)

BEGIN_EVENT_TABLE(wxPlotCtrlArea, wxWindow)
EVT_ERASE_BACKGROUND(wxPlotCtrlArea::OnEraseBackground)
EVT_PAINT(wxPlotCtrlArea::OnPaint)
EVT_MOUSE_EVENTS(wxPlotCtrlArea::OnMouse)
EVT_CHAR(wxPlotCtrlArea::OnChar)
EVT_KEY_DOWN(wxPlotCtrlArea::OnKeyDown)
EVT_KEY_UP(wxPlotCtrlArea::OnKeyUp)
END_EVENT_TABLE()

wxPlotCtrlArea::wxPlotCtrlArea(wxWindow* parent, wxWindowID win_id, const wxPoint& pos, const wxSize& size, long style,
                               const wxString& name) {
    _owner = wxDynamicCast(parent, wxPlotCtrl);

    if (!wxWindow::Create(parent, win_id, pos, size, style, name)) return;

    SetSizeHints(4, 4);  // Don't allow window to get smaller than this!
}

void wxPlotCtrlArea::OnChar(wxKeyEvent& event) {
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlArea"));
    _owner->ProcessAreaEVT_CHAR(event);
}

void wxPlotCtrlArea::OnKeyDown(wxKeyEvent& event) {
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlArea"));
    _owner->ProcessAreaEVT_KEY_DOWN(event);
}

void wxPlotCtrlArea::OnKeyUp(wxKeyEvent& event) {
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlArea"));
    _owner->ProcessAreaEVT_KEY_UP(event);
}

void wxPlotCtrlArea::OnMouse(wxMouseEvent& event) {
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlArea"));
    _owner->ProcessAreaEVT_MOUSE_EVENTS(event);
}

void wxPlotCtrlArea::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlArea"));

    _owner->ProcessAreaEVT_PAINT(event, dc, this);
}

//-----------------------------------------------------------------------------
// wxPlotCtrlAxis
//-----------------------------------------------------------------------------
IMPLEMENT_ABSTRACT_CLASS(wxPlotCtrlAxis, wxWindow)

BEGIN_EVENT_TABLE(wxPlotCtrlAxis, wxWindow)
EVT_ERASE_BACKGROUND(wxPlotCtrlAxis::OnEraseBackground)
EVT_PAINT(wxPlotCtrlAxis::OnPaint)
EVT_MOUSE_EVENTS(wxPlotCtrlAxis::OnMouse)
EVT_CHAR(wxPlotCtrlAxis::OnChar)
END_EVENT_TABLE()

wxPlotCtrlAxis::wxPlotCtrlAxis(wxPlotCtrlAxis_Type axis_type, wxWindow* parent, wxWindowID win_id, const wxPoint& pos,
                               const wxSize& size, long style, const wxString& name) {
    _owner = wxDynamicCast(parent, wxPlotCtrl);
    _axis_type = axis_type;

    if (!wxWindow::Create(parent, win_id, pos, size, style, name)) return;

    SetSizeHints(4, 4);  // Don't allow window to get smaller than this!

    if (_axis_type == wxPLOTCTRL_Y_AXIS)
        SetCursor(wxCursor(wxCURSOR_SIZENS));
    else
        SetCursor(wxCursor(wxCURSOR_SIZEWE));
}

void wxPlotCtrlAxis::OnChar(wxKeyEvent& event) {
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlAxis"));
    _owner->ProcessAxisEVT_CHAR(event);
}

void wxPlotCtrlAxis::OnMouse(wxMouseEvent& event) {
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlAxis"));
    _owner->ProcessAxisEVT_MOUSE_EVENTS(event);
}

void wxPlotCtrlAxis::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);
    wxCHECK_RET(_owner, wxT("Invalid parent of wxPlotCtrlAxis"));

    _owner->ProcessAxisEVT_PAINT(event, dc, this);
}

//-----------------------------------------------------------------------------
// wxPlotCtrl
//-----------------------------------------------------------------------------
IMPLEMENT_ABSTRACT_CLASS(wxPlotCtrl, wxWindow)

BEGIN_EVENT_TABLE(wxPlotCtrl, wxWindow)
// EVT_ERASE_BACKGROUND ( wxPlotCtrl::OnEraseBackground ) // clear for MSW
EVT_SIZE(wxPlotCtrl::OnSize)
EVT_PAINT(wxPlotCtrl::OnPaint)
EVT_CHAR(wxPlotCtrl::OnChar)
EVT_SCROLL(wxPlotCtrl::OnScroll)
EVT_IDLE(wxPlotCtrl::OnIdle)
EVT_MOUSE_EVENTS(wxPlotCtrl::OnMouse)
EVT_TIMER(wxID_ANY, wxPlotCtrl::OnTimer)

EVT_TEXT_ENTER(wxID_ANY, wxPlotCtrl::OnTextEnter)
END_EVENT_TABLE()

void wxPlotCtrl::Init() {
    _drawOnScreen = true;

    _activeCurve = NULL;
    _active_index = -1;

    _cursorMarker.CreateEllipseMarker(wxPoint2DDouble(0, 0), wxSize(2, 2), wxGenericPen(wxGenericColour(0, 255, 0)));
    _cursor_curve = -1;
    _cursor_index = -1;

    _selection_type = wxPLOTCTRL_SELECT_MULTIPLE;

    _show_key = true;

    _show_title = _show_xlabel = _show_ylabel = false;
    _title = wxT("Title");
    _xLabel = wxT("X-Axis");
    _yLabel = wxT("Y-Axis");

    _titleFont = *wxSWISS_FONT;
    _titleColour = *wxBLACK;
    _borderColour = *wxBLACK;

    _scroll_on_thumb_release = false;
    _crosshair_cursor = false;
    _draw_symbols = true;
    _draw_lines = true;
    _draw_spline = false;
    _draw_grid = true;
    _fit_on_new_curve = true;
    _show_xAxis = true;
    _show_yAxis = true;

    _zoom = wxPoint2DDouble(1.0, 1.0);
    _history_views_index = -1;

    _fix_aspectratio = false;
    _aspectratio = 1.0;

    _defaultPlotRect = wxRect2DDouble(-10.0, -10.0, 20.0, 20.0);
    _viewRect = _defaultPlotRect;
    _curveBoundingRect = _defaultPlotRect;
    _areaClientRect = wxRect(0, 0, 10, 10);

    _xAxisTickType = _yAxisTickType = wxPLOTCTRL_VALUE;
    _xAxisTickFormat = _yAxisTickFormat = wxT("%lf");
    _xAxisTick_step = _yAxisTick_step = 1.0;
    _xAxisTick_step_fix = _yAxisTick_step_fix = -1;
    _xAxisTick_count = _yAxisTick_count = 4;
    _correct_ticks = true;

    _areaDrawer = NULL;
    _xAxisDrawer = NULL;
    _yAxisDrawer = NULL;
    _keyDrawer = NULL;
    _curveDrawer = NULL;
    _dataCurveDrawer = NULL;
    _markerDrawer = NULL;

    _xAxis = NULL;
    _yAxis = NULL;
    _area = NULL;
    _xAxisScrollbar = NULL;
    _yAxisScrollbar = NULL;
    _textCtrl = NULL;

    _focusedWin = NULL;
    _greedy_focus = false;

    _redraw_type = wxPLOTCTRL_REDRAW_BLOCKER;
    _batch_count = 0;

    _axisFontSize.x = 6;
    _axisFontSize.y = 12;
    _y_axis_text_width = 20;
    _area_border_width = 1;
    _border = 4;
    _min_exponential = 1000;
    _pen_print_width = 0.4;

    _timer = NULL;
    _winCapture = NULL;

    _area_mouse_marker = wxPLOTCTRL_MARKER_RECT;
    _area_mouse_func = wxPLOTCTRL_MOUSE_ZOOM;
    _area_mouse_cursorid = wxCURSOR_CROSS;

    _mouse_cursorid = wxCURSOR_ARROW;
}

bool wxPlotCtrl::Create(wxWindow* parent, wxWindowID win_id, const wxPoint& pos, const wxSize& size,
                        wxPlotCtrlAxis_Type WXUNUSED(flag), const wxString& name) {
    _redraw_type = wxPLOTCTRL_REDRAW_BLOCKER;  // no paints until finished

    if (!wxWindow::Create(parent, win_id, pos, wxSize(size.x > 20 ? size.x : 20, size.y > 20 ? size.y : 20),
                          wxWANTS_CHARS | wxCLIP_CHILDREN, name))
        return false;

    SetSizeHints(20, 20);  // Don't allow window to get smaller than this!

    if (!s_handCursor.Ok()) {
        wxImage image(wxBitmap(hand_xpm).ConvertToImage());
        image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_X, image.GetWidth() / 2);
        image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_Y, image.GetHeight() / 2);
        s_handCursor = wxCursor(image);
    }
    if (!s_grabCursor.Ok()) {
        wxImage image(wxBitmap(grab_xpm).ConvertToImage());
        image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_X, image.GetWidth() / 2);
        image.SetOption(wxIMAGE_OPTION_CUR_HOTSPOT_Y, image.GetHeight() / 2);
        s_grabCursor = wxCursor(image);
    }

    _areaDrawer = new wxPlotDrawerArea(this);
    _xAxisDrawer = new wxPlotDrawerXAxis(this);
    _yAxisDrawer = new wxPlotDrawerYAxis(this);
    _keyDrawer = new wxPlotDrawerKey(this);
    _curveDrawer = new wxPlotDrawerCurve(this);
    _dataCurveDrawer = new wxPlotDrawerDataCurve(this);
    _markerDrawer = new wxPlotDrawerMarker(this);

    wxFont axisFont(GetFont());
    GetTextExtent(wxT("5"), &_axisFontSize.x, &_axisFontSize.y, NULL, NULL, &axisFont);
    if ((_axisFontSize.x < 2) || (_axisFontSize.y < 2))  // don't want to divide by 0
    {
        _axisFontSize.x = 6;
        _axisFontSize.y = 12;
        wxFAIL_MSG(wxT("Can't determine the font size for the axis! I'll guess.\n")
                       wxT("The display might be corrupted, however you may continue."));
    }

    _xAxisDrawer->SetTickFont(axisFont);
    _yAxisDrawer->SetTickFont(axisFont);
    //    _xAxisDrawer->SetLabelFont(*wxSWISS_FONT); // needs to be rotated
    //    _yAxisDrawer->SetLabelFont(*wxSWISS_FONT); //   swiss works

    _xAxis = new wxPlotCtrlAxis(wxPLOTCTRL_X_AXIS, this, ID_PLOTCTRL_X_AXIS);
    _yAxis = new wxPlotCtrlAxis(wxPLOTCTRL_Y_AXIS, this, ID_PLOTCTRL_Y_AXIS);
    _area = new wxPlotCtrlArea(this, ID_PLOTCTRL_AREA);
    _xAxisScrollbar = new wxScrollBar(this, ID_PLOTCTRL_X_SCROLLBAR, wxDefaultPosition, wxDefaultSize,
                                       wxSB_HORIZONTAL);
    _yAxisScrollbar = new wxScrollBar(this, ID_PLOTCTRL_Y_SCROLLBAR, wxDefaultPosition, wxDefaultSize, wxSB_VERTICAL);

    _area->SetCursor(wxCURSOR_CROSS);
    _area->SetBackgroundColour(*wxWHITE);
    _xAxis->SetBackgroundColour(*wxWHITE);
    _yAxis->SetBackgroundColour(*wxWHITE);
    wxWindow::SetBackgroundColour(*wxWHITE);

    _area->SetForegroundColour(*wxLIGHT_GREY);

    // update the sizes of the title and axis labels
    SetPlotTitle(GetPlotTitle());
    SetXAxisLabel(GetXAxisLabel());
    SetYAxisLabel(GetYAxisLabel());

    _redraw_type = 0;  // redraw when all done
    Redraw(wxPLOTCTRL_REDRAW_WHOLEPLOT);

    return true;
}

wxPlotCtrl::~wxPlotCtrl() {
    delete _areaDrawer;
    delete _xAxisDrawer;
    delete _yAxisDrawer;
    delete _keyDrawer;
    delete _curveDrawer;
    delete _dataCurveDrawer;
    delete _markerDrawer;
}

void wxPlotCtrl::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    wxPaintDC dc(this);

    // DrawActiveBitmap(&dc);
    DrawPlotCtrl(&dc);
}

void wxPlotCtrl::DrawPlotCtrl(wxDC* dc) {
    wxCHECK_RET(dc, wxT("invalid dc"));

    if (_show_title && !_title.IsEmpty()) {
        dc->SetFont(GetPlotTitleFont());
        dc->SetTextForeground(GetPlotTitleColour());
        dc->DrawText(_title, _titleRect.x, _titleRect.y);
    }

    bool draw_xlabel = (_show_xlabel && !_xLabel.IsEmpty());
    bool draw_ylabel = (_show_ylabel && !_yLabel.IsEmpty());

    if (draw_xlabel || draw_ylabel) {
        dc->SetFont(GetAxisLabelFont());
        dc->SetTextForeground(GetAxisLabelColour());

        if (draw_xlabel) dc->DrawText(_xLabel, _xLabelRect.x, _xLabelRect.y);
        if (draw_ylabel) dc->DrawRotatedText(_yLabel, _yLabelRect.x, _yLabelRect.y + _yLabelRect.height, 90);
    }

#ifdef DRAW_BORDERS
    // Test code for sizing to show the extent of the axes
    dc->SetBrush(*wxTRANSPARENT_BRUSH);
    dc->SetPen(wxPen(GetBorderColour(), 1, wxPENSTYLE_SOLID));
    dc->DrawRectangle(_titleRect);
    dc->DrawRectangle(_xLabelRect);
    dc->DrawRectangle(_yLabelRect);
#endif  // DRAW_BORDERS
}

void wxPlotCtrl::SetPlotWinMouseCursor(int cursorid) {
    if (cursorid == _mouse_cursorid) return;
    _mouse_cursorid = cursorid;
    SetCursor(wxCursor((wxStockCursor)cursorid));
}

void wxPlotCtrl::OnMouse(wxMouseEvent& event) {
    if (event.ButtonDown() && IsTextCtrlShown()) {
        HideTextCtrl(true, true);
        return;
    }

    wxSize size(GetClientSize());
    wxPoint mousePt(event.GetPosition());

    if ((_show_title && _titleRect.Contains(mousePt)) || (_show_xlabel && _xLabelRect.Contains(mousePt)) ||
        (_show_ylabel && _yLabelRect.Contains(mousePt))) {
        SetPlotWinMouseCursor(wxCURSOR_IBEAM);
    } else
        SetPlotWinMouseCursor(wxCURSOR_ARROW);

    if (event.ButtonDClick(1) && !IsTextCtrlShown()) {
        if (_show_title && _titleRect.Contains(mousePt))
            ShowTextCtrl(wxPLOTCTRL_EDIT_TITLE, true);
        else if (_show_xlabel && _xLabelRect.Contains(mousePt))
            ShowTextCtrl(wxPLOTCTRL_EDIT_XAXIS, true);
        else if (_show_ylabel && _yLabelRect.Contains(mousePt))
            ShowTextCtrl(wxPLOTCTRL_EDIT_YAXIS, true);
    }
}

void wxPlotCtrl::ShowTextCtrl(wxPlotCtrlTextCtrl_Type type, bool send_event) {
    switch (type) {
        case wxPLOTCTRL_EDIT_TITLE: {
            if (_textCtrl) {
                if (_textCtrl->GetId() != wxEVT_PLOTCTRL_END_TITLE_EDIT)
                    HideTextCtrl(true, true);
                else
                    return;  // already shown
            }

            if (send_event) {
                wxPlotCtrlEvent pevent(wxEVT_PLOTCTRL_BEGIN_TITLE_EDIT, GetId(), this);
                pevent.SetString(_title);
                if (!DoSendEvent(pevent)) return;
            }

            _textCtrl = new wxTextCtrl(this, wxEVT_PLOTCTRL_END_TITLE_EDIT, GetPlotTitle(), wxPoint(_areaRect.x, 0),
                                        wxSize(_areaRect.width, _titleRect.height + 2 * _border),
                                        wxTE_PROCESS_ENTER);

            _textCtrl->SetFont(GetPlotTitleFont());
            _textCtrl->SetForegroundColour(GetPlotTitleColour());
            _textCtrl->SetBackgroundColour(GetBackgroundColour());
            break;
        }
        case wxPLOTCTRL_EDIT_XAXIS: {
            if (_textCtrl) {
                if (_textCtrl->GetId() != wxEVT_PLOTCTRL_END_X_LABEL_EDIT)
                    HideTextCtrl(true, true);
                else
                    return;  // already shown
            }

            if (send_event) {
                wxPlotCtrlEvent pevent(wxEVT_PLOTCTRL_BEGIN_X_LABEL_EDIT, GetId(), this);
                pevent.SetString(_xLabel);
                if (!DoSendEvent(pevent)) return;
            }

            _textCtrl = new wxTextCtrl(
                this, wxEVT_PLOTCTRL_END_X_LABEL_EDIT, GetXAxisLabel(), wxPoint(_areaRect.x, _xAxisRect.GetBottom()),
                wxSize(_areaRect.width, _xLabelRect.height + 2 * _border), wxTE_PROCESS_ENTER);

            _textCtrl->SetFont(GetAxisLabelFont());
            _textCtrl->SetForegroundColour(GetAxisLabelColour());
            _textCtrl->SetBackgroundColour(GetBackgroundColour());
            break;
        }
        case wxPLOTCTRL_EDIT_YAXIS: {
            if (_textCtrl) {
                if (_textCtrl->GetId() != wxEVT_PLOTCTRL_END_Y_LABEL_EDIT)
                    HideTextCtrl(true, true);
                else
                    return;  // already shown
            }

            if (send_event) {
                wxPlotCtrlEvent pevent(wxEVT_PLOTCTRL_BEGIN_Y_LABEL_EDIT, GetId(), this);
                pevent.SetString(_yLabel);
                if (!DoSendEvent(pevent)) return;
            }

            _textCtrl = new wxTextCtrl(
                this, wxEVT_PLOTCTRL_END_Y_LABEL_EDIT, GetYAxisLabel(),
                wxPoint(0, _areaRect.y + _areaRect.height / 2),
                wxSize(_clientRect.width - _axisFontSize.y / 2, _yLabelRect.width + 2 * _border),
                wxTE_PROCESS_ENTER);

            _textCtrl->SetFont(GetAxisLabelFont());
            _textCtrl->SetForegroundColour(GetAxisLabelColour());
            _textCtrl->SetBackgroundColour(GetBackgroundColour());
            break;
        }
    }
}

void wxPlotCtrl::HideTextCtrl(bool save_value, bool send_event) {
    wxCHECK_RET(_textCtrl, wxT("HideTextCtrl, but textctrl is not shown"));

    long event_type = _textCtrl->GetId();
    wxString value = _textCtrl->GetValue();

    _textCtrl->Destroy();
    _textCtrl = NULL;

    if (!save_value) return;

    bool changed = false;

    if (event_type == wxEVT_PLOTCTRL_END_TITLE_EDIT)
        changed = (value != GetPlotTitle());
    else if (event_type == wxEVT_PLOTCTRL_END_X_LABEL_EDIT)
        changed = (value != GetXAxisLabel());
    else if (event_type == wxEVT_PLOTCTRL_END_Y_LABEL_EDIT)
        changed = (value != GetYAxisLabel());

    if (!changed) return;

    if (send_event) {
        wxPlotCtrlEvent event(event_type, GetId(), this);
        event.SetString(value);
        if (!DoSendEvent(event)) return;
    }

    if (event_type == wxEVT_PLOTCTRL_END_TITLE_EDIT)
        SetPlotTitle(value);
    else if (event_type == wxEVT_PLOTCTRL_END_X_LABEL_EDIT)
        SetXAxisLabel(value);
    else if (event_type == wxEVT_PLOTCTRL_END_Y_LABEL_EDIT)
        SetYAxisLabel(value);
}

bool wxPlotCtrl::IsTextCtrlShown() const {
    return _textCtrl && _textCtrl->IsShown();
}

void wxPlotCtrl::OnTextEnter(wxCommandEvent& event) {
    // we send a fake event so that we can destroy the textctrl the second time
    if (event.GetId() == 1)
        HideTextCtrl(true, true);
    else {
        wxCommandEvent newevt(wxEVT_COMMAND_TEXT_ENTER, 1);
        GetEventHandler()->AddPendingEvent(newevt);
    }
}

void wxPlotCtrl::OnIdle(wxIdleEvent& event) {
    CheckFocus();
    event.Skip();
}

bool wxPlotCtrl::CheckFocus() {
    wxWindow* win = FindFocus();

    if (win == _focusedWin) return true;

    if ((win == _area) || (win == _xAxis) || (win == _yAxis) || (win == this)) {
        if (!_focusedWin) {
            _focusedWin = win;
        }
    } else if (_focusedWin) {
        _focusedWin = NULL;
    }
    return _focusedWin != NULL;
}

void wxPlotCtrl::EndBatch(bool force_refresh) {
    if (_batch_count > 0) {
        _batch_count--;
        if ((_batch_count <= 0) && force_refresh) {
            Redraw(wxPLOTCTRL_REDRAW_WHOLEPLOT);
            AdjustScrollBars();
        }
    }
}

bool wxPlotCtrl::SetBackgroundColour(const wxColour& colour) {
    wxCHECK_MSG(colour.Ok(), false, wxT("invalid colour"));
    _area->SetBackgroundColour(colour);
    _xAxis->SetBackgroundColour(colour);
    _yAxis->SetBackgroundColour(colour);
    wxWindow::SetBackgroundColour(colour);

    Redraw(wxPLOTCTRL_REDRAW_EVERYTHING);
    return true;
}

void wxPlotCtrl::SetGridColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    _area->SetForegroundColour(colour);
    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

void wxPlotCtrl::SetBorderColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    _borderColour = colour;
    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

void wxPlotCtrl::SetCursorColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    _cursorMarker.GetPen().SetColour(colour);
    wxClientDC dc(_area);
    DrawCurveCursor(&dc);
}

wxColour wxPlotCtrl::GetCursorColour() const {
    return _cursorMarker.GetPen().GetColour();
}

int wxPlotCtrl::GetCursorSize() const {
    return _cursorMarker.GetSize().x;
}

void wxPlotCtrl::SetCursorSize(int size) {
    _cursorMarker.SetSize(wxSize(size, size));
}

wxFont wxPlotCtrl::GetAxisFont() const {
    return _xAxisDrawer->_tickFont;  // FIXME
}

wxColour wxPlotCtrl::GetAxisColour() const {
    return _xAxisDrawer->_tickColour.GetColour();  // FIXME
}

void wxPlotCtrl::SetAxisFont(const wxFont& font) {
    wxCHECK_RET(font.Ok(), wxT("invalid font"));

    if (_xAxisDrawer) _xAxisDrawer->SetTickFont(font);
    if (_yAxisDrawer) _yAxisDrawer->SetTickFont(font);

    int x = 6, y = 12, decent = 0, leading = 0;

    GetTextExtent(wxT("5"), &x, &y, &decent, &leading, &font);
    _axisFontSize.x = x + leading;
    _axisFontSize.y = y + decent;

    GetTextExtent(wxT("99.99"), &x, &y, &decent, &leading, &font);
    _y_axis_text_width = x + leading;

    // _axisFontSize.x = _xAxis->GetCharWidth();
    // _axisFontSize.y = _xAxis->GetCharHeight();
    if ((_axisFontSize.x < 2) || (_axisFontSize.y < 2))  // don't want to divide by 0
    {
        static bool first_try = false;

        _axisFontSize.x = 6;
        _axisFontSize.y = 12;
        wxMessageBox(wxT("Can't determine the font size for the axis.\n") wxT("Reverting to a default font."),
                     wxT("Font error"), wxICON_ERROR, this);

        if (!first_try) {
            first_try = true;
            SetAxisFont(*wxNORMAL_FONT);
        } else
            first_try = false;
    }

    DoSize();
    Redraw(wxPLOTCTRL_REDRAW_XAXIS | wxPLOTCTRL_REDRAW_YAXIS);
}

void wxPlotCtrl::SetAxisColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    if (_xAxisDrawer) _xAxisDrawer->SetTickColour(colour);
    if (_yAxisDrawer) _yAxisDrawer->SetTickColour(colour);
    Redraw(wxPLOTCTRL_REDRAW_XAXIS | wxPLOTCTRL_REDRAW_YAXIS);
}

wxFont wxPlotCtrl::GetAxisLabelFont() const {
    return _xAxisDrawer->_labelFont;  // FIXME
}

wxColour wxPlotCtrl::GetAxisLabelColour() const {
    return _xAxisDrawer->_labelColour.GetColour();  // FIXME
}

void wxPlotCtrl::SetAxisLabelFont(const wxFont& font) {
    wxCHECK_RET(font.Ok(), wxT("invalid font"));
    if (_xAxisDrawer) _xAxisDrawer->SetLabelFont(font);
    if (_yAxisDrawer) _yAxisDrawer->SetLabelFont(font);
    SetXAxisLabel(GetXAxisLabel());  // FIXME - lazy hack
    SetYAxisLabel(GetYAxisLabel());
}

void wxPlotCtrl::SetAxisLabelColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    if (_xAxisDrawer) _xAxisDrawer->SetLabelColour(colour);
    if (_yAxisDrawer) _yAxisDrawer->SetLabelColour(colour);
    SetXAxisLabel(GetXAxisLabel());  // FIXME - lazy hack
    SetYAxisLabel(GetYAxisLabel());
}

void wxPlotCtrl::SetPlotTitleFont(const wxFont& font) {
    wxCHECK_RET(font.Ok(), wxT("invalid font"));
    _titleFont = font;
    SetPlotTitle(GetPlotTitle());
}

void wxPlotCtrl::SetPlotTitleColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    _titleColour = colour;
    SetPlotTitle(GetPlotTitle());
}

wxFont wxPlotCtrl::GetKeyFont() const {
    return _keyDrawer->_font;  // FIXME
}

wxColour wxPlotCtrl::GetKeyColour() const {
    return _keyDrawer->_fontColour.GetColour();  // FIXME
}

void wxPlotCtrl::SetKeyFont(const wxFont& font) {
    wxCHECK_RET(font.Ok(), wxT("invalid font"));
    _keyDrawer->SetFont(font);
    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

void wxPlotCtrl::SetKeyColour(const wxColour& colour) {
    wxCHECK_RET(colour.Ok(), wxT("invalid colour"));
    _keyDrawer->SetFontColour(colour);
    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

// ------------------------------------------------------------------------
// Title, axis labels, and key
// ------------------------------------------------------------------------

void wxPlotCtrl::SetXAxisLabel(const wxString& label) {
    if (label.IsEmpty())
        _xLabel = wxT("X - Axis");
    else
        _xLabel = label;

    wxFont font = GetAxisLabelFont();
    GetTextExtent(_xLabel, &_xLabelRect.width, &_xLabelRect.height, NULL, NULL, &font);

    _xLabel = label;
    Refresh();
    DoSize();
}

void wxPlotCtrl::SetYAxisLabel(const wxString& label) {
    if (label.IsEmpty())
        _yLabel = wxT("Y - Axis");
    else
        _yLabel = label;

    wxFont font = GetAxisLabelFont();
    GetTextExtent(_yLabel, &_yLabelRect.height, &_yLabelRect.width, NULL, NULL, &font);

    _yLabel = label;

    Refresh();
    DoSize();
}

void wxPlotCtrl::SetPlotTitle(const wxString& title) {
    if (title.IsEmpty())
        _title = wxT("Title");
    else
        _title = title;

    wxFont font = GetPlotTitleFont();
    GetTextExtent(_title, &_titleRect.width, &_titleRect.height, NULL, NULL, &font);

    _title = title;

    Refresh();
    DoSize();
}

wxPoint wxPlotCtrl::GetKeyPosition() const {
    return _keyDrawer->_keyPosition;
}

bool wxPlotCtrl::GetKeyInside() const {
    return _keyDrawer->_key_inside;
}

void wxPlotCtrl::SetKeyPosition(const wxPoint& pos, bool stay_inside) {
    _keyDrawer->_keyPosition = pos;
    _keyDrawer->_key_inside = stay_inside;
    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

void wxPlotCtrl::SetKeyBottom(bool put_bottom) {
    _keyDrawer->_key_bottom = put_bottom;
    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

void wxPlotCtrl::CreateKeyString() {
    _keyString.Clear();
    int n, count = _curves.GetCount();
    for (n = 0; n < count; n++) {
        wxString key;
        if (GetDataCurve(n))
            key = GetDataCurve(n)->GetFilename();
        else if (GetFunctionCurve(n))
            key = GetFunctionCurve(n)->GetFunctionString();
        else
            key.Printf(wxT("Curve %d"), n);

        _keyString += (key + wxT("\n"));
    }
}

// ------------------------------------------------------------------------
// Curve Accessors
// ------------------------------------------------------------------------

bool wxPlotCtrl::AddCurve(wxPlotCurve* curve, bool select, bool send_event) {
    if (!curve || !curve->Ok()) {
        if (curve) delete curve;
        wxCHECK_MSG(false, false, wxT("Invalid curve"));
    }

    _curves.Add(curve);
    _curveSelections.Add(new wxRangeDoubleSelection());
    _dataSelections.Add(new wxRangeIntSelection());

    CalcBoundingPlotRect();
    CreateKeyString();

    if (send_event) {
        wxPlotCtrlEvent event(wxEVT_PLOTCTRL_ADD_CURVE, GetId(), this);
        event.SetCurve(curve, _curves.GetCount() - 1);
        (void)DoSendEvent(event);
    }

    _batch_count++;
    if (select) SetActiveCurve(curve, send_event);
    _batch_count--;

    if (_fit_on_new_curve)
        SetZoom(-1, -1, 0, 0, true);
    else
        Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

bool wxPlotCtrl::AddCurve(const wxPlotCurve& curve, bool select, bool send_event) {
    wxCHECK_MSG(curve.Ok(), false, wxT("invalid wxPlotCurve"));

    return AddCurve(curve.Clone(), select, send_event);
}

bool wxPlotCtrl::DeleteCurve(wxPlotCurve* curve, bool send_event) {
    wxCHECK_MSG(curve, false, wxT("invalid plotcurve"));

    int index = _curves.Index(*curve);
    wxCHECK_MSG(index != wxNOT_FOUND, false, wxT("Unknown wxPlotCurve"));

    return DeleteCurve(index, send_event);
}

bool wxPlotCtrl::DeleteCurve(int n, bool send_event) {
    wxCHECK_MSG((n >= -1) && (n < int(_curves.GetCount())), false, wxT("Invalid curve index"));

    if (send_event) {
        wxPlotCtrlEvent event(wxEVT_PLOTCTRL_DELETING_CURVE, GetId(), this);
        event.SetCurveIndex(n);  // can't set curve since index may be -1 for all curves
        if (!DoSendEvent(event)) return false;
    }

    BeginBatch();  // don't redraw yet

    if (n < 0) {
        InvalidateCursor(send_event);
        ClearSelectedRanges(-1, send_event);
        _dataSelections.Clear();
        _curveSelections.Clear();
        _curves.Clear();
    } else {
        if (_cursor_curve == n)
            InvalidateCursor(send_event);
        else if (_cursor_curve > n)
            _cursor_curve--;

        ClearSelectedRanges(n, send_event);
        _dataSelections.RemoveAt(n);
        _curveSelections.RemoveAt(n);
        _curves.RemoveAt(n);
    }

    int old_active_index = _active_index;
    _active_index = -1;
    _activeCurve = NULL;

    if (old_active_index >= int(_curves.GetCount())) {
        // force this invalid, can't override this, the curve is "gone"
        SetActiveIndex(_curves.GetCount() - 1, send_event);
    } else if (old_active_index >= 0) {
        SetActiveIndex(old_active_index, send_event);
    }

    EndBatch(false);  // still don't redraw

    CalcBoundingPlotRect();
    CreateKeyString();
    Redraw(wxPLOTCTRL_REDRAW_PLOT);

    if (send_event) {
        wxPlotCtrlEvent event1(wxEVT_PLOTCTRL_DELETED_CURVE, GetId(), this);
        event1.SetCurveIndex(n);
        (void)DoSendEvent(event1);
    }

    return true;
}

wxPlotCurve* wxPlotCtrl::GetCurve(int n) const {
    wxCHECK_MSG((n >= 0) && (n < GetCurveCount()), NULL, wxT("Invalid index"));
    return &(_curves.Item(n));
}

void wxPlotCtrl::SetActiveCurve(wxPlotCurve* current, bool send_event) {
    wxCHECK_RET(current, wxT("Invalid curve"));

    int index = _curves.Index(*current);
    wxCHECK_RET(index != wxNOT_FOUND, wxT("Unknown PlotCurve"));

    SetActiveIndex(index, send_event);
}

void wxPlotCtrl::SetActiveIndex(int curve_index, bool send_event) {
    wxCHECK_RET((curve_index < GetCurveCount()), wxT("Invalid index"));

    if (send_event) {
        wxPlotCtrlEvent event(wxEVT_PLOTCTRL_CURVE_SEL_CHANGING, GetId(), this);
        event.SetCurve(_activeCurve, _active_index);
        if (!DoSendEvent(event)) return;
    }

    if ((curve_index >= 0) && _curves.Item(curve_index).Ok()) {
        _active_index = curve_index;
        _activeCurve = &(_curves.Item(curve_index));
    } else {
        _active_index = -1;
        _activeCurve = NULL;
    }

    if (send_event) {
        wxPlotCtrlEvent event(wxEVT_PLOTCTRL_CURVE_SEL_CHANGED, GetId(), this);
        event.SetCurve(_activeCurve, _active_index);
        (void)DoSendEvent(event);
    }

    Redraw(wxPLOTCTRL_REDRAW_PLOT);
}

wxArrayInt wxPlotCtrl::GetPlotDataIndexes() const {
    wxArrayInt array;
    size_t n, count = _curves.GetCount();
    for (n = 0; n < count; n++) {
        if (wxDynamicCast(&_curves.Item(n), wxPlotData)) array.Add(n);
    }
    return array;
}

wxArrayInt wxPlotCtrl::GetPlotFunctionIndexes() const {
    wxArrayInt array;
    size_t n, count = _curves.GetCount();
    for (n = 0; n < count; n++) {
        if (wxDynamicCast(&_curves.Item(n), wxPlotFunction)) array.Add(n);
    }
    return array;
}

//-------------------------------------------------------------------------
// Markers
//-------------------------------------------------------------------------

int wxPlotCtrl::AddMarker(const wxPlotMarker& marker) {
    _plotMarkers.Add(marker);
    return _plotMarkers.GetCount() - 1;
}

void wxPlotCtrl::RemoveMarker(int marker) {
    wxCHECK_RET((marker >= 0) && (marker < (int)_plotMarkers.GetCount()), wxT("Invalid marker number"));
    _plotMarkers.RemoveAt(marker);
}

void wxPlotCtrl::ClearMarkers() {
    _plotMarkers.Clear();
}

wxPlotMarker wxPlotCtrl::GetMarker(int marker) const {
    wxCHECK_MSG((marker >= 0) && (marker < (int)_plotMarkers.GetCount()), wxPlotMarker(),
                wxT("Invalid marker number"));
    return _plotMarkers[marker];
}

//-------------------------------------------------------------------------
// Cursor position
//-------------------------------------------------------------------------

void wxPlotCtrl::InvalidateCursor(bool send_event) {
    bool changed = _cursor_curve >= 0;
    _cursor_curve = -1;
    _cursor_index = -1;
    _cursorMarker.SetPlotPosition(wxPoint2DDouble(0, 0));

    if (send_event && changed) {
        wxPlotCtrlEvent plotEvent(wxEVT_PLOTCTRL_CURSOR_CHANGED, GetId(), this);
        (void)DoSendEvent(plotEvent);
    }
}

bool wxPlotCtrl::IsCursorValid() {
    if (_cursor_curve < 0) return false;

    // sanity check
    if (_cursor_curve >= int(_curves.GetCount())) {
        wxFAIL_MSG(wxT("Invalid cursor index"));
        InvalidateCursor(true);
        return false;
    }

    wxPlotData* plotData = GetDataCurve(_cursor_curve);
    if (plotData) {
        // sanity check
        if (_cursor_index < 0) {
            wxFAIL_MSG(wxT("Invalid cursor data index"));
            InvalidateCursor(true);
            return false;
        }
        // if the curve shrinks or is bad
        if (!plotData->Ok() || (_cursor_index >= (int)plotData->GetCount())) {
            InvalidateCursor(true);
            return false;
        }

        _cursorMarker.SetPlotPosition(plotData->GetPoint(_cursor_index));
    } else {
        wxDouble x = _cursorMarker.GetPlotRect()._x;
        _cursorMarker.GetPlotRect()._y = GetCurve(_cursor_curve)->GetY(x);
    }

    return true;
}

wxPoint2DDouble wxPlotCtrl::GetCursorPoint() {
    wxCHECK_MSG(IsCursorValid(), wxPoint2DDouble(0, 0), wxT("invalid cursor"));
    return _cursorMarker.GetPlotPosition();
}

bool wxPlotCtrl::SetCursorDataIndex(int curve_index, int cursor_index, bool send_event) {
    wxCHECK_MSG(CurveIndexOk(curve_index) && GetDataCurve(curve_index), false, wxT("invalid curve index"));

    wxPlotData* plotData = GetDataCurve(curve_index);

    wxCHECK_MSG((cursor_index >= 0) && plotData->Ok() && (cursor_index < (int)plotData->GetCount()), false,
                wxT("invalid index"));

    // do nothing if already set
    if ((_cursor_curve == curve_index) && (_cursor_index == cursor_index)) return false;

    wxPoint2DDouble cursorPt(plotData->GetPoint(cursor_index));

    if (send_event) {
        wxPlotCtrlEvent cursor_event(wxEVT_PLOTCTRL_CURSOR_CHANGING, GetId(), this);
        cursor_event.SetPosition(cursorPt._x, cursorPt._y);
        cursor_event.SetCurve(plotData, curve_index);
        cursor_event.SetCurveDataIndex(cursor_index);
        if (!DoSendEvent(cursor_event)) return false;
    }

    int old_cursor_curve = _cursor_curve;
    int old_cursor_index = _cursor_index;
    _cursorMarker.SetPlotPosition(cursorPt);
    _cursor_curve = curve_index;
    _cursor_index = cursor_index;

    if (send_event) {
        wxPlotCtrlEvent cursor_event(wxEVT_PLOTCTRL_CURSOR_CHANGED, GetId(), this);
        cursor_event.SetPosition(cursorPt._x, cursorPt._y);
        cursor_event.SetCurve(plotData, curve_index);
        cursor_event.SetCurveDataIndex(cursor_index);
        (void)DoSendEvent(cursor_event);
    }

    if ((_active_index == old_cursor_curve) && (_active_index == _cursor_curve)) {
        RedrawDataCurve(curve_index, old_cursor_index, old_cursor_index);
        RedrawDataCurve(curve_index, _cursor_index, _cursor_index);
    } else
        Redraw(wxPLOTCTRL_REDRAW_PLOT);

    return true;
}

bool wxPlotCtrl::SetCursorXPoint(int curve_index, double x, bool send_event) {
    wxCHECK_MSG(CurveIndexOk(curve_index), false, wxT("invalid curve index"));

    if (GetDataCurve(curve_index))
        return SetCursorDataIndex(curve_index, GetDataCurve(curve_index)->GetIndexFromX(x), send_event);

    // do nothing if already set
    if ((_cursor_curve == curve_index) && (_cursorMarker.GetPlotRect()._x == x)) return false;

    wxPlotCurve* plotCurve = GetCurve(curve_index);
    wxPoint2DDouble cursorPt(x, plotCurve->GetY(x));

    if (send_event) {
        wxPlotCtrlEvent cursor_event(wxEVT_PLOTCTRL_CURSOR_CHANGING, GetId(), this);
        cursor_event.SetPosition(cursorPt._x, cursorPt._y);
        cursor_event.SetCurve(plotCurve, curve_index);
        if (!DoSendEvent(cursor_event)) return false;
    }

    _cursorMarker.SetPlotPosition(cursorPt);
    _cursor_curve = curve_index;
    _cursor_index = -1;

    if (send_event) {
        wxPlotCtrlEvent cursor_event(wxEVT_PLOTCTRL_CURSOR_CHANGED, GetId(), this);
        cursor_event.SetPosition(cursorPt._x, cursorPt._y);
        cursor_event.SetCurve(plotCurve, curve_index);
        (void)DoSendEvent(cursor_event);
    }

    Redraw(wxPLOTCTRL_REDRAW_PLOT);
    return true;
}

void wxPlotCtrl::MakeCursorVisible(bool center, bool send_event) {
    wxCHECK_RET(IsCursorValid(), wxT("invalid plot cursor"));

    if (center) {
        wxPoint2DDouble origin = _viewRect.GetLeftTop() - _viewRect.GetCentre() + GetCursorPoint();

        SetOrigin(origin._x, origin._y, send_event);
        return;
    }

    wxPoint2DDouble origin = GetCursorPoint();

    if (_viewRect.Contains(origin)) return;

    double dx = 4 / _zoom._x;
    double dy = 4 / _zoom._y;

    if (origin._x < _viewRect._x)
        origin._x -= dx;
    else if (origin._x > _viewRect.GetRight())
        origin._x = _viewRect._x + (origin._x - _viewRect.GetRight()) + dx;
    else
        origin._x = _viewRect._x;

    if (origin._y < _viewRect._y)
        origin._y -= dy;
    else if (origin._y > _viewRect.GetBottom())
        origin._y = _viewRect._y + (origin._y - _viewRect.GetBottom()) + dy;
    else
        origin._y = _viewRect._y;

    SetOrigin(origin._x, origin._y, send_event);
}

//-------------------------------------------------------------------------
// Selected points, data curves use
//-------------------------------------------------------------------------
bool wxPlotCtrl::HasSelection(int curve_index) const {
    if (curve_index == -1) {
        int n, count = _curveSelections.GetCount();
        for (n = 0; n < count; n++) {
            if ((_curveSelections[n].GetCount() > 0) || (_dataSelections[n].GetCount() > 0)) return true;
        }
        return false;
    }

    wxCHECK_MSG(CurveIndexOk(curve_index), false, wxT("invalid curve index"));
    return (_curveSelections[curve_index].GetCount() > 0) || (_dataSelections[curve_index].GetCount() > 0);
}

wxRangeDoubleSelection* wxPlotCtrl::GetCurveSelection(int curve_index) const {
    wxCHECK_MSG(CurveIndexOk(curve_index), NULL, wxT("invalid curve index"));
    return &_curveSelections[curve_index];
}

wxRangeIntSelection* wxPlotCtrl::GetDataCurveSelection(int curve_index) const {
    wxCHECK_MSG(CurveIndexOk(curve_index), NULL, wxT("invalid curve index"));
    return &_dataSelections[curve_index];
}

bool wxPlotCtrl::UpdateSelectionState(int curve_index, bool send_event) {
    wxCHECK_MSG(CurveIndexOk(curve_index), false, wxT("invalid curve index"));
    switch (_selection_type) {
        case wxPLOTCTRL_SELECT_NONE:
            break;  // should have been handled
        case wxPLOTCTRL_SELECT_SINGLE: {
            if (HasSelection()) return ClearSelectedRanges(-1, send_event);

            break;
        }
        case wxPLOTCTRL_SELECT_SINGLE_CURVE: {
            int n, count = _curves.GetCount();
            bool done = false;
            for (n = 0; n < count; n++) {
                if ((n != curve_index) && HasSelection(n)) done |= ClearSelectedRanges(n, send_event);
            }
            return done;
        }
        case wxPLOTCTRL_SELECT_SINGLE_PER_CURVE: {
            if (HasSelection(curve_index)) return ClearSelectedRanges(curve_index, send_event);

            break;
        }
        case wxPLOTCTRL_SELECT_MULTIPLE:
            break;  // anything goes
        default:
            break;
    }

    return false;
}

bool wxPlotCtrl::DoSelectRectangle(int curve_index, const wxRect2DDouble& rect, bool select, bool send_event) {
    wxCHECK_MSG((curve_index >= -1) && (curve_index < int(_curves.GetCount())), false, wxT("invalid plotcurve index"));
    wxCHECK_MSG((rect._width > 0) || (rect._height > 0), false, wxT("invalid selection range"));

    if (_selection_type == wxPLOTCTRL_SELECT_NONE) return false;

    if (!IsFinite(rect._x, wxT("Selection x is NaN")) || !IsFinite(rect._y, wxT("Selection y is NaN")) ||
        !IsFinite(rect._width, wxT("Selection width is NaN")) ||
        !IsFinite(rect._height, wxT("Selection height is NaN")))
        return false;

    bool done = false;

    // Run this code for all the curves if curve == -1 then exit
    if (curve_index == -1) {
        size_t n, curve_count = _curves.GetCount();

        for (n = 0; n < curve_count; n++) done |= DoSelectRectangle(n, rect, select, send_event);

        return done;
    }

    // check the selection type and clear previous selections if necessary
    if (select) UpdateSelectionState(curve_index, send_event);

    bool is_x_range = rect._height <= 0;
    bool is_y_range = rect._width <= 0;
    wxRangeDouble xRange(rect._x, rect.GetRight());
    wxRangeDouble yRange(rect._y, rect.GetBottom());

    wxPlotData* plotData = GetDataCurve(curve_index);
    if (plotData) {
        wxCHECK_MSG(plotData->Ok(), false, wxT("Invalid data curve"));
        wxRect2DDouble r(plotData->GetBoundingRect());

        if ((xRange._max < r.GetLeft()) || (xRange._min > r.GetRight())) return false;

        if (is_x_range && plotData->GetIsXOrdered()) {
            int min_ = plotData->GetIndexFromX(xRange._min);
            int max_ = plotData->GetIndexFromX(xRange._max);
            int count = plotData->GetCount();

            if ((plotData->GetXValue(min_) > xRange._min) && (min_ > 0) &&
                (plotData->GetXValue(min_ - 1) > xRange._min))
                min_--;
            if ((plotData->GetXValue(min_) < xRange._min) && (min_ < count - 1)) min_++;

            if ((plotData->GetXValue(max_) > xRange._max) && (max_ > 0)) max_--;
            if ((plotData->GetXValue(max_) < xRange._max) && (max_ < count - 1) &&
                (plotData->GetXValue(max_ + 1) < xRange._max))
                max_++;

            wxRangeInt sel(min_, max_);  // always check if max < min! - not a bug

            if (!sel.IsEmpty()) {
                if (select)
                    _curveSelections[curve_index].SelectRange(wxRangeDouble(rect._x, rect.GetRight()));
                else
                    _curveSelections[curve_index].DeselectRange(wxRangeDouble(rect._x, rect.GetRight()));

                return DoSelectDataRange(curve_index, sel, select, send_event);
            } else
                return false;
        } else  // not ordered or not just an x selection
        {
            int i, count = plotData->GetCount();
            int first_sel = -1;
            double* x_data = plotData->GetXData();
            double* y_data = plotData->GetYData();

            int min_ = plotData->GetCount() - 1, max_ = 0;

            wxRangeIntSelection ranges;

            for (i = 0; i < count; i++) {
                if ((is_x_range && xRange.Contains(*x_data)) || (is_y_range && yRange.Contains(*y_data)) ||
                    (!is_x_range && !is_y_range && wxPlotRect2DDoubleContains(*x_data, *y_data, rect))) {
                    if (select) {
                        if (_dataSelections[curve_index].SelectRange(wxRangeInt(i, i))) {
                            ranges.SelectRange(wxRangeInt(i, i));
                            done = true;
                        }
                    } else {
                        if (_dataSelections[curve_index].DeselectRange(wxRangeInt(i, i))) {
                            ranges.SelectRange(wxRangeInt(i, i));
                            done = true;
                        }
                    }

                    min_ = wxMin(min_, i);
                    max_ = wxMin(max_, i);

                    if (done && (first_sel == -1)) first_sel = i;
                }

                x_data++;
                y_data++;
            }

            if (done && (min_ <= max_)) RedrawDataCurve(curve_index, min_, max_);

            if (done) {
                if (select)
                    _curveSelections[curve_index].SelectRange(wxRangeDouble(rect._x, rect.GetRight()));
                else
                    _curveSelections[curve_index].DeselectRange(wxRangeDouble(rect._x, rect.GetRight()));
            }

            if (send_event && done) {
                wxPlotCtrlSelEvent event(wxEVT_PLOTCTRL_RANGE_SEL_CHANGED, GetId(), this);
                event.SetCurve(GetCurve(curve_index), curve_index);
                event.SetDataSelectionRange(wxRangeInt(first_sel, first_sel), select);
                event.SetDataSelections(ranges);
                (void)DoSendEvent(event);
            }

            return done;
        }
    } else {
        if (select)
            done = _curveSelections[curve_index].SelectRange(wxRangeDouble(rect._x, rect.GetRight()));
        else
            done = _curveSelections[curve_index].DeselectRange(wxRangeDouble(rect._x, rect.GetRight()));

        if (send_event && done) {
            wxPlotCtrlSelEvent event(wxEVT_PLOTCTRL_RANGE_SEL_CHANGED, GetId(), this);
            event.SetCurve(GetCurve(curve_index), curve_index);
            event.SetCurveSelectionRange(xRange, select);
            (void)DoSendEvent(event);
        }

        if (done) RedrawCurve(curve_index, xRange._min, xRange._max);

        return done;
    }
}

bool wxPlotCtrl::DoSelectDataRange(int curve_index, const wxRangeInt& range, bool select, bool send_event) {
    wxCHECK_MSG(CurveIndexOk(curve_index), false, wxT("invalid plotcurve index"));
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("invalid selection range"));

    if (_selection_type == wxPLOTCTRL_SELECT_NONE) return false;

    wxPlotData* plotData = GetDataCurve(curve_index);
    wxCHECK_MSG(plotData && (range._min >= 0) && (range._max < (int)plotData->GetCount()), false,
                wxT("invalid index"));

    // check the selection type and clear previous selections if necessary
    if (select) UpdateSelectionState(curve_index, send_event);

    bool done = false;

    if (select)
        done = _dataSelections[curve_index].SelectRange(range);
    else
        done = _dataSelections[curve_index].DeselectRange(range);

    if (send_event && done) {
        wxPlotCtrlSelEvent event(wxEVT_PLOTCTRL_RANGE_SEL_CHANGED, GetId(), this);
        event.SetCurve(GetCurve(curve_index), curve_index);
        event.SetDataSelectionRange(range, select);
        event.GetDataSelections().SelectRange(range);
        (void)DoSendEvent(event);
    }

    if (done) RedrawDataCurve(curve_index, range._min, range._max);

    return done;
}

int wxPlotCtrl::GetSelectedRangeCount(int curve_index) const {
    wxCHECK_MSG(CurveIndexOk(curve_index), 0, wxT("invalid plotcurve index"));

    if (GetDataCurve(curve_index))
        return _dataSelections[curve_index].GetCount();
    else
        return _curveSelections[curve_index].GetCount();
}

bool wxPlotCtrl::ClearSelectedRanges(int curve_index, bool send_event) {
    wxCHECK_MSG((curve_index >= -1) && (curve_index < int(_curves.GetCount())), false, wxT("invalid plotcurve index"));

    bool done = false;

    if (curve_index == -1) {
        for (size_t n = 0; n < _curves.GetCount(); n++) done |= ClearSelectedRanges(n, send_event);

        return done;
    } else {
        if (IsDataCurve(curve_index)) {
            done = _dataSelections[curve_index].GetCount() > 0;
            _dataSelections[curve_index].Clear();
            _curveSelections[curve_index].Clear();
            if (done) RedrawDataCurve(curve_index, 0, GetDataCurve(curve_index)->GetCount() - 1);
        } else {
            done = _curveSelections[curve_index].GetCount() > 0;
            _curveSelections[curve_index].Clear();
            _dataSelections[curve_index].Clear();
            if (done) RedrawCurve(curve_index, _viewRect._x, _viewRect.GetRight());
        }
    }

    if (send_event && done) {
        wxPlotCtrlSelEvent event(wxEVT_PLOTCTRL_RANGE_SEL_CHANGED, GetId(), this);
        event.SetCurve(GetCurve(curve_index), curve_index);

        if (IsDataCurve(curve_index))
            event.SetDataSelectionRange(wxRangeInt(0, GetDataCurve(curve_index)->GetCount() - 1), false);
        else
            event.SetCurveSelectionRange(wxEmptyRangeDouble, false);

        (void)DoSendEvent(event);
    }
    return done;
}

// ------------------------------------------------------------------------
// Get/Set origin, size, and Zoom in/out of view, set scaling, size...
// ------------------------------------------------------------------------
/*

// FIXME - can't shift the bitmap due to off by one errors in ClipLineToRect

void wxPlotCtrl::ShiftOrigin( int dx, int dy, bool send_event )
{
    if ((dx == 0) && (dy == 0)) return;

    if (send_event)
    {
        wxPlotCtrlEvent event( wxEVT_PLOTCTRL_VIEW_CHANGING, GetId(), this);
        event.SetCurve(_activeCurve, _active_index);
        if (DoSendEvent(event)) return;
    }

    {
        wxBitmap tempBitmap(_areaClientRect.width, _areaClientRect.height);
        wxMemoryDC mdc;
        mdc.SelectObject(tempBitmap);
        mdc.DrawBitmap( _area->_bitmap, dx, dy, false );
        mdc.SelectObject(wxNullBitmap);
        _area->_bitmap = tempBitmap;
    }
    wxRect rx, ry;

    _viewRect._x -= dx / _zoom._x;
    _viewRect._y += dy / _zoom._y;

    if (dx != 0)
    {
        rx = wxRect((dx>0 ? -5 : _areaClientRect.width+dx-5), 0, labs(dx)+10, _areaClientRect.height);
        RedrawXAxis(false);
    }
    if (dy != 0)
    {
        ry = wxRect(0, (dy>0 ? -5 : _areaClientRect.height+dy-5), _areaClientRect.width, labs(dy)+10);
        RedrawYAxis(false);
    }

    printf("Shift %d %d rx %d %d %d %d, ry %d %d %d %d\n", dx, dy, rx.x, rx.y, rx.width, rx.height, ry.x, ry.y,
ry.width, ry.height); fflush(stdout);

    if (rx.width > 0) _area->CreateBitmap( rx );
        //_area->Refresh(false, &rx);
    if (ry.height > 0) _area->CreateBitmap( ry );
        //_area->Refresh(false, &ry);

    {
        wxClientDC cdc(_area);
        cdc.DrawBitmap(_area->_bitmap, 0, 0);
    }

    AdjustScrollBars();

    if (send_event)
    {
        wxPlotCtrlEvent event( wxEVT_PLOTCTRL_VIEW_CHANGED, GetId(), this);
        event.SetCurve(_activeCurve, _active_index);
        (void)DoSendEvent( event );
    }
}
*/

bool wxPlotCtrl::MakeCurveVisible(int curve_index, bool send_event) {
    if (curve_index < 0) return SetZoom(-1, -1, 0, 0, send_event);

    wxCHECK_MSG(curve_index < GetCurveCount(), false, wxT("Invalid curve index"));
    wxPlotCurve* curve = GetCurve(curve_index);
    wxCHECK_MSG(curve && curve->Ok(), false, wxT("Invalid curve"));

    // If the curve is a straight line we need to expand it
    wxRect2DDouble curveRect(curve->GetBoundingRect());
    if (curveRect._width == 0) {
        curveRect._x -= .1;
        curveRect._width = .2;
    }
    if (curveRect._height == 0) {
        curveRect._y -= .1;
        curveRect._height = .2;
    }

    return SetViewRect(curveRect, send_event);
}

bool wxPlotCtrl::SetViewRect(const wxRect2DDouble& view, bool send_event) {
    double zoom_x = _areaClientRect.width / view._width;
    double zoom_y = _areaClientRect.height / view._height;
    return SetZoom(zoom_x, zoom_y, view._x, view._y, send_event);
}

bool wxPlotCtrl::SetZoom(const wxPoint2DDouble& zoom, bool around_center, bool send_event) {
    if (around_center && (zoom._x > 0) && (zoom._y > 0)) {
        double origin_x = (_viewRect.GetLeft() + _viewRect._width / 2.0);
        origin_x -= (_viewRect._width / 2.0) * _zoom._x / zoom._x;
        double origin_y = (_viewRect.GetTop() + _viewRect._height / 2.0);
        origin_y -= (_viewRect._height / 2.0) * _zoom._y / zoom._y;
        return SetZoom(zoom._x, zoom._y, origin_x, origin_y, send_event);
    } else
        return SetZoom(zoom._x, zoom._y, _viewRect.GetLeft(), _viewRect.GetTop(), send_event);
}

bool wxPlotCtrl::SetZoom(const wxRect& window, bool send_event) {
    if ((window.GetHeight() < 1) || (window.GetWidth() < 1)) return false;

    double origin_x = GetPlotCoordFromClientX(window.GetX());
    double origin_y = GetPlotCoordFromClientY(window.GetY() + window.GetHeight());
    double zoom_x = _zoom._x * double(_areaClientRect.width) / (window.GetWidth());
    double zoom_y = _zoom._y * double(_areaClientRect.height) / (window.GetHeight());

    bool ok = SetZoom(zoom_x, zoom_y, origin_x, origin_y, send_event);
    if (ok) AddHistoryView();
    return ok;
}

bool wxPlotCtrl::SetZoom(double zoom_x, double zoom_y, double origin_x, double origin_y, bool send_event) {
    // fit to window if zoom <= 0
    if (zoom_x <= 0) {
        zoom_x = double(_areaClientRect.width) / (_curveBoundingRect._width);
        origin_x = _curveBoundingRect._x;
    }
    if (zoom_y <= 0) {
        zoom_y = double(_areaClientRect.height) / (_curveBoundingRect._height);
        origin_y = _curveBoundingRect._y;
    }

    if (_fix_aspectratio) FixAspectRatio(&zoom_x, &zoom_y, &origin_x, &origin_y);

    double view_width = _areaClientRect.width / zoom_x;
    double view_height = _areaClientRect.height / zoom_y;

    if (!IsFinite(zoom_x, wxT("X zoom is NaN"))) return false;
    if (!IsFinite(zoom_y, wxT("Y zoom is NaN"))) return false;
    if (!IsFinite(origin_x, wxT("X origin is not finite"))) return false;
    if (!IsFinite(origin_y, wxT("Y origin is not finite"))) return false;
    if (!IsFinite(view_width, wxT("Plot width is NaN"))) return false;
    if (!IsFinite(view_height, wxT("Plot height is NaN"))) return false;

    bool x_changed = false, y_changed = false;

    if ((_viewRect._x != origin_x) || (_zoom._x != zoom_x)) x_changed = true;
    if ((_viewRect._y != origin_y) || (_zoom._y != zoom_y)) y_changed = true;

    if (x_changed || y_changed) {
        if (send_event) {
            wxPlotCtrlEvent event(wxEVT_PLOTCTRL_VIEW_CHANGING, GetId(), this);
            event.SetCurve(_activeCurve, _active_index);
            event.SetPosition(origin_x, origin_y);
            if (!DoSendEvent(event)) return false;
        }

        _zoom._x = zoom_x;
        _zoom._y = zoom_y;

        _viewRect._x = origin_x;
        _viewRect._y = origin_y;
        _viewRect._width = view_width;
        _viewRect._height = view_height;
    }

    // redraw even if unchanged since we expect that it should be different
    Redraw(wxPLOTCTRL_REDRAW_PLOT | (x_changed ? wxPLOTCTRL_REDRAW_XAXIS : 0) |
           (y_changed ? wxPLOTCTRL_REDRAW_YAXIS : 0));

    if (!_batch_count) AdjustScrollBars();

    if (send_event && (x_changed || y_changed)) {
        wxPlotCtrlEvent event(wxEVT_PLOTCTRL_VIEW_CHANGED, GetId(), this);
        event.SetCurve(_activeCurve, _active_index);
        event.SetPosition(origin_x, origin_y);
        (void)DoSendEvent(event);
    }

    return true;
}

void wxPlotCtrl::SetFixAspectRatio(bool fixed_ratio, double ratio) {
    wxCHECK_RET(ratio > 0, wxT("Invalid aspect ratio"));
    _fix_aspectratio = fixed_ratio;
    _aspectratio = ratio;
}

void wxPlotCtrl::FixAspectRatio(double* zoom_x, double* zoom_y, double* origin_x, double* origin_y) const {
    wxCHECK_RET(zoom_x && zoom_y && origin_x && origin_y, wxT("Invalid parameters"));

    // get the width and height of the view in plot coordinates
    double view_width = _areaClientRect.width / (*zoom_x);
    double view_height = _areaClientRect.height / (*zoom_y);

    // get the centre of the visible area in plot coordinates
    double x_centre = (*origin_x) + view_width / 2;
    double y_centre = (*origin_y) + view_height / 2;

    // if zoom in one direction is more than in the other, reduce both to the lower value
    if ((*zoom_x) * _aspectratio > (*zoom_y)) {
        (*zoom_x) = (*zoom_y) * _aspectratio;
        (*zoom_y) = (*zoom_y);
    } else {
        (*zoom_x) = (*zoom_x);
        (*zoom_y) = (*zoom_x) / _aspectratio;
    }

    // update the plot coordinate view width and height based on the new zooms
    view_width = _areaClientRect.width / (*zoom_x);
    view_height = _areaClientRect.height / (*zoom_y);

    // create the new bottom-left corner of the view in plot coordinates
    *origin_x = x_centre - (view_width / 2);
    *origin_y = y_centre - (view_height / 2);
}

void wxPlotCtrl::SetDefaultBoundingRect(const wxRect2DDouble& rect, bool send_event) {
    wxCHECK_RET(wxFinite(rect._x) && wxFinite(rect._y) && wxFinite(rect.GetRight()) && wxFinite(rect.GetBottom()),
                wxT("bounding rect is NaN"));
    wxCHECK_RET((rect._width > 0) && (rect._height > 0), wxT("Plot Size < 0"));
    _defaultPlotRect = rect;
    CalcBoundingPlotRect();
    SetZoom(_areaClientRect.width / rect._width, _areaClientRect.height / rect._height, rect._x, rect._y,
            send_event);
}

void wxPlotCtrl::AddHistoryView() {
    if (!(wxFinite(_viewRect.GetLeft()) && wxFinite(_viewRect.GetRight()) && wxFinite(_viewRect.GetTop()) &&
          wxFinite(_viewRect.GetBottom())))
        return;

    if ((_history_views_index >= 0) && (_history_views_index < int(_historyViews.GetCount())) &&
        WXRECT2DDOUBLE_EQUAL(_viewRect, _historyViews[_history_views_index]))
        return;

    if (int(_historyViews.GetCount()) >= MAX_PLOT_ZOOMS) {
        if (_history_views_index < int(_historyViews.GetCount()) - 1) {
            _historyViews[_history_views_index] = _viewRect;
        } else {
            _historyViews.RemoveAt(0);
            _historyViews.Add(_viewRect);
        }
    } else {
        _historyViews.Add(_viewRect);
        _history_views_index++;
    }
}

void wxPlotCtrl::NextHistoryView(bool foward, bool send_event) {
    int count = _historyViews.GetCount();

    // try to set it to the "current" history view
    if ((_history_views_index > -1) && (_history_views_index < count)) {
        if (!WXRECT2DDOUBLE_EQUAL(_viewRect, _historyViews[_history_views_index]))
            SetViewRect(_historyViews[_history_views_index], send_event);
    }

    if (foward) {
        if ((count > 0) && (_history_views_index < count - 1)) {
            _history_views_index++;
            SetViewRect(_historyViews[_history_views_index], send_event);
        }
    } else {
        if (_history_views_index > 0) {
            _history_views_index--;
            SetViewRect(_historyViews[_history_views_index], send_event);
        } else
            SetZoom(-1, -1, 0, 0, send_event);
    }
}

void wxPlotCtrl::SetAreaMouseFunction(wxPlotCtrlMouse_Type func, bool send_event) {
    if (func == _area_mouse_func) return;

    if (send_event) {
        wxPlotCtrlEvent event1(wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGING, GetId(), this);
        event1.SetMouseFunction(func);
        if (!DoSendEvent(event1)) return;
    }

    _area_mouse_func = func;

    switch (func) {
        case wxPLOTCTRL_MOUSE_ZOOM: {
            SetAreaMouseCursor(wxCURSOR_MAGNIFIER);  // wxCURSOR_CROSS);
            break;
        }
        case wxPLOTCTRL_MOUSE_SELECT:
        case wxPLOTCTRL_MOUSE_DESELECT: {
            SetAreaMouseCursor(wxCURSOR_ARROW);
            break;
        }
        case wxPLOTCTRL_MOUSE_PAN: {
            SetAreaMouseCursor(wxCURSOR_HAND);
            SetAreaMouseMarker(wxPLOTCTRL_MARKER_NONE);
            break;
        }
        case wxPLOTCTRL_MOUSE_NOTHING:
        default: {
            SetAreaMouseCursor(wxCURSOR_CROSS);
            SetAreaMouseMarker(wxPLOTCTRL_MARKER_NONE);
            break;
        }
    }

    if (send_event) {
        wxPlotCtrlEvent event2(wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGED, GetId(), this);
        event2.SetMouseFunction(func);
        (void)DoSendEvent(event2);
    }
}

void wxPlotCtrl::SetAreaMouseMarker(wxPlotCtrlMarker_Type type) {
    if (type == _area_mouse_marker) return;

    wxClientDC dc(_area);
    DrawMouseMarker(&dc, _area_mouse_marker, _area->_mouseRect);
    _area_mouse_marker = type;
    DrawMouseMarker(&dc, _area_mouse_marker, _area->_mouseRect);
}

void wxPlotCtrl::SetAreaMouseCursor(int cursorid) {
    if (cursorid == _area_mouse_cursorid) return;

    _area_mouse_cursorid = cursorid;

    if (cursorid == wxCURSOR_HAND)
        _area->SetCursor(s_handCursor);
    else if (cursorid == wxPLOTCTRL_CURSOR_GRAB)
        _area->SetCursor(s_grabCursor);
    else
        _area->SetCursor(wxCursor((wxStockCursor)cursorid));
}

void wxPlotCtrl::OnSize(wxSizeEvent&) {
    DoSize();
}

void wxPlotCtrl::DoSize(const wxRect& boundingRect, bool set_window_sizes) {
    if (!_yAxisScrollbar) return;  // we're not created yet

    _redraw_type = wxPLOTCTRL_REDRAW_BLOCKER;  // block OnPaints until done

    wxSize size;

    if (boundingRect == wxRect(0, 0, 0, 0)) {
        UpdateWindowSize();
        size = GetClientSize();
    } else {
        size.x = boundingRect.width;
        size.y = boundingRect.height;
    }

    // wait until we have a normal size
    if ((size.x < 2) || (size.y < 2)) return;

    int sb_width = _yAxisScrollbar->GetSize().GetWidth();

    _clientRect = wxRect(0, 0, size.x - sb_width, size.y - sb_width);

    // title and label positions, add padding here
    wxRect titleRect = _show_title ? wxRect(_titleRect).Inflate(_border) : wxRect(0, 0, 1, 1);
    wxRect xLabelRect = _show_xlabel ? wxRect(_xLabelRect).Inflate(_border) : wxRect(0, 0, 1, 1);
    wxRect yLabelRect = _show_ylabel ? wxRect(_yLabelRect).Inflate(_border) : wxRect(0, 0, 1, 1);

    // this is the border around the area, it lets you see about 1 digit extra on axis
    int area_border = _axisFontSize.y / 2;

    // use the area_border between top of y-axis and area as bottom border of title
    if (_show_title) titleRect.height -= _border;

    int yaxis_width = GetShowYAxis() ? _y_axis_text_width : 1;
    int xaxis_height = GetShowXAxis() ? _axisFontSize.y : area_border;

    int area_width = _clientRect.width - yLabelRect.GetRight() - yaxis_width - 2 * area_border;
    int area_height = _clientRect.height - titleRect.GetBottom() - xaxis_height - xLabelRect.height - area_border;

    _yAxisRect = wxRect(yLabelRect.GetRight(), titleRect.GetBottom(), yaxis_width, area_height + 2 * area_border);

    _xAxisRect = wxRect(_yAxisRect.GetRight(), _yAxisRect.GetBottom() - area_border + 1,
                         area_width + 2 * area_border, xaxis_height);

    _areaRect = wxRect(_yAxisRect.GetRight() + area_border, _yAxisRect.GetTop() + area_border, area_width,
                        area_height);

    // scrollbar to right and bottom
    if (set_window_sizes) {
        _yAxisScrollbar->SetSize(_clientRect.width, 0, sb_width, _clientRect.height);
        _xAxisScrollbar->SetSize(0, _clientRect.height, _clientRect.width, sb_width);

        _yAxis->Show(GetShowYAxis());
        _xAxis->Show(GetShowXAxis());
        if (GetShowYAxis()) _yAxis->SetSize(_yAxisRect);
        if (GetShowXAxis()) _xAxis->SetSize(_xAxisRect);

        _area->SetSize(_areaRect);
        UpdateWindowSize();
    } else
        _areaClientRect = wxRect(wxPoint(0, 0), _areaRect.GetSize());

    _titleRect.x = _areaRect.x + (_areaRect.width - _titleRect.GetWidth()) / 2;
    // _titleRect.x = _clientRect.width/2-_titleRect.GetWidth()/2; center on whole plot
    _titleRect.y = _border;

    _xLabelRect.x = _areaRect.x + _areaRect.width / 2 - _xLabelRect.width / 2;
    _xLabelRect.y = _xAxisRect.GetBottom() + _border;

    _yLabelRect.x = _border;
    _yLabelRect.y = _areaRect.y + _areaRect.height / 2 - _yLabelRect.height / 2;

    double zoom_x = _areaClientRect.width / _viewRect._width;
    double zoom_y = _areaClientRect.height / _viewRect._height;
    if (!IsFinite(zoom_x, wxT("Plot zoom is NaN"))) return;
    if (!IsFinite(zoom_y, wxT("Plot zoom is NaN"))) return;

    if (_fix_aspectratio) {
        FixAspectRatio(&zoom_x, &zoom_y, &_viewRect._x, &_viewRect._y);

        _viewRect._width = _areaClientRect.width / zoom_x;
        _viewRect._height = _areaClientRect.height / zoom_y;
    }

    _zoom._x = zoom_x;
    _zoom._y = zoom_y;

    wxPlotCtrlEvent event(wxEVT_PLOTCTRL_VIEW_CHANGED, GetId(), this);
    event.SetCurve(_activeCurve, _active_index);
    (void)DoSendEvent(event);

    _redraw_type = 0;
    Redraw(wxPLOTCTRL_REDRAW_EVERYTHING);
}

void wxPlotCtrl::CalcBoundingPlotRect() {
    int i, count = _curves.GetCount();

    if (count > 0) {
        bool valid_rect = false;

        wxRect2DDouble rect = _curves[0].GetBoundingRect();

        if (IsFinite(rect._x, wxT("left curve boundary is NaN")) &&
            IsFinite(rect._y, wxT("bottom curve boundary is NaN")) &&
            IsFinite(rect.GetRight(), wxT("right curve boundary is NaN")) &&
            IsFinite(rect.GetBottom(), wxT("top curve boundary is NaN")) && (rect._width >= 0) &&
            (rect._height >= 0)) {
            valid_rect = true;
        } else
            rect = wxNullPlotBounds;

        for (i = 1; i < count; i++) {
            wxRect2DDouble curveRect = _curves[i].GetBoundingRect();

            if ((curveRect._width) <= 0 || (curveRect._height <= 0)) continue;

            wxRect2DDouble newRect;
            if (!valid_rect)
                newRect = curveRect;
            else
                newRect = rect.CreateUnion(curveRect);

            if (IsFinite(newRect._x, wxT("left curve boundary is NaN")) &&
                IsFinite(newRect._y, wxT("bottom curve boundary is NaN")) &&
                IsFinite(newRect.GetRight(), wxT("right curve boundary is NaN")) &&
                IsFinite(newRect.GetBottom(), wxT("top curve boundary is NaN")) && (newRect._width >= 0) &&
                (newRect._height >= 0)) {
                if (!valid_rect) valid_rect = true;
                rect = newRect;
            }
        }

        // maybe just a single point, center it using default size
        bool zeroWidth = false, zeroHeight = false;

        if (rect._width == 0.0) {
            zeroWidth = true;
            rect._x -= 1;
            rect._width = 2;
        }
        if (rect._height == 0.0) {
            zeroHeight = true;
            rect._y -= 1;
            rect._height = 2;
        }

        _curveBoundingRect = rect;

        // add some padding so the edge points can be seen
        double w = (!zeroWidth) ? rect._width / 50.0 : 0.0;
        double h = (!zeroHeight) ? rect._height / 50.0 : 0.0;
        _curveBoundingRect.Inset(-w, -h, -w, -h);
    } else
        _curveBoundingRect = _defaultPlotRect;

    AdjustScrollBars();
}

void wxPlotCtrl::Redraw(int type) {
    if (_batch_count) return;

    if (WXPC_HASBIT(type, wxPLOTCTRL_REDRAW_XAXIS)) {
        _redraw_type |= wxPLOTCTRL_REDRAW_XAXIS;
        AutoCalcXAxisTicks();
        if (_correct_ticks == true) CorrectXAxisTicks();
        CalcXAxisTickPositions();
    }
    if (WXPC_HASBIT(type, wxPLOTCTRL_REDRAW_YAXIS)) {
        _redraw_type |= wxPLOTCTRL_REDRAW_YAXIS;
        AutoCalcYAxisTicks();
        if (_correct_ticks == true) CorrectYAxisTicks();
        CalcYAxisTickPositions();
    }

    if (WXPC_HASBIT(type, wxPLOTCTRL_REDRAW_PLOT)) {
        _redraw_type |= wxPLOTCTRL_REDRAW_PLOT;
        _area->Refresh(false);
    }

    if (WXPC_HASBIT(type, wxPLOTCTRL_REDRAW_XAXIS)) _xAxis->Refresh(false);
    if (WXPC_HASBIT(type, wxPLOTCTRL_REDRAW_YAXIS)) _yAxis->Refresh(false);

    if (WXPC_HASBIT(type, wxPLOTCTRL_REDRAW_WINDOW)) Refresh();
}

void wxPlotCtrl::DrawAreaWindow(wxDC* dc, const wxRect& rect) {
    wxCHECK_RET(dc, wxT("invalid dc"));

    // GTK doesn't like invalid parameters
    wxRect refreshRect = rect;
    wxRect clientRect(GetPlotAreaRect());
    refreshRect.Intersect(clientRect);

    if ((refreshRect.width == 0) || (refreshRect.height == 0)) return;

    dc->SetClippingRegion(refreshRect);

    dc->SetBrush(wxBrush(GetBackgroundColour(), wxBRUSHSTYLE_SOLID));
    dc->SetPen(wxPen(GetBorderColour(), _area_border_width, wxPENSTYLE_SOLID));
    dc->DrawRectangle(clientRect);

    DrawTickMarks(dc, refreshRect);
    DrawMarkers(dc, refreshRect);

    dc->DestroyClippingRegion();

    if (_drawOnScreen) {
        wxMemoryDC mdc;
        wxBitmap bmp = dc->GetAsBitmap();
        mdc.SelectObject(bmp);
        wxGraphicsContext* gc = wxGraphicsContext::Create(mdc);
        wxASSERT(gc);

        int i;
        wxPlotCurve* curve;
        wxPlotCurve* activeCurve = GetActiveCurve();
        for (i = 0; i < GetCurveCount(); i++) {
            curve = GetCurve(i);

            if (curve != activeCurve) {
                if (wxDynamicCast(curve, wxPlotData))
                    DrawDataCurve(gc, wxDynamicCast(curve, wxPlotData), i, refreshRect);
                else
                    DrawCurve(gc, curve, i, refreshRect);
            }
        }
        // active curve is drawn on top
        if (activeCurve) {
            if (wxDynamicCast(activeCurve, wxPlotData))
                DrawDataCurve(gc, wxDynamicCast(activeCurve, wxPlotData), GetActiveIndex(), refreshRect);
            else
                DrawCurve(gc, activeCurve, GetActiveIndex(), refreshRect);
        }

        dc->Blit(rect.x, rect.y, rect.width, rect.height, &mdc, rect.x, rect.y);
        mdc.SelectObject(wxNullBitmap);

        wxDELETE(gc);
    } else {
        int i;
        wxPlotCurve* curve;
        wxPlotCurve* activeCurve = GetActiveCurve();
        for (i = 0; i < GetCurveCount(); i++) {
            curve = GetCurve(i);

            if (curve != activeCurve) {
                if (wxDynamicCast(curve, wxPlotData))
                    DrawDataCurve(dc, wxDynamicCast(curve, wxPlotData), i, refreshRect);
                else
                    DrawCurve(dc, curve, i, refreshRect);
            }
        }
        // active curve is drawn on top
        if (activeCurve) {
            if (wxDynamicCast(activeCurve, wxPlotData))
                DrawDataCurve(dc, wxDynamicCast(activeCurve, wxPlotData), GetActiveIndex(), refreshRect);
            else
                DrawCurve(dc, activeCurve, GetActiveIndex(), refreshRect);
        }
    }

    DrawCurveCursor(dc);
    DrawKey(dc);

    // refresh border
    dc->SetBrush(*wxTRANSPARENT_BRUSH);
    dc->SetPen(wxPen(GetBorderColour(), _area_border_width, wxPENSTYLE_SOLID));
    dc->DrawRectangle(clientRect);

    dc->SetPen(wxNullPen);
    dc->SetBrush(wxNullBrush);
}

void wxPlotCtrl::DrawMouseMarker(wxDC* dc, int type, const wxRect& rect) {
    wxCHECK_RET(dc, wxT("invalid window"));

    if ((rect.width == 0) || (rect.height == 0)) return;

    wxRasterOperationMode logical_fn = dc->GetLogicalFunction();
    dc->SetLogicalFunction(wxINVERT);
    dc->SetBrush(*wxTRANSPARENT_BRUSH);
    dc->SetPen(*wxThePenList->FindOrCreatePen(*wxBLACK, 1, wxPENSTYLE_DOT));

    switch (type) {
        case wxPLOTCTRL_MARKER_NONE:
            break;

        case wxPLOTCTRL_MARKER_RECT: {
            // rects are drawn to width and height - 1, doesn't line up w/ cursor, who cares?
            dc->DrawRectangle(rect.x, rect.y, rect.width, rect.height);
            break;
        }
        case wxPLOTCTRL_MARKER_VERT: {
            if (rect.width != 0) {
                int height = GetClientSize().y;
                dc->DrawLine(rect.x, 1, rect.x, height - 2);
                dc->DrawLine(rect.GetRight() + 1, 1, rect.GetRight() + 1, height - 2);
            }
            break;
        }
        case wxPLOTCTRL_MARKER_HORIZ: {
            if (rect.height != 0) {
                int width = GetClientSize().x;
                dc->DrawLine(1, rect.y, width - 2, rect.y);
                dc->DrawLine(1, rect.GetBottom() + 1, width - 2, rect.GetBottom() + 1);
            }
            break;
        }
        default:
            break;
    }

    dc->SetBrush(wxNullBrush);
    dc->SetPen(wxNullPen);
    dc->SetLogicalFunction(logical_fn);
}

void wxPlotCtrl::DrawCrosshairCursor(wxDC* dc, const wxPoint& pos) {
    wxCHECK_RET(dc, wxT("invalid window"));

    dc->SetPen(*wxBLACK_PEN);
    wxRasterOperationMode logical_fn = dc->GetLogicalFunction();
    dc->SetLogicalFunction(wxINVERT);

    dc->CrossHair(pos.x, pos.y);

    dc->SetPen(wxNullPen);
    dc->SetLogicalFunction(logical_fn);
}

void wxPlotCtrl::DrawDataCurve(wxGraphicsContext* gc, wxPlotData* curve, int curve_index, const wxRect& rect) {
    wxCHECK_RET(gc && _dataCurveDrawer && curve && curve->Ok(), wxT("invalid curve"));

    _dataCurveDrawer->SetDCRect(rect);
    _dataCurveDrawer->SetPlotViewRect(_viewRect);
    _dataCurveDrawer->Draw(gc, curve, curve_index);
}

void wxPlotCtrl::DrawDataCurve(wxDC* dc, wxPlotData* curve, int curve_index, const wxRect& rect) {
    wxCHECK_RET(dc && _dataCurveDrawer && curve && curve->Ok(), wxT("invalid curve"));

    _dataCurveDrawer->SetDCRect(rect);
    _dataCurveDrawer->SetPlotViewRect(_viewRect);
    _dataCurveDrawer->Draw(dc, curve, curve_index);
}

void wxPlotCtrl::DrawCurve(wxGraphicsContext* gc, wxPlotCurve* curve, int curve_index, const wxRect& rect) {
    wxCHECK_RET(gc && _curveDrawer && curve && curve->Ok(), wxT("invalid curve"));

    _curveDrawer->SetDCRect(rect);
    _curveDrawer->SetPlotViewRect(_viewRect);
    _curveDrawer->Draw(gc, curve, curve_index);
}

void wxPlotCtrl::DrawCurve(wxDC* dc, wxPlotCurve* curve, int curve_index, const wxRect& rect) {
    wxCHECK_RET(dc && _curveDrawer && curve && curve->Ok(), wxT("invalid curve"));

    _curveDrawer->SetDCRect(rect);
    _curveDrawer->SetPlotViewRect(_viewRect);
    _curveDrawer->Draw(dc, curve, curve_index);
}

void wxPlotCtrl::RedrawDataCurve(int index, int min_index, int max_index) {
    if (_batch_count) return;

    wxCHECK_RET((index >= 0) && (index < (int)_curves.GetCount()), wxT("invalid curve index"));

    wxPlotData* plotData = GetDataCurve(index);
    wxCHECK_RET(plotData, wxT("not a data curve"));

    int count = plotData->GetCount();
    wxCHECK_RET(
        (min_index <= max_index) && (min_index >= 0) && (max_index >= 0) && (min_index < count) && (max_index < count),
        wxT("invalid data index"));

    wxRect rect(_areaClientRect);
    int cursor_size = GetCursorSize();

    if (plotData->GetIsXOrdered()) {
        double x = plotData->GetXValue(wxMax(min_index - 1, 0));

        if (x > _viewRect.GetRight())
            return;
        else if (x < _viewRect._x)
            rect.x = 0;
        else
            rect.x = GetClientCoordFromPlotX(x) - cursor_size / 2 - 1;

        x = plotData->GetXValue(wxMin(max_index + 1, (int)plotData->GetCount() - 1));

        if (x < _viewRect._x)
            return;
        else if (x > _viewRect.GetRight())
            rect.SetRight(_areaClientRect.width);
        else
            rect.SetRight(GetClientCoordFromPlotX(x) + cursor_size / 2 + 1);

        rect.Intersect(_areaClientRect);
    }

    wxMemoryDC mdc;
    mdc.SelectObject(_area->_bitmap);
    wxGraphicsContext* gc = wxGraphicsContext::Create(mdc);
    wxASSERT(gc);
    DrawDataCurve(gc, plotData, index, rect);
    DrawCurveCursor(&mdc);
    wxClientDC dc(_area);
    dc.Blit(rect.x, rect.y, rect.width, rect.height, &mdc, rect.x, rect.y);
    mdc.SelectObject(wxNullBitmap);
    wxDELETE(gc);
}

void wxPlotCtrl::RedrawCurve(int index, double min_x, double max_x) {
    if (_batch_count) return;

    wxCHECK_RET((min_x <= max_x) && (index >= 0) && (index < (int)_curves.GetCount()), wxT("invalid curve index"));
    wxCHECK_RET(!GetDataCurve(index), wxT("invalid curve"));
    wxRect rect(_areaClientRect);

    if (min_x > _viewRect.GetRight()) return;
    if (min_x < _viewRect._x) min_x = _viewRect._x;

    rect.x = GetClientCoordFromPlotX(min_x);

    if (max_x < _viewRect._x) return;
    if (max_x > _viewRect.GetRight()) max_x = _viewRect.GetRight();

    rect.width = GetClientCoordFromPlotX(max_x) - rect.x;

    if (rect.width < 1) return;

    wxMemoryDC mdc;
    mdc.SelectObject(_area->_bitmap);
    wxGraphicsContext* gc = wxGraphicsContext::Create(mdc);
    wxASSERT(gc);
    DrawCurve(gc, GetCurve(index), index, rect);
    DrawCurveCursor(&mdc);
    wxClientDC dc(_area);
    dc.Blit(rect.x, rect.y, rect.width, rect.height, &mdc, rect.x, rect.y);
    mdc.SelectObject(wxNullBitmap);
    wxDELETE(gc);
}

void wxPlotCtrl::DrawKey(wxDC* dc) {
    wxCHECK_RET(dc && _keyDrawer, wxT("invalid window"));
    if (!GetShowKey() || _keyString.IsEmpty()) return;

    wxRect dcRect(wxPoint(0, 0), GetPlotAreaRect().GetSize());
    _keyDrawer->SetDCRect(dcRect);
    _keyDrawer->SetPlotViewRect(_viewRect);
    _keyDrawer->Draw(dc, _keyString);
}

void wxPlotCtrl::DrawCurveCursor(wxDC* dc) {
    wxCHECK_RET(dc, wxT("invalid window"));
    if (!IsCursorValid()) return;

    _markerDrawer->SetPlotViewRect(_viewRect);
    _markerDrawer->SetDCRect(wxRect(wxPoint(0, 0), _area->GetClientSize()));
    _markerDrawer->Draw(dc, _cursorMarker);
}

void wxPlotCtrl::DrawTickMarks(wxDC* dc, const wxRect& rect) {
    wxRect clientRect(GetPlotAreaRect());
    dc->SetPen(wxPen(GetGridColour(), 1, wxPENSTYLE_SOLID));

    int xtick_length = GetDrawGrid() ? clientRect.height : 10;
    int ytick_length = GetDrawGrid() ? clientRect.width : 10;

    int tick_pos, i;
    // X-axis ticks
    int tick_count = _xAxisTicks.GetCount();
    for (i = 0; i < tick_count; i++) {
        tick_pos = _xAxisTicks[i];
        if (tick_pos < rect.x)
            continue;
        else if (tick_pos > rect.GetRight())
            break;

        dc->DrawLine(tick_pos, clientRect.height, tick_pos, clientRect.height - xtick_length);
    }

    // Y-axis ticks
    tick_count = _yAxisTicks.GetCount();
    for (i = 0; i < tick_count; i++) {
        tick_pos = _yAxisTicks[i];
        if (tick_pos < rect.y)
            break;
        else if (tick_pos > rect.GetBottom())
            continue;

        dc->DrawLine(0, tick_pos, ytick_length, tick_pos);
    }
}

void wxPlotCtrl::DrawMarkers(wxDC* dc, const wxRect& rect) {
    wxCHECK_RET(_markerDrawer, wxT("Invalid marker drawer"));
    _markerDrawer->SetPlotViewRect(_viewRect);
    _markerDrawer->SetDCRect(rect);
    _markerDrawer->Draw(dc, _plotMarkers);
}

void wxPlotCtrl::DrawXAxis(wxDC* dc, bool refresh) {
    wxCHECK_RET(_xAxisDrawer, wxT("Invalid x axis drawer"));

    _xAxisDrawer->SetTickPositions(_xAxisTicks);
    _xAxisDrawer->SetTickLabels(_xAxisTickLabels);
    _xAxisDrawer->SetPlotViewRect(_viewRect);
    wxSize clientSize = _xAxisRect.GetSize();
    _xAxisDrawer->SetDCRect(wxRect(wxPoint(0, 0), clientSize));
    _xAxisDrawer->Draw(dc, refresh);
}

void wxPlotCtrl::DrawYAxis(wxDC* dc, bool refresh) {
    wxCHECK_RET(_yAxisDrawer, wxT("Invalid y axis drawer"));

    _yAxisDrawer->SetTickPositions(_yAxisTicks);
    _yAxisDrawer->SetTickLabels(_yAxisTickLabels);
    _yAxisDrawer->SetPlotViewRect(_viewRect);
    wxSize clientSize = _yAxisRect.GetSize();
    _yAxisDrawer->SetDCRect(wxRect(wxPoint(0, 0), clientSize));
    _yAxisDrawer->Draw(dc, refresh);
}

wxRect ScaleRect(const wxRect& rect, double x_scale, double y_scale) {
    return wxRect(int(rect.x * x_scale + 0.5), int(rect.y * y_scale + 0.5), int(rect.width * x_scale + 0.5),
                  int(rect.height * y_scale + 0.5));
}

void wxPlotCtrl::DrawWholePlot(wxDC* dc, const wxRect& boundingRect, double dpi) {
    _drawOnScreen = false;

    wxCHECK_RET(dc, wxT("invalid dc"));
    wxCHECK_RET(dpi > 0, wxT("Invalid dpi for plot drawing"));

    // set font scale so 1pt = 1pixel at 72dpi
    double fontScale = (double)dpi / 72.0;
    // one pixel wide line equals (_pen_print_width) millimeters wide
    double penScale = (double)_pen_print_width * dpi / 25.4;

    // save old values
    wxFont oldAxisFont = GetAxisFont();
    wxFont oldAxisLabelFont = GetAxisLabelFont();
    wxFont oldPlotTitleFont = GetPlotTitleFont();
    wxFont oldKeyFont = GetKeyFont();

    int old_area_border_width = _area_border_width;
    int old_border = _border;
    int old_cursor_size = _cursorMarker.GetSize().x;
    wxPoint2DDouble old_zoom = _zoom;
    wxRect2DDouble old_view = _viewRect;
    wxRect old_areaClientRect = _areaClientRect;

    // resize border and border pen
    _area_border_width = RINT(_area_border_width * penScale);
    _border = RINT(_border * penScale);

    // resize the curve cursor
    _cursorMarker.SetSize(wxSize(int(old_cursor_size * penScale), int(old_cursor_size * penScale)));

    // resize the fonts
    wxFont axisFont = GetAxisFont();
    axisFont.SetPointSize(wxMax(2, RINT(axisFont.GetPointSize() * fontScale)));
    SetAxisFont(axisFont);

    wxFont axisLabelFont = GetAxisLabelFont();
    axisLabelFont.SetPointSize(wxMax(2, RINT(axisLabelFont.GetPointSize() * fontScale)));
    SetAxisLabelFont(axisLabelFont);

    wxFont plotTitleFont = GetPlotTitleFont();
    plotTitleFont.SetPointSize(wxMax(2, RINT(plotTitleFont.GetPointSize() * fontScale)));
    SetPlotTitleFont(plotTitleFont);

    wxFont keyFont = GetKeyFont();
    keyFont.SetPointSize(wxMax(2, RINT(keyFont.GetPointSize() * fontScale)));
    SetKeyFont(keyFont);

    // reload the original zoom and view rect in case it was changed by any of the font changes
    _zoom = old_zoom;
    _viewRect = old_view;

    // resize all window component rects to the bounding rect
    DoSize(boundingRect, false);
    // AutoCalcTicks();  // don't reset ticks since it might not be WYSIWYG

    // reload the original zoom and view rect in case it was changed by any of the font changes
    _zoom = wxPoint2DDouble(old_zoom._x * double(_areaClientRect.width) / old_areaClientRect.width,
                             old_zoom._y * double(_areaClientRect.height) / old_areaClientRect.height);

    // wxPrintf(wxT("DPI %g, font %g pen%g\n"), dpi, fontScale, penScale);
    // PRINT_WXRECT(wxT("Whole plot"), boundingRect);
    // PRINT_WXRECT(wxT("Area plot"), _areaRect);
    // PRINT_WXRECT(wxT("Xaxis plot"), _xAxisRect);
    // PRINT_WXRECT(wxT("Yaxis plot"), _yAxisRect);

    // draw all components to the provided dc
    dc->SetDeviceOrigin(long(boundingRect.x + _xAxisRect.GetLeft()), long(boundingRect.y + _xAxisRect.GetTop()));
    CalcXAxisTickPositions();
    DrawXAxis(dc, false);

    dc->SetDeviceOrigin(long(boundingRect.x + _yAxisRect.GetLeft()), long(boundingRect.y + _yAxisRect.GetTop()));
    CalcYAxisTickPositions();
    DrawYAxis(dc, false);

    dc->SetDeviceOrigin(long(boundingRect.x + _areaRect.GetLeft()), long(boundingRect.y + _areaRect.GetTop()));
    DrawAreaWindow(dc, _areaClientRect);

    dc->SetDeviceOrigin(boundingRect.x, boundingRect.y);
    DrawPlotCtrl(dc);

    // dc->SetBrush(*wxTRANSPARENT_BRUSH);
    // dc->SetPen(*wxRED_PEN);
    // dc->SetDeviceOrigin(boundingRect.x, boundingRect.y);
    // dc->DrawRectangle(_xAxisRect);
    // dc->DrawRectangle(_yAxisRect);
    // dc->DrawRectangle(_areaRect);

    // restore old values
    _area_border_width = old_area_border_width;
    _border = old_border;
    _cursorMarker.SetSize(wxSize(old_cursor_size, old_cursor_size));

    SetAxisFont(oldAxisFont);
    SetAxisLabelFont(oldAxisLabelFont);
    SetPlotTitleFont(oldPlotTitleFont);
    SetKeyFont(oldKeyFont);
    _zoom = old_zoom;
    _viewRect = old_view;

    // update to window instead of printer
    UpdateWindowSize();
    Redraw(wxPLOTCTRL_REDRAW_WHOLEPLOT);  // recalc ticks for this window
}

// ----------------------------------------------------------------------------
// Axis tick calculations
// ----------------------------------------------------------------------------

void wxPlotCtrl::DoAutoCalcTicks(bool x_axis) {
    double start = 0.0, end = 1.0;
    int i, n, window = 100;

    double* tick_step = NULL;
    double tick_step_fix = -1;
    int* tick_count = NULL;
    wxString* tickFormat = NULL;

    if (x_axis) {
        tick_step = &_xAxisTick_step;
        tick_step_fix = _xAxisTick_step_fix;
        tick_count = &_xAxisTick_count;
        tickFormat = &_xAxisTickFormat;

        window = GetPlotAreaRect().width;
        _xAxisTicks.Clear();  // kill it in case something goes wrong
        start = _viewRect.GetLeft();
        end = _viewRect.GetRight();
        *tick_count = window / (_axisFontSize.x * 10);
    } else {
        tick_step = &_yAxisTick_step;
        tick_step_fix = _yAxisTick_step_fix;
        tick_count = &_yAxisTick_count;
        tickFormat = &_yAxisTickFormat;

        window = GetPlotAreaRect().height;
        _yAxisTicks.Clear();
        start = _viewRect.GetTop();
        end = _viewRect.GetBottom();
        double tick_count_scale = window / (_axisFontSize.y * 2.0) > 2.0 ? 2.0 : 1.5;
        *tick_count = int(window / (_axisFontSize.y * tick_count_scale) + 0.5);
    }

    if (window < 5) return;  // FIXME

    if (!IsFinite(start, wxT("axis range is not finite")) || !IsFinite(end, wxT("axis range is not finite"))) {
        *tick_count = 0;
        return;
    }

    double range = end - start;
    double max = fabs(start) > fabs(end) ? fabs(start) : fabs(end);
    double min = fabs(start) < fabs(end) ? fabs(start) : fabs(end);
    bool exponential = (min >= _min_exponential) || (max < 1.0 / _min_exponential) ? true : false;
    int places = exponential ? 1 : int(floor(fabs(log10(max))));

    if (!IsFinite(range, wxT("axis range is not finite")) || !IsFinite(min, wxT("axis range is not finite")) ||
        !IsFinite(max, wxT("axis range is not finite"))) {
        *tick_count = 0;
        return;
    }

    *tick_step = 1.0;
    int int_log_range = int(log10(range));
    if (int_log_range > 0) {
        for (i = 0; i < int_log_range; i++) (*tick_step) *= 10;
    } else if (int_log_range < 0) {
        for (i = 0; i < -int_log_range; i++) (*tick_step) /= 10;
    }

    double stepsizes[TIC_STEPS] = {.1, .2, .5};
    double step10 = (*tick_step) / 10.0;
    int sigFigs = 0;
    int digits = 0;

    for (n = 0; n < 4; n++) {
        for (i = 0; i < TIC_STEPS; i++) {
            if (tick_step_fix > 0) {
                *tick_step = tick_step_fix;
            } else {
                *tick_step = step10 * stepsizes[i];
            }

            if (exponential)
                sigFigs = labs(int(log10(max)) - int(log10(*tick_step)));
            else
                sigFigs = (*tick_step) >= 1.0 ? 0 : int(ceil(-log10(*tick_step)));

            if (x_axis) {
                digits = 1 + places + (sigFigs > 0 ? 1 + sigFigs : 0) + (exponential ? 4 : 0);
                *tick_count = int(double(window) / double((digits + 3) * _axisFontSize.x) + 0.5);
            }

            if ((range / (*tick_step)) <= (*tick_count)) break;
        }
        if ((range / (*tick_step)) <= (*tick_count)) break;
        step10 *= 10.0;
    }

    // if (!x_axis) wxPrintf(wxT("Ticks %d %lf, %d\n"), n, *tick_step, *tick_count);

    if (sigFigs > 9) sigFigs = 9;  // FIXME

    if (exponential)
        tickFormat->Printf(wxT("%%.%dle"), sigFigs);
    else
        tickFormat->Printf(wxT("%%.%dlf"), sigFigs);

    *tick_count = int(ceil(range / (*tick_step))) + 1;

    //  note : first_tick = ceil(start / tick_step) * tick_step;
}

void wxPlotCtrl::CorrectXAxisTicks() {
    double start = ceil(_viewRect.GetLeft() / _xAxisTick_step) * _xAxisTick_step;
    wxString label;
    label.Printf(_xAxisTickFormat.c_str(), start);
    if (label.ToDouble(&start)) {
        double x = GetClientCoordFromPlotX(start);
        double zoom_x = (GetClientCoordFromPlotX(start + _xAxisTick_step) - x) / _xAxisTick_step;
        double origin_x = start - x / zoom_x;
        BeginBatch();
        if (!SetZoom(zoom_x, _zoom._y, origin_x, _viewRect.GetTop(), true)) _xAxisTick_count = 0;  // oops

        EndBatch(false);  // don't draw just block
    }
}

void wxPlotCtrl::CorrectYAxisTicks() {
    double start = ceil(_viewRect.GetTop() / _yAxisTick_step) * _yAxisTick_step;
    wxString label;
    label.Printf(_yAxisTickFormat.c_str(), start);
    if (label.ToDouble(&start)) {
        double y = GetClientCoordFromPlotY(start);
        double zoom_y = (y - GetClientCoordFromPlotY(start + _yAxisTick_step)) / _yAxisTick_step;
        double origin_y = start - (GetPlotAreaRect().height - y) / zoom_y;
        BeginBatch();
        if (!SetZoom(_zoom._x, zoom_y, _viewRect.GetLeft(), origin_y, true)) _yAxisTick_count = 0;  // oops

        EndBatch(false);
    }
}

void wxPlotCtrl::CalcXAxisTickPositions() {
    double current = ceil(_viewRect.GetLeft() / _xAxisTick_step) * _xAxisTick_step;
    _xAxisTicks.Clear();
    _xAxisTickLabels.Clear();
    int i, x, windowWidth = GetPlotAreaRect().width;
    for (i = 0; i < _xAxisTick_count; i++) {
        if (!IsFinite(current, wxT("axis label is not finite"))) return;

        x = GetClientCoordFromPlotX(current);

        if ((x >= -1) && (x < windowWidth + 2)) {
            _xAxisTicks.Add(x);
            FormatAxisTickLables(_xAxisTickLabels, current, _xAxisTickFormat, _xAxisTickType);
        }

        current += _xAxisTick_step;
    }
}

void wxPlotCtrl::CalcYAxisTickPositions() {
    double current = ceil(_viewRect.GetTop() / _yAxisTick_step) * _yAxisTick_step;
    _yAxisTicks.Clear();
    _yAxisTickLabels.Clear();
    int i, y, windowWidth = GetPlotAreaRect().height;
    for (i = 0; i < _yAxisTick_count; i++) {
        if (!IsFinite(current, wxT("axis label is not finite"))) return;

        y = GetClientCoordFromPlotY(current);

        if ((y >= -1) && (y < windowWidth + 2)) {
            _yAxisTicks.Add(y);
            FormatAxisTickLables(_yAxisTickLabels, current, _yAxisTickFormat, _yAxisTickType);
        }

        current += _yAxisTick_step;
    }
}

void wxPlotCtrl::FormatAxisTickLables(wxArrayString& axisLabels, double current, wxString tickFormat,
                                      wxPlotCtrlAxis_TicksType tickType) {
    switch (tickType) {
        case (wxPLOTCTRL_VALUE):
            axisLabels.Add(wxString::Format(tickFormat.c_str(), current));
            break;
        case (wxPLOTCTRL_DATE_DDMM_FROMMJD):
            axisLabels.Add(GetDateStringFromMJD(current, "DD.MM"));
            break;
        case (wxPLOTCTRL_DATE_MMDD_FROMMJD):
            axisLabels.Add(GetDateStringFromMJD(current, "MM/DD"));
            break;
        case (wxPLOTCTRL_DATE_DDMMYY_FROMMJD):
            axisLabels.Add(GetDateStringFromMJD(current, "DD.MM.YY"));
            break;
        case (wxPLOTCTRL_DATE_MMDDYY_FROMMJD):
            axisLabels.Add(GetDateStringFromMJD(current, "MM/DD/YY"));
            break;
        case (wxPLOTCTRL_DATE_DDMMYYYY_FROMMJD):
            axisLabels.Add(GetDateStringFromMJD(current, "DD.MM.YYYY"));
            break;
        case (wxPLOTCTRL_DATE_MMDDYYYY_FROMMJD):
            axisLabels.Add(GetDateStringFromMJD(current, "MM/DD/YYYY"));
            break;
        default:
            axisLabels.Add(wxString::Format(tickFormat.c_str(), current));
    }
}

wxString wxPlotCtrl::GetDateStringFromMJD(double date, const wxString& format) {
    // To Julian day
    date += 2400001;  // And not 2400000.5 (don't know why)

    int sec, min, hour, day, month, year;

    // Remaining seconds
    double rest = date - floor(date);
    sec = floor(rest * 86400 + 0.5);  // round ti
    hour = floor((float)(sec / 3600));
    sec -= hour * 3600;
    min = floor((float)(sec / 60));
    sec -= min * 60;
    sec = sec;

    // Convertion
    long a, b, c, d, e, z, alpha;

    z = date;
    if (z < 2299161L)
        a = z;
    else {
        alpha = (long)((z - 1867216.25) / 36524.25);
        a = z + 1 + alpha - alpha / 4;
    }
    b = a + 1524;
    c = (long)((b - 122.1) / 365.25);
    d = (long)(365.25 * c);
    e = (long)((b - d) / 30.6001);
    day = (int)b - d - (long)(30.6001 * e);
    month = (int)(e < 13.5) ? e - 1 : e - 13;
    year = (int)(month > 2.5) ? (c - 4716) : c - 4715;
    if (year <= 0) year -= 1;

    // Format the string
    wxString datestr = format;

    wxString yearStr = wxString::Format("%d", year);
    wxString monthStr = wxString::Format("%d", month);
    if (monthStr.Length() < 2) monthStr = "0" + monthStr;
    wxString dayStr = wxString::Format("%d", day);
    if (dayStr.Length() < 2) dayStr = "0" + dayStr;
    wxString hourStr = wxString::Format("%d", hour);
    if (hourStr.Length() < 2) hourStr = "0" + hourStr;
    wxString minuteStr = wxString::Format("%d", min);
    if (minuteStr.Length() < 2) minuteStr = "0" + minuteStr;

    datestr.Replace("YYYY", yearStr);
    datestr.Replace("YY", yearStr.SubString(2, 2));
    datestr.Replace("MM", monthStr, false);
    datestr.Replace("DD", dayStr);
    datestr.Replace("hh", hourStr);
    datestr.Replace("HH", hourStr);
    datestr.Replace("mm", minuteStr);
    datestr.Replace("HH", minuteStr);

    return datestr;
}

// ----------------------------------------------------------------------------
// Event processing
// ----------------------------------------------------------------------------

void wxPlotCtrl::ProcessAreaEVT_MOUSE_EVENTS(wxMouseEvent& event) {
    wxPoint& _mousePt = _area->_mousePt;
    wxRect& _mouseRect = _area->_mouseRect;

    wxPoint lastMousePt = _mousePt;
    _mousePt = event.GetPosition();

    if (event.ButtonDown() && IsTextCtrlShown()) {
        HideTextCtrl(true, true);
        return;
    }

    if (GetGreedyFocus() && (FindFocus() != _area)) _area->SetFocus();

    double plotX = GetPlotCoordFromClientX(_mousePt.x), plotY = GetPlotCoordFromClientY(_mousePt.y);

    wxClientDC dc(_area);

    // Mouse motion
    if (lastMousePt != _area->_mousePt) {
        wxPlotCtrlEvent evt_motion(wxEVT_PLOTCTRL_MOUSE_MOTION, GetId(), this);
        evt_motion.SetPosition(plotX, plotY);
        (void)DoSendEvent(evt_motion);

        // Draw the crosshair cursor
        if (GetCrossHairCursor()) {
            if (!event.Entering() || _area->HasCapture()) DrawCrosshairCursor(&dc, lastMousePt);
            if (!event.Leaving() || _area->HasCapture()) DrawCrosshairCursor(&dc, _mousePt);
        }
    }

    // Wheel scrolling up and down
    if (event.GetWheelRotation() != 0) {
        double dir = event.GetWheelRotation() > 0 ? 0.25 : -0.25;
        SetOrigin(_viewRect.GetLeft(), _viewRect.GetTop() + dir * _viewRect._height, true);
    }

    int active_index = GetActiveIndex();

    // Initial Left down selection
    if (event.LeftDown() || event.LeftDClick()) {
        if (FindFocus() != _area)  // fixme MSW focus problems
            _area->SetFocus();

        if (_area_mouse_cursorid == wxCURSOR_HAND) SetAreaMouseCursor(wxPLOTCTRL_CURSOR_GRAB);

        // send a click or doubleclick event
        wxPlotCtrlEvent click_event(event.ButtonDClick() ? wxEVT_PLOTCTRL_DOUBLECLICKED : wxEVT_PLOTCTRL_CLICKED,
                                    GetId(), this);
        click_event.SetPosition(plotX, plotY);
        (void)DoSendEvent(click_event);

        if (!event.ButtonDClick()) _mouseRect = wxRect(_mousePt, wxSize(0, 0));

        int data_index = -1;
        int curve_index = -1;

        wxPoint2DDouble dpt(2.0 / _zoom._x, 2.0 / _zoom._y);
        wxPoint2DDouble curvePt;

        if (FindCurve(wxPoint2DDouble(plotX, plotY), dpt, curve_index, data_index, &curvePt)) {
            wxPlotCurve* plotCurve = GetCurve(curve_index);
            wxPlotData* plotData = wxDynamicCast(plotCurve, wxPlotData);

            if (plotCurve) {
                wxPlotCtrlEvent pt_click_event(
                    event.ButtonDClick() ? wxEVT_PLOTCTRL_POINT_DOUBLECLICKED : wxEVT_PLOTCTRL_POINT_CLICKED, GetId(),
                    this);
                pt_click_event.SetPosition(curvePt._x, curvePt._y);
                pt_click_event.SetCurve(plotCurve, curve_index);
                pt_click_event.SetCurveDataIndex(data_index);
                (void)DoSendEvent(pt_click_event);

                // send curve selection switched event
                if (curve_index != GetActiveIndex()) SetActiveIndex(curve_index, true);

                if (!event.LeftDClick() && (_area_mouse_func == wxPLOTCTRL_MOUSE_SELECT)) {
                    if (plotData)
                        SelectDataRange(curve_index, wxRangeInt(data_index, data_index), true);
                    else
                        SelectXRange(curve_index, wxRangeDouble(curvePt._x, curvePt._x), true);
                } else if (!event.LeftDClick() && (_area_mouse_func == wxPLOTCTRL_MOUSE_DESELECT)) {
                    if (plotData)
                        DeselectDataRange(curve_index, wxRangeInt(data_index, data_index), true);
                    else
                        DeselectXRange(curve_index, wxRangeDouble(curvePt._x, curvePt._x), true);
                } else {
                    if (plotData)
                        SetCursorDataIndex(curve_index, data_index, true);
                    else
                        SetCursorXPoint(curve_index, curvePt._x, true);
                }

                return;
            }
        }
    }
    // Finished marking rectangle or scrolling, perhaps
    else if (event.LeftUp()) {
        SetCaptureWindow(NULL);

        if (_area_mouse_cursorid == wxPLOTCTRL_CURSOR_GRAB) SetAreaMouseCursor(wxCURSOR_HAND);

        StopMouseTimer();

        if (_mouseRect == wxRect(0, 0, 0, 0)) return;

        wxRect rightedRect = _mouseRect;

        // rightedRect always goes from upper-left to lower-right
        //   don't fix _mouseRect since redrawing will be off
        if (rightedRect.width < 0) {
            rightedRect.x += rightedRect.width;
            rightedRect.width = -rightedRect.width;
        }
        if (rightedRect.height < 0) {
            rightedRect.y += rightedRect.height;
            rightedRect.height = -rightedRect.height;
        }

        // Zoom into image
        if (_area_mouse_func == wxPLOTCTRL_MOUSE_ZOOM) {
            if ((_area_mouse_marker == wxPLOTCTRL_MARKER_RECT) &&
                ((rightedRect.width > 10) && (rightedRect.height > 10)))
                SetZoom(rightedRect, true);
            else if ((_area_mouse_marker == wxPLOTCTRL_MARKER_VERT) && (rightedRect.width > 10))
                SetZoom(wxRect(rightedRect.x, 0, rightedRect.width, _areaClientRect.height), true);
            else if ((_area_mouse_marker == wxPLOTCTRL_MARKER_HORIZ) && (rightedRect.height > 10))
                SetZoom(wxRect(0, rightedRect.y, _areaClientRect.width, rightedRect.height), true);
            else
                DrawMouseMarker(&dc, _area_mouse_marker, _mouseRect);
        }
        // Select a range of points
        else if ((_area_mouse_func == wxPLOTCTRL_MOUSE_SELECT) && (active_index >= 0)) {
            BeginBatch();  // if you select nothing, you don't get a refresh

            wxRect2DDouble plotRect = GetPlotRectFromClientRect(rightedRect);

            if ((_area_mouse_marker == wxPLOTCTRL_MARKER_VERT) && (plotRect._width > 0))
                SelectXRange(active_index, wxRangeDouble(plotRect._x, plotRect.GetRight()), true);
            else if ((_area_mouse_marker == wxPLOTCTRL_MARKER_HORIZ) && (plotRect._height > 0))
                SelectYRange(active_index, wxRangeDouble(plotRect._y, plotRect.GetBottom()), true);
            else if ((plotRect._width > 0) || (plotRect._height > 0))
                SelectRectangle(active_index, plotRect, true);

            _mouseRect = wxRect(0, 0, 0, 0);
            EndBatch();
        }
        // Deselect a range of points
        else if ((_area_mouse_func == wxPLOTCTRL_MOUSE_DESELECT) && (active_index >= 0)) {
            BeginBatch();

            wxRect2DDouble plotRect = GetPlotRectFromClientRect(rightedRect);

            if ((_area_mouse_marker == wxPLOTCTRL_MARKER_VERT) && (plotRect._width > 0))
                DeselectXRange(active_index, wxRangeDouble(plotRect._x, plotRect.GetRight()), true);
            else if ((_area_mouse_marker == wxPLOTCTRL_MARKER_HORIZ) && (plotRect._height > 0))
                DeselectYRange(active_index, wxRangeDouble(plotRect._y, plotRect.GetBottom()), true);
            else if ((plotRect._width > 0) || (plotRect._height > 0))
                DeselectRectangle(active_index, plotRect, true);

            _mouseRect = wxRect(0, 0, 0, 0);
            EndBatch();
        }
        // Nothing to do - erase the rect
        else {
            DrawMouseMarker(&dc, _area_mouse_marker, _mouseRect);
        }

        _mouseRect = wxRect(0, 0, 0, 0);
        return;
    }
    // Marking the rectangle or panning around
    else if (event.LeftIsDown() && event.Dragging()) {
        SetCaptureWindow(_area);

        if (_area_mouse_cursorid == wxCURSOR_HAND) SetAreaMouseCursor(wxPLOTCTRL_CURSOR_GRAB);

        // Move the origin
        if (_area_mouse_func == wxPLOTCTRL_MOUSE_PAN) {
            if (!_areaClientRect.Contains(event.GetPosition())) {
                StartMouseTimer(ID_AREA_TIMER);
            }

            _mouseRect = wxRect(0, 0, 0, 0);  // no marker

            double dx = _mousePt.x - lastMousePt.x;
            double dy = _mousePt.y - lastMousePt.y;
            SetOrigin(_viewRect.GetLeft() - dx / _zoom._x, _viewRect.GetTop() + dy / _zoom._y, true);
            return;
        } else {
            if (_mouseRect != wxRect(0, 0, 0, 0))
                DrawMouseMarker(&dc, _area_mouse_marker, _mouseRect);
            else
                _mouseRect = wxRect(_mousePt, wxSize(1, 1));

            _mouseRect.width = _mousePt.x - _mouseRect.x;
            _mouseRect.height = _mousePt.y - _mouseRect.y;

            DrawMouseMarker(&dc, _area_mouse_marker, _mouseRect);
        }

        return;
    }
    return;
}

void wxPlotCtrl::ProcessAxisEVT_MOUSE_EVENTS(wxMouseEvent& event) {
    if (event.ButtonDown() && IsTextCtrlShown()) {
        HideTextCtrl(true, true);
        return;
    }

    wxPoint pos = event.GetPosition();
    wxPlotCtrlAxis* axisWin = (wxPlotCtrlAxis*)event.GetEventObject();
    wxCHECK_RET(axisWin, wxT("Unknown window"));

    wxPoint& _mousePt = axisWin->_mousePt;

    if (event.LeftIsDown() && (axisWin != GetCaptureWindow())) {
        SetCaptureWindow(axisWin);
        _mousePt = pos;
        return;
    } else if (!event.LeftIsDown()) {
        SetCaptureWindow(NULL);
        StopMouseTimer();
    } else if (event.LeftIsDown()) {
        wxSize winSize = axisWin->GetSize();

        if ((axisWin->IsXAxis() && ((pos.x < 0) || (pos.x > winSize.x))) ||
            (!axisWin->IsXAxis() && ((pos.y < 0) || (pos.y > winSize.y)))) {
            _mousePt = pos;
            StartMouseTimer(axisWin->IsXAxis() ? ID_XAXIS_TIMER : ID_YAXIS_TIMER);
        } else if (IsTimerRunning())
            _mousePt = pos;
    }

    int wheel = event.GetWheelRotation();

    if (wheel != 0) {
        wheel = wheel > 0 ? 1 : wheel < 0 ? -1 : 0;
        double dx = 0, dy = 0;

        if (axisWin->IsXAxis())
            dx = wheel * _viewRect._width / 4.0;
        else
            dy = wheel * _viewRect._height / 4.0;

        SetOrigin(_viewRect.GetLeft() + dx, _viewRect.GetTop() + dy, true);
    }

    if ((!GetScrollOnThumbRelease() && event.LeftIsDown() && event.Dragging()) ||
        (GetScrollOnThumbRelease() && event.LeftUp())) {
        double x = _viewRect.GetLeft();
        double y = _viewRect.GetTop();

        if (axisWin->IsXAxis())
            x += (pos.x - _mousePt.x) / _zoom._x;
        else
            y += (_mousePt.y - pos.y) / _zoom._y;

        SetOrigin(x, y, true);
    }

    if (!GetScrollOnThumbRelease()) _mousePt = pos;
}

void wxPlotCtrl::ProcessAreaEVT_KEY_DOWN(wxKeyEvent& event) {
    // wxPrintf(wxT("wxPlotCtrl::ProcessAreaEVT_KEY_DOWN %d `%c` S%dC%dA%d\n"), int(event.GetKeyCode()),
    // (wxChar)event.GetKeyCode(), event.ShiftDown(), event.ControlDown(), event.AltDown());
    event.Skip(true);

    int code = event.GetKeyCode();
    bool alt = event.AltDown() || (code == WXK_ALT);
    bool ctrl = event.ControlDown() || (code == WXK_CONTROL);
    bool shift = event.ShiftDown() || (code == WXK_SHIFT);

    if (shift && !alt && !ctrl)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_SELECT, true);
    else if (!shift && ctrl && !alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_DESELECT, true);
    else if (ctrl && shift && alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_PAN, true);
    else  // if (!ctrl || !shift || !alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_ZOOM, true);
}

void wxPlotCtrl::ProcessAreaEVT_KEY_UP(wxKeyEvent& event) {
    // wxPrintf(wxT("wxPlotCtrl::ProcessAreaEVT_KEY_UP %d `%c` S%dC%dA%d\n"), int(event.GetKeyCode()),
    // (wxChar)event.GetKeyCode(), event.ShiftDown(), event.ControlDown(), event.AltDown());
    event.Skip(true);

    int code = event.GetKeyCode();
    bool alt = event.AltDown() && (code != WXK_ALT);
    bool ctrl = event.ControlDown() && (code != WXK_CONTROL);
    bool shift = event.ShiftDown() && (code != WXK_SHIFT);

    if (shift && !ctrl && !alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_SELECT, true);
    else if (!shift && ctrl && !alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_DESELECT, true);
    else if (shift && ctrl && alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_PAN, true);
    else  // if (!shift && !ctrl && !alt)
        SetAreaMouseFunction(wxPLOTCTRL_MOUSE_ZOOM, true);
}

void wxPlotCtrl::ProcessAreaEVT_PAINT(wxPaintEvent& WXUNUSED(event), wxPaintDC& dc, wxPlotCtrlArea* areaWin) {
    int redraw_type = GetRedrawType();

    if (WXPC_HASBIT(redraw_type, wxPLOTCTRL_REDRAW_BLOCKER)) return;

    /*
        wxRegionIterator upd( areaWin->GetUpdateRegion() );
        while (upd)
        {
            //wxPrintf(wxT("Region %d %d %d %d \n"), upd.GetX(), upd.GetY(), upd.GetWidth(), upd.GetHeight() );
            upd++;
        }
    */

    if (WXPC_HASBIT(redraw_type, wxPLOTCTRL_REDRAW_PLOT)) {
        wxRect refreshRect(GetPlotAreaRect());
        wxRect clientRect(GetPlotAreaRect());
        refreshRect.Intersect(clientRect);

        if ((refreshRect.width == 0) || (refreshRect.height == 0)) return;

        // if the bitmap need to be recreated then refresh everything
        if (!areaWin->_bitmap.Ok() || (clientRect.width != areaWin->_bitmap.GetWidth()) ||
            (clientRect.height != areaWin->_bitmap.GetHeight())) {
            areaWin->_bitmap.Create(clientRect.width, clientRect.height);
            refreshRect = clientRect;
        }

        wxMemoryDC mdc;
        mdc.SelectObject(areaWin->_bitmap);
        DrawAreaWindow(&mdc, refreshRect);
        mdc.SelectObject(wxNullBitmap);

        SetRedrawType(redraw_type & ~wxPLOTCTRL_REDRAW_PLOT);
    }

    if (areaWin->_bitmap.Ok()) dc.DrawBitmap(areaWin->_bitmap, 0, 0, false);

    if (GetCrossHairCursor() && GetPlotAreaRect().Contains(areaWin->_mousePt))
        DrawCrosshairCursor(&dc, areaWin->_mousePt);

    DrawMouseMarker(&dc, GetAreaMouseMarker(), areaWin->_mouseRect);
}

void wxPlotCtrl::ProcessAxisEVT_PAINT(wxPaintEvent& WXUNUSED(event), wxPaintDC& dc, wxPlotCtrlAxis* axisWin) {
    int redraw_type = GetRedrawType();
    if (WXPC_HASBIT(redraw_type, wxPLOTCTRL_REDRAW_BLOCKER)) return;

    bool redraw = false;

    if (axisWin->IsXAxis() && WXPC_HASBIT(redraw_type, wxPLOTCTRL_REDRAW_XAXIS)) {
        SetRedrawType(redraw_type & ~wxPLOTCTRL_REDRAW_XAXIS);
        redraw = true;
    } else if (!axisWin->IsXAxis() && WXPC_HASBIT(redraw_type, wxPLOTCTRL_REDRAW_YAXIS)) {
        SetRedrawType(redraw_type & ~wxPLOTCTRL_REDRAW_YAXIS);
        redraw = true;
    }

    if (redraw) {
        UpdateWindowSize();
        wxSize clientSize(axisWin->GetClientSize());
        if ((clientSize.x < 2) || (clientSize.y < 2)) return;

        if (!axisWin->_bitmap.Ok() || (clientSize.x != axisWin->_bitmap.GetWidth()) ||
            (clientSize.y != axisWin->_bitmap.GetHeight())) {
            axisWin->_bitmap.Create(clientSize.x, clientSize.y);
        }

        wxMemoryDC mdc;
        mdc.SelectObject(axisWin->_bitmap);
        if (axisWin->IsXAxis())
            DrawXAxis(&mdc, true);
        else
            DrawYAxis(&mdc, true);

        mdc.SelectObject(wxNullBitmap);
    }

    if (axisWin->_bitmap.Ok()) dc.DrawBitmap(axisWin->_bitmap, 0, 0, false);
}

void wxPlotCtrl::OnChar(wxKeyEvent& event) {
    // wxPrintf(wxT("wxPlotCtrl::OnChar %d `%c` S%dC%dA%d\n"), int(event.GetKeyCode()), (wxChar)event.GetKeyCode(),
    // event.ShiftDown(), event.ControlDown(), event.AltDown());

    // select the next curve if possible, or cursor point like left mouse
    if (event.GetKeyCode() == WXK_SPACE) {
        if (event.ShiftDown() || event.ControlDown()) {
            if (IsCursorValid()) {
                if (GetDataCurve(_cursor_curve))
                    DoSelectDataRange(_cursor_curve, wxRangeInt(_cursor_index, _cursor_index), !event.ControlDown(),
                                      true);
                else {
                    wxPoint2DDouble pt(_cursorMarker.GetPlotPosition());
                    DoSelectRectangle(_cursor_curve, wxRect2DDouble(pt._x, 0, pt._x, 0), !event.ControlDown(), true);
                }
            }
        } else {
            int count = GetCurveCount();
            if ((count < 1) || ((count == 1) && (_active_index == 0))) return;
            int index = (_active_index + 1 > count - 1) ? 0 : _active_index + 1;
            SetActiveIndex(index, true);
        }
        return;
    }

    // These are reserved for the program
    if (event.ControlDown() || event.AltDown()) {
        event.Skip(true);
        return;
    }

    switch (event.GetKeyCode()) {
        // cursor keys moves the plot origin around
        case WXK_LEFT:
            SetOrigin(_viewRect.GetLeft() - _viewRect._width / 10.0, _viewRect.GetTop(), true);
            return;
        case WXK_RIGHT:
            SetOrigin(_viewRect.GetLeft() + _viewRect._width / 10.0, _viewRect.GetTop(), true);
            return;
        case WXK_UP:
            SetOrigin(_viewRect.GetLeft(), _viewRect.GetTop() + _viewRect._height / 10.0, true);
            return;
        case WXK_DOWN:
            SetOrigin(_viewRect.GetLeft(), _viewRect.GetTop() - _viewRect._height / 10.0, true);
            return;
        case WXK_PAGEUP:
            SetOrigin(_viewRect.GetLeft(), _viewRect.GetTop() + _viewRect._height / 2.0, true);
            return;
        case WXK_PAGEDOWN:
            SetOrigin(_viewRect.GetLeft(), _viewRect.GetTop() - _viewRect._height / 2.0, true);
            return;

            // Center the plot on the cursor point, or 0,0
        case WXK_HOME: {
            if (IsCursorValid())
                MakeCursorVisible(true, true);
            else
                SetOrigin(-_viewRect._width / 2.0, -_viewRect._height / 2.0, true);

            return;
        }
            // try to make the current curve fully visible
        case WXK_END: {
            wxPlotData* plotData = GetActiveDataCurve();
            if (plotData) {
                wxRect2DDouble bound = plotData->GetBoundingRect();
                bound.Inset(-bound._width / 80.0, -bound._height / 80.0);
                SetViewRect(bound, true);
            } else if (GetActiveCurve()) {
                wxPlotCurve* curve = GetActiveCurve();
                double y, min, max;

                y = max = min = curve->GetY(GetPlotCoordFromClientX(0));

                for (int i = 1; i < _areaClientRect.width; i++) {
                    y = curve->GetY(GetPlotCoordFromClientX(i));

                    if (wxFinite(y) != 0) {
                        if (y > max) max = y;
                        if (y < min) min = y;
                    }
                }

                if (max == min) {
                    min -= 5;
                    max += 5;
                }

                wxRect2DDouble bound(_viewRect._x, min, _viewRect._width, max - min);
                SetViewRect(bound, true);
            }

            return;
        }

            // zoom in and out
        case wxT('a'):
            SetZoom(wxPoint2DDouble(_zoom._x / 1.5, _zoom._y), true);
            return;
        case wxT('d'):
            SetZoom(wxPoint2DDouble(_zoom._x * 1.5, _zoom._y), true);
            return;
        case wxT('w'):
            SetZoom(wxPoint2DDouble(_zoom._x, _zoom._y * 1.5), true);
            return;
        case wxT('x'):
            SetZoom(wxPoint2DDouble(_zoom._x, _zoom._y / 1.5), true);
            return;

        case wxT('q'):
            SetZoom(wxPoint2DDouble(_zoom._x / 1.5, _zoom._y * 1.5), true);
            return;
        case wxT('e'):
            SetZoom(wxPoint2DDouble(_zoom._x * 1.5, _zoom._y * 1.5), true);
            return;
        case wxT('z'):
            SetZoom(wxPoint2DDouble(_zoom._x / 1.5, _zoom._y / 1.5), true);
            return;
        case wxT('c'):
            SetZoom(wxPoint2DDouble(_zoom._x * 1.5, _zoom._y / 1.5), true);
            return;

        case wxT('='): {
            wxRect2DDouble r = GetViewRect();
            r.Scale(.67);
            r.SetCentre(GetAreaMousePoint());
            SetViewRect(r, true);
            return;
        }
        case wxT('-'): {
            wxRect2DDouble r = GetViewRect();
            r.Scale(1.5);
            r.SetCentre(GetAreaMousePoint());
            SetViewRect(r, true);
            return;
        }

        case wxT('s'):
            MakeCurveVisible(GetActiveIndex(), true);
            break;

            // Select previous/next point in a curve
        case wxT('<'):
        case wxT(','): {
            double x = GetPlotCoordFromClientX(_areaClientRect.width - 1);
            wxPlotData* plotData = GetActiveDataCurve();
            if (plotData) {
                if (!IsCursorValid())
                    SetCursorDataIndex(_active_index, plotData->GetIndexFromX(x, wxPlotData::index_floor), true);
                else if (_cursor_index > 0)
                    SetCursorDataIndex(_cursor_curve, _cursor_index - 1, true);
            } else if (_active_index >= 0) {
                if (!IsCursorValid())
                    SetCursorXPoint(_active_index, x, true);
                else {
                    x = GetPlotCoordFromClientX((GetClientCoordFromPlotX(_cursorMarker.GetPlotRect()._x) - 1));
                    SetCursorXPoint(_cursor_curve, x, true);
                }
            }

            MakeCursorVisible(false, true);

            return;
        }
        case wxT('>'):
        case wxT('.'): {
            double x = GetPlotCoordFromClientX(0);
            wxPlotData* plotData = GetActiveDataCurve();
            if (plotData) {
                int count = plotData->GetCount();

                if (!IsCursorValid())
                    SetCursorDataIndex(_active_index, plotData->GetIndexFromX(x, wxPlotData::index_ceil), true);
                else if (_cursor_index < count - 1)
                    SetCursorDataIndex(_cursor_curve, _cursor_index + 1, true);
            } else if (_active_index >= 0) {
                if (!IsCursorValid())
                    SetCursorXPoint(_active_index, x, true);
                else {
                    x = GetPlotCoordFromClientX((GetClientCoordFromPlotX(_cursorMarker.GetPlotRect()._x) + 1));
                    SetCursorXPoint(_cursor_curve, x, true);
                }
            }

            MakeCursorVisible(false, true);

            return;
        }

            // go to the last or next zoom
        case wxT('['):
            NextHistoryView(false, true);
            return;
        case wxT(']'):
            NextHistoryView(true, true);
            return;

            // delete the selected curve
        case WXK_DELETE: {
            if (_activeCurve) DeleteCurve(_activeCurve, true);
            return;
        }
            // delete current selection or go to next curve and delete it's selection
            //   finally invalidate cursor
        case WXK_ESCAPE: {
            BeginBatch();
            if ((_active_index >= 0) && (GetSelectedRangeCount(_active_index) > 0)) {
                ClearSelectedRanges(_active_index, true);
            } else {
                bool has_cleared = false;

                for (int i = 0; i < GetCurveCount(); i++) {
                    if (GetSelectedRangeCount(i) > 0) {
                        ClearSelectedRanges(i, true);
                        has_cleared = true;
                        break;
                    }
                }

                if (!has_cleared) {
                    if (IsCursorValid())
                        InvalidateCursor(true);
                    else if (_active_index > -1)
                        SetActiveIndex(-1, true);
                }
            }
            EndBatch();  // ESC is also a generic clean up routine too!
            break;
        }

        default:
            event.Skip(true);
            break;
    }
}

void wxPlotCtrl::UpdateWindowSize() {
    _areaClientRect = wxRect(wxPoint(0, 0), _area->GetClientSize());
    // If something happens to make these true, there's a problem
    if (_areaClientRect.width < 10) _areaClientRect.width = 10;
    if (_areaClientRect.height < 10) _areaClientRect.height = 10;
}

void wxPlotCtrl::AdjustScrollBars() {
    double range, thumbsize, position;
    double pagesize;

    range = (_curveBoundingRect._width * _zoom._x);
    if (!IsFinite(range, wxT("plot's x range is NaN"))) return;
    if (range > 32000)
        range = 32000;
    else if (range < 1)
        range = 1;

    thumbsize = (range * (_viewRect._width / _curveBoundingRect._width));
    if (!IsFinite(thumbsize, wxT("plot's x range is NaN"))) return;
    if (thumbsize > range)
        thumbsize = range;
    else if (thumbsize < 1)
        thumbsize = 1;

    position = (range * ((_viewRect.GetLeft() - _curveBoundingRect.GetLeft()) / _curveBoundingRect._width));
    if (!IsFinite(position, wxT("plot's x range is NaN"))) return;
    if (position > range - thumbsize)
        position = range - thumbsize;
    else if (position < 0)
        position = 0;
    pagesize = thumbsize;

    _xAxisScrollbar->SetScrollbar(int(position), int(thumbsize), int(range), int(pagesize));

    range = (_curveBoundingRect._height * _zoom._y);
    if (!IsFinite(range, wxT("plot's y range is NaN"))) return;
    if (range > 32000)
        range = 32000;
    else if (range < 1)
        range = 1;

    thumbsize = (range * (_viewRect._height / _curveBoundingRect._height));
    if (!IsFinite(thumbsize, wxT("plot's x range is NaN"))) return;
    if (thumbsize > range)
        thumbsize = range;
    else if (thumbsize < 1)
        thumbsize = 1;

    position = (range - range * ((_viewRect.GetTop() - _curveBoundingRect.GetTop()) / _curveBoundingRect._height) -
                thumbsize);
    if (!IsFinite(position, wxT("plot's x range is NaN"))) return;
    if (position > range - thumbsize)
        position = range - thumbsize;
    else if (position < 0)
        position = 0;
    pagesize = thumbsize;

    _yAxisScrollbar->SetScrollbar(int(position), int(thumbsize), int(range), int(pagesize));
}

void wxPlotCtrl::HideScrollBars() {
    _xAxisScrollbar->SetSize(0, 0);
    _xAxisScrollbar->Hide();
    _yAxisScrollbar->SetSize(0, 0);
    _yAxisScrollbar->Hide();
}

void wxPlotCtrl::OnScroll(wxScrollEvent& event) {
    if (_scroll_on_thumb_release && (event.GetEventType() == wxEVT_SCROLL_THUMBTRACK)) return;

    if (event.GetId() == ID_PLOTCTRL_X_SCROLLBAR) {
        double range = _xAxisScrollbar->GetRange();
        if (range < 1) return;
        double position = _xAxisScrollbar->GetThumbPosition();
        double origin_x = _curveBoundingRect.GetLeft() + _curveBoundingRect._width * (position / range);
        if (!IsFinite(origin_x, wxT("plot's x-origin is NaN"))) return;
        _viewRect._x = origin_x;
        Redraw(wxPLOTCTRL_REDRAW_PLOT | wxPLOTCTRL_REDRAW_XAXIS);
    } else if (event.GetId() == ID_PLOTCTRL_Y_SCROLLBAR) {
        double range = _yAxisScrollbar->GetRange();
        if (range < 1) return;
        double position = _yAxisScrollbar->GetThumbPosition();
        double thumbsize = _yAxisScrollbar->GetThumbSize();
        double origin_y = _curveBoundingRect.GetTop() +
                          _curveBoundingRect._height * ((range - position - thumbsize) / range);
        if (!IsFinite(origin_y, wxT("plot's y-origin is NaN"))) return;
        _viewRect._y = origin_y;
        Redraw(wxPLOTCTRL_REDRAW_PLOT | wxPLOTCTRL_REDRAW_YAXIS);
    }
}

bool wxPlotCtrl::IsFinite(double n, const wxString& msg) const {
    if (!wxFinite(n)) {
        if (!msg.IsEmpty()) {
            wxPlotCtrlEvent event(wxEVT_PLOTCTRL_ERROR, GetId(), (wxPlotCtrl*)this);
            event.SetString(msg);
            (void)DoSendEvent(event);
        }

        return false;
    }

    return true;
}

bool wxPlotCtrl::FindCurve(const wxPoint2DDouble& pt, const wxPoint2DDouble& dpt, int& curve_index, int& data_index,
                           wxPoint2DDouble* curvePt) const {
    curve_index = data_index = -1;

    if (!IsFinite(pt._x, wxT("point is not finite"))) return false;
    if (!IsFinite(pt._y, wxT("point is not finite"))) return false;
    if (!IsFinite(dpt._x, wxT("point is not finite"))) return false;
    if (!IsFinite(dpt._y, wxT("point is not finite"))) return false;

    int curve_count = GetCurveCount();
    if (curve_count < 1) return false;

    for (int n = -1; n < curve_count; n++) {
        // find the point in the selected curve first
        if (n == -1) {
            if (_active_index >= 0)
                n = _active_index;
            else
                n = 0;
        } else if (n == _active_index)
            continue;

        wxPlotCurve* plotCurve = GetCurve(n);
        wxPlotData* plotData = wxDynamicCast(plotCurve, wxPlotData);

        // find the index of the closest point in a wxPlotData curve
        if (plotData) {
            // check if curve has BoundingRect
            wxRect2DDouble rect = plotData->GetBoundingRect();
            if (((rect._width > 0) && ((pt._x + dpt._x < rect.GetLeft()) || (pt._x - dpt._x > rect.GetRight()))) ||
                ((rect._height > 0) &&
                 ((pt._y + dpt._y < rect.GetTop()) || (pt._y - dpt._y > rect.GetBottom())))) {
                if ((n == _active_index) && (n > 0)) n = -1;  // start back at 0
                continue;
            }

            int index = plotData->GetIndexFromXY(pt._x, pt._y, dpt._x);

            double x = plotData->GetXValue(index);
            double y = plotData->GetYValue(index);

            if ((fabs(x - pt._x) <= dpt._x) && (fabs(y - pt._y) <= dpt._y)) {
                curve_index = n;
                data_index = index;
                if (curvePt) *curvePt = wxPoint2DDouble(x, y);
                return true;
            }
        } else  // not a data curve, just find y at this x pos
        {
            wxRect2DDouble rect = plotCurve->GetBoundingRect();
            if ((rect._width <= 0) ||
                ((pt._x + dpt._x >= rect.GetLeft()) && (pt._x - dpt._x <= rect.GetRight()))) {
                if ((rect._height <= 0) || ((pt._y >= rect.GetTop()) && (pt._y - dpt._y <= rect.GetBottom()))) {
                    double y = plotCurve->GetY(pt._x);
                    if (fabs(y - pt._y) <= dpt._y) {
                        curve_index = n;
                        if (curvePt) *curvePt = wxPoint2DDouble(pt._x, y);
                        return true;
                    }
                }
            }
        }

        // continue searching through curves
        // if on the current then start back at the beginning if not already at 0
        if ((n == _active_index) && (n > 0)) n = -1;
    }
    return false;
}

bool wxPlotCtrl::DoSendEvent(wxPlotCtrlEvent& event) const {
    /*
        if (event.GetEventType() != wxEVT_PLOTCTRL_MOUSE_MOTION)
        {
            wxLogDebug(wxT("wxPlotCtrlEvent '%s' CurveIndex: %d, DataIndex: %d, Pos: %lf %lf, MouseFn %d"),
                wxPlotCtrl_GetEventName(event.GetEventType).c_str(),
                event.GetCurveIndex(), event.GetCurveDataIndex(),
                event.GetX(), event.GetY(), event.GetMouseFunction());
        }
    */
    return !GetEventHandler()->ProcessEvent(event) || event.IsAllowed();
}

void wxPlotCtrl::StartMouseTimer(wxWindowID win_id) {
#if wxCHECK_VERSION(2, 5, 0)
    if (_timer && (_timer->GetId() != win_id)) StopMouseTimer();
#else
    StopMouseTimer();  // always stop it I guess
#endif  // wxCHECK_VERSION(2,5,0)

    if (!_timer) _timer = new wxTimer(this, win_id);

    if (!_timer->IsRunning()) _timer->Start(200, true);  // one shot timer
}

void wxPlotCtrl::StopMouseTimer() {
    if (_timer) {
        if (_timer->IsRunning()) _timer->Stop();

        delete _timer;
        _timer = NULL;
    }
}

bool wxPlotCtrl::IsTimerRunning() {
    return (_timer && _timer->IsRunning());
}

void wxPlotCtrl::OnTimer(wxTimerEvent& event) {
    wxPoint mousePt;

    switch (event.GetId()) {
        case ID_AREA_TIMER:
            mousePt = _area->_mousePt;
            break;
        case ID_XAXIS_TIMER:
            mousePt = _xAxis->_mousePt;
            break;
        case ID_YAXIS_TIMER:
            mousePt = _yAxis->_mousePt;
            break;
        default: {
            event.Skip();  // someone else's timer?
            return;
        }
    }

    double dx = (mousePt.x < 0) ? -20 : (mousePt.x > GetPlotAreaRect().width) ? 20 : 0;
    double dy = (mousePt.y < 0) ? 20 : (mousePt.y > GetPlotAreaRect().height) ? -20 : 0;
    dx /= _zoom._x;
    dy /= _zoom._y;

    if (((dx == 0) && (dy == 0)) || !SetOrigin(GetViewRect().GetLeft() + dx, GetViewRect().GetTop() + dy, true)) {
        StopMouseTimer();
    } else
        StartMouseTimer(event.GetId());  // restart timer for another round
}

void wxPlotCtrl::SetCaptureWindow(wxWindow* win) {
    if (_winCapture && (_winCapture != win) && _winCapture->HasCapture()) _winCapture->ReleaseMouse();

    _winCapture = win;

    if (_winCapture && (!_winCapture->HasCapture())) _winCapture->CaptureMouse();
}
