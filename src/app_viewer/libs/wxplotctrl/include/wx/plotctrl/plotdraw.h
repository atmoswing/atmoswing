/////////////////////////////////////////////////////////////////////////////
// Name:        plotdraw.h
// Purpose:     wxPlotDrawer and friends
// Author:      John Labenski
// Modified by:
// Created:     6/5/2002
// Copyright:   (c) John Labenski
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PLOTDRAW_H_
#define _WX_PLOTDRAW_H_

#if defined(__GNUG__) && !defined(NO_GCC_PRAGMA)
#pragma interface "plotdraw.h"
#endif

#include "wx/plotctrl/plotdefs.h"
#include "wx/plotctrl/plotmark.h"
#include "wx/things/genergdi.h"
#include "wx/things/range.h"

class WXDLLIMPEXP_FWD_CORE wxDC;

class WXDLLIMPEXP_FWD_CORE wxGraphicsContext;

class WXDLLIMPEXP_FWD_CORE wxMemoryDC;

class WXDLLIMPEXP_THINGS wxRangeIntSelection;

class WXDLLIMPEXP_THINGS wxRangeDoubleSelection;

class WXDLLIMPEXP_THINGS wxArrayRangeIntSelection;

class WXDLLIMPEXP_THINGS wxArrayRangeDoubleSelection;

class WXDLLIMPEXP_PLOTCTRL wxPlotCtrl;

class WXDLLIMPEXP_PLOTCTRL wxPlotCurve;

class WXDLLIMPEXP_PLOTCTRL wxPlotData;

class WXDLLIMPEXP_PLOTCTRL wxPlotFunction;

class WXDLLIMPEXP_PLOTCTRL wxPlotMarker;

//-----------------------------------------------------------------------------
// wxPlotDrawerBase
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerBase : public wxObject {
  public:
    wxPlotDrawerBase(wxPlotCtrl* owner)
        : wxObject(),
          _owner(owner),
          _pen_scale(1),
          _font_scale(1) {}

    virtual void Draw(wxDC* dc, bool refresh) = 0;

    // Get/Set the owner plotctrl
    wxPlotCtrl* GetOwner() const {
        return _owner;
    }

    void SetOwner(wxPlotCtrl* owner) {
        _owner = owner;
    }

    // Get/Set the rect in the DC to draw on
    void SetDCRect(const wxRect& rect) {
        _dcRect = rect;
    }

    const wxRect& GetDCRect() const {
        return _dcRect;
    }

    // Get/Set the rect of the visible area in the plot window
    void SetPlotViewRect(const wxRect2DDouble& rect) {
        _plotViewRect = rect;
    }

    const wxRect2DDouble& GetPlotViewRect() const {
        return _plotViewRect;
    }

    // Get/Set the scaling for drawing, fonts, pens, etc are scaled
    void SetPenScale(double scale) {
        _pen_scale = scale;
    }

    double GetPenScale() const {
        return _pen_scale;
    }

    void SetFontScale(double scale) {
        _font_scale = scale;
    }

    double GetFontScale() const {
        return _font_scale;
    }

  protected:
    wxPlotCtrl* _owner;
    wxRect _dcRect;
    wxRect2DDouble _plotViewRect;
    double _pen_scale;   // width scaling factor for pens
    double _font_scale;  // scaling factor for font sizes

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerBase)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerArea
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerArea : public wxPlotDrawerBase {
  public:
    wxPlotDrawerArea(wxPlotCtrl* owner)
        : wxPlotDrawerBase(owner) {}

    virtual void Draw(wxDC* dc, bool refresh);

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerArea)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerAxisBase
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerAxisBase : public wxPlotDrawerBase {
  public:
    wxPlotDrawerAxisBase(wxPlotCtrl* owner);

    virtual void Draw(wxDC* dc, bool refresh) = 0;

    void SetTickFont(const wxFont& font) {
        _tickFont = font;
    }

    void SetLabelFont(const wxFont& font) {
        _labelFont = font;
    }

    void SetTickColour(const wxGenericColour& colour) {
        _tickColour = colour;
    }

    void SetLabelColour(const wxGenericColour& colour) {
        _labelColour = colour;
    }

    void SetTickPen(const wxGenericPen& pen) {
        _tickPen = pen;
    }

    void SetBackgroundBrush(const wxGenericBrush& brush) {
        _backgroundBrush = brush;
    }

    void SetTickPositions(const wxArrayInt& pos) {
        _tickPositions = pos;
    }

    void SetTickLabels(const wxArrayString& labels) {
        _tickLabels = labels;
    }

    void SetLabel(const wxString& label) {
        _label = label;
    }

    // implementation
    wxArrayInt _tickPositions;
    wxArrayString _tickLabels;

    wxString _label;

    wxFont _tickFont;
    wxFont _labelFont;
    wxGenericColour _tickColour;
    wxGenericColour _labelColour;

    wxGenericPen _tickPen;
    wxGenericBrush _backgroundBrush;

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerAxisBase)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerXAxis
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerXAxis : public wxPlotDrawerAxisBase {
  public:
    wxPlotDrawerXAxis(wxPlotCtrl* owner)
        : wxPlotDrawerAxisBase(owner) {}

    virtual void Draw(wxDC* dc, bool refresh);

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerXAxis)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerYAxis
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerYAxis : public wxPlotDrawerAxisBase {
  public:
    wxPlotDrawerYAxis(wxPlotCtrl* owner)
        : wxPlotDrawerAxisBase(owner) {}

    virtual void Draw(wxDC* dc, bool refresh);

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerYAxis)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerKey
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerKey : public wxPlotDrawerBase {
  public:
    wxPlotDrawerKey(wxPlotCtrl* owner);

    virtual void Draw(wxDC* WXUNUSED(dc), bool WXUNUSED(refresh)) {}  // unused
    virtual void Draw(wxDC* dc, const wxString& keyString);

    void SetFont(const wxFont& font) {
        _font = font;
    }

    void SetFontColour(const wxGenericColour& colour) {
        _fontColour = colour;
    }

    void SetKeyPosition(const wxPoint& pos) {
        _keyPosition = pos;
    }

    // implementation
    wxFont _font;
    wxGenericColour _fontColour;

    wxPoint _keyPosition;
    bool _key_inside;
    bool _key_bottom;
    int _border;
    int _key_line_width;   // length of line to draw for curve
    int _key_line_margin;  // margin between line and key text

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerKey)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerCurve
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerCurve : public wxPlotDrawerBase {
  public:
    wxPlotDrawerCurve(wxPlotCtrl* owner)
        : wxPlotDrawerBase(owner) {}

    virtual void Draw(wxDC* WXUNUSED(dc), bool WXUNUSED(refresh)) {}  // unused
    virtual void Draw(wxGraphicsContext* gc, wxPlotCurve* curve, int curve_index);

    virtual void Draw(wxDC* dc, wxPlotCurve* curve, int curve_index);

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerCurve)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerDataCurve
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerDataCurve : public wxPlotDrawerBase {
  public:
    wxPlotDrawerDataCurve(wxPlotCtrl* owner)
        : wxPlotDrawerBase(owner) {}

    virtual void Draw(wxDC* WXUNUSED(dc), bool WXUNUSED(refresh)) {}  // unused
    virtual void Draw(wxGraphicsContext* gc, wxPlotData* plotData, int curve_index);

    virtual void Draw(wxDC* dc, wxPlotData* plotData, int curve_index);

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerDataCurve)
};

//-----------------------------------------------------------------------------
// wxPlotDrawerMarkers
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_PLOTCTRL wxPlotDrawerMarker : public wxPlotDrawerBase {
  public:
    wxPlotDrawerMarker(wxPlotCtrl* owner)
        : wxPlotDrawerBase(owner) {}

    virtual void Draw(wxDC* WXUNUSED(dc), bool WXUNUSED(refresh)) {}  // unused
    virtual void Draw(wxDC* dc, const wxArrayPlotMarker& markers);

    virtual void Draw(wxDC* dc, const wxPlotMarker& marker);

  private:
    DECLARE_ABSTRACT_CLASS(wxPlotDrawerMarker)
};

#endif  // _WX_PLOTDRAW_H_
