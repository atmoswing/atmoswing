/////////////////////////////////////////////////////////////////////////////
// Name:            mathplot.cpp
// Purpose:         Framework for plotting in wxWindows
// Original Author: David Schalig
// Maintainer:      Davide Rondini
// Contributors:    Jose Luis Blanco, Val Greene
// Created:         21/07/2003
// Last edit:       09/09/2007
// Copyright:       (c) David Schalig, Davide Rondini
// Licence:         wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifdef __GNUG__
// #pragma implementation "plot.h"
#pragma implementation "mathplot.h"
#endif

// For compilers that support precompilation, includes "wx.h".
#include <wx/window.h>
// #include <wx/wxprec.h>

// Comment out for release operation:
// (Added by J.L.Blanco, Aug 2007)
// #define MATHPLOT_DO_LOGGING

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/colour.h"
#include "wx/cursor.h"
#include "wx/dcclient.h"
#include "wx/font.h"
#include "wx/intl.h"
#include "wx/log.h"
#include "wx/object.h"
#include "wx/settings.h"
#include "wx/sizer.h"

#endif

#include <wx/bmpbuttn.h>
#include <wx/image.h>
#include <wx/module.h>
#include <wx/msgdlg.h>
#include <wx/tipwin.h>

#include <cmath>
#include <cstdio>  // used only for debug
#include <ctime>   // used for representation of x axes involving date

#include "mathplot.h"

// #include "pixel.xpm"

// Memory leak debugging
#ifdef _DEBUG
#ifdef __WXMSW__
#define new DEBUG_NEW
#endif
#endif

// Legend margins
#define mpLEGEND_MARGIN 5
#define mpLEGEND_LINEWIDTH 10

// Minimum axis label separation
#define mpMIN_X_AXIS_LABEL_SEPARATION 64
#define mpMIN_Y_AXIS_LABEL_SEPARATION 32

// Number of pixels to scroll when scrolling by a line
#define mpSCROLL_NUM_PIXELS_PER_LINE 10

// See doxygen comments.
double mpWindow::zoomIncrementalFactor = 1.5;

#ifdef _MSC_VER
#pragma warning(disable : 4125)  // C4125: decimal digit terminates octal escape sequence
#pragma warning(disable : 4100)  // C4100: unreferenced formal parameter
#endif

//-----------------------------------------------------------------------------
// mpLayer
//-----------------------------------------------------------------------------

IMPLEMENT_ABSTRACT_CLASS(mpLayer, wxObject)

mpLayer::mpLayer()
    : _type(mpLAYER_UNDEF) {
    SetPen((wxPen&)*wxBLACK_PEN);
    SetFont((wxFont&)*wxNORMAL_FONT);
    _continuous = FALSE;  // Default
    _showName = TRUE;     // Default
    _drawOutsideMargins = TRUE;
    _visible = true;
}

wxBitmap mpLayer::GetColourSquare(int side) {
    wxBitmap square(side, side, -1);
    wxColour filler = _pen.GetColour();
    wxBrush brush(filler);
    wxMemoryDC dc;
    dc.SelectObject(square);
    dc.SetBackground(brush);
    dc.Clear();
    dc.SelectObject(wxNullBitmap);
    return square;
}

//-----------------------------------------------------------------------------
// mpInfoLayer
//-----------------------------------------------------------------------------
IMPLEMENT_DYNAMIC_CLASS(mpInfoLayer, mpLayer)

mpInfoLayer::mpInfoLayer() {
    _dim = wxRect(0, 0, 1, 1);
    _brush = *wxTRANSPARENT_BRUSH;
    _reference.x = 0;
    _reference.y = 0;
    _winX = 1;  // parent->GetScrX();
    _winY = 1;  // parent->GetScrY();
    _type = mpLAYER_INFO;
}

mpInfoLayer::mpInfoLayer(wxRect rect, const wxBrush* brush)
    : _dim(rect) {
    _brush = *brush;
    _reference.x = rect.x;
    _reference.y = rect.y;
    _winX = 1;  // parent->GetScrX();
    _winY = 1;  // parent->GetScrY();
    _type = mpLAYER_INFO;
}

mpInfoLayer::~mpInfoLayer() {}

void mpInfoLayer::UpdateInfo(mpWindow& w, wxEvent& event) {}

bool mpInfoLayer::Inside(wxPoint& point) {
    return _dim.Contains(point);
}

void mpInfoLayer::Move(wxPoint delta) {
    _dim.SetX(_reference.x + delta.x);
    _dim.SetY(_reference.y + delta.y);
}

void mpInfoLayer::UpdateReference() {
    _reference.x = _dim.x;
    _reference.y = _dim.y;
}

void mpInfoLayer::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        // Adjust relative position inside the window
        int scrx = w.GetScrX();
        int scry = w.GetScrY();
        // Avoid dividing by 0
        if (scrx == 0) scrx = 1;
        if (scry == 0) scry = 1;

        if ((_winX != scrx) || (_winY != scry)) {
#ifdef MATHPLOT_DO_LOGGING
            // wxLogMessage(_("mpInfoLayer::Plot() screen size has changed from %d x %d to %d x %d"), _winX, _winY,
            // scrx, scry);
#endif
            if (_winX != 1) _dim.x = (int)floor((double)(_dim.x * scrx / _winX));
            if (_winY != 1) {
                _dim.y = (int)floor((double)(_dim.y * scry / _winY));
                UpdateReference();
            }
            // Finally update window size
            _winX = scrx;
            _winY = scry;
        }
        dc.SetPen(_pen);
        //     wxImage image0(wxT("pixel.png"), wxBITMAP_TYPE_PNG);
        //     wxBitmap image1(image0);
        //     wxBrush semiWhite(image1);
        dc.SetBrush(_brush);
        dc.DrawRectangle(_dim.x, _dim.y, _dim.width, _dim.height);
    }
}

wxPoint mpInfoLayer::GetPosition() {
    return _dim.GetPosition();
}

wxSize mpInfoLayer::GetSize() {
    return _dim.GetSize();
}

mpInfoCoords::mpInfoCoords()
    : mpInfoLayer() {}

mpInfoCoords::mpInfoCoords(wxRect rect, const wxBrush* brush)
    : mpInfoLayer(rect, brush) {}

mpInfoCoords::~mpInfoCoords() {}

void mpInfoCoords::UpdateInfo(mpWindow& w, wxEvent& event) {
    if (event.GetEventType() == wxEVT_MOTION) {
        int mouseX = ((wxMouseEvent&)event).GetX();
        int mouseY = ((wxMouseEvent&)event).GetY();
        /* It seems that Windows port of wxWidgets don't support multi-line test to be drawn in a wxDC.
           wxGTK instead works perfectly with it.
           Info on wxForum: http://wxforum.shadonet.com/viewtopic.php?t=3451&highlight=drawtext+eol */
#ifdef _WINDOWS
        _content.Printf(wxT("x = %f y = %f"), w.p2x(mouseX), w.p2y(mouseY));
#else
        _content.Printf(wxT("x = %f\ny = %f"), w.p2x(mouseX), w.p2y(mouseY));
#endif
    }
}

void mpInfoCoords::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        // Adjust relative position inside the window
        int scrx = w.GetScrX();
        int scry = w.GetScrY();
        if ((_winX != scrx) || (_winY != scry)) {
#ifdef MATHPLOT_DO_LOGGING
            // wxLogMessage(_("mpInfoLayer::Plot() screen size has changed from %d x %d to %d x %d"), _winX, _winY,
            // scrx, scry);
#endif
            if (_winX != 1) _dim.x = (int)floor((double)(_dim.x * scrx / _winX));
            if (_winY != 1) {
                _dim.y = (int)floor((double)(_dim.y * scry / _winY));
                UpdateReference();
            }
            // Finally update window size
            _winX = scrx;
            _winY = scry;
        }
        dc.SetPen(_pen);
        //     wxImage image0(wxT("pixel.png"), wxBITMAP_TYPE_PNG);
        //     wxBitmap image1(image0);
        //     wxBrush semiWhite(image1);
        dc.SetBrush(_brush);
        dc.SetFont(_font);
        int textX, textY;
        dc.GetTextExtent(_content, &textX, &textY);
        if (_dim.width < textX + 10) _dim.width = textX + 10;
        if (_dim.height < textY + 10) _dim.height = textY + 10;
        dc.DrawRectangle(_dim.x, _dim.y, _dim.width, _dim.height);
        dc.DrawText(_content, _dim.x + 5, _dim.y + 5);
    }
}

mpInfoLegend::mpInfoLegend()
    : mpInfoLayer() {}

mpInfoLegend::mpInfoLegend(wxRect rect, const wxBrush* brush)
    : mpInfoLayer(rect, brush) {}

mpInfoLegend::~mpInfoLegend() {}

void mpInfoLegend::UpdateInfo(mpWindow& w, wxEvent& event) {}

void mpInfoLegend::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        // Adjust relative position inside the window
        int scrx = w.GetScrX();
        int scry = w.GetScrY();
        if ((_winX != scrx) || (_winY != scry)) {
#ifdef MATHPLOT_DO_LOGGING
            // wxLogMessage(_("mpInfoLayer::Plot() screen size has changed from %d x %d to %d x %d"), _winX, _winY,
            // scrx, scry);
#endif
            if (_winX != 1) _dim.x = (int)floor((double)(_dim.x * scrx / _winX));
            if (_winY != 1) {
                _dim.y = (int)floor((double)(_dim.y * scry / _winY));
                UpdateReference();
            }
            // Finally update window size
            _winX = scrx;
            _winY = scry;
        }
        //     wxImage image0(wxT("pixel.png"), wxBITMAP_TYPE_PNG);
        //     wxBitmap image1(image0);
        //     wxBrush semiWhite(image1);
        dc.SetBrush(_brush);
        dc.SetFont(_font);
        const int baseWidth = (mpLEGEND_MARGIN * 2 + mpLEGEND_LINEWIDTH);
        int textX = baseWidth, textY = mpLEGEND_MARGIN;
        int plotCount = 0;
        int posY = 0;
        int tmpX = 0, tmpY = 0;
        mpLayer* ly = NULL;
        wxPen lpen;
        wxString label;
        for (unsigned int p = 0; p < w.CountAllLayers(); p++) {
            ly = w.GetLayer(p);
            if ((ly->GetLayerType() == mpLAYER_PLOT) && (ly->IsVisible())) {
                label = ly->GetName();
                dc.GetTextExtent(label, &tmpX, &tmpY);
                textX = (textX > (tmpX + baseWidth)) ? textX : (tmpX + baseWidth + mpLEGEND_MARGIN);
                textY += (tmpY);
#ifdef MATHPLOT_DO_LOGGING
                // wxLogMessage(_("mpInfoLegend::Plot() Adding layer %d: %s"), p, label.c_str());
#endif
            }
        }
        dc.SetPen(_pen);
        dc.SetBrush(_brush);
        _dim.width = textX;
        if (textY != mpLEGEND_MARGIN) {  // Don't draw any thing if there are no visible layers
            textY += mpLEGEND_MARGIN;
            _dim.height = textY;
            dc.DrawRectangle(_dim.x, _dim.y, _dim.width, _dim.height);
            for (unsigned int p2 = 0; p2 < w.CountAllLayers(); p2++) {
                ly = w.GetLayer(p2);
                if ((ly->GetLayerType() == mpLAYER_PLOT) && (ly->IsVisible())) {
                    label = ly->GetName();
                    lpen = ly->GetPen();
                    dc.GetTextExtent(label, &tmpX, &tmpY);
                    dc.SetPen(lpen);
                    // textX = (textX > (tmpX + baseWidth)) ? textX : (tmpX + baseWidth);
                    // textY += (tmpY + mpLEGEND_MARGIN);
                    posY = _dim.y + mpLEGEND_MARGIN + plotCount * tmpY + (tmpY >> 1);
                    dc.DrawLine(_dim.x + mpLEGEND_MARGIN,                       // X start coord
                                posY,                                           // Y start coord
                                _dim.x + mpLEGEND_LINEWIDTH + mpLEGEND_MARGIN,  // X end coord
                                posY);
                    // dc.DrawRectangle(_dim.x + 5, _dim.y + 5 + plotCount*tmpY, 5, 5);
                    dc.DrawText(label, _dim.x + baseWidth, _dim.y + mpLEGEND_MARGIN + plotCount * tmpY);
                    plotCount++;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// mpLayer implementations - functions
//-----------------------------------------------------------------------------

IMPLEMENT_ABSTRACT_CLASS(mpFX, mpLayer)

mpFX::mpFX(wxString name, int flags) {
    SetName(name);
    _flags = flags;
    _type = mpLAYER_PLOT;
}

void mpFX::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);

        wxCoord startPx = _drawOutsideMargins ? 0 : w.GetMarginLeft();
        wxCoord endPx = _drawOutsideMargins ? w.GetScrX() : w.GetScrX() - w.GetMarginRight();
        wxCoord minYpx = _drawOutsideMargins ? 0 : w.GetMarginTop();
        wxCoord maxYpx = _drawOutsideMargins ? w.GetScrY() : w.GetScrY() - w.GetMarginBottom();

        wxCoord iy = 0;
        if (_pen.GetWidth() <= 1) {
            for (wxCoord i = startPx; i < endPx; ++i) {
                iy = w.y2p(GetY(w.p2x(i)));
                // Draw the point only if you can draw outside margins or if the point is inside margins
                if (_drawOutsideMargins || ((iy >= minYpx) && (iy <= maxYpx)))
                    dc.DrawPoint(i,
                                 iy);  // (wxCoord) ((w.GetPosY() - GetY( (double)i / w.GetScaleX() + w.GetPosX()) ) *
                                       // w.GetScaleY()));
            }
        } else {
            for (wxCoord i = startPx; i < endPx; ++i) {
                iy = w.y2p(GetY(w.p2x(i)));
                // Draw the point only if you can draw outside margins or if the point is inside margins
                if (_drawOutsideMargins || ((iy >= minYpx) && (iy <= maxYpx))) dc.DrawLine(i, iy, i, iy);
                //             wxCoord c = w.y2p( GetY(w.p2x(i)) ); //(wxCoord) ((w.GetPosY() - GetY( (double)i /
                //             w.GetScaleX() + w.GetPosX()) ) * w.GetScaleY());
            }
        }

        if (!_name.IsEmpty() && _showName) {
            dc.SetFont(_font);

            wxCoord tx, ty;
            dc.GetTextExtent(_name, &tx, &ty);

            /*if ((_flags & mpALIGNMASK) == mpALIGN_RIGHT)
                tx = (w.GetScrX()>>1) - tx - 8;
            else if ((_flags & mpALIGNMASK) == mpALIGN_CENTER)
                tx = -tx/2;
            else
                tx = -(w.GetScrX()>>1) + 8;
            */
            if ((_flags & mpALIGNMASK) == mpALIGN_RIGHT)
                tx = (w.GetScrX() - tx) - w.GetMarginRight() - 8;
            else if ((_flags & mpALIGNMASK) == mpALIGN_CENTER)
                tx = ((w.GetScrX() - w.GetMarginRight() - w.GetMarginLeft() - tx) / 2) + w.GetMarginLeft();
            else
                tx = w.GetMarginLeft() + 8;
            dc.DrawText(_name, tx, w.y2p(GetY(w.p2x(tx))));  // (wxCoord) ((w.GetPosY() - GetY( (double)tx /
                                                             // w.GetScaleX() + w.GetPosX())) * w.GetScaleY()) );
        }
    }
}

IMPLEMENT_ABSTRACT_CLASS(mpFY, mpLayer)

mpFY::mpFY(wxString name, int flags) {
    SetName(name);
    _flags = flags;
    _type = mpLAYER_PLOT;
}

void mpFY::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);

        wxCoord i, ix;

        wxCoord startPx = _drawOutsideMargins ? 0 : w.GetMarginLeft();
        wxCoord endPx = _drawOutsideMargins ? w.GetScrX() : w.GetScrX() - w.GetMarginRight();
        wxCoord minYpx = _drawOutsideMargins ? 0 : w.GetMarginTop();
        wxCoord maxYpx = _drawOutsideMargins ? w.GetScrY() : w.GetScrY() - w.GetMarginBottom();

        if (_pen.GetWidth() <= 1) {
            for (i = minYpx; i < maxYpx; ++i) {
                ix = w.x2p(GetX(w.p2y(i)));
                if (_drawOutsideMargins || ((ix >= startPx) && (ix <= endPx))) dc.DrawPoint(ix, i);
            }
        } else {
            for (i = 0; i < w.GetScrY(); ++i) {
                ix = w.x2p(GetX(w.p2y(i)));
                if (_drawOutsideMargins || ((ix >= startPx) && (ix <= endPx))) dc.DrawLine(ix, i, ix, i);
                //             wxCoord c =  w.x2p(GetX(w.p2y(i))); //(wxCoord) ((GetX( (double)i / w.GetScaleY() +
                //             w.GetPosY()) - w.GetPosX()) * w.GetScaleX()); dc.DrawLine(c, i, c, i);
            }
        }

        if (!_name.IsEmpty() && _showName) {
            dc.SetFont(_font);

            wxCoord tx, ty;
            dc.GetTextExtent(_name, &tx, &ty);

            if ((_flags & mpALIGNMASK) == mpALIGN_TOP)
                ty = w.GetMarginTop() + 8;
            else if ((_flags & mpALIGNMASK) == mpALIGN_CENTER)
                ty = ((w.GetScrY() - w.GetMarginTop() - w.GetMarginBottom() - ty) / 2) + w.GetMarginTop();
            else
                ty = w.GetScrY() - 8 - ty - w.GetMarginBottom();

            dc.DrawText(_name, w.x2p(GetX(w.p2y(ty))),
                        ty);  // (wxCoord) ((GetX( (double)i / w.GetScaleY() + w.GetPosY()) - w.GetPosX()) *
                              // w.GetScaleX()), -ty);
        }
    }
}

IMPLEMENT_ABSTRACT_CLASS(mpFXY, mpLayer)

mpFXY::mpFXY(wxString name, int flags) {
    SetName(name);
    _flags = flags;
    _type = mpLAYER_PLOT;
}

void mpFXY::UpdateViewBoundary(wxCoord xnew, wxCoord ynew) {
    // Keep track of how many points have been drawn and the bouding box
    maxDrawX = (xnew > maxDrawX) ? xnew : maxDrawX;
    minDrawX = (xnew < minDrawX) ? xnew : minDrawX;
    maxDrawY = (maxDrawY > ynew) ? maxDrawY : ynew;
    minDrawY = (minDrawY < ynew) ? minDrawY : ynew;
    // drawnPoints++;
}

void mpFXY::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);

        double x, y;
        // Do this to reset the counters to evaluate bounding box for label positioning
        Rewind();
        GetNextXY(x, y);
        maxDrawX = x;
        minDrawX = x;
        maxDrawY = y;
        minDrawY = y;
        // drawnPoints = 0;
        Rewind();

        wxCoord startPx = _drawOutsideMargins ? 0 : w.GetMarginLeft();
        wxCoord endPx = _drawOutsideMargins ? w.GetScrX() : w.GetScrX() - w.GetMarginRight();
        wxCoord minYpx = _drawOutsideMargins ? 0 : w.GetMarginTop();
        wxCoord maxYpx = _drawOutsideMargins ? w.GetScrY() : w.GetScrY() - w.GetMarginBottom();

        wxCoord ix = 0, iy = 0;

        if (!_continuous) {
            // for some reason DrawPoint does not use the current pen,
            // so we use DrawLine for fat pens
            if (_pen.GetWidth() <= 1) {
                while (GetNextXY(x, y)) {
                    ix = w.x2p(x);
                    iy = w.y2p(y);
                    if (_drawOutsideMargins || ((ix >= startPx) && (ix <= endPx) && (iy >= minYpx) && (iy <= maxYpx))) {
                        dc.DrawPoint(ix, iy);
                        UpdateViewBoundary(ix, iy);
                    };
                }
            } else {
                while (GetNextXY(x, y)) {
                    ix = w.x2p(x);
                    iy = w.y2p(y);
                    if (_drawOutsideMargins || ((ix >= startPx) && (ix <= endPx) && (iy >= minYpx) && (iy <= maxYpx))) {
                        dc.DrawLine(ix, iy, ix, iy);
                        UpdateViewBoundary(ix, iy);
                    }
                    //                dc.DrawLine(cx, cy, cx, cy);
                }
            }
        } else {
            // Old code
            wxCoord x0 = 0, c0 = 0;
            bool first = TRUE;
            while (GetNextXY(x, y)) {
                wxCoord x1 = w.x2p(x);  // (wxCoord) ((x - w.GetPosX()) * w.GetScaleX());
                wxCoord c1 = w.y2p(y);  // (wxCoord) ((w.GetPosY() - y) * w.GetScaleY());
                if (first) {
                    first = FALSE;
                    x0 = x1;
                    c0 = c1;
                }
                bool outUp, outDown;
                if ((x1 >= startPx) && (x0 <= endPx)) {
                    outDown = (c0 > maxYpx) && (c1 > maxYpx);
                    outUp = (c0 < minYpx) && (c1 < minYpx);
                    if (!outUp && !outDown) {
                        if (c1 != c0) {
                            if (c0 < minYpx) {
                                x0 = (int)(((float)(minYpx - c0)) / ((float)(c1 - c0)) * (x1 - x0)) + x0;
                                c0 = minYpx;
                            }
                            if (c0 > maxYpx) {
                                x0 = (int)(((float)(maxYpx - c0)) / ((float)(c1 - c0)) * (x1 - x0)) + x0;
                                // wxLogDebug(wxT("old x0 = %d, new x0 = %d"), x0, newX0);
                                // x0 = newX0;
                                c0 = maxYpx;
                            }
                            if (c1 < minYpx) {
                                x1 = (int)(((float)(minYpx - c0)) / ((float)(c1 - c0)) * (x1 - x0)) + x0;
                                c1 = minYpx;
                            }
                            if (c1 > maxYpx) {
                                x1 = (int)(((float)(maxYpx - c0)) / ((float)(c1 - c0)) * (x1 - x0)) + x0;
                                // wxLogDebug(wxT("old x0 = %d, old x1 = %d, new x1 = %d, c0 = %d, c1 = %d, maxYpx =
                                // %d"), x0, x1, newX1, c0, c1, maxYpx); x1 = newX1;
                                c1 = maxYpx;
                            }
                        }
                        if (x1 != x0) {
                            if (x0 < startPx) {
                                c0 = (int)(((float)(startPx - x0)) / ((float)(x1 - x0)) * (c1 - c0)) + c0;
                                x0 = startPx;
                            }
                            if (x1 > endPx) {
                                c1 = (int)(((float)(endPx - x0)) / ((float)(x1 - x0)) * (c1 - c0)) + c0;
                                x1 = endPx;
                            }
                        }
                        dc.DrawLine(x0, c0, x1, c1);
                        UpdateViewBoundary(x1, c1);
                    }
                }
                x0 = x1;
                c0 = c1;
            }
        }

        if (!_name.IsEmpty() && _showName) {
            dc.SetFont(_font);

            wxCoord tx, ty;
            dc.GetTextExtent(_name, &tx, &ty);

            // xxx implement else ... if (!HasBBox())
            {
                // const int sx = w.GetScrX();
                // const int sy = w.GetScrY();

                if ((_flags & mpALIGNMASK) == mpALIGN_NW) {
                    tx = minDrawX + 8;
                    ty = maxDrawY + 8;
                } else if ((_flags & mpALIGNMASK) == mpALIGN_NE) {
                    tx = maxDrawX - tx - 8;
                    ty = maxDrawY + 8;
                } else if ((_flags & mpALIGNMASK) == mpALIGN_SE) {
                    tx = maxDrawX - tx - 8;
                    ty = minDrawY - ty - 8;
                } else {  // mpALIGN_SW
                    tx = minDrawX + 8;
                    ty = minDrawY - ty - 8;
                }
            }

            dc.DrawText(_name, tx, ty);
        }
    }
}

//-----------------------------------------------------------------------------
// mpProfile implementation
//-----------------------------------------------------------------------------

IMPLEMENT_ABSTRACT_CLASS(mpProfile, mpLayer)

mpProfile::mpProfile(wxString name, int flags) {
    SetName(name);
    _flags = flags;
    _type = mpLAYER_PLOT;
}

void mpProfile::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);

        wxCoord startPx = _drawOutsideMargins ? 0 : w.GetMarginLeft();
        wxCoord endPx = _drawOutsideMargins ? w.GetScrX() : w.GetScrX() - w.GetMarginRight();
        wxCoord minYpx = _drawOutsideMargins ? 0 : w.GetMarginTop();
        wxCoord maxYpx = _drawOutsideMargins ? w.GetScrY() : w.GetScrY() - w.GetMarginBottom();

        // Plot profile linking subsequent point of the profile, instead of mpFY, which plots simple points.
        for (wxCoord i = startPx; i < endPx; ++i) {
            wxCoord c0 = w.y2p(GetY(
                w.p2x(i)));  // (wxCoord) ((w.GetYpos() - GetY( (double)i / w.GetXscl() + w.GetXpos()) ) * w.GetYscl());
            wxCoord c1 = w.y2p(GetY(w.p2x(i + 1)));  //(wxCoord) ((w.GetYpos() - GetY( (double)(i+1) / w.GetXscl() +
                                                     //(w.GetXpos() ) ) ) * w.GetYscl());
            // c0 = (c0 <= maxYpx) ? ((c0 >= minYpx) ? c0 : minYpx) : maxYpx;
            // c1 = (c1 <= maxYpx) ? ((c1 >= minYpx) ? c1 : minYpx) : maxYpx;
            if (!_drawOutsideMargins) {
                c0 = (c0 <= maxYpx) ? ((c0 >= minYpx) ? c0 : minYpx) : maxYpx;
                c1 = (c1 <= maxYpx) ? ((c1 >= minYpx) ? c1 : minYpx) : maxYpx;
            }
            dc.DrawLine(i, c0, i + 1, c1);
        };
        if (!_name.IsEmpty()) {
            dc.SetFont(_font);

            wxCoord tx, ty;
            dc.GetTextExtent(_name, &tx, &ty);

            if ((_flags & mpALIGNMASK) == mpALIGN_RIGHT)
                tx = (w.GetScrX() - tx) - w.GetMarginRight() - 8;
            else if ((_flags & mpALIGNMASK) == mpALIGN_CENTER)
                tx = ((w.GetScrX() - w.GetMarginRight() - w.GetMarginLeft() - tx) / 2) + w.GetMarginLeft();
            else
                tx = w.GetMarginLeft() + 8;

            dc.DrawText(_name, tx, w.y2p(GetY(w.p2x(tx))));  //(wxCoord) ((w.GetPosY() - GetY( (double)tx /
                                                             // w.GetScaleX() + w.GetPosX())) * w.GetScaleY()) );
        }
    }
}

//-----------------------------------------------------------------------------
// mpLayer implementations - furniture (scales, ...)
//-----------------------------------------------------------------------------

#define mpLN10 2.3025850929940456840179914546844

IMPLEMENT_DYNAMIC_CLASS(mpScaleX, mpLayer)

mpScaleX::mpScaleX(wxString name, int flags, bool ticks, unsigned int type) {
    SetName(name);
    SetFont((wxFont&)*wxSMALL_FONT);
    SetPen((wxPen&)*wxGREY_PEN);
    _flags = flags;
    _ticks = ticks;
    _labelType = type;
    _type = mpLAYER_AXIS;
    _labelFormat = wxT("");
}

void mpScaleX::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);
        dc.SetFont(_font);
        int orgy = 0;

        const int extend = w.GetScrX();                 //  /2;
        if (_flags == mpALIGN_CENTER) orgy = w.y2p(0);  //(int)(w.GetPosY() * w.GetScaleY());
        if (_flags == mpALIGN_TOP) {
            if (_drawOutsideMargins)
                orgy = X_BORDER_SEPARATION;
            else
                orgy = w.GetMarginTop();
        }
        if (_flags == mpALIGN_BOTTOM) {
            if (_drawOutsideMargins)
                orgy = X_BORDER_SEPARATION;
            else
                orgy = w.GetScrY() - w.GetMarginBottom();
        }
        if (_flags == mpALIGN_BORDER_BOTTOM) orgy = w.GetScrY() - 1;  // dc.LogicalToDeviceY(0) - 1;
        if (_flags == mpALIGN_BORDER_TOP) orgy = 1;                   //-dc.LogicalToDeviceY(0);

        dc.DrawLine(0, orgy, w.GetScrX(), orgy);

        // To cut the axis line when draw outside margin is false, use this code
        /*if (_drawOutsideMargins == true)
            dc.DrawLine( 0, orgy, w.GetScrX(), orgy);
        else
            dc.DrawLine( w.GetMarginLeft(), orgy, w.GetScrX() - w.GetMarginRight(), orgy); */

        const double dig = floor(log(128.0 / w.GetScaleX()) / mpLN10);
        const double step = exp(mpLN10 * dig);
        const double end = w.GetPosX() + (double)extend / w.GetScaleX();

        wxCoord tx, ty;
        wxString s;
        wxString fmt;
        int tmp = (int)dig;
        if (_labelType == mpX_NORMAL) {
            if (!_labelFormat.IsEmpty()) {
                fmt = _labelFormat;
            } else {
                if (tmp >= 1) {
                    fmt = wxT("%.f");
                } else {
                    tmp = 8 - tmp;
                    fmt.Printf(wxT("%%.%df"), tmp >= -1 ? 2 : -tmp);
                }
            }
        } else {
            // Date and/or time axis representation
            if (_labelType == mpX_DATETIME) {
                fmt = (wxT("%04.0f-%02.0f-%02.0fT%02.0f:%02.0f:%02.0f"));
            } else if (_labelType == mpX_DATE) {
                fmt = (wxT("%04.0f-%02.0f-%02.0f"));
            } else if ((_labelType == mpX_TIME) && (end / 60 < 2)) {
                fmt = (wxT("%02.0f:%02.3f"));
            } else {
                fmt = (wxT("%02.0f:%02.0f:%02.0f"));
            }
        }

        // double n = floor( (w.GetPosX() - (double)extend / w.GetScaleX()) / step ) * step ;
        double n0 = floor(
                        (w.GetPosX() /* - (double)(extend - w.GetMarginLeft() - w.GetMarginRight())/ w.GetScaleX() */) /
                        step) *
                    step;
        double n = 0;
#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(wxT("mpScaleX::Plot: dig: %f , step: %f, end: %f, n: %f"), dig, step, end, n0);
#endif
        wxCoord startPx = _drawOutsideMargins ? 0 : w.GetMarginLeft();
        wxCoord endPx = _drawOutsideMargins ? w.GetScrX() : w.GetScrX() - w.GetMarginRight();
        wxCoord minYpx = _drawOutsideMargins ? 0 : w.GetMarginTop();
        wxCoord maxYpx = _drawOutsideMargins ? w.GetScrY() : w.GetScrY() - w.GetMarginBottom();

        tmp = -65535;
        int labelH = 0;  // Control labels heigth to decide where to put axis name (below labels or on top of axis)
        int maxExtent = 0;
        for (n = n0; n < end; n += step) {
            const int p = (int)((n - w.GetPosX()) * w.GetScaleX());
#ifdef MATHPLOT_DO_LOGGING
            wxLogMessage(wxT("mpScaleX::Plot: n: %f -> p = %d"), n, p);
#endif
            if ((p >= startPx) && (p <= endPx)) {
                if (_ticks) {  // draw axis ticks
                    if (_flags == mpALIGN_BORDER_BOTTOM)
                        dc.DrawLine(p, orgy, p, orgy - 4);
                    else
                        dc.DrawLine(p, orgy, p, orgy + 4);
                } else {  // draw grid dotted lines
                    _pen.SetStyle(wxPENSTYLE_DOT);
                    dc.SetPen(_pen);
                    if ((_flags == mpALIGN_BOTTOM) && !_drawOutsideMargins) {
                        dc.DrawLine(p, orgy + 4, p, minYpx);
                    } else {
                        if ((_flags == mpALIGN_TOP) && !_drawOutsideMargins) {
                            dc.DrawLine(p, orgy - 4, p, maxYpx);
                        } else {
                            dc.DrawLine(p, 0 /*-w.GetScrY()*/, p, w.GetScrY());
                        }
                    }
                    _pen.SetStyle(wxPENSTYLE_SOLID);
                    dc.SetPen(_pen);
                }
                // Write ticks labels in s string
                if (_labelType == mpX_NORMAL)
                    s.Printf(fmt, n);
                else if (_labelType == mpX_DATETIME) {
                    time_t when = (time_t)n;
                    struct tm tm = *localtime(&when);
                    s.Printf(fmt, (double)tm.tm_year + 1900, (double)tm.tm_mon + 1, (double)tm.tm_mday,
                             (double)tm.tm_hour, (double)tm.tm_min, (double)tm.tm_sec);
                } else if (_labelType == mpX_DATE) {
                    time_t when = (time_t)n;
                    struct tm tm = *localtime(&when);
                    s.Printf(fmt, (double)tm.tm_year + 1900, (double)tm.tm_mon + 1, (double)tm.tm_mday);
                } else if ((_labelType == mpX_TIME) || (_labelType == mpX_HOURS)) {
                    double modulus = fabs(n);
                    double sign = n / modulus;
                    double hh = floor(modulus / 3600);
                    double mm = floor((modulus - hh * 3600) / 60);
                    double ss = modulus - hh * 3600 - mm * 60;
#ifdef MATHPLOT_DO_LOGGING
                    wxLogMessage(wxT("%02.0f Hours, %02.0f minutes, %02.0f seconds"), sign * hh, mm, ss);
#endif                                    // MATHPLOT_DO_LOGGING
                    if (fmt.Len() == 20)  // Format with hours has 11 chars
                        s.Printf(fmt, sign * hh, mm, floor(ss));
                    else
                        s.Printf(fmt, sign * mm, ss);
                }
                dc.GetTextExtent(s, &tx, &ty);
                labelH = (labelH <= ty) ? ty : labelH;
                /*                if ((p-tx/2-tmp) > 64) { // Problem about non-regular axis labels
                                    if ((_flags == mpALIGN_BORDER_BOTTOM) || (_flags == mpALIGN_TOP)) {
                                        dc.DrawText( s, p-tx/2, orgy-4-ty);
                                    } else {
                                        dc.DrawText( s, p-tx/2, orgy+4);
                                    }
                                    tmp=p+tx/2;
                                }
                                */
                maxExtent = (tx > maxExtent) ? tx : maxExtent;  // Keep in mind max label width
            }
        }
        // Actually draw labels, taking care of not overlapping them, and distributing them regularly
        double labelStep = ceil((maxExtent + mpMIN_X_AXIS_LABEL_SEPARATION) / (w.GetScaleX() * step)) * step;
        for (n = n0; n < end; n += labelStep) {
            const int p = (int)((n - w.GetPosX()) * w.GetScaleX());
#ifdef MATHPLOT_DO_LOGGING
            wxLogMessage(wxT("mpScaleX::Plot: n_label = %f -> p_label = %d"), n, p);
#endif
            if ((p >= startPx) && (p <= endPx)) {
                // Write ticks labels in s string
                if (_labelType == mpX_NORMAL)
                    s.Printf(fmt, n);
                else if (_labelType == mpX_DATETIME) {
                    time_t when = (time_t)n;
                    struct tm tm = *localtime(&when);
                    s.Printf(fmt, (double)tm.tm_year + 1900, (double)tm.tm_mon + 1, (double)tm.tm_mday,
                             (double)tm.tm_hour, (double)tm.tm_min, (double)tm.tm_sec);
                } else if (_labelType == mpX_DATE) {
                    time_t when = (time_t)n;
                    struct tm tm = *localtime(&when);
                    s.Printf(fmt, (double)tm.tm_year + 1900, (double)tm.tm_mon + 1, (double)tm.tm_mday);
                } else if ((_labelType == mpX_TIME) || (_labelType == mpX_HOURS)) {
                    double modulus = fabs(n);
                    double sign = n / modulus;
                    double hh = floor(modulus / 3600);
                    double mm = floor((modulus - hh * 3600) / 60);
                    double ss = modulus - hh * 3600 - mm * 60;
#ifdef MATHPLOT_DO_LOGGING
                    wxLogMessage(wxT("%02.0f Hours, %02.0f minutes, %02.0f seconds"), sign * hh, mm, ss);
#endif                                    // MATHPLOT_DO_LOGGING
                    if (fmt.Len() == 20)  // Format with hours has 11 chars
                        s.Printf(fmt, sign * hh, mm, floor(ss));
                    else
                        s.Printf(fmt, sign * mm, ss);
                }
                dc.GetTextExtent(s, &tx, &ty);
                if ((_flags == mpALIGN_BORDER_BOTTOM) || (_flags == mpALIGN_TOP)) {
                    dc.DrawText(s, p - tx / 2, orgy - 4 - ty);
                } else {
                    dc.DrawText(s, p - tx / 2, orgy + 4);
                }
            }
        }

        // Draw axis name
        dc.GetTextExtent(_name, &tx, &ty);
        switch (_flags) {
            case mpALIGN_BORDER_BOTTOM:
                dc.DrawText(_name, extend - tx - 4, orgy - 8 - ty - labelH);
                break;
            case mpALIGN_BOTTOM: {
                if ((!_drawOutsideMargins) && (w.GetMarginBottom() > (ty + labelH + 8))) {
                    dc.DrawText(_name, (endPx - startPx - tx) >> 1, orgy + 6 + labelH);
                } else {
                    dc.DrawText(_name, extend - tx - 4, orgy - 4 - ty);
                }
            } break;
            case mpALIGN_CENTER:
                dc.DrawText(_name, extend - tx - 4, orgy - 4 - ty);
                break;
            case mpALIGN_TOP: {
                if ((!_drawOutsideMargins) && (w.GetMarginTop() > (ty + labelH + 8))) {
                    dc.DrawText(_name, (endPx - startPx - tx) >> 1, orgy - 6 - ty - labelH);
                } else {
                    dc.DrawText(_name, extend - tx - 4, orgy + 4);
                }
            } break;
            case mpALIGN_BORDER_TOP:
                dc.DrawText(_name, extend - tx - 4, orgy + 6 + labelH);
                break;
            default:
                break;
        }
    }
    /*    if (_flags != mpALIGN_TOP) {

            if ((_flags == mpALIGN_BORDER_BOTTOM) || (_flags == mpALIGN_TOP)) {
                dc.DrawText( _name, extend - tx - 4, orgy - 4 - (ty*2));
            } else {
                dc.DrawText( _name, extend - tx - 4, orgy - 4 - ty); //orgy + 4 + ty);
            }
        }; */
}

IMPLEMENT_DYNAMIC_CLASS(mpScaleY, mpLayer)

mpScaleY::mpScaleY(wxString name, int flags, bool ticks) {
    SetName(name);
    SetFont((wxFont&)*wxSMALL_FONT);
    SetPen((wxPen&)*wxGREY_PEN);
    _flags = flags;
    _ticks = ticks;
    _type = mpLAYER_AXIS;
    _labelFormat = wxT("");
}

void mpScaleY::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);
        dc.SetFont(_font);

        int orgx = 0;
        const int extend = w.GetScrY();                 // /2;
        if (_flags == mpALIGN_CENTER) orgx = w.x2p(0);  //(int)(w.GetPosX() * w.GetScaleX());
        if (_flags == mpALIGN_LEFT) {
            if (_drawOutsideMargins)
                orgx = Y_BORDER_SEPARATION;
            else
                orgx = w.GetMarginLeft();
        }
        if (_flags == mpALIGN_RIGHT) {
            if (_drawOutsideMargins)
                orgx = w.GetScrX() - Y_BORDER_SEPARATION;
            else
                orgx = w.GetScrX() - w.GetMarginRight();
        }
        if (_flags == mpALIGN_BORDER_RIGHT) orgx = w.GetScrX() - 1;  // dc.LogicalToDeviceX(0) - 1;
        if (_flags == mpALIGN_BORDER_LEFT) orgx = 1;                 //-dc.LogicalToDeviceX(0);

        // Draw line
        dc.DrawLine(orgx, 0, orgx, extend);

        // To cut the axis line when draw outside margin is false, use this code
        /* if (_drawOutsideMargins == true)
            dc.DrawLine( orgx, 0, orgx, extend);
        else
            dc.DrawLine( orgx, w.GetMarginTop(), orgx, w.GetScrY() - w.GetMarginBottom()); */

        const double dig = floor(log(128.0 / w.GetScaleY()) / mpLN10);
        const double step = exp(mpLN10 * dig);
        const double end = w.GetPosY() + (double)extend / w.GetScaleY();

        wxCoord tx, ty;
        wxString s;
        wxString fmt;
        int tmp = (int)dig;
        double maxScaleAbs = fabs(w.GetDesiredYmax());
        double minScaleAbs = fabs(w.GetDesiredYmin());
        double endscale = (maxScaleAbs > minScaleAbs) ? maxScaleAbs : minScaleAbs;
        if (_labelFormat.IsEmpty()) {
            if ((endscale < 1e4) && (endscale > 1e-3))
                fmt = wxT("%.2f");
            else
                fmt = wxT("%.1e");
        } else {
            fmt = _labelFormat;
        }
        /*    if (tmp>=1)
            {*/
        //    fmt = wxT("%7.5g");
        //     }
        //     else
        //     {
        //         tmp=8-tmp;
        //         fmt.Printf(wxT("%%.%dg"), (tmp >= -1) ? 2 : -tmp);
        //     }

        double n = floor((w.GetPosY() - (double)(extend - w.GetMarginTop() - w.GetMarginBottom()) / w.GetScaleY()) /
                         step) *
                   step;

        /* wxCoord startPx = _drawOutsideMargins ? 0 : w.GetMarginLeft(); */
        wxCoord endPx = _drawOutsideMargins ? w.GetScrX() : w.GetScrX() - w.GetMarginRight();
        wxCoord minYpx = _drawOutsideMargins ? 0 : w.GetMarginTop();
        wxCoord maxYpx = _drawOutsideMargins ? w.GetScrY() : w.GetScrY() - w.GetMarginBottom();

        tmp = 65536;
        int labelW = 0;
        // Before staring cycle, calculate label height
        int labelHeigth = 0;
        s.Printf(fmt, n);
        dc.GetTextExtent(s, &tx, &labelHeigth);
        for (; n < end; n += step) {
            const int p = (int)((w.GetPosY() - n) * w.GetScaleY());
            if ((p >= minYpx) && (p <= maxYpx)) {
                if (_ticks) {  // Draw axis ticks
                    if (_flags == mpALIGN_BORDER_LEFT) {
                        dc.DrawLine(orgx, p, orgx + 4, p);
                    } else {
                        dc.DrawLine(orgx - 4, p, orgx, p);  //( orgx, p, orgx+4, p);
                    }
                } else {
                    _pen.SetStyle(wxPENSTYLE_DOT);
                    dc.SetPen(_pen);
                    if ((_flags == mpALIGN_LEFT) && !_drawOutsideMargins) {
                        dc.DrawLine(orgx - 4, p, endPx, p);
                    } else {
                        if ((_flags == mpALIGN_RIGHT) && !_drawOutsideMargins) {
                            dc.DrawLine(minYpx, p, orgx + 4, p);
                        } else {
                            dc.DrawLine(0 /*-w.GetScrX()*/, p, w.GetScrX(), p);
                        }
                    }
                    _pen.SetStyle(wxPENSTYLE_SOLID);
                    dc.SetPen(_pen);
                }
                // Print ticks labels
                s.Printf(fmt, n);
                dc.GetTextExtent(s, &tx, &ty);
#ifdef MATHPLOT_DO_LOGGING
                if (ty != labelHeigth)
                    wxLogMessage(wxT("mpScaleY::Plot: ty(%f) and labelHeigth(%f) differ!"), ty, labelHeigth);
#endif
                labelW = (labelW <= tx) ? tx : labelW;
                if ((tmp - p + labelHeigth / 2) > mpMIN_Y_AXIS_LABEL_SEPARATION) {
                    if ((_flags == mpALIGN_BORDER_LEFT) || (_flags == mpALIGN_RIGHT))
                        dc.DrawText(s, orgx + 4, p - ty / 2);
                    else
                        dc.DrawText(s, orgx - 4 - tx, p - ty / 2);  //( s, orgx+4, p-ty/2);
                    tmp = p - labelHeigth / 2;
                }
            }
        }
        // Draw axis name

        dc.GetTextExtent(_name, &tx, &ty);
        switch (_flags) {
            case mpALIGN_BORDER_LEFT:
                dc.DrawText(_name, labelW + 8, 4);
                break;
            case mpALIGN_LEFT: {
                if ((!_drawOutsideMargins) && (w.GetMarginLeft() > (ty + labelW + 8))) {
                    dc.DrawRotatedText(_name, orgx - 6 - labelW - ty, (maxYpx - minYpx + tx) >> 1, 90);
                } else {
                    dc.DrawText(_name, orgx + 4, 4);
                }
            } break;
            case mpALIGN_CENTER:
                dc.DrawText(_name, orgx + 4, 4);
                break;
            case mpALIGN_RIGHT: {
                if ((!_drawOutsideMargins) && (w.GetMarginRight() > (ty + labelW + 8))) {
                    dc.DrawRotatedText(_name, orgx + 6 + labelW, (maxYpx - minYpx + tx) >> 1, 90);
                } else {
                    dc.DrawText(_name, orgx - tx - 4, 4);
                }
            } break;
            case mpALIGN_BORDER_RIGHT:
                dc.DrawText(_name, orgx - 6 - tx - labelW, 4);
                break;
            default:
                break;
        }
    }

    /*    if (_flags != mpALIGN_RIGHT) {
        dc.GetTextExtent(_name, &tx, &ty);
        if (_flags == mpALIGN_BORDER_LEFT) {
                dc.DrawText( _name, orgx-tx-4, -extend + ty + 4);
            } else {
                if (_flags == mpALIGN_BORDER_RIGHT )
                    dc.DrawText( _name, orgx-(tx*2)-4, -extend + ty + 4);
                else
                    dc.DrawText( _name, orgx + 4, -extend + 4);
            }
        }; */
}

//-----------------------------------------------------------------------------
// mpWindow
//-----------------------------------------------------------------------------

IMPLEMENT_DYNAMIC_CLASS(mpWindow, wxWindow)

BEGIN_EVENT_TABLE(mpWindow, wxWindow)
EVT_PAINT(mpWindow::OnPaint)
EVT_SIZE(mpWindow::OnSize)
EVT_SCROLLWIN_THUMBTRACK(mpWindow::OnScrollThumbTrack)
EVT_SCROLLWIN_PAGEUP(mpWindow::OnScrollPageUp)
EVT_SCROLLWIN_PAGEDOWN(mpWindow::OnScrollPageDown)
EVT_SCROLLWIN_LINEUP(mpWindow::OnScrollLineUp)
EVT_SCROLLWIN_LINEDOWN(mpWindow::OnScrollLineDown)
EVT_SCROLLWIN_TOP(mpWindow::OnScrollTop)
EVT_SCROLLWIN_BOTTOM(mpWindow::OnScrollBottom)

    EVT_MIDDLE_UP(mpWindow::OnShowPopupMenu) EVT_RIGHT_DOWN(mpWindow::OnMouseRightDown)  // JLB
    EVT_RIGHT_UP(mpWindow::OnShowPopupMenu) EVT_MOUSEWHEEL(mpWindow::OnMouseWheel)       // JLB
    EVT_MOTION(mpWindow::OnMouseMove)                                                    // JLB
    EVT_LEFT_DOWN(mpWindow::OnMouseLeftDown) EVT_LEFT_UP(mpWindow::OnMouseLeftRelease)

        EVT_MENU(mpID_CENTER, mpWindow::OnCenter) EVT_MENU(mpID_FIT, mpWindow::OnFit)
            EVT_MENU(mpID_ZOOM_IN, mpWindow::OnZoomIn) EVT_MENU(mpID_ZOOM_OUT, mpWindow::OnZoomOut)
                EVT_MENU(mpID_LOCKASPECT, mpWindow::OnLockAspect) EVT_MENU(mpID_HELP_MOUSE, mpWindow::OnMouseHelp)
                    END_EVENT_TABLE()

                        mpWindow::mpWindow(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size,
                                           long flag)
    : wxWindow(parent, id, pos, size, flag, wxT("mathplot")) {
    _scaleX = _scaleY = 1.0;
    _posX = _posY = 0;
    _desiredXmin = _desiredYmin = 0;
    _desiredXmax = _desiredYmax = 1;
    _scrX = _scrY = 64;  // Fixed from _scrX = _scrX = 64;
    _minX = _minY = 0;
    _maxX = _maxY = 0;
    _last_lx = _last_ly = 0;
    _buff_bmp = NULL;
    _enableDoubleBuffer = FALSE;
    _enableMouseNavigation = TRUE;
    _mouseMovedAfterRightClick = FALSE;
    _movingInfoLayer = NULL;
    // Set margins to 0
    _marginTop = 0;
    _marginRight = 0;
    _marginBottom = 0;
    _marginLeft = 0;

    _lockaspect = FALSE;

    _popmenu.Append(mpID_CENTER, _("Center"), _("Center plot view to this position"));
    _popmenu.Append(mpID_FIT, _("Fit"), _("Set plot view to show all items"));
    _popmenu.Append(mpID_ZOOM_IN, _("Zoom in"), _("Zoom in plot view."));
    _popmenu.Append(mpID_ZOOM_OUT, _("Zoom out"), _("Zoom out plot view."));
    _popmenu.AppendCheckItem(mpID_LOCKASPECT, _("Lock aspect"), _("Lock horizontal and vertical zoom aspect."));
    _popmenu.Append(mpID_HELP_MOUSE, _("Show mouse commands..."), _("Show help about the mouse commands."));

    _layers.clear();
    SetBackgroundColour(*wxWHITE);
    _bgColour = *wxWHITE;
    _fgColour = *wxBLACK;

    _enableScrollBars = false;
    SetSizeHints(128, 128);

    // J.L.Blanco: Eliminates the "flick" with the double buffer.
    SetBackgroundStyle(wxBG_STYLE_CUSTOM);

    UpdateAll();
}

mpWindow::~mpWindow() {
    // Free all the layers:
    DelAllLayers(true, false);

    if (_buff_bmp) {
        delete _buff_bmp;
        _buff_bmp = NULL;
    }
}

// Mouse handler, for detecting when the user drag with the right button or just "clicks" for the menu
// JLB
void mpWindow::OnMouseRightDown(wxMouseEvent& event) {
    _mouseMovedAfterRightClick = FALSE;
    _mouseRClick_X = event.GetX();
    _mouseRClick_Y = event.GetY();
    if (_enableMouseNavigation) {
        SetCursor(*wxCROSS_CURSOR);
    }
}

// Process mouse wheel events
// JLB
void mpWindow::OnMouseWheel(wxMouseEvent& event) {
    if (!_enableMouseNavigation) {
        event.Skip();
        return;
    }

    //     GetClientSize( &_scrX,&_scrY);

    if (event._controlDown) {
        wxPoint clickPt(event.GetX(), event.GetY());
        // CTRL key hold: Zoom in/out:
        if (event.GetWheelRotation() > 0)
            ZoomIn(clickPt);
        else
            ZoomOut(clickPt);
    } else {
        // Scroll vertically or horizontally (this is SHIFT is hold down).
        int change = -event.GetWheelRotation();  // Opposite direction (More intuitive)!
        double changeUnitsX = change / _scaleX;
        double changeUnitsY = change / _scaleY;

        if (event._shiftDown) {
            _posX += changeUnitsX;
            _desiredXmax += changeUnitsX;
            _desiredXmin += changeUnitsX;
        } else {
            _posY -= changeUnitsY;
            _desiredYmax -= changeUnitsY;
            _desiredYmax -= changeUnitsY;
        }

        UpdateAll();
    }
}

// If the user "drags" with the right buttom pressed, do "pan"
// JLB
void mpWindow::OnMouseMove(wxMouseEvent& event) {
    if (!_enableMouseNavigation) {
        event.Skip();
        return;
    }

    if (event._rightDown) {
        _mouseMovedAfterRightClick = TRUE;  // Hides the popup menu after releasing the button!

        // The change:
        int Ax = _mouseRClick_X - event.GetX();
        int Ay = _mouseRClick_Y - event.GetY();

        // For the next event, use relative to this coordinates.
        _mouseRClick_X = event.GetX();
        _mouseRClick_Y = event.GetY();

        double Ax_units = Ax / _scaleX;
        double Ay_units = -Ay / _scaleY;

        _posX += Ax_units;
        _posY += Ay_units;
        _desiredXmax += Ax_units;
        _desiredXmin += Ax_units;
        _desiredYmax += Ay_units;
        _desiredYmin += Ay_units;

        UpdateAll();

#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(_("[mpWindow::OnMouseMove] Ax:%i Ay:%i _posX:%f _posY:%f"), Ax, Ay, _posX, _posY);
#endif
    } else {
        if (event._leftDown) {
            if (_movingInfoLayer == NULL) {
                wxClientDC dc(this);
                wxPen pen(*wxBLACK, 1, wxPENSTYLE_DOT);
                dc.SetPen(pen);
                dc.SetBrush(*wxTRANSPARENT_BRUSH);
                dc.DrawRectangle(_mouseLClick_X, _mouseLClick_Y, event.GetX() - _mouseLClick_X,
                                 event.GetY() - _mouseLClick_Y);
            } else {
                wxPoint moveVector(event.GetX() - _mouseLClick_X, event.GetY() - _mouseLClick_Y);
                _movingInfoLayer->Move(moveVector);
            }
            UpdateAll();
        } else {
            wxLayerList::iterator li;
            for (li = _layers.begin(); li != _layers.end(); li++) {
                if ((*li)->IsInfo() && (*li)->IsVisible()) {
                    mpInfoLayer* tmpLyr = (mpInfoLayer*)(*li);
                    tmpLyr->UpdateInfo(*this, event);
                    // UpdateAll();
                    RefreshRect(tmpLyr->GetRectangle());
                }
            }
            /* if (_coordTooltip) {
                wxString toolTipContent;
                toolTipContent.Printf(_("X = %f\nY = %f"), p2x(event.GetX()), p2y(event.GetY()));
                wxTipWindow** ptr = NULL;
                wxRect rectBounds(event.GetX(), event.GetY(), 5, 5);
                wxTipWindow* tip = new wxTipWindow(this, toolTipContent, 100, ptr, &rectBounds);

            } */
        }
    }
    event.Skip();
}

void mpWindow::OnMouseLeftDown(wxMouseEvent& event) {
    _mouseLClick_X = event.GetX();
    _mouseLClick_Y = event.GetY();
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::OnMouseLeftDown() X = %d , Y = %d"), event.GetX(),
                 event.GetY()); /*_mouseLClick_X, _mouseLClick_Y);*/
#endif
    wxPoint pointClicked = event.GetPosition();
    _movingInfoLayer = IsInsideInfoLayer(pointClicked);
    if (_movingInfoLayer != NULL) {
#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(_("mpWindow::OnMouseLeftDown() started moving layer %lx"),
                     (long int)_movingInfoLayer); /*_mouseLClick_X, _mouseLClick_Y);*/
#endif
    }
    event.Skip();
}

void mpWindow::OnMouseLeftRelease(wxMouseEvent& event) {
    wxPoint release(event.GetX(), event.GetY());
    wxPoint press(_mouseLClick_X, _mouseLClick_Y);
    if (_movingInfoLayer != NULL) {
        _movingInfoLayer->UpdateReference();
        _movingInfoLayer = NULL;
    } else {
        if (release != press) {
            ZoomRect(press, release);
        } /*else {
            if (_coordTooltip) {
                wxString toolTipContent;
                toolTipContent.Printf(_("X = %f\nY = %f"), p2x(event.GetX()), p2y(event.GetY()));
                SetToolTip(toolTipContent);
            }
        } */
    }
    event.Skip();
}

void mpWindow::Fit() {
    if (UpdateBBox()) Fit(_minX, _maxX, _minY, _maxY);
}

// JL
void mpWindow::Fit(double xMin, double xMax, double yMin, double yMax, wxCoord* printSizeX, wxCoord* printSizeY) {
    // Save desired borders:
    _desiredXmin = xMin;
    _desiredXmax = xMax;
    _desiredYmin = yMin;
    _desiredYmax = yMax;

    if (printSizeX != NULL && printSizeY != NULL) {
        // Printer:
        _scrX = *printSizeX;
        _scrY = *printSizeY;
    } else {
        // Normal case (screen):
        GetClientSize(&_scrX, &_scrY);
    }

    double Ax, Ay;

    Ax = xMax - xMin;
    Ay = yMax - yMin;

    _scaleX = (Ax != 0) ? (_scrX - _marginLeft - _marginRight) / Ax : 1;  // _scaleX = (Ax!=0) ? _scrX/Ax : 1;
    _scaleY = (Ay != 0) ? (_scrY - _marginTop - _marginBottom) / Ay : 1;  // _scaleY = (Ay!=0) ? _scrY/Ay : 1;

    if (_lockaspect) {
#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(_("mpWindow::Fit()(lock) _scaleX=%f,_scaleY=%f"), _scaleX, _scaleY);
#endif
        // Keep the lowest "scale" to fit the whole range required by that axis (to actually "fit"!):
        double s = _scaleX < _scaleY ? _scaleX : _scaleY;
        _scaleX = s;
        _scaleY = s;
    }

    // Adjusts corner coordinates: This should be simply:
    //   _posX = _minX;
    //   _posY = _maxY;
    // But account for centering if we have lock aspect:
    _posX = (xMin + xMax) / 2 - ((_scrX - _marginLeft - _marginRight) / 2 + _marginLeft) /
                                    _scaleX;  // _posX = (xMin+xMax)/2 - (_scrX/2)/_scaleX;
    //    _posY = (yMin+yMax)/2 + ((_scrY - _marginTop - _marginBottom)/2 - _marginTop)/_scaleY;  // _posY =
    //    (yMin+yMax)/2 + (_scrY/2)/_scaleY;
    _posY = (yMin + yMax) / 2 + ((_scrY - _marginTop - _marginBottom) / 2 + _marginTop) /
                                    _scaleY;  // _posY = (yMin+yMax)/2 + (_scrY/2)/_scaleY;

#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::Fit() _desiredXmin=%f _desiredXmax=%f  _desiredYmin=%f _desiredYmax=%f"), xMin, xMax,
                 yMin, yMax);
    wxLogMessage(_("mpWindow::Fit() _scaleX = %f , _scrX = %d,_scrY=%d, Ax=%f, Ay=%f, _posX=%f, _posY=%f"), _scaleX,
                 _scrX, _scrY, Ax, Ay, _posX, _posY);
#endif

    // It is VERY IMPORTANT to DO NOT call Refresh if we are drawing to the printer!!
    // Otherwise, the DC dimensions will be those of the window instead of the printer device
    if (printSizeX == NULL || printSizeY == NULL) UpdateAll();
}

// Patch ngpaton
void mpWindow::DoZoomInXCalc(const int staticXpixel) {
    // Preserve the position of the clicked point:
    double staticX = p2x(staticXpixel);
    // Zoom in:
    _scaleX = _scaleX * zoomIncrementalFactor;
    // Adjust the new _posx
    _posX = staticX - (staticXpixel / _scaleX);
    // Adjust desired
    _desiredXmin = _posX;
    _desiredXmax = _posX + (_scrX - (_marginLeft + _marginRight)) / _scaleX;
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::DoZoomInXCalc() prior X coord: (%f), new X coord: (%f) SHOULD BE EQUAL!!"), staticX,
                 p2x(staticXpixel));
#endif
}

void mpWindow::DoZoomInYCalc(const int staticYpixel) {
    // Preserve the position of the clicked point:
    double staticY = p2y(staticYpixel);
    // Zoom in:
    _scaleY = _scaleY * zoomIncrementalFactor;
    // Adjust the new _posy:
    _posY = staticY + (staticYpixel / _scaleY);
    // Adjust desired
    _desiredYmax = _posY;
    _desiredYmin = _posY - (_scrY - (_marginTop + _marginBottom)) / _scaleY;
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::DoZoomInYCalc() prior Y coord: (%f), new Y coord: (%f) SHOULD BE EQUAL!!"), staticY,
                 p2y(staticYpixel));
#endif
}

void mpWindow::DoZoomOutXCalc(const int staticXpixel) {
    // Preserve the position of the clicked point:
    double staticX = p2x(staticXpixel);
    // Zoom out:
    _scaleX = _scaleX / zoomIncrementalFactor;
    // Adjust the new _posx/y:
    _posX = staticX - (staticXpixel / _scaleX);
    // Adjust desired
    _desiredXmin = _posX;
    _desiredXmax = _posX + (_scrX - (_marginLeft + _marginRight)) / _scaleX;
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::DoZoomOutXCalc() prior X coord: (%f), new X coord: (%f) SHOULD BE EQUAL!!"), staticX,
                 p2x(staticXpixel));
#endif
}

void mpWindow::DoZoomOutYCalc(const int staticYpixel) {
    // Preserve the position of the clicked point:
    double staticY = p2y(staticYpixel);
    // Zoom out:
    _scaleY = _scaleY / zoomIncrementalFactor;
    // Adjust the new _posx/y:
    _posY = staticY + (staticYpixel / _scaleY);
    // Adjust desired
    _desiredYmax = _posY;
    _desiredYmin = _posY - (_scrY - (_marginTop + _marginBottom)) / _scaleY;
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::DoZoomOutYCalc() prior Y coord: (%f), new Y coord: (%f) SHOULD BE EQUAL!!"), staticY,
                 p2y(staticYpixel));
#endif
}

void mpWindow::ZoomIn(const wxPoint& centerPoint) {
    wxPoint c(centerPoint);
    if (c == wxDefaultPosition) {
        GetClientSize(&_scrX, &_scrY);
        c.x = (_scrX - _marginLeft - _marginRight) / 2 + _marginLeft;  // c.x = _scrX/2;
        c.y = (_scrY - _marginTop - _marginBottom) / 2 - _marginTop;   // c.y = _scrY/2;
    }

    // Preserve the position of the clicked point:
    double prior_layer_x = p2x(c.x);
    double prior_layer_y = p2y(c.y);

    // Zoom in:
    _scaleX = _scaleX * zoomIncrementalFactor;
    _scaleY = _scaleY * zoomIncrementalFactor;

    // Adjust the new _posx/y:
    _posX = prior_layer_x - c.x / _scaleX;
    _posY = prior_layer_y + c.y / _scaleY;

    _desiredXmin = _posX;
    _desiredXmax = _posX + (_scrX - _marginLeft - _marginRight) / _scaleX;  // _desiredXmax = _posX + _scrX / _scaleX;
    _desiredYmax = _posY;
    _desiredYmin = _posY - (_scrY - _marginTop - _marginBottom) / _scaleY;  // _desiredYmin = _posY - _scrY / _scaleY;

#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::ZoomIn() prior coords: (%f,%f), new coords: (%f,%f) SHOULD BE EQUAL!!"), prior_layer_x,
                 prior_layer_y, p2x(c.x), p2y(c.y));
#endif

    UpdateAll();
}

void mpWindow::ZoomOut(const wxPoint& centerPoint) {
    wxPoint c(centerPoint);
    if (c == wxDefaultPosition) {
        GetClientSize(&_scrX, &_scrY);
        c.x = (_scrX - _marginLeft - _marginRight) / 2 + _marginLeft;  // c.x = _scrX/2;
        c.y = (_scrY - _marginTop - _marginBottom) / 2 - _marginTop;   // c.y = _scrY/2;
    }

    // Preserve the position of the clicked point:
    double prior_layer_x = p2x(c.x);
    double prior_layer_y = p2y(c.y);

    // Zoom out:
    _scaleX = _scaleX / zoomIncrementalFactor;
    _scaleY = _scaleY / zoomIncrementalFactor;

    // Adjust the new _posx/y:
    _posX = prior_layer_x - c.x / _scaleX;
    _posY = prior_layer_y + c.y / _scaleY;

    _desiredXmin = _posX;
    _desiredXmax = _posX + (_scrX - _marginLeft - _marginRight) / _scaleX;  // _desiredXmax = _posX + _scrX / _scaleX;
    _desiredYmax = _posY;
    _desiredYmin = _posY - (_scrY - _marginTop - _marginBottom) / _scaleY;  // _desiredYmin = _posY - _scrY / _scaleY;

#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::ZoomOut() prior coords: (%f,%f), new coords: (%f,%f) SHOULD BE EQUAL!!"), prior_layer_x,
                 prior_layer_y, p2x(c.x), p2y(c.y));
#endif
    UpdateAll();
}

void mpWindow::ZoomInX() {
    _scaleX = _scaleX * zoomIncrementalFactor;
    UpdateAll();
}

void mpWindow::ZoomOutX() {
    _scaleX = _scaleX / zoomIncrementalFactor;
    UpdateAll();
}

void mpWindow::ZoomInY() {
    _scaleY = _scaleY * zoomIncrementalFactor;
    UpdateAll();
}

void mpWindow::ZoomOutY() {
    _scaleY = _scaleY / zoomIncrementalFactor;
    UpdateAll();
}

void mpWindow::ZoomRect(wxPoint p0, wxPoint p1) {
    // Compute the 2 corners in graph coordinates:
    double p0x = p2x(p0.x);
    double p0y = p2y(p0.y);
    double p1x = p2x(p1.x);
    double p1y = p2y(p1.y);

    // Order them:
    double zoom_x_min = p0x < p1x ? p0x : p1x;
    double zoom_x_max = p0x > p1x ? p0x : p1x;
    double zoom_y_min = p0y < p1y ? p0y : p1y;
    double zoom_y_max = p0y > p1y ? p0y : p1y;

#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("Zoom: (%f,%f)-(%f,%f)"), zoom_x_min, zoom_y_min, zoom_x_max, zoom_y_max);
#endif

    Fit(zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max);
}

void mpWindow::LockAspect(bool enable) {
    _lockaspect = enable;
    _popmenu.Check(mpID_LOCKASPECT, enable);

    // Try to fit again with the new config:
    Fit(_desiredXmin, _desiredXmax, _desiredYmin, _desiredYmax);
}

void mpWindow::OnShowPopupMenu(wxMouseEvent& event) {
    // Only display menu if the user has not "dragged" the figure
    if (_enableMouseNavigation) {
        SetCursor(*wxSTANDARD_CURSOR);
    }

    if (!_mouseMovedAfterRightClick)  // JLB
    {
        _clickedX = event.GetX();
        _clickedY = event.GetY();
        PopupMenu(&_popmenu, event.GetX(), event.GetY());
    }
}

void mpWindow::OnLockAspect(wxCommandEvent& WXUNUSED(event)) {
    LockAspect(!_lockaspect);
}

void mpWindow::OnMouseHelp(wxCommandEvent& WXUNUSED(event)) {
    wxMessageBox(_("Supported Mouse commands:\n \
        - Left button down + Mark area: Rectangular zoom\n \
        - Right button down + Move: Pan (Move)\n \
        - Wheel: Vertical scroll\n \
        - Wheel + SHIFT: Horizontal scroll\n \
        - Wheel + CTRL: Zoom in/out"),
                 _("wxMathPlot help"), wxOK, this);
}

void mpWindow::OnFit(wxCommandEvent& WXUNUSED(event)) {
    Fit();
}

void mpWindow::OnCenter(wxCommandEvent& WXUNUSED(event)) {
    GetClientSize(&_scrX, &_scrY);
    int centerX = (_scrX - _marginLeft - _marginRight) / 2;  // + _marginLeft; // c.x = _scrX/2;
    int centerY = (_scrY - _marginTop - _marginBottom) / 2;  // - _marginTop; // c.y = _scrY/2;
    SetPos(p2x(_clickedX - centerX), p2y(_clickedY - centerY));
    // SetPos( p2x(_clickedX-_scrX/2), p2y(_clickedY-_scrY/2) );  //SetPos( (double)(_clickedX-_scrX/2) / _scaleX
    // + _posX, (double)(_scrY/2-_clickedY) / _scaleY + _posY);
}

void mpWindow::OnZoomIn(wxCommandEvent& WXUNUSED(event)) {
    ZoomIn(wxPoint(_mouseRClick_X, _mouseRClick_Y));
}

void mpWindow::OnZoomOut(wxCommandEvent& WXUNUSED(event)) {
    ZoomOut();
}

void mpWindow::OnSize(wxSizeEvent& WXUNUSED(event)) {
    // Try to fit again with the new window size:
    Fit(_desiredXmin, _desiredXmax, _desiredYmin, _desiredYmax);
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(_("mpWindow::OnSize() _scrX = %d, _scrY = %d"), _scrX, _scrY);
#endif  // MATHPLOT_DO_LOGGING
}

bool mpWindow::AddLayer(mpLayer* layer, bool refreshDisplay) {
    if (layer != NULL) {
        _layers.push_back(layer);
        if (refreshDisplay) UpdateAll();
        return true;
    };
    return false;
}

bool mpWindow::DelLayer(mpLayer* layer, bool alsoDeleteObject, bool refreshDisplay) {
    wxLayerList::iterator layIt;
    for (layIt = _layers.begin(); layIt != _layers.end(); layIt++) {
        if (*layIt == layer) {
            // Also delete the object?
            if (alsoDeleteObject) delete *layIt;
            _layers.erase(layIt);  // this deleted the reference only
            if (refreshDisplay) UpdateAll();
            return true;
        }
    }
    return false;
}

void mpWindow::DelAllLayers(bool alsoDeleteObject, bool refreshDisplay) {
    while (_layers.size() > 0) {
        // Also delete the object?
        if (alsoDeleteObject) delete _layers[0];
        _layers.erase(_layers.begin());  // this deleted the reference only
    }
    if (refreshDisplay) UpdateAll();
}

// void mpWindow::DoPrepareDC(wxDC& dc)
// {
//     dc.SetDeviceOrigin(x2p(_minX), y2p(_maxY));
// }

void mpWindow::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    wxPaintDC dc(this);
    dc.GetSize(&_scrX, &_scrY);  // This is the size of the visible area only!
                                 //     DoPrepareDC(dc);

#ifdef MATHPLOT_DO_LOGGING
    {
        int px, py;
        GetViewStart(&px, &py);
        wxLogMessage(_("[mpWindow::OnPaint] vis.area:%ix%i px=%i py=%i"), _scrX, _scrY, px, py);
    }
#endif

    // Selects direct or buffered draw:
    wxDC* trgDc;

    // J.L.Blanco @ Aug 2007: Added double buffer support
    if (_enableDoubleBuffer) {
        if (_last_lx != _scrX || _last_ly != _scrY) {
            if (_buff_bmp) delete _buff_bmp;
            _buff_bmp = new wxBitmap(_scrX, _scrY);
            _buff_dc.SelectObject(*_buff_bmp);
            _last_lx = _scrX;
            _last_ly = _scrY;
        }
        trgDc = &_buff_dc;
    } else {
        trgDc = &dc;
    }

    // Draw background:
    // trgDc->SetDeviceOrigin(0,0);
    trgDc->SetPen(*wxTRANSPARENT_PEN);
    wxBrush brush(GetBackgroundColour());
    trgDc->SetBrush(brush);
    trgDc->SetTextForeground(_fgColour);
    trgDc->DrawRectangle(0, 0, _scrX, _scrY);

    // Draw all the layers:
    // trgDc->SetDeviceOrigin( _scrX>>1, _scrY>>1);  // Origin at the center
    wxLayerList::iterator li;
    for (li = _layers.begin(); li != _layers.end(); li++) {
        (*li)->Plot(*trgDc, *this);
    };

    // If doublebuffer, draw now to the window:
    if (_enableDoubleBuffer) {
        // trgDc->SetDeviceOrigin(0,0);
        // dc.SetDeviceOrigin(0,0);  // Origin at the center
        dc.Blit(0, 0, _scrX, _scrY, trgDc, 0, 0);
    }

    /*    if (_coordTooltip) {
            wxString toolTipContent;
            wxPoint mousePoint =  wxGetMousePosition();
            toolTipContent.Printf(_("X = %f\nY = %f"), p2x(mousePoint.x), p2y(mousePoint.y));
            SetToolTip(toolTipContent);
        }*/
    // If scrollbars are enabled, refresh them
    if (_enableScrollBars) {
        /*       _scrollX = (int) floor((_posX - _minX)*_scaleX);
               _scrollY = (int) floor((_maxY - _posY )*_scaleY);
               Scroll(_scrollX, _scrollY);*/
        // Scroll(x2p(_posX), y2p(_posY));
        //             SetVirtualSize((int) ((_maxX - _minX)*_scaleX), (int) ((_maxY - _minY)*_scaleY));
        //         int centerX = (_scrX - _marginLeft - _marginRight)/2; // + _marginLeft; // c.x = _scrX/2;
        //     int centerY = (_scrY - _marginTop - _marginBottom)/2; // - _marginTop; // c.y = _scrY/2;
        /*SetScrollbars(1, 1, (int) ((_maxX - _minX)*_scaleX), (int) ((_maxY - _minY)*_scaleY));*/  //, x2p(_posX
                                                                                                    //+
                                                                                                    // centerX/_scaleX),
                                                                                                    // y2p(_posY
                                                                                                    // -
                                                                                                    // centerY/_scaleY),
                                                                                                    // true);
    }
}

// void mpWindow::OnScroll2(wxScrollWinEvent &event)
// {
// #ifdef MATHPLOT_DO_LOGGING
//     wxLogMessage(_("[mpWindow::OnScroll2] Init: _posX=%f _posY=%f, sc_pos = %d"),_posX,_posY,
//     event.GetPosition());
// #endif
//     // If scrollbars are not enabled, Skip operation
//     if (!_enableScrollBars) {
//         event.Skip();
//         return;
//     }
// //     _scrollX = (int) floor((_posX - _minX)*_scaleX);
// //     _scrollY = (int) floor((_maxY - _posY /*- _minY*/)*_scaleY);
// //     Scroll(_scrollX, _scrollY);
//
// //     GetClientSize( &_scrX, &_scrY);
//     //Scroll(x2p(_desiredXmin), y2p(_desiredYmin));
//     int pixelStep = 1;
//     if (event.GetOrientation() == wxHORIZONTAL) {
//         //_desiredXmin -= (_scrollX - event.GetPosition())/_scaleX;
//         //_desiredXmax -= (_scrollX - event.GetPosition())/_scaleX;
//         _posX -= (_scrollX - event.GetPosition())/_scaleX;
//         _scrollX = event.GetPosition();
//     }
//     Fit(_desiredXmin, _desiredXmax, _desiredYmin, _desiredYmax);
// // /*    int pixelStep = 1;
// //     if (event.GetOrientation() == wxHORIZONTAL) {
// //         _posX         -= (px - event.GetPosition())/_scaleX;//(pixelStep/_scaleX);
// //     _desiredXmax     -= (px - event.GetPosition())/_scaleX;//(pixelStep/_scaleX);
// //     _desiredXmin     -= (px - event.GetPosition())/_scaleX;//(pixelStep/_scaleX);
// //         //SetPosX( (double)px / GetScaleX() + _minX + (double)(width>>1)/GetScaleX());
// // //         _posX = p2x(px); //_minX + (double)(px /*+ (_scrX)*/)/GetScaleX();
// //     } else {
// //         _posY         += (py - event.GetPosition())/_scaleY;//(pixelStep/_scaleY);
// //     _desiredYmax    += (py - event.GetPosition())/_scaleY;//(pixelStep/_scaleY);
// //     _desiredYmax    += (py - event.GetPosition())/_scaleY;//(pixelStep/_scaleY);
// //         //SetPosY( _maxY - (double)py / GetScaleY() - (double)(height>>1)/GetScaleY());
// //         //_posY = _maxY - (double)py / GetScaleY() - (double)(height>>1)/GetScaleY();
// // //         _posY = p2y(py);//_maxY - (double)(py /*+ (_scrY)*/)/GetScaleY();
// //     }*/
// #ifdef MATHPLOT_DO_LOGGING
//     int px, py;
//     GetViewStart( &px, &py);
//     wxLogMessage(_("[mpWindow::OnScroll2] End:  _posX = %f, _posY = %f, px = %f, py = %f"),_posX, _posY, px, py);
// #endif
//
//     UpdateAll();
// //     event.Skip();
// }

void mpWindow::SetMPScrollbars(bool status) {
    // Temporary behaviour: always disable scrollbars
    _enableScrollBars = status;  // false;
    if (status == false) {
        SetScrollbar(wxHORIZONTAL, 0, 0, 0);
        SetScrollbar(wxVERTICAL, 0, 0, 0);
    }
    // else the scroll bars will be updated in UpdateAll();
    UpdateAll();

    //     EnableScrolling(false, false);
    //     _enableScrollBars = status;
    //     EnableScrolling(status, status);
    /*    _scrollX = (int) floor((_posX - _minX)*_scaleX);
        _scrollY = (int) floor((_posY - _minY)*_scaleY);*/
    //     int scrollWidth = (int) floor((_maxX - _minX)*_scaleX) - _scrX;
    //     int scrollHeight = (int) floor((_minY - _maxY)*_scaleY) - _scrY;

    // /*    _scrollX = (int) floor((_posX - _minX)*_scaleX);
    //     _scrollY = (int) floor((_maxY - _posY /*- _minY*/)*_scaleY);
    //     int scrollWidth = (int) floor(((_maxX - _minX) - (_desiredXmax - _desiredXmin))*_scaleX);
    //     int scrollHeight = (int) floor(((_maxY - _minY) - (_desiredYmax - _desiredYmin))*_scaleY);
    // #ifdef MATHPLOT_DO_LOGGING
    //     wxLogMessage(_("mpWindow::SetMPScrollbars() scrollWidth = %d, scrollHeight = %d"), scrollWidth,
    //     scrollHeight);
    // #endif
    //     if(status) {
    //         SetScrollbars(1,
    //                       1,
    //                       scrollWidth,
    //                       scrollHeight,
    //                       _scrollX,
    //                       _scrollY);
    // //         SetVirtualSize((int) (_maxX - _minX), (int) (_maxY - _minY));
    //     }
    //     Refresh(false);*/
};

bool mpWindow::UpdateBBox() {
    bool first = TRUE;

    for (wxLayerList::iterator li = _layers.begin(); li != _layers.end(); li++) {
        mpLayer* f = *li;

        if (f->HasBBox()) {
            if (first) {
                first = FALSE;
                _minX = f->GetMinX();
                _maxX = f->GetMaxX();
                _minY = f->GetMinY();
                _maxY = f->GetMaxY();
            } else {
                if (f->GetMinX() < _minX) _minX = f->GetMinX();
                if (f->GetMaxX() > _maxX) _maxX = f->GetMaxX();
                if (f->GetMinY() < _minY) _minY = f->GetMinY();
                if (f->GetMaxY() > _maxY) _maxY = f->GetMaxY();
            }
        }
        // node = node->GetNext();
    }
#ifdef MATHPLOT_DO_LOGGING
    wxLogDebug(wxT("[mpWindow::UpdateBBox] Bounding box: Xmin = %f, Xmax = %f, Ymin = %f, YMax = %f"), _minX, _maxX,
               _minY, _maxY);
#endif  // MATHPLOT_DO_LOGGING
    return first == FALSE;
}

// void mpWindow::UpdateAll()
// {
// GetClientSize( &_scrX,&_scrY);
/*    if (_enableScrollBars) {
        // The "virtual size" of the scrolled window:
        const int sx = (int)((_maxX - _minX) * GetScaleX());
        const int sy = (int)((_maxY - _minY) * GetScaleY());
    SetVirtualSize(sx, sy);
    SetScrollRate(1, 1);*/
//         const int px = (int)((GetPosX() - _minX) * GetScaleX());// - _scrX); //(cx>>1));

// J.L.Blanco, Aug 2007: Formula fixed:
//         const int py = (int)((_maxY - GetPosY()) * GetScaleY());// - _scrY); //(cy>>1));
//         int px, py;
//         GetViewStart(&px0, &py0);
//     px = (int)((_posX - _minX)*_scaleX);
//     py = (int)((_maxY - _posY)*_scaleY);

//         SetScrollbars( 1, 1, sx - _scrX, sy - _scrY, px, py, TRUE);
//     }

// Working code
//     UpdateBBox();
//    Refresh( FALSE );
// end working code

// Old version
/*   bool box = UpdateBBox();
    if (box)
{
        int cx, cy;
        GetClientSize( &cx, &cy);

        // The "virtual size" of the scrolled window:
        const int sx = (int)((_maxX - _minX) * GetScaleX());
        const int sy = (int)((_maxY - _minY) * GetScaleY());

        const int px = (int)((GetPosX() - _minX) * GetScaleX() - (cx>>1));

        // J.L.Blanco, Aug 2007: Formula fixed:
        const int py = (int)((_maxY - GetPosY()) * GetScaleY() - (cy>>1));

        SetScrollbars( 1, 1, sx, sy, px, py, TRUE);

#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(_("[mpWindow::UpdateAll] Size:%ix%i ScrollBars:%i,%i"),sx,sy,px,py);
#endif
}

    FitInside();
    Refresh( FALSE );
*/
// }

void mpWindow::UpdateAll() {
    if (UpdateBBox()) {
        if (_enableScrollBars) {
            int cx, cy;
            GetClientSize(&cx, &cy);
            // Do x scroll bar
            {
                // Convert margin sizes from pixels to coordinates
                double leftMargin = _marginLeft / _scaleX;
                // Calculate the range in coords that we want to scroll over
                double maxX = (_desiredXmax > _maxX) ? _desiredXmax : _maxX;
                double minX = (_desiredXmin < _minX) ? _desiredXmin : _minX;
                if ((_posX + leftMargin) < minX) minX = _posX + leftMargin;
                // Calculate scroll bar size and thumb position
                int sizeX = (int)((maxX - minX) * _scaleX);
                int thumbX = (int)(((_posX + leftMargin) - minX) * _scaleX);
                SetScrollbar(wxHORIZONTAL, thumbX, cx - (_marginRight + _marginLeft), sizeX);
            }
            // Do y scroll bar
            {
                // Convert margin sizes from pixels to coordinates
                double topMargin = _marginTop / _scaleY;
                // Calculate the range in coords that we want to scroll over
                double maxY = (_desiredYmax > _maxY) ? _desiredYmax : _maxY;
                if ((_posY - topMargin) > maxY) maxY = _posY - topMargin;
                double minY = (_desiredYmin < _minY) ? _desiredYmin : _minY;
                // Calculate scroll bar size and thumb position
                int sizeY = (int)((maxY - minY) * _scaleY);
                int thumbY = (int)((maxY - (_posY - topMargin)) * _scaleY);
                SetScrollbar(wxVERTICAL, thumbY, cy - (_marginTop + _marginBottom), sizeY);
            }
        }
    }

    Refresh(FALSE);
}

void mpWindow::DoScrollCalc(const int position, const int orientation) {
    if (orientation == wxVERTICAL) {
        // Y axis
        // Get top margin in coord units
        double topMargin = _marginTop / _scaleY;
        // Calculate maximum Y coord to be shown in the graph
        double maxY = _desiredYmax > _maxY ? _desiredYmax : _maxY;
        // Set new position
        SetPosY((maxY - (position / _scaleY)) + topMargin);
    } else {
        // X Axis
        // Get left margin in coord units
        double leftMargin = _marginLeft / _scaleX;
        // Calculate minimum X coord to be shown in the graph
        double minX = (_desiredXmin < _minX) ? _desiredXmin : _minX;
        // Set new position
        SetPosX((minX + (position / _scaleX)) - leftMargin);
    }
}

void mpWindow::OnScrollThumbTrack(wxScrollWinEvent& event) {
    DoScrollCalc(event.GetPosition(), event.GetOrientation());
}

void mpWindow::OnScrollPageUp(wxScrollWinEvent& event) {
    int scrollOrientation = event.GetOrientation();
    // Get position before page up
    int position = GetScrollPos(scrollOrientation);
    // Get thumb size
    int thumbSize = GetScrollThumb(scrollOrientation);
    // Need to adjust position by a page
    position -= thumbSize;
    if (position < 0) position = 0;

    DoScrollCalc(position, scrollOrientation);
}

void mpWindow::OnScrollPageDown(wxScrollWinEvent& event) {
    int scrollOrientation = event.GetOrientation();
    // Get position before page up
    int position = GetScrollPos(scrollOrientation);
    // Get thumb size
    int thumbSize = GetScrollThumb(scrollOrientation);
    // Get scroll range
    int scrollRange = GetScrollRange(scrollOrientation);
    // Need to adjust position by a page
    position += thumbSize;
    if (position > (scrollRange - thumbSize)) position = scrollRange - thumbSize;

    DoScrollCalc(position, scrollOrientation);
}

void mpWindow::OnScrollLineUp(wxScrollWinEvent& event) {
    int scrollOrientation = event.GetOrientation();
    // Get position before page up
    int position = GetScrollPos(scrollOrientation);
    // Need to adjust position by a line
    position -= mpSCROLL_NUM_PIXELS_PER_LINE;
    if (position < 0) position = 0;

    DoScrollCalc(position, scrollOrientation);
}

void mpWindow::OnScrollLineDown(wxScrollWinEvent& event) {
    int scrollOrientation = event.GetOrientation();
    // Get position before page up
    int position = GetScrollPos(scrollOrientation);
    // Get thumb size
    int thumbSize = GetScrollThumb(scrollOrientation);
    // Get scroll range
    int scrollRange = GetScrollRange(scrollOrientation);
    // Need to adjust position by a page
    position += mpSCROLL_NUM_PIXELS_PER_LINE;
    if (position > (scrollRange - thumbSize)) position = scrollRange - thumbSize;

    DoScrollCalc(position, scrollOrientation);
}

void mpWindow::OnScrollTop(wxScrollWinEvent& event) {
    DoScrollCalc(0, event.GetOrientation());
}

void mpWindow::OnScrollBottom(wxScrollWinEvent& event) {
    int scrollOrientation = event.GetOrientation();
    // Get thumb size
    int thumbSize = GetScrollThumb(scrollOrientation);
    // Get scroll range
    int scrollRange = GetScrollRange(scrollOrientation);

    DoScrollCalc(scrollRange - thumbSize, scrollOrientation);
}
// End patch ngpaton

void mpWindow::SetScaleX(double scaleX) {
    if (scaleX != 0) _scaleX = scaleX;
    UpdateAll();
}

// New methods implemented by Davide Rondini

unsigned int mpWindow::CountLayers() {
    // wxNode *node = _layers.GetFirst();
    unsigned int layerNo = 0;
    for (wxLayerList::iterator li = _layers.begin(); li != _layers.end(); li++)  // while(node)
    {
        if ((*li)->HasBBox()) layerNo++;
        // node = node->GetNext();
    };
    return layerNo;
}

mpLayer* mpWindow::GetLayer(int position) {
    if ((position >= (int)_layers.size()) || position < 0) return NULL;
    return _layers[position];
}

mpLayer* mpWindow::GetLayerByName(const wxString& name) {
    for (wxLayerList::iterator it = _layers.begin(); it != _layers.end(); it++)
        if (!(*it)->GetName().Cmp(name)) return *it;
    return NULL;  // Not found
}

void mpWindow::GetBoundingBox(double* bbox) {
    bbox[0] = _minX;
    bbox[1] = _maxX;
    bbox[2] = _minY;
    bbox[3] = _maxY;
}

bool mpWindow::SaveScreenshot(const wxString& filename, int type, wxSize imageSize, bool fit) {
    int sizeX, sizeY;
    int bk_scrX = 0;
    int bk_scrY = 0;
    if (imageSize == wxDefaultSize) {
        sizeX = _scrX;
        sizeY = _scrY;
    } else {
        sizeX = imageSize.x;
        sizeY = imageSize.y;
        bk_scrX = _scrX;
        bk_scrY = _scrY;
        SetScr(sizeX, sizeY);
    }

    wxBitmap screenBuffer(sizeX, sizeY);
    wxMemoryDC screenDC;
    screenDC.SelectObject(screenBuffer);
    screenDC.SetPen(*wxTRANSPARENT_PEN);
    wxBrush brush(GetBackgroundColour());
    screenDC.SetBrush(brush);
    screenDC.DrawRectangle(0, 0, sizeX, sizeY);

    if (fit) {
        Fit(_minX, _maxX, _minY, _maxY, &sizeX, &sizeY);
    } else {
        Fit(_desiredXmin, _desiredXmax, _desiredYmin, _desiredYmax, &sizeX, &sizeY);
    }
    // Draw all the layers:
    wxLayerList::iterator li;
    for (li = _layers.begin(); li != _layers.end(); li++) (*li)->Plot(screenDC, *this);

    if (imageSize != wxDefaultSize) {
        // Restore dimensions
        SetScr(bk_scrX, bk_scrY);
        Fit(_desiredXmin, _desiredXmax, _desiredYmin, _desiredYmax, &bk_scrX, &bk_scrY);
        UpdateAll();
    }
    // Once drawing is complete, actually save screen shot
    wxImage screenImage = screenBuffer.ConvertToImage();
    return screenImage.SaveFile(filename, (wxBitmapType)type);
}

void mpWindow::SetMargins(int top, int right, int bottom, int left) {
    _marginTop = top;
    _marginRight = right;
    _marginBottom = bottom;
    _marginLeft = left;
}

mpInfoLayer* mpWindow::IsInsideInfoLayer(wxPoint& point) {
    wxLayerList::iterator li;
    for (li = _layers.begin(); li != _layers.end(); li++) {
#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(_("mpWindow::IsInsideInfoLayer() examinining layer = %p"), (*li));
#endif  // MATHPLOT_DO_LOGGING
        if ((*li)->IsInfo()) {
            mpInfoLayer* tmpLyr = (mpInfoLayer*)(*li);
#ifdef MATHPLOT_DO_LOGGING
            wxLogMessage(_("mpWindow::IsInsideInfoLayer() layer = %p"), (*li));
#endif  // MATHPLOT_DO_LOGGING
            if (tmpLyr->Inside(point)) {
                return tmpLyr;
            }
        }
    }
    return NULL;
}

void mpWindow::SetLayerVisible(const wxString& name, bool viewable) {
    mpLayer* lx = GetLayerByName(name);
    if (lx) {
        lx->SetVisible(viewable);
        UpdateAll();
    }
}

bool mpWindow::IsLayerVisible(const wxString& name) {
    mpLayer* lx = GetLayerByName(name);
    return (lx) ? lx->IsVisible() : false;
}

void mpWindow::SetLayerVisible(const unsigned int position, bool viewable) {
    mpLayer* lx = GetLayer(position);
    if (lx) {
        lx->SetVisible(viewable);
        UpdateAll();
    }
}

bool mpWindow::IsLayerVisible(const unsigned int position) {
    mpLayer* lx = GetLayer(position);
    return (lx) ? lx->IsVisible() : false;
}

void mpWindow::SetColourTheme(const wxColour& bgColour, const wxColour& drawColour, const wxColour& axesColour) {
    SetBackgroundColour(bgColour);
    SetForegroundColour(drawColour);
    _bgColour = bgColour;
    _fgColour = drawColour;
    _axColour = axesColour;
    // cycle between layers to set colours and properties to them
    wxLayerList::iterator li;
    for (li = _layers.begin(); li != _layers.end(); li++) {
        if ((*li)->GetLayerType() == mpLAYER_AXIS) {
            wxPen axisPen = (*li)->GetPen();  // Get the old pen to modify only colour, not style or width
            axisPen.SetColour(axesColour);
            (*li)->SetPen(axisPen);
        }
        if ((*li)->GetLayerType() == mpLAYER_INFO) {
            wxPen infoPen = (*li)->GetPen();  // Get the old pen to modify only colour, not style or width
            infoPen.SetColour(drawColour);
            (*li)->SetPen(infoPen);
        }
    }
}

// void mpWindow::EnableCoordTooltip(bool value)
// {
//      _coordTooltip = value;
// //      if (value) GetToolTip()->SetDelay(100);
// }

/*
double mpWindow::p2x(wxCoord pixelCoordX, bool drawOutside )
{
    if (drawOutside) {
        return _posX + pixelCoordX/_scaleX;
    }
    // Draw inside margins
    double marginScaleX = ((double)(_scrX - _marginLeft - _marginRight))/_scrX;
    return _marginLeft + (_posX + pixelCoordX/_scaleX)/marginScaleX;
}

double mpWindow::p2y(wxCoord pixelCoordY, bool drawOutside )
{
    if (drawOutside) {
        return _posY - pixelCoordY/_scaleY;
    }
    // Draw inside margins
    double marginScaleY = ((double)(_scrY - _marginTop - _marginBottom))/_scrY;
    return _marginTop + (_posY - pixelCoordY/_scaleY)/marginScaleY;
}

wxCoord mpWindow::x2p(double x, bool drawOutside)
{
    if (drawOutside) {
        return (wxCoord) ((x-_posX) * _scaleX);
    }
    // Draw inside margins
    double marginScaleX = ((double)(_scrX - _marginLeft - _marginRight))/_scrX;
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(wxT("x2p ScrX = %d, marginRight = %d, marginLeft = %d, marginScaleX = %f"), _scrX, _marginRight,
_marginLeft,  marginScaleX); #endif // MATHPLOT_DO_LOGGING return (wxCoord) (int)(((x-_posX) * _scaleX)*marginScaleX)
- _marginLeft;
}

wxCoord mpWindow::y2p(double y, bool drawOutside)
{
    if (drawOutside) {
        return (wxCoord) ( (_posY-y) * _scaleY);
    }
    // Draw inside margins
    double marginScaleY = ((double)(_scrY - _marginTop - _marginBottom))/_scrY;
#ifdef MATHPLOT_DO_LOGGING
    wxLogMessage(wxT("y2p ScrY = %d, marginTop = %d, marginBottom = %d, marginScaleY = %f"), _scrY, _marginTop,
_marginBottom, marginScaleY); #endif // MATHPLOT_DO_LOGGING return (wxCoord) ((int)((_posY-y) *
_scaleY)*marginScaleY) - _marginTop;
}
*/

//-----------------------------------------------------------------------------
// mpFXYVector implementation - by Jose Luis Blanco (AGO-2007)
//-----------------------------------------------------------------------------

IMPLEMENT_DYNAMIC_CLASS(mpFXYVector, mpFXY)

// Constructor
mpFXYVector::mpFXYVector(wxString name, int flags)
    : mpFXY(name, flags) {
    _index = 0;
    _minX = -1;
    _maxX = 1;
    _minY = -1;
    _maxY = 1;
    _type = mpLAYER_PLOT;
}

void mpFXYVector::Rewind() {
    _index = 0;
}

bool mpFXYVector::GetNextXY(double& x, double& y) {
    if (_index >= _xs.size())
        return FALSE;
    else {
        x = _xs[_index];
        y = _ys[_index++];
        return _index <= _xs.size();
    }
}

void mpFXYVector::Clear() {
    _xs.clear();
    _ys.clear();
}

void mpFXYVector::SetData(const std::vector<double>& xs, const std::vector<double>& ys) {
    // Check if the data vectora are of the same size
    if (xs.size() != ys.size()) {
        wxLogError(_("wxMathPlot error: X and Y vector are not of the same length!"));
        return;
    }
    // Copy the data:
    _xs = xs;
    _ys = ys;

    // Update internal variables for the bounding box.
    if (xs.size() > 0) {
        _minX = xs[0];
        _maxX = xs[0];
        _minY = ys[0];
        _maxY = ys[0];

        std::vector<double>::const_iterator it;

        for (it = xs.begin(); it != xs.end(); it++) {
            if (*it < _minX) _minX = *it;
            if (*it > _maxX) _maxX = *it;
        }
        for (it = ys.begin(); it != ys.end(); it++) {
            if (*it < _minY) _minY = *it;
            if (*it > _maxY) _maxY = *it;
        }
        _minX -= 0.5f;
        _minY -= 0.5f;
        _maxX += 0.5f;
        _maxY += 0.5f;
    } else {
        _minX = -1;
        _maxX = 1;
        _minY = -1;
        _maxY = 1;
    }
}

//-----------------------------------------------------------------------------
// mpText - provided by Val Greene
//-----------------------------------------------------------------------------

IMPLEMENT_DYNAMIC_CLASS(mpText, mpLayer)

/** @param name text to be displayed
@param offsetx x position in percentage (0-100)
@param offsetx y position in percentage (0-100)
*/
mpText::mpText(wxString name, int offsetx, int offsety) {
    SetName(name);

    if (offsetx >= 0 && offsetx <= 100)
        _offsetx = offsetx;
    else
        _offsetx = 5;

    if (offsety >= 0 && offsety <= 100)
        _offsety = offsety;
    else
        _offsetx = 50;
    _type = mpLAYER_INFO;
}

/** mpText Layer plot handler.
This implementation will plot the text adjusted to the visible area.
*/

void mpText::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);
        dc.SetFont(_font);

        wxCoord tw = 0, th = 0;
        dc.GetTextExtent(GetName(), &tw, &th);

        //     int left = -dc.LogicalToDeviceX(0);
        //     int width = dc.LogicalToDeviceX(0) - left;
        //     int bottom = dc.LogicalToDeviceY(0);
        //     int height = bottom - -dc.LogicalToDeviceY(0);

        /*    dc.DrawText( GetName(),
            (int)((((float)width/100.0) * _offsety) + left - (tw/2)),
            (int)((((float)height/100.0) * _offsetx) - bottom) );*/
        int px = _offsetx * (w.GetScrX() - w.GetMarginLeft() - w.GetMarginRight()) / 100;
        int py = _offsety * (w.GetScrY() - w.GetMarginTop() - w.GetMarginBottom()) / 100;
        dc.DrawText(GetName(), px, py);
    }
}

//-----------------------------------------------------------------------------
// mpPrintout - provided by Davide Rondini
//-----------------------------------------------------------------------------

mpPrintout::mpPrintout(mpWindow* drawWindow, const wxChar* title)
    : wxPrintout(title) {
    drawn = false;
    plotWindow = drawWindow;
}

bool mpPrintout::OnPrintPage(int page) {
    wxDC* trgDc = GetDC();
    if ((trgDc) && (page == 1)) {
        wxCoord _prnX, _prnY;
        int marginX = 50;
        int marginY = 50;
        trgDc->GetSize(&_prnX, &_prnY);

        _prnX -= (2 * marginX);
        _prnY -= (2 * marginY);
        trgDc->SetDeviceOrigin(marginX, marginY);

#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(wxT("Print Size: %d x %d\n"), _prnX, _prnY);
        wxLogMessage(wxT("Screen Size: %d x %d\n"), plotWindow->GetScrX(), plotWindow->GetScrY());
#endif

        // Set the scale according to the page:
        plotWindow->Fit(plotWindow->GetDesiredXmin(), plotWindow->GetDesiredXmax(), plotWindow->GetDesiredYmin(),
                        plotWindow->GetDesiredYmax(), &_prnX, &_prnY);

        // Get the colours of the plotWindow to restore them ath the end
        wxColour oldBgColour = plotWindow->GetBackgroundColour();
        wxColour oldFgColour = plotWindow->GetForegroundColour();
        wxColour oldAxColour = plotWindow->GetAxesColour();

        // Draw background, ensuring to use white background for printing.
        trgDc->SetPen(*wxTRANSPARENT_PEN);
        // wxBrush brush( plotWindow->GetBackgroundColour() );
        wxBrush brush = *wxWHITE_BRUSH;
        trgDc->SetBrush(brush);
        trgDc->DrawRectangle(0, 0, _prnX, _prnY);

        // Draw all the layers:
        // trgDc->SetDeviceOrigin( _prnX>>1, _prnY>>1);  // Origin at the center
        mpLayer* layer;
        for (unsigned int li = 0; li < plotWindow->CountAllLayers(); li++) {
            layer = plotWindow->GetLayer(li);
            layer->Plot(*trgDc, *plotWindow);
        };
        // Restore device origin
        // trgDc->SetDeviceOrigin(0, 0);
        // Restore colours
        plotWindow->SetColourTheme(oldBgColour, oldFgColour, oldAxColour);
        // Restore drawing
        plotWindow->Fit(plotWindow->GetDesiredXmin(), plotWindow->GetDesiredXmax(), plotWindow->GetDesiredYmin(),
                        plotWindow->GetDesiredYmax(), NULL, NULL);
        plotWindow->UpdateAll();
    }
    return true;
}

bool mpPrintout::HasPage(int page) {
    return (page == 1);
}

//-----------------------------------------------------------------------------
// mpMovableObject - provided by Jose Luis Blanco
//-----------------------------------------------------------------------------
void mpMovableObject::TranslatePoint(double x, double y, double& out_x, double& out_y) {
    double ccos = cos(_reference_phi);  // Avoid computing cos/sin twice.
    double csin = sin(_reference_phi);

    out_x = _reference_x + ccos * x - csin * y;
    out_y = _reference_y + csin * x + ccos * y;
}

// This method updates the buffers _trans_shape_xs/ys, and the precomputed bounding box.
void mpMovableObject::ShapeUpdated() {
    // Just in case...
    if (_shape_xs.size() != _shape_ys.size()) {
        wxLogError(wxT("[mpMovableObject::ShapeUpdated] Error, _shape_xs and _shape_ys have different lengths!"));
    } else {
        double ccos = cos(_reference_phi);  // Avoid computing cos/sin twice.
        double csin = sin(_reference_phi);

        _trans_shape_xs.resize(_shape_xs.size());
        _trans_shape_ys.resize(_shape_xs.size());

        std::vector<double>::iterator itXi, itXo;
        std::vector<double>::iterator itYi, itYo;

        _bbox_min_x = 1e300;
        _bbox_max_x = -1e300;
        _bbox_min_y = 1e300;
        _bbox_max_y = -1e300;

        for (itXo = _trans_shape_xs.begin(), itYo = _trans_shape_ys.begin(), itXi = _shape_xs.begin(),
            itYi = _shape_ys.begin();
             itXo != _trans_shape_xs.end(); itXo++, itYo++, itXi++, itYi++) {
            *itXo = _reference_x + ccos * (*itXi) - csin * (*itYi);
            *itYo = _reference_y + csin * (*itXi) + ccos * (*itYi);

            // Keep BBox:
            if (*itXo < _bbox_min_x) _bbox_min_x = *itXo;
            if (*itXo > _bbox_max_x) _bbox_max_x = *itXo;
            if (*itYo < _bbox_min_y) _bbox_min_y = *itYo;
            if (*itYo > _bbox_max_y) _bbox_max_y = *itYo;
        }
    }
}

void mpMovableObject::Plot(wxDC& dc, mpWindow& w) {
    if (_visible) {
        dc.SetPen(_pen);

        std::vector<double>::iterator itX = _trans_shape_xs.begin();
        std::vector<double>::iterator itY = _trans_shape_ys.begin();

        if (!_continuous) {
            // for some reason DrawPoint does not use the current pen,
            // so we use DrawLine for fat pens
            if (_pen.GetWidth() <= 1) {
                while (itX != _trans_shape_xs.end()) {
                    dc.DrawPoint(w.x2p(*(itX++)), w.y2p(*(itY++)));
                }
            } else {
                while (itX != _trans_shape_xs.end()) {
                    wxCoord cx = w.x2p(*(itX++));
                    wxCoord cy = w.y2p(*(itY++));
                    dc.DrawLine(cx, cy, cx, cy);
                }
            }
        } else {
            wxCoord cx0 = 0, cy0 = 0;
            bool first = TRUE;
            while (itX != _trans_shape_xs.end()) {
                wxCoord cx = w.x2p(*(itX++));
                wxCoord cy = w.y2p(*(itY++));
                if (first) {
                    first = FALSE;
                    cx0 = cx;
                    cy0 = cy;
                }
                dc.DrawLine(cx0, cy0, cx, cy);
                cx0 = cx;
                cy0 = cy;
            }
        }

        if (!_name.IsEmpty() && _showName) {
            dc.SetFont(_font);

            wxCoord tx, ty;
            dc.GetTextExtent(_name, &tx, &ty);

            if (HasBBox()) {
                wxCoord sx = (wxCoord)((_bbox_max_x - w.GetPosX()) * w.GetScaleX());
                wxCoord sy = (wxCoord)((w.GetPosY() - _bbox_max_y) * w.GetScaleY());

                tx = sx - tx - 8;
                ty = sy - 8 - ty;
            } else {
                const int sx = w.GetScrX() >> 1;
                const int sy = w.GetScrY() >> 1;

                if ((_flags & mpALIGNMASK) == mpALIGN_NE) {
                    tx = sx - tx - 8;
                    ty = -sy + 8;
                } else if ((_flags & mpALIGNMASK) == mpALIGN_NW) {
                    tx = -sx + 8;
                    ty = -sy + 8;
                } else if ((_flags & mpALIGNMASK) == mpALIGN_SW) {
                    tx = -sx + 8;
                    ty = sy - 8 - ty;
                } else {
                    tx = sx - tx - 8;
                    ty = sy - 8 - ty;
                }
            }

            dc.DrawText(_name, tx, ty);
        }
    }
}

//-----------------------------------------------------------------------------
// mpCovarianceEllipse - provided by Jose Luis Blanco
//-----------------------------------------------------------------------------

// Called to update the _shape_xs, _shape_ys vectors, whenever a parameter changes.
void mpCovarianceEllipse::RecalculateShape() {
    _shape_xs.clear();
    _shape_ys.clear();

    // Preliminar checks:
    if (_quantiles < 0) {
        wxLogError(wxT("[mpCovarianceEllipse] Error: quantiles must be non-negative"));
        return;
    }
    if (_cov_00 < 0) {
        wxLogError(wxT("[mpCovarianceEllipse] Error: cov(0,0) must be non-negative"));
        return;
    }
    if (_cov_11 < 0) {
        wxLogError(wxT("[mpCovarianceEllipse] Error: cov(1,1) must be non-negative"));
        return;
    }

    _shape_xs.resize(_segments, 0);
    _shape_ys.resize(_segments, 0);

    // Compute the two eigenvalues of the covariance:
    // -------------------------------------------------
    double b = -_cov_00 - _cov_11;
    double c = _cov_00 * _cov_11 - _cov_01 * _cov_01;

    double D = b * b - 4 * c;

    if (D < 0) {
        wxLogError(wxT("[mpCovarianceEllipse] Error: cov is not positive definite"));
        return;
    }

    double eigenVal0 = 0.5 * (-b + sqrt(D));
    double eigenVal1 = 0.5 * (-b - sqrt(D));

    // Compute the two corresponding eigenvectors:
    // -------------------------------------------------
    double eigenVec0_x, eigenVec0_y;
    double eigenVec1_x, eigenVec1_y;

    if (fabs(eigenVal0 - _cov_00) > 1e-6) {
        double k1x = _cov_01 / (eigenVal0 - _cov_00);
        eigenVec0_y = 1;
        eigenVec0_x = eigenVec0_y * k1x;
    } else {
        double k1y = _cov_01 / (eigenVal0 - _cov_11);
        eigenVec0_x = 1;
        eigenVec0_y = eigenVec0_x * k1y;
    }

    if (fabs(eigenVal1 - _cov_00) > 1e-6) {
        double k2x = _cov_01 / (eigenVal1 - _cov_00);
        eigenVec1_y = 1;
        eigenVec1_x = eigenVec1_y * k2x;
    } else {
        double k2y = _cov_01 / (eigenVal1 - _cov_11);
        eigenVec1_x = 1;
        eigenVec1_y = eigenVec1_x * k2y;
    }

    // Normalize the eigenvectors:
    double len = sqrt(eigenVec0_x * eigenVec0_x + eigenVec0_y * eigenVec0_y);
    eigenVec0_x /= len;  // It *CANNOT* be zero
    eigenVec0_y /= len;

    len = sqrt(eigenVec1_x * eigenVec1_x + eigenVec1_y * eigenVec1_y);
    eigenVec1_x /= len;  // It *CANNOT* be zero
    eigenVec1_y /= len;

    // Take the sqrt of the eigenvalues (required for the ellipse scale):
    eigenVal0 = sqrt(eigenVal0);
    eigenVal1 = sqrt(eigenVal1);

    // Compute the 2x2 matrix M = diag(eigVal) * (~eigVec)  (each eigen vector is a row):
    double M_00 = eigenVec0_x * eigenVal0;
    double M_01 = eigenVec0_y * eigenVal0;

    double M_10 = eigenVec1_x * eigenVal1;
    double M_11 = eigenVec1_y * eigenVal1;

    // The points of the 2D ellipse:
    double ang;
    double Aang = 6.283185308 / (_segments - 1);
    int i;
    for (i = 0, ang = 0; i < _segments; i++, ang += Aang) {
        double ccos = cos(ang);
        double csin = sin(ang);

        _shape_xs[i] = _quantiles * (ccos * M_00 + csin * M_10);
        _shape_ys[i] = _quantiles * (ccos * M_01 + csin * M_11);
    }  // end for points on ellipse

    ShapeUpdated();
}

//-----------------------------------------------------------------------------
// mpPolygon - provided by Jose Luis Blanco
//-----------------------------------------------------------------------------
void mpPolygon::setPoints(const std::vector<double>& points_xs, const std::vector<double>& points_ys,
                          bool closedShape) {
    if (points_xs.size() != points_ys.size()) {
        wxLogError(wxT("[mpPolygon] Error: points_xs and points_ys must have the same number of elements"));
    } else {
        _shape_xs = points_xs;
        _shape_ys = points_ys;

        if (closedShape && points_xs.size()) {
            _shape_xs.push_back(points_xs[0]);
            _shape_ys.push_back(points_ys[0]);
        }

        ShapeUpdated();
    }
}

//-----------------------------------------------------------------------------
// mpBitmapLayer - provided by Jose Luis Blanco
//-----------------------------------------------------------------------------
void mpBitmapLayer::GetBitmapCopy(wxImage& outBmp) const {
    if (_validImg) outBmp = _bitmap;
}

void mpBitmapLayer::SetBitmap(const wxImage& inBmp, double x, double y, double lx, double ly) {
    if (!inBmp.Ok()) {
        wxLogError(wxT("[mpBitmapLayer] Assigned bitmap is not Ok()!"));
    } else {
        _bitmap = inBmp;  //.GetSubBitmap( wxRect(0, 0, inBmp.GetWidth(), inBmp.GetHeight()));
        _min_x = x;
        _min_y = y;
        _max_x = x + lx;
        _max_y = y + ly;
        _validImg = true;
    }
}

void mpBitmapLayer::Plot(wxDC& dc, mpWindow& w) {
    if (_visible && _validImg) {
        /*    1st: We compute (x0,y0)-(x1,y1), the pixel coordinates of the real outer limits
                 of the image rectangle within the (screen) mpWindow. Note that these coordinates
                 might fall well far away from the real view limits when the user zoom in.

            2nd: We compute (dx0,dy0)-(dx1,dy1), the pixel coordinates the rectangle that will
                 be actually drawn into the mpWindow, i.e. the clipped real rectangle that
                 avoids the non-visible parts. (offset_x,offset_y) are the pixel coordinates
                 that correspond to the window point (dx0,dy0) within the image "_bitmap", and
                 (b_width,b_height) is the size of the bitmap patch that will be drawn.

        (x0,y0) .................  (x1,y0)
            .                          .
            .                          .
        (x0,y1) ................   (x1,y1)
                      (In pixels!!)
        */

        // 1st step -------------------------------
        wxCoord x0 = w.x2p(_min_x);
        wxCoord y0 = w.y2p(_max_y);
        wxCoord x1 = w.x2p(_max_x);
        wxCoord y1 = w.y2p(_min_y);

        // 2nd step -------------------------------
        // Precompute the size of the actual bitmap pixel on the screen (e.g. will be >1 if zoomed in)
        double screenPixelX = (x1 - x0) / (double)_bitmap.GetWidth();
        double screenPixelY = (y1 - y0) / (double)_bitmap.GetHeight();

        // The minimum number of pixels that the streched image will overpass the actual mpWindow borders:
        wxCoord borderMarginX = (wxCoord)(screenPixelX + 1);  // ceil
        wxCoord borderMarginY = (wxCoord)(screenPixelY + 1);  // ceil

        // The actual drawn rectangle (dx0,dy0)-(dx1,dy1) is (x0,y0)-(x1,y1) clipped:
        wxCoord dx0 = x0, dx1 = x1, dy0 = y0, dy1 = y1;
        if (dx0 < 0) dx0 = -borderMarginX;
        if (dy0 < 0) dy0 = -borderMarginY;
        if (dx1 > w.GetScrX()) dx1 = w.GetScrX() + borderMarginX;
        if (dy1 > w.GetScrY()) dy1 = w.GetScrY() + borderMarginY;

        // For convenience, compute the width/height of the rectangle to be actually drawn:
        wxCoord d_width = dx1 - dx0 + 1;
        wxCoord d_height = dy1 - dy0 + 1;

        // Compute the pixel offsets in the internally stored bitmap:
        wxCoord offset_x = (wxCoord)((dx0 - x0) / screenPixelX);
        wxCoord offset_y = (wxCoord)((dy0 - y0) / screenPixelY);

        // and the size in pixel of the area to be actually drawn from the internally stored bitmap:
        wxCoord b_width = (wxCoord)((dx1 - dx0 + 1) / screenPixelX);
        wxCoord b_height = (wxCoord)((dy1 - dy0 + 1) / screenPixelY);

#ifdef MATHPLOT_DO_LOGGING
        wxLogMessage(_("[mpBitmapLayer::Plot] screenPixel: x=%f y=%f  d_width=%ix%i"), screenPixelX, screenPixelY,
                     d_width, d_height);
        wxLogMessage(_("[mpBitmapLayer::Plot] offset: x=%i y=%i  bmpWidth=%ix%i"), offset_x, offset_y, b_width,
                     b_height);
#endif

        // Is there any visible region?
        if (d_width > 0 && d_height > 0) {
            // Build the scaled bitmap from the image, only if it has changed:
            if (_scaledBitmap.GetWidth() != d_width || _scaledBitmap.GetHeight() != d_height ||
                _scaledBitmap_offset_x != offset_x || _scaledBitmap_offset_y != offset_y) {
                wxRect r(wxRect(offset_x, offset_y, b_width, b_height));
                // Just for the case....
                if (r.x < 0) r.x = 0;
                if (r.y < 0) r.y = 0;
                if (r.width > _bitmap.GetWidth()) r.width = _bitmap.GetWidth();
                if (r.height > _bitmap.GetHeight()) r.height = _bitmap.GetHeight();

                _scaledBitmap = wxBitmap(wxBitmap(_bitmap).GetSubBitmap(r).ConvertToImage().Scale(d_width, d_height));
                _scaledBitmap_offset_x = offset_x;
                _scaledBitmap_offset_y = offset_y;
            }

            // Draw it:
            dc.DrawBitmap(_scaledBitmap, dx0, dy0, true);
        }
    }

    // Draw the name label
    if (!_name.IsEmpty() && _showName) {
        dc.SetFont(_font);

        wxCoord tx, ty;
        dc.GetTextExtent(_name, &tx, &ty);

        if (HasBBox()) {
            wxCoord sx = (wxCoord)((_max_x - w.GetPosX()) * w.GetScaleX());
            wxCoord sy = (wxCoord)((w.GetPosY() - _max_y) * w.GetScaleY());

            tx = sx - tx - 8;
            ty = sy - 8 - ty;
        } else {
            const int sx = w.GetScrX() >> 1;
            const int sy = w.GetScrY() >> 1;

            if ((_flags & mpALIGNMASK) == mpALIGN_NE) {
                tx = sx - tx - 8;
                ty = -sy + 8;
            } else if ((_flags & mpALIGNMASK) == mpALIGN_NW) {
                tx = -sx + 8;
                ty = -sy + 8;
            } else if ((_flags & mpALIGNMASK) == mpALIGN_SW) {
                tx = -sx + 8;
                ty = sy - 8 - ty;
            } else {
                tx = sx - tx - 8;
                ty = sy - 8 - ty;
            }
        }

        dc.DrawText(_name, tx, ty);
    }
}
