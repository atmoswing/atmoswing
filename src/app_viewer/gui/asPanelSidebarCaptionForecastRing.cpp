/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asPanelSidebarCaptionForecastRing.h"

#include "images.h"

/*
 * asPanelSidebarCaptionForecastRing
 */

asPanelSidebarCaptionForecastRing::asPanelSidebarCaptionForecastRing(wxWindow *parent, wxWindowID id,
                                                                     const wxPoint &pos, const wxSize &size, long style)
        : asPanelSidebar(parent, id, pos, size, style)
{
    m_header->SetLabelText(_("Forecast caption"));

    m_panelDrawing = new asPanelSidebarCaptionForecastRingDrawing(this, wxID_ANY, wxDefaultPosition,
                                                                  wxSize(240 * g_ppiScaleDc, 240 * g_ppiScaleDc),
                                                                  wxTAB_TRAVERSAL);
    m_sizerContent->Add(m_panelDrawing, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRing::OnPaint), NULL, this);

    Layout();
    m_sizerMain->Fit(this);
    FitInside();
}

asPanelSidebarCaptionForecastRing::~asPanelSidebarCaptionForecastRing()
{
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRing::OnPaint), NULL, this);
}

void asPanelSidebarCaptionForecastRing::OnPaint(wxPaintEvent &event)
{
    event.Skip();
}

void asPanelSidebarCaptionForecastRing::SetDates(Array1DFloat &dates)
{
    m_panelDrawing->DrawDates(dates);
}

void asPanelSidebarCaptionForecastRing::SetColorbarMax(double valmax)
{
    m_panelDrawing->DrawColorbar(valmax);
}


/*
 * asPanelSidebarCaptionForecastRingDrawing
 */

asPanelSidebarCaptionForecastRingDrawing::asPanelSidebarCaptionForecastRingDrawing(wxWindow *parent, wxWindowID id,
                                                                                   const wxPoint &pos,
                                                                                   const wxSize &size, long style)
        : wxPanel(parent, id, pos, size, style)
{
    m_bmpDates = NULL;
    m_bmpColorbar = NULL;
    m_gdc = NULL;

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRingDrawing::OnPaint), NULL, this);

    Array1DFloat emptyDates;
    DrawDates(emptyDates);
    DrawColorbar(1);

    Layout();
}

asPanelSidebarCaptionForecastRingDrawing::~asPanelSidebarCaptionForecastRingDrawing()
{
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRingDrawing::OnPaint), NULL, this);
    wxDELETE(m_bmpDates);
    wxDELETE(m_bmpColorbar);
}

void asPanelSidebarCaptionForecastRingDrawing::DrawDates(Array1DFloat &dates)
{
    wxBitmap *bmp = new wxBitmap(int(240 * g_ppiScaleDc), int(182 * g_ppiScaleDc));
    wxASSERT(bmp);

    // Create device context
    wxMemoryDC dc(*bmp);
    dc.SetBackground(wxBrush(GetBackgroundColour()));
#if defined(__UNIX__)
    dc.SetBackground(wxBrush(g_linuxBgColour));
#endif
    dc.Clear();

    // Create graphics context
    wxGraphicsContext *gc = wxGraphicsContext::Create(dc);

    if (gc) {
        gc->SetPen(*wxBLACK);
        wxFont defFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(defFont, *wxBLACK);
        wxGraphicsPath path = gc->CreatePath();

        wxPoint center(120 * g_ppiScaleDc, 91 * g_ppiScaleDc); // Looks better than 105

        int segmentsTot = dates.size();
        const double scale = 0.9 * g_ppiScaleDc;

        if (segmentsTot == 0) {
            CreateDatesPath(path, center, scale, 1, 0);
        } else {
            for (int i_leadtime = 0; i_leadtime < segmentsTot; i_leadtime++) {
                CreateDatesPath(path, center, scale, segmentsTot, i_leadtime);
                wxString dateStr = asTime::GetStringTime(dates[i_leadtime], "DD.MM");
                CreateDatesText(gc, center, scale, segmentsTot, i_leadtime, dateStr);
            }
        }

        gc->StrokePath(path);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmapDates(bmp);
    wxDELETE(bmp);
    wxASSERT(!bmp);

    Refresh();
}

void asPanelSidebarCaptionForecastRingDrawing::DrawColorbar(double valmax)
{
    wxBitmap *bmp = new wxBitmap(int(240 * g_ppiScaleDc), int(50 * g_ppiScaleDc));
    wxASSERT(bmp);

    // Create device context
    wxMemoryDC dc(*bmp);
    dc.SetBackground(wxBrush(GetBackgroundColour()));
#if defined(__UNIX__)
    dc.SetBackground(wxBrush(g_linuxBgColour));
#endif
    dc.Clear();

    // Create graphics context
    wxGraphicsContext *gc = wxGraphicsContext::Create(dc);

    if (gc) {
        gc->SetPen(*wxBLACK);
        wxFont defFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(defFont, *wxBLACK);
        wxGraphicsPath path = gc->CreatePath();

        CreateColorbarPath(path);
        FillColorbar(gc, path);
        CreateColorbarText(gc, path, valmax);
        CreateColorbarOtherClasses(gc, path);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmapColorbar(bmp);
    wxDELETE(bmp);
    wxASSERT(!bmp);

    Refresh();
}

void asPanelSidebarCaptionForecastRingDrawing::SetBitmapDates(wxBitmap *bmp)
{
    wxDELETE(m_bmpDates);
    wxASSERT(!m_bmpDates);

    if (bmp != NULL) {
        wxASSERT(bmp);
        m_bmpDates = new wxBitmap(*bmp);
        wxASSERT(m_bmpDates);
    }
}

void asPanelSidebarCaptionForecastRingDrawing::SetBitmapColorbar(wxBitmap *bmp)
{
    wxDELETE(m_bmpColorbar);
    wxASSERT(!m_bmpColorbar);

    if (bmp != NULL) {
        wxASSERT(bmp);
        m_bmpColorbar = new wxBitmap(*bmp);
        wxASSERT(m_bmpColorbar);
    }
}

void asPanelSidebarCaptionForecastRingDrawing::OnPaint(wxPaintEvent &event)
{
    wxPaintDC dc(this);

    if (m_bmpDates != NULL) {
        dc.DrawBitmap(*m_bmpDates, 0, 0, true);
    }

    if (m_bmpColorbar != NULL) {
        dc.DrawBitmap(*m_bmpColorbar, 0, 190 * g_ppiScaleDc, true);
    }

    Layout();

    event.Skip();
}

void asPanelSidebarCaptionForecastRingDrawing::CreateDatesPath(wxGraphicsPath &path, const wxPoint &center,
                                                               double scale, int segmentsTotNb, int segmentNb)
{
    const wxDouble radiusOut = 100 * scale;
    const wxDouble radiusIn = 40 * scale;

    wxDouble segmentStart = -0.5 * M_PI + ((double) segmentNb / (double) segmentsTotNb) * (1.5 * M_PI);
    wxDouble segmentEnd = -0.5 * M_PI + ((double) (segmentNb + 1) / (double) segmentsTotNb) * (1.5 * M_PI);
    wxDouble centerX = (wxDouble) center.x;
    wxDouble centerY = (wxDouble) center.y;

    // Get starting point
    double dX = cos(segmentStart) * radiusOut;
    double dY = sin(segmentStart) * radiusOut;
    wxDouble startPointX = centerX + dX;
    wxDouble startPointY = centerY + dY;

    path.MoveToPoint(startPointX, startPointY);

    path.AddArc(centerX, centerY, radiusOut, segmentStart, segmentEnd, true);

    const wxDouble radiusRatio = ((radiusOut - radiusIn) / radiusOut);
    wxPoint2DDouble currentPoint = path.GetCurrentPoint();
    wxDouble newPointX = currentPoint.m_x - (currentPoint.m_x - centerX) * radiusRatio;
    wxDouble newPointY = currentPoint.m_y - (currentPoint.m_y - centerY) * radiusRatio;

    path.AddLineToPoint(newPointX, newPointY);

    path.AddArc(centerX, centerY, radiusIn, segmentEnd, segmentStart, false);

    path.CloseSubpath();
}

void asPanelSidebarCaptionForecastRingDrawing::CreateDatesText(wxGraphicsContext *gc, const wxPoint &center,
                                                               double scale, int segmentsTotNb, int segmentNb,
                                                               const wxString &label)
{
    // Get geometric elements
    const wxDouble radiusMean = 70 * scale;
    wxDouble segmentMean = -0.5 * M_PI + ((double) (segmentNb + 0.5) / (double) segmentsTotNb) * (1.5 * M_PI);
    wxDouble centerX = (wxDouble) center.x;
    wxDouble centerY = (wxDouble) center.y;

    // Text extent
    wxDouble w, h;
    gc->GetTextExtent(label, &w, &h);

    // Get point coordinates
    double dX = cos(segmentMean) * radiusMean;
    double dY = sin(segmentMean) * radiusMean;
    wxDouble newPointX = centerX + dX - w / 2.0;
    wxDouble newPointY = centerY + dY - h / 2.0;

    // Draw text
    gc->DrawText(label, newPointX, newPointY);
}

void asPanelSidebarCaptionForecastRingDrawing::CreateColorbarPath(wxGraphicsPath &path)
{
    int startX = 30;
    int endX = 210 * g_ppiScaleDc;
    int startY = 2 * g_ppiScaleDc;
    int endY = 11 * g_ppiScaleDc;

    path.MoveToPoint(startX, startY);
    path.AddLineToPoint(endX, startY);
    path.AddLineToPoint(endX, endY);
    path.AddLineToPoint(startX, endY);
    path.CloseSubpath();
}

void asPanelSidebarCaptionForecastRingDrawing::FillColorbar(wxGraphicsContext *gc, wxGraphicsPath &path)
{
    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    wxGraphicsGradientStops stops(wxColour(200, 255, 200), wxColour(255, 0, 0));
    stops.Add(wxColour(255, 255, 0), 0.5);
    wxGraphicsBrush brush = gc->CreateLinearGradientBrush(x, y, x + w, y, stops);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asPanelSidebarCaptionForecastRingDrawing::CreateColorbarText(wxGraphicsContext *gc, wxGraphicsPath &path,
                                                                  double valmax)
{
    gc->SetPen(*wxBLACK);

    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    // Correction of the coordinates needed on Linux
#if defined(__WXMSW__)
    int corr = 0;
#elif defined(__WXMAC__)
    int corr = 0;
#elif defined(__UNIX__)
    int corr = 1;
#endif

    // Set ticks
    wxGraphicsPath pathTickStart = gc->CreatePath();
    pathTickStart.MoveToPoint(x + corr, y + corr);
    pathTickStart.AddLineToPoint(x + corr, y + corr + h + 5);
    gc->StrokePath(pathTickStart);
    wxGraphicsPath pathTickMid = gc->CreatePath();
    pathTickMid.MoveToPoint(x + w / 2, y + corr);
    pathTickMid.AddLineToPoint(x + w / 2, y + corr + h + 5);
    gc->StrokePath(pathTickMid);
    wxGraphicsPath pathTickEnd = gc->CreatePath();
    pathTickEnd.MoveToPoint(x - corr + w, y + corr);
    pathTickEnd.AddLineToPoint(x - corr + w, y + corr + h + 5);
    gc->StrokePath(pathTickEnd);

    // Set labels
    wxString labelStart = "0";
    wxString labelMid = wxString::Format("%g", valmax / 2.0);
    wxString labelEnd = wxString::Format("%g", valmax);

    // Draw text
    int dy = 12 * g_ppiScaleDc;
    gc->DrawText(labelStart, x + 4, y + dy);
    gc->DrawText(labelMid, x + w / 2 + 4, y + dy);
    gc->DrawText(labelEnd, x + w + 4, y + dy);

}

void asPanelSidebarCaptionForecastRingDrawing::CreateColorbarOtherClasses(wxGraphicsContext *gc, wxGraphicsPath &path)
{
    gc->SetPen(*wxBLACK);

    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);
    int dh1 = 20 * g_ppiScaleDc;
    int dh2 = 10 * g_ppiScaleDc;
    int dw = 10 * g_ppiScaleDc;
    int halfWidth = w / 2;

    // Create first box
    wxGraphicsPath pathBox1 = gc->CreatePath();
    pathBox1.MoveToPoint(x, y + h + dh1);
    pathBox1.AddLineToPoint(x, y + h + dh1 + dh2);
    pathBox1.AddLineToPoint(x + dw, y + h + dh1 + dh2);
    pathBox1.AddLineToPoint(x + dw, y + h + dh1);
    pathBox1.CloseSubpath();

    wxColour colour = wxColour();
    colour.Set(255, 255, 255);
    wxBrush brush1(colour, wxSOLID);
    gc->SetBrush(brush1);
    gc->DrawPath(pathBox1);

    // Create second box
    wxGraphicsPath pathBox2 = gc->CreatePath();
    pathBox2.MoveToPoint(x + halfWidth, y + h + dh1);
    pathBox2.AddLineToPoint(x + halfWidth, y + h + dh1 + dh2);
    pathBox2.AddLineToPoint(x + halfWidth + dw, y + h + dh1 + dh2);
    pathBox2.AddLineToPoint(x + halfWidth + dw, y + h + dh1);
    pathBox2.CloseSubpath();

    colour.Set(150, 150, 150);
    wxBrush brush2(colour, wxSOLID);
    gc->SetBrush(brush2);
    gc->DrawPath(pathBox2);

    // Set labels
    wxString label1 = _("No rainfall");
    wxString label2 = _("Missing data");

    // Draw text
    int dwLabel = 14 * g_ppiScaleDc;
    gc->DrawText(label1, x + dwLabel, y + h + dh1 - 1);
    gc->DrawText(label2, x + halfWidth + dwLabel, y + h + dh1 - 1);
}
