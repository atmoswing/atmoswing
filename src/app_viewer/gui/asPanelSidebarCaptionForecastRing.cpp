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

/*
 * asPanelSidebarCaptionForecastRing
 */

asPanelSidebarCaptionForecastRing::asPanelSidebarCaptionForecastRing(wxWindow* parent, wxWindowID id,
                                                                     const wxPoint& pos, const wxSize& size, long style)
    : asPanelSidebar(parent, id, pos, size, style) {
    m_header->SetLabelText(_("Forecast caption"));

    m_panelDrawing = new asPanelSidebarCaptionForecastRingDrawing(
        this, wxID_ANY, wxDefaultPosition, wxSize(240 * g_ppiScaleDc, 260 * g_ppiScaleDc), wxTAB_TRAVERSAL);
    m_sizerContent->Add(m_panelDrawing, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRing::OnPaint), nullptr, this);

    Layout();
    m_sizerMain->Fit(this);
    FitInside();
}

asPanelSidebarCaptionForecastRing::~asPanelSidebarCaptionForecastRing() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRing::OnPaint), nullptr, this);
}

void asPanelSidebarCaptionForecastRing::OnPaint(wxPaintEvent& event) {
    event.Skip();
}

void asPanelSidebarCaptionForecastRing::SetDates(a1f& dates) {
    m_panelDrawing->DrawDates(dates);
}

void asPanelSidebarCaptionForecastRing::SetColorbarMax(double valmax) {
    m_panelDrawing->DrawColorbar(valmax);
}

/*
 * asPanelSidebarCaptionForecastRingDrawing
 */

asPanelSidebarCaptionForecastRingDrawing::asPanelSidebarCaptionForecastRingDrawing(wxWindow* parent, wxWindowID id,
                                                                                   const wxPoint& pos,
                                                                                   const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style) {
    m_bmpDates = nullptr;
    m_bmpColorbar = nullptr;
    m_gdc = nullptr;

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRingDrawing::OnPaint), nullptr, this);

    a1f emptyDates;
    DrawDates(emptyDates);
    DrawColorbar(1);

    Layout();
}

asPanelSidebarCaptionForecastRingDrawing::~asPanelSidebarCaptionForecastRingDrawing() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarCaptionForecastRingDrawing::OnPaint), nullptr, this);
    wxDELETE(m_bmpDates);
    wxDELETE(m_bmpColorbar);
}

void asPanelSidebarCaptionForecastRingDrawing::DrawDates(a1f& dates) {
    wxDELETE(m_bmpDates);
    m_bmpDates = new wxBitmap(int(240 * g_ppiScaleDc), int(200 * g_ppiScaleDc));
    wxASSERT(m_bmpDates);

    // Set the default pens
    wxPen greyPen(*wxLIGHT_GREY_PEN);
    wxPen blackPen(*wxBLACK_PEN);

    // Create device context
    wxMemoryDC dc(*m_bmpDates);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();

    // Create graphics context
    wxGraphicsContext* gc = wxGraphicsContext::Create(dc);

    if (gc) {
        gc->SetPen(*wxBLACK);
        wxFont defFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(defFont, *wxBLACK);
        wxGraphicsPath path = gc->CreatePath();

        wxPoint center(120 * g_ppiScaleDc, 91 * g_ppiScaleDc);  // Looks better than 105

        int leadTimeSize = dates.size();

        if (leadTimeSize == 0) {
            dc.SelectObject(wxNullBitmap);
            Refresh();
            return;
        }

        const double scale = 0.9 * g_ppiScaleDc;

        // Draw ticks
        gc->SetBrush(*wxTRANSPARENT_BRUSH);
        double prevLeadTimeDate = dates[0];
        for (int iLead = 1; iLead < leadTimeSize; iLead++) {
            double date = dates[iLead];
            if (floor(prevLeadTimeDate) == floor(date)) {
                gc->SetPen(greyPen);
            } else {
                gc->SetPen(blackPen);
            }

            prevLeadTimeDate = date;

            // Create shape
            path = gc->CreatePath();
            CreatePathTick(path, center, scale, leadTimeSize, iLead);
            gc->DrawPath(path);
        }

        // Write dates
        gc->SetPen(blackPen);
        prevLeadTimeDate = dates[0];
        int prevIdx = 0;
        for (int iLead = 0; iLead <= leadTimeSize; iLead++) {
            double date = 0;
            if (iLead < leadTimeSize) {
                date = dates[iLead];
                if (floor(prevLeadTimeDate) == floor(date)) {
                    continue;
                }
            }

            // Write date
            int count = iLead - prevIdx;
            wxString dateStr = asTime::GetStringTime(prevLeadTimeDate, "DD.MM");
            CreateDatesText(gc, center, scale, leadTimeSize, prevIdx, count, dateStr);

            prevLeadTimeDate = date;
            prevIdx = iLead;
        }

        // Set the pen
        gc->SetPen(blackPen);
        gc->SetBrush(*wxTRANSPARENT_BRUSH);

        // Draw overall box
        path = gc->CreatePath();
        CreatePathAround(path, center, scale);
        gc->DrawPath(path);

        gc->StrokePath(path);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    Refresh();
}

void asPanelSidebarCaptionForecastRingDrawing::DrawColorbar(double valmax) {
    wxDELETE(m_bmpColorbar);
    m_bmpColorbar = new wxBitmap(int(240 * g_ppiScaleDc), int(70 * g_ppiScaleDc));
    wxASSERT(m_bmpColorbar);

    // Create device context
    wxMemoryDC dc(*m_bmpColorbar);
    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();

    // Create graphics context
    wxGraphicsContext* gc = wxGraphicsContext::Create(dc);

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

    Refresh();
}

void asPanelSidebarCaptionForecastRingDrawing::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);

    if (m_bmpDates != nullptr) {
        dc.DrawBitmap(*m_bmpDates, 0, 0, true);
    }

    if (m_bmpColorbar != nullptr) {
        dc.DrawBitmap(*m_bmpColorbar, 0, 190 * g_ppiScaleDc, true);
    }

    Layout();

    event.Skip();
}

void asPanelSidebarCaptionForecastRingDrawing::CreatePathTick(wxGraphicsPath& path, const wxPoint& center, double scale,
                                                              int segmentsTotNb, int segmentNb) {
    const wxDouble radiusOut = 100 * scale;
    const wxDouble radiusIn = 40 * scale;

    wxDouble segmentStart = -0.5 * M_PI + ((double)segmentNb / (double)segmentsTotNb) * (1.5 * M_PI);
    wxDouble centerX = (wxDouble)center.x;
    wxDouble centerY = (wxDouble)center.y;

    // Get starting point
    double dXin = cos(segmentStart) * radiusIn;
    double dXout = cos(segmentStart) * radiusOut;
    double dYin = sin(segmentStart) * radiusIn;
    double dYout = sin(segmentStart) * radiusOut;

    path.MoveToPoint(centerX + dXin, centerY + dYin);
    path.AddLineToPoint(centerX + dXout, centerY + dYout);
}

void asPanelSidebarCaptionForecastRingDrawing::CreatePathAround(wxGraphicsPath& path, const wxPoint& center, double scale) {
    const wxDouble radiusOut = 100 * scale;
    const wxDouble radiusIn = 40 * scale;

    wxDouble segmentStart = -0.5 * M_PI;
    wxDouble segmentEnd = M_PI;
    wxDouble centerX = (wxDouble)center.x;
    wxDouble centerY = (wxDouble)center.y;

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

void asPanelSidebarCaptionForecastRingDrawing::CreateDatesText(wxGraphicsContext* gc, const wxPoint& center,
                                                               double scale, int segmentsTotNb, int segmentNb,
                                                               int count, const wxString& label) {
    // Get geometric elements
    const wxDouble radiusMean = 70 * scale;
    wxDouble segmentMean = -0.5 * M_PI + ((segmentNb + double(count) / 2.0) / segmentsTotNb) * (1.5 * M_PI);
    auto centerX = (wxDouble)center.x;
    auto centerY = (wxDouble)center.y;

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

void asPanelSidebarCaptionForecastRingDrawing::CreateColorbarPath(wxGraphicsPath& path) {
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

void asPanelSidebarCaptionForecastRingDrawing::FillColorbar(wxGraphicsContext* gc, wxGraphicsPath& path) {
    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    wxGraphicsGradientStops stops(wxColour(200, 255, 200), wxColour(255, 0, 0));
    stops.Add(wxColour(255, 255, 0), 0.5);
    wxGraphicsBrush brush = gc->CreateLinearGradientBrush(x, y, x + w, y, stops);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asPanelSidebarCaptionForecastRingDrawing::CreateColorbarText(wxGraphicsContext* gc, wxGraphicsPath& path,
                                                                  double valmax) {
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
    wxString labelMid = asStrF("%g", valmax / 2.0);
    wxString labelEnd = asStrF("%g", valmax);

    // Draw text
    int dy = 12 * g_ppiScaleDc;
    gc->DrawText(labelStart, x + 4, y + dy);
    gc->DrawText(labelMid, x + w / 2 + 4, y + dy);
    gc->DrawText(labelEnd, x + w + 4, y + dy);
}

void asPanelSidebarCaptionForecastRingDrawing::CreateColorbarOtherClasses(wxGraphicsContext* gc, wxGraphicsPath& path) {
    gc->SetPen(*wxBLACK);

    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);
    int dh1 = 22 * g_ppiScaleDc;
    int dh2 = 10 * g_ppiScaleDc;
    int dh3 = 18 * g_ppiScaleDc;
    int dw = 10 * g_ppiScaleDc;

    // Create first box
    wxGraphicsPath pathBox1 = gc->CreatePath();
    pathBox1.MoveToPoint(x, y + h + dh1);
    pathBox1.AddLineToPoint(x, y + h + dh1 + dh2);
    pathBox1.AddLineToPoint(x + dw, y + h + dh1 + dh2);
    pathBox1.AddLineToPoint(x + dw, y + h + dh1);
    pathBox1.CloseSubpath();

    wxColour colour = wxColour();
    colour.Set(255, 255, 255);
    wxBrush brush1(colour, wxBRUSHSTYLE_SOLID);
    gc->SetBrush(brush1);
    gc->DrawPath(pathBox1);

    // Create second box
    wxGraphicsPath pathBox2 = gc->CreatePath();
    pathBox2.MoveToPoint(x, y + h + dh1 + dh3);
    pathBox2.AddLineToPoint(x, y + h + dh1 + dh2 + dh3);
    pathBox2.AddLineToPoint(x + dw, y + h + dh1 + dh2 + dh3);
    pathBox2.AddLineToPoint(x + dw, y + h + dh1 + dh3);
    pathBox2.CloseSubpath();

    colour.Set(150, 150, 150);
    wxBrush brush2(colour, wxBRUSHSTYLE_SOLID);
    gc->SetBrush(brush2);
    gc->DrawPath(pathBox2);

    // Set labels
    wxString label1 = _("No rainfall");
    wxString label2 = _("Missing data");

    // Draw text
    int dwLabel = 14 * g_ppiScaleDc;
    gc->DrawText(label1, x + dwLabel, y + h + dh1 - 1);
    gc->DrawText(label2, x + dwLabel, y + h + dh1 + dh3 - 1);
}
