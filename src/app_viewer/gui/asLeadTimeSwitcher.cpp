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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asLeadTimeSwitcher.h"
#include "asFrameForecast.h"

wxDEFINE_EVENT(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, wxCommandEvent);

asLeadTimeSwitcher::asLeadTimeSwitcher(wxWindow *parent, asWorkspace *workspace, asForecastManager *forecastManager,
                                       wxWindowID id, const wxPoint &pos, const wxSize &size, long style)
        : wxPanel(parent, id, pos, size, style),
          m_parent(NULL),
          m_workspace(workspace),
          m_forecastManager(forecastManager),
          m_bmp(NULL),
          m_gdc(NULL),
          m_cellWidth(int(40 * g_ppiScaleDc)),
          m_leadTime(0)
{
    Connect(wxEVT_PAINT, wxPaintEventHandler(asLeadTimeSwitcher::OnPaint), NULL, this);
    Connect(wxEVT_LEFT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), NULL, this);
    Connect(wxEVT_RIGHT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), NULL, this);
    Connect(wxEVT_MIDDLE_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), NULL, this);

    Layout();
}

asLeadTimeSwitcher::~asLeadTimeSwitcher()
{
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asLeadTimeSwitcher::OnPaint), NULL, this);
    Disconnect(wxEVT_LEFT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), NULL, this);
    Disconnect(wxEVT_RIGHT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), NULL, this);
    Disconnect(wxEVT_MIDDLE_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), NULL, this);

    wxDELETE(m_bmp);
}

void asLeadTimeSwitcher::SetParent(wxWindow *parent)
{
    m_parent = parent;
}

void asLeadTimeSwitcher::OnLeadTimeSlctChange(wxMouseEvent &event)
{
    wxPoint position = event.GetPosition();
    int val = floor(position.x / m_cellWidth);
    wxCommandEvent eventSlct(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED);
    eventSlct.SetInt(val);
    GetParent()->ProcessWindowEvent(eventSlct);
}

void asLeadTimeSwitcher::Draw(a1f &dates)
{
    // Required size
    int margin = 5 * g_ppiScaleDc;
    int width = (dates.size() + 1) * m_cellWidth;
    int height = m_cellWidth + margin;

    // Get color values
    int returnPeriodRef = m_workspace->GetAlarmsPanelReturnPeriod();
    float quantileThreshold = m_workspace->GetAlarmsPanelQuantile();
    a1f values = m_forecastManager->GetAggregator()->GetOverallMaxValues(dates, returnPeriodRef, quantileThreshold);
    wxASSERT(values.size() == dates.size());

    // Create bitmap
    wxBitmap *bmp = new wxBitmap(width, height);
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

    if (gc && values.size() > 0) {
        gc->SetPen(*wxBLACK);
        wxFont datesFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(datesFont, *wxBLACK);

        wxPoint startText(margin, m_cellWidth / 2 - 10);

        // For every lead time
        for (int iLead = 0; iLead < dates.size(); iLead++) {
            gc->SetPen(wxPen(GetBackgroundColour(), 3, wxPENSTYLE_SOLID));

            wxGraphicsPath path = gc->CreatePath();
            CreatePath(path, iLead);
            FillPath(gc, path, values[iLead]);

            wxString dateStr = asTime::GetStringTime(dates[iLead], "DD.MM");
            gc->SetFont(datesFont, *wxBLACK);
            CreateDatesText(gc, startText, iLead, dateStr);
        }

        // For the global view option
        gc->SetPen(wxPen(*wxWHITE, 1, wxPENSTYLE_SOLID));
        wxGraphicsPath path = gc->CreatePath();

        int segmentsTot = 7;
        const double scale = 0.16;
        wxPoint center(width - m_cellWidth / 2, m_cellWidth / 2);

        for (int i = 0; i < segmentsTot; i++) {
            CreatePathRing(path, center, scale, segmentsTot, i);
        }

        gc->StrokePath(path);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmap(bmp);
    wxDELETE(bmp);
    wxASSERT(!bmp);

    Refresh();

}

void asLeadTimeSwitcher::SetLeadTime(int leadTime)
{
    m_leadTime = leadTime;
}

void asLeadTimeSwitcher::SetLeadTimeMarker(int leadTime)
{
    // Clear overlay
    {
        wxClientDC dc(this);
        wxDCOverlay overlaydc(m_overlay, &dc);
        overlaydc.Clear();
    }
    m_overlay.Reset();

    // Get overlay
    wxClientDC dc(this);
    wxDCOverlay overlaydc(m_overlay, &dc);
    overlaydc.Clear();

    // Create graphics context
    wxGraphicsContext *gc = wxGraphicsContext::Create(dc);

    if (gc && leadTime >= 0) {
        // Set Lead time marker
        wxGraphicsPath markerPath = gc->CreatePath();
        CreatePathMarker(markerPath, leadTime);
        gc->SetBrush(wxBrush(*wxWHITE, wxBRUSHSTYLE_SOLID));
        gc->SetPen(wxPen(*wxBLACK, 1, wxPENSTYLE_SOLID));
        gc->DrawPath(markerPath);

        wxDELETE(gc);
    }
}

void asLeadTimeSwitcher::SetBitmap(wxBitmap *bmp)
{
    wxDELETE(m_bmp);
    wxASSERT(!m_bmp);

    if (bmp != NULL) {
        wxASSERT(bmp);
        m_bmp = new wxBitmap(*bmp);
        wxASSERT(m_bmp);
    }
}

void asLeadTimeSwitcher::OnPaint(wxPaintEvent &event)
{
    if (m_bmp != NULL) {
        wxPaintDC dc(this);
        dc.DrawBitmap(*m_bmp, 0, 0, true);
    }

    Layout();

    SetLeadTimeMarker(m_leadTime);

    event.Skip();
}

void asLeadTimeSwitcher::CreatePath(wxGraphicsPath &path, int iCol)
{
    wxPoint start(0, 0);

    double startPointX = (double) start.x + iCol * m_cellWidth;

    double startPointY = (double) start.y;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint(startPointX + m_cellWidth, startPointY);
    path.AddLineToPoint(startPointX + m_cellWidth, startPointY + m_cellWidth - 1);
    path.AddLineToPoint(startPointX, startPointY + m_cellWidth - 1);
    path.AddLineToPoint(startPointX, startPointY);

    path.CloseSubpath();
}

void asLeadTimeSwitcher::CreatePathRing(wxGraphicsPath &path, const wxPoint &center, double scale, int segmentsTotNb,
                                        int segmentNb)
{
    const wxDouble radiusOut = 100 * g_ppiScaleDc * scale;
    const wxDouble radiusIn = 40 * g_ppiScaleDc * scale;

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

void asLeadTimeSwitcher::FillPath(wxGraphicsContext *gc, wxGraphicsPath &path, float value)
{
    wxColour colour;

    if (asIsNaN(value)) // NaN -> gray
    {
        colour.Set(150, 150, 150);
    } else if (value == 0) // No rain -> white
    {
        colour.Set(255, 255, 255);
    } else if (value <= 0.5) // light green to yellow
    {
        int baseVal = 200;
        int valColour = ((value / (0.5))) * baseVal;
        int valColourCompl = ((value / (0.5))) * (255 - baseVal);
        if (valColour > baseVal)
            valColour = baseVal;
        if (valColourCompl + baseVal > 255)
            valColourCompl = 255 - baseVal;
        colour.Set((baseVal + valColourCompl), 255, (baseVal - valColour));
    } else // Yellow to red
    {
        int valColour = ((value - 0.5) / (0.5)) * 255;
        if (valColour > 255)
            valColour = 255;
        colour.Set(255, (255 - valColour), 0);
    }

    wxBrush brush(colour, wxBRUSHSTYLE_SOLID);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asLeadTimeSwitcher::CreateDatesText(wxGraphicsContext *gc, const wxPoint &start, int iCol, const wxString &label)
{
    double pointX = (double) start.x + iCol * m_cellWidth;
    double pointY = (double) start.y;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}

void asLeadTimeSwitcher::CreatePathMarker(wxGraphicsPath &path, int iCol)
{
    int outlier = g_ppiScaleDc * 5;
    int markerHeight = g_ppiScaleDc * 13;
    wxPoint start(m_cellWidth / 2, m_cellWidth - markerHeight);

    double startPointX = (double) start.x + iCol * m_cellWidth;
    double startPointY = (double) start.y;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint(startPointX - m_cellWidth / (g_ppiScaleDc * 5),
                        startPointY + markerHeight + outlier - g_ppiScaleDc * 1);
    path.AddLineToPoint(startPointX + m_cellWidth / (g_ppiScaleDc * 5),
                        startPointY + markerHeight + outlier - g_ppiScaleDc * 1);
    path.AddLineToPoint(startPointX, startPointY);

    path.CloseSubpath();
}