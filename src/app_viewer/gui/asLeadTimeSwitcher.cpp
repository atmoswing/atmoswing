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

#include "asFrameViewer.h"

wxDEFINE_EVENT(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, wxCommandEvent);

asLeadTimeSwitcher::asLeadTimeSwitcher(wxWindow* parent, asWorkspace* workspace, asForecastManager* forecastManager,
                                       wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style),
      m_parent(nullptr),
      m_workspace(workspace),
      m_forecastManager(forecastManager),
      m_bmp(nullptr),
      m_gdc(nullptr),
      m_subDailyMode(false),
      m_subDailyFraction(1.0),
      m_cellWidth(int(40 * g_ppiScaleDc)),
      m_cellHeight(int(40 * g_ppiScaleDc)),
      m_margin(5 * g_ppiScaleDc),
      m_leadTime(0) {

    m_hasSubDaily = m_forecastManager->HasSubDailyForecasts();
    if (m_hasSubDaily) {
        m_cellWidth = int(50 * g_ppiScaleDc);
    }

    // Required size
    int width = (m_forecastManager->GetFullTargetDates().size() + 1) * m_cellWidth * g_ppiScaleDc;
    int height = m_cellHeight * g_ppiScaleDc + m_margin;
    SetSize(wxSize(width, height));

    Connect(wxEVT_PAINT, wxPaintEventHandler(asLeadTimeSwitcher::OnPaint), nullptr, this);
    Connect(wxEVT_LEFT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), nullptr, this);
    Connect(wxEVT_RIGHT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), nullptr, this);
    Connect(wxEVT_MIDDLE_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), nullptr, this);
}

asLeadTimeSwitcher::~asLeadTimeSwitcher() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asLeadTimeSwitcher::OnPaint), nullptr, this);
    Disconnect(wxEVT_LEFT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), nullptr, this);
    Disconnect(wxEVT_RIGHT_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), nullptr, this);
    Disconnect(wxEVT_MIDDLE_UP, wxMouseEventHandler(asLeadTimeSwitcher::OnLeadTimeSlctChange), nullptr, this);

    wxDELETE(m_bmp);
}

void asLeadTimeSwitcher::SetForecastSelection(int iMethod, int iForecast) {
    int i = wxMax(iMethod, 0);
    int j = wxMax(iForecast, 0);

    if (m_forecastManager->GetMethodsNb() < i) return;
    if (m_forecastManager->GetForecastsNb(i) < j) return;

    m_subDailyMode = m_forecastManager->GetForecast(i, j)->IsSubDaily();
}

void asLeadTimeSwitcher::OnLeadTimeSlctChange(wxMouseEvent& event) {
    wxBusyCursor wait;

    wxPoint position = event.GetPosition();
    int val = 0;

    // Check if forecast ring display
    if (position.x > GetSize().GetWidth() - m_cellWidth) {
        val = -1;
    } else {
        val = floor(position.x / m_cellWidth);
        if (m_subDailyMode) {
            val = floor(position.x / (m_cellWidth * m_subDailyFraction));
        }
    }

    wxCommandEvent eventSlct(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED);
    eventSlct.SetInt(val);
    GetParent()->ProcessWindowEvent(eventSlct);
}

void asLeadTimeSwitcher::Draw(a1f& dates) {
    // Required size
    int width = (dates.size() + 1) * m_cellWidth;
    int height = m_cellHeight + m_margin;

    // Get values at a daily time step
    int returnPeriodRef = m_workspace->GetAlarmsPanelReturnPeriod();
    float quantileThreshold = m_workspace->GetAlarmsPanelQuantile();
    a1f values = m_forecastManager->GetAggregator()->GetOverallMaxValues(dates, returnPeriodRef, quantileThreshold);
    wxASSERT(values.size() == dates.size());

    // Get values at a sub-daily time step
    a1f valuesSubDaily;
    if (m_hasSubDaily) {
        for (int iMethod = 0; iMethod < m_forecastManager->GetMethodsNb(); iMethod++) {
            if (!m_forecastManager->GetAggregator()->GetForecast(iMethod, 0)->IsSubDaily()) {
                continue;
            }
            a1f methodMaxValues = m_forecastManager->GetAggregator()->GetMethodMaxValues(
                dates, iMethod, returnPeriodRef, quantileThreshold);
            methodMaxValues = (methodMaxValues.isFinite()).select(methodMaxValues, 0);
            if (valuesSubDaily.size() == 0) {
                valuesSubDaily = methodMaxValues;
                continue;
            }
            if (valuesSubDaily.size() != methodMaxValues.size()) {
                wxLogError(_("Combination of sub-daily time steps not yet implemented."));
                continue;
            }
            valuesSubDaily = valuesSubDaily.cwiseMax(methodMaxValues);
        }
        valuesSubDaily = (valuesSubDaily > 0).select(valuesSubDaily, NAN);
    }

    // Handle sub-daily time steps
    m_subDailyFraction = double(dates.size()) / double(valuesSubDaily.size());
    if (m_subDailyFraction < 0.2) {
        wxLogError(_("Too small time steps are not supported in the lead time switcher."));
        return;
    }

    // Create bitmap
    wxDELETE(m_bmp);
    m_bmp = new wxBitmap(width, height);
    wxASSERT(m_bmp);

    // Create device context
    wxMemoryDC dc(*m_bmp);
    dc.SetBackground(wxBrush(GetBackgroundColour()));
    dc.Clear();

    // Create graphics context
    wxGraphicsContext* gc = wxGraphicsContext::Create(dc);

    if (gc && values.size() > 0) {
        gc->SetPen(*wxBLACK);
        int fontSize = 10;
#ifdef __LINUX__
        fontSize = 8;
#endif
        wxFont datesFont(fontSize, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(datesFont, *wxBLACK);

        wxPoint startText(m_margin + (m_cellWidth - 40) / 2, m_cellHeight / 2.5 - fontSize);

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

        if (m_hasSubDaily) {
            for (int iLead = 0; iLead < valuesSubDaily.size(); iLead++) {
                gc->SetPen(wxPen(GetBackgroundColour(), 1, wxPENSTYLE_SOLID));

                wxGraphicsPath path = gc->CreatePath();
                CreatePathSubDaily(path, iLead);
                FillPath(gc, path, valuesSubDaily[iLead]);
            }
        }

        // For the global view option
        gc->SetPen(wxPen(*wxWHITE, 1, wxPENSTYLE_SOLID));
        wxGraphicsPath path = gc->CreatePath();

        int segmentsTot = 7;
        const double scale = 0.16;
        wxPoint center(width - m_cellWidth / 2, m_cellHeight / 2);

        for (int i = 0; i < segmentsTot; i++) {
            CreatePathRing(path, center, scale, segmentsTot, i);
        }

        gc->StrokePath(path);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    Refresh();
}

void asLeadTimeSwitcher::SetLeadTime(int leadTime) {
    m_leadTime = leadTime;
}

void asLeadTimeSwitcher::SetLeadTimeMarker(int leadTime) {
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
    wxGraphicsContext* gc = wxGraphicsContext::Create(dc);

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

void asLeadTimeSwitcher::OnPaint(wxPaintEvent& event) {
    if (m_bmp != nullptr) {
        wxPaintDC dc(this);
        dc.DrawBitmap(*m_bmp, 0, 0, true);
    }

    Layout();

    SetLeadTimeMarker(m_leadTime);

    event.Skip();
}

void asLeadTimeSwitcher::CreatePath(wxGraphicsPath& path, int iCol) const {
    wxPoint start(0, 0);

    double startPointX = (double)start.x + iCol * m_cellWidth;

    auto startPointY = (double)start.y;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint(startPointX + m_cellWidth, startPointY);
    path.AddLineToPoint(startPointX + m_cellWidth, startPointY + m_cellHeight - 1);
    path.AddLineToPoint(startPointX, startPointY + m_cellHeight - 1);
    path.AddLineToPoint(startPointX, startPointY);

    path.CloseSubpath();
}

void asLeadTimeSwitcher::CreatePathSubDaily(wxGraphicsPath& path, int iCol) const {
    double heightFraction = 0.4;
    wxPoint start(0, 0);
    int fullDaysNb = floor(iCol * m_subDailyFraction);
    int subDaysNb = iCol - fullDaysNb / m_subDailyFraction;

    int smallCellWidth = (m_cellWidth - 2) * m_subDailyFraction;

    double startPointX = (double)start.x + fullDaysNb * m_cellWidth + subDaysNb * smallCellWidth + 1;

    auto startPointY = (double)start.y + (1 - heightFraction) * m_cellHeight;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint(startPointX + smallCellWidth, startPointY);
    path.AddLineToPoint(startPointX + smallCellWidth, startPointY + m_cellHeight * heightFraction - 1);
    path.AddLineToPoint(startPointX, startPointY + m_cellHeight * heightFraction - 1);
    path.AddLineToPoint(startPointX, startPointY);

    path.CloseSubpath();
}

void asLeadTimeSwitcher::CreatePathRing(wxGraphicsPath& path, const wxPoint& center, double scale, int segmentsTotNb,
                                        int segmentNb) {
    const wxDouble radiusOut = 100 * g_ppiScaleDc * scale;
    const wxDouble radiusIn = 40 * g_ppiScaleDc * scale;

    wxDouble segmentStart = -0.5 * M_PI + ((double)segmentNb / (double)segmentsTotNb) * (1.5 * M_PI);
    wxDouble segmentEnd = -0.5 * M_PI + ((double)(segmentNb + 1) / (double)segmentsTotNb) * (1.5 * M_PI);
    auto centerX = (wxDouble)center.x;
    auto centerY = (wxDouble)center.y;

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

void asLeadTimeSwitcher::FillPath(wxGraphicsContext* gc, wxGraphicsPath& path, float value) {
    wxColour colour;

    if (isnan(value))  // NaN -> gray
    {
        colour.Set(150, 150, 150);
    } else if (value == 0)  // No rain -> white
    {
        colour.Set(255, 255, 255);
    } else if (value <= 0.5)  // light green to yellow
    {
        int baseVal = 200;
        int valColour = ((value / (0.5))) * baseVal;
        int valColourCompl = ((value / (0.5))) * (255 - baseVal);
        if (valColour > baseVal) valColour = baseVal;
        if (valColourCompl + baseVal > 255) valColourCompl = 255 - baseVal;
        colour.Set((baseVal + valColourCompl), 255, (baseVal - valColour));
    } else  // Yellow to red
    {
        int valColour = ((value - 0.5) / (0.5)) * 255;
        if (valColour > 255) valColour = 255;
        colour.Set(255, (255 - valColour), 0);
    }

    wxBrush brush(colour, wxBRUSHSTYLE_SOLID);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asLeadTimeSwitcher::CreateDatesText(wxGraphicsContext* gc, const wxPoint& start, int iCol, const wxString& label) {
    double pointX = (double)start.x + iCol * m_cellWidth;
    auto pointY = (double)start.y;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}

void asLeadTimeSwitcher::CreatePathMarker(wxGraphicsPath& path, int iCol) {
    int outlier = g_ppiScaleDc * 5;
    int markerHeight = g_ppiScaleDc * 13;
    int cellWidth = m_cellWidth;

    double startPointX = (double)cellWidth / 2 + iCol * cellWidth;
    auto startPointY = (double)m_cellHeight - markerHeight;

    if (m_subDailyMode) {
        cellWidth = (m_cellWidth - 2) * m_subDailyFraction;
        int fullDaysNb = floor(iCol * m_subDailyFraction);
        int subDaysNb = iCol - fullDaysNb / m_subDailyFraction;

        startPointX = fullDaysNb * m_cellWidth + subDaysNb * cellWidth + 1 + cellWidth / 2;
    }

    int halfWidth = cellWidth / (g_ppiScaleDc * 5);
    if (m_subDailyMode) {
        halfWidth = cellWidth / (g_ppiScaleDc * 2);
    }

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint(startPointX - halfWidth, startPointY + markerHeight + outlier - g_ppiScaleDc * 1);
    path.AddLineToPoint(startPointX + halfWidth, startPointY + markerHeight + outlier - g_ppiScaleDc * 1);
    path.AddLineToPoint(startPointX, startPointY);

    path.CloseSubpath();
}