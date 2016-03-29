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

#include "asPanelSidebarAlarms.h"

#include "images.h"

/*
 * asPanelSidebarAlarms
 */

asPanelSidebarAlarms::asPanelSidebarAlarms(wxWindow *parent, asWorkspace *workspace, asForecastManager *forecastManager,
                                           wxWindowID id, const wxPoint &pos, const wxSize &size, long style)
        : asPanelSidebar(parent, id, pos, size, style)
{
    m_workspace = workspace;
    m_forecastManager = forecastManager;

    m_header->SetLabelText(_("Alarms"));

    m_mode = 1; // 1: values
    // 2: thresholds -> not available yet

    m_panelDrawing = NULL;

    m_sizerContent->Fit(this);

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarAlarms::OnPaint), NULL, this);

    Layout();
    m_sizerMain->Fit(this);
}

asPanelSidebarAlarms::~asPanelSidebarAlarms()
{
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarAlarms::OnPaint), NULL, this);
}

void asPanelSidebarAlarms::OnPaint(wxPaintEvent &event)
{
    event.Skip();
}

void asPanelSidebarAlarms::Update()
{
    int returnPeriodRef = m_workspace->GetAlarmsPanelReturnPeriod();
    float quantileThreshold = m_workspace->GetAlarmsPanelQuantile();

    m_header->SetLabelText(wxString::Format(_("Alarms (T=%d, q=%g)"), returnPeriodRef, quantileThreshold));

    Array1DFloat dates = m_forecastManager->GetFullTargetDates();

    switch (m_mode) {
        case (1): {
            wxASSERT(returnPeriodRef >= 2);
            wxASSERT(quantileThreshold > 0);
            wxASSERT(quantileThreshold < 1);
            if (returnPeriodRef < 2)
                returnPeriodRef = 2;
            if (quantileThreshold <= 0)
                quantileThreshold = (float) 0.9;
            if (quantileThreshold > 1)
                quantileThreshold = (float) 0.9;

            Array2DFloat values = Array2DFloat::Ones(m_forecastManager->GetMethodsNb(), dates.size());
            values *= NaNFloat;

            for (int methodRow = 0; methodRow < m_forecastManager->GetMethodsNb(); methodRow++) {

                Array1DFloat methodMaxValues = m_forecastManager->GetAggregator()->GetMethodMaxValues(dates, methodRow,
                                                                                                      returnPeriodRef,
                                                                                                      quantileThreshold);
                values.row(methodRow) = methodMaxValues;
            }

            SetData(dates, values);
            break;
        }

        case (2): {
            // Not yet implemented
        }
    }
}

void asPanelSidebarAlarms::SetData(Array1DFloat &dates, Array2DFloat &values)
{
    VectorString names = m_forecastManager->GetAllMethodNames();

    // Required size
    int rows = values.rows();
    int cellHeight = 20 * g_ppiScaleDc;
    int totHeight = cellHeight * rows + cellHeight;
    int width = 240 * g_ppiScaleDc;

    // Delete and recreate the panel. Cannot get it work with a resize...
    wxDELETE(m_panelDrawing);
    m_panelDrawing = new asPanelSidebarAlarmsDrawing(this, wxID_ANY, wxDefaultPosition, wxSize(width, totHeight),
                                                     wxTAB_TRAVERSAL);
    m_panelDrawing->SetParent(this);
    m_panelDrawing->Layout();
    m_panelDrawing->DrawAlarms(dates, names, values);

    m_sizerContent->Add(m_panelDrawing, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);
    m_sizerContent->Fit(this);

    GetParent()->FitInside();
}

/*
 * asPanelSidebarAlarmsDrawing
 */

asPanelSidebarAlarmsDrawing::asPanelSidebarAlarmsDrawing(wxWindow *parent, wxWindowID id, const wxPoint &pos,
                                                         const wxSize &size, long style)
        : wxPanel(parent, id, pos, size, style)
{
    m_bmpAlarms = NULL;
    m_gdc = NULL;

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarAlarmsDrawing::OnPaint), NULL, this);

    Layout();
}

asPanelSidebarAlarmsDrawing::~asPanelSidebarAlarmsDrawing()
{
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelSidebarAlarmsDrawing::OnPaint), NULL, this);
    wxDELETE(m_bmpAlarms);
}

void asPanelSidebarAlarmsDrawing::SetParent(asPanelSidebarAlarms *parent)
{
    m_parent = parent;
}

void asPanelSidebarAlarmsDrawing::DrawAlarms(Array1DFloat &dates, const VectorString &names, Array2DFloat &values)
{
    // Get sizes
    int cols = dates.size();
    int rows = names.size();
    wxASSERT_MSG((values.cols() == cols), wxString::Format("values.cols()=%d, cols=%d", (int) values.cols(), cols));
    wxASSERT_MSG((values.rows() == rows), wxString::Format("values.rows()=%d, rows=%d", (int) values.rows(), rows));

    // Height of a grid row
    int cellHeight = 20 * g_ppiScaleDc;
    int width = 240 * g_ppiScaleDc;

    // Create bitmap
    int totHeight = cellHeight * rows + cellHeight;
    wxBitmap *bmp = new wxBitmap(width, totHeight);
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

    if (gc && cols > 0 && rows > 0) {
        gc->SetPen(*wxBLACK);
        wxFont datesFont(6, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        wxFont numFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(datesFont, *wxBLACK);

        wxPoint startText(17 * g_ppiScaleDc, 0);
        wxPoint startNb(0, 14 * g_ppiScaleDc);
        wxPoint startGrid(12 * g_ppiScaleDc, 10 * g_ppiScaleDc);
        int cellWitdh = (226 * g_ppiScaleDc) / dates.size();

        for (int i_leadtime = 0; i_leadtime < dates.size(); i_leadtime++) {
            wxString dateStr = asTime::GetStringTime(dates[i_leadtime], "DD.MM");
            gc->SetFont(datesFont, *wxBLACK);
            CreateDatesText(gc, startText, cellWitdh, i_leadtime, dateStr);

            for (int i_forecast = 0; (unsigned) i_forecast < names.size(); i_forecast++) {
                if (i_leadtime == 0) {
                    wxString forecastStr = wxString::Format("%d", i_forecast + 1);
                    gc->SetFont(numFont, *wxBLACK);
                    CreateNbText(gc, startNb, cellHeight, i_forecast, forecastStr);
                }

                wxGraphicsPath path = gc->CreatePath();
                CreatePath(path, startGrid, cellWitdh, cellHeight, i_leadtime, i_forecast, cols, rows);
                float value = values(i_forecast, i_leadtime);
                FillPath(gc, path, value);
            }
        }

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmapAlarms(bmp);
    wxDELETE(bmp);
    wxASSERT(!bmp);

    Refresh();
}

void asPanelSidebarAlarmsDrawing::SetBitmapAlarms(wxBitmap *bmp)
{
    wxDELETE(m_bmpAlarms);
    wxASSERT(!m_bmpAlarms);

    if (bmp != NULL) {
        wxASSERT(bmp);
        m_bmpAlarms = new wxBitmap(*bmp);
        wxASSERT(m_bmpAlarms);
    }
}

void asPanelSidebarAlarmsDrawing::OnPaint(wxPaintEvent &event)
{
    if (m_bmpAlarms != NULL) {
        wxPaintDC dc(this);
        dc.DrawBitmap(*m_bmpAlarms, 0, 0, true);
    }

    Layout();

    event.Skip();
}

void asPanelSidebarAlarmsDrawing::CreatePath(wxGraphicsPath &path, const wxPoint &start, int cellWitdh, int cellHeight,
                                             int i_col, int i_row, int cols, int rows)
{
    double startPointX = (double) start.x + i_col * cellWitdh;

    double startPointY = (double) start.y + i_row * cellHeight;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint(startPointX + cellWitdh, startPointY);
    path.AddLineToPoint(startPointX + cellWitdh, startPointY + cellHeight);
    path.AddLineToPoint(startPointX, startPointY + cellHeight);
    path.AddLineToPoint(startPointX, startPointY);

    path.CloseSubpath();
}

void asPanelSidebarAlarmsDrawing::FillPath(wxGraphicsContext *gc, wxGraphicsPath &path, float value)
{
    wxColour colour;

    switch (m_parent->GetMode()) {
        case (1): {
            if (asTools::IsNaN(value)) // NaN -> gray
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
            break;
        }

        case (2): {
            if (value == 1) // Green
            {
                colour.Set(200, 255, 200);
            } else if (value == 2) // Yellow
            {
                colour.Set(255, 255, 118);
            } else if (value == 3) // Red
            {
                colour.Set(255, 80, 80);
            } else {
                colour.Set(150, 150, 150);
            }
            break;
        }
    }

    wxBrush brush(colour, wxSOLID);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asPanelSidebarAlarmsDrawing::CreateDatesText(wxGraphicsContext *gc, const wxPoint &start, int cellWitdh, int i_col,
                                                  const wxString &label)
{
    double pointX = (double) start.x + i_col * cellWitdh;
    double pointY = (double) start.y;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}

void asPanelSidebarAlarmsDrawing::CreateNbText(wxGraphicsContext *gc, const wxPoint &start, int cellHeight, int i_row,
                                               const wxString &label)
{
    double pointX = (double) start.x;
    double pointY = (double) start.y + i_row * cellHeight;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}
