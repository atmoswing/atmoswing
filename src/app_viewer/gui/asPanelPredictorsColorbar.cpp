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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include "asPanelPredictorsColorbar.h"

/**
 * The constructor of the panel for the colorbar (asPanelPredictorsColorbar).
 *
 * @param parent The parent window.
 * @param id The window ID.
 * @param pos The position.
 * @param size The size.
 * @param style The style (flags).
 */
asPanelPredictorsColorbar::asPanelPredictorsColorbar(wxWindow* parent, wxWindowID id, const wxPoint& pos,
                                                     const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style),
      m_valMin(0),
      m_valMax(100),
      m_step(10) {
    m_sizerContent = new wxBoxSizer(wxVERTICAL);
    m_panelDrawing = new asPanelPredictorsColorbarDrawing(this, wxID_ANY, wxDefaultPosition,
                                                          wxSize(-1, 50 * g_ppiScaleDc), wxTAB_TRAVERSAL);
    m_sizerContent->Add(m_panelDrawing, 1, wxEXPAND, 0);

    SetSizer(m_sizerContent);
    Layout();
    m_sizerContent->Fit(this);

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbar::OnPaint), nullptr, this);
}

/**
 * The destructor of the panel for the colorbar (asPanelPredictorsColorbar).
 */
asPanelPredictorsColorbar::~asPanelPredictorsColorbar() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbar::OnPaint), nullptr, this);
}

/**
 * Paint event.
 *
 * @param event The event.
 */
void asPanelPredictorsColorbar::OnPaint(wxPaintEvent& event) {
    m_panelDrawing->DrawColorbar(m_valMin, m_valMax, m_step);
    event.Skip();
}

/**
 * Set the range of the colorbar.
 *
 * @param valMin The minimum value.
 * @param valMax The maximum value.
 */
void asPanelPredictorsColorbar::SetRange(double valMin, double valMax) {
    m_valMin = valMin;
    m_valMax = valMax;
}

/**
 * Set the step of the colorbar.
 *
 * @param step The step for the ticks.
 */
void asPanelPredictorsColorbar::SetStep(double step) {
    m_step = step;
}

/**
 * Set the render.
 *
 * @param render The render.
 */
void asPanelPredictorsColorbar::SetRender(vrRenderRasterPredictor* render) {
    m_panelDrawing->SetRender(render);
}

/**
 * The constructor of the panel for the colorbar drawing (asPanelPredictorsColorbarDrawing).
 *
 * @param parent The parent window.
 * @param id The window ID.
 * @param pos The position.
 * @param size The size.
 * @param style The style (flags).
 */
asPanelPredictorsColorbarDrawing::asPanelPredictorsColorbarDrawing(wxWindow* parent, wxWindowID id, const wxPoint& pos,
                                                                   const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style),
      m_render(nullptr),
      m_bmpColorbar(nullptr),
      m_gdc(nullptr) {
    Layout();

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbarDrawing::OnPaint), nullptr, this);
}

/**
 * The destructor of the panel for the colorbar drawing (asPanelPredictorsColorbarDrawing).
 */
asPanelPredictorsColorbarDrawing::~asPanelPredictorsColorbarDrawing() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbarDrawing::OnPaint), nullptr, this);
    wxDELETE(m_bmpColorbar);
}

/**
 * Draw the colorbar.
 *
 * @param valMin The minimum value.
 * @param valMax The maximum value.
 * @param step The step for the ticks.
 */
void asPanelPredictorsColorbarDrawing::DrawColorbar(double valMin, double valMax, double step) {
    if (!m_render) {
        return;
    }

    wxSize sizePanel = GetSize();

    wxDELETE(m_bmpColorbar);
    m_bmpColorbar = new wxBitmap(sizePanel.x, sizePanel.y);
    wxASSERT(m_bmpColorbar);

    // Create device context
    wxMemoryDC dc(*m_bmpColorbar);
    wxColor bg = GetBackgroundColour();
    dc.SetBackground(bg);
    dc.Clear();

    // Create graphics context
    wxGraphicsContext* gc = wxGraphicsContext::Create(dc);

    if (gc) {
        gc->SetPen(*wxBLACK);
        wxFont defFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        gc->SetFont(defFont, *wxBLACK);
        wxGraphicsPath path = gc->CreatePath();

        CreateColorbarPath(path);
        FillColorbar(gc, path, valMin, valMax);
        CreateColorbarText(gc, path, valMin, valMax, step);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    Refresh();
}

/**
 * The paint event.
 *
 * @param event The event.
 */
void asPanelPredictorsColorbarDrawing::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);

    if (m_bmpColorbar != nullptr) {
        dc.DrawBitmap(*m_bmpColorbar, 0, 0, true);
    }

    Layout();

    event.Skip();
}

/**
 * Creates the colorbar path.
 *
 * @param path The path.
 */
void asPanelPredictorsColorbarDrawing::CreateColorbarPath(wxGraphicsPath& path) {
    wxSize sizePanel = GetSize();

    int startX = 30;
    int endX = sizePanel.x - 30;
    int startY = 0;
    int endY = 15;

    path.MoveToPoint(startX, startY);
    path.AddLineToPoint(endX, startY);
    path.AddLineToPoint(endX, endY);
    path.AddLineToPoint(startX, endY);
    path.CloseSubpath();
}

/**
 * Fills the colorbar.
 *
 * @param gc The graphics context.
 * @param path The path.
 * @param valMin The minimum value.
 * @param valMax The maximum value.
 */
void asPanelPredictorsColorbarDrawing::FillColorbar(wxGraphicsContext* gc, wxGraphicsPath& path, double valMin,
                                                    double valMax) {
    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    wxASSERT(m_render);
    wxImage::RGBValue startColor = m_render->GetColorFromTable(valMin, valMin, valMax - valMin);
    wxImage::RGBValue endColor = m_render->GetColorFromTable(valMax, valMin, valMax - valMin);

    wxGraphicsGradientStops stops(wxColour(startColor.red, startColor.green, startColor.blue),
                                  wxColour(endColor.red, endColor.green, endColor.blue));

    for (int i = 1; i < 256; i++) {
        float ratio = float(i) / 256.0f;
        double val = valMin + ratio * (valMax - valMin);
        wxImage::RGBValue color = m_render->GetColorFromTable(val, valMin, valMax - valMin);
        stops.Add(wxColour(color.red, color.green, color.blue), ratio);
    }

    wxGraphicsBrush brush = gc->CreateLinearGradientBrush(x, y, x + w, y, stops);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

/**
 * Creates the colorbar text.
 *
 * @param gc The graphics context.
 * @param path The path.
 * @param valMin The minimum value.
 * @param valMax The maximum value.
 * @param step The step for the ticks.
 */
void asPanelPredictorsColorbarDrawing::CreateColorbarText(wxGraphicsContext* gc, wxGraphicsPath& path, double valMin,
                                                          double valMax, double step) {
    gc->SetPen(*wxBLACK);

    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    // Set ticks and labels
    double val = step * ceil(valMin / step);
    int dy = 18 * g_ppiScaleDc;
    while (val <= valMax) {
        int xTick = floor(w * (val - valMin) / (valMax - valMin));
        wxGraphicsPath pathTick = gc->CreatePath();
        pathTick.MoveToPoint(x + xTick, y);
        pathTick.AddLineToPoint(x + xTick, y + h + 5);
        gc->StrokePath(pathTick);

        wxString label = asStrF("%g", val);
        gc->DrawText(label, x + xTick + 2, y + dy);

        val += step;
    }
}

/**
 * Sets the render.
 *
 * @param render The render.
 */
void asPanelPredictorsColorbarDrawing::SetRender(vrRenderRasterPredictor* render) {
    m_render = render;
}