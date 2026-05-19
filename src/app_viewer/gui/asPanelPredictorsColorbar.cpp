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

asPanelPredictorsColorbar::asPanelPredictorsColorbar(wxWindow* parent, wxWindowID id, const wxPoint& pos,
                                                     const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style),
      _valMin(0),
      _valMax(100),
      _step(10) {
    _sizerContent = new wxBoxSizer(wxVERTICAL);
    _panelDrawing = new asPanelPredictorsColorbarDrawing(this, wxID_ANY, wxDefaultPosition,
                                                          wxSize(-1, 50 * g_ppiScaleDc), wxTAB_TRAVERSAL);
    _sizerContent->Add(_panelDrawing, 1, wxEXPAND, 0);

    SetSizer(_sizerContent);
    Layout();
    _sizerContent->Fit(this);

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbar::OnPaint), nullptr, this);
}

asPanelPredictorsColorbar::~asPanelPredictorsColorbar() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbar::OnPaint), nullptr, this);
}

void asPanelPredictorsColorbar::OnPaint(wxPaintEvent& event) {
    _panelDrawing->DrawColorbar(_valMin, _valMax, _step);
    event.Skip();
}

void asPanelPredictorsColorbar::SetRange(double valMin, double valMax) {
    _valMin = valMin;
    _valMax = valMax;
}

void asPanelPredictorsColorbar::SetStep(double step) {
    _step = step;
}

void asPanelPredictorsColorbar::SetRender(vrRenderRasterPredictor* render) {
    _panelDrawing->SetRender(render);
}

asPanelPredictorsColorbarDrawing::asPanelPredictorsColorbarDrawing(wxWindow* parent, wxWindowID id, const wxPoint& pos,
                                                                   const wxSize& size, long style)
    : wxPanel(parent, id, pos, size, style),
      _bmpColorbar(nullptr),
      _gdc(nullptr),
      _render(nullptr) {
    Layout();

    Connect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbarDrawing::OnPaint), nullptr, this);
}

asPanelPredictorsColorbarDrawing::~asPanelPredictorsColorbarDrawing() {
    Disconnect(wxEVT_PAINT, wxPaintEventHandler(asPanelPredictorsColorbarDrawing::OnPaint), nullptr, this);
    wxDELETE(_bmpColorbar);
}

void asPanelPredictorsColorbarDrawing::DrawColorbar(double valMin, double valMax, double step) {
    if (!_render) {
        return;
    }

    wxSize sizePanel = GetSize();

    wxDELETE(_bmpColorbar);
    _bmpColorbar = new wxBitmap(sizePanel.x, sizePanel.y);
    wxASSERT(_bmpColorbar);

    // Create device context
    wxMemoryDC dc(*_bmpColorbar);
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

void asPanelPredictorsColorbarDrawing::OnPaint(wxPaintEvent& event) {
    wxPaintDC dc(this);

    if (_bmpColorbar != nullptr) {
        dc.DrawBitmap(*_bmpColorbar, 0, 0, true);
    }

    Layout();

    event.Skip();
}

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

void asPanelPredictorsColorbarDrawing::FillColorbar(wxGraphicsContext* gc, wxGraphicsPath& path, double valMin,
                                                    double valMax) {
    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    wxASSERT(_render);
    wxImage::RGBValue startColor = _render->GetColorFromTable(valMin, valMin, valMax - valMin);
    wxImage::RGBValue endColor = _render->GetColorFromTable(valMax, valMin, valMax - valMin);

    wxGraphicsGradientStops stops(wxColour(startColor.red, startColor.green, startColor.blue),
                                  wxColour(endColor.red, endColor.green, endColor.blue));

    for (int i = 1; i < 256; i++) {
        float ratio = float(i) / 256.0f;
        double val = valMin + ratio * (valMax - valMin);
        wxImage::RGBValue color = _render->GetColorFromTable(val, valMin, valMax - valMin);
        stops.Add(wxColour(color.red, color.green, color.blue), ratio);
    }

    wxGraphicsBrush brush = gc->CreateLinearGradientBrush(x, y, x + w, y, stops);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

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

void asPanelPredictorsColorbarDrawing::SetRender(vrRenderRasterPredictor* render) {
    _render = render;
}