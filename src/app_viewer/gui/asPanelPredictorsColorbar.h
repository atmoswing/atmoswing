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

#ifndef AS_PANEL_PREDICTORS_COLORBAR_H
#define AS_PANEL_PREDICTORS_COLORBAR_H

#include <wx/graphics.h>

#include "asIncludes.h"
#include "vrRenderRasterPredictor.h"

class asPanelPredictorsColorbarDrawing : public wxPanel {
  public:
    /**
     * The constructor of the panel for the colorbar drawing (asPanelPredictorsColorbarDrawing).
     *
     * @param parent The parent window.
     * @param id The window ID.
     * @param pos The position.
     * @param size The size.
     * @param style The style (flags).
     */
    explicit asPanelPredictorsColorbarDrawing(wxWindow* parent, wxWindowID id = wxID_ANY,
                                              const wxPoint& pos = wxDefaultPosition,
                                              const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    /**
     * The destructor of the panel for the colorbar drawing (asPanelPredictorsColorbarDrawing).
     */
    ~asPanelPredictorsColorbarDrawing() override;

    /**
     * Draw the colorbar.
     *
     * @param valMin The minimum value.
     * @param valMax The maximum value.
     * @param step The step for the ticks.
     */
    void DrawColorbar(double valMin, double valMax, double step);

    /**
     * Sets the render.
     *
     * @param render The render.
     */
    void SetRender(vrRenderRasterPredictor* render);

  private:
    wxBitmap* m_bmpColorbar;
    wxGraphicsContext* m_gdc;
    vrRenderRasterPredictor* m_render;

    /**
     * Creates the colorbar path.
     *
     * @param path The path.
     */
    void CreateColorbarPath(wxGraphicsPath& path);

    /**
     * Creates the colorbar text.
     *
     * @param gc The graphics context.
     * @param path The path.
     * @param valMin The minimum value.
     * @param valMax The maximum value.
     * @param step The step for the ticks.
     */
    void CreateColorbarText(wxGraphicsContext* gc, wxGraphicsPath& path, double valMin, double valMax, double step);

    /**
     * Fills the colorbar.
     *
     * @param gc The graphics context.
     * @param path The path.
     * @param valMin The minimum value.
     * @param valMax The maximum value.
     */
    void FillColorbar(wxGraphicsContext* gdc, wxGraphicsPath& path, double valMin, double valMax);

    /**
     * The paint event.
     *
     * @param event The event.
     */
    void OnPaint(wxPaintEvent& event);
};

class asPanelPredictorsColorbar : public wxPanel {
  public:
    /**
     * The constructor of the panel for the colorbar (asPanelPredictorsColorbar).
     *
     * @param parent The parent window.
     * @param id The window ID.
     * @param pos The position.
     * @param size The size.
     * @param style The style (flags).
     */
    explicit asPanelPredictorsColorbar(wxWindow* parent, wxWindowID id = wxID_ANY,
                                       const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize,
                                       long style = wxTAB_TRAVERSAL);

    /**
     * The destructor of the panel for the colorbar (asPanelPredictorsColorbar).
     */
    ~asPanelPredictorsColorbar() override;

    /**
     * Set the range of the colorbar.
     *
     * @param valMin The minimum value.
     * @param valMax The maximum value.
     */
    void SetRange(double valMin, double valMax);

    /**
     * Set the step of the colorbar.
     *
     * @param step The step for the ticks.
     */
    void SetStep(double step);

    /**
     * Set the render.
     *
     * @param render The render.
     */
    void SetRender(vrRenderRasterPredictor* render);

  private:
    wxBoxSizer* m_sizerContent;
    asPanelPredictorsColorbarDrawing* m_panelDrawing;
    double m_valMin;
    double m_valMax;
    double m_step;

    /**
     * Paint event.
     *
     * @param event The event.
     */
    void OnPaint(wxPaintEvent& event);
};

#endif
