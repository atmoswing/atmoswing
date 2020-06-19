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

#ifndef AS_PANEL_SIDEBAR_CAPTION_FORECAST_DOTS_H
#define AS_PANEL_SIDEBAR_CAPTION_FORECAST_DOTS_H

#include <wx/graphics.h>

#include "asIncludes.h"
#include "asPanelSidebar.h"

class asPanelSidebarCaptionForecastDotsDrawing : public wxPanel {
   public:
    explicit asPanelSidebarCaptionForecastDotsDrawing(wxWindow *parent, wxWindowID id = wxID_ANY,
                                                      const wxPoint &pos = wxDefaultPosition,
                                                      const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    ~asPanelSidebarCaptionForecastDotsDrawing() override;

    void DrawColorbar(double maxval);

   private:
    wxBitmap *m_bmpColorbar;
    wxGraphicsContext *m_gdc;

    void SetBitmapColorbar(wxBitmap *bmp);

    void CreateColorbarPath(wxGraphicsPath &path);

    void CreateColorbarText(wxGraphicsContext *gc, wxGraphicsPath &path, double valmax);

    void CreateColorbarOtherClasses(wxGraphicsContext *gc, wxGraphicsPath &path);

    void FillColorbar(wxGraphicsContext *gdc, wxGraphicsPath &path);

    void OnPaint(wxPaintEvent &event);
};

class asPanelSidebarCaptionForecastDots : public asPanelSidebar {
   public:
    explicit asPanelSidebarCaptionForecastDots(wxWindow *parent, wxWindowID id = wxID_ANY,
                                               const wxPoint &pos = wxDefaultPosition,
                                               const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    ~asPanelSidebarCaptionForecastDots() override;

    void SetColorbarMax(double maxval);

   private:
    asPanelSidebarCaptionForecastDotsDrawing *m_panelDrawing;

    void OnPaint(wxPaintEvent &event);
};

#endif
