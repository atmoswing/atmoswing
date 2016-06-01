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

#ifndef __ASLEADTIMESWITCHER__
#define __ASLEADTIMESWITCHER__

#include "asIncludes.h"
#include "asWorkspace.h"
#include "asForecastManager.h"
#include <wx/graphics.h>
#include <wx/panel.h>
#include <wx/overlay.h>

class asLeadTimeSwitcher
        : public wxPanel
{
public:
    asLeadTimeSwitcher(wxWindow *parent, asWorkspace *workspace, asForecastManager *forecastManager,
                       wxWindowID id = wxID_ANY, const wxPoint &pos = wxDefaultPosition,
                       const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

    ~asLeadTimeSwitcher();

    void Draw(Array1DFloat &dates);

    void SetLeadTime(int leadTime);

    void SetParent(wxWindow *parent);

private:
    int m_cellWidth;
    int m_leadTime;
    asForecastManager *m_forecastManager;
    asWorkspace *m_workspace;
    wxBitmap *m_bmp;
    wxOverlay m_overlay;
    wxGraphicsContext *m_gdc;
    wxWindow *m_parent;

    void OnLeadTimeSlctChange(wxMouseEvent &event);

    void SetBitmap(wxBitmap *bmp);

    void SetLeadTimeMarker(int leadTime);

    wxBitmap *GetBitmap() const;

    void CreatePath(wxGraphicsPath &path, int i_col);

    void CreatePathRing(wxGraphicsPath &path, const wxPoint &center, double scale, int segmentsTotNb, int segmentNb);

    void FillPath(wxGraphicsContext *gc, wxGraphicsPath &path, float value);

    void CreateDatesText(wxGraphicsContext *gc, const wxPoint &start, int i_col, const wxString &label);

    void CreatePathMarker(wxGraphicsPath &path, int i_col);

    void OnPaint(wxPaintEvent &event);

};

#endif // __ASLEADTIMESWITCHER__
