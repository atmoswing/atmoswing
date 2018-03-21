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

    ~asLeadTimeSwitcher() override;

    void Draw(a1f &dates);

    void SetLeadTime(int leadTime);

    void SetParent(wxWindow *parent);

private:
    wxWindow *m_parent;
    asWorkspace *m_workspace;
    asForecastManager *m_forecastManager;
    wxBitmap *m_bmp;
    wxGraphicsContext *m_gdc;
    wxOverlay m_overlay;
    int m_cellWidth;
    int m_leadTime;

    void OnLeadTimeSlctChange(wxMouseEvent &event);

    void SetBitmap(wxBitmap *bmp);

    void SetLeadTimeMarker(int leadTime);

    void CreatePath(wxGraphicsPath &path, int iCol);

    void CreatePathRing(wxGraphicsPath &path, const wxPoint &center, double scale, int segmentsTotNb, int segmentNb);

    void FillPath(wxGraphicsContext *gc, wxGraphicsPath &path, float value);

    void CreateDatesText(wxGraphicsContext *gc, const wxPoint &start, int iCol, const wxString &label);

    void CreatePathMarker(wxGraphicsPath &path, int iCol);

    void OnPaint(wxPaintEvent &event);

};

#endif // __ASLEADTIMESWITCHER__
