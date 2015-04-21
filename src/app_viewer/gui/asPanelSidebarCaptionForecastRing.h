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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef __asPanelSidebarCaptionForecastRing__
#define __asPanelSidebarCaptionForecastRing__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include <wx/graphics.h>


/** Implementing asPanelSidebarCaptionForecastRingDrawing */
class asPanelSidebarCaptionForecastRingDrawing : public wxPanel
{
public:
    /** Constructor */
    asPanelSidebarCaptionForecastRingDrawing( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarCaptionForecastRingDrawing();

    void DrawDates( Array1DFloat & dates );
    void DrawColorbar( double maxval );

private:
    wxBitmap *m_bmpDates;
    wxBitmap *m_bmpColorbar;
    wxGraphicsContext* m_gdc;
    void SetBitmapDates(wxBitmap * bmp);
    void SetBitmapColorbar(wxBitmap * bmp);
    void CreateDatesPath( wxGraphicsPath & path, const wxPoint & center, double scale, int segmentsTotNb, int segmentNb );
    void CreateDatesText( wxGraphicsContext * gc, const wxPoint & center, double scale, int segmentsTotNb, int segmentNb, const wxString &label);
    void CreateColorbarPath( wxGraphicsPath & path );
    void CreateColorbarText( wxGraphicsContext * gc, wxGraphicsPath & path, double valmax);
    void CreateColorbarOtherClasses(wxGraphicsContext * gc, wxGraphicsPath & path );
    void FillColorbar(wxGraphicsContext * gdc, wxGraphicsPath & path );
    void OnPaint( wxPaintEvent & event );
};


/** Implementing asPanelSidebarCaptionForecastRing */
class asPanelSidebarCaptionForecastRing : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarCaptionForecastRing( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarCaptionForecastRing();

    void SetDates(Array1DFloat & dates);
    void SetColorbarMax(double maxval);

private:
    asPanelSidebarCaptionForecastRingDrawing *m_panelDrawing;
    void OnPaint( wxPaintEvent & event );
};


#endif // __asPanelSidebar__
