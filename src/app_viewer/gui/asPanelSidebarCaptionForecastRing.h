/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef __asPanelSidebarCaptionForecastRing__
#define __asPanelSidebarCaptionForecastRing__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include <wx/graphics.h>

/** Implementing asPanelSidebarCaptionForecastRing */
class asPanelSidebarCaptionForecastRing : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarCaptionForecastRing( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarCaptionForecastRing();

    void SetBitmapDates(wxBitmap * bmp);
    void SetBitmapColorbar(wxBitmap * bmp);
    void SetDates(Array1DFloat & dates);
    void SetColorbarMax(double maxval);

private:
    wxPanel *m_PanelDrawing;
    wxBitmap *m_BmpDates;
    wxBitmap *m_BmpColorbar;
    wxGraphicsContext* m_Gdc;
    void DrawDates( Array1DFloat & dates );
    void DrawColorbar( double maxval );
    void CreateDatesPath( wxGraphicsPath & path, const wxPoint & center, double scale, int segmentsTotNb, int segmentNb );
    void CreateDatesText( wxGraphicsContext * gc, const wxPoint & center, double scale, int segmentsTotNb, int segmentNb, const wxString &label);
    void CreateColorbarPath( wxGraphicsPath & path );
    void CreateColorbarText( wxGraphicsContext * gc, wxGraphicsPath & path, double valmax);
    void CreateColorbarOtherClasses(wxGraphicsContext * gc, wxGraphicsPath & path );
    void FillColorbar(wxGraphicsContext * gdc, wxGraphicsPath & path );
    void OnPaint( wxPaintEvent & event );
};

#endif // __asPanelSidebar__
