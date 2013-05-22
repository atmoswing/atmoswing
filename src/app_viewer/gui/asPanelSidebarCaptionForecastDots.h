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
 
#ifndef __asPanelSidebarCaptionForecastDots__
#define __asPanelSidebarCaptionForecastDots__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include <wx/graphics.h>

/** Implementing asPanelSidebarCaptionForecastDots */
class asPanelSidebarCaptionForecastDots : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarCaptionForecastDots( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarCaptionForecastDots();

    void SetBitmapColorbar(wxBitmap * bmp);
    void SetColorbarMax(double maxval);

private:
    wxPanel *m_PanelDrawing;
    wxBitmap *m_BmpColorbar;
    wxGraphicsContext* m_Gdc;
    void DrawColorbar( double maxval );
    void CreateColorbarPath( wxGraphicsPath & path );
    void CreateColorbarText( wxGraphicsContext * gc, wxGraphicsPath & path, double valmax);
    void CreateColorbarOtherClasses(wxGraphicsContext * gc, wxGraphicsPath & path );
    void FillColorbar(wxGraphicsContext * gdc, wxGraphicsPath & path );
    void OnPaint( wxPaintEvent & event );
};

#endif // __asPanelSidebar__
