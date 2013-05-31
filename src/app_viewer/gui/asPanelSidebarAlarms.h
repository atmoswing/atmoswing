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
 
#ifndef __asPanelSidebarAlarms__
#define __asPanelSidebarAlarms__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include "asResultsAnalogsForecast.h"
#include <wx/graphics.h>

/** Implementing asPanelSidebarAlarms */
class asPanelSidebarAlarms : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarAlarms( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarAlarms();

    void SetData( Array1DFloat &dates, const VectorString &models, Array2DFloat &values );
    void SetBitmapAlarms(wxBitmap *bmp);
    void UpdateAlarms(Array1DFloat &dates, VectorString &models, std::vector <asResultsAnalogsForecast*> forecasts);

private:
    wxPanel *m_PanelDrawing;
    wxBitmap *m_BmpAlarms;
    wxGraphicsContext* m_Gdc;
    int m_Mode;
    void DrawAlarms( Array1DFloat &dates, const VectorString &models, Array2DFloat &values );
    void CreatePath(wxGraphicsPath &path, const wxPoint &start, int witdh, int height, int i_col, int i_row, int cols, int rows);
    void FillPath( wxGraphicsContext *gc, wxGraphicsPath & path, float value );
    void CreateDatesText( wxGraphicsContext *gc, const wxPoint& start, int cellWitdh, int i_col, const wxString &label);
    void CreateNbText( wxGraphicsContext *gc, const wxPoint& start, int cellHeight, int i_row, const wxString &label);
    void OnPaint( wxPaintEvent &event );
};

#endif // __asPanelSidebar__
