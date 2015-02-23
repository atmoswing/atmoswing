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
 
#ifndef __asPanelSidebarAlarms__
#define __asPanelSidebarAlarms__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include "asForecastManager.h"
#include "asWorkspace.h"
#include <wx/graphics.h>

class asPanelSidebarAlarms; // predefinition

/** Implementing asPanelSidebarAlarmsDrawing */
class asPanelSidebarAlarmsDrawing : public wxPanel
{
public:
    /** Constructor */
    asPanelSidebarAlarmsDrawing( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarAlarmsDrawing();
    
    void DrawAlarms( Array1DFloat &dates, const VectorString &forecasts, Array2DFloat &values );
    void SetParent( asPanelSidebarAlarms* parent );

private:
    wxBitmap *m_BmpAlarms;
    wxGraphicsContext* m_Gdc;
    asPanelSidebarAlarms* m_Parent;
    void SetBitmapAlarms(wxBitmap *bmp);
    void CreatePath(wxGraphicsPath &path, const wxPoint &start, int witdh, int height, int i_col, int i_row, int cols, int rows);
    void FillPath( wxGraphicsContext *gc, wxGraphicsPath & path, float value );
    void CreateDatesText( wxGraphicsContext *gc, const wxPoint& start, int cellWitdh, int i_col, const wxString &label);
    void CreateNbText( wxGraphicsContext *gc, const wxPoint& start, int cellHeight, int i_row, const wxString &label);
    void OnPaint( wxPaintEvent &event );
};

/** Implementing asPanelSidebarAlarms */
class asPanelSidebarAlarms : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarAlarms( wxWindow* parent, asWorkspace* workspace, asForecastManager * forecastManager, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarAlarms();

    void SetData( Array1DFloat &dates, Array2DFloat &values );
    void Update();
    int GetMode()
    {
        return m_Mode;
    }

private:
    asWorkspace* m_Workspace;
    asForecastManager* m_ForecastManager;
    asPanelSidebarAlarmsDrawing *m_PanelDrawing;
    int m_Mode;
    void OnPaint( wxPaintEvent &event );
};

#endif // __asPanelSidebar__
