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

#ifndef AS_PANEL_SIDEBAR_ALARMS_H
#define AS_PANEL_SIDEBAR_ALARMS_H

#include <wx/graphics.h>

#include "asForecastManager.h"
#include "asIncludes.h"
#include "asPanelSidebar.h"
#include "asWorkspace.h"

class asPanelSidebarAlarms;

class asPanelSidebarAlarmsDrawing : public wxPanel {
 public:
  asPanelSidebarAlarmsDrawing(wxWindow *parent, wxWindowID id = wxID_ANY, const wxPoint &pos = wxDefaultPosition,
                              const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

  ~asPanelSidebarAlarmsDrawing() override;

  void DrawAlarms(a1f &dates, const vwxs &forecasts, a2f &values);

  void SetParent(asPanelSidebarAlarms *parent);

 private:
  wxBitmap *m_bmpAlarms;
  wxGraphicsContext *m_gdc;
  asPanelSidebarAlarms *m_parent;

  void SetBitmapAlarms(wxBitmap *bmp);

  void CreatePath(wxGraphicsPath &path, const wxPoint &start, int witdh, int height, int iCol, int iRow);

  void FillPath(wxGraphicsContext *gc, wxGraphicsPath &path, float value);

  void CreateDatesText(wxGraphicsContext *gc, const wxPoint &start, int cellWitdh, int iCol, const wxString &label);

  void CreateNbText(wxGraphicsContext *gc, const wxPoint &start, int cellHeight, int iRow, const wxString &label);

  void OnPaint(wxPaintEvent &event);
};

class asPanelSidebarAlarms : public asPanelSidebar {
 public:
  asPanelSidebarAlarms(wxWindow *parent, asWorkspace *workspace, asForecastManager *forecastManager,
                       wxWindowID id = wxID_ANY, const wxPoint &pos = wxDefaultPosition,
                       const wxSize &size = wxDefaultSize, long style = wxTAB_TRAVERSAL);

  ~asPanelSidebarAlarms() override;

  void SetData(a1f &dates, a2f &values);

  void Update() override;

  int GetMode() {
    return m_mode;
  }

 private:
  asWorkspace *m_workspace;
  asForecastManager *m_forecastManager;
  asPanelSidebarAlarmsDrawing *m_panelDrawing;
  int m_mode;

  void OnPaint(wxPaintEvent &event);
};

#endif
