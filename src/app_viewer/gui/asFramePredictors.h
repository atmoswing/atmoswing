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
 * Portions Copyright 2022-2023 Pascal Horton, Terranum.
 */

#ifndef AS_FRAME_PREDICTORS_H
#define AS_FRAME_PREDICTORS_H

#include "AtmoswingViewerGui.h"
#include "asForecastManager.h"
#include "asIncludes.h"
#include "asPredictorsRenderer.h"
#include "vroomgis.h"
#include "wx/dnd.h"


/** Implementing vroomDropFiles */
class asFramePredictors;
class vroomDropFilesPredictors : public wxFileDropTarget {
  private:
    asFramePredictors* m_LoaderFrame;

  public:
    vroomDropFilesPredictors(asFramePredictors* parent);
    virtual bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames);
};

class asFramePredictors : public asFramePredictorsVirtual {
  public:
    asFramePredictors(wxWindow* parent, asForecastManager* forecastManager, asWorkspace* workspace,
                      int selectedForecast, wxWindowID id = asWINDOW_PREDICTORS);

    ~asFramePredictors();

    void Init();

    void InitExtent();

    void OpenDefaultLayers();

  protected:
    wxKeyboardState m_KeyBoardState;

    void OpenFramePreferences(wxCommandEvent& event);

    void OnToolZoomIn(wxCommandEvent& event);

    void OnToolZoomOut(wxCommandEvent& event);

    void OnToolPan(wxCommandEvent& event);

    void OnToolAction(wxCommandEvent& event);

    void OnKeyDown(wxKeyEvent& event);

    void OnKeyUp(wxKeyEvent& event);

  private:
    asForecastManager* m_forecastManager;
    asPredictorsRenderer* m_predictorsViewer;
    asPredictorsManager* m_predictorsManager;
    asWorkspace* m_workspace;

    DECLARE_EVENT_TABLE()
};

#endif
