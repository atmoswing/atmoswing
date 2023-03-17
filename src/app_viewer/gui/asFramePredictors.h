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
                      int methodRow, int forecastRow, wxWindowID id = asWINDOW_PREDICTORS);

    ~asFramePredictors();

    void Init();

    void InitExtent();

    bool OpenLayers(const wxArrayString& names);

    void OpenDefaultLayers();

  protected:
    wxKeyboardState m_KeyBoardState;

    void OpenFramePreferences(wxCommandEvent& event);

    void OnSwitchRight(wxCommandEvent& event);

    void OnSwitchLeft(wxCommandEvent& event);

    void OnPredictorSelectionChange(wxCommandEvent& event);

    void OnMethodChange(wxCommandEvent& event);

    void OnForecastChange(wxCommandEvent& event);

    void OnTargetDateChange(wxCommandEvent& event);

    void OnAnalogDateChange(wxCommandEvent& event);

    void OnOpenLayer(wxCommandEvent& event);

    void OnCloseLayer(wxCommandEvent& event);

    void OnToolZoomIn(wxCommandEvent& event);

    void OnToolZoomOut(wxCommandEvent& event);

    void OnToolPan(wxCommandEvent& event);

    void OnToolSight(wxCommandEvent& event);

    void OnToolAction(wxCommandEvent& event);

    void OnKeyDown(wxKeyEvent& event);

    void OnKeyUp(wxKeyEvent& event);

    void UpdateLayers();

    void ReloadViewerLayerManagerLeft();

    void ReloadViewerLayerManagerRight();

  private:
    asForecastManager* m_forecastManager;
    asPredictorsRenderer* m_predictorsViewer;
    asPredictorsManager* m_predictorsManager;
    asWorkspace* m_workspace;
    int m_selectedMethod;
    int m_selectedForecast;
    int m_selectedTargetDate;
    int m_selectedAnalogDate;
    bool m_syncroTool;
    bool m_displayPanelLeft;
    bool m_displayPanelRight;
    wxOverlay m_overlay;
#if defined(__WIN32__)
    wxCriticalSection m_critSectionViewerLayerManager;
#endif

    // Vroomgis
    vrLayerManager* m_layerManager;
    vrViewerTOCList* m_tocCtrlLeft;
    vrViewerTOCList* m_tocCtrlRight;
    vrViewerLayerManager* m_viewerLayerManagerLeft;
    vrViewerLayerManager* m_viewerLayerManagerRight;
    vrViewerDisplay* m_displayCtrlLeft;
    vrViewerDisplay* m_displayCtrlRight;

    DECLARE_EVENT_TABLE()
};

#endif
