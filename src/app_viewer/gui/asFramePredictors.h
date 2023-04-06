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
#include "asPanelPredictorsColorbar.h"
#include "asPredictorsRenderer.h"
#include "vroomgis.h"
#include "wx/dnd.h"

/** Implementing vroomDropFiles */
class asFramePredictors;
class vroomDropFilesPredictors : public wxFileDropTarget {
  public:
    explicit vroomDropFilesPredictors(asFramePredictors* parent);

    bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) override;

  private:
    asFramePredictors* m_LoaderFrame;
};

class asFramePredictors : public asFramePredictorsVirtual {
  public:
    asFramePredictors(wxWindow* parent, asForecastManager* forecastManager, asWorkspace* workspace,
                      int methodRow, int forecastRow, wxWindowID id = asWINDOW_PREDICTORS);

    ~asFramePredictors() override;

    void Init();

    void InitExtent();

    bool OpenLayers(const wxArrayString& names);

    void OpenDefaultLayers();

  protected:
    wxKeyboardState m_KeyBoardState;

    virtual void OnRightClick(wxMouseEvent& event) {
        event.Skip();
    }

  private:
    asForecastManager* m_forecastManager;
    asPredictorsRenderer* m_predictorsRenderer;
    asPredictorsManager* m_predictorsManagerTarget;
    asPredictorsManager* m_predictorsManagerAnalog;
    asWorkspace* m_workspace;
    asPanelPredictorsColorbar* m_panelPredictorsColorbarLeft;
    asPanelPredictorsColorbar* m_panelPredictorsColorbarRight;
    int m_selectedMethod;
    int m_selectedForecast;
    int m_selectedTargetDate;
    int m_selectedAnalogDate;
    int m_selectedPredictor;
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

    void UpdateMethodsList();

    void UpdateForecastList();

    void UpdatePredictorsList();

    void UpdatePredictorsProperties();

    void UpdateTargetDatesList();

    void UpdateAnalogDatesList();

    void OpenFramePreferences(wxCommandEvent& event);

    void OnSwitchRight(wxCommandEvent& event) override;

    void OnSwitchLeft(wxCommandEvent& event) override;

    void OnPredictorSelectionChange(wxCommandEvent& event) override;

    void OnMethodChange(wxCommandEvent& event) override;

    void OnForecastChange(wxCommandEvent& event) override;

    void OnTargetDateChange(wxCommandEvent& event) override;

    void OnAnalogDateChange(wxCommandEvent& event) override;

    void OnOpenLayer(wxCommandEvent& event) override;

    void OnCloseLayer(wxCommandEvent& event) override;

    void OnSyncroToolSwitch(wxCommandEvent& event);

    void OnToolZoomIn(wxCommandEvent& event);

    void OnToolZoomOut(wxCommandEvent& event);

    void OnToolPan(wxCommandEvent& event);

    void OnToolSight(wxCommandEvent& event);

    void OnToolZoomToFit(wxCommandEvent& event);

    void OnToolAction(wxCommandEvent& event);

    void OnKeyDown(wxKeyEvent& event);

    void OnKeyUp(wxKeyEvent& event);

    void UpdateLayers();

    void ReloadViewerLayerManagerLeft();

    void ReloadViewerLayerManagerRight();

    vrRealRect getDesiredExtent() const;

    DECLARE_EVENT_TABLE()
};

#endif
