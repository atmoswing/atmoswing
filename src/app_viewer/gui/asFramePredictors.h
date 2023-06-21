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

#include "AtmoSwingViewerGui.h"
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
    /**
     * A class to handle the drop of files on the frame.
     *
     * @param parent The parent window.
     */
    explicit vroomDropFilesPredictors(asFramePredictors* parent);

    /**
     * Handle the drop of files on the frame.
     *
     * @param x The x coordinate of the drop.
     * @param y The y coordinate of the drop.
     * @param filenames The list of files dropped.
     *
     * @return True if the drop was handled.
     */
    bool OnDropFiles(wxCoord x, wxCoord y, const wxArrayString& filenames) override;

  private:
    asFramePredictors* m_LoaderFrame;
};

class asFramePredictors : public asFramePredictorsVirtual {
  public:
    /**
     * Constructor of the frame to plot predictors.
     *
     * @param parent The parent window.
     * @param forecastManager The forecast manager.
     * @param workspace The workspace.
     * @param methodRow The selected method.
     * @param forecastRow The selected forecast.
     * @param id The window identifier.
     */
    asFramePredictors(wxWindow* parent, asForecastManager* forecastManager, asWorkspace* workspace, int methodRow,
                      int forecastRow, wxWindowID id = asWINDOW_PREDICTORS);

    /**
     * The destructor.
     */
    ~asFramePredictors() override;

    /**
     * Initialize the frame.
     */
    void Init();

    /**
     * Initialize the map extent.
     */
    void InitExtent();

    /**
     * Open a list of layers.
     *
     * @param names Array of layer names to open.
     * @return True if successful.
     */
    bool OpenLayers(const wxArrayString& names);

    /**
     * Open the default maps layers (background).
     */
    void OpenDefaultLayers();

    /**
     * Move the map separator to the right.
     */
    void SwitchPanelRight();

    /**
     * Move the map separator to the left.
     */
    void SwitchPanelLeft();

    /**
     * Get a pointer to the left panel.
     * @return A pointer to the left panel.
     */
    wxPanel* GetPanelRight() {
        return m_panelRight;
    }

    /**
     * Get a pointer to the right panel.
     * @return A pointer to the right panel.
     */
    wxPanel* GetPanelLeft() {
        return m_panelLeft;
    }

    /**
     * Get a pointer to the list of predictors.
     * @return A pointer to the list of predictors.
     */
    wxListBox* GetListPredictors() {
        return m_listPredictors;
    }

  protected:
    wxKeyboardState m_KeyBoardState; /**< Keyboard state. */

    /**
     * Handle the right click on the map.
     * 
     * @param event The mouse event.
     */
    virtual void OnRightClick(wxMouseEvent& event) {
        event.Skip();
    }

  private:
    asForecastManager* m_forecastManager; /**< The forecast manager. */
    asPredictorsRenderer* m_predictorsRenderer; /**< The predictors renderer. */
    asPredictorsManager* m_predictorsManagerTarget; /**< The predictors manager for the target date. */
    asPredictorsManager* m_predictorsManagerAnalog; /**< The predictors manager for the analog date. */
    asWorkspace* m_workspace; /**< The workspace. */
    asPanelPredictorsColorbar* m_panelPredictorsColorbarLeft; /**< The left colorbar. */
    asPanelPredictorsColorbar* m_panelPredictorsColorbarRight; /**< The right colorbar. */
    int m_selectedMethod; /**< The selected method (index). */
    int m_selectedForecast; /**< The selected forecast (index). */
    int m_selectedTargetDate; /**< The selected target date (index). */
    int m_selectedAnalogDate; /**< The selected analog date (index). */
    int m_selectedPredictor; /**< The selected predictor (index). */
    bool m_syncroTool; /**< A flag to indicate if the syncro tool is active. */
    bool m_displayPanelLeft; /**< A flag to indicate if the left panel is displayed. */
    bool m_displayPanelRight; /**< A flag to indicate if the right panel is displayed. */
    wxOverlay m_overlay; /**< The overlay. */
#if defined(__WIN32__)
    wxCriticalSection m_critSectionViewerLayerManager; /**< The critical section for the viewer layer manager. */
#endif

    // Vroomgis
    vrLayerManager* m_layerManager; /**< The layer manager. */
    vrViewerTOCList* m_tocCtrlLeft; /**< The left TOC control. */
    vrViewerTOCList* m_tocCtrlRight; /**< The right TOC control. */
    vrViewerLayerManager* m_viewerLayerManagerLeft; /**< The left viewer layer manager. */
    vrViewerLayerManager* m_viewerLayerManagerRight; /**< The right viewer layer manager. */
    vrViewerDisplay* m_displayCtrlLeft; /**< The left display control. */
    vrViewerDisplay* m_displayCtrlRight; /**< The right display control. */

    /**
     * Update the methods list.
     */
    void UpdateMethodsList();

    /**
     * Update the forecasts list.
     */
    void UpdateForecastList();

    /**
     * Update the available predictors list.
     */
    void UpdatePredictorsList();

    /**
     * Update the predictors properties.
     */
    void UpdatePredictorsProperties();

    /**
     * Update the target dates list.
     */
    void UpdateTargetDatesList();

    /**
     * Update the analog dates list.
     */
    void UpdateAnalogDatesList();

    /**
     * Open the preferences frame.
     *
     * @param event The command event.
     */
    void OpenFramePreferences(wxCommandEvent& event);

    /**
     * Move the map separator to the right.
     *
     * @param event The command event.
     */
    void OnSwitchRight(wxCommandEvent& event) override;

    /**
     * Move the map separator to the left.
     *
     * @param event The command event.
     */
    void OnSwitchLeft(wxCommandEvent& event) override;

    /**
     * Update the map when the predictor selection changes.
     *
     * @param event The command event.
     */
    void OnPredictorSelectionChange(wxCommandEvent& event) override;

    /**
     * Update the map and the list of forecasts when the method changes.
     *
     * @param event The command event.
     */
    void OnMethodChange(wxCommandEvent& event) override;

    /**
     * Update the map and the list of target dates when the forecast changes.
     *
     * @param event The command event.
     */
    void OnForecastChange(wxCommandEvent& event) override;

    /**
     * Update the map and the analog dates when the target date changes.
     *
     * @param event The command event.
     */
    void OnTargetDateChange(wxCommandEvent& event) override;

    /**
     * Update the map when the analog date changes.
     *
     * @param event The command event.
     */
    void OnAnalogDateChange(wxCommandEvent& event) override;

    /**
     * Open a dialog to select a layer.
     *
     * @param event The command event.
     */
    void OnOpenLayer(wxCommandEvent& event) override;

    /**
     * Open a dialog to select a layer to close.
     *
     * @param event The command event.
     */
    void OnCloseLayer(wxCommandEvent& event) override;

    /**
     * Activate or deactivates the syncro mode between the two maps.
     *
     * @param event The command event.
     */
    void OnSyncroToolSwitch(wxCommandEvent& event);

    /**
     * Set the zoom in tool.
     *
     * @param event The command event.
     */
    void OnToolZoomIn(wxCommandEvent& event);

    /**
     * Set the zoom out tool.
     *
     * @param event The command event.
     */
    void OnToolZoomOut(wxCommandEvent& event);

    /**
     * Set the pan tool.
     *
     * @param event The command event.
     */
    void OnToolPan(wxCommandEvent& event);

    /**
     * Set the sight tool.
     *
     * @param event The command event.
     */
    void OnToolSight(wxCommandEvent& event);

    /**
     * Handle the zoom to fit event.
     *
     * @param event The command event.
     */
    void OnToolZoomToFit(wxCommandEvent& event);

    /**
     * Handle the different tool actions (zoom, pan, etc).
     *
     * @param event The command event.
     */
    void OnToolAction(wxCommandEvent& event);

    /**
     * Key down event to handle the zoom in and out.
     *
     * @param event The key event.
     */
    void OnKeyDown(wxKeyEvent& event);

    /**
     * Key up event to handle the zoom in and out.
     *
     * @param event The key event.
     */
    void OnKeyUp(wxKeyEvent& event);

    /**
     * Set target and analog dates in lists and refreshes the map.
     */
    void UpdateLayers();

    /**
     * Get the mean coordinates of the stations in WGS84.
     *
     * @param forecast The forecast object.
     * @return The mean coordinates of the stations in WGS84.
     */
    Coo GetStationsMeanCoordinatesWgs84(asResultsForecast* forecast);

    /**
     * Reload the left viewer layer manager.
     */
    void ReloadViewerLayerManagerLeft();

    /**
     * Reload the right viewer layer manager.
     */
    void ReloadViewerLayerManagerRight();

    /**
     * Get the desired extent for the map.
     *
     * @return The desired extent.
     */
    vrRealRect GetDesiredExtent() const;

    DECLARE_EVENT_TABLE()
};

#endif
