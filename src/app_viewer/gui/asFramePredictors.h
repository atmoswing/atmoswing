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
     * Initializes the frame.
     */
    void Init();

    /**
     * Initializes the map extent.
     */
    void InitExtent();

    /**
     * Opens a list of layers.
     *
     * @param names Array of layer names to open.
     * @return True if successful.
     */
    bool OpenLayers(const wxArrayString& names);

    /**
     * Opens the default maps layers (background).
     */
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

    /**
     * Updates the methods list.
     */
    void UpdateMethodsList();

    /**
     * Updates the forecasts list.
     */
    void UpdateForecastList();

    /**
     * Updates the available predictors list.
     */
    void UpdatePredictorsList();

    /**
     * Updates the predictors properties.
     */
    void UpdatePredictorsProperties();

    /**
     * Updates the target dates list.
     */
    void UpdateTargetDatesList();

    /**
     * Updates the analog dates list.
     */
    void UpdateAnalogDatesList();

    /**
     * Opens the preferences frame.
     *
     * @param event The command event.
     */
    void OpenFramePreferences(wxCommandEvent& event);

    /**
     * Moves the map separator to the right.
     *
     * @param event The command event.
     */
    void OnSwitchRight(wxCommandEvent& event) override;

    /**
     * Moves the map separator to the left.
     *
     * @param event The command event.
     */
    void OnSwitchLeft(wxCommandEvent& event) override;

    /**
     * Updates the map when the predictor selection changes.
     *
     * @param event The command event.
     */
    void OnPredictorSelectionChange(wxCommandEvent& event) override;

    /**
     * Updates the map and the list of forecasts when the method changes.
     *
     * @param event The command event.
     */
    void OnMethodChange(wxCommandEvent& event) override;

    /**
     * Updates the map and the list of target dates when the forecast changes.
     *
     * @param event The command event.
     */
    void OnForecastChange(wxCommandEvent& event) override;

    /**
     * Updates the map and the analog dates when the target date changes.
     *
     * @param event The command event.
     */
    void OnTargetDateChange(wxCommandEvent& event) override;

    /**
     * Updates the map when the analog date changes.
     *
     * @param event The command event.
     */
    void OnAnalogDateChange(wxCommandEvent& event) override;

    /**
     * Opens a dialog to select a layer.
     *
     * @param event The command event.
     */
    void OnOpenLayer(wxCommandEvent& event) override;

    /**
     * Opens a dialog to select a layer to close.
     *
     * @param event The command event.
     */
    void OnCloseLayer(wxCommandEvent& event) override;

    /**
     * Activates or deactivates the syncro mode between the two maps.
     *
     * @param event The command event.
     */
    void OnSyncroToolSwitch(wxCommandEvent& event);

    /**
     * Sets the zoom in tool.
     *
     * @param event The command event.
     */
    void OnToolZoomIn(wxCommandEvent& event);

    /**
     * Sets the zoom out tool.
     *
     * @param event The command event.
     */
    void OnToolZoomOut(wxCommandEvent& event);

    /**
     * Sets the pan tool.
     *
     * @param event The command event.
     */
    void OnToolPan(wxCommandEvent& event);

    /**
     * Sets the sight tool.
     *
     * @param event The command event.
     */
    void OnToolSight(wxCommandEvent& event);

    /**
     * Handles the zoom to fit event.
     *
     * @param event The command event.
     */
    void OnToolZoomToFit(wxCommandEvent& event);

    /**
     * Handles the different tool actions (zoom, pan, etc).
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
     * Sets target and analog dates in lists and refreshes the map.
     */
    void UpdateLayers();

    /**
     * Gets the mean coordinates of the stations in WGS84.
     *
     * @param forecast The forecast object.
     * @return The mean coordinates of the stations in WGS84.
     */
    Coo GetStationsMeanCoordinatesWgs84(asResultsForecast* forecast);

    /**
     * Reloads the left viewer layer manager.
     */
    void ReloadViewerLayerManagerLeft();

    /**
     * Reloads the right viewer layer manager.
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
