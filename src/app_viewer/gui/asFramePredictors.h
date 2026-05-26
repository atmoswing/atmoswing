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

#include <wx/overlay.h>  // wxOverlay
#include <wx/thread.h>   // wxCriticalSection (windows)

#include "AtmoSwingViewerGui.h"
#include "asForecastManager.h"
#include "asHeadersBase.h"
#include "asPanelPredictorsColorbar.h"
#include "asPredictorsRenderer.h"
#include "vroomgis.h"
#include "wx/dnd.h"

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
    asFramePredictors* _loaderFrame;
};

/**
 * @brief The frame to plot the predictors.
 *
 * This class is the frame used to plot the predictors.
 */
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
        return _panelRight;
    }

    /**
     * Get a pointer to the right panel.
     * @return A pointer to the right panel.
     */
    wxPanel* GetPanelLeft() {
        return _panelLeft;
    }

    /**
     * Get a pointer to the list of predictors.
     * @return A pointer to the list of predictors.
     */
    wxListBox* GetListPredictors() {
        return _listPredictors;
    }

    /**
     * Get a pointer to the method choice.
     * @return A pointer to the method choice.
     */
    wxChoice* GetChoiceMethod() {
        return _choiceMethod;
    }

    /**
     * Get a pointer to the forecast choice.
     * @return A pointer to the forecast choice.
     */
    wxChoice* GetChoiceForecast() {
        return _choiceForecast;
    }

    /**
     * Get a pointer to the target date choice.
     * @return A pointer to the target date choice.
     */
    wxChoice* GetChoiceTargetDates() {
        return _choiceTargetDates;
    }

    /**
     * Get a pointer to the analog date choice.
     * @return A pointer to the analog date choice.
     */
    wxChoice* GetChoiceAnalogDates() {
        return _choiceAnalogDates;
    }

    /**
     * Get a pointer to the target predictors manager.
     * @return A pointer to the target predictors manager.
     */
    asPredictorsManager* GetPredictorsManagerTarget() {
        return _predictorsManagerTarget;
    }

    /**
     * Get a pointer to the analog predictors manager.
     * @return A pointer to the analog predictors manager.
     */
    asPredictorsManager* GetPredictorsManagerAnalog() {
        return _predictorsManagerAnalog;
    }

    /**
     * Get a pointer to the forecast manager.
     * @return A pointer to the forecast manager.
     */
    asForecastManager* GetForecastManager() {
        return _forecastManager;
    }

  protected:
    wxKeyboardState _keyBoardState; /**< Keyboard state. */

    /**
     * Handle the right click on the map.
     *
     * @param event The mouse event.
     */
    virtual void OnRightClick(wxMouseEvent& event) {
        event.Skip();
    }

  private:
    asForecastManager* _forecastManager;                      /**< The forecast manager. */
    asPredictorsRenderer* _predictorsRenderer;                /**< The predictors renderer. */
    asPredictorsManager* _predictorsManagerTarget;            /**< The predictors manager for the target date. */
    asPredictorsManager* _predictorsManagerAnalog;            /**< The predictors manager for the analog date. */
    asWorkspace* _workspace;                                  /**< The workspace. */
    asPanelPredictorsColorbar* _panelPredictorsColorbarLeft;  /**< The left colorbar. */
    asPanelPredictorsColorbar* _panelPredictorsColorbarRight; /**< The right colorbar. */
    int _selectedMethod;                                      /**< The selected method (index). */
    int _selectedForecast;                                    /**< The selected forecast (index). */
    int _selectedTargetDate;                                  /**< The selected target date (index). */
    int _selectedAnalogDate;                                  /**< The selected analog date (index). */
    int _selectedPredictor;                                   /**< The selected predictor (index). */
    bool _syncroTool;                                         /**< A flag to indicate if the syncro tool is active. */
    bool _displayPanelLeft;                                   /**< A flag to indicate if the left panel is displayed. */
    bool _displayPanelRight; /**< A flag to indicate if the right panel is displayed. */
    wxOverlay _overlay;      /**< The overlay. */
#if defined(__WIN32__)
    wxCriticalSection _critSectionViewerLayerManager; /**< The critical section for the viewer layer manager. */
#endif

    // Vroomgis
    vrLayerManager* _layerManager;                  /**< The layer manager. */
    vrViewerTOCList* _tocCtrlLeft;                  /**< The left TOC control. */
    vrViewerTOCList* _tocCtrlRight;                 /**< The right TOC control. */
    vrViewerLayerManager* _viewerLayerManagerLeft;  /**< The left viewer layer manager. */
    vrViewerLayerManager* _viewerLayerManagerRight; /**< The right viewer layer manager. */
    vrViewerDisplay* _displayCtrlLeft;              /**< The left display control. */
    vrViewerDisplay* _displayCtrlRight;             /**< The right display control. */

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
