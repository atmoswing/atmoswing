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

#ifndef AS_FRAME_PLOT_DISTRIBUTIONS_H
#define AS_FRAME_PLOT_DISTRIBUTIONS_H

#include "AtmoswingViewerGui.h"
#include "asPanelPlot.h"

class asForecastManager;

class asFramePlotDistributions : public asFramePlotDistributionsVirutal {
  public:
    /**
     * Constructor.
     * 
     * @param parent Parent window.
     * @param methodRow The selected method row.
     * @param forecastRow The selected forecast row.
     * @param forecastManager The forecast manager.
     * @param id Window ID.
     */
    asFramePlotDistributions(wxWindow* parent, int methodRow, int forecastRow, asForecastManager* forecastManager,
                             wxWindowID id = asWINDOW_PLOTS_DISTRIBUTIONS);

    /**
     * Destructor.
     */
    ~asFramePlotDistributions() override;

    /**
     * Initialize the frame.
     */
    void Init();

    /**
     * Plot the distributions.
     *
     * @return True if plotted successfully.
     */
    bool Plot();

    /**
     * Plot the predictands.
     *
     * @return True if plotted successfully.
     */
    bool PlotPredictands();

    /**
     * Reset the extent of the plot to the default value.
     *
     * @param event The command event.
     */
    void ResetExtent(wxCommandEvent& event) override;

    /**
     * Plot the analogy criteria.
     *
     * @return True if plotted successfully.
     */
    bool PlotCriteria();

  protected:
  private:
    enum PlotPredictandsData {
        ClassicQuantiles,
        AllAnalogsPoints,
        AllAnalogsCurve,
        BestAnalogs10Points,
        BestAnalogs10Curve,
        BestAnalogs5Points,
        BestAnalogs5Curve,
        ClassicReturnPeriod,
        AllReturnPeriods
    };

    asPanelPlot* m_panelPlotPredictands; /**< The panel for the predictands plot. */
    asPanelPlot* m_panelPlotCriteria; /**< The panel for the criteria plot. */
    asForecastManager* m_forecastManager; /**< The forecast manager. */
    int m_selectedMethod; /**< The selected method (index). */
    int m_selectedForecast; /**< The selected forecast (index). */
    int m_selectedStation; /**< The selected station (index). */
    int m_selectedDate;  /**< The selected date (index). */
    int m_xmaxPredictands; /**< The maximum value for the x-axis of the predictands plot. */

    /**
     * Rebuild the choice of the forecast.
     */
    void RebuildChoiceForecast();

    /**
     * Rebuild the choice of the station.
     */
    void InitPredictandsCheckListBox();

    /**
     * Initialize the predictands plot control.
     */
    void InitPredictandsPlotCtrl();

    /**
     * Initialize the criteria plot control.
     */
    void InitCriteriaPlotCtrl();

    /**
     * Plot all return periods.
     */
    void PlotAllReturnPeriods();

    /**
     * Plot the given return period.
     *
     * @param returnPeriod The given return period.
     */
    void PlotReturnPeriod(int returnPeriod);

    /**
     * Plot all analogs as points.
     */
    void PlotAllAnalogsPoints();

    /**
     * Plot all analogs as a curve.
     */
    void PlotAllAnalogsCurve();

    /**
     * Plot the best analogs as points.
     *
     * @param analogsNb The number of analogs.
     */
    void PlotBestAnalogsPoints(int analogsNb);

    /**
     * Plot the best analogs as a curve.
     *
     * @param analogsNb The number of analogs.
     */
    void PlotBestAnalogsCurve(int analogsNb);

    /**
     * Plot the classic quantiles.
     */
    void PlotClassicQuantiles();

    /**
     * Plot the criteria curve.
     */
    void PlotCriteriaCurve();

    /**
     * When the choice of the method changes.
     * 
     * @param event The command event.
     */
    void OnChoiceForecastChange(wxCommandEvent& event) override;

    /**
     * Event triggered when the choice of the station changes.
     * 
     * @param event The command event.
     */
    void OnChoiceStationChange(wxCommandEvent& event) override;

    /**
     * Event triggered when the choice of the date changes.
     * 
     * @param event The command event.
     */
    void OnChoiceDateChange(wxCommandEvent& event) override;

    /**
     * Event triggered when the choice of the items to show has changed in the table of content.
     * 
     * @param event The command event.
     */
    void OnTocSelectionChange(wxCommandEvent& event) override;

    /** 
     * Event triggered when the frame is being closed.
     * 
     * @param evt The event.
    */
    void OnClose(wxCloseEvent& evt);

    DECLARE_EVENT_TABLE()
};

#endif
