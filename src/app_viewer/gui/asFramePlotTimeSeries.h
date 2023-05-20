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

#ifndef AS_FRAME_PLOT_TIMESERIES_H
#define AS_FRAME_PLOT_TIMESERIES_H

#include "AtmoswingViewerGui.h"
#include "asPanelPlot.h"

class asForecastManager;

class asFramePlotTimeSeries : public asFramePlotTimeSeriesVirtual {
  public:
    /**
     * Constructor.
     * 
     * @param parent Parent window.
     * @param selectedMethod The selected method.
     * @param selectedForecast The selected forecast.
     * @param selectedStation The selected station.
     * @param forecastManager The forecast manager.
     * @param id Window ID.
     */
    asFramePlotTimeSeries(wxWindow* parent, int selectedMethod, int selectedForecast, int selectedStation,
                          asForecastManager* forecastManager, wxWindowID id = asWINDOW_PLOTS_TIMESERIES);

    /**
     * Destructor.
     */
    ~asFramePlotTimeSeries() override = default;

    /**
     * Initialize the frame.
     */ 
    void Init();

    /**
     * Plot the data.
     * 
     * @return True if plotted successfully.
     */
    bool Plot();

  protected:
  private:
    enum PlotData {
        ClassicQuantiles,
        AllQuantiles,
        AllAnalogs,
        BestAnalogs10,
        BestAnalogs5,
        ClassicReturnPeriod,
        AllReturnPeriods,
        PreviousForecasts,
        Interpretation
    };

    asPanelPlot* m_panelPlot; /**< The plot panel. */
    asForecastManager* m_forecastManager; /**< The forecast manager. */
    int m_selectedStation; /**< The selected station (index). */
    int m_selectedMethod; /**< The selected method (index). */
    int m_selectedForecast; /**< The selected forecast (index). */
    float m_maxVal; /**< The maximum value of the time series. */
    vd m_leadTimes; /**< The lead times. */

    /**
     * Event triggered when the frame is closed.
     * 
     * @param evt The event.
     */
    void OnClose(wxCloseEvent& evt);

    /**
     * Event triggered when the choice of the items to show has changed in the table of content.
     * 
     * @param event The command event.
     */
    void OnTocSelectionChange(wxCommandEvent& event) override;

    /**
     * Event triggered when the export to TXT button is pressed.
     * 
     * @param event The command event.
     */
    void OnExportTXT(wxCommandEvent& event) override;

    /**
     * Event triggered when the export to SVG button is pressed.
     * 
     * @param event The command event.
     */
    void OnExportSVG(wxCommandEvent& event);

    /**
     * Event triggered when the preview button is pressed.
     * 
     * @param event The command event.
     */
    void OnPreview(wxCommandEvent& event) override;

    /**
     * Event triggered when the print button is pressed.
     * 
     * @param event The command event.
     */
    void OnPrint(wxCommandEvent& event) override;

    /**
     * Initialize the check list box.
     */
    void InitCheckListBox();

    /**
     * Initialize the plot control.
     */
    void InitPlotCtrl();

    /**
     * Reset the extent of the plot to the default value.
     *
     * @param event The command event.
     */
    void ResetExtent(wxCommandEvent& event) override;

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
     * Plot all analogs.
     */
    void PlotAllAnalogs();

    /**
     * Plot the best analogs as points.
     *
     * @param pointsNb The number of analogs.
     */
    void PlotBestAnalogs(int pointsNb);

    /**
     * Plot the classic quantiles.
     */
    void PlotClassicQuantiles();

    /** 
     * Plot the past forecasts.
     */
    void PlotPastForecasts();

    /**
     * Plot a past forecast.
     *
     * @param i The index of the forecast.
     */
    void PlotPastForecast(int i);

    /**
     * Plot all quantiles.
     */
    void PlotAllQuantiles();

    /**
     * Plot the interpretation curve (not enabled yet).
     */
    void PlotInterpretation();

    DECLARE_EVENT_TABLE()
};

#endif
