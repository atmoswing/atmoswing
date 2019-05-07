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

class asFramePlotDistributions
        : public asFramePlotDistributionsVirutal
{
public:
    asFramePlotDistributions(wxWindow *parent, int methodRow, int forecastRow, asForecastManager *forecastManager,
                             wxWindowID id = asWINDOW_PLOTS_DISTRIBUTIONS);

    ~asFramePlotDistributions() override;

    void Init();

    bool Plot();

    bool PlotPredictands();

    bool PlotCriteria();

protected:

private:
    enum PlotPredictandsData
    {
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

    asPanelPlot *m_panelPlotPredictands;
    asPanelPlot *m_panelPlotCriteria;
    asForecastManager *m_forecastManager;
    int m_selectedMethod;
    int m_selectedForecast;
    int m_selectedStation;
    int m_selectedDate;
    int m_xmaxPredictands;

    void RebuildChoiceForecast();

    void InitPredictandsCheckListBox();

    void InitPredictandsPlotCtrl();

    void InitCriteriaPlotCtrl();

    void PlotAllReturnPeriods();

    void PlotReturnPeriod(int returnPeriod);

    void PlotAllAnalogsPoints();

    void PlotAllAnalogsCurve();

    void PlotBestAnalogsPoints(int analogsNb);

    void PlotBestAnalogsCurve(int analogsNb);

    void PlotClassicQuantiles();

    void PlotCriteriaCurve();

    void OnChoiceForecastChange(wxCommandEvent &event) override;

    void OnChoiceStationChange(wxCommandEvent &event) override;

    void OnChoiceDateChange(wxCommandEvent &event) override;

    void OnTocSelectionChange(wxCommandEvent &event) override;

    void OnClose(wxCloseEvent &evt);

DECLARE_EVENT_TABLE()
};

#endif
