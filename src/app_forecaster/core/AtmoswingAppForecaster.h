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

#ifndef AS_APP_FORECASTER_H
#define AS_APP_FORECASTER_H

#include <wx/app.h>
#include <wx/cmdline.h>
#include <wx/snglinst.h>
#include <wx/socket.h>

#include "asIncludes.h"

#if USE_GUI

class AtmoswingAppForecaster : public wxApp
#else

class AtmoswingAppForecaster : public wxAppConsole
#endif
{
  public:
    bool OnInit() override;

    int OnRun() override;

    int OnExit() override;

    void OnInitCmdLine(wxCmdLineParser& parser) override;

    bool InitLog();

    bool SetUseAsCmdLine();

    bool OnCmdLineParsed(wxCmdLineParser& parser) override;

    /**
     * Initialize the language support.
     */
    static void InitLanguageSupport();

  private:
    bool m_doConfig; /**< Do we want to configure the software? */
    bool m_doForecast; /**< Do we want to run the forecast? */
    bool m_doForecastPast; /**< Do we want to run the forecast for past dates? */
    double m_forecastDate; /**< Date of the forecast. */
    int m_forecastPastDays; /**< Number of days to forecast in the past. */
#if USE_GUI
    wxSingleInstanceChecker* m_singleInstanceChecker; /**< The single instance checker. */
#endif
};

DECLARE_APP(AtmoswingAppForecaster);

#endif
