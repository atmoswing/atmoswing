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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#ifndef ATMOSWINGAPPDOWNSCALER_H
#define ATMOSWINGAPPDOWNSCALER_H

#include <wx/app.h>
#include <wx/cmdline.h>
#include <wx/snglinst.h>
#include <wx/socket.h>

#include "asIncludes.h"

#if USE_GUI

class AtmoSwingAppDownscaler : public wxApp
#else

class AtmoSwingAppDownscaler : public wxAppConsole
#endif
{
  public:
    virtual ~AtmoSwingAppDownscaler(){};

    virtual bool OnInit();

    virtual int OnRun();

    virtual int OnExit();

    void CleanUp();

    virtual void OnInitCmdLine(wxCmdLineParser& parser);

    wxString GetLocalPath();

    bool InitLog();

    bool SetUseAsCmdLine();

    bool InitForCmdLineOnly();

    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

    static void InitLanguageSupport();

    virtual bool OnExceptionInMainLoop();

    virtual void OnFatalException();

    virtual void OnUnhandledException();

  private:
    wxString m_downscalingParamsFile;
    wxString m_downscalingMethod;
    wxString m_predictandDB;
    wxString m_predictorsArchiveDir;
    wxString m_predictorsScenarioDir;
    vi m_predictandStationIds;
    bool m_doProcessing;
#if USE_GUI
    wxSingleInstanceChecker* m_singleInstanceChecker;
#endif
};

DECLARE_APP(AtmoSwingAppDownscaler);

#endif
