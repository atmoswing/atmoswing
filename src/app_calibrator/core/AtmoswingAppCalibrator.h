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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */
 
#ifndef AtmoswingAPPCalibrator_H
#define AtmoswingAPPCalibrator_H

#include <wx/app.h>
#include <wx/snglinst.h>
#include <wx/cmdline.h>
#include <wx/socket.h>
#include <asIncludes.h>

#if wxUSE_GUI
class AtmoswingAppCalibrator : public wxApp
#else
class AtmoswingAppCalibrator : public wxAppConsole
#endif
{
public:
    //AtmoswingAppCalibrator();
    virtual ~AtmoswingAppCalibrator(){};
    virtual bool OnInit();
    virtual int OnRun();
    virtual int OnExit();
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    bool InitForCmdLineOnly();
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);
    bool CommonInit();
    virtual bool OnExceptionInMainLoop();
    virtual void OnFatalException();
    virtual void OnUnhandledException();

private:
    wxString m_calibParamsFile;
    wxString m_predictandDB;
    wxString m_predictorsDir;
	VectorInt m_predictandStationIds;
    wxString m_calibMethod;
	bool m_forceQuit;
    #if wxUSE_GUI
        wxSingleInstanceChecker* m_singleInstanceChecker;
    #endif
};

DECLARE_APP(AtmoswingAppCalibrator);

#endif
