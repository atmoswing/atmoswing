/**
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch).
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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
    virtual int OnExit();
    virtual int OnRun();
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    bool InitForCmdLineOnly();
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);
    bool CommonInit();
    virtual bool OnExceptionInMainLoop();
    virtual void OnFatalException();
    virtual void OnUnhandledException();

private:
    wxString m_CalibParamsFile;
    wxString m_PredictandDB;
    wxString m_PredictorsDir;
    wxString m_CalibMethod;
    #if wxUSE_GUI
        wxSingleInstanceChecker* m_SingleInstanceChecker;
    #endif
};

DECLARE_APP(AtmoswingAppCalibrator);

#endif
