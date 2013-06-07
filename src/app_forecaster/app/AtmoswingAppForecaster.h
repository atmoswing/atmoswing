/***************************************************************
 * Name:      AtmoswingAppForecaster.h
 * Purpose:   Defines Application Class
 * Author:    Pascal Horton (pascal.horton@unil.ch)
 * Created:   2009-06-08
 * Copyright: Pascal Horton (www.unil.ch/igar)
 * License:
 **************************************************************/

#ifndef AtmoswingAPPFORECASTER_H
#define AtmoswingAPPFORECASTER_H

#include <wx/app.h>
#include <wx/snglinst.h>
#include <wx/cmdline.h>
#include <wx/socket.h>
#include <asIncludes.h>

#if wxUSE_GUI
class AtmoswingAppForecaster : public wxApp
#else
class AtmoswingAppForecaster : public wxAppConsole
#endif
{
public:
    virtual bool OnInit();
    virtual int OnExit();
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    bool InitForCmdLineOnly(long logLevel);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);
    bool CommonInit();

private:
	#if wxUSE_GUI
		wxSingleInstanceChecker* m_SingleInstanceChecker;
	#endif
};

DECLARE_APP(AtmoswingAppForecaster);

#endif
