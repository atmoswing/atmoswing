/***************************************************************
 * Name:      AtmoswingAppViewer.h
 * Purpose:   Defines Application Class
 * Author:    Pascal Horton (pascal.horton@unil.ch)
 * Created:   2009-06-08
 * Copyright: Pascal Horton (www.unil.ch/igar)
 * License:
 **************************************************************/

#ifndef AtmoswingAPPVIEWER_H
#define AtmoswingAPPVIEWER_H

#include <wx/app.h>
#include <wx/snglinst.h>
#include <wx/cmdline.h>
#include <wx/socket.h>
#include <asIncludes.h>

// Forward declaration
class asThreadsManager;

class AtmoswingAppViewer : public wxApp
{
public:
    virtual bool OnInit();
    virtual int OnExit();
    virtual void OnInitCmdLine(wxCmdLineParser& parser);
    bool InitForCmdLineOnly(long logLevel);
    virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

private:
    wxSingleInstanceChecker* m_SingleInstanceChecker;
};

DECLARE_APP(AtmoswingAppViewer);

#endif
