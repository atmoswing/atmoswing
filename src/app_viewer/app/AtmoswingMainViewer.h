/***************************************************************
 * Name:      AtmoswingMainViewer.h
 * Purpose:   Defines Application Frame
 * Author:    Pascal Horton (pascal.horton@unil.ch)
 * Created:   2009-06-08
 * Copyright: Pascal Horton (www.unil.ch/igar)
 * License:
 **************************************************************/

#ifndef AtmoswingMAINVIEWER_H
#define AtmoswingMAINVIEWER_H

//#include "version.h"
#include "asIncludes.h"
#include "AtmoswingAppViewer.h"
#include "asFrameForecastRings.h"


class AtmoswingFrameViewer: public asFrameForecastRings
{
public:
    AtmoswingFrameViewer(wxFrame *frame);
    ~AtmoswingFrameViewer();
private:
    asLogWindow *m_LogWindow;
    virtual void OnClose(wxCloseEvent& event);
    virtual void OnQuit(wxCommandEvent& event);
    void OnShowLog( wxCommandEvent& event );
    void ProcessTest();
    void SetDefaultOptions();
};

#endif // AtmoswingMAINVIEWER_H
