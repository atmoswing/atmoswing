/***************************************************************
 * Name:      AtmoswingMainForecaster.h
 * Purpose:   Defines Application Frame
 * Author:    Pascal Horton (pascal.horton@unil.ch)
 * Created:   2009-06-08
 * Copyright: Pascal Horton (www.unil.ch/igar)
 * License:
 **************************************************************/

#ifndef AtmoswingMAINFORECATSER_H
#define AtmoswingMAINFORECATSER_H

//#include "version.h"
#include "asIncludes.h"
#include "AtmoswingAppForecaster.h"
#include "asFrameMain.h"


class AtmoswingFrameForecaster: public asFrameMain
{
public:
    AtmoswingFrameForecaster(wxFrame *frame);
    ~AtmoswingFrameForecaster();
private:
    virtual void OnClose(wxCloseEvent& event);
    virtual void OnQuit(wxCommandEvent& event);
    void ProcessTest();
    void SetDefaultOptions();
};

#endif // AtmoswingMAINFORECATSER_H
