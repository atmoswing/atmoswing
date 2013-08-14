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

#ifndef __asFrameCalibration__
#define __asFrameCalibration__

#include "AtmoswingCalibratorGui.h"
#include <asIncludes.h>
#include "asMethodCalibrator.h"
#include "asLogWindow.h"


/** Implementing asFrameCalibrationVirtual */
class asFrameCalibration : public asFrameCalibrationVirtual
{
public:
    /** Constructor */
    asFrameCalibration( wxWindow* parent );
    ~asFrameCalibration();
    void OnInit();


protected:
    asLogWindow *m_LogWindow;
    asMethodCalibrator *m_MethodCalibrator;

    // Handlers for asFrameCalibrationVirtual events.
    void Update();
    void OnSaveDefault( wxCommandEvent& event );
    void Launch( wxCommandEvent& event );
    void LoadOptions();
    void SaveOptions();
    void OpenFramePreferences( wxCommandEvent& event );
    void OpenFrameAbout( wxCommandEvent& event );
    void OnShowLog( wxCommandEvent& event );
    void OnLogLevel1( wxCommandEvent& event );
    void OnLogLevel2( wxCommandEvent& event );
    void OnLogLevel3( wxCommandEvent& event );
    void DisplayLogLevelMenu();
    void Cancel( wxCommandEvent& event );
//    void OnIdle( wxCommandEvent& event );

};

#endif // __asFrameCalibration__
