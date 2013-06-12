#ifndef __asFrameCalibration__
#define __asFrameCalibration__

/**
@file
Subclass of asFrameCalibrationVirtual, which is generated by wxFormBuilder.
*/

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
