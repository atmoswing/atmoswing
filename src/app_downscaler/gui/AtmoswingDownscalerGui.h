///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct 26 2018)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/bitmap.h>
#include <wx/button.h>
#include <wx/checkbox.h>
#include <wx/choice.h>
#include <wx/colour.h>
#include <wx/filepicker.h>
#include <wx/font.h>
#include <wx/frame.h>
#include <wx/gdicmn.h>
#include <wx/icon.h>
#include <wx/image.h>
#include <wx/intl.h>
#include <wx/menu.h>
#include <wx/notebook.h>
#include <wx/panel.h>
#include <wx/radiobox.h>
#include <wx/radiobut.h>
#include <wx/settings.h>
#include <wx/sizer.h>
#include <wx/slider.h>
#include <wx/statbox.h>
#include <wx/stattext.h>
#include <wx/statusbr.h>
#include <wx/string.h>
#include <wx/textctrl.h>
#include <wx/toolbar.h>
#include <wx/xrc/xmlres.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameDownscalerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameDownscalerVirtual : public wxFrame {
   private:
   protected:
    wxPanel* m_panelMain;
    wxPanel* m_panelControls;
    wxStaticText* m_staticTextMethod;
    wxChoice* m_choiceMethod;
    wxStaticText* m_staticTextFileParameters;
    wxFilePickerCtrl* m_filePickerParameters;
    wxStaticText* m_staticTextFilePredictand;
    wxFilePickerCtrl* m_filePickerPredictand;
    wxStaticText* m_staticTextArchivePredictorDir;
    wxDirPickerCtrl* m_dirPickerArchivePredictor;
    wxStaticText* m_staticTextScenarioPredictorDir;
    wxDirPickerCtrl* m_dirPickerScenarioPredictor;
    wxStaticText* m_staticTextDownscalingResultsDir;
    wxDirPickerCtrl* m_dirPickerDownscalingResults;
    wxCheckBox* m_checkBoxParallelEvaluations;
    wxStaticText* m_staticTextStateLabel;
    wxStaticText* m_staticTextState;
    wxButton* m_buttonSaveDefault;
    wxMenuBar* m_menuBar;
    wxMenu* m_menuOptions;
    wxMenu* m_menuTools;
    wxMenu* m_menuLog;
    wxMenu* m_menuLogLevel;
    wxMenu* m_menuHelp;
    wxToolBar* m_toolBar;
    wxStatusBar* m_statusBar1;

    // Virtual event handlers, overide them in your derived class
    virtual void OnSaveDefault(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OpenFramePreferences(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OpenFramePredictandDB(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OnShowLog(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OnLogLevel1(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OnLogLevel2(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OnLogLevel3(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void OpenFrameAbout(wxCommandEvent& event) {
        event.Skip();
    }

   public:
    asFrameDownscalerVirtual(wxWindow* parent, wxWindowID id = wxID_ANY,
                             const wxString& title = _("AtmoSwing Downscaler"), const wxPoint& pos = wxDefaultPosition,
                             const wxSize& size = wxSize(606, 500),
                             long style = wxDEFAULT_FRAME_STYLE | wxTAB_TRAVERSAL);

    ~asFrameDownscalerVirtual();
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesDownscalerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesDownscalerVirtual : public wxFrame {
   private:
   protected:
    wxPanel* m_panelBase;
    wxNotebook* m_notebookBase;
    wxPanel* m_panelGeneralCommon;
    wxRadioButton* m_radioBtnLogLevel1;
    wxRadioButton* m_radioBtnLogLevel2;
    wxRadioButton* m_radioBtnLogLevel3;
    wxCheckBox* m_checkBoxDisplayLogWindow;
    wxCheckBox* m_checkBoxSaveLogFile;
    wxStaticText* m_staticTextArchivePredictorsDir;
    wxDirPickerCtrl* m_dirPickerArchivePredictors;
    wxStaticText* m_staticTextScenarioPredictorsDir;
    wxDirPickerCtrl* m_dirPickerScenarioPredictors;
    wxStaticText* m_staticTextPredictandDBDir;
    wxDirPickerCtrl* m_dirPickerPredictandDB;
    wxPanel* m_panelAdvanced;
    wxNotebook* m_notebookAdvanced;
    wxPanel* m_panelGeneral;
    wxRadioBox* m_radioBoxGui;
    wxCheckBox* m_checkBoxResponsiveness;
    wxPanel* m_panelProcessing;
    wxCheckBox* m_checkBoxAllowMultithreading;
    wxStaticText* m_staticTextThreadsNb;
    wxTextCtrl* m_textCtrlThreadsNb;
    wxStaticText* m_staticTextThreadsPriority;
    wxSlider* m_sliderThreadsPriority;
    wxRadioBox* m_radioBoxProcessingMethods;
    wxPanel* m_panelUserDirectories;
    wxStaticText* m_staticTextIntermediateResultsDir;
    wxDirPickerCtrl* m_dirPickerIntermediateResults;
    wxStaticText* m_staticTextUserDirLabel;
    wxStaticText* m_staticTextUserDir;
    wxStaticText* m_staticTextLogFileLabels;
    wxStaticText* m_staticTextLogFile;
    wxStaticText* m_staticTextPrefFileLabel;
    wxStaticText* m_staticTextPrefFile;
    wxStdDialogButtonSizer* m_buttonsConfirmation;
    wxButton* m_buttonsConfirmationOK;
    wxButton* m_buttonsConfirmationApply;
    wxButton* m_buttonsConfirmationCancel;

    // Virtual event handlers, overide them in your derived class
    virtual void OnChangeMultithreadingCheckBox(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void ApplyChanges(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void CloseFrame(wxCommandEvent& event) {
        event.Skip();
    }
    virtual void SaveAndClose(wxCommandEvent& event) {
        event.Skip();
    }

   public:
    asFramePreferencesDownscalerVirtual(wxWindow* parent, wxWindowID id = wxID_ANY,
                                        const wxString& title = _("Preferences"),
                                        const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize(482, 534),
                                        long style = wxDEFAULT_FRAME_STYLE | wxTAB_TRAVERSAL);

    ~asFramePreferencesDownscalerVirtual();
};
