///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
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
    wxPanel* _panelMain;
    wxPanel* _panelControls;
    wxStaticText* _staticTextMethod;
    wxChoice* _choiceMethod;
    wxStaticText* _staticTextFileParameters;
    wxFilePickerCtrl* _filePickerParameters;
    wxStaticText* _staticTextFilePredictand;
    wxFilePickerCtrl* _filePickerPredictand;
    wxStaticText* _staticTextArchivePredictorDir;
    wxDirPickerCtrl* _dirPickerArchivePredictor;
    wxStaticText* _staticTextScenarioPredictorDir;
    wxDirPickerCtrl* _dirPickerScenarioPredictor;
    wxStaticText* _staticTextDownscalingResultsDir;
    wxDirPickerCtrl* _dirPickerDownscalingResults;
    wxCheckBox* _checkBoxParallelEvaluations;
    wxStaticText* _staticTextStateLabel;
    wxStaticText* _staticTextState;
    wxButton* _buttonSaveDefault;
    wxMenuBar* _menuBar;
    wxMenu* _menuOptions;
    wxMenu* _menuTools;
    wxMenu* _menuLog;
    wxMenu* _menuLogLevel;
    wxMenu* _menuHelp;
    wxToolBar* _toolBar;
    wxStatusBar* _statusBar1;

    // Virtual event handlers, override them in your derived class
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
    wxPanel* _panelBase;
    wxNotebook* _notebookBase;
    wxPanel* _panelGeneralCommon;
    wxChoice* _choiceLocale;
    wxStaticText* _staticText21;
    wxRadioButton* _radioBtnLogLevel1;
    wxRadioButton* _radioBtnLogLevel2;
    wxRadioButton* _radioBtnLogLevel3;
    wxCheckBox* _checkBoxDisplayLogWindow;
    wxCheckBox* _checkBoxSaveLogFile;
    wxStaticText* _staticTextArchivePredictorsDir;
    wxDirPickerCtrl* _dirPickerArchivePredictors;
    wxStaticText* _staticTextScenarioPredictorsDir;
    wxDirPickerCtrl* _dirPickerScenarioPredictors;
    wxStaticText* _staticTextPredictandDBDir;
    wxDirPickerCtrl* _dirPickerPredictandDB;
    wxPanel* _panelAdvanced;
    wxNotebook* _notebookAdvanced;
    wxPanel* _panelGeneral;
    wxRadioBox* _radioBoxGui;
    wxCheckBox* _checkBoxResponsiveness;
    wxPanel* _panelProcessing;
    wxCheckBox* _checkBoxAllowMultithreading;
    wxStaticText* _staticTextThreadsNb;
    wxTextCtrl* _textCtrlThreadsNb;
    wxStaticText* _staticTextThreadsPriority;
    wxSlider* _sliderThreadsPriority;
    wxRadioBox* _radioBoxProcessingMethods;
    wxPanel* _panelUserDirectories;
    wxStaticText* _staticTextIntermediateResultsDir;
    wxDirPickerCtrl* _dirPickerIntermediateResults;
    wxStaticText* _staticTextUserDirLabel;
    wxStaticText* _staticTextUserDir;
    wxStaticText* _staticTextLogFileLabels;
    wxStaticText* _staticTextLogFile;
    wxStaticText* _staticTextPrefFileLabel;
    wxStaticText* _staticTextPrefFile;
    wxStdDialogButtonSizer* _buttonsConfirmation;
    wxButton* _buttonsConfirmationOK;
    wxButton* _buttonsConfirmationApply;
    wxButton* _buttonsConfirmationCancel;

    // Virtual event handlers, override them in your derived class
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
