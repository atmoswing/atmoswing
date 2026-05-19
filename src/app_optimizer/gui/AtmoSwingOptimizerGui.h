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
/// Class asFrameOptimizerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameOptimizerVirtual : public wxFrame {
  private:
  protected:
    wxPanel* _panelMain;
    wxNotebook* _notebookBase;
    wxPanel* _panelControls;
    wxStaticText* _staticTextMethod;
    wxChoice* _choiceMethod;
    wxStaticText* _staticTextFileParameters;
    wxFilePickerCtrl* _filePickerParameters;
    wxStaticText* _staticTextFilePredictand;
    wxFilePickerCtrl* _filePickerPredictand;
    wxStaticText* _staticTextPredictorDir;
    wxDirPickerCtrl* _dirPickerPredictor;
    wxStaticText* _staticTextCalibrationResultsDir;
    wxDirPickerCtrl* _dirPickerCalibrationResults;
    wxStaticText* _staticTextStateLabel;
    wxStaticText* _staticTextState;
    wxPanel* _panelOptions;
    wxNotebook* _notebookOptions;
    wxPanel* _panelSingle;
    wxStaticText* _staticTextClassicPlusStepsLonPertinenceMap;
    wxTextCtrl* _textCtrlClassicPlusStepsLonPertinenceMap;
    wxStaticText* _staticTextClassicPlusStepsLatPertinenceMap;
    wxTextCtrl* _textCtrlClassicPlusStepsLatPertinenceMap;
    wxStaticText* _staticTextClassicPlusResizingIterations;
    wxTextCtrl* _textCtrlClassicPlusResizingIterations;
    wxCheckBox* _checkBoxProceedSequentially;
    wxStaticText* _staticTextSpacer;
    wxCheckBox* _checkBoxClassicPlusResize;
    wxStaticText* _staticTextMonteCarloRandomNb;
    wxTextCtrl* _textCtrlMonteCarloRandomNb;
    wxStaticText* _staticTextVarExploStepToExplore;
    wxTextCtrl* _textCtrlVarExploStepToExplore;
    wxPanel* _panelGeneticAlgoritms;
    wxStaticText* _staticTextGAsNaturalSelectionOperator;
    wxChoice* _choiceGAsNaturalSelectionOperator;
    wxStaticText* _staticTextGAsCouplesSelectionOperator;
    wxChoice* _choiceGAsCouplesSelectionOperator;
    wxStaticText* _staticTextGAsCrossoverOperator;
    wxChoice* _choiceGAsCrossoverOperator;
    wxStaticText* _staticTextGAsMutationOperator;
    wxChoice* _choiceGAsMutationOperator;
    wxStaticText* _staticTextGAsRunNumbers;
    wxTextCtrl* _textCtrlGAsRunNumbers;
    wxStaticText* _staticTextGAsPopulationSize;
    wxTextCtrl* _textCtrlGAsPopulationSize;
    wxStaticText* _staticTextGAsConvergenceNb;
    wxTextCtrl* _textCtrlGAsConvergenceNb;
    wxStaticText* _staticTextGAsRatioIntermGen;
    wxTextCtrl* _textCtrlGAsRatioIntermGen;
    wxCheckBox* _checkBoxGAsAllowElitism;
    wxNotebook* _notebookGAoptions;
    wxPanel* _panelSelections;
    wxStaticText* _staticTextGAsNaturalSlctTournamentProb;
    wxTextCtrl* _textCtrlGAsNaturalSlctTournamentProb;
    wxStaticText* _staticTextGAsCouplesSlctTournamentNb;
    wxTextCtrl* _textCtrlGAsCouplesSlctTournamentNb;
    wxPanel* _panelCrossover;
    wxStaticText* _staticTextGAsCrossoverMultipleNbPts;
    wxTextCtrl* _textCtrlGAsCrossoverMultipleNbPts;
    wxStaticText* _staticTextGAsCrossoverBlendingNbPts;
    wxTextCtrl* _textCtrlGAsCrossoverBlendingNbPts;
    wxStaticText* _staticTextGAsCrossoverBlendingShareBeta;
    wxCheckBox* _checkBoxGAsCrossoverBlendingShareBeta;
    wxStaticText* _staticTextGAsCrossoverLinearNbPts;
    wxTextCtrl* _textCtrlGAsCrossoverLinearNbPts;
    wxStaticText* _staticTextGAsCrossoverHeuristicNbPts;
    wxTextCtrl* _textCtrlGAsCrossoverHeuristicNbPts;
    wxStaticText* _staticTextGAsCrossoverHeuristicShareBeta;
    wxCheckBox* _checkBoxGAsCrossoverHeuristicShareBeta;
    wxStaticText* _staticTextGAsCrossoverBinLikeNbPts;
    wxTextCtrl* _textCtrlGAsCrossoverBinLikeNbPts;
    wxStaticText* _staticTextGAsCrossoverBinLikeShareBeta;
    wxCheckBox* _checkBoxGAsCrossoverBinLikeShareBeta;
    wxPanel* _panelMutation;
    wxStaticText* _staticTextGAsMutationsUniformCstProb;
    wxTextCtrl* _textCtrlGAsMutationsUniformCstProb;
    wxStaticText* _staticTextGAsMutationsNormalCstProb;
    wxTextCtrl* _textCtrlGAsMutationsNormalCstProb;
    wxStaticText* _staticTextGAsMutationsNormalCstStdDev;
    wxTextCtrl* _textCtrlGAsMutationsNormalCstStdDev;
    wxStaticText* _staticTextGAsMutationsUniformVarMaxGensNb;
    wxTextCtrl* _textCtrlGAsMutationsUniformVarMaxGensNb;
    wxStaticText* _staticTextGAsMutationsUniformVarProbStart;
    wxTextCtrl* _textCtrlGAsMutationsUniformVarProbStart;
    wxStaticText* _staticTextGAsMutationsUniformVarProbEnd;
    wxTextCtrl* _textCtrlGAsMutationsUniformVarProbEnd;
    wxStaticText* _staticTextGAsMutationsMultiScaleProb;
    wxTextCtrl* _textCtrlGAsMutationsMultiScaleProb;
    wxStaticText* _staticTextGAsMutationsNormalVarMaxGensNbProb;
    wxTextCtrl* _textCtrlGAsMutationsNormalVarMaxGensNbProb;
    wxStaticText* _staticTextGAsMutationsNormalVarMaxGensNbStdDev;
    wxTextCtrl* _textCtrlGAsMutationsNormalVarMaxGensNbStdDev;
    wxStaticText* _staticTextGAsMutationsNormalVarProbStart;
    wxTextCtrl* _textCtrlGAsMutationsNormalVarProbStart;
    wxStaticText* _staticTextGAsMutationsNormalVarProbEnd;
    wxTextCtrl* _textCtrlGAsMutationsNormalVarProbEnd;
    wxStaticText* _staticTextGAsMutationsNormalVarStdDevStart;
    wxTextCtrl* _textCtrlGAsMutationsNormalVarStdDevStart;
    wxStaticText* _staticTextGAsMutationsNormalVarStdDevEnd;
    wxTextCtrl* _textCtrlGAsMutationsNormalVarStdDevEnd;
    wxStaticText* _staticTextGAsMutationsNonUniformProb;
    wxTextCtrl* _textCtrlGAsMutationsNonUniformProb;
    wxStaticText* _staticTextGAsMutationsNonUniformGensNb;
    wxTextCtrl* _textCtrlGAsMutationsNonUniformGensNb;
    wxStaticText* _staticTextGAsMutationsNonUniformMinRate;
    wxTextCtrl* _textCtrlGAsMutationsNonUniformMinRate;
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
    asFrameOptimizerVirtual(wxWindow* parent, wxWindowID id = wxID_ANY,
                            const wxString& title = _("AtmoSwing Optimizer"), const wxPoint& pos = wxDefaultPosition,
                            const wxSize& size = wxSize(-1, -1), long style = wxDEFAULT_FRAME_STYLE | wxTAB_TRAVERSAL);

    ~asFrameOptimizerVirtual();
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesOptimizerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesOptimizerVirtual : public wxFrame {
  private:
  protected:
    wxPanel* _panelBase;
    wxNotebook* _notebookBase;
    wxPanel* _panelGeneralCommon;
    wxChoice* _choiceLocale;
    wxStaticText* _staticText59;
    wxRadioButton* _radioBtnLogLevel1;
    wxRadioButton* _radioBtnLogLevel2;
    wxRadioButton* _radioBtnLogLevel3;
    wxCheckBox* _checkBoxDisplayLogWindow;
    wxCheckBox* _checkBoxSaveLogFile;
    wxStaticText* _staticTextArchivePredictorsDir;
    wxDirPickerCtrl* _dirPickerArchivePredictors;
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
    asFramePreferencesOptimizerVirtual(wxWindow* parent, wxWindowID id = wxID_ANY,
                                       const wxString& title = _("Preferences"), const wxPoint& pos = wxDefaultPosition,
                                       const wxSize& size = wxSize(482, 534),
                                       long style = wxDEFAULT_FRAME_STYLE | wxTAB_TRAVERSAL);

    ~asFramePreferencesOptimizerVirtual();
};
