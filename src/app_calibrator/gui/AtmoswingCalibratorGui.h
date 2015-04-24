///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun  5 2014)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __ATMOSWINGCALIBRATORGUI_H__
#define __ATMOSWINGCALIBRATORGUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/intl.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/choice.h>
#include <wx/filepicker.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>
#include <wx/statbox.h>
#include <wx/panel.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/textctrl.h>
#include <wx/notebook.h>
#include <wx/button.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/radiobox.h>
#include <wx/slider.h>

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class asFrameCalibrationVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameCalibrationVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelMain;
		wxNotebook* m_notebookBase;
		wxPanel* m_panelControls;
		wxStaticText* m_staticTextMethod;
		wxChoice* m_choiceMethod;
		wxStaticText* m_staticTextFileParameters;
		wxFilePickerCtrl* m_filePickerParameters;
		wxStaticText* m_staticTextFilePredictand;
		wxFilePickerCtrl* m_filePickerPredictand;
		wxStaticText* m_staticTextPredictorDir;
		wxDirPickerCtrl* m_dirPickerPredictor;
		wxStaticText* m_staticTextCalibrationResultsDir;
		wxDirPickerCtrl* m_dirPickerCalibrationResults;
		wxCheckBox* m_checkBoxParallelEvaluations;
		wxStaticText* m_staticTextSaveAnalogDates;
		wxCheckBox* m_checkBoxSaveAnalogDatesStep1;
		wxCheckBox* m_checkBoxSaveAnalogDatesStep2;
		wxCheckBox* m_checkBoxSaveAnalogDatesStep3;
		wxCheckBox* m_checkBoxSaveAnalogDatesStep4;
		wxCheckBox* m_checkBoxSaveAnalogDatesAllSteps;
		wxCheckBox* m_checkBoxSaveAnalogValues;
		wxCheckBox* m_checkBoxSaveForecastScores;
		wxCheckBox* m_checkBoxSaveFinalForecastScore;
		wxStaticText* m_staticText60;
		wxStaticText* m_staticTextLoadAnalogDates;
		wxCheckBox* m_checkBoxLoadAnalogDatesStep1;
		wxCheckBox* m_checkBoxLoadAnalogDatesStep2;
		wxCheckBox* m_checkBoxLoadAnalogDatesStep3;
		wxCheckBox* m_checkBoxLoadAnalogDatesStep4;
		wxCheckBox* m_checkBoxLoadAnalogDatesAllSteps;
		wxCheckBox* m_checkBoxLoadAnalogValues;
		wxCheckBox* m_checkBoxLoadForecastScores;
		wxStaticText* m_staticText61;
		wxStaticText* m_staticTextStateLabel;
		wxStaticText* m_staticTextState;
		wxPanel* m_panelOptions;
		wxNotebook* m_notebookOptions;
		wxPanel* m_panelSingle;
		wxStaticText* m_staticTextClassicPlusStepsLonPertinenceMap;
		wxTextCtrl* m_textCtrlClassicPlusStepsLonPertinenceMap;
		wxStaticText* m_staticTextClassicPlusStepsLatPertinenceMap;
		wxTextCtrl* m_textCtrlClassicPlusStepsLatPertinenceMap;
		wxStaticText* m_staticTextClassicPlusResizingIterations;
		wxTextCtrl* m_textCtrlClassicPlusResizingIterations;
		wxCheckBox* m_checkBoxProceedSequentially;
		wxStaticText* m_staticTextSpacer;
		wxCheckBox* m_checkBoxClassicPlusResize;
		wxStaticText* m_staticText66;
		wxStaticText* m_staticText67;
		wxStaticText* m_StaticTextMonteCarloRandomNb;
		wxTextCtrl* m_TextCtrlMonteCarloRandomNb;
		wxStaticText* m_StaticTextVarExploStepToExplore;
		wxTextCtrl* m_TextCtrlVarExploStepToExplore;
		wxPanel* m_PanelGeneticAlgoritms;
		wxStaticText* m_StaticTextGAsNaturalSelectionOperator;
		wxChoice* m_ChoiceGAsNaturalSelectionOperator;
		wxStaticText* m_StaticTextGAsCouplesSelectionOperator;
		wxChoice* m_ChoiceGAsCouplesSelectionOperator;
		wxStaticText* m_StaticTextGAsCrossoverOperator;
		wxChoice* m_ChoiceGAsCrossoverOperator;
		wxStaticText* m_StaticTextGAsMutationOperator;
		wxChoice* m_ChoiceGAsMutationOperator;
		wxStaticText* m_StaticTextGAsRunNumbers;
		wxTextCtrl* m_TextCtrlGAsRunNumbers;
		wxStaticText* m_StaticTextGAsPopulationSize;
		wxTextCtrl* m_TextCtrlGAsPopulationSize;
		wxStaticText* m_StaticTextGAsConvergenceNb;
		wxTextCtrl* m_TextCtrlGAsConvergenceNb;
		wxStaticText* m_StaticTextGAsRatioIntermGen;
		wxTextCtrl* m_TextCtrlGAsRatioIntermGen;
		wxCheckBox* m_CheckBoxGAsAllowElitism;
		wxNotebook* m_NotebookGAoptions;
		wxPanel* m_PanelSelections;
		wxStaticText* m_StaticTextGAsNaturalSlctTournamentProb;
		wxTextCtrl* m_TextCtrlGAsNaturalSlctTournamentProb;
		wxStaticText* m_StaticTextGAsCouplesSlctTournamentNb;
		wxTextCtrl* m_TextCtrlGAsCouplesSlctTournamentNb;
		wxPanel* m_PanelCrossover;
		wxStaticText* m_StaticTextGAsCrossoverMultipleNbPts;
		wxTextCtrl* m_TextCtrlGAsCrossoverMultipleNbPts;
		wxStaticText* m_StaticTextGAsCrossoverBlendingNbPts;
		wxTextCtrl* m_TextCtrlGAsCrossoverBlendingNbPts;
		wxStaticText* m_StaticTextGAsCrossoverBlendingShareBeta;
		wxCheckBox* m_CheckBoxGAsCrossoverBlendingShareBeta;
		wxStaticText* m_StaticTextGAsCrossoverLinearNbPts;
		wxTextCtrl* m_TextCtrlGAsCrossoverLinearNbPts;
		wxStaticText* m_StaticTextGAsCrossoverHeuristicNbPts;
		wxTextCtrl* m_TextCtrlGAsCrossoverHeuristicNbPts;
		wxStaticText* m_StaticTextGAsCrossoverHeuristicShareBeta;
		wxCheckBox* m_CheckBoxGAsCrossoverHeuristicShareBeta;
		wxStaticText* m_StaticTextGAsCrossoverBinLikeNbPts;
		wxTextCtrl* m_TextCtrlGAsCrossoverBinLikeNbPts;
		wxStaticText* m_StaticTextGAsCrossoverBinLikeShareBeta;
		wxCheckBox* m_CheckBoxGAsCrossoverBinLikeShareBeta;
		wxPanel* m_PanelMutation;
		wxStaticText* m_StaticTextGAsMutationsUniformCstProb;
		wxTextCtrl* m_TextCtrlGAsMutationsUniformCstProb;
		wxStaticText* m_StaticTextGAsMutationsNormalCstProb;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalCstProb;
		wxStaticText* m_StaticTextGAsMutationsNormalCstStdDev;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalCstStdDev;
		wxStaticText* m_StaticTextGAsMutationsUniformVarMaxGensNb;
		wxTextCtrl* m_TextCtrlGAsMutationsUniformVarMaxGensNb;
		wxStaticText* m_StaticTextGAsMutationsUniformVarProbStart;
		wxTextCtrl* m_TextCtrlGAsMutationsUniformVarProbStart;
		wxStaticText* m_StaticTextGAsMutationsUniformVarProbEnd;
		wxTextCtrl* m_TextCtrlGAsMutationsUniformVarProbEnd;
		wxStaticText* m_StaticTextGAsMutationsMultiScaleProb;
		wxTextCtrl* m_TextCtrlGAsMutationsMultiScaleProb;
		wxStaticText* m_StaticTextGAsMutationsNormalVarMaxGensNbProb;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalVarMaxGensNbProb;
		wxStaticText* m_StaticTextGAsMutationsNormalVarMaxGensNbStdDev;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalVarMaxGensNbStdDev;
		wxStaticText* m_StaticTextGAsMutationsNormalVarProbStart;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalVarProbStart;
		wxStaticText* m_StaticTextGAsMutationsNormalVarProbEnd;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalVarProbEnd;
		wxStaticText* m_StaticTextGAsMutationsNormalVarStdDevStart;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalVarStdDevStart;
		wxStaticText* m_StaticTextGAsMutationsNormalVarStdDevEnd;
		wxTextCtrl* m_TextCtrlGAsMutationsNormalVarStdDevEnd;
		wxStaticText* m_StaticTextGAsMutationsNonUniformProb;
		wxTextCtrl* m_TextCtrlGAsMutationsNonUniformProb;
		wxStaticText* m_StaticTextGAsMutationsNonUniformGensNb;
		wxTextCtrl* m_TextCtrlGAsMutationsNonUniformGensNb;
		wxStaticText* m_StaticTextGAsMutationsNonUniformMinRate;
		wxTextCtrl* m_TextCtrlGAsMutationsNonUniformMinRate;
		wxButton* m_ButtonSaveDefault;
		wxMenuBar* m_MenuBar;
		wxMenu* m_MenuOptions;
		wxMenu* m_MenuLog;
		wxMenu* m_MenuLogLevel;
		wxMenu* m_MenuHelp;
		wxToolBar* m_ToolBar;
		wxStatusBar* m_statusBar1;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnSaveDefault( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePreferences( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameCalibrationVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Atmoswing Calibrator"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameCalibrationVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesCalibratorVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesCalibratorVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelBase;
		wxNotebook* m_notebookBase;
		wxPanel* m_panelGeneralCommon;
		wxRadioBox* m_radioBoxLogLevel;
		wxCheckBox* m_checkBoxDisplayLogWindow;
		wxCheckBox* m_checkBoxSaveLogFile;
		wxStaticText* m_staticTextArchivePredictorsDir;
		wxDirPickerCtrl* m_dirPickerArchivePredictors;
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
		wxRadioBox* m_radioBoxLinearAlgebra;
		wxPanel* m_panelUserDirectories;
		wxStaticText* m_staticTextIntermediateResultsDir;
		wxDirPickerCtrl* m_dirPickerIntermediateResults;
		wxStaticText* m_staticTextUserDirLabel;
		wxStaticText* m_staticTextUserDir;
		wxStaticText* m_staticTextLogFileLabel;
		wxStaticText* m_staticTextLogFile;
		wxStaticText* m_staticTextPrefFileLabel;
		wxStaticText* m_staticTextPrefFile;
		wxStdDialogButtonSizer* m_buttonsConfirmation;
		wxButton* m_buttonsConfirmationOK;
		wxButton* m_buttonsConfirmationApply;
		wxButton* m_buttonsConfirmationCancel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnChangeMultithreadingCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void ApplyChanges( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void SaveAndClose( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePreferencesCalibratorVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Preferences"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 482,534 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePreferencesCalibratorVirtual();
	
};

#endif //__ATMOSWINGCALIBRATORGUI_H__
