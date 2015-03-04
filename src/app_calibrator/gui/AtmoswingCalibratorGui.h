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
		wxStaticText* m_staticTextMonteCarloRandomNb;
		wxTextCtrl* m_textCtrlMonteCarloRandomNb;
		wxStaticText* m_staticTextVarExploStepToExplore;
		wxTextCtrl* m_textCtrlVarExploStepToExplore;
		wxButton* m_buttonSaveDefault;
		wxMenuBar* m_menuBar;
		wxMenu* m_menuOptions;
		wxMenu* m_menuLog;
		wxMenu* m_menuLogLevel;
		wxMenu* m_menuHelp;
		wxToolBar* m_toolBar;
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
