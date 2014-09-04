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
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/menu.h>
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
		wxPanel* m_PanelMain;
		wxNotebook* m_NotebookBase;
		wxPanel* m_PanelControls;
		wxStaticText* m_StaticTextMethod;
		wxChoice* m_ChoiceMethod;
		wxStaticText* m_StaticTextFileParameters;
		wxFilePickerCtrl* m_FilePickerParameters;
		wxStaticText* m_StaticTextFilePredictand;
		wxFilePickerCtrl* m_FilePickerPredictand;
		wxStaticText* m_StaticTextPredictorDir;
		wxDirPickerCtrl* m_DirPickerPredictor;
		wxStaticText* m_StaticTextCalibrationResultsDir;
		wxDirPickerCtrl* m_DirPickerCalibrationResults;
		wxCheckBox* m_CheckBoxParallelEvaluations;
		wxStaticText* m_StaticTextSaveAnalogDates;
		wxCheckBox* m_CheckBoxSaveAnalogDatesStep1;
		wxCheckBox* m_CheckBoxSaveAnalogDatesStep2;
		wxCheckBox* m_CheckBoxSaveAnalogDatesStep3;
		wxCheckBox* m_CheckBoxSaveAnalogDatesStep4;
		wxCheckBox* m_CheckBoxSaveAnalogDatesAllSteps;
		wxCheckBox* m_CheckBoxSaveAnalogValues;
		wxCheckBox* m_CheckBoxSaveForecastScores;
		wxCheckBox* m_CheckBoxSaveFinalForecastScore;
		wxStaticText* m_staticText60;
		wxStaticText* m_StaticTextLoadAnalogDates;
		wxCheckBox* m_CheckBoxLoadAnalogDatesStep1;
		wxCheckBox* m_CheckBoxLoadAnalogDatesStep2;
		wxCheckBox* m_CheckBoxLoadAnalogDatesStep3;
		wxCheckBox* m_CheckBoxLoadAnalogDatesStep4;
		wxCheckBox* m_CheckBoxLoadAnalogDatesAllSteps;
		wxCheckBox* m_CheckBoxLoadAnalogValues;
		wxCheckBox* m_CheckBoxLoadForecastScores;
		wxStaticText* m_staticText61;
		wxStaticText* m_StaticTextStateLabel;
		wxStaticText* m_StaticTextState;
		wxPanel* m_PanelOptions;
		wxNotebook* m_NotebookOptions;
		wxPanel* m_PanelSingle;
		wxStaticText* m_StaticTextClassicPlusStepsLonPertinenceMap;
		wxTextCtrl* m_TextCtrlClassicPlusStepsLonPertinenceMap;
		wxStaticText* m_StaticTextClassicPlusStepsLatPertinenceMap;
		wxTextCtrl* m_TextCtrlClassicPlusStepsLatPertinenceMap;
		wxStaticText* m_StaticTextClassicPlusResizingIterations;
		wxTextCtrl* m_TextCtrlClassicPlusResizingIterations;
		wxCheckBox* m_CheckBoxProceedSequentially;
		wxStaticText* m_StaticTextSpacer;
		wxCheckBox* m_CheckBoxClassicPlusResize;
		wxStaticText* m_staticText66;
		wxStaticText* m_staticText67;
		wxStaticText* m_StaticTextMonteCarloRandomNb;
		wxTextCtrl* m_TextCtrlMonteCarloRandomNb;
		wxStaticText* m_StaticTextVarExploStepToExplore;
		wxTextCtrl* m_TextCtrlVarExploStepToExplore;
		wxButton* m_ButtonSaveDefault;
		wxToolBar* m_ToolBar;
		wxStatusBar* m_statusBar1;
		wxMenuBar* m_MenuBar;
		wxMenu* m_MenuOptions;
		wxMenu* m_MenuLog;
		wxMenu* m_MenuLogLevel;
		wxMenu* m_MenuHelp;
		wxMenu* m_MenuControls;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnSaveDefault( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePreferences( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }
		virtual void Launch( wxCommandEvent& event ) { event.Skip(); }
		
	
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
		wxPanel* m_PanelBase;
		wxNotebook* m_NotebookBase;
		wxPanel* m_PanelGeneralCommon;
		wxRadioBox* m_RadioBoxLogLevel;
		wxCheckBox* m_CheckBoxDisplayLogWindow;
		wxCheckBox* m_CheckBoxSaveLogFile;
		wxStaticText* m_StaticTextParametersDir;
		wxDirPickerCtrl* m_DirPickerParameters;
		wxStaticText* m_StaticTextArchivePredictorsDir;
		wxDirPickerCtrl* m_DirPickerArchivePredictors;
		wxStaticText* m_StaticTextPredictandDBDir;
		wxDirPickerCtrl* m_DirPickerPredictandDB;
		wxPanel* m_PanelAdvanced;
		wxNotebook* m_NotebookAdvanced;
		wxPanel* m_PanelGeneral;
		wxRadioBox* m_RadioBoxGui;
		wxCheckBox* m_CheckBoxResponsiveness;
		wxPanel* m_PanelProcessing;
		wxCheckBox* m_CheckBoxAllowMultithreading;
		wxStaticText* m_StaticTextThreadsNb;
		wxTextCtrl* m_TextCtrlThreadsNb;
		wxStaticText* m_StaticTextThreadsPriority;
		wxSlider* m_SliderThreadsPriority;
		wxRadioBox* m_RadioBoxProcessingMethods;
		wxRadioBox* m_RadioBoxLinearAlgebra;
		wxPanel* m_PanelUserDirectories;
		wxStaticText* m_StaticTextIntermediateResultsDir;
		wxDirPickerCtrl* m_DirPickerIntermediateResults;
		wxStaticText* m_StaticTextUserDirLabel;
		wxStaticText* m_StaticTextUserDir;
		wxStaticText* m_StaticTextLogFileLabel;
		wxStaticText* m_StaticTextLogFile;
		wxStaticText* m_StaticTextPrefFileLabel;
		wxStaticText* m_StaticTextPrefFile;
		wxStdDialogButtonSizer* m_ButtonsConfirmation;
		wxButton* m_ButtonsConfirmationOK;
		wxButton* m_ButtonsConfirmationApply;
		wxButton* m_ButtonsConfirmationCancel;
		
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
