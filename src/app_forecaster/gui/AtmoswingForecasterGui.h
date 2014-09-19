///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun  5 2014)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __ATMOSWINGFORECASTERGUI_H__
#define __ATMOSWINGFORECASTERGUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/intl.h>
#include <wx/calctrl.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/bmpbuttn.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/statbox.h>
#include <wx/scrolwin.h>
#include <wx/panel.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/filepicker.h>
#include <wx/radiobox.h>
#include <wx/slider.h>
#include <wx/notebook.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class asFrameMainVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameMainVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelMain;
		wxCalendarCtrl* m_CalendarForecastDate;
		wxStaticText* m_StaticTextForecastHour;
		wxTextCtrl* m_TextCtrlForecastHour;
		wxBitmapButton* m_BpButtonNow;
		wxFlexGridSizer* m_SizerLeds;
		wxButton* m_button2;
		wxScrolledWindow* m_ScrolledWindowModels;
		wxBoxSizer* m_SizerModels;
		wxBitmapButton* m_BpButtonAdd;
		wxMenuBar* m_MenuBar;
		wxMenu* m_MenuFile;
		wxMenu* m_MenuOptions;
		wxMenu* m_MenuTools;
		wxMenu* m_MenuLog;
		wxMenu* m_MenuLogLevel;
		wxMenu* m_MenuHelp;
		wxToolBar* m_ToolBar;
		wxStatusBar* m_statusBar1;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnSetPresentDate( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnConfigureDirectories( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddForecastingModel( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenBatchForecasts( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveBatchForecasts( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveBatchForecastsAs( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNewBatchForecasts( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePreferences( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePredictandDB( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameMainVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("AtmoSwing Forecaster"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameMainVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePredictandDBVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePredictandDBVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panel2;
		wxStaticText* m_StaticTextDataParam;
		wxChoice* m_ChoiceDataParam;
		wxStaticText* m_StaticTextDataTempResol;
		wxChoice* m_ChoiceDataTempResol;
		wxStaticText* m_StaticTextDataSpatAggreg;
		wxChoice* m_ChoiceDataSpatAggreg;
		wxPanel* m_PanelDataProcessing;
		wxCheckBox* m_CheckBoxReturnPeriod;
		wxTextCtrl* m_TextCtrlReturnPeriod;
		wxStaticText* m_StaticTextYears;
		wxCheckBox* m_CheckBoxSqrt;
		wxStaticText* m_StaticTextCatalogPath;
		wxFilePickerCtrl* m_FilePickerCatalogPath;
		wxStaticText* m_StaticTextDataDir;
		wxDirPickerCtrl* m_DirPickerDataDir;
		wxStaticText* m_StaticTextPatternsDir;
		wxDirPickerCtrl* m_DirPickerPatternsDir;
		wxStaticText* m_StaticDestinationDir;
		wxDirPickerCtrl* m_DirPickerDestinationDir;
		wxButton* m_ButtonSaveDefault;
		wxStdDialogButtonSizer* m_ButtonsConfirmation;
		wxButton* m_ButtonsConfirmationOK;
		wxButton* m_ButtonsConfirmationCancel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnDataSelection( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveDefault( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void BuildDatabase( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePredictandDBVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Predictand database generator"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePredictandDBVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelForecastingModelVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelForecastingModelVirtual : public wxPanel 
{
	private:
	
	protected:
		wxBoxSizer* m_SizerPanel;
		wxBoxSizer* m_SizerHeader;
		wxStaticText* m_StaticTextModelName;
		wxBitmapButton* m_BpButtonReduce;
		wxBitmapButton* m_BpButtonClose;
		wxBoxSizer* m_SizerFields;
		wxStaticText* m_StaticTextModelNameInput;
		wxTextCtrl* m_TextCtrlModelName;
		wxStaticText* m_StaticTextModelDescriptionInput;
		wxTextCtrl* m_TextCtrlModelDescription;
		wxStaticText* m_StaticTextParametersFileName;
		wxTextCtrl* m_TextCtrlParametersFileName;
		wxStaticText* m_StaticTextPredictandDB;
		wxTextCtrl* m_TextCtrlPredictandDB;
		
		// Virtual event handlers, overide them in your derived class
		virtual void ReducePanel( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClosePanel( wxCommandEvent& event ) { event.Skip(); }
		virtual void ChangeModelName( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asPanelForecastingModelVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL ); 
		~asPanelForecastingModelVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesForecasterVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesForecasterVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelBase;
		wxNotebook* m_NotebookBase;
		wxPanel* m_PanelPathsCommon;
		wxBoxSizer* m_SizerPanelPaths;
		wxStaticText* m_StaticTextParametersDir;
		wxDirPickerCtrl* m_DirPickerParameters;
		wxStaticText* m_StaticTextPredictandDBDir;
		wxDirPickerCtrl* m_DirPickerPredictandDB;
		wxStaticText* m_StaticTextArchivePredictorsDir;
		wxDirPickerCtrl* m_DirPickerArchivePredictors;
		wxStaticText* m_StaticTextRealtimePredictorSavingDir;
		wxDirPickerCtrl* m_DirPickerRealtimePredictorSaving;
		wxStaticText* m_StaticTextForecastResultsDir;
		wxDirPickerCtrl* m_DirPickerForecastResults;
		wxPanel* m_PanelGeneralCommon;
		wxRadioBox* m_RadioBoxLogLevel;
		wxCheckBox* m_CheckBoxDisplayLogWindow;
		wxCheckBox* m_CheckBoxSaveLogFile;
		wxCheckBox* m_CheckBoxProxy;
		wxStaticText* m_StaticTextProxyAddress;
		wxTextCtrl* m_TextCtrlProxyAddress;
		wxStaticText* m_StaticTextProxyPort;
		wxTextCtrl* m_TextCtrlProxyPort;
		wxStaticText* m_StaticTextProxyUser;
		wxTextCtrl* m_TextCtrlProxyUser;
		wxStaticText* m_StaticTextProxyPasswd;
		wxTextCtrl* m_TextCtrlProxyPasswd;
		wxPanel* m_PanelAdvanced;
		wxNotebook* m_NotebookAdvanced;
		wxPanel* m_PanelGeneral;
		wxRadioBox* m_RadioBoxGui;
		wxStaticText* m_StaticTextNumberFails;
		wxTextCtrl* m_TextCtrlMaxPrevStepsNb;
		wxStaticText* m_StaticTextMaxRequestsNb;
		wxTextCtrl* m_TextCtrlMaxRequestsNb;
		wxCheckBox* m_CheckBoxRestrictDownloads;
		wxCheckBox* m_CheckBoxResponsiveness;
		wxCheckBox* m_CheckBoxMultiInstancesForecaster;
		wxPanel* m_PanelProcessing;
		wxCheckBox* m_CheckBoxAllowMultithreading;
		wxStaticText* m_StaticTextThreadsNb;
		wxTextCtrl* m_TextCtrlThreadsNb;
		wxStaticText* m_StaticTextThreadsPriority;
		wxSlider* m_SliderThreadsPriority;
		wxRadioBox* m_RadioBoxProcessingMethods;
		wxRadioBox* m_RadioBoxLinearAlgebra;
		wxPanel* m_PanelUserDirectories;
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
		
		asFramePreferencesForecasterVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Preferences"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 482,534 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePreferencesForecasterVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asWizardBatchForecastsVirtual
///////////////////////////////////////////////////////////////////////////////
class asWizardBatchForecastsVirtual : public wxWizard 
{
	private:
	
	protected:
		wxStaticText* m_staticText37;
		wxStaticText* m_staticText35;
		wxButton* m_button4;
		wxStaticText* m_staticText46;
		wxStaticText* m_staticText36;
		wxStaticText* m_staticText43;
		wxFilePickerCtrl* m_FilePickerBatchFile;
		wxStaticText* m_staticText45;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnWizardFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnLoadExistingBatchForecasts( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asWizardBatchForecastsVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Batch file creation wizard"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;
		~asWizardBatchForecastsVirtual();
	
};

#endif //__ATMOSWINGFORECASTERGUI_H__
