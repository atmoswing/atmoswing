///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Nov  6 2017)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
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
#include <wx/gauge.h>
#include <wx/scrolwin.h>
#include <wx/panel.h>
#include <wx/menu.h>
#include <wx/statusbr.h>
#include <wx/toolbar.h>
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
		wxPanel* m_panelMain;
		wxCalendarCtrl* m_calendarForecastDate;
		wxStaticText* m_staticTextForecastHour;
		wxTextCtrl* m_textCtrlForecastHour;
		wxBitmapButton* m_bpButtonNow;
		wxGauge* m_gauge;
		wxStaticText* m_staticTextProgressActual;
		wxStaticText* m_staticText38;
		wxStaticText* m_staticTextProgressTot;
		wxFlexGridSizer* m_sizerLeds;
		wxButton* m_button2;
		wxScrolledWindow* m_scrolledWindowForecasts;
		wxBoxSizer* m_sizerForecasts;
		wxBitmapButton* m_bpButtonAdd;
		wxMenuBar* m_menuBar;
		wxMenu* m_menuFile;
		wxMenu* m_menuOptions;
		wxMenu* m_menuTools;
		wxMenu* m_menuLog;
		wxMenu* m_menuLogLevel;
		wxMenu* m_menuHelp;
		wxStatusBar* m_statusBar1;
		wxToolBar* m_toolBar;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnSetPresentDate( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnConfigureDirectories( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddForecast( wxCommandEvent& event ) { event.Skip(); }
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
		
		asFrameMainVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("AtmoSwing Forecaster"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,700 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
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
		wxStaticText* m_staticTextDataParam;
		wxChoice* m_choiceDataParam;
		wxStaticText* m_staticTextDataTempResol;
		wxChoice* m_choiceDataTempResol;
		wxStaticText* m_staticTextDataSpatAggreg;
		wxChoice* m_choiceDataSpatAggreg;
		wxPanel* m_panelDataProcessing;
		wxCheckBox* m_checkBoxReturnPeriod;
		wxTextCtrl* m_textCtrlReturnPeriod;
		wxStaticText* m_staticTextYears;
		wxCheckBox* m_checkBoxSqrt;
		wxStaticText* m_staticTextCatalogPath;
		wxFilePickerCtrl* m_filePickerCatalogPath;
		wxStaticText* m_staticTextDataDir;
		wxDirPickerCtrl* m_dirPickerDataDir;
		wxStaticText* m_staticTextPatternsDir;
		wxDirPickerCtrl* m_dirPickerPatternsDir;
		wxStaticText* m_staticDestinationDir;
		wxDirPickerCtrl* m_dirPickerDestinationDir;
		wxButton* m_buttonSaveDefault;
		wxStdDialogButtonSizer* m_buttonsConfirmation;
		wxButton* m_buttonsConfirmationOK;
		wxButton* m_buttonsConfirmationCancel;
		
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
/// Class asPanelForecastVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelForecastVirtual : public wxPanel 
{
	private:
		wxBoxSizer* m_sizerFilename;
	
	protected:
		wxBoxSizer* m_sizerPanel;
		wxBoxSizer* m_sizerHeader;
		wxTextCtrl* m_textCtrlParametersFileName;
		wxBitmapButton* m_bpButtonClose;
		
		// Virtual event handlers, overide them in your derived class
		virtual void ClosePanel( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asPanelForecastVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL ); 
		~asPanelForecastVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesForecasterVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesForecasterVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelBase;
		wxNotebook* m_notebookBase;
		wxPanel* m_panelPathsCommon;
		wxBoxSizer* m_sizerPanelPaths;
		wxStaticText* m_staticTextParametersDir;
		wxDirPickerCtrl* m_dirPickerParameters;
		wxStaticText* m_staticTextPredictandDBDir;
		wxDirPickerCtrl* m_dirPickerPredictandDB;
		wxStaticText* m_staticTextArchivePredictorsDir;
		wxDirPickerCtrl* m_dirPickerArchivePredictors;
		wxStaticText* m_staticTextRealtimePredictorSavingDir;
		wxDirPickerCtrl* m_dirPickerRealtimePredictorSaving;
		wxStaticText* m_staticTextForecastResultsDir;
		wxDirPickerCtrl* m_dirPickerForecastResults;
		wxStaticText* m_staticTextForecastResultsExportsDir;
		wxDirPickerCtrl* m_dirPickerForecastResultsExports;
		wxCheckBox* m_checkBoxExportSyntheticXml;
		wxPanel* m_panelGeneralCommon;
		wxRadioBox* m_radioBoxLogLevel;
		wxCheckBox* m_checkBoxDisplayLogWindow;
		wxCheckBox* m_checkBoxSaveLogFile;
		wxCheckBox* m_checkBoxProxy;
		wxStaticText* m_staticTextProxyAddress;
		wxTextCtrl* m_textCtrlProxyAddress;
		wxStaticText* m_staticTextProxyPort;
		wxTextCtrl* m_textCtrlProxyPort;
		wxStaticText* m_staticTextProxyUser;
		wxTextCtrl* m_textCtrlProxyUser;
		wxStaticText* m_staticTextProxyPasswd;
		wxTextCtrl* m_textCtrlProxyPasswd;
		wxPanel* m_panelAdvanced;
		wxNotebook* m_notebookAdvanced;
		wxPanel* m_panelGeneral;
		wxRadioBox* m_radioBoxGui;
		wxStaticText* m_staticTextNumberFails;
		wxTextCtrl* m_textCtrlMaxPrevStepsNb;
		wxStaticText* m_staticTextMaxRequestsNb;
		wxTextCtrl* m_textCtrlMaxRequestsNb;
		wxCheckBox* m_checkBoxRestrictDownloads;
		wxCheckBox* m_checkBoxResponsiveness;
		wxCheckBox* m_checkBoxMultiInstancesForecaster;
		wxPanel* m_panelProcessing;
		wxCheckBox* m_checkBoxAllowMultithreading;
		wxStaticText* m_staticTextThreadsNb;
		wxTextCtrl* m_textCtrlThreadsNb;
		wxStaticText* m_staticTextThreadsPriority;
		wxSlider* m_sliderThreadsPriority;
		wxRadioBox* m_radioBoxProcessingMethods;
		wxPanel* m_panelUserDirectories;
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
		wxFilePickerCtrl* m_filePickerBatchFile;
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
