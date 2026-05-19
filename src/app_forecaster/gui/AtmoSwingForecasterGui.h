///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/intl.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/calctrl.h>
#include <wx/textctrl.h>
#include <wx/bmpbuttn.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/gauge.h>
#include <wx/scrolwin.h>
#include <wx/panel.h>
#include <wx/menu.h>
#include <wx/statusbr.h>
#include <wx/toolbar.h>
#include <wx/frame.h>
#include <wx/filepicker.h>
#include <wx/choice.h>
#include <wx/statbox.h>
#include <wx/radiobut.h>
#include <wx/checkbox.h>
#include <wx/radiobox.h>
#include <wx/slider.h>
#include <wx/notebook.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );
#include <wx/stc/stc.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameForecasterVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameForecasterVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panelMain;
		wxStaticText* _staticText41;
		wxCalendarCtrl* _calendarForecastDate;
		wxStaticText* _staticTextForecastHour;
		wxTextCtrl* _textCtrlForecastHour;
		wxBitmapButton* _bpButtonNow;
		wxGauge* _gauge;
		wxStaticText* _staticTextProgressActual;
		wxStaticText* _staticText38;
		wxStaticText* _staticTextProgressTot;
		wxFlexGridSizer* _sizerLeds;
		wxButton* _button2;
		wxStaticText* _staticText34;
		wxStaticText* _staticTextbatchFile;
		wxScrolledWindow* _scrolledWindowForecasts;
		wxBoxSizer* _sizerForecasts;
		wxBitmapButton* _bpButtonAdd;
		wxMenuBar* _menuBar;
		wxMenu* _menuFile;
		wxMenu* _menuOptions;
		wxMenu* _menuTools;
		wxMenu* _menuLog;
		wxMenu* _menuLogLevel;
		wxMenu* _menuHelp;
		wxStatusBar* _statusBar1;
		wxToolBar* _toolBar;

		// Virtual event handlers, override them in your derived class
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

		asFrameForecasterVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("AtmoSwing Forecaster"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 600,700 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFrameForecasterVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelForecastVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelForecastVirtual : public wxPanel
{
	private:

	protected:
		wxBoxSizer* _sizerPanel;
		wxBoxSizer* _sizerHeader;
		wxStaticText* _textParametersFileName;
		wxBitmapButton* _bpButtonWarning;
		wxBitmapButton* _bpButtonEdit;
		wxBitmapButton* _bpButtonInfo;
		wxBitmapButton* _bpButtonDetails;
		wxBitmapButton* _bpButtonClose;

		// Virtual event handlers, override them in your derived class
		virtual void OnEditForecastFile( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnDetailsForecastFile( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClosePanel( wxCommandEvent& event ) { event.Skip(); }


	public:

		asPanelForecastVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxBORDER_NONE|wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~asPanelForecastVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesForecasterVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesForecasterVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panelBase;
		wxNotebook* _notebookBase;
		wxPanel* _panelPathsCommon;
		wxBoxSizer* _sizerPanelPaths;
		wxStaticText* _staticTextParametersDir;
		wxDirPickerCtrl* _dirPickerParameters;
		wxStaticText* _staticTextPredictandDBDir;
		wxDirPickerCtrl* _dirPickerPredictandDB;
		wxStaticText* _staticTextArchivePredictorsDir;
		wxDirPickerCtrl* _dirPickerArchivePredictors;
		wxStaticText* _staticTextRealtimePredictorSavingDir;
		wxDirPickerCtrl* _dirPickerRealtimePredictorSaving;
		wxStaticText* _staticTextForecastResultsDir;
		wxDirPickerCtrl* _dirPickerForecastResults;
		wxStaticText* _staticTextForecastResultsExportsDir;
		wxDirPickerCtrl* _dirPickerForecastResultsExports;
		wxStaticText* _staticTextExport;
		wxChoice* _choiceExports;
		wxPanel* _panelGeneralCommon;
		wxChoice* _choiceLocale;
		wxStaticText* _staticText34;
		wxRadioButton* _radioBtnLogLevel1;
		wxRadioButton* _radioBtnLogLevel2;
		wxRadioButton* _radioBtnLogLevel3;
		wxCheckBox* _checkBoxDisplayLogWindow;
		wxCheckBox* _checkBoxSaveLogFile;
		wxCheckBox* _checkBoxProxy;
		wxStaticText* _staticTextProxyAddress;
		wxTextCtrl* _textCtrlProxyAddress;
		wxStaticText* _staticTextProxyPort;
		wxTextCtrl* _textCtrlProxyPort;
		wxStaticText* _staticTextProxyUser;
		wxTextCtrl* _textCtrlProxyUser;
		wxStaticText* _staticTextProxyPasswd;
		wxTextCtrl* _textCtrlProxyPasswd;
		wxStaticText* _staticTextEcCodesDefs;
		wxTextCtrl* _textCtrlEcCodesDefs;
		wxPanel* _panelAdvanced;
		wxNotebook* _notebookAdvanced;
		wxPanel* _panelGeneral;
		wxRadioBox* _radioBoxGui;
		wxStaticText* _staticTextNumberFails;
		wxTextCtrl* _textCtrlMaxPrevStepsNb;
		wxCheckBox* _checkBoxRestrictDownloads;
		wxCheckBox* _checkBoxResponsiveness;
		wxCheckBox* _checkBoxMultiInstancesForecaster;
		wxPanel* _panelProcessing;
		wxCheckBox* _checkBoxAllowMultithreading;
		wxStaticText* _staticTextThreadsNb;
		wxTextCtrl* _textCtrlThreadsNb;
		wxStaticText* _staticTextThreadsPriority;
		wxSlider* _sliderThreadsPriority;
		wxRadioBox* _radioBoxProcessingMethods;
		wxPanel* _panelUserDirectories;
		wxStaticText* _staticTextUserDirLabel;
		wxStaticText* _staticTextUserDir;
		wxStaticText* _staticTextLogFileLabel;
		wxStaticText* _staticTextLogFile;
		wxStaticText* _staticTextPrefFileLabel;
		wxStaticText* _staticTextPrefFile;
		wxStdDialogButtonSizer* _buttonsConfirmation;
		wxButton* _buttonsConfirmationOK;
		wxButton* _buttonsConfirmationApply;
		wxButton* _buttonsConfirmationCancel;

		// Virtual event handlers, override them in your derived class
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
		wxStaticText* _staticText37;
		wxStaticText* _staticText35;
		wxButton* _button4;
		wxStaticText* _staticText46;
		wxStaticText* _staticText36;
		wxStaticText* _staticText43;
		wxFilePickerCtrl* _filePickerBatchFile;
		wxStaticText* _staticText45;

		// Virtual event handlers, override them in your derived class
		virtual void OnWizardFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnLoadExistingBatchForecasts( wxCommandEvent& event ) { event.Skip(); }


	public:

		asWizardBatchForecastsVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Batch file creation wizard"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~asWizardBatchForecastsVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameStyledTextCtrlVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameStyledTextCtrlVirtual : public wxFrame
{
	private:

	protected:
		wxStyledTextCtrl* _scintilla;

	public:

		asFrameStyledTextCtrlVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,500 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFrameStyledTextCtrlVirtual();

};

