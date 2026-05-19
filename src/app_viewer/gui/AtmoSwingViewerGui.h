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
#include <wx/sizer.h>
#include <wx/gdicmn.h>
#include <wx/scrolwin.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/button.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/panel.h>
#include <wx/splitter.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/filepicker.h>
#include <wx/choice.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );
#include <wx/statbmp.h>
#include <wx/checklst.h>
#include <wx/notebook.h>
#include <wx/grid.h>
#include <wx/listbox.h>
#include <wx/bmpbuttn.h>
#include <wx/textctrl.h>
#include <wx/statbox.h>
#include <wx/radiobut.h>
#include <wx/checkbox.h>

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameViewerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameViewerVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panelMain;
		wxSplitterWindow* _splitterGIS;
		wxScrolledWindow* _scrolledWindowOptions;
		wxBoxSizer* _sizerScrolledWindow;
		wxPanel* _panelContent;
		wxBoxSizer* _sizerContent;
		wxPanel* _panelTop;
		wxBoxSizer* _sizerTop;
		wxBoxSizer* _sizerTopLeft;
		wxStaticText* _staticTextForecastDate;
		wxButton* _button51;
		wxButton* _button5;
		wxButton* _button6;
		wxButton* _button61;
		wxStaticText* _staticTextForecast;
		wxBoxSizer* _sizerTopRight;
		wxBoxSizer* _sizerLeadTimeSwitch;
		wxPanel* _panelGIS;
		wxBoxSizer* _sizerGIS;
		wxMenuBar* _menuBar;
		wxMenu* _menuFile;
		wxMenu* _menuOptions;
		wxMenu* _menuTools;
		wxMenu* _menuLog;
		wxMenu* _menuLogLevel;
		wxMenu* _menuHelp;
		wxToolBar* _toolBar;
		wxStatusBar* _statusBar;

		// Virtual event handlers, override them in your derived class
		virtual void OnLoadPreviousDay( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLoadPreviousForecast( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLoadNextForecast( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLoadNextDay( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenWorkspace( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveWorkspace( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveWorkspaceAs( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnNewWorkspace( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenForecast( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenLayer( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCloseLayer( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnMoveLayer( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnQuit( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePreferences( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePredictandDB( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }


	public:

		asFrameViewerVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("AtmoSwing Viewer"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFrameViewerVirtual();

		void _splitterGISOnIdle( wxIdleEvent& )
		{
			_splitterGIS->SetSashPosition( 270 );
			_splitterGIS->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFrameViewerVirtual::_splitterGISOnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class asWizardWorkspaceVirtual
///////////////////////////////////////////////////////////////////////////////
class asWizardWorkspaceVirtual : public wxWizard
{
	private:

	protected:
		wxStaticText* _staticText37;
		wxStaticText* _staticText35;
		wxButton* _button4;
		wxStaticText* _staticText46;
		wxStaticText* _staticText36;
		wxStaticText* _staticText43;
		wxFilePickerCtrl* _filePickerWorkspaceFile;
		wxStaticText* _staticText44;
		wxStaticText* _staticTextForecastResultsDir;
		wxDirPickerCtrl* _dirPickerForecastResults;
		wxStaticText* _staticText42;
		wxStaticText* _staticText45;
		wxStaticText* _staticText40;
		wxChoice* _choiceBaseMap;
		wxStaticText* _staticText41;

		// Virtual event handlers, override them in your derived class
		virtual void OnWizardFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnLoadExistingWorkspace( wxCommandEvent& event ) { event.Skip(); }


	public:

		asWizardWorkspaceVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Workspace creation wizard"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;

		~asWizardWorkspaceVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelSidebarVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelSidebarVirtual : public wxPanel
{
	private:

	protected:
		wxBoxSizer* _sizerMain;
		wxPanel* _panel28;
		wxPanel* _panelHeader;
		wxStaticText* _header;
		wxStaticBitmap* _bitmapCaret;
		wxBoxSizer* _sizerContent;

		// Virtual event handlers, override them in your derived class
		virtual void OnReducePanel( wxMouseEvent& event ) { event.Skip(); }


	public:

		asPanelSidebarVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~asPanelSidebarVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePlotTimeSeriesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePlotTimeSeriesVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panelStationName;
		wxStaticText* _staticTextStationName;
		wxButton* _buttonSaveTxt;
		wxButton* _buttonPreview;
		wxButton* _buttonPrint;
		wxButton* _buttonReset;
		wxSplitterWindow* _splitter;
		wxPanel* _panelLeft;
		wxCheckListBox* _checkListToc;
		wxCheckListBox* _checkListPast;
		wxPanel* _panelRight;
		wxBoxSizer* _sizerPlot;

		// Virtual event handlers, override them in your derived class
		virtual void OnExportTXT( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreview( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPrint( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetExtent( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTocSelectionChange( wxCommandEvent& event ) { event.Skip(); }


	public:

		asFramePlotTimeSeriesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Forecast plots"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,400 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxTAB_TRAVERSAL );

		~asFramePlotTimeSeriesVirtual();

		void _splitterOnIdle( wxIdleEvent& )
		{
			_splitter->SetSashPosition( 150 );
			_splitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotTimeSeriesVirtual::_splitterOnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePlotDistributionsVirutal
///////////////////////////////////////////////////////////////////////////////
class asFramePlotDistributionsVirutal : public wxFrame
{
	private:

	protected:
		wxPanel* _panelOptions;
		wxStaticText* _staticTextForecast;
		wxStaticText* _staticTextStation;
		wxStaticText* _staticTextDate;
		wxChoice* _choiceForecast;
		wxChoice* _choiceStation;
		wxChoice* _choiceDate;
		wxNotebook* _notebook;
		wxPanel* _panelPredictands;
		wxSplitterWindow* _splitter4;
		wxPanel* _panelPredictandsLeft;
		wxCheckListBox* _checkListTocPredictands;
		wxButton* _buttonResetZoom;
		wxPanel* _panelPredictandsRight;
		wxBoxSizer* _sizerPlotPredictands;
		wxPanel* _panelCriteria;
		wxBoxSizer* _sizerPlotCriteria;

		// Virtual event handlers, override them in your derived class
		virtual void OnChoiceForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceStationChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTocSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void ResetExtent( wxCommandEvent& event ) { event.Skip(); }


	public:

		asFramePlotDistributionsVirutal( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Distribution plots"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 900,500 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFramePlotDistributionsVirutal();

		void _splitter4OnIdle( wxIdleEvent& )
		{
			_splitter4->SetSashPosition( 178 );
			_splitter4->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::_splitter4OnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameGridAnalogsValuesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameGridAnalogsValuesVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panelOptions;
		wxStaticText* _staticTextForecast;
		wxChoice* _choiceForecast;
		wxStaticText* _staticTextStation;
		wxChoice* _choiceStation;
		wxStaticText* _staticTextDate;
		wxChoice* _choiceDate;
		wxGrid* _grid;

		// Virtual event handlers, override them in your derived class
		virtual void OnChoiceForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceStationChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void SortGrid( wxGridEvent& event ) { event.Skip(); }


	public:

		asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Analogs details"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,500 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFrameGridAnalogsValuesVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePredictorsVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePredictorsVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panel15;
		wxSplitterWindow* _splitterToc;
		wxScrolledWindow* _scrolledWindowOptions;
		wxBoxSizer* _sizerScrolledWindow;
		wxStaticText* _staticTextChoiceMethod;
		wxChoice* _choiceMethod;
		wxStaticText* _staticTextChoiceForecast;
		wxChoice* _choiceForecast;
		wxStaticText* _staticTextCheckListPredictors;
		wxListBox* _listPredictors;
		wxStaticText* _staticTextTocLeft;
		wxStaticText* _staticTextTocRight;
		wxPanel* _panelGIS;
		wxBoxSizer* _sizerGIS;
		wxPanel* _panelLeft;
		wxStaticText* _staticTextTargetDates;
		wxChoice* _choiceTargetDates;
		wxPanel* _panelGISLeft;
		wxBoxSizer* _sizerGISLeft;
		wxPanel* _panelColorbarLeft;
		wxBoxSizer* _sizerColorbarLeft;
		wxPanel* _panelSwitch;
		wxBitmapButton* _bpButtonSwitchRight;
		wxBitmapButton* _bpButtonSwitchLeft;
		wxPanel* _panelRight;
		wxStaticText* _staticTextAnalogDates;
		wxChoice* _choiceAnalogDates;
		wxPanel* _panelGISRight;
		wxBoxSizer* _sizerGISRight;
		wxPanel* _panelColorbarRight;
		wxBoxSizer* _sizerColorbarRight;
		wxMenuBar* _menubar;
		wxMenu* _menuFile;
		wxMenu* _menuTools;
		wxToolBar* _toolBar;

		// Virtual event handlers, override them in your derived class
		virtual void OnMethodChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPredictorSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTargetDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSwitchRight( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSwitchLeft( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAnalogDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenLayer( wxCommandEvent& event ) { event.Skip(); }


	public:

		asFramePredictorsVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Predictors overview"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFramePredictorsVirtual();

		void _splitterTocOnIdle( wxIdleEvent& )
		{
			_splitterToc->SetSashPosition( 220 );
			_splitterToc->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::_splitterTocOnIdle ), NULL, this );
		}

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesViewerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesViewerVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panelBase;
		wxNotebook* _notebookBase;
		wxPanel* _panelWorkspace;
		wxStaticText* _staticTextForecastResultsDir;
		wxDirPickerCtrl* _dirPickerForecastResults;
		wxStaticText* _staticTextColorbarMaxValue;
		wxTextCtrl* _textCtrlColorbarMaxValue;
		wxStaticText* _staticTextColorbarMaxUnit;
		wxStaticText* _staticTextPastDaysNb;
		wxTextCtrl* _textCtrlPastDaysNb;
		wxStaticText* _staticTextAlarmsReturnPeriod;
		wxChoice* _choiceAlarmsReturnPeriod;
		wxStaticText* _staticTextAlarmsReturnPeriodYears;
		wxStaticText* _staticTextAlarmsQuantile;
		wxTextCtrl* _textCtrlAlarmsQuantile;
		wxStaticText* _staticTextAlarmsQuantileRange;
		wxStaticText* _staticText581;
		wxStaticText* _staticText541;
		wxTextCtrl* _textCtrlMaxLengthDaily;
		wxStaticText* _staticText56;
		wxStaticText* _staticText55;
		wxTextCtrl* _textCtrlMaxLengthSubDaily;
		wxStaticText* _staticText571;
		wxPanel* _panelPaths;
		wxStaticText* _staticPredictorID;
		wxStaticText* _staticPredictorPaths;
		wxTextCtrl* _textCtrlDatasetId1;
		wxDirPickerCtrl* _dirPickerDataset1;
		wxTextCtrl* _textCtrlDatasetId2;
		wxDirPickerCtrl* _dirPickerDataset2;
		wxTextCtrl* _textCtrlDatasetId3;
		wxDirPickerCtrl* _dirPickerDataset3;
		wxTextCtrl* _textCtrlDatasetId4;
		wxDirPickerCtrl* _dirPickerDataset4;
		wxTextCtrl* _textCtrlDatasetId5;
		wxDirPickerCtrl* _dirPickerDataset5;
		wxTextCtrl* _textCtrlDatasetId6;
		wxDirPickerCtrl* _dirPickerDataset6;
		wxTextCtrl* _textCtrlDatasetId7;
		wxDirPickerCtrl* _dirPickerDataset7;
		wxPanel* _panelColors;
		wxStaticText* _staticText54;
		wxFilePickerCtrl* _filePickerColorZ;
		wxStaticText* RelativeHumidity;
		wxFilePickerCtrl* _filePickerColorPwat;
		wxStaticText* _staticText57;
		wxFilePickerCtrl* _filePickerColorRh;
		wxStaticText* _staticText58;
		wxFilePickerCtrl* _filePickerColorSh;
		wxPanel* _panelGeneralCommon;
		wxChoice* _choiceLocale;
		wxStaticText* _staticText53;
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
		wxPanel* _panelAdvanced;
		wxCheckBox* _checkBoxMultiInstancesViewer;
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
		virtual void ApplyChanges( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void SaveAndClose( wxCommandEvent& event ) { event.Skip(); }


	public:

		asFramePreferencesViewerVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Preferences"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 482,534 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFramePreferencesViewerVirtual();

};

