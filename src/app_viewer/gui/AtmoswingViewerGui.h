///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun  5 2014)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __ATMOSWINGVIEWERGUI_H__
#define __ATMOSWINGVIEWERGUI_H__

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
#include <wx/panel.h>
#include <wx/splitter.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/checklst.h>
#include <wx/choice.h>
#include <wx/notebook.h>
#include <wx/grid.h>
#include <wx/bmpbuttn.h>
#include <wx/filepicker.h>
#include <wx/statbox.h>
#include <wx/textctrl.h>
#include <wx/radiobox.h>
#include <wx/checkbox.h>
#include <wx/wizard.h>
#include <wx/dynarray.h>
WX_DEFINE_ARRAY_PTR( wxWizardPageSimple*, WizardPages );

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class asFrameForecastVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameForecastVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelMain;
		wxSplitterWindow* m_splitterGIS;
		wxScrolledWindow* m_scrolledWindowOptions;
		wxBoxSizer* m_sizerScrolledWindow;
		wxPanel* m_panelContent;
		wxBoxSizer* m_sizerContent;
		wxPanel* m_panelTop;
		wxBoxSizer* m_sizerTop;
		wxBoxSizer* m_sizerTopLeft;
		wxStaticText* m_staticTextForecastDate;
		wxButton* m_button51;
		wxButton* m_button5;
		wxButton* m_button6;
		wxButton* m_button61;
		wxStaticText* m_staticTextForecast;
		wxBoxSizer* m_sizerTopRight;
		wxBoxSizer* m_sizerLeadTimeSwitch;
		wxPanel* m_panelGIS;
		wxBoxSizer* m_sizerGIS;
		wxMenuBar* m_menuBar;
		wxMenu* m_menuFile;
		wxMenu* m_menuOptions;
		wxMenu* m_menuLog;
		wxMenu* m_menuLogLevel;
		wxMenu* m_menuHelp;
		wxToolBar* m_toolBar;
		wxStatusBar* m_statusBar;
		
		// Virtual event handlers, overide them in your derived class
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
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameForecastVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("AtmoSwing Viewer"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameForecastVirtual();
		
		void m_splitterGISOnIdle( wxIdleEvent& )
		{
			m_splitterGIS->SetSashPosition( 270 );
			m_splitterGIS->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFrameForecastVirtual::m_splitterGISOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePlotTimeSeriesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePlotTimeSeriesVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelStationName;
		wxStaticText* m_staticTextStationName;
		wxButton* m_buttonSaveTxt;
		wxButton* m_buttonPreview;
		wxButton* m_buttonPrint;
		wxSplitterWindow* m_splitter;
		wxPanel* m_panelLeft;
		wxCheckListBox* m_checkListToc;
		wxCheckListBox* m_checkListPast;
		wxPanel* m_panelRight;
		wxBoxSizer* m_sizerPlot;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnExportTXT( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreview( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPrint( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTocSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePlotTimeSeriesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Forecast plots"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,400 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxTAB_TRAVERSAL );
		
		~asFramePlotTimeSeriesVirtual();
		
		void m_splitterOnIdle( wxIdleEvent& )
		{
			m_splitter->SetSashPosition( 150 );
			m_splitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotTimeSeriesVirtual::m_splitterOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePlotDistributionsVirutal
///////////////////////////////////////////////////////////////////////////////
class asFramePlotDistributionsVirutal : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelOptions;
		wxStaticText* m_staticTextForecast;
		wxStaticText* m_staticTextStation;
		wxStaticText* m_staticTextDate;
		wxChoice* m_choiceForecast;
		wxChoice* m_choiceStation;
		wxChoice* m_choiceDate;
		wxNotebook* m_notebook;
		wxPanel* m_panelPredictands;
		wxSplitterWindow* m_splitterPredictands;
		wxPanel* m_panelPredictandsLeft;
		wxCheckListBox* m_checkListTocPredictands;
		wxPanel* m_panelPredictandsRight;
		wxBoxSizer* m_sizerPlotPredictands;
		wxPanel* m_panelCriteria;
		wxBoxSizer* m_sizerPlotCriteria;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnChoiceForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceStationChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTocSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePlotDistributionsVirutal( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Distribution plots"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 800,500 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePlotDistributionsVirutal();
		
		void m_splitterPredictandsOnIdle( wxIdleEvent& )
		{
			m_splitterPredictands->SetSashPosition( 170 );
			m_splitterPredictands->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::m_splitterPredictandsOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameGridAnalogsValuesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameGridAnalogsValuesVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelOptions;
		wxStaticText* m_staticTextForecast;
		wxChoice* m_choiceForecast;
		wxStaticText* m_staticTextStation;
		wxChoice* m_choiceStation;
		wxStaticText* m_staticTextDate;
		wxChoice* m_choiceDate;
		wxGrid* m_grid;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnChoiceForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceStationChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void SortGrid( wxGridEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Analogs details"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,500 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameGridAnalogsValuesVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePredictorsVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePredictorsVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panel15;
		wxSplitterWindow* m_splitterToc;
		wxScrolledWindow* m_scrolledWindowOptions;
		wxBoxSizer* m_sizerScrolledWindow;
		wxStaticText* m_staticTextChoiceForecast;
		wxChoice* m_choiceForecast;
		wxStaticText* m_staticTextCheckListPredictors;
		wxCheckListBox* m_checkListPredictors;
		wxStaticText* m_staticTextTocLeft;
		wxStaticText* m_staticTextTocRight;
		wxPanel* m_panelGIS;
		wxBoxSizer* m_sizerGIS;
		wxPanel* m_panelLeft;
		wxStaticText* m_staticTextTargetDates;
		wxChoice* m_choiceTargetDates;
		wxPanel* m_panelGISLeft;
		wxBoxSizer* m_sizerGISLeft;
		wxPanel* m_panelSwitch;
		wxBitmapButton* m_bpButtonSwitchRight;
		wxBitmapButton* m_bpButtonSwitchLeft;
		wxPanel* m_panelRight;
		wxStaticText* m_staticTextAnalogDates;
		wxChoice* m_choiceAnalogDates;
		wxPanel* m_panelGISRight;
		wxBoxSizer* m_sizerGISRight;
		wxMenuBar* m_menubar;
		wxMenu* m_menuFile;
		wxMenu* m_menuTools;
		wxToolBar* m_toolBar;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPredictorSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTargetDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSwitchRight( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSwitchLeft( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAnalogDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenLayer( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePredictorsVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Predictors overview"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 800,600 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePredictorsVirtual();
		
		void m_splitterTocOnIdle( wxIdleEvent& )
		{
			m_splitterToc->SetSashPosition( 170 );
			m_splitterToc->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::m_splitterTocOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelSidebarVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelSidebarVirtual : public wxPanel 
{
	private:
	
	protected:
		wxBoxSizer* m_sizerMain;
		wxPanel* m_panelHeader;
		wxStaticText* m_header;
		wxBitmapButton* m_bpButtonReduce;
		wxBoxSizer* m_sizerContent;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnReducePanel( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asPanelSidebarVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxSIMPLE_BORDER|wxTAB_TRAVERSAL ); 
		~asPanelSidebarVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesViewerVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesViewerVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panelBase;
		wxNotebook* m_notebookBase;
		wxPanel* m_panelWorkspace;
		wxStaticText* m_staticTextForecastResultsDir;
		wxDirPickerCtrl* m_dirPickerForecastResults;
		wxStaticText* m_staticTextColorbarMaxValue;
		wxTextCtrl* m_textCtrlColorbarMaxValue;
		wxStaticText* m_staticTextColorbarMaxUnit;
		wxStaticText* m_staticTextPastDaysNb;
		wxTextCtrl* m_textCtrlPastDaysNb;
		wxStaticText* m_staticTextAlarmsReturnPeriod;
		wxChoice* m_choiceAlarmsReturnPeriod;
		wxStaticText* m_staticTextAlarmsReturnPeriodYears;
		wxStaticText* m_staticTextAlarmsQuantile;
		wxTextCtrl* m_textCtrlAlarmsQuantile;
		wxStaticText* m_staticTextAlarmsQuantileRange;
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
		wxCheckBox* m_checkBoxMultiInstancesViewer;
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
		virtual void ApplyChanges( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void SaveAndClose( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePreferencesViewerVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Preferences"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 482,534 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePreferencesViewerVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asWizardWorkspaceVirtual
///////////////////////////////////////////////////////////////////////////////
class asWizardWorkspaceVirtual : public wxWizard 
{
	private:
	
	protected:
		wxStaticText* m_staticText37;
		wxStaticText* m_staticText35;
		wxButton* m_button4;
		wxStaticText* m_staticText46;
		wxStaticText* m_staticText36;
		wxStaticText* m_staticText43;
		wxFilePickerCtrl* m_filePickerWorkspaceFile;
		wxStaticText* m_staticText44;
		wxStaticText* m_staticTextForecastResultsDir;
		wxDirPickerCtrl* m_dirPickerForecastResults;
		wxStaticText* m_staticText42;
		wxStaticText* m_staticText45;
		wxStaticText* m_staticText40;
		wxChoice* m_choiceBaseMap;
		wxStaticText* m_staticText41;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnWizardFinished( wxWizardEvent& event ) { event.Skip(); }
		virtual void OnLoadExistingWorkspace( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asWizardWorkspaceVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Workspace creation wizard"), const wxBitmap& bitmap = wxNullBitmap, const wxPoint& pos = wxDefaultPosition, long style = wxDEFAULT_DIALOG_STYLE );
		WizardPages m_pages;
		~asWizardWorkspaceVirtual();
	
};

#endif //__ATMOSWINGVIEWERGUI_H__
