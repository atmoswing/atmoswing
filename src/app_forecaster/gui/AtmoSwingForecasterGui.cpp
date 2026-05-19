///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif //WX_PRECOMP

#include "AtmoSwingForecasterGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameForecasterVirtual::asFrameForecasterVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,700 ), wxDefaultSize );

	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );

	_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer18;
	bSizer18 = new wxBoxSizer( wxVERTICAL );

	_staticText41 = new wxStaticText( _panelMain, wxID_ANY, _("Start the forecast for a given date"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText41->Wrap( -1 );
	_staticText41->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer18->Add( _staticText41, 0, wxALL|wxEXPAND, 10 );

	wxBoxSizer* bSizer19;
	bSizer19 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer36;
	bSizer36 = new wxBoxSizer( wxVERTICAL );

	_calendarForecastDate = new wxCalendarCtrl( _panelMain, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxDefaultSize, wxCAL_MONDAY_FIRST|wxCAL_SHOW_HOLIDAYS|wxCAL_SHOW_SURROUNDING_WEEKS );
	bSizer36->Add( _calendarForecastDate, 0, wxALL, 5 );

	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextForecastHour = new wxStaticText( _panelMain, wxID_ANY, _("Hour (UTM)"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecastHour->Wrap( -1 );
	bSizer35->Add( _staticTextForecastHour, 0, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlForecastHour = new wxTextCtrl( _panelMain, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	#ifdef __WXGTK__
	if ( !_textCtrlForecastHour->HasFlag( wxTE_MULTILINE ) )
	{
	_textCtrlForecastHour->SetMaxLength( 2 );
	}
	#else
	_textCtrlForecastHour->SetMaxLength( 2 );
	#endif
	bSizer35->Add( _textCtrlForecastHour, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_bpButtonNow = new wxBitmapButton( _panelMain, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	_bpButtonNow->SetToolTip( _("Set current date.") );

	bSizer35->Add( _bpButtonNow, 0, wxTOP|wxBOTTOM|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer36->Add( bSizer35, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	bSizer19->Add( bSizer36, 0, wxRIGHT, 5 );

	wxBoxSizer* bSizer341;
	bSizer341 = new wxBoxSizer( wxVERTICAL );

	_gauge = new wxGauge( _panelMain, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	_gauge->SetValue( 0 );
	bSizer341->Add( _gauge, 0, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer351;
	bSizer351 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextProgressActual = new wxStaticText( _panelMain, wxID_ANY, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextProgressActual->Wrap( -1 );
	bSizer351->Add( _staticTextProgressActual, 0, wxTOP|wxBOTTOM, 5 );

	_staticText38 = new wxStaticText( _panelMain, wxID_ANY, _("/"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText38->Wrap( -1 );
	bSizer351->Add( _staticText38, 0, wxALL, 5 );

	_staticTextProgressTot = new wxStaticText( _panelMain, wxID_ANY, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextProgressTot->Wrap( -1 );
	bSizer351->Add( _staticTextProgressTot, 0, wxTOP|wxBOTTOM, 5 );


	bSizer341->Add( bSizer351, 0, wxALIGN_CENTER_HORIZONTAL, 5 );

	_sizerLeds = new wxFlexGridSizer( 4, 2, 0, 0 );
	_sizerLeds->SetFlexibleDirection( wxBOTH );
	_sizerLeds->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );


	bSizer341->Add( _sizerLeds, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );

	_button2 = new wxButton( _panelMain, wxID_ANY, _("Configure directories"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer38->Add( _button2, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );


	bSizer341->Add( bSizer38, 0, wxEXPAND, 5 );


	bSizer19->Add( bSizer341, 1, wxLEFT|wxEXPAND, 5 );


	bSizer18->Add( bSizer19, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxBoxSizer* bSizer352;
	bSizer352 = new wxBoxSizer( wxHORIZONTAL );

	_staticText34 = new wxStaticText( _panelMain, wxID_ANY, _("Opened batch file:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText34->Wrap( -1 );
	bSizer352->Add( _staticText34, 0, wxALL, 5 );

	_staticTextbatchFile = new wxStaticText( _panelMain, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextbatchFile->Wrap( -1 );
	bSizer352->Add( _staticTextbatchFile, 0, wxALL, 5 );


	bSizer18->Add( bSizer352, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );

	bSizer22->SetMinSize( wxSize( -1,200 ) );
	_scrolledWindowForecasts = new wxScrolledWindow( _panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	_scrolledWindowForecasts->SetScrollRate( 5, 5 );
	_scrolledWindowForecasts->SetBackgroundColour( wxColour( 144, 144, 144 ) );
	_scrolledWindowForecasts->SetMinSize( wxSize( -1,200 ) );

	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );

	_sizerForecasts = new wxBoxSizer( wxVERTICAL );


	bSizer32->Add( _sizerForecasts, 0, wxEXPAND|wxTOP, 5 );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );

	_bpButtonAdd = new wxBitmapButton( _scrolledWindowForecasts, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 28,28 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	_bpButtonAdd->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	_bpButtonAdd->SetToolTip( _("Add a parameters file.") );

	bSizer34->Add( _bpButtonAdd, 0, wxALL, 8 );


	bSizer32->Add( bSizer34, 0, wxLEFT, 5 );


	_scrolledWindowForecasts->SetSizer( bSizer32 );
	_scrolledWindowForecasts->Layout();
	bSizer32->Fit( _scrolledWindowForecasts );
	bSizer22->Add( _scrolledWindowForecasts, 1, wxEXPAND|wxTOP, 5 );


	bSizer18->Add( bSizer22, 1, wxEXPAND, 5 );


	_panelMain->SetSizer( bSizer18 );
	_panelMain->Layout();
	bSizer18->Fit( _panelMain );
	bSizer3->Add( _panelMain, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer3 );
	this->Layout();
	_menuBar = new wxMenuBar( 0 );
	_menuFile = new wxMenu();
	wxMenuItem* _menuItemOpenBatchFile;
	_menuItemOpenBatchFile = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Open a batch file") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemOpenBatchFile );

	wxMenuItem* _menuItemSaveBatchFile;
	_menuItemSaveBatchFile = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Save batch file") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemSaveBatchFile );

	wxMenuItem* _menuItemSaveBatchFileAs;
	_menuItemSaveBatchFileAs = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Save batch file as") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemSaveBatchFileAs );

	wxMenuItem* _menuItemNewBatchFile;
	_menuItemNewBatchFile = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Create a new batch file") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemNewBatchFile );

	_menuBar->Append( _menuFile, _("File") );

	_menuOptions = new wxMenu();
	wxMenuItem* _menuItemPreferences;
	_menuItemPreferences = new wxMenuItem( _menuOptions, wxID_ANY, wxString( _("Preferences") ) , wxEmptyString, wxITEM_NORMAL );
	_menuOptions->Append( _menuItemPreferences );

	_menuBar->Append( _menuOptions, _("Options") );

	_menuTools = new wxMenu();
	wxMenuItem* _menuItemBuildPredictandDB;
	_menuItemBuildPredictandDB = new wxMenuItem( _menuTools, wxID_ANY, wxString( _("Build predictand DB") ) , wxEmptyString, wxITEM_NORMAL );
	_menuTools->Append( _menuItemBuildPredictandDB );

	_menuBar->Append( _menuTools, _("Tools") );

	_menuLog = new wxMenu();
	wxMenuItem* _menuItemShowLog;
	_menuItemShowLog = new wxMenuItem( _menuLog, wxID_ANY, wxString( _("Show Log Window") ) , wxEmptyString, wxITEM_NORMAL );
	_menuLog->Append( _menuItemShowLog );

	_menuLogLevel = new wxMenu();
	wxMenuItem* _menuLogLevelItem = new wxMenuItem( _menuLog, wxID_ANY, _("Log level"), wxEmptyString, wxITEM_NORMAL, _menuLogLevel );
	wxMenuItem* _menuItemLogLevel1;
	_menuItemLogLevel1 = new wxMenuItem( _menuLogLevel, wxID_ANY, wxString( _("Only errors") ) , wxEmptyString, wxITEM_CHECK );
	_menuLogLevel->Append( _menuItemLogLevel1 );

	wxMenuItem* _menuItemLogLevel2;
	_menuItemLogLevel2 = new wxMenuItem( _menuLogLevel, wxID_ANY, wxString( _("Errors and warnings") ) , wxEmptyString, wxITEM_CHECK );
	_menuLogLevel->Append( _menuItemLogLevel2 );

	wxMenuItem* _menuItemLogLevel3;
	_menuItemLogLevel3 = new wxMenuItem( _menuLogLevel, wxID_ANY, wxString( _("Verbose") ) , wxEmptyString, wxITEM_CHECK );
	_menuLogLevel->Append( _menuItemLogLevel3 );

	_menuLog->Append( _menuLogLevelItem );

	_menuBar->Append( _menuLog, _("Log") );

	_menuHelp = new wxMenu();
	wxMenuItem* _menuItemAbout;
	_menuItemAbout = new wxMenuItem( _menuHelp, wxID_ANY, wxString( _("About") ) , wxEmptyString, wxITEM_NORMAL );
	_menuHelp->Append( _menuItemAbout );

	_menuBar->Append( _menuHelp, _("Help") );

	this->SetMenuBar( _menuBar );

	_statusBar1 = this->CreateStatusBar( 1, wxSTB_SIZEGRIP, wxID_ANY );
	_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	_toolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	_toolBar->Realize();


	this->Centre( wxBOTH );

	// Connect Events
	_bpButtonNow->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnSetPresentDate ), NULL, this );
	_button2->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnConfigureDirectories ), NULL, this );
	_bpButtonAdd->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::AddForecast ), NULL, this );
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnOpenBatchForecasts ), this, _menuItemOpenBatchFile->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnSaveBatchForecasts ), this, _menuItemSaveBatchFile->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnSaveBatchForecastsAs ), this, _menuItemSaveBatchFileAs->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnNewBatchForecasts ), this, _menuItemNewBatchFile->GetId());
	_menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OpenFramePreferences ), this, _menuItemPreferences->GetId());
	_menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OpenFramePredictandDB ), this, _menuItemBuildPredictandDB->GetId());
	_menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnShowLog ), this, _menuItemShowLog->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnLogLevel1 ), this, _menuItemLogLevel1->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnLogLevel2 ), this, _menuItemLogLevel2->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnLogLevel3 ), this, _menuItemLogLevel3->GetId());
	_menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OpenFrameAbout ), this, _menuItemAbout->GetId());
}

asFrameForecasterVirtual::~asFrameForecasterVirtual()
{
	// Disconnect Events
	_bpButtonNow->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnSetPresentDate ), NULL, this );
	_button2->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnConfigureDirectories ), NULL, this );
	_bpButtonAdd->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::AddForecast ), NULL, this );

}

asPanelForecastVirtual::asPanelForecastVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	_sizerPanel = new wxBoxSizer( wxVERTICAL );

	_sizerHeader = new wxBoxSizer( wxHORIZONTAL );

	_textParametersFileName = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	_textParametersFileName->Wrap( -1 );
	_sizerHeader->Add( _textParametersFileName, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_bpButtonWarning = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0|wxBORDER_NONE );
	_bpButtonWarning->SetToolTip( _("File not found") );

	_sizerHeader->Add( _bpButtonWarning, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );

	_bpButtonEdit = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0|wxBORDER_NONE );
	_bpButtonEdit->SetToolTip( _("Edit path") );

	_sizerHeader->Add( _bpButtonEdit, 0, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxLEFT, 5 );

	_bpButtonInfo = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0|wxBORDER_NONE );
	_sizerHeader->Add( _bpButtonInfo, 0, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxLEFT, 5 );

	_bpButtonDetails = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW|0|wxBORDER_NONE );
	_bpButtonDetails->SetToolTip( _("See details") );

	_sizerHeader->Add( _bpButtonDetails, 0, wxALL, 5 );

	_bpButtonClose = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	_bpButtonClose->SetToolTip( _("Close") );

	_sizerHeader->Add( _bpButtonClose, 0, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxLEFT, 5 );


	_sizerPanel->Add( _sizerHeader, 0, wxEXPAND, 5 );


	this->SetSizer( _sizerPanel );
	this->Layout();
	_sizerPanel->Fit( this );

	// Connect Events
	_bpButtonEdit->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::OnEditForecastFile ), NULL, this );
	_bpButtonDetails->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::OnDetailsForecastFile ), NULL, this );
	_bpButtonClose->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::ClosePanel ), NULL, this );
}

asPanelForecastVirtual::~asPanelForecastVirtual()
{
	// Disconnect Events
	_bpButtonEdit->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::OnEditForecastFile ), NULL, this );
	_bpButtonDetails->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::OnDetailsForecastFile ), NULL, this );
	_bpButtonClose->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::ClosePanel ), NULL, this );

}

asFramePreferencesForecasterVirtual::asFramePreferencesForecasterVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );

	_panelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );

	_notebookBase = new wxNotebook( _panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelPathsCommon = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerPanelPaths = new wxBoxSizer( wxVERTICAL );

	_staticTextParametersDir = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Directory containing the parameters files"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextParametersDir->Wrap( -1 );
	_sizerPanelPaths->Add( _staticTextParametersDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerParameters = new wxDirPickerCtrl( _panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerPanelPaths->Add( _dirPickerParameters, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	_staticTextPredictandDBDir = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Directory containing the predictand DB"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPredictandDBDir->Wrap( -1 );
	_sizerPanelPaths->Add( _staticTextPredictandDBDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerPredictandDB = new wxDirPickerCtrl( _panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerPanelPaths->Add( _dirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextArchivePredictorsDir = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextArchivePredictorsDir->Wrap( -1 );
	_sizerPanelPaths->Add( _staticTextArchivePredictorsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerArchivePredictors = new wxDirPickerCtrl( _panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerPanelPaths->Add( _dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	_staticTextRealtimePredictorSavingDir = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Directory to save downloaded real-time predictors (GCM forecasts)"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextRealtimePredictorSavingDir->Wrap( -1 );
	_sizerPanelPaths->Add( _staticTextRealtimePredictorSavingDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerRealtimePredictorSaving = new wxDirPickerCtrl( _panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerPanelPaths->Add( _dirPickerRealtimePredictorSaving, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextForecastResultsDir = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Directory to save forecast outputs (netCDF)"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecastResultsDir->Wrap( -1 );
	_sizerPanelPaths->Add( _staticTextForecastResultsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerForecastResults = new wxDirPickerCtrl( _panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerPanelPaths->Add( _dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextForecastResultsExportsDir = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Directory to save forecast exports (xml)"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecastResultsExportsDir->Wrap( -1 );
	_sizerPanelPaths->Add( _staticTextForecastResultsExportsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerForecastResultsExports = new wxDirPickerCtrl( _panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerPanelPaths->Add( _dirPickerForecastResultsExports, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextExport = new wxStaticText( _panelPathsCommon, wxID_ANY, _("Export:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextExport->Wrap( -1 );
	bSizer33->Add( _staticTextExport, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString _choiceExportsChoices[] = { _("None"), _("Full XML"), _("Small CSV"), _("Custom CSV for FVG") };
	int _choiceExportsNChoices = sizeof( _choiceExportsChoices ) / sizeof( wxString );
	_choiceExports = new wxChoice( _panelPathsCommon, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceExportsNChoices, _choiceExportsChoices, 0 );
	_choiceExports->SetSelection( 0 );
	bSizer33->Add( _choiceExports, 0, wxALL, 5 );


	_sizerPanelPaths->Add( bSizer33, 0, wxEXPAND, 5 );


	_panelPathsCommon->SetSizer( _sizerPanelPaths );
	_panelPathsCommon->Layout();
	_sizerPanelPaths->Fit( _panelPathsCommon );
	_notebookBase->AddPage( _panelPathsCommon, _("Batch file properties"), true );
	_panelGeneralCommon = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer16;
	bSizer16 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer71;
	sbSizer71 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Language") ), wxVERTICAL );

	wxString _choiceLocaleChoices[] = { _("English"), _("French") };
	int _choiceLocaleNChoices = sizeof( _choiceLocaleChoices ) / sizeof( wxString );
	_choiceLocale = new wxChoice( sbSizer71->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceLocaleNChoices, _choiceLocaleChoices, 0 );
	_choiceLocale->SetSelection( 0 );
	sbSizer71->Add( _choiceLocale, 0, wxALL, 5 );

	_staticText34 = new wxStaticText( sbSizer71->GetStaticBox(), wxID_ANY, _("Restart AtmoSwing for the change to take effect."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText34->Wrap( -1 );
	sbSizer71->Add( _staticText34, 0, wxALL, 5 );


	bSizer16->Add( sbSizer71, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );

	_radioBtnLogLevel1 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer38->Add( _radioBtnLogLevel1, 0, wxALL, 5 );

	_radioBtnLogLevel2 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer38->Add( _radioBtnLogLevel2, 0, wxALL, 5 );

	_radioBtnLogLevel3 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer38->Add( _radioBtnLogLevel3, 0, wxALL, 5 );


	bSizer20->Add( bSizer38, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );

	_checkBoxDisplayLogWindow = new wxCheckBox( sbSizer7->GetStaticBox(), wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxDisplayLogWindow->SetValue(true);
	bSizer21->Add( _checkBoxDisplayLogWindow, 0, wxALL, 5 );

	_checkBoxSaveLogFile = new wxCheckBox( sbSizer7->GetStaticBox(), wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxSaveLogFile->SetValue(true);
	_checkBoxSaveLogFile->Enable( false );

	bSizer21->Add( _checkBoxSaveLogFile, 0, wxALL, 5 );


	bSizer20->Add( bSizer21, 1, wxEXPAND, 5 );


	sbSizer7->Add( bSizer20, 1, wxEXPAND, 5 );


	bSizer16->Add( sbSizer7, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer14;
	sbSizer14 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Proxy configuration") ), wxVERTICAL );

	_checkBoxProxy = new wxCheckBox( sbSizer14->GetStaticBox(), wxID_ANY, _("Internet connection uses a proxy"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer14->Add( _checkBoxProxy, 0, wxALL, 5 );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextProxyAddress = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Proxy address"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextProxyAddress->Wrap( -1 );
	bSizer34->Add( _staticTextProxyAddress, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlProxyAddress = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 180,-1 ), 0 );
	bSizer34->Add( _textCtrlProxyAddress, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextProxyPort = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Port"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextProxyPort->Wrap( -1 );
	bSizer34->Add( _staticTextProxyPort, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlProxyPort = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer34->Add( _textCtrlProxyPort, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer14->Add( bSizer34, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextProxyUser = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Username"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextProxyUser->Wrap( -1 );
	bSizer35->Add( _staticTextProxyUser, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlProxyUser = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	bSizer35->Add( _textCtrlProxyUser, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextProxyPasswd = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Password"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextProxyPasswd->Wrap( -1 );
	bSizer35->Add( _staticTextProxyPasswd, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlProxyPasswd = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD );
	bSizer35->Add( _textCtrlProxyPasswd, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer14->Add( bSizer35, 1, wxEXPAND, 5 );


	bSizer16->Add( sbSizer14, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Libraries") ), wxVERTICAL );

	wxBoxSizer* bSizer341;
	bSizer341 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextEcCodesDefs = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("ecCodes definitions"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextEcCodesDefs->Wrap( -1 );
	bSizer341->Add( _staticTextEcCodesDefs, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlEcCodesDefs = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer341->Add( _textCtrlEcCodesDefs, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer6->Add( bSizer341, 1, wxEXPAND, 5 );


	bSizer16->Add( sbSizer6, 0, wxEXPAND|wxALL, 5 );


	_panelGeneralCommon->SetSizer( bSizer16 );
	_panelGeneralCommon->Layout();
	bSizer16->Fit( _panelGeneralCommon );
	_notebookBase->AddPage( _panelGeneralCommon, _("General options"), false );
	_panelAdvanced = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	_notebookAdvanced = new wxNotebook( _panelAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelGeneral = new wxPanel( _notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer271;
	bSizer271 = new wxBoxSizer( wxVERTICAL );

	wxString _radioBoxGuiChoices[] = { _("Silent (no progressbar, much faster)"), _("Standard (recommanded)"), _("Verbose (not much used)") };
	int _radioBoxGuiNChoices = sizeof( _radioBoxGuiChoices ) / sizeof( wxString );
	_radioBoxGui = new wxRadioBox( _panelGeneral, wxID_ANY, _("GUI options"), wxDefaultPosition, wxDefaultSize, _radioBoxGuiNChoices, _radioBoxGuiChoices, 1, wxRA_SPECIFY_COLS );
	_radioBoxGui->SetSelection( 1 );
	bSizer271->Add( _radioBoxGui, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer11;
	sbSizer11 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneral, wxID_ANY, _("Predictor download") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextNumberFails = new wxStaticText( sbSizer11->GetStaticBox(), wxID_ANY, _("Maximum number of previous time steps if download fails"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextNumberFails->Wrap( -1 );
	fgSizer2->Add( _staticTextNumberFails, 0, wxALL, 5 );

	_textCtrlMaxPrevStepsNb = new wxTextCtrl( sbSizer11->GetStaticBox(), wxID_ANY, _("5"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	#ifdef __WXGTK__
	if ( !_textCtrlMaxPrevStepsNb->HasFlag( wxTE_MULTILINE ) )
	{
	_textCtrlMaxPrevStepsNb->SetMaxLength( 1 );
	}
	#else
	_textCtrlMaxPrevStepsNb->SetMaxLength( 1 );
	#endif
	fgSizer2->Add( _textCtrlMaxPrevStepsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_checkBoxRestrictDownloads = new wxCheckBox( sbSizer11->GetStaticBox(), wxID_ANY, _("Restrict downloads to needed lead times."), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxRestrictDownloads->SetValue(true);
	_checkBoxRestrictDownloads->Enable( false );
	_checkBoxRestrictDownloads->Hide();

	fgSizer2->Add( _checkBoxRestrictDownloads, 0, wxALL, 5 );


	sbSizer11->Add( fgSizer2, 1, wxEXPAND, 5 );


	bSizer271->Add( sbSizer11, 0, wxALL|wxEXPAND, 5 );

	_checkBoxResponsiveness = new wxCheckBox( _panelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxResponsiveness->SetValue(true);
	bSizer271->Add( _checkBoxResponsiveness, 0, wxALL, 5 );

	_checkBoxMultiInstancesForecaster = new wxCheckBox( _panelGeneral, wxID_ANY, _("Allow multiple instances of the forecaster"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer271->Add( _checkBoxMultiInstancesForecaster, 0, wxALL, 5 );


	_panelGeneral->SetSizer( bSizer271 );
	_panelGeneral->Layout();
	bSizer271->Fit( _panelGeneral );
	_notebookAdvanced->AddPage( _panelGeneral, _("General"), true );
	_panelProcessing = new wxPanel( _notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer1611;
	bSizer1611 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer15;
	sbSizer15 = new wxStaticBoxSizer( new wxStaticBox( _panelProcessing, wxID_ANY, _("Multithreading") ), wxVERTICAL );

	_checkBoxAllowMultithreading = new wxCheckBox( sbSizer15->GetStaticBox(), wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxAllowMultithreading->SetValue(true);
	sbSizer15->Add( _checkBoxAllowMultithreading, 0, wxALL, 5 );

	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextThreadsNb = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Max nb of threads"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextThreadsNb->Wrap( -1 );
	bSizer221->Add( _staticTextThreadsNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlThreadsNb = new wxTextCtrl( sbSizer15->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 30,-1 ), 0 );
	bSizer221->Add( _textCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer15->Add( bSizer221, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextThreadsPriority = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( _staticTextThreadsPriority, 0, wxALL, 5 );

	_sliderThreadsPriority = new wxSlider( sbSizer15->GetStaticBox(), wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( _sliderThreadsPriority, 1, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer15->Add( bSizer241, 0, wxEXPAND, 5 );


	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );

	wxString _radioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Standard (slower)") };
	int _radioBoxProcessingMethodsNChoices = sizeof( _radioBoxProcessingMethodsChoices ) / sizeof( wxString );
	_radioBoxProcessingMethods = new wxRadioBox( _panelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition, wxDefaultSize, _radioBoxProcessingMethodsNChoices, _radioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS );
	_radioBoxProcessingMethods->SetSelection( 0 );
	_radioBoxProcessingMethods->SetToolTip( _("These options don't affect the results, only the processor efficiency.") );

	bSizer1611->Add( _radioBoxProcessingMethods, 0, wxALL|wxEXPAND, 5 );


	_panelProcessing->SetSizer( bSizer1611 );
	_panelProcessing->Layout();
	bSizer1611->Fit( _panelProcessing );
	_notebookAdvanced->AddPage( _panelProcessing, _("Processing"), false );
	_panelUserDirectories = new wxPanel( _notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer24;
	bSizer24 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( _panelUserDirectories, wxID_ANY, _("User specific paths") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer9;
	fgSizer9 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer9->SetFlexibleDirection( wxBOTH );
	fgSizer9->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextUserDirLabel = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("User working directory:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextUserDirLabel->Wrap( -1 );
	fgSizer9->Add( _staticTextUserDirLabel, 0, wxALL, 5 );

	_staticTextUserDir = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextUserDir->Wrap( -1 );
	fgSizer9->Add( _staticTextUserDir, 0, wxALL, 5 );

	_staticTextLogFileLabel = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextLogFileLabel->Wrap( -1 );
	fgSizer9->Add( _staticTextLogFileLabel, 0, wxALL, 5 );

	_staticTextLogFile = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextLogFile->Wrap( -1 );
	fgSizer9->Add( _staticTextLogFile, 0, wxALL, 5 );

	_staticTextPrefFileLabel = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("Preferences file:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPrefFileLabel->Wrap( -1 );
	fgSizer9->Add( _staticTextPrefFileLabel, 0, wxALL, 5 );

	_staticTextPrefFile = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPrefFile->Wrap( -1 );
	fgSizer9->Add( _staticTextPrefFile, 0, wxALL, 5 );


	sbSizer17->Add( fgSizer9, 1, wxEXPAND, 5 );


	bSizer24->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );


	_panelUserDirectories->SetSizer( bSizer24 );
	_panelUserDirectories->Layout();
	bSizer24->Fit( _panelUserDirectories );
	_notebookAdvanced->AddPage( _panelUserDirectories, _("User paths"), false );

	bSizer26->Add( _notebookAdvanced, 1, wxEXPAND | wxALL, 5 );


	_panelAdvanced->SetSizer( bSizer26 );
	_panelAdvanced->Layout();
	bSizer26->Fit( _panelAdvanced );
	_notebookBase->AddPage( _panelAdvanced, _("Advanced options"), false );

	bSizer15->Add( _notebookBase, 1, wxEXPAND | wxALL, 5 );

	_buttonsConfirmation = new wxStdDialogButtonSizer();
	_buttonsConfirmationOK = new wxButton( _panelBase, wxID_OK );
	_buttonsConfirmation->AddButton( _buttonsConfirmationOK );
	_buttonsConfirmationApply = new wxButton( _panelBase, wxID_APPLY );
	_buttonsConfirmation->AddButton( _buttonsConfirmationApply );
	_buttonsConfirmationCancel = new wxButton( _panelBase, wxID_CANCEL );
	_buttonsConfirmation->AddButton( _buttonsConfirmationCancel );
	_buttonsConfirmation->Realize();

	bSizer15->Add( _buttonsConfirmation, 0, wxEXPAND|wxALL, 5 );


	_panelBase->SetSizer( bSizer15 );
	_panelBase->Layout();
	bSizer15->Fit( _panelBase );
	bSizer14->Add( _panelBase, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer14 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	_checkBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::ApplyChanges ), NULL, this );
	_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesForecasterVirtual::~asFramePreferencesForecasterVirtual()
{
	// Disconnect Events
	_checkBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::ApplyChanges ), NULL, this );
	_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::SaveAndClose ), NULL, this );

}

asWizardBatchForecastsVirtual::asWizardBatchForecastsVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );

	wxWizardPageSimple* _wizPage1 = new wxWizardPageSimple( this );
	m_pages.Add( _wizPage1 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxVERTICAL );

	_staticText37 = new wxStaticText( _wizPage1, wxID_ANY, _("Load / create a batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText37->Wrap( -1 );
	_staticText37->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer48->Add( _staticText37, 0, wxALL|wxEXPAND, 5 );

	_staticText35 = new wxStaticText( _wizPage1, wxID_ANY, _("Provide the path to an existing file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText35->Wrap( -1 );
	bSizer48->Add( _staticText35, 0, wxALL|wxEXPAND, 5 );

	_button4 = new wxButton( _wizPage1, wxID_ANY, _("Load existing batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer48->Add( _button4, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	_staticText46 = new wxStaticText( _wizPage1, wxID_ANY, _("or continue to create a new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText46->Wrap( -1 );
	bSizer48->Add( _staticText46, 0, wxALL, 5 );


	_wizPage1->SetSizer( bSizer48 );
	_wizPage1->Layout();
	bSizer48->Fit( _wizPage1 );
	wxWizardPageSimple* _wizPage2 = new wxWizardPageSimple( this );
	m_pages.Add( _wizPage2 );

	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxVERTICAL );

	_staticText36 = new wxStaticText( _wizPage2, wxID_ANY, _("Create a new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText36->Wrap( -1 );
	_staticText36->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer49->Add( _staticText36, 0, wxALL|wxEXPAND, 5 );

	_staticText43 = new wxStaticText( _wizPage2, wxID_ANY, _("Path to save the new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText43->Wrap( -1 );
	bSizer49->Add( _staticText43, 0, wxALL|wxEXPAND, 5 );

	_filePickerBatchFile = new wxFilePickerCtrl( _wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), _("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizer49->Add( _filePickerBatchFile, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticText45 = new wxStaticText( _wizPage2, wxID_ANY, _("The preferences frame will open to configure the required directories."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText45->Wrap( -1 );
	bSizer49->Add( _staticText45, 1, wxALL|wxEXPAND, 5 );


	_wizPage2->SetSizer( bSizer49 );
	_wizPage2->Layout();
	bSizer49->Fit( _wizPage2 );

	this->Centre( wxBOTH );

	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}

	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardBatchForecastsVirtual::OnWizardFinished ) );
	_button4->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardBatchForecastsVirtual::OnLoadExistingBatchForecasts ), NULL, this );
}

asWizardBatchForecastsVirtual::~asWizardBatchForecastsVirtual()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardBatchForecastsVirtual::OnWizardFinished ) );
	_button4->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardBatchForecastsVirtual::OnLoadExistingBatchForecasts ), NULL, this );

	m_pages.Clear();
}

asFrameStyledTextCtrlVirtual::asFrameStyledTextCtrlVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxVERTICAL );

	_scintilla = new wxStyledTextCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0, wxEmptyString );
	_scintilla->SetUseTabs( false );
	_scintilla->SetTabWidth( 4 );
	_scintilla->SetIndent( 4 );
	_scintilla->SetTabIndents( true );
	_scintilla->SetBackSpaceUnIndents( true );
	_scintilla->SetViewEOL( false );
	_scintilla->SetViewWhiteSpace( false );
	_scintilla->SetMarginWidth( 2, 0 );
	_scintilla->SetIndentationGuides( true );
	_scintilla->SetReadOnly( false );
	_scintilla->SetMarginType( 1, wxSTC_MARGIN_SYMBOL );
	_scintilla->SetMarginMask( 1, wxSTC_MASK_FOLDERS );
	_scintilla->SetMarginWidth( 1, 16);
	_scintilla->SetMarginSensitive( 1, true );
	_scintilla->SetProperty( wxT("fold"), wxT("1") );
	_scintilla->SetFoldFlags( wxSTC_FOLDFLAG_LINEBEFORE_CONTRACTED | wxSTC_FOLDFLAG_LINEAFTER_CONTRACTED );
	_scintilla->SetMarginType( 0, wxSTC_MARGIN_NUMBER );
	_scintilla->SetMarginWidth( 0, _scintilla->TextWidth( wxSTC_STYLE_LINENUMBER, wxT("_99999") ) );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDER, wxSTC_MARK_BOXPLUS );
	_scintilla->MarkerSetBackground( wxSTC_MARKNUM_FOLDER, wxColour( wxT("BLACK") ) );
	_scintilla->MarkerSetForeground( wxSTC_MARKNUM_FOLDER, wxColour( wxT("WHITE") ) );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDEROPEN, wxSTC_MARK_BOXMINUS );
	_scintilla->MarkerSetBackground( wxSTC_MARKNUM_FOLDEROPEN, wxColour( wxT("BLACK") ) );
	_scintilla->MarkerSetForeground( wxSTC_MARKNUM_FOLDEROPEN, wxColour( wxT("WHITE") ) );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDERSUB, wxSTC_MARK_EMPTY );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDEREND, wxSTC_MARK_BOXPLUS );
	_scintilla->MarkerSetBackground( wxSTC_MARKNUM_FOLDEREND, wxColour( wxT("BLACK") ) );
	_scintilla->MarkerSetForeground( wxSTC_MARKNUM_FOLDEREND, wxColour( wxT("WHITE") ) );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDEROPENMID, wxSTC_MARK_BOXMINUS );
	_scintilla->MarkerSetBackground( wxSTC_MARKNUM_FOLDEROPENMID, wxColour( wxT("BLACK") ) );
	_scintilla->MarkerSetForeground( wxSTC_MARKNUM_FOLDEROPENMID, wxColour( wxT("WHITE") ) );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDERMIDTAIL, wxSTC_MARK_EMPTY );
	_scintilla->MarkerDefine( wxSTC_MARKNUM_FOLDERTAIL, wxSTC_MARK_EMPTY );
	_scintilla->SetSelBackground( true, wxSystemSettings::GetColour( wxSYS_COLOUR_HIGHLIGHT ) );
	_scintilla->SetSelForeground( true, wxSystemSettings::GetColour( wxSYS_COLOUR_HIGHLIGHTTEXT ) );
	bSizer37->Add( _scintilla, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer37 );
	this->Layout();

	this->Centre( wxBOTH );
}

asFrameStyledTextCtrlVirtual::~asFrameStyledTextCtrlVirtual()
{
}
