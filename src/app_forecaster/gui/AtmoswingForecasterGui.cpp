///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
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

#include "AtmoswingForecasterGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameForecasterVirtual::asFrameForecasterVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,700 ), wxDefaultSize );

	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );

	m_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer18;
	bSizer18 = new wxBoxSizer( wxVERTICAL );

	m_staticText41 = new wxStaticText( m_panelMain, wxID_ANY, _("Start the forecast for a given date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText41->Wrap( -1 );
	m_staticText41->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer18->Add( m_staticText41, 0, wxALL|wxEXPAND, 10 );

	wxBoxSizer* bSizer19;
	bSizer19 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer36;
	bSizer36 = new wxBoxSizer( wxVERTICAL );

	m_calendarForecastDate = new wxCalendarCtrl( m_panelMain, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxDefaultSize, wxCAL_MONDAY_FIRST|wxCAL_SHOW_HOLIDAYS|wxCAL_SHOW_SURROUNDING_WEEKS );
	bSizer36->Add( m_calendarForecastDate, 0, wxALL, 5 );

	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextForecastHour = new wxStaticText( m_panelMain, wxID_ANY, _("Hour (UTM)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastHour->Wrap( -1 );
	bSizer35->Add( m_staticTextForecastHour, 0, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlForecastHour = new wxTextCtrl( m_panelMain, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	#ifdef __WXGTK__
	if ( !m_textCtrlForecastHour->HasFlag( wxTE_MULTILINE ) )
	{
	m_textCtrlForecastHour->SetMaxLength( 2 );
	}
	#else
	m_textCtrlForecastHour->SetMaxLength( 2 );
	#endif
	bSizer35->Add( m_textCtrlForecastHour, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_bpButtonNow = new wxBitmapButton( m_panelMain, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	m_bpButtonNow->SetToolTip( _("Set current date.") );

	bSizer35->Add( m_bpButtonNow, 0, wxTOP|wxBOTTOM|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer36->Add( bSizer35, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	bSizer19->Add( bSizer36, 0, wxRIGHT, 5 );

	wxBoxSizer* bSizer341;
	bSizer341 = new wxBoxSizer( wxVERTICAL );

	m_gauge = new wxGauge( m_panelMain, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	m_gauge->SetValue( 0 );
	bSizer341->Add( m_gauge, 0, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer351;
	bSizer351 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextProgressActual = new wxStaticText( m_panelMain, wxID_ANY, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProgressActual->Wrap( -1 );
	bSizer351->Add( m_staticTextProgressActual, 0, wxTOP|wxBOTTOM, 5 );

	m_staticText38 = new wxStaticText( m_panelMain, wxID_ANY, _("/"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText38->Wrap( -1 );
	bSizer351->Add( m_staticText38, 0, wxALL, 5 );

	m_staticTextProgressTot = new wxStaticText( m_panelMain, wxID_ANY, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProgressTot->Wrap( -1 );
	bSizer351->Add( m_staticTextProgressTot, 0, wxTOP|wxBOTTOM, 5 );


	bSizer341->Add( bSizer351, 0, wxALIGN_CENTER_HORIZONTAL, 5 );

	m_sizerLeds = new wxFlexGridSizer( 4, 2, 0, 0 );
	m_sizerLeds->SetFlexibleDirection( wxBOTH );
	m_sizerLeds->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );


	bSizer341->Add( m_sizerLeds, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );

	m_button2 = new wxButton( m_panelMain, wxID_ANY, _("Configure directories"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer38->Add( m_button2, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );


	bSizer341->Add( bSizer38, 0, wxEXPAND, 5 );


	bSizer19->Add( bSizer341, 1, wxLEFT|wxEXPAND, 5 );


	bSizer18->Add( bSizer19, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );

	bSizer22->SetMinSize( wxSize( -1,200 ) );
	m_scrolledWindowForecasts = new wxScrolledWindow( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	m_scrolledWindowForecasts->SetScrollRate( 5, 5 );
	m_scrolledWindowForecasts->SetBackgroundColour( wxColour( 144, 144, 144 ) );
	m_scrolledWindowForecasts->SetMinSize( wxSize( -1,200 ) );

	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );

	m_sizerForecasts = new wxBoxSizer( wxVERTICAL );


	bSizer32->Add( m_sizerForecasts, 0, wxEXPAND|wxTOP, 5 );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );

	m_bpButtonAdd = new wxBitmapButton( m_scrolledWindowForecasts, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 28,28 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	m_bpButtonAdd->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_bpButtonAdd->SetToolTip( _("Add a parameters file.") );

	bSizer34->Add( m_bpButtonAdd, 0, wxALL, 8 );


	bSizer32->Add( bSizer34, 0, wxLEFT, 5 );


	m_scrolledWindowForecasts->SetSizer( bSizer32 );
	m_scrolledWindowForecasts->Layout();
	bSizer32->Fit( m_scrolledWindowForecasts );
	bSizer22->Add( m_scrolledWindowForecasts, 1, wxEXPAND|wxTOP, 5 );


	bSizer18->Add( bSizer22, 1, wxEXPAND, 5 );


	m_panelMain->SetSizer( bSizer18 );
	m_panelMain->Layout();
	bSizer18->Fit( m_panelMain );
	bSizer3->Add( m_panelMain, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer3 );
	this->Layout();
	m_menuBar = new wxMenuBar( 0 );
	m_menuFile = new wxMenu();
	wxMenuItem* m_menuItemOpenBatchFile;
	m_menuItemOpenBatchFile = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Open a batch file") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemOpenBatchFile );

	wxMenuItem* m_menuItemSaveBatchFile;
	m_menuItemSaveBatchFile = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Save batch file") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemSaveBatchFile );

	wxMenuItem* m_menuItemSaveBatchFileAs;
	m_menuItemSaveBatchFileAs = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Save batch file as") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemSaveBatchFileAs );

	wxMenuItem* m_menuItemNewBatchFile;
	m_menuItemNewBatchFile = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Create a new batch file") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemNewBatchFile );

	m_menuBar->Append( m_menuFile, _("File") );

	m_menuOptions = new wxMenu();
	wxMenuItem* m_menuItemPreferences;
	m_menuItemPreferences = new wxMenuItem( m_menuOptions, wxID_ANY, wxString( _("Preferences") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuOptions->Append( m_menuItemPreferences );

	m_menuBar->Append( m_menuOptions, _("Options") );

	m_menuTools = new wxMenu();
	wxMenuItem* m_menuItemBuildPredictandDB;
	m_menuItemBuildPredictandDB = new wxMenuItem( m_menuTools, wxID_ANY, wxString( _("Build predictand DB") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuTools->Append( m_menuItemBuildPredictandDB );

	m_menuBar->Append( m_menuTools, _("Tools") );

	m_menuLog = new wxMenu();
	wxMenuItem* m_menuItemShowLog;
	m_menuItemShowLog = new wxMenuItem( m_menuLog, wxID_ANY, wxString( _("Show Log Window") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuLog->Append( m_menuItemShowLog );

	m_menuLogLevel = new wxMenu();
	wxMenuItem* m_menuLogLevelItem = new wxMenuItem( m_menuLog, wxID_ANY, _("Log level"), wxEmptyString, wxITEM_NORMAL, m_menuLogLevel );
	wxMenuItem* m_MenuItemLogLevel1;
	m_MenuItemLogLevel1 = new wxMenuItem( m_menuLogLevel, wxID_ANY, wxString( _("Only errors") ) , wxEmptyString, wxITEM_CHECK );
	m_menuLogLevel->Append( m_MenuItemLogLevel1 );

	wxMenuItem* m_MenuItemLogLevel2;
	m_MenuItemLogLevel2 = new wxMenuItem( m_menuLogLevel, wxID_ANY, wxString( _("Errors and warnings") ) , wxEmptyString, wxITEM_CHECK );
	m_menuLogLevel->Append( m_MenuItemLogLevel2 );

	wxMenuItem* m_MenuItemLogLevel3;
	m_MenuItemLogLevel3 = new wxMenuItem( m_menuLogLevel, wxID_ANY, wxString( _("Verbose") ) , wxEmptyString, wxITEM_CHECK );
	m_menuLogLevel->Append( m_MenuItemLogLevel3 );

	m_menuLog->Append( m_menuLogLevelItem );

	m_menuBar->Append( m_menuLog, _("Log") );

	m_menuHelp = new wxMenu();
	wxMenuItem* m_menuItemAbout;
	m_menuItemAbout = new wxMenuItem( m_menuHelp, wxID_ANY, wxString( _("About") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuHelp->Append( m_menuItemAbout );

	m_menuBar->Append( m_menuHelp, _("Help") );

	this->SetMenuBar( m_menuBar );

	m_statusBar1 = this->CreateStatusBar( 1, wxSTB_SIZEGRIP, wxID_ANY );
	m_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	m_toolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_toolBar->Realize();


	this->Centre( wxBOTH );

	// Connect Events
	m_bpButtonNow->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnSetPresentDate ), NULL, this );
	m_button2->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnConfigureDirectories ), NULL, this );
	m_bpButtonAdd->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::AddForecast ), NULL, this );
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnOpenBatchForecasts ), this, m_menuItemOpenBatchFile->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnSaveBatchForecasts ), this, m_menuItemSaveBatchFile->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnSaveBatchForecastsAs ), this, m_menuItemSaveBatchFileAs->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnNewBatchForecasts ), this, m_menuItemNewBatchFile->GetId());
	m_menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OpenFramePreferences ), this, m_menuItemPreferences->GetId());
	m_menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OpenFramePredictandDB ), this, m_menuItemBuildPredictandDB->GetId());
	m_menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnShowLog ), this, m_menuItemShowLog->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnLogLevel1 ), this, m_MenuItemLogLevel1->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnLogLevel2 ), this, m_MenuItemLogLevel2->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OnLogLevel3 ), this, m_MenuItemLogLevel3->GetId());
	m_menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecasterVirtual::OpenFrameAbout ), this, m_menuItemAbout->GetId());
}

asFrameForecasterVirtual::~asFrameForecasterVirtual()
{
	// Disconnect Events
	m_bpButtonNow->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnSetPresentDate ), NULL, this );
	m_button2->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::OnConfigureDirectories ), NULL, this );
	m_bpButtonAdd->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecasterVirtual::AddForecast ), NULL, this );

}

asPanelForecastVirtual::asPanelForecastVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	m_sizerPanel = new wxBoxSizer( wxVERTICAL );

	m_sizerHeader = new wxBoxSizer( wxHORIZONTAL );

	m_sizerFilename = new wxBoxSizer( wxVERTICAL );

	m_textCtrlParametersFileName = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_textCtrlParametersFileName->SetToolTip( _("Enter the parameters file name...") );

	m_sizerFilename->Add( m_textCtrlParametersFileName, 0, wxEXPAND|wxALL, 3 );


	m_sizerHeader->Add( m_sizerFilename, 1, wxEXPAND, 5 );

	m_bpButtonClose = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	m_sizerHeader->Add( m_bpButtonClose, 0, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxLEFT, 5 );


	m_sizerPanel->Add( m_sizerHeader, 0, wxEXPAND, 5 );


	this->SetSizer( m_sizerPanel );
	this->Layout();
	m_sizerPanel->Fit( this );

	// Connect Events
	m_bpButtonClose->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::ClosePanel ), NULL, this );
}

asPanelForecastVirtual::~asPanelForecastVirtual()
{
	// Disconnect Events
	m_bpButtonClose->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::ClosePanel ), NULL, this );

}

asFramePreferencesForecasterVirtual::asFramePreferencesForecasterVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );

	m_panelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );

	m_notebookBase = new wxNotebook( m_panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelPathsCommon = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_sizerPanelPaths = new wxBoxSizer( wxVERTICAL );

	m_staticTextParametersDir = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Directory containing the parameters files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextParametersDir->Wrap( -1 );
	m_sizerPanelPaths->Add( m_staticTextParametersDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerParameters = new wxDirPickerCtrl( m_panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_sizerPanelPaths->Add( m_dirPickerParameters, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	m_staticTextPredictandDBDir = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Directory containing the predictand DB"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPredictandDBDir->Wrap( -1 );
	m_sizerPanelPaths->Add( m_staticTextPredictandDBDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerPredictandDB = new wxDirPickerCtrl( m_panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_sizerPanelPaths->Add( m_dirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextArchivePredictorsDir = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextArchivePredictorsDir->Wrap( -1 );
	m_sizerPanelPaths->Add( m_staticTextArchivePredictorsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerArchivePredictors = new wxDirPickerCtrl( m_panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_sizerPanelPaths->Add( m_dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	m_staticTextRealtimePredictorSavingDir = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Directory to save downloaded real-time predictors (GCM forecasts)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextRealtimePredictorSavingDir->Wrap( -1 );
	m_sizerPanelPaths->Add( m_staticTextRealtimePredictorSavingDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerRealtimePredictorSaving = new wxDirPickerCtrl( m_panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_sizerPanelPaths->Add( m_dirPickerRealtimePredictorSaving, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextForecastResultsDir = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Directory to save forecast outputs (netCDF)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsDir->Wrap( -1 );
	m_sizerPanelPaths->Add( m_staticTextForecastResultsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerForecastResults = new wxDirPickerCtrl( m_panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_sizerPanelPaths->Add( m_dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextForecastResultsExportsDir = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Directory to save forecast exports (xml)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsExportsDir->Wrap( -1 );
	m_sizerPanelPaths->Add( m_staticTextForecastResultsExportsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerForecastResultsExports = new wxDirPickerCtrl( m_panelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_sizerPanelPaths->Add( m_dirPickerForecastResultsExports, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextExport = new wxStaticText( m_panelPathsCommon, wxID_ANY, _("Export:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextExport->Wrap( -1 );
	bSizer33->Add( m_staticTextExport, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString m_choiceExportsChoices[] = { _("None"), _("Full XML"), _("Small CSV"), _("Custom CSV for FVG") };
	int m_choiceExportsNChoices = sizeof( m_choiceExportsChoices ) / sizeof( wxString );
	m_choiceExports = new wxChoice( m_panelPathsCommon, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceExportsNChoices, m_choiceExportsChoices, 0 );
	m_choiceExports->SetSelection( 0 );
	bSizer33->Add( m_choiceExports, 0, wxALL, 5 );


	m_sizerPanelPaths->Add( bSizer33, 0, wxEXPAND, 5 );


	m_panelPathsCommon->SetSizer( m_sizerPanelPaths );
	m_panelPathsCommon->Layout();
	m_sizerPanelPaths->Fit( m_panelPathsCommon );
	m_notebookBase->AddPage( m_panelPathsCommon, _("Batch file properties"), false );
	m_panelGeneralCommon = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer16;
	bSizer16 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer71;
	sbSizer71 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Language") ), wxVERTICAL );

	wxString m_choiceLocaleChoices[] = { _("English"), _("French") };
	int m_choiceLocaleNChoices = sizeof( m_choiceLocaleChoices ) / sizeof( wxString );
	m_choiceLocale = new wxChoice( sbSizer71->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceLocaleNChoices, m_choiceLocaleChoices, 0 );
	m_choiceLocale->SetSelection( 0 );
	sbSizer71->Add( m_choiceLocale, 0, wxALL, 5 );

	m_staticText34 = new wxStaticText( sbSizer71->GetStaticBox(), wxID_ANY, _("Restart AtmoSwing for the change to take effect."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText34->Wrap( -1 );
	sbSizer71->Add( m_staticText34, 0, wxALL, 5 );


	bSizer16->Add( sbSizer71, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );

	m_radioBtnLogLevel1 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer38->Add( m_radioBtnLogLevel1, 0, wxALL, 5 );

	m_radioBtnLogLevel2 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer38->Add( m_radioBtnLogLevel2, 0, wxALL, 5 );

	m_radioBtnLogLevel3 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer38->Add( m_radioBtnLogLevel3, 0, wxALL, 5 );


	bSizer20->Add( bSizer38, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );

	m_checkBoxDisplayLogWindow = new wxCheckBox( sbSizer7->GetStaticBox(), wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxDisplayLogWindow->SetValue(true);
	bSizer21->Add( m_checkBoxDisplayLogWindow, 0, wxALL, 5 );

	m_checkBoxSaveLogFile = new wxCheckBox( sbSizer7->GetStaticBox(), wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxSaveLogFile->SetValue(true);
	m_checkBoxSaveLogFile->Enable( false );

	bSizer21->Add( m_checkBoxSaveLogFile, 0, wxALL, 5 );


	bSizer20->Add( bSizer21, 1, wxEXPAND, 5 );


	sbSizer7->Add( bSizer20, 1, wxEXPAND, 5 );


	bSizer16->Add( sbSizer7, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer14;
	sbSizer14 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Proxy configuration") ), wxVERTICAL );

	m_checkBoxProxy = new wxCheckBox( sbSizer14->GetStaticBox(), wxID_ANY, _("Internet connection uses a proxy"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer14->Add( m_checkBoxProxy, 0, wxALL, 5 );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextProxyAddress = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Proxy address"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyAddress->Wrap( -1 );
	bSizer34->Add( m_staticTextProxyAddress, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlProxyAddress = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 180,-1 ), 0 );
	bSizer34->Add( m_textCtrlProxyAddress, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextProxyPort = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Port"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyPort->Wrap( -1 );
	bSizer34->Add( m_staticTextProxyPort, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlProxyPort = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer34->Add( m_textCtrlProxyPort, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer14->Add( bSizer34, 1, wxEXPAND, 5 );

	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextProxyUser = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Username"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyUser->Wrap( -1 );
	bSizer35->Add( m_staticTextProxyUser, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlProxyUser = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	bSizer35->Add( m_textCtrlProxyUser, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextProxyPasswd = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Password"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyPasswd->Wrap( -1 );
	bSizer35->Add( m_staticTextProxyPasswd, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlProxyPasswd = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD );
	bSizer35->Add( m_textCtrlProxyPasswd, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer14->Add( bSizer35, 1, wxEXPAND, 5 );


	bSizer16->Add( sbSizer14, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Libraries") ), wxVERTICAL );

	wxBoxSizer* bSizer341;
	bSizer341 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextEcCodesDefs = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("ecCodes definitions"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextEcCodesDefs->Wrap( -1 );
	bSizer341->Add( m_staticTextEcCodesDefs, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlEcCodesDefs = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer341->Add( m_textCtrlEcCodesDefs, 1, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer6->Add( bSizer341, 1, wxEXPAND, 5 );


	bSizer16->Add( sbSizer6, 0, wxEXPAND|wxALL, 5 );


	m_panelGeneralCommon->SetSizer( bSizer16 );
	m_panelGeneralCommon->Layout();
	bSizer16->Fit( m_panelGeneralCommon );
	m_notebookBase->AddPage( m_panelGeneralCommon, _("General options"), true );
	m_panelAdvanced = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	m_notebookAdvanced = new wxNotebook( m_panelAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelGeneral = new wxPanel( m_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer271;
	bSizer271 = new wxBoxSizer( wxVERTICAL );

	wxString m_radioBoxGuiChoices[] = { _("Silent (no progressbar, much faster)"), _("Standard (recommanded)"), _("Verbose (not much used)") };
	int m_radioBoxGuiNChoices = sizeof( m_radioBoxGuiChoices ) / sizeof( wxString );
	m_radioBoxGui = new wxRadioBox( m_panelGeneral, wxID_ANY, _("GUI options"), wxDefaultPosition, wxDefaultSize, m_radioBoxGuiNChoices, m_radioBoxGuiChoices, 1, wxRA_SPECIFY_COLS );
	m_radioBoxGui->SetSelection( 1 );
	bSizer271->Add( m_radioBoxGui, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer11;
	sbSizer11 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneral, wxID_ANY, _("Predictor download") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticTextNumberFails = new wxStaticText( sbSizer11->GetStaticBox(), wxID_ANY, _("Maximum number of previous time steps if download fails"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextNumberFails->Wrap( -1 );
	fgSizer2->Add( m_staticTextNumberFails, 0, wxALL, 5 );

	m_textCtrlMaxPrevStepsNb = new wxTextCtrl( sbSizer11->GetStaticBox(), wxID_ANY, _("5"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	#ifdef __WXGTK__
	if ( !m_textCtrlMaxPrevStepsNb->HasFlag( wxTE_MULTILINE ) )
	{
	m_textCtrlMaxPrevStepsNb->SetMaxLength( 1 );
	}
	#else
	m_textCtrlMaxPrevStepsNb->SetMaxLength( 1 );
	#endif
	fgSizer2->Add( m_textCtrlMaxPrevStepsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextMaxRequestsNb = new wxStaticText( sbSizer11->GetStaticBox(), wxID_ANY, _("Maximum parallel requests number"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextMaxRequestsNb->Wrap( -1 );
	fgSizer2->Add( m_staticTextMaxRequestsNb, 0, wxALL, 5 );

	m_textCtrlMaxRequestsNb = new wxTextCtrl( sbSizer11->GetStaticBox(), wxID_ANY, _("3"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	#ifdef __WXGTK__
	if ( !m_textCtrlMaxRequestsNb->HasFlag( wxTE_MULTILINE ) )
	{
	m_textCtrlMaxRequestsNb->SetMaxLength( 1 );
	}
	#else
	m_textCtrlMaxRequestsNb->SetMaxLength( 1 );
	#endif
	fgSizer2->Add( m_textCtrlMaxRequestsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_checkBoxRestrictDownloads = new wxCheckBox( sbSizer11->GetStaticBox(), wxID_ANY, _("Restrict downloads to needed lead times."), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxRestrictDownloads->SetValue(true);
	m_checkBoxRestrictDownloads->Enable( false );

	fgSizer2->Add( m_checkBoxRestrictDownloads, 0, wxALL, 5 );


	sbSizer11->Add( fgSizer2, 1, wxEXPAND, 5 );


	bSizer271->Add( sbSizer11, 0, wxALL|wxEXPAND, 5 );

	m_checkBoxResponsiveness = new wxCheckBox( m_panelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxResponsiveness->SetValue(true);
	bSizer271->Add( m_checkBoxResponsiveness, 0, wxALL, 5 );

	m_checkBoxMultiInstancesForecaster = new wxCheckBox( m_panelGeneral, wxID_ANY, _("Allow multiple instances of the forecaster"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer271->Add( m_checkBoxMultiInstancesForecaster, 0, wxALL, 5 );


	m_panelGeneral->SetSizer( bSizer271 );
	m_panelGeneral->Layout();
	bSizer271->Fit( m_panelGeneral );
	m_notebookAdvanced->AddPage( m_panelGeneral, _("General"), true );
	m_panelProcessing = new wxPanel( m_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer1611;
	bSizer1611 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer15;
	sbSizer15 = new wxStaticBoxSizer( new wxStaticBox( m_panelProcessing, wxID_ANY, _("Multithreading") ), wxVERTICAL );

	m_checkBoxAllowMultithreading = new wxCheckBox( sbSizer15->GetStaticBox(), wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxAllowMultithreading->SetValue(true);
	sbSizer15->Add( m_checkBoxAllowMultithreading, 0, wxALL, 5 );

	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextThreadsNb = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Max nb of threads"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsNb->Wrap( -1 );
	bSizer221->Add( m_staticTextThreadsNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlThreadsNb = new wxTextCtrl( sbSizer15->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 30,-1 ), 0 );
	bSizer221->Add( m_textCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer15->Add( bSizer221, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextThreadsPriority = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( m_staticTextThreadsPriority, 0, wxALL, 5 );

	m_sliderThreadsPriority = new wxSlider( sbSizer15->GetStaticBox(), wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( m_sliderThreadsPriority, 1, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer15->Add( bSizer241, 0, wxEXPAND, 5 );


	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );

	wxString m_radioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Standard (slower)") };
	int m_radioBoxProcessingMethodsNChoices = sizeof( m_radioBoxProcessingMethodsChoices ) / sizeof( wxString );
	m_radioBoxProcessingMethods = new wxRadioBox( m_panelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition, wxDefaultSize, m_radioBoxProcessingMethodsNChoices, m_radioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS );
	m_radioBoxProcessingMethods->SetSelection( 0 );
	m_radioBoxProcessingMethods->SetToolTip( _("These options don't affect the results, only the processor efficiency.") );

	bSizer1611->Add( m_radioBoxProcessingMethods, 0, wxALL|wxEXPAND, 5 );


	m_panelProcessing->SetSizer( bSizer1611 );
	m_panelProcessing->Layout();
	bSizer1611->Fit( m_panelProcessing );
	m_notebookAdvanced->AddPage( m_panelProcessing, _("Processing"), false );
	m_panelUserDirectories = new wxPanel( m_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer24;
	bSizer24 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( m_panelUserDirectories, wxID_ANY, _("User specific paths") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer9;
	fgSizer9 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer9->SetFlexibleDirection( wxBOTH );
	fgSizer9->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticTextUserDirLabel = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("User working directory:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextUserDirLabel->Wrap( -1 );
	fgSizer9->Add( m_staticTextUserDirLabel, 0, wxALL, 5 );

	m_staticTextUserDir = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextUserDir->Wrap( -1 );
	fgSizer9->Add( m_staticTextUserDir, 0, wxALL, 5 );

	m_staticTextLogFileLabel = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextLogFileLabel->Wrap( -1 );
	fgSizer9->Add( m_staticTextLogFileLabel, 0, wxALL, 5 );

	m_staticTextLogFile = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextLogFile->Wrap( -1 );
	fgSizer9->Add( m_staticTextLogFile, 0, wxALL, 5 );

	m_staticTextPrefFileLabel = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("Preferences file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPrefFileLabel->Wrap( -1 );
	fgSizer9->Add( m_staticTextPrefFileLabel, 0, wxALL, 5 );

	m_staticTextPrefFile = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPrefFile->Wrap( -1 );
	fgSizer9->Add( m_staticTextPrefFile, 0, wxALL, 5 );


	sbSizer17->Add( fgSizer9, 1, wxEXPAND, 5 );


	bSizer24->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );


	m_panelUserDirectories->SetSizer( bSizer24 );
	m_panelUserDirectories->Layout();
	bSizer24->Fit( m_panelUserDirectories );
	m_notebookAdvanced->AddPage( m_panelUserDirectories, _("User paths"), false );

	bSizer26->Add( m_notebookAdvanced, 1, wxEXPAND | wxALL, 5 );


	m_panelAdvanced->SetSizer( bSizer26 );
	m_panelAdvanced->Layout();
	bSizer26->Fit( m_panelAdvanced );
	m_notebookBase->AddPage( m_panelAdvanced, _("Advanced options"), false );

	bSizer15->Add( m_notebookBase, 1, wxEXPAND | wxALL, 5 );

	m_buttonsConfirmation = new wxStdDialogButtonSizer();
	m_buttonsConfirmationOK = new wxButton( m_panelBase, wxID_OK );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationOK );
	m_buttonsConfirmationApply = new wxButton( m_panelBase, wxID_APPLY );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationApply );
	m_buttonsConfirmationCancel = new wxButton( m_panelBase, wxID_CANCEL );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationCancel );
	m_buttonsConfirmation->Realize();

	bSizer15->Add( m_buttonsConfirmation, 0, wxEXPAND|wxALL, 5 );


	m_panelBase->SetSizer( bSizer15 );
	m_panelBase->Layout();
	bSizer15->Fit( m_panelBase );
	bSizer14->Add( m_panelBase, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer14 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	m_checkBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesForecasterVirtual::~asFramePreferencesForecasterVirtual()
{
	// Disconnect Events
	m_checkBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::SaveAndClose ), NULL, this );

}

asWizardBatchForecastsVirtual::asWizardBatchForecastsVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );

	wxWizardPageSimple* m_wizPage1 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage1 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxVERTICAL );

	m_staticText37 = new wxStaticText( m_wizPage1, wxID_ANY, _("Load / create a batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText37->Wrap( -1 );
	m_staticText37->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer48->Add( m_staticText37, 0, wxALL|wxEXPAND, 5 );

	m_staticText35 = new wxStaticText( m_wizPage1, wxID_ANY, _("Provide the path to an existing file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText35->Wrap( -1 );
	bSizer48->Add( m_staticText35, 0, wxALL|wxEXPAND, 5 );

	m_button4 = new wxButton( m_wizPage1, wxID_ANY, _("Load existing batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer48->Add( m_button4, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	m_staticText46 = new wxStaticText( m_wizPage1, wxID_ANY, _("or continue to create a new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	bSizer48->Add( m_staticText46, 0, wxALL, 5 );


	m_wizPage1->SetSizer( bSizer48 );
	m_wizPage1->Layout();
	bSizer48->Fit( m_wizPage1 );
	wxWizardPageSimple* m_wizPage2 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage2 );

	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxVERTICAL );

	m_staticText36 = new wxStaticText( m_wizPage2, wxID_ANY, _("Create a new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText36->Wrap( -1 );
	m_staticText36->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer49->Add( m_staticText36, 0, wxALL|wxEXPAND, 5 );

	m_staticText43 = new wxStaticText( m_wizPage2, wxID_ANY, _("Path to save the new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	bSizer49->Add( m_staticText43, 0, wxALL|wxEXPAND, 5 );

	m_filePickerBatchFile = new wxFilePickerCtrl( m_wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), _("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizer49->Add( m_filePickerBatchFile, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticText45 = new wxStaticText( m_wizPage2, wxID_ANY, _("The preferences frame will open to configure the required directories."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	bSizer49->Add( m_staticText45, 1, wxALL|wxEXPAND, 5 );


	m_wizPage2->SetSizer( bSizer49 );
	m_wizPage2->Layout();
	bSizer49->Fit( m_wizPage2 );

	this->Centre( wxBOTH );

	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}

	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardBatchForecastsVirtual::OnWizardFinished ) );
	m_button4->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardBatchForecastsVirtual::OnLoadExistingBatchForecasts ), NULL, this );
}

asWizardBatchForecastsVirtual::~asWizardBatchForecastsVirtual()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardBatchForecastsVirtual::OnWizardFinished ) );
	m_button4->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardBatchForecastsVirtual::OnLoadExistingBatchForecasts ), NULL, this );

	m_pages.Clear();
}
