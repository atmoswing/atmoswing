///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun 17 2015)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
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

asFrameMainVirtual::asFrameMainVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,700 ), wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer18;
	bSizer18 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer19;
	bSizer19 = new wxBoxSizer( wxHORIZONTAL );
	
	wxStaticBoxSizer* sbSizer13;
	sbSizer13 = new wxStaticBoxSizer( new wxStaticBox( m_panelMain, wxID_ANY, _("Day of the forecast") ), wxVERTICAL );
	
	m_calendarForecastDate = new wxCalendarCtrl( sbSizer13->GetStaticBox(), wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxDefaultSize, wxCAL_MONDAY_FIRST|wxCAL_SHOW_HOLIDAYS|wxCAL_SHOW_SURROUNDING_WEEKS );
	sbSizer13->Add( m_calendarForecastDate, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextForecastHour = new wxStaticText( sbSizer13->GetStaticBox(), wxID_ANY, _("Hour (UTM)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastHour->Wrap( -1 );
	bSizer35->Add( m_staticTextForecastHour, 0, wxTOP|wxBOTTOM|wxLEFT, 5 );
	
	m_textCtrlForecastHour = new wxTextCtrl( sbSizer13->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_textCtrlForecastHour->SetMaxLength( 2 ); 
	bSizer35->Add( m_textCtrlForecastHour, 0, wxALL, 5 );
	
	m_bpButtonNow = new wxBitmapButton( sbSizer13->GetStaticBox(), wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_bpButtonNow->SetToolTip( _("Set current date.") );
	
	bSizer35->Add( m_bpButtonNow, 0, wxTOP|wxBOTTOM, 5 );
	
	
	sbSizer13->Add( bSizer35, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer19->Add( sbSizer13, 0, wxEXPAND|wxALL, 5 );
	
	wxBoxSizer* bSizer341;
	bSizer341 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer131;
	sbSizer131 = new wxStaticBoxSizer( new wxStaticBox( m_panelMain, wxID_ANY, _("Overall progress") ), wxVERTICAL );
	
	m_gauge = new wxGauge( sbSizer131->GetStaticBox(), wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	m_gauge->SetValue( 0 ); 
	sbSizer131->Add( m_gauge, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer351;
	bSizer351 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextProgressActual = new wxStaticText( sbSizer131->GetStaticBox(), wxID_ANY, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProgressActual->Wrap( -1 );
	bSizer351->Add( m_staticTextProgressActual, 0, wxTOP|wxBOTTOM, 5 );
	
	m_staticText38 = new wxStaticText( sbSizer131->GetStaticBox(), wxID_ANY, _("/"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText38->Wrap( -1 );
	bSizer351->Add( m_staticText38, 0, wxALL, 5 );
	
	m_staticTextProgressTot = new wxStaticText( sbSizer131->GetStaticBox(), wxID_ANY, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProgressTot->Wrap( -1 );
	bSizer351->Add( m_staticTextProgressTot, 0, wxTOP|wxBOTTOM, 5 );
	
	
	sbSizer131->Add( bSizer351, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer341->Add( sbSizer131, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( m_panelMain, wxID_ANY, _("Current forecast state") ), wxVERTICAL );
	
	m_sizerLeds = new wxFlexGridSizer( 4, 2, 0, 0 );
	m_sizerLeds->SetFlexibleDirection( wxBOTH );
	m_sizerLeds->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	
	sbSizer5->Add( m_sizerLeds, 0, wxEXPAND, 5 );
	
	
	bSizer341->Add( sbSizer5, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer19->Add( bSizer341, 1, wxEXPAND, 5 );
	
	
	bSizer18->Add( bSizer19, 0, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( m_panelMain, wxID_ANY, _("List of the forecasts") ), wxVERTICAL );
	
	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );
	
	bSizer22->SetMinSize( wxSize( -1,200 ) ); 
	m_button2 = new wxButton( sbSizer6->GetStaticBox(), wxID_ANY, _("Configure directories"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer22->Add( m_button2, 0, wxRIGHT|wxLEFT, 5 );
	
	m_scrolledWindowForecasts = new wxScrolledWindow( sbSizer6->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxVSCROLL );
	m_scrolledWindowForecasts->SetScrollRate( 5, 5 );
	m_scrolledWindowForecasts->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_scrolledWindowForecasts->SetMinSize( wxSize( -1,200 ) );
	
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );
	
	m_sizerForecasts = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer32->Add( m_sizerForecasts, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_bpButtonAdd = new wxBitmapButton( m_scrolledWindowForecasts, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 28,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_bpButtonAdd->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_bpButtonAdd->SetToolTip( _("Add a parameters file.") );
	
	bSizer34->Add( m_bpButtonAdd, 0, wxALL, 5 );
	
	
	bSizer32->Add( bSizer34, 0, wxLEFT, 5 );
	
	
	m_scrolledWindowForecasts->SetSizer( bSizer32 );
	m_scrolledWindowForecasts->Layout();
	bSizer32->Fit( m_scrolledWindowForecasts );
	bSizer22->Add( m_scrolledWindowForecasts, 1, wxEXPAND | wxALL, 5 );
	
	
	sbSizer6->Add( bSizer22, 1, wxEXPAND, 5 );
	
	
	bSizer18->Add( sbSizer6, 1, wxALL|wxEXPAND, 5 );
	
	
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
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	m_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	m_toolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_toolBar->Realize(); 
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_bpButtonNow->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnSetPresentDate ), NULL, this );
	m_button2->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnConfigureDirectories ), NULL, this );
	m_bpButtonAdd->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::AddForecast ), NULL, this );
	this->Connect( m_menuItemOpenBatchFile->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnOpenBatchForecasts ) );
	this->Connect( m_menuItemSaveBatchFile->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnSaveBatchForecasts ) );
	this->Connect( m_menuItemSaveBatchFileAs->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnSaveBatchForecastsAs ) );
	this->Connect( m_menuItemNewBatchFile->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnNewBatchForecasts ) );
	this->Connect( m_menuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFramePreferences ) );
	this->Connect( m_menuItemBuildPredictandDB->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFramePredictandDB ) );
	this->Connect( m_menuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnShowLog ) );
	this->Connect( m_MenuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel1 ) );
	this->Connect( m_MenuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel2 ) );
	this->Connect( m_MenuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel3 ) );
	this->Connect( m_menuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFrameAbout ) );
}

asFrameMainVirtual::~asFrameMainVirtual()
{
	// Disconnect Events
	m_bpButtonNow->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnSetPresentDate ), NULL, this );
	m_button2->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnConfigureDirectories ), NULL, this );
	m_bpButtonAdd->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::AddForecast ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnOpenBatchForecasts ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnSaveBatchForecasts ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnSaveBatchForecastsAs ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnNewBatchForecasts ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFramePreferences ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFramePredictandDB ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnShowLog ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel1 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel2 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel3 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFrameAbout ) );
	
}

asFramePredictandDBVirtual::asFramePredictandDBVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxVERTICAL );
	
	m_panel2 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer6;
	bSizer6 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextDataParam = new wxStaticText( m_panel2, wxID_ANY, _("Predictand parameter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataParam->Wrap( -1 );
	fgSizer2->Add( m_staticTextDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_choiceDataParamChoices[] = { _("Precipitation"), _("Temperature"), _("Lightnings"), _("Other") };
	int m_choiceDataParamNChoices = sizeof( m_choiceDataParamChoices ) / sizeof( wxString );
	m_choiceDataParam = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDataParamNChoices, m_choiceDataParamChoices, 0 );
	m_choiceDataParam->SetSelection( 0 );
	fgSizer2->Add( m_choiceDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_staticTextDataTempResol = new wxStaticText( m_panel2, wxID_ANY, _("Temporal resolution"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataTempResol->Wrap( -1 );
	fgSizer2->Add( m_staticTextDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_choiceDataTempResolChoices[] = { _("24 hours"), _("6 hours"), _("1-hr MTW"), _("3-hr MTW"), _("6-hr MTW"), _("12-hr MTW") };
	int m_choiceDataTempResolNChoices = sizeof( m_choiceDataTempResolChoices ) / sizeof( wxString );
	m_choiceDataTempResol = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDataTempResolNChoices, m_choiceDataTempResolChoices, 0 );
	m_choiceDataTempResol->SetSelection( 0 );
	fgSizer2->Add( m_choiceDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_staticTextDataSpatAggreg = new wxStaticText( m_panel2, wxID_ANY, _("Spatial aggregation"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataSpatAggreg->Wrap( -1 );
	fgSizer2->Add( m_staticTextDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_choiceDataSpatAggregChoices[] = { _("Station"), _("Groupment"), _("Catchment") };
	int m_choiceDataSpatAggregNChoices = sizeof( m_choiceDataSpatAggregChoices ) / sizeof( wxString );
	m_choiceDataSpatAggreg = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDataSpatAggregNChoices, m_choiceDataSpatAggregChoices, 0 );
	m_choiceDataSpatAggreg->SetSelection( 0 );
	fgSizer2->Add( m_choiceDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer6->Add( fgSizer2, 1, wxTOP|wxBOTTOM|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxVERTICAL );
	
	m_panelDataProcessing = new wxPanel( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxStaticBoxSizer* sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( m_panelDataProcessing, wxID_ANY, _("Data processing (precipitation only)") ), wxVERTICAL );
	
	wxBoxSizer* bSizer9;
	bSizer9 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );
	
	m_checkBoxReturnPeriod = new wxCheckBox( sbSizer1->GetStaticBox(), wxID_ANY, _("Normalize by the return period of"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxReturnPeriod->SetValue(true); 
	bSizer11->Add( m_checkBoxReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_textCtrlReturnPeriod = new wxTextCtrl( sbSizer1->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlReturnPeriod->SetMaxLength( 0 ); 
	bSizer11->Add( m_textCtrlReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticTextYears = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextYears->Wrap( -1 );
	bSizer11->Add( m_staticTextYears, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer9->Add( bSizer11, 1, wxEXPAND, 5 );
	
	m_checkBoxSqrt = new wxCheckBox( sbSizer1->GetStaticBox(), wxID_ANY, _("Process the square root"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer9->Add( m_checkBoxSqrt, 0, wxALL, 5 );
	
	
	sbSizer1->Add( bSizer9, 1, wxEXPAND, 5 );
	
	
	m_panelDataProcessing->SetSizer( sbSizer1 );
	m_panelDataProcessing->Layout();
	sbSizer1->Fit( m_panelDataProcessing );
	bSizer12->Add( m_panelDataProcessing, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer6->Add( bSizer12, 0, wxEXPAND, 5 );
	
	m_staticTextCatalogPath = new wxStaticText( m_panel2, wxID_ANY, _("Select the predictand catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextCatalogPath->Wrap( -1 );
	bSizer6->Add( m_staticTextCatalogPath, 0, wxALL, 5 );
	
	m_filePickerCatalogPath = new wxFilePickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL );
	bSizer6->Add( m_filePickerCatalogPath, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextDataDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the predictand data directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataDir->Wrap( -1 );
	bSizer6->Add( m_staticTextDataDir, 0, wxALL, 5 );
	
	m_dirPickerDataDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_dirPickerDataDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextPatternsDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the directory containing the file patterns description"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPatternsDir->Wrap( -1 );
	bSizer6->Add( m_staticTextPatternsDir, 0, wxALL, 5 );
	
	m_dirPickerPatternsDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_dirPickerPatternsDir, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_staticDestinationDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the destination directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticDestinationDir->Wrap( -1 );
	bSizer6->Add( m_staticDestinationDir, 0, wxALL, 5 );
	
	m_dirPickerDestinationDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_dirPickerDestinationDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );
	
	m_buttonSaveDefault = new wxButton( m_panel2, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( m_buttonSaveDefault, 0, wxALIGN_RIGHT, 5 );
	
	m_buttonsConfirmation = new wxStdDialogButtonSizer();
	m_buttonsConfirmationOK = new wxButton( m_panel2, wxID_OK );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationOK );
	m_buttonsConfirmationCancel = new wxButton( m_panel2, wxID_CANCEL );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationCancel );
	m_buttonsConfirmation->Realize();
	
	bSizer15->Add( m_buttonsConfirmation, 0, 0, 5 );
	
	
	bSizer6->Add( bSizer15, 0, wxALIGN_RIGHT|wxTOP|wxBOTTOM|wxRIGHT, 5 );
	
	
	m_panel2->SetSizer( bSizer6 );
	m_panel2->Layout();
	bSizer6->Fit( m_panel2 );
	bSizer5->Add( m_panel2, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer5 );
	this->Layout();
	bSizer5->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_choiceDataParam->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	m_buttonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::OnSaveDefault ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
}

asFramePredictandDBVirtual::~asFramePredictandDBVirtual()
{
	// Disconnect Events
	m_choiceDataParam->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	m_buttonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::OnSaveDefault ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
	
}

asPanelForecastVirtual::asPanelForecastVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	m_sizerPanel = new wxBoxSizer( wxVERTICAL );
	
	m_sizerHeader = new wxBoxSizer( wxHORIZONTAL );
	
	m_sizerFilename = new wxBoxSizer( wxVERTICAL );
	
	m_textCtrlParametersFileName = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_textCtrlParametersFileName->SetMaxLength( 0 ); 
	m_textCtrlParametersFileName->SetToolTip( _("Enter the parameters file name...") );
	
	m_sizerFilename->Add( m_textCtrlParametersFileName, 0, wxEXPAND|wxRIGHT|wxLEFT, 5 );
	
	
	m_sizerHeader->Add( m_sizerFilename, 1, wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_bpButtonClose = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_sizerHeader->Add( m_bpButtonClose, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	
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
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_panelPathsCommon, wxID_ANY, _("Directories for real-time forecasting") ), wxVERTICAL );
	
	m_staticTextParametersDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing the parameters files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextParametersDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextParametersDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerParameters = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerParameters, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_staticTextPredictandDBDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing the predictand DB"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerPredictandDB = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextArchivePredictorsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerArchivePredictors = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_staticTextRealtimePredictorSavingDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory to save downloaded real-time predictors (GCM forecasts)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextRealtimePredictorSavingDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextRealtimePredictorSavingDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerRealtimePredictorSaving = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerRealtimePredictorSaving, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextForecastResultsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory to save forecast outputs (netCDF)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextForecastResultsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerForecastResults = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextForecastResultsExportsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory to save forecast exports (xml)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsExportsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextForecastResultsExportsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerForecastResultsExports = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerForecastResultsExports, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_checkBoxExportSyntheticXml = new wxCheckBox( sbSizer18->GetStaticBox(), wxID_ANY, _("Export forecasts as synthetic xml"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer18->Add( m_checkBoxExportSyntheticXml, 0, wxALL, 5 );
	
	
	m_sizerPanelPaths->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelPathsCommon->SetSizer( m_sizerPanelPaths );
	m_panelPathsCommon->Layout();
	m_sizerPanelPaths->Fit( m_panelPathsCommon );
	m_notebookBase->AddPage( m_panelPathsCommon, _("Batch file properties"), true );
	m_panelGeneralCommon = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer16;
	bSizer16 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );
	
	wxString m_radioBoxLogLevelChoices[] = { _("Errors only (recommanded)"), _("Errors and warnings"), _("Verbose") };
	int m_radioBoxLogLevelNChoices = sizeof( m_radioBoxLogLevelChoices ) / sizeof( wxString );
	m_radioBoxLogLevel = new wxRadioBox( sbSizer7->GetStaticBox(), wxID_ANY, _("Level"), wxDefaultPosition, wxDefaultSize, m_radioBoxLogLevelNChoices, m_radioBoxLogLevelChoices, 1, wxRA_SPECIFY_COLS );
	m_radioBoxLogLevel->SetSelection( 0 );
	bSizer20->Add( m_radioBoxLogLevel, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( sbSizer7->GetStaticBox(), wxID_ANY, _("Outputs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );
	
	m_checkBoxDisplayLogWindow = new wxCheckBox( sbSizer8->GetStaticBox(), wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxDisplayLogWindow->SetValue(true); 
	bSizer21->Add( m_checkBoxDisplayLogWindow, 0, wxALL, 5 );
	
	m_checkBoxSaveLogFile = new wxCheckBox( sbSizer8->GetStaticBox(), wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxSaveLogFile->SetValue(true); 
	m_checkBoxSaveLogFile->Enable( false );
	
	bSizer21->Add( m_checkBoxSaveLogFile, 0, wxALL, 5 );
	
	
	sbSizer8->Add( bSizer21, 1, wxEXPAND, 5 );
	
	
	bSizer20->Add( sbSizer8, 1, wxALL|wxEXPAND, 5 );
	
	
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
	bSizer34->Add( m_staticTextProxyAddress, 0, wxALL, 5 );
	
	m_textCtrlProxyAddress = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 180,-1 ), 0 );
	m_textCtrlProxyAddress->SetMaxLength( 0 ); 
	bSizer34->Add( m_textCtrlProxyAddress, 1, wxALL, 5 );
	
	m_staticTextProxyPort = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Port"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyPort->Wrap( -1 );
	bSizer34->Add( m_staticTextProxyPort, 0, wxALL, 5 );
	
	m_textCtrlProxyPort = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_textCtrlProxyPort->SetMaxLength( 0 ); 
	bSizer34->Add( m_textCtrlProxyPort, 0, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer34, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextProxyUser = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Username"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyUser->Wrap( -1 );
	bSizer35->Add( m_staticTextProxyUser, 0, wxALL, 5 );
	
	m_textCtrlProxyUser = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_textCtrlProxyUser->SetMaxLength( 0 ); 
	bSizer35->Add( m_textCtrlProxyUser, 1, wxALL, 5 );
	
	m_staticTextProxyPasswd = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Password"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyPasswd->Wrap( -1 );
	bSizer35->Add( m_staticTextProxyPasswd, 0, wxALL, 5 );
	
	m_textCtrlProxyPasswd = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD );
	m_textCtrlProxyPasswd->SetMaxLength( 0 ); 
	bSizer35->Add( m_textCtrlProxyPasswd, 1, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer35, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer14, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelGeneralCommon->SetSizer( bSizer16 );
	m_panelGeneralCommon->Layout();
	bSizer16->Fit( m_panelGeneralCommon );
	m_notebookBase->AddPage( m_panelGeneralCommon, _("General options"), false );
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
	m_textCtrlMaxPrevStepsNb->SetMaxLength( 1 ); 
	fgSizer2->Add( m_textCtrlMaxPrevStepsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextMaxRequestsNb = new wxStaticText( sbSizer11->GetStaticBox(), wxID_ANY, _("Maximum parallel requests number"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextMaxRequestsNb->Wrap( -1 );
	fgSizer2->Add( m_staticTextMaxRequestsNb, 0, wxALL, 5 );
	
	m_textCtrlMaxRequestsNb = new wxTextCtrl( sbSizer11->GetStaticBox(), wxID_ANY, _("3"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_textCtrlMaxRequestsNb->SetMaxLength( 1 ); 
	fgSizer2->Add( m_textCtrlMaxRequestsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_checkBoxRestrictDownloads = new wxCheckBox( sbSizer11->GetStaticBox(), wxID_ANY, _("Restrict downloads to needed lead times."), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxRestrictDownloads->SetValue(true); 
	m_checkBoxRestrictDownloads->Enable( false );
	
	fgSizer2->Add( m_checkBoxRestrictDownloads, 0, wxALL, 5 );
	
	
	sbSizer11->Add( fgSizer2, 1, wxEXPAND, 5 );
	
	
	bSizer271->Add( sbSizer11, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneral, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_checkBoxResponsiveness = new wxCheckBox( sbSizer151->GetStaticBox(), wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxResponsiveness->SetValue(true); 
	sbSizer151->Add( m_checkBoxResponsiveness, 0, wxALL, 5 );
	
	m_checkBoxMultiInstancesForecaster = new wxCheckBox( sbSizer151->GetStaticBox(), wxID_ANY, _("Allow multiple instances of the forecaster"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer151->Add( m_checkBoxMultiInstancesForecaster, 0, wxALL, 5 );
	
	
	bSizer271->Add( sbSizer151, 0, wxEXPAND|wxALL, 5 );
	
	
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
	m_textCtrlThreadsNb->SetMaxLength( 0 ); 
	bSizer221->Add( m_textCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	sbSizer15->Add( bSizer221, 0, wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextThreadsPriority = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( m_staticTextThreadsPriority, 0, wxALL, 5 );
	
	m_sliderThreadsPriority = new wxSlider( sbSizer15->GetStaticBox(), wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( m_sliderThreadsPriority, 1, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer15->Add( bSizer241, 0, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_radioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Date array insertions (slower)"), _("Date array splitting (slower)") };
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
	m_staticText37->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer48->Add( m_staticText37, 0, wxALL, 5 );
	
	m_staticText35 = new wxStaticText( m_wizPage1, wxID_ANY, _("Press the button below to load an existing file"), wxDefaultPosition, wxDefaultSize, 0 );
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
	m_staticText36->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer49->Add( m_staticText36, 0, wxALL, 5 );
	
	m_staticText43 = new wxStaticText( m_wizPage2, wxID_ANY, _("Path to save the new batch file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	bSizer49->Add( m_staticText43, 0, wxALL, 5 );
	
	m_filePickerBatchFile = new wxFilePickerCtrl( m_wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
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
