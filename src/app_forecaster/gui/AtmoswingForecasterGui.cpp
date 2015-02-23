///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun  5 2014)
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
	this->SetSizeHints( wxSize( 500,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer18;
	bSizer18 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer19;
	bSizer19 = new wxBoxSizer( wxHORIZONTAL );
	
	wxStaticBoxSizer* sbSizer13;
	sbSizer13 = new wxStaticBoxSizer( new wxStaticBox( m_PanelMain, wxID_ANY, _("Day of the forecast") ), wxVERTICAL );
	
	m_CalendarForecastDate = new wxCalendarCtrl( m_PanelMain, wxID_ANY, wxDefaultDateTime, wxDefaultPosition, wxDefaultSize, wxCAL_MONDAY_FIRST|wxCAL_SHOW_HOLIDAYS|wxCAL_SHOW_SURROUNDING_WEEKS );
	sbSizer13->Add( m_CalendarForecastDate, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextForecastHour = new wxStaticText( m_PanelMain, wxID_ANY, _("Hour (UTM)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastHour->Wrap( -1 );
	bSizer35->Add( m_StaticTextForecastHour, 0, wxTOP|wxBOTTOM|wxLEFT, 5 );
	
	m_TextCtrlForecastHour = new wxTextCtrl( m_PanelMain, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlForecastHour->SetMaxLength( 2 ); 
	bSizer35->Add( m_TextCtrlForecastHour, 0, wxALL, 5 );
	
	m_BpButtonNow = new wxBitmapButton( m_PanelMain, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( -1,-1 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_BpButtonNow->SetToolTip( _("Set current date.") );
	
	bSizer35->Add( m_BpButtonNow, 0, wxTOP|wxBOTTOM, 5 );
	
	
	sbSizer13->Add( bSizer35, 1, wxEXPAND, 5 );
	
	
	bSizer19->Add( sbSizer13, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( m_PanelMain, wxID_ANY, _("Current forecast state") ), wxVERTICAL );
	
	m_SizerLeds = new wxFlexGridSizer( 4, 2, 0, 0 );
	m_SizerLeds->SetFlexibleDirection( wxBOTH );
	m_SizerLeds->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	
	sbSizer5->Add( m_SizerLeds, 0, wxEXPAND, 5 );
	
	
	bSizer19->Add( sbSizer5, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer18->Add( bSizer19, 0, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( m_PanelMain, wxID_ANY, _("List of the forecasts") ), wxVERTICAL );
	
	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );
	
	bSizer22->SetMinSize( wxSize( -1,200 ) ); 
	m_button2 = new wxButton( m_PanelMain, wxID_ANY, _("Configure directories"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer22->Add( m_button2, 0, wxALL, 5 );
	
	m_ScrolledWindowForecasts = new wxScrolledWindow( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxVSCROLL );
	m_ScrolledWindowForecasts->SetScrollRate( 5, 5 );
	m_ScrolledWindowForecasts->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_ScrolledWindowForecasts->SetMinSize( wxSize( -1,200 ) );
	
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );
	
	m_SizerForecasts = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer32->Add( m_SizerForecasts, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_BpButtonAdd = new wxBitmapButton( m_ScrolledWindowForecasts, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 28,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_BpButtonAdd->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_BpButtonAdd->SetToolTip( _("Add a parameters file.") );
	
	bSizer34->Add( m_BpButtonAdd, 0, wxALL, 5 );
	
	
	bSizer32->Add( bSizer34, 0, wxLEFT, 5 );
	
	
	m_ScrolledWindowForecasts->SetSizer( bSizer32 );
	m_ScrolledWindowForecasts->Layout();
	bSizer32->Fit( m_ScrolledWindowForecasts );
	bSizer22->Add( m_ScrolledWindowForecasts, 1, wxEXPAND | wxALL, 5 );
	
	
	sbSizer6->Add( bSizer22, 1, wxEXPAND, 5 );
	
	
	bSizer18->Add( sbSizer6, 1, wxALL|wxEXPAND, 5 );
	
	
	m_PanelMain->SetSizer( bSizer18 );
	m_PanelMain->Layout();
	bSizer18->Fit( m_PanelMain );
	bSizer3->Add( m_PanelMain, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer3 );
	this->Layout();
	bSizer3->Fit( this );
	m_MenuBar = new wxMenuBar( 0 );
	m_MenuFile = new wxMenu();
	wxMenuItem* m_MenuItemOpenBatchFile;
	m_MenuItemOpenBatchFile = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Open a batch file") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenBatchFile );
	
	wxMenuItem* m_MenuItemSaveBatchFile;
	m_MenuItemSaveBatchFile = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Save batch file") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemSaveBatchFile );
	
	wxMenuItem* m_MenuItemSaveBatchFileAs;
	m_MenuItemSaveBatchFileAs = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Save batch file as") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemSaveBatchFileAs );
	
	wxMenuItem* m_MenuItemNewBatchFile;
	m_MenuItemNewBatchFile = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Create a new batch file") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemNewBatchFile );
	
	m_MenuBar->Append( m_MenuFile, _("File") ); 
	
	m_MenuOptions = new wxMenu();
	wxMenuItem* m_MenuItemPreferences;
	m_MenuItemPreferences = new wxMenuItem( m_MenuOptions, wxID_ANY, wxString( _("Preferences") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuOptions->Append( m_MenuItemPreferences );
	
	m_MenuBar->Append( m_MenuOptions, _("Options") ); 
	
	m_MenuTools = new wxMenu();
	wxMenuItem* m_MenuItemBuildPredictandDB;
	m_MenuItemBuildPredictandDB = new wxMenuItem( m_MenuTools, wxID_ANY, wxString( _("Build predictand DB") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuTools->Append( m_MenuItemBuildPredictandDB );
	
	m_MenuBar->Append( m_MenuTools, _("Tools") ); 
	
	m_MenuLog = new wxMenu();
	wxMenuItem* m_MenuItemShowLog;
	m_MenuItemShowLog = new wxMenuItem( m_MenuLog, wxID_ANY, wxString( _("Show Log Window") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuLog->Append( m_MenuItemShowLog );
	
	m_MenuLogLevel = new wxMenu();
	wxMenuItem* m_MenuLogLevelItem = new wxMenuItem( m_MenuLog, wxID_ANY, _("Log level"), wxEmptyString, wxITEM_NORMAL, m_MenuLogLevel );
	wxMenuItem* m_MenuItemLogLevel1;
	m_MenuItemLogLevel1 = new wxMenuItem( m_MenuLogLevel, wxID_ANY, wxString( _("Only errors") ) , wxEmptyString, wxITEM_CHECK );
	m_MenuLogLevel->Append( m_MenuItemLogLevel1 );
	
	wxMenuItem* m_MenuItemLogLevel2;
	m_MenuItemLogLevel2 = new wxMenuItem( m_MenuLogLevel, wxID_ANY, wxString( _("Errors and warnings") ) , wxEmptyString, wxITEM_CHECK );
	m_MenuLogLevel->Append( m_MenuItemLogLevel2 );
	
	wxMenuItem* m_MenuItemLogLevel3;
	m_MenuItemLogLevel3 = new wxMenuItem( m_MenuLogLevel, wxID_ANY, wxString( _("Verbose") ) , wxEmptyString, wxITEM_CHECK );
	m_MenuLogLevel->Append( m_MenuItemLogLevel3 );
	
	m_MenuLog->Append( m_MenuLogLevelItem );
	
	m_MenuBar->Append( m_MenuLog, _("Log") ); 
	
	m_MenuHelp = new wxMenu();
	wxMenuItem* m_MenuItemAbout;
	m_MenuItemAbout = new wxMenuItem( m_MenuHelp, wxID_ANY, wxString( _("About") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuHelp->Append( m_MenuItemAbout );
	
	m_MenuBar->Append( m_MenuHelp, _("Help") ); 
	
	this->SetMenuBar( m_MenuBar );
	
	m_ToolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	m_ToolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_ToolBar->Realize(); 
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_BpButtonNow->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnSetPresentDate ), NULL, this );
	m_button2->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnConfigureDirectories ), NULL, this );
	m_BpButtonAdd->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::AddForecast ), NULL, this );
	this->Connect( m_MenuItemOpenBatchFile->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnOpenBatchForecasts ) );
	this->Connect( m_MenuItemSaveBatchFile->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnSaveBatchForecasts ) );
	this->Connect( m_MenuItemSaveBatchFileAs->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnSaveBatchForecastsAs ) );
	this->Connect( m_MenuItemNewBatchFile->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnNewBatchForecasts ) );
	this->Connect( m_MenuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFramePreferences ) );
	this->Connect( m_MenuItemBuildPredictandDB->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFramePredictandDB ) );
	this->Connect( m_MenuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnShowLog ) );
	this->Connect( m_MenuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel1 ) );
	this->Connect( m_MenuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel2 ) );
	this->Connect( m_MenuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OnLogLevel3 ) );
	this->Connect( m_MenuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::OpenFrameAbout ) );
}

asFrameMainVirtual::~asFrameMainVirtual()
{
	// Disconnect Events
	m_BpButtonNow->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnSetPresentDate ), NULL, this );
	m_button2->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::OnConfigureDirectories ), NULL, this );
	m_BpButtonAdd->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::AddForecast ), NULL, this );
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
	
	m_StaticTextDataParam = new wxStaticText( m_panel2, wxID_ANY, _("Predictand parameter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDataParam->Wrap( -1 );
	fgSizer2->Add( m_StaticTextDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_ChoiceDataParamChoices[] = { _("Precipitation"), _("Temperature"), _("Lightnings"), _("Other") };
	int m_ChoiceDataParamNChoices = sizeof( m_ChoiceDataParamChoices ) / sizeof( wxString );
	m_ChoiceDataParam = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceDataParamNChoices, m_ChoiceDataParamChoices, 0 );
	m_ChoiceDataParam->SetSelection( 0 );
	fgSizer2->Add( m_ChoiceDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_StaticTextDataTempResol = new wxStaticText( m_panel2, wxID_ANY, _("Temporal resolution"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDataTempResol->Wrap( -1 );
	fgSizer2->Add( m_StaticTextDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_ChoiceDataTempResolChoices[] = { _("24 hours"), _("6 hours"), _("Moving temporal window (6/24 hours)") };
	int m_ChoiceDataTempResolNChoices = sizeof( m_ChoiceDataTempResolChoices ) / sizeof( wxString );
	m_ChoiceDataTempResol = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceDataTempResolNChoices, m_ChoiceDataTempResolChoices, 0 );
	m_ChoiceDataTempResol->SetSelection( 0 );
	fgSizer2->Add( m_ChoiceDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_StaticTextDataSpatAggreg = new wxStaticText( m_panel2, wxID_ANY, _("Spatial aggregation"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDataSpatAggreg->Wrap( -1 );
	fgSizer2->Add( m_StaticTextDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_ChoiceDataSpatAggregChoices[] = { _("Station"), _("Groupment"), _("Catchment") };
	int m_ChoiceDataSpatAggregNChoices = sizeof( m_ChoiceDataSpatAggregChoices ) / sizeof( wxString );
	m_ChoiceDataSpatAggreg = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceDataSpatAggregNChoices, m_ChoiceDataSpatAggregChoices, 0 );
	m_ChoiceDataSpatAggreg->SetSelection( 0 );
	fgSizer2->Add( m_ChoiceDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer6->Add( fgSizer2, 1, wxTOP|wxBOTTOM|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelDataProcessing = new wxPanel( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxStaticBoxSizer* sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( m_PanelDataProcessing, wxID_ANY, _("Data processing (precipitation only)") ), wxVERTICAL );
	
	wxBoxSizer* bSizer9;
	bSizer9 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );
	
	m_CheckBoxReturnPeriod = new wxCheckBox( m_PanelDataProcessing, wxID_ANY, _("Normalize by the return period of"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxReturnPeriod->SetValue(true); 
	bSizer11->Add( m_CheckBoxReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_TextCtrlReturnPeriod = new wxTextCtrl( m_PanelDataProcessing, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlReturnPeriod->SetMaxLength( 0 ); 
	bSizer11->Add( m_TextCtrlReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_StaticTextYears = new wxStaticText( m_PanelDataProcessing, wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextYears->Wrap( -1 );
	bSizer11->Add( m_StaticTextYears, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer9->Add( bSizer11, 1, wxEXPAND, 5 );
	
	m_CheckBoxSqrt = new wxCheckBox( m_PanelDataProcessing, wxID_ANY, _("Process the square root"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer9->Add( m_CheckBoxSqrt, 0, wxALL, 5 );
	
	
	sbSizer1->Add( bSizer9, 1, wxEXPAND, 5 );
	
	
	m_PanelDataProcessing->SetSizer( sbSizer1 );
	m_PanelDataProcessing->Layout();
	sbSizer1->Fit( m_PanelDataProcessing );
	bSizer12->Add( m_PanelDataProcessing, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer6->Add( bSizer12, 0, wxEXPAND, 5 );
	
	m_StaticTextCatalogPath = new wxStaticText( m_panel2, wxID_ANY, _("Select the predictand catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextCatalogPath->Wrap( -1 );
	bSizer6->Add( m_StaticTextCatalogPath, 0, wxALL, 5 );
	
	m_FilePickerCatalogPath = new wxFilePickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL );
	bSizer6->Add( m_FilePickerCatalogPath, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextDataDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the predictand data directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDataDir->Wrap( -1 );
	bSizer6->Add( m_StaticTextDataDir, 0, wxALL, 5 );
	
	m_DirPickerDataDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_DirPickerDataDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextPatternsDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the directory containing the file patterns description"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPatternsDir->Wrap( -1 );
	bSizer6->Add( m_StaticTextPatternsDir, 0, wxALL, 5 );
	
	m_DirPickerPatternsDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_DirPickerPatternsDir, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticDestinationDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the destination directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticDestinationDir->Wrap( -1 );
	bSizer6->Add( m_StaticDestinationDir, 0, wxALL, 5 );
	
	m_DirPickerDestinationDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_DirPickerDestinationDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );
	
	m_ButtonSaveDefault = new wxButton( m_panel2, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( m_ButtonSaveDefault, 0, wxALIGN_RIGHT, 5 );
	
	m_ButtonsConfirmation = new wxStdDialogButtonSizer();
	m_ButtonsConfirmationOK = new wxButton( m_panel2, wxID_OK );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationOK );
	m_ButtonsConfirmationCancel = new wxButton( m_panel2, wxID_CANCEL );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationCancel );
	m_ButtonsConfirmation->Realize();
	
	bSizer15->Add( m_ButtonsConfirmation, 0, 0, 5 );
	
	
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
	m_ChoiceDataParam->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	m_ButtonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::OnSaveDefault ), NULL, this );
	m_ButtonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
}

asFramePredictandDBVirtual::~asFramePredictandDBVirtual()
{
	// Disconnect Events
	m_ChoiceDataParam->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	m_ButtonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::OnSaveDefault ), NULL, this );
	m_ButtonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
	
}

asPanelForecastVirtual::asPanelForecastVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	m_SizerPanel = new wxBoxSizer( wxVERTICAL );
	
	m_SizerHeader = new wxBoxSizer( wxHORIZONTAL );
	
	m_SizerFilename = new wxBoxSizer( wxVERTICAL );
	
	m_TextCtrlParametersFileName = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlParametersFileName->SetMaxLength( 0 ); 
	m_TextCtrlParametersFileName->SetToolTip( _("Enter the parameters file name...") );
	
	m_SizerFilename->Add( m_TextCtrlParametersFileName, 0, wxEXPAND|wxRIGHT|wxLEFT, 5 );
	
	
	m_SizerHeader->Add( m_SizerFilename, 1, wxEXPAND, 5 );
	
	m_BpButtonClose = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_SizerHeader->Add( m_BpButtonClose, 0, 0, 5 );
	
	
	m_SizerPanel->Add( m_SizerHeader, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( m_SizerPanel );
	this->Layout();
	m_SizerPanel->Fit( this );
	
	// Connect Events
	m_BpButtonClose->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::ClosePanel ), NULL, this );
}

asPanelForecastVirtual::~asPanelForecastVirtual()
{
	// Disconnect Events
	m_BpButtonClose->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastVirtual::ClosePanel ), NULL, this );
	
}

asFramePreferencesForecasterVirtual::asFramePreferencesForecasterVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookBase = new wxNotebook( m_PanelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelPathsCommon = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerPanelPaths = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_PanelPathsCommon, wxID_ANY, _("Directories for real-time forecasting") ), wxVERTICAL );
	
	m_StaticTextParametersDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory containing the parameters files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextParametersDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextParametersDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerParameters = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerParameters, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextPredictandDBDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory containing the predictand DB"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerPredictandDB = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextArchivePredictorsDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerArchivePredictors = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextRealtimePredictorSavingDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory to save downloaded real-time predictors (GCM forecasts)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextRealtimePredictorSavingDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextRealtimePredictorSavingDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerRealtimePredictorSaving = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerRealtimePredictorSaving, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextForecastResultsDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory to save forecast outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastResultsDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextForecastResultsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerForecastResults = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	m_SizerPanelPaths->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelPathsCommon->SetSizer( m_SizerPanelPaths );
	m_PanelPathsCommon->Layout();
	m_SizerPanelPaths->Fit( m_PanelPathsCommon );
	m_NotebookBase->AddPage( m_PanelPathsCommon, _("Batch file properties"), true );
	m_PanelGeneralCommon = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer16;
	bSizer16 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );
	
	wxString m_RadioBoxLogLevelChoices[] = { _("Errors only (recommanded)"), _("Errors and warnings"), _("Verbose") };
	int m_RadioBoxLogLevelNChoices = sizeof( m_RadioBoxLogLevelChoices ) / sizeof( wxString );
	m_RadioBoxLogLevel = new wxRadioBox( m_PanelGeneralCommon, wxID_ANY, _("Level"), wxDefaultPosition, wxDefaultSize, m_RadioBoxLogLevelNChoices, m_RadioBoxLogLevelChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxLogLevel->SetSelection( 0 );
	bSizer20->Add( m_RadioBoxLogLevel, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Outputs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );
	
	m_CheckBoxDisplayLogWindow = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxDisplayLogWindow->SetValue(true); 
	bSizer21->Add( m_CheckBoxDisplayLogWindow, 0, wxALL, 5 );
	
	m_CheckBoxSaveLogFile = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxSaveLogFile->SetValue(true); 
	m_CheckBoxSaveLogFile->Enable( false );
	
	bSizer21->Add( m_CheckBoxSaveLogFile, 0, wxALL, 5 );
	
	
	sbSizer8->Add( bSizer21, 1, wxEXPAND, 5 );
	
	
	bSizer20->Add( sbSizer8, 1, wxALL|wxEXPAND, 5 );
	
	
	sbSizer7->Add( bSizer20, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer7, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer14;
	sbSizer14 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Proxy configuration") ), wxVERTICAL );
	
	m_CheckBoxProxy = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Internet connection uses a proxy"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer14->Add( m_CheckBoxProxy, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextProxyAddress = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Proxy address"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyAddress->Wrap( -1 );
	bSizer34->Add( m_StaticTextProxyAddress, 0, wxALL, 5 );
	
	m_TextCtrlProxyAddress = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 180,-1 ), 0 );
	m_TextCtrlProxyAddress->SetMaxLength( 0 ); 
	bSizer34->Add( m_TextCtrlProxyAddress, 1, wxALL, 5 );
	
	m_StaticTextProxyPort = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Port"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyPort->Wrap( -1 );
	bSizer34->Add( m_StaticTextProxyPort, 0, wxALL, 5 );
	
	m_TextCtrlProxyPort = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlProxyPort->SetMaxLength( 0 ); 
	bSizer34->Add( m_TextCtrlProxyPort, 0, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer34, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextProxyUser = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Username"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyUser->Wrap( -1 );
	bSizer35->Add( m_StaticTextProxyUser, 0, wxALL, 5 );
	
	m_TextCtrlProxyUser = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_TextCtrlProxyUser->SetMaxLength( 0 ); 
	bSizer35->Add( m_TextCtrlProxyUser, 1, wxALL, 5 );
	
	m_StaticTextProxyPasswd = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Password"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyPasswd->Wrap( -1 );
	bSizer35->Add( m_StaticTextProxyPasswd, 0, wxALL, 5 );
	
	m_TextCtrlProxyPasswd = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD );
	m_TextCtrlProxyPasswd->SetMaxLength( 0 ); 
	bSizer35->Add( m_TextCtrlProxyPasswd, 1, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer35, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer14, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelGeneralCommon->SetSizer( bSizer16 );
	m_PanelGeneralCommon->Layout();
	bSizer16->Fit( m_PanelGeneralCommon );
	m_NotebookBase->AddPage( m_PanelGeneralCommon, _("General options"), false );
	m_PanelAdvanced = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookAdvanced = new wxNotebook( m_PanelAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelGeneral = new wxPanel( m_NotebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer271;
	bSizer271 = new wxBoxSizer( wxVERTICAL );
	
	wxString m_RadioBoxGuiChoices[] = { _("Silent (no progressbar, much faster)"), _("Standard (recommanded)"), _("Verbose (not much used)") };
	int m_RadioBoxGuiNChoices = sizeof( m_RadioBoxGuiChoices ) / sizeof( wxString );
	m_RadioBoxGui = new wxRadioBox( m_PanelGeneral, wxID_ANY, _("GUI options"), wxDefaultPosition, wxDefaultSize, m_RadioBoxGuiNChoices, m_RadioBoxGuiChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxGui->SetSelection( 1 );
	bSizer271->Add( m_RadioBoxGui, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer11;
	sbSizer11 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneral, wxID_ANY, _("Predictor download") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextNumberFails = new wxStaticText( m_PanelGeneral, wxID_ANY, _("Maximum number of previous time steps if download fails"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNumberFails->Wrap( -1 );
	fgSizer2->Add( m_StaticTextNumberFails, 0, wxALL, 5 );
	
	m_TextCtrlMaxPrevStepsNb = new wxTextCtrl( m_PanelGeneral, wxID_ANY, _("5"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_TextCtrlMaxPrevStepsNb->SetMaxLength( 1 ); 
	fgSizer2->Add( m_TextCtrlMaxPrevStepsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextMaxRequestsNb = new wxStaticText( m_PanelGeneral, wxID_ANY, _("Maximum parallel requests number"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextMaxRequestsNb->Wrap( -1 );
	fgSizer2->Add( m_StaticTextMaxRequestsNb, 0, wxALL, 5 );
	
	m_TextCtrlMaxRequestsNb = new wxTextCtrl( m_PanelGeneral, wxID_ANY, _("3"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_TextCtrlMaxRequestsNb->SetMaxLength( 1 ); 
	fgSizer2->Add( m_TextCtrlMaxRequestsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_CheckBoxRestrictDownloads = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Restrict downloads to needed lead times."), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxRestrictDownloads->SetValue(true); 
	m_CheckBoxRestrictDownloads->Enable( false );
	
	fgSizer2->Add( m_CheckBoxRestrictDownloads, 0, wxALL, 5 );
	
	
	sbSizer11->Add( fgSizer2, 1, wxEXPAND, 5 );
	
	
	bSizer271->Add( sbSizer11, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneral, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_CheckBoxResponsiveness = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxResponsiveness->SetValue(true); 
	sbSizer151->Add( m_CheckBoxResponsiveness, 0, wxALL, 5 );
	
	m_CheckBoxMultiInstancesForecaster = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Allow multiple instances of the forecaster"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer151->Add( m_CheckBoxMultiInstancesForecaster, 0, wxALL, 5 );
	
	
	bSizer271->Add( sbSizer151, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelGeneral->SetSizer( bSizer271 );
	m_PanelGeneral->Layout();
	bSizer271->Fit( m_PanelGeneral );
	m_NotebookAdvanced->AddPage( m_PanelGeneral, _("General"), true );
	m_PanelProcessing = new wxPanel( m_NotebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer1611;
	bSizer1611 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer15;
	sbSizer15 = new wxStaticBoxSizer( new wxStaticBox( m_PanelProcessing, wxID_ANY, _("Multithreading") ), wxVERTICAL );
	
	m_CheckBoxAllowMultithreading = new wxCheckBox( m_PanelProcessing, wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxAllowMultithreading->SetValue(true); 
	sbSizer15->Add( m_CheckBoxAllowMultithreading, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextThreadsNb = new wxStaticText( m_PanelProcessing, wxID_ANY, _("Max nb of threads"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextThreadsNb->Wrap( -1 );
	bSizer221->Add( m_StaticTextThreadsNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_TextCtrlThreadsNb = new wxTextCtrl( m_PanelProcessing, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_TextCtrlThreadsNb->SetMaxLength( 0 ); 
	bSizer221->Add( m_TextCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	sbSizer15->Add( bSizer221, 0, wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextThreadsPriority = new wxStaticText( m_PanelProcessing, wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( m_StaticTextThreadsPriority, 0, wxALL, 5 );
	
	m_SliderThreadsPriority = new wxSlider( m_PanelProcessing, wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( m_SliderThreadsPriority, 1, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer15->Add( bSizer241, 0, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_RadioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Date array insertions (slower)"), _("Date array splitting (slower)") };
	int m_RadioBoxProcessingMethodsNChoices = sizeof( m_RadioBoxProcessingMethodsChoices ) / sizeof( wxString );
	m_RadioBoxProcessingMethods = new wxRadioBox( m_PanelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition, wxDefaultSize, m_RadioBoxProcessingMethodsNChoices, m_RadioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxProcessingMethods->SetSelection( 0 );
	m_RadioBoxProcessingMethods->SetToolTip( _("These options don't affect the results, only the processor efficiency.") );
	
	bSizer1611->Add( m_RadioBoxProcessingMethods, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_RadioBoxLinearAlgebraChoices[] = { _("Direct access to the coefficients"), _("Direct access to the coefficients and minimizing variable declarations"), _("Linear algebra using Eigen"), _("Linear algebra using Eigen and minimizing variable declarations (recommended)") };
	int m_RadioBoxLinearAlgebraNChoices = sizeof( m_RadioBoxLinearAlgebraChoices ) / sizeof( wxString );
	m_RadioBoxLinearAlgebra = new wxRadioBox( m_PanelProcessing, wxID_ANY, _("Linear algebra options"), wxDefaultPosition, wxDefaultSize, m_RadioBoxLinearAlgebraNChoices, m_RadioBoxLinearAlgebraChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxLinearAlgebra->SetSelection( 3 );
	m_RadioBoxLinearAlgebra->Enable( false );
	
	bSizer1611->Add( m_RadioBoxLinearAlgebra, 0, wxALL|wxEXPAND, 5 );
	
	
	m_PanelProcessing->SetSizer( bSizer1611 );
	m_PanelProcessing->Layout();
	bSizer1611->Fit( m_PanelProcessing );
	m_NotebookAdvanced->AddPage( m_PanelProcessing, _("Processing"), false );
	m_PanelUserDirectories = new wxPanel( m_NotebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer24;
	bSizer24 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( m_PanelUserDirectories, wxID_ANY, _("User specific paths") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer9;
	fgSizer9 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer9->SetFlexibleDirection( wxBOTH );
	fgSizer9->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextUserDirLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("User working directory:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextUserDirLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextUserDirLabel, 0, wxALL, 5 );
	
	m_StaticTextUserDir = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextUserDir->Wrap( -1 );
	fgSizer9->Add( m_StaticTextUserDir, 0, wxALL, 5 );
	
	m_StaticTextLogFileLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFileLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFileLabel, 0, wxALL, 5 );
	
	m_StaticTextLogFile = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFile->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFile, 0, wxALL, 5 );
	
	m_StaticTextPrefFileLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Preferences file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFileLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFileLabel, 0, wxALL, 5 );
	
	m_StaticTextPrefFile = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFile->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFile, 0, wxALL, 5 );
	
	
	sbSizer17->Add( fgSizer9, 1, wxEXPAND, 5 );
	
	
	bSizer24->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );
	
	
	m_PanelUserDirectories->SetSizer( bSizer24 );
	m_PanelUserDirectories->Layout();
	bSizer24->Fit( m_PanelUserDirectories );
	m_NotebookAdvanced->AddPage( m_PanelUserDirectories, _("User paths"), false );
	
	bSizer26->Add( m_NotebookAdvanced, 1, wxEXPAND | wxALL, 5 );
	
	
	m_PanelAdvanced->SetSizer( bSizer26 );
	m_PanelAdvanced->Layout();
	bSizer26->Fit( m_PanelAdvanced );
	m_NotebookBase->AddPage( m_PanelAdvanced, _("Advanced options"), false );
	
	bSizer15->Add( m_NotebookBase, 1, wxEXPAND | wxALL, 5 );
	
	m_ButtonsConfirmation = new wxStdDialogButtonSizer();
	m_ButtonsConfirmationOK = new wxButton( m_PanelBase, wxID_OK );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationOK );
	m_ButtonsConfirmationApply = new wxButton( m_PanelBase, wxID_APPLY );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationApply );
	m_ButtonsConfirmationCancel = new wxButton( m_PanelBase, wxID_CANCEL );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationCancel );
	m_ButtonsConfirmation->Realize();
	
	bSizer15->Add( m_ButtonsConfirmation, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelBase->SetSizer( bSizer15 );
	m_PanelBase->Layout();
	bSizer15->Fit( m_PanelBase );
	bSizer14->Add( m_PanelBase, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer14 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_CheckBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_ButtonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesForecasterVirtual::~asFramePreferencesForecasterVirtual()
{
	// Disconnect Events
	m_CheckBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_ButtonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesForecasterVirtual::SaveAndClose ), NULL, this );
	
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
	
	m_FilePickerBatchFile = new wxFilePickerCtrl( m_wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizer49->Add( m_FilePickerBatchFile, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
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
