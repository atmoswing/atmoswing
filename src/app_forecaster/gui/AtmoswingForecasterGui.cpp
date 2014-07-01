///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Nov  6 2013)
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
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( m_PanelMain, wxID_ANY, _("Current model state") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_LedDownloading = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedDownloading->SetState( awxLED_OFF );
	fgSizer1->Add( m_LedDownloading, 0, wxALL, 5 );
	
	m_StaticTextDownloading = new wxStaticText( m_PanelMain, wxID_ANY, _("Downloading predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDownloading->Wrap( -1 );
	fgSizer1->Add( m_StaticTextDownloading, 0, wxALL, 5 );
	
	m_LedLoading = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedLoading->SetState( awxLED_OFF );
	fgSizer1->Add( m_LedLoading, 0, wxALL, 5 );
	
	m_StaticTextLoading = new wxStaticText( m_PanelMain, wxID_ANY, _("Loading data"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLoading->Wrap( -1 );
	fgSizer1->Add( m_StaticTextLoading, 0, wxALL, 5 );
	
	m_LedProcessing = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedProcessing->SetState( awxLED_OFF );
	fgSizer1->Add( m_LedProcessing, 0, wxALL, 5 );
	
	m_StaticTextProcessing = new wxStaticText( m_PanelMain, wxID_ANY, _("Processing"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProcessing->Wrap( -1 );
	fgSizer1->Add( m_StaticTextProcessing, 0, wxALL, 5 );
	
	m_LedSaving = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedSaving->SetState( awxLED_OFF );
	fgSizer1->Add( m_LedSaving, 0, wxALL, 5 );
	
	m_StaticTextSaving = new wxStaticText( m_PanelMain, wxID_ANY, _("Saving results"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextSaving->Wrap( -1 );
	fgSizer1->Add( m_StaticTextSaving, 0, wxALL, 5 );
	
	
	sbSizer5->Add( fgSizer1, 0, wxEXPAND, 5 );
	
	
	bSizer19->Add( sbSizer5, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer18->Add( bSizer19, 0, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( m_PanelMain, wxID_ANY, _("List of the forecasting models") ), wxVERTICAL );
	
	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );
	
	bSizer22->SetMinSize( wxSize( -1,200 ) ); 
	m_ScrolledWindowModels = new wxScrolledWindow( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxVSCROLL );
	m_ScrolledWindowModels->SetScrollRate( 5, 5 );
	m_ScrolledWindowModels->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_ScrolledWindowModels->SetMinSize( wxSize( -1,200 ) );
	
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );
	
	m_SizerModels = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer32->Add( m_SizerModels, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_BpButtonAdd = new wxBitmapButton( m_ScrolledWindowModels, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 28,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_BpButtonAdd->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_ACTIVEBORDER ) );
	m_BpButtonAdd->SetToolTip( _("Add a model.") );
	
	bSizer34->Add( m_BpButtonAdd, 0, wxALL, 5 );
	
	
	bSizer32->Add( bSizer34, 0, wxLEFT, 5 );
	
	
	m_ScrolledWindowModels->SetSizer( bSizer32 );
	m_ScrolledWindowModels->Layout();
	bSizer32->Fit( m_ScrolledWindowModels );
	bSizer22->Add( m_ScrolledWindowModels, 1, wxEXPAND | wxALL, 5 );
	
	
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
	wxMenuItem* m_MenuItemSaveDefaultModelsList;
	m_MenuItemSaveDefaultModelsList = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Save models list as default") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemSaveDefaultModelsList );
	
	wxMenuItem* m_MenuItemLoadDefaultModelsList;
	m_MenuItemLoadDefaultModelsList = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Load default models list") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemLoadDefaultModelsList );
	
	wxMenuItem* m_MenuItemSaveModelList;
	m_MenuItemSaveModelList = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Save models list") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemSaveModelList );
	
	wxMenuItem* m_MenuItemLoadModelsList;
	m_MenuItemLoadModelsList = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Load models list") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemLoadModelsList );
	
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
	wxMenuItem* m_MenuItemLogLevel1;
	m_MenuItemLogLevel1 = new wxMenuItem( m_MenuLogLevel, wxID_ANY, wxString( _("Only errors") ) , wxEmptyString, wxITEM_CHECK );
	m_MenuLogLevel->Append( m_MenuItemLogLevel1 );
	
	wxMenuItem* m_MenuItemLogLevel2;
	m_MenuItemLogLevel2 = new wxMenuItem( m_MenuLogLevel, wxID_ANY, wxString( _("Errors and warnings") ) , wxEmptyString, wxITEM_CHECK );
	m_MenuLogLevel->Append( m_MenuItemLogLevel2 );
	
	wxMenuItem* m_MenuItemLogLevel3;
	m_MenuItemLogLevel3 = new wxMenuItem( m_MenuLogLevel, wxID_ANY, wxString( _("Verbose") ) , wxEmptyString, wxITEM_CHECK );
	m_MenuLogLevel->Append( m_MenuItemLogLevel3 );
	
	m_MenuLog->Append( -1, _("Log level"), m_MenuLogLevel );
	
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
	m_BpButtonAdd->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::AddForecastingModel ), NULL, this );
	this->Connect( m_MenuItemSaveDefaultModelsList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListSaveAsDefault ) );
	this->Connect( m_MenuItemLoadDefaultModelsList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListLoadDefault ) );
	this->Connect( m_MenuItemSaveModelList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListSave ) );
	this->Connect( m_MenuItemLoadModelsList->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListLoad ) );
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
	m_BpButtonAdd->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameMainVirtual::AddForecastingModel ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListSaveAsDefault ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListLoadDefault ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListSave ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMainVirtual::ModelsListLoad ) );
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

asPanelForecastingModelVirtual::asPanelForecastingModelVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	m_SizerPanel = new wxBoxSizer( wxVERTICAL );
	
	m_SizerHeader = new wxBoxSizer( wxHORIZONTAL );
	
	m_Led = new awxLed( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_RED, 0 );
	m_Led->SetState( awxLED_OFF );
	m_SizerHeader->Add( m_Led, 0, wxALL, 5 );
	
	m_StaticTextModelName = new wxStaticText( this, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextModelName->Wrap( -1 );
	m_SizerHeader->Add( m_StaticTextModelName, 1, wxALL, 5 );
	
	m_BpButtonReduce = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_SizerHeader->Add( m_BpButtonReduce, 0, 0, 5 );
	
	m_BpButtonClose = new wxBitmapButton( this, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_SizerHeader->Add( m_BpButtonClose, 0, 0, 5 );
	
	
	m_SizerPanel->Add( m_SizerHeader, 0, wxEXPAND, 5 );
	
	m_SizerFields = new wxBoxSizer( wxVERTICAL );
	
	m_StaticTextModelNameInput = new wxStaticText( this, wxID_ANY, _("Model tag name (short!)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextModelNameInput->Wrap( -1 );
	m_SizerFields->Add( m_StaticTextModelNameInput, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_TextCtrlModelName = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlModelName->SetMaxLength( 0 ); 
	m_SizerFields->Add( m_TextCtrlModelName, 0, wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextModelDescriptionInput = new wxStaticText( this, wxID_ANY, _("Model description (no accent!)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextModelDescriptionInput->Wrap( -1 );
	m_SizerFields->Add( m_StaticTextModelDescriptionInput, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_TextCtrlModelDescription = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlModelDescription->SetMaxLength( 0 ); 
	m_SizerFields->Add( m_TextCtrlModelDescription, 0, wxEXPAND|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextParametersFileName = new wxStaticText( this, wxID_ANY, _("Parameters file name"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextParametersFileName->Wrap( -1 );
	m_SizerFields->Add( m_StaticTextParametersFileName, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_TextCtrlParametersFileName = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlParametersFileName->SetMaxLength( 0 ); 
	m_SizerFields->Add( m_TextCtrlParametersFileName, 0, wxEXPAND|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextPredictandDB = new wxStaticText( this, wxID_ANY, _("Predictand database"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPredictandDB->Wrap( -1 );
	m_SizerFields->Add( m_StaticTextPredictandDB, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_TextCtrlPredictandDB = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlPredictandDB->SetMaxLength( 0 ); 
	m_SizerFields->Add( m_TextCtrlPredictandDB, 0, wxEXPAND|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextPredictorsArchiveDir = new wxStaticText( this, wxID_ANY, _("Predictors archive directory (if different from the preferences)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPredictorsArchiveDir->Wrap( -1 );
	m_SizerFields->Add( m_StaticTextPredictorsArchiveDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerPredictorsArchive = new wxDirPickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	m_SizerFields->Add( m_DirPickerPredictorsArchive, 0, wxEXPAND|wxRIGHT|wxLEFT, 5 );
	
	
	m_SizerPanel->Add( m_SizerFields, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( m_SizerPanel );
	this->Layout();
	m_SizerPanel->Fit( this );
	
	// Connect Events
	m_BpButtonReduce->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastingModelVirtual::ReducePanel ), NULL, this );
	m_BpButtonClose->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastingModelVirtual::ClosePanel ), NULL, this );
	m_TextCtrlModelName->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( asPanelForecastingModelVirtual::ChangeModelName ), NULL, this );
}

asPanelForecastingModelVirtual::~asPanelForecastingModelVirtual()
{
	// Disconnect Events
	m_BpButtonReduce->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastingModelVirtual::ReducePanel ), NULL, this );
	m_BpButtonClose->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelForecastingModelVirtual::ClosePanel ), NULL, this );
	m_TextCtrlModelName->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( asPanelForecastingModelVirtual::ChangeModelName ), NULL, this );
	
}
