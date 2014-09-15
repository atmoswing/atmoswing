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

#include "AtmoswingViewerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameForecastVirtual::asFrameForecastVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxVERTICAL );
	
	m_SplitterGIS = new wxSplitterWindow( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_SplitterGIS->Connect( wxEVT_IDLE, wxIdleEventHandler( asFrameForecastVirtual::m_SplitterGISOnIdle ), NULL, this );
	m_SplitterGIS->SetMinimumPaneSize( 270 );
	
	m_ScrolledWindowOptions = new wxScrolledWindow( m_SplitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_ScrolledWindowOptions->SetScrollRate( 5, 5 );
	m_SizerScrolledWindow = new wxBoxSizer( wxVERTICAL );
	
	
	m_ScrolledWindowOptions->SetSizer( m_SizerScrolledWindow );
	m_ScrolledWindowOptions->Layout();
	m_SizerScrolledWindow->Fit( m_ScrolledWindowOptions );
	m_PanelContent = new wxPanel( m_SplitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerContent = new wxBoxSizer( wxVERTICAL );
	
	m_PanelTop = new wxPanel( m_PanelContent, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxTAB_TRAVERSAL );
	m_PanelTop->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	
	m_SizerTop = new wxBoxSizer( wxHORIZONTAL );
	
	m_SizerTopLeft = new wxBoxSizer( wxVERTICAL );
	
	m_StaticTextForecastDate = new wxStaticText( m_PanelTop, wxID_ANY, _("No forecast opened"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastDate->Wrap( -1 );
	m_StaticTextForecastDate->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	m_StaticTextForecastDate->SetForegroundColour( wxColour( 255, 255, 255 ) );
	
	m_SizerTopLeft->Add( m_StaticTextForecastDate, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextForecastModel = new wxStaticText( m_PanelTop, wxID_ANY, _("No model selected"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastModel->Wrap( -1 );
	m_StaticTextForecastModel->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	m_StaticTextForecastModel->SetForegroundColour( wxColour( 255, 255, 255 ) );
	
	m_SizerTopLeft->Add( m_StaticTextForecastModel, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	m_SizerTop->Add( m_SizerTopLeft, 0, wxALIGN_LEFT|wxEXPAND, 5 );
	
	m_SizerTopRight = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer39;
	bSizer39 = new wxBoxSizer( wxVERTICAL );
	
	m_SizerLeadTimeSwitch = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer39->Add( m_SizerLeadTimeSwitch, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_SizerTopRight->Add( bSizer39, 1, wxALIGN_RIGHT, 5 );
	
	
	m_SizerTop->Add( m_SizerTopRight, 1, wxALIGN_RIGHT|wxEXPAND, 5 );
	
	
	m_PanelTop->SetSizer( m_SizerTop );
	m_PanelTop->Layout();
	m_SizerTop->Fit( m_PanelTop );
	m_SizerContent->Add( m_PanelTop, 0, wxEXPAND, 5 );
	
	m_PanelGIS = new wxPanel( m_PanelContent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerGIS = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelGIS->SetSizer( m_SizerGIS );
	m_PanelGIS->Layout();
	m_SizerGIS->Fit( m_PanelGIS );
	m_SizerContent->Add( m_PanelGIS, 1, wxEXPAND, 5 );
	
	
	m_PanelContent->SetSizer( m_SizerContent );
	m_PanelContent->Layout();
	m_SizerContent->Fit( m_PanelContent );
	m_SplitterGIS->SplitVertically( m_ScrolledWindowOptions, m_PanelContent, 270 );
	bSizer11->Add( m_SplitterGIS, 1, wxEXPAND|wxALL, 4 );
	
	
	m_PanelMain->SetSizer( bSizer11 );
	m_PanelMain->Layout();
	bSizer11->Fit( m_PanelMain );
	bSizer3->Add( m_PanelMain, 1, wxEXPAND, 2 );
	
	
	this->SetSizer( bSizer3 );
	this->Layout();
	bSizer3->Fit( this );
	m_MenuBar = new wxMenuBar( 0 );
	m_MenuFile = new wxMenu();
	wxMenuItem* m_MenuItemOpenWorkspace;
	m_MenuItemOpenWorkspace = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Open a workspace") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenWorkspace );
	
	wxMenuItem* m_MenuItemSaveWorkspace;
	m_MenuItemSaveWorkspace = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Save the workspace") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemSaveWorkspace );
	
	wxMenuItem* m_MenuItemSaveWorkspaceAs;
	m_MenuItemSaveWorkspaceAs = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Save the workspace as") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemSaveWorkspaceAs );
	
	wxMenuItem* m_MenuItemNewWorkspace;
	m_MenuItemNewWorkspace = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Create a new workspace") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemNewWorkspace );
	
	m_MenuFile->AppendSeparator();
	
	wxMenuItem* m_MenuItemOpenForecast;
	m_MenuItemOpenForecast = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Open a forecast file") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenForecast );
	
	m_MenuFile->AppendSeparator();
	
	wxMenuItem* m_MenuItemOpenGISLayer;
	m_MenuItemOpenGISLayer = new wxMenuItem( m_MenuFile, wxID_OPEN, wxString( _("Open a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenGISLayer );
	
	wxMenuItem* m_MenuItemCloseGISLayer;
	m_MenuItemCloseGISLayer = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Close a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemCloseGISLayer );
	
	wxMenuItem* m_MenuItemMoveGISLayer;
	m_MenuItemMoveGISLayer = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Move the selected layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemMoveGISLayer );
	
	m_MenuFile->AppendSeparator();
	
	wxMenuItem* m_MenuItemQuit;
	m_MenuItemQuit = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Quit") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemQuit );
	
	m_MenuBar->Append( m_MenuFile, _("File") ); 
	
	m_MenuOptions = new wxMenu();
	wxMenuItem* m_MenuItemPreferences;
	m_MenuItemPreferences = new wxMenuItem( m_MenuOptions, wxID_ANY, wxString( _("Preferences") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuOptions->Append( m_MenuItemPreferences );
	
	m_MenuBar->Append( m_MenuOptions, _("Options") ); 
	
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
	m_ToolBar->Realize(); 
	
	m_StatusBar = this->CreateStatusBar( 2, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( m_MenuItemOpenWorkspace->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenWorkspace ) );
	this->Connect( m_MenuItemSaveWorkspace->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnSaveWorkspace ) );
	this->Connect( m_MenuItemSaveWorkspaceAs->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnSaveWorkspaceAs ) );
	this->Connect( m_MenuItemNewWorkspace->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnNewWorkspace ) );
	this->Connect( m_MenuItemOpenForecast->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenForecast ) );
	this->Connect( m_MenuItemOpenGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenLayer ) );
	this->Connect( m_MenuItemCloseGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnCloseLayer ) );
	this->Connect( m_MenuItemMoveGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnMoveLayer ) );
	this->Connect( m_MenuItemQuit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnQuit ) );
	this->Connect( m_MenuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OpenFramePreferences ) );
	this->Connect( m_MenuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnShowLog ) );
	this->Connect( m_MenuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel1 ) );
	this->Connect( m_MenuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel2 ) );
	this->Connect( m_MenuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel3 ) );
	this->Connect( m_MenuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OpenFrameAbout ) );
}

asFrameForecastVirtual::~asFrameForecastVirtual()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenWorkspace ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnSaveWorkspace ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnSaveWorkspaceAs ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnNewWorkspace ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenForecast ) );
	this->Disconnect( wxID_OPEN, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenLayer ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnCloseLayer ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnMoveLayer ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnQuit ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OpenFramePreferences ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnShowLog ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel1 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel2 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel3 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OpenFrameAbout ) );
	
}

asFramePlotTimeSeriesVirtual::asFramePlotTimeSeriesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 500,300 ), wxDefaultSize );
	
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelStationName = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextStationName = new wxStaticText( m_PanelStationName, wxID_ANY, _("Station name"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextStationName->Wrap( -1 );
	m_StaticTextStationName->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 90, false, wxEmptyString ) );
	
	bSizer37->Add( m_StaticTextStationName, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_ButtonSaveTxt = new wxButton( m_PanelStationName, wxID_ANY, _("Export as txt"), wxDefaultPosition, wxSize( 80,20 ), 0 );
	m_ButtonSaveTxt->SetFont( wxFont( 8, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer37->Add( m_ButtonSaveTxt, 0, wxALL, 5 );
	
	m_ButtonPreview = new wxButton( m_PanelStationName, wxID_ANY, _("Preview"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	m_ButtonPreview->Enable( false );
	m_ButtonPreview->Hide();
	
	bSizer37->Add( m_ButtonPreview, 0, wxALL, 5 );
	
	m_ButtonPrint = new wxButton( m_PanelStationName, wxID_ANY, _("Print"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	m_ButtonPrint->Enable( false );
	m_ButtonPrint->Hide();
	
	bSizer37->Add( m_ButtonPrint, 0, wxALL, 5 );
	
	
	bSizer29->Add( bSizer37, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_PanelStationName->SetSizer( bSizer29 );
	m_PanelStationName->Layout();
	bSizer29->Fit( m_PanelStationName );
	bSizer13->Add( m_PanelStationName, 0, wxEXPAND, 5 );
	
	m_Splitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_Splitter->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotTimeSeriesVirtual::m_SplitterOnIdle ), NULL, this );
	m_Splitter->SetMinimumPaneSize( 150 );
	
	m_PanelLeft = new wxPanel( m_Splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	wxArrayString m_CheckListTocChoices;
	m_CheckListToc = new wxCheckListBox( m_PanelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_CheckListTocChoices, 0 );
	bSizer27->Add( m_CheckListToc, 1, wxEXPAND, 5 );
	
	wxArrayString m_CheckListPastChoices;
	m_CheckListPast = new wxCheckListBox( m_PanelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_CheckListPastChoices, 0 );
	bSizer27->Add( m_CheckListPast, 1, wxEXPAND|wxTOP, 5 );
	
	
	m_PanelLeft->SetSizer( bSizer27 );
	m_PanelLeft->Layout();
	bSizer27->Fit( m_PanelLeft );
	m_PanelRight = new wxPanel( m_Splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_SizerPlot = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelRight->SetSizer( m_SizerPlot );
	m_PanelRight->Layout();
	m_SizerPlot->Fit( m_PanelRight );
	m_Splitter->SplitVertically( m_PanelLeft, m_PanelRight, 150 );
	bSizer13->Add( m_Splitter, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer13 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_ButtonSaveTxt->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	m_ButtonPreview->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	m_ButtonPrint->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	m_CheckListToc->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	m_CheckListPast->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
}

asFramePlotTimeSeriesVirtual::~asFramePlotTimeSeriesVirtual()
{
	// Disconnect Events
	m_ButtonSaveTxt->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	m_ButtonPreview->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	m_ButtonPrint->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	m_CheckListToc->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	m_CheckListPast->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	
}

asFramePlotDistributionsVirutal::asFramePlotDistributionsVirutal( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextForecast = new wxStaticText( m_PanelOptions, wxID_ANY, _("Select forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecast->Wrap( -1 );
	fgSizer1->Add( m_StaticTextForecast, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextStation = new wxStaticText( m_PanelOptions, wxID_ANY, _("Select station"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextStation->Wrap( -1 );
	fgSizer1->Add( m_StaticTextStation, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextDate = new wxStaticText( m_PanelOptions, wxID_ANY, _("Select date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDate->Wrap( -1 );
	fgSizer1->Add( m_StaticTextDate, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_ChoiceForecastChoices;
	m_ChoiceForecast = new wxChoice( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceForecastChoices, 0 );
	m_ChoiceForecast->SetSelection( 0 );
	fgSizer1->Add( m_ChoiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_ChoiceStationChoices;
	m_ChoiceStation = new wxChoice( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceStationChoices, 0 );
	m_ChoiceStation->SetSelection( 0 );
	fgSizer1->Add( m_ChoiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_ChoiceDateChoices;
	m_ChoiceDate = new wxChoice( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceDateChoices, 0 );
	m_ChoiceDate->SetSelection( 0 );
	fgSizer1->Add( m_ChoiceDate, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer29->Add( fgSizer1, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_PanelOptions->SetSizer( bSizer29 );
	m_PanelOptions->Layout();
	bSizer29->Fit( m_PanelOptions );
	bSizer13->Add( m_PanelOptions, 0, wxEXPAND, 5 );
	
	m_Notebook = new wxNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelPredictands = new wxPanel( m_Notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );
	
	m_SplitterPredictands = new wxSplitterWindow( m_PanelPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_SplitterPredictands->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::m_SplitterPredictandsOnIdle ), NULL, this );
	m_SplitterPredictands->SetMinimumPaneSize( 150 );
	
	m_PanelPredictandsLeft = new wxPanel( m_SplitterPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	wxArrayString m_CheckListTocPredictandsChoices;
	m_CheckListTocPredictands = new wxCheckListBox( m_PanelPredictandsLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_CheckListTocPredictandsChoices, 0 );
	bSizer27->Add( m_CheckListTocPredictands, 1, wxEXPAND, 5 );
	
	
	m_PanelPredictandsLeft->SetSizer( bSizer27 );
	m_PanelPredictandsLeft->Layout();
	bSizer27->Fit( m_PanelPredictandsLeft );
	m_PanelPredictandsRight = new wxPanel( m_SplitterPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_SizerPlotPredictands = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelPredictandsRight->SetSizer( m_SizerPlotPredictands );
	m_PanelPredictandsRight->Layout();
	m_SizerPlotPredictands->Fit( m_PanelPredictandsRight );
	m_SplitterPredictands->SplitVertically( m_PanelPredictandsLeft, m_PanelPredictandsRight, 150 );
	bSizer22->Add( m_SplitterPredictands, 1, wxEXPAND|wxTOP, 5 );
	
	
	m_PanelPredictands->SetSizer( bSizer22 );
	m_PanelPredictands->Layout();
	bSizer22->Fit( m_PanelPredictands );
	m_Notebook->AddPage( m_PanelPredictands, _("Predictands distribution"), true );
	m_PanelCriteria = new wxPanel( m_Notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerPlotCriteria = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelCriteria->SetSizer( m_SizerPlotCriteria );
	m_PanelCriteria->Layout();
	m_SizerPlotCriteria->Fit( m_PanelCriteria );
	m_Notebook->AddPage( m_PanelCriteria, _("Criteria distribution"), false );
	
	bSizer13->Add( m_Notebook, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer13 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_ChoiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	m_ChoiceStation->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	m_ChoiceDate->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	m_CheckListTocPredictands->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
}

asFramePlotDistributionsVirutal::~asFramePlotDistributionsVirutal()
{
	// Disconnect Events
	m_ChoiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	m_ChoiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	m_ChoiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	m_CheckListTocPredictands->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
	
}

asFrameGridAnalogsValuesVirtual::asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 550,-1 ), wxSize( 550,-1 ) );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer291;
	bSizer291 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextForecast = new wxStaticText( m_PanelOptions, wxID_ANY, _("Select forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecast->Wrap( -1 );
	fgSizer1->Add( m_StaticTextForecast, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextStation = new wxStaticText( m_PanelOptions, wxID_ANY, _("Select station"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextStation->Wrap( -1 );
	fgSizer1->Add( m_StaticTextStation, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextDate = new wxStaticText( m_PanelOptions, wxID_ANY, _("Select date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDate->Wrap( -1 );
	fgSizer1->Add( m_StaticTextDate, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_ChoiceForecastChoices;
	m_ChoiceForecast = new wxChoice( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceForecastChoices, 0 );
	m_ChoiceForecast->SetSelection( 0 );
	fgSizer1->Add( m_ChoiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_ChoiceStationChoices;
	m_ChoiceStation = new wxChoice( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceStationChoices, 0 );
	m_ChoiceStation->SetSelection( 0 );
	fgSizer1->Add( m_ChoiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_ChoiceDateChoices;
	m_ChoiceDate = new wxChoice( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceDateChoices, 0 );
	m_ChoiceDate->SetSelection( 0 );
	fgSizer1->Add( m_ChoiceDate, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer291->Add( fgSizer1, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_PanelOptions->SetSizer( bSizer291 );
	m_PanelOptions->Layout();
	bSizer291->Fit( m_PanelOptions );
	bSizer29->Add( m_PanelOptions, 0, wxEXPAND, 5 );
	
	m_Grid = new wxGrid( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	
	// Grid
	m_Grid->CreateGrid( 5, 4 );
	m_Grid->EnableEditing( false );
	m_Grid->EnableGridLines( true );
	m_Grid->EnableDragGridSize( false );
	m_Grid->SetMargins( 0, 0 );
	
	// Columns
	m_Grid->SetColSize( 0, 95 );
	m_Grid->SetColSize( 1, 122 );
	m_Grid->SetColSize( 2, 130 );
	m_Grid->SetColSize( 3, 130 );
	m_Grid->EnableDragColMove( false );
	m_Grid->EnableDragColSize( true );
	m_Grid->SetColLabelSize( 20 );
	m_Grid->SetColLabelValue( 0, _("Analog") );
	m_Grid->SetColLabelValue( 1, _("Date") );
	m_Grid->SetColLabelValue( 2, _("Precipitation (mm)") );
	m_Grid->SetColLabelValue( 3, _("Criteria") );
	m_Grid->SetColLabelAlignment( wxALIGN_CENTRE, wxALIGN_CENTRE );
	
	// Rows
	m_Grid->EnableDragRowSize( true );
	m_Grid->SetRowLabelSize( 40 );
	m_Grid->SetRowLabelAlignment( wxALIGN_CENTRE, wxALIGN_CENTRE );
	
	// Label Appearance
	
	// Cell Defaults
	m_Grid->SetDefaultCellAlignment( wxALIGN_RIGHT, wxALIGN_TOP );
	bSizer29->Add( m_Grid, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer29 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_ChoiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceForecastChange ), NULL, this );
	m_ChoiceStation->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceStationChange ), NULL, this );
	m_ChoiceDate->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceDateChange ), NULL, this );
	m_Grid->Connect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_Grid->Connect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_Grid->Connect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_Grid->Connect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
}

asFrameGridAnalogsValuesVirtual::~asFrameGridAnalogsValuesVirtual()
{
	// Disconnect Events
	m_ChoiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceForecastChange ), NULL, this );
	m_ChoiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceStationChange ), NULL, this );
	m_ChoiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceDateChange ), NULL, this );
	m_Grid->Disconnect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_Grid->Disconnect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_Grid->Disconnect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_Grid->Disconnect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	
}

asFramePredictorsVirtual::asFramePredictorsVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 800,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );
	
	m_panel15 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	m_SplitterToc = new wxSplitterWindow( m_panel15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_SplitterToc->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::m_SplitterTocOnIdle ), NULL, this );
	m_SplitterToc->SetMinimumPaneSize( 165 );
	
	m_ScrolledWindowOptions = new wxScrolledWindow( m_SplitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_ScrolledWindowOptions->SetScrollRate( 5, 5 );
	m_SizerScrolledWindow = new wxBoxSizer( wxVERTICAL );
	
	m_StaticTextChoiceForecast = new wxStaticText( m_ScrolledWindowOptions, wxID_ANY, _("Forecast model"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextChoiceForecast->Wrap( -1 );
	m_SizerScrolledWindow->Add( m_StaticTextChoiceForecast, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	wxArrayString m_ChoiceForecastChoices;
	m_ChoiceForecast = new wxChoice( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceForecastChoices, 0 );
	m_ChoiceForecast->SetSelection( 0 );
	m_ChoiceForecast->SetMaxSize( wxSize( 150,-1 ) );
	
	m_SizerScrolledWindow->Add( m_ChoiceForecast, 0, wxEXPAND|wxALL, 5 );
	
	m_StaticTextCheckListPredictors = new wxStaticText( m_ScrolledWindowOptions, wxID_ANY, _("Possible predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextCheckListPredictors->Wrap( -1 );
	m_SizerScrolledWindow->Add( m_StaticTextCheckListPredictors, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	wxArrayString m_CheckListPredictorsChoices;
	m_CheckListPredictors = new wxCheckListBox( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_CheckListPredictorsChoices, 0 );
	m_SizerScrolledWindow->Add( m_CheckListPredictors, 1, wxEXPAND, 5 );
	
	m_StaticTextTocLeft = new wxStaticText( m_ScrolledWindowOptions, wxID_ANY, _("Layers of the left panel"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextTocLeft->Wrap( -1 );
	m_SizerScrolledWindow->Add( m_StaticTextTocLeft, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextTocRight = new wxStaticText( m_ScrolledWindowOptions, wxID_ANY, _("Layers of the right panel"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextTocRight->Wrap( -1 );
	m_SizerScrolledWindow->Add( m_StaticTextTocRight, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_ScrolledWindowOptions->SetSizer( m_SizerScrolledWindow );
	m_ScrolledWindowOptions->Layout();
	m_SizerScrolledWindow->Fit( m_ScrolledWindowOptions );
	m_PanelGIS = new wxPanel( m_SplitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerGIS = new wxBoxSizer( wxHORIZONTAL );
	
	m_PanelLeft = new wxPanel( m_PanelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer371;
	bSizer371 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextTargetDates = new wxStaticText( m_PanelLeft, wxID_ANY, _("Forecast date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextTargetDates->Wrap( -1 );
	bSizer34->Add( m_StaticTextTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxArrayString m_ChoiceTargetDatesChoices;
	m_ChoiceTargetDates = new wxChoice( m_PanelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceTargetDatesChoices, 0 );
	m_ChoiceTargetDates->SetSelection( 0 );
	bSizer34->Add( m_ChoiceTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer371->Add( bSizer34, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_PanelGISLeft = new wxPanel( m_PanelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_SizerGISLeft = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelGISLeft->SetSizer( m_SizerGISLeft );
	m_PanelGISLeft->Layout();
	m_SizerGISLeft->Fit( m_PanelGISLeft );
	bSizer371->Add( m_PanelGISLeft, 1, wxEXPAND, 5 );
	
	
	m_PanelLeft->SetSizer( bSizer371 );
	m_PanelLeft->Layout();
	bSizer371->Fit( m_PanelLeft );
	m_SizerGIS->Add( m_PanelLeft, 1, wxEXPAND, 5 );
	
	m_PanelSwitch = new wxPanel( m_PanelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer40;
	bSizer40 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* m_SizerSwitch;
	m_SizerSwitch = new wxBoxSizer( wxVERTICAL );
	
	m_BpButtonSwitchRight = new wxBitmapButton( m_PanelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_SizerSwitch->Add( m_BpButtonSwitchRight, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_BpButtonSwitchLeft = new wxBitmapButton( m_PanelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_SizerSwitch->Add( m_BpButtonSwitchLeft, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer40->Add( m_SizerSwitch, 1, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	m_PanelSwitch->SetSizer( bSizer40 );
	m_PanelSwitch->Layout();
	bSizer40->Fit( m_PanelSwitch );
	m_SizerGIS->Add( m_PanelSwitch, 0, wxEXPAND, 5 );
	
	m_PanelRight = new wxPanel( m_PanelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextAnalogDates = new wxStaticText( m_PanelRight, wxID_ANY, _("Analogs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAnalogDates->Wrap( -1 );
	bSizer35->Add( m_StaticTextAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxArrayString m_ChoiceAnalogDatesChoices;
	m_ChoiceAnalogDates = new wxChoice( m_PanelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceAnalogDatesChoices, 0 );
	m_ChoiceAnalogDates->SetSelection( 0 );
	bSizer35->Add( m_ChoiceAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer38->Add( bSizer35, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_PanelGISRight = new wxPanel( m_PanelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_SizerGISRight = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelGISRight->SetSizer( m_SizerGISRight );
	m_PanelGISRight->Layout();
	m_SizerGISRight->Fit( m_PanelGISRight );
	bSizer38->Add( m_PanelGISRight, 1, wxEXPAND, 5 );
	
	
	m_PanelRight->SetSizer( bSizer38 );
	m_PanelRight->Layout();
	bSizer38->Fit( m_PanelRight );
	m_SizerGIS->Add( m_PanelRight, 1, wxEXPAND, 5 );
	
	
	m_PanelGIS->SetSizer( m_SizerGIS );
	m_PanelGIS->Layout();
	m_SizerGIS->Fit( m_PanelGIS );
	m_SplitterToc->SplitVertically( m_ScrolledWindowOptions, m_PanelGIS, 170 );
	bSizer26->Add( m_SplitterToc, 1, wxEXPAND, 5 );
	
	
	m_panel15->SetSizer( bSizer26 );
	m_panel15->Layout();
	bSizer26->Fit( m_panel15 );
	bSizer25->Add( m_panel15, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer25 );
	this->Layout();
	m_Menubar = new wxMenuBar( 0 );
	m_MenuFile = new wxMenu();
	wxMenuItem* m_MenuItemOpenGisLayer;
	m_MenuItemOpenGisLayer = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Open GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenGisLayer );
	
	m_Menubar->Append( m_MenuFile, _("File") ); 
	
	m_MenuTools = new wxMenu();
	m_Menubar->Append( m_MenuTools, _("Tools") ); 
	
	this->SetMenuBar( m_Menubar );
	
	m_ToolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY ); 
	m_ToolBar->Realize(); 
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_ChoiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	m_CheckListPredictors->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	m_ChoiceTargetDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	m_BpButtonSwitchRight->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	m_BpButtonSwitchLeft->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	m_ChoiceAnalogDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );
	this->Connect( m_MenuItemOpenGisLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnOpenLayer ) );
}

asFramePredictorsVirtual::~asFramePredictorsVirtual()
{
	// Disconnect Events
	m_ChoiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	m_CheckListPredictors->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	m_ChoiceTargetDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	m_BpButtonSwitchRight->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	m_BpButtonSwitchLeft->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	m_ChoiceAnalogDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnOpenLayer ) );
	
}

asFrameMeteorologicalSituationVirtual::asFrameMeteorologicalSituationVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 800,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );
	
	m_Splitter = new wxSplitterWindow( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_Splitter->Connect( wxEVT_IDLE, wxIdleEventHandler( asFrameMeteorologicalSituationVirtual::m_SplitterOnIdle ), NULL, this );
	m_Splitter->SetMinimumPaneSize( 270 );
	
	m_ScrolledWindow = new wxScrolledWindow( m_Splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_ScrolledWindow->SetScrollRate( 5, 5 );
	m_SizerScrolledWindow = new wxBoxSizer( wxVERTICAL );
	
	
	m_ScrolledWindow->SetSizer( m_SizerScrolledWindow );
	m_ScrolledWindow->Layout();
	m_SizerScrolledWindow->Fit( m_ScrolledWindow );
	m_PanelContent = new wxPanel( m_Splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerContent = new wxBoxSizer( wxVERTICAL );
	
	m_PanelGIS = new wxPanel( m_PanelContent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerGIS = new wxBoxSizer( wxVERTICAL );
	
	
	m_PanelGIS->SetSizer( m_SizerGIS );
	m_PanelGIS->Layout();
	m_SizerGIS->Fit( m_PanelGIS );
	m_SizerContent->Add( m_PanelGIS, 1, wxEXPAND, 5 );
	
	
	m_PanelContent->SetSizer( m_SizerContent );
	m_PanelContent->Layout();
	m_SizerContent->Fit( m_PanelContent );
	m_Splitter->SplitVertically( m_ScrolledWindow, m_PanelContent, 270 );
	bSizer38->Add( m_Splitter, 1, wxEXPAND|wxALL, 5 );
	
	
	m_PanelMain->SetSizer( bSizer38 );
	m_PanelMain->Layout();
	bSizer38->Fit( m_PanelMain );
	bSizer37->Add( m_PanelMain, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer37 );
	this->Layout();
	m_Menubar = new wxMenuBar( 0 );
	m_MenuFile = new wxMenu();
	wxMenuItem* m_MenuItemOpenGisLayer;
	m_MenuItemOpenGisLayer = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Open GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenGisLayer );
	
	m_Menubar->Append( m_MenuFile, _("File") ); 
	
	this->SetMenuBar( m_Menubar );
	
	m_ToolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY ); 
	m_ToolBar->Realize(); 
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( m_MenuItemOpenGisLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMeteorologicalSituationVirtual::OnOpenLayer ) );
}

asFrameMeteorologicalSituationVirtual::~asFrameMeteorologicalSituationVirtual()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameMeteorologicalSituationVirtual::OnOpenLayer ) );
	
}

asPanelSidebarVirtual::asPanelSidebarVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	m_SizerMain = new wxBoxSizer( wxVERTICAL );
	
	m_PanelHeader = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_PanelHeader->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_SCROLLBAR ) );
	
	wxBoxSizer* m_SizerHeader;
	m_SizerHeader = new wxBoxSizer( wxHORIZONTAL );
	
	m_Header = new wxStaticText( m_PanelHeader, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_Header->Wrap( -1 );
	m_Header->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 90, false, wxEmptyString ) );
	
	m_SizerHeader->Add( m_Header, 1, wxALL|wxEXPAND, 5 );
	
	m_BpButtonReduce = new wxBitmapButton( m_PanelHeader, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_BpButtonReduce->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_SCROLLBAR ) );
	
	m_SizerHeader->Add( m_BpButtonReduce, 0, 0, 5 );
	
	
	m_PanelHeader->SetSizer( m_SizerHeader );
	m_PanelHeader->Layout();
	m_SizerHeader->Fit( m_PanelHeader );
	m_SizerMain->Add( m_PanelHeader, 1, wxEXPAND, 5 );
	
	m_SizerContent = new wxBoxSizer( wxVERTICAL );
	
	
	m_SizerMain->Add( m_SizerContent, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( m_SizerMain );
	this->Layout();
	m_SizerMain->Fit( this );
	
	// Connect Events
	m_BpButtonReduce->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
}

asPanelSidebarVirtual::~asPanelSidebarVirtual()
{
	// Disconnect Events
	m_BpButtonReduce->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	
}

asFramePreferencesViewerVirtual::asFramePreferencesViewerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );
	
	m_PanelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookBase = new wxNotebook( m_PanelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelWorkspace = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_PanelWorkspace, wxID_ANY, _("Directories for real-time forecasting") ), wxVERTICAL );
	
	m_StaticTextForecastResultsDir = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("Directory to save forecast outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastResultsDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextForecastResultsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerForecastResults = new wxDirPickerCtrl( m_PanelWorkspace, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer55->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer141;
	sbSizer141 = new wxStaticBoxSizer( new wxStaticBox( m_PanelWorkspace, wxID_ANY, _("Forecast display options") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer81;
	fgSizer81 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer81->SetFlexibleDirection( wxBOTH );
	fgSizer81->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextColorbarMaxValue = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("Set the maximum rainfall value for the colorbar"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextColorbarMaxValue->Wrap( -1 );
	fgSizer81->Add( m_StaticTextColorbarMaxValue, 0, wxALL, 5 );
	
	m_TextCtrlColorbarMaxValue = new wxTextCtrl( m_PanelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlColorbarMaxValue->SetMaxLength( 0 ); 
	fgSizer81->Add( m_TextCtrlColorbarMaxValue, 0, wxALL, 5 );
	
	m_StaticTextColorbarMaxUnit = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("mm/d"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextColorbarMaxUnit->Wrap( -1 );
	fgSizer81->Add( m_StaticTextColorbarMaxUnit, 0, wxALL, 5 );
	
	m_StaticTextPastDaysNb = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("Number of past days to display on the timeseries"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPastDaysNb->Wrap( -1 );
	fgSizer81->Add( m_StaticTextPastDaysNb, 0, wxALL, 5 );
	
	m_TextCtrlPastDaysNb = new wxTextCtrl( m_PanelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlPastDaysNb->SetMaxLength( 0 ); 
	fgSizer81->Add( m_TextCtrlPastDaysNb, 0, wxALL, 5 );
	
	
	sbSizer141->Add( fgSizer81, 1, wxEXPAND, 5 );
	
	
	bSizer55->Add( sbSizer141, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer191;
	sbSizer191 = new wxStaticBoxSizer( new wxStaticBox( m_PanelWorkspace, wxID_ANY, _("Alarms panel") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextAlarmsReturnPeriod = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("Return period to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsReturnPeriod->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsReturnPeriod, 0, wxALL, 5 );
	
	wxString m_ChoiceAlarmsReturnPeriodChoices[] = { _("2"), _("5"), _("10"), _("20"), _("50"), _("100") };
	int m_ChoiceAlarmsReturnPeriodNChoices = sizeof( m_ChoiceAlarmsReturnPeriodChoices ) / sizeof( wxString );
	m_ChoiceAlarmsReturnPeriod = new wxChoice( m_PanelWorkspace, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceAlarmsReturnPeriodNChoices, m_ChoiceAlarmsReturnPeriodChoices, 0 );
	m_ChoiceAlarmsReturnPeriod->SetSelection( 0 );
	fgSizer13->Add( m_ChoiceAlarmsReturnPeriod, 0, wxALL, 5 );
	
	m_StaticTextAlarmsReturnPeriodYears = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsReturnPeriodYears->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsReturnPeriodYears, 0, wxALL, 5 );
	
	m_StaticTextAlarmsPercentile = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("Percentile to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsPercentile->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsPercentile, 0, wxALL, 5 );
	
	m_TextCtrlAlarmsPercentile = new wxTextCtrl( m_PanelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlAlarmsPercentile->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlAlarmsPercentile, 0, wxALL, 5 );
	
	m_StaticTextAlarmsPercentileRange = new wxStaticText( m_PanelWorkspace, wxID_ANY, _("(in between 0 - 1)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsPercentileRange->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsPercentileRange, 0, wxALL, 5 );
	
	
	sbSizer191->Add( fgSizer13, 1, wxEXPAND, 5 );
	
	
	bSizer55->Add( sbSizer191, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelWorkspace->SetSizer( bSizer55 );
	m_PanelWorkspace->Layout();
	bSizer55->Fit( m_PanelWorkspace );
	m_NotebookBase->AddPage( m_PanelWorkspace, _("Workspace"), true );
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
	m_NotebookBase->AddPage( m_PanelGeneralCommon, _("General"), false );
	m_PanelAdvanced = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_PanelAdvanced, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_CheckBoxMultiInstancesViewer = new wxCheckBox( m_PanelAdvanced, wxID_ANY, _("Allow multiple instances of the viewer"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer151->Add( m_CheckBoxMultiInstancesViewer, 0, wxALL, 5 );
	
	
	bSizer26->Add( sbSizer151, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( m_PanelAdvanced, wxID_ANY, _("User specific paths") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer9;
	fgSizer9 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer9->SetFlexibleDirection( wxBOTH );
	fgSizer9->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextUserDirLabel = new wxStaticText( m_PanelAdvanced, wxID_ANY, _("User working directory:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextUserDirLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextUserDirLabel, 0, wxALL, 5 );
	
	m_StaticTextUserDir = new wxStaticText( m_PanelAdvanced, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextUserDir->Wrap( -1 );
	fgSizer9->Add( m_StaticTextUserDir, 0, wxALL, 5 );
	
	m_StaticTextLogFileLabel = new wxStaticText( m_PanelAdvanced, wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFileLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFileLabel, 0, wxALL, 5 );
	
	m_StaticTextLogFile = new wxStaticText( m_PanelAdvanced, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFile->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFile, 0, wxALL, 5 );
	
	m_StaticTextPrefFileLabel = new wxStaticText( m_PanelAdvanced, wxID_ANY, _("Preferences file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFileLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFileLabel, 0, wxALL, 5 );
	
	m_StaticTextPrefFile = new wxStaticText( m_PanelAdvanced, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFile->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFile, 0, wxALL, 5 );
	
	
	sbSizer17->Add( fgSizer9, 1, wxEXPAND, 5 );
	
	
	bSizer26->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );
	
	
	m_PanelAdvanced->SetSizer( bSizer26 );
	m_PanelAdvanced->Layout();
	bSizer26->Fit( m_PanelAdvanced );
	m_NotebookBase->AddPage( m_PanelAdvanced, _("Advanced"), false );
	
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
	m_ButtonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesViewerVirtual::~asFramePreferencesViewerVirtual()
{
	// Disconnect Events
	m_ButtonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::SaveAndClose ), NULL, this );
	
}

asWizardWorkspaceVirtual::asWizardWorkspaceVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );
	
	wxWizardPageSimple* m_wizPage1 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage1 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText37 = new wxStaticText( m_wizPage1, wxID_ANY, _("Load / create a workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText37->Wrap( -1 );
	m_staticText37->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer48->Add( m_staticText37, 0, wxALL, 5 );
	
	m_staticText35 = new wxStaticText( m_wizPage1, wxID_ANY, _("Press the button below to load an existing file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText35->Wrap( -1 );
	bSizer48->Add( m_staticText35, 0, wxALL|wxEXPAND, 5 );
	
	m_button4 = new wxButton( m_wizPage1, wxID_ANY, _("Load existing workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer48->Add( m_button4, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticText46 = new wxStaticText( m_wizPage1, wxID_ANY, _("or continue to create a new workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	bSizer48->Add( m_staticText46, 0, wxALL, 5 );
	
	
	m_wizPage1->SetSizer( bSizer48 );
	m_wizPage1->Layout();
	bSizer48->Fit( m_wizPage1 );
	wxWizardPageSimple* m_wizPage2 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage2 );
	
	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText36 = new wxStaticText( m_wizPage2, wxID_ANY, _("Create a new workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText36->Wrap( -1 );
	m_staticText36->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer49->Add( m_staticText36, 0, wxALL, 5 );
	
	m_staticText43 = new wxStaticText( m_wizPage2, wxID_ANY, _("Path to save the new workspace file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	bSizer49->Add( m_staticText43, 0, wxALL, 5 );
	
	m_FilePickerWorkspaceFile = new wxFilePickerCtrl( m_wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizer49->Add( m_FilePickerWorkspaceFile, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	m_wizPage2->SetSizer( bSizer49 );
	m_wizPage2->Layout();
	bSizer49->Fit( m_wizPage2 );
	wxWizardPageSimple* m_wizPage3 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage3 );
	
	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText44 = new wxStaticText( m_wizPage3, wxID_ANY, _("Workspace options"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText44->Wrap( -1 );
	m_staticText44->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer50->Add( m_staticText44, 0, wxALL, 5 );
	
	m_StaticTextForecastResultsDir = new wxStaticText( m_wizPage3, wxID_ANY, _("Provide the path to the forecasts directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastResultsDir->Wrap( -1 );
	bSizer50->Add( m_StaticTextForecastResultsDir, 0, wxALL, 5 );
	
	m_DirPickerForecastResults = new wxDirPickerCtrl( m_wizPage3, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer50->Add( m_DirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticText42 = new wxStaticText( m_wizPage3, wxID_ANY, _("Other workspace parameters can be defined in the preferences frame."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText42->Wrap( -1 );
	bSizer50->Add( m_staticText42, 1, wxALL|wxEXPAND, 5 );
	
	
	m_wizPage3->SetSizer( bSizer50 );
	m_wizPage3->Layout();
	bSizer50->Fit( m_wizPage3 );
	wxWizardPageSimple* m_wizPage4 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage4 );
	
	wxBoxSizer* bSizer51;
	bSizer51 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText45 = new wxStaticText( m_wizPage4, wxID_ANY, _("Base map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	m_staticText45->SetFont( wxFont( 13, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer51->Add( m_staticText45, 0, wxALL, 5 );
	
	m_staticText40 = new wxStaticText( m_wizPage4, wxID_ANY, _("Choose the base map for your project"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText40->Wrap( -1 );
	bSizer51->Add( m_staticText40, 0, wxALL, 5 );
	
	wxString m_ChoiceBaseMapChoices[] = { _("Custom layers"), _("Terrain from Google maps (recommended)"), _("Map from Google maps"), _("Map from Openstreetmap"), _("Map from ArcGIS Mapserver"), _("Satellite imagery from Google maps"), _("Satellite imagery from VirtualEarth") };
	int m_ChoiceBaseMapNChoices = sizeof( m_ChoiceBaseMapChoices ) / sizeof( wxString );
	m_ChoiceBaseMap = new wxChoice( m_wizPage4, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceBaseMapNChoices, m_ChoiceBaseMapChoices, 0 );
	m_ChoiceBaseMap->SetSelection( 0 );
	bSizer51->Add( m_ChoiceBaseMap, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_staticText41 = new wxStaticText( m_wizPage4, wxID_ANY, _("Other GIS layers can be added in the viewer frame directly, and be saved to the workspace."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText41->Wrap( -1 );
	bSizer51->Add( m_staticText41, 1, wxALL|wxEXPAND, 5 );
	
	
	m_wizPage4->SetSizer( bSizer51 );
	m_wizPage4->Layout();
	bSizer51->Fit( m_wizPage4 );
	
	this->Centre( wxBOTH );
	
	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}
	
	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardWorkspaceVirtual::OnWizardFinished ) );
	m_button4->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardWorkspaceVirtual::OnLoadExistingWorkspace ), NULL, this );
}

asWizardWorkspaceVirtual::~asWizardWorkspaceVirtual()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardWorkspaceVirtual::OnWizardFinished ) );
	m_button4->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardWorkspaceVirtual::OnLoadExistingWorkspace ), NULL, this );
	
	m_pages.Clear();
}
