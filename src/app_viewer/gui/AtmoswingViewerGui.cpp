///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Nov  6 2017)
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

#include "AtmoswingViewerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameForecastVirtual::asFrameForecastVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxVERTICAL );
	
	m_splitterGIS = new wxSplitterWindow( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitterGIS->Connect( wxEVT_IDLE, wxIdleEventHandler( asFrameForecastVirtual::m_splitterGISOnIdle ), NULL, this );
	m_splitterGIS->SetMinimumPaneSize( 270 );
	
	m_scrolledWindowOptions = new wxScrolledWindow( m_splitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_scrolledWindowOptions->SetScrollRate( 5, 5 );
	m_sizerScrolledWindow = new wxBoxSizer( wxVERTICAL );
	
	
	m_scrolledWindowOptions->SetSizer( m_sizerScrolledWindow );
	m_scrolledWindowOptions->Layout();
	m_sizerScrolledWindow->Fit( m_scrolledWindowOptions );
	m_panelContent = new wxPanel( m_splitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_sizerContent = new wxBoxSizer( wxVERTICAL );
	
	m_panelTop = new wxPanel( m_panelContent, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxTAB_TRAVERSAL );
	m_panelTop->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	
	m_sizerTop = new wxBoxSizer( wxHORIZONTAL );
	
	m_sizerTopLeft = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer52;
	bSizer52 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextForecastDate = new wxStaticText( m_panelTop, wxID_ANY, _("No forecast opened"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastDate->Wrap( -1 );
	m_staticTextForecastDate->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	m_staticTextForecastDate->SetForegroundColour( wxColour( 255, 255, 255 ) );
	
	bSizer52->Add( m_staticTextForecastDate, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_button51 = new wxButton( m_panelTop, wxID_ANY, _("<<"), wxDefaultPosition, wxSize( 20,20 ), 0|wxNO_BORDER );
	m_button51->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button51->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	
	bSizer52->Add( m_button51, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_button5 = new wxButton( m_panelTop, wxID_ANY, _("<"), wxDefaultPosition, wxSize( 20,20 ), 0|wxNO_BORDER );
	m_button5->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button5->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	
	bSizer52->Add( m_button5, 0, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_VERTICAL, 3 );
	
	m_button6 = new wxButton( m_panelTop, wxID_ANY, _(">"), wxDefaultPosition, wxSize( 20,20 ), 0|wxNO_BORDER );
	m_button6->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button6->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	
	bSizer52->Add( m_button6, 0, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_VERTICAL, 3 );
	
	m_button61 = new wxButton( m_panelTop, wxID_ANY, _(">>"), wxDefaultPosition, wxSize( 20,20 ), 0|wxNO_BORDER );
	m_button61->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button61->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	
	bSizer52->Add( m_button61, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	m_sizerTopLeft->Add( bSizer52, 1, wxEXPAND, 5 );
	
	m_staticTextForecast = new wxStaticText( m_panelTop, wxID_ANY, _("No forecast selected"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecast->Wrap( -1 );
	m_staticTextForecast->SetFont( wxFont( 11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	m_staticTextForecast->SetForegroundColour( wxColour( 255, 255, 255 ) );
	
	m_sizerTopLeft->Add( m_staticTextForecast, 0, wxALL, 5 );
	
	
	m_sizerTop->Add( m_sizerTopLeft, 0, wxALIGN_LEFT|wxEXPAND, 5 );
	
	m_sizerTopRight = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer39;
	bSizer39 = new wxBoxSizer( wxVERTICAL );
	
	m_sizerLeadTimeSwitch = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer39->Add( m_sizerLeadTimeSwitch, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_sizerTopRight->Add( bSizer39, 1, wxALIGN_RIGHT, 5 );
	
	
	m_sizerTop->Add( m_sizerTopRight, 1, wxALIGN_RIGHT|wxEXPAND, 5 );
	
	
	m_panelTop->SetSizer( m_sizerTop );
	m_panelTop->Layout();
	m_sizerTop->Fit( m_panelTop );
	m_sizerContent->Add( m_panelTop, 0, wxEXPAND, 5 );
	
	m_panelGIS = new wxPanel( m_panelContent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_sizerGIS = new wxBoxSizer( wxVERTICAL );
	
	
	m_panelGIS->SetSizer( m_sizerGIS );
	m_panelGIS->Layout();
	m_sizerGIS->Fit( m_panelGIS );
	m_sizerContent->Add( m_panelGIS, 1, wxEXPAND, 5 );
	
	
	m_panelContent->SetSizer( m_sizerContent );
	m_panelContent->Layout();
	m_sizerContent->Fit( m_panelContent );
	m_splitterGIS->SplitVertically( m_scrolledWindowOptions, m_panelContent, 270 );
	bSizer11->Add( m_splitterGIS, 1, wxEXPAND|wxALL, 4 );
	
	
	m_panelMain->SetSizer( bSizer11 );
	m_panelMain->Layout();
	bSizer11->Fit( m_panelMain );
	bSizer3->Add( m_panelMain, 1, wxEXPAND, 2 );
	
	
	this->SetSizer( bSizer3 );
	this->Layout();
	bSizer3->Fit( this );
	m_menuBar = new wxMenuBar( 0 );
	m_menuFile = new wxMenu();
	wxMenuItem* m_menuItemOpenWorkspace;
	m_menuItemOpenWorkspace = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Open a workspace") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemOpenWorkspace );
	
	wxMenuItem* m_menuItemSaveWorkspace;
	m_menuItemSaveWorkspace = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Save the workspace") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemSaveWorkspace );
	
	wxMenuItem* m_menuItemSaveWorkspaceAs;
	m_menuItemSaveWorkspaceAs = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Save the workspace as") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemSaveWorkspaceAs );
	
	wxMenuItem* m_menuItemNewWorkspace;
	m_menuItemNewWorkspace = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Create a new workspace") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemNewWorkspace );
	
	m_menuFile->AppendSeparator();
	
	wxMenuItem* m_menuItemOpenForecast;
	m_menuItemOpenForecast = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Open a forecast file") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemOpenForecast );
	
	m_menuFile->AppendSeparator();
	
	wxMenuItem* m_menuItemOpenGISLayer;
	m_menuItemOpenGISLayer = new wxMenuItem( m_menuFile, wxID_OPEN, wxString( _("Open a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemOpenGISLayer );
	
	wxMenuItem* m_menuItemCloseGISLayer;
	m_menuItemCloseGISLayer = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Close a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemCloseGISLayer );
	
	wxMenuItem* m_menuItemMoveGISLayer;
	m_menuItemMoveGISLayer = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Move the selected layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemMoveGISLayer );
	
	m_menuFile->AppendSeparator();
	
	wxMenuItem* m_menuItemQuit;
	m_menuItemQuit = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Quit") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemQuit );
	
	m_menuBar->Append( m_menuFile, _("File") ); 
	
	m_menuOptions = new wxMenu();
	wxMenuItem* m_menuItemPreferences;
	m_menuItemPreferences = new wxMenuItem( m_menuOptions, wxID_ANY, wxString( _("Preferences") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuOptions->Append( m_menuItemPreferences );
	
	m_menuBar->Append( m_menuOptions, _("Options") ); 
	
	m_menuLog = new wxMenu();
	wxMenuItem* m_menuItemShowLog;
	m_menuItemShowLog = new wxMenuItem( m_menuLog, wxID_ANY, wxString( _("Show Log Window") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuLog->Append( m_menuItemShowLog );
	
	m_menuLogLevel = new wxMenu();
	wxMenuItem* m_menuLogLevelItem = new wxMenuItem( m_menuLog, wxID_ANY, _("Log level"), wxEmptyString, wxITEM_NORMAL, m_menuLogLevel );
	wxMenuItem* m_menuItemLogLevel1;
	m_menuItemLogLevel1 = new wxMenuItem( m_menuLogLevel, wxID_ANY, wxString( _("Only errors") ) , wxEmptyString, wxITEM_CHECK );
	m_menuLogLevel->Append( m_menuItemLogLevel1 );
	
	wxMenuItem* m_menuItemLogLevel2;
	m_menuItemLogLevel2 = new wxMenuItem( m_menuLogLevel, wxID_ANY, wxString( _("Errors and warnings") ) , wxEmptyString, wxITEM_CHECK );
	m_menuLogLevel->Append( m_menuItemLogLevel2 );
	
	wxMenuItem* m_menuItemLogLevel3;
	m_menuItemLogLevel3 = new wxMenuItem( m_menuLogLevel, wxID_ANY, wxString( _("Verbose") ) , wxEmptyString, wxITEM_CHECK );
	m_menuLogLevel->Append( m_menuItemLogLevel3 );
	
	m_menuLog->Append( m_menuLogLevelItem );
	
	m_menuBar->Append( m_menuLog, _("Log") ); 
	
	m_menuHelp = new wxMenu();
	wxMenuItem* m_menuItemAbout;
	m_menuItemAbout = new wxMenuItem( m_menuHelp, wxID_ANY, wxString( _("About") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuHelp->Append( m_menuItemAbout );
	
	m_menuBar->Append( m_menuHelp, _("Help") ); 
	
	this->SetMenuBar( m_menuBar );
	
	m_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY ); 
	m_toolBar->Realize(); 
	
	m_statusBar = this->CreateStatusBar( 2, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button51->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadPreviousDay ), NULL, this );
	m_button5->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadPreviousForecast ), NULL, this );
	m_button6->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadNextForecast ), NULL, this );
	m_button61->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadNextDay ), NULL, this );
	this->Connect( m_menuItemOpenWorkspace->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenWorkspace ) );
	this->Connect( m_menuItemSaveWorkspace->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnSaveWorkspace ) );
	this->Connect( m_menuItemSaveWorkspaceAs->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnSaveWorkspaceAs ) );
	this->Connect( m_menuItemNewWorkspace->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnNewWorkspace ) );
	this->Connect( m_menuItemOpenForecast->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenForecast ) );
	this->Connect( m_menuItemOpenGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenLayer ) );
	this->Connect( m_menuItemCloseGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnCloseLayer ) );
	this->Connect( m_menuItemMoveGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnMoveLayer ) );
	this->Connect( m_menuItemQuit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnQuit ) );
	this->Connect( m_menuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OpenFramePreferences ) );
	this->Connect( m_menuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnShowLog ) );
	this->Connect( m_menuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel1 ) );
	this->Connect( m_menuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel2 ) );
	this->Connect( m_menuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnLogLevel3 ) );
	this->Connect( m_menuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OpenFrameAbout ) );
}

asFrameForecastVirtual::~asFrameForecastVirtual()
{
	// Disconnect Events
	m_button51->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadPreviousDay ), NULL, this );
	m_button5->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadPreviousForecast ), NULL, this );
	m_button6->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadNextForecast ), NULL, this );
	m_button61->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameForecastVirtual::OnLoadNextDay ), NULL, this );
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
	
	m_panelStationName = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextStationName = new wxStaticText( m_panelStationName, wxID_ANY, _("Station name"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextStationName->Wrap( -1 );
	m_staticTextStationName->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	bSizer37->Add( m_staticTextStationName, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_buttonSaveTxt = new wxButton( m_panelStationName, wxID_ANY, _("Export as txt"), wxDefaultPosition, wxSize( -1,25 ), 0 );
	m_buttonSaveTxt->SetFont( wxFont( 8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	bSizer37->Add( m_buttonSaveTxt, 0, wxALL, 5 );
	
	m_buttonPreview = new wxButton( m_panelStationName, wxID_ANY, _("Preview"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	m_buttonPreview->Enable( false );
	m_buttonPreview->Hide();
	
	bSizer37->Add( m_buttonPreview, 0, wxALL, 5 );
	
	m_buttonPrint = new wxButton( m_panelStationName, wxID_ANY, _("Print"), wxDefaultPosition, wxSize( 50,20 ), 0 );
	m_buttonPrint->Enable( false );
	m_buttonPrint->Hide();
	
	bSizer37->Add( m_buttonPrint, 0, wxALL, 5 );
	
	
	bSizer29->Add( bSizer37, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_panelStationName->SetSizer( bSizer29 );
	m_panelStationName->Layout();
	bSizer29->Fit( m_panelStationName );
	bSizer13->Add( m_panelStationName, 0, wxEXPAND, 5 );
	
	m_splitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotTimeSeriesVirtual::m_splitterOnIdle ), NULL, this );
	m_splitter->SetMinimumPaneSize( 150 );
	
	m_panelLeft = new wxPanel( m_splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	wxArrayString m_checkListTocChoices;
	m_checkListToc = new wxCheckListBox( m_panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_checkListTocChoices, 0 );
	bSizer27->Add( m_checkListToc, 1, wxEXPAND, 5 );
	
	wxArrayString m_checkListPastChoices;
	m_checkListPast = new wxCheckListBox( m_panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_checkListPastChoices, 0 );
	bSizer27->Add( m_checkListPast, 1, wxEXPAND|wxTOP, 5 );
	
	
	m_panelLeft->SetSizer( bSizer27 );
	m_panelLeft->Layout();
	bSizer27->Fit( m_panelLeft );
	m_panelRight = new wxPanel( m_splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_sizerPlot = new wxBoxSizer( wxVERTICAL );
	
	
	m_panelRight->SetSizer( m_sizerPlot );
	m_panelRight->Layout();
	m_sizerPlot->Fit( m_panelRight );
	m_splitter->SplitVertically( m_panelLeft, m_panelRight, 150 );
	bSizer13->Add( m_splitter, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer13 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_buttonSaveTxt->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	m_buttonPreview->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	m_buttonPrint->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	m_checkListToc->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	m_checkListPast->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
}

asFramePlotTimeSeriesVirtual::~asFramePlotTimeSeriesVirtual()
{
	// Disconnect Events
	m_buttonSaveTxt->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	m_buttonPreview->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	m_buttonPrint->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	m_checkListToc->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	m_checkListPast->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	
}

asFramePlotDistributionsVirutal::asFramePlotDistributionsVirutal( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxVERTICAL );
	
	m_panelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextForecast = new wxStaticText( m_panelOptions, wxID_ANY, _("Select forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecast->Wrap( -1 );
	fgSizer1->Add( m_staticTextForecast, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextStation = new wxStaticText( m_panelOptions, wxID_ANY, _("Select station"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextStation->Wrap( -1 );
	fgSizer1->Add( m_staticTextStation, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextDate = new wxStaticText( m_panelOptions, wxID_ANY, _("Select date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDate->Wrap( -1 );
	fgSizer1->Add( m_staticTextDate, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_choiceForecastChoices;
	m_choiceForecast = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceForecastChoices, 0 );
	m_choiceForecast->SetSelection( 0 );
	fgSizer1->Add( m_choiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_choiceStationChoices;
	m_choiceStation = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceStationChoices, 0 );
	m_choiceStation->SetSelection( 0 );
	fgSizer1->Add( m_choiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_choiceDateChoices;
	m_choiceDate = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDateChoices, 0 );
	m_choiceDate->SetSelection( 0 );
	fgSizer1->Add( m_choiceDate, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer29->Add( fgSizer1, 1, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_panelOptions->SetSizer( bSizer29 );
	m_panelOptions->Layout();
	bSizer29->Fit( m_panelOptions );
	bSizer13->Add( m_panelOptions, 0, wxEXPAND, 5 );
	
	m_notebook = new wxNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelPredictands = new wxPanel( m_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxVERTICAL );
	
	m_splitterPredictands = new wxSplitterWindow( m_panelPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitterPredictands->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::m_splitterPredictandsOnIdle ), NULL, this );
	m_splitterPredictands->SetMinimumPaneSize( 170 );
	
	m_panelPredictandsLeft = new wxPanel( m_splitterPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	wxArrayString m_checkListTocPredictandsChoices;
	m_checkListTocPredictands = new wxCheckListBox( m_panelPredictandsLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_checkListTocPredictandsChoices, 0 );
	bSizer27->Add( m_checkListTocPredictands, 1, wxEXPAND, 5 );
	
	
	m_panelPredictandsLeft->SetSizer( bSizer27 );
	m_panelPredictandsLeft->Layout();
	bSizer27->Fit( m_panelPredictandsLeft );
	m_panelPredictandsRight = new wxPanel( m_splitterPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_sizerPlotPredictands = new wxBoxSizer( wxVERTICAL );
	
	
	m_panelPredictandsRight->SetSizer( m_sizerPlotPredictands );
	m_panelPredictandsRight->Layout();
	m_sizerPlotPredictands->Fit( m_panelPredictandsRight );
	m_splitterPredictands->SplitVertically( m_panelPredictandsLeft, m_panelPredictandsRight, 170 );
	bSizer22->Add( m_splitterPredictands, 1, wxEXPAND|wxTOP, 5 );
	
	
	m_panelPredictands->SetSizer( bSizer22 );
	m_panelPredictands->Layout();
	bSizer22->Fit( m_panelPredictands );
	m_notebook->AddPage( m_panelPredictands, _("Predictands distribution"), true );
	m_panelCriteria = new wxPanel( m_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_sizerPlotCriteria = new wxBoxSizer( wxVERTICAL );
	
	
	m_panelCriteria->SetSizer( m_sizerPlotCriteria );
	m_panelCriteria->Layout();
	m_sizerPlotCriteria->Fit( m_panelCriteria );
	m_notebook->AddPage( m_panelCriteria, _("Criteria distribution"), false );
	
	bSizer13->Add( m_notebook, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer13 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	m_choiceStation->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	m_choiceDate->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	m_checkListTocPredictands->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
}

asFramePlotDistributionsVirutal::~asFramePlotDistributionsVirutal()
{
	// Disconnect Events
	m_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	m_choiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	m_choiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	m_checkListTocPredictands->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
	
}

asFrameGridAnalogsValuesVirtual::asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 630,-1 ), wxSize( 630,-1 ) );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	m_panelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer291;
	bSizer291 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextForecast = new wxStaticText( m_panelOptions, wxID_ANY, _("Select forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecast->Wrap( -1 );
	fgSizer1->Add( m_staticTextForecast, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_choiceForecastChoices;
	m_choiceForecast = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceForecastChoices, 0 );
	m_choiceForecast->SetSelection( 0 );
	fgSizer1->Add( m_choiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextStation = new wxStaticText( m_panelOptions, wxID_ANY, _("Select station"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextStation->Wrap( -1 );
	fgSizer1->Add( m_staticTextStation, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_choiceStationChoices;
	m_choiceStation = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceStationChoices, 0 );
	m_choiceStation->SetSelection( 0 );
	fgSizer1->Add( m_choiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextDate = new wxStaticText( m_panelOptions, wxID_ANY, _("Select date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDate->Wrap( -1 );
	fgSizer1->Add( m_staticTextDate, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );
	
	wxArrayString m_choiceDateChoices;
	m_choiceDate = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDateChoices, 0 );
	m_choiceDate->SetSelection( 0 );
	fgSizer1->Add( m_choiceDate, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer291->Add( fgSizer1, 1, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	
	m_panelOptions->SetSizer( bSizer291 );
	m_panelOptions->Layout();
	bSizer291->Fit( m_panelOptions );
	bSizer29->Add( m_panelOptions, 0, wxEXPAND, 5 );
	
	m_grid = new wxGrid( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	
	// Grid
	m_grid->CreateGrid( 5, 4 );
	m_grid->EnableEditing( false );
	m_grid->EnableGridLines( true );
	m_grid->EnableDragGridSize( false );
	m_grid->SetMargins( 0, 0 );
	
	// Columns
	m_grid->SetColSize( 0, 95 );
	m_grid->SetColSize( 1, 122 );
	m_grid->SetColSize( 2, 190 );
	m_grid->SetColSize( 3, 130 );
	m_grid->EnableDragColMove( false );
	m_grid->EnableDragColSize( true );
	m_grid->SetColLabelSize( 40 );
	m_grid->SetColLabelValue( 0, _("Analog") );
	m_grid->SetColLabelValue( 1, _("Date") );
	m_grid->SetColLabelValue( 2, _("Precipitation (mm)") );
	m_grid->SetColLabelValue( 3, _("Criteria") );
	m_grid->SetColLabelAlignment( wxALIGN_CENTRE, wxALIGN_CENTRE );
	
	// Rows
	m_grid->EnableDragRowSize( true );
	m_grid->SetRowLabelSize( 40 );
	m_grid->SetRowLabelAlignment( wxALIGN_CENTRE, wxALIGN_CENTRE );
	
	// Label Appearance
	
	// Cell Defaults
	m_grid->SetDefaultCellAlignment( wxALIGN_RIGHT, wxALIGN_TOP );
	bSizer29->Add( m_grid, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer29 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceForecastChange ), NULL, this );
	m_choiceStation->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceStationChange ), NULL, this );
	m_choiceDate->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceDateChange ), NULL, this );
	m_grid->Connect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_grid->Connect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_grid->Connect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_grid->Connect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
}

asFrameGridAnalogsValuesVirtual::~asFrameGridAnalogsValuesVirtual()
{
	// Disconnect Events
	m_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceForecastChange ), NULL, this );
	m_choiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceStationChange ), NULL, this );
	m_choiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceDateChange ), NULL, this );
	m_grid->Disconnect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_grid->Disconnect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_grid->Disconnect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	m_grid->Disconnect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	
}

asFramePredictorsVirtual::asFramePredictorsVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 800,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );
	
	m_panel15 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	m_splitterToc = new wxSplitterWindow( m_panel15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitterToc->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::m_splitterTocOnIdle ), NULL, this );
	m_splitterToc->SetMinimumPaneSize( 165 );
	
	m_scrolledWindowOptions = new wxScrolledWindow( m_splitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_scrolledWindowOptions->SetScrollRate( 5, 5 );
	m_sizerScrolledWindow = new wxBoxSizer( wxVERTICAL );
	
	m_staticTextChoiceForecast = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextChoiceForecast->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextChoiceForecast, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	wxArrayString m_choiceForecastChoices;
	m_choiceForecast = new wxChoice( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceForecastChoices, 0 );
	m_choiceForecast->SetSelection( 0 );
	m_choiceForecast->SetMaxSize( wxSize( 150,-1 ) );
	
	m_sizerScrolledWindow->Add( m_choiceForecast, 0, wxEXPAND|wxALL, 5 );
	
	m_staticTextCheckListPredictors = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Possible predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextCheckListPredictors->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextCheckListPredictors, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	wxArrayString m_checkListPredictorsChoices;
	m_checkListPredictors = new wxCheckListBox( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_checkListPredictorsChoices, 0 );
	m_sizerScrolledWindow->Add( m_checkListPredictors, 1, wxEXPAND, 5 );
	
	m_staticTextTocLeft = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Layers of the left panel"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextTocLeft->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextTocLeft, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextTocRight = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Layers of the right panel"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextTocRight->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextTocRight, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	m_scrolledWindowOptions->SetSizer( m_sizerScrolledWindow );
	m_scrolledWindowOptions->Layout();
	m_sizerScrolledWindow->Fit( m_scrolledWindowOptions );
	m_panelGIS = new wxPanel( m_splitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_sizerGIS = new wxBoxSizer( wxHORIZONTAL );
	
	m_panelLeft = new wxPanel( m_panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer371;
	bSizer371 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextTargetDates = new wxStaticText( m_panelLeft, wxID_ANY, _("Forecast date"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextTargetDates->Wrap( -1 );
	bSizer34->Add( m_staticTextTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxArrayString m_choiceTargetDatesChoices;
	m_choiceTargetDates = new wxChoice( m_panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceTargetDatesChoices, 0 );
	m_choiceTargetDates->SetSelection( 0 );
	bSizer34->Add( m_choiceTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer371->Add( bSizer34, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_panelGISLeft = new wxPanel( m_panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_sizerGISLeft = new wxBoxSizer( wxVERTICAL );
	
	
	m_panelGISLeft->SetSizer( m_sizerGISLeft );
	m_panelGISLeft->Layout();
	m_sizerGISLeft->Fit( m_panelGISLeft );
	bSizer371->Add( m_panelGISLeft, 1, wxEXPAND, 5 );
	
	
	m_panelLeft->SetSizer( bSizer371 );
	m_panelLeft->Layout();
	bSizer371->Fit( m_panelLeft );
	m_sizerGIS->Add( m_panelLeft, 1, wxEXPAND, 5 );
	
	m_panelSwitch = new wxPanel( m_panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer40;
	bSizer40 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* m_sizerSwitch;
	m_sizerSwitch = new wxBoxSizer( wxVERTICAL );
	
	m_bpButtonSwitchRight = new wxBitmapButton( m_panelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_sizerSwitch->Add( m_bpButtonSwitchRight, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_bpButtonSwitchLeft = new wxBitmapButton( m_panelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_sizerSwitch->Add( m_bpButtonSwitchLeft, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer40->Add( m_sizerSwitch, 1, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	m_panelSwitch->SetSizer( bSizer40 );
	m_panelSwitch->Layout();
	bSizer40->Fit( m_panelSwitch );
	m_sizerGIS->Add( m_panelSwitch, 0, wxEXPAND, 5 );
	
	m_panelRight = new wxPanel( m_panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextAnalogDates = new wxStaticText( m_panelRight, wxID_ANY, _("Analogs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAnalogDates->Wrap( -1 );
	bSizer35->Add( m_staticTextAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxArrayString m_choiceAnalogDatesChoices;
	m_choiceAnalogDates = new wxChoice( m_panelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceAnalogDatesChoices, 0 );
	m_choiceAnalogDates->SetSelection( 0 );
	bSizer35->Add( m_choiceAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer38->Add( bSizer35, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_panelGISRight = new wxPanel( m_panelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_sizerGISRight = new wxBoxSizer( wxVERTICAL );
	
	
	m_panelGISRight->SetSizer( m_sizerGISRight );
	m_panelGISRight->Layout();
	m_sizerGISRight->Fit( m_panelGISRight );
	bSizer38->Add( m_panelGISRight, 1, wxEXPAND, 5 );
	
	
	m_panelRight->SetSizer( bSizer38 );
	m_panelRight->Layout();
	bSizer38->Fit( m_panelRight );
	m_sizerGIS->Add( m_panelRight, 1, wxEXPAND, 5 );
	
	
	m_panelGIS->SetSizer( m_sizerGIS );
	m_panelGIS->Layout();
	m_sizerGIS->Fit( m_panelGIS );
	m_splitterToc->SplitVertically( m_scrolledWindowOptions, m_panelGIS, 170 );
	bSizer26->Add( m_splitterToc, 1, wxEXPAND, 5 );
	
	
	m_panel15->SetSizer( bSizer26 );
	m_panel15->Layout();
	bSizer26->Fit( m_panel15 );
	bSizer25->Add( m_panel15, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer25 );
	this->Layout();
	m_menubar = new wxMenuBar( 0 );
	m_menuFile = new wxMenu();
	wxMenuItem* m_menuItemOpenGisLayer;
	m_menuItemOpenGisLayer = new wxMenuItem( m_menuFile, wxID_ANY, wxString( _("Open GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuFile->Append( m_menuItemOpenGisLayer );
	
	m_menubar->Append( m_menuFile, _("File") ); 
	
	m_menuTools = new wxMenu();
	m_menubar->Append( m_menuTools, _("Tools") ); 
	
	this->SetMenuBar( m_menubar );
	
	m_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY ); 
	m_toolBar->Realize(); 
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	m_checkListPredictors->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	m_choiceTargetDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	m_bpButtonSwitchRight->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	m_bpButtonSwitchLeft->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	m_choiceAnalogDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );
	this->Connect( m_menuItemOpenGisLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnOpenLayer ) );
}

asFramePredictorsVirtual::~asFramePredictorsVirtual()
{
	// Disconnect Events
	m_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	m_checkListPredictors->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	m_choiceTargetDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	m_bpButtonSwitchRight->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	m_bpButtonSwitchLeft->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	m_choiceAnalogDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnOpenLayer ) );
	
}

asPanelSidebarVirtual::asPanelSidebarVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	m_sizerMain = new wxBoxSizer( wxVERTICAL );
	
	m_panelHeader = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_panelHeader->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_SCROLLBAR ) );
	
	wxBoxSizer* m_sizerHeader;
	m_sizerHeader = new wxBoxSizer( wxHORIZONTAL );
	
	m_header = new wxStaticText( m_panelHeader, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_header->Wrap( -1 );
	m_header->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	m_sizerHeader->Add( m_header, 1, wxALL|wxEXPAND, 5 );
	
	m_bpButtonReduce = new wxBitmapButton( m_panelHeader, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), wxBU_AUTODRAW|wxNO_BORDER );
	m_bpButtonReduce->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_SCROLLBAR ) );
	
	m_sizerHeader->Add( m_bpButtonReduce, 0, 0, 5 );
	
	
	m_panelHeader->SetSizer( m_sizerHeader );
	m_panelHeader->Layout();
	m_sizerHeader->Fit( m_panelHeader );
	m_sizerMain->Add( m_panelHeader, 1, wxEXPAND, 5 );
	
	m_sizerContent = new wxBoxSizer( wxVERTICAL );
	
	
	m_sizerMain->Add( m_sizerContent, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( m_sizerMain );
	this->Layout();
	m_sizerMain->Fit( this );
	
	// Connect Events
	m_bpButtonReduce->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
}

asPanelSidebarVirtual::~asPanelSidebarVirtual()
{
	// Disconnect Events
	m_bpButtonReduce->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	
}

asFramePreferencesViewerVirtual::asFramePreferencesViewerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );
	
	m_panelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );
	
	m_notebookBase = new wxNotebook( m_panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelWorkspace = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_panelWorkspace, wxID_ANY, _("Directories for real-time forecasting") ), wxVERTICAL );
	
	m_staticTextForecastResultsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory to save forecast outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextForecastResultsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerForecastResults = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer55->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer141;
	sbSizer141 = new wxStaticBoxSizer( new wxStaticBox( m_panelWorkspace, wxID_ANY, _("Forecast display options") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer81;
	fgSizer81 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer81->SetFlexibleDirection( wxBOTH );
	fgSizer81->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextColorbarMaxValue = new wxStaticText( sbSizer141->GetStaticBox(), wxID_ANY, _("Set the maximum rainfall value for the colorbar"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextColorbarMaxValue->Wrap( -1 );
	fgSizer81->Add( m_staticTextColorbarMaxValue, 0, wxALL, 5 );
	
	m_textCtrlColorbarMaxValue = new wxTextCtrl( sbSizer141->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer81->Add( m_textCtrlColorbarMaxValue, 0, wxALL, 5 );
	
	m_staticTextColorbarMaxUnit = new wxStaticText( sbSizer141->GetStaticBox(), wxID_ANY, _("mm/d"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextColorbarMaxUnit->Wrap( -1 );
	fgSizer81->Add( m_staticTextColorbarMaxUnit, 0, wxALL, 5 );
	
	m_staticTextPastDaysNb = new wxStaticText( sbSizer141->GetStaticBox(), wxID_ANY, _("Number of past days to display on the timeseries"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPastDaysNb->Wrap( -1 );
	fgSizer81->Add( m_staticTextPastDaysNb, 0, wxALL, 5 );
	
	m_textCtrlPastDaysNb = new wxTextCtrl( sbSizer141->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer81->Add( m_textCtrlPastDaysNb, 0, wxALL, 5 );
	
	
	sbSizer141->Add( fgSizer81, 1, wxEXPAND, 5 );
	
	
	bSizer55->Add( sbSizer141, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer191;
	sbSizer191 = new wxStaticBoxSizer( new wxStaticBox( m_panelWorkspace, wxID_ANY, _("Alarms panel") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextAlarmsReturnPeriod = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("Return period to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsReturnPeriod->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsReturnPeriod, 0, wxALL, 5 );
	
	wxString m_choiceAlarmsReturnPeriodChoices[] = { _("2"), _("5"), _("10"), _("20"), _("50"), _("100") };
	int m_choiceAlarmsReturnPeriodNChoices = sizeof( m_choiceAlarmsReturnPeriodChoices ) / sizeof( wxString );
	m_choiceAlarmsReturnPeriod = new wxChoice( sbSizer191->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceAlarmsReturnPeriodNChoices, m_choiceAlarmsReturnPeriodChoices, 0 );
	m_choiceAlarmsReturnPeriod->SetSelection( 0 );
	fgSizer13->Add( m_choiceAlarmsReturnPeriod, 0, wxALL, 5 );
	
	m_staticTextAlarmsReturnPeriodYears = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsReturnPeriodYears->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsReturnPeriodYears, 0, wxALL, 5 );
	
	m_staticTextAlarmsQuantile = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("Quantile to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsQuantile->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsQuantile, 0, wxALL, 5 );
	
	m_textCtrlAlarmsQuantile = new wxTextCtrl( sbSizer191->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer13->Add( m_textCtrlAlarmsQuantile, 0, wxALL, 5 );
	
	m_staticTextAlarmsQuantileRange = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("(in between 0 - 1)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsQuantileRange->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsQuantileRange, 0, wxALL, 5 );
	
	
	sbSizer191->Add( fgSizer13, 1, wxEXPAND, 5 );
	
	
	bSizer55->Add( sbSizer191, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelWorkspace->SetSizer( bSizer55 );
	m_panelWorkspace->Layout();
	bSizer55->Fit( m_panelWorkspace );
	m_notebookBase->AddPage( m_panelWorkspace, _("Workspace"), true );
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
	bSizer34->Add( m_textCtrlProxyAddress, 1, wxALL, 5 );
	
	m_staticTextProxyPort = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Port"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyPort->Wrap( -1 );
	bSizer34->Add( m_staticTextProxyPort, 0, wxALL, 5 );
	
	m_textCtrlProxyPort = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer34->Add( m_textCtrlProxyPort, 0, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer34, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextProxyUser = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Username"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyUser->Wrap( -1 );
	bSizer35->Add( m_staticTextProxyUser, 0, wxALL, 5 );
	
	m_textCtrlProxyUser = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	bSizer35->Add( m_textCtrlProxyUser, 1, wxALL, 5 );
	
	m_staticTextProxyPasswd = new wxStaticText( sbSizer14->GetStaticBox(), wxID_ANY, _("Password"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextProxyPasswd->Wrap( -1 );
	bSizer35->Add( m_staticTextProxyPasswd, 0, wxALL, 5 );
	
	m_textCtrlProxyPasswd = new wxTextCtrl( sbSizer14->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD );
	bSizer35->Add( m_textCtrlProxyPasswd, 1, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer35, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer14, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelGeneralCommon->SetSizer( bSizer16 );
	m_panelGeneralCommon->Layout();
	bSizer16->Fit( m_panelGeneralCommon );
	m_notebookBase->AddPage( m_panelGeneralCommon, _("General"), false );
	m_panelAdvanced = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_panelAdvanced, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_checkBoxMultiInstancesViewer = new wxCheckBox( sbSizer151->GetStaticBox(), wxID_ANY, _("Allow multiple instances of the viewer"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer151->Add( m_checkBoxMultiInstancesViewer, 0, wxALL, 5 );
	
	
	bSizer26->Add( sbSizer151, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( m_panelAdvanced, wxID_ANY, _("User specific paths") ), wxVERTICAL );
	
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
	
	
	bSizer26->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );
	
	
	m_panelAdvanced->SetSizer( bSizer26 );
	m_panelAdvanced->Layout();
	bSizer26->Fit( m_panelAdvanced );
	m_notebookBase->AddPage( m_panelAdvanced, _("Advanced"), false );
	
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
	m_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesViewerVirtual::~asFramePreferencesViewerVirtual()
{
	// Disconnect Events
	m_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::SaveAndClose ), NULL, this );
	
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
	m_staticText37->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
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
	m_staticText36->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	bSizer49->Add( m_staticText36, 0, wxALL, 5 );
	
	m_staticText43 = new wxStaticText( m_wizPage2, wxID_ANY, _("Path to save the new workspace file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	bSizer49->Add( m_staticText43, 0, wxALL, 5 );
	
	m_filePickerWorkspaceFile = new wxFilePickerCtrl( m_wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.asvw"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizer49->Add( m_filePickerWorkspaceFile, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	m_wizPage2->SetSizer( bSizer49 );
	m_wizPage2->Layout();
	bSizer49->Fit( m_wizPage2 );
	wxWizardPageSimple* m_wizPage3 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage3 );
	
	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText44 = new wxStaticText( m_wizPage3, wxID_ANY, _("Workspace options"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText44->Wrap( -1 );
	m_staticText44->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	bSizer50->Add( m_staticText44, 0, wxALL, 5 );
	
	m_staticTextForecastResultsDir = new wxStaticText( m_wizPage3, wxID_ANY, _("Provide the path to the forecasts directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsDir->Wrap( -1 );
	bSizer50->Add( m_staticTextForecastResultsDir, 0, wxALL, 5 );
	
	m_dirPickerForecastResults = new wxDirPickerCtrl( m_wizPage3, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer50->Add( m_dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
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
	m_staticText45->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	bSizer51->Add( m_staticText45, 0, wxALL, 5 );
	
	m_staticText40 = new wxStaticText( m_wizPage4, wxID_ANY, _("Choose the base map for your project"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText40->Wrap( -1 );
	bSizer51->Add( m_staticText40, 0, wxALL, 5 );
	
	wxString m_choiceBaseMapChoices[] = { _("Custom layers"), _("Terrain from Google maps (recommended)"), _("Map from Google maps"), _("Map from Openstreetmap"), _("Map from ArcGIS Mapserver"), _("Satellite imagery from Google maps"), _("Satellite imagery from VirtualEarth") };
	int m_choiceBaseMapNChoices = sizeof( m_choiceBaseMapChoices ) / sizeof( wxString );
	m_choiceBaseMap = new wxChoice( m_wizPage4, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceBaseMapNChoices, m_choiceBaseMapChoices, 0 );
	m_choiceBaseMap->SetSelection( 0 );
	bSizer51->Add( m_choiceBaseMap, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
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
