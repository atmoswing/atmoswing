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

#include "AtmoSwingViewerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameViewerVirtual::asFrameViewerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,600 ), wxDefaultSize );

	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );

	m_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxVERTICAL );

	m_splitterGIS = new wxSplitterWindow( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_NOBORDER );
	m_splitterGIS->Connect( wxEVT_IDLE, wxIdleEventHandler( asFrameViewerVirtual::m_splitterGISOnIdle ), NULL, this );
	m_splitterGIS->SetMinimumPaneSize( 270 );

	m_scrolledWindowOptions = new wxScrolledWindow( m_splitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_scrolledWindowOptions->SetScrollRate( 5, 5 );
	m_scrolledWindowOptions->SetBackgroundColour( wxColour( 255, 255, 255 ) );

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

	m_button51 = new wxButton( m_panelTop, wxID_ANY, _("<<"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	m_button51->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button51->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( m_button51, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_button5 = new wxButton( m_panelTop, wxID_ANY, _("<"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	m_button5->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button5->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( m_button5, 0, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_VERTICAL, 3 );

	m_button6 = new wxButton( m_panelTop, wxID_ANY, _(">"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	m_button6->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button6->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( m_button6, 0, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_VERTICAL, 3 );

	m_button61 = new wxButton( m_panelTop, wxID_ANY, _(">>"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	m_button61->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_button61->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( m_button61, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	m_sizerTopLeft->Add( bSizer52, 1, wxEXPAND, 5 );

	m_staticTextForecast = new wxStaticText( m_panelTop, wxID_ANY, _("No forecast selected"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecast->Wrap( -1 );
	m_staticTextForecast->SetFont( wxFont( 11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	m_staticTextForecast->SetForegroundColour( wxColour( 255, 255, 255 ) );

	m_sizerTopLeft->Add( m_staticTextForecast, 0, wxALL, 5 );


	m_sizerTop->Add( m_sizerTopLeft, 0, wxEXPAND, 5 );

	m_sizerTopRight = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer39;
	bSizer39 = new wxBoxSizer( wxVERTICAL );

	m_sizerLeadTimeSwitch = new wxBoxSizer( wxHORIZONTAL );


	bSizer39->Add( m_sizerLeadTimeSwitch, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	m_sizerTopRight->Add( bSizer39, 1, wxALIGN_RIGHT, 5 );


	m_sizerTop->Add( m_sizerTopRight, 1, wxEXPAND, 5 );


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
	bSizer11->Add( m_splitterGIS, 1, wxEXPAND|wxTOP, 4 );


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

	m_statusBar = this->CreateStatusBar( 2, wxSTB_SIZEGRIP, wxID_ANY );

	this->Centre( wxBOTH );

	// Connect Events
	m_button51->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousDay ), NULL, this );
	m_button5->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousForecast ), NULL, this );
	m_button6->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextForecast ), NULL, this );
	m_button61->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextDay ), NULL, this );
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnOpenWorkspace ), this, m_menuItemOpenWorkspace->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnSaveWorkspace ), this, m_menuItemSaveWorkspace->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnSaveWorkspaceAs ), this, m_menuItemSaveWorkspaceAs->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnNewWorkspace ), this, m_menuItemNewWorkspace->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnOpenForecast ), this, m_menuItemOpenForecast->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnOpenLayer ), this, m_menuItemOpenGISLayer->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnCloseLayer ), this, m_menuItemCloseGISLayer->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnMoveLayer ), this, m_menuItemMoveGISLayer->GetId());
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnQuit ), this, m_menuItemQuit->GetId());
	m_menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OpenFramePreferences ), this, m_menuItemPreferences->GetId());
	m_menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OpenFramePredictandDB ), this, m_menuItemBuildPredictandDB->GetId());
	m_menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnShowLog ), this, m_menuItemShowLog->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnLogLevel1 ), this, m_menuItemLogLevel1->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnLogLevel2 ), this, m_menuItemLogLevel2->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnLogLevel3 ), this, m_menuItemLogLevel3->GetId());
	m_menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OpenFrameAbout ), this, m_menuItemAbout->GetId());
}

asFrameViewerVirtual::~asFrameViewerVirtual()
{
	// Disconnect Events
	m_button51->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousDay ), NULL, this );
	m_button5->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousForecast ), NULL, this );
	m_button6->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextForecast ), NULL, this );
	m_button61->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextDay ), NULL, this );

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

	bSizer48->Add( m_staticText37, 0, wxALL|wxEXPAND, 5 );

	m_staticText35 = new wxStaticText( m_wizPage1, wxID_ANY, _("Load an existing file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText35->Wrap( -1 );
	bSizer48->Add( m_staticText35, 0, wxALL|wxEXPAND, 5 );

	m_button4 = new wxButton( m_wizPage1, wxID_ANY, _("Load workspace"), wxDefaultPosition, wxDefaultSize, 0 );
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

	bSizer49->Add( m_staticText36, 0, wxALL|wxEXPAND, 5 );

	m_staticText43 = new wxStaticText( m_wizPage2, wxID_ANY, _("Path to save the new file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	bSizer49->Add( m_staticText43, 0, wxALL, 5 );

	m_filePickerWorkspaceFile = new wxFilePickerCtrl( m_wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), _("*.asvw"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
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

	bSizer50->Add( m_staticText44, 0, wxALL|wxEXPAND, 5 );

	m_staticTextForecastResultsDir = new wxStaticText( m_wizPage3, wxID_ANY, _("Path to the forecasts directory"), wxDefaultPosition, wxDefaultSize, 0 );
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

	bSizer51->Add( m_staticText45, 0, wxALL|wxEXPAND, 5 );

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

asPanelSidebarVirtual::asPanelSidebarVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	this->SetBackgroundColour( wxColour( 255, 255, 255 ) );

	m_sizerMain = new wxBoxSizer( wxVERTICAL );

	m_panel28 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxSize( -1,2 ), wxTAB_TRAVERSAL );
	m_panel28->SetBackgroundColour( wxColour( 102, 102, 102 ) );

	m_sizerMain->Add( m_panel28, 0, wxEXPAND, 5 );

	m_panelHeader = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_panelHeader->SetBackgroundColour( wxColour( 150, 150, 150 ) );

	wxBoxSizer* m_sizerHeader;
	m_sizerHeader = new wxBoxSizer( wxHORIZONTAL );

	m_header = new wxStaticText( m_panelHeader, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_header->Wrap( -1 );
	m_header->SetFont( wxFont( 10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	m_header->SetForegroundColour( wxColour( 255, 255, 255 ) );

	m_sizerHeader->Add( m_header, 1, wxALL|wxEXPAND, 5 );

	m_bitmapCaret = new wxStaticBitmap( m_panelHeader, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), 0 );
	m_bitmapCaret->SetBackgroundColour( wxColour( 150, 150, 150 ) );

	m_sizerHeader->Add( m_bitmapCaret, 0, 0, 2 );


	m_panelHeader->SetSizer( m_sizerHeader );
	m_panelHeader->Layout();
	m_sizerHeader->Fit( m_panelHeader );
	m_sizerMain->Add( m_panelHeader, 0, wxEXPAND, 5 );

	m_sizerContent = new wxBoxSizer( wxVERTICAL );


	m_sizerMain->Add( m_sizerContent, 1, wxEXPAND, 5 );


	this->SetSizer( m_sizerMain );
	this->Layout();
	m_sizerMain->Fit( this );

	// Connect Events
	m_panelHeader->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	m_header->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	m_bitmapCaret->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
}

asPanelSidebarVirtual::~asPanelSidebarVirtual()
{
	// Disconnect Events
	m_panelHeader->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	m_header->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	m_bitmapCaret->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );

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

	bSizer37->Add( m_staticTextStationName, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_buttonSaveTxt = new wxButton( m_panelStationName, wxID_ANY, _("Export as txt"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_buttonSaveTxt->SetFont( wxFont( 8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer37->Add( m_buttonSaveTxt, 0, wxALL, 5 );

	m_buttonPreview = new wxButton( m_panelStationName, wxID_ANY, _("Preview"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_buttonPreview->Enable( false );
	m_buttonPreview->Hide();

	bSizer37->Add( m_buttonPreview, 0, wxALL, 5 );

	m_buttonPrint = new wxButton( m_panelStationName, wxID_ANY, _("Print"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_buttonPrint->Enable( false );
	m_buttonPrint->Hide();

	bSizer37->Add( m_buttonPrint, 0, wxALL, 5 );

	m_buttonReset = new wxButton( m_panelStationName, wxID_ANY, _("Reset zoom"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer37->Add( m_buttonReset, 0, wxALL, 5 );


	bSizer29->Add( bSizer37, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	m_panelStationName->SetSizer( bSizer29 );
	m_panelStationName->Layout();
	bSizer29->Fit( m_panelStationName );
	bSizer13->Add( m_panelStationName, 0, wxEXPAND, 5 );

	m_splitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
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
	m_panelRight = new wxPanel( m_splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_SIMPLE );
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
	m_buttonReset->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::ResetExtent ), NULL, this );
	m_checkListToc->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	m_checkListPast->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
}

asFramePlotTimeSeriesVirtual::~asFramePlotTimeSeriesVirtual()
{
	// Disconnect Events
	m_buttonSaveTxt->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	m_buttonPreview->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	m_buttonPrint->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	m_buttonReset->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::ResetExtent ), NULL, this );
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
	bSizer22 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer52;
	bSizer52 = new wxBoxSizer( wxVERTICAL );


	bSizer22->Add( bSizer52, 1, wxEXPAND, 5 );

	m_splitter4 = new wxSplitterWindow( m_panelPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::m_splitter4OnIdle ), NULL, this );

	m_panelPredictandsLeft = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxVERTICAL );

	wxArrayString m_checkListTocPredictandsChoices;
	m_checkListTocPredictands = new wxCheckListBox( m_panelPredictandsLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_checkListTocPredictandsChoices, 0|wxBORDER_NONE );
	bSizer55->Add( m_checkListTocPredictands, 1, wxEXPAND|wxTOP|wxBOTTOM|wxLEFT, 5 );

	m_buttonResetZoom = new wxButton( m_panelPredictandsLeft, wxID_ANY, _("Reset zoom"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer55->Add( m_buttonResetZoom, 0, wxALL|wxEXPAND, 5 );


	m_panelPredictandsLeft->SetSizer( bSizer55 );
	m_panelPredictandsLeft->Layout();
	bSizer55->Fit( m_panelPredictandsLeft );
	m_panelPredictandsRight = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_sizerPlotPredictands = new wxBoxSizer( wxVERTICAL );


	m_panelPredictandsRight->SetSizer( m_sizerPlotPredictands );
	m_panelPredictandsRight->Layout();
	m_sizerPlotPredictands->Fit( m_panelPredictandsRight );
	m_splitter4->SplitVertically( m_panelPredictandsLeft, m_panelPredictandsRight, 178 );
	bSizer22->Add( m_splitter4, 1, wxEXPAND, 5 );


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
	m_buttonResetZoom->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotDistributionsVirutal::ResetExtent ), NULL, this );
}

asFramePlotDistributionsVirutal::~asFramePlotDistributionsVirutal()
{
	// Disconnect Events
	m_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	m_choiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	m_choiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	m_checkListTocPredictands->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
	m_buttonResetZoom->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotDistributionsVirutal::ResetExtent ), NULL, this );

}

asFrameGridAnalogsValuesVirtual::asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );

	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );

	m_panelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer291;
	bSizer291 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticTextForecast = new wxStaticText( m_panelOptions, wxID_ANY, _("Forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecast->Wrap( -1 );
	fgSizer1->Add( m_staticTextForecast, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

	wxArrayString m_choiceForecastChoices;
	m_choiceForecast = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceForecastChoices, 0 );
	m_choiceForecast->SetSelection( 0 );
	fgSizer1->Add( m_choiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextStation = new wxStaticText( m_panelOptions, wxID_ANY, _("Station"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextStation->Wrap( -1 );
	fgSizer1->Add( m_staticTextStation, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

	wxArrayString m_choiceStationChoices;
	m_choiceStation = new wxChoice( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceStationChoices, 0 );
	m_choiceStation->SetSelection( 0 );
	fgSizer1->Add( m_choiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextDate = new wxStaticText( m_panelOptions, wxID_ANY, _("Lead time"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDate->Wrap( -1 );
	fgSizer1->Add( m_staticTextDate, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

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
	m_grid->SetColSize( 0, 100 );
	m_grid->SetColSize( 1, 100 );
	m_grid->SetColSize( 2, 100 );
	m_grid->SetColSize( 3, 100 );
	m_grid->EnableDragColMove( false );
	m_grid->EnableDragColSize( true );
	m_grid->SetColLabelValue( 0, _("Analog") );
	m_grid->SetColLabelValue( 1, _("Date") );
	m_grid->SetColLabelValue( 2, _("Value") );
	m_grid->SetColLabelValue( 3, _("Criteria") );
	m_grid->SetColLabelSize( 30 );
	m_grid->SetColLabelAlignment( wxALIGN_CENTER, wxALIGN_CENTER );

	// Rows
	m_grid->EnableDragRowSize( true );
	m_grid->SetRowLabelSize( 40 );
	m_grid->SetRowLabelAlignment( wxALIGN_CENTER, wxALIGN_CENTER );

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
	this->SetSizeHints( wxSize( 1300,600 ), wxDefaultSize );

	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );

	m_panel15 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	m_splitterToc = new wxSplitterWindow( m_panel15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitterToc->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::m_splitterTocOnIdle ), NULL, this );
	m_splitterToc->SetMinimumPaneSize( 200 );

	m_scrolledWindowOptions = new wxScrolledWindow( m_splitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	m_scrolledWindowOptions->SetScrollRate( 5, 5 );
	m_sizerScrolledWindow = new wxBoxSizer( wxVERTICAL );

	m_staticTextChoiceMethod = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Method"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextChoiceMethod->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextChoiceMethod, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	wxArrayString m_choiceMethodChoices;
	m_choiceMethod = new wxChoice( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxSize( 220,-1 ), m_choiceMethodChoices, 0 );
	m_choiceMethod->SetSelection( 0 );
	m_sizerScrolledWindow->Add( m_choiceMethod, 0, wxEXPAND|wxBOTTOM, 5 );

	m_staticTextChoiceForecast = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Configuration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextChoiceForecast->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextChoiceForecast, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	wxArrayString m_choiceForecastChoices;
	m_choiceForecast = new wxChoice( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxSize( 220,-1 ), m_choiceForecastChoices, 0 );
	m_choiceForecast->SetSelection( 0 );
	m_sizerScrolledWindow->Add( m_choiceForecast, 0, wxEXPAND|wxBOTTOM, 5 );

	m_staticTextCheckListPredictors = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Possible predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextCheckListPredictors->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextCheckListPredictors, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_listPredictors = new wxListBox( m_scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	m_sizerScrolledWindow->Add( m_listPredictors, 1, wxEXPAND, 5 );

	m_staticTextTocLeft = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Layers of the left panel"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextTocLeft->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextTocLeft, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_staticTextTocRight = new wxStaticText( m_scrolledWindowOptions, wxID_ANY, _("Layers of the right panel"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextTocRight->Wrap( -1 );
	m_sizerScrolledWindow->Add( m_staticTextTocRight, 0, wxTOP|wxRIGHT|wxLEFT, 5 );


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

	m_staticTextTargetDates = new wxStaticText( m_panelLeft, wxID_ANY, _("Forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextTargetDates->Wrap( -1 );
	bSizer34->Add( m_staticTextTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxArrayString m_choiceTargetDatesChoices;
	m_choiceTargetDates = new wxChoice( m_panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceTargetDatesChoices, 0 );
	m_choiceTargetDates->SetSelection( 0 );
	m_choiceTargetDates->SetMinSize( wxSize( 100,-1 ) );

	bSizer34->Add( m_choiceTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer371->Add( bSizer34, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );

	m_panelGISLeft = new wxPanel( m_panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_SIMPLE );
	m_sizerGISLeft = new wxBoxSizer( wxVERTICAL );


	m_panelGISLeft->SetSizer( m_sizerGISLeft );
	m_panelGISLeft->Layout();
	m_sizerGISLeft->Fit( m_panelGISLeft );
	bSizer371->Add( m_panelGISLeft, 1, wxEXPAND, 5 );

	m_panelColorbarLeft = new wxPanel( m_panelLeft, wxID_ANY, wxDefaultPosition, wxSize( -1,30 ), wxTAB_TRAVERSAL );
	m_sizerColorbarLeft = new wxBoxSizer( wxVERTICAL );


	m_panelColorbarLeft->SetSizer( m_sizerColorbarLeft );
	m_panelColorbarLeft->Layout();
	bSizer371->Add( m_panelColorbarLeft, 0, wxALL|wxEXPAND, 5 );


	m_panelLeft->SetSizer( bSizer371 );
	m_panelLeft->Layout();
	bSizer371->Fit( m_panelLeft );
	m_sizerGIS->Add( m_panelLeft, 1, wxEXPAND, 5 );

	m_panelSwitch = new wxPanel( m_panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer40;
	bSizer40 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* m_sizerSwitch;
	m_sizerSwitch = new wxBoxSizer( wxVERTICAL );

	m_bpButtonSwitchRight = new wxBitmapButton( m_panelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	m_sizerSwitch->Add( m_bpButtonSwitchRight, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_CENTER_HORIZONTAL|wxRIGHT|wxLEFT, 1 );

	m_bpButtonSwitchLeft = new wxBitmapButton( m_panelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	m_sizerSwitch->Add( m_bpButtonSwitchLeft, 0, wxALIGN_CENTER_HORIZONTAL|wxRIGHT|wxLEFT, 1 );


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
	m_choiceAnalogDates->SetMinSize( wxSize( 120,-1 ) );

	bSizer35->Add( m_choiceAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer38->Add( bSizer35, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );

	m_panelGISRight = new wxPanel( m_panelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_SIMPLE );
	m_sizerGISRight = new wxBoxSizer( wxVERTICAL );


	m_panelGISRight->SetSizer( m_sizerGISRight );
	m_panelGISRight->Layout();
	m_sizerGISRight->Fit( m_panelGISRight );
	bSizer38->Add( m_panelGISRight, 1, wxEXPAND, 5 );

	m_panelColorbarRight = new wxPanel( m_panelRight, wxID_ANY, wxDefaultPosition, wxSize( -1,30 ), wxTAB_TRAVERSAL );
	m_sizerColorbarRight = new wxBoxSizer( wxVERTICAL );


	m_panelColorbarRight->SetSizer( m_sizerColorbarRight );
	m_panelColorbarRight->Layout();
	bSizer38->Add( m_panelColorbarRight, 0, wxALL|wxEXPAND, 5 );


	m_panelRight->SetSizer( bSizer38 );
	m_panelRight->Layout();
	bSizer38->Fit( m_panelRight );
	m_sizerGIS->Add( m_panelRight, 1, wxEXPAND, 5 );


	m_panelGIS->SetSizer( m_sizerGIS );
	m_panelGIS->Layout();
	m_sizerGIS->Fit( m_panelGIS );
	m_splitterToc->SplitVertically( m_scrolledWindowOptions, m_panelGIS, 220 );
	bSizer26->Add( m_splitterToc, 1, wxEXPAND, 5 );


	m_panel15->SetSizer( bSizer26 );
	m_panel15->Layout();
	bSizer26->Fit( m_panel15 );
	bSizer25->Add( m_panel15, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer25 );
	this->Layout();
	bSizer25->Fit( this );
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
	m_choiceMethod->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnMethodChange ), NULL, this );
	m_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	m_listPredictors->Connect( wxEVT_COMMAND_LISTBOX_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	m_choiceTargetDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	m_bpButtonSwitchRight->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	m_bpButtonSwitchLeft->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	m_choiceAnalogDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );
	m_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnOpenLayer ), this, m_menuItemOpenGisLayer->GetId());
}

asFramePredictorsVirtual::~asFramePredictorsVirtual()
{
	// Disconnect Events
	m_choiceMethod->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnMethodChange ), NULL, this );
	m_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	m_listPredictors->Disconnect( wxEVT_COMMAND_LISTBOX_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	m_choiceTargetDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	m_bpButtonSwitchRight->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	m_bpButtonSwitchLeft->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	m_choiceAnalogDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );

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

	m_staticTextForecastResultsDir = new wxStaticText( m_panelWorkspace, wxID_ANY, _("Directory containing the forecasts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextForecastResultsDir->Wrap( -1 );
	bSizer55->Add( m_staticTextForecastResultsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	m_dirPickerForecastResults = new wxDirPickerCtrl( m_panelWorkspace, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer55->Add( m_dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxFlexGridSizer* fgSizer81;
	fgSizer81 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer81->SetFlexibleDirection( wxBOTH );
	fgSizer81->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticTextColorbarMaxValue = new wxStaticText( m_panelWorkspace, wxID_ANY, _("Set the maximum rainfall value for the colorbar"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextColorbarMaxValue->Wrap( -1 );
	fgSizer81->Add( m_staticTextColorbarMaxValue, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlColorbarMaxValue = new wxTextCtrl( m_panelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer81->Add( m_textCtrlColorbarMaxValue, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextColorbarMaxUnit = new wxStaticText( m_panelWorkspace, wxID_ANY, _("mm/d"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextColorbarMaxUnit->Wrap( -1 );
	fgSizer81->Add( m_staticTextColorbarMaxUnit, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextPastDaysNb = new wxStaticText( m_panelWorkspace, wxID_ANY, _("Number of past days to display on the timeseries"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPastDaysNb->Wrap( -1 );
	fgSizer81->Add( m_staticTextPastDaysNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlPastDaysNb = new wxTextCtrl( m_panelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer81->Add( m_textCtrlPastDaysNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer55->Add( fgSizer81, 0, wxEXPAND|wxBOTTOM, 5 );

	wxStaticBoxSizer* sbSizer191;
	sbSizer191 = new wxStaticBoxSizer( new wxStaticBox( m_panelWorkspace, wxID_ANY, _("Alarms panel") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticTextAlarmsReturnPeriod = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("Return period to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsReturnPeriod->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString m_choiceAlarmsReturnPeriodChoices[] = { _("2"), _("5"), _("10"), _("20"), _("50"), _("100") };
	int m_choiceAlarmsReturnPeriodNChoices = sizeof( m_choiceAlarmsReturnPeriodChoices ) / sizeof( wxString );
	m_choiceAlarmsReturnPeriod = new wxChoice( sbSizer191->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceAlarmsReturnPeriodNChoices, m_choiceAlarmsReturnPeriodChoices, 0 );
	m_choiceAlarmsReturnPeriod->SetSelection( 0 );
	fgSizer13->Add( m_choiceAlarmsReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextAlarmsReturnPeriodYears = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsReturnPeriodYears->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsReturnPeriodYears, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextAlarmsQuantile = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("Quantile to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsQuantile->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsQuantile, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlAlarmsQuantile = new wxTextCtrl( sbSizer191->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer13->Add( m_textCtrlAlarmsQuantile, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticTextAlarmsQuantileRange = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("(in between 0 - 1)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextAlarmsQuantileRange->Wrap( -1 );
	fgSizer13->Add( m_staticTextAlarmsQuantileRange, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer191->Add( fgSizer13, 1, wxEXPAND, 5 );


	bSizer55->Add( sbSizer191, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( m_panelWorkspace, wxID_ANY, _("Maximum length of time series to display") ), wxVERTICAL );

	m_staticText581 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("Requires a restart or opening new forecasts."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText581->Wrap( -1 );
	sbSizer8->Add( m_staticText581, 0, wxALL, 5 );

	wxFlexGridSizer* fgSizer8;
	fgSizer8 = new wxFlexGridSizer( 0, 3, 0, 0 );
	fgSizer8->SetFlexibleDirection( wxBOTH );
	fgSizer8->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText541 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("Daily forecasts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText541->Wrap( -1 );
	fgSizer8->Add( m_staticText541, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlMaxLengthDaily = new wxTextCtrl( sbSizer8->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer8->Add( m_textCtrlMaxLengthDaily, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticText56 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("days"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText56->Wrap( -1 );
	fgSizer8->Add( m_staticText56, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticText55 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("Sub-daily forecasts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText55->Wrap( -1 );
	fgSizer8->Add( m_staticText55, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_textCtrlMaxLengthSubDaily = new wxTextCtrl( sbSizer8->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer8->Add( m_textCtrlMaxLengthSubDaily, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_staticText571 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("hours"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText571->Wrap( -1 );
	fgSizer8->Add( m_staticText571, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer8->Add( fgSizer8, 1, wxEXPAND, 5 );


	bSizer55->Add( sbSizer8, 0, wxEXPAND|wxALL, 5 );


	m_panelWorkspace->SetSizer( bSizer55 );
	m_panelWorkspace->Layout();
	bSizer55->Fit( m_panelWorkspace );
	m_notebookBase->AddPage( m_panelWorkspace, _("Workspace"), true );
	m_panelPaths = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer551;
	bSizer551 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( m_panelPaths, wxID_ANY, _("Path to the predictor datasets") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer6;
	fgSizer6 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer6->AddGrowableCol( 1 );
	fgSizer6->SetFlexibleDirection( wxBOTH );
	fgSizer6->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticPredictorID = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Dataset ID"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticPredictorID->Wrap( -1 );
	fgSizer6->Add( m_staticPredictorID, 0, wxALL, 5 );

	m_staticPredictorPaths = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Path to the directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticPredictorPaths->Wrap( -1 );
	fgSizer6->Add( m_staticPredictorPaths, 0, wxALL, 5 );

	m_textCtrlDatasetId1 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId1, 1, wxALL|wxEXPAND, 5 );

	m_dirPickerDataset1 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset1, 0, wxALL|wxEXPAND, 5 );

	m_textCtrlDatasetId2 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId2, 0, wxALL, 5 );

	m_dirPickerDataset2 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset2, 0, wxALL|wxEXPAND, 5 );

	m_textCtrlDatasetId3 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId3, 0, wxALL, 5 );

	m_dirPickerDataset3 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset3, 0, wxALL|wxEXPAND, 5 );

	m_textCtrlDatasetId4 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId4, 0, wxALL, 5 );

	m_dirPickerDataset4 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset4, 0, wxALL|wxEXPAND, 5 );

	m_textCtrlDatasetId5 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId5, 0, wxALL, 5 );

	m_dirPickerDataset5 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset5, 0, wxALL|wxEXPAND, 5 );

	m_textCtrlDatasetId6 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId6, 0, wxALL, 5 );

	m_dirPickerDataset6 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset6, 0, wxALL|wxEXPAND, 5 );

	m_textCtrlDatasetId7 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( m_textCtrlDatasetId7, 0, wxALL, 5 );

	m_dirPickerDataset7 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( m_dirPickerDataset7, 0, wxALL|wxEXPAND, 5 );


	sbSizer5->Add( fgSizer6, 1, wxEXPAND, 5 );


	bSizer551->Add( sbSizer5, 1, wxEXPAND|wxALL, 5 );


	m_panelPaths->SetSizer( bSizer551 );
	m_panelPaths->Layout();
	bSizer551->Fit( m_panelPaths );
	m_notebookBase->AddPage( m_panelPaths, _("Paths"), false );
	m_panelColors = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer5511;
	bSizer5511 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer51;
	sbSizer51 = new wxStaticBoxSizer( new wxStaticBox( m_panelColors, wxID_ANY, _("Paths to the color tables") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer61;
	fgSizer61 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer61->AddGrowableCol( 1 );
	fgSizer61->SetFlexibleDirection( wxBOTH );
	fgSizer61->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	m_staticText54 = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Geopotential height"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText54->Wrap( -1 );
	fgSizer61->Add( m_staticText54, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_filePickerColorZ = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( m_filePickerColorZ, 0, wxALL|wxEXPAND, 5 );

	RelativeHumidity = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Precipitable water"), wxDefaultPosition, wxDefaultSize, 0 );
	RelativeHumidity->Wrap( -1 );
	fgSizer61->Add( RelativeHumidity, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_filePickerColorPwat = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( m_filePickerColorPwat, 0, wxALL|wxEXPAND, 5 );

	m_staticText57 = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Relative humidity"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText57->Wrap( -1 );
	fgSizer61->Add( m_staticText57, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_filePickerColorRh = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( m_filePickerColorRh, 0, wxALL|wxEXPAND, 5 );

	m_staticText58 = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Specific humidity"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText58->Wrap( -1 );
	fgSizer61->Add( m_staticText58, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_filePickerColorSh = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( m_filePickerColorSh, 0, wxALL|wxEXPAND, 5 );


	sbSizer51->Add( fgSizer61, 1, wxEXPAND, 5 );


	bSizer5511->Add( sbSizer51, 1, wxEXPAND|wxALL, 5 );


	m_panelColors->SetSizer( bSizer5511 );
	m_panelColors->Layout();
	bSizer5511->Fit( m_panelColors );
	m_notebookBase->AddPage( m_panelColors, _("Colors"), false );
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

	m_staticText53 = new wxStaticText( sbSizer71->GetStaticBox(), wxID_ANY, _("Restart AtmoSwing for the change to take effect."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText53->Wrap( -1 );
	sbSizer71->Add( m_staticText53, 0, wxALL, 5 );


	bSizer16->Add( sbSizer71, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer62;
	bSizer62 = new wxBoxSizer( wxVERTICAL );

	m_radioBtnLogLevel1 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer62->Add( m_radioBtnLogLevel1, 0, wxALL, 5 );

	m_radioBtnLogLevel2 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer62->Add( m_radioBtnLogLevel2, 0, wxALL, 5 );

	m_radioBtnLogLevel3 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer62->Add( m_radioBtnLogLevel3, 0, wxALL, 5 );


	bSizer20->Add( bSizer62, 1, wxEXPAND, 5 );

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


	m_panelGeneralCommon->SetSizer( bSizer16 );
	m_panelGeneralCommon->Layout();
	bSizer16->Fit( m_panelGeneralCommon );
	m_notebookBase->AddPage( m_panelGeneralCommon, _("General"), false );
	m_panelAdvanced = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	m_checkBoxMultiInstancesViewer = new wxCheckBox( m_panelAdvanced, wxID_ANY, _("Allow multiple instances of the viewer"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26->Add( m_checkBoxMultiInstancesViewer, 0, wxALL, 5 );

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
