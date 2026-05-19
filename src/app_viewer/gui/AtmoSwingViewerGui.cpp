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

#include "AtmoSwingViewerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameViewerVirtual::asFrameViewerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1000,600 ), wxDefaultSize );

	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );

	_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxVERTICAL );

	_splitterGIS = new wxSplitterWindow( _panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_NOBORDER );
	_splitterGIS->Connect( wxEVT_IDLE, wxIdleEventHandler( asFrameViewerVirtual::_splitterGISOnIdle ), NULL, this );
	_splitterGIS->SetMinimumPaneSize( 270 );

	_scrolledWindowOptions = new wxScrolledWindow( _splitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	_scrolledWindowOptions->SetScrollRate( 5, 5 );
	_scrolledWindowOptions->SetBackgroundColour( wxColour( 255, 255, 255 ) );

	_sizerScrolledWindow = new wxBoxSizer( wxVERTICAL );


	_scrolledWindowOptions->SetSizer( _sizerScrolledWindow );
	_scrolledWindowOptions->Layout();
	_sizerScrolledWindow->Fit( _scrolledWindowOptions );
	_panelContent = new wxPanel( _splitterGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerContent = new wxBoxSizer( wxVERTICAL );

	_panelTop = new wxPanel( _panelContent, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxTAB_TRAVERSAL );
	_panelTop->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	_sizerTop = new wxBoxSizer( wxHORIZONTAL );

	_sizerTopLeft = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer52;
	bSizer52 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextForecastDate = new wxStaticText( _panelTop, wxID_ANY, _("No forecast opened"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecastDate->Wrap( -1 );
	_staticTextForecastDate->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	_staticTextForecastDate->SetForegroundColour( wxColour( 255, 255, 255 ) );

	bSizer52->Add( _staticTextForecastDate, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );

	_button51 = new wxButton( _panelTop, wxID_ANY, _("<<"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	_button51->SetForegroundColour( wxColour( 255, 255, 255 ) );
	_button51->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( _button51, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_button5 = new wxButton( _panelTop, wxID_ANY, _("<"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	_button5->SetForegroundColour( wxColour( 255, 255, 255 ) );
	_button5->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( _button5, 0, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_VERTICAL, 3 );

	_button6 = new wxButton( _panelTop, wxID_ANY, _(">"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	_button6->SetForegroundColour( wxColour( 255, 255, 255 ) );
	_button6->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( _button6, 0, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_VERTICAL, 3 );

	_button61 = new wxButton( _panelTop, wxID_ANY, _(">>"), wxDefaultPosition, wxSize( 20,20 ), 0|wxBORDER_NONE );
	_button61->SetForegroundColour( wxColour( 255, 255, 255 ) );
	_button61->SetBackgroundColour( wxColour( 77, 77, 77 ) );

	bSizer52->Add( _button61, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	_sizerTopLeft->Add( bSizer52, 1, wxEXPAND, 5 );

	_staticTextForecast = new wxStaticText( _panelTop, wxID_ANY, _("No forecast selected"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecast->Wrap( -1 );
	_staticTextForecast->SetFont( wxFont( 11, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	_staticTextForecast->SetForegroundColour( wxColour( 255, 255, 255 ) );

	_sizerTopLeft->Add( _staticTextForecast, 0, wxALL, 5 );


	_sizerTop->Add( _sizerTopLeft, 0, wxEXPAND, 5 );

	_sizerTopRight = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer39;
	bSizer39 = new wxBoxSizer( wxVERTICAL );

	_sizerLeadTimeSwitch = new wxBoxSizer( wxHORIZONTAL );


	bSizer39->Add( _sizerLeadTimeSwitch, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	_sizerTopRight->Add( bSizer39, 1, wxALIGN_RIGHT, 5 );


	_sizerTop->Add( _sizerTopRight, 1, wxEXPAND, 5 );


	_panelTop->SetSizer( _sizerTop );
	_panelTop->Layout();
	_sizerTop->Fit( _panelTop );
	_sizerContent->Add( _panelTop, 0, wxEXPAND, 5 );

	_panelGIS = new wxPanel( _panelContent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerGIS = new wxBoxSizer( wxVERTICAL );


	_panelGIS->SetSizer( _sizerGIS );
	_panelGIS->Layout();
	_sizerGIS->Fit( _panelGIS );
	_sizerContent->Add( _panelGIS, 1, wxEXPAND, 5 );


	_panelContent->SetSizer( _sizerContent );
	_panelContent->Layout();
	_sizerContent->Fit( _panelContent );
	_splitterGIS->SplitVertically( _scrolledWindowOptions, _panelContent, 270 );
	bSizer11->Add( _splitterGIS, 1, wxEXPAND|wxTOP, 4 );


	_panelMain->SetSizer( bSizer11 );
	_panelMain->Layout();
	bSizer11->Fit( _panelMain );
	bSizer3->Add( _panelMain, 1, wxEXPAND, 2 );


	this->SetSizer( bSizer3 );
	this->Layout();
	bSizer3->Fit( this );
	_menuBar = new wxMenuBar( 0 );
	_menuFile = new wxMenu();
	wxMenuItem* _menuItemOpenWorkspace;
	_menuItemOpenWorkspace = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Open a workspace") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemOpenWorkspace );

	wxMenuItem* _menuItemSaveWorkspace;
	_menuItemSaveWorkspace = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Save the workspace") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemSaveWorkspace );

	wxMenuItem* _menuItemSaveWorkspaceAs;
	_menuItemSaveWorkspaceAs = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Save the workspace as") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemSaveWorkspaceAs );

	wxMenuItem* _menuItemNewWorkspace;
	_menuItemNewWorkspace = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Create a new workspace") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemNewWorkspace );

	_menuFile->AppendSeparator();

	wxMenuItem* _menuItemOpenForecast;
	_menuItemOpenForecast = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Open a forecast file") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemOpenForecast );

	_menuFile->AppendSeparator();

	wxMenuItem* _menuItemOpenGISLayer;
	_menuItemOpenGISLayer = new wxMenuItem( _menuFile, wxID_OPEN, wxString( _("Open a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemOpenGISLayer );

	wxMenuItem* _menuItemCloseGISLayer;
	_menuItemCloseGISLayer = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Close a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemCloseGISLayer );

	wxMenuItem* _menuItemMoveGISLayer;
	_menuItemMoveGISLayer = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Move the selected layer") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemMoveGISLayer );

	_menuFile->AppendSeparator();

	wxMenuItem* _menuItemQuit;
	_menuItemQuit = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Quit") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemQuit );

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

	_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	_toolBar->Realize();

	_statusBar = this->CreateStatusBar( 2, wxSTB_SIZEGRIP, wxID_ANY );

	this->Centre( wxBOTH );

	// Connect Events
	_button51->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousDay ), NULL, this );
	_button5->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousForecast ), NULL, this );
	_button6->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextForecast ), NULL, this );
	_button61->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextDay ), NULL, this );
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnOpenWorkspace ), this, _menuItemOpenWorkspace->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnSaveWorkspace ), this, _menuItemSaveWorkspace->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnSaveWorkspaceAs ), this, _menuItemSaveWorkspaceAs->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnNewWorkspace ), this, _menuItemNewWorkspace->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnOpenForecast ), this, _menuItemOpenForecast->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnOpenLayer ), this, _menuItemOpenGISLayer->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnCloseLayer ), this, _menuItemCloseGISLayer->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnMoveLayer ), this, _menuItemMoveGISLayer->GetId());
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnQuit ), this, _menuItemQuit->GetId());
	_menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OpenFramePreferences ), this, _menuItemPreferences->GetId());
	_menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OpenFramePredictandDB ), this, _menuItemBuildPredictandDB->GetId());
	_menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnShowLog ), this, _menuItemShowLog->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnLogLevel1 ), this, _menuItemLogLevel1->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnLogLevel2 ), this, _menuItemLogLevel2->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OnLogLevel3 ), this, _menuItemLogLevel3->GetId());
	_menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameViewerVirtual::OpenFrameAbout ), this, _menuItemAbout->GetId());
}

asFrameViewerVirtual::~asFrameViewerVirtual()
{
	// Disconnect Events
	_button51->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousDay ), NULL, this );
	_button5->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadPreviousForecast ), NULL, this );
	_button6->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextForecast ), NULL, this );
	_button61->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameViewerVirtual::OnLoadNextDay ), NULL, this );

}

asWizardWorkspaceVirtual::asWizardWorkspaceVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style )
{
	this->Create( parent, id, title, bitmap, pos, style );

	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );

	wxWizardPageSimple* _wizPage1 = new wxWizardPageSimple( this );
	m_pages.Add( _wizPage1 );

	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxVERTICAL );

	_staticText37 = new wxStaticText( _wizPage1, wxID_ANY, _("Load / create a workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText37->Wrap( -1 );
	_staticText37->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer48->Add( _staticText37, 0, wxALL|wxEXPAND, 5 );

	_staticText35 = new wxStaticText( _wizPage1, wxID_ANY, _("Load an existing file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText35->Wrap( -1 );
	bSizer48->Add( _staticText35, 0, wxALL|wxEXPAND, 5 );

	_button4 = new wxButton( _wizPage1, wxID_ANY, _("Load workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer48->Add( _button4, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	_staticText46 = new wxStaticText( _wizPage1, wxID_ANY, _("or continue to create a new workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText46->Wrap( -1 );
	bSizer48->Add( _staticText46, 0, wxALL, 5 );


	_wizPage1->SetSizer( bSizer48 );
	_wizPage1->Layout();
	bSizer48->Fit( _wizPage1 );
	wxWizardPageSimple* _wizPage2 = new wxWizardPageSimple( this );
	m_pages.Add( _wizPage2 );

	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxVERTICAL );

	_staticText36 = new wxStaticText( _wizPage2, wxID_ANY, _("Create a new workspace"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText36->Wrap( -1 );
	_staticText36->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer49->Add( _staticText36, 0, wxALL|wxEXPAND, 5 );

	_staticText43 = new wxStaticText( _wizPage2, wxID_ANY, _("Path to save the new file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText43->Wrap( -1 );
	bSizer49->Add( _staticText43, 0, wxALL, 5 );

	_filePickerWorkspaceFile = new wxFilePickerCtrl( _wizPage2, wxID_ANY, wxEmptyString, _("Select a file"), _("*.asvw"), wxDefaultPosition, wxDefaultSize, wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizer49->Add( _filePickerWorkspaceFile, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	_wizPage2->SetSizer( bSizer49 );
	_wizPage2->Layout();
	bSizer49->Fit( _wizPage2 );
	wxWizardPageSimple* _wizPage3 = new wxWizardPageSimple( this );
	m_pages.Add( _wizPage3 );

	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxVERTICAL );

	_staticText44 = new wxStaticText( _wizPage3, wxID_ANY, _("Workspace options"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText44->Wrap( -1 );
	_staticText44->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer50->Add( _staticText44, 0, wxALL|wxEXPAND, 5 );

	_staticTextForecastResultsDir = new wxStaticText( _wizPage3, wxID_ANY, _("Path to the forecasts directory"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecastResultsDir->Wrap( -1 );
	bSizer50->Add( _staticTextForecastResultsDir, 0, wxALL, 5 );

	_dirPickerForecastResults = new wxDirPickerCtrl( _wizPage3, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer50->Add( _dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticText42 = new wxStaticText( _wizPage3, wxID_ANY, _("Other workspace parameters can be defined in the preferences frame."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText42->Wrap( -1 );
	bSizer50->Add( _staticText42, 1, wxALL|wxEXPAND, 5 );


	_wizPage3->SetSizer( bSizer50 );
	_wizPage3->Layout();
	bSizer50->Fit( _wizPage3 );
	wxWizardPageSimple* _wizPage4 = new wxWizardPageSimple( this );
	m_pages.Add( _wizPage4 );

	wxBoxSizer* bSizer51;
	bSizer51 = new wxBoxSizer( wxVERTICAL );

	_staticText45 = new wxStaticText( _wizPage4, wxID_ANY, _("Base map"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText45->Wrap( -1 );
	_staticText45->SetFont( wxFont( 13, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer51->Add( _staticText45, 0, wxALL|wxEXPAND, 5 );

	_staticText40 = new wxStaticText( _wizPage4, wxID_ANY, _("Choose the base map for your project"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText40->Wrap( -1 );
	bSizer51->Add( _staticText40, 0, wxALL, 5 );

	wxString _choiceBaseMapChoices[] = { _("Custom layers"), _("Terrain from Google maps (recommended)"), _("Map from Google maps"), _("Map from Openstreetmap"), _("Map from ArcGIS Mapserver"), _("Satellite imagery from Google maps"), _("Satellite imagery from VirtualEarth") };
	int _choiceBaseMapNChoices = sizeof( _choiceBaseMapChoices ) / sizeof( wxString );
	_choiceBaseMap = new wxChoice( _wizPage4, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceBaseMapNChoices, _choiceBaseMapChoices, 0 );
	_choiceBaseMap->SetSelection( 0 );
	bSizer51->Add( _choiceBaseMap, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	_staticText41 = new wxStaticText( _wizPage4, wxID_ANY, _("Other GIS layers can be added in the viewer frame directly, and be saved to the workspace."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText41->Wrap( -1 );
	bSizer51->Add( _staticText41, 1, wxALL|wxEXPAND, 5 );


	_wizPage4->SetSizer( bSizer51 );
	_wizPage4->Layout();
	bSizer51->Fit( _wizPage4 );

	this->Centre( wxBOTH );

	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}

	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardWorkspaceVirtual::OnWizardFinished ) );
	_button4->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardWorkspaceVirtual::OnLoadExistingWorkspace ), NULL, this );
}

asWizardWorkspaceVirtual::~asWizardWorkspaceVirtual()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( asWizardWorkspaceVirtual::OnWizardFinished ) );
	_button4->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asWizardWorkspaceVirtual::OnLoadExistingWorkspace ), NULL, this );

	m_pages.Clear();
}

asPanelSidebarVirtual::asPanelSidebarVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	this->SetBackgroundColour( wxColour( 255, 255, 255 ) );

	_sizerMain = new wxBoxSizer( wxVERTICAL );

	_panel28 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxSize( -1,2 ), wxTAB_TRAVERSAL );
	_panel28->SetBackgroundColour( wxColour( 102, 102, 102 ) );

	_sizerMain->Add( _panel28, 0, wxEXPAND, 5 );

	_panelHeader = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_panelHeader->SetBackgroundColour( wxColour( 150, 150, 150 ) );

	wxBoxSizer* _sizerHeader;
	_sizerHeader = new wxBoxSizer( wxHORIZONTAL );

	_header = new wxStaticText( _panelHeader, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	_header->Wrap( -1 );
	_header->SetFont( wxFont( 10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	_header->SetForegroundColour( wxColour( 255, 255, 255 ) );

	_sizerHeader->Add( _header, 1, wxALL|wxEXPAND, 5 );

	_bitmapCaret = new wxStaticBitmap( _panelHeader, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 22,22 ), 0 );
	_bitmapCaret->SetBackgroundColour( wxColour( 150, 150, 150 ) );

	_sizerHeader->Add( _bitmapCaret, 0, 0, 2 );


	_panelHeader->SetSizer( _sizerHeader );
	_panelHeader->Layout();
	_sizerHeader->Fit( _panelHeader );
	_sizerMain->Add( _panelHeader, 0, wxEXPAND, 5 );

	_sizerContent = new wxBoxSizer( wxVERTICAL );


	_sizerMain->Add( _sizerContent, 1, wxEXPAND, 5 );


	this->SetSizer( _sizerMain );
	this->Layout();
	_sizerMain->Fit( this );

	// Connect Events
	_panelHeader->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	_header->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	_bitmapCaret->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
}

asPanelSidebarVirtual::~asPanelSidebarVirtual()
{
	// Disconnect Events
	_panelHeader->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	_header->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );
	_bitmapCaret->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( asPanelSidebarVirtual::OnReducePanel ), NULL, this );

}

asFramePlotTimeSeriesVirtual::asFramePlotTimeSeriesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 500,300 ), wxDefaultSize );

	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxVERTICAL );

	_panelStationName = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextStationName = new wxStaticText( _panelStationName, wxID_ANY, _("Station name"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextStationName->Wrap( -1 );
	_staticTextStationName->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer37->Add( _staticTextStationName, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_buttonSaveTxt = new wxButton( _panelStationName, wxID_ANY, _("Export as txt"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	_buttonSaveTxt->SetFont( wxFont( 8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer37->Add( _buttonSaveTxt, 0, wxALL, 5 );

	_buttonPreview = new wxButton( _panelStationName, wxID_ANY, _("Preview"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	_buttonPreview->Enable( false );
	_buttonPreview->Hide();

	bSizer37->Add( _buttonPreview, 0, wxALL, 5 );

	_buttonPrint = new wxButton( _panelStationName, wxID_ANY, _("Print"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	_buttonPrint->Enable( false );
	_buttonPrint->Hide();

	bSizer37->Add( _buttonPrint, 0, wxALL, 5 );

	_buttonReset = new wxButton( _panelStationName, wxID_ANY, _("Reset zoom"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer37->Add( _buttonReset, 0, wxALL, 5 );


	bSizer29->Add( bSizer37, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	_panelStationName->SetSizer( bSizer29 );
	_panelStationName->Layout();
	bSizer29->Fit( _panelStationName );
	bSizer13->Add( _panelStationName, 0, wxEXPAND, 5 );

	_splitter = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_splitter->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotTimeSeriesVirtual::_splitterOnIdle ), NULL, this );
	_splitter->SetMinimumPaneSize( 150 );

	_panelLeft = new wxPanel( _splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );

	wxArrayString _checkListTocChoices;
	_checkListToc = new wxCheckListBox( _panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, _checkListTocChoices, 0 );
	bSizer27->Add( _checkListToc, 1, wxEXPAND, 5 );

	wxArrayString _checkListPastChoices;
	_checkListPast = new wxCheckListBox( _panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, _checkListPastChoices, 0 );
	bSizer27->Add( _checkListPast, 1, wxEXPAND|wxTOP, 5 );


	_panelLeft->SetSizer( bSizer27 );
	_panelLeft->Layout();
	bSizer27->Fit( _panelLeft );
	_panelRight = new wxPanel( _splitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_SIMPLE );
	_sizerPlot = new wxBoxSizer( wxVERTICAL );


	_panelRight->SetSizer( _sizerPlot );
	_panelRight->Layout();
	_sizerPlot->Fit( _panelRight );
	_splitter->SplitVertically( _panelLeft, _panelRight, 150 );
	bSizer13->Add( _splitter, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer13 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	_buttonSaveTxt->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	_buttonPreview->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	_buttonPrint->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	_buttonReset->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::ResetExtent ), NULL, this );
	_checkListToc->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	_checkListPast->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
}

asFramePlotTimeSeriesVirtual::~asFramePlotTimeSeriesVirtual()
{
	// Disconnect Events
	_buttonSaveTxt->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	_buttonPreview->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	_buttonPrint->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	_buttonReset->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::ResetExtent ), NULL, this );
	_checkListToc->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	_checkListPast->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );

}

asFramePlotDistributionsVirutal::asFramePlotDistributionsVirutal( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,400 ), wxDefaultSize );

	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxVERTICAL );

	_panelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextForecast = new wxStaticText( _panelOptions, wxID_ANY, _("Select forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecast->Wrap( -1 );
	fgSizer1->Add( _staticTextForecast, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );

	_staticTextStation = new wxStaticText( _panelOptions, wxID_ANY, _("Select station"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextStation->Wrap( -1 );
	fgSizer1->Add( _staticTextStation, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );

	_staticTextDate = new wxStaticText( _panelOptions, wxID_ANY, _("Select date"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDate->Wrap( -1 );
	fgSizer1->Add( _staticTextDate, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 5 );

	wxArrayString _choiceForecastChoices;
	_choiceForecast = new wxChoice( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceForecastChoices, 0 );
	_choiceForecast->SetSelection( 0 );
	fgSizer1->Add( _choiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxArrayString _choiceStationChoices;
	_choiceStation = new wxChoice( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceStationChoices, 0 );
	_choiceStation->SetSelection( 0 );
	fgSizer1->Add( _choiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxArrayString _choiceDateChoices;
	_choiceDate = new wxChoice( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceDateChoices, 0 );
	_choiceDate->SetSelection( 0 );
	fgSizer1->Add( _choiceDate, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	bSizer29->Add( fgSizer1, 1, wxALIGN_CENTER_HORIZONTAL, 5 );


	_panelOptions->SetSizer( bSizer29 );
	_panelOptions->Layout();
	bSizer29->Fit( _panelOptions );
	bSizer13->Add( _panelOptions, 0, wxEXPAND, 5 );

	_notebook = new wxNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelPredictands = new wxPanel( _notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer22;
	bSizer22 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer52;
	bSizer52 = new wxBoxSizer( wxVERTICAL );


	bSizer22->Add( bSizer52, 1, wxEXPAND, 5 );

	_splitter4 = new wxSplitterWindow( _panelPredictands, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::_splitter4OnIdle ), NULL, this );

	_panelPredictandsLeft = new wxPanel( _splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxVERTICAL );

	wxArrayString _checkListTocPredictandsChoices;
	_checkListTocPredictands = new wxCheckListBox( _panelPredictandsLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, _checkListTocPredictandsChoices, 0|wxBORDER_NONE );
	bSizer55->Add( _checkListTocPredictands, 1, wxEXPAND|wxTOP|wxBOTTOM|wxLEFT, 5 );

	_buttonResetZoom = new wxButton( _panelPredictandsLeft, wxID_ANY, _("Reset zoom"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer55->Add( _buttonResetZoom, 0, wxALL|wxEXPAND, 5 );


	_panelPredictandsLeft->SetSizer( bSizer55 );
	_panelPredictandsLeft->Layout();
	bSizer55->Fit( _panelPredictandsLeft );
	_panelPredictandsRight = new wxPanel( _splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerPlotPredictands = new wxBoxSizer( wxVERTICAL );


	_panelPredictandsRight->SetSizer( _sizerPlotPredictands );
	_panelPredictandsRight->Layout();
	_sizerPlotPredictands->Fit( _panelPredictandsRight );
	_splitter4->SplitVertically( _panelPredictandsLeft, _panelPredictandsRight, 178 );
	bSizer22->Add( _splitter4, 1, wxEXPAND, 5 );


	_panelPredictands->SetSizer( bSizer22 );
	_panelPredictands->Layout();
	bSizer22->Fit( _panelPredictands );
	_notebook->AddPage( _panelPredictands, _("Predictands distribution"), true );
	_panelCriteria = new wxPanel( _notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerPlotCriteria = new wxBoxSizer( wxVERTICAL );


	_panelCriteria->SetSizer( _sizerPlotCriteria );
	_panelCriteria->Layout();
	_sizerPlotCriteria->Fit( _panelCriteria );
	_notebook->AddPage( _panelCriteria, _("Criteria distribution"), false );

	bSizer13->Add( _notebook, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer13 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	_choiceStation->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	_choiceDate->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	_checkListTocPredictands->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
	_buttonResetZoom->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotDistributionsVirutal::ResetExtent ), NULL, this );
}

asFramePlotDistributionsVirutal::~asFramePlotDistributionsVirutal()
{
	// Disconnect Events
	_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceForecastChange ), NULL, this );
	_choiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceStationChange ), NULL, this );
	_choiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnChoiceDateChange ), NULL, this );
	_checkListTocPredictands->Disconnect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotDistributionsVirutal::OnTocSelectionChange ), NULL, this );
	_buttonResetZoom->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotDistributionsVirutal::ResetExtent ), NULL, this );

}

asFrameGridAnalogsValuesVirtual::asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxSize( -1,-1 ) );

	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );

	_panelOptions = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer291;
	bSizer291 = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextForecast = new wxStaticText( _panelOptions, wxID_ANY, _("Forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecast->Wrap( -1 );
	fgSizer1->Add( _staticTextForecast, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

	wxArrayString _choiceForecastChoices;
	_choiceForecast = new wxChoice( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceForecastChoices, 0 );
	_choiceForecast->SetSelection( 0 );
	fgSizer1->Add( _choiceForecast, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextStation = new wxStaticText( _panelOptions, wxID_ANY, _("Station"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextStation->Wrap( -1 );
	fgSizer1->Add( _staticTextStation, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

	wxArrayString _choiceStationChoices;
	_choiceStation = new wxChoice( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceStationChoices, 0 );
	_choiceStation->SetSelection( 0 );
	fgSizer1->Add( _choiceStation, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextDate = new wxStaticText( _panelOptions, wxID_ANY, _("Lead time"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDate->Wrap( -1 );
	fgSizer1->Add( _staticTextDate, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

	wxArrayString _choiceDateChoices;
	_choiceDate = new wxChoice( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceDateChoices, 0 );
	_choiceDate->SetSelection( 0 );
	fgSizer1->Add( _choiceDate, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	bSizer291->Add( fgSizer1, 1, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );


	_panelOptions->SetSizer( bSizer291 );
	_panelOptions->Layout();
	bSizer291->Fit( _panelOptions );
	bSizer29->Add( _panelOptions, 0, wxEXPAND, 5 );

	_grid = new wxGrid( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), 0 );

	// Grid
	_grid->CreateGrid( 5, 4 );
	_grid->EnableEditing( false );
	_grid->EnableGridLines( true );
	_grid->EnableDragGridSize( false );
	_grid->SetMargins( 0, 0 );

	// Columns
	_grid->SetColSize( 0, 100 );
	_grid->SetColSize( 1, 100 );
	_grid->SetColSize( 2, 100 );
	_grid->SetColSize( 3, 100 );
	_grid->EnableDragColMove( false );
	_grid->EnableDragColSize( true );
	_grid->SetColLabelValue( 0, _("Analog") );
	_grid->SetColLabelValue( 1, _("Date") );
	_grid->SetColLabelValue( 2, _("Value") );
	_grid->SetColLabelValue( 3, _("Criteria") );
	_grid->SetColLabelSize( 30 );
	_grid->SetColLabelAlignment( wxALIGN_CENTER, wxALIGN_CENTER );

	// Rows
	_grid->EnableDragRowSize( true );
	_grid->SetRowLabelSize( 40 );
	_grid->SetRowLabelAlignment( wxALIGN_CENTER, wxALIGN_CENTER );

	// Label Appearance

	// Cell Defaults
	_grid->SetDefaultCellAlignment( wxALIGN_RIGHT, wxALIGN_TOP );
	bSizer29->Add( _grid, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer29 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceForecastChange ), NULL, this );
	_choiceStation->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceStationChange ), NULL, this );
	_choiceDate->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceDateChange ), NULL, this );
	_grid->Connect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	_grid->Connect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	_grid->Connect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	_grid->Connect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
}

asFrameGridAnalogsValuesVirtual::~asFrameGridAnalogsValuesVirtual()
{
	// Disconnect Events
	_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceForecastChange ), NULL, this );
	_choiceStation->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceStationChange ), NULL, this );
	_choiceDate->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFrameGridAnalogsValuesVirtual::OnChoiceDateChange ), NULL, this );
	_grid->Disconnect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	_grid->Disconnect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	_grid->Disconnect( wxEVT_GRID_LABEL_LEFT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );
	_grid->Disconnect( wxEVT_GRID_LABEL_RIGHT_CLICK, wxGridEventHandler( asFrameGridAnalogsValuesVirtual::SortGrid ), NULL, this );

}

asFramePredictorsVirtual::asFramePredictorsVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1300,600 ), wxDefaultSize );

	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );

	_panel15 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	_splitterToc = new wxSplitterWindow( _panel15, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	_splitterToc->Connect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::_splitterTocOnIdle ), NULL, this );
	_splitterToc->SetMinimumPaneSize( 200 );

	_scrolledWindowOptions = new wxScrolledWindow( _splitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	_scrolledWindowOptions->SetScrollRate( 5, 5 );
	_sizerScrolledWindow = new wxBoxSizer( wxVERTICAL );

	_staticTextChoiceMethod = new wxStaticText( _scrolledWindowOptions, wxID_ANY, _("Method"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextChoiceMethod->Wrap( -1 );
	_sizerScrolledWindow->Add( _staticTextChoiceMethod, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	wxArrayString _choiceMethodChoices;
	_choiceMethod = new wxChoice( _scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxSize( 220,-1 ), _choiceMethodChoices, 0 );
	_choiceMethod->SetSelection( 0 );
	_sizerScrolledWindow->Add( _choiceMethod, 0, wxEXPAND|wxBOTTOM, 5 );

	_staticTextChoiceForecast = new wxStaticText( _scrolledWindowOptions, wxID_ANY, _("Configuration"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextChoiceForecast->Wrap( -1 );
	_sizerScrolledWindow->Add( _staticTextChoiceForecast, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	wxArrayString _choiceForecastChoices;
	_choiceForecast = new wxChoice( _scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxSize( 220,-1 ), _choiceForecastChoices, 0 );
	_choiceForecast->SetSelection( 0 );
	_sizerScrolledWindow->Add( _choiceForecast, 0, wxEXPAND|wxBOTTOM, 5 );

	_staticTextCheckListPredictors = new wxStaticText( _scrolledWindowOptions, wxID_ANY, _("Possible predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextCheckListPredictors->Wrap( -1 );
	_sizerScrolledWindow->Add( _staticTextCheckListPredictors, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_listPredictors = new wxListBox( _scrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	_sizerScrolledWindow->Add( _listPredictors, 1, wxEXPAND, 5 );

	_staticTextTocLeft = new wxStaticText( _scrolledWindowOptions, wxID_ANY, _("Layers of the left panel"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextTocLeft->Wrap( -1 );
	_sizerScrolledWindow->Add( _staticTextTocLeft, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_staticTextTocRight = new wxStaticText( _scrolledWindowOptions, wxID_ANY, _("Layers of the right panel"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextTocRight->Wrap( -1 );
	_sizerScrolledWindow->Add( _staticTextTocRight, 0, wxTOP|wxRIGHT|wxLEFT, 5 );


	_scrolledWindowOptions->SetSizer( _sizerScrolledWindow );
	_scrolledWindowOptions->Layout();
	_sizerScrolledWindow->Fit( _scrolledWindowOptions );
	_panelGIS = new wxPanel( _splitterToc, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerGIS = new wxBoxSizer( wxHORIZONTAL );

	_panelLeft = new wxPanel( _panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer371;
	bSizer371 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextTargetDates = new wxStaticText( _panelLeft, wxID_ANY, _("Forecast"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextTargetDates->Wrap( -1 );
	bSizer34->Add( _staticTextTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxArrayString _choiceTargetDatesChoices;
	_choiceTargetDates = new wxChoice( _panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceTargetDatesChoices, 0 );
	_choiceTargetDates->SetSelection( 0 );
	_choiceTargetDates->SetMinSize( wxSize( 100,-1 ) );

	bSizer34->Add( _choiceTargetDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer371->Add( bSizer34, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );

	_panelGISLeft = new wxPanel( _panelLeft, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_SIMPLE );
	_sizerGISLeft = new wxBoxSizer( wxVERTICAL );


	_panelGISLeft->SetSizer( _sizerGISLeft );
	_panelGISLeft->Layout();
	_sizerGISLeft->Fit( _panelGISLeft );
	bSizer371->Add( _panelGISLeft, 1, wxEXPAND, 5 );

	_panelColorbarLeft = new wxPanel( _panelLeft, wxID_ANY, wxDefaultPosition, wxSize( -1,30 ), wxTAB_TRAVERSAL );
	_sizerColorbarLeft = new wxBoxSizer( wxVERTICAL );


	_panelColorbarLeft->SetSizer( _sizerColorbarLeft );
	_panelColorbarLeft->Layout();
	bSizer371->Add( _panelColorbarLeft, 0, wxALL|wxEXPAND, 5 );


	_panelLeft->SetSizer( bSizer371 );
	_panelLeft->Layout();
	bSizer371->Fit( _panelLeft );
	_sizerGIS->Add( _panelLeft, 1, wxEXPAND, 5 );

	_panelSwitch = new wxPanel( _panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer40;
	bSizer40 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* _sizerSwitch;
	_sizerSwitch = new wxBoxSizer( wxVERTICAL );

	_bpButtonSwitchRight = new wxBitmapButton( _panelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	_sizerSwitch->Add( _bpButtonSwitchRight, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_CENTER_HORIZONTAL|wxRIGHT|wxLEFT, 1 );

	_bpButtonSwitchLeft = new wxBitmapButton( _panelSwitch, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxSize( 10,28 ), wxBU_AUTODRAW|0|wxBORDER_NONE );
	_sizerSwitch->Add( _bpButtonSwitchLeft, 0, wxALIGN_CENTER_HORIZONTAL|wxRIGHT|wxLEFT, 1 );


	bSizer40->Add( _sizerSwitch, 1, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5 );


	_panelSwitch->SetSizer( bSizer40 );
	_panelSwitch->Layout();
	bSizer40->Fit( _panelSwitch );
	_sizerGIS->Add( _panelSwitch, 0, wxEXPAND, 5 );

	_panelRight = new wxPanel( _panelGIS, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextAnalogDates = new wxStaticText( _panelRight, wxID_ANY, _("Analogs"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextAnalogDates->Wrap( -1 );
	bSizer35->Add( _staticTextAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxArrayString _choiceAnalogDatesChoices;
	_choiceAnalogDates = new wxChoice( _panelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceAnalogDatesChoices, 0 );
	_choiceAnalogDates->SetSelection( 0 );
	_choiceAnalogDates->SetMinSize( wxSize( 120,-1 ) );

	bSizer35->Add( _choiceAnalogDates, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer38->Add( bSizer35, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL, 5 );

	_panelGISRight = new wxPanel( _panelRight, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL|wxBORDER_SIMPLE );
	_sizerGISRight = new wxBoxSizer( wxVERTICAL );


	_panelGISRight->SetSizer( _sizerGISRight );
	_panelGISRight->Layout();
	_sizerGISRight->Fit( _panelGISRight );
	bSizer38->Add( _panelGISRight, 1, wxEXPAND, 5 );

	_panelColorbarRight = new wxPanel( _panelRight, wxID_ANY, wxDefaultPosition, wxSize( -1,30 ), wxTAB_TRAVERSAL );
	_sizerColorbarRight = new wxBoxSizer( wxVERTICAL );


	_panelColorbarRight->SetSizer( _sizerColorbarRight );
	_panelColorbarRight->Layout();
	bSizer38->Add( _panelColorbarRight, 0, wxALL|wxEXPAND, 5 );


	_panelRight->SetSizer( bSizer38 );
	_panelRight->Layout();
	bSizer38->Fit( _panelRight );
	_sizerGIS->Add( _panelRight, 1, wxEXPAND, 5 );


	_panelGIS->SetSizer( _sizerGIS );
	_panelGIS->Layout();
	_sizerGIS->Fit( _panelGIS );
	_splitterToc->SplitVertically( _scrolledWindowOptions, _panelGIS, 220 );
	bSizer26->Add( _splitterToc, 1, wxEXPAND, 5 );


	_panel15->SetSizer( bSizer26 );
	_panel15->Layout();
	bSizer26->Fit( _panel15 );
	bSizer25->Add( _panel15, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer25 );
	this->Layout();
	bSizer25->Fit( this );
	_menubar = new wxMenuBar( 0 );
	_menuFile = new wxMenu();
	wxMenuItem* _menuItemOpenGisLayer;
	_menuItemOpenGisLayer = new wxMenuItem( _menuFile, wxID_ANY, wxString( _("Open GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	_menuFile->Append( _menuItemOpenGisLayer );

	_menubar->Append( _menuFile, _("File") );

	_menuTools = new wxMenu();
	_menubar->Append( _menuTools, _("Tools") );

	this->SetMenuBar( _menubar );

	_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	_toolBar->Realize();


	this->Centre( wxBOTH );

	// Connect Events
	_choiceMethod->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnMethodChange ), NULL, this );
	_choiceForecast->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	_listPredictors->Connect( wxEVT_COMMAND_LISTBOX_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	_choiceTargetDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	_bpButtonSwitchRight->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	_bpButtonSwitchLeft->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	_choiceAnalogDates->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );
	_menuFile->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnOpenLayer ), this, _menuItemOpenGisLayer->GetId());
}

asFramePredictorsVirtual::~asFramePredictorsVirtual()
{
	// Disconnect Events
	_choiceMethod->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnMethodChange ), NULL, this );
	_choiceForecast->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnForecastChange ), NULL, this );
	_listPredictors->Disconnect( wxEVT_COMMAND_LISTBOX_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnPredictorSelectionChange ), NULL, this );
	_choiceTargetDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnTargetDateChange ), NULL, this );
	_bpButtonSwitchRight->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchRight ), NULL, this );
	_bpButtonSwitchLeft->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictorsVirtual::OnSwitchLeft ), NULL, this );
	_choiceAnalogDates->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictorsVirtual::OnAnalogDateChange ), NULL, this );

}

asFramePreferencesViewerVirtual::asFramePreferencesViewerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );

	_panelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );

	_notebookBase = new wxNotebook( _panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelWorkspace = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxVERTICAL );

	_staticTextForecastResultsDir = new wxStaticText( _panelWorkspace, wxID_ANY, _("Directory containing the forecasts"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextForecastResultsDir->Wrap( -1 );
	bSizer55->Add( _staticTextForecastResultsDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );

	_dirPickerForecastResults = new wxDirPickerCtrl( _panelWorkspace, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer55->Add( _dirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxFlexGridSizer* fgSizer81;
	fgSizer81 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer81->SetFlexibleDirection( wxBOTH );
	fgSizer81->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextColorbarMaxValue = new wxStaticText( _panelWorkspace, wxID_ANY, _("Set the maximum rainfall value for the colorbar"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextColorbarMaxValue->Wrap( -1 );
	fgSizer81->Add( _staticTextColorbarMaxValue, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlColorbarMaxValue = new wxTextCtrl( _panelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer81->Add( _textCtrlColorbarMaxValue, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextColorbarMaxUnit = new wxStaticText( _panelWorkspace, wxID_ANY, _("mm/d"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextColorbarMaxUnit->Wrap( -1 );
	fgSizer81->Add( _staticTextColorbarMaxUnit, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextPastDaysNb = new wxStaticText( _panelWorkspace, wxID_ANY, _("Number of past days to display on the timeseries"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPastDaysNb->Wrap( -1 );
	fgSizer81->Add( _staticTextPastDaysNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlPastDaysNb = new wxTextCtrl( _panelWorkspace, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer81->Add( _textCtrlPastDaysNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer55->Add( fgSizer81, 0, wxEXPAND|wxBOTTOM, 5 );

	wxStaticBoxSizer* sbSizer191;
	sbSizer191 = new wxStaticBoxSizer( new wxStaticBox( _panelWorkspace, wxID_ANY, _("Alarms panel") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextAlarmsReturnPeriod = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("Return period to display"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextAlarmsReturnPeriod->Wrap( -1 );
	fgSizer13->Add( _staticTextAlarmsReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString _choiceAlarmsReturnPeriodChoices[] = { _("2"), _("5"), _("10"), _("20"), _("50"), _("100") };
	int _choiceAlarmsReturnPeriodNChoices = sizeof( _choiceAlarmsReturnPeriodChoices ) / sizeof( wxString );
	_choiceAlarmsReturnPeriod = new wxChoice( sbSizer191->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceAlarmsReturnPeriodNChoices, _choiceAlarmsReturnPeriodChoices, 0 );
	_choiceAlarmsReturnPeriod->SetSelection( 0 );
	fgSizer13->Add( _choiceAlarmsReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextAlarmsReturnPeriodYears = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextAlarmsReturnPeriodYears->Wrap( -1 );
	fgSizer13->Add( _staticTextAlarmsReturnPeriodYears, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextAlarmsQuantile = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("Quantile to display"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextAlarmsQuantile->Wrap( -1 );
	fgSizer13->Add( _staticTextAlarmsQuantile, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlAlarmsQuantile = new wxTextCtrl( sbSizer191->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	fgSizer13->Add( _textCtrlAlarmsQuantile, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextAlarmsQuantileRange = new wxStaticText( sbSizer191->GetStaticBox(), wxID_ANY, _("(in between 0 - 1)"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextAlarmsQuantileRange->Wrap( -1 );
	fgSizer13->Add( _staticTextAlarmsQuantileRange, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer191->Add( fgSizer13, 1, wxEXPAND, 5 );


	bSizer55->Add( sbSizer191, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( _panelWorkspace, wxID_ANY, _("Maximum length of time series to display") ), wxVERTICAL );

	_staticText581 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("Requires a restart or opening new forecasts."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText581->Wrap( -1 );
	sbSizer8->Add( _staticText581, 0, wxALL, 5 );

	wxFlexGridSizer* fgSizer8;
	fgSizer8 = new wxFlexGridSizer( 0, 3, 0, 0 );
	fgSizer8->SetFlexibleDirection( wxBOTH );
	fgSizer8->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticText541 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("Daily forecasts"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText541->Wrap( -1 );
	fgSizer8->Add( _staticText541, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlMaxLengthDaily = new wxTextCtrl( sbSizer8->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer8->Add( _textCtrlMaxLengthDaily, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticText56 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("days"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText56->Wrap( -1 );
	fgSizer8->Add( _staticText56, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticText55 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("Sub-daily forecasts"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText55->Wrap( -1 );
	fgSizer8->Add( _staticText55, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlMaxLengthSubDaily = new wxTextCtrl( sbSizer8->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer8->Add( _textCtrlMaxLengthSubDaily, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticText571 = new wxStaticText( sbSizer8->GetStaticBox(), wxID_ANY, _("hours"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText571->Wrap( -1 );
	fgSizer8->Add( _staticText571, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer8->Add( fgSizer8, 1, wxEXPAND, 5 );


	bSizer55->Add( sbSizer8, 0, wxEXPAND|wxALL, 5 );


	_panelWorkspace->SetSizer( bSizer55 );
	_panelWorkspace->Layout();
	bSizer55->Fit( _panelWorkspace );
	_notebookBase->AddPage( _panelWorkspace, _("Workspace"), true );
	_panelPaths = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer551;
	bSizer551 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( _panelPaths, wxID_ANY, _("Path to the predictor datasets") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer6;
	fgSizer6 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer6->AddGrowableCol( 1 );
	fgSizer6->SetFlexibleDirection( wxBOTH );
	fgSizer6->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticPredictorID = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Dataset ID"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticPredictorID->Wrap( -1 );
	fgSizer6->Add( _staticPredictorID, 0, wxALL, 5 );

	_staticPredictorPaths = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Path to the directory"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticPredictorPaths->Wrap( -1 );
	fgSizer6->Add( _staticPredictorPaths, 0, wxALL, 5 );

	_textCtrlDatasetId1 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId1, 1, wxALL|wxEXPAND, 5 );

	_dirPickerDataset1 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset1, 0, wxALL|wxEXPAND, 5 );

	_textCtrlDatasetId2 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId2, 0, wxALL, 5 );

	_dirPickerDataset2 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset2, 0, wxALL|wxEXPAND, 5 );

	_textCtrlDatasetId3 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId3, 0, wxALL, 5 );

	_dirPickerDataset3 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset3, 0, wxALL|wxEXPAND, 5 );

	_textCtrlDatasetId4 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId4, 0, wxALL, 5 );

	_dirPickerDataset4 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset4, 0, wxALL|wxEXPAND, 5 );

	_textCtrlDatasetId5 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId5, 0, wxALL, 5 );

	_dirPickerDataset5 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset5, 0, wxALL|wxEXPAND, 5 );

	_textCtrlDatasetId6 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId6, 0, wxALL, 5 );

	_dirPickerDataset6 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset6, 0, wxALL|wxEXPAND, 5 );

	_textCtrlDatasetId7 = new wxTextCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 150,-1 ), 0 );
	fgSizer6->Add( _textCtrlDatasetId7, 0, wxALL, 5 );

	_dirPickerDataset7 = new wxDirPickerCtrl( sbSizer5->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	fgSizer6->Add( _dirPickerDataset7, 0, wxALL|wxEXPAND, 5 );


	sbSizer5->Add( fgSizer6, 1, wxEXPAND, 5 );


	bSizer551->Add( sbSizer5, 1, wxEXPAND|wxALL, 5 );


	_panelPaths->SetSizer( bSizer551 );
	_panelPaths->Layout();
	bSizer551->Fit( _panelPaths );
	_notebookBase->AddPage( _panelPaths, _("Paths"), false );
	_panelColors = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer5511;
	bSizer5511 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer51;
	sbSizer51 = new wxStaticBoxSizer( new wxStaticBox( _panelColors, wxID_ANY, _("Paths to the color tables") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer61;
	fgSizer61 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer61->AddGrowableCol( 1 );
	fgSizer61->SetFlexibleDirection( wxBOTH );
	fgSizer61->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticText54 = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Geopotential height"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText54->Wrap( -1 );
	fgSizer61->Add( _staticText54, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_filePickerColorZ = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( _filePickerColorZ, 0, wxALL|wxEXPAND, 5 );

	RelativeHumidity = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Precipitable water"), wxDefaultPosition, wxDefaultSize, 0 );
	RelativeHumidity->Wrap( -1 );
	fgSizer61->Add( RelativeHumidity, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_filePickerColorPwat = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( _filePickerColorPwat, 0, wxALL|wxEXPAND, 5 );

	_staticText57 = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Relative humidity"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText57->Wrap( -1 );
	fgSizer61->Add( _staticText57, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_filePickerColorRh = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( _filePickerColorRh, 0, wxALL|wxEXPAND, 5 );

	_staticText58 = new wxStaticText( sbSizer51->GetStaticBox(), wxID_ANY, _("Specific humidity"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText58->Wrap( -1 );
	fgSizer61->Add( _staticText58, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_filePickerColorSh = new wxFilePickerCtrl( sbSizer51->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	fgSizer61->Add( _filePickerColorSh, 0, wxALL|wxEXPAND, 5 );


	sbSizer51->Add( fgSizer61, 1, wxEXPAND, 5 );


	bSizer5511->Add( sbSizer51, 1, wxEXPAND|wxALL, 5 );


	_panelColors->SetSizer( bSizer5511 );
	_panelColors->Layout();
	bSizer5511->Fit( _panelColors );
	_notebookBase->AddPage( _panelColors, _("Colors"), false );
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

	_staticText53 = new wxStaticText( sbSizer71->GetStaticBox(), wxID_ANY, _("Restart AtmoSwing for the change to take effect."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText53->Wrap( -1 );
	sbSizer71->Add( _staticText53, 0, wxALL, 5 );


	bSizer16->Add( sbSizer71, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer62;
	bSizer62 = new wxBoxSizer( wxVERTICAL );

	_radioBtnLogLevel1 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer62->Add( _radioBtnLogLevel1, 0, wxALL, 5 );

	_radioBtnLogLevel2 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer62->Add( _radioBtnLogLevel2, 0, wxALL, 5 );

	_radioBtnLogLevel3 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer62->Add( _radioBtnLogLevel3, 0, wxALL, 5 );


	bSizer20->Add( bSizer62, 1, wxEXPAND, 5 );

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


	_panelGeneralCommon->SetSizer( bSizer16 );
	_panelGeneralCommon->Layout();
	bSizer16->Fit( _panelGeneralCommon );
	_notebookBase->AddPage( _panelGeneralCommon, _("General"), false );
	_panelAdvanced = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	_checkBoxMultiInstancesViewer = new wxCheckBox( _panelAdvanced, wxID_ANY, _("Allow multiple instances of the viewer"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer26->Add( _checkBoxMultiInstancesViewer, 0, wxALL, 5 );

	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( _panelAdvanced, wxID_ANY, _("User specific paths") ), wxVERTICAL );

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


	bSizer26->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );


	_panelAdvanced->SetSizer( bSizer26 );
	_panelAdvanced->Layout();
	bSizer26->Fit( _panelAdvanced );
	_notebookBase->AddPage( _panelAdvanced, _("Advanced"), false );

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
	_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::ApplyChanges ), NULL, this );
	_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesViewerVirtual::~asFramePreferencesViewerVirtual()
{
	// Disconnect Events
	_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::ApplyChanges ), NULL, this );
	_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesViewerVirtual::SaveAndClose ), NULL, this );

}
