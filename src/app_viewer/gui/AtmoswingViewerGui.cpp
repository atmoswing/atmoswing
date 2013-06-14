///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct  8 2012)
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
	this->SetSizeHints( wxSize( 800,600 ), wxDefaultSize );
	
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
	
	
	m_SizerTop->Add( m_SizerTopLeft, 1, wxALIGN_LEFT|wxEXPAND, 5 );
	
	m_SizerTopRight = new wxBoxSizer( wxVERTICAL );
	
	
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
	wxMenuItem* m_MenuItemQuit;
	m_MenuItemQuit = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Quit") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemQuit );
	
	wxMenuItem* m_MenuItemOpenGISLayer;
	m_MenuItemOpenGISLayer = new wxMenuItem( m_MenuFile, wxID_OPEN, wxString( _("Open a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenGISLayer );
	
	wxMenuItem* m_MenuItemCloseGISLayer;
	m_MenuItemCloseGISLayer = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Close a GIS layer") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemCloseGISLayer );
	
	wxMenuItem* m_MenuItemOpenForecast;
	m_MenuItemOpenForecast = new wxMenuItem( m_MenuFile, wxID_ANY, wxString( _("Open a forecast file") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuFile->Append( m_MenuItemOpenForecast );
	
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
	m_ToolBar->Realize(); 
	
	m_StatusBar = this->CreateStatusBar( 2, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( m_MenuItemQuit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnQuit ) );
	this->Connect( m_MenuItemOpenGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenLayer ) );
	this->Connect( m_MenuItemCloseGISLayer->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnCloseLayer ) );
	this->Connect( m_MenuItemOpenForecast->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenForecast ) );
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
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnQuit ) );
	this->Disconnect( wxID_OPEN, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenLayer ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnCloseLayer ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameForecastVirtual::OnOpenForecast ) );
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
	
	m_ButtonSVG = new wxButton( m_PanelStationName, wxID_ANY, _("Export as svg"), wxDefaultPosition, wxSize( 80,20 ), 0 );
	m_ButtonSVG->SetFont( wxFont( 8, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer37->Add( m_ButtonSVG, 0, wxALL, 5 );
	
	m_ButtonSavePdf = new wxButton( m_PanelStationName, wxID_ANY, _("Export as txt"), wxDefaultPosition, wxSize( 80,20 ), 0 );
	m_ButtonSavePdf->SetFont( wxFont( 8, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer37->Add( m_ButtonSavePdf, 0, wxALL, 5 );
	
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
	m_ButtonSVG->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportSVG ), NULL, this );
	m_ButtonSavePdf->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
	m_ButtonPreview->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPreview ), NULL, this );
	m_ButtonPrint->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnPrint ), NULL, this );
	m_CheckListToc->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
	m_CheckListPast->Connect( wxEVT_COMMAND_CHECKLISTBOX_TOGGLED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnTocSelectionChange ), NULL, this );
}

asFramePlotTimeSeriesVirtual::~asFramePlotTimeSeriesVirtual()
{
	// Disconnect Events
	m_ButtonSVG->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportSVG ), NULL, this );
	m_ButtonSavePdf->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePlotTimeSeriesVirtual::OnExportTXT ), NULL, this );
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

asPanelPlotVirtual::asPanelPlotVirtual( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	m_PlotCtrl = new wxPlotCtrl(this, wxID_ANY, wxDefaultPosition, wxDefaultSize );
	m_PlotCtrl->SetScrollOnThumbRelease( false );
	m_PlotCtrl->SetDrawSymbols( false );
	m_PlotCtrl->SetDrawLines( true );
	m_PlotCtrl->SetDrawSpline( false );
	m_PlotCtrl->SetDrawGrid( true );
	m_PlotCtrl->SetAreaMouseFunction( wxPLOTCTRL_MOUSE_PAN );
	m_PlotCtrl->SetAreaMouseMarker( wxPLOTCTRL_MARKER_RECT );
	m_PlotCtrl->SetCrossHairCursor( false );
	m_PlotCtrl->SetShowXAxis( true );
	m_PlotCtrl->SetShowXAxisLabel( true );
	m_PlotCtrl->SetXAxisLabel( _("X Axis") );
	m_PlotCtrl->SetShowYAxis( true );
	m_PlotCtrl->SetShowYAxisLabel( true );
	m_PlotCtrl->SetYAxisLabel( _("Y Axis") );
	m_PlotCtrl->SetShowPlotTitle( false );
	m_PlotCtrl->SetPlotTitle( _("Title") );
	m_PlotCtrl->SetShowKey( true );
	m_PlotCtrl->SetKeyPosition( wxPoint( 100,100 ) );
	
	bSizer26->Add( m_PlotCtrl, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	bSizer26->Fit( this );
}

asPanelPlotVirtual::~asPanelPlotVirtual()
{
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
