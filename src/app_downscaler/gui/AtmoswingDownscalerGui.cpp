///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct 26 2018)
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

#include "AtmoswingDownscalerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameDownscalerVirtual::asFrameDownscalerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,500 ), wxDefaultSize );

	wxBoxSizer* bSizer4;
	bSizer4 = new wxBoxSizer( wxVERTICAL );

	m_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );

	m_panelControls = new wxPanel( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxVERTICAL );

	m_staticTextMethod = new wxStaticText( m_panelControls, wxID_ANY, _("Select the downscaling method"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextMethod->Wrap( -1 );
	m_staticTextMethod->Hide();

	bSizer5->Add( m_staticTextMethod, 0, wxALL, 5 );

	wxString m_choiceMethodChoices[] = { _("Classic downscaling") };
	int m_choiceMethodNChoices = sizeof( m_choiceMethodChoices ) / sizeof( wxString );
	m_choiceMethod = new wxChoice( m_panelControls, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceMethodNChoices, m_choiceMethodChoices, 0 );
	m_choiceMethod->SetSelection( 0 );
	m_choiceMethod->Hide();

	bSizer5->Add( m_choiceMethod, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextFileParameters = new wxStaticText( m_panelControls, wxID_ANY, _("Select the parameters file for the downscaling"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextFileParameters->Wrap( -1 );
	bSizer5->Add( m_staticTextFileParameters, 0, wxALL, 5 );

	m_filePickerParameters = new wxFilePickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a file"), _("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	bSizer5->Add( m_filePickerParameters, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextFilePredictand = new wxStaticText( m_panelControls, wxID_ANY, _("Select the predictand DB file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextFilePredictand->Wrap( -1 );
	bSizer5->Add( m_staticTextFilePredictand, 0, wxALL, 5 );

	m_filePickerPredictand = new wxFilePickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a file"), _("*.nc"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_DEFAULT_STYLE );
	bSizer5->Add( m_filePickerPredictand, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextArchivePredictorDir = new wxStaticText( m_panelControls, wxID_ANY, _("Select the archive predictors directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextArchivePredictorDir->Wrap( -1 );
	bSizer5->Add( m_staticTextArchivePredictorDir, 0, wxALL, 5 );

	m_dirPickerArchivePredictor = new wxDirPickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_dirPickerArchivePredictor, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextScenarioPredictorDir = new wxStaticText( m_panelControls, wxID_ANY, _("Select the predictors directory for the target period"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextScenarioPredictorDir->Wrap( -1 );
	bSizer5->Add( m_staticTextScenarioPredictorDir, 0, wxALL, 5 );

	m_dirPickerScenarioPredictor = new wxDirPickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_dirPickerScenarioPredictor, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextDownscalingResultsDir = new wxStaticText( m_panelControls, wxID_ANY, _("Directory to save downscaling outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDownscalingResultsDir->Wrap( -1 );
	bSizer5->Add( m_staticTextDownscalingResultsDir, 0, wxALL, 5 );

	m_dirPickerDownscalingResults = new wxDirPickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_dirPickerDownscalingResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_checkBoxParallelEvaluations = new wxCheckBox( m_panelControls, wxID_ANY, _("Parallel evaluations when possible (competes with multithreading in the processor)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5->Add( m_checkBoxParallelEvaluations, 0, wxALL, 5 );

	wxBoxSizer* bSizer23;
	bSizer23 = new wxBoxSizer( wxHORIZONTAL );

	m_staticTextStateLabel = new wxStaticText( m_panelControls, wxID_ANY, _("Downscaling state: "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextStateLabel->Wrap( -1 );
	m_staticTextStateLabel->Hide();

	bSizer23->Add( m_staticTextStateLabel, 0, wxALL, 5 );

	m_staticTextState = new wxStaticText( m_panelControls, wxID_ANY, _("Not running."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextState->Wrap( 350 );
	m_staticTextState->Hide();

	bSizer23->Add( m_staticTextState, 1, wxALL, 5 );


	bSizer5->Add( bSizer23, 0, wxEXPAND, 5 );


	m_panelControls->SetSizer( bSizer5 );
	m_panelControls->Layout();
	bSizer5->Fit( m_panelControls );
	bSizer29->Add( m_panelControls, 1, wxEXPAND | wxALL, 5 );

	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );

	m_buttonSaveDefault = new wxButton( m_panelMain, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( m_buttonSaveDefault, 0, 0, 5 );


	bSizer29->Add( bSizer15, 0, wxALIGN_RIGHT|wxTOP|wxBOTTOM|wxRIGHT, 5 );


	m_panelMain->SetSizer( bSizer29 );
	m_panelMain->Layout();
	bSizer29->Fit( m_panelMain );
	bSizer4->Add( m_panelMain, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer4 );
	this->Layout();
	m_menuBar = new wxMenuBar( 0 );
	m_menuOptions = new wxMenu();
	wxMenuItem* m_menuItemPreferences;
	m_menuItemPreferences = new wxMenuItem( m_menuOptions, wxID_ANY, wxString( _("Preferences") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuOptions->Append( m_menuItemPreferences );

	m_menuBar->Append( m_menuOptions, _("Options") );

	m_menuTools = new wxMenu();
	wxMenuItem* menuItemBuildPredictandDB;
	menuItemBuildPredictandDB = new wxMenuItem( m_menuTools, wxID_ANY, wxString( _("Build predictand DB") ) , wxEmptyString, wxITEM_NORMAL );
	m_menuTools->Append( menuItemBuildPredictandDB );

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

	m_toolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	m_toolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_toolBar->Realize();

	m_statusBar1 = this->CreateStatusBar( 1, wxSTB_SIZEGRIP, wxID_ANY );

	this->Centre( wxBOTH );

	// Connect Events
	m_buttonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameDownscalerVirtual::OnSaveDefault ), NULL, this );
	m_menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OpenFramePreferences ), this, m_menuItemPreferences->GetId());
	m_menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OpenFramePredictandDB ), this, menuItemBuildPredictandDB->GetId());
	m_menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OnShowLog ), this, m_menuItemShowLog->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OnLogLevel1 ), this, m_MenuItemLogLevel1->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OnLogLevel2 ), this, m_MenuItemLogLevel2->GetId());
	m_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OnLogLevel3 ), this, m_MenuItemLogLevel3->GetId());
	m_menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameDownscalerVirtual::OpenFrameAbout ), this, m_menuItemAbout->GetId());
}

asFrameDownscalerVirtual::~asFrameDownscalerVirtual()
{
	// Disconnect Events
	m_buttonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameDownscalerVirtual::OnSaveDefault ), NULL, this );

}

asFramePreferencesDownscalerVirtual::asFramePreferencesDownscalerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );

	m_panelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );

	m_notebookBase = new wxNotebook( m_panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelGeneralCommon = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer16;
	bSizer16 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer17;
	bSizer17 = new wxBoxSizer( wxVERTICAL );

	m_radioBtnLogLevel1 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer17->Add( m_radioBtnLogLevel1, 0, wxALL, 5 );

	m_radioBtnLogLevel2 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer17->Add( m_radioBtnLogLevel2, 0, wxALL, 5 );

	m_radioBtnLogLevel3 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer17->Add( m_radioBtnLogLevel3, 0, wxALL, 5 );


	bSizer20->Add( bSizer17, 1, wxEXPAND, 5 );

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

	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Directories") ), wxVERTICAL );

	m_staticTextArchivePredictorsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );

	m_dirPickerArchivePredictors = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	m_staticTextScenarioPredictorsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing scenario predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextScenarioPredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextScenarioPredictorsDir, 0, wxRIGHT|wxLEFT, 5 );

	m_dirPickerScenarioPredictors = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerScenarioPredictors, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	m_staticTextPredictandDBDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Default predictand DB directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );

	m_dirPickerPredictandDB = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	bSizer16->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );


	m_panelGeneralCommon->SetSizer( bSizer16 );
	m_panelGeneralCommon->Layout();
	bSizer16->Fit( m_panelGeneralCommon );
	m_notebookBase->AddPage( m_panelGeneralCommon, _("General"), true );
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

	m_checkBoxResponsiveness = new wxCheckBox( m_panelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxResponsiveness->SetValue(true);
	bSizer271->Add( m_checkBoxResponsiveness, 0, wxALL, 5 );


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
	bSizer241->Add( m_sliderThreadsPriority, 1, wxRIGHT|wxLEFT, 5 );


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

	wxStaticBoxSizer* sbSizer411;
	sbSizer411 = new wxStaticBoxSizer( new wxStaticBox( m_panelUserDirectories, wxID_ANY, _("Working directories") ), wxVERTICAL );

	m_staticTextIntermediateResultsDir = new wxStaticText( sbSizer411->GetStaticBox(), wxID_ANY, _("Directory to save intermediate temporary results"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextIntermediateResultsDir->Wrap( -1 );
	sbSizer411->Add( m_staticTextIntermediateResultsDir, 0, wxALL, 5 );

	m_dirPickerIntermediateResults = new wxDirPickerCtrl( sbSizer411->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer411->Add( m_dirPickerIntermediateResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	bSizer24->Add( sbSizer411, 0, wxEXPAND|wxALL, 5 );

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

	m_staticTextLogFileLabels = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextLogFileLabels->Wrap( -1 );
	fgSizer9->Add( m_staticTextLogFileLabels, 0, wxALL, 5 );

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

	bSizer15->Add( m_buttonsConfirmation, 0, wxALL|wxEXPAND, 5 );


	m_panelBase->SetSizer( bSizer15 );
	m_panelBase->Layout();
	bSizer15->Fit( m_panelBase );
	bSizer14->Add( m_panelBase, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer14 );
	this->Layout();

	this->Centre( wxBOTH );

	// Connect Events
	m_checkBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesDownscalerVirtual::~asFramePreferencesDownscalerVirtual()
{
	// Disconnect Events
	m_checkBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesDownscalerVirtual::SaveAndClose ), NULL, this );

}
