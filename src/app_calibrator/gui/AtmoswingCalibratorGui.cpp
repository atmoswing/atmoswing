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

#include "AtmoswingCalibratorGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameCalibrationVirtual::asFrameCalibrationVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,600 ), wxDefaultSize );
	
	wxBoxSizer* bSizer4;
	bSizer4 = new wxBoxSizer( wxVERTICAL );
	
	m_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	m_notebookBase = new wxNotebook( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_LEFT );
	m_panelControls = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxVERTICAL );
	
	m_staticTextMethod = new wxStaticText( m_panelControls, wxID_ANY, _("Select the calibration method"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextMethod->Wrap( -1 );
	bSizer5->Add( m_staticTextMethod, 0, wxALL, 5 );
	
	wxString m_choiceMethodChoices[] = { _("Single assessment"), _("Classic calibration"), _("Classic+ calibration"), _("Variables exploration Classic+"), _("Evaluate all scores"), _("Only predictand values") };
	int m_choiceMethodNChoices = sizeof( m_choiceMethodChoices ) / sizeof( wxString );
	m_choiceMethod = new wxChoice( m_panelControls, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceMethodNChoices, m_choiceMethodChoices, 0 );
	m_choiceMethod->SetSelection( 0 );
	bSizer5->Add( m_choiceMethod, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextFileParameters = new wxStaticText( m_panelControls, wxID_ANY, _("Select the parameters file for the calibration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextFileParameters->Wrap( -1 );
	bSizer5->Add( m_staticTextFileParameters, 0, wxALL, 5 );
	
	m_filePickerParameters = new wxFilePickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	bSizer5->Add( m_filePickerParameters, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextFilePredictand = new wxStaticText( m_panelControls, wxID_ANY, _("Select the predictand DB file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextFilePredictand->Wrap( -1 );
	bSizer5->Add( m_staticTextFilePredictand, 0, wxALL, 5 );
	
	m_filePickerPredictand = new wxFilePickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.nc"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_DEFAULT_STYLE );
	bSizer5->Add( m_filePickerPredictand, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextPredictorDir = new wxStaticText( m_panelControls, wxID_ANY, _("Select the predictors directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPredictorDir->Wrap( -1 );
	bSizer5->Add( m_staticTextPredictorDir, 0, wxALL, 5 );
	
	m_dirPickerPredictor = new wxDirPickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_dirPickerPredictor, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextCalibrationResultsDir = new wxStaticText( m_panelControls, wxID_ANY, _("Directory to save calibration outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextCalibrationResultsDir->Wrap( -1 );
	bSizer5->Add( m_staticTextCalibrationResultsDir, 0, wxALL, 5 );
	
	m_dirPickerCalibrationResults = new wxDirPickerCtrl( m_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_dirPickerCalibrationResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_checkBoxParallelEvaluations = new wxCheckBox( m_panelControls, wxID_ANY, _("Parallel evaluations when possible (competes with multithreading in the processor)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5->Add( m_checkBoxParallelEvaluations, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer131;
	bSizer131 = new wxBoxSizer( wxHORIZONTAL );
	
	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( m_panelControls, wxID_ANY, _("Intermediate results saving options") ), wxVERTICAL );
	
	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextSaveAnalogDates = new wxStaticText( m_panelControls, wxID_ANY, _("Analog dates steps:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSaveAnalogDates->Wrap( -1 );
	bSizer141->Add( m_staticTextSaveAnalogDates, 0, wxALL, 5 );
	
	m_checkBoxSaveAnalogDatesStep1 = new wxCheckBox( m_panelControls, wxID_ANY, _("1"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_checkBoxSaveAnalogDatesStep1, 0, wxALL, 5 );
	
	m_checkBoxSaveAnalogDatesStep2 = new wxCheckBox( m_panelControls, wxID_ANY, _("2"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_checkBoxSaveAnalogDatesStep2, 0, wxALL, 5 );
	
	m_checkBoxSaveAnalogDatesStep3 = new wxCheckBox( m_panelControls, wxID_ANY, _("3"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_checkBoxSaveAnalogDatesStep3, 0, wxALL, 5 );
	
	m_checkBoxSaveAnalogDatesStep4 = new wxCheckBox( m_panelControls, wxID_ANY, _("4"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_checkBoxSaveAnalogDatesStep4, 0, wxALL, 5 );
	
	
	sbSizer8->Add( bSizer141, 0, wxEXPAND, 5 );
	
	m_checkBoxSaveAnalogDatesAllSteps = new wxCheckBox( m_panelControls, wxID_ANY, _("All analog dates steps (overwrite previous)"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_checkBoxSaveAnalogDatesAllSteps, 0, wxALL, 5 );
	
	m_checkBoxSaveAnalogValues = new wxCheckBox( m_panelControls, wxID_ANY, _("Analog values"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_checkBoxSaveAnalogValues, 0, wxALL, 5 );
	
	m_checkBoxSaveForecastScores = new wxCheckBox( m_panelControls, wxID_ANY, _("Forecast scores"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_checkBoxSaveForecastScores, 0, wxALL, 5 );
	
	m_checkBoxSaveFinalForecastScore = new wxCheckBox( m_panelControls, wxID_ANY, _("Final forecast score"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_checkBoxSaveFinalForecastScore, 0, wxALL, 5 );
	
	m_staticText60 = new wxStaticText( m_panelControls, wxID_ANY, _("Options are always desactivated at initialization !"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText60->Wrap( -1 );
	sbSizer8->Add( m_staticText60, 0, wxALL, 5 );
	
	
	bSizer131->Add( sbSizer8, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer81;
	sbSizer81 = new wxStaticBoxSizer( new wxStaticBox( m_panelControls, wxID_ANY, _("Intermediate results loading options") ), wxVERTICAL );
	
	wxBoxSizer* bSizer1411;
	bSizer1411 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextLoadAnalogDates = new wxStaticText( m_panelControls, wxID_ANY, _("Analog dates steps:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextLoadAnalogDates->Wrap( -1 );
	bSizer1411->Add( m_staticTextLoadAnalogDates, 0, wxALL, 5 );
	
	m_checkBoxLoadAnalogDatesStep1 = new wxCheckBox( m_panelControls, wxID_ANY, _("1"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_checkBoxLoadAnalogDatesStep1, 0, wxALL, 5 );
	
	m_checkBoxLoadAnalogDatesStep2 = new wxCheckBox( m_panelControls, wxID_ANY, _("2"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_checkBoxLoadAnalogDatesStep2, 0, wxALL, 5 );
	
	m_checkBoxLoadAnalogDatesStep3 = new wxCheckBox( m_panelControls, wxID_ANY, _("3"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_checkBoxLoadAnalogDatesStep3, 0, wxALL, 5 );
	
	m_checkBoxLoadAnalogDatesStep4 = new wxCheckBox( m_panelControls, wxID_ANY, _("4"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_checkBoxLoadAnalogDatesStep4, 0, wxALL, 5 );
	
	
	sbSizer81->Add( bSizer1411, 0, wxEXPAND, 5 );
	
	m_checkBoxLoadAnalogDatesAllSteps = new wxCheckBox( m_panelControls, wxID_ANY, _("All analog dates steps (overwrite previous)"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer81->Add( m_checkBoxLoadAnalogDatesAllSteps, 0, wxALL, 5 );
	
	m_checkBoxLoadAnalogValues = new wxCheckBox( m_panelControls, wxID_ANY, _("Analog values"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer81->Add( m_checkBoxLoadAnalogValues, 0, wxALL, 5 );
	
	m_checkBoxLoadForecastScores = new wxCheckBox( m_panelControls, wxID_ANY, _("Forecast scores"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer81->Add( m_checkBoxLoadForecastScores, 0, wxALL, 5 );
	
	m_staticText61 = new wxStaticText( m_panelControls, wxID_ANY, _("Options are always desactivated at initialization !"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText61->Wrap( -1 );
	sbSizer81->Add( m_staticText61, 0, wxALL, 5 );
	
	
	bSizer131->Add( sbSizer81, 1, wxEXPAND|wxALL, 5 );
	
	
	bSizer5->Add( bSizer131, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer23;
	bSizer23 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextStateLabel = new wxStaticText( m_panelControls, wxID_ANY, _("Calibration state: "), wxDefaultPosition, wxDefaultSize, 0 );
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
	m_notebookBase->AddPage( m_panelControls, _("Controls"), true );
	m_panelOptions = new wxPanel( m_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );
	
	m_notebookOptions = new wxNotebook( m_panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_MULTILINE );
	m_panelSingle = new wxPanel( m_notebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer3;
	fgSizer3 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer3->SetFlexibleDirection( wxBOTH );
	fgSizer3->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	wxStaticBoxSizer* sbSizer10;
	sbSizer10 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("Classic+ calibration") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer21;
	fgSizer21 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer21->SetFlexibleDirection( wxBOTH );
	fgSizer21->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextClassicPlusStepsLonPertinenceMap = new wxStaticText( m_panelSingle, wxID_ANY, _("Multiple of the steps in lon for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextClassicPlusStepsLonPertinenceMap->Wrap( -1 );
	fgSizer21->Add( m_staticTextClassicPlusStepsLonPertinenceMap, 0, wxALL, 5 );
	
	m_textCtrlClassicPlusStepsLonPertinenceMap = new wxTextCtrl( m_panelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlClassicPlusStepsLonPertinenceMap->SetMaxLength( 0 ); 
	fgSizer21->Add( m_textCtrlClassicPlusStepsLonPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextClassicPlusStepsLatPertinenceMap = new wxStaticText( m_panelSingle, wxID_ANY, _("Multiple of the steps in lat for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextClassicPlusStepsLatPertinenceMap->Wrap( -1 );
	fgSizer21->Add( m_staticTextClassicPlusStepsLatPertinenceMap, 0, wxALL, 5 );
	
	m_textCtrlClassicPlusStepsLatPertinenceMap = new wxTextCtrl( m_panelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlClassicPlusStepsLatPertinenceMap->SetMaxLength( 0 ); 
	fgSizer21->Add( m_textCtrlClassicPlusStepsLatPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextClassicPlusResizingIterations = new wxStaticText( m_panelSingle, wxID_ANY, _("Iterations in final resizing attempts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextClassicPlusResizingIterations->Wrap( -1 );
	fgSizer21->Add( m_staticTextClassicPlusResizingIterations, 0, wxALL, 5 );
	
	m_textCtrlClassicPlusResizingIterations = new wxTextCtrl( m_panelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlClassicPlusResizingIterations->SetMaxLength( 0 ); 
	fgSizer21->Add( m_textCtrlClassicPlusResizingIterations, 0, wxRIGHT|wxLEFT, 5 );
	
	m_checkBoxProceedSequentially = new wxCheckBox( m_panelSingle, wxID_ANY, _("Proceed sequentially (standard)"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer21->Add( m_checkBoxProceedSequentially, 0, wxALL, 5 );
	
	m_staticTextSpacer = new wxStaticText( m_panelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSpacer->Wrap( -1 );
	fgSizer21->Add( m_staticTextSpacer, 0, wxALL, 5 );
	
	m_checkBoxClassicPlusResize = new wxCheckBox( m_panelSingle, wxID_ANY, _("Resize the spatial windows separately"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxClassicPlusResize->Enable( false );
	
	fgSizer21->Add( m_checkBoxClassicPlusResize, 0, wxALL, 5 );
	
	
	sbSizer10->Add( fgSizer21, 1, wxEXPAND, 5 );
	
	
	fgSizer3->Add( sbSizer10, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer9;
	sbSizer9 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("No option for") ), wxVERTICAL );
	
	m_staticText66 = new wxStaticText( m_panelSingle, wxID_ANY, _("Single assessment"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText66->Wrap( -1 );
	sbSizer9->Add( m_staticText66, 0, wxALL, 5 );
	
	m_staticText67 = new wxStaticText( m_panelSingle, wxID_ANY, _("Classic calibration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText67->Wrap( -1 );
	sbSizer9->Add( m_staticText67, 0, wxALL, 5 );
	
	
	fgSizer3->Add( sbSizer9, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer12;
	sbSizer12 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("Monte-Carlo") ), wxVERTICAL );
	
	m_staticTextMonteCarloRandomNb = new wxStaticText( m_panelSingle, wxID_ANY, _("Number of random param. sets"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextMonteCarloRandomNb->Wrap( -1 );
	sbSizer12->Add( m_staticTextMonteCarloRandomNb, 0, wxALL, 5 );
	
	m_textCtrlMonteCarloRandomNb = new wxTextCtrl( m_panelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlMonteCarloRandomNb->SetMaxLength( 0 ); 
	sbSizer12->Add( m_textCtrlMonteCarloRandomNb, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	fgSizer3->Add( sbSizer12, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer91;
	sbSizer91 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("Variables exploration") ), wxVERTICAL );
	
	m_staticTextVarExploStepToExplore = new wxStaticText( m_panelSingle, wxID_ANY, _("Step to explore (0-based)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextVarExploStepToExplore->Wrap( -1 );
	sbSizer91->Add( m_staticTextVarExploStepToExplore, 0, wxALL, 5 );
	
	m_textCtrlVarExploStepToExplore = new wxTextCtrl( m_panelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlVarExploStepToExplore->SetMaxLength( 0 ); 
	sbSizer91->Add( m_textCtrlVarExploStepToExplore, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	fgSizer3->Add( sbSizer91, 1, wxEXPAND|wxALL, 5 );
	
	
	m_panelSingle->SetSizer( fgSizer3 );
	m_panelSingle->Layout();
	fgSizer3->Fit( m_panelSingle );
	m_notebookOptions->AddPage( m_panelSingle, _("Calibration"), true );
	
	bSizer28->Add( m_notebookOptions, 1, wxEXPAND | wxALL, 5 );
	
	
	m_panelOptions->SetSizer( bSizer28 );
	m_panelOptions->Layout();
	bSizer28->Fit( m_panelOptions );
	m_notebookBase->AddPage( m_panelOptions, _("Options"), false );
	
	bSizer29->Add( m_notebookBase, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );
	
	m_buttonSaveDefault = new wxButton( m_panelMain, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( m_buttonSaveDefault, 0, wxALIGN_RIGHT, 5 );
	
	
	bSizer29->Add( bSizer15, 0, wxALIGN_RIGHT|wxTOP|wxBOTTOM|wxRIGHT, 5 );
	
	
	m_panelMain->SetSizer( bSizer29 );
	m_panelMain->Layout();
	bSizer29->Fit( m_panelMain );
	bSizer4->Add( m_panelMain, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer4 );
	this->Layout();
	bSizer4->Fit( this );
	m_menuBar = new wxMenuBar( 0 );
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
	m_toolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_toolBar->Realize(); 
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_buttonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameCalibrationVirtual::OnSaveDefault ), NULL, this );
	this->Connect( m_menuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFramePreferences ) );
	this->Connect( m_menuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnShowLog ) );
	this->Connect( m_menuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel1 ) );
	this->Connect( m_menuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel2 ) );
	this->Connect( m_menuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel3 ) );
	this->Connect( m_menuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFrameAbout ) );
}

asFrameCalibrationVirtual::~asFrameCalibrationVirtual()
{
	// Disconnect Events
	m_buttonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameCalibrationVirtual::OnSaveDefault ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFramePreferences ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnShowLog ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel1 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel2 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel3 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFrameAbout ) );
	
}

asFramePreferencesCalibratorVirtual::asFramePreferencesCalibratorVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
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
	
	wxString m_radioBoxLogLevelChoices[] = { _("Errors only (recommanded)"), _("Errors and warnings"), _("Verbose") };
	int m_radioBoxLogLevelNChoices = sizeof( m_radioBoxLogLevelChoices ) / sizeof( wxString );
	m_radioBoxLogLevel = new wxRadioBox( m_panelGeneralCommon, wxID_ANY, _("Level"), wxDefaultPosition, wxDefaultSize, m_radioBoxLogLevelNChoices, m_radioBoxLogLevelChoices, 1, wxRA_SPECIFY_COLS );
	m_radioBoxLogLevel->SetSelection( 0 );
	bSizer20->Add( m_radioBoxLogLevel, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Outputs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );
	
	m_checkBoxDisplayLogWindow = new wxCheckBox( m_panelGeneralCommon, wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxDisplayLogWindow->SetValue(true); 
	bSizer21->Add( m_checkBoxDisplayLogWindow, 0, wxALL, 5 );
	
	m_checkBoxSaveLogFile = new wxCheckBox( m_panelGeneralCommon, wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxSaveLogFile->SetValue(true); 
	m_checkBoxSaveLogFile->Enable( false );
	
	bSizer21->Add( m_checkBoxSaveLogFile, 0, wxALL, 5 );
	
	
	sbSizer8->Add( bSizer21, 1, wxEXPAND, 5 );
	
	
	bSizer20->Add( sbSizer8, 1, wxALL|wxEXPAND, 5 );
	
	
	sbSizer7->Add( bSizer20, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer7, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Directories") ), wxVERTICAL );
	
	m_staticTextArchivePredictorsDir = new wxStaticText( m_panelGeneralCommon, wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerArchivePredictors = new wxDirPickerCtrl( m_panelGeneralCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_staticTextPredictandDBDir = new wxStaticText( m_panelGeneralCommon, wxID_ANY, _("Default predictand DB directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerPredictandDB = new wxDirPickerCtrl( m_panelGeneralCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
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
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneral, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_checkBoxResponsiveness = new wxCheckBox( m_panelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxResponsiveness->SetValue(true); 
	sbSizer151->Add( m_checkBoxResponsiveness, 0, wxALL, 5 );
	
	
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
	
	m_checkBoxAllowMultithreading = new wxCheckBox( m_panelProcessing, wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxAllowMultithreading->SetValue(true); 
	sbSizer15->Add( m_checkBoxAllowMultithreading, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextThreadsNb = new wxStaticText( m_panelProcessing, wxID_ANY, _("Max nb of threads"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsNb->Wrap( -1 );
	bSizer221->Add( m_staticTextThreadsNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_textCtrlThreadsNb = new wxTextCtrl( m_panelProcessing, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_textCtrlThreadsNb->SetMaxLength( 0 ); 
	bSizer221->Add( m_textCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	sbSizer15->Add( bSizer221, 0, wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextThreadsPriority = new wxStaticText( m_panelProcessing, wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( m_staticTextThreadsPriority, 0, wxALL, 5 );
	
	m_sliderThreadsPriority = new wxSlider( m_panelProcessing, wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( m_sliderThreadsPriority, 1, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer15->Add( bSizer241, 0, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_radioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Date array insertions (slower)"), _("Date array splitting (slower)") };
	int m_radioBoxProcessingMethodsNChoices = sizeof( m_radioBoxProcessingMethodsChoices ) / sizeof( wxString );
	m_radioBoxProcessingMethods = new wxRadioBox( m_panelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition, wxDefaultSize, m_radioBoxProcessingMethodsNChoices, m_radioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS );
	m_radioBoxProcessingMethods->SetSelection( 0 );
	m_radioBoxProcessingMethods->SetToolTip( _("These options don't affect the results, only the processor efficiency.") );
	
	bSizer1611->Add( m_radioBoxProcessingMethods, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_radioBoxLinearAlgebraChoices[] = { _("Direct access to the coefficients"), _("Direct access to the coefficients and minimizing variable declarations"), _("Linear algebra using Eigen"), _("Linear algebra using Eigen and minimizing variable declarations (recommended)") };
	int m_radioBoxLinearAlgebraNChoices = sizeof( m_radioBoxLinearAlgebraChoices ) / sizeof( wxString );
	m_radioBoxLinearAlgebra = new wxRadioBox( m_panelProcessing, wxID_ANY, _("Linear algebra options"), wxDefaultPosition, wxDefaultSize, m_radioBoxLinearAlgebraNChoices, m_radioBoxLinearAlgebraChoices, 1, wxRA_SPECIFY_COLS );
	m_radioBoxLinearAlgebra->SetSelection( 3 );
	m_radioBoxLinearAlgebra->Enable( false );
	
	bSizer1611->Add( m_radioBoxLinearAlgebra, 0, wxALL|wxEXPAND, 5 );
	
	
	m_panelProcessing->SetSizer( bSizer1611 );
	m_panelProcessing->Layout();
	bSizer1611->Fit( m_panelProcessing );
	m_notebookAdvanced->AddPage( m_panelProcessing, _("Processing"), false );
	m_panelUserDirectories = new wxPanel( m_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer24;
	bSizer24 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer411;
	sbSizer411 = new wxStaticBoxSizer( new wxStaticBox( m_panelUserDirectories, wxID_ANY, _("Working directories") ), wxVERTICAL );
	
	m_staticTextIntermediateResultsDir = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("Directory to save intermediate temporary results"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextIntermediateResultsDir->Wrap( -1 );
	sbSizer411->Add( m_staticTextIntermediateResultsDir, 0, wxALL, 5 );
	
	m_dirPickerIntermediateResults = new wxDirPickerCtrl( m_panelUserDirectories, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer411->Add( m_dirPickerIntermediateResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer24->Add( sbSizer411, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( m_panelUserDirectories, wxID_ANY, _("User specific paths") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer9;
	fgSizer9 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer9->SetFlexibleDirection( wxBOTH );
	fgSizer9->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextUserDirLabel = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("User working directory:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextUserDirLabel->Wrap( -1 );
	fgSizer9->Add( m_staticTextUserDirLabel, 0, wxALL, 5 );
	
	m_staticTextUserDir = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextUserDir->Wrap( -1 );
	fgSizer9->Add( m_staticTextUserDir, 0, wxALL, 5 );
	
	m_staticTextLogFileLabel = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextLogFileLabel->Wrap( -1 );
	fgSizer9->Add( m_staticTextLogFileLabel, 0, wxALL, 5 );
	
	m_staticTextLogFile = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextLogFile->Wrap( -1 );
	fgSizer9->Add( m_staticTextLogFile, 0, wxALL, 5 );
	
	m_staticTextPrefFileLabel = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("Preferences file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPrefFileLabel->Wrap( -1 );
	fgSizer9->Add( m_staticTextPrefFileLabel, 0, wxALL, 5 );
	
	m_staticTextPrefFile = new wxStaticText( m_panelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
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
	
	bSizer15->Add( m_buttonsConfirmation, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelBase->SetSizer( bSizer15 );
	m_panelBase->Layout();
	bSizer15->Fit( m_panelBase );
	bSizer14->Add( m_panelBase, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer14 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_checkBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesCalibratorVirtual::~asFramePreferencesCalibratorVirtual()
{
	// Disconnect Events
	m_checkBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::SaveAndClose ), NULL, this );
	
}
