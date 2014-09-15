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
	
	m_PanelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookBase = new wxNotebook( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_LEFT );
	m_PanelControls = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxVERTICAL );
	
	m_StaticTextMethod = new wxStaticText( m_PanelControls, wxID_ANY, _("Select the calibration method"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextMethod->Wrap( -1 );
	bSizer5->Add( m_StaticTextMethod, 0, wxALL, 5 );
	
	wxString m_ChoiceMethodChoices[] = { _("Single assessment"), _("Classic calibration"), _("Classic+ calibration"), _("Variables exploration Classic+"), _("Evaluate all scores"), _("Only predictand values") };
	int m_ChoiceMethodNChoices = sizeof( m_ChoiceMethodChoices ) / sizeof( wxString );
	m_ChoiceMethod = new wxChoice( m_PanelControls, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceMethodNChoices, m_ChoiceMethodChoices, 0 );
	m_ChoiceMethod->SetSelection( 0 );
	bSizer5->Add( m_ChoiceMethod, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextFileParameters = new wxStaticText( m_PanelControls, wxID_ANY, _("Select the parameters file for the calibration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextFileParameters->Wrap( -1 );
	bSizer5->Add( m_StaticTextFileParameters, 0, wxALL, 5 );
	
	m_FilePickerParameters = new wxFilePickerCtrl( m_PanelControls, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	bSizer5->Add( m_FilePickerParameters, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextFilePredictand = new wxStaticText( m_PanelControls, wxID_ANY, _("Select the predictand DB file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextFilePredictand->Wrap( -1 );
	bSizer5->Add( m_StaticTextFilePredictand, 0, wxALL, 5 );
	
	m_FilePickerPredictand = new wxFilePickerCtrl( m_PanelControls, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.nc"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_DEFAULT_STYLE );
	bSizer5->Add( m_FilePickerPredictand, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextPredictorDir = new wxStaticText( m_PanelControls, wxID_ANY, _("Select the predictors directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPredictorDir->Wrap( -1 );
	bSizer5->Add( m_StaticTextPredictorDir, 0, wxALL, 5 );
	
	m_DirPickerPredictor = new wxDirPickerCtrl( m_PanelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_DirPickerPredictor, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextCalibrationResultsDir = new wxStaticText( m_PanelControls, wxID_ANY, _("Directory to save calibration outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextCalibrationResultsDir->Wrap( -1 );
	bSizer5->Add( m_StaticTextCalibrationResultsDir, 0, wxALL, 5 );
	
	m_DirPickerCalibrationResults = new wxDirPickerCtrl( m_PanelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( m_DirPickerCalibrationResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_CheckBoxParallelEvaluations = new wxCheckBox( m_PanelControls, wxID_ANY, _("Parallel evaluations when possible (competes with multithreading in the processor)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer5->Add( m_CheckBoxParallelEvaluations, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer131;
	bSizer131 = new wxBoxSizer( wxHORIZONTAL );
	
	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( m_PanelControls, wxID_ANY, _("Intermediate results saving options") ), wxVERTICAL );
	
	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextSaveAnalogDates = new wxStaticText( m_PanelControls, wxID_ANY, _("Analog dates steps:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextSaveAnalogDates->Wrap( -1 );
	bSizer141->Add( m_StaticTextSaveAnalogDates, 0, wxALL, 5 );
	
	m_CheckBoxSaveAnalogDatesStep1 = new wxCheckBox( m_PanelControls, wxID_ANY, _("1"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_CheckBoxSaveAnalogDatesStep1, 0, wxALL, 5 );
	
	m_CheckBoxSaveAnalogDatesStep2 = new wxCheckBox( m_PanelControls, wxID_ANY, _("2"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_CheckBoxSaveAnalogDatesStep2, 0, wxALL, 5 );
	
	m_CheckBoxSaveAnalogDatesStep3 = new wxCheckBox( m_PanelControls, wxID_ANY, _("3"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_CheckBoxSaveAnalogDatesStep3, 0, wxALL, 5 );
	
	m_CheckBoxSaveAnalogDatesStep4 = new wxCheckBox( m_PanelControls, wxID_ANY, _("4"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer141->Add( m_CheckBoxSaveAnalogDatesStep4, 0, wxALL, 5 );
	
	
	sbSizer8->Add( bSizer141, 0, wxEXPAND, 5 );
	
	m_CheckBoxSaveAnalogDatesAllSteps = new wxCheckBox( m_PanelControls, wxID_ANY, _("All analog dates steps (overwrite previous)"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_CheckBoxSaveAnalogDatesAllSteps, 0, wxALL, 5 );
	
	m_CheckBoxSaveAnalogValues = new wxCheckBox( m_PanelControls, wxID_ANY, _("Analog values"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_CheckBoxSaveAnalogValues, 0, wxALL, 5 );
	
	m_CheckBoxSaveForecastScores = new wxCheckBox( m_PanelControls, wxID_ANY, _("Forecast scores"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_CheckBoxSaveForecastScores, 0, wxALL, 5 );
	
	m_CheckBoxSaveFinalForecastScore = new wxCheckBox( m_PanelControls, wxID_ANY, _("Final forecast score"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer8->Add( m_CheckBoxSaveFinalForecastScore, 0, wxALL, 5 );
	
	m_staticText60 = new wxStaticText( m_PanelControls, wxID_ANY, _("Options are always desactivated at initialization !"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText60->Wrap( -1 );
	sbSizer8->Add( m_staticText60, 0, wxALL, 5 );
	
	
	bSizer131->Add( sbSizer8, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer81;
	sbSizer81 = new wxStaticBoxSizer( new wxStaticBox( m_PanelControls, wxID_ANY, _("Intermediate results loading options") ), wxVERTICAL );
	
	wxBoxSizer* bSizer1411;
	bSizer1411 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextLoadAnalogDates = new wxStaticText( m_PanelControls, wxID_ANY, _("Analog dates steps:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLoadAnalogDates->Wrap( -1 );
	bSizer1411->Add( m_StaticTextLoadAnalogDates, 0, wxALL, 5 );
	
	m_CheckBoxLoadAnalogDatesStep1 = new wxCheckBox( m_PanelControls, wxID_ANY, _("1"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_CheckBoxLoadAnalogDatesStep1, 0, wxALL, 5 );
	
	m_CheckBoxLoadAnalogDatesStep2 = new wxCheckBox( m_PanelControls, wxID_ANY, _("2"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_CheckBoxLoadAnalogDatesStep2, 0, wxALL, 5 );
	
	m_CheckBoxLoadAnalogDatesStep3 = new wxCheckBox( m_PanelControls, wxID_ANY, _("3"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_CheckBoxLoadAnalogDatesStep3, 0, wxALL, 5 );
	
	m_CheckBoxLoadAnalogDatesStep4 = new wxCheckBox( m_PanelControls, wxID_ANY, _("4"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer1411->Add( m_CheckBoxLoadAnalogDatesStep4, 0, wxALL, 5 );
	
	
	sbSizer81->Add( bSizer1411, 0, wxEXPAND, 5 );
	
	m_CheckBoxLoadAnalogDatesAllSteps = new wxCheckBox( m_PanelControls, wxID_ANY, _("All analog dates steps (overwrite previous)"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer81->Add( m_CheckBoxLoadAnalogDatesAllSteps, 0, wxALL, 5 );
	
	m_CheckBoxLoadAnalogValues = new wxCheckBox( m_PanelControls, wxID_ANY, _("Analog values"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer81->Add( m_CheckBoxLoadAnalogValues, 0, wxALL, 5 );
	
	m_CheckBoxLoadForecastScores = new wxCheckBox( m_PanelControls, wxID_ANY, _("Forecast scores"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer81->Add( m_CheckBoxLoadForecastScores, 0, wxALL, 5 );
	
	m_staticText61 = new wxStaticText( m_PanelControls, wxID_ANY, _("Options are always desactivated at initialization !"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText61->Wrap( -1 );
	sbSizer81->Add( m_staticText61, 0, wxALL, 5 );
	
	
	bSizer131->Add( sbSizer81, 1, wxEXPAND|wxALL, 5 );
	
	
	bSizer5->Add( bSizer131, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer23;
	bSizer23 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextStateLabel = new wxStaticText( m_PanelControls, wxID_ANY, _("Calibration state: "), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextStateLabel->Wrap( -1 );
	m_StaticTextStateLabel->Hide();
	
	bSizer23->Add( m_StaticTextStateLabel, 0, wxALL, 5 );
	
	m_StaticTextState = new wxStaticText( m_PanelControls, wxID_ANY, _("Not running."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextState->Wrap( 350 );
	m_StaticTextState->Hide();
	
	bSizer23->Add( m_StaticTextState, 1, wxALL, 5 );
	
	
	bSizer5->Add( bSizer23, 0, wxEXPAND, 5 );
	
	
	m_PanelControls->SetSizer( bSizer5 );
	m_PanelControls->Layout();
	bSizer5->Fit( m_PanelControls );
	m_NotebookBase->AddPage( m_PanelControls, _("Controls"), true );
	m_PanelOptions = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookOptions = new wxNotebook( m_PanelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_MULTILINE );
	m_PanelSingle = new wxPanel( m_NotebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer3;
	fgSizer3 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer3->SetFlexibleDirection( wxBOTH );
	fgSizer3->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	wxStaticBoxSizer* sbSizer10;
	sbSizer10 = new wxStaticBoxSizer( new wxStaticBox( m_PanelSingle, wxID_ANY, _("Classic+ calibration") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer21;
	fgSizer21 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer21->SetFlexibleDirection( wxBOTH );
	fgSizer21->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextClassicPlusStepsLonPertinenceMap = new wxStaticText( m_PanelSingle, wxID_ANY, _("Multiple of the steps in lon for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextClassicPlusStepsLonPertinenceMap->Wrap( -1 );
	fgSizer21->Add( m_StaticTextClassicPlusStepsLonPertinenceMap, 0, wxALL, 5 );
	
	m_TextCtrlClassicPlusStepsLonPertinenceMap = new wxTextCtrl( m_PanelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlClassicPlusStepsLonPertinenceMap->SetMaxLength( 0 ); 
	fgSizer21->Add( m_TextCtrlClassicPlusStepsLonPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextClassicPlusStepsLatPertinenceMap = new wxStaticText( m_PanelSingle, wxID_ANY, _("Multiple of the steps in lat for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextClassicPlusStepsLatPertinenceMap->Wrap( -1 );
	fgSizer21->Add( m_StaticTextClassicPlusStepsLatPertinenceMap, 0, wxALL, 5 );
	
	m_TextCtrlClassicPlusStepsLatPertinenceMap = new wxTextCtrl( m_PanelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlClassicPlusStepsLatPertinenceMap->SetMaxLength( 0 ); 
	fgSizer21->Add( m_TextCtrlClassicPlusStepsLatPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextClassicPlusResizingIterations = new wxStaticText( m_PanelSingle, wxID_ANY, _("Iterations in final resizing attempts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextClassicPlusResizingIterations->Wrap( -1 );
	fgSizer21->Add( m_StaticTextClassicPlusResizingIterations, 0, wxALL, 5 );
	
	m_TextCtrlClassicPlusResizingIterations = new wxTextCtrl( m_PanelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlClassicPlusResizingIterations->SetMaxLength( 0 ); 
	fgSizer21->Add( m_TextCtrlClassicPlusResizingIterations, 0, wxRIGHT|wxLEFT, 5 );
	
	m_CheckBoxProceedSequentially = new wxCheckBox( m_PanelSingle, wxID_ANY, _("Proceed sequentially (standard)"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer21->Add( m_CheckBoxProceedSequentially, 0, wxALL, 5 );
	
	m_StaticTextSpacer = new wxStaticText( m_PanelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextSpacer->Wrap( -1 );
	fgSizer21->Add( m_StaticTextSpacer, 0, wxALL, 5 );
	
	m_CheckBoxClassicPlusResize = new wxCheckBox( m_PanelSingle, wxID_ANY, _("Resize the spatial windows separately"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxClassicPlusResize->Enable( false );
	
	fgSizer21->Add( m_CheckBoxClassicPlusResize, 0, wxALL, 5 );
	
	
	sbSizer10->Add( fgSizer21, 1, wxEXPAND, 5 );
	
	
	fgSizer3->Add( sbSizer10, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer9;
	sbSizer9 = new wxStaticBoxSizer( new wxStaticBox( m_PanelSingle, wxID_ANY, _("No option for") ), wxVERTICAL );
	
	m_staticText66 = new wxStaticText( m_PanelSingle, wxID_ANY, _("Single assessment"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText66->Wrap( -1 );
	sbSizer9->Add( m_staticText66, 0, wxALL, 5 );
	
	m_staticText67 = new wxStaticText( m_PanelSingle, wxID_ANY, _("Classic calibration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText67->Wrap( -1 );
	sbSizer9->Add( m_staticText67, 0, wxALL, 5 );
	
	
	fgSizer3->Add( sbSizer9, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer12;
	sbSizer12 = new wxStaticBoxSizer( new wxStaticBox( m_PanelSingle, wxID_ANY, _("Monte-Carlo") ), wxVERTICAL );
	
	m_StaticTextMonteCarloRandomNb = new wxStaticText( m_PanelSingle, wxID_ANY, _("Number of random param. sets"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextMonteCarloRandomNb->Wrap( -1 );
	sbSizer12->Add( m_StaticTextMonteCarloRandomNb, 0, wxALL, 5 );
	
	m_TextCtrlMonteCarloRandomNb = new wxTextCtrl( m_PanelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlMonteCarloRandomNb->SetMaxLength( 0 ); 
	sbSizer12->Add( m_TextCtrlMonteCarloRandomNb, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	fgSizer3->Add( sbSizer12, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer91;
	sbSizer91 = new wxStaticBoxSizer( new wxStaticBox( m_PanelSingle, wxID_ANY, _("Variables exploration") ), wxVERTICAL );
	
	m_StaticTextVarExploStepToExplore = new wxStaticText( m_PanelSingle, wxID_ANY, _("Step to explore (0-based)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextVarExploStepToExplore->Wrap( -1 );
	sbSizer91->Add( m_StaticTextVarExploStepToExplore, 0, wxALL, 5 );
	
	m_TextCtrlVarExploStepToExplore = new wxTextCtrl( m_PanelSingle, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlVarExploStepToExplore->SetMaxLength( 0 ); 
	sbSizer91->Add( m_TextCtrlVarExploStepToExplore, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	fgSizer3->Add( sbSizer91, 1, wxEXPAND|wxALL, 5 );
	
	
	m_PanelSingle->SetSizer( fgSizer3 );
	m_PanelSingle->Layout();
	fgSizer3->Fit( m_PanelSingle );
	m_NotebookOptions->AddPage( m_PanelSingle, _("Calibration"), true );
	
	bSizer28->Add( m_NotebookOptions, 1, wxEXPAND | wxALL, 5 );
	
	
	m_PanelOptions->SetSizer( bSizer28 );
	m_PanelOptions->Layout();
	bSizer28->Fit( m_PanelOptions );
	m_NotebookBase->AddPage( m_PanelOptions, _("Options"), false );
	
	bSizer29->Add( m_NotebookBase, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );
	
	m_ButtonSaveDefault = new wxButton( m_PanelMain, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( m_ButtonSaveDefault, 0, wxALIGN_RIGHT, 5 );
	
	
	bSizer29->Add( bSizer15, 0, wxALIGN_RIGHT|wxTOP|wxBOTTOM|wxRIGHT, 5 );
	
	
	m_PanelMain->SetSizer( bSizer29 );
	m_PanelMain->Layout();
	bSizer29->Fit( m_PanelMain );
	bSizer4->Add( m_PanelMain, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer4 );
	this->Layout();
	bSizer4->Fit( this );
	m_MenuBar = new wxMenuBar( 0 );
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
	m_ToolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_ToolBar->Realize(); 
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_ButtonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameCalibrationVirtual::OnSaveDefault ), NULL, this );
	this->Connect( m_MenuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFramePreferences ) );
	this->Connect( m_MenuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnShowLog ) );
	this->Connect( m_MenuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel1 ) );
	this->Connect( m_MenuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel2 ) );
	this->Connect( m_MenuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel3 ) );
	this->Connect( m_MenuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFrameAbout ) );
}

asFrameCalibrationVirtual::~asFrameCalibrationVirtual()
{
	// Disconnect Events
	m_ButtonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameCalibrationVirtual::OnSaveDefault ), NULL, this );
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
	
	m_PanelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookBase = new wxNotebook( m_PanelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
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
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Directories") ), wxVERTICAL );
	
	m_StaticTextArchivePredictorsDir = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerArchivePredictors = new wxDirPickerCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextPredictandDBDir = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Default predictand DB directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerPredictandDB = new wxDirPickerCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer16->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelGeneralCommon->SetSizer( bSizer16 );
	m_PanelGeneralCommon->Layout();
	bSizer16->Fit( m_PanelGeneralCommon );
	m_NotebookBase->AddPage( m_PanelGeneralCommon, _("General"), true );
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
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneral, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_CheckBoxResponsiveness = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxResponsiveness->SetValue(true); 
	sbSizer151->Add( m_CheckBoxResponsiveness, 0, wxALL, 5 );
	
	
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
	
	wxStaticBoxSizer* sbSizer411;
	sbSizer411 = new wxStaticBoxSizer( new wxStaticBox( m_PanelUserDirectories, wxID_ANY, _("Working directories") ), wxVERTICAL );
	
	m_StaticTextIntermediateResultsDir = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Directory to save intermediate temporary results"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextIntermediateResultsDir->Wrap( -1 );
	sbSizer411->Add( m_StaticTextIntermediateResultsDir, 0, wxALL, 5 );
	
	m_DirPickerIntermediateResults = new wxDirPickerCtrl( m_PanelUserDirectories, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer411->Add( m_DirPickerIntermediateResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer24->Add( sbSizer411, 0, wxEXPAND|wxALL, 5 );
	
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
	m_CheckBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_ButtonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesCalibratorVirtual::~asFramePreferencesCalibratorVirtual()
{
	// Disconnect Events
	m_CheckBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_ButtonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesCalibratorVirtual::SaveAndClose ), NULL, this );
	
}
