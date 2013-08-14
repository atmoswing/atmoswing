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
	
	wxString m_ChoiceMethodChoices[] = { _("Single assessment"), _("(Exhaustive exploration)"), _("Classic calibration"), _("Classic+ calibration"), _("Variables exploration Classic+"), _("Nelder-Mead optimization"), _("Monte-Carlo"), _("Genetic algorithms"), _("Evaluate all scores"), _("Only predictand values") };
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
	
	m_StaticTextPredictorDir = new wxStaticText( m_PanelControls, wxID_ANY, _("Select the predictors directory (if not set in the catalog file)"), wxDefaultPosition, wxDefaultSize, 0 );
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
	
	m_staticText68 = new wxStaticText( m_PanelSingle, wxID_ANY, _("Exhaustive exploration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText68->Wrap( -1 );
	sbSizer9->Add( m_staticText68, 0, wxALL, 5 );
	
	
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
	m_PanelNelderMead = new wxPanel( m_NotebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer19;
	fgSizer19 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer19->SetFlexibleDirection( wxBOTH );
	fgSizer19->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	wxStaticBoxSizer* sbSizer13;
	sbSizer13 = new wxStaticBoxSizer( new wxStaticBox( m_PanelNelderMead, wxID_ANY, _("Nelder-Mead options") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer10;
	fgSizer10 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer10->SetFlexibleDirection( wxBOTH );
	fgSizer10->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextNelderMeadNbRuns = new wxStaticText( m_PanelNelderMead, wxID_ANY, _("Number of independant runs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNelderMeadNbRuns->Wrap( -1 );
	fgSizer10->Add( m_StaticTextNelderMeadNbRuns, 0, wxALL, 5 );
	
	m_TextCtrlNelderMeadNbRuns = new wxTextCtrl( m_PanelNelderMead, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlNelderMeadNbRuns->SetMaxLength( 0 ); 
	fgSizer10->Add( m_TextCtrlNelderMeadNbRuns, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextNelderMeadRho = new wxStaticText( m_PanelNelderMead, wxID_ANY, _("Rho (reflexion factor)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNelderMeadRho->Wrap( -1 );
	fgSizer10->Add( m_StaticTextNelderMeadRho, 0, wxALL, 5 );
	
	m_TextCtrlNelderMeadRho = new wxTextCtrl( m_PanelNelderMead, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlNelderMeadRho->SetMaxLength( 0 ); 
	fgSizer10->Add( m_TextCtrlNelderMeadRho, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextNelderMeadChi = new wxStaticText( m_PanelNelderMead, wxID_ANY, _("Chi (expansion)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNelderMeadChi->Wrap( -1 );
	fgSizer10->Add( m_StaticTextNelderMeadChi, 0, wxALL, 5 );
	
	m_TextCtrlNelderMeadChi = new wxTextCtrl( m_PanelNelderMead, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlNelderMeadChi->SetMaxLength( 0 ); 
	fgSizer10->Add( m_TextCtrlNelderMeadChi, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextNelderMeadGamma = new wxStaticText( m_PanelNelderMead, wxID_ANY, _("Gamma (contraction)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNelderMeadGamma->Wrap( -1 );
	fgSizer10->Add( m_StaticTextNelderMeadGamma, 0, wxALL, 5 );
	
	m_TextCtrlNelderMeadGamma = new wxTextCtrl( m_PanelNelderMead, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlNelderMeadGamma->SetMaxLength( 0 ); 
	fgSizer10->Add( m_TextCtrlNelderMeadGamma, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextNelderMeadSigma = new wxStaticText( m_PanelNelderMead, wxID_ANY, _("Sigma (reduction)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNelderMeadSigma->Wrap( -1 );
	fgSizer10->Add( m_StaticTextNelderMeadSigma, 0, wxALL, 5 );
	
	m_TextCtrlNelderMeadSigma = new wxTextCtrl( m_PanelNelderMead, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlNelderMeadSigma->SetMaxLength( 0 ); 
	fgSizer10->Add( m_TextCtrlNelderMeadSigma, 0, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer13->Add( fgSizer10, 1, wxEXPAND, 5 );
	
	
	fgSizer19->Add( sbSizer13, 1, wxEXPAND|wxALL, 5 );
	
	
	m_PanelNelderMead->SetSizer( fgSizer19 );
	m_PanelNelderMead->Layout();
	fgSizer19->Fit( m_PanelNelderMead );
	m_NotebookOptions->AddPage( m_PanelNelderMead, _("Nelder-Mead"), false );
	m_PanelGeneticAlgoritms = new wxPanel( m_NotebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer111;
	bSizer111 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );
	
	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneticAlgoritms, wxID_ANY, _("Operators") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsNaturalSelectionOperator = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Natural selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsNaturalSelectionOperator->Wrap( -1 );
	fgSizer11->Add( m_StaticTextGAsNaturalSelectionOperator, 0, wxALL, 5 );
	
	wxString m_ChoiceGAsNaturalSelectionOperatorChoices[] = { _("Ratio elitism"), _("Tournament") };
	int m_ChoiceGAsNaturalSelectionOperatorNChoices = sizeof( m_ChoiceGAsNaturalSelectionOperatorChoices ) / sizeof( wxString );
	m_ChoiceGAsNaturalSelectionOperator = new wxChoice( m_PanelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceGAsNaturalSelectionOperatorNChoices, m_ChoiceGAsNaturalSelectionOperatorChoices, 0 );
	m_ChoiceGAsNaturalSelectionOperator->SetSelection( 0 );
	fgSizer11->Add( m_ChoiceGAsNaturalSelectionOperator, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGAsCouplesSelectionOperator = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Couples selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCouplesSelectionOperator->Wrap( -1 );
	fgSizer11->Add( m_StaticTextGAsCouplesSelectionOperator, 0, wxALL, 5 );
	
	wxString m_ChoiceGAsCouplesSelectionOperatorChoices[] = { _("Rank pairing"), _("Randomly"), _("Roulette wheel on rank"), _("Roulette wheel on score"), _("Tournament") };
	int m_ChoiceGAsCouplesSelectionOperatorNChoices = sizeof( m_ChoiceGAsCouplesSelectionOperatorChoices ) / sizeof( wxString );
	m_ChoiceGAsCouplesSelectionOperator = new wxChoice( m_PanelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceGAsCouplesSelectionOperatorNChoices, m_ChoiceGAsCouplesSelectionOperatorChoices, 0 );
	m_ChoiceGAsCouplesSelectionOperator->SetSelection( 0 );
	fgSizer11->Add( m_ChoiceGAsCouplesSelectionOperator, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGAsCrossoverOperator = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Crossover"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverOperator->Wrap( -1 );
	fgSizer11->Add( m_StaticTextGAsCrossoverOperator, 0, wxALL, 5 );
	
	wxString m_ChoiceGAsCrossoverOperatorChoices[] = { _("Single point crossover"), _("Double points crossover"), _("Multiple points crossover"), _("Uniform crossover"), _("Limited blending"), _("Linear crossover"), _("Heuristic crossover"), _("Binary-like crossover"), _("Linear interpolation"), _("Free interpolation") };
	int m_ChoiceGAsCrossoverOperatorNChoices = sizeof( m_ChoiceGAsCrossoverOperatorChoices ) / sizeof( wxString );
	m_ChoiceGAsCrossoverOperator = new wxChoice( m_PanelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceGAsCrossoverOperatorNChoices, m_ChoiceGAsCrossoverOperatorChoices, 0 );
	m_ChoiceGAsCrossoverOperator->SetSelection( 0 );
	fgSizer11->Add( m_ChoiceGAsCrossoverOperator, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGAsMutationOperator = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Mutation"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationOperator->Wrap( -1 );
	fgSizer11->Add( m_StaticTextGAsMutationOperator, 0, wxALL, 5 );
	
	wxString m_ChoiceGAsMutationOperatorChoices[] = { _("Uniform constant"), _("Uniform variable"), _("Normal constant"), _("Normal variable"), _("Non-uniform"), _("Self adaptation rate"), _("Self adaptation radius"), _("Self adaptation rate chromosome"), _("Self adaptation radius chromosome"), _("Multi scale") };
	int m_ChoiceGAsMutationOperatorNChoices = sizeof( m_ChoiceGAsMutationOperatorChoices ) / sizeof( wxString );
	m_ChoiceGAsMutationOperator = new wxChoice( m_PanelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceGAsMutationOperatorNChoices, m_ChoiceGAsMutationOperatorChoices, 0 );
	m_ChoiceGAsMutationOperator->SetSelection( 0 );
	fgSizer11->Add( m_ChoiceGAsMutationOperator, 0, wxALL|wxEXPAND, 5 );
	
	
	sbSizer5->Add( fgSizer11, 1, wxEXPAND, 5 );
	
	
	bSizer12->Add( sbSizer5, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneticAlgoritms, wxID_ANY, _("General options") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer121;
	fgSizer121 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer121->SetFlexibleDirection( wxBOTH );
	fgSizer121->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsRunNumbers = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Runs number"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsRunNumbers->Wrap( -1 );
	fgSizer121->Add( m_StaticTextGAsRunNumbers, 0, wxALL, 5 );
	
	m_TextCtrlGAsRunNumbers = new wxTextCtrl( m_PanelGeneticAlgoritms, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsRunNumbers->SetMaxLength( 0 ); 
	fgSizer121->Add( m_TextCtrlGAsRunNumbers, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsPopulationSize = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Population's size"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsPopulationSize->Wrap( -1 );
	fgSizer121->Add( m_StaticTextGAsPopulationSize, 0, wxALL, 5 );
	
	m_TextCtrlGAsPopulationSize = new wxTextCtrl( m_PanelGeneticAlgoritms, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsPopulationSize->SetMaxLength( 0 ); 
	fgSizer121->Add( m_TextCtrlGAsPopulationSize, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsConvergenceNb = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Convergence after"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsConvergenceNb->Wrap( -1 );
	fgSizer121->Add( m_StaticTextGAsConvergenceNb, 0, wxALL, 5 );
	
	m_TextCtrlGAsConvergenceNb = new wxTextCtrl( m_PanelGeneticAlgoritms, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsConvergenceNb->SetMaxLength( 0 ); 
	fgSizer121->Add( m_TextCtrlGAsConvergenceNb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsRatioIntermGen = new wxStaticText( m_PanelGeneticAlgoritms, wxID_ANY, _("Ratio interm. gen."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsRatioIntermGen->Wrap( -1 );
	fgSizer121->Add( m_StaticTextGAsRatioIntermGen, 0, wxALL, 5 );
	
	m_TextCtrlGAsRatioIntermGen = new wxTextCtrl( m_PanelGeneticAlgoritms, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsRatioIntermGen->SetMaxLength( 0 ); 
	fgSizer121->Add( m_TextCtrlGAsRatioIntermGen, 0, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer6->Add( fgSizer121, 0, wxEXPAND, 5 );
	
	m_CheckBoxGAsAllowElitism = new wxCheckBox( m_PanelGeneticAlgoritms, wxID_ANY, _("Allow elitism for the best"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer6->Add( m_CheckBoxGAsAllowElitism, 0, wxALL, 5 );
	
	
	bSizer12->Add( sbSizer6, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer111->Add( bSizer12, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookGAoptions = new wxNotebook( m_PanelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelSelections = new wxPanel( m_NotebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer141;
	fgSizer141 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer141->SetFlexibleDirection( wxBOTH );
	fgSizer141->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsNaturalSlctTournamentProb = new wxStaticText( m_PanelSelections, wxID_ANY, _("Natural slct tournament: prob"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsNaturalSlctTournamentProb->Wrap( -1 );
	fgSizer141->Add( m_StaticTextGAsNaturalSlctTournamentProb, 0, wxALL, 5 );
	
	m_TextCtrlGAsNaturalSlctTournamentProb = new wxTextCtrl( m_PanelSelections, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsNaturalSlctTournamentProb->SetMaxLength( 0 ); 
	fgSizer141->Add( m_TextCtrlGAsNaturalSlctTournamentProb, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer13->Add( fgSizer141, 1, wxEXPAND|wxALL, 5 );
	
	wxFlexGridSizer* fgSizer151;
	fgSizer151 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer151->SetFlexibleDirection( wxBOTH );
	fgSizer151->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsCouplesSlctTournamentNb = new wxStaticText( m_PanelSelections, wxID_ANY, _("Couples slct tournament: nb ind."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCouplesSlctTournamentNb->Wrap( -1 );
	fgSizer151->Add( m_StaticTextGAsCouplesSlctTournamentNb, 0, wxALL, 5 );
	
	m_TextCtrlGAsCouplesSlctTournamentNb = new wxTextCtrl( m_PanelSelections, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsCouplesSlctTournamentNb->SetMaxLength( 0 ); 
	fgSizer151->Add( m_TextCtrlGAsCouplesSlctTournamentNb, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer13->Add( fgSizer151, 1, wxEXPAND|wxALL, 5 );
	
	
	m_PanelSelections->SetSizer( bSizer13 );
	m_PanelSelections->Layout();
	bSizer13->Fit( m_PanelSelections );
	m_NotebookGAoptions->AddPage( m_PanelSelections, _("Natural and couple selections"), false );
	m_PanelCrossover = new wxPanel( m_NotebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer10;
	bSizer10 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer14;
	fgSizer14 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer14->SetFlexibleDirection( wxBOTH );
	fgSizer14->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsCrossoverMultipleNbPts = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Multiple crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverMultipleNbPts->Wrap( -1 );
	fgSizer14->Add( m_StaticTextGAsCrossoverMultipleNbPts, 0, wxALL, 5 );
	
	m_TextCtrlGAsCrossoverMultipleNbPts = new wxTextCtrl( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsCrossoverMultipleNbPts->SetMaxLength( 0 ); 
	fgSizer14->Add( m_TextCtrlGAsCrossoverMultipleNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsCrossoverBlendingNbPts = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Blending crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverBlendingNbPts->Wrap( -1 );
	fgSizer14->Add( m_StaticTextGAsCrossoverBlendingNbPts, 0, wxALL, 5 );
	
	m_TextCtrlGAsCrossoverBlendingNbPts = new wxTextCtrl( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsCrossoverBlendingNbPts->SetMaxLength( 0 ); 
	fgSizer14->Add( m_TextCtrlGAsCrossoverBlendingNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsCrossoverBlendingShareBeta = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Blending crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverBlendingShareBeta->Wrap( -1 );
	fgSizer14->Add( m_StaticTextGAsCrossoverBlendingShareBeta, 0, wxALL, 5 );
	
	m_CheckBoxGAsCrossoverBlendingShareBeta = new wxCheckBox( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer14->Add( m_CheckBoxGAsCrossoverBlendingShareBeta, 0, wxALL, 5 );
	
	m_StaticTextGAsCrossoverLinearNbPts = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Linear crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverLinearNbPts->Wrap( -1 );
	fgSizer14->Add( m_StaticTextGAsCrossoverLinearNbPts, 0, wxALL, 5 );
	
	m_TextCtrlGAsCrossoverLinearNbPts = new wxTextCtrl( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsCrossoverLinearNbPts->SetMaxLength( 0 ); 
	fgSizer14->Add( m_TextCtrlGAsCrossoverLinearNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer10->Add( fgSizer14, 1, wxEXPAND|wxALL, 5 );
	
	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsCrossoverHeuristicNbPts = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Heuristic crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverHeuristicNbPts->Wrap( -1 );
	fgSizer15->Add( m_StaticTextGAsCrossoverHeuristicNbPts, 0, wxALL, 5 );
	
	m_TextCtrlGAsCrossoverHeuristicNbPts = new wxTextCtrl( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsCrossoverHeuristicNbPts->SetMaxLength( 0 ); 
	fgSizer15->Add( m_TextCtrlGAsCrossoverHeuristicNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsCrossoverHeuristicShareBeta = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Heuristic crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverHeuristicShareBeta->Wrap( -1 );
	fgSizer15->Add( m_StaticTextGAsCrossoverHeuristicShareBeta, 0, wxALL, 5 );
	
	m_CheckBoxGAsCrossoverHeuristicShareBeta = new wxCheckBox( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( m_CheckBoxGAsCrossoverHeuristicShareBeta, 0, wxALL, 5 );
	
	m_StaticTextGAsCrossoverBinLikeNbPts = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Binary-like crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverBinLikeNbPts->Wrap( -1 );
	fgSizer15->Add( m_StaticTextGAsCrossoverBinLikeNbPts, 0, wxALL, 5 );
	
	m_TextCtrlGAsCrossoverBinLikeNbPts = new wxTextCtrl( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsCrossoverBinLikeNbPts->SetMaxLength( 0 ); 
	fgSizer15->Add( m_TextCtrlGAsCrossoverBinLikeNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsCrossoverBinLikeShareBeta = new wxStaticText( m_PanelCrossover, wxID_ANY, _("Binary-like crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsCrossoverBinLikeShareBeta->Wrap( -1 );
	fgSizer15->Add( m_StaticTextGAsCrossoverBinLikeShareBeta, 0, wxALL, 5 );
	
	m_CheckBoxGAsCrossoverBinLikeShareBeta = new wxCheckBox( m_PanelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( m_CheckBoxGAsCrossoverBinLikeShareBeta, 0, wxALL, 5 );
	
	
	bSizer10->Add( fgSizer15, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelCrossover->SetSizer( bSizer10 );
	m_PanelCrossover->Layout();
	bSizer10->Fit( m_PanelCrossover );
	m_NotebookGAoptions->AddPage( m_PanelCrossover, _("Crossover"), false );
	m_PanelMutation = new wxPanel( m_NotebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 7, 2, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsMutationsUniformCstProb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Uniform constant: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsUniformCstProb->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsUniformCstProb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsUniformCstProb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsUniformCstProb->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsUniformCstProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalCstProb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal constant: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalCstProb->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsNormalCstProb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalCstProb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalCstProb->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsNormalCstProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalCstStdDev = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal constant: std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalCstStdDev->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsNormalCstStdDev, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalCstStdDev = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalCstStdDev->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsNormalCstStdDev, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsUniformVarMaxGensNb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Uniform variable: on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsUniformVarMaxGensNb->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsUniformVarMaxGensNb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsUniformVarMaxGensNb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsUniformVarMaxGensNb->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsUniformVarMaxGensNb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsUniformVarProbStart = new wxStaticText( m_PanelMutation, wxID_ANY, _("Uniform variable: starting probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsUniformVarProbStart->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsUniformVarProbStart, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsUniformVarProbStart = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsUniformVarProbStart->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsUniformVarProbStart, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsUniformVarProbEnd = new wxStaticText( m_PanelMutation, wxID_ANY, _("Uniform variable: ending probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsUniformVarProbEnd->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsUniformVarProbEnd, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsUniformVarProbEnd = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsUniformVarProbEnd->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsUniformVarProbEnd, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsMultiScaleProb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Multi-scale: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsMultiScaleProb->Wrap( -1 );
	fgSizer13->Add( m_StaticTextGAsMutationsMultiScaleProb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsMultiScaleProb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsMultiScaleProb->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlGAsMutationsMultiScaleProb, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer11->Add( fgSizer13, 1, wxEXPAND|wxALL, 5 );
	
	wxFlexGridSizer* fgSizer191;
	fgSizer191 = new wxFlexGridSizer( 9, 2, 0, 0 );
	fgSizer191->SetFlexibleDirection( wxBOTH );
	fgSizer191->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGAsMutationsNormalVarMaxGensNbProb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal variable: prob on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalVarMaxGensNbProb->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNormalVarMaxGensNbProb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalVarMaxGensNbProb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalVarMaxGensNbProb->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNormalVarMaxGensNbProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalVarMaxGensNbStdDev = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal variable: std dev on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalVarMaxGensNbStdDev->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNormalVarMaxGensNbStdDev, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalVarMaxGensNbStdDev = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalVarMaxGensNbStdDev->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNormalVarMaxGensNbStdDev, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalVarProbStart = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal variable: starting probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalVarProbStart->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNormalVarProbStart, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalVarProbStart = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalVarProbStart->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNormalVarProbStart, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalVarProbEnd = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal variable: ending probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalVarProbEnd->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNormalVarProbEnd, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalVarProbEnd = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalVarProbEnd->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNormalVarProbEnd, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalVarStdDevStart = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal variable: starting std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalVarStdDevStart->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNormalVarStdDevStart, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalVarStdDevStart = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalVarStdDevStart->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNormalVarStdDevStart, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNormalVarStdDevEnd = new wxStaticText( m_PanelMutation, wxID_ANY, _("Normal variable: ending std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNormalVarStdDevEnd->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNormalVarStdDevEnd, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNormalVarStdDevEnd = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNormalVarStdDevEnd->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNormalVarStdDevEnd, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNonUniformProb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Non-uniform: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNonUniformProb->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNonUniformProb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNonUniformProb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNonUniformProb->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNonUniformProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNonUniformGensNb = new wxStaticText( m_PanelMutation, wxID_ANY, _("Non-uniform: on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNonUniformGensNb->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNonUniformGensNb, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNonUniformGensNb = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNonUniformGensNb->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNonUniformGensNb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextGAsMutationsNonUniformMinRate = new wxStaticText( m_PanelMutation, wxID_ANY, _("Non-uniform: minimum rate"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGAsMutationsNonUniformMinRate->Wrap( -1 );
	fgSizer191->Add( m_StaticTextGAsMutationsNonUniformMinRate, 0, wxALL, 5 );
	
	m_TextCtrlGAsMutationsNonUniformMinRate = new wxTextCtrl( m_PanelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_TextCtrlGAsMutationsNonUniformMinRate->SetMaxLength( 0 ); 
	fgSizer191->Add( m_TextCtrlGAsMutationsNonUniformMinRate, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer11->Add( fgSizer191, 1, wxEXPAND|wxALL, 5 );
	
	
	m_PanelMutation->SetSizer( bSizer11 );
	m_PanelMutation->Layout();
	bSizer11->Fit( m_PanelMutation );
	m_NotebookGAoptions->AddPage( m_PanelMutation, _("Mutation"), true );
	
	bSizer14->Add( m_NotebookGAoptions, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer111->Add( bSizer14, 0, wxEXPAND, 5 );
	
	
	m_PanelGeneticAlgoritms->SetSizer( bSizer111 );
	m_PanelGeneticAlgoritms->Layout();
	bSizer111->Fit( m_PanelGeneticAlgoritms );
	m_NotebookOptions->AddPage( m_PanelGeneticAlgoritms, _("Genetic algoritms"), false );
	
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
	m_ToolBar = this->CreateToolBar( wxTB_HORIZONTAL, wxID_ANY );
	m_ToolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	m_ToolBar->Realize(); 
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
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
	
	m_MenuControls = new wxMenu();
	wxMenuItem* m_MenuItemLaunch;
	m_MenuItemLaunch = new wxMenuItem( m_MenuControls, wxID_ANY, wxString( _("Launch") ) , wxEmptyString, wxITEM_NORMAL );
	m_MenuControls->Append( m_MenuItemLaunch );
	
	m_MenuBar->Append( m_MenuControls, _("Controls") ); 
	
	this->SetMenuBar( m_MenuBar );
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_ButtonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameCalibrationVirtual::OnSaveDefault ), NULL, this );
	this->Connect( m_MenuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFramePreferences ) );
	this->Connect( m_MenuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnShowLog ) );
	this->Connect( m_MenuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel1 ) );
	this->Connect( m_MenuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel2 ) );
	this->Connect( m_MenuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OnLogLevel3 ) );
	this->Connect( m_MenuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::OpenFrameAbout ) );
	this->Connect( m_MenuItemLaunch->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::Launch ) );
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
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameCalibrationVirtual::Launch ) );
	
}
