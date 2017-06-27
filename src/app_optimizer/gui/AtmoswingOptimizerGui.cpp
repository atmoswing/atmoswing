///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun 17 2015)
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

#include "AtmoswingOptimizerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameOptimizerVirtual::asFrameOptimizerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 600,500 ), wxDefaultSize );
	
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
	
	wxString m_choiceMethodChoices[] = { _("Single assessment"), _("Classic calibration"), _("Classic+ calibration"), _("Variables exploration Classic+"), _("Monte-Carlo"), _("Genetic algorithms"), _("Evaluate all scores"), _("Only predictand values") };
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
	
	m_staticTextClassicPlusStepsLonPertinenceMap = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, _("Multiple of the steps in lon for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextClassicPlusStepsLonPertinenceMap->Wrap( -1 );
	fgSizer21->Add( m_staticTextClassicPlusStepsLonPertinenceMap, 0, wxALL, 5 );
	
	m_textCtrlClassicPlusStepsLonPertinenceMap = new wxTextCtrl( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlClassicPlusStepsLonPertinenceMap->SetMaxLength( 0 ); 
	fgSizer21->Add( m_textCtrlClassicPlusStepsLonPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextClassicPlusStepsLatPertinenceMap = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, _("Multiple of the steps in lat for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextClassicPlusStepsLatPertinenceMap->Wrap( -1 );
	fgSizer21->Add( m_staticTextClassicPlusStepsLatPertinenceMap, 0, wxALL, 5 );
	
	m_textCtrlClassicPlusStepsLatPertinenceMap = new wxTextCtrl( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlClassicPlusStepsLatPertinenceMap->SetMaxLength( 0 ); 
	fgSizer21->Add( m_textCtrlClassicPlusStepsLatPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextClassicPlusResizingIterations = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, _("Iterations in final resizing attempts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextClassicPlusResizingIterations->Wrap( -1 );
	fgSizer21->Add( m_staticTextClassicPlusResizingIterations, 0, wxALL, 5 );
	
	m_textCtrlClassicPlusResizingIterations = new wxTextCtrl( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlClassicPlusResizingIterations->SetMaxLength( 0 ); 
	fgSizer21->Add( m_textCtrlClassicPlusResizingIterations, 0, wxRIGHT|wxLEFT, 5 );
	
	m_checkBoxProceedSequentially = new wxCheckBox( sbSizer10->GetStaticBox(), wxID_ANY, _("Proceed sequentially (standard)"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer21->Add( m_checkBoxProceedSequentially, 0, wxALL, 5 );
	
	m_staticTextSpacer = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSpacer->Wrap( -1 );
	fgSizer21->Add( m_staticTextSpacer, 0, wxALL, 5 );
	
	m_checkBoxClassicPlusResize = new wxCheckBox( sbSizer10->GetStaticBox(), wxID_ANY, _("Resize the spatial windows separately"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxClassicPlusResize->Enable( false );
	
	fgSizer21->Add( m_checkBoxClassicPlusResize, 0, wxALL, 5 );
	
	
	sbSizer10->Add( fgSizer21, 1, wxEXPAND, 5 );
	
	
	fgSizer3->Add( sbSizer10, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer9;
	sbSizer9 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("No option for") ), wxVERTICAL );
	
	m_staticText66 = new wxStaticText( sbSizer9->GetStaticBox(), wxID_ANY, _("Single assessment"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText66->Wrap( -1 );
	sbSizer9->Add( m_staticText66, 0, wxALL, 5 );
	
	m_staticText67 = new wxStaticText( sbSizer9->GetStaticBox(), wxID_ANY, _("Classic calibration"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText67->Wrap( -1 );
	sbSizer9->Add( m_staticText67, 0, wxALL, 5 );
	
	
	fgSizer3->Add( sbSizer9, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer12;
	sbSizer12 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("Monte-Carlo") ), wxVERTICAL );
	
	m_staticTextMonteCarloRandomNb = new wxStaticText( sbSizer12->GetStaticBox(), wxID_ANY, _("Number of random param. sets"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextMonteCarloRandomNb->Wrap( -1 );
	sbSizer12->Add( m_staticTextMonteCarloRandomNb, 0, wxALL, 5 );
	
	m_textCtrlMonteCarloRandomNb = new wxTextCtrl( sbSizer12->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlMonteCarloRandomNb->SetMaxLength( 0 ); 
	sbSizer12->Add( m_textCtrlMonteCarloRandomNb, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	fgSizer3->Add( sbSizer12, 1, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer91;
	sbSizer91 = new wxStaticBoxSizer( new wxStaticBox( m_panelSingle, wxID_ANY, _("Variables exploration") ), wxVERTICAL );
	
	m_staticTextVarExploStepToExplore = new wxStaticText( sbSizer91->GetStaticBox(), wxID_ANY, _("Step to explore (0-based)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextVarExploStepToExplore->Wrap( -1 );
	sbSizer91->Add( m_staticTextVarExploStepToExplore, 0, wxALL, 5 );
	
	m_textCtrlVarExploStepToExplore = new wxTextCtrl( sbSizer91->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	m_textCtrlVarExploStepToExplore->SetMaxLength( 0 ); 
	sbSizer91->Add( m_textCtrlVarExploStepToExplore, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	fgSizer3->Add( sbSizer91, 1, wxEXPAND|wxALL, 5 );
	
	
	m_panelSingle->SetSizer( fgSizer3 );
	m_panelSingle->Layout();
	fgSizer3->Fit( m_panelSingle );
	m_notebookOptions->AddPage( m_panelSingle, _("Calibration"), false );
	m_panelGeneticAlgoritms = new wxPanel( m_notebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer111;
	bSizer111 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );
	
	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneticAlgoritms, wxID_ANY, _("Operators") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsNaturalSelectionOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Natural selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsNaturalSelectionOperator->Wrap( -1 );
	fgSizer11->Add( m_staticTextGAsNaturalSelectionOperator, 0, wxALL, 5 );
	
	wxString m_choiceGAsNaturalSelectionOperatorChoices[] = { _("Ratio elitism"), _("Tournament") };
	int m_choiceGAsNaturalSelectionOperatorNChoices = sizeof( m_choiceGAsNaturalSelectionOperatorChoices ) / sizeof( wxString );
	m_choiceGAsNaturalSelectionOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceGAsNaturalSelectionOperatorNChoices, m_choiceGAsNaturalSelectionOperatorChoices, 0 );
	m_choiceGAsNaturalSelectionOperator->SetSelection( 0 );
	fgSizer11->Add( m_choiceGAsNaturalSelectionOperator, 0, wxALL|wxEXPAND, 5 );
	
	m_staticTextGAsCouplesSelectionOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Couples selection"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCouplesSelectionOperator->Wrap( -1 );
	fgSizer11->Add( m_staticTextGAsCouplesSelectionOperator, 0, wxALL, 5 );
	
	wxString m_choiceGAsCouplesSelectionOperatorChoices[] = { _("Rank pairing"), _("Randomly"), _("Roulette wheel on rank"), _("Roulette wheel on score"), _("Tournament") };
	int m_choiceGAsCouplesSelectionOperatorNChoices = sizeof( m_choiceGAsCouplesSelectionOperatorChoices ) / sizeof( wxString );
	m_choiceGAsCouplesSelectionOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceGAsCouplesSelectionOperatorNChoices, m_choiceGAsCouplesSelectionOperatorChoices, 0 );
	m_choiceGAsCouplesSelectionOperator->SetSelection( 0 );
	fgSizer11->Add( m_choiceGAsCouplesSelectionOperator, 0, wxALL|wxEXPAND, 5 );
	
	m_staticTextGAsCrossoverOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Crossover"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverOperator->Wrap( -1 );
	fgSizer11->Add( m_staticTextGAsCrossoverOperator, 0, wxALL, 5 );
	
	wxString m_choiceGAsCrossoverOperatorChoices[] = { _("Single point crossover"), _("Double points crossover"), _("Multiple points crossover"), _("Uniform crossover"), _("Limited blending"), _("Linear crossover"), _("Heuristic crossover"), _("Binary-like crossover"), _("Linear interpolation"), _("Free interpolation") };
	int m_choiceGAsCrossoverOperatorNChoices = sizeof( m_choiceGAsCrossoverOperatorChoices ) / sizeof( wxString );
	m_choiceGAsCrossoverOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceGAsCrossoverOperatorNChoices, m_choiceGAsCrossoverOperatorChoices, 0 );
	m_choiceGAsCrossoverOperator->SetSelection( 0 );
	fgSizer11->Add( m_choiceGAsCrossoverOperator, 0, wxALL|wxEXPAND, 5 );
	
	m_staticTextGAsMutationOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Mutation"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationOperator->Wrap( -1 );
	fgSizer11->Add( m_staticTextGAsMutationOperator, 0, wxALL, 5 );
	
	wxString m_choiceGAsMutationOperatorChoices[] = { _("Uniform constant"), _("Uniform variable"), _("Normal constant"), _("Normal variable"), _("Non-uniform"), _("Self adaptation rate"), _("Self adaptation radius"), _("Self adaptation rate chromosome"), _("Self adaptation radius chromosome"), _("Multi scale") };
	int m_choiceGAsMutationOperatorNChoices = sizeof( m_choiceGAsMutationOperatorChoices ) / sizeof( wxString );
	m_choiceGAsMutationOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceGAsMutationOperatorNChoices, m_choiceGAsMutationOperatorChoices, 0 );
	m_choiceGAsMutationOperator->SetSelection( 0 );
	fgSizer11->Add( m_choiceGAsMutationOperator, 0, wxALL|wxEXPAND, 5 );
	
	
	sbSizer5->Add( fgSizer11, 1, wxEXPAND, 5 );
	
	
	bSizer12->Add( sbSizer5, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneticAlgoritms, wxID_ANY, _("General options") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer121;
	fgSizer121 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer121->SetFlexibleDirection( wxBOTH );
	fgSizer121->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsRunNumbers = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Runs number"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsRunNumbers->Wrap( -1 );
	fgSizer121->Add( m_staticTextGAsRunNumbers, 0, wxALL, 5 );
	
	m_textCtrlGAsRunNumbers = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( m_textCtrlGAsRunNumbers, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsPopulationSize = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Population's size"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsPopulationSize->Wrap( -1 );
	fgSizer121->Add( m_staticTextGAsPopulationSize, 0, wxALL, 5 );
	
	m_textCtrlGAsPopulationSize = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( m_textCtrlGAsPopulationSize, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsConvergenceNb = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Convergence after"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsConvergenceNb->Wrap( -1 );
	fgSizer121->Add( m_staticTextGAsConvergenceNb, 0, wxALL, 5 );
	
	m_textCtrlGAsConvergenceNb = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( m_textCtrlGAsConvergenceNb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsRatioIntermGen = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Ratio interm. gen."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsRatioIntermGen->Wrap( -1 );
	fgSizer121->Add( m_staticTextGAsRatioIntermGen, 0, wxALL, 5 );
	
	m_textCtrlGAsRatioIntermGen = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( m_textCtrlGAsRatioIntermGen, 0, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer6->Add( fgSizer121, 0, wxEXPAND, 5 );
	
	m_checkBoxGAsAllowElitism = new wxCheckBox( sbSizer6->GetStaticBox(), wxID_ANY, _("Allow elitism for the best"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer6->Add( m_checkBoxGAsAllowElitism, 0, wxALL, 5 );
	
	
	bSizer12->Add( sbSizer6, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer111->Add( bSizer12, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );
	
	m_notebookGAoptions = new wxNotebook( m_panelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelSelections = new wxPanel( m_notebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer141;
	fgSizer141 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer141->SetFlexibleDirection( wxBOTH );
	fgSizer141->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsNaturalSlctTournamentProb = new wxStaticText( m_panelSelections, wxID_ANY, _("Natural slct tournament: prob"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsNaturalSlctTournamentProb->Wrap( -1 );
	fgSizer141->Add( m_staticTextGAsNaturalSlctTournamentProb, 0, wxALL, 5 );
	
	m_textCtrlGAsNaturalSlctTournamentProb = new wxTextCtrl( m_panelSelections, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer141->Add( m_textCtrlGAsNaturalSlctTournamentProb, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer13->Add( fgSizer141, 1, wxEXPAND|wxALL, 5 );
	
	wxFlexGridSizer* fgSizer151;
	fgSizer151 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer151->SetFlexibleDirection( wxBOTH );
	fgSizer151->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsCouplesSlctTournamentNb = new wxStaticText( m_panelSelections, wxID_ANY, _("Couples slct tournament: nb ind."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCouplesSlctTournamentNb->Wrap( -1 );
	fgSizer151->Add( m_staticTextGAsCouplesSlctTournamentNb, 0, wxALL, 5 );
	
	m_textCtrlGAsCouplesSlctTournamentNb = new wxTextCtrl( m_panelSelections, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer151->Add( m_textCtrlGAsCouplesSlctTournamentNb, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer13->Add( fgSizer151, 1, wxEXPAND|wxALL, 5 );
	
	
	m_panelSelections->SetSizer( bSizer13 );
	m_panelSelections->Layout();
	bSizer13->Fit( m_panelSelections );
	m_notebookGAoptions->AddPage( m_panelSelections, _("Natural and couple selections"), false );
	m_panelCrossover = new wxPanel( m_notebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer10;
	bSizer10 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer14;
	fgSizer14 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer14->SetFlexibleDirection( wxBOTH );
	fgSizer14->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsCrossoverMultipleNbPts = new wxStaticText( m_panelCrossover, wxID_ANY, _("Multiple crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverMultipleNbPts->Wrap( -1 );
	fgSizer14->Add( m_staticTextGAsCrossoverMultipleNbPts, 0, wxALL, 5 );
	
	m_textCtrlGAsCrossoverMultipleNbPts = new wxTextCtrl( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer14->Add( m_textCtrlGAsCrossoverMultipleNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsCrossoverBlendingNbPts = new wxStaticText( m_panelCrossover, wxID_ANY, _("Blending crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverBlendingNbPts->Wrap( -1 );
	fgSizer14->Add( m_staticTextGAsCrossoverBlendingNbPts, 0, wxALL, 5 );
	
	m_textCtrlGAsCrossoverBlendingNbPts = new wxTextCtrl( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer14->Add( m_textCtrlGAsCrossoverBlendingNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsCrossoverBlendingShareBeta = new wxStaticText( m_panelCrossover, wxID_ANY, _("Blending crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverBlendingShareBeta->Wrap( -1 );
	fgSizer14->Add( m_staticTextGAsCrossoverBlendingShareBeta, 0, wxALL, 5 );
	
	m_checkBoxGAsCrossoverBlendingShareBeta = new wxCheckBox( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer14->Add( m_checkBoxGAsCrossoverBlendingShareBeta, 0, wxALL, 5 );
	
	m_staticTextGAsCrossoverLinearNbPts = new wxStaticText( m_panelCrossover, wxID_ANY, _("Linear crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverLinearNbPts->Wrap( -1 );
	fgSizer14->Add( m_staticTextGAsCrossoverLinearNbPts, 0, wxALL, 5 );
	
	m_textCtrlGAsCrossoverLinearNbPts = new wxTextCtrl( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer14->Add( m_textCtrlGAsCrossoverLinearNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer10->Add( fgSizer14, 1, wxEXPAND|wxALL, 5 );
	
	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsCrossoverHeuristicNbPts = new wxStaticText( m_panelCrossover, wxID_ANY, _("Heuristic crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverHeuristicNbPts->Wrap( -1 );
	fgSizer15->Add( m_staticTextGAsCrossoverHeuristicNbPts, 0, wxALL, 5 );
	
	m_textCtrlGAsCrossoverHeuristicNbPts = new wxTextCtrl( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer15->Add( m_textCtrlGAsCrossoverHeuristicNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsCrossoverHeuristicShareBeta = new wxStaticText( m_panelCrossover, wxID_ANY, _("Heuristic crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverHeuristicShareBeta->Wrap( -1 );
	fgSizer15->Add( m_staticTextGAsCrossoverHeuristicShareBeta, 0, wxALL, 5 );
	
	m_checkBoxGAsCrossoverHeuristicShareBeta = new wxCheckBox( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( m_checkBoxGAsCrossoverHeuristicShareBeta, 0, wxALL, 5 );
	
	m_staticTextGAsCrossoverBinLikeNbPts = new wxStaticText( m_panelCrossover, wxID_ANY, _("Binary-like crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverBinLikeNbPts->Wrap( -1 );
	fgSizer15->Add( m_staticTextGAsCrossoverBinLikeNbPts, 0, wxALL, 5 );
	
	m_textCtrlGAsCrossoverBinLikeNbPts = new wxTextCtrl( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer15->Add( m_textCtrlGAsCrossoverBinLikeNbPts, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsCrossoverBinLikeShareBeta = new wxStaticText( m_panelCrossover, wxID_ANY, _("Binary-like crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsCrossoverBinLikeShareBeta->Wrap( -1 );
	fgSizer15->Add( m_staticTextGAsCrossoverBinLikeShareBeta, 0, wxALL, 5 );
	
	m_checkBoxGAsCrossoverBinLikeShareBeta = new wxCheckBox( m_panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( m_checkBoxGAsCrossoverBinLikeShareBeta, 0, wxALL, 5 );
	
	
	bSizer10->Add( fgSizer15, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelCrossover->SetSizer( bSizer10 );
	m_panelCrossover->Layout();
	bSizer10->Fit( m_panelCrossover );
	m_notebookGAoptions->AddPage( m_panelCrossover, _("Crossover"), false );
	m_panelMutation = new wxPanel( m_notebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 7, 2, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsMutationsUniformCstProb = new wxStaticText( m_panelMutation, wxID_ANY, _("Uniform constant: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsUniformCstProb->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsUniformCstProb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsUniformCstProb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsUniformCstProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalCstProb = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal constant: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalCstProb->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsNormalCstProb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalCstProb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsNormalCstProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalCstStdDev = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal constant: std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalCstStdDev->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsNormalCstStdDev, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalCstStdDev = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsNormalCstStdDev, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsUniformVarMaxGensNb = new wxStaticText( m_panelMutation, wxID_ANY, _("Uniform variable: on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsUniformVarMaxGensNb->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsUniformVarMaxGensNb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsUniformVarMaxGensNb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsUniformVarMaxGensNb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsUniformVarProbStart = new wxStaticText( m_panelMutation, wxID_ANY, _("Uniform variable: starting probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsUniformVarProbStart->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsUniformVarProbStart, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsUniformVarProbStart = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsUniformVarProbStart, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsUniformVarProbEnd = new wxStaticText( m_panelMutation, wxID_ANY, _("Uniform variable: ending probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsUniformVarProbEnd->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsUniformVarProbEnd, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsUniformVarProbEnd = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsUniformVarProbEnd, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsMultiScaleProb = new wxStaticText( m_panelMutation, wxID_ANY, _("Multi-scale: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsMultiScaleProb->Wrap( -1 );
	fgSizer13->Add( m_staticTextGAsMutationsMultiScaleProb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsMultiScaleProb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( m_textCtrlGAsMutationsMultiScaleProb, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer11->Add( fgSizer13, 1, wxEXPAND|wxALL, 5 );
	
	wxFlexGridSizer* fgSizer191;
	fgSizer191 = new wxFlexGridSizer( 9, 2, 0, 0 );
	fgSizer191->SetFlexibleDirection( wxBOTH );
	fgSizer191->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextGAsMutationsNormalVarMaxGensNbProb = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal variable: prob on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalVarMaxGensNbProb->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNormalVarMaxGensNbProb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalVarMaxGensNbProb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNormalVarMaxGensNbProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalVarMaxGensNbStdDev = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal variable: std dev on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalVarMaxGensNbStdDev->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNormalVarMaxGensNbStdDev, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNormalVarMaxGensNbStdDev, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalVarProbStart = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal variable: starting probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalVarProbStart->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNormalVarProbStart, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalVarProbStart = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNormalVarProbStart, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalVarProbEnd = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal variable: ending probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalVarProbEnd->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNormalVarProbEnd, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalVarProbEnd = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNormalVarProbEnd, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalVarStdDevStart = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal variable: starting std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalVarStdDevStart->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNormalVarStdDevStart, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalVarStdDevStart = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNormalVarStdDevStart, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNormalVarStdDevEnd = new wxStaticText( m_panelMutation, wxID_ANY, _("Normal variable: ending std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNormalVarStdDevEnd->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNormalVarStdDevEnd, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNormalVarStdDevEnd = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNormalVarStdDevEnd, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNonUniformProb = new wxStaticText( m_panelMutation, wxID_ANY, _("Non-uniform: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNonUniformProb->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNonUniformProb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNonUniformProb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNonUniformProb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNonUniformGensNb = new wxStaticText( m_panelMutation, wxID_ANY, _("Non-uniform: on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNonUniformGensNb->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNonUniformGensNb, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNonUniformGensNb = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNonUniformGensNb, 0, wxRIGHT|wxLEFT, 5 );
	
	m_staticTextGAsMutationsNonUniformMinRate = new wxStaticText( m_panelMutation, wxID_ANY, _("Non-uniform: minimum rate"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextGAsMutationsNonUniformMinRate->Wrap( -1 );
	fgSizer191->Add( m_staticTextGAsMutationsNonUniformMinRate, 0, wxALL, 5 );
	
	m_textCtrlGAsMutationsNonUniformMinRate = new wxTextCtrl( m_panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( m_textCtrlGAsMutationsNonUniformMinRate, 0, wxRIGHT|wxLEFT, 5 );
	
	
	bSizer11->Add( fgSizer191, 1, wxEXPAND|wxALL, 5 );
	
	
	m_panelMutation->SetSizer( bSizer11 );
	m_panelMutation->Layout();
	bSizer11->Fit( m_panelMutation );
	m_notebookGAoptions->AddPage( m_panelMutation, _("Mutation"), true );
	
	bSizer14->Add( m_notebookGAoptions, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer111->Add( bSizer14, 0, wxEXPAND, 5 );
	
	
	m_panelGeneticAlgoritms->SetSizer( bSizer111 );
	m_panelGeneticAlgoritms->Layout();
	bSizer111->Fit( m_panelGeneticAlgoritms );
	m_notebookOptions->AddPage( m_panelGeneticAlgoritms, _("Genetic algoritms"), false );
	
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
	
	m_statusBar1 = this->CreateStatusBar( 1, wxST_SIZEGRIP, wxID_ANY );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_buttonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameOptimizerVirtual::OnSaveDefault ), NULL, this );
	this->Connect( m_menuItemPreferences->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFramePreferences ) );
	this->Connect( m_menuItemShowLog->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnShowLog ) );
	this->Connect( m_MenuItemLogLevel1->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel1 ) );
	this->Connect( m_MenuItemLogLevel2->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel2 ) );
	this->Connect( m_MenuItemLogLevel3->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel3 ) );
	this->Connect( m_menuItemAbout->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFrameAbout ) );
}

asFrameOptimizerVirtual::~asFrameOptimizerVirtual()
{
	// Disconnect Events
	m_buttonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameOptimizerVirtual::OnSaveDefault ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFramePreferences ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnShowLog ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel1 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel2 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel3 ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFrameAbout ) );
	
}

asFramePreferencesOptimizerVirtual::asFramePreferencesOptimizerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
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
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneralCommon, wxID_ANY, _("Directories") ), wxVERTICAL );
	
	m_staticTextArchivePredictorsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_staticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_dirPickerArchivePredictors = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
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
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_panelGeneral, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_checkBoxResponsiveness = new wxCheckBox( sbSizer151->GetStaticBox(), wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
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
	
	m_checkBoxAllowMultithreading = new wxCheckBox( sbSizer15->GetStaticBox(), wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxAllowMultithreading->SetValue(true); 
	sbSizer15->Add( m_checkBoxAllowMultithreading, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextThreadsNb = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Max nb of threads"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsNb->Wrap( -1 );
	bSizer221->Add( m_staticTextThreadsNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_textCtrlThreadsNb = new wxTextCtrl( sbSizer15->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_textCtrlThreadsNb->SetMaxLength( 0 ); 
	bSizer221->Add( m_textCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	sbSizer15->Add( bSizer221, 0, wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticTextThreadsPriority = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( m_staticTextThreadsPriority, 0, wxALL, 5 );
	
	m_sliderThreadsPriority = new wxSlider( sbSizer15->GetStaticBox(), wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( m_sliderThreadsPriority, 1, wxRIGHT|wxLEFT, 5 );
	
	
	sbSizer15->Add( bSizer241, 0, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_radioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Date array insertions (slower)"), _("Date array splitting (slower)") };
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
	
	bSizer15->Add( m_buttonsConfirmation, 0, wxEXPAND|wxALL, 5 );
	
	
	m_panelBase->SetSizer( bSizer15 );
	m_panelBase->Layout();
	bSizer15->Fit( m_panelBase );
	bSizer14->Add( m_panelBase, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer14 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_checkBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesOptimizerVirtual::~asFramePreferencesOptimizerVirtual()
{
	// Disconnect Events
	m_checkBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::ApplyChanges ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::SaveAndClose ), NULL, this );
	
}
