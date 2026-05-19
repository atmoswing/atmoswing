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

#include "AtmoSwingOptimizerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameOptimizerVirtual::asFrameOptimizerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 650,500 ), wxDefaultSize );

	wxBoxSizer* bSizer4;
	bSizer4 = new wxBoxSizer( wxVERTICAL );

	_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxVERTICAL );

	_notebookBase = new wxNotebook( _panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_LEFT );
	_panelControls = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxVERTICAL );

	_staticTextMethod = new wxStaticText( _panelControls, wxID_ANY, _("Select the calibration method"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextMethod->Wrap( -1 );
	bSizer5->Add( _staticTextMethod, 0, wxALL, 5 );

	wxString _choiceMethodChoices[] = { _("Single assessment"), _("Classic calibration"), _("Classic+ calibration"), _("Variables exploration Classic+"), _("Monte-Carlo"), _("Genetic algorithms"), _("Evaluate all scores"), _("Only predictand values"), _("Only analog dates (and criteria)") };
	int _choiceMethodNChoices = sizeof( _choiceMethodChoices ) / sizeof( wxString );
	_choiceMethod = new wxChoice( _panelControls, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceMethodNChoices, _choiceMethodChoices, 0 );
	_choiceMethod->SetSelection( 0 );
	bSizer5->Add( _choiceMethod, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextFileParameters = new wxStaticText( _panelControls, wxID_ANY, _("Select the parameters file for the calibration"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextFileParameters->Wrap( -1 );
	bSizer5->Add( _staticTextFileParameters, 0, wxALL, 5 );

	_filePickerParameters = new wxFilePickerCtrl( _panelControls, wxID_ANY, wxEmptyString, _("Select a file"), _("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE );
	bSizer5->Add( _filePickerParameters, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextFilePredictand = new wxStaticText( _panelControls, wxID_ANY, _("Select the predictand DB file"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextFilePredictand->Wrap( -1 );
	bSizer5->Add( _staticTextFilePredictand, 0, wxALL, 5 );

	_filePickerPredictand = new wxFilePickerCtrl( _panelControls, wxID_ANY, wxEmptyString, _("Select a file"), _("*.nc"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_DEFAULT_STYLE );
	bSizer5->Add( _filePickerPredictand, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextPredictorDir = new wxStaticText( _panelControls, wxID_ANY, _("Select the predictors directory"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPredictorDir->Wrap( -1 );
	bSizer5->Add( _staticTextPredictorDir, 0, wxALL, 5 );

	_dirPickerPredictor = new wxDirPickerCtrl( _panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( _dirPickerPredictor, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextCalibrationResultsDir = new wxStaticText( _panelControls, wxID_ANY, _("Directory to save calibration outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextCalibrationResultsDir->Wrap( -1 );
	bSizer5->Add( _staticTextCalibrationResultsDir, 0, wxALL, 5 );

	_dirPickerCalibrationResults = new wxDirPickerCtrl( _panelControls, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	bSizer5->Add( _dirPickerCalibrationResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxBoxSizer* bSizer23;
	bSizer23 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextStateLabel = new wxStaticText( _panelControls, wxID_ANY, _("Calibration state: "), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextStateLabel->Wrap( -1 );
	_staticTextStateLabel->Hide();

	bSizer23->Add( _staticTextStateLabel, 0, wxALL, 5 );

	_staticTextState = new wxStaticText( _panelControls, wxID_ANY, _("Not running."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextState->Wrap( 350 );
	_staticTextState->Hide();

	bSizer23->Add( _staticTextState, 1, wxALL, 5 );


	bSizer5->Add( bSizer23, 0, wxEXPAND, 5 );


	_panelControls->SetSizer( bSizer5 );
	_panelControls->Layout();
	bSizer5->Fit( _panelControls );
	_notebookBase->AddPage( _panelControls, _("Controls"), true );
	_panelOptions = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );

	_notebookOptions = new wxNotebook( _panelOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_MULTILINE );
	_panelSingle = new wxPanel( _notebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer24;
	bSizer24 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer10;
	sbSizer10 = new wxStaticBoxSizer( new wxStaticBox( _panelSingle, wxID_ANY, _("Classic calibration") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer21;
	fgSizer21 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer21->SetFlexibleDirection( wxBOTH );
	fgSizer21->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextClassicPlusStepsLonPertinenceMap = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, _("Multiple of the steps in lon for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextClassicPlusStepsLonPertinenceMap->Wrap( -1 );
	fgSizer21->Add( _staticTextClassicPlusStepsLonPertinenceMap, 0, wxALL, 5 );

	_textCtrlClassicPlusStepsLonPertinenceMap = new wxTextCtrl( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer21->Add( _textCtrlClassicPlusStepsLonPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextClassicPlusStepsLatPertinenceMap = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, _("Multiple of the steps in lat for pertinence map"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextClassicPlusStepsLatPertinenceMap->Wrap( -1 );
	fgSizer21->Add( _staticTextClassicPlusStepsLatPertinenceMap, 0, wxALL, 5 );

	_textCtrlClassicPlusStepsLatPertinenceMap = new wxTextCtrl( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer21->Add( _textCtrlClassicPlusStepsLatPertinenceMap, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextClassicPlusResizingIterations = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, _("Iterations in final resizing attempts"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextClassicPlusResizingIterations->Wrap( -1 );
	fgSizer21->Add( _staticTextClassicPlusResizingIterations, 0, wxALL, 5 );

	_textCtrlClassicPlusResizingIterations = new wxTextCtrl( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer21->Add( _textCtrlClassicPlusResizingIterations, 0, wxRIGHT|wxLEFT, 5 );

	_checkBoxProceedSequentially = new wxCheckBox( sbSizer10->GetStaticBox(), wxID_ANY, _("Proceed sequentially (standard)"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer21->Add( _checkBoxProceedSequentially, 0, wxALL, 5 );

	_staticTextSpacer = new wxStaticText( sbSizer10->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextSpacer->Wrap( -1 );
	fgSizer21->Add( _staticTextSpacer, 0, wxALL, 5 );

	_checkBoxClassicPlusResize = new wxCheckBox( sbSizer10->GetStaticBox(), wxID_ANY, _("Resize the spatial windows separately"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxClassicPlusResize->Enable( false );

	fgSizer21->Add( _checkBoxClassicPlusResize, 0, wxALL, 5 );


	sbSizer10->Add( fgSizer21, 1, wxEXPAND, 5 );


	bSizer24->Add( sbSizer10, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer12;
	sbSizer12 = new wxStaticBoxSizer( new wxStaticBox( _panelSingle, wxID_ANY, _("Monte-Carlo") ), wxHORIZONTAL );

	_staticTextMonteCarloRandomNb = new wxStaticText( sbSizer12->GetStaticBox(), wxID_ANY, _("Number of random param. sets"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextMonteCarloRandomNb->Wrap( -1 );
	sbSizer12->Add( _staticTextMonteCarloRandomNb, 0, wxALL, 5 );

	_textCtrlMonteCarloRandomNb = new wxTextCtrl( sbSizer12->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	sbSizer12->Add( _textCtrlMonteCarloRandomNb, 0, wxBOTTOM|wxLEFT|wxRIGHT, 5 );


	bSizer24->Add( sbSizer12, 0, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer91;
	sbSizer91 = new wxStaticBoxSizer( new wxStaticBox( _panelSingle, wxID_ANY, _("Variables exploration") ), wxHORIZONTAL );

	_staticTextVarExploStepToExplore = new wxStaticText( sbSizer91->GetStaticBox(), wxID_ANY, _("Step to explore (0-based)"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextVarExploStepToExplore->Wrap( -1 );
	sbSizer91->Add( _staticTextVarExploStepToExplore, 0, wxALL, 5 );

	_textCtrlVarExploStepToExplore = new wxTextCtrl( sbSizer91->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	sbSizer91->Add( _textCtrlVarExploStepToExplore, 0, wxBOTTOM|wxLEFT|wxRIGHT, 5 );


	bSizer24->Add( sbSizer91, 0, wxEXPAND|wxALL, 5 );


	_panelSingle->SetSizer( bSizer24 );
	_panelSingle->Layout();
	bSizer24->Fit( _panelSingle );
	_notebookOptions->AddPage( _panelSingle, _("Calibration"), true );
	_panelGeneticAlgoritms = new wxPanel( _notebookOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer111;
	bSizer111 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );

	wxStaticBoxSizer* sbSizer5;
	sbSizer5 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneticAlgoritms, wxID_ANY, _("Operators") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsNaturalSelectionOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Natural selection"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsNaturalSelectionOperator->Wrap( -1 );
	fgSizer11->Add( _staticTextGAsNaturalSelectionOperator, 0, wxALL, 5 );

	wxString _choiceGAsNaturalSelectionOperatorChoices[] = { _("Ratio elitism"), _("Tournament") };
	int _choiceGAsNaturalSelectionOperatorNChoices = sizeof( _choiceGAsNaturalSelectionOperatorChoices ) / sizeof( wxString );
	_choiceGAsNaturalSelectionOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceGAsNaturalSelectionOperatorNChoices, _choiceGAsNaturalSelectionOperatorChoices, 0 );
	_choiceGAsNaturalSelectionOperator->SetSelection( 0 );
	fgSizer11->Add( _choiceGAsNaturalSelectionOperator, 0, wxALL|wxEXPAND, 5 );

	_staticTextGAsCouplesSelectionOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Couples selection"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCouplesSelectionOperator->Wrap( -1 );
	fgSizer11->Add( _staticTextGAsCouplesSelectionOperator, 0, wxALL, 5 );

	wxString _choiceGAsCouplesSelectionOperatorChoices[] = { _("Rank pairing"), _("Randomly"), _("Roulette wheel on rank"), _("Roulette wheel on score"), _("Tournament") };
	int _choiceGAsCouplesSelectionOperatorNChoices = sizeof( _choiceGAsCouplesSelectionOperatorChoices ) / sizeof( wxString );
	_choiceGAsCouplesSelectionOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceGAsCouplesSelectionOperatorNChoices, _choiceGAsCouplesSelectionOperatorChoices, 0 );
	_choiceGAsCouplesSelectionOperator->SetSelection( 0 );
	fgSizer11->Add( _choiceGAsCouplesSelectionOperator, 0, wxALL|wxEXPAND, 5 );

	_staticTextGAsCrossoverOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Crossover"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverOperator->Wrap( -1 );
	fgSizer11->Add( _staticTextGAsCrossoverOperator, 0, wxALL, 5 );

	wxString _choiceGAsCrossoverOperatorChoices[] = { _("Single point crossover"), _("Double points crossover"), _("Multiple points crossover"), _("Uniform crossover"), _("Limited blending"), _("Linear crossover"), _("Heuristic crossover"), _("Binary-like crossover"), _("Linear interpolation"), _("Free interpolation") };
	int _choiceGAsCrossoverOperatorNChoices = sizeof( _choiceGAsCrossoverOperatorChoices ) / sizeof( wxString );
	_choiceGAsCrossoverOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceGAsCrossoverOperatorNChoices, _choiceGAsCrossoverOperatorChoices, 0 );
	_choiceGAsCrossoverOperator->SetSelection( 0 );
	fgSizer11->Add( _choiceGAsCrossoverOperator, 0, wxALL|wxEXPAND, 5 );

	_staticTextGAsMutationOperator = new wxStaticText( sbSizer5->GetStaticBox(), wxID_ANY, _("Mutation"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationOperator->Wrap( -1 );
	fgSizer11->Add( _staticTextGAsMutationOperator, 0, wxALL, 5 );

	wxString _choiceGAsMutationOperatorChoices[] = { _("Uniform constant"), _("Uniform variable"), _("Normal constant"), _("Normal variable"), _("Non-uniform"), _("Self adaptation rate"), _("Self adaptation radius"), _("Self adaptation rate chromosome"), _("Self adaptation radius chromosome"), _("Multi scale") };
	int _choiceGAsMutationOperatorNChoices = sizeof( _choiceGAsMutationOperatorChoices ) / sizeof( wxString );
	_choiceGAsMutationOperator = new wxChoice( sbSizer5->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceGAsMutationOperatorNChoices, _choiceGAsMutationOperatorChoices, 0 );
	_choiceGAsMutationOperator->SetSelection( 0 );
	fgSizer11->Add( _choiceGAsMutationOperator, 0, wxALL|wxEXPAND, 5 );


	sbSizer5->Add( fgSizer11, 1, wxEXPAND, 5 );


	bSizer12->Add( sbSizer5, 1, wxALL|wxEXPAND, 5 );

	wxStaticBoxSizer* sbSizer6;
	sbSizer6 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneticAlgoritms, wxID_ANY, _("General options") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer121;
	fgSizer121 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer121->SetFlexibleDirection( wxBOTH );
	fgSizer121->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsRunNumbers = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Runs number"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsRunNumbers->Wrap( -1 );
	fgSizer121->Add( _staticTextGAsRunNumbers, 0, wxALL, 5 );

	_textCtrlGAsRunNumbers = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( _textCtrlGAsRunNumbers, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsPopulationSize = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Population's size"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsPopulationSize->Wrap( -1 );
	fgSizer121->Add( _staticTextGAsPopulationSize, 0, wxALL, 5 );

	_textCtrlGAsPopulationSize = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( _textCtrlGAsPopulationSize, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsConvergenceNb = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Convergence after"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsConvergenceNb->Wrap( -1 );
	fgSizer121->Add( _staticTextGAsConvergenceNb, 0, wxALL, 5 );

	_textCtrlGAsConvergenceNb = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( _textCtrlGAsConvergenceNb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsRatioIntermGen = new wxStaticText( sbSizer6->GetStaticBox(), wxID_ANY, _("Ratio interm. gen."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsRatioIntermGen->Wrap( -1 );
	fgSizer121->Add( _staticTextGAsRatioIntermGen, 0, wxALL, 5 );

	_textCtrlGAsRatioIntermGen = new wxTextCtrl( sbSizer6->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer121->Add( _textCtrlGAsRatioIntermGen, 0, wxRIGHT|wxLEFT, 5 );


	sbSizer6->Add( fgSizer121, 0, wxEXPAND, 5 );

	_checkBoxGAsAllowElitism = new wxCheckBox( sbSizer6->GetStaticBox(), wxID_ANY, _("Allow elitism for the best"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer6->Add( _checkBoxGAsAllowElitism, 0, wxALL, 5 );


	bSizer12->Add( sbSizer6, 0, wxALL|wxEXPAND, 5 );


	bSizer111->Add( bSizer12, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );

	_notebookGAoptions = new wxNotebook( _panelGeneticAlgoritms, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelSelections = new wxPanel( _notebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer13;
	bSizer13 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer141;
	fgSizer141 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer141->SetFlexibleDirection( wxBOTH );
	fgSizer141->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsNaturalSlctTournamentProb = new wxStaticText( _panelSelections, wxID_ANY, _("Natural slct tournament: prob"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsNaturalSlctTournamentProb->Wrap( -1 );
	fgSizer141->Add( _staticTextGAsNaturalSlctTournamentProb, 0, wxALL, 5 );

	_textCtrlGAsNaturalSlctTournamentProb = new wxTextCtrl( _panelSelections, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer141->Add( _textCtrlGAsNaturalSlctTournamentProb, 0, wxRIGHT|wxLEFT, 5 );


	bSizer13->Add( fgSizer141, 1, wxEXPAND|wxALL, 5 );

	wxFlexGridSizer* fgSizer151;
	fgSizer151 = new wxFlexGridSizer( 2, 2, 0, 0 );
	fgSizer151->SetFlexibleDirection( wxBOTH );
	fgSizer151->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsCouplesSlctTournamentNb = new wxStaticText( _panelSelections, wxID_ANY, _("Couples slct tournament: nb ind."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCouplesSlctTournamentNb->Wrap( -1 );
	fgSizer151->Add( _staticTextGAsCouplesSlctTournamentNb, 0, wxALL, 5 );

	_textCtrlGAsCouplesSlctTournamentNb = new wxTextCtrl( _panelSelections, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer151->Add( _textCtrlGAsCouplesSlctTournamentNb, 0, wxRIGHT|wxLEFT, 5 );


	bSizer13->Add( fgSizer151, 1, wxEXPAND|wxALL, 5 );


	_panelSelections->SetSizer( bSizer13 );
	_panelSelections->Layout();
	bSizer13->Fit( _panelSelections );
	_notebookGAoptions->AddPage( _panelSelections, _("Natural and couple selections"), false );
	_panelCrossover = new wxPanel( _notebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer10;
	bSizer10 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer14;
	fgSizer14 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer14->SetFlexibleDirection( wxBOTH );
	fgSizer14->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsCrossoverMultipleNbPts = new wxStaticText( _panelCrossover, wxID_ANY, _("Multiple crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverMultipleNbPts->Wrap( -1 );
	fgSizer14->Add( _staticTextGAsCrossoverMultipleNbPts, 0, wxALL, 5 );

	_textCtrlGAsCrossoverMultipleNbPts = new wxTextCtrl( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer14->Add( _textCtrlGAsCrossoverMultipleNbPts, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsCrossoverBlendingNbPts = new wxStaticText( _panelCrossover, wxID_ANY, _("Blending crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverBlendingNbPts->Wrap( -1 );
	fgSizer14->Add( _staticTextGAsCrossoverBlendingNbPts, 0, wxALL, 5 );

	_textCtrlGAsCrossoverBlendingNbPts = new wxTextCtrl( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer14->Add( _textCtrlGAsCrossoverBlendingNbPts, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsCrossoverBlendingShareBeta = new wxStaticText( _panelCrossover, wxID_ANY, _("Blending crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverBlendingShareBeta->Wrap( -1 );
	fgSizer14->Add( _staticTextGAsCrossoverBlendingShareBeta, 0, wxALL, 5 );

	_checkBoxGAsCrossoverBlendingShareBeta = new wxCheckBox( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer14->Add( _checkBoxGAsCrossoverBlendingShareBeta, 0, wxALL, 5 );

	_staticTextGAsCrossoverLinearNbPts = new wxStaticText( _panelCrossover, wxID_ANY, _("Linear crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverLinearNbPts->Wrap( -1 );
	fgSizer14->Add( _staticTextGAsCrossoverLinearNbPts, 0, wxALL, 5 );

	_textCtrlGAsCrossoverLinearNbPts = new wxTextCtrl( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer14->Add( _textCtrlGAsCrossoverLinearNbPts, 0, wxRIGHT|wxLEFT, 5 );


	bSizer10->Add( fgSizer14, 1, wxEXPAND|wxALL, 5 );

	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsCrossoverHeuristicNbPts = new wxStaticText( _panelCrossover, wxID_ANY, _("Heuristic crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverHeuristicNbPts->Wrap( -1 );
	fgSizer15->Add( _staticTextGAsCrossoverHeuristicNbPts, 0, wxALL, 5 );

	_textCtrlGAsCrossoverHeuristicNbPts = new wxTextCtrl( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer15->Add( _textCtrlGAsCrossoverHeuristicNbPts, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsCrossoverHeuristicShareBeta = new wxStaticText( _panelCrossover, wxID_ANY, _("Heuristic crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverHeuristicShareBeta->Wrap( -1 );
	fgSizer15->Add( _staticTextGAsCrossoverHeuristicShareBeta, 0, wxALL, 5 );

	_checkBoxGAsCrossoverHeuristicShareBeta = new wxCheckBox( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( _checkBoxGAsCrossoverHeuristicShareBeta, 0, wxALL, 5 );

	_staticTextGAsCrossoverBinLikeNbPts = new wxStaticText( _panelCrossover, wxID_ANY, _("Binary-like crossover: nb points"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverBinLikeNbPts->Wrap( -1 );
	fgSizer15->Add( _staticTextGAsCrossoverBinLikeNbPts, 0, wxALL, 5 );

	_textCtrlGAsCrossoverBinLikeNbPts = new wxTextCtrl( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer15->Add( _textCtrlGAsCrossoverBinLikeNbPts, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsCrossoverBinLikeShareBeta = new wxStaticText( _panelCrossover, wxID_ANY, _("Binary-like crossover: share beta"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsCrossoverBinLikeShareBeta->Wrap( -1 );
	fgSizer15->Add( _staticTextGAsCrossoverBinLikeShareBeta, 0, wxALL, 5 );

	_checkBoxGAsCrossoverBinLikeShareBeta = new wxCheckBox( _panelCrossover, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer15->Add( _checkBoxGAsCrossoverBinLikeShareBeta, 0, wxALL, 5 );


	bSizer10->Add( fgSizer15, 0, wxEXPAND|wxALL, 5 );


	_panelCrossover->SetSizer( bSizer10 );
	_panelCrossover->Layout();
	bSizer10->Fit( _panelCrossover );
	_notebookGAoptions->AddPage( _panelCrossover, _("Crossover"), false );
	_panelMutation = new wxPanel( _notebookGAoptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );

	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 7, 2, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsMutationsUniformCstProb = new wxStaticText( _panelMutation, wxID_ANY, _("Uniform constant: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsUniformCstProb->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsUniformCstProb, 0, wxALL, 5 );

	_textCtrlGAsMutationsUniformCstProb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsUniformCstProb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalCstProb = new wxStaticText( _panelMutation, wxID_ANY, _("Normal constant: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalCstProb->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsNormalCstProb, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalCstProb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsNormalCstProb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalCstStdDev = new wxStaticText( _panelMutation, wxID_ANY, _("Normal constant: std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalCstStdDev->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsNormalCstStdDev, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalCstStdDev = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsNormalCstStdDev, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsUniformVarMaxGensNb = new wxStaticText( _panelMutation, wxID_ANY, _("Uniform variable: on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsUniformVarMaxGensNb->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsUniformVarMaxGensNb, 0, wxALL, 5 );

	_textCtrlGAsMutationsUniformVarMaxGensNb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsUniformVarMaxGensNb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsUniformVarProbStart = new wxStaticText( _panelMutation, wxID_ANY, _("Uniform variable: starting probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsUniformVarProbStart->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsUniformVarProbStart, 0, wxALL, 5 );

	_textCtrlGAsMutationsUniformVarProbStart = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsUniformVarProbStart, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsUniformVarProbEnd = new wxStaticText( _panelMutation, wxID_ANY, _("Uniform variable: ending probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsUniformVarProbEnd->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsUniformVarProbEnd, 0, wxALL, 5 );

	_textCtrlGAsMutationsUniformVarProbEnd = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsUniformVarProbEnd, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsMultiScaleProb = new wxStaticText( _panelMutation, wxID_ANY, _("Multi-scale: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsMultiScaleProb->Wrap( -1 );
	fgSizer13->Add( _staticTextGAsMutationsMultiScaleProb, 0, wxALL, 5 );

	_textCtrlGAsMutationsMultiScaleProb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer13->Add( _textCtrlGAsMutationsMultiScaleProb, 0, wxRIGHT|wxLEFT, 5 );


	bSizer11->Add( fgSizer13, 1, wxEXPAND|wxALL, 5 );

	wxFlexGridSizer* fgSizer191;
	fgSizer191 = new wxFlexGridSizer( 9, 2, 0, 0 );
	fgSizer191->SetFlexibleDirection( wxBOTH );
	fgSizer191->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextGAsMutationsNormalVarMaxGensNbProb = new wxStaticText( _panelMutation, wxID_ANY, _("Normal variable: prob on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalVarMaxGensNbProb->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNormalVarMaxGensNbProb, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalVarMaxGensNbProb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNormalVarMaxGensNbProb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalVarMaxGensNbStdDev = new wxStaticText( _panelMutation, wxID_ANY, _("Normal variable: std dev on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalVarMaxGensNbStdDev->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNormalVarMaxGensNbStdDev, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalVarMaxGensNbStdDev = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNormalVarMaxGensNbStdDev, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalVarProbStart = new wxStaticText( _panelMutation, wxID_ANY, _("Normal variable: starting probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalVarProbStart->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNormalVarProbStart, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalVarProbStart = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNormalVarProbStart, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalVarProbEnd = new wxStaticText( _panelMutation, wxID_ANY, _("Normal variable: ending probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalVarProbEnd->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNormalVarProbEnd, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalVarProbEnd = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNormalVarProbEnd, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalVarStdDevStart = new wxStaticText( _panelMutation, wxID_ANY, _("Normal variable: starting std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalVarStdDevStart->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNormalVarStdDevStart, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalVarStdDevStart = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNormalVarStdDevStart, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNormalVarStdDevEnd = new wxStaticText( _panelMutation, wxID_ANY, _("Normal variable: ending std dev"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNormalVarStdDevEnd->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNormalVarStdDevEnd, 0, wxALL, 5 );

	_textCtrlGAsMutationsNormalVarStdDevEnd = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNormalVarStdDevEnd, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNonUniformProb = new wxStaticText( _panelMutation, wxID_ANY, _("Non-uniform: probability"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNonUniformProb->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNonUniformProb, 0, wxALL, 5 );

	_textCtrlGAsMutationsNonUniformProb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNonUniformProb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNonUniformGensNb = new wxStaticText( _panelMutation, wxID_ANY, _("Non-uniform: on # generations"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNonUniformGensNb->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNonUniformGensNb, 0, wxALL, 5 );

	_textCtrlGAsMutationsNonUniformGensNb = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNonUniformGensNb, 0, wxRIGHT|wxLEFT, 5 );

	_staticTextGAsMutationsNonUniformMinRate = new wxStaticText( _panelMutation, wxID_ANY, _("Non-uniform: minimum rate"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextGAsMutationsNonUniformMinRate->Wrap( -1 );
	fgSizer191->Add( _staticTextGAsMutationsNonUniformMinRate, 0, wxALL, 5 );

	_textCtrlGAsMutationsNonUniformMinRate = new wxTextCtrl( _panelMutation, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	fgSizer191->Add( _textCtrlGAsMutationsNonUniformMinRate, 0, wxRIGHT|wxLEFT, 5 );


	bSizer11->Add( fgSizer191, 1, wxEXPAND|wxALL, 5 );


	_panelMutation->SetSizer( bSizer11 );
	_panelMutation->Layout();
	bSizer11->Fit( _panelMutation );
	_notebookGAoptions->AddPage( _panelMutation, _("Mutation"), true );

	bSizer14->Add( _notebookGAoptions, 0, wxALL|wxEXPAND, 5 );


	bSizer111->Add( bSizer14, 0, wxEXPAND, 5 );


	_panelGeneticAlgoritms->SetSizer( bSizer111 );
	_panelGeneticAlgoritms->Layout();
	bSizer111->Fit( _panelGeneticAlgoritms );
	_notebookOptions->AddPage( _panelGeneticAlgoritms, _("Genetic algoritms"), false );

	bSizer28->Add( _notebookOptions, 1, wxEXPAND | wxALL, 5 );


	_panelOptions->SetSizer( bSizer28 );
	_panelOptions->Layout();
	bSizer28->Fit( _panelOptions );
	_notebookBase->AddPage( _panelOptions, _("Options"), false );

	bSizer29->Add( _notebookBase, 1, wxALL|wxEXPAND, 5 );

	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );

	_buttonSaveDefault = new wxButton( _panelMain, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( _buttonSaveDefault, 0, 0, 5 );


	bSizer29->Add( bSizer15, 0, wxALIGN_RIGHT|wxBOTTOM|wxRIGHT|wxTOP, 5 );


	_panelMain->SetSizer( bSizer29 );
	_panelMain->Layout();
	bSizer29->Fit( _panelMain );
	bSizer4->Add( _panelMain, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer4 );
	this->Layout();
	bSizer4->Fit( this );
	_menuBar = new wxMenuBar( 0 );
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
	_toolBar->SetToolBitmapSize( wxSize( 32,32 ) );
	_toolBar->Realize();

	_statusBar1 = this->CreateStatusBar( 1, wxSTB_SIZEGRIP, wxID_ANY );

	this->Centre( wxBOTH );

	// Connect Events
	_buttonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameOptimizerVirtual::OnSaveDefault ), NULL, this );
	_menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFramePreferences ), this, _menuItemPreferences->GetId());
	_menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFramePredictandDB ), this, _menuItemBuildPredictandDB->GetId());
	_menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnShowLog ), this, _menuItemShowLog->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel1 ), this, _menuItemLogLevel1->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel2 ), this, _menuItemLogLevel2->GetId());
	_menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OnLogLevel3 ), this, _menuItemLogLevel3->GetId());
	_menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( asFrameOptimizerVirtual::OpenFrameAbout ), this, _menuItemAbout->GetId());
}

asFrameOptimizerVirtual::~asFrameOptimizerVirtual()
{
	// Disconnect Events
	_buttonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFrameOptimizerVirtual::OnSaveDefault ), NULL, this );

}

asFramePreferencesOptimizerVirtual::asFramePreferencesOptimizerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );

	wxBoxSizer* bSizer14;
	bSizer14 = new wxBoxSizer( wxVERTICAL );

	_panelBase = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );

	_notebookBase = new wxNotebook( _panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelGeneralCommon = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer16;
	bSizer16 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer11;
	sbSizer11 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Language") ), wxVERTICAL );

	wxString _choiceLocaleChoices[] = { _("English"), _("French") };
	int _choiceLocaleNChoices = sizeof( _choiceLocaleChoices ) / sizeof( wxString );
	_choiceLocale = new wxChoice( sbSizer11->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceLocaleNChoices, _choiceLocaleChoices, 0 );
	_choiceLocale->SetSelection( 0 );
	sbSizer11->Add( _choiceLocale, 0, wxALL, 5 );

	_staticText59 = new wxStaticText( sbSizer11->GetStaticBox(), wxID_ANY, _("Restart AtmoSwing for the change to take effect."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText59->Wrap( -1 );
	sbSizer11->Add( _staticText59, 0, wxALL, 5 );


	bSizer16->Add( sbSizer11, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer7;
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Logs") ), wxVERTICAL );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );

	_radioBtnLogLevel1 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer25->Add( _radioBtnLogLevel1, 0, wxALL, 5 );

	_radioBtnLogLevel2 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer25->Add( _radioBtnLogLevel2, 0, wxALL, 5 );

	_radioBtnLogLevel3 = new wxRadioButton( sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer25->Add( _radioBtnLogLevel3, 0, wxALL, 5 );


	bSizer20->Add( bSizer25, 1, wxEXPAND, 5 );

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

	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( _panelGeneralCommon, wxID_ANY, _("Directories") ), wxVERTICAL );

	_staticTextArchivePredictorsDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( _staticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );

	_dirPickerArchivePredictors = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( _dirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	_staticTextPredictandDBDir = new wxStaticText( sbSizer18->GetStaticBox(), wxID_ANY, _("Default predictand DB directory"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( _staticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );

	_dirPickerPredictandDB = new wxDirPickerCtrl( sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( _dirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	bSizer16->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );


	_panelGeneralCommon->SetSizer( bSizer16 );
	_panelGeneralCommon->Layout();
	bSizer16->Fit( _panelGeneralCommon );
	_notebookBase->AddPage( _panelGeneralCommon, _("General"), true );
	_panelAdvanced = new wxPanel( _notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );

	_notebookAdvanced = new wxNotebook( _panelAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelGeneral = new wxPanel( _notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer271;
	bSizer271 = new wxBoxSizer( wxVERTICAL );

	wxString _radioBoxGuiChoices[] = { _("Silent (no progressbar, much faster)"), _("Standard (recommanded)"), _("Verbose (not much used)") };
	int _radioBoxGuiNChoices = sizeof( _radioBoxGuiChoices ) / sizeof( wxString );
	_radioBoxGui = new wxRadioBox( _panelGeneral, wxID_ANY, _("GUI options"), wxDefaultPosition, wxDefaultSize, _radioBoxGuiNChoices, _radioBoxGuiChoices, 1, wxRA_SPECIFY_COLS );
	_radioBoxGui->SetSelection( 1 );
	bSizer271->Add( _radioBoxGui, 0, wxALL|wxEXPAND, 5 );

	_checkBoxResponsiveness = new wxCheckBox( _panelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxResponsiveness->SetValue(true);
	bSizer271->Add( _checkBoxResponsiveness, 0, wxALL, 5 );


	_panelGeneral->SetSizer( bSizer271 );
	_panelGeneral->Layout();
	bSizer271->Fit( _panelGeneral );
	_notebookAdvanced->AddPage( _panelGeneral, _("General"), false );
	_panelProcessing = new wxPanel( _notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer1611;
	bSizer1611 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer15;
	sbSizer15 = new wxStaticBoxSizer( new wxStaticBox( _panelProcessing, wxID_ANY, _("Multithreading") ), wxVERTICAL );

	_checkBoxAllowMultithreading = new wxCheckBox( sbSizer15->GetStaticBox(), wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxAllowMultithreading->SetValue(true);
	sbSizer15->Add( _checkBoxAllowMultithreading, 0, wxALL, 5 );

	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextThreadsNb = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Max nb of threads"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextThreadsNb->Wrap( -1 );
	bSizer221->Add( _staticTextThreadsNb, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_textCtrlThreadsNb = new wxTextCtrl( sbSizer15->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 30,-1 ), 0 );
	bSizer221->Add( _textCtrlThreadsNb, 0, wxRIGHT|wxLEFT|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer15->Add( bSizer221, 0, wxEXPAND, 5 );

	wxBoxSizer* bSizer241;
	bSizer241 = new wxBoxSizer( wxHORIZONTAL );

	_staticTextThreadsPriority = new wxStaticText( sbSizer15->GetStaticBox(), wxID_ANY, _("Threads priority"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextThreadsPriority->Wrap( -1 );
	bSizer241->Add( _staticTextThreadsPriority, 0, wxALL, 5 );

	_sliderThreadsPriority = new wxSlider( sbSizer15->GetStaticBox(), wxID_ANY, 95, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL|wxSL_LABELS );
	bSizer241->Add( _sliderThreadsPriority, 1, wxRIGHT|wxLEFT, 5 );


	sbSizer15->Add( bSizer241, 0, wxEXPAND, 5 );


	bSizer1611->Add( sbSizer15, 0, wxALL|wxEXPAND, 5 );

	wxString _radioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Standard (slower)") };
	int _radioBoxProcessingMethodsNChoices = sizeof( _radioBoxProcessingMethodsChoices ) / sizeof( wxString );
	_radioBoxProcessingMethods = new wxRadioBox( _panelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition, wxDefaultSize, _radioBoxProcessingMethodsNChoices, _radioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS );
	_radioBoxProcessingMethods->SetSelection( 0 );
	_radioBoxProcessingMethods->SetToolTip( _("These options don't affect the results, only the processor efficiency.") );

	bSizer1611->Add( _radioBoxProcessingMethods, 0, wxALL|wxEXPAND, 5 );


	_panelProcessing->SetSizer( bSizer1611 );
	_panelProcessing->Layout();
	bSizer1611->Fit( _panelProcessing );
	_notebookAdvanced->AddPage( _panelProcessing, _("Processing"), true );
	_panelUserDirectories = new wxPanel( _notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer24;
	bSizer24 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer411;
	sbSizer411 = new wxStaticBoxSizer( new wxStaticBox( _panelUserDirectories, wxID_ANY, _("Working directories") ), wxVERTICAL );

	_staticTextIntermediateResultsDir = new wxStaticText( sbSizer411->GetStaticBox(), wxID_ANY, _("Directory to save intermediate temporary results"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextIntermediateResultsDir->Wrap( -1 );
	sbSizer411->Add( _staticTextIntermediateResultsDir, 0, wxALL, 5 );

	_dirPickerIntermediateResults = new wxDirPickerCtrl( sbSizer411->GetStaticBox(), wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer411->Add( _dirPickerIntermediateResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );


	bSizer24->Add( sbSizer411, 0, wxEXPAND|wxALL, 5 );

	wxStaticBoxSizer* sbSizer17;
	sbSizer17 = new wxStaticBoxSizer( new wxStaticBox( _panelUserDirectories, wxID_ANY, _("User specific paths") ), wxVERTICAL );

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

	_staticTextLogFileLabels = new wxStaticText( sbSizer17->GetStaticBox(), wxID_ANY, _("Log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextLogFileLabels->Wrap( -1 );
	fgSizer9->Add( _staticTextLogFileLabels, 0, wxALL, 5 );

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


	bSizer24->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );


	_panelUserDirectories->SetSizer( bSizer24 );
	_panelUserDirectories->Layout();
	bSizer24->Fit( _panelUserDirectories );
	_notebookAdvanced->AddPage( _panelUserDirectories, _("User paths"), false );

	bSizer26->Add( _notebookAdvanced, 1, wxEXPAND | wxALL, 5 );


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
	_checkBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	_buttonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::ApplyChanges ), NULL, this );
	_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesOptimizerVirtual::~asFramePreferencesOptimizerVirtual()
{
	// Disconnect Events
	_checkBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	_buttonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::ApplyChanges ), NULL, this );
	_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesOptimizerVirtual::SaveAndClose ), NULL, this );

}
