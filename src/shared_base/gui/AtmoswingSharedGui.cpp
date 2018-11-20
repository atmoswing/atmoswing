///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jan 23 2018)
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

#include "AtmoswingSharedGui.h"

///////////////////////////////////////////////////////////////////////////

asDialogFilePickerVirtual::asDialogFilePickerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizerMain;
	bSizerMain = new wxBoxSizer( wxVERTICAL );
	
	m_staticTextDescription = new wxStaticText( this, wxID_ANY, _("Please select the file."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDescription->Wrap( -1 );
	bSizerMain->Add( m_staticTextDescription, 0, wxALL, 5 );
	
	m_filePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_USE_TEXTCTRL );
	bSizerMain->Add( m_filePicker, 0, wxALL|wxEXPAND, 5 );
	
	m_buttonsConfirmation = new wxStdDialogButtonSizer();
	m_buttonsConfirmationOK = new wxButton( this, wxID_OK );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationOK );
	m_buttonsConfirmationCancel = new wxButton( this, wxID_CANCEL );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationCancel );
	m_buttonsConfirmation->Realize();
	
	bSizerMain->Add( m_buttonsConfirmation, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizerMain );
	this->Layout();
}

asDialogFilePickerVirtual::~asDialogFilePickerVirtual()
{
}

asDialogFileSaverVirtual::asDialogFileSaverVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizerMain;
	bSizerMain = new wxBoxSizer( wxVERTICAL );
	
	m_staticTextDescription = new wxStaticText( this, wxID_ANY, _("Please select the directory and the file name."), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDescription->Wrap( -1 );
	bSizerMain->Add( m_staticTextDescription, 0, wxALL, 5 );
	
	m_filePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizerMain->Add( m_filePicker, 0, wxALL|wxEXPAND, 5 );
	
	m_buttonsConfirmation = new wxStdDialogButtonSizer();
	m_buttonsConfirmationSave = new wxButton( this, wxID_SAVE );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationSave );
	m_buttonsConfirmationCancel = new wxButton( this, wxID_CANCEL );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationCancel );
	m_buttonsConfirmation->Realize();
	
	bSizerMain->Add( m_buttonsConfirmation, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizerMain );
	this->Layout();
}

asDialogFileSaverVirtual::~asDialogFileSaverVirtual()
{
}

asFramePredictandDBVirtual::asFramePredictandDBVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );
	
	wxBoxSizer* bSizer5;
	bSizer5 = new wxBoxSizer( wxVERTICAL );
	
	m_panel2 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer6;
	bSizer6 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticTextDataParam = new wxStaticText( m_panel2, wxID_ANY, _("Predictand parameter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataParam->Wrap( -1 );
	fgSizer2->Add( m_staticTextDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_choiceDataParamChoices[] = { _("Precipitation"), _("Temperature"), _("Lightnings"), _("Other") };
	int m_choiceDataParamNChoices = sizeof( m_choiceDataParamChoices ) / sizeof( wxString );
	m_choiceDataParam = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDataParamNChoices, m_choiceDataParamChoices, 0 );
	m_choiceDataParam->SetSelection( 0 );
	fgSizer2->Add( m_choiceDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_staticTextDataTempResol = new wxStaticText( m_panel2, wxID_ANY, _("Temporal resolution"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataTempResol->Wrap( -1 );
	fgSizer2->Add( m_staticTextDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_choiceDataTempResolChoices[] = { _("24 hours"), _("6 hours"), _("1-hr MTW"), _("3-hr MTW"), _("6-hr MTW"), _("12-hr MTW") };
	int m_choiceDataTempResolNChoices = sizeof( m_choiceDataTempResolChoices ) / sizeof( wxString );
	m_choiceDataTempResol = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDataTempResolNChoices, m_choiceDataTempResolChoices, 0 );
	m_choiceDataTempResol->SetSelection( 0 );
	fgSizer2->Add( m_choiceDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_staticTextDataSpatAggreg = new wxStaticText( m_panel2, wxID_ANY, _("Spatial aggregation"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataSpatAggreg->Wrap( -1 );
	fgSizer2->Add( m_staticTextDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	wxString m_choiceDataSpatAggregChoices[] = { _("Station"), _("Groupment"), _("Catchment") };
	int m_choiceDataSpatAggregNChoices = sizeof( m_choiceDataSpatAggregChoices ) / sizeof( wxString );
	m_choiceDataSpatAggreg = new wxChoice( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_choiceDataSpatAggregNChoices, m_choiceDataSpatAggregChoices, 0 );
	m_choiceDataSpatAggreg->SetSelection( 0 );
	fgSizer2->Add( m_choiceDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	
	bSizer6->Add( fgSizer2, 1, wxTOP|wxBOTTOM|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxVERTICAL );
	
	m_panelDataProcessing = new wxPanel( m_panel2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxStaticBoxSizer* sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( m_panelDataProcessing, wxID_ANY, _("Data processing (precipitation only)") ), wxVERTICAL );
	
	wxBoxSizer* bSizer9;
	bSizer9 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );
	
	m_checkBoxReturnPeriod = new wxCheckBox( sbSizer1->GetStaticBox(), wxID_ANY, _("Normalize by the return period of"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBoxReturnPeriod->SetValue(true); 
	bSizer11->Add( m_checkBoxReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_textCtrlReturnPeriod = new wxTextCtrl( sbSizer1->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 50,-1 ), 0 );
	bSizer11->Add( m_textCtrlReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	m_staticTextYears = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextYears->Wrap( -1 );
	bSizer11->Add( m_staticTextYears, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer9->Add( bSizer11, 1, wxEXPAND, 5 );
	
	m_checkBoxSqrt = new wxCheckBox( sbSizer1->GetStaticBox(), wxID_ANY, _("Process the square root"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer9->Add( m_checkBoxSqrt, 0, wxALL, 5 );
	
	
	sbSizer1->Add( bSizer9, 1, wxEXPAND, 5 );
	
	
	m_panelDataProcessing->SetSizer( sbSizer1 );
	m_panelDataProcessing->Layout();
	sbSizer1->Fit( m_panelDataProcessing );
	bSizer12->Add( m_panelDataProcessing, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer6->Add( bSizer12, 0, wxEXPAND, 5 );
	
	m_staticTextCatalogPath = new wxStaticText( m_panel2, wxID_ANY, _("Select the predictand catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextCatalogPath->Wrap( -1 );
	bSizer6->Add( m_staticTextCatalogPath, 0, wxALL, 5 );
	
	m_filePickerCatalogPath = new wxFilePickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL );
	bSizer6->Add( m_filePickerCatalogPath, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextDataDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the predictand data directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDataDir->Wrap( -1 );
	bSizer6->Add( m_staticTextDataDir, 0, wxALL, 5 );
	
	m_dirPickerDataDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_dirPickerDataDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextPatternsDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the directory containing the file patterns description"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextPatternsDir->Wrap( -1 );
	bSizer6->Add( m_staticTextPatternsDir, 0, wxALL, 5 );
	
	m_dirPickerPatternsDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_dirPickerPatternsDir, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_staticDestinationDir = new wxStaticText( m_panel2, wxID_ANY, _("Select the destination directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticDestinationDir->Wrap( -1 );
	bSizer6->Add( m_staticDestinationDir, 0, wxALL, 5 );
	
	m_dirPickerDestinationDir = new wxDirPickerCtrl( m_panel2, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	bSizer6->Add( m_dirPickerDestinationDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );
	
	m_buttonSaveDefault = new wxButton( m_panel2, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer15->Add( m_buttonSaveDefault, 0, 0, 5 );
	
	m_buttonsConfirmation = new wxStdDialogButtonSizer();
	m_buttonsConfirmationOK = new wxButton( m_panel2, wxID_OK );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationOK );
	m_buttonsConfirmationCancel = new wxButton( m_panel2, wxID_CANCEL );
	m_buttonsConfirmation->AddButton( m_buttonsConfirmationCancel );
	m_buttonsConfirmation->Realize();
	
	bSizer15->Add( m_buttonsConfirmation, 0, 0, 5 );
	
	
	bSizer6->Add( bSizer15, 0, wxALIGN_RIGHT|wxBOTTOM|wxRIGHT|wxTOP, 5 );
	
	
	m_panel2->SetSizer( bSizer6 );
	m_panel2->Layout();
	bSizer6->Fit( m_panel2 );
	bSizer5->Add( m_panel2, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer5 );
	this->Layout();
	bSizer5->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_choiceDataParam->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	m_buttonSaveDefault->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::OnSaveDefault ), NULL, this );
	m_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
}

asFramePredictandDBVirtual::~asFramePredictandDBVirtual()
{
	// Disconnect Events
	m_choiceDataParam->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	m_buttonSaveDefault->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::OnSaveDefault ), NULL, this );
	m_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	m_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
	
}

asFrameAboutVirtual::asFrameAboutVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 350,450 ), wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_Panel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	m_logo = new wxStaticBitmap( m_Panel, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_logo, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 20 );
	
	m_staticTextVersion = new wxStaticText( m_Panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextVersion->Wrap( -1 );
	m_staticTextVersion->SetFont( wxFont( 12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );
	
	bSizer27->Add( m_staticTextVersion, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextChangeset = new wxStaticText( m_Panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextChangeset->Wrap( -1 );
	bSizer27->Add( m_staticTextChangeset, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 20 );
	
	m_notebook = new wxNotebook( m_Panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelCredits = new wxPanel( m_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );
	
	m_staticTextDevelopers = new wxStaticText( m_panelCredits, wxID_ANY, _("Main developer:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDevelopers->Wrap( -1 );
	m_staticTextDevelopers->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextDevelopers, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 20 );
	
	m_staticTextDevelopersList = new wxStaticText( m_panelCredits, wxID_ANY, _("Pascal Horton"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextDevelopersList->Wrap( -1 );
	bSizer28->Add( m_staticTextDevelopersList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextSupervision = new wxStaticText( m_panelCredits, wxID_ANY, _("Developed at:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSupervision->Wrap( -1 );
	m_staticTextSupervision->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextSupervision, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_staticTextSupervisionList = new wxStaticText( m_panelCredits, wxID_ANY, _("University of Lausanne\nTerranum\nUniversity of Bern"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextSupervisionList->Wrap( -1 );
	bSizer28->Add( m_staticTextSupervisionList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextThanks = new wxStaticText( m_panelCredits, wxID_ANY, _("Special thanks to:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThanks->Wrap( -1 );
	m_staticTextThanks->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextThanks, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 10 );
	
	m_staticTextThanksList = new wxStaticText( m_panelCredits, wxID_ANY, _("Charles Obled\nMichel Jaboyedoff\nLucien Schreiber\nRenaud Marty\nRichard Metzger"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextThanksList->Wrap( -1 );
	bSizer28->Add( m_staticTextThanksList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextSpacer = new wxStaticText( m_panelCredits, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSpacer->Wrap( -1 );
	bSizer28->Add( m_staticTextSpacer, 0, wxALL, 5 );
	
	
	m_panelCredits->SetSizer( bSizer28 );
	m_panelCredits->Layout();
	bSizer28->Fit( m_panelCredits );
	m_notebook->AddPage( m_panelCredits, _("Credits"), true );
	m_panelLicense = new wxPanel( m_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );
	
	m_textCtrlLicense = new wxTextCtrl( m_panelLicense, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer32->Add( m_textCtrlLicense, 1, wxEXPAND, 5 );
	
	
	m_panelLicense->SetSizer( bSizer32 );
	m_panelLicense->Layout();
	bSizer32->Fit( m_panelLicense );
	m_notebook->AddPage( m_panelLicense, _("License"), false );
	m_panelLibraries = new wxPanel( m_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer31;
	bSizer31 = new wxBoxSizer( wxVERTICAL );
	
	m_textCtrlLibraries = new wxTextCtrl( m_panelLibraries, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer31->Add( m_textCtrlLibraries, 1, wxEXPAND, 5 );
	
	
	m_panelLibraries->SetSizer( bSizer31 );
	m_panelLibraries->Layout();
	bSizer31->Fit( m_panelLibraries );
	m_notebook->AddPage( m_panelLibraries, _("Libraries"), false );
	
	bSizer27->Add( m_notebook, 1, wxEXPAND, 5 );
	
	
	m_Panel->SetSizer( bSizer27 );
	m_Panel->Layout();
	bSizer27->Fit( m_Panel );
	bSizer3->Add( m_Panel, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer3 );
	this->Layout();
	
	this->Centre( wxBOTH );
}

asFrameAboutVirtual::~asFrameAboutVirtual()
{
}
