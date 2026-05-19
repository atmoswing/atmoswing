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

#include "AtmoSwingSharedGui.h"

///////////////////////////////////////////////////////////////////////////

asDialogFilePickerVirtual::asDialogFilePickerVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );

	wxBoxSizer* bSizerMain;
	bSizerMain = new wxBoxSizer( wxVERTICAL );

	_staticTextDescription = new wxStaticText( this, wxID_ANY, _("Please select the file."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDescription->Wrap( -1 );
	bSizerMain->Add( _staticTextDescription, 0, wxALL, 5 );

	_filePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_USE_TEXTCTRL );
	bSizerMain->Add( _filePicker, 0, wxALL|wxEXPAND, 5 );

	_buttonsConfirmation = new wxStdDialogButtonSizer();
	_buttonsConfirmationOK = new wxButton( this, wxID_OK );
	_buttonsConfirmation->AddButton( _buttonsConfirmationOK );
	_buttonsConfirmationCancel = new wxButton( this, wxID_CANCEL );
	_buttonsConfirmation->AddButton( _buttonsConfirmationCancel );
	_buttonsConfirmation->Realize();

	bSizerMain->Add( _buttonsConfirmation, 0, wxEXPAND, 5 );


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

	_staticTextDescription = new wxStaticText( this, wxID_ANY, _("Please select the directory and the file name."), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDescription->Wrap( -1 );
	bSizerMain->Add( _staticTextDescription, 0, wxALL, 5 );

	_filePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizerMain->Add( _filePicker, 0, wxALL|wxEXPAND, 5 );

	_buttonsConfirmation = new wxStdDialogButtonSizer();
	_buttonsConfirmationSave = new wxButton( this, wxID_SAVE );
	_buttonsConfirmation->AddButton( _buttonsConfirmationSave );
	_buttonsConfirmationCancel = new wxButton( this, wxID_CANCEL );
	_buttonsConfirmation->AddButton( _buttonsConfirmationCancel );
	_buttonsConfirmation->Realize();

	bSizerMain->Add( _buttonsConfirmation, 0, wxEXPAND, 5 );


	this->SetSizer( bSizerMain );
	this->Layout();
}

asDialogFileSaverVirtual::~asDialogFileSaverVirtual()
{
}

asFramePredictandDBVirtual::asFramePredictandDBVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 400,400 ), wxDefaultSize );

	_sizerMain = new wxBoxSizer( wxVERTICAL );

	_panelMain = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	_sizerMainPanel = new wxBoxSizer( wxVERTICAL );

	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	_staticTextDataParam = new wxStaticText( _panelMain, wxID_ANY, _("Predictand parameter"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDataParam->Wrap( -1 );
	fgSizer2->Add( _staticTextDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString _choiceDataParamChoices[] = { _("Precipitation"), _("Temperature"), _("Lightning"), _("Other") };
	int _choiceDataParamNChoices = sizeof( _choiceDataParamChoices ) / sizeof( wxString );
	_choiceDataParam = new wxChoice( _panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceDataParamNChoices, _choiceDataParamChoices, 0 );
	_choiceDataParam->SetSelection( 1 );
	fgSizer2->Add( _choiceDataParam, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	_staticTextDataTempResol = new wxStaticText( _panelMain, wxID_ANY, _("Temporal resolution"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDataTempResol->Wrap( -1 );
	fgSizer2->Add( _staticTextDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString _choiceDataTempResolChoices[] = { _("24 hours"), _("6 hours"), _("1-hr MTW"), _("3-hr MTW"), _("6-hr MTW"), _("12-hr MTW") };
	int _choiceDataTempResolNChoices = sizeof( _choiceDataTempResolChoices ) / sizeof( wxString );
	_choiceDataTempResol = new wxChoice( _panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceDataTempResolNChoices, _choiceDataTempResolChoices, 0 );
	_choiceDataTempResol->SetSelection( 0 );
	fgSizer2->Add( _choiceDataTempResol, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	_staticTextDataSpatAggreg = new wxStaticText( _panelMain, wxID_ANY, _("Spatial aggregation"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDataSpatAggreg->Wrap( -1 );
	fgSizer2->Add( _staticTextDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString _choiceDataSpatAggregChoices[] = { _("Station"), _("Groupment"), _("Catchment"), _("Region") };
	int _choiceDataSpatAggregNChoices = sizeof( _choiceDataSpatAggregChoices ) / sizeof( wxString );
	_choiceDataSpatAggreg = new wxChoice( _panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceDataSpatAggregNChoices, _choiceDataSpatAggregChoices, 0 );
	_choiceDataSpatAggreg->SetSelection( 0 );
	fgSizer2->Add( _choiceDataSpatAggreg, 0, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );


	_sizerMainPanel->Add( fgSizer2, 1, wxTOP|wxBOTTOM|wxEXPAND, 5 );

	_sizerProcessing = new wxBoxSizer( wxVERTICAL );


	_sizerMainPanel->Add( _sizerProcessing, 0, 0, 5 );

	_staticTextCatalogPath = new wxStaticText( _panelMain, wxID_ANY, _("Select the predictand catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextCatalogPath->Wrap( -1 );
	_sizerMainPanel->Add( _staticTextCatalogPath, 0, wxALL, 5 );

	_filePickerCatalogPath = new wxFilePickerCtrl( _panelMain, wxID_ANY, wxEmptyString, _("Select a file"), _("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL );
	_sizerMainPanel->Add( _filePickerCatalogPath, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextDataDir = new wxStaticText( _panelMain, wxID_ANY, _("Select the predictand data directory"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDataDir->Wrap( -1 );
	_sizerMainPanel->Add( _staticTextDataDir, 0, wxALL, 5 );

	_dirPickerDataDir = new wxDirPickerCtrl( _panelMain, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerMainPanel->Add( _dirPickerDataDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_staticTextPatternsDir = new wxStaticText( _panelMain, wxID_ANY, _("Select the directory containing the file patterns description"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextPatternsDir->Wrap( -1 );
	_sizerMainPanel->Add( _staticTextPatternsDir, 0, wxALL, 5 );

	_dirPickerPatternsDir = new wxDirPickerCtrl( _panelMain, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerMainPanel->Add( _dirPickerPatternsDir, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );

	_staticDestinationDir = new wxStaticText( _panelMain, wxID_ANY, _("Select the destination directory"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticDestinationDir->Wrap( -1 );
	_sizerMainPanel->Add( _staticDestinationDir, 0, wxALL, 5 );

	_dirPickerDestinationDir = new wxDirPickerCtrl( _panelMain, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	_sizerMainPanel->Add( _dirPickerDestinationDir, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxHORIZONTAL );

	_buttonsConfirmation = new wxStdDialogButtonSizer();
	_buttonsConfirmationOK = new wxButton( _panelMain, wxID_OK );
	_buttonsConfirmation->AddButton( _buttonsConfirmationOK );
	_buttonsConfirmationCancel = new wxButton( _panelMain, wxID_CANCEL );
	_buttonsConfirmation->AddButton( _buttonsConfirmationCancel );
	_buttonsConfirmation->Realize();

	bSizer15->Add( _buttonsConfirmation, 0, 0, 5 );


	_sizerMainPanel->Add( bSizer15, 0, wxALIGN_RIGHT|wxBOTTOM|wxRIGHT|wxTOP, 5 );


	_panelMain->SetSizer( _sizerMainPanel );
	_panelMain->Layout();
	_sizerMainPanel->Fit( _panelMain );
	_sizerMain->Add( _panelMain, 1, wxEXPAND, 5 );


	this->SetSizer( _sizerMain );
	this->Layout();
	_sizerMain->Fit( this );

	this->Centre( wxBOTH );

	// Connect Events
	_choiceDataParam->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	_buttonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );
}

asFramePredictandDBVirtual::~asFramePredictandDBVirtual()
{
	// Disconnect Events
	_choiceDataParam->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( asFramePredictandDBVirtual::OnDataSelection ), NULL, this );
	_buttonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::CloseFrame ), NULL, this );
	_buttonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePredictandDBVirtual::BuildDatabase ), NULL, this );

}

asPanelProcessingPrecipitation::asPanelProcessingPrecipitation( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer18;
	bSizer18 = new wxBoxSizer( wxVERTICAL );

	_staticText22 = new wxStaticText( this, wxID_ANY, _("Precipitation data normalization"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText22->Wrap( -1 );
	bSizer18->Add( _staticText22, 0, 0, 5 );

	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxVERTICAL );

	wxBoxSizer* bSizer11;
	bSizer11 = new wxBoxSizer( wxHORIZONTAL );

	_checkBoxReturnPeriod = new wxCheckBox( this, wxID_ANY, _("Normalize by the return period of"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxReturnPeriod->SetValue(true);
	bSizer11->Add( _checkBoxReturnPeriod, 0, wxALIGN_CENTER_VERTICAL|wxTOP|wxBOTTOM|wxRIGHT, 5 );

	_textCtrlReturnPeriod = new wxTextCtrl( this, wxID_ANY, _("10"), wxDefaultPosition, wxSize( 50,-1 ), 0 );
	bSizer11->Add( _textCtrlReturnPeriod, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	_staticTextYears = new wxStaticText( this, wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextYears->Wrap( -1 );
	bSizer11->Add( _staticTextYears, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );


	bSizer20->Add( bSizer11, 0, 0, 5 );

	_checkBoxSqrt = new wxCheckBox( this, wxID_ANY, _("Process the square root"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer20->Add( _checkBoxSqrt, 0, wxTOP|wxBOTTOM|wxRIGHT, 5 );


	bSizer18->Add( bSizer20, 0, wxLEFT|wxEXPAND, 15 );


	this->SetSizer( bSizer18 );
	this->Layout();
	bSizer18->Fit( this );
}

asPanelProcessingPrecipitation::~asPanelProcessingPrecipitation()
{
}

asPanelProcessingLightning::asPanelProcessingLightning( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name ) : wxPanel( parent, id, pos, size, style, name )
{
	wxBoxSizer* bSizer19;
	bSizer19 = new wxBoxSizer( wxVERTICAL );

	_staticText23 = new wxStaticText( this, wxID_ANY, _("Lightning data normalization"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticText23->Wrap( -1 );
	bSizer19->Add( _staticText23, 0, 0, 5 );

	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );

	_checkBoxLog = new wxCheckBox( this, wxID_ANY, _("Process log10(nb+1)"), wxDefaultPosition, wxDefaultSize, 0 );
	_checkBoxLog->SetValue(true);
	bSizer21->Add( _checkBoxLog, 0, wxTOP|wxBOTTOM|wxRIGHT, 9 );


	bSizer19->Add( bSizer21, 0, wxLEFT, 15 );


	this->SetSizer( bSizer19 );
	this->Layout();
	bSizer19->Fit( this );
}

asPanelProcessingLightning::~asPanelProcessingLightning()
{
}

asFrameAboutVirtual::asFrameAboutVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 350,450 ), wxDefaultSize );

	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );

	_panel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );

	_logo = new wxStaticBitmap( _panel, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( _logo, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 20 );

	_staticTextVersion = new wxStaticText( _panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	_staticTextVersion->Wrap( -1 );
	_staticTextVersion->SetFont( wxFont( 12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL, false, wxEmptyString ) );

	bSizer27->Add( _staticTextVersion, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );

	_notebook = new wxNotebook( _panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	_panelCredits = new wxPanel( _notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );

	_staticTextDevelopers = new wxStaticText( _panelCredits, wxID_ANY, _("Main developer:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextDevelopers->Wrap( -1 );
	_staticTextDevelopers->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	bSizer28->Add( _staticTextDevelopers, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 20 );

	_staticTextDevelopersList = new wxStaticText( _panelCredits, wxID_ANY, _("Pascal Horton"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	_staticTextDevelopersList->Wrap( -1 );
	bSizer28->Add( _staticTextDevelopersList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	_staticTextSupervision = new wxStaticText( _panelCredits, wxID_ANY, _("Developed at:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextSupervision->Wrap( -1 );
	_staticTextSupervision->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	bSizer28->Add( _staticTextSupervision, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );

	_staticTextSupervisionList = new wxStaticText( _panelCredits, wxID_ANY, _("University of Lausanne\nTerranum\nUniversity of Bern"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	_staticTextSupervisionList->Wrap( -1 );
	bSizer28->Add( _staticTextSupervisionList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	_staticTextThanks = new wxStaticText( _panelCredits, wxID_ANY, _("Special thanks to:"), wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextThanks->Wrap( -1 );
	_staticTextThanks->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD, false, wxEmptyString ) );

	bSizer28->Add( _staticTextThanks, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 10 );

	_staticTextThanksList = new wxStaticText( _panelCredits, wxID_ANY, _("Charles Obled\nMichel Jaboyedoff\nLucien Schreiber\nRenaud Marty\nRichard Metzger"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTER_HORIZONTAL );
	_staticTextThanksList->Wrap( -1 );
	bSizer28->Add( _staticTextThanksList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );

	_staticTextSpacer = new wxStaticText( _panelCredits, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	_staticTextSpacer->Wrap( -1 );
	bSizer28->Add( _staticTextSpacer, 0, wxALL, 5 );


	_panelCredits->SetSizer( bSizer28 );
	_panelCredits->Layout();
	bSizer28->Fit( _panelCredits );
	_notebook->AddPage( _panelCredits, _("Credits"), true );
	_panelLicense = new wxPanel( _notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );

	_textCtrlLicense = new wxTextCtrl( _panelLicense, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer32->Add( _textCtrlLicense, 1, wxEXPAND, 5 );


	_panelLicense->SetSizer( bSizer32 );
	_panelLicense->Layout();
	bSizer32->Fit( _panelLicense );
	_notebook->AddPage( _panelLicense, _("License"), false );

	bSizer27->Add( _notebook, 1, wxEXPAND, 5 );


	_panel->SetSizer( bSizer27 );
	_panel->Layout();
	bSizer27->Fit( _panel );
	bSizer3->Add( _panel, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer3 );
	this->Layout();

	this->Centre( wxBOTH );
}

asFrameAboutVirtual::~asFrameAboutVirtual()
{
}
