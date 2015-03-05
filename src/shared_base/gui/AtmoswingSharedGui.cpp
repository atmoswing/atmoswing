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

asFrameXmlEditorVirtual::asFrameXmlEditorVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxVERTICAL );
	
	m_toolBar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL ); 
	m_toolSave = m_toolBar->AddTool( wxID_ANY, _("save"), wxNullBitmap, wxNullBitmap, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL ); 
	
	m_toolBar->Realize(); 
	
	bSizer2->Add( m_toolBar, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer2 );
	this->Layout();
	
	this->Centre( wxBOTH );
}

asFrameXmlEditorVirtual::~asFrameXmlEditorVirtual()
{
}

asFrameAboutVirtual::asFrameAboutVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 350,-1 ), wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_panel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	m_logo = new wxStaticBitmap( m_panel, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_logo, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 20 );
	
	m_staticTextVersion = new wxStaticText( m_panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextVersion->Wrap( -1 );
	m_staticTextVersion->SetFont( wxFont( 12, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer27->Add( m_staticTextVersion, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_staticTextChangeset = new wxStaticText( m_panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextChangeset->Wrap( -1 );
	bSizer27->Add( m_staticTextChangeset, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 20 );
	
	m_notebook = new wxNotebook( m_panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_panelCredits = new wxPanel( m_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );
	
	m_staticTextDevelopers = new wxStaticText( m_panelCredits, wxID_ANY, _("Developed by:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextDevelopers->Wrap( -1 );
	m_staticTextDevelopers->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextDevelopers, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 20 );
	
	m_staticTextDevelopersList = new wxStaticText( m_panelCredits, wxID_ANY, _("Pascal Horton (University of Lausanne)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextDevelopersList->Wrap( -1 );
	bSizer28->Add( m_staticTextDevelopersList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextSupervision = new wxStaticText( m_panelCredits, wxID_ANY, _("Under the supervision of:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSupervision->Wrap( -1 );
	m_staticTextSupervision->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextSupervision, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_staticTextSupervisionList = new wxStaticText( m_panelCredits, wxID_ANY, _("Michel Jaboyedoff (Unil) and Charles Obled (INPG)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextSupervisionList->Wrap( -1 );
	bSizer28->Add( m_staticTextSupervisionList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextThanks = new wxStaticText( m_panelCredits, wxID_ANY, _("Special thanks to:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThanks->Wrap( -1 );
	m_staticTextThanks->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextThanks, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 10 );
	
	m_staticTextThanksList = new wxStaticText( m_panelCredits, wxID_ANY, _("Lucien Schreiber (CREALP), Richard Metzger (Terr@num)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextThanksList->Wrap( -1 );
	bSizer28->Add( m_staticTextThanksList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_staticTextOtherCredits = new wxStaticText( m_panelCredits, wxID_ANY, _("Other credits:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextOtherCredits->Wrap( -1 );
	m_staticTextOtherCredits->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_staticTextOtherCredits, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_staticTextOtherCreditsList = new wxStaticText( m_panelCredits, wxID_ANY, _("Icons by FatCow Web Hosting (http://www.fatcow.com/)\nand Gasyoun (http://twitter.com/gasyoun)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_staticTextOtherCreditsList->Wrap( -1 );
	bSizer28->Add( m_staticTextOtherCreditsList, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
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
	
	bSizer27->Add( m_notebook, 1, wxEXPAND | wxALL, 5 );
	
	
	m_panel->SetSizer( bSizer27 );
	m_panel->Layout();
	bSizer27->Fit( m_panel );
	bSizer3->Add( m_panel, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer3 );
	this->Layout();
	
	this->Centre( wxBOTH );
}

asFrameAboutVirtual::~asFrameAboutVirtual()
{
}
