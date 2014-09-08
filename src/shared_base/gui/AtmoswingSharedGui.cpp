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
	
	m_StaticTextDescription = new wxStaticText( this, wxID_ANY, _("Please select the file."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDescription->Wrap( -1 );
	bSizerMain->Add( m_StaticTextDescription, 0, wxALL, 5 );
	
	m_FilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_USE_TEXTCTRL );
	bSizerMain->Add( m_FilePicker, 0, wxALL|wxEXPAND, 5 );
	
	m_ButtonsConfirmation = new wxStdDialogButtonSizer();
	m_ButtonsConfirmationOK = new wxButton( this, wxID_OK );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationOK );
	m_ButtonsConfirmationCancel = new wxButton( this, wxID_CANCEL );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationCancel );
	m_ButtonsConfirmation->Realize();
	
	bSizerMain->Add( m_ButtonsConfirmation, 0, wxEXPAND, 5 );
	
	
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
	
	m_StaticTextDescription = new wxStaticText( this, wxID_ANY, _("Please select the directory and the file name."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDescription->Wrap( -1 );
	bSizerMain->Add( m_StaticTextDescription, 0, wxALL, 5 );
	
	m_FilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	bSizerMain->Add( m_FilePicker, 0, wxALL|wxEXPAND, 5 );
	
	m_ButtonsConfirmation = new wxStdDialogButtonSizer();
	m_ButtonsConfirmationSave = new wxButton( this, wxID_SAVE );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationSave );
	m_ButtonsConfirmationCancel = new wxButton( this, wxID_CANCEL );
	m_ButtonsConfirmation->AddButton( m_ButtonsConfirmationCancel );
	m_ButtonsConfirmation->Realize();
	
	bSizerMain->Add( m_ButtonsConfirmation, 0, wxEXPAND, 5 );
	
	
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
	
	m_ToolBar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL ); 
	m_ToolSave = m_ToolBar->AddTool( wxID_ANY, _("save"), wxNullBitmap, wxNullBitmap, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL ); 
	
	m_ToolBar->Realize(); 
	
	bSizer2->Add( m_ToolBar, 0, wxEXPAND, 5 );
	
	
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
	
	m_Panel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	m_Logo = new wxStaticBitmap( m_Panel, wxID_ANY, wxNullBitmap, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_Logo, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 20 );
	
	m_StaticTextVersion = new wxStaticText( m_Panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextVersion->Wrap( -1 );
	m_StaticTextVersion->SetFont( wxFont( 12, 70, 90, 90, false, wxEmptyString ) );
	
	bSizer27->Add( m_StaticTextVersion, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextChangeset = new wxStaticText( m_Panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextChangeset->Wrap( -1 );
	bSizer27->Add( m_StaticTextChangeset, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 20 );
	
	m_Notebook = new wxNotebook( m_Panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelCredits = new wxPanel( m_Notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxVERTICAL );
	
	m_StaticTextDevelopers = new wxStaticText( m_PanelCredits, wxID_ANY, _("Developed by:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDevelopers->Wrap( -1 );
	m_StaticTextDevelopers->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_StaticTextDevelopers, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 20 );
	
	m_StaticTextDevelopersList = new wxStaticText( m_PanelCredits, wxID_ANY, _("Pascal Horton (University of Lausanne)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextDevelopersList->Wrap( -1 );
	bSizer28->Add( m_StaticTextDevelopersList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextSupervision = new wxStaticText( m_PanelCredits, wxID_ANY, _("Under the supervision of:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextSupervision->Wrap( -1 );
	m_StaticTextSupervision->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_StaticTextSupervision, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_StaticTextSupervisionList = new wxStaticText( m_PanelCredits, wxID_ANY, _("Michel Jaboyedoff (Unil) and Charles Obled (INPG)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextSupervisionList->Wrap( -1 );
	bSizer28->Add( m_StaticTextSupervisionList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextThanks = new wxStaticText( m_PanelCredits, wxID_ANY, _("Special thanks to:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextThanks->Wrap( -1 );
	m_StaticTextThanks->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_StaticTextThanks, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 10 );
	
	m_StaticTextThanksList = new wxStaticText( m_PanelCredits, wxID_ANY, _("Lucien Schreiber (CREALP), Richard Metzger (Terr@num)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextThanksList->Wrap( -1 );
	bSizer28->Add( m_StaticTextThanksList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextOtherCredits = new wxStaticText( m_PanelCredits, wxID_ANY, _("Other credits:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextOtherCredits->Wrap( -1 );
	m_StaticTextOtherCredits->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer28->Add( m_StaticTextOtherCredits, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_StaticTextOtherCreditsList = new wxStaticText( m_PanelCredits, wxID_ANY, _("Icons by FatCow Web Hosting (http://www.fatcow.com/)\nand Gasyoun (http://twitter.com/gasyoun)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextOtherCreditsList->Wrap( -1 );
	bSizer28->Add( m_StaticTextOtherCreditsList, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	m_staticTextSpacer = new wxStaticText( m_PanelCredits, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticTextSpacer->Wrap( -1 );
	bSizer28->Add( m_staticTextSpacer, 0, wxALL, 5 );
	
	
	m_PanelCredits->SetSizer( bSizer28 );
	m_PanelCredits->Layout();
	bSizer28->Fit( m_PanelCredits );
	m_Notebook->AddPage( m_PanelCredits, _("Credits"), true );
	m_PanelLicense = new wxPanel( m_Notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxVERTICAL );
	
	m_TextCtrlLicense = new wxTextCtrl( m_PanelLicense, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer32->Add( m_TextCtrlLicense, 1, wxEXPAND, 5 );
	
	
	m_PanelLicense->SetSizer( bSizer32 );
	m_PanelLicense->Layout();
	bSizer32->Fit( m_PanelLicense );
	m_Notebook->AddPage( m_PanelLicense, _("License"), false );
	m_PanelLibraries = new wxPanel( m_Notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer31;
	bSizer31 = new wxBoxSizer( wxVERTICAL );
	
	m_TextCtrlLibraries = new wxTextCtrl( m_PanelLibraries, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer31->Add( m_TextCtrlLibraries, 1, wxEXPAND, 5 );
	
	
	m_PanelLibraries->SetSizer( bSizer31 );
	m_PanelLibraries->Layout();
	bSizer31->Fit( m_PanelLibraries );
	m_Notebook->AddPage( m_PanelLibraries, _("Libraries"), false );
	
	bSizer27->Add( m_Notebook, 1, wxEXPAND | wxALL, 5 );
	
	
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
