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
	
	m_FilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxSize( -1,-1 ), wxFLP_DEFAULT_STYLE );
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
	m_ToolBar->AddTool( wxID_ANY, _("save"), wxNullBitmap, wxNullBitmap, wxITEM_NORMAL, wxEmptyString, wxEmptyString, NULL ); 
	
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
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxVERTICAL );
	
	m_Panel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer4;
	bSizer4 = new wxBoxSizer( wxVERTICAL );
	
	m_StaticTextTitle = new wxStaticText( m_Panel, wxID_ANY, _("Atmoswing"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextTitle->Wrap( -1 );
	m_StaticTextTitle->SetFont( wxFont( 15, 70, 90, 92, false, wxT("Arial") ) );
	
	bSizer4->Add( m_StaticTextTitle, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 20 );
	
	m_StaticTextVersion = new wxStaticText( m_Panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextVersion->Wrap( -1 );
	bSizer4->Add( m_StaticTextVersion, 0, wxALIGN_CENTER_HORIZONTAL|wxBOTTOM|wxRIGHT|wxLEFT, 20 );
	
	m_bitmap1 = new wxStaticBitmap( m_Panel, wxID_ANY, wxBitmap( wxT("../../../art/icon/icon.png"), wxBITMAP_TYPE_ANY ), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer4->Add( m_bitmap1, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextDevelopers = new wxStaticText( m_Panel, wxID_ANY, _("Developed by:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextDevelopers->Wrap( -1 );
	m_StaticTextDevelopers->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer4->Add( m_StaticTextDevelopers, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 20 );
	
	m_StaticTextDevelopersList = new wxStaticText( m_Panel, wxID_ANY, _("University of Lausanne - Pascal Horton"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextDevelopersList->Wrap( -1 );
	bSizer4->Add( m_StaticTextDevelopersList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextSupervision = new wxStaticText( m_Panel, wxID_ANY, _("Under the supervision of:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextSupervision->Wrap( -1 );
	m_StaticTextSupervision->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer4->Add( m_StaticTextSupervision, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_StaticTextSupervisionList = new wxStaticText( m_Panel, wxID_ANY, _("Michel Jaboyedoff (Unil) and Charles Obled (INPG)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextSupervisionList->Wrap( -1 );
	bSizer4->Add( m_StaticTextSupervisionList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextThanks = new wxStaticText( m_Panel, wxID_ANY, _("Special thanks to:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextThanks->Wrap( -1 );
	m_StaticTextThanks->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer4->Add( m_StaticTextThanks, 0, wxTOP|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 10 );
	
	m_StaticTextThanksList = new wxStaticText( m_Panel, wxID_ANY, _("Lucien Schreiber (Crealp), Richard Metzger (Terr@num)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextThanksList->Wrap( -1 );
	bSizer4->Add( m_StaticTextThanksList, 0, wxALL|wxALIGN_CENTER_HORIZONTAL, 5 );
	
	m_StaticTextOtherCredits = new wxStaticText( m_Panel, wxID_ANY, _("Other credits:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextOtherCredits->Wrap( -1 );
	m_StaticTextOtherCredits->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer4->Add( m_StaticTextOtherCredits, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_StaticTextOtherCreditsList = new wxStaticText( m_Panel, wxID_ANY, _("Icons by FatCow Web Hosting (http://www.fatcow.com/)\nand Gasyoun (http://twitter.com/gasyoun)"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	m_StaticTextOtherCreditsList->Wrap( -1 );
	bSizer4->Add( m_StaticTextOtherCreditsList, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	m_StaticTextLibraries = new wxStaticText( m_Panel, wxID_ANY, _("Libraries:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLibraries->Wrap( -1 );
	m_StaticTextLibraries->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	bSizer4->Add( m_StaticTextLibraries, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxRIGHT|wxLEFT, 10 );
	
	m_GridSizer = new wxGridSizer( 5, 2, 0, 0 );
	
	
	bSizer4->Add( m_GridSizer, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP, 5 );
	
	m_StaticTextSpacer = new wxStaticText( m_Panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextSpacer->Wrap( -1 );
	bSizer4->Add( m_StaticTextSpacer, 0, wxALL, 5 );
	
	
	m_Panel->SetSizer( bSizer4 );
	m_Panel->Layout();
	bSizer4->Fit( m_Panel );
	bSizer3->Add( m_Panel, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer3 );
	this->Layout();
	bSizer3->Fit( this );
	
	this->Centre( wxBOTH );
}

asFrameAboutVirtual::~asFrameAboutVirtual()
{
}

asFramePreferencesVirtual::asFramePreferencesVirtual( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
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
	sbSizer7 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Logs forecaster") ), wxVERTICAL );
	
	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );
	
	wxString m_RadioBoxLogFLevelChoices[] = { _("Errors only (recommanded)"), _("Errors and warnings"), _("Verbose") };
	int m_RadioBoxLogFLevelNChoices = sizeof( m_RadioBoxLogFLevelChoices ) / sizeof( wxString );
	m_RadioBoxLogFLevel = new wxRadioBox( m_PanelGeneralCommon, wxID_ANY, _("Level"), wxDefaultPosition, wxDefaultSize, m_RadioBoxLogFLevelNChoices, m_RadioBoxLogFLevelChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxLogFLevel->SetSelection( 0 );
	bSizer20->Add( m_RadioBoxLogFLevel, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer8;
	sbSizer8 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Outputs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer21;
	bSizer21 = new wxBoxSizer( wxVERTICAL );
	
	m_CheckBoxDisplayLogFWindow = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxDisplayLogFWindow->SetValue(true); 
	bSizer21->Add( m_CheckBoxDisplayLogFWindow, 0, wxALL, 5 );
	
	m_CheckBoxSaveLogFFile = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxSaveLogFFile->SetValue(true); 
	m_CheckBoxSaveLogFFile->Enable( false );
	
	bSizer21->Add( m_CheckBoxSaveLogFFile, 0, wxALL, 5 );
	
	
	sbSizer8->Add( bSizer21, 1, wxEXPAND, 5 );
	
	
	bSizer20->Add( sbSizer8, 1, wxALL|wxEXPAND, 5 );
	
	
	sbSizer7->Add( bSizer20, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer7, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer71;
	sbSizer71 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Logs viewer") ), wxVERTICAL );
	
	wxBoxSizer* bSizer201;
	bSizer201 = new wxBoxSizer( wxHORIZONTAL );
	
	wxString m_RadioBoxLogVLevelChoices[] = { _("Errors only (recommanded)"), _("Errors and warnings"), _("Verbose") };
	int m_RadioBoxLogVLevelNChoices = sizeof( m_RadioBoxLogVLevelChoices ) / sizeof( wxString );
	m_RadioBoxLogVLevel = new wxRadioBox( m_PanelGeneralCommon, wxID_ANY, _("Level"), wxDefaultPosition, wxDefaultSize, m_RadioBoxLogVLevelNChoices, m_RadioBoxLogVLevelChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxLogVLevel->SetSelection( 0 );
	bSizer201->Add( m_RadioBoxLogVLevel, 1, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer81;
	sbSizer81 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Outputs") ), wxVERTICAL );
	
	wxBoxSizer* bSizer211;
	bSizer211 = new wxBoxSizer( wxVERTICAL );
	
	m_CheckBoxDisplayLogVWindow = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Display window"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxDisplayLogVWindow->SetValue(true); 
	bSizer211->Add( m_CheckBoxDisplayLogVWindow, 0, wxALL, 5 );
	
	m_CheckBoxSaveLogVFile = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Save to a file"), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxSaveLogVFile->SetValue(true); 
	m_CheckBoxSaveLogVFile->Enable( false );
	
	bSizer211->Add( m_CheckBoxSaveLogVFile, 0, wxALL, 5 );
	
	
	sbSizer81->Add( bSizer211, 1, wxEXPAND, 5 );
	
	
	bSizer201->Add( sbSizer81, 1, wxEXPAND|wxALL, 5 );
	
	
	sbSizer71->Add( bSizer201, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer71, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer14;
	sbSizer14 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneralCommon, wxID_ANY, _("Proxy configuration") ), wxVERTICAL );
	
	m_CheckBoxProxy = new wxCheckBox( m_PanelGeneralCommon, wxID_ANY, _("Internet connection uses a proxy"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer14->Add( m_CheckBoxProxy, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextProxyAddress = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Proxy address"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyAddress->Wrap( -1 );
	bSizer34->Add( m_StaticTextProxyAddress, 0, wxALL, 5 );
	
	m_TextCtrlProxyAddress = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 180,-1 ), 0 );
	m_TextCtrlProxyAddress->SetMaxLength( 0 ); 
	bSizer34->Add( m_TextCtrlProxyAddress, 1, wxALL, 5 );
	
	m_StaticTextProxyPort = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Port"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyPort->Wrap( -1 );
	bSizer34->Add( m_StaticTextProxyPort, 0, wxALL, 5 );
	
	m_TextCtrlProxyPort = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlProxyPort->SetMaxLength( 0 ); 
	bSizer34->Add( m_TextCtrlProxyPort, 0, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer34, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxHORIZONTAL );
	
	m_StaticTextProxyUser = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Username"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyUser->Wrap( -1 );
	bSizer35->Add( m_StaticTextProxyUser, 0, wxALL, 5 );
	
	m_TextCtrlProxyUser = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_TextCtrlProxyUser->SetMaxLength( 0 ); 
	bSizer35->Add( m_TextCtrlProxyUser, 1, wxALL, 5 );
	
	m_StaticTextProxyPasswd = new wxStaticText( m_PanelGeneralCommon, wxID_ANY, _("Password"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextProxyPasswd->Wrap( -1 );
	bSizer35->Add( m_StaticTextProxyPasswd, 0, wxALL, 5 );
	
	m_TextCtrlProxyPasswd = new wxTextCtrl( m_PanelGeneralCommon, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD );
	m_TextCtrlProxyPasswd->SetMaxLength( 0 ); 
	bSizer35->Add( m_TextCtrlProxyPasswd, 1, wxALL, 5 );
	
	
	sbSizer14->Add( bSizer35, 1, wxEXPAND, 5 );
	
	
	bSizer16->Add( sbSizer14, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelGeneralCommon->SetSizer( bSizer16 );
	m_PanelGeneralCommon->Layout();
	bSizer16->Fit( m_PanelGeneralCommon );
	m_NotebookBase->AddPage( m_PanelGeneralCommon, _("General"), false );
	m_PanelPathsCommon = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	m_SizerPanelPaths = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer132;
	sbSizer132 = new wxStaticBoxSizer( new wxStaticBox( m_PanelPathsCommon, wxID_ANY, _("Executables paths") ), wxVERTICAL );
	
	m_StaticTextForecasterPath = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Forecaster"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecasterPath->Wrap( -1 );
	sbSizer132->Add( m_StaticTextForecasterPath, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_FilePickerForecaster = new wxFilePickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	sbSizer132->Add( m_FilePickerForecaster, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextViewerPath = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Viewer"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextViewerPath->Wrap( -1 );
	sbSizer132->Add( m_StaticTextViewerPath, 0, wxRIGHT|wxLEFT, 5 );
	
	m_FilePickerViewer = new wxFilePickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	sbSizer132->Add( m_FilePickerViewer, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	
	m_SizerPanelPaths->Add( sbSizer132, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer18;
	sbSizer18 = new wxStaticBoxSizer( new wxStaticBox( m_PanelPathsCommon, wxID_ANY, _("Directories for real-time forecasting") ), wxVERTICAL );
	
	m_StaticTextParametersDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory containing the parameters files"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextParametersDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextParametersDir, 0, wxTOP|wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerParameters = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer18->Add( m_DirPickerParameters, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextArchivePredictorsDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory containing archive predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextArchivePredictorsDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextArchivePredictorsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerArchivePredictors = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer18->Add( m_DirPickerArchivePredictors, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxEXPAND, 5 );
	
	m_StaticTextRealtimePredictorSavingDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory to save downloaded real-time predictors (GCM forecasts)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextRealtimePredictorSavingDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextRealtimePredictorSavingDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerRealtimePredictorSaving = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerRealtimePredictorSaving, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextForecastResultsDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Directory to save forecast outputs"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextForecastResultsDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextForecastResultsDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerForecastResults = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_USE_TEXTCTRL );
	sbSizer18->Add( m_DirPickerForecastResults, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextPredictandDBDir = new wxStaticText( m_PanelPathsCommon, wxID_ANY, _("Default predictand DB directory"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPredictandDBDir->Wrap( -1 );
	sbSizer18->Add( m_StaticTextPredictandDBDir, 0, wxRIGHT|wxLEFT, 5 );
	
	m_DirPickerPredictandDB = new wxDirPickerCtrl( m_PanelPathsCommon, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer18->Add( m_DirPickerPredictandDB, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	m_SizerPanelPaths->Add( sbSizer18, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelPathsCommon->SetSizer( m_SizerPanelPaths );
	m_PanelPathsCommon->Layout();
	m_SizerPanelPaths->Fit( m_PanelPathsCommon );
	m_NotebookBase->AddPage( m_PanelPathsCommon, _("Paths"), false );
	m_PanelViewer = new wxPanel( m_NotebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer39;
	bSizer39 = new wxBoxSizer( wxVERTICAL );
	
	m_NotebookViewer = new wxNotebook( m_PanelViewer, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelForecastDisplay = new wxPanel( m_NotebookViewer, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer261;
	bSizer261 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer141;
	sbSizer141 = new wxStaticBoxSizer( new wxStaticBox( m_PanelForecastDisplay, wxID_ANY, _("Forecast display options") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer81;
	fgSizer81 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer81->SetFlexibleDirection( wxBOTH );
	fgSizer81->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextColorbarMaxValue = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("Set the maximum rainfall value for the colorbar"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextColorbarMaxValue->Wrap( -1 );
	fgSizer81->Add( m_StaticTextColorbarMaxValue, 0, wxALL, 5 );
	
	m_TextCtrlColorbarMaxValue = new wxTextCtrl( m_PanelForecastDisplay, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlColorbarMaxValue->SetMaxLength( 0 ); 
	fgSizer81->Add( m_TextCtrlColorbarMaxValue, 0, wxALL, 5 );
	
	m_StaticTextColorbarMaxUnit = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("mm/d"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextColorbarMaxUnit->Wrap( -1 );
	fgSizer81->Add( m_StaticTextColorbarMaxUnit, 0, wxALL, 5 );
	
	m_StaticTextPastDaysNb = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("Number of past days to display on the timeseries"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPastDaysNb->Wrap( -1 );
	fgSizer81->Add( m_StaticTextPastDaysNb, 0, wxALL, 5 );
	
	m_TextCtrlPastDaysNb = new wxTextCtrl( m_PanelForecastDisplay, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlPastDaysNb->SetMaxLength( 0 ); 
	fgSizer81->Add( m_TextCtrlPastDaysNb, 0, wxALL, 5 );
	
	
	sbSizer141->Add( fgSizer81, 1, wxEXPAND, 5 );
	
	
	bSizer261->Add( sbSizer141, 0, wxEXPAND|wxALL, 5 );
	
	wxStaticBoxSizer* sbSizer191;
	sbSizer191 = new wxStaticBoxSizer( new wxStaticBox( m_PanelForecastDisplay, wxID_ANY, _("Alarms panel") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer13;
	fgSizer13 = new wxFlexGridSizer( 2, 3, 0, 0 );
	fgSizer13->SetFlexibleDirection( wxBOTH );
	fgSizer13->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextAlarmsReturnPeriod = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("Return period to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsReturnPeriod->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsReturnPeriod, 0, wxALL, 5 );
	
	wxString m_ChoiceAlarmsReturnPeriodChoices[] = { _("2"), _("5"), _("10"), _("20"), _("50"), _("100") };
	int m_ChoiceAlarmsReturnPeriodNChoices = sizeof( m_ChoiceAlarmsReturnPeriodChoices ) / sizeof( wxString );
	m_ChoiceAlarmsReturnPeriod = new wxChoice( m_PanelForecastDisplay, wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ChoiceAlarmsReturnPeriodNChoices, m_ChoiceAlarmsReturnPeriodChoices, 0 );
	m_ChoiceAlarmsReturnPeriod->SetSelection( 0 );
	fgSizer13->Add( m_ChoiceAlarmsReturnPeriod, 0, wxALL, 5 );
	
	m_StaticTextAlarmsReturnPeriodYears = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("years"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsReturnPeriodYears->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsReturnPeriodYears, 0, wxALL, 5 );
	
	m_StaticTextAlarmsPercentile = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("Percentile to display"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsPercentile->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsPercentile, 0, wxALL, 5 );
	
	m_TextCtrlAlarmsPercentile = new wxTextCtrl( m_PanelForecastDisplay, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize( 40,-1 ), 0 );
	m_TextCtrlAlarmsPercentile->SetMaxLength( 0 ); 
	fgSizer13->Add( m_TextCtrlAlarmsPercentile, 0, wxALL, 5 );
	
	m_StaticTextAlarmsPercentileRange = new wxStaticText( m_PanelForecastDisplay, wxID_ANY, _("(in between 0 - 1)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextAlarmsPercentileRange->Wrap( -1 );
	fgSizer13->Add( m_StaticTextAlarmsPercentileRange, 0, wxALL, 5 );
	
	
	sbSizer191->Add( fgSizer13, 1, wxEXPAND, 5 );
	
	
	bSizer261->Add( sbSizer191, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelForecastDisplay->SetSizer( bSizer261 );
	m_PanelForecastDisplay->Layout();
	bSizer261->Fit( m_PanelForecastDisplay );
	m_NotebookViewer->AddPage( m_PanelForecastDisplay, _("Forecast display"), false );
	m_PanelGISForecast = new wxPanel( m_NotebookViewer, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer23;
	bSizer23 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer13;
	sbSizer13 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGISForecast, wxID_ANY, _("Default map layers") ), wxVERTICAL );
	
	m_notebook5 = new wxNotebook( m_PanelGISForecast, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	m_PanelLayerHillshade = new wxPanel( m_notebook5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer4;
	fgSizer4 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer4->AddGrowableCol( 1 );
	fgSizer4->SetFlexibleDirection( wxBOTH );
	fgSizer4->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGISLayerHillshadeVisibility = new wxStaticText( m_PanelLayerHillshade, wxID_ANY, _("Visibility"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHillshadeVisibility->Wrap( -1 );
	fgSizer4->Add( m_StaticTextGISLayerHillshadeVisibility, 0, wxALL, 5 );
	
	m_CheckBoxGISLayerHillshadeVisibility = new wxCheckBox( m_PanelLayerHillshade, wxID_ANY, _("display layer"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer4->Add( m_CheckBoxGISLayerHillshadeVisibility, 0, wxALL, 5 );
	
	m_StaticTextGISLayerHillshadeFile = new wxStaticText( m_PanelLayerHillshade, wxID_ANY, _("File"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHillshadeFile->Wrap( -1 );
	fgSizer4->Add( m_StaticTextGISLayerHillshadeFile, 0, wxALL, 5 );
	
	m_FilePickerGISLayerHillshade = new wxFilePickerCtrl( m_PanelLayerHillshade, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_OPEN|wxFLP_USE_TEXTCTRL );
	fgSizer4->Add( m_FilePickerGISLayerHillshade, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGISLayerHillshadeTransp = new wxStaticText( m_PanelLayerHillshade, wxID_ANY, _("Transparency (%)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHillshadeTransp->Wrap( -1 );
	fgSizer4->Add( m_StaticTextGISLayerHillshadeTransp, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerHillshadeTransp = new wxTextCtrl( m_PanelLayerHillshade, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerHillshadeTransp->SetMaxLength( 3 ); 
	fgSizer4->Add( m_TextCtrlGISLayerHillshadeTransp, 0, wxALL, 5 );
	
	
	m_PanelLayerHillshade->SetSizer( fgSizer4 );
	m_PanelLayerHillshade->Layout();
	fgSizer4->Fit( m_PanelLayerHillshade );
	m_notebook5->AddPage( m_PanelLayerHillshade, _("Hillshade"), true );
	m_PanelLayerCatchments = new wxPanel( m_notebook5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer5;
	fgSizer5 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer5->AddGrowableCol( 1 );
	fgSizer5->SetFlexibleDirection( wxBOTH );
	fgSizer5->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGISLayerCatchmentsVisibility = new wxStaticText( m_PanelLayerCatchments, wxID_ANY, _("Visibility"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerCatchmentsVisibility->Wrap( -1 );
	fgSizer5->Add( m_StaticTextGISLayerCatchmentsVisibility, 0, wxALL, 5 );
	
	m_CheckBoxGISLayerCatchmentsVisibility = new wxCheckBox( m_PanelLayerCatchments, wxID_ANY, _("display layer"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer5->Add( m_CheckBoxGISLayerCatchmentsVisibility, 0, wxALL, 5 );
	
	m_StaticTextGISLayerCatchmentsFile = new wxStaticText( m_PanelLayerCatchments, wxID_ANY, _("File"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerCatchmentsFile->Wrap( -1 );
	fgSizer5->Add( m_StaticTextGISLayerCatchmentsFile, 0, wxALL, 5 );
	
	m_FilePickerGISLayerCatchments = new wxFilePickerCtrl( m_PanelLayerCatchments, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_OPEN|wxFLP_USE_TEXTCTRL );
	fgSizer5->Add( m_FilePickerGISLayerCatchments, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGISLayerCatchmentsTransp = new wxStaticText( m_PanelLayerCatchments, wxID_ANY, _("Transparency (%)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerCatchmentsTransp->Wrap( -1 );
	fgSizer5->Add( m_StaticTextGISLayerCatchmentsTransp, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerCatchmentsTransp = new wxTextCtrl( m_PanelLayerCatchments, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerCatchmentsTransp->SetMaxLength( 3 ); 
	fgSizer5->Add( m_TextCtrlGISLayerCatchmentsTransp, 0, wxALL, 5 );
	
	m_StaticTextGISLayerCatchmentsColor = new wxStaticText( m_PanelLayerCatchments, wxID_ANY, _("Color"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerCatchmentsColor->Wrap( -1 );
	fgSizer5->Add( m_StaticTextGISLayerCatchmentsColor, 0, wxALL, 5 );
	
	m_ColourPickerGISLayerCatchmentsColor = new wxColourPickerCtrl( m_PanelLayerCatchments, wxID_ANY, *wxBLACK, wxDefaultPosition, wxDefaultSize, wxCLRP_DEFAULT_STYLE );
	fgSizer5->Add( m_ColourPickerGISLayerCatchmentsColor, 0, wxALL, 5 );
	
	m_StaticTextGISLayerCatchmentsSize = new wxStaticText( m_PanelLayerCatchments, wxID_ANY, _("Line width"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerCatchmentsSize->Wrap( -1 );
	fgSizer5->Add( m_StaticTextGISLayerCatchmentsSize, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerCatchmentsSize = new wxTextCtrl( m_PanelLayerCatchments, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerCatchmentsSize->SetMaxLength( 2 ); 
	fgSizer5->Add( m_TextCtrlGISLayerCatchmentsSize, 0, wxALL, 5 );
	
	
	m_PanelLayerCatchments->SetSizer( fgSizer5 );
	m_PanelLayerCatchments->Layout();
	fgSizer5->Fit( m_PanelLayerCatchments );
	m_notebook5->AddPage( m_PanelLayerCatchments, _("Catchments"), false );
	m_PanelLayerHydrography = new wxPanel( m_notebook5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer6;
	fgSizer6 = new wxFlexGridSizer( 5, 2, 0, 0 );
	fgSizer6->AddGrowableCol( 1 );
	fgSizer6->SetFlexibleDirection( wxBOTH );
	fgSizer6->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGISLayerHydroVisibility = new wxStaticText( m_PanelLayerHydrography, wxID_ANY, _("Visibility"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHydroVisibility->Wrap( -1 );
	fgSizer6->Add( m_StaticTextGISLayerHydroVisibility, 0, wxALL, 5 );
	
	m_CheckBoxGISLayerHydroVisibility = new wxCheckBox( m_PanelLayerHydrography, wxID_ANY, _("display layer"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer6->Add( m_CheckBoxGISLayerHydroVisibility, 0, wxALL, 5 );
	
	m_StaticTextGISLayerHydroFile = new wxStaticText( m_PanelLayerHydrography, wxID_ANY, _("File"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHydroFile->Wrap( -1 );
	fgSizer6->Add( m_StaticTextGISLayerHydroFile, 0, wxALL, 5 );
	
	m_FilePickerGISLayerHydro = new wxFilePickerCtrl( m_PanelLayerHydrography, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_OPEN|wxFLP_USE_TEXTCTRL );
	fgSizer6->Add( m_FilePickerGISLayerHydro, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGISLayerHydroTransp = new wxStaticText( m_PanelLayerHydrography, wxID_ANY, _("Transparency (%)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHydroTransp->Wrap( -1 );
	fgSizer6->Add( m_StaticTextGISLayerHydroTransp, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerHydroTransp = new wxTextCtrl( m_PanelLayerHydrography, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerHydroTransp->SetMaxLength( 3 ); 
	fgSizer6->Add( m_TextCtrlGISLayerHydroTransp, 0, wxALL, 5 );
	
	m_StaticTextGISLayerHydroColor = new wxStaticText( m_PanelLayerHydrography, wxID_ANY, _("Color"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHydroColor->Wrap( -1 );
	fgSizer6->Add( m_StaticTextGISLayerHydroColor, 0, wxALL, 5 );
	
	m_ColourPickerGISLayerHydroColor = new wxColourPickerCtrl( m_PanelLayerHydrography, wxID_ANY, *wxBLACK, wxDefaultPosition, wxDefaultSize, wxCLRP_DEFAULT_STYLE );
	fgSizer6->Add( m_ColourPickerGISLayerHydroColor, 0, wxALL, 5 );
	
	m_StaticTextGISLayerHydroSize = new wxStaticText( m_PanelLayerHydrography, wxID_ANY, _("Line width"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerHydroSize->Wrap( -1 );
	fgSizer6->Add( m_StaticTextGISLayerHydroSize, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerHydroSize = new wxTextCtrl( m_PanelLayerHydrography, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerHydroSize->SetMaxLength( 2 ); 
	fgSizer6->Add( m_TextCtrlGISLayerHydroSize, 0, wxALL, 5 );
	
	
	m_PanelLayerHydrography->SetSizer( fgSizer6 );
	m_PanelLayerHydrography->Layout();
	fgSizer6->Fit( m_PanelLayerHydrography );
	m_notebook5->AddPage( m_PanelLayerHydrography, _("Hydrography"), false );
	m_PanelLayerLakes = new wxPanel( m_notebook5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer7;
	fgSizer7 = new wxFlexGridSizer( 4, 2, 0, 0 );
	fgSizer7->AddGrowableCol( 1 );
	fgSizer7->SetFlexibleDirection( wxBOTH );
	fgSizer7->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGISLayerLakesVisibility = new wxStaticText( m_PanelLayerLakes, wxID_ANY, _("Visibility"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerLakesVisibility->Wrap( -1 );
	fgSizer7->Add( m_StaticTextGISLayerLakesVisibility, 0, wxALL, 5 );
	
	m_CheckBoxGISLayerLakesVisibility = new wxCheckBox( m_PanelLayerLakes, wxID_ANY, _("display layer"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer7->Add( m_CheckBoxGISLayerLakesVisibility, 0, wxALL, 5 );
	
	m_StaticTextGISLayerLakesFile = new wxStaticText( m_PanelLayerLakes, wxID_ANY, _("File"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerLakesFile->Wrap( -1 );
	fgSizer7->Add( m_StaticTextGISLayerLakesFile, 0, wxALL, 5 );
	
	m_FilePickerGISLayerLakes = new wxFilePickerCtrl( m_PanelLayerLakes, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_OPEN|wxFLP_USE_TEXTCTRL );
	fgSizer7->Add( m_FilePickerGISLayerLakes, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGISLayerLakesTransp = new wxStaticText( m_PanelLayerLakes, wxID_ANY, _("Transparency (%)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerLakesTransp->Wrap( -1 );
	fgSizer7->Add( m_StaticTextGISLayerLakesTransp, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerLakesTransp = new wxTextCtrl( m_PanelLayerLakes, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerLakesTransp->SetMaxLength( 3 ); 
	fgSizer7->Add( m_TextCtrlGISLayerLakesTransp, 0, wxALL, 5 );
	
	m_StaticTextGISLayerLakesColor = new wxStaticText( m_PanelLayerLakes, wxID_ANY, _("Color"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerLakesColor->Wrap( -1 );
	fgSizer7->Add( m_StaticTextGISLayerLakesColor, 0, wxALL, 5 );
	
	m_ColourPickerGISLayerLakesColor = new wxColourPickerCtrl( m_PanelLayerLakes, wxID_ANY, *wxBLACK, wxDefaultPosition, wxDefaultSize, wxCLRP_DEFAULT_STYLE );
	fgSizer7->Add( m_ColourPickerGISLayerLakesColor, 0, wxALL, 5 );
	
	
	m_PanelLayerLakes->SetSizer( fgSizer7 );
	m_PanelLayerLakes->Layout();
	fgSizer7->Fit( m_PanelLayerLakes );
	m_notebook5->AddPage( m_PanelLayerLakes, _("Lakes"), false );
	m_PanelLayerBasemap = new wxPanel( m_notebook5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxFlexGridSizer* fgSizer8;
	fgSizer8 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer8->AddGrowableCol( 1 );
	fgSizer8->SetFlexibleDirection( wxBOTH );
	fgSizer8->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextGISLayerBasemapVisibility = new wxStaticText( m_PanelLayerBasemap, wxID_ANY, _("Visibility"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerBasemapVisibility->Wrap( -1 );
	fgSizer8->Add( m_StaticTextGISLayerBasemapVisibility, 0, wxALL, 5 );
	
	m_CheckBoxGISLayerBasemapVisibility = new wxCheckBox( m_PanelLayerBasemap, wxID_ANY, _("display layer"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer8->Add( m_CheckBoxGISLayerBasemapVisibility, 0, wxALL, 5 );
	
	m_StaticTextGISLayerBasemapFile = new wxStaticText( m_PanelLayerBasemap, wxID_ANY, _("File"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerBasemapFile->Wrap( -1 );
	fgSizer8->Add( m_StaticTextGISLayerBasemapFile, 0, wxALL, 5 );
	
	m_FilePickerGISLayerBasemap = new wxFilePickerCtrl( m_PanelLayerBasemap, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.*"), wxDefaultPosition, wxDefaultSize, wxFLP_OPEN|wxFLP_USE_TEXTCTRL );
	fgSizer8->Add( m_FilePickerGISLayerBasemap, 0, wxALL|wxEXPAND, 5 );
	
	m_StaticTextGISLayerBasemapTransp = new wxStaticText( m_PanelLayerBasemap, wxID_ANY, _("Transparency (%)"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextGISLayerBasemapTransp->Wrap( -1 );
	fgSizer8->Add( m_StaticTextGISLayerBasemapTransp, 0, wxALL, 5 );
	
	m_TextCtrlGISLayerBasemapTransp = new wxTextCtrl( m_PanelLayerBasemap, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_TextCtrlGISLayerBasemapTransp->SetMaxLength( 3 ); 
	fgSizer8->Add( m_TextCtrlGISLayerBasemapTransp, 0, wxALL, 5 );
	
	
	m_PanelLayerBasemap->SetSizer( fgSizer8 );
	m_PanelLayerBasemap->Layout();
	fgSizer8->Fit( m_PanelLayerBasemap );
	m_notebook5->AddPage( m_PanelLayerBasemap, _("Basemap"), false );
	
	sbSizer13->Add( m_notebook5, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer23->Add( sbSizer13, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelGISForecast->SetSizer( bSizer23 );
	m_PanelGISForecast->Layout();
	bSizer23->Fit( m_PanelGISForecast );
	m_NotebookViewer->AddPage( m_PanelGISForecast, _("Forecast GIS options"), false );
	
	bSizer39->Add( m_NotebookViewer, 1, wxEXPAND | wxALL, 5 );
	
	
	m_PanelViewer->SetSizer( bSizer39 );
	m_PanelViewer->Layout();
	bSizer39->Fit( m_PanelViewer );
	m_NotebookBase->AddPage( m_PanelViewer, _("Viewer"), true );
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
	
	wxStaticBoxSizer* sbSizer11;
	sbSizer11 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneral, wxID_ANY, _("Predictor download") ), wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_StaticTextNumberFails = new wxStaticText( m_PanelGeneral, wxID_ANY, _("Maximum number of previous time steps if download fails"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextNumberFails->Wrap( -1 );
	fgSizer2->Add( m_StaticTextNumberFails, 0, wxALL, 5 );
	
	m_TextCtrlMaxPrevStepsNb = new wxTextCtrl( m_PanelGeneral, wxID_ANY, _("5"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_TextCtrlMaxPrevStepsNb->SetMaxLength( 1 ); 
	fgSizer2->Add( m_TextCtrlMaxPrevStepsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextMaxRequestsNb = new wxStaticText( m_PanelGeneral, wxID_ANY, _("Maximum parallel requests number"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextMaxRequestsNb->Wrap( -1 );
	fgSizer2->Add( m_StaticTextMaxRequestsNb, 0, wxALL, 5 );
	
	m_TextCtrlMaxRequestsNb = new wxTextCtrl( m_PanelGeneral, wxID_ANY, _("3"), wxDefaultPosition, wxSize( 30,-1 ), 0 );
	m_TextCtrlMaxRequestsNb->SetMaxLength( 1 ); 
	fgSizer2->Add( m_TextCtrlMaxRequestsNb, 0, wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_CheckBoxRestrictDownloads = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Restrict downloads to needed lead times."), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxRestrictDownloads->SetValue(true); 
	m_CheckBoxRestrictDownloads->Enable( false );
	
	fgSizer2->Add( m_CheckBoxRestrictDownloads, 0, wxALL, 5 );
	
	
	sbSizer11->Add( fgSizer2, 1, wxEXPAND, 5 );
	
	
	bSizer271->Add( sbSizer11, 0, wxALL|wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer151;
	sbSizer151 = new wxStaticBoxSizer( new wxStaticBox( m_PanelGeneral, wxID_ANY, _("Advanced options") ), wxVERTICAL );
	
	m_CheckBoxResponsiveness = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Let the software be responsive while processing (recommended)."), wxDefaultPosition, wxDefaultSize, 0 );
	m_CheckBoxResponsiveness->SetValue(true); 
	sbSizer151->Add( m_CheckBoxResponsiveness, 0, wxALL, 5 );
	
	m_CheckBoxMultiInstancesForecaster = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Allow multiple instances of the forecaster"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer151->Add( m_CheckBoxMultiInstancesForecaster, 0, wxALL, 5 );
	
	m_CheckBoxMultiInstancesViewer = new wxCheckBox( m_PanelGeneral, wxID_ANY, _("Allow multiple instances of the viewer"), wxDefaultPosition, wxDefaultSize, 0 );
	sbSizer151->Add( m_CheckBoxMultiInstancesViewer, 0, wxALL, 5 );
	
	
	bSizer271->Add( sbSizer151, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelGeneral->SetSizer( bSizer271 );
	m_PanelGeneral->Layout();
	bSizer271->Fit( m_PanelGeneral );
	m_NotebookAdvanced->AddPage( m_PanelGeneral, _("General"), false );
	m_PanelProcessing = new wxPanel( m_NotebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer1611;
	bSizer1611 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer15;
	sbSizer15 = new wxStaticBoxSizer( new wxStaticBox( m_PanelProcessing, wxID_ANY, _("Multithreading") ), wxVERTICAL );
	
	m_CheckBoxAllowMultithreading = new wxCheckBox( m_PanelProcessing, wxID_ANY, _("Allow multithreading"), wxDefaultPosition, wxDefaultSize, 0 );
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
	
	wxString m_RadioBoxProcessingMethodsChoices[] = { _("Multithreaded (only if allowed hereabove)"), _("Date array insertions (safer)"), _("Date array splitting (slower)") };
	int m_RadioBoxProcessingMethodsNChoices = sizeof( m_RadioBoxProcessingMethodsChoices ) / sizeof( wxString );
	m_RadioBoxProcessingMethods = new wxRadioBox( m_PanelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition, wxDefaultSize, m_RadioBoxProcessingMethodsNChoices, m_RadioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxProcessingMethods->SetSelection( 1 );
	m_RadioBoxProcessingMethods->SetToolTip( _("These options don't affect the results, only the processor efficiency.") );
	
	bSizer1611->Add( m_RadioBoxProcessingMethods, 0, wxALL|wxEXPAND, 5 );
	
	wxString m_RadioBoxLinearAlgebraChoices[] = { _("Direct access to the coefficients (recommanded)"), _("Direct access to the coefficients and minimizing variable declarations"), _("Linear algebra using Eigen"), _("Linear algebra using Eigen and minimizing variable declarations") };
	int m_RadioBoxLinearAlgebraNChoices = sizeof( m_RadioBoxLinearAlgebraChoices ) / sizeof( wxString );
	m_RadioBoxLinearAlgebra = new wxRadioBox( m_PanelProcessing, wxID_ANY, _("Linear algebra options"), wxDefaultPosition, wxDefaultSize, m_RadioBoxLinearAlgebraNChoices, m_RadioBoxLinearAlgebraChoices, 1, wxRA_SPECIFY_COLS );
	m_RadioBoxLinearAlgebra->SetSelection( 0 );
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
	
	m_DirPickerIntermediateResults = new wxDirPickerCtrl( m_PanelUserDirectories, wxID_ANY, wxEmptyString, _("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
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
	
	m_StaticTextLogFileForecasterLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Forecaster log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFileForecasterLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFileForecasterLabel, 0, wxALL, 5 );
	
	m_StaticTextLogFileForecaster = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFileForecaster->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFileForecaster, 0, wxALL, 5 );
	
	m_StaticTextLogFileViewerLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Viewer log file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFileViewerLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFileViewerLabel, 0, wxALL, 5 );
	
	m_StaticTextLogFileViewer = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLogFileViewer->Wrap( -1 );
	fgSizer9->Add( m_StaticTextLogFileViewer, 0, wxALL, 5 );
	
	m_StaticTextPrefFileForecasterLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Forecaster ini file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFileForecasterLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFileForecasterLabel, 0, wxALL, 5 );
	
	m_StaticTextPrefFileForecaster = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFileForecaster->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFileForecaster, 0, wxALL, 5 );
	
	m_StaticTextPrefFileViewerLabel = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("Viewer ini file:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFileViewerLabel->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFileViewerLabel, 0, wxALL, 5 );
	
	m_StaticTextPrefFileViewer = new wxStaticText( m_PanelUserDirectories, wxID_ANY, _("..."), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextPrefFileViewer->Wrap( -1 );
	fgSizer9->Add( m_StaticTextPrefFileViewer, 0, wxALL, 5 );
	
	
	sbSizer17->Add( fgSizer9, 1, wxEXPAND, 5 );
	
	
	bSizer24->Add( sbSizer17, 0, wxALL|wxEXPAND, 5 );
	
	
	m_PanelUserDirectories->SetSizer( bSizer24 );
	m_PanelUserDirectories->Layout();
	bSizer24->Fit( m_PanelUserDirectories );
	m_NotebookAdvanced->AddPage( m_PanelUserDirectories, _("User paths"), true );
	m_PanelPathsCatalogs = new wxPanel( m_NotebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer171;
	bSizer171 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer61;
	sbSizer61 = new wxStaticBoxSizer( new wxStaticBox( m_PanelPathsCatalogs, wxID_ANY, _("Catalog files paths") ), wxVERTICAL );
	
	m_StaticTextCatalogPredictorArchive = new wxStaticText( m_PanelPathsCatalogs, wxID_ANY, _("Path to the archive predictors catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextCatalogPredictorArchive->Wrap( -1 );
	sbSizer61->Add( m_StaticTextCatalogPredictorArchive, 0, wxALL, 5 );
	
	m_FilePickerCatalogPredictorsArchive = new wxFilePickerCtrl( m_PanelPathsCatalogs, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	sbSizer61->Add( m_FilePickerCatalogPredictorsArchive, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextCatalogPredictorRealtime = new wxStaticText( m_PanelPathsCatalogs, wxID_ANY, _("Path to the real-time predictors catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextCatalogPredictorRealtime->Wrap( -1 );
	sbSizer61->Add( m_StaticTextCatalogPredictorRealtime, 0, wxALL, 5 );
	
	m_FilePickerCatalogPredictorsRealtime = new wxFilePickerCtrl( m_PanelPathsCatalogs, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	sbSizer61->Add( m_FilePickerCatalogPredictorsRealtime, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	m_StaticTextCatalogPredictand = new wxStaticText( m_PanelPathsCatalogs, wxID_ANY, _("Path to the predictand catalog"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextCatalogPredictand->Wrap( -1 );
	sbSizer61->Add( m_StaticTextCatalogPredictand, 0, wxALL, 5 );
	
	m_FilePickerCatalogPredictands = new wxFilePickerCtrl( m_PanelPathsCatalogs, wxID_ANY, wxEmptyString, _("Select a file"), wxT("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	sbSizer61->Add( m_FilePickerCatalogPredictands, 0, wxEXPAND|wxBOTTOM|wxRIGHT|wxLEFT, 5 );
	
	
	bSizer171->Add( sbSizer61, 0, wxEXPAND|wxALL, 5 );
	
	
	m_PanelPathsCatalogs->SetSizer( bSizer171 );
	m_PanelPathsCatalogs->Layout();
	bSizer171->Fit( m_PanelPathsCatalogs );
	m_NotebookAdvanced->AddPage( m_PanelPathsCatalogs, _("Catalogs"), false );
	
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
	m_CheckBoxAllowMultithreading->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_ButtonsConfirmationApply->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::SaveAndClose ), NULL, this );
}

asFramePreferencesVirtual::~asFramePreferencesVirtual()
{
	// Disconnect Events
	m_CheckBoxAllowMultithreading->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::OnChangeMultithreadingCheckBox ), NULL, this );
	m_ButtonsConfirmationApply->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::ApplyChanges ), NULL, this );
	m_ButtonsConfirmationCancel->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::CloseFrame ), NULL, this );
	m_ButtonsConfirmationOK->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( asFramePreferencesVirtual::SaveAndClose ), NULL, this );
	
}
