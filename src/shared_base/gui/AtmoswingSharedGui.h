///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun  5 2014)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __ATMOSWINGSHAREDGUI_H__
#define __ATMOSWINGSHAREDGUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/intl.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/filepicker.h>
#include <wx/sizer.h>
#include <wx/button.h>
#include <wx/dialog.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/toolbar.h>
#include <wx/frame.h>
#include <wx/statbmp.h>
#include <wx/panel.h>
#include <wx/textctrl.h>
#include <wx/notebook.h>

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class asDialogFilePickerVirtual
///////////////////////////////////////////////////////////////////////////////
class asDialogFilePickerVirtual : public wxDialog 
{
	private:
	
	protected:
		wxStaticText* m_StaticTextDescription;
		wxFilePickerCtrl* m_FilePicker;
		wxStdDialogButtonSizer* m_ButtonsConfirmation;
		wxButton* m_ButtonsConfirmationOK;
		wxButton* m_ButtonsConfirmationCancel;
	
	public:
		
		asDialogFilePickerVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Select a file"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 374,130 ), long style = wxDEFAULT_DIALOG_STYLE ); 
		~asDialogFilePickerVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asDialogFileSaverVirtual
///////////////////////////////////////////////////////////////////////////////
class asDialogFileSaverVirtual : public wxDialog 
{
	private:
	
	protected:
		wxStaticText* m_StaticTextDescription;
		wxFilePickerCtrl* m_FilePicker;
		wxStdDialogButtonSizer* m_ButtonsConfirmation;
		wxButton* m_ButtonsConfirmationSave;
		wxButton* m_ButtonsConfirmationCancel;
	
	public:
		
		asDialogFileSaverVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Save to a file"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 374,130 ), long style = wxDEFAULT_DIALOG_STYLE ); 
		~asDialogFileSaverVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameXmlEditorVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameXmlEditorVirtual : public wxFrame 
{
	private:
	
	protected:
		wxToolBar* m_ToolBar;
		wxToolBarToolBase* m_ToolSave; 
	
	public:
		
		asFrameXmlEditorVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Xml Editor"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 532,423 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameXmlEditorVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameAboutVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameAboutVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_Panel;
		wxStaticBitmap* m_Logo;
		wxStaticText* m_StaticTextVersion;
		wxStaticText* m_StaticTextChangeset;
		wxNotebook* m_Notebook;
		wxPanel* m_PanelCredits;
		wxStaticText* m_StaticTextDevelopers;
		wxStaticText* m_StaticTextDevelopersList;
		wxStaticText* m_StaticTextSupervision;
		wxStaticText* m_StaticTextSupervisionList;
		wxStaticText* m_StaticTextThanks;
		wxStaticText* m_StaticTextThanksList;
		wxStaticText* m_StaticTextOtherCredits;
		wxStaticText* m_StaticTextOtherCreditsList;
		wxStaticText* m_staticTextSpacer;
		wxPanel* m_PanelLicense;
		wxTextCtrl* m_TextCtrlLicense;
		wxPanel* m_PanelLibraries;
		wxTextCtrl* m_TextCtrlLibraries;
	
	public:
		
		asFrameAboutVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("About"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 350,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxSTAY_ON_TOP|wxTAB_TRAVERSAL );
		
		~asFrameAboutVirtual();
	
};

#endif //__ATMOSWINGSHAREDGUI_H__
