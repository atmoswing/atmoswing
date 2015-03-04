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
		wxStaticText* m_staticTextDescription;
		wxFilePickerCtrl* m_filePicker;
		wxStdDialogButtonSizer* m_buttonsConfirmation;
		wxButton* m_buttonsConfirmationOK;
		wxButton* m_buttonsConfirmationCancel;
	
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
		wxStaticText* m_staticTextDescription;
		wxFilePickerCtrl* m_filePicker;
		wxStdDialogButtonSizer* m_buttonsConfirmation;
		wxButton* m_buttonsConfirmationSave;
		wxButton* m_buttonsConfirmationCancel;
	
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
		wxToolBar* m_toolBar;
		wxToolBarToolBase* m_toolSave; 
	
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
		wxPanel* m_panel;
		wxStaticBitmap* m_logo;
		wxStaticText* m_staticTextVersion;
		wxStaticText* m_staticTextChangeset;
		wxNotebook* m_notebook;
		wxPanel* m_panelCredits;
		wxStaticText* m_staticTextDevelopers;
		wxStaticText* m_staticTextDevelopersList;
		wxStaticText* m_staticTextSupervision;
		wxStaticText* m_staticTextSupervisionList;
		wxStaticText* m_staticTextThanks;
		wxStaticText* m_staticTextThanksList;
		wxStaticText* m_staticTextOtherCredits;
		wxStaticText* m_staticTextOtherCreditsList;
		wxStaticText* m_staticTextSpacer;
		wxPanel* m_panelLicense;
		wxTextCtrl* m_textCtrlLicense;
		wxPanel* m_panelLibraries;
		wxTextCtrl* m_textCtrlLibraries;
	
	public:
		
		asFrameAboutVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("About"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 350,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxSTAY_ON_TOP|wxTAB_TRAVERSAL );
		
		~asFrameAboutVirtual();
	
};

#endif //__ATMOSWINGSHAREDGUI_H__
