///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct 26 2018)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

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
#include <wx/choice.h>
#include <wx/panel.h>
#include <wx/frame.h>
#include <wx/checkbox.h>
#include <wx/textctrl.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/statbmp.h>
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
/// Class asFramePredictandDBVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePredictandDBVirtual : public wxFrame
{
	private:

	protected:
		wxBoxSizer* m_sizerMain;
		wxPanel* m_panelMain;
		wxBoxSizer* m_sizerMainPanel;
		wxStaticText* m_staticTextDataParam;
		wxChoice* m_choiceDataParam;
		wxStaticText* m_staticTextDataTempResol;
		wxChoice* m_choiceDataTempResol;
		wxStaticText* m_staticTextDataSpatAggreg;
		wxChoice* m_choiceDataSpatAggreg;
		wxBoxSizer* m_sizerProcessing;
		wxStaticText* m_staticTextCatalogPath;
		wxFilePickerCtrl* m_filePickerCatalogPath;
		wxStaticText* m_staticTextDataDir;
		wxDirPickerCtrl* m_dirPickerDataDir;
		wxStaticText* m_staticTextPatternsDir;
		wxDirPickerCtrl* m_dirPickerPatternsDir;
		wxStaticText* m_staticDestinationDir;
		wxDirPickerCtrl* m_dirPickerDestinationDir;
		wxStdDialogButtonSizer* m_buttonsConfirmation;
		wxButton* m_buttonsConfirmationOK;
		wxButton* m_buttonsConfirmationCancel;

		// Virtual event handlers, overide them in your derived class
		virtual void OnDataSelection( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void BuildDatabase( wxCommandEvent& event ) { event.Skip(); }


	public:

		asFramePredictandDBVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Predictand database generator"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );

		~asFramePredictandDBVirtual();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelProcessingPrecipitation
///////////////////////////////////////////////////////////////////////////////
class asPanelProcessingPrecipitation : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText22;
		wxStaticText* m_staticTextYears;

	public:
		wxCheckBox* m_checkBoxReturnPeriod;
		wxTextCtrl* m_textCtrlReturnPeriod;
		wxCheckBox* m_checkBoxSqrt;

		asPanelProcessingPrecipitation( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );
		~asPanelProcessingPrecipitation();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelProcessingLightnings
///////////////////////////////////////////////////////////////////////////////
class asPanelProcessingLightnings : public wxPanel
{
	private:

	protected:
		wxStaticText* m_staticText23;

	public:
		wxCheckBox* m_checkBoxLog;

		asPanelProcessingLightnings( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );
		~asPanelProcessingLightnings();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameAboutVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameAboutVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* m_Panel;
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
		wxStaticText* m_staticTextSpacer;
		wxPanel* m_panelLicense;
		wxTextCtrl* m_textCtrlLicense;
		wxPanel* m_panelLibraries;
		wxTextCtrl* m_textCtrlLibraries;

	public:

		asFrameAboutVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("About"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 350,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxSTAY_ON_TOP|wxTAB_TRAVERSAL );

		~asFrameAboutVirtual();

};

