///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
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
		wxStaticText* _staticTextDescription;
		wxFilePickerCtrl* _filePicker;
		wxStdDialogButtonSizer* _buttonsConfirmation;
		wxButton* _buttonsConfirmationOK;
		wxButton* _buttonsConfirmationCancel;

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
		wxStaticText* _staticTextDescription;
		wxFilePickerCtrl* _filePicker;
		wxStdDialogButtonSizer* _buttonsConfirmation;
		wxButton* _buttonsConfirmationSave;
		wxButton* _buttonsConfirmationCancel;

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
		wxBoxSizer* _sizerMain;
		wxPanel* _panelMain;
		wxBoxSizer* _sizerMainPanel;
		wxStaticText* _staticTextDataParam;
		wxChoice* _choiceDataParam;
		wxStaticText* _staticTextDataTempResol;
		wxChoice* _choiceDataTempResol;
		wxStaticText* _staticTextDataSpatAggreg;
		wxChoice* _choiceDataSpatAggreg;
		wxBoxSizer* _sizerProcessing;
		wxStaticText* _staticTextCatalogPath;
		wxFilePickerCtrl* _filePickerCatalogPath;
		wxStaticText* _staticTextDataDir;
		wxDirPickerCtrl* _dirPickerDataDir;
		wxStaticText* _staticTextPatternsDir;
		wxDirPickerCtrl* _dirPickerPatternsDir;
		wxStaticText* _staticDestinationDir;
		wxDirPickerCtrl* _dirPickerDestinationDir;
		wxStdDialogButtonSizer* _buttonsConfirmation;
		wxButton* _buttonsConfirmationOK;
		wxButton* _buttonsConfirmationCancel;

		// Virtual event handlers, override them in your derived class
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
		wxStaticText* _staticText22;
		wxStaticText* _staticTextYears;

	public:
		wxCheckBox* _checkBoxReturnPeriod;
		wxTextCtrl* _textCtrlReturnPeriod;
		wxCheckBox* _checkBoxSqrt;

		asPanelProcessingPrecipitation( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~asPanelProcessingPrecipitation();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelProcessingLightning
///////////////////////////////////////////////////////////////////////////////
class asPanelProcessingLightning : public wxPanel
{
	private:

	protected:
		wxStaticText* _staticText23;

	public:
		wxCheckBox* _checkBoxLog;

		asPanelProcessingLightning( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL, const wxString& name = wxEmptyString );

		~asPanelProcessingLightning();

};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameAboutVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameAboutVirtual : public wxFrame
{
	private:

	protected:
		wxPanel* _panel;
		wxStaticBitmap* _logo;
		wxStaticText* _staticTextVersion;
		wxNotebook* _notebook;
		wxPanel* _panelCredits;
		wxStaticText* _staticTextDevelopers;
		wxStaticText* _staticTextDevelopersList;
		wxStaticText* _staticTextSupervision;
		wxStaticText* _staticTextSupervisionList;
		wxStaticText* _staticTextThanks;
		wxStaticText* _staticTextThanksList;
		wxStaticText* _staticTextSpacer;
		wxPanel* _panelLicense;
		wxTextCtrl* _textCtrlLicense;

	public:

		asFrameAboutVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("About"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 350,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxSTAY_ON_TOP|wxTAB_TRAVERSAL );

		~asFrameAboutVirtual();

};

