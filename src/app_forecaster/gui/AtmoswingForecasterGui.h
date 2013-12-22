///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct  8 2012)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __ATMOSWINGFORECASTERGUI_H__
#define __ATMOSWINGFORECASTERGUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/intl.h>
#include <wx/calctrl.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/bmpbuttn.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/statbox.h>
#include <wx/awx/led.h>
#ifdef __VISUALC__
#include <wx/link_additions.h>
#endif //__VISUALC__
#include <wx/scrolwin.h>
#include <wx/panel.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/filepicker.h>

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class asFrameMainVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameMainVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelMain;
		wxCalendarCtrl* m_CalendarForecastDate;
		wxStaticText* m_StaticTextForecastHour;
		wxTextCtrl* m_TextCtrlForecastHour;
		wxBitmapButton* m_BpButtonNow;
		awxLed* m_LedDownloading;
		wxStaticText* m_StaticTextDownloading;
		awxLed* m_LedLoading;
		wxStaticText* m_StaticTextLoading;
		awxLed* m_LedProcessing;
		wxStaticText* m_StaticTextProcessing;
		awxLed* m_LedSaving;
		wxStaticText* m_StaticTextSaving;
		wxScrolledWindow* m_ScrolledWindowModels;
		wxBoxSizer* m_SizerModels;
		wxBitmapButton* m_BpButtonAdd;
		wxMenuBar* m_MenuBar;
		wxMenu* m_MenuFile;
		wxMenu* m_MenuOptions;
		wxMenu* m_MenuTools;
		wxMenu* m_MenuLog;
		wxMenu* m_MenuLogLevel;
		wxMenu* m_MenuHelp;
		wxToolBar* m_ToolBar;
		wxStatusBar* m_statusBar1;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnSetPresentDate( wxCommandEvent& event ) { event.Skip(); }
		virtual void AddForecastingModel( wxCommandEvent& event ) { event.Skip(); }
		virtual void ModelsListSaveAsDefault( wxCommandEvent& event ) { event.Skip(); }
		virtual void ModelsListLoadDefault( wxCommandEvent& event ) { event.Skip(); }
		virtual void ModelsListSave( wxCommandEvent& event ) { event.Skip(); }
		virtual void ModelsListLoad( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePreferences( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePredictandDB( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameMainVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("AtmoSwing Forecaster"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameMainVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePredictandDBVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePredictandDBVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panel2;
		wxStaticText* m_StaticTextDataParam;
		wxChoice* m_ChoiceDataParam;
		wxStaticText* m_StaticTextDataTempResol;
		wxChoice* m_ChoiceDataTempResol;
		wxStaticText* m_StaticTextDataSpatAggreg;
		wxChoice* m_ChoiceDataSpatAggreg;
		wxPanel* m_PanelDataProcessing;
		wxCheckBox* m_CheckBoxReturnPeriod;
		wxTextCtrl* m_TextCtrlReturnPeriod;
		wxStaticText* m_StaticTextYears;
		wxCheckBox* m_CheckBoxSqrt;
		wxStaticText* m_StaticTextCatalogPath;
		wxFilePickerCtrl* m_FilePickerCatalogPath;
		wxStaticText* m_StaticTextDataDir;
		wxDirPickerCtrl* m_DirPickerDataDir;
		wxStaticText* m_StaticTextPatternsDir;
		wxDirPickerCtrl* m_DirPickerPatternsDir;
		wxStaticText* m_StaticDestinationDir;
		wxDirPickerCtrl* m_DirPickerDestinationDir;
		wxButton* m_ButtonSaveDefault;
		wxStdDialogButtonSizer* m_ButtonsConfirmation;
		wxButton* m_ButtonsConfirmationOK;
		wxButton* m_ButtonsConfirmationCancel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnDataSelection( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSaveDefault( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void BuildDatabase( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePredictandDBVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Predictand database generator"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePredictandDBVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelForecastingModelVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelForecastingModelVirtual : public wxPanel 
{
	private:
	
	protected:
		wxBoxSizer* m_SizerPanel;
		wxBoxSizer* m_SizerHeader;
		awxLed* m_Led;
		wxStaticText* m_StaticTextModelName;
		wxBitmapButton* m_BpButtonReduce;
		wxBitmapButton* m_BpButtonClose;
		wxBoxSizer* m_SizerFields;
		wxStaticText* m_StaticTextModelNameInput;
		wxTextCtrl* m_TextCtrlModelName;
		wxStaticText* m_StaticTextModelDescriptionInput;
		wxTextCtrl* m_TextCtrlModelDescription;
		wxStaticText* m_StaticTextParametersFileName;
		wxTextCtrl* m_TextCtrlParametersFileName;
		wxStaticText* m_StaticTextPredictandDB;
		wxTextCtrl* m_TextCtrlPredictandDB;
		wxStaticText* m_StaticTextPredictorsArchiveDir;
		wxDirPickerCtrl* m_DirPickerPredictorsArchive;
		
		// Virtual event handlers, overide them in your derived class
		virtual void ReducePanel( wxCommandEvent& event ) { event.Skip(); }
		virtual void ClosePanel( wxCommandEvent& event ) { event.Skip(); }
		virtual void ChangeModelName( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asPanelForecastingModelVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL ); 
		~asPanelForecastingModelVirtual();
	
};

#endif //__ATMOSWINGFORECASTERGUI_H__
