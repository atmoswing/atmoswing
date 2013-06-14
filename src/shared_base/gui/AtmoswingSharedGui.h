///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct  8 2012)
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
#include <wx/radiobox.h>
#include <wx/checkbox.h>
#include <wx/statbox.h>
#include <wx/textctrl.h>
#include <wx/choice.h>
#include <wx/clrpicker.h>
#include <wx/notebook.h>
#include <wx/slider.h>

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
		wxStaticText* m_StaticTextTitle;
		wxStaticText* m_StaticTextVersion;
		wxStaticBitmap* m_bitmap1;
		wxStaticText* m_StaticTextDevelopers;
		wxStaticText* m_StaticTextDevelopersList;
		wxStaticText* m_StaticTextSupervision;
		wxStaticText* m_StaticTextSupervisionList;
		wxStaticText* m_StaticTextThanks;
		wxStaticText* m_StaticTextThanksList;
		wxStaticText* m_StaticTextOtherCredits;
		wxStaticText* m_StaticTextOtherCreditsList;
		wxStaticText* m_StaticTextLibraries;
		wxGridSizer* m_GridSizer;
		wxStaticText* m_StaticTextSpacer;
	
	public:
		
		asFrameAboutVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("About"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxSTAY_ON_TOP|wxTAB_TRAVERSAL );
		
		~asFrameAboutVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePreferencesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePreferencesVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelBase;
		wxNotebook* m_NotebookBase;
		wxPanel* m_PanelGeneralCommon;
		wxRadioBox* m_RadioBoxLogFLevel;
		wxCheckBox* m_CheckBoxDisplayLogFWindow;
		wxCheckBox* m_CheckBoxSaveLogFFile;
		wxRadioBox* m_RadioBoxLogVLevel;
		wxCheckBox* m_CheckBoxDisplayLogVWindow;
		wxCheckBox* m_CheckBoxSaveLogVFile;
		wxCheckBox* m_CheckBoxProxy;
		wxStaticText* m_StaticTextProxyAddress;
		wxTextCtrl* m_TextCtrlProxyAddress;
		wxStaticText* m_StaticTextProxyPort;
		wxTextCtrl* m_TextCtrlProxyPort;
		wxStaticText* m_StaticTextProxyUser;
		wxTextCtrl* m_TextCtrlProxyUser;
		wxStaticText* m_StaticTextProxyPasswd;
		wxTextCtrl* m_TextCtrlProxyPasswd;
		wxPanel* m_PanelPathsCommon;
		wxBoxSizer* m_SizerPanelPaths;
		wxStaticText* m_StaticTextForecasterPath;
		wxFilePickerCtrl* m_FilePickerForecaster;
		wxStaticText* m_StaticTextViewerPath;
		wxFilePickerCtrl* m_FilePickerViewer;
		wxStaticText* m_StaticTextParametersDir;
		wxDirPickerCtrl* m_DirPickerParameters;
		wxStaticText* m_StaticTextArchivePredictorsDir;
		wxDirPickerCtrl* m_DirPickerArchivePredictors;
		wxStaticText* m_StaticTextRealtimePredictorSavingDir;
		wxDirPickerCtrl* m_DirPickerRealtimePredictorSaving;
		wxStaticText* m_StaticTextForecastResultsDir;
		wxDirPickerCtrl* m_DirPickerForecastResults;
		wxStaticText* m_StaticTextPredictandDBDir;
		wxDirPickerCtrl* m_DirPickerPredictandDB;
		wxPanel* m_PanelViewer;
		wxNotebook* m_NotebookViewer;
		wxPanel* m_PanelForecastDisplay;
		wxStaticText* m_StaticTextColorbarMaxValue;
		wxTextCtrl* m_TextCtrlColorbarMaxValue;
		wxStaticText* m_StaticTextColorbarMaxUnit;
		wxStaticText* m_StaticTextPastDaysNb;
		wxTextCtrl* m_TextCtrlPastDaysNb;
		wxStaticText* m_StaticTextAlarmsReturnPeriod;
		wxChoice* m_ChoiceAlarmsReturnPeriod;
		wxStaticText* m_StaticTextAlarmsReturnPeriodYears;
		wxStaticText* m_StaticTextAlarmsPercentile;
		wxTextCtrl* m_TextCtrlAlarmsPercentile;
		wxStaticText* m_StaticTextAlarmsPercentileRange;
		wxPanel* m_PanelGISForecast;
		wxNotebook* m_notebook5;
		wxPanel* m_PanelLayerHillshade;
		wxStaticText* m_StaticTextGISLayerHillshadeVisibility;
		wxCheckBox* m_CheckBoxGISLayerHillshadeVisibility;
		wxStaticText* m_StaticTextGISLayerHillshadeFile;
		wxFilePickerCtrl* m_FilePickerGISLayerHillshade;
		wxStaticText* m_StaticTextGISLayerHillshadeTransp;
		wxTextCtrl* m_TextCtrlGISLayerHillshadeTransp;
		wxPanel* m_PanelLayerCatchments;
		wxStaticText* m_StaticTextGISLayerCatchmentsVisibility;
		wxCheckBox* m_CheckBoxGISLayerCatchmentsVisibility;
		wxStaticText* m_StaticTextGISLayerCatchmentsFile;
		wxFilePickerCtrl* m_FilePickerGISLayerCatchments;
		wxStaticText* m_StaticTextGISLayerCatchmentsTransp;
		wxTextCtrl* m_TextCtrlGISLayerCatchmentsTransp;
		wxStaticText* m_StaticTextGISLayerCatchmentsColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerCatchmentsColor;
		wxStaticText* m_StaticTextGISLayerCatchmentsSize;
		wxTextCtrl* m_TextCtrlGISLayerCatchmentsSize;
		wxPanel* m_PanelLayerHydrography;
		wxStaticText* m_StaticTextGISLayerHydroVisibility;
		wxCheckBox* m_CheckBoxGISLayerHydroVisibility;
		wxStaticText* m_StaticTextGISLayerHydroFile;
		wxFilePickerCtrl* m_FilePickerGISLayerHydro;
		wxStaticText* m_StaticTextGISLayerHydroTransp;
		wxTextCtrl* m_TextCtrlGISLayerHydroTransp;
		wxStaticText* m_StaticTextGISLayerHydroColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerHydroColor;
		wxStaticText* m_StaticTextGISLayerHydroSize;
		wxTextCtrl* m_TextCtrlGISLayerHydroSize;
		wxPanel* m_PanelLayerLakes;
		wxStaticText* m_StaticTextGISLayerLakesVisibility;
		wxCheckBox* m_CheckBoxGISLayerLakesVisibility;
		wxStaticText* m_StaticTextGISLayerLakesFile;
		wxFilePickerCtrl* m_FilePickerGISLayerLakes;
		wxStaticText* m_StaticTextGISLayerLakesTransp;
		wxTextCtrl* m_TextCtrlGISLayerLakesTransp;
		wxStaticText* m_StaticTextGISLayerLakesColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerLakesColor;
		wxPanel* m_PanelLayerBasemap;
		wxStaticText* m_StaticTextGISLayerBasemapVisibility;
		wxCheckBox* m_CheckBoxGISLayerBasemapVisibility;
		wxStaticText* m_StaticTextGISLayerBasemapFile;
		wxFilePickerCtrl* m_FilePickerGISLayerBasemap;
		wxStaticText* m_StaticTextGISLayerBasemapTransp;
		wxTextCtrl* m_TextCtrlGISLayerBasemapTransp;
		wxPanel* m_PanelGISPredictors;
		wxNotebook* m_notebook51;
		wxPanel* m_PanelLayerContinents;
		wxStaticText* m_StaticTextGISLayerContinentsVisibility;
		wxCheckBox* m_CheckBoxGISLayerContinentsVisibility;
		wxStaticText* m_StaticTextGISLayerContinentsFile;
		wxFilePickerCtrl* m_FilePickerGISLayerContinents;
		wxStaticText* m_StaticTextGISLayerContinentsTransp;
		wxTextCtrl* m_TextCtrlGISLayerContinentsTransp;
		wxStaticText* m_StaticTextGISLayerContinentsColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerContinentsColor;
		wxStaticText* m_StaticTextGISLayerContinentsSize;
		wxTextCtrl* m_TextCtrlGISLayerContinentsSize;
		wxPanel* m_PanelLayerCountries;
		wxStaticText* m_StaticTextGISLayerCountriesVisibility;
		wxCheckBox* m_CheckBoxGISLayerCountriesVisibility;
		wxStaticText* m_StaticTextGISLayerCountriesFile;
		wxFilePickerCtrl* m_FilePickerGISLayerCountries;
		wxStaticText* m_StaticTextGISLayerCountriesTransp;
		wxTextCtrl* m_TextCtrlGISLayerCountriesTransp;
		wxStaticText* m_StaticTextGISLayerCountriesColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerCountriesColor;
		wxStaticText* m_StaticTextGISLayerCountriesSize;
		wxTextCtrl* m_TextCtrlGISLayerCountriesSize;
		wxPanel* m_PanelLayerGeogrid;
		wxStaticText* m_StaticTextGISLayerGeogridVisibility;
		wxCheckBox* m_CheckBoxGISLayerGeogridVisibility;
		wxStaticText* m_StaticTextGISLayerGeogridFile;
		wxFilePickerCtrl* m_FilePickerGISLayerGeogrid;
		wxStaticText* m_StaticTextGISLayerGeogridTransp;
		wxTextCtrl* m_TextCtrlGISLayerGeogridTransp;
		wxStaticText* m_StaticTextGISLayerGeogridColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerGeogridColor;
		wxStaticText* m_StaticTextGISLayerGeogridSize;
		wxTextCtrl* m_TextCtrlGISLayerGeogridSize;
		wxPanel* m_PanelLayerLatLong;
		wxStaticText* m_StaticTextGISLayerLatLongVisibility;
		wxCheckBox* m_CheckBoxGISLayerLatLongVisibility;
		wxStaticText* m_StaticTextGISLayerLatLongFile;
		wxFilePickerCtrl* m_FilePickerGISLayerLatLong;
		wxStaticText* m_StaticTextGISLayerLatLongTransp;
		wxTextCtrl* m_TextCtrlGISLayerLatLongTransp;
		wxStaticText* m_StaticTextGISLayerLatLongColor;
		wxColourPickerCtrl* m_ColourPickerGISLayerLatLongColor;
		wxStaticText* m_StaticTextGISLayerLatLongSize;
		wxTextCtrl* m_TextCtrlGISLayerLatLongSize;
		wxPanel* m_PanelAdvanced;
		wxNotebook* m_NotebookAdvanced;
		wxPanel* m_PanelGeneral;
		wxRadioBox* m_RadioBoxGui;
		wxStaticText* m_StaticTextNumberFails;
		wxTextCtrl* m_TextCtrlMaxPrevStepsNb;
		wxStaticText* m_StaticTextMaxRequestsNb;
		wxTextCtrl* m_TextCtrlMaxRequestsNb;
		wxCheckBox* m_CheckBoxRestrictDownloads;
		wxCheckBox* m_CheckBoxResponsiveness;
		wxCheckBox* m_CheckBoxMultiInstancesForecaster;
		wxCheckBox* m_CheckBoxMultiInstancesViewer;
		wxPanel* m_PanelProcessing;
		wxCheckBox* m_CheckBoxAllowMultithreading;
		wxStaticText* m_StaticTextThreadsNb;
		wxTextCtrl* m_TextCtrlThreadsNb;
		wxStaticText* m_StaticTextThreadsPriority;
		wxSlider* m_SliderThreadsPriority;
		wxRadioBox* m_RadioBoxProcessingMethods;
		wxRadioBox* m_RadioBoxLinearAlgebra;
		wxPanel* m_PanelUserDirectories;
		wxStaticText* m_StaticTextIntermediateResultsDir;
		wxDirPickerCtrl* m_DirPickerIntermediateResults;
		wxStaticText* m_StaticTextUserDirLabel;
		wxStaticText* m_StaticTextUserDir;
		wxStaticText* m_StaticTextLogFileForecasterLabel;
		wxStaticText* m_StaticTextLogFileForecaster;
		wxStaticText* m_StaticTextLogFileViewerLabel;
		wxStaticText* m_StaticTextLogFileViewer;
		wxStaticText* m_StaticTextPrefFileForecasterLabel;
		wxStaticText* m_StaticTextPrefFileForecaster;
		wxStaticText* m_StaticTextPrefFileViewerLabel;
		wxStaticText* m_StaticTextPrefFileViewer;
		wxPanel* m_PanelPathsCatalogs;
		wxStaticText* m_StaticTextCatalogPredictorArchive;
		wxFilePickerCtrl* m_FilePickerCatalogPredictorsArchive;
		wxStaticText* m_StaticTextCatalogPredictorRealtime;
		wxFilePickerCtrl* m_FilePickerCatalogPredictorsRealtime;
		wxStaticText* m_StaticTextCatalogPredictand;
		wxFilePickerCtrl* m_FilePickerCatalogPredictands;
		wxStdDialogButtonSizer* m_ButtonsConfirmation;
		wxButton* m_ButtonsConfirmationOK;
		wxButton* m_ButtonsConfirmationApply;
		wxButton* m_ButtonsConfirmationCancel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnChangeMultithreadingCheckBox( wxCommandEvent& event ) { event.Skip(); }
		virtual void ApplyChanges( wxCommandEvent& event ) { event.Skip(); }
		virtual void CloseFrame( wxCommandEvent& event ) { event.Skip(); }
		virtual void SaveAndClose( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePreferencesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Preferences"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 482,534 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePreferencesVirtual();
	
};

#endif //__ATMOSWINGSHAREDGUI_H__
