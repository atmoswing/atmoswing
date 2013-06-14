///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct  8 2012)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __ATMOSWINGVIEWERGUI_H__
#define __ATMOSWINGVIEWERGUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/intl.h>
#include <wx/sizer.h>
#include <wx/gdicmn.h>
#include <wx/scrolwin.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/panel.h>
#include <wx/splitter.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/menu.h>
#include <wx/toolbar.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/button.h>
#include <wx/checklst.h>
#include <wx/choice.h>
#include <wx/notebook.h>
#include <wx/grid.h>
#include <wx/bmpbuttn.h>
#include <wx/plotctrl/plotctrl.h>
#ifdef __VISUALC__
#include <wx/link_additions.h>
#endif //__VISUALC__

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class asFrameForecastVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameForecastVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelMain;
		wxSplitterWindow* m_SplitterGIS;
		wxScrolledWindow* m_ScrolledWindowOptions;
		wxBoxSizer* m_SizerScrolledWindow;
		wxPanel* m_PanelContent;
		wxBoxSizer* m_SizerContent;
		wxPanel* m_PanelTop;
		wxBoxSizer* m_SizerTop;
		wxBoxSizer* m_SizerTopLeft;
		wxStaticText* m_StaticTextForecastDate;
		wxStaticText* m_StaticTextForecastModel;
		wxBoxSizer* m_SizerTopRight;
		wxPanel* m_PanelGIS;
		wxBoxSizer* m_SizerGIS;
		wxMenuBar* m_MenuBar;
		wxMenu* m_MenuFile;
		wxMenu* m_MenuOptions;
		wxMenu* m_MenuLog;
		wxMenu* m_MenuLogLevel;
		wxMenu* m_MenuHelp;
		wxToolBar* m_ToolBar;
		wxStatusBar* m_StatusBar;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnQuit( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenLayer( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnCloseLayer( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenForecast( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFramePreferences( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnShowLog( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel1( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel2( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnLogLevel3( wxCommandEvent& event ) { event.Skip(); }
		virtual void OpenFrameAbout( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameForecastVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Atmoswing Viewer"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameForecastVirtual();
		
		void m_SplitterGISOnIdle( wxIdleEvent& )
		{
			m_SplitterGIS->SetSashPosition( 270 );
			m_SplitterGIS->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFrameForecastVirtual::m_SplitterGISOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePlotTimeSeriesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePlotTimeSeriesVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelStationName;
		wxStaticText* m_StaticTextStationName;
		wxButton* m_ButtonSVG;
		wxButton* m_ButtonSavePdf;
		wxButton* m_ButtonPreview;
		wxButton* m_ButtonPrint;
		wxSplitterWindow* m_Splitter;
		wxPanel* m_PanelLeft;
		wxCheckListBox* m_CheckListToc;
		wxCheckListBox* m_CheckListPast;
		wxPanel* m_PanelRight;
		wxBoxSizer* m_SizerPlot;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnExportSVG( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnExportTXT( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPreview( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPrint( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTocSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePlotTimeSeriesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Forecast plots"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 700,400 ), long style = wxDEFAULT_FRAME_STYLE|wxFRAME_FLOAT_ON_PARENT|wxTAB_TRAVERSAL );
		
		~asFramePlotTimeSeriesVirtual();
		
		void m_SplitterOnIdle( wxIdleEvent& )
		{
			m_Splitter->SetSashPosition( 150 );
			m_Splitter->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotTimeSeriesVirtual::m_SplitterOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePlotDistributionsVirutal
///////////////////////////////////////////////////////////////////////////////
class asFramePlotDistributionsVirutal : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelOptions;
		wxStaticText* m_StaticTextForecast;
		wxStaticText* m_StaticTextStation;
		wxStaticText* m_StaticTextDate;
		wxChoice* m_ChoiceForecast;
		wxChoice* m_ChoiceStation;
		wxChoice* m_ChoiceDate;
		wxNotebook* m_Notebook;
		wxPanel* m_PanelPredictands;
		wxSplitterWindow* m_SplitterPredictands;
		wxPanel* m_PanelPredictandsLeft;
		wxCheckListBox* m_CheckListTocPredictands;
		wxPanel* m_PanelPredictandsRight;
		wxBoxSizer* m_SizerPlotPredictands;
		wxPanel* m_PanelCriteria;
		wxBoxSizer* m_SizerPlotCriteria;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnChoiceForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceStationChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTocSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePlotDistributionsVirutal( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Distribution plots"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 800,500 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePlotDistributionsVirutal();
		
		void m_SplitterPredictandsOnIdle( wxIdleEvent& )
		{
			m_SplitterPredictands->SetSashPosition( 150 );
			m_SplitterPredictands->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePlotDistributionsVirutal::m_SplitterPredictandsOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFrameGridAnalogsValuesVirtual
///////////////////////////////////////////////////////////////////////////////
class asFrameGridAnalogsValuesVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_PanelOptions;
		wxStaticText* m_StaticTextForecast;
		wxStaticText* m_StaticTextStation;
		wxStaticText* m_StaticTextDate;
		wxChoice* m_ChoiceForecast;
		wxChoice* m_ChoiceStation;
		wxChoice* m_ChoiceDate;
		wxGrid* m_Grid;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnChoiceForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceStationChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnChoiceDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void SortGrid( wxGridEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFrameGridAnalogsValuesVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Analogs details"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 550,300 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFrameGridAnalogsValuesVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asFramePredictorsVirtual
///////////////////////////////////////////////////////////////////////////////
class asFramePredictorsVirtual : public wxFrame 
{
	private:
	
	protected:
		wxPanel* m_panel15;
		wxSplitterWindow* m_SplitterToc;
		wxScrolledWindow* m_ScrolledWindowOptions;
		wxBoxSizer* m_SizerScrolledWindow;
		wxStaticText* m_StaticTextChoiceForecast;
		wxChoice* m_ChoiceForecast;
		wxStaticText* m_StaticTextCheckListPredictors;
		wxCheckListBox* m_CheckListPredictors;
		wxStaticText* m_StaticTextTocLeft;
		wxStaticText* m_StaticTextTocRight;
		wxPanel* m_PanelGIS;
		wxBoxSizer* m_SizerGIS;
		wxPanel* m_PanelLeft;
		wxStaticText* m_StaticTextTargetDates;
		wxChoice* m_ChoiceTargetDates;
		wxPanel* m_PanelGISLeft;
		wxBoxSizer* m_SizerGISLeft;
		wxPanel* m_PanelSwitch;
		wxBitmapButton* m_BpButtonSwitchRight;
		wxBitmapButton* m_BpButtonSwitchLeft;
		wxPanel* m_PanelRight;
		wxStaticText* m_StaticTextAnalogDates;
		wxChoice* m_ChoiceAnalogDates;
		wxPanel* m_PanelGISRight;
		wxBoxSizer* m_SizerGISRight;
		wxMenuBar* m_Menubar;
		wxMenu* m_MenuFile;
		wxMenu* m_MenuTools;
		wxToolBar* m_ToolBar;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnForecastChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnPredictorSelectionChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnTargetDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSwitchRight( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnSwitchLeft( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnAnalogDateChange( wxCommandEvent& event ) { event.Skip(); }
		virtual void OnOpenLayer( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asFramePredictorsVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = _("Predictors overview"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 800,600 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~asFramePredictorsVirtual();
		
		void m_SplitterTocOnIdle( wxIdleEvent& )
		{
			m_SplitterToc->SetSashPosition( 170 );
			m_SplitterToc->Disconnect( wxEVT_IDLE, wxIdleEventHandler( asFramePredictorsVirtual::m_SplitterTocOnIdle ), NULL, this );
		}
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelPlotVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelPlotVirtual : public wxPanel 
{
	private:
	
	protected:
		wxPlotCtrl* m_PlotCtrl;
	
	public:
		
		asPanelPlotVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxTAB_TRAVERSAL ); 
		~asPanelPlotVirtual();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class asPanelSidebarVirtual
///////////////////////////////////////////////////////////////////////////////
class asPanelSidebarVirtual : public wxPanel 
{
	private:
	
	protected:
		wxBoxSizer* m_SizerMain;
		wxPanel* m_PanelHeader;
		wxStaticText* m_Header;
		wxBitmapButton* m_BpButtonReduce;
		wxBoxSizer* m_SizerContent;
		
		// Virtual event handlers, overide them in your derived class
		virtual void OnReducePanel( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		asPanelSidebarVirtual( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( -1,-1 ), long style = wxSIMPLE_BORDER|wxTAB_TRAVERSAL ); 
		~asPanelSidebarVirtual();
	
};

#endif //__ATMOSWINGVIEWERGUI_H__
