/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asFrameForecastDots.h"

#include "asFrameForecastRings.h"
#include "asResultsAnalogsForecast.h"


BEGIN_EVENT_TABLE(asFrameForecastDots, wxFrame)
    EVT_END_PROCESS(wxID_ANY, asFrameForecast::OnForecastProcessTerminate)
/*    EVT_MENU(wxID_EXIT,  asFrameForecast::OnQuit)
    EVT_MENU(wxID_ABOUT, asFrameForecast::OnAbout)
	EVT_MENU(wxID_OPEN, asFrameForecast::OnOpenLayer)
	EVT_MENU(wxID_REMOVE, asFrameForecast::OnCloseLayer)
	EVT_MENU (wxID_INFO, asFrameForecast::OnShowLog)*/
	EVT_MENU (asID_SELECT, asFrameForecast::OnToolSelect)
	EVT_MENU (asID_ZOOM_IN, asFrameForecast::OnToolZoomIn)
	EVT_MENU (asID_ZOOM_OUT, asFrameForecast::OnToolZoomOut)
	EVT_MENU (asID_ZOOM_FIT, asFrameForecast::OnToolZoomToFit)
	EVT_MENU (asID_PAN, asFrameForecast::OnToolPan)/*
	EVT_MENU (vlID_MOVE_LAYER, asFrameForecast::OnMoveLayer)
*/
    EVT_KEY_DOWN(asFrameForecast::OnKeyDown)
	EVT_KEY_UP(asFrameForecast::OnKeyUp)

	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOM, asFrameForecast::OnToolAction)
	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOMOUT, asFrameForecast::OnToolAction)
	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_SELECT, asFrameForecast::OnToolAction)
	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_PAN, asFrameForecast::OnToolAction)

    EVT_COMMAND(wxID_ANY, asEVT_ACTION_STATION_SELECTION_CHANGED, asFrameForecast::OnStationSelection)
//    EVT_COMMAND(wxID_ANY, asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED, asFrameForecast::OnStationSelection)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_CLEAR, asFrameForecast::OnForecastClear)
	EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_NEW_ADDED, asFrameForecastDots::OnForecastNewAdded)
	EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, asFrameForecastDots::OnForecastRatioSelectionChange)
	EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, asFrameForecastDots::OnForecastModelSelectionChange)
	EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_PERCENTILE_SELECTION_CHANGED, asFrameForecastDots::OnForecastPercentileSelectionChange)
END_EVENT_TABLE()


asFrameForecastDots::asFrameForecastDots( wxWindow* parent, asFrameForecastRings* frameRings, wxWindowID id )
:
asFrameForecast( parent, id )
{
    m_FrameRings = frameRings;

    m_ForecastViewer->SetDisplayType(asForecastViewer::ForecastDots);

    // Add slider
    m_SliderLeadTime = new wxSlider( m_PanelTop, wxID_ANY, 0, 0, 1, wxDefaultPosition, wxDefaultSize, 0 );
	m_SliderLeadTime->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_SliderLeadTime->SetBackgroundColour( wxColour( 77, 77, 77 ) );
	m_SizerTopRight->Add( m_SliderLeadTime, 0, wxALIGN_RIGHT|wxEXPAND|wxTOP|wxRIGHT|wxLEFT, 5 );

    // Add date display text
    m_StaticTextLeadTime = new wxStaticText( m_PanelTop, wxID_ANY, _("No data"), wxDefaultPosition, wxDefaultSize, 0 );
	m_StaticTextLeadTime->Wrap( -1 );
	m_StaticTextLeadTime->SetForegroundColour( wxColour( 255, 255, 255 ) );
	m_SizerTopRight->Add( m_StaticTextLeadTime, 0, wxBOTTOM|wxRIGHT|wxLEFT|wxALIGN_CENTER_HORIZONTAL, 5 );

    m_PanelTop->Layout();
	m_SizerTop->Fit( m_PanelTop );
	m_SizerContent->Fit(m_PanelContent);
	this->Layout();
	Refresh();

	// Analog dates sidebar
	m_PanelSidebarAnalogDates = new asPanelSidebarAnalogDates( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarAnalogDates->Layout();
	m_SizerScrolledWindow->Add( m_PanelSidebarAnalogDates, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Caption panel
    m_PanelSidebarCaptionForecastDots = new asPanelSidebarCaptionForecastDots( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
	m_PanelSidebarCaptionForecastDots->Layout();
	m_SizerScrolledWindow->Add( m_PanelSidebarCaptionForecastDots, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Connect Events
	m_SliderLeadTime->Connect( wxEVT_SCROLL_TOP, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this  );
    m_SliderLeadTime->Connect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Connect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
}

asFrameForecastDots::~asFrameForecastDots()
{
    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/SidebarPanelsDisplay/AnalogDates", !m_PanelSidebarAnalogDates->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/CaptionForecastDots", !m_PanelSidebarCaptionForecastDots->IsReduced());

    // Disconnect Events
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_TOP, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_BOTTOM, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_LINEUP, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_LINEDOWN, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_PAGEUP, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_PAGEDOWN, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_THUMBTRACK, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_THUMBRELEASE, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );
	m_SliderLeadTime->Disconnect( wxEVT_SCROLL_CHANGED, wxScrollEventHandler( asFrameForecastDots::OnLeadtimeChange ), NULL, this );

	m_FrameRings->NullFrameDotsPointer();
}

void asFrameForecastDots::OnInit()
{
    asFrameForecast::OnInit();

    // Reduce some panels
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool display = true;
    pConfig->Read("/SidebarPanelsDisplay/AnalogDates", &display, true);
    if (!display)
    {
        m_PanelSidebarAnalogDates->ReducePanel();
        m_PanelSidebarAnalogDates->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/CaptionForecastDots", &display, true);
    if (!display)
    {
        m_PanelSidebarCaptionForecastDots->ReducePanel();
        m_PanelSidebarCaptionForecastDots->Layout();
    }

    m_ScrolledWindowOptions->Layout();
    Layout();
    Refresh();
}

void asFrameForecastDots::OnQuit( wxCommandEvent& event )
{
	event.Skip();
}

void asFrameForecastDots::OnLeadtimeChange(wxScrollEvent &event)
{
    m_ForecastViewer->ChangeLeadTime(event.GetInt());

    UpdateHeaderTexts();
    UpdatePanelAnalogDates();
}

void asFrameForecastDots::OnForecastRatioSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetForecastDisplay(event.GetInt());

    UpdatePanelCaption();
}

void asFrameForecastDots::OnForecastModelSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetModel(event.GetInt());

    UpdateHeaderTexts();
    UpdatePanelCaption();
    UpdatePanelAnalogDates();
    UpdateLeadTimeSlider();
    UpdatePanelStationsList();
}

void asFrameForecastDots::OnForecastPercentileSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetPercentile(event.GetInt());
}

void asFrameForecastDots::OnForecastSelectionChange( wxCommandEvent& event )
{




    /*
    // Update header texts
    UpdateHeaderTexts();

    // Update caption
    m_PanelSidebarCaptionForecastDots->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());

    // Update stations list
    wxArrayString arrayStation = m_ForecastManager->GetStationNamesWithHeights(m_ForecastViewer->GetModelSelection());
    m_PanelSidebarStationsList->SetChoices(arrayStation);

    // Update analog dates list
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection());
    Array1DFloat arrayDate = forecast->GetAnalogsDates(m_ForecastViewer->GetLeadTimeIndex());
    Array1DFloat arrayCriteria = forecast->GetAnalogsCriteria(m_ForecastViewer->GetLeadTimeIndex());
    m_PanelSidebarAnalogDates->SetChoices(arrayDate, arrayCriteria);

    // Update the lead time slider
    int leadtimeNb = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection())->GetTargetDatesLength();
    m_SliderLeadTime->SetMax(leadtimeNb-1);
    m_SliderLeadTime->Layout();*/
}

void asFrameForecastDots::OnForecastNewAdded( wxCommandEvent& event )
{
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(event.GetInt());
	m_PanelSidebarForecasts->AddForecast(forecast->GetModelName(), forecast->GetLeadTimeOriginString(), forecast->GetPredictandParameter(), forecast->GetPredictandTemporalResolution());

    m_ForecastViewer->SetModel(event.GetInt());

    UpdateHeaderTexts();
    UpdatePanelCaption();
    UpdatePanelAnalogDates();
    UpdateLeadTimeSlider();
    UpdatePanelStationsList();
}

void asFrameForecastDots::UpdateHeaderTexts()
{
    // Set header text
	wxString dateForecast = asTime::GetStringTime(m_ForecastManager->GetLeadTimeOrigin(), "DD.MM.YYYY");
	wxString dateTarget = asTime::GetStringTime(m_ForecastViewer->GetSelectedTargetDate(), "DD.MM.YYYY");
	wxString dateStr = wxString::Format(_("Forecast of the %s for the %s"), dateForecast.c_str(), dateTarget.c_str());
	m_StaticTextForecastDate->SetLabel(dateStr);
	wxString dateTargetStr = wxString::Format(_("%s"), dateTarget.c_str());
	m_StaticTextLeadTime->SetLabel(dateTargetStr);

	wxString model = m_ForecastManager->GetModelName(m_ForecastViewer->GetModelSelection());
	wxString modelStr = wxString::Format(_("Model selected : %s"), model.c_str());
    m_StaticTextForecastModel->SetLabel(modelStr);

    m_PanelTop->Layout();
    m_PanelTop->Refresh();
}

void asFrameForecastDots::UpdateLeadTimeSlider()
{
    int leadtimeNb = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection())->GetTargetDatesLength();
    m_SliderLeadTime->SetMax(leadtimeNb-1);
    m_SliderLeadTime->Layout();
}

void asFrameForecastDots::UpdatePanelCaption()
{
    m_PanelSidebarCaptionForecastDots->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
}

void asFrameForecastDots::UpdatePanelAnalogDates()
{
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection());
    Array1DFloat arrayDate = forecast->GetAnalogsDates(m_ForecastViewer->GetLeadTimeIndex());
    Array1DFloat arrayCriteria = forecast->GetAnalogsCriteria(m_ForecastViewer->GetLeadTimeIndex());
    m_PanelSidebarAnalogDates->SetChoices(arrayDate, arrayCriteria);
}

void asFrameForecastDots::UpdatePanelStationsList()
{
    wxArrayString arrayStation = m_ForecastManager->GetStationNamesWithHeights(m_ForecastViewer->GetModelSelection());
    m_PanelSidebarStationsList->SetChoices(arrayStation);
}
