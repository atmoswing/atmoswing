/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include "asFrameForecastRings.h"

#include "asFrameForecastDots.h"
#include "asResultsAnalogsForecast.h"
#include "img_toolbar.h"


BEGIN_EVENT_TABLE(asFrameForecastRings, wxFrame)
    EVT_END_PROCESS(wxID_ANY, asFrameForecast::OnForecastProcessTerminate)
    EVT_MENU(wxID_EXIT,  asFrameForecast::OnQuit)
/*    EVT_MENU(wxID_ABOUT, asFrameForecast::OnAbout)
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
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_CLEAR, asFrameForecast::OnForecastClear)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_NEW_ADDED, asFrameForecastRings::OnForecastNewAdded)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, asFrameForecastRings::OnForecastRatioSelectionChange)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, asFrameForecastRings::OnForecastModelSelectionChange)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_FORECAST_PERCENTILE_SELECTION_CHANGED, asFrameForecastRings::OnForecastPercentileSelectionChange)
END_EVENT_TABLE()


asFrameForecastRings::asFrameForecastRings( wxWindow* parent, wxWindowID id )
:
asFrameForecast( parent, id )
{
    m_FrameDots = NULL;

    // Toolbar
    m_ToolBar->InsertTool(13, asID_FRAME_DOTS, wxT("Open time evolution"), img_frame_dots, img_frame_dots, wxITEM_NORMAL, _("Open time evolution"), _("Open time evolution"), NULL);
    m_ToolBar->Realize();

    // Alarms
    m_PanelSidebarAlarms = new asPanelSidebarAlarms( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarAlarms->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarAlarms, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    // Caption panel
    m_PanelSidebarCaptionForecastRing = new asPanelSidebarCaptionForecastRing( m_ScrolledWindowOptions, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxTAB_TRAVERSAL );
    m_PanelSidebarCaptionForecastRing->Layout();
    m_SizerScrolledWindow->Add( m_PanelSidebarCaptionForecastRing, 0, wxEXPAND|wxTOP|wxBOTTOM, 2 );

    m_SizerScrolledWindow->Fit(m_ScrolledWindowOptions);

    Layout();

    // Events
    this->Connect( asID_FRAME_DOTS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecastRings::OpenFrameDots ) );
}

asFrameForecastRings::~asFrameForecastRings()
{
    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/SidebarPanelsDisplay/Alarms", !m_PanelSidebarAlarms->IsReduced());
    pConfig->Write("/SidebarPanelsDisplay/CaptionForecastRing", !m_PanelSidebarCaptionForecastRing->IsReduced());

    if (m_FrameDots!=NULL)
    {
        m_FrameDots->Destroy();
        m_FrameDots = NULL;
    }

    this->Disconnect( asID_FRAME_DOTS, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameForecastRings::OpenFrameDots ) );
}

void asFrameForecastRings::OnInit()
{
    asFrameForecast::OnInit();

    // Reduce some panels
    wxConfigBase *pConfig = wxFileConfig::Get();
    bool display = true;
    pConfig->Read("/SidebarPanelsDisplay/Alarms", &display, true);
    if (!display)
    {
        m_PanelSidebarAlarms->ReducePanel();
        m_PanelSidebarAlarms->Layout();
    }
    pConfig->Read("/SidebarPanelsDisplay/CaptionForecastRing", &display, true);
    if (!display)
    {
        m_PanelSidebarCaptionForecastRing->ReducePanel();
        m_PanelSidebarCaptionForecastRing->Layout();
    }

    m_ScrolledWindowOptions->Layout();
    Layout();
    Refresh();
}

void asFrameForecastRings::OnQuit( wxCommandEvent& event )
{
    event.Skip();
}

void asFrameForecastRings::OnForecastRatioSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetForecastDisplay(event.GetInt());

    UpdatePanelCaptionColorbar();
}

void asFrameForecastRings::OnForecastModelSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetModel(event.GetInt());

    UpdateHeaderTexts();
    UpdatePanelCaptionAll();
    UpdatePanelStationsList();
}

void asFrameForecastRings::OnForecastPercentileSelectionChange( wxCommandEvent& event )
{
    m_ForecastViewer->SetPercentile(event.GetInt());
}

void asFrameForecastRings::OnForecastSelectionChange( wxCommandEvent& event )
{
    // Already processed.
}

void asFrameForecastRings::OnForecastNewAdded( wxCommandEvent& event )
{
    asResultsAnalogsForecast* forecastLast = m_ForecastManager->GetCurrentForecast(event.GetInt());
    m_PanelSidebarForecasts->AddForecast(forecastLast->GetModelName(), forecastLast->GetLeadTimeOriginString(), forecastLast->GetPredictandParameter(), forecastLast->GetPredictandTemporalResolution());

    if (event.GetString().IsSameAs("last"))
    {
        m_ForecastViewer->SetModel(event.GetInt());

        UpdatePanelAlarms();
        UpdateHeaderTexts();
        UpdatePanelCaptionAll();
        UpdatePanelStationsList();
    }

}

void asFrameForecastRings::UpdateHeaderTexts()
{
    // Set header text
    wxString date = asTime::GetStringTime(m_ForecastManager->GetLeadTimeOrigin(), "DD.MM.YYYY");
    int length = m_ForecastManager->GetLeadTimeLength(m_ForecastViewer->GetModelSelection());
    wxString dateStr = wxString::Format(_("Forecast of the %s for the next %d days"), date.c_str(), length);
    m_StaticTextForecastDate->SetLabel(dateStr);

    wxString model = m_ForecastManager->GetModelName(m_ForecastViewer->GetModelSelection());
    wxString modelStr = wxString::Format(_("Model selected : %s"), model.c_str());
    m_StaticTextForecastModel->SetLabel(modelStr);
}

void asFrameForecastRings::UpdatePanelCaptionAll()
{
    m_PanelSidebarCaptionForecastRing->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_ForecastViewer->GetModelSelection());
    Array1DFloat dates = forecast->GetTargetDates();
    m_PanelSidebarCaptionForecastRing->SetDates(dates);
}

void asFrameForecastRings::UpdatePanelCaptionColorbar()
{
    m_PanelSidebarCaptionForecastRing->SetColorbarMax(m_ForecastViewer->GetLayerMaxValue());
}

void asFrameForecastRings::UpdatePanelStationsList()
{
    wxArrayString arrayStation = m_ForecastManager->GetStationNamesWithHeights(m_ForecastViewer->GetModelSelection());
    m_PanelSidebarStationsList->SetChoices(arrayStation);
}

void asFrameForecastRings::UpdatePanelAlarms()
{
    Array1DFloat datesFull = m_ForecastManager->GetFullTargetDatesVector();
    VectorString models = m_ForecastManager->GetModelsNames();
    std::vector <asResultsAnalogsForecast*> forecasts = m_ForecastManager->GetCurrentForecasts();
    m_PanelSidebarAlarms->UpdateAlarms(datesFull, models, forecasts);
}

void asFrameForecastRings::OpenFrameDots( wxCommandEvent& event )
{
    if (m_FrameDots!=NULL)
    {
        m_FrameDots->Destroy();
        m_FrameDots = NULL;
    }

    m_FrameDots = new asFrameForecastDots(0L, this);
    m_FrameDots->OnInit();
// TODO (phorton#9#): Set the icon
//    m_FrameDots->SetIcon(wxICON(aaaa)); // To Set App Icon


    // Open actuel forecasts.
    if (m_ForecastManager->GetModelsNb()>0)
    {
        m_FrameDots->OpenForecast(m_ForecastManager->GetFilePathsWxArray());
    }

    m_FrameDots->Show();
}

void asFrameForecastRings::NullFrameDotsPointer()
{
    m_FrameDots = NULL;
}
