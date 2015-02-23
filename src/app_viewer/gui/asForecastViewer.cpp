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
 
#include "asForecastViewer.h"

#include "asForecastManager.h"
#include "asResultsAnalogsForecast.h"
#include "asFrameForecast.h"
#include "vrLayerVectorFcstRing.h"
#include "vrLayerVectorFcstDots.h"
#include "vrlayervector.h"
#include "vrrender.h"


wxDEFINE_EVENT(asEVT_ACTION_FORECAST_SELECT_FIRST, wxCommandEvent);

asForecastViewer::asForecastViewer( asFrameForecast* parent, asForecastManager *forecastManager, vrLayerManager *layerManager, vrViewerLayerManager *viewerLayerManager)
{
    m_Parent = parent;
    m_ForecastManager = forecastManager;
    m_LayerManager = layerManager;
    m_ViewerLayerManager = viewerLayerManager;
    m_LeadTimeIndex = 0;
    m_LeadTimeDate = 0;
    m_LayerMaxValue = 1;

    m_DisplayForecast.Add(_("Value"));
    m_DisplayForecast.Add(_("Ratio P/P2"));
    m_DisplayForecast.Add(_("Ratio P/P5"));
    m_DisplayForecast.Add(_("Ratio P/P10"));
    m_DisplayForecast.Add(_("Ratio P/P20"));
    m_DisplayForecast.Add(_("Ratio P/P50"));
    m_DisplayForecast.Add(_("Ratio P/P100"));
    m_DisplayForecast.Add(_("Ratio P/P200"));
    m_DisplayForecast.Add(_("Ratio P/P300"));
    m_DisplayForecast.Add(_("Ratio P/P500"));

    m_ReturnPeriods.push_back(0);
    m_ReturnPeriods.push_back(2);
    m_ReturnPeriods.push_back(5);
    m_ReturnPeriods.push_back(10);
    m_ReturnPeriods.push_back(20);
    m_ReturnPeriods.push_back(50);
    m_ReturnPeriods.push_back(100);
    m_ReturnPeriods.push_back(200);
    m_ReturnPeriods.push_back(300);
    m_ReturnPeriods.push_back(500);

    //m_DisplayPercentiles.Add(_("interpretation"));
    m_DisplayPercentiles.Add(_("q90"));
    m_DisplayPercentiles.Add(_("q60"));
    m_DisplayPercentiles.Add(_("q20"));

    //m_Percentiles.push_back(-1);
    m_Percentiles.push_back(0.9f);
    m_Percentiles.push_back(0.6f);
    m_Percentiles.push_back(0.2f);

    m_MethodSelection = -1;
    m_ForecastSelection = -1;

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/ForecastViewer/DisplaySelection", &m_ForecastDisplaySelection, 3);
    pConfig->Read("/ForecastViewer/PercentileSelection", &m_PercentileSelection, 0);
    if ((unsigned)m_ForecastDisplaySelection>=m_ReturnPeriods.size())
    {
        m_ForecastDisplaySelection = 1;
    }
    if ((unsigned)m_PercentileSelection>=m_Percentiles.size())
    {
        m_PercentileSelection = 0;
    }

    m_Opened = false;
}

asForecastViewer::~asForecastViewer()
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ForecastViewer/DisplaySelection", m_ForecastDisplaySelection);
    pConfig->Write("/ForecastViewer/PercentileSelection", m_PercentileSelection);
}

wxArrayString asForecastViewer::GetForecastDisplayStringArray()
{
    return m_DisplayForecast;
}

wxArrayString asForecastViewer::GetPercentilesStringArray()
{
    return m_DisplayPercentiles;
}

void asForecastViewer::FixForecastSelection()
{
    if (m_MethodSelection<0)
    {
        wxCommandEvent eventSlct (asEVT_ACTION_FORECAST_SELECT_FIRST);
        m_Parent->ProcessWindowEvent(eventSlct);
    }
}

void asForecastViewer::SetForecast(int methodRow, int forecastRow)
{
    m_MethodSelection = methodRow;
    m_ForecastSelection = forecastRow;

    Redraw();
}

wxString asForecastViewer::GetStationName(int i_stat)
{
    return m_ForecastManager->GetStationName(m_MethodSelection, m_ForecastSelection, i_stat);
}

float asForecastViewer::GetSelectedTargetDate()
{
    Array1DFloat targetDates;

    if(m_ForecastSelection>0)
    {
        targetDates = m_ForecastManager->GetTargetDates(m_MethodSelection, m_ForecastSelection);
    }
    else
    {
        targetDates = m_ForecastManager->GetTargetDates(m_MethodSelection);
    }

    wxASSERT(m_LeadTimeIndex>=0);
    if (m_LeadTimeIndex>=targetDates.size())
    {
        return 0;
    }
    return targetDates[m_LeadTimeIndex];
}

void asForecastViewer::SetLeadTimeDate(float date)
{
    if (date>0 && (m_MethodSelection>0))
    {
        Array1DFloat targetDates;

        if(m_ForecastSelection>0)
        {
            targetDates = m_ForecastManager->GetTargetDates(m_MethodSelection, m_ForecastSelection);
        }
        else
        {
            targetDates = m_ForecastManager->GetTargetDates(m_MethodSelection);
        }

        int index = asTools::SortedArraySearchClosest(&targetDates[0], &targetDates[targetDates.size()-1], date);
        if (index>=0)
        {
            m_LeadTimeIndex = index;
        }
    }
}

void asForecastViewer::SetForecastDisplay(int i)
{
    m_ForecastDisplaySelection = i;

    wxString display = m_DisplayForecast.Item(m_ForecastDisplaySelection);
    asLogMessage(wxString::Format(_("Selected display : %s."), display.c_str()));

    Redraw();
}

void asForecastViewer::SetPercentile(int i)
{
    m_PercentileSelection = i;

    wxString percentile = m_DisplayPercentiles.Item(m_PercentileSelection);
    asLogMessage(wxString::Format(_("Selected percentile : %s."), percentile.c_str()));

    Redraw();
}

void asForecastViewer::LoadPastForecast()
{
    wxBusyCursor wait;

    // Check that elements are selected
    if ( (m_MethodSelection==-1) || (m_ForecastDisplaySelection==-1) || (m_PercentileSelection==-1) ) return;
    if ( m_MethodSelection >= m_ForecastManager->GetMethodsNb() ) return;
    
    if (m_ForecastSelection>0)
    {
        m_ForecastManager->LoadPastForecast(m_MethodSelection, m_ForecastSelection);
    }
    else
    {
        m_ForecastManager->LoadPastForecast(m_MethodSelection);
    }
}

void asForecastViewer::Redraw()
{
    // Check that elements are selected
    if ( (m_MethodSelection==-1) || (m_ForecastDisplaySelection==-1) || (m_PercentileSelection==-1) ) return;
    if ( m_MethodSelection >= m_ForecastManager->GetMethodsNb() ) return;
    if ( (unsigned)m_ForecastDisplaySelection >= m_DisplayForecast.size() ) return;
    if ( m_Percentiles.size() != m_DisplayPercentiles.size() ) return;
    if ( m_ReturnPeriods.size() != m_DisplayForecast.size() ) return;
    
    // Get data
    vector <asResultsAnalogsForecast*> forecasts;

    if (m_ForecastSelection<0)
    {
        for (int i=0; i<m_ForecastManager->GetForecastsNb(m_MethodSelection); i++)
        {
            forecasts.push_back(m_ForecastManager->GetForecast(m_MethodSelection, i));
        }
    }
    else
    {
        forecasts.push_back(m_ForecastManager->GetForecast(m_MethodSelection, m_ForecastSelection));
    }

    // Create a memory layer
    wxFileName memoryLayerNameSpecific ("", "Forecast - specific", "memory");
    wxFileName memoryLayerNameOther ("", "Forecast - other", "memory");

    // Check if memory layer already added
    m_ViewerLayerManager->FreezeBegin();
    for (int i = 0; i < m_ViewerLayerManager->GetCount(); i++)
    {
        if (m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerNameSpecific)
        {
            vrRenderer *renderer = m_ViewerLayerManager->GetRenderer(i);
            vrLayer *layer = renderer->GetLayer();
            wxASSERT(renderer);
            m_ViewerLayerManager->Remove(renderer);
            // Close layer
            m_LayerManager->Close(layer);
        }
    }
    for (int i = 0; i < m_ViewerLayerManager->GetCount(); i++)
    {
        if (m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerNameOther)
        {
            vrRenderer *renderer = m_ViewerLayerManager->GetRenderer(i);
            vrLayer *layer = renderer->GetLayer();
            wxASSERT(renderer);
            m_ViewerLayerManager->Remove(renderer);
            // Close layer
            m_LayerManager->Close(layer);
        }
    }

    // Get display option
    float percentile = m_Percentiles[m_PercentileSelection];
    float returnPeriod = m_ReturnPeriods[m_ForecastDisplaySelection];

    // Get reference axis index
    int indexReferenceAxis = asNOT_FOUND;
    if (forecasts[0]->HasReferenceValues() && returnPeriod!=0)
    {
        Array1DFloat forecastReferenceAxis = forecasts[0]->GetReferenceAxis();

        indexReferenceAxis = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriod);
        if ( (indexReferenceAxis==asNOT_FOUND) || (indexReferenceAxis==asOUT_OF_RANGE) )
        {
            asLogError(_("The desired reference value is not available in the forecast file."));
            m_ViewerLayerManager->FreezeEnd();
            return;
        }
    }

    // Get the maximum value
    double colorbarMaxValue = m_Parent->GetWorkspace()->GetColorbarMaxValue();

    wxASSERT(m_LeadTimeIndex>=0);

    // Display according to the chosen display type
    if (m_LeadTimeIndex==m_ForecastManager->GetLeadTimeLengthMax())
    {
        // Create the layers
        vrLayerVectorFcstRing * layerSpecific = new vrLayerVectorFcstRing();
        vrLayerVectorFcstRing * layerOther = new vrLayerVectorFcstRing();
        if(layerSpecific->Create(memoryLayerNameSpecific, wkbPoint)==false)
        {
            wxFAIL;
            m_ViewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }
        if(layerOther->Create(memoryLayerNameOther, wkbPoint)==false)
        {
            wxFAIL;
            m_ViewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }

        // Set the maximum value
        if (m_ForecastDisplaySelection==0) // Only if the value option is selected, and not the ratio
        {
            layerSpecific->SetMaxValue(colorbarMaxValue);
            layerOther->SetMaxValue(colorbarMaxValue);
            m_LayerMaxValue = colorbarMaxValue;
        }
        else
        {
            layerSpecific->SetMaxValue(1.0);
            layerOther->SetMaxValue(1.0);
            m_LayerMaxValue = 1.0;
        }

        // Length of the lead time
        int leadTimeSize = forecasts[0]->GetTargetDatesLength();

        // Adding fields
        OGRFieldDefn fieldStationRow ("stationrow", OFTReal);
        layerSpecific->AddField(fieldStationRow);
        layerOther->AddField(fieldStationRow);
        OGRFieldDefn fieldStationId ("stationid", OFTReal);
        layerSpecific->AddField(fieldStationId);
        layerOther->AddField(fieldStationId);
        OGRFieldDefn fieldLeadTimeSize ("leadtimesize", OFTReal);
        layerSpecific->AddField(fieldLeadTimeSize);
        layerOther->AddField(fieldLeadTimeSize);

        // Adding a field for every lead time
        for (int i=0; i<leadTimeSize; i++)
        {
            OGRFieldDefn fieldLeadTime (wxString::Format("leadtime%d", i), OFTReal);
            layerSpecific->AddField(fieldLeadTime);
            layerOther->AddField(fieldLeadTime);
        }

        // Adding features to the layer
        for (int i_stat=0; i_stat<forecasts[0]->GetStationsNb(); i_stat++)
        {
            int currentId = forecasts[0]->GetStationId(i_stat);

            // Select the accurate forecast
            bool accurateForecast = false;
            asResultsAnalogsForecast* forecast = NULL;
            if (m_ForecastSelection>=0)
            {
                forecast = forecasts[0];
                accurateForecast = forecast->IsSpecificForStation(currentId);
            }
            else
            {
                for (int i=0; i<forecasts.size(); i++)
                {   
                    accurateForecast = forecasts[i]->IsSpecificForStation(currentId);
                    if (accurateForecast)
                    {
                        forecast = forecasts[i];
                        break;
                    }
                }
            }

            if(!forecast) {
                asLogWarning(wxString::Format(_("%s is not associated to any forecast"), forecast->GetStationName(i_stat).c_str()));
                continue;
            }

            OGRPoint station;
            station.setX( forecast->GetStationLocCoordX(i_stat) );
            station.setY( forecast->GetStationLocCoordY(i_stat) );

            // Field container
            wxArrayDouble data;
            data.Add((double)i_stat);
            data.Add((double)currentId);
            data.Add((double)leadTimeSize);

            // For normalization by the return period
            double factor = 1;
            if (forecast->HasReferenceValues() && returnPeriod!=0)
            {
                float precip = forecast->GetReferenceValue(i_stat, indexReferenceAxis);
                wxASSERT(precip>0);
                wxASSERT(precip<500);
                factor = 1.0/precip;
                wxASSERT(factor>0);
            }

            // Loop over the lead times
            for (int i_leadtime=0; i_leadtime<leadTimeSize; i_leadtime++)
            {
                Array1DFloat values = forecast->GetAnalogsValuesGross(i_leadtime, i_stat);

                if(asTools::HasNaN(&values[0], &values[values.size()-1]))
                {
                    data.Add(NaNDouble);
                }
                else
                {
                    if (percentile>=0)
                    {
                        double forecastVal = asTools::Percentile(&values[0], &values[values.size()-1], percentile);
                        wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                        forecastVal *= factor;
                        data.Add(forecastVal);
                    }
                    else
                    {
                        // Interpretatio
                        double forecastVal = 0;
                        double forecastVal30 = asTools::Percentile(&values[0], &values[values.size()-1], 0.3f);
                        double forecastVal60 = asTools::Percentile(&values[0], &values[values.size()-1], 0.6f);
                        double forecastVal90 = asTools::Percentile(&values[0], &values[values.size()-1], 0.9f);

                        if(forecastVal60==0)
                        {
                            forecastVal = 0;
                        }
                        else if(forecastVal30>0)
                        {
                            forecastVal = forecastVal90;
                        }
                        else
                        {
                            forecastVal = forecastVal60;
                        }

                        wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                        forecastVal *= factor;
                        data.Add(forecastVal);
                    }
                }
            }

            if (accurateForecast)
            {
                layerSpecific->AddFeature(&station, &data);
            }
            else
            {
                layerOther->AddFeature(&station, &data);
            }
        }

        wxASSERT(layerSpecific);
        wxASSERT(layerOther);

        if(layerOther->GetFeatureCount()>0) {
            m_LayerManager->Add(layerOther);
            vrRenderVector * renderOther = new vrRenderVector();
            renderOther->SetSize(1);
            renderOther->SetColorPen(wxColor(150, 150, 150));
            m_ViewerLayerManager->Add(-1, layerOther, renderOther);
        }
        else {
            wxDELETE(layerOther);
        }

        m_LayerManager->Add(layerSpecific);
        vrRenderVector * renderSpecific = new vrRenderVector();
        renderSpecific->SetSize(1);
        renderSpecific->SetColorPen(*wxBLACK);
        m_ViewerLayerManager->Add(-1, layerSpecific, renderSpecific);
        m_ViewerLayerManager->FreezeEnd();

    }
    else
    {
        // Create the layer
        vrLayerVectorFcstDots * layerSpecific = new vrLayerVectorFcstDots();
        vrLayerVectorFcstDots * layerOther = new vrLayerVectorFcstDots();
        if(layerSpecific->Create(memoryLayerNameSpecific, wkbPoint)==false)
        {
            wxFAIL;
            m_ViewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }
        if(layerOther->Create(memoryLayerNameOther, wkbPoint)==false)
        {
            wxFAIL;
            m_ViewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }

        // Set the maximum value
        if (m_ForecastDisplaySelection==0) // Only if the value option is selected, and not the ratio
        {
            layerSpecific->SetMaxValue(colorbarMaxValue);
            layerOther->SetMaxValue(colorbarMaxValue);
            m_LayerMaxValue = colorbarMaxValue;
        }
        else
        {
            layerSpecific->SetMaxValue(1.0);
            layerOther->SetMaxValue(1.0);
            m_LayerMaxValue = 1.0;
        }

        // Adding fields
        OGRFieldDefn fieldStationRow ("stationrow", OFTReal);
        layerSpecific->AddField(fieldStationRow);
        layerOther->AddField(fieldStationRow);
        OGRFieldDefn fieldStationId ("stationid", OFTReal);
        layerSpecific->AddField(fieldStationId);
        layerOther->AddField(fieldStationId);
        OGRFieldDefn fieldValueReal ("valueReal", OFTReal);
        layerSpecific->AddField(fieldValueReal);
        layerOther->AddField(fieldValueReal);
        OGRFieldDefn fieldValueNorm ("valueNorm", OFTReal);
        layerSpecific->AddField(fieldValueNorm);
        layerOther->AddField(fieldValueNorm);

        // Adding features to the layer
        for (int i_stat=0; i_stat<forecasts[0]->GetStationsNb(); i_stat++)
        {
            int currentId = forecasts[0]->GetStationId(i_stat);

            // Select the accurate forecast
            bool accurateForecast = false;
            asResultsAnalogsForecast* forecast = NULL;
            if (m_ForecastSelection>=0)
            {
                forecast = forecasts[0];
                accurateForecast = forecast->IsSpecificForStation(currentId);
            }
            else
            {
                for (int i=0; i<forecasts.size(); i++)
                {   
                    accurateForecast = forecasts[i]->IsSpecificForStation(currentId);
                    if (accurateForecast)
                    {
                        forecast = forecasts[i];
                        break;
                    }
                }
            }

            if(!forecast) {
                asLogWarning(wxString::Format(_("%s is not associated to any forecast"), forecasts[0]->GetStationName(i_stat).c_str()));
                continue;
            }

            OGRPoint station;
            station.setX( forecast->GetStationLocCoordX(i_stat) );
            station.setY( forecast->GetStationLocCoordY(i_stat) );

            // Field container
            wxArrayDouble data;
            data.Add((double)i_stat);
            data.Add((double)currentId);

            // For normalization by the return period
            double factor = 1;
            if (forecast->HasReferenceValues() && returnPeriod!=0)
            {
                float precip = forecast->GetReferenceValue(i_stat, indexReferenceAxis);
                wxASSERT(precip>0);
                wxASSERT(precip<500);
                factor = 1.0/precip;
                wxASSERT(factor>0);
            }

            // Check available lead times
            if(forecast->GetTargetDatesLength()<=m_LeadTimeIndex)
            {
                asLogError(_("Lead time not available for this forecast."));
                m_LeadTimeIndex = forecast->GetTargetDatesLength()-1;
            }

            Array1DFloat values = forecast->GetAnalogsValuesGross(m_LeadTimeIndex, i_stat);

            if(asTools::HasNaN(&values[0], &values[values.size()-1]))
            {
                data.Add(NaNDouble); // 1st real value
                data.Add(NaNDouble); // 2nd normalized
            }
            else
            {
                if (percentile>=0)
                {
                    double forecastVal = asTools::Percentile(&values[0], &values[values.size()-1], percentile);
                    wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                    data.Add(forecastVal); // 1st real value
                    forecastVal *= factor;
                    data.Add(forecastVal); // 2nd normalized
                }
                else
                {
                    // Interpretatio
                    double forecastVal = 0;
                    double forecastVal30 = asTools::Percentile(&values[0], &values[values.size()-1], 0.3f);
                    double forecastVal60 = asTools::Percentile(&values[0], &values[values.size()-1], 0.6f);
                    double forecastVal90 = asTools::Percentile(&values[0], &values[values.size()-1], 0.9f);

                    if(forecastVal60==0)
                    {
                        forecastVal = 0;
                    }
                    else if(forecastVal30>0)
                    {
                        forecastVal = forecastVal90;
                    }
                    else
                    {
                        forecastVal = forecastVal60;
                    }

                    wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                    data.Add(forecastVal); // 1st real value
                    forecastVal *= factor;
                    data.Add(forecastVal); // 2nd normalized
                }
            }

            if (accurateForecast)
            {
                layerSpecific->AddFeature(&station, &data);
            }
            else
            {
                layerOther->AddFeature(&station, &data);
            }
        }
        
        wxASSERT(layerSpecific);
        wxASSERT(layerOther);

        if(layerOther->GetFeatureCount()>0) {
            m_LayerManager->Add(layerOther);
            vrRenderVector * renderOther = new vrRenderVector();
            renderOther->SetSize(1);
            renderOther->SetColorPen(wxColor(150, 150, 150));
            m_ViewerLayerManager->Add(-1, layerOther, renderOther);
        }
        else {
            wxDELETE(layerOther);
        }

        m_LayerManager->Add(layerSpecific);
        vrRenderVector * renderSpecific = new vrRenderVector();
        renderSpecific->SetSize(1);
        renderSpecific->SetColorPen(*wxBLACK);
        m_ViewerLayerManager->Add(-1, layerSpecific, renderSpecific);
        m_ViewerLayerManager->FreezeEnd();
    }
}

void asForecastViewer::ChangeLeadTime( int val )
{
    if (m_LeadTimeIndex==val) // Already selected
        return;

    m_LeadTimeIndex = val;
    wxASSERT(m_LeadTimeIndex>=0);

    m_LeadTimeDate = GetSelectedTargetDate();

    Redraw();
}
