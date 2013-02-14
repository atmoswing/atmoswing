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
 
#include "asForecastViewer.h"

#include "asForecastManager.h"
#include "asResultsAnalogsForecast.h"
#include "vrLayerVectorFcstRing.h"
#include "vrLayerVectorFcstDots.h"
#include "vrlayervector.h"
#include "vrrender.h"


asForecastViewer::asForecastViewer( wxWindow* parent, asForecastManager *forecastManager, vrLayerManager *layerManager, vrViewerLayerManager *viewerLayerManager)
{
    m_DisplayType = asForecastViewer::ForecastRings;
    m_Parent = parent;
    m_ForecastManager = forecastManager;
    m_LayerManager = layerManager;
	m_ViewerLayerManager = viewerLayerManager;
	m_LeadTimeIndex = 0;
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

    m_ModelSelection = -1;

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

void asForecastViewer::SetModel(int i)
{
    m_ModelSelection = i;

    wxString model = m_ForecastManager->GetModelName(m_ModelSelection);
    asLogMessage(wxString::Format(_("Selected model : %s."), model.c_str()));

    Redraw();
}

void asForecastViewer::SetLastModel()
{
    SetModel(m_ForecastManager->GetModelsNb()-1);
}

wxString asForecastViewer::GetStationName(int i_stat)
{
    return m_ForecastManager->GetStationName(m_ModelSelection, i_stat);
}

float asForecastViewer::GetSelectedTargetDate()
{
    Array1DFloat targetDates = m_ForecastManager->GetCurrentForecast(m_ModelSelection)->GetTargetDates();
    wxASSERT(m_LeadTimeIndex>=0);
    return targetDates[m_LeadTimeIndex];
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
    // Check that elements are selected
    if ( (m_ModelSelection==-1) || (m_ForecastDisplaySelection==-1) || (m_PercentileSelection==-1) ) return;
    if ( m_ModelSelection >= m_ForecastManager->GetModelsNb() ) return;

    m_ForecastManager->LoadPastForecast(m_ModelSelection);
}

void asForecastViewer::Redraw()
{
    // Check that elements are selected
    if ( (m_ModelSelection==-1) || (m_ForecastDisplaySelection==-1) || (m_PercentileSelection==-1) ) return;
    if ( m_ModelSelection >= m_ForecastManager->GetModelsNb() ) return;
    if ( (unsigned)m_ForecastDisplaySelection >= m_DisplayForecast.size() ) return;
    if ( m_Percentiles.size() != m_DisplayPercentiles.size() ) return;
    if ( m_ReturnPeriods.size() != m_DisplayForecast.size() ) return;

    // Create a memory layer
    wxFileName memoryLayerName ("", _("Forecast"), "memory");
    wxASSERT(memoryLayerName.GetExt() == "memory");

    // Check if memory layer already added
    m_ViewerLayerManager->FreezeBegin();
	for (int i = 0; i < m_ViewerLayerManager->GetCount(); i++)
    {
		if (m_ViewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerName)
        {
			vrRenderer *renderer = m_ViewerLayerManager->GetRenderer(i);
			vrLayer *layer = renderer->GetLayer();
			wxASSERT(renderer);
			m_ViewerLayerManager->Remove(renderer);
			// Close layer
			m_LayerManager->Close(layer);
		}
	}

	// Get data
	asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_ModelSelection);

    // Get display option
    float percentile = m_Percentiles[m_PercentileSelection];
    float returnPeriod = m_ReturnPeriods[m_ForecastDisplaySelection];

    // Get return period index
    int indexReturnPeriod;
    if (returnPeriod!=0)
    {
        Array1DFloat forecastReturnPeriod = forecast->GetReturnPeriods();

        indexReturnPeriod = asTools::SortedArraySearch(&forecastReturnPeriod[0], &forecastReturnPeriod[forecastReturnPeriod.size()-1], returnPeriod);
        if ( (indexReturnPeriod==asNOT_FOUND) || (indexReturnPeriod==asOUT_OF_RANGE) )
        {
            asLogError(_("The desired return period is not available in the forecast file."));
            m_ViewerLayerManager->FreezeEnd();
            return;
        }
    }

    // Get the maximum value
	wxConfigBase *pConfig = wxFileConfig::Get();
    double colorbarMaxValue = pConfig->ReadDouble("/GIS/ColorbarMaxValue", 50.0);

    // Display according to the chosen display type
    switch (m_DisplayType)
    {
    case ForecastRings:
        {
            // Create the layer
            vrLayerVectorFcstRing * layer = new vrLayerVectorFcstRing();
            if(layer->Create(memoryLayerName, wkbPoint)==false)
            {
                wxFAIL;
                m_ViewerLayerManager->FreezeEnd();
                return;
            }

            // Set the maximum value
            if (m_ForecastDisplaySelection==0) // Only if the value option is selected, and not the ratio
            {
                layer->SetMaxValue(colorbarMaxValue);
                m_LayerMaxValue = colorbarMaxValue;
            }
            else
            {
                layer->SetMaxValue(1.0);
                m_LayerMaxValue = 1.0;
            }

            // Length of the lead time
            int leadTimeSize = forecast->GetTargetDatesLength();

            // Adding size field
            OGRFieldDefn fieldLeadTimeSize ("leadtimesize", OFTReal);
            layer->AddField(fieldLeadTimeSize);

            // Adding a field for every lead time
            for (int i=0; i<leadTimeSize; i++)
            {
                OGRFieldDefn fieldLeadTime (wxString::Format("leadtime%d", i), OFTReal);
                layer->AddField(fieldLeadTime);
            }

            // Adding features to the layer
            for (int i_stat=0; i_stat<forecast->GetStationsNb(); i_stat++)
            {
                OGRPoint station;
                station.setX( forecast->GetStationLocCoordU(i_stat) );
                station.setY( forecast->GetStationLocCoordV(i_stat) );

                // Field container
                wxArrayDouble data;
                data.Add((double)leadTimeSize);

                // For normalization by the return period
                double factor = 1;
                if (returnPeriod!=0)
                {
                    float precip = forecast->GetDailyPrecipitationForReturnPeriod(i_stat, indexReturnPeriod);
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

                layer->AddFeature(&station, &data);
            }

            wxASSERT(layer);
            m_LayerManager->Add(layer);

            // Change default render
            vrRenderVector * render = new vrRenderVector();
            render->SetSize(1);
            render->SetColorPen(*wxBLACK);

            m_ViewerLayerManager->Add(-1, layer, render);
            m_ViewerLayerManager->FreezeEnd();

            break;
        }
    case ForecastDots:
        {
            wxASSERT(m_LeadTimeIndex>=0);

            // Create the layer
            vrLayerVectorFcstDots * layer = new vrLayerVectorFcstDots();
            if(layer->Create(memoryLayerName, wkbPoint)==false)
            {
                wxFAIL;
                m_ViewerLayerManager->FreezeEnd();
                return;
            }

            // Set the maximum value
            if (m_ForecastDisplaySelection==0) // Only if the value option is selected, and not the ratio
            {
                layer->SetMaxValue(colorbarMaxValue);
                m_LayerMaxValue = colorbarMaxValue;
            }
            else
            {
                layer->SetMaxValue(1.0);
                m_LayerMaxValue = 1.0;
            }

            // Adding size field
            OGRFieldDefn fieldValueReal ("valueReal", OFTReal);
            layer->AddField(fieldValueReal);
            OGRFieldDefn fieldValueNorm ("valueNorm", OFTReal);
            layer->AddField(fieldValueNorm);

            // Adding features to the layer
            for (int i_stat=0; i_stat<forecast->GetStationsNb(); i_stat++)
            {
                OGRPoint station;
                station.setX( forecast->GetStationLocCoordU(i_stat) );
                station.setY( forecast->GetStationLocCoordV(i_stat) );

                // Field container
                wxArrayDouble data;

                // For normalization by the return period
                double factor = 1;
                if (returnPeriod!=0)
                {
                    float precip = forecast->GetDailyPrecipitationForReturnPeriod(i_stat, indexReturnPeriod);
                    wxASSERT(precip>0);
                    wxASSERT(precip<500);
                    factor = 1.0/precip;
                    wxASSERT(factor>0);
                }

                // Check available lead times
                if(forecast->GetTargetDatesLength()<=m_LeadTimeIndex)
                {
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

                layer->AddFeature(&station, &data);
            }

            wxASSERT(layer);
            m_LayerManager->Add(layer);

            // Change default render
            vrRenderVector * render = new vrRenderVector();
            render->SetSize(1);
            render->SetColorPen(*wxBLACK);

            m_ViewerLayerManager->Add(-1, layer, render);
            m_ViewerLayerManager->FreezeEnd();

            break;
        }
    }
}

void asForecastViewer::ChangeLeadTime( int val )
{
    if (m_LeadTimeIndex==val) // Already selected
        return;

    m_LeadTimeIndex = val;
    wxASSERT(m_LeadTimeIndex>=0);

    Redraw();
}
