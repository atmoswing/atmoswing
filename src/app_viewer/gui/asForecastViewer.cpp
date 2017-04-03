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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
 */

#include "asForecastViewer.h"

#include "asForecastManager.h"
#include "asFrameForecast.h"
#include "vrLayerVectorFcstRing.h"
#include "vrLayerVectorFcstDots.h"


wxDEFINE_EVENT(asEVT_ACTION_FORECAST_SELECT_FIRST, wxCommandEvent);

asForecastViewer::asForecastViewer(asFrameForecast *parent, asForecastManager *forecastManager,
                                   vrLayerManager *layerManager, vrViewerLayerManager *viewerLayerManager)
{
    m_parent = parent;
    m_forecastManager = forecastManager;
    m_layerManager = layerManager;
    m_viewerLayerManager = viewerLayerManager;
    m_leadTimeIndex = 0;
    m_leadTimeDate = 0;
    m_layerMaxValue = 1;

    m_displayForecast.Add(_("Value"));
    m_displayForecast.Add(_("Ratio P/P2"));
    m_displayForecast.Add(_("Ratio P/P5"));
    m_displayForecast.Add(_("Ratio P/P10"));
    m_displayForecast.Add(_("Ratio P/P20"));
    m_displayForecast.Add(_("Ratio P/P50"));
    m_displayForecast.Add(_("Ratio P/P100"));
    m_displayForecast.Add(_("Ratio P/P200"));
    m_displayForecast.Add(_("Ratio P/P300"));
    m_displayForecast.Add(_("Ratio P/P500"));

    m_returnPeriods.push_back(0);
    m_returnPeriods.push_back(2);
    m_returnPeriods.push_back(5);
    m_returnPeriods.push_back(10);
    m_returnPeriods.push_back(20);
    m_returnPeriods.push_back(50);
    m_returnPeriods.push_back(100);
    m_returnPeriods.push_back(200);
    m_returnPeriods.push_back(300);
    m_returnPeriods.push_back(500);

    //m_displayQuantiles.Add(_("interpretation"));
    m_displayQuantiles.Add(_("q90"));
    m_displayQuantiles.Add(_("q60"));
    m_displayQuantiles.Add(_("q20"));

    //m_quantiles.push_back(-1);
    m_quantiles.push_back(0.9f);
    m_quantiles.push_back(0.6f);
    m_quantiles.push_back(0.2f);

    m_methodSelection = -1;
    m_forecastSelection = -1;

    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/ForecastViewer/DisplaySelection", &m_forecastDisplaySelection, 3);
    pConfig->Read("/ForecastViewer/QuantileSelection", &m_quantileSelection, 0);
    if ((unsigned) m_forecastDisplaySelection >= m_returnPeriods.size()) {
        m_forecastDisplaySelection = 1;
    }
    if ((unsigned) m_quantileSelection >= m_quantiles.size()) {
        m_quantileSelection = 0;
    }

    m_opened = false;
}

asForecastViewer::~asForecastViewer()
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/ForecastViewer/DisplaySelection", m_forecastDisplaySelection);
    pConfig->Write("/ForecastViewer/QuantileSelection", m_quantileSelection);
}

void asForecastViewer::FixForecastSelection()
{
    if (m_methodSelection < 0) {
        wxCommandEvent eventSlct(asEVT_ACTION_FORECAST_SELECT_FIRST);
        m_parent->ProcessWindowEvent(eventSlct);
    }
}

void asForecastViewer::ResetForecastSelection()
{
    m_methodSelection = -1;
    m_forecastSelection = -1;
}

void asForecastViewer::SetForecast(int methodRow, int forecastRow)
{
    m_methodSelection = methodRow;
    m_forecastSelection = forecastRow;

    Redraw();
}

wxString asForecastViewer::GetStationName(int i_stat) const
{
    return m_forecastManager->GetStationName(m_methodSelection, m_forecastSelection, i_stat);
}

float asForecastViewer::GetSelectedTargetDate()
{
    Array1DFloat targetDates;

    if (m_methodSelection < 0) {
        m_methodSelection = 0;
    }

    if (m_forecastSelection > 0) {
        targetDates = m_forecastManager->GetTargetDates(m_methodSelection, m_forecastSelection);
    } else {
        targetDates = m_forecastManager->GetTargetDates(m_methodSelection);
    }

    wxASSERT(m_leadTimeIndex >= 0);
    if (m_leadTimeIndex >= targetDates.size()) {
        return 0;
    }
    return targetDates[m_leadTimeIndex];
}

void asForecastViewer::SetLeadTimeDate(float date)
{
    if (date > 0 && (m_methodSelection > 0)) {
        Array1DFloat targetDates;

        if (m_forecastSelection > 0) {
            targetDates = m_forecastManager->GetTargetDates(m_methodSelection, m_forecastSelection);
        } else {
            targetDates = m_forecastManager->GetTargetDates(m_methodSelection);
        }

        int index = asTools::SortedArraySearchClosest(&targetDates[0], &targetDates[targetDates.size() - 1], date);
        if (index >= 0) {
            m_leadTimeIndex = index;
        }
    }
}

void asForecastViewer::SetForecastDisplay(int i)
{
    m_forecastDisplaySelection = i;

    wxString display = m_displayForecast.Item((size_t) m_forecastDisplaySelection);
    wxLogVerbose(_("Selected display : %s."), display);

    Redraw();
}

void asForecastViewer::SetQuantile(int i)
{
    m_quantileSelection = i;

    wxString quantile = m_displayQuantiles.Item((size_t) m_quantileSelection);
    wxLogVerbose(_("Selected quantile : %s."), quantile);

    Redraw();
}

void asForecastViewer::LoadPastForecast()
{
    wxBusyCursor wait;

    // Check that elements are selected
    if ((m_methodSelection == -1) || (m_forecastDisplaySelection == -1) || (m_quantileSelection == -1))
        return;
    if (m_methodSelection >= m_forecastManager->GetMethodsNb())
        return;

    if (m_forecastSelection > 0) {
        m_forecastManager->LoadPastForecast(m_methodSelection, m_forecastSelection);
    } else {
        m_forecastManager->LoadPastForecast(m_methodSelection);
    }
}

void asForecastViewer::Redraw()
{
    // Check that elements are selected
    if ((m_methodSelection == -1) || (m_forecastDisplaySelection == -1) || (m_quantileSelection == -1))
        return;
    if (m_methodSelection >= m_forecastManager->GetMethodsNb())
        return;
    if ((unsigned) m_forecastDisplaySelection >= m_displayForecast.size())
        return;
    if (m_quantiles.size() != m_displayQuantiles.size())
        return;
    if (m_returnPeriods.size() != m_displayForecast.size())
        return;

    // Get data
    std::vector<asResultsAnalogsForecast *> forecasts;

    if (m_forecastSelection < 0) {
        for (int i = 0; i < m_forecastManager->GetForecastsNb(m_methodSelection); i++) {
            forecasts.push_back(m_forecastManager->GetForecast(m_methodSelection, i));
        }
    } else {
        forecasts.push_back(m_forecastManager->GetForecast(m_methodSelection, m_forecastSelection));
    }

    // Create a memory layer
    wxFileName memoryLayerNameSpecific("", "Forecast - specific", "memory");
    wxFileName memoryLayerNameOther("", "Forecast - other", "memory");

    // Check if memory layer already added
    m_viewerLayerManager->FreezeBegin();
    for (unsigned int i = 0; i < m_viewerLayerManager->GetCount(); i++) {
        if (m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerNameSpecific) {
            vrRenderer *renderer = m_viewerLayerManager->GetRenderer(i);
            vrLayer *layer = renderer->GetLayer();
            wxASSERT(renderer);
            m_viewerLayerManager->Remove(renderer);
            // Close layer
            m_layerManager->Close(layer);
        }
    }
    for (unsigned int i = 0; i < m_viewerLayerManager->GetCount(); i++) {
        if (m_viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerNameOther) {
            vrRenderer *renderer = m_viewerLayerManager->GetRenderer(i);
            vrLayer *layer = renderer->GetLayer();
            wxASSERT(renderer);
            m_viewerLayerManager->Remove(renderer);
            // Close layer
            m_layerManager->Close(layer);
        }
    }

    // Get display option
    float quantile = m_quantiles[m_quantileSelection];
    float returnPeriod = m_returnPeriods[m_forecastDisplaySelection];

    // Get reference axis index
    int indexReferenceAxis = asNOT_FOUND;
    if (forecasts[0]->HasReferenceValues() && returnPeriod != 0) {
        Array1DFloat forecastReferenceAxis = forecasts[0]->GetReferenceAxis();

        indexReferenceAxis = asTools::SortedArraySearch(&forecastReferenceAxis[0],
                                                        &forecastReferenceAxis[forecastReferenceAxis.size() - 1],
                                                        returnPeriod);
        if ((indexReferenceAxis == asNOT_FOUND) || (indexReferenceAxis == asOUT_OF_RANGE)) {
            wxLogError(_("The desired reference value is not available in the forecast file."));
            m_viewerLayerManager->FreezeEnd();
            return;
        }
    }

    // Get the maximum value
    double colorbarMaxValue = m_parent->GetWorkspace()->GetColorbarMaxValue();

    wxASSERT(m_leadTimeIndex >= 0);

    // Display according to the chosen display type
    if (m_leadTimeIndex == m_forecastManager->GetLeadTimeLengthMax()) {
        // Create the layers
        vrLayerVectorFcstRing *layerSpecific = new vrLayerVectorFcstRing();
        vrLayerVectorFcstRing *layerOther = new vrLayerVectorFcstRing();
        if (!layerSpecific->Create(memoryLayerNameSpecific, wkbPoint)) {
            wxFAIL;
            m_viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }
        if (!layerOther->Create(memoryLayerNameOther, wkbPoint)) {
            wxFAIL;
            m_viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }

        // Set the maximum value
        if (m_forecastDisplaySelection == 0) // Only if the value option is selected, and not the ratio
        {
            layerSpecific->SetMaxValue(colorbarMaxValue);
            layerOther->SetMaxValue(colorbarMaxValue);
            m_layerMaxValue = colorbarMaxValue;
        } else {
            layerSpecific->SetMaxValue(1.0);
            layerOther->SetMaxValue(1.0);
            m_layerMaxValue = 1.0;
        }

        // Length of the lead time
        int leadTimeSize = forecasts[0]->GetTargetDatesLength();

        // Adding fields
        OGRFieldDefn fieldStationRow("stationrow", OFTReal);
        layerSpecific->AddField(fieldStationRow);
        layerOther->AddField(fieldStationRow);
        OGRFieldDefn fieldStationId("stationid", OFTReal);
        layerSpecific->AddField(fieldStationId);
        layerOther->AddField(fieldStationId);
        OGRFieldDefn fieldLeadTimeSize("leadtimesize", OFTReal);
        layerSpecific->AddField(fieldLeadTimeSize);
        layerOther->AddField(fieldLeadTimeSize);

        // Adding a field for every lead time
        for (int i = 0; i < leadTimeSize; i++) {
            OGRFieldDefn fieldLeadTime(wxString::Format("leadtime%d", i), OFTReal);
            layerSpecific->AddField(fieldLeadTime);
            layerOther->AddField(fieldLeadTime);
        }

        // Adding features to the layer
        for (int i_stat = 0; i_stat < forecasts[0]->GetStationsNb(); i_stat++) {
            int currentId = forecasts[0]->GetStationId(i_stat);

            // Select the accurate forecast
            bool accurateForecast = false;
            asResultsAnalogsForecast *forecast = NULL;
            if (m_forecastSelection >= 0) {
                forecast = forecasts[0];
                accurateForecast = forecast->IsSpecificForStationId(currentId);
            } else {
                for (int i = 0; i < (int) forecasts.size(); i++) {
                    accurateForecast = forecasts[i]->IsSpecificForStationId(currentId);
                    if (accurateForecast) {
                        forecast = forecasts[i];
                        break;
                    }
                }
            }

            if (m_forecastManager->GetForecastsNb(m_methodSelection) == 1) {
                forecast = forecasts[0];
                accurateForecast = true;
            }

            if (!forecast) {
                wxLogWarning(_("%s is not associated to any forecast"), forecasts[0]->GetStationName(i_stat));
                continue;
            }

            OGRPoint station;
            station.setX(forecast->GetStationXCoord(i_stat));
            station.setY(forecast->GetStationYCoord(i_stat));

            // Field container
            wxArrayDouble data;
            data.Add((double) i_stat);
            data.Add((double) currentId);
            data.Add((double) leadTimeSize);

            // For normalization by the return period
            double factor = 1;
            if (forecast->HasReferenceValues() && returnPeriod != 0) {
                float precip = forecast->GetReferenceValue(i_stat, indexReferenceAxis);
                wxASSERT(precip > 0);
                wxASSERT(precip < 500);
                factor = 1.0 / precip;
                wxASSERT(factor > 0);
            }

            // Loop over the lead times
            for (unsigned int i_leadtime = 0; i_leadtime < leadTimeSize; i_leadtime++) {
                Array1DFloat values = forecast->GetAnalogsValuesGross(i_leadtime, i_stat);

                if (asTools::HasNaN(&values[0], &values[values.size() - 1])) {
                    data.Add(NaNDouble);
                } else {
                    if (quantile >= 0) {
                        double forecastVal = asTools::GetValueForQuantile(values, quantile);
                        wxASSERT_MSG(forecastVal >= 0, wxString::Format("Forecast value = %g", forecastVal));
                        forecastVal *= factor;
                        data.Add(forecastVal);
                    } else {
                        // Interpretatio
                        double forecastVal = 0;
                        double forecastVal30 = asTools::GetValueForQuantile(values, 0.2f);
                        double forecastVal60 = asTools::GetValueForQuantile(values, 0.6f);
                        double forecastVal90 = asTools::GetValueForQuantile(values, 0.9f);

                        if (forecastVal60 == 0) {
                            forecastVal = 0;
                        } else if (forecastVal30 > 0) {
                            forecastVal = forecastVal90;
                        } else {
                            forecastVal = forecastVal60;
                        }

                        wxASSERT_MSG(forecastVal >= 0, wxString::Format("Forecast value = %g", forecastVal));
                        forecastVal *= factor;
                        data.Add(forecastVal);
                    }
                }
            }

            if (accurateForecast) {
                layerSpecific->AddFeature(&station, &data);
            } else {
                layerOther->AddFeature(&station, &data);
            }
        }

        wxASSERT(layerSpecific);
        wxASSERT(layerOther);

        if (layerOther->GetFeatureCount() > 0) {
            m_layerManager->Add(layerOther);
            vrRenderVector *renderOther = new vrRenderVector();
            renderOther->SetSize(1);
            renderOther->SetColorPen(wxColor(150, 150, 150));
            m_viewerLayerManager->Add(-1, layerOther, renderOther);
        } else {
            wxDELETE(layerOther);
        }

        m_layerManager->Add(layerSpecific);
        vrRenderVector *renderSpecific = new vrRenderVector();
        renderSpecific->SetSize(1);
        renderSpecific->SetColorPen(*wxBLACK);
        m_viewerLayerManager->Add(-1, layerSpecific, renderSpecific);
        m_viewerLayerManager->FreezeEnd();

    } else {
        // Create the layer
        vrLayerVectorFcstDots *layerSpecific = new vrLayerVectorFcstDots();
        vrLayerVectorFcstDots *layerOther = new vrLayerVectorFcstDots();
        if (!layerSpecific->Create(memoryLayerNameSpecific, wkbPoint)) {
            wxFAIL;
            m_viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }
        if (!layerOther->Create(memoryLayerNameOther, wkbPoint)) {
            wxFAIL;
            m_viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }

        // Set the maximum value
        if (m_forecastDisplaySelection == 0) // Only if the value option is selected, and not the ratio
        {
            layerSpecific->SetMaxValue(colorbarMaxValue);
            layerOther->SetMaxValue(colorbarMaxValue);
            m_layerMaxValue = colorbarMaxValue;
        } else {
            layerSpecific->SetMaxValue(1.0);
            layerOther->SetMaxValue(1.0);
            m_layerMaxValue = 1.0;
        }

        // Adding fields
        OGRFieldDefn fieldStationRow("stationrow", OFTReal);
        layerSpecific->AddField(fieldStationRow);
        layerOther->AddField(fieldStationRow);
        OGRFieldDefn fieldStationId("stationid", OFTReal);
        layerSpecific->AddField(fieldStationId);
        layerOther->AddField(fieldStationId);
        OGRFieldDefn fieldValueReal("valueReal", OFTReal);
        layerSpecific->AddField(fieldValueReal);
        layerOther->AddField(fieldValueReal);
        OGRFieldDefn fieldValueNorm("valueNorm", OFTReal);
        layerSpecific->AddField(fieldValueNorm);
        layerOther->AddField(fieldValueNorm);

        // Adding features to the layer
        for (int i_stat = 0; i_stat < forecasts[0]->GetStationsNb(); i_stat++) {
            int currentId = forecasts[0]->GetStationId(i_stat);

            // Select the accurate forecast
            bool accurateForecast = false;
            asResultsAnalogsForecast *forecast = NULL;
            if (m_forecastSelection >= 0) {
                forecast = forecasts[0];
                accurateForecast = forecast->IsSpecificForStationId(currentId);
            } else {
                for (int i = 0; i < (int) forecasts.size(); i++) {
                    accurateForecast = forecasts[i]->IsSpecificForStationId(currentId);
                    if (accurateForecast) {
                        forecast = forecasts[i];
                        break;
                    }
                }
            }

            if (m_forecastManager->GetForecastsNb(m_methodSelection) == 1) {
                forecast = forecasts[0];
                accurateForecast = true;
            }

            if (!forecast) {
                wxLogWarning(_("%s is not associated to any forecast"), forecasts[0]->GetStationName(i_stat));
                continue;
            }

            OGRPoint station;
            station.setX(forecast->GetStationXCoord(i_stat));
            station.setY(forecast->GetStationYCoord(i_stat));

            // Field container
            wxArrayDouble data;
            data.Add((double) i_stat);
            data.Add((double) currentId);

            // For normalization by the return period
            double factor = 1;
            if (forecast->HasReferenceValues() && returnPeriod != 0) {
                float precip = forecast->GetReferenceValue(i_stat, indexReferenceAxis);
                wxASSERT(precip > 0);
                wxASSERT(precip < 500);
                factor = 1.0 / precip;
                wxASSERT(factor > 0);
            }

            // Check available lead times
            if (forecast->GetTargetDatesLength() <= m_leadTimeIndex) {
                wxLogError(_("Lead time not available for this forecast."));
                m_leadTimeIndex = forecast->GetTargetDatesLength() - 1;
            }

            Array1DFloat values = forecast->GetAnalogsValuesGross(m_leadTimeIndex, i_stat);

            if (asTools::HasNaN(&values[0], &values[values.size() - 1])) {
                data.Add(NaNDouble); // 1st real value
                data.Add(NaNDouble); // 2nd normalized
            } else {
                if (quantile >= 0) {
                    double forecastVal = asTools::GetValueForQuantile(values, quantile);
                    wxASSERT_MSG(forecastVal >= 0, wxString::Format("Forecast value = %g", forecastVal));
                    data.Add(forecastVal); // 1st real value
                    forecastVal *= factor;
                    data.Add(forecastVal); // 2nd normalized
                } else {
                    // Interpretatio
                    double forecastVal = 0;
                    double forecastVal30 = asTools::GetValueForQuantile(values, 0.3f);
                    double forecastVal60 = asTools::GetValueForQuantile(values, 0.6f);
                    double forecastVal90 = asTools::GetValueForQuantile(values, 0.9f);

                    if (forecastVal60 == 0) {
                        forecastVal = 0;
                    } else if (forecastVal30 > 0) {
                        forecastVal = forecastVal90;
                    } else {
                        forecastVal = forecastVal60;
                    }

                    wxASSERT_MSG(forecastVal >= 0, wxString::Format("Forecast value = %g", forecastVal));
                    data.Add(forecastVal); // 1st real value
                    forecastVal *= factor;
                    data.Add(forecastVal); // 2nd normalized
                }
            }

            if (accurateForecast) {
                layerSpecific->AddFeature(&station, &data);
            } else {
                layerOther->AddFeature(&station, &data);
            }
        }

        wxASSERT(layerSpecific);
        wxASSERT(layerOther);

        if (layerOther->GetFeatureCount() > 0) {
            m_layerManager->Add(layerOther);
            vrRenderVector *renderOther = new vrRenderVector();
            renderOther->SetSize(1);
            renderOther->SetColorPen(wxColor(150, 150, 150));
            m_viewerLayerManager->Add(-1, layerOther, renderOther);
        } else {
            wxDELETE(layerOther);
        }

        m_layerManager->Add(layerSpecific);
        vrRenderVector *renderSpecific = new vrRenderVector();
        renderSpecific->SetSize(1);
        renderSpecific->SetColorPen(*wxBLACK);
        m_viewerLayerManager->Add(-1, layerSpecific, renderSpecific);
        m_viewerLayerManager->FreezeEnd();
    }
}

void asForecastViewer::ChangeLeadTime(int val)
{
    if (m_leadTimeIndex == val) // Already selected
        return;

    m_leadTimeIndex = val;
    wxASSERT(m_leadTimeIndex >= 0);

    m_leadTimeDate = GetSelectedTargetDate();

    Redraw();
}
