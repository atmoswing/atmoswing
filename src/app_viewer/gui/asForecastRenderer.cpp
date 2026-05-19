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

#include "asForecastRenderer.h"

#include "asForecastManager.h"
#include "asFrameViewer.h"
#include "vrLayerVectorFcstDots.h"
#include "vrLayerVectorFcstRing.h"

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_SELECT_FIRST, wxCommandEvent);

asForecastRenderer::asForecastRenderer(asFrameViewer* parent, asForecastManager* forecastManager,
                                       vrLayerManager* layerManager, vrViewerLayerManager* viewerLayerManager)
    : _parent(parent),
      _forecastManager(forecastManager),
      _layerManager(layerManager),
      _viewerLayerManager(viewerLayerManager),
      _leadTimeIndex(0),
      _leadTimeDate(0),
      _leadTimeStep(24),
      _layerMaxValue(1),
      _methodSelection(-1),
      _forecastSelection(-1),
      _opened(false) {
    _displayForecast.Add(_("Value"));
    _displayForecast.Add(_("Ratio P/P2"));
    _displayForecast.Add(_("Ratio P/P5"));
    _displayForecast.Add(_("Ratio P/P10"));
    _displayForecast.Add(_("Ratio P/P20"));
    _displayForecast.Add(_("Ratio P/P50"));
    _displayForecast.Add(_("Ratio P/P100"));
    _displayForecast.Add(_("Ratio P/P200"));
    _displayForecast.Add(_("Ratio P/P300"));
    _displayForecast.Add(_("Ratio P/P500"));

    _returnPeriods.push_back(0);
    _returnPeriods.push_back(2);
    _returnPeriods.push_back(5);
    _returnPeriods.push_back(10);
    _returnPeriods.push_back(20);
    _returnPeriods.push_back(50);
    _returnPeriods.push_back(100);
    _returnPeriods.push_back(200);
    _returnPeriods.push_back(300);
    _returnPeriods.push_back(500);

    // _displayQuantiles.Add(_("interpretation"));
    _displayQuantiles.Add(_("q90"));
    _displayQuantiles.Add(_("q60"));
    _displayQuantiles.Add(_("q20"));

    // _quantiles.push_back(-1);
    _quantiles.push_back(0.9f);
    _quantiles.push_back(0.6f);
    _quantiles.push_back(0.2f);

    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Read("/ForecastViewer/DisplaySelection", &_forecastDisplaySelection, 3);
    pConfig->Read("/ForecastViewer/QuantileSelection", &_quantileSelection, 0);
    if (_forecastDisplaySelection >= _returnPeriods.size()) {
        _forecastDisplaySelection = 1;
    }
    if (_quantileSelection >= _quantiles.size()) {
        _quantileSelection = 0;
    }
}

asForecastRenderer::~asForecastRenderer() {
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Write("/ForecastViewer/DisplaySelection", _forecastDisplaySelection);
    pConfig->Write("/ForecastViewer/QuantileSelection", _quantileSelection);
}

void asForecastRenderer::FixForecastSelection() {
    if (_methodSelection < 0) {
        wxCommandEvent eventSlct(asEVT_ACTION_FORECAST_SELECT_FIRST);
        _parent->ProcessWindowEvent(eventSlct);
    }
}

void asForecastRenderer::ResetForecastSelection() {
    _methodSelection = -1;
    _forecastSelection = -1;
}

void asForecastRenderer::SetForecast(int methodRow, int forecastRow) {
    _methodSelection = methodRow;
    _forecastSelection = forecastRow;

    AdaptLeadTimeIndex();

    Redraw();
}

void asForecastRenderer::AdaptLeadTimeIndex() {
    if (_methodSelection < 0) return;

    int forecast = wxMax(_forecastSelection, 0);
    float timeStep = _forecastManager->GetForecast(_methodSelection, forecast)->GetForecastTimeStepHours();

    if (_leadTimeIndex == -1) {
        _leadTimeStep = timeStep;
        return;
    }

    if (timeStep != _leadTimeStep) {
        _leadTimeIndex = int(_leadTimeIndex * float(_leadTimeStep) / float(timeStep));
        _leadTimeStep = timeStep;
    }
}

float asForecastRenderer::GetSelectedTargetDate() {
    if (_leadTimeIndex < 0) {
        return 0;
    }

    a1f targetDates = _forecastManager->GetTargetDates(wxMax(_methodSelection, 0), wxMax(_forecastSelection, 0));

    if (_leadTimeIndex >= targetDates.size()) {
        return 0;
    }
    return targetDates[_leadTimeIndex];
}

void asForecastRenderer::SetLeadTimeDate(float date) {
    if (date > 0 && (_methodSelection > 0)) {
        a1f targetDates = _forecastManager->GetTargetDates(_methodSelection, wxMax(_forecastSelection, 0));

        int index = asFindClosest(&targetDates[0], &targetDates[targetDates.size() - 1], date);
        if (index >= 0) {
            _leadTimeIndex = index;
        }
    }
}

void asForecastRenderer::SetForecastDisplay(int i) {
    _forecastDisplaySelection = i;

    wxString display = _displayForecast.Item((size_t)_forecastDisplaySelection);
    wxLogVerbose(_("Selected display : %s."), display);

    Redraw();
}

void asForecastRenderer::SetQuantile(int i) {
    _quantileSelection = i;

    wxString quantile = _displayQuantiles.Item((size_t)_quantileSelection);
    wxLogVerbose(_("Selected quantile : %s."), quantile);

    Redraw();
}

void asForecastRenderer::LoadPastForecast() {
    wxBusyCursor wait;

    // Check that elements are selected
    if ((_methodSelection == -1) || (_forecastDisplaySelection == -1) || (_quantileSelection == -1)) return;
    if (_methodSelection >= _forecastManager->GetMethodsNb()) return;

    if (_forecastSelection > 0) {
        _forecastManager->LoadPastForecast(_methodSelection, _forecastSelection);
    } else {
        _forecastManager->LoadPastForecast(_methodSelection);
    }
}

void asForecastRenderer::Redraw() {
    // Check that elements are selected
    if ((_methodSelection == -1) || (_forecastDisplaySelection == -1) || (_quantileSelection == -1)) return;
    if (_methodSelection >= _forecastManager->GetMethodsNb()) return;
    if (_forecastDisplaySelection >= _displayForecast.size()) return;
    if (_quantiles.size() != _displayQuantiles.size()) return;
    if (_returnPeriods.size() != _displayForecast.size()) return;

    // Get data
    vector<asResultsForecast*> forecasts;

    if (_forecastSelection < 0) {
        for (int i = 0; i < _forecastManager->GetForecastsNb(_methodSelection); i++) {
            forecasts.push_back(_forecastManager->GetForecast(_methodSelection, i));
        }
    } else {
        forecasts.push_back(_forecastManager->GetForecast(_methodSelection, _forecastSelection));
    }

    // Create a memory layer
    wxFileName memoryLayerNameSpecific("", _("Forecast - specific"), "memory");
    wxFileName memoryLayerNameOther("", _("Forecast - other"), "memory");

    // Check if memory layer already added
    _viewerLayerManager->FreezeBegin();
    for (int i = 0; i < _viewerLayerManager->GetCount(); i++) {
        if (_viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerNameSpecific) {
            vrRenderer* renderer = _viewerLayerManager->GetRenderer(i);
            vrLayer* layer = renderer->GetLayer();
            wxASSERT(renderer);
            _viewerLayerManager->Remove(renderer);
            // Close layer
            _layerManager->Close(layer);
        }
    }
    for (int i = 0; i < _viewerLayerManager->GetCount(); i++) {
        if (_viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryLayerNameOther) {
            vrRenderer* renderer = _viewerLayerManager->GetRenderer(i);
            vrLayer* layer = renderer->GetLayer();
            wxASSERT(renderer);
            _viewerLayerManager->Remove(renderer);
            // Close layer
            _layerManager->Close(layer);
        }
    }

    // Get display option
    float quantile = _quantiles[_quantileSelection];
    float returnPeriod = _returnPeriods[_forecastDisplaySelection];

    // Get reference axis index
    int indexReferenceAxis = asNOT_FOUND;
    if (forecasts[0]->HasReferenceValues() && returnPeriod != 0) {
        a1f forecastReferenceAxis = forecasts[0]->GetReferenceAxis();

        indexReferenceAxis = asFind(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size() - 1],
                                    returnPeriod);
        if ((indexReferenceAxis == asNOT_FOUND) || (indexReferenceAxis == asOUT_OF_RANGE)) {
            wxLogError(_("The desired reference value is not available in the forecast file."));
            _viewerLayerManager->FreezeEnd();
            return;
        }
    }

    // Get the maximum value
    double colorbarMaxValue = _parent->GetWorkspace()->GetColorbarMaxValue();

    // Display according to the chosen display type
    if (_leadTimeIndex == -1) {
        // Create the layers
        auto layerSpecific = new vrLayerVectorFcstRing();
        auto layerOther = new vrLayerVectorFcstRing();
        if (!layerSpecific->Create(memoryLayerNameSpecific, wkbPoint)) {
            wxFAIL;
            _viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }
        if (!layerOther->Create(memoryLayerNameOther, wkbPoint)) {
            wxFAIL;
            _viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }

        // Set the maximum value
        if (_forecastDisplaySelection == 0)  // Only if the value option is selected, and not the ratio
        {
            layerSpecific->SetMaxValue(colorbarMaxValue);
            layerOther->SetMaxValue(colorbarMaxValue);
            _layerMaxValue = colorbarMaxValue;
        } else {
            layerSpecific->SetMaxValue(1.0);
            layerOther->SetMaxValue(1.0);
            _layerMaxValue = 1.0;
        }

        // Length of the lead time
        int leadTimeSize = forecasts[0]->GetTargetDatesLength();

        // Check if a time shift
        bool timeShiftEndAccumulation = false;
        if (forecasts[0]->IsSubDaily()) {
            timeShiftEndAccumulation = true;
            leadTimeSize -= 1;
        }

        // Adding fields
        OGRFieldDefn fieldStationRow("stationRow", OFTReal);
        layerSpecific->AddField(fieldStationRow);
        layerOther->AddField(fieldStationRow);
        OGRFieldDefn fieldStationId("stationId", OFTReal);
        layerSpecific->AddField(fieldStationId);
        layerOther->AddField(fieldStationId);
        OGRFieldDefn fieldLeadTimeSize("leadTimeSize", OFTReal);
        layerSpecific->AddField(fieldLeadTimeSize);
        layerOther->AddField(fieldLeadTimeSize);

        // Adding a field for every lead time
        for (int i = 0; i < leadTimeSize; i++) {
            OGRFieldDefn fieldLeadTimeDate(asStrF("leadTimeDate%d", i), OFTReal);
            layerSpecific->AddField(fieldLeadTimeDate);
            layerOther->AddField(fieldLeadTimeDate);
            OGRFieldDefn fieldLeadTimeVal(asStrF("leadTimeVal%d", i), OFTReal);
            layerSpecific->AddField(fieldLeadTimeVal);
            layerOther->AddField(fieldLeadTimeVal);
        }

        // Adding features to the layer
        for (int iStat = 0; iStat < forecasts[0]->GetStationsNb(); iStat++) {
            int currentId = forecasts[0]->GetStationId(iStat);

            // Select the accurate forecast
            bool accurateForecast = false;
            asResultsForecast* forecast = nullptr;
            if (_forecastSelection >= 0) {
                forecast = forecasts[0];
                accurateForecast = forecast->IsSpecificForStationId(currentId);
            } else {
                for (auto& fcst : forecasts) {
                    accurateForecast = fcst->IsSpecificForStationId(currentId);
                    if (accurateForecast) {
                        forecast = fcst;
                        break;
                    }
                }
            }

            if (_forecastManager->GetForecastsNb(_methodSelection) == 1) {
                forecast = forecasts[0];
                accurateForecast = true;
            }

            if (!forecast) {
                wxLogWarning(_("%s is not associated to any forecast"), forecasts[0]->GetStationName(iStat));
                continue;
            }

            OGRPoint station;
            station.setX(forecast->GetStationXCoord(iStat));
            station.setY(forecast->GetStationYCoord(iStat));

            // Field container
            wxArrayDouble data;
            data.Add((double)iStat);
            data.Add((double)currentId);
            data.Add((double)leadTimeSize);

            // For normalization by the return period
            double factor = 1;
            if (forecast->HasReferenceValues() && returnPeriod != 0) {
                float precip = forecast->GetReferenceValue(iStat, indexReferenceAxis);
                wxASSERT(precip > 0);
                wxASSERT(precip < 500);
                factor = 1.0 / precip;
                wxASSERT(factor > 0);
            }

            // Loop over the lead times
            a1f dates = forecast->GetTargetDates();
            for (int iLead = 0; iLead < leadTimeSize; iLead++) {
                int idx = iLead;
                if (timeShiftEndAccumulation) {
                    idx += 1;
                }
                data.Add(dates[iLead]);

                a1f values = forecast->GetAnalogsValuesRaw(idx, iStat);

                if (asHasNaN(&values[0], &values[values.size() - 1])) {
                    data.Add(NAN);
                } else {
                    if (quantile >= 0) {
                        double forecastVal = asGetValueForQuantile(values, quantile);
                        wxASSERT_MSG(forecastVal >= 0, asStrF("Forecast value = %g", forecastVal));
                        forecastVal *= factor;
                        data.Add(forecastVal);
                    } else {
                        // Interpretation
                        double forecastVal = 0;
                        double forecastVal30 = asGetValueForQuantile(values, 0.2f);
                        double forecastVal60 = asGetValueForQuantile(values, 0.6f);
                        double forecastVal90 = asGetValueForQuantile(values, 0.9f);

                        if (forecastVal60 == 0) {
                            forecastVal = 0;
                        } else if (forecastVal30 > 0) {
                            forecastVal = forecastVal90;
                        } else {
                            forecastVal = forecastVal60;
                        }

                        wxASSERT_MSG(forecastVal >= 0, asStrF("Forecast value = %g", forecastVal));
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
            _layerManager->Add(layerOther);
            auto renderOther = new vrRenderVector();
            renderOther->SetSize(1);
            renderOther->SetColorPen(wxColor(150, 150, 150));
            _viewerLayerManager->Add(-1, layerOther, renderOther);
        } else {
            wxDELETE(layerOther);
        }

        _layerManager->Add(layerSpecific);
        auto renderSpecific = new vrRenderVector();
        renderSpecific->SetSize(1);
        renderSpecific->SetColorPen(*wxBLACK);
        _viewerLayerManager->Add(-1, layerSpecific, renderSpecific);
        _viewerLayerManager->FreezeEnd();

    } else {
        // Create the layer
        auto layerSpecific = new vrLayerVectorFcstDots();
        auto layerOther = new vrLayerVectorFcstDots();
        if (!layerSpecific->Create(memoryLayerNameSpecific, wkbPoint)) {
            wxFAIL;
            _viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }
        if (!layerOther->Create(memoryLayerNameOther, wkbPoint)) {
            wxFAIL;
            _viewerLayerManager->FreezeEnd();
            wxDELETE(layerSpecific);
            wxDELETE(layerOther);
            return;
        }

        // Set the maximum value
        if (_forecastDisplaySelection == 0)  // Only if the value option is selected, and not the ratio
        {
            layerSpecific->SetMaxValue(colorbarMaxValue);
            layerOther->SetMaxValue(colorbarMaxValue);
            _layerMaxValue = colorbarMaxValue;
        } else {
            layerSpecific->SetMaxValue(1.0);
            layerOther->SetMaxValue(1.0);
            _layerMaxValue = 1.0;
        }

        // Adding fields
        OGRFieldDefn fieldStationRow("stationRow", OFTReal);
        layerSpecific->AddField(fieldStationRow);
        layerOther->AddField(fieldStationRow);
        OGRFieldDefn fieldStationId("stationId", OFTReal);
        layerSpecific->AddField(fieldStationId);
        layerOther->AddField(fieldStationId);
        OGRFieldDefn fieldValueReal("valueReal", OFTReal);
        layerSpecific->AddField(fieldValueReal);
        layerOther->AddField(fieldValueReal);
        OGRFieldDefn fieldValueNorm("valueNorm", OFTReal);
        layerSpecific->AddField(fieldValueNorm);
        layerOther->AddField(fieldValueNorm);

        // Adding features to the layer
        for (int iStat = 0; iStat < forecasts[0]->GetStationsNb(); iStat++) {
            int currentId = forecasts[0]->GetStationId(iStat);

            // Select the accurate forecast
            bool accurateForecast = false;
            asResultsForecast* forecast = nullptr;
            if (_forecastSelection >= 0) {
                forecast = forecasts[0];
                accurateForecast = forecast->IsSpecificForStationId(currentId);
            } else {
                for (auto& fcst : forecasts) {
                    accurateForecast = fcst->IsSpecificForStationId(currentId);
                    if (accurateForecast) {
                        forecast = fcst;
                        break;
                    }
                }
            }

            if (_forecastManager->GetForecastsNb(_methodSelection) == 1) {
                forecast = forecasts[0];
                accurateForecast = true;
            }

            if (!forecast) {
                wxLogWarning(_("%s is not associated to any forecast"), forecasts[0]->GetStationName(iStat));
                continue;
            }

            OGRPoint station;
            station.setX(forecast->GetStationXCoord(iStat));
            station.setY(forecast->GetStationYCoord(iStat));

            // Field container
            wxArrayDouble data;
            data.Add((double)iStat);
            data.Add((double)currentId);

            // For normalization by the return period
            double factor = 1;
            if (forecast->HasReferenceValues() && returnPeriod != 0) {
                float precip = forecast->GetReferenceValue(iStat, indexReferenceAxis);
                wxASSERT(precip > 0);
                wxASSERT(precip < 500);
                factor = 1.0 / precip;
                wxASSERT(factor > 0);
            }

            // Check available lead times
            if (forecast->GetTargetDatesLength() <= _leadTimeIndex) {
                wxLogError(_("Lead time not available for this forecast."));
                _leadTimeIndex = forecast->GetTargetDatesLength() - 1;
            }

            a1f values = forecast->GetAnalogsValuesRaw(_leadTimeIndex, iStat);

            if (asHasNaN(&values[0], &values[values.size() - 1])) {
                data.Add(NAN);  // 1st real value
                data.Add(NAN);  // 2nd normalized
            } else {
                if (quantile >= 0) {
                    double forecastVal = asGetValueForQuantile(values, quantile);
                    wxASSERT_MSG(forecastVal >= 0, asStrF("Forecast value = %g", forecastVal));
                    data.Add(forecastVal);  // 1st real value
                    forecastVal *= factor;
                    data.Add(forecastVal);  // 2nd normalized
                } else {
                    // Interpretatio
                    double forecastVal = 0;
                    double forecastVal30 = asGetValueForQuantile(values, 0.3f);
                    double forecastVal60 = asGetValueForQuantile(values, 0.6f);
                    double forecastVal90 = asGetValueForQuantile(values, 0.9f);

                    if (forecastVal60 == 0) {
                        forecastVal = 0;
                    } else if (forecastVal30 > 0) {
                        forecastVal = forecastVal90;
                    } else {
                        forecastVal = forecastVal60;
                    }

                    wxASSERT_MSG(forecastVal >= 0, asStrF("Forecast value = %g", forecastVal));
                    data.Add(forecastVal);  // 1st real value
                    forecastVal *= factor;
                    data.Add(forecastVal);  // 2nd normalized
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
            _layerManager->Add(layerOther);
            auto renderOther = new vrRenderVector();
            renderOther->SetSize(1);
            renderOther->SetColorPen(wxColor(150, 150, 150));
            _viewerLayerManager->Add(-1, layerOther, renderOther);
        } else {
            wxDELETE(layerOther);
        }

        _layerManager->Add(layerSpecific);
        auto renderSpecific = new vrRenderVector();
        renderSpecific->SetSize(1);
        renderSpecific->SetColorPen(*wxBLACK);
        _viewerLayerManager->Add(-1, layerSpecific, renderSpecific);
        _viewerLayerManager->FreezeEnd();
    }
}

void asForecastRenderer::ChangeLeadTime(int val) {
    if (_leadTimeIndex == val)  // Already selected
        return;

    _leadTimeIndex = val;
    _leadTimeDate = GetSelectedTargetDate();

    Redraw();
}

void asForecastRenderer::FixMethodSelection() {
    if (_methodSelection >= _forecastManager->GetMethodsNb()) {
        _methodSelection = _forecastManager->GetMethodsNb() - 1;
    }
}