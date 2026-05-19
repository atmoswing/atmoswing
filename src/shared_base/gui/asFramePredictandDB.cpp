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
 */

#include "asFramePredictandDB.h"

#include "asPredictandLightning.h"
#include "asPredictandPrecipitation.h"
#include "asPredictandTemperature.h"

asFramePredictandDB::asFramePredictandDB(wxWindow* parent, wxWindowID id)
    : asFramePredictandDBVirtual(parent, id) {
    SetLabel(_("Predictand database generator"));

    // Set the defaults
    wxConfigBase* pConfig = wxFileConfig::Get();
    _choiceDataParam->SetSelection((int)0);

    _panelProcessing = new asPanelProcessingPrecipitation(_panelMain);
    _sizerProcessing->Add(_panelProcessing, 1, wxALL | wxEXPAND, 5);

    _filePickerCatalogPath->SetPath(pConfig->Read("/PredictandDBToolbox/CatalogPath", wxEmptyString));
    _dirPickerDataDir->SetPath(pConfig->Read("/PredictandDBToolbox/PredictandDataDir", wxEmptyString));
    _dirPickerDestinationDir->SetPath(pConfig->Read("/PredictandDBToolbox/DestinationDir", wxEmptyString));
    _dirPickerPatternsDir->SetPath(pConfig->Read("/PredictandDBToolbox/PatternsDir", wxEmptyString));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePredictandDB::OnClose(wxCloseEvent&) {
    wxBusyCursor wait;

    // Save as defaults
    wxConfigBase* pConfig = wxFileConfig::Get();

    wxString catalogPath = _filePickerCatalogPath->GetPath();
    pConfig->Write("/PredictandDBToolbox/CatalogPath", catalogPath);
    wxString predictandDataDir = _dirPickerDataDir->GetPath();
    pConfig->Write("/PredictandDBToolbox/PredictandDataDir", predictandDataDir);
    wxString destinationDir = _dirPickerDestinationDir->GetPath();
    pConfig->Write("/PredictandDBToolbox/DestinationDir", destinationDir);
    wxString patternsDir = _dirPickerPatternsDir->GetPath();
    pConfig->Write("/PredictandDBToolbox/PatternsDir", patternsDir);

    pConfig->Flush();
}

void asFramePredictandDB::CloseFrame(wxCommandEvent& event) {
    Close();
}

void asFramePredictandDB::FixFrameSize() {
    int w = -1;
    int h = -1;
    GetSize(&w, &h);
    SetMinSize(wxSize(w, -1));
    SetMaxSize(wxSize(w, -1));

    _panelMain->Layout();
    _sizerMainPanel->Fit(_panelMain);
    Layout();
    _sizerMain->Fit(this);

    GetSize(&w, &h);
    SetMinSize(wxSize(w, h));
    SetMaxSize(wxSize(w, h));
}

void asFramePredictandDB::OnDataSelection(wxCommandEvent& event) {
    Freeze();

    _sizerProcessing->Clear();
    wxDELETE(_panelProcessing);

    switch (_choiceDataParam->GetSelection()) {
        case 0:  // precipitation
        {
            _panelProcessing = new asPanelProcessingPrecipitation(_panelMain);
            _sizerProcessing->Add(_panelProcessing, 1, wxALL | wxEXPAND, 5);
            break;
        }
        case 2:  // lightning
        {
            _panelProcessing = new asPanelProcessingLightning(_panelMain);
            _sizerProcessing->Add(_panelProcessing, 1, wxALL | wxEXPAND, 5);
            break;
        }
        default:  // other
        {
            // Nothing to do
            break;
        }
    }

    FixFrameSize();
    Thaw();
}

void asFramePredictandDB::BuildDatabase(wxCommandEvent& event) {
    wxBusyCursor wait;

    try {
        // Get paths
        wxString catalogFilePath = _filePickerCatalogPath->GetPath();
        if (catalogFilePath.IsEmpty()) {
            wxLogError(_("The given path for the predictand catalog is empty."));
            return;
        }
        wxString pathDataDir = _dirPickerDataDir->GetPath();
        if (pathDataDir.IsEmpty()) {
            wxLogError(_("The given path for the data directory is empty."));
            return;
        }
        wxString pathDestinationDir = _dirPickerDestinationDir->GetPath();
        if (pathDestinationDir.IsEmpty()) {
            wxLogError(_("The given path for the output destination is empty."));
            return;
        }
        wxString pathPatternsDir = _dirPickerPatternsDir->GetPath();
        if (pathPatternsDir.IsEmpty()) {
            wxLogError(_("The given path for the patterns directory is empty."));
            return;
        }

        // Get temporal resolution
        asPredictand::TemporalResolution temporalResol = asPredictand::Daily;
        switch (_choiceDataTempResol->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong selection of the temporal resolution option."));
                break;
            }
            case 0:  // 24 hours
            {
                temporalResol = asPredictand::Daily;
                break;
            }
            case 1:  // 6 hours
            {
                temporalResol = asPredictand::SixHourly;
                break;
            }
            case 2:  // Moving temporal window (1-hourly)
            {
                temporalResol = asPredictand::OneHourlyMTW;
                break;
            }
            case 3:  // Moving temporal window (3-hourly)
            {
                temporalResol = asPredictand::ThreeHourlyMTW;
                break;
            }
            case 4:  // Moving temporal window (6-hourly)
            {
                temporalResol = asPredictand::SixHourlyMTW;
                break;
            }
            case 5:  // Moving temporal window (12-hourly)
            {
                temporalResol = asPredictand::TwelveHourlyMTW;
                break;
            }
            default:
                wxLogError(_("Wrong selection of the temporal resolution option."));
        }

        // Get temporal resolution
        asPredictand::SpatialAggregation spatialAggr = asPredictand::Station;
        switch (_choiceDataSpatAggreg->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong selection of the spatial aggregation option."));
                break;
            }
            case 0:  // Station
            {
                spatialAggr = asPredictand::Station;
                break;
            }
            case 1:  // Groupment
            {
                spatialAggr = asPredictand::Groupment;
                break;
            }
            case 2:  // Catchment
            {
                spatialAggr = asPredictand::Catchment;
                break;
            }
            case 3:  // Region
            {
                spatialAggr = asPredictand::Region;
                break;
            }
            default:
                wxLogError(_("Wrong selection of the spatial aggregation option."));
        }

        // Get data parameter
        switch (_choiceDataParam->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong selection of the data parameter option."));
                break;
            }
            case 0:  // Precipitation
            {
                wxASSERT(_panelProcessing);
                auto panel = dynamic_cast<asPanelProcessingPrecipitation*>(_panelProcessing);
                wxASSERT(panel->_checkBoxReturnPeriod);
                wxASSERT(panel->_textCtrlReturnPeriod);
                wxASSERT(panel->_checkBoxSqrt);

                // Return period
                double valReturnPeriod = 0;
                if (panel->_checkBoxReturnPeriod->GetValue()) {
                    wxString valReturnPeriodString = panel->_textCtrlReturnPeriod->GetValue();
                    valReturnPeriodString.ToDouble(&valReturnPeriod);
                    if ((valReturnPeriod < 1) | (valReturnPeriod > 1000)) {
                        wxLogError(_("The given return period is not consistent."));
                        return;
                    }
                }

                // Instantiate a predictand object
                asPredictandPrecipitation predictand(asPredictand::Precipitation, temporalResol, spatialAggr);
                predictand.SetHasReferenceValues(panel->_checkBoxReturnPeriod->GetValue());
                predictand.SetIsSqrt(panel->_checkBoxSqrt->GetValue());
                predictand.BuildPredictandDB(catalogFilePath, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 1:  // Temperature
            {
                // Instantiate a predictand object
                asPredictandTemperature predictand(asPredictand::AirTemperature, temporalResol, spatialAggr);
                predictand.BuildPredictandDB(catalogFilePath, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 2:  // Lightning
            {
                wxASSERT(_panelProcessing);
                auto panel = dynamic_cast<asPanelProcessingLightning*>(_panelProcessing);
                wxASSERT(panel->_checkBoxLog);

                // Instantiate a predictand object
                asPredictandLightning predictand(asPredictand::Lightning, temporalResol, spatialAggr);
                predictand.SetHasReferenceValues(panel->_checkBoxLog->GetValue());
                predictand.BuildPredictandDB(catalogFilePath, pathDataDir, pathPatternsDir, pathDestinationDir);
                break;
            }
            case 3:  // Other
            {
                wxLogError(_("Generic predictand database not yet implemented."));
                break;
            }
            default:
                wxLogError(_("Wrong selection of the data parameter option."));
        }
    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
    }
}
