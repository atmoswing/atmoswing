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
 * Portions Copyright 2022-2023 Pascal Horton, Terranum.
 */

#include "asPredictorsManager.h"

#include "asAreaGridFull.h"
#include "asIncludes.h"
#include "asPredictorOper.h"

asPredictorsManager::asPredictorsManager(asWorkspace* workspace, bool isTargetPredictor)
    : _workspace(workspace),
      _predictor(nullptr),
      _isTargetPredictor(isTargetPredictor),
      _date(-1),
      _forecastTimeStepHours(6),
      _needsDataReload(true) {}

asPredictorsManager::~asPredictorsManager() = default;

asPredictor::Parameter asPredictorsManager::GetParameter() {
    return _predictor->GetParameter();
}

bool asPredictorsManager::LoadData(int selection) {
    if (!_needsDataReload) return true;

    wxDELETE(_predictor);

    if (selection < 0) {
        return false;
    }

    asAreaGridFull area = asAreaGridFull(true);

    if (_isTargetPredictor) {
        wxString directory = _workspace->GetPredictorDir(_datasetIds[selection]);
        asPredictorOper* predictor = asPredictorOper::GetInstance(_datasetIds[selection], _dataIds[selection]);
        if (!predictor) {
            wxLogError(_("Failed to get an instance of %s from %s."), _dataIds[selection], _datasetIds[selection]);
            return false;
        }
        predictor->SetPredictorsRealtimeDirectory(directory);
        predictor->SetRunDateInUse(_forecastDate);
        predictor->SetLevel(_levels[selection]);

        double dataHour = 0;
        if (_forecastTimeStepHours >= 24) {
            dataHour = (_date - floor(_forecastDate)) * 24 + _hours[selection];
        } else {
            dataHour = (_date - _forecastDate) * 24 + _hours[selection];
        }

        if (!predictor->BuildFilenamesAndUrls(dataHour, _forecastTimeStepHours, 1)) {
            return false;
        }

        if (!predictor->Load(area, _date + _hours[selection] / 24, _levels[selection])) {
            wxLogError(_("The variable %s from %s could not be loaded."), _dataIds[selection], _datasetIds[selection]);
            wxDELETE(predictor);
            return false;
        }

        _predictor = predictor;

    } else {
        wxString directory = _workspace->GetPredictorDir(_datasetIds[selection]);
        _predictor = asPredictor::GetInstance(_datasetIds[selection], _dataIds[selection], directory);
        if (!_predictor) {
            wxLogError(_("Failed to get an instance of %s from %s."), _dataIds[selection], _datasetIds[selection]);
            return false;
        }

        if (!_predictor->Load(area, _date + _hours[selection] / 24, _levels[selection])) {
            wxLogError(_("The variable %s from %s could not be loaded."), _dataIds[selection], _datasetIds[selection]);
            wxDELETE(_predictor);
            return false;
        }
    }

    if (!_predictor->HasSingleArray()) {
        wxFAIL;
        return false;
    }

    _data = _predictor->GetData(0, 0);
    _longitudes = _predictor->GetLonAxisPt();
    _latitudes = _predictor->GetLatAxisPt();

    _needsDataReload = false;

    return true;
}

float* asPredictorsManager::GetData() {
    return _data->data();
}

float* asPredictorsManager::GetDataRow(int row) {
    wxASSERT(_data->rows() > row);
    return &(*_data)(row, 0);
}

float asPredictorsManager::GetDataMin() {
    return _data->minCoeff();
}

float asPredictorsManager::GetDataMax() {
    return _data->maxCoeff();
}

void asPredictorsManager::SetDate(double date) {
    if (_date == date) return;
    _date = date;
    _needsDataReload = true;
}
