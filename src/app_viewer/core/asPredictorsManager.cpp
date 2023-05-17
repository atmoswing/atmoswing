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
#include "asPredictorOper.h"

/**
 * The constructor of the class handling the predictor data for the Viewer.
 *
 * @param listPredictors The list of predictors from the interface.
 * @param workspace The opened workspace.
 * @param isTargetPredictor A boolean indicating if the predictor is the target.
 */
asPredictorsManager::asPredictorsManager(wxListBox* listPredictors, asWorkspace* workspace, bool isTargetPredictor)
    : m_workspace(workspace),
      m_listPredictors(listPredictors),
      m_isTargetPredictor(isTargetPredictor),
      m_predictor(nullptr),
      m_date(-1),
      m_forecastTimeStepHours(6),
      m_leadTimeNb(5),
      m_needsDataReload(true) {}

/**
 * The destructor of the class handling the predictor data for the Viewer.
 */
asPredictorsManager::~asPredictorsManager() = default;

/**
 * Get the meteorological parameter in use.
 *
 * @return The meteorological parameter.
 */
asPredictor::Parameter asPredictorsManager::GetParameter() {
    return m_predictor->GetParameter();
}

/**
 * Load the data for the selected predictor.
 *
 * @return True if the data was loaded successfully.
 */
bool asPredictorsManager::LoadData() {
    if (!m_needsDataReload) return true;

    wxDELETE(m_predictor);

    int selection = m_listPredictors->GetSelection();
    if (selection < 0) {
        return false;
    }

    asAreaGridFull area = asAreaGridFull(true);

    if (m_isTargetPredictor) {
        wxString directory = m_workspace->GetPredictorDir(m_datasetIds[selection]);
        asPredictorOper* predictor = asPredictorOper::GetInstance(m_datasetIds[selection], m_dataIds[selection]);
        if (!predictor) {
            wxLogError(_("Failed to get an instance of %s from %s."), m_dataIds[selection], m_datasetIds[selection]);
            return false;
        }
        predictor->SetPredictorsRealtimeDirectory(directory);
        predictor->SetRunDateInUse(m_forecastDate);

        double dataHour = 0;
        if (m_forecastTimeStepHours >= 24) {
            dataHour = (m_date - floor(m_forecastDate)) * 24 + m_hours[selection];
        } else {
            dataHour = (m_date - m_forecastDate) * 24 + m_hours[selection];
        }

        if (!predictor->BuildFilenamesAndUrls(dataHour, m_forecastTimeStepHours, 1)) {
            return false;
        }

        if (!predictor->Load(area, m_date + m_hours[selection] / 24, m_levels[selection])) {
            wxLogError(_("The variable %s from %s could not be loaded."), m_dataIds[selection],
                       m_datasetIds[selection]);
            wxDELETE(predictor);
            return false;
        }

        m_predictor = predictor;

    } else {
        wxString directory = m_workspace->GetPredictorDir(m_datasetIds[selection]);
        m_predictor = asPredictor::GetInstance(m_datasetIds[selection], m_dataIds[selection], directory);
        if (!m_predictor) {
            wxLogError(_("Failed to get an instance of %s from %s."), m_dataIds[selection], m_datasetIds[selection]);
            return false;
        }

        if (!m_predictor->Load(area, m_date + m_hours[selection] / 24, m_levels[selection])) {
            wxLogError(_("The variable %s from %s could not be loaded."), m_dataIds[selection],
                       m_datasetIds[selection]);
            wxDELETE(m_predictor);
            return false;
        }
    }

    if (!m_predictor->HasSingleArray()) {
        wxFAIL;
        return false;
    }

    m_data = m_predictor->GetData(0, 0);
    m_longitudes = m_predictor->GetLonAxisPt();
    m_latitudes = m_predictor->GetLatAxisPt();

    m_needsDataReload = false;

    return true;
}

/**
 * Access to a pointer to the loaded data.
 *
 * @return A pointer to the loaded data.
 */
float* asPredictorsManager::GetData() {
    return m_data->data();
}

/**
 * Access to a row of the loaded data array.
 *
 * @param row The row to access.
 * @return A row of the loaded data array.
 */
float* asPredictorsManager::GetDataRow(int row) {
    wxASSERT(m_data->rows() > row);
    return &(*m_data)(row, 0);
}

/**
 * Get the minimum value of the loaded data.
 *
 * @return The minimum value of the loaded data.
 */
float asPredictorsManager::GetDataMin() {
    return m_data->minCoeff();
}

/**
 * Get the maximum value of the loaded data.
 *
 * @return The maximum value of the loaded data.
 */
float asPredictorsManager::GetDataMax() {
    return m_data->maxCoeff();
}

/**
 * Set the desired date for the predictor data.
 *
 * @param date The desired date.
 */
void asPredictorsManager::SetDate(double date) {
    if (m_date == date) return;
    m_date = date;
    m_needsDataReload = true;
}
