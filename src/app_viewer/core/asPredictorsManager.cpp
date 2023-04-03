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

asPredictorsManager::asPredictorsManager(wxListBox* listPredictors)
    : m_listPredictors(listPredictors),
      m_date(-1) {}

asPredictorsManager::~asPredictorsManager() = default;

asPredictor::Parameter asPredictorsManager::GetParameter() {
    return m_predictor->GetParameter();
}

bool asPredictorsManager::LoadData() {
    if (!m_needsDataReload) return true;

    wxDELETE(m_predictor);

    int selection = m_listPredictors->GetSelection();
    if (selection < 0) {
        return false;
    }

    if (m_isTargetPredictor) {
    } else {
        m_predictor = asPredictor::GetInstance(m_datasetIds[selection], m_dataIds[selection], directory);
    }

    if (!m_predictor) {
        wxLogError(_("Failed to get an instance of %s from %s."), m_dataIds[selection], m_datasetIds[selection]);
        return false;
    }

    asAreaGridFull area = asAreaGridFull(true);
    if (!m_predictor->Load(area, m_date + m_hours[selection] / 24, m_levels[selection])) {
        wxLogError(_("The variable %s from %s could not be loaded."), m_dataIds[selection], m_datasetIds[selection]);
        wxDELETE(m_predictor);
        return false;
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

float* asPredictorsManager::GetData() {
    return m_data->data();
}

float* asPredictorsManager::GetDataRow(int row) {
    wxASSERT(m_data->rows() > row);
    return &(*m_data)(row,0);
}

void asPredictorsManager::SetDate(double date) {
    if (m_date == date) return;
    m_date = date;
    m_needsDataReload = true;
}
