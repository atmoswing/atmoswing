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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 * Portions Copyright 2016-2020 Pascal Horton, University of Bern.
 */

#include "asResultsParametersArray.h"

#include "asFileText.h"

asResultsParametersArray::asResultsParametersArray()
    : asResults(),
      m_analogsExcludeDays(0),
      m_medianScore(NAN) {
    m_scores.quantile = NAN;
    m_scores.threshold = NAN;
}

asResultsParametersArray::~asResultsParametersArray() = default;

void asResultsParametersArray::Init(const wxString& fileTag) {
    BuildFileName(fileTag);
    wxASSERT(m_scoresCalib.empty());
}

void asResultsParametersArray::Clear() {
    m_parameters.clear();
    m_predictandStationIds.clear();
    m_analogsIntervalDays.clear();
    m_scoresCalib.clear();
    m_scoresValid.clear();
    m_scoresCalibForScoreOnArray.clear();
    m_scoresValidForScoreOnArray.clear();
}

void asResultsParametersArray::StoreValues(asParametersScoring& params) {
    asParameters::VectorParamsStep p = params.GetParameters();

    for (auto& steps : p) {
        for (auto& predictor : steps.predictors) {
            predictor.preloadDataIds.clear();
            predictor.preloadHours.clear();
            predictor.preloadLevels.clear();
        }
    }

    m_parameters.push_back(p);
    m_predictandStationIds.push_back(params.GetPredictandStationIds());
    m_analogsIntervalDays.push_back(params.GetAnalogsIntervalDays());

    if (m_scores.name.IsEmpty()) {
        m_scores = params.GetScore();
        m_analogsExcludeDays = params.GetAnalogsExcludeDays();
    }
}

void asResultsParametersArray::BuildFileName(const wxString& fileTag) {
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), YYYYMMDD_hhmm);
    m_filePath.Append(asStrF("/%s_%s.txt", time, fileTag));
}

void asResultsParametersArray::AddWithoutProcessingMedian(asParametersScoring& params, float scoreCalib) {
    StoreValues(params);
    m_scoresCalib.push_back(scoreCalib);
    m_scoresValid.push_back(NAN);
}

void asResultsParametersArray::Add(asParametersScoring& params, float scoreCalib, float scoreValid) {
    StoreValues(params);
    m_scoresCalib.push_back(scoreCalib);
    m_scoresValid.push_back(scoreValid);

    ProcessMedianScores();
}

void asResultsParametersArray::Add(asParametersScoring& params, const a1f& scoreCalib, const a1f& scoreValid) {
    StoreValues(params);
    m_scoresCalibForScoreOnArray.push_back(scoreCalib);
    m_scoresValidForScoreOnArray.push_back(scoreValid);
}

void asResultsParametersArray::ProcessMedianScores() {
    vf scores = m_scoresCalib;

    // Does not need to be super precise, so no need to handle even numbers.
    unsigned long mid = scores.size() / 2;
    auto median_it = scores.begin() + mid;
    std::nth_element(scores.begin(), median_it, scores.end());

    m_medianScore = scores[mid];
}

bool asResultsParametersArray::HasBeenAssessed(asParametersScoring& params, float& score) {
    for (int i = 0; i < m_parameters.size(); ++i) {
        if (params.IsSameAs(m_parameters[i], m_predictandStationIds[i], m_analogsIntervalDays[i])) {
            score = m_scoresCalib[i];
            return true;
        }
    }

    return false;
}

bool asResultsParametersArray::HasCloseOneBeenAssessed(asParametersScoring& params, float& score) {
    for (int i = 0; i < m_parameters.size(); ++i) {
        if (params.IsCloseTo(m_parameters[i], m_predictandStationIds[i], m_analogsIntervalDays[i])) {
            score = m_scoresCalib[i];
            return true;
        }
    }

    return false;
}

bool asResultsParametersArray::Print(int fromIndex) const {
    bool fileExists = wxFileName::FileExists(m_filePath);

    asFile::FileMode mode = asFile::Replace;
    if (fileExists) {
        mode = asFile::Append;
    }

    // Create a file
    asFileText fileRes(m_filePath, mode);
    if (!fileRes.Open()) return false;

    if (!fileExists) {
        wxString header;
        header = asStrF(_("Optimization processed %s\n"), asTime::GetStringTime(asTime::NowMJD(asLOCAL)));
        fileRes.AddContent(header);
    }

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (int iParam = fromIndex; iParam < m_scoresCalib.size(); iParam++) {
        content.Append(PrintParams(iParam));

        content.Append(asStrF("|||| Score\t%s\t", m_scores.name));
        if (!isnan(m_scores.quantile)) {
            content.Append(asStrF("quantile\t%f\t", m_scores.quantile));
        }
        if (!isnan(m_scores.threshold)) {
            content.Append(asStrF("threshold\t%f\t", m_scores.threshold));
        }
        content.Append(asStrF("TimeArray\t%s\t", m_scores.timeArrayMode));

        content.Append(asStrF("Calib\t%e\t", m_scoresCalib[iParam]));
        content.Append(asStrF("Valid\t%e", m_scoresValid[iParam]));
        content.Append("\n");
    }

    // Write every parameter for scores on array one after the other
    for (int iParam = fromIndex; iParam < m_scoresCalibForScoreOnArray.size(); iParam++) {
        content.Append(PrintParams(iParam));
        content.Append("Calib\t");
        for (int iRow = 0; iRow < m_scoresCalibForScoreOnArray[iParam].size(); iRow++) {
            content.Append(asStrF("%e\t", m_scoresCalibForScoreOnArray[iParam][iRow]));
        }
        content.Append("Valid\t");
        for (int iRow = 0; iRow < m_scoresValidForScoreOnArray[iParam].size(); iRow++) {
            content.Append(asStrF("%e\t", m_scoresValidForScoreOnArray[iParam][iRow]));
        }
        content.Append("\n");
    }

    fileRes.AddContent(content);

    fileRes.Close();

    return true;
}

wxString asResultsParametersArray::PrintParams(int iParam) const {
    // Create content string
    wxString content = wxEmptyString;

    content.Append(asStrF("Station\t%s\t", asParameters::PredictandStationIdsToString(m_predictandStationIds[iParam])));
    content.Append(asStrF("DaysInt\t%d\t", m_analogsIntervalDays[iParam]));
    content.Append(asStrF("ExcludeDays\t%d\t", m_analogsExcludeDays));

    asParametersScoring::VectorParamsStep params = m_parameters[iParam];

    for (int iStep = 0; iStep < params.size(); iStep++) {
        content.Append(asStrF("|||| Step(%d)\t", iStep));
        content.Append(asStrF("Anb\t%d\t", params[iStep].analogsNumber));

        for (int iPtor = 0; iPtor < params[iStep].predictors.size(); iPtor++) {
            content.Append(asStrF("|| Ptor(%d)\t", iPtor));

            asParameters::ParamsPredictor ptor = params[iStep].predictors[iPtor];

            if (ptor.preprocess) {
                content.Append(asStrF("%s\t", ptor.preprocessMethod));

                for (int iPre = 0; iPre < ptor.preprocessDataIds.size(); iPre++) {
                    content.Append(asStrF("| %s %s\t", ptor.preprocessDatasetIds[iPre], ptor.preprocessDataIds[iPre]));
                    content.Append(asStrF("Level\t%g\t", ptor.preprocessLevels[iPre]));
                    content.Append(asStrF("Time\t%g\t", ptor.preprocessHours[iPre]));
                }
            } else {
                content.Append(asStrF("%s %s\t", ptor.datasetId, ptor.dataId));
                content.Append(asStrF("Level\t%g\t", ptor.level));
                content.Append(asStrF("Time\t%g\t", ptor.hour));
            }

            content.Append(asStrF("GridType\t%s\t", ptor.gridType));
            content.Append(asStrF("xMin\t%g\t", ptor.xMin));
            content.Append(asStrF("xPtsNb\t%d\t", ptor.xPtsNb));
            content.Append(asStrF("xStep\t%g\t", ptor.xStep));
            content.Append(asStrF("yMin\t%g\t", ptor.yMin));
            content.Append(asStrF("yPtsNb\t%d\t", ptor.yPtsNb));
            content.Append(asStrF("yStep\t%g\t", ptor.yStep));
            content.Append(asStrF("Weight\t%e\t", ptor.weight));
            if (!ptor.preprocessMethod.empty()) {
                content.Append(asStrF("%s\t", ptor.preprocessMethod));
            } else {
                content.Append("NoPreprocessing\t");
            }
            content.Append(asStrF("Criteria\t%s\t", ptor.criteria));
        }
    }

    return content;
}