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
      _analogsExcludeDays(0),
      _medianScore(NAN) {
    _scores.quantile = NAN;
    _scores.threshold = NAN;
}

asResultsParametersArray::~asResultsParametersArray() = default;

void asResultsParametersArray::Init(const wxString& fileTag) {
    BuildFileName(fileTag);
    wxASSERT(_scoresCalib.empty());
}

void asResultsParametersArray::Clear() {
    _parameters.clear();
    _predictandStationIds.clear();
    _analogsIntervalDays.clear();
    _scoresCalib.clear();
    _scoresValid.clear();
    _scoresCalibForScoreOnArray.clear();
    _scoresValidForScoreOnArray.clear();
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

    _parameters.push_back(p);
    _predictandStationIds.push_back(params.GetPredictandStationIds());
    _analogsIntervalDays.push_back(params.GetAnalogsIntervalDays());

    if (_scores.name.IsEmpty()) {
        _scores = params.GetScore();
        _analogsExcludeDays = params.GetAnalogsExcludeDays();
    }
}

void asResultsParametersArray::BuildFileName(const wxString& fileTag) {
    ThreadsManager().CritSectionConfig().Enter();
    _filePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    wxString time = asTime::GetStringTime(asTime::NowMJD(asLOCAL), YYYYMMDD_hhmm);
    _filePath.Append(asStrF("/%s_%s.txt", time, fileTag));
}

void asResultsParametersArray::AddWithoutProcessingMedian(asParametersScoring& params, float scoreCalib) {
    StoreValues(params);
    _scoresCalib.push_back(scoreCalib);
    _scoresValid.push_back(NAN);
}

void asResultsParametersArray::Add(asParametersScoring& params, float scoreCalib, float scoreValid) {
    StoreValues(params);
    _scoresCalib.push_back(scoreCalib);
    _scoresValid.push_back(scoreValid);

    ProcessMedianScores();
}

void asResultsParametersArray::Add(asParametersScoring& params, const a1f& scoreCalib, const a1f& scoreValid) {
    StoreValues(params);
    _scoresCalibForScoreOnArray.push_back(scoreCalib);
    _scoresValidForScoreOnArray.push_back(scoreValid);
}

void asResultsParametersArray::ProcessMedianScores() {
    vf scores = _scoresCalib;

    // Does not need to be super precise, so no need to handle even numbers.
    unsigned long mid = scores.size() / 2;
    auto median_it = scores.begin() + mid;
    std::nth_element(scores.begin(), median_it, scores.end());

    _medianScore = scores[mid];
}

bool asResultsParametersArray::HasBeenAssessed(asParametersScoring& params, float& score) {
    for (int i = 0; i < _parameters.size(); ++i) {
        if (params.IsSameAs(_parameters[i], _predictandStationIds[i], _analogsIntervalDays[i])) {
            score = _scoresCalib[i];
            return true;
        }
    }

    return false;
}

bool asResultsParametersArray::HasCloseOneBeenAssessed(asParametersScoring& params, float& score) {
    for (int i = 0; i < _parameters.size(); ++i) {
        if (params.IsCloseTo(_parameters[i], _predictandStationIds[i], _analogsIntervalDays[i])) {
            score = _scoresCalib[i];
            return true;
        }
    }

    return false;
}

bool asResultsParametersArray::Print(int fromIndex) const {
    bool fileExists = wxFileName::FileExists(_filePath);

    asFile::FileMode mode = asFile::Replace;
    if (fileExists) {
        mode = asFile::Append;
    }

    // Create a file
    asFileText fileRes(_filePath, mode);
    if (!fileRes.Open()) return false;

    if (!fileExists) {
        wxString header;
        header = asStrF(_("Optimization processed %s\n"), asTime::GetStringTime(asTime::NowMJD(asLOCAL)));
        fileRes.AddContent(header);
    }

    wxString content = wxEmptyString;

    // Write every parameter one after the other
    for (int iParam = fromIndex; iParam < _scoresCalib.size(); iParam++) {
        content.Append(PrintParams(iParam));

        content.Append(asStrF("|||| Score\t%s\t", _scores.name));
        if (!isnan(_scores.quantile)) {
            content.Append(asStrF("quantile\t%f\t", _scores.quantile));
        }
        if (!isnan(_scores.threshold)) {
            content.Append(asStrF("threshold\t%f\t", _scores.threshold));
        }
        content.Append(asStrF("TimeArray\t%s\t", _scores.timeArrayMode));

        content.Append(asStrF("Calib\t%e\t", _scoresCalib[iParam]));
        content.Append(asStrF("Valid\t%e", _scoresValid[iParam]));
        content.Append("\n");
    }

    // Write every parameter for scores on array one after the other
    for (int iParam = fromIndex; iParam < _scoresCalibForScoreOnArray.size(); iParam++) {
        content.Append(PrintParams(iParam));
        content.Append("Calib\t");
        for (int iRow = 0; iRow < _scoresCalibForScoreOnArray[iParam].size(); iRow++) {
            content.Append(asStrF("%e\t", _scoresCalibForScoreOnArray[iParam][iRow]));
        }
        content.Append("Valid\t");
        for (int iRow = 0; iRow < _scoresValidForScoreOnArray[iParam].size(); iRow++) {
            content.Append(asStrF("%e\t", _scoresValidForScoreOnArray[iParam][iRow]));
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

    content.Append(asStrF("Station\t%s\t", asParameters::PredictandStationIdsToString(_predictandStationIds[iParam])));
    content.Append(asStrF("DaysInt\t%d\t", _analogsIntervalDays[iParam]));
    content.Append(asStrF("ExcludeDays\t%d\t", _analogsExcludeDays));

    asParametersScoring::VectorParamsStep params = _parameters[iParam];

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