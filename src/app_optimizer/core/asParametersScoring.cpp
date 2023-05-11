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
 * Portions Copyright 2013-2014 Pascal Horton, Terranum.
 */

#include "asParametersScoring.h"

#include "asFileParametersCalibration.h"

asParametersScoring::asParametersScoring()
    : asParameters(),
      m_calibrationStart(NAN),
      m_calibrationEnd(NAN) {}

asParametersScoring::~asParametersScoring() {}

void asParametersScoring::AddPredictorVect(ParamsStepVect& step) {
    ParamsPredictorVect predictor;

    predictor.dataId.emplace_back("");
    predictor.level.push_back(0);
    predictor.xMin.push_back(0);
    predictor.xPtsNb.push_back(0);
    predictor.yMin.push_back(0);
    predictor.yPtsNb.push_back(0);
    predictor.hours.push_back(0);
    predictor.criteria.emplace_back("");
    predictor.weight.push_back(1);

    step.predictors.push_back(predictor);
}

bool asParametersScoring::GenerateSimpleParametersFile(const wxString& filePath) const {
    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::Replace);
    if (!fileParams.Open()) return false;

    // Create root nodes
    if (!fileParams.EditRootElement()) return false;

    // Description
    auto nodeDescr = new wxXmlNode(wxXML_ELEMENT_NODE, "description");
    nodeDescr->AddChild(fileParams.CreateNode("method_id", GetMethodId()));
    nodeDescr->AddChild(fileParams.CreateNode("method_id_display", GetMethodIdDisplay()));
    nodeDescr->AddChild(fileParams.CreateNode("specific_tag", GetSpecificTag()));
    nodeDescr->AddChild(fileParams.CreateNode("specific_tag_display", GetSpecificTagDisplay()));
    nodeDescr->AddChild(fileParams.CreateNode("description", GetDescription()));

    fileParams.AddChild(nodeDescr);

    // Time properties
    auto nodeTime = new wxXmlNode(wxXML_ELEMENT_NODE, "time_properties");

    auto nodeTimeArchivePeriod = new wxXmlNode(wxXML_ELEMENT_NODE, "archive_period");
    nodeTime->AddChild(nodeTimeArchivePeriod);
    wxString archiveStart = asTime::GetStringTime(GetArchiveStart(), "DD.MM.YYYY");
    nodeTimeArchivePeriod->AddChild(fileParams.CreateNode("start", archiveStart));
    wxString archiveEnd = asTime::GetStringTime(GetArchiveEnd(), "DD.MM.YYYY");
    nodeTimeArchivePeriod->AddChild(fileParams.CreateNode("end", archiveEnd));

    auto nodeTimeCalibrationPeriod = new wxXmlNode(wxXML_ELEMENT_NODE, "calibration_period");
    nodeTime->AddChild(nodeTimeCalibrationPeriod);
    wxString calibrationStart = asTime::GetStringTime(GetCalibrationStart(), "DD.MM.YYYY");
    nodeTimeCalibrationPeriod->AddChild(fileParams.CreateNode("start", calibrationStart));
    wxString calibrationEnd = asTime::GetStringTime(GetCalibrationEnd(), "DD.MM.YYYY");
    nodeTimeCalibrationPeriod->AddChild(fileParams.CreateNode("end", calibrationEnd));

    if (HasValidationPeriod()) {
        auto nodeTimeValidationPeriod = new wxXmlNode(wxXML_ELEMENT_NODE, "validation_period");
        nodeTime->AddChild(nodeTimeValidationPeriod);
        wxString validationYears;
        vi validationYearsVect = GetValidationYearsVector();
        for (int i = 0; i < (int)validationYearsVect.size(); i++) {
            validationYears << validationYearsVect[i];
            if (i != (int)validationYearsVect.size() - 1) {
                validationYears << ", ";
            }
        }
        nodeTimeValidationPeriod->AddChild(fileParams.CreateNode("years", validationYears));
    }

    wxString timeArrayTimeStep;
    timeArrayTimeStep << GetTargetTimeStepHours();
    nodeTime->AddChild(fileParams.CreateNode("time_step", timeArrayTimeStep));

    auto nodeTimeArrayTarget = new wxXmlNode(wxXML_ELEMENT_NODE, "time_array_target");
    nodeTime->AddChild(nodeTimeArrayTarget);
    nodeTimeArrayTarget->AddChild(fileParams.CreateNode("time_array", GetTimeArrayTargetMode()));
    if (GetTimeArrayTargetMode().IsSameAs("predictand_thresholds")) {
        nodeTimeArrayTarget->AddChild(
            fileParams.CreateNode("predictand_serie_name", GetTimeArrayTargetPredictandSerieName()));
        nodeTimeArrayTarget->AddChild(
            fileParams.CreateNode("predictand_min_threshold", GetTimeArrayTargetPredictandMinThreshold()));
        nodeTimeArrayTarget->AddChild(
            fileParams.CreateNode("predictand_max_threshold", GetTimeArrayTargetPredictandMaxThreshold()));
    }

    auto nodeTimeArrayAnalogs = new wxXmlNode(wxXML_ELEMENT_NODE, "time_array_analogs");
    nodeTime->AddChild(nodeTimeArrayAnalogs);
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNode("time_array", GetTimeArrayAnalogsMode()));
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNode("interval_days", GetAnalogsIntervalDays()));
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNode("exclude_days", GetAnalogsExcludeDays()));

    fileParams.AddChild(nodeTime);

    // Analog dates
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        auto nodeAnalogDates = new wxXmlNode(wxXML_ELEMENT_NODE, "analog_dates");

        nodeAnalogDates->AddChild(fileParams.CreateNode("analogs_number", GetAnalogsNumber(iStep)));

        // Predictors
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            auto nodePredictor = new wxXmlNode(wxXML_ELEMENT_NODE, "predictor");
            nodeAnalogDates->AddChild(nodePredictor);

            nodePredictor->AddChild(fileParams.CreateNode("preload", NeedsPreloading(iStep, iPtor)));
            nodePredictor->AddChild(fileParams.CreateNode("standardize", GetStandardize(iStep, iPtor)));
            nodePredictor->AddChild(fileParams.CreateNode("standardize_mean", GetStandardizeMean(iStep, iPtor)));
            nodePredictor->AddChild(fileParams.CreateNode("standardize_sd", GetStandardizeSd(iStep, iPtor)));

            if (NeedsPreprocessing(iStep, iPtor)) {
                auto nodePreprocessing = new wxXmlNode(wxXML_ELEMENT_NODE, "preprocessing");
                nodePredictor->AddChild(nodePreprocessing);

                nodePreprocessing->AddChild(
                    fileParams.CreateNode("preprocessing_method", GetPreprocessMethod(iStep, iPtor)));

                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    auto nodePreprocessingData = new wxXmlNode(wxXML_ELEMENT_NODE, "preprocessing_data");
                    nodePreprocessing->AddChild(nodePreprocessingData);

                    nodePreprocessingData->AddChild(
                        fileParams.CreateNode("dataset_id", GetPreprocessDatasetId(iStep, iPtor, iPre)));
                    nodePreprocessingData->AddChild(
                        fileParams.CreateNode("data_id", GetPreprocessDataId(iStep, iPtor, iPre)));
                    nodePreprocessingData->AddChild(
                        fileParams.CreateNode("level", GetPreprocessLevel(iStep, iPtor, iPre)));
                    nodePreprocessingData->AddChild(
                        fileParams.CreateNode("time", GetPreprocessHour(iStep, iPtor, iPre)));
                }
            } else {
                nodePredictor->AddChild(fileParams.CreateNode("dataset_id", GetPredictorDatasetId(iStep, iPtor)));
                nodePredictor->AddChild(fileParams.CreateNode("data_id", GetPredictorDataId(iStep, iPtor)));
                nodePredictor->AddChild(fileParams.CreateNode("level", GetPredictorLevel(iStep, iPtor)));
                nodePredictor->AddChild(fileParams.CreateNode("time", GetPredictorHour(iStep, iPtor)));
            }

            auto nodeWindow = new wxXmlNode(wxXML_ELEMENT_NODE, "spatial_window");
            nodePredictor->AddChild(nodeWindow);

            nodeWindow->AddChild(fileParams.CreateNode("grid_type", GetPredictorGridType(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNode("x_min", GetPredictorXmin(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNode("x_points_nb", GetPredictorXptsnb(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNode("x_step", GetPredictorXstep(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNode("y_min", GetPredictorYmin(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNode("y_points_nb", GetPredictorYptsnb(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNode("y_step", GetPredictorYstep(iStep, iPtor)));

            nodePredictor->AddChild(fileParams.CreateNode("criteria", GetPredictorCriteria(iStep, iPtor)));
            nodePredictor->AddChild(fileParams.CreateNode("weight", GetPredictorWeight(iStep, iPtor)));
        }

        fileParams.AddChild(nodeAnalogDates);
    }

    // Analogs values
    auto nodeAnalogValues = new wxXmlNode(wxXML_ELEMENT_NODE, "analog_values");

    auto nodePredictand = new wxXmlNode(wxXML_ELEMENT_NODE, "predictand");
    nodeAnalogValues->AddChild(nodePredictand);

    vvi predictandStationIdsVect = GetPredictandStationIdsVector();
    wxString predictandStationIds = GetPredictandStationIdsVectorString(predictandStationIdsVect);
    nodePredictand->AddChild(fileParams.CreateNode("station_id", predictandStationIds));

    fileParams.AddChild(nodeAnalogValues);

    // Forecast scores
    auto nodeAnalogScore = new wxXmlNode(wxXML_ELEMENT_NODE, "evaluation");

    nodeAnalogScore->AddChild(fileParams.CreateNode("score", GetScoreName()));

    float fsThreshold = GetScoreThreshold();
    if (!isnan(fsThreshold)) {
        nodeAnalogScore->AddChild(fileParams.CreateNode("threshold", fsThreshold));
    }

    float fsQuantile = GetScoreQuantile();
    if (!isnan(fsQuantile)) {
        nodeAnalogScore->AddChild(fileParams.CreateNode("quantile", fsQuantile));
    }

    nodeAnalogScore->AddChild(fileParams.CreateNode("time_array", GetScoreTimeArrayMode()));

    fileParams.AddChild(nodeAnalogScore);

    if (!fileParams.Save()) return false;
    if (!fileParams.Close()) return false;

    return true;
}

bool asParametersScoring::PreprocessingDataIdsOk() {
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                int preprocSize = GetPreprocessSize(iStep, iPtor);

                // Check that the data ID is unique
                for (int iPre = 0; iPre < preprocSize; iPre++) {
                    if (GetPreprocessDataIdVectorSize(iStep, iPtor, iPre) != 1) {
                        wxLogError(_("The preprocess dataId must be unique with the preload option."));
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

wxString asParametersScoring::GetPredictandStationIdsVectorString(vvi& predictandStationIdsVect) const {
    wxString ids;

    for (int i = 0; i < (int)predictandStationIdsVect.size(); i++) {
        vi predictandStationIds = predictandStationIdsVect[i];

        if (predictandStationIds.size() == 1) {
            ids << predictandStationIds[0];
        } else {
            ids.Append("(");

            for (int j = 0; j < (int)predictandStationIds.size(); j++) {
                ids << predictandStationIds[j];

                if (j < (int)predictandStationIds.size() - 1) {
                    ids.Append(",");
                }
            }

            ids.Append(")");
        }

        if (i < (int)predictandStationIdsVect.size() - 1) {
            ids.Append(",");
        }
    }

    return ids;
}

wxString asParametersScoring::Print() const {
    // Create content string
    wxString content = asParameters::Print();

    content.Append(asStrF("|||| Score\t%s\t", GetScoreName()));
    if (!isnan(GetScoreQuantile())) {
        content.Append(asStrF("quantile\t%f\t", GetScoreQuantile()));
    }
    if (!isnan(GetScoreThreshold())) {
        content.Append(asStrF("threshold\t%f\t", GetScoreThreshold()));
    }
    content.Append(asStrF("TimeArray\t%s\t", GetScoreTimeArrayMode()));

    return content;
}

bool asParametersScoring::GetValuesFromString(wxString stringVals) {
    // Get the parameters values
    if (!asParameters::GetValuesFromString(stringVals)) {
        return false;
    }

    // Check that the score is similar
    wxString strVal = asExtractParamValueAndCut(stringVals, "Score");
    if (!strVal.IsSameAs(GetScoreName())) {
        wxLogError(_("The current score (%s) doesn't correspond to the previous one (%s)."), GetScoreName(), strVal);
        asLog::PrintToConsole(asStrF(_("Error: The current score (%s) doesn't correspond to the previous one (%s).\n"),
                                     GetScoreName(), strVal));
        return false;
    }

    return true;
}
