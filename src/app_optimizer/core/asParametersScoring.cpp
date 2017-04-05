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

#include <asFileParametersCalibration.h>

asParametersScoring::asParametersScoring()
        : asParameters()
{
    m_calibrationStart = 0;
    m_calibrationEnd = 0;
    m_forecastScore.name = wxEmptyString;
    m_forecastScore.timeArrayMode = wxEmptyString;
    m_forecastScore.timeArrayDate = 0;
    m_forecastScore.timeArrayIntervalDays = 0;
    m_forecastScore.postprocess = false;
    m_forecastScore.postprocessDupliExp = 0;
    m_forecastScore.postprocessMethod = wxEmptyString;
    m_forecastScore.threshold = NaNFloat; // initialization required
    m_forecastScore.quantile = NaNFloat; // initialization required
}

asParametersScoring::~asParametersScoring()
{
    //dtor
}

void asParametersScoring::AddPredictorVect(ParamsStepVect &step)
{
    ParamsPredictorVect predictor;

    predictor.dataId.push_back("");
    predictor.level.push_back(0);
    predictor.xMin.push_back(0);
    predictor.xPtsNb.push_back(0);
    predictor.yMin.push_back(0);
    predictor.yPtsNb.push_back(0);
    predictor.timeHours.push_back(0);
    predictor.criteria.push_back("");
    predictor.weight.push_back(1);

    step.predictors.push_back(predictor);
}


bool asParametersScoring::GenerateSimpleParametersFile(const wxString &filePath) const
{
    wxLogVerbose(_("Generating parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::Replace);
    if (!fileParams.Open())
        return false;

    // Create root nodes
    if (!fileParams.EditRootElement())
        return false;

    // Description
    wxXmlNode *nodeDescr = new wxXmlNode(wxXML_ELEMENT_NODE, "description");
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("method_id", GetMethodId()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("method_id_display", GetMethodIdDisplay()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("specific_tag", GetSpecificTag()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("specific_tag_display", GetSpecificTagDisplay()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("description", GetDescription()));

    fileParams.AddChild(nodeDescr);

    // Time properties
    wxXmlNode *nodeTime = new wxXmlNode(wxXML_ELEMENT_NODE, "time_properties");

    wxXmlNode *nodeTimeArchivePeriod = new wxXmlNode(wxXML_ELEMENT_NODE, "archive_period");
    nodeTime->AddChild(nodeTimeArchivePeriod);
    wxString archiveStart = asTime::GetStringTime(GetArchiveStart(), "DD.MM.YYYY");
    nodeTimeArchivePeriod->AddChild(fileParams.CreateNodeWithValue("start", archiveStart));
    wxString archiveEnd = asTime::GetStringTime(GetArchiveEnd(), "DD.MM.YYYY");
    nodeTimeArchivePeriod->AddChild(fileParams.CreateNodeWithValue("end", archiveEnd));

    wxXmlNode *nodeTimeCalibrationPeriod = new wxXmlNode(wxXML_ELEMENT_NODE, "calibration_period");
    nodeTime->AddChild(nodeTimeCalibrationPeriod);
    wxString calibrationStart = asTime::GetStringTime(GetCalibrationStart(), "DD.MM.YYYY");
    nodeTimeCalibrationPeriod->AddChild(fileParams.CreateNodeWithValue("start", calibrationStart));
    wxString calibrationEnd = asTime::GetStringTime(GetCalibrationEnd(), "DD.MM.YYYY");
    nodeTimeCalibrationPeriod->AddChild(fileParams.CreateNodeWithValue("end", calibrationEnd));

    if (HasValidationPeriod()) {
        wxXmlNode *nodeTimeValidationPeriod = new wxXmlNode(wxXML_ELEMENT_NODE, "validation_period");
        nodeTime->AddChild(nodeTimeValidationPeriod);
        wxString validationYears;
        VectorInt validationYearsVect = GetValidationYearsVector();
        for (int i = 0; i < (int) validationYearsVect.size(); i++) {
            validationYears << validationYearsVect[i];
            if (i != (int) validationYearsVect.size() - 1) {
                validationYears << ", ";
            }
        }
        nodeTimeValidationPeriod->AddChild(fileParams.CreateNodeWithValue("years", validationYears));
    }

    wxString timeArrayTimeStep;
    timeArrayTimeStep << GetTimeArrayTargetTimeStepHours();
    nodeTime->AddChild(fileParams.CreateNodeWithValue("time_step", timeArrayTimeStep));

    wxXmlNode *nodeTimeArrayTarget = new wxXmlNode(wxXML_ELEMENT_NODE, "time_array_target");
    nodeTime->AddChild(nodeTimeArrayTarget);
    nodeTimeArrayTarget->AddChild(fileParams.CreateNodeWithValue("time_array", GetTimeArrayTargetMode()));
    if (GetTimeArrayTargetMode().IsSameAs("predictand_thresholds")) {
        nodeTimeArrayTarget->AddChild(
                fileParams.CreateNodeWithValue("predictand_serie_name", GetTimeArrayTargetPredictandSerieName()));
        nodeTimeArrayTarget->AddChild(
                fileParams.CreateNodeWithValue("predictand_min_threshold", GetTimeArrayTargetPredictandMinThreshold()));
        nodeTimeArrayTarget->AddChild(
                fileParams.CreateNodeWithValue("predictand_max_threshold", GetTimeArrayTargetPredictandMaxThreshold()));
    }

    wxXmlNode *nodeTimeArrayAnalogs = new wxXmlNode(wxXML_ELEMENT_NODE, "time_array_analogs");
    nodeTime->AddChild(nodeTimeArrayAnalogs);
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNodeWithValue("time_array", GetTimeArrayAnalogsMode()));
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNodeWithValue("interval_days", GetTimeArrayAnalogsIntervalDays()));
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNodeWithValue("exclude_days", GetTimeArrayAnalogsExcludeDays()));

    fileParams.AddChild(nodeTime);


    // Analog dates
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        wxXmlNode *nodeAnalogDates = new wxXmlNode(wxXML_ELEMENT_NODE, "analog_dates");

        nodeAnalogDates->AddChild(fileParams.CreateNodeWithValue("analogs_number", GetAnalogsNumber(iStep)));

        // Predictors
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            wxXmlNode *nodePredictor = new wxXmlNode(wxXML_ELEMENT_NODE, "predictor");
            nodeAnalogDates->AddChild(nodePredictor);

            nodePredictor->AddChild(fileParams.CreateNodeWithValue("preload", NeedsPreloading(iStep, iPtor)));

            if (NeedsPreprocessing(iStep, iPtor)) {
                wxXmlNode *nodePreprocessing = new wxXmlNode(wxXML_ELEMENT_NODE, "preprocessing");
                nodePredictor->AddChild(nodePreprocessing);

                nodePreprocessing->AddChild(
                        fileParams.CreateNodeWithValue("preprocessing_method", GetPreprocessMethod(iStep, iPtor)));

                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    wxXmlNode *nodePreprocessingData = new wxXmlNode(wxXML_ELEMENT_NODE, "preprocessing_data");
                    nodePreprocessing->AddChild(nodePreprocessingData);

                    nodePreprocessingData->AddChild(fileParams.CreateNodeWithValue("dataset_id",
                                                                                   GetPreprocessDatasetId(iStep,
                                                                                                          iPtor,
                                                                                                          iPre)));
                    nodePreprocessingData->AddChild(
                            fileParams.CreateNodeWithValue("data_id", GetPreprocessDataId(iStep, iPtor, iPre)));
                    nodePreprocessingData->AddChild(
                            fileParams.CreateNodeWithValue("level", GetPreprocessLevel(iStep, iPtor, iPre)));
                    nodePreprocessingData->AddChild(
                            fileParams.CreateNodeWithValue("time", GetPreprocessTimeHours(iStep, iPtor, iPre)));
                }
            } else {
                nodePredictor->AddChild(
                        fileParams.CreateNodeWithValue("dataset_id", GetPredictorDatasetId(iStep, iPtor)));
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("data_id", GetPredictorDataId(iStep, iPtor)));
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("level", GetPredictorLevel(iStep, iPtor)));
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("time", GetPredictorTimeHours(iStep, iPtor)));
            }

            wxXmlNode *nodeWindow = new wxXmlNode(wxXML_ELEMENT_NODE, "spatial_window");
            nodePredictor->AddChild(nodeWindow);

            nodeWindow->AddChild(fileParams.CreateNodeWithValue("grid_type", GetPredictorGridType(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("x_min", GetPredictorXmin(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("x_points_nb", GetPredictorXptsnb(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("x_step", GetPredictorXstep(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("y_min", GetPredictorYmin(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("y_points_nb", GetPredictorYptsnb(iStep, iPtor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("y_step", GetPredictorYstep(iStep, iPtor)));

            nodePredictor->AddChild(fileParams.CreateNodeWithValue("criteria", GetPredictorCriteria(iStep, iPtor)));
            nodePredictor->AddChild(fileParams.CreateNodeWithValue("weight", GetPredictorWeight(iStep, iPtor)));
        }

        fileParams.AddChild(nodeAnalogDates);
    }


    // Analogs values
    wxXmlNode *nodeAnalogValues = new wxXmlNode(wxXML_ELEMENT_NODE, "analog_values");

    wxXmlNode *nodePredictand = new wxXmlNode(wxXML_ELEMENT_NODE, "predictand");
    nodeAnalogValues->AddChild(nodePredictand);

    VVectorInt predictandStationIdsVect = GetPredictandStationIdsVector();
    wxString predictandStationIds = GetPredictandStationIdsVectorString(predictandStationIdsVect);
    nodePredictand->AddChild(fileParams.CreateNodeWithValue("station_id", predictandStationIds));

    fileParams.AddChild(nodeAnalogValues);


    // Forecast scores
    wxXmlNode *nodeAnalogScore = new wxXmlNode(wxXML_ELEMENT_NODE, "analog_forecast_score");

    nodeAnalogScore->AddChild(fileParams.CreateNodeWithValue("score", GetForecastScoreName()));

    float fsThreshold = GetForecastScoreThreshold();
    if (!asTools::IsNaN(fsThreshold)) {
        nodeAnalogScore->AddChild(fileParams.CreateNodeWithValue("threshold", fsThreshold));
    }

    float fsQuantile = GetForecastScoreQuantile();
    if (!asTools::IsNaN(fsQuantile)) {
        nodeAnalogScore->AddChild(fileParams.CreateNodeWithValue("quantile", fsQuantile));
    }

    fileParams.AddChild(nodeAnalogScore);


    // Forecast score final
    wxXmlNode *nodeAnalogScoreFinal = new wxXmlNode(wxXML_ELEMENT_NODE, "analog_forecast_score_final");

    nodeAnalogScoreFinal->AddChild(fileParams.CreateNodeWithValue("time_array", GetForecastScoreTimeArrayMode()));

    fileParams.AddChild(nodeAnalogScoreFinal);


    if (!fileParams.Save())
        return false;
    if (!fileParams.Close())
        return false;

    wxLogVerbose(_("Parameters file generated."));

    return true;
}

bool asParametersScoring::PreprocessingPropertiesOk()
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                int preprocSize = GetPreprocessSize(iStep, iPtor);

                // Check that the data ID is unique
                for (int iPre = 0; iPre < preprocSize; iPre++) {
                    if (GetPreprocessDataIdVectorSize(iStep, iPtor, iPre) != 1) {
                        wxLogError(_("The preprocess dataId must be unique with the preload option."));
                        return false;
                    }
                }

                // Different actions depending on the preprocessing method.
                wxString msg = _("The size of the provided predictors (%d) does not match the requirements (%d) in the preprocessing %s method.");
                if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") || method.IsSameAs("Addition") ||
                    method.IsSameAs("Average")) {
                    // No constraints
                } else if (method.IsSameAs("Gradients")) {
                    if (preprocSize != 1) {
                        wxLogError(msg, preprocSize, 1, "Gradient");
                        return false;
                    }
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "HumidityFlux");
                        return false;
                    }
                } else if (method.IsSameAs("HumidityIndex")) {
                    if (preprocSize != 2) {
                        wxLogError(msg, preprocSize, 2, "HumidityIndex");
                        return false;
                    }
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "FormerHumidityIndex");
                        return false;
                    }
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }
            }
        }
    }

    return true;
}

wxString asParametersScoring::GetPredictandStationIdsVectorString(VVectorInt &predictandStationIdsVect) const
{
    wxString Ids;

    for (int i = 0; i < (int) predictandStationIdsVect.size(); i++) {
        VectorInt predictandStationIds = predictandStationIdsVect[i];

        if (predictandStationIds.size() == 1) {
            Ids << predictandStationIds[0];
        } else {
            Ids.Append("(");

            for (int j = 0; j < (int) predictandStationIds.size(); j++) {
                Ids << predictandStationIds[j];

                if (j < (int) predictandStationIds.size() - 1) {
                    Ids.Append(",");
                }
            }

            Ids.Append(")");
        }

        if (i < (int) predictandStationIdsVect.size() - 1) {
            Ids.Append(",");
        }
    }

    return Ids;
}

wxString asParametersScoring::Print() const
{
    // Create content string
    wxString content = asParameters::Print();

    content.Append(wxString::Format("|||| Score \t%s\t", GetForecastScoreName()));
    if (!asTools::IsNaN(GetForecastScoreQuantile())) {
        content.Append(wxString::Format("quantile \t%f\t", GetForecastScoreQuantile()));
    }
    if (!asTools::IsNaN(GetForecastScoreThreshold())) {
        content.Append(wxString::Format("threshold \t%f\t", GetForecastScoreThreshold()));
    }
    content.Append(wxString::Format("TimeArray\t%s\t", GetForecastScoreTimeArrayMode()));

    return content;
}

bool asParametersScoring::GetValuesFromString(wxString stringVals)
{
    // Get the parameters values
    if (!asParameters::GetValuesFromString(stringVals)) {
        return false;
    }

    unsigned int iLeft, iRight;
    wxString strVal;

    // Check that the score is similar
    iLeft = (unsigned int) stringVals.Find("Score");
    stringVals = stringVals.SubString(iLeft + 7, stringVals.Length());
    iLeft = 0;
    iRight = (unsigned int) stringVals.Find("\t");
    strVal = stringVals.SubString(iLeft, iRight - 1);
    if (!strVal.IsSameAs(GetForecastScoreName())) {
        wxLogError(_("The current score (%s) doesn't correspond to the previous one (%s)."), GetForecastScoreName(),
                   strVal);
        wxPrintf(wxString::Format(_("Error: The current score (%s) doesn't correspond to the previous one (%s).\n"),
                                  GetForecastScoreName(), strVal));
        return false;
    }

    return true;
}
