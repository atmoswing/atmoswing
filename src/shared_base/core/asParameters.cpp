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
 */

#include "asParameters.h"

#include "asFileAscii.h"
#include <wx/tokenzr.h>
#include "asFileParameters.h"


asParameters::asParameters()
        : m_archiveStart(NaNd),
          m_archiveEnd(NaNd),
          m_timeArrayAnalogsIntervalDays(0),
          m_predictandStationIds(),
          m_timeMinHours(0),
          m_timeMaxHours(0),
          m_dateProcessed(asTime::GetStringTime(asTime::NowTimeStruct(asLOCAL))),
          m_timeArrayTargetMode("simple"),
          m_timeArrayTargetTimeStepHours(0),
          m_timeArrayTargetPredictandMinThreshold(0),
          m_timeArrayTargetPredictandMaxThreshold(0),
          m_timeArrayAnalogsMode("days_interval"),
          m_timeArrayAnalogsTimeStepHours(0),
          m_timeArrayAnalogsExcludeDays(0),
          m_predictandParameter(asPredictand::Precipitation),
          m_predictandTemporalResolution(asPredictand::Daily),
          m_predictandSpatialAggregation(asPredictand::Station),
          m_predictandTimeHours(0)
{

}

void asParameters::AddStep()
{
    ParamsStep step;

    step.analogsNumber = 0;

    m_steps.push_back(step);
}

void asParameters::AddPredictor()
{
    AddPredictor(m_steps[m_steps.size() - 1]);
}

void asParameters::AddPredictor(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.datasetId = wxEmptyString;
    predictor.dataId = wxEmptyString;
    predictor.preload = false;
    predictor.preloadXmin = 0;
    predictor.preloadXptsnb = 0;
    predictor.preloadYmin = 0;
    predictor.preloadYptsnb = 0;
    predictor.preprocess = false;
    predictor.preprocessMethod = wxEmptyString;
    predictor.standardize = false;
    predictor.level = 0;
    predictor.xMin = 0;
    predictor.xPtsNb = 1;
    predictor.xStep = 0;
    predictor.xShift = 0;
    predictor.yMin = 0;
    predictor.yPtsNb = 1;
    predictor.yStep = 0;
    predictor.yShift = 0;
    predictor.flatAllowed = asFLAT_FORBIDDEN;
    predictor.timeHours = 0;
    predictor.criteria = wxEmptyString;
    predictor.weight = 1;
    predictor.membersNb = 0;

    step.predictors.push_back(predictor);
}

void asParameters::AddPredictor(int iStep)
{
    ParamsPredictor predictor;

    predictor.datasetId = wxEmptyString;
    predictor.dataId = wxEmptyString;
    predictor.preload = false;
    predictor.preloadXmin = 0;
    predictor.preloadXptsnb = 0;
    predictor.preloadYmin = 0;
    predictor.preloadYptsnb = 0;
    predictor.preprocess = false;
    predictor.preprocessMethod = wxEmptyString;
    predictor.standardize = false;
    predictor.level = 0;
    predictor.gridType = "regular";
    predictor.xMin = 0;
    predictor.xPtsNb = 1;
    predictor.xStep = 0;
    predictor.xShift = 0;
    predictor.yMin = 0;
    predictor.yPtsNb = 1;
    predictor.yStep = 0;
    predictor.yShift = 0;
    predictor.flatAllowed = asFLAT_FORBIDDEN;
    predictor.timeHours = 0;
    predictor.criteria = wxEmptyString;
    predictor.weight = 1;
    predictor.membersNb = 0;

    m_steps[iStep].predictors.push_back(predictor);
}

bool asParameters::LoadFromFile(const wxString &filePath)
{
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParameters fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open())
        return false;

    if (!fileParams.CheckRootElement())
        return false;

    int iStep = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {

        if (nodeProcess->GetName() == "description") {
            if (!ParseDescription(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "time_properties") {
            if (!ParseTimeProperties(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_dates") {
            AddStep();
            if (!ParseAnalogDatesParams(fileParams, iStep, nodeProcess))
                return false;
            iStep++;

        } else if (nodeProcess->GetName() == "analog_values") {
            if (!ParseAnalogValuesParams(fileParams, nodeProcess))
                return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    if (!InputsOK())
        return false;

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    wxLogVerbose(_("Parameters file loaded."));

    return true;
}

bool asParameters::ParseDescription(asFileParameters &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParam = nodeProcess->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "method_id") {
            SetMethodId(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "method_id_display") {
            SetMethodIdDisplay(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag") {
            SetSpecificTag(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "specific_tag_display") {
            SetSpecificTagDisplay(asFileParameters::GetString(nodeParam));
        } else if (nodeParam->GetName() == "description") {
            SetDescription(asFileParameters::GetString(nodeParam));
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }
    return true;
}

bool asParameters::ParseTimeProperties(asFileParameters &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "archive_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    if (!SetArchiveYearStart(asFileParameters::GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end_year") {
                    if (!SetArchiveYearEnd(asFileParameters::GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "start") {
                    if (!SetArchiveStart(asFileParameters::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end") {
                    if (!SetArchiveEnd(asFileParameters::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetTimeArrayAnalogsTimeStepHours(asFileParameters::GetDouble(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_step") {
            if (!SetTimeArrayTargetTimeStepHours(asFileParameters::GetDouble(nodeParamBlock)))
                return false;
            if (!SetTimeArrayAnalogsTimeStepHours(asFileParameters::GetDouble(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "time_array_target") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayTargetMode(asFileParameters::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "predictand_serie_name") {
                    if (!SetTimeArrayTargetPredictandSerieName(asFileParameters::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "predictand_min_threshold") {
                    if (!SetTimeArrayTargetPredictandMinThreshold(asFileParameters::GetFloat(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "predictand_max_threshold") {
                    if (!SetTimeArrayTargetPredictandMaxThreshold(asFileParameters::GetFloat(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "time_array_analogs") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "time_array") {
                    if (!SetTimeArrayAnalogsMode(asFileParameters::GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "interval_days") {
                    if (!SetTimeArrayAnalogsIntervalDays(asFileParameters::GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "exclude_days") {
                    if (!SetTimeArrayAnalogsExcludeDays(asFileParameters::GetInt(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParameters::ParseAnalogDatesParams(asFileParameters &fileParams, int iStep, const wxXmlNode *nodeProcess)
{
    int iPtor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            if (!SetAnalogsNumber(iStep, asFileParameters::GetInt(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            SetStandardize(iStep, iPtor, false);
            if (!ParsePredictors(fileParams, iStep, iPtor, nodeParamBlock))
                return false;
            iPtor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }

    return true;
}

bool asParameters::ParsePredictors(asFileParameters &fileParams, int iStep, int iPtor, const wxXmlNode *nodeParamBlock)
{
    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
    while (nodeParam) {
        if (nodeParam->GetName() == "preload") {
            SetPreload(iStep, iPtor, asFileParameters::GetBool(nodeParam));
        } else if (nodeParam->GetName() == "standardize") {
            SetStandardize(iStep, iPtor, asFileParameters::GetBool(nodeParam));
        } else if (nodeParam->GetName() == "preprocessing") {
            SetPreprocess(iStep, iPtor, true);
            if (!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam))
                return false;
        } else if (nodeParam->GetName() == "dataset_id") {
            if (!SetPredictorDatasetId(iStep, iPtor, asFileParameters::GetString(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "data_id") {
            if (!SetPredictorDataId(iStep, iPtor, asFileParameters::GetString(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "level") {
            if (!SetPredictorLevel(iStep, iPtor, asFileParameters::GetFloat(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "time") {
            if (!SetPredictorTimeHours(iStep, iPtor, asFileParameters::GetDouble(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "members") {
            if (!SetPredictorMembersNb(iStep, iPtor, asFileParameters::GetInt(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "spatial_window") {
            wxXmlNode *nodeWindow = nodeParam->GetChildren();
            while (nodeWindow) {
                if (nodeWindow->GetName() == "grid_type") {
                    if (!SetPredictorGridType(iStep, iPtor, asFileParameters::GetString(nodeWindow, "regular")))
                        return false;
                } else if (nodeWindow->GetName() == "x_min") {
                    if (!SetPredictorXmin(iStep, iPtor, asFileParameters::GetDouble(nodeWindow)))
                        return false;
                } else if (nodeWindow->GetName() == "x_points_nb") {
                    if (!SetPredictorXptsnb(iStep, iPtor, asFileParameters::GetInt(nodeWindow)))
                        return false;
                } else if (nodeWindow->GetName() == "x_step") {
                    if (!SetPredictorXstep(iStep, iPtor, asFileParameters::GetDouble(nodeWindow)))
                        return false;
                } else if (nodeWindow->GetName() == "y_min") {
                    if (!SetPredictorYmin(iStep, iPtor, asFileParameters::GetDouble(nodeWindow)))
                        return false;
                } else if (nodeWindow->GetName() == "y_points_nb") {
                    if (!SetPredictorYptsnb(iStep, iPtor, asFileParameters::GetInt(nodeWindow)))
                        return false;
                } else if (nodeWindow->GetName() == "y_step") {
                    if (!SetPredictorYstep(iStep, iPtor, asFileParameters::GetDouble(nodeWindow)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeWindow);
                }
                nodeWindow = nodeWindow->GetNext();
            }
        } else if (nodeParam->GetName() == "criteria") {
            if (!SetPredictorCriteria(iStep, iPtor, asFileParameters::GetString(nodeParam)))
                return false;
        } else if (nodeParam->GetName() == "weight") {
            if (!SetPredictorWeight(iStep, iPtor, asFileParameters::GetFloat(nodeParam)))
                return false;
        } else {
            fileParams.UnknownNode(nodeParam);
        }
        nodeParam = nodeParam->GetNext();
    }

    return true;
}

bool asParameters::ParsePreprocessedPredictors(asFileParameters &fileParams, int iStep, int iPtor,
                                               const wxXmlNode *nodeParam)
{
    int iPre = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(iStep, iPtor, asFileParameters::GetString(nodePreprocess)))
                return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "dataset_id") {
                    if (!SetPreprocessDatasetId(iStep, iPtor, iPre, asFileParameters::GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "data_id") {
                    if (!SetPreprocessDataId(iStep, iPtor, iPre, asFileParameters::GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "level") {
                    if (!SetPreprocessLevel(iStep, iPtor, iPre, asFileParameters::GetFloat(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "time") {
                    if (!SetPreprocessTimeHours(iStep, iPtor, iPre, asFileParameters::GetDouble(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "members") {
                    if (!SetPreprocessMembersNb(iStep, iPtor, iPre, asFileParameters::GetInt(nodeParamPreprocess)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParamPreprocess);
                }
                nodeParamPreprocess = nodeParamPreprocess->GetNext();
            }
            iPre++;
        } else {
            fileParams.UnknownNode(nodePreprocess);
        }
        nodePreprocess = nodePreprocess->GetNext();
    }
    return true;
}

bool asParameters::ParseAnalogValuesParams(asFileParameters &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id") {
                    if (!SetPredictandStationIds(asFileParameters::GetStationIds(asFileParameters::GetString(nodeParam))))
                        return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictandTimeHours(asFileParameters::GetDouble(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }

    return true;
}

bool asParameters::SetSpatialWindowProperties()
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (GetPredictorXptsnb(iStep, iPtor) == 0)
                SetPredictorXptsnb(iStep, iPtor, 1);
            if (GetPredictorYptsnb(iStep, iPtor) == 0)
                SetPredictorYptsnb(iStep, iPtor, 1);

            double xShift = std::fmod(GetPredictorXmin(iStep, iPtor), GetPredictorXstep(iStep, iPtor));
            if (xShift < 0)
                xShift += GetPredictorXstep(iStep, iPtor);
            if (!SetPredictorXshift(iStep, iPtor, xShift))
                return false;

            double yShift = std::fmod(GetPredictorYmin(iStep, iPtor), GetPredictorYstep(iStep, iPtor));
            if (yShift < 0)
                yShift += GetPredictorYstep(iStep, iPtor);
            if (!SetPredictorYshift(iStep, iPtor, yShift))
                return false;

            if (GetPredictorXptsnb(iStep, iPtor) == 1 || GetPredictorYptsnb(iStep, iPtor) == 1) {
                SetPredictorFlatAllowed(iStep, iPtor, asFLAT_ALLOWED);
            }
        }
    }

    return true;
}

bool asParameters::SetPreloadingProperties()
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            // Set maximum extent
            if (NeedsPreloading(iStep, iPtor)) {
                if (!SetPreloadXmin(iStep, iPtor, GetPredictorXmin(iStep, iPtor)))
                    return false;
                if (!SetPreloadYmin(iStep, iPtor, GetPredictorYmin(iStep, iPtor)))
                    return false;
                if (!SetPreloadXptsnb(iStep, iPtor, GetPredictorXptsnb(iStep, iPtor)))
                    return false;
                if (!SetPreloadYptsnb(iStep, iPtor, GetPredictorYptsnb(iStep, iPtor)))
                    return false;
            }

            // Change predictor properties when preprocessing
            if (NeedsPreprocessing(iStep, iPtor)) {
                if (GetPreprocessSize(iStep, iPtor) == 1) {
                    SetPredictorDatasetId(iStep, iPtor, GetPreprocessDatasetId(iStep, iPtor, 0));
                    SetPredictorDataId(iStep, iPtor, GetPreprocessDataId(iStep, iPtor, 0));
                    SetPredictorLevel(iStep, iPtor, GetPreprocessLevel(iStep, iPtor, 0));
                    SetPredictorTimeHours(iStep, iPtor, GetPreprocessTimeHours(iStep, iPtor, 0));
                } else {
                    SetPredictorDatasetId(iStep, iPtor, "mix");
                    SetPredictorDataId(iStep, iPtor, "mix");
                    SetPredictorLevel(iStep, iPtor, 0);
                    SetPredictorTimeHours(iStep, iPtor, 0);
                }
            }

            // Set levels and time for preloading
            if (NeedsPreloading(iStep, iPtor) && !NeedsPreprocessing(iStep, iPtor)) {
                if (!SetPreloadDataIds(iStep, iPtor, GetPredictorDataId(iStep, iPtor)))
                    return false;
                if (!SetPreloadLevels(iStep, iPtor, GetPredictorLevel(iStep, iPtor)))
                    return false;
                if (!SetPreloadTimeHours(iStep, iPtor, GetPredictorTimeHours(iStep, iPtor)))
                    return false;
            } else if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                vf preprocLevels;
                vd preprocTimeHours;
                int preprocSize = GetPreprocessSize(iStep, iPtor);

                // Different actions depending on the preprocessing method.
                wxString msg = _("The size of the provided predictors (%d) does not match the requirements (%d) in the preprocessing %s method.");
                if (NeedsGradientPreprocessing(iStep, iPtor)) {
                    if (preprocSize != 1) {
                        wxLogError(msg, preprocSize, 1, "Gradient");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(iStep, iPtor, 0));
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "HumidityFlux");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(iStep, iPtor, 0));
                } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                           method.IsSameAs("HumidityIndex")) {
                    if (preprocSize != 2) {
                        wxLogError(msg, preprocSize, 2, "HumidityIndex");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(iStep, iPtor, 0));
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (preprocSize != 4) {
                        wxLogError(msg, preprocSize, 4, "FormerHumidityIndex");
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(iStep, iPtor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(iStep, iPtor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(iStep, iPtor, 1));
                } else {
                    wxLogWarning(_("The %s preprocessing method is not yet handled with the preload option."), method);
                }

                if (!SetPreloadLevels(iStep, iPtor, preprocLevels))
                    return false;
                if (!SetPreloadTimeHours(iStep, iPtor, preprocTimeHours))
                    return false;
            }
        }
    }

    return true;
}

bool asParameters::InputsOK() const
{
    // Time properties
    if (asIsNaN(GetArchiveStart())) {
        wxLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (asIsNaN(GetArchiveEnd())) {
        wxLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayTargetTimeStepHours() <= 0) {
        wxLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsTimeStepHours() <= 0) {
        wxLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
        GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0) {
        if (GetTimeArrayTargetPredictandSerieName().IsEmpty()) {
            wxLogError(_("The predictand time series (for the threshold preselection) was not provided in the parameters file."));
            return false;
        }
        if (GetTimeArrayTargetPredictandMinThreshold() == GetTimeArrayTargetPredictandMaxThreshold()) {
            wxLogError(_("The provided min/max predictand thresholds are equal in the parameters file."));
            return false;
        }
    }

    if (GetTimeArrayAnalogsMode().CmpNoCase("interval_days") == 0 ||
        GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays") == 0) {
        if (GetTimeArrayAnalogsIntervalDays() <= 0) {
            wxLogError(_("The interval days for the analogs preselection was not provided in the parameters file."));
            return false;
        }
        if (GetTimeArrayAnalogsExcludeDays() <= 0) {
            wxLogError(_("The number of days to exclude around the target date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumber(i) <= 0) {
            wxLogError(_("The number of analogs (step %d) was not provided in the parameters file."), i);
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    wxLogError(_("The preprocessing method (step %d, predictor %d) was not provided in the parameters file."),
                               i, j);
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessDatasetId(i, j, k).IsEmpty()) {
                        wxLogError(_("The dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessDataId(i, j, k).IsEmpty()) {
                        wxLogError(_("The data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                }
            } else {
                if (GetPredictorDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The dataset (step %d, predictor %d) was not provided in the parameters file."), i, j);
                    return false;
                }
                if (GetPredictorDataId(i, j).IsEmpty()) {
                    wxLogError(_("The data (step %d, predictor %d) was not provided in the parameters file."), i, j);
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                wxLogError(_("The grid type (step %d, predictor %d) is empty in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorXptsnb(i, j) == 0) {
                wxLogError(_("The X points nb value (step %d, predictor %d) was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorYptsnb(i, j) == 0) {
                wxLogError(_("The Y points nb value (step %d, predictor %d) was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorCriteria(i, j).IsEmpty()) {
                wxLogError(_("The criteria (step %d, predictor %d) was not provided in the parameters file."), i, j);
                return false;
            }
        }
    }

    return true;
}

bool asParameters::PreprocessingPropertiesOk() const
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                int preprocSize = GetPreprocessSize(iStep, iPtor);

                // Different actions depending on the preprocessing method.
                wxString msg = _("The size of the provided predictors (%d) does not match the requirements (%d) in the preprocessing %s method.");
                if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") || method.IsSameAs("Addition") ||
                    method.IsSameAs("Average")) {
                    // No constraints
                } else if (NeedsGradientPreprocessing(iStep, iPtor)) {
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

bool asParameters::FixAnalogsNb()
{
    // Check analogs number coherence
    int analogsNb = GetAnalogsNumber(0);
    for (unsigned int iStep = 1; iStep < m_steps.size(); iStep++) {
        if (GetAnalogsNumber(iStep) > analogsNb) {
            SetAnalogsNumber(iStep, analogsNb);
        } else {
            analogsNb = GetAnalogsNumber(iStep);
        }
    }

    return true;
}

void asParameters::SortLevelsAndTime()
{
    // Sort levels on every analogy level
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        // Get the predictors vector
        VectorParamsPredictors oldPtors = GetVectorParamsPredictors(iStep);
        VectorParamsPredictors newPtors;

        // Sort
        while (true) {
            if (oldPtors.empty()) {
                break;
            }

            // Find the smallest level and hour combination
            int lowestIndex = 0;
            float level;
            double hour;
            if (oldPtors[0].preprocess) {
                level = oldPtors[0].preprocessLevels[0];
                hour = oldPtors[0].preprocessTimeHours[0];
            } else {
                level = oldPtors[0].level;
                hour = oldPtors[0].timeHours;
            }

            for (unsigned int i = 1; i < oldPtors.size(); i++) {
                // Get next level and hour
                float nextLevel;
                double nextHour;
                if (oldPtors[i].preprocess) {
                    nextLevel = oldPtors[i].preprocessLevels[0];
                    nextHour = oldPtors[i].preprocessTimeHours[0];
                } else {
                    nextLevel = oldPtors[i].level;
                    nextHour = oldPtors[i].timeHours;
                }

                // Compare to previous one
                if (nextLevel < level) {
                    lowestIndex = i;
                    level = nextLevel;
                    hour = nextHour;
                } else if (nextLevel == level) {
                    if (nextHour < hour) {
                        lowestIndex = i;
                        level = nextLevel;
                        hour = nextHour;
                    }
                }
            }

            // Store in the new container and remove from the old one
            newPtors.push_back(oldPtors[lowestIndex]);
            oldPtors.erase(oldPtors.begin() + lowestIndex);

            // Store the sorted vector
            SetVectorParamsPredictors(iStep, newPtors);
        }
    }
}

vi asParameters::GetFileStationIds(wxString stationIdsString)
{
    // Trim
    stationIdsString.Trim(true);
    stationIdsString.Trim(false);

    vi ids;

    if (stationIdsString.IsEmpty()) {
        wxLogError(_("The station ID was not provided."));
        return ids;
    }

    // Multivariate
    if (stationIdsString.SubString(0, 0).IsSameAs("(") || stationIdsString.SubString(0, 1).IsSameAs("'(")) {
        wxString subStr = wxEmptyString;
        if (stationIdsString.SubString(0, 0).IsSameAs("(")) {
            subStr = stationIdsString.SubString(1, stationIdsString.Len() - 1);
        } else {
            subStr = stationIdsString.SubString(2, stationIdsString.Len() - 1);
        }

        // Check that it contains only 1 opening bracket
        if (subStr.Find("(") != wxNOT_FOUND) {
            wxLogError(_("The format of the station ID is not correct (more than one opening bracket)."));
            return ids;
        }

        // Check that it contains 1 closing bracket at the end
        if (subStr.Find(")") != subStr.size() - 1 && subStr.Find(")'") != subStr.size() - 2) {
            wxLogError(_("The format of the station ID is not correct (location of the closing bracket)."));
            return ids;
        }

        // Extract content
        wxChar separator = ',';
        while (subStr.Find(separator) != wxNOT_FOUND) {
            wxString strBefore = subStr.BeforeFirst(separator);
            subStr = subStr.AfterFirst(separator);
            int id = wxAtoi(strBefore);
            ids.push_back(id);
        }
        if (!subStr.IsEmpty()) {
            int id = wxAtoi(subStr);
            ids.push_back(id);
        }
    } else {
        // Check for single value
        if (stationIdsString.Find("(") != wxNOT_FOUND || stationIdsString.Find(")") != wxNOT_FOUND ||
            stationIdsString.Find(",") != wxNOT_FOUND) {
            wxLogError(_("The format of the station ID is not correct (should be only digits)."));
            return ids;
        }
        int id = wxAtoi(stationIdsString);
        ids.push_back(id);
    }

    return ids;
}

wxString asParameters::GetPredictandStationIdsString() const
{
    wxString ids;

    if (m_predictandStationIds.size() == 1) {
        ids << m_predictandStationIds[0];
    } else {
        ids = "(";

        for (int i = 0; i < (int) m_predictandStationIds.size(); i++) {
            ids << m_predictandStationIds[i];

            if (i < (int) m_predictandStationIds.size() - 1) {
                ids.Append(",");
            }
        }

        ids.Append(")");
    }

    return ids;
}

bool asParameters::FixTimeLimits()
{
    double minHour = 1000.0, maxHour = -1000.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                double minHourPredictor = 1000.0, maxHourPredictor = -1000.0;

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(m_steps[i].predictors[j].preprocessTimeHours[k], minHour);
                    maxHour = wxMax(m_steps[i].predictors[j].preprocessTimeHours[k], maxHour);
                    minHourPredictor = wxMin(m_steps[i].predictors[j].preprocessTimeHours[k], minHourPredictor);
                    maxHourPredictor = wxMax(m_steps[i].predictors[j].preprocessTimeHours[k], maxHourPredictor);
                    m_steps[i].predictors[j].timeHours = minHourPredictor;
                }
            } else {
                minHour = wxMin(m_steps[i].predictors[j].timeHours, minHour);
                maxHour = wxMax(m_steps[i].predictors[j].timeHours, maxHour);
            }
        }
    }

    m_timeMinHours = minHour;
    m_timeMaxHours = maxHour;

    return true;
}

bool asParameters::FixWeights()
{
    for (int i = 0; i < GetStepsNb(); i++) {
        // Sum the weights
        float totWeight = 0;
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            totWeight += m_steps[i].predictors[j].weight;
        }

        // Correct to set the total to 1
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            m_steps[i].predictors[j].weight /= totWeight;
        }
    }

    return true;
}

bool asParameters::FixCoordinates()
{
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (m_steps[i].predictors[j].gridType.IsSameAs("regular", false)) {

                // Check that the coordinates are a multiple of the steps
                if (std::abs(std::fmod(m_steps[i].predictors[j].xMin - m_steps[i].predictors[j].xShift,
                                       m_steps[i].predictors[j].xStep)) > 0) {
                    double factor = (m_steps[i].predictors[j].xMin - m_steps[i].predictors[j].xShift) /
                                    m_steps[i].predictors[j].xStep;
                    factor = asRound(factor);
                    m_steps[i].predictors[j].xMin =
                            factor * m_steps[i].predictors[j].xStep + m_steps[i].predictors[j].xShift;
                }

                if (std::abs(std::fmod(m_steps[i].predictors[j].yMin - m_steps[i].predictors[j].yShift,
                                       m_steps[i].predictors[j].yStep)) > 0) {
                    double factor = (m_steps[i].predictors[j].yMin - m_steps[i].predictors[j].yShift) /
                                    m_steps[i].predictors[j].yStep;
                    factor = asRound(factor);
                    m_steps[i].predictors[j].yMin =
                            factor * m_steps[i].predictors[j].yStep + m_steps[i].predictors[j].yShift;
                }
            }

            if (m_steps[i].predictors[j].flatAllowed == asFLAT_FORBIDDEN) {
                // Check that the size is larger than 1 point
                if (m_steps[i].predictors[j].xPtsNb < 2) {
                    m_steps[i].predictors[j].xPtsNb = 2;
                }

                if (m_steps[i].predictors[j].yPtsNb < 2) {
                    m_steps[i].predictors[j].yPtsNb = 2;
                }
            } else {
                // Check that the size is larger than 0
                if (m_steps[i].predictors[j].xPtsNb < 1) {
                    m_steps[i].predictors[j].xPtsNb = 1;
                }

                if (m_steps[i].predictors[j].yPtsNb < 1) {
                    m_steps[i].predictors[j].yPtsNb = 1;
                }
            }
        }
    }

    return true;
}

wxString asParameters::Print() const
{
    // Create content string
    wxString content = wxEmptyString;

    content.Append(wxString::Format("Station\t%s\t", GetPredictandStationIdsString()));
    content.Append(wxString::Format("DaysInt\t%d\t", GetTimeArrayAnalogsIntervalDays()));
    content.Append(wxString::Format("ExcludeDays\t%d\t", GetTimeArrayAnalogsExcludeDays()));

    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        content.Append(wxString::Format("|||| Step(%d)\t", iStep));
        content.Append(wxString::Format("Anb\t%d\t", GetAnalogsNumber(iStep)));

        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            content.Append(wxString::Format("|| Ptor(%d)\t", iPtor));

            if (NeedsPreprocessing(iStep, iPtor)) {
                content.Append(wxString::Format("%s\t", GetPreprocessMethod(iStep, iPtor)));

                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    content.Append(wxString::Format("| %s %s\t", GetPreprocessDatasetId(iStep, iPtor, iPre),
                                                    GetPreprocessDataId(iStep, iPtor, iPre)));
                    content.Append(wxString::Format("Level\t%g\t", GetPreprocessLevel(iStep, iPtor, iPre)));
                    content.Append(wxString::Format("Time\t%g\t", GetPreprocessTimeHours(iStep, iPtor, iPre)));
                }
            } else {
                content.Append(wxString::Format("%s %s\t", GetPredictorDatasetId(iStep, iPtor),
                                                GetPredictorDataId(iStep, iPtor)));
                content.Append(wxString::Format("Level\t%g\t", GetPredictorLevel(iStep, iPtor)));
                content.Append(wxString::Format("Time\t%g\t", GetPredictorTimeHours(iStep, iPtor)));
            }

            content.Append(wxString::Format("GridType\t%s\t", GetPredictorGridType(iStep, iPtor)));
            content.Append(wxString::Format("xMin\t%g\t", GetPredictorXmin(iStep, iPtor)));
            content.Append(wxString::Format("xPtsNb\t%d\t", GetPredictorXptsnb(iStep, iPtor)));
            content.Append(wxString::Format("xStep\t%g\t", GetPredictorXstep(iStep, iPtor)));
            content.Append(wxString::Format("yMin\t%g\t", GetPredictorYmin(iStep, iPtor)));
            content.Append(wxString::Format("yPtsNb\t%d\t", GetPredictorYptsnb(iStep, iPtor)));
            content.Append(wxString::Format("yStep\t%g\t", GetPredictorYstep(iStep, iPtor)));
            content.Append(wxString::Format("Weight\t%e\t", GetPredictorWeight(iStep, iPtor)));
            if (!GetPreprocessMethod(iStep, iPtor).IsEmpty()) {
                content.Append(wxString::Format("%s\t", GetPreprocessMethod(iStep, iPtor)));
            } else {
                content.Append("NoPreprocessing\t");
            }
            content.Append(wxString::Format("Criteria\t%s\t", GetPredictorCriteria(iStep, iPtor)));
        }
    }

    return content;
}

bool asParameters::PrintAndSaveTemp(const wxString &filePath) const
{
    wxString saveFilePath;

    if (filePath.IsEmpty()) {
        saveFilePath = asConfig::GetTempDir() + "/AtmoSwingCurrentParameters.txt";
    } else {
        saveFilePath = filePath;
    }

    asFileAscii fileRes(saveFilePath, asFileAscii::Replace);
    if (!fileRes.Open())
        return false;

    wxString content = Print();

    wxString header;
    header = _("AtmoSwing current parameters, run ") + asTime::GetStringTime(asTime::NowMJD(asLOCAL));
    fileRes.AddLineContent(header);
    fileRes.AddLineContent(content);
    fileRes.Close();

    return true;
}

bool asParameters::GetValuesFromString(wxString stringVals)
{
    int iLeft, iRight;
    wxString strVal;
    double dVal;
    long lVal;

    wxString errMsg(_("Error when parsing the parameters file"));

    iLeft = stringVals.Find("DaysInt");
    iRight = stringVals.Find("||||");
    if (iLeft < 0 || iRight < 0) {
        wxLogError(errMsg);
        return false;
    }
    strVal = stringVals.SubString((size_t) iLeft + 8, (size_t) iRight - 2);
    if (!strVal.ToLong(&lVal)) {
        wxLogError(errMsg);
        return false;
    }
    if (!SetTimeArrayAnalogsIntervalDays(static_cast<int>(lVal))) {
        wxLogError(errMsg);
        return false;
    }
    stringVals = stringVals.SubString((size_t) iRight + 5, stringVals.Length());

    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        iLeft = stringVals.Find("Anb");
        iRight = stringVals.Find("||");
        if (iLeft < 0 || iRight < 0) {
            wxLogError(errMsg);
            return false;
        }
        strVal = stringVals.SubString((size_t) iLeft + 4, (size_t) iRight - 2);
        if (!strVal.ToLong(&lVal)) {
            wxLogError(errMsg);
            return false;
        }
        if (!SetAnalogsNumber(iStep, static_cast<int>(lVal))) {
            wxLogError(errMsg);
            return false;
        }
        stringVals = stringVals.SubString(iRight, stringVals.Length());

        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (NeedsPreprocessing(iStep, iPtor)) {
                for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); iPre++) {
                    iLeft = stringVals.Find("Level");
                    iRight = stringVals.Find("Time");
                    if (iLeft < 0 || iRight < 0) {
                        wxLogError(errMsg);
                        return false;
                    }
                    strVal = stringVals.SubString((size_t) iLeft + 6, (size_t) iRight - 2);
                    if (!strVal.ToDouble(&dVal)) {
                        wxLogError(errMsg);
                        return false;
                    }
                    if (!SetPreprocessLevel(iStep, iPtor, iPre, static_cast<float>(dVal))) {
                        wxLogError(errMsg);
                        return false;
                    }
                    stringVals = stringVals.SubString((size_t) iRight + 5, stringVals.Length());

                    iLeft = 0;
                    iRight = stringVals.Find("\t");
                    if (iLeft < 0 || iRight < 0) {
                        wxLogError(errMsg);
                        return false;
                    }
                    strVal = stringVals.SubString((size_t) iLeft, (size_t) iRight - 1);
                    if (!strVal.ToDouble(&dVal)) {
                        wxLogError(errMsg);
                        return false;
                    }
                    if (!SetPreprocessTimeHours(iStep, iPtor, iPre, static_cast<float>(dVal))) {
                        wxLogError(errMsg);
                        return false;
                    }
                    stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());
                }
            } else {
                iLeft = stringVals.Find("Level");
                iRight = stringVals.Find("Time");
                if (iLeft < 0 || iRight < 0) {
                    wxLogError(errMsg);
                    return false;
                }
                strVal = stringVals.SubString((size_t) iLeft + 6, (size_t) iRight - 2);
                if (!strVal.ToDouble(&dVal)) {
                    wxLogError(errMsg);
                    return false;
                }
                if (!SetPredictorLevel(iStep, iPtor, static_cast<float>(dVal))) {
                    wxLogError(errMsg);
                    return false;
                }
                stringVals = stringVals.SubString((size_t) iRight + 5, stringVals.Length());

                iLeft = 0;
                iRight = stringVals.Find("\t");
                if (iLeft < 0 || iRight < 0) {
                    wxLogError(errMsg);
                    return false;
                }
                strVal = stringVals.SubString((size_t) iLeft, (size_t) iRight - 1);
                if (!strVal.ToDouble(&dVal)) {
                    wxLogError(errMsg);
                    return false;
                }
                if (!SetPredictorTimeHours(iStep, iPtor, static_cast<float>(dVal))) {
                    wxLogError(errMsg);
                    return false;
                }
                stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());
            }

            iLeft = stringVals.Find("xMin");
            if (iLeft < 0)
                iLeft = stringVals.Find("Umin");
            iRight = stringVals.Find("xPtsNb");
            if (iRight < 0)
                iRight = stringVals.Find("Uptsnb");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 5, (size_t) iRight - 2);
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorXmin(iStep, iPtor, dVal)) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());

            iLeft = stringVals.Find("xPtsNb");
            if (iLeft < 0)
                iLeft = stringVals.Find("Uptsnb");
            iRight = stringVals.Find("xStep");
            if (iRight < 0)
                iRight = stringVals.Find("Ustep");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 7, (size_t) iRight - 2);
            if (!strVal.ToLong(&lVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorXptsnb(iStep, iPtor, static_cast<int>(lVal))) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("xStep");
            if (iLeft < 0)
                iLeft = stringVals.Find("Ustep");
            iRight = stringVals.Find("yMin");
            if (iRight < 0)
                iRight = stringVals.Find("Vmin");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 6, (size_t) iRight - 2);
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorXstep(iStep, iPtor, dVal)) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());

            iLeft = stringVals.Find("yMin");
            if (iLeft < 0)
                iLeft = stringVals.Find("Vmin");
            iRight = stringVals.Find("yPtsNb");
            if (iRight < 0)
                iRight = stringVals.Find("Vptsnb");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 5, (size_t) iRight - 2);
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorYmin(iStep, iPtor, dVal)) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());

            iLeft = stringVals.Find("yPtsNb");
            if (iLeft < 0)
                iLeft = stringVals.Find("Vptsnb");
            iRight = stringVals.Find("yStep");
            if (iRight < 0)
                iRight = stringVals.Find("Vstep");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 7, (size_t) iRight - 2);
            if (!strVal.ToLong(&lVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorYptsnb(iStep, iPtor, static_cast<int>(lVal))) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());

            iLeft = stringVals.Find("yStep");
            if (iLeft < 0)
                iLeft = stringVals.Find("Vstep");
            iRight = stringVals.Find("Weight");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 6, (size_t) iRight - 2);
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorYstep(iStep, iPtor, dVal)) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());

            iLeft = stringVals.Find("Weight");
            iRight = stringVals.Find("Criteria");
            if (iLeft < 0 || iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            strVal = stringVals.SubString((size_t) iLeft + 7, (size_t) iRight - 2);
            if (!strVal.ToDouble(&dVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (!SetPredictorWeight(iStep, iPtor, dVal)) {
                wxLogError(errMsg);
                return false;
            }
            if (iRight < 0) {
                wxLogError(errMsg);
                return false;
            }
            stringVals = stringVals.SubString((size_t) iRight, stringVals.Length());
        }
    }

    return true;
}

bool asParameters::SetTimeArrayTargetTimeStepHours(double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the target time step is null"));
        return false;
    }
    m_timeArrayTargetTimeStepHours = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsTimeStepHours(double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the analogs time step is null"));
        return false;
    }
    m_timeArrayAnalogsTimeStepHours = val;
    return true;
}

bool asParameters::SetTimeArrayTargetMode(const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the target time array mode is null"));
        return false;
    }
    m_timeArrayTargetMode = val;
    return true;
}

bool asParameters::SetTimeArrayTargetPredictandSerieName(const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictand serie name is null"));
        return false;
    }
    m_timeArrayTargetPredictandSerieName = val;
    return true;
}

bool asParameters::SetTimeArrayTargetPredictandMinThreshold(float val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictand min threshold is null"));
        return false;
    }
    m_timeArrayTargetPredictandMinThreshold = val;
    return true;
}

bool asParameters::SetTimeArrayTargetPredictandMaxThreshold(float val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictand max threshold is null"));
        return false;
    }
    m_timeArrayTargetPredictandMaxThreshold = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsMode(const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the analogy time array mode is null"));
        return false;
    }
    m_timeArrayAnalogsMode = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsExcludeDays(int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the 'exclude days' is null"));
        return false;
    }
    m_timeArrayAnalogsExcludeDays = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsIntervalDays(int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the analogs interval days is null"));
        return false;
    }
    m_timeArrayAnalogsIntervalDays = val;
    return true;
}

bool asParameters::SetPredictandStationIds(vi val)
{
    for (int v : val) {
        if (asIsNaN(v)) {
            wxLogError(_("The provided value for the predictand ID is null"));
            return false;
        }
    }
    m_predictandStationIds = val;
    return true;
}

bool asParameters::SetPredictandStationIds(wxString val)
{
    wxStringTokenizer tokenizer(val, ":,; ");
    while (tokenizer.HasMoreTokens()) {
        wxString token = tokenizer.GetNextToken();
        long stationId;
        if (token.ToLong(&stationId)) {
            m_predictandStationIds.push_back(static_cast<int>(stationId));
        }
    }
    return true;
}

bool asParameters::SetPredictandTimeHours(double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictand time (hours) is null"));
        return false;
    }
    m_predictandTimeHours = val;
    return true;
}

bool asParameters::SetAnalogsNumber(int iStep, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the analogs number is null"));
        return false;
    }
    m_steps[iStep].analogsNumber = val;
    return true;
}

bool asParameters::SetPreloadDataIds(int iStep, int iPtor, vwxs val)
{
    if (val.empty()) {
        wxLogError(_("The provided preload data IDs vector is empty."));
        return false;
    } else {
        for (auto &v : val) {
            if (v.IsEmpty()) {
                wxLogError(_("There are empty values in the provided preload data IDs vector."));
                return false;
            }
        }
    }
    m_steps[iStep].predictors[iPtor].preloadDataIds = val;
    return true;
}

bool asParameters::SetPreloadDataIds(int iStep, int iPtor, wxString val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided preload data id parameter is empty."));
        return false;
    }

    m_steps[iStep].predictors[iPtor].preloadDataIds.clear();
    m_steps[iStep].predictors[iPtor].preloadDataIds.push_back(val);

    return true;
}

bool asParameters::SetPreloadTimeHours(int iStep, int iPtor, vd val)
{
    if (val.empty()) {
        wxLogError(_("The provided preload time (hours) vector is empty."));
        return false;
    } else {
        for (double v : val) {
            if (asIsNaN(v)) {
                wxLogError(_("There are NaN values in the provided preload time (hours) vector."));
                return false;
            }
        }
    }
    m_steps[iStep].predictors[iPtor].preloadTimeHours = val;
    return true;
}

bool asParameters::SetPreloadTimeHours(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided preload time parameter is a NaN."));
        return false;
    }

    m_steps[iStep].predictors[iPtor].preloadTimeHours.clear();
    m_steps[iStep].predictors[iPtor].preloadTimeHours.push_back(val);
    return true;
}

bool asParameters::SetPreloadLevels(int iStep, int iPtor, vf val)
{
    if (val.empty()) {
        wxLogError(_("The provided 'preload levels' vector is empty."));
        return false;
    } else {
        for (float v : val) {
            if (asIsNaN(v)) {
                wxLogError(_("There are NaN values in the provided 'preload levels' vector."));
                return false;
            }
        }
    }
    m_steps[iStep].predictors[iPtor].preloadLevels = val;
    return true;
}

bool asParameters::SetPreloadLevels(int iStep, int iPtor, float val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided preload level parameter is a NaN."));
        return false;
    }

    m_steps[iStep].predictors[iPtor].preloadLevels.clear();
    m_steps[iStep].predictors[iPtor].preloadLevels.push_back(val);
    return true;
}

bool asParameters::SetPreloadXmin(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preload xMin is null"));
        return false;
    }
    m_steps[iStep].predictors[iPtor].preloadXmin = val;
    return true;
}

bool asParameters::SetPreloadXptsnb(int iStep, int iPtor, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preload points number on X is null"));
        return false;
    }
    m_steps[iStep].predictors[iPtor].preloadXptsnb = val;
    return true;
}

bool asParameters::SetPreloadYmin(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preload yMin is null"));
        return false;
    }
    m_steps[iStep].predictors[iPtor].preloadYmin = val;
    return true;
}

bool asParameters::SetPreloadYptsnb(int iStep, int iPtor, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preload points number on Y is null"));
        return false;
    }
    m_steps[iStep].predictors[iPtor].preloadYptsnb = val;
    return true;
}

bool asParameters::SetPreprocessMethod(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess method is null"));
        return false;
    }
    m_steps[iStep].predictors[iPtor].preprocessMethod = val;
    return true;
}

bool asParameters::NeedsGradientPreprocessing(int iStep, int iPtor) const
{
    wxString method = m_steps[iStep].predictors[iPtor].preprocessMethod;

    return method.IsSameAs("Gradients", false) ||
           method.IsSameAs("SimpleGradients", false) ||
           method.IsSameAs("RealGradients", false) ||
           method.IsSameAs("SimpleGradientsWithGaussianWeights", false) ||
           method.IsSameAs("RealGradientsWithGaussianWeights", false);
}

bool asParameters::IsCriteriaUsingGradients(int iStep, int iPtor) const
{
    if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1grads") ||
        GetPredictorCriteria(iStep, iPtor).IsSameAs("S2grads")) {
        return true;
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1grads") ||
               GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2grads")) {
        return true;
    }

    return false;
}

void asParameters::FixCriteriaIfGradientsPreprocessed(int iStep, int iPtor)
{
    if (NeedsGradientPreprocessing(iStep, iPtor)) {
        if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1") ||
             GetPredictorCriteria(iStep, iPtor).IsSameAs("S1r") ||
             GetPredictorCriteria(iStep, iPtor).IsSameAs("S1s") ||
             GetPredictorCriteria(iStep, iPtor).IsSameAs("S1G") ||
             GetPredictorCriteria(iStep, iPtor).IsSameAs("S1rG") ||
             GetPredictorCriteria(iStep, iPtor).IsSameAs("S1sG")) {
            SetPredictorCriteria(iStep, iPtor, "S1grads");
        } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1r") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1s") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1G") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1rG") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1sG")) {
            SetPredictorCriteria(iStep, iPtor, "NS1grads");
        } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2r") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2s") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2G") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2rG") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("S2sG")) {
            SetPredictorCriteria(iStep, iPtor, "S2grads");
        } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2r") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2s") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2G") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2rG") ||
                   GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2sG")) {
            SetPredictorCriteria(iStep, iPtor, "NS2grads");
        }
    }
}

void asParameters::ForceUsingGradientsPreprocessing(int iStep, int iPtor)
{
    // Gradients - S1
    if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1") || GetPredictorCriteria(iStep, iPtor).IsSameAs("S1r")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "RealGradients");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1s")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleGradients");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1G") || GetPredictorCriteria(iStep, iPtor).IsSameAs("S1rG")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "RealGradientsWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S1sG")) {
        SetPredictorCriteria(iStep, iPtor, "S1grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleGradientsWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1") || GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1r")) {
        SetPredictorCriteria(iStep, iPtor, "NS1grads");
        SetPreprocessMethod(iStep, iPtor, "RealGradients");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1s")) {
        SetPredictorCriteria(iStep, iPtor, "NS1grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleGradients");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1G") || GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1rG")) {
        SetPredictorCriteria(iStep, iPtor, "NS1grads");
        SetPreprocessMethod(iStep, iPtor, "RealGradientsWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS1sG")) {
        SetPredictorCriteria(iStep, iPtor, "NS1grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleGradientsWithGaussianWeights");
    }

    // Curvature - S2
    if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2") || GetPredictorCriteria(iStep, iPtor).IsSameAs("S2r")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "RealCurvature");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2s")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleCurvature");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2G") || GetPredictorCriteria(iStep, iPtor).IsSameAs("S2rG")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "RealCurvatureWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("S2sG")) {
        SetPredictorCriteria(iStep, iPtor, "S2grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleCurvatureWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2") || GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2r")) {
        SetPredictorCriteria(iStep, iPtor, "NS2grads");
        SetPreprocessMethod(iStep, iPtor, "RealCurvature");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2s")) {
        SetPredictorCriteria(iStep, iPtor, "NS2grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleCurvature");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2G") || GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2rG")) {
        SetPredictorCriteria(iStep, iPtor, "NS2grads");
        SetPreprocessMethod(iStep, iPtor, "RealCurvatureWithGaussianWeights");
    } else if (GetPredictorCriteria(iStep, iPtor).IsSameAs("NS2sG")) {
        SetPredictorCriteria(iStep, iPtor, "NS2grads");
        SetPreprocessMethod(iStep, iPtor, "SimpleCurvatureWithGaussianWeights");
    }
}

wxString asParameters::GetPreprocessDatasetId(int iStep, int iPtor, int iPre) const
{
    if (m_steps[iStep].predictors[iPtor].preprocessDatasetIds.size() >= (unsigned) (iPre + 1)) {
        return m_steps[iStep].predictors[iPtor].preprocessDatasetIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParameters::SetPreprocessDatasetId(int iStep, int iPtor, int iPre, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess dataset ID is null"));
        return false;
    }

    if (m_steps[iStep].predictors[iPtor].preprocessDatasetIds.size() >= (unsigned) (iPre + 1)) {
        m_steps[iStep].predictors[iPtor].preprocessDatasetIds[iPre] = val;
    } else {
        wxASSERT((int) m_steps[iStep].predictors[iPtor].preprocessDatasetIds.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessDatasetIds.push_back(val);
    }

    return true;
}

wxString asParameters::GetPreprocessDataId(int iStep, int iPtor, int iPre) const
{
    if (m_steps[iStep].predictors[iPtor].preprocessDataIds.size() >= (unsigned) (iPre + 1)) {
        return m_steps[iStep].predictors[iPtor].preprocessDataIds[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessDataIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParameters::SetPreprocessDataId(int iStep, int iPtor, int iPre, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the preprocess data ID is null"));
        return false;
    }

    if (m_steps[iStep].predictors[iPtor].preprocessDataIds.size() >= (unsigned) (iPre + 1)) {
        m_steps[iStep].predictors[iPtor].preprocessDataIds[iPre] = val;
    } else {
        wxASSERT((int) m_steps[iStep].predictors[iPtor].preprocessDataIds.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessDataIds.push_back(val);
    }

    return true;
}

float asParameters::GetPreprocessLevel(int iStep, int iPtor, int iPre) const
{
    if (m_steps[iStep].predictors[iPtor].preprocessLevels.size() >= (unsigned) (iPre + 1)) {
        return m_steps[iStep].predictors[iPtor].preprocessLevels[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessLevels in the parameters object."));
        return NaNf;
    }
}

bool asParameters::SetPreprocessLevel(int iStep, int iPtor, int iPre, float val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preprocess level is null"));
        return false;
    }

    if (m_steps[iStep].predictors[iPtor].preprocessLevels.size() >= (unsigned) (iPre + 1)) {
        m_steps[iStep].predictors[iPtor].preprocessLevels[iPre] = val;
    } else {
        wxASSERT((int) m_steps[iStep].predictors[iPtor].preprocessLevels.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessLevels.push_back(val);
    }

    return true;
}

double asParameters::GetPreprocessTimeHours(int iStep, int iPtor, int iPre) const
{
    if (m_steps[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
        return m_steps[iStep].predictors[iPtor].preprocessTimeHours[iPre];
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessTimeHours (std) in the parameters object."));
        return NaNd;
    }
}

bool asParameters::SetPreprocessTimeHours(int iStep, int iPtor, int iPre, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preprocess time (hours) is null"));
        return false;
    }

    if (m_steps[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
        m_steps[iStep].predictors[iPtor].preprocessTimeHours[iPre] = val;
    } else {
        wxASSERT((int) m_steps[iStep].predictors[iPtor].preprocessTimeHours.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessTimeHours.push_back(val);
    }

    return true;
}

int asParameters::GetPreprocessMembersNb(int iStep, int iPtor, int iPre) const
{
    if (m_steps[iStep].predictors[iPtor].preprocessMembersNb.size() >= (unsigned) (iPre + 1)) {
        return m_steps[iStep].predictors[iPtor].preprocessMembersNb[iPre];
    } else {
        wxLogWarning(_("Trying to access to an element outside of preprocessMembersNb (std) in the parameters object."));
        return 1;
    }
}

bool asParameters::SetPreprocessMembersNb(int iStep, int iPtor, int iPre, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the preprocess members is null"));
        return false;
    }

    if (m_steps[iStep].predictors[iPtor].preprocessMembersNb.size() >= (unsigned) (iPre + 1)) {
        m_steps[iStep].predictors[iPtor].preprocessMembersNb[iPre] = val;
    } else {
        wxASSERT((int) m_steps[iStep].predictors[iPtor].preprocessMembersNb.size() == iPre);
        m_steps[iStep].predictors[iPtor].preprocessMembersNb.push_back(val);
    }

    return true;
}

bool asParameters::SetPredictorDatasetId(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor dataset is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].datasetId = val;

    return true;
}

bool asParameters::SetPredictorDataId(int iStep, int iPtor, wxString val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor data is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].dataId = val;

    return true;
}

bool asParameters::SetPredictorLevel(int iStep, int iPtor, float val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor level is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].level = val;

    return true;
}

bool asParameters::SetPredictorGridType(int iStep, int iPtor, wxString val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor grid type is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].gridType = val;

    return true;
}

bool asParameters::SetPredictorXmin(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor xMin is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].xMin = val;

    return true;
}

bool asParameters::SetPredictorXptsnb(int iStep, int iPtor, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor points number on X is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].xPtsNb = val;

    return true;
}

bool asParameters::SetPredictorXstep(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor X step is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].xStep = val;

    return true;
}

bool asParameters::SetPredictorXshift(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor X shift is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].xShift = val;

    return true;
}

bool asParameters::SetPredictorYmin(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor yMin is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].yMin = val;

    return true;
}

bool asParameters::SetPredictorYptsnb(int iStep, int iPtor, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor points number on Y is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].yPtsNb = val;

    return true;
}

bool asParameters::SetPredictorYstep(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor Y step is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].yStep = val;

    return true;
}

bool asParameters::SetPredictorYshift(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor Y shift is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].yShift = val;

    return true;
}

bool asParameters::SetPredictorFlatAllowed(int iStep, int iPtor, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the 'flat allowed' property is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].flatAllowed = val;

    return true;
}

bool asParameters::SetPredictorTimeHours(int iStep, int iPtor, double val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor time (hours) is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].timeHours = val;

    return true;
}

bool asParameters::SetPredictorMembersNb(int iStep, int iPtor, int val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor members is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].membersNb = val;

    return true;
}

bool asParameters::SetPredictorCriteria(int iStep, int iPtor, const wxString &val)
{
    if (val.IsEmpty()) {
        wxLogError(_("The provided value for the predictor criteria is null"));
        return false;
    }
    m_steps[iStep].predictors[iPtor].criteria = val;
    return true;
}

bool asParameters::SetPredictorWeight(int iStep, int iPtor, float val)
{
    if (asIsNaN(val)) {
        wxLogError(_("The provided value for the predictor weight is null"));
        return false;
    }

    m_steps[iStep].predictors[iPtor].weight = val;

    return true;
}
