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

#include "asFileParametersStandard.h"
#include "asFileAscii.h"
#include <wx/tokenzr.h>


asParameters::asParameters()
{
    m_dateProcessed = asTime::GetStringTime(asTime::NowTimeStruct(asLOCAL));
    m_archiveStart = 0;
    m_archiveEnd = 0;
    m_timeMinHours = 0;
    m_timeMaxHours = 0;
    m_timeArrayTargetMode = "simple";
    m_timeArrayTargetTimeStepHours = 0;
    m_timeArrayAnalogsMode = "days_interval";
    m_timeArrayAnalogsTimeStepHours = 0;
    m_timeArrayAnalogsExcludeDays = 0;
    m_timeArrayAnalogsIntervalDays = 0;
    m_predictandTimeHours = 0;
    m_predictandParameter = (asDataPredictand::Parameter) 0;
    m_predictandTemporalResolution = (asDataPredictand::TemporalResolution) 0;
    m_predictandSpatialAggregation = (asDataPredictand::SpatialAggregation) 0;
    m_predictandDatasetId = wxEmptyString;
    m_timeArrayTargetPredictandMinThreshold = 0;
    m_timeArrayTargetPredictandMaxThreshold = 0;
}

asParameters::~asParameters()
{
    //dtor
}

void asParameters::AddStep()
{
    ParamsStep step;

    step.AnalogsNumber = 0;

    m_steps.push_back(step);
}

void asParameters::AddPredictor()
{
    AddPredictor(m_steps[m_steps.size() - 1]);
}

void asParameters::AddPredictor(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.DatasetId = wxEmptyString;
    predictor.DataId = wxEmptyString;
    predictor.Preload = false;
    predictor.PreloadXmin = 0;
    predictor.PreloadXptsnb = 0;
    predictor.PreloadYmin = 0;
    predictor.PreloadYptsnb = 0;
    predictor.Preprocess = false;
    predictor.PreprocessMethod = wxEmptyString;
    predictor.Level = 0;
    predictor.Xmin = 0;
    predictor.Xptsnb = 1;
    predictor.Xstep = 0;
    predictor.Xshift = 0;
    predictor.Ymin = 0;
    predictor.Yptsnb = 1;
    predictor.Ystep = 0;
    predictor.Yshift = 0;
    predictor.FlatAllowed = asFLAT_FORBIDDEN;
    predictor.TimeHours = 0;
    predictor.Criteria = wxEmptyString;
    predictor.Weight = 1;

    step.Predictors.push_back(predictor);
}

void asParameters::AddPredictor(int i_step)
{
    ParamsPredictor predictor;

    predictor.DatasetId = wxEmptyString;
    predictor.DataId = wxEmptyString;
    predictor.Preload = false;
    predictor.PreloadXmin = 0;
    predictor.PreloadXptsnb = 0;
    predictor.PreloadYmin = 0;
    predictor.PreloadYptsnb = 0;
    predictor.Preprocess = false;
    predictor.PreprocessMethod = wxEmptyString;
    predictor.Level = 0;
    predictor.GridType = "regular";
    predictor.Xmin = 0;
    predictor.Xptsnb = 1;
    predictor.Xstep = 0;
    predictor.Xshift = 0;
    predictor.Ymin = 0;
    predictor.Yptsnb = 1;
    predictor.Ystep = 0;
    predictor.Yshift = 0;
    predictor.FlatAllowed = asFLAT_FORBIDDEN;
    predictor.TimeHours = 0;
    predictor.Criteria = wxEmptyString;
    predictor.Weight = 1;

    m_steps[i_step].Predictors.push_back(predictor);
}

bool asParameters::LoadFromFile(const wxString &filePath)
{
    asLogMessage(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersStandard fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open())
        return false;

    if (!fileParams.CheckRootElement())
        return false;

    int i_step = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {

        // Description
        if (nodeProcess->GetName() == "description") {
            wxXmlNode *nodeParam = nodeProcess->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "method_id") {
                    SetMethodId(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "method_id_display") {
                    SetMethodIdDisplay(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "specific_tag") {
                    SetSpecificTag(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "specific_tag_display") {
                    SetSpecificTagDisplay(fileParams.GetString(nodeParam));
                } else if (nodeParam->GetName() == "description") {
                    SetDescription(fileParams.GetString(nodeParam));
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }

            // Time properties
        } else if (nodeProcess->GetName() == "time_properties") {
            wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
            while (nodeParamBlock) {
                if (nodeParamBlock->GetName() == "archive_period") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "start_year") {
                            if (!SetArchiveYearStart(fileParams.GetInt(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "end_year") {
                            if (!SetArchiveYearEnd(fileParams.GetInt(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "start") {
                            if (!SetArchiveStart(fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "end") {
                            if (!SetArchiveEnd(fileParams.GetString(nodeParam)))
                                return false;
                        } else {
                            fileParams.UnknownNode(nodeParam);
                        }
                        nodeParam = nodeParam->GetNext();
                    }
                } else if (nodeParamBlock->GetName() == "time_step") {
                    if (!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock)))
                        return false;
                    if (!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock)))
                        return false;
                } else if (nodeParamBlock->GetName() == "time_array_target") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "time_array") {
                            if (!SetTimeArrayTargetMode(fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "predictand_serie_name") {
                            if (!SetTimeArrayTargetPredictandSerieName(fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "predictand_min_threshold") {
                            if (!SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFloat(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "predictand_max_threshold") {
                            if (!SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFloat(nodeParam)))
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
                            if (!SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "interval_days") {
                            if (!SetTimeArrayAnalogsIntervalDays(fileParams.GetInt(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "exclude_days") {
                            if (!SetTimeArrayAnalogsExcludeDays(fileParams.GetInt(nodeParam)))
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

            // Analog dates
        } else if (nodeProcess->GetName() == "analog_dates") {
            AddStep();
            int i_ptor = 0;
            wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
            while (nodeParamBlock) {
                if (nodeParamBlock->GetName() == "analogs_number") {
                    if (!SetAnalogsNumber(i_step, fileParams.GetInt(nodeParamBlock)))
                        return false;
                } else if (nodeParamBlock->GetName() == "predictor") {
                    AddPredictor(i_step);
                    SetPreprocess(i_step, i_ptor, false);
                    SetPreload(i_step, i_ptor, false);
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "preload") {
                            SetPreload(i_step, i_ptor, fileParams.GetBool(nodeParam));
                        } else if (nodeParam->GetName() == "preprocessing") {
                            SetPreprocess(i_step, i_ptor, true);
                            int i_dataset = 0;
                            wxXmlNode *nodePreprocess = nodeParam->GetChildren();
                            while (nodePreprocess) {
                                if (nodePreprocess->GetName() == "preprocessing_method") {
                                    if (!SetPreprocessMethod(i_step, i_ptor, fileParams.GetString(nodePreprocess)))
                                        return false;
                                } else if (nodePreprocess->GetName() == "preprocessing_data") {
                                    wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
                                    while (nodeParamPreprocess) {
                                        if (nodeParamPreprocess->GetName() == "dataset_id") {
                                            if (!SetPreprocessDatasetId(i_step, i_ptor, i_dataset,
                                                                        fileParams.GetString(nodeParamPreprocess)))
                                                return false;
                                        } else if (nodeParamPreprocess->GetName() == "data_id") {
                                            if (!SetPreprocessDataId(i_step, i_ptor, i_dataset,
                                                                     fileParams.GetString(nodeParamPreprocess)))
                                                return false;
                                        } else if (nodeParamPreprocess->GetName() == "level") {
                                            if (!SetPreprocessLevel(i_step, i_ptor, i_dataset,
                                                                    fileParams.GetFloat(nodeParamPreprocess)))
                                                return false;
                                        } else if (nodeParamPreprocess->GetName() == "time") {
                                            if (!SetPreprocessTimeHours(i_step, i_ptor, i_dataset,
                                                                        fileParams.GetDouble(nodeParamPreprocess)))
                                                return false;
                                        } else {
                                            fileParams.UnknownNode(nodeParamPreprocess);
                                        }
                                        nodeParamPreprocess = nodeParamPreprocess->GetNext();
                                    }
                                    i_dataset++;
                                } else {
                                    fileParams.UnknownNode(nodePreprocess);
                                }
                                nodePreprocess = nodePreprocess->GetNext();
                            }
                        } else if (nodeParam->GetName() == "dataset_id") {
                            if (!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "data_id") {
                            if (!SetPredictorDataId(i_step, i_ptor, fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "level") {
                            if (!SetPredictorLevel(i_step, i_ptor, fileParams.GetFloat(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "time") {
                            if (!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetDouble(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "spatial_window") {
                            wxXmlNode *nodeWindow = nodeParam->GetChildren();
                            while (nodeWindow) {
                                if (nodeWindow->GetName() == "grid_type") {
                                    if (!SetPredictorGridType(i_step, i_ptor,
                                                              fileParams.GetString(nodeWindow, "regular")))
                                        return false;
                                } else if (nodeWindow->GetName() == "x_min") {
                                    if (!SetPredictorXmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                        return false;
                                } else if (nodeWindow->GetName() == "x_points_nb") {
                                    if (!SetPredictorXptsnb(i_step, i_ptor, fileParams.GetInt(nodeWindow)))
                                        return false;
                                } else if (nodeWindow->GetName() == "x_step") {
                                    if (!SetPredictorXstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                        return false;
                                } else if (nodeWindow->GetName() == "y_min") {
                                    if (!SetPredictorYmin(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                        return false;
                                } else if (nodeWindow->GetName() == "y_points_nb") {
                                    if (!SetPredictorYptsnb(i_step, i_ptor, fileParams.GetInt(nodeWindow)))
                                        return false;
                                } else if (nodeWindow->GetName() == "y_step") {
                                    if (!SetPredictorYstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow)))
                                        return false;
                                } else {
                                    fileParams.UnknownNode(nodeWindow);
                                }
                                nodeWindow = nodeWindow->GetNext();
                            }
                        } else if (nodeParam->GetName() == "criteria") {
                            if (!SetPredictorCriteria(i_step, i_ptor, fileParams.GetString(nodeParam)))
                                return false;
                        } else if (nodeParam->GetName() == "weight") {
                            if (!SetPredictorWeight(i_step, i_ptor, fileParams.GetFloat(nodeParam)))
                                return false;
                        } else {
                            fileParams.UnknownNode(nodeParam);
                        }
                        nodeParam = nodeParam->GetNext();
                    }
                    i_ptor++;
                } else {
                    fileParams.UnknownNode(nodeParamBlock);
                }
                nodeParamBlock = nodeParamBlock->GetNext();
            }
            i_step++;

            // Analog values
        } else if (nodeProcess->GetName() == "analog_values") {
            wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
            while (nodeParamBlock) {
                if (nodeParamBlock->GetName() == "predictand") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "station_id") {
                            if (!SetPredictandStationIds(fileParams.GetStationIds(fileParams.GetString(nodeParam))))
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

    asLogMessage(_("Parameters file loaded."));

    return true;
}

bool asParameters::SetSpatialWindowProperties()
{
    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        for (int i_ptor = 0; i_ptor < GetPredictorsNb(i_step); i_ptor++) {
            if (GetPredictorXptsnb(i_step, i_ptor) == 0)
                SetPredictorXptsnb(i_step, i_ptor, 1);
            if (GetPredictorYptsnb(i_step, i_ptor) == 0)
                SetPredictorYptsnb(i_step, i_ptor, 1);

            double Xshift = std::fmod(GetPredictorXmin(i_step, i_ptor), GetPredictorXstep(i_step, i_ptor));
            if (Xshift < 0)
                Xshift += GetPredictorXstep(i_step, i_ptor);
            if (!SetPredictorXshift(i_step, i_ptor, Xshift))
                return false;

            double Yshift = std::fmod(GetPredictorYmin(i_step, i_ptor), GetPredictorYstep(i_step, i_ptor));
            if (Yshift < 0)
                Yshift += GetPredictorYstep(i_step, i_ptor);
            if (!SetPredictorYshift(i_step, i_ptor, Yshift))
                return false;

            if (GetPredictorXptsnb(i_step, i_ptor) == 1 || GetPredictorYptsnb(i_step, i_ptor) == 1) {
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            }
        }
    }

    return true;
}

bool asParameters::SetPreloadingProperties()
{
    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        for (int i_ptor = 0; i_ptor < GetPredictorsNb(i_step); i_ptor++) {
            // Set maximum extent
            if (NeedsPreloading(i_step, i_ptor)) {
                if (!SetPreloadXmin(i_step, i_ptor, GetPredictorXmin(i_step, i_ptor)))
                    return false;
                if (!SetPreloadYmin(i_step, i_ptor, GetPredictorYmin(i_step, i_ptor)))
                    return false;
                if (!SetPreloadXptsnb(i_step, i_ptor, GetPredictorXptsnb(i_step, i_ptor)))
                    return false;
                if (!SetPreloadYptsnb(i_step, i_ptor, GetPredictorYptsnb(i_step, i_ptor)))
                    return false;
            }

            // Change predictor properties when preprocessing
            if (NeedsPreprocessing(i_step, i_ptor)) {
                if (GetPreprocessSize(i_step, i_ptor) == 1) {
                    SetPredictorDatasetId(i_step, i_ptor, GetPreprocessDatasetId(i_step, i_ptor, 0));
                    SetPredictorDataId(i_step, i_ptor, GetPreprocessDataId(i_step, i_ptor, 0));
                    SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
                    SetPredictorTimeHours(i_step, i_ptor, GetPreprocessTimeHours(i_step, i_ptor, 0));
                } else {
                    SetPredictorDatasetId(i_step, i_ptor, "mix");
                    SetPredictorDataId(i_step, i_ptor, "mix");
                    SetPredictorLevel(i_step, i_ptor, 0);
                    SetPredictorTimeHours(i_step, i_ptor, 0);
                }
            }

            // Set levels and time for preloading
            if (NeedsPreloading(i_step, i_ptor) && !NeedsPreprocessing(i_step, i_ptor)) {
                if (!SetPreloadDataIds(i_step, i_ptor, GetPredictorDataId(i_step, i_ptor)))
                    return false;
                if (!SetPreloadLevels(i_step, i_ptor, GetPredictorLevel(i_step, i_ptor)))
                    return false;
                if (!SetPreloadTimeHours(i_step, i_ptor, GetPredictorTimeHours(i_step, i_ptor)))
                    return false;
            } else if (NeedsPreloading(i_step, i_ptor) && NeedsPreprocessing(i_step, i_ptor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(i_step, i_ptor);
                VectorFloat preprocLevels;
                VectorDouble preprocTimeHours;
                int preprocSize = GetPreprocessSize(i_step, i_ptor);

                // Different actions depending on the preprocessing method.
                if (method.IsSameAs("Gradients")) {
                    if (preprocSize != 1) {
                        asLogError(wxString::Format(
                                _("The size of the provided predictors (%d) does not match the requirements (1) in the preprocessing Gradients method."),
                                preprocSize));
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(i_step, i_ptor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(i_step, i_ptor, 0));
                } else if (method.IsSameAs("HumidityFlux")) {
                    if (preprocSize != 4) {
                        asLogError(wxString::Format(
                                _("The size of the provided predictors (%d) does not match the requirements (4) in the preprocessing HumidityFlux method."),
                                preprocSize));
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(i_step, i_ptor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(i_step, i_ptor, 0));
                } else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                           method.IsSameAs("HumidityIndex")) {
                    if (preprocSize != 2) {
                        asLogError(wxString::Format(
                                _("The size of the provided predictors (%d) does not match the requirements (2) in the preprocessing Multiply method."),
                                preprocSize));
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(i_step, i_ptor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(i_step, i_ptor, 0));
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    if (preprocSize != 4) {
                        asLogError(wxString::Format(
                                _("The size of the provided predictors (%d) does not match the requirements (4) in the preprocessing FormerHumidityIndex method."),
                                preprocSize));
                        return false;
                    }
                    preprocLevels.push_back(GetPreprocessLevel(i_step, i_ptor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(i_step, i_ptor, 0));
                    preprocTimeHours.push_back(GetPreprocessTimeHours(i_step, i_ptor, 1));
                } else {
                    asLogWarning(wxString::Format(
                            _("The %s preprocessing method is not yet handled with the preload option."), method));
                }

                if (!SetPreloadLevels(i_step, i_ptor, preprocLevels))
                    return false;
                if (!SetPreloadTimeHours(i_step, i_ptor, preprocTimeHours))
                    return false;
            }
        }
    }

    return true;
}

bool asParameters::InputsOK() const
{
    // Time properties
    if (GetArchiveStart() <= 0) {
        asLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetArchiveEnd() <= 0) {
        asLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayTargetTimeStepHours() <= 0) {
        asLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayAnalogsTimeStepHours() <= 0) {
        asLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if (GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds") == 0 ||
        GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds") == 0) {
        if (GetTimeArrayTargetPredictandSerieName().IsEmpty()) {
            asLogError(
                    _("The predictand time series (for the threshold preselection) was not provided in the parameters file."));
            return false;
        }
        if (GetTimeArrayTargetPredictandMinThreshold() == GetTimeArrayTargetPredictandMaxThreshold()) {
            asLogError(_("The provided min/max predictand thresholds are equal in the parameters file."));
            return false;
        }
    }

    if (GetTimeArrayAnalogsMode().CmpNoCase("interval_days") == 0 ||
        GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays") == 0) {
        if (GetTimeArrayAnalogsIntervalDays() <= 0) {
            asLogError(_("The interval days for the analogs preselection was not provided in the parameters file."));
            return false;
        }
        if (GetTimeArrayAnalogsExcludeDays() <= 0) {
            asLogError(
                    _("The number of days to exclude around the target date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for (int i = 0; i < GetStepsNb(); i++) {
        if (GetAnalogsNumber(i) <= 0) {
            asLogError(
                    wxString::Format(_("The number of analogs (step %d) was not provided in the parameters file."), i));
            return false;
        }

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                if (GetPreprocessMethod(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The preprocessing method (step %d, predictor %d) was not provided in the parameters file."),
                            i, j));
                    return false;
                }

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    if (GetPreprocessDatasetId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(
                                _("The dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                i, j));
                        return false;
                    }
                    if (GetPreprocessDataId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(
                                _("The data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                i, j));
                        return false;
                    }
                }
            } else {
                if (GetPredictorDatasetId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The dataset (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }
                if (GetPredictorDataId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(
                            _("The data (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                asLogError(
                        wxString::Format(_("The grid type (step %d, predictor %d) is empty in the parameters file."), i,
                                         j));
                return false;
            }
            if (GetPredictorXptsnb(i, j) == 0) {
                asLogError(wxString::Format(
                        _("The X points nb value (step %d, predictor %d) was not provided in the parameters file."), i,
                        j));
                return false;
            }
            if (GetPredictorYptsnb(i, j) == 0) {
                asLogError(wxString::Format(
                        _("The Y points nb value (step %d, predictor %d) was not provided in the parameters file."), i,
                        j));
                return false;
            }
            if (GetPredictorCriteria(i, j).IsEmpty()) {
                asLogError(wxString::Format(
                        _("The criteria (step %d, predictor %d) was not provided in the parameters file."), i, j));
                return false;
            }
        }
    }

    return true;
}

bool asParameters::FixAnalogsNb()
{
    // Check analogs number coherence
    int analogsNb = GetAnalogsNumber(0);
    for (unsigned int i_step = 1; i_step < m_steps.size(); i_step++) {
        if (GetAnalogsNumber(i_step) > analogsNb) {
            SetAnalogsNumber(i_step, analogsNb);
        } else {
            analogsNb = GetAnalogsNumber(i_step);
        }
    }

    return true;
}

void asParameters::SortLevelsAndTime()
{
    // Sort levels on every analogy level
    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        // Get the predictors vector
        VectorParamsPredictors oldPtors = GetVectorParamsPredictors(i_step);
        VectorParamsPredictors newPtors;

        // Sort
        while (true) {
            if (oldPtors.size() == 0) {
                break;
            }

            // Find the smallest level and hour combination
            int lowestIndex = 0;
            float level;
            double hour;
            if (oldPtors[0].Preprocess) {
                level = oldPtors[0].PreprocessLevels[0];
                hour = oldPtors[0].PreprocessTimeHours[0];
            } else {
                level = oldPtors[0].Level;
                hour = oldPtors[0].TimeHours;
            }

            for (unsigned int i = 1; i < oldPtors.size(); i++) {
                // Get next level and hour
                float nextLevel;
                double nextHour;
                if (oldPtors[i].Preprocess) {
                    nextLevel = oldPtors[i].PreprocessLevels[0];
                    nextHour = oldPtors[i].PreprocessTimeHours[0];
                } else {
                    nextLevel = oldPtors[i].Level;
                    nextHour = oldPtors[i].TimeHours;
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
            SetVectorParamsPredictors(i_step, newPtors);
        }
    }
}

VectorInt asParameters::GetFileStationIds(wxString stationIdsString)
{
    // Trim
    stationIdsString.Trim(true);
    stationIdsString.Trim(false);

    VectorInt ids;

    if (stationIdsString.IsEmpty()) {
        asLogError(_("The station ID was not provided."));
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
            asLogError(_("The format of the station ID is not correct (more than one opening bracket)."));
            return ids;
        }

        // Check that it contains 1 closing bracket at the end
        if (subStr.Find(")") != subStr.size() - 1 && subStr.Find(")'") != subStr.size() - 2) {
            asLogError(_("The format of the station ID is not correct (location of the closing bracket)."));
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
            asLogError(_("The format of the station ID is not correct (should be only digits)."));
            return ids;
        }
        int id = wxAtoi(stationIdsString);
        ids.push_back(id);
    }

    return ids;
}

wxString asParameters::GetPredictandStationIdsString() const
{
    wxString Ids;

    if (m_predictandStationIds.size() == 1) {
        Ids << m_predictandStationIds[0];
    } else {
        Ids = "(";

        for (int i = 0; i < (int) m_predictandStationIds.size(); i++) {
            Ids << m_predictandStationIds[i];

            if (i < (int) m_predictandStationIds.size() - 1) {
                Ids.Append(",");
            }
        }

        Ids.Append(")");
    }

    return Ids;
}

bool asParameters::FixTimeLimits()
{
    double minHour = 1000.0, maxHour = -1000.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                double minHourPredictor = 1000.0, maxHourPredictor = -1000.0;

                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(m_steps[i].Predictors[j].PreprocessTimeHours[k], minHour);
                    maxHour = wxMax(m_steps[i].Predictors[j].PreprocessTimeHours[k], maxHour);
                    minHourPredictor = wxMin(m_steps[i].Predictors[j].PreprocessTimeHours[k], minHourPredictor);
                    maxHourPredictor = wxMax(m_steps[i].Predictors[j].PreprocessTimeHours[k], maxHourPredictor);
                    m_steps[i].Predictors[j].TimeHours = minHourPredictor;
                }
            } else {
                minHour = wxMin(m_steps[i].Predictors[j].TimeHours, minHour);
                maxHour = wxMax(m_steps[i].Predictors[j].TimeHours, maxHour);
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
            totWeight += m_steps[i].Predictors[j].Weight;
        }

        // Correct to set the total to 1
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            m_steps[i].Predictors[j].Weight /= totWeight;
        }
    }

    return true;
}

bool asParameters::FixCoordinates()
{
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (m_steps[i].Predictors[j].GridType.IsSameAs("regular", false)) {

                // Check that the coordinates are a multiple of the steps
                if (std::abs(std::fmod(m_steps[i].Predictors[j].Xmin - m_steps[i].Predictors[j].Xshift,
                                       m_steps[i].Predictors[j].Xstep)) > 0) {
                    double factor = (m_steps[i].Predictors[j].Xmin - m_steps[i].Predictors[j].Xshift) /
                                    m_steps[i].Predictors[j].Xstep;
                    factor = asTools::Round(factor);
                    m_steps[i].Predictors[j].Xmin =
                            factor * m_steps[i].Predictors[j].Xstep + m_steps[i].Predictors[j].Xshift;
                }

                if (std::abs(std::fmod(m_steps[i].Predictors[j].Ymin - m_steps[i].Predictors[j].Yshift,
                                       m_steps[i].Predictors[j].Ystep)) > 0) {
                    double factor = (m_steps[i].Predictors[j].Ymin - m_steps[i].Predictors[j].Yshift) /
                                    m_steps[i].Predictors[j].Ystep;
                    factor = asTools::Round(factor);
                    m_steps[i].Predictors[j].Ymin =
                            factor * m_steps[i].Predictors[j].Ystep + m_steps[i].Predictors[j].Yshift;
                }
            }

            if (m_steps[i].Predictors[j].FlatAllowed == asFLAT_FORBIDDEN) {
                // Check that the size is larger than 1 point
                if (m_steps[i].Predictors[j].Xptsnb < 2) {
                    m_steps[i].Predictors[j].Xptsnb = 2;
                }

                if (m_steps[i].Predictors[j].Yptsnb < 2) {
                    m_steps[i].Predictors[j].Yptsnb = 2;
                }
            } else {
                // Check that the size is larger than 0
                if (m_steps[i].Predictors[j].Xptsnb < 1) {
                    m_steps[i].Predictors[j].Xptsnb = 1;
                }

                if (m_steps[i].Predictors[j].Yptsnb < 1) {
                    m_steps[i].Predictors[j].Yptsnb = 1;
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

    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        content.Append(wxString::Format("|||| Step(%d)\t", i_step));
        content.Append(wxString::Format("Anb\t%d\t", GetAnalogsNumber(i_step)));

        for (int i_ptor = 0; i_ptor < GetPredictorsNb(i_step); i_ptor++) {
            content.Append(wxString::Format("|| Ptor(%d)\t", i_ptor));

            if (NeedsPreprocessing(i_step, i_ptor)) {
                content.Append(wxString::Format("%s\t", GetPreprocessMethod(i_step, i_ptor)));

                for (int i_dataset = 0; i_dataset < GetPreprocessSize(i_step, i_ptor); i_dataset++) {
                    content.Append(wxString::Format("| %s %s\t", GetPreprocessDatasetId(i_step, i_ptor, i_dataset),
                                                    GetPreprocessDataId(i_step, i_ptor, i_dataset)));
                    content.Append(wxString::Format("Level\t%g\t", GetPreprocessLevel(i_step, i_ptor, i_dataset)));
                    content.Append(wxString::Format("Time\t%g\t", GetPreprocessTimeHours(i_step, i_ptor, i_dataset)));
                }
            } else {
                content.Append(wxString::Format("%s %s\t", GetPredictorDatasetId(i_step, i_ptor),
                                                GetPredictorDataId(i_step, i_ptor)));
                content.Append(wxString::Format("Level\t%g\t", GetPredictorLevel(i_step, i_ptor)));
                content.Append(wxString::Format("Time\t%g\t", GetPredictorTimeHours(i_step, i_ptor)));
            }

            content.Append(wxString::Format("GridType\t%s\t", GetPredictorGridType(i_step, i_ptor)));
            content.Append(wxString::Format("Xmin\t%g\t", GetPredictorXmin(i_step, i_ptor)));
            content.Append(wxString::Format("Xptsnb\t%d\t", GetPredictorXptsnb(i_step, i_ptor)));
            content.Append(wxString::Format("Xstep\t%g\t", GetPredictorXstep(i_step, i_ptor)));
            content.Append(wxString::Format("Ymin\t%g\t", GetPredictorYmin(i_step, i_ptor)));
            content.Append(wxString::Format("Yptsnb\t%d\t", GetPredictorYptsnb(i_step, i_ptor)));
            content.Append(wxString::Format("Ystep\t%g\t", GetPredictorYstep(i_step, i_ptor)));
            content.Append(wxString::Format("Weight\t%e\t", GetPredictorWeight(i_step, i_ptor)));
            content.Append(wxString::Format("Criteria\t%s\t", GetPredictorCriteria(i_step, i_ptor)));
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
    size_t iLeft, iRight;
    wxString strVal;
    double dVal;
    long lVal;

    iLeft = stringVals.Find("DaysInt");
    iRight = stringVals.Find("||||");
    strVal = stringVals.SubString(iLeft + 8, iRight - 2);
    strVal.ToLong(&lVal);
    SetTimeArrayAnalogsIntervalDays(int(lVal));
    stringVals = stringVals.SubString(iRight + 5, stringVals.Length());

    for (int i_step = 0; i_step < GetStepsNb(); i_step++) {
        iLeft = stringVals.Find("Anb");
        iRight = stringVals.Find("||");
        strVal = stringVals.SubString(iLeft + 4, iRight - 2);
        strVal.ToLong(&lVal);
        SetAnalogsNumber(i_step, int(lVal));
        stringVals = stringVals.SubString(iRight, stringVals.Length());

        for (int i_ptor = 0; i_ptor < GetPredictorsNb(i_step); i_ptor++) {
            if (NeedsPreprocessing(i_step, i_ptor)) {
                for (int i_dataset = 0; i_dataset < GetPreprocessSize(i_step, i_ptor); i_dataset++) {
                    iLeft = stringVals.Find("Level");
                    iRight = stringVals.Find("Time");
                    strVal = stringVals.SubString(iLeft + 6, iRight - 2);
                    strVal.ToDouble(&dVal);
                    SetPreprocessLevel(i_step, i_ptor, i_dataset, float(dVal));
                    stringVals = stringVals.SubString(iRight + 5, stringVals.Length());

                    iLeft = 0;
                    iRight = stringVals.Find("\t");
                    strVal = stringVals.SubString(iLeft, iRight - 1);
                    strVal.ToDouble(&dVal);
                    SetPreprocessTimeHours(i_step, i_ptor, i_dataset, float(dVal));
                    stringVals = stringVals.SubString(iRight, stringVals.Length());
                }
            } else {
                iLeft = stringVals.Find("Level");
                iRight = stringVals.Find("Time");
                strVal = stringVals.SubString(iLeft + 6, iRight - 2);
                strVal.ToDouble(&dVal);
                SetPredictorLevel(i_step, i_ptor, float(dVal));
                stringVals = stringVals.SubString(iRight + 5, stringVals.Length());

                iLeft = 0;
                iRight = stringVals.Find("\t");
                strVal = stringVals.SubString(iLeft, iRight - 1);
                strVal.ToDouble(&dVal);
                SetPredictorTimeHours(i_step, i_ptor, float(dVal));
                stringVals = stringVals.SubString(iRight, stringVals.Length());
            }

            iLeft = stringVals.Find("Xmin");
            if (iLeft < 0)
                iLeft = stringVals.Find("Umin");
            iRight = stringVals.Find("Xptsnb");
            if (iRight < 0)
                iRight = stringVals.Find("Uptsnb");
            strVal = stringVals.SubString(iLeft + 5, iRight - 2);
            strVal.ToDouble(&dVal);
            SetPredictorXmin(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Xptsnb");
            if (iLeft < 0)
                iLeft = stringVals.Find("Uptsnb");
            iRight = stringVals.Find("Xstep");
            if (iRight < 0)
                iRight = stringVals.Find("Ustep");
            strVal = stringVals.SubString(iLeft + 7, iRight - 2);
            strVal.ToLong(&lVal);
            SetPredictorXptsnb(i_step, i_ptor, int(lVal));
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Xstep");
            if (iLeft < 0)
                iLeft = stringVals.Find("Ustep");
            iRight = stringVals.Find("Ymin");
            if (iRight < 0)
                iRight = stringVals.Find("Vmin");
            strVal = stringVals.SubString(iLeft + 6, iRight - 2);
            strVal.ToDouble(&dVal);
            SetPredictorXstep(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Ymin");
            if (iLeft < 0)
                iLeft = stringVals.Find("Vmin");
            iRight = stringVals.Find("Yptsnb");
            if (iRight < 0)
                iRight = stringVals.Find("Vptsnb");
            strVal = stringVals.SubString(iLeft + 5, iRight - 2);
            strVal.ToDouble(&dVal);
            SetPredictorYmin(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Yptsnb");
            if (iLeft < 0)
                iLeft = stringVals.Find("Vptsnb");
            iRight = stringVals.Find("Ystep");
            if (iRight < 0)
                iRight = stringVals.Find("Vstep");
            strVal = stringVals.SubString(iLeft + 7, iRight - 2);
            strVal.ToLong(&lVal);
            SetPredictorYptsnb(i_step, i_ptor, int(lVal));
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Ystep");
            if (iLeft < 0)
                iLeft = stringVals.Find("Vstep");
            iRight = stringVals.Find("Weight");
            strVal = stringVals.SubString(iLeft + 6, iRight - 2);
            strVal.ToDouble(&dVal);
            SetPredictorYstep(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Weight");
            iRight = stringVals.Find("Criteria");
            strVal = stringVals.SubString(iLeft + 7, iRight - 2);
            strVal.ToDouble(&dVal);
            SetPredictorWeight(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());
        }
    }

    return true;
}

bool asParameters::SetTimeArrayTargetTimeStepHours(double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the target time step is null"));
        return false;
    }
    m_timeArrayTargetTimeStepHours = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsTimeStepHours(double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the analogs time step is null"));
        return false;
    }
    m_timeArrayAnalogsTimeStepHours = val;
    return true;
}

bool asParameters::SetTimeArrayTargetMode(const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the target time array mode is null"));
        return false;
    }
    m_timeArrayTargetMode = val;
    return true;
}

bool asParameters::SetTimeArrayTargetPredictandSerieName(const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictand serie name is null"));
        return false;
    }
    m_timeArrayTargetPredictandSerieName = val;
    return true;
}

bool asParameters::SetTimeArrayTargetPredictandMinThreshold(float val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictand min threshold is null"));
        return false;
    }
    m_timeArrayTargetPredictandMinThreshold = val;
    return true;
}

bool asParameters::SetTimeArrayTargetPredictandMaxThreshold(float val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictand max threshold is null"));
        return false;
    }
    m_timeArrayTargetPredictandMaxThreshold = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsMode(const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the analogy time array mode is null"));
        return false;
    }
    m_timeArrayAnalogsMode = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsExcludeDays(int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the 'exclude days' is null"));
        return false;
    }
    m_timeArrayAnalogsExcludeDays = val;
    return true;
}

bool asParameters::SetTimeArrayAnalogsIntervalDays(int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the analogs interval days is null"));
        return false;
    }
    m_timeArrayAnalogsIntervalDays = val;
    return true;
}

bool asParameters::SetPredictandStationIds(VectorInt val)
{
    for (int i = 0; i < (int) val.size(); i++) {
        if (asTools::IsNaN(val[i])) {
            asLogError(_("The provided value for the predictand ID is null"));
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
        token.ToLong(&stationId);
        m_predictandStationIds.push_back(stationId);
    }
    return true;
}

bool asParameters::SetPredictandDatasetId(const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictand dataset ID is null"));
        return false;
    }
    m_predictandDatasetId = val;
    return true;
}

bool asParameters::SetPredictandTimeHours(double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictand time (hours) is null"));
        return false;
    }
    m_predictandTimeHours = val;
    return true;
}

bool asParameters::SetAnalogsNumber(int i_step, int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the analogs number is null"));
        return false;
    }
    m_steps[i_step].AnalogsNumber = val;
    return true;
}

bool asParameters::SetPreloadDataIds(int i_step, int i_predictor, VectorString val)
{
    if (val.size() < 1) {
        asLogError(_("The provided preload data IDs vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (val[i].IsEmpty()) {
                asLogError(_("There are empty values in the provided preload data IDs vector."));
                return false;
            }
        }
    }
    m_steps[i_step].Predictors[i_predictor].PreloadDataIds = val;
    return true;
}

bool asParameters::SetPreloadDataIds(int i_step, int i_predictor, wxString val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided preload data id parameter is empty."));
        return false;
    }

    m_steps[i_step].Predictors[i_predictor].PreloadDataIds.clear();
    m_steps[i_step].Predictors[i_predictor].PreloadDataIds.push_back(val);

    return true;
}

bool asParameters::SetPreloadTimeHours(int i_step, int i_predictor, VectorDouble val)
{
    if (val.size() < 1) {
        asLogError(_("The provided preload time (hours) vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                asLogError(_("There are NaN values in the provided preload time (hours) vector."));
                return false;
            }
        }
    }
    m_steps[i_step].Predictors[i_predictor].PreloadTimeHours = val;
    return true;
}

bool asParameters::SetPreloadTimeHours(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided preload time parameter is a NaN."));
        return false;
    }

    m_steps[i_step].Predictors[i_predictor].PreloadTimeHours.clear();
    m_steps[i_step].Predictors[i_predictor].PreloadTimeHours.push_back(val);
    return true;
}

bool asParameters::SetPreloadLevels(int i_step, int i_predictor, VectorFloat val)
{
    if (val.size() < 1) {
        asLogError(_("The provided 'preload levels' vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                asLogError(_("There are NaN values in the provided 'preload levels' vector."));
                return false;
            }
        }
    }
    m_steps[i_step].Predictors[i_predictor].PreloadLevels = val;
    return true;
}

bool asParameters::SetPreloadLevels(int i_step, int i_predictor, float val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided preload level parameter is a NaN."));
        return false;
    }

    m_steps[i_step].Predictors[i_predictor].PreloadLevels.clear();
    m_steps[i_step].Predictors[i_predictor].PreloadLevels.push_back(val);
    return true;
}

bool asParameters::SetPreloadXmin(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the preload Xmin is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].PreloadXmin = val;
    return true;
}

bool asParameters::SetPreloadXptsnb(int i_step, int i_predictor, int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the preload points number on X is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].PreloadXptsnb = val;
    return true;
}

bool asParameters::SetPreloadYmin(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the preload Ymin is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].PreloadYmin = val;
    return true;
}

bool asParameters::SetPreloadYptsnb(int i_step, int i_predictor, int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the preload points number on Y is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].PreloadYptsnb = val;
    return true;
}

bool asParameters::SetPreprocessMethod(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess method is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].PreprocessMethod = val;
    return true;
}

wxString asParameters::GetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset) const
{
    if (m_steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size() >= (unsigned) (i_dataset + 1)) {
        return m_steps[i_step].Predictors[i_predictor].PreprocessDatasetIds[i_dataset];
    } else {
        asLogError(_("Trying to access to an element outside of PreprocessDatasetIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParameters::SetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess dataset ID is null"));
        return false;
    }

    if (m_steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size() >= (unsigned) (i_dataset + 1)) {
        m_steps[i_step].Predictors[i_predictor].PreprocessDatasetIds[i_dataset] = val;
    } else {
        wxASSERT((int) m_steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size() == i_dataset);
        m_steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.push_back(val);
    }

    return true;
}

wxString asParameters::GetPreprocessDataId(int i_step, int i_predictor, int i_dataset) const
{
    if (m_steps[i_step].Predictors[i_predictor].PreprocessDataIds.size() >= (unsigned) (i_dataset + 1)) {
        return m_steps[i_step].Predictors[i_predictor].PreprocessDataIds[i_dataset];
    } else {
        asLogError(_("Trying to access to an element outside of PreprocessDataIds in the parameters object."));
        return wxEmptyString;
    }
}

bool asParameters::SetPreprocessDataId(int i_step, int i_predictor, int i_dataset, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the preprocess data ID is null"));
        return false;
    }

    if (m_steps[i_step].Predictors[i_predictor].PreprocessDataIds.size() >= (unsigned) (i_dataset + 1)) {
        m_steps[i_step].Predictors[i_predictor].PreprocessDataIds[i_dataset] = val;
    } else {
        wxASSERT((int) m_steps[i_step].Predictors[i_predictor].PreprocessDataIds.size() == i_dataset);
        m_steps[i_step].Predictors[i_predictor].PreprocessDataIds.push_back(val);
    }

    return true;
}

float asParameters::GetPreprocessLevel(int i_step, int i_predictor, int i_dataset) const
{
    if (m_steps[i_step].Predictors[i_predictor].PreprocessLevels.size() >= (unsigned) (i_dataset + 1)) {
        return m_steps[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
    } else {
        asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
        return NaNFloat;
    }
}

bool asParameters::SetPreprocessLevel(int i_step, int i_predictor, int i_dataset, float val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the preprocess level is null"));
        return false;
    }

    if (m_steps[i_step].Predictors[i_predictor].PreprocessLevels.size() >= (unsigned) (i_dataset + 1)) {
        m_steps[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
    } else {
        wxASSERT((int) m_steps[i_step].Predictors[i_predictor].PreprocessLevels.size() == i_dataset);
        m_steps[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
    }

    return true;
}

double asParameters::GetPreprocessTimeHours(int i_step, int i_predictor, int i_dataset) const
{
    if (m_steps[i_step].Predictors[i_predictor].PreprocessTimeHours.size() >= (unsigned) (i_dataset + 1)) {
        return m_steps[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    } else {
        asLogError(_("Trying to access to an element outside of PreprocessTimeHours (std) in the parameters object."));
        return NaNDouble;
    }
}

bool asParameters::SetPreprocessTimeHours(int i_step, int i_predictor, int i_dataset, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the preprocess time (hours) is null"));
        return false;
    }

    if (m_steps[i_step].Predictors[i_predictor].PreprocessTimeHours.size() >= (unsigned) (i_dataset + 1)) {
        m_steps[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
    } else {
        wxASSERT((int) m_steps[i_step].Predictors[i_predictor].PreprocessTimeHours.size() == i_dataset);
        m_steps[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
    }

    return true;
}

bool asParameters::SetPredictorDatasetId(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor dataset is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].DatasetId = val;
    return true;
}

bool asParameters::SetPredictorDataId(int i_step, int i_predictor, wxString val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor data is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].DataId = val;
    return true;
}

bool asParameters::SetPredictorLevel(int i_step, int i_predictor, float val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor level is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Level = val;
    return true;
}

bool asParameters::SetPredictorGridType(int i_step, int i_predictor, wxString val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor grid type is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].GridType = val;
    return true;
}

bool asParameters::SetPredictorXmin(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor Xmin is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Xmin = val;
    return true;
}

bool asParameters::SetPredictorXptsnb(int i_step, int i_predictor, int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor points number on X is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Xptsnb = val;
    return true;
}

bool asParameters::SetPredictorXstep(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor X step is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Xstep = val;
    return true;
}

bool asParameters::SetPredictorXshift(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor X shift is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Xshift = val;
    return true;
}

bool asParameters::SetPredictorYmin(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor Ymin is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Ymin = val;
    return true;
}

bool asParameters::SetPredictorYptsnb(int i_step, int i_predictor, int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor points number on Y is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Yptsnb = val;
    return true;
}

bool asParameters::SetPredictorYstep(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor Y step is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Ystep = val;
    return true;
}

bool asParameters::SetPredictorYshift(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor Y shift is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Yshift = val;
    return true;
}

bool asParameters::SetPredictorFlatAllowed(int i_step, int i_predictor, int val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the 'flat allowed' property is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].FlatAllowed = val;
    return true;
}

bool asParameters::SetPredictorTimeHours(int i_step, int i_predictor, double val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor time (hours) is null"));
        return false;
    }

    m_steps[i_step].Predictors[i_predictor].TimeHours = val;

    return true;
}

bool asParameters::SetPredictorCriteria(int i_step, int i_predictor, const wxString &val)
{
    if (val.IsEmpty()) {
        asLogError(_("The provided value for the predictor criteria is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Criteria = val;
    return true;
}

bool asParameters::SetPredictorWeight(int i_step, int i_predictor, float val)
{
    if (asTools::IsNaN(val)) {
        asLogError(_("The provided value for the predictor weight is null"));
        return false;
    }
    m_steps[i_step].Predictors[i_predictor].Weight = val;
    return true;
}
