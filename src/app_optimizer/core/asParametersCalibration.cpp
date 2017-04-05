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

#include "asParametersCalibration.h"

#include <asFileParametersCalibration.h>
#include <asGeoAreaCompositeGrid.h>

asParametersCalibration::asParametersCalibration()
        : asParametersScoring()
{
    //ctor
}

asParametersCalibration::~asParametersCalibration()
{
    //dtor
}

void asParametersCalibration::AddStep()
{
    asParameters::AddStep();
    ParamsStepVect stepVect;
    stepVect.analogsNumber.push_back(0);
    m_stepsVect.push_back(stepVect);
}

bool asParametersCalibration::LoadFromFile(const wxString &filePath)
{
    wxLogVerbose(_("Loading parameters file."));

    if (filePath.IsEmpty()) {
        wxLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::ReadOnly);
    if (!fileParams.Open())
        return false;

    if (!fileParams.CheckRootElement())
        return false;

    int iStep = 0;
    wxXmlNode *nodeProcess = fileParams.GetRoot()->GetChildren();
    while (nodeProcess) {

        if (nodeProcess->GetName() == "description") {
            if(!ParseDescription(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "time_properties") {
            if(!ParseTimeProperties(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_dates") {
            AddStep();
            if(!ParseAnalogDatesParams(fileParams, iStep, nodeProcess))
                return false;
            iStep++;

        } else if (nodeProcess->GetName() == "analog_values") {
            if(!ParseAnalogValuesParams(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_forecast_score") {
            if(!ParseForecastScore(fileParams, nodeProcess))
                return false;

        } else if (nodeProcess->GetName() == "analog_forecast_score_final") {
            if(!ParseForecastScoreFinal(fileParams, nodeProcess))
                return false;

        } else {
            fileParams.UnknownNode(nodeProcess);
        }

        nodeProcess = nodeProcess->GetNext();
    }

    // Set properties
    if (!PreprocessingPropertiesOk())
        return false;
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Check inputs and init parameters
    if (!InputsOK())
        return false;
    InitValues();

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    wxLogVerbose(_("Parameters file loaded."));

    return true;
}

bool asParametersCalibration::ParseDescription(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess)
{
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

    return true;
}

bool asParametersCalibration::ParseTimeProperties(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess)
{
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
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "calibration_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "start_year") {
                    if (!SetCalibrationYearStart(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end_year") {
                    if (!SetCalibrationYearEnd(fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "start") {
                    if (!SetCalibrationStart(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "end") {
                    if (!SetCalibrationEnd(fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time_step") {
                    if (!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
        } else if (nodeParamBlock->GetName() == "validation_period") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "years") {
                    if (!SetValidationYearsVector(fileParams.GetVectorInt(nodeParam)))
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
                    if (!SetTimeArrayAnalogsIntervalDaysVector(fileParams.GetVectorInt(nodeParam)))
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
    return true;
}

bool asParametersCalibration::ParseAnalogDatesParams(asFileParametersCalibration &fileParams, int iStep,
                                                     const wxXmlNode *nodeProcess)
{
    int iPtor = 0;
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "analogs_number") {
            if (!SetAnalogsNumberVector(iStep, fileParams.GetVectorInt(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "predictor") {
            AddPredictor(iStep);
            AddPredictorVect(m_stepsVect[iStep]);
            SetPreprocess(iStep, iPtor, false);
            SetPreload(iStep, iPtor, false);
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "preload") {
                    SetPreload(iStep, iPtor, fileParams.GetBool(nodeParam));
                } else if (nodeParam->GetName() == "preprocessing") {
                    SetPreprocess(iStep, iPtor, true);
                    if(!ParsePreprocessedPredictors(fileParams, iStep, iPtor, nodeParam))
                        return false;
                } else if (nodeParam->GetName() == "dataset_id") {
                    if (!SetPredictorDatasetId(iStep, iPtor, fileParams.GetString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "data_id") {
                    if (!SetPredictorDataIdVector(iStep, iPtor, fileParams.GetVectorString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "level") {
                    if (!SetPredictorLevelVector(iStep, iPtor, fileParams.GetVectorFloat(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictorTimeHoursVector(iStep, iPtor, fileParams.GetVectorDouble(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "members") {
                    if (!SetPredictorMembersNb(iStep, iPtor, fileParams.GetInt(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "spatial_window") {
                    wxXmlNode *nodeWindow = nodeParam->GetChildren();
                    while (nodeWindow) {
                        if (nodeWindow->GetName() == "grid_type") {
                            if (!SetPredictorGridType(iStep, iPtor, fileParams.GetString(nodeWindow, "regular")))
                                return false;
                        } else if (nodeWindow->GetName() == "x_min") {
                            if (!SetPredictorXminVector(iStep, iPtor, GetVectorXmin(fileParams, nodeWindow, iStep, iPtor)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_points_nb") {
                            if (!SetPredictorXptsnbVector(iStep, iPtor, fileParams.GetVectorInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "x_step") {
                            if (!SetPredictorXstep(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_min") {
                            if (!SetPredictorYminVector(iStep, iPtor, GetVectorYmin(fileParams, nodeWindow, iStep, iPtor)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_points_nb") {
                            if (!SetPredictorYptsnbVector(iStep, iPtor, fileParams.GetVectorInt(nodeWindow)))
                                return false;
                        } else if (nodeWindow->GetName() == "y_step") {
                            if (!SetPredictorYstep(iStep, iPtor, fileParams.GetDouble(nodeWindow)))
                                return false;
                        } else {
                            fileParams.UnknownNode(nodeWindow);
                        }
                        nodeWindow = nodeWindow->GetNext();
                    }
                } else if (nodeParam->GetName() == "criteria") {
                    if (!SetPredictorCriteriaVector(iStep, iPtor, fileParams.GetVectorString(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "weight") {
                    if (!SetPredictorWeightVector(iStep, iPtor, fileParams.GetVectorFloat(nodeParam)))
                        return false;
                } else {
                    fileParams.UnknownNode(nodeParam);
                }
                nodeParam = nodeParam->GetNext();
            }
            iPtor++;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersCalibration::ParsePreprocessedPredictors(asFileParametersCalibration &fileParams, int iStep,
                                                          int iPtor, const wxXmlNode *nodeParam)
{
    int iPre = 0;
    wxXmlNode *nodePreprocess = nodeParam->GetChildren();
    while (nodePreprocess) {
        if (nodePreprocess->GetName() == "preprocessing_method") {
            if (!SetPreprocessMethod(iStep, iPtor, fileParams.GetString(nodePreprocess)))
                return false;
        } else if (nodePreprocess->GetName() == "preprocessing_data") {
            wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
            while (nodeParamPreprocess) {
                if (nodeParamPreprocess->GetName() == "dataset_id") {
                    if (!SetPreprocessDatasetId(iStep, iPtor, iPre, fileParams.GetString(nodeParamPreprocess)))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "data_id") {
                    if (!SetPreprocessDataIdVector(iStep, iPtor, iPre,
                                                   fileParams.GetVectorString(nodeParamPreprocess)))
                        return false;
                    if (!SetPreprocessDataId(iStep, iPtor, iPre,
                                             fileParams.GetVectorString(nodeParamPreprocess)[0]))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "level") {
                    if (!SetPreprocessLevelVector(iStep, iPtor, iPre,
                                                  fileParams.GetVectorFloat(nodeParamPreprocess)))
                        return false;
                    if (!SetPreprocessLevel(iStep, iPtor, iPre,
                                            fileParams.GetVectorFloat(nodeParamPreprocess)[0]))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "time") {
                    if (!SetPreprocessTimeHoursVector(iStep, iPtor, iPre,
                                                      fileParams.GetVectorDouble(nodeParamPreprocess)))
                        return false;
                    if (!SetPreprocessTimeHours(iStep, iPtor, iPre,
                                                fileParams.GetVectorDouble(nodeParamPreprocess)[0]))
                        return false;
                } else if (nodeParamPreprocess->GetName() == "members") {
                    if (!SetPreprocessMembersNb(iStep, iPtor, iPre, fileParams.GetInt(nodeParamPreprocess)))
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

bool asParametersCalibration::ParseAnalogValuesParams(asFileParametersCalibration &fileParams,
                                                      const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "predictand") {
            wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
            while (nodeParam) {
                if (nodeParam->GetName() == "station_id") {
                    if (!SetPredictandStationIdsVector(fileParams.GetStationIdsVector(nodeParam)))
                        return false;
                } else if (nodeParam->GetName() == "time") {
                    if (!SetPredictandTimeHours(fileParams.GetDouble(nodeParam)))
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

bool asParametersCalibration::ParseForecastScore(asFileParametersCalibration &fileParams, const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "score") {
            if (!SetForecastScoreNameVector(fileParams.GetVectorString(nodeParamBlock)))
                return false;
        } else if (nodeParamBlock->GetName() == "threshold") {
            SetForecastScoreThreshold(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "quantile") {
            SetForecastScoreQuantile(fileParams.GetFloat(nodeParamBlock));
        } else if (nodeParamBlock->GetName() == "postprocessing") {
            wxLogError(_("The postptocessing is not yet fully implemented."));
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersCalibration::ParseForecastScoreFinal(asFileParametersCalibration &fileParams,
                                                      const wxXmlNode *nodeProcess)
{
    wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
    while (nodeParamBlock) {
        if (nodeParamBlock->GetName() == "time_array") {
            if (!SetForecastScoreTimeArrayModeVector(fileParams.GetVectorString(nodeParamBlock)))
                return false;
        } else {
            fileParams.UnknownNode(nodeParamBlock);
        }
        nodeParamBlock = nodeParamBlock->GetNext();
    }
    return true;
}

bool asParametersCalibration::SetSpatialWindowProperties()
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            if (GetPredictorXstep(iStep, iPtor) == 0) {
                SetPredictorXshift(iStep, iPtor, 0);
            } else {
                double Xshift = std::fmod(GetPredictorXminVector(iStep, iPtor)[0], GetPredictorXstep(iStep, iPtor));
                if (Xshift < 0)
                    Xshift += GetPredictorXstep(iStep, iPtor);
                if (!SetPredictorXshift(iStep, iPtor, Xshift))
                    return false;
            }

            if(GetPredictorYstep(iStep, iPtor) == 0) {
                SetPredictorYshift(iStep, iPtor, 0);
            } else {
                double Yshift = std::fmod(GetPredictorYminVector(iStep, iPtor)[0], GetPredictorYstep(iStep, iPtor));
                if (Yshift < 0)
                    Yshift += GetPredictorYstep(iStep, iPtor);
                if (!SetPredictorYshift(iStep, iPtor, Yshift))
                    return false;
            }

            vi uptsnbs = GetPredictorXptsnbVector(iStep, iPtor);
            vi vptsnbs = GetPredictorYptsnbVector(iStep, iPtor);
            if (asTools::MinArray(&uptsnbs[0], &uptsnbs[uptsnbs.size() - 1]) <= 1 ||
                asTools::MinArray(&vptsnbs[0], &vptsnbs[vptsnbs.size() - 1]) <= 1) {
                SetPredictorFlatAllowed(iStep, iPtor, asFLAT_ALLOWED);
            }
        }
    }

    return true;
}

bool asParametersCalibration::SetPreloadingProperties()
{
    for (int iStep = 0; iStep < GetStepsNb(); iStep++) {
        for (int iPtor = 0; iPtor < GetPredictorsNb(iStep); iPtor++) {
            // Set maximum extent
            if (NeedsPreloading(iStep, iPtor)) {
                if (!SetPreloadXmin(iStep, iPtor, GetPredictorXminVector(iStep, iPtor)[0]))
                    return false;
                if (!SetPreloadYmin(iStep, iPtor, GetPredictorYminVector(iStep, iPtor)[0]))
                    return false;
                if (!SetPreloadXptsnb(iStep, iPtor, (int) GetPredictorXminVector(iStep, iPtor).size() +
                        GetPredictorXptsnbVector(iStep, iPtor)[GetPredictorXptsnbVector(iStep, iPtor).size() - 1]))
                    return false;
                if (!SetPreloadYptsnb(iStep, iPtor, (int) GetPredictorYminVector(iStep, iPtor).size() +
                        GetPredictorYptsnbVector(iStep, iPtor)[GetPredictorYptsnbVector(iStep, iPtor).size() - 1]))
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
                if (!SetPreloadDataIds(iStep, iPtor, GetPredictorDataIdVector(iStep, iPtor)))
                    return false;
                if (!SetPreloadLevels(iStep, iPtor, GetPredictorLevelVector(iStep, iPtor)))
                    return false;
                if (!SetPreloadTimeHours(iStep, iPtor, GetPredictorTimeHoursVector(iStep, iPtor)))
                    return false;
            } else if (NeedsPreloading(iStep, iPtor) && NeedsPreprocessing(iStep, iPtor)) {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(iStep, iPtor);
                vf preprocLevels;
                vd preprocTimeHours;

                // Different actions depending on the preprocessing method.
                if (method.IsSameAs("Gradients") || method.IsSameAs("Multiplication") || method.IsSameAs("Multiply") ||
                    method.IsSameAs("Addition") || method.IsSameAs("Average")) {
                    // Get them all
                    GetAllPreprocessTimesAndLevels(iStep, iPtor, preprocLevels, preprocTimeHours);
                } else if (method.IsSameAs("HumidityFlux")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(iStep, iPtor, 0);
                } else if (method.IsSameAs("HumidityIndex")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(iStep, iPtor, 0);
                } else if (method.IsSameAs("FormerHumidityIndex")) {
                    preprocLevels = GetPreprocessLevelVector(iStep, iPtor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(iStep, iPtor, 0);
                    vd preprocTimeHours2 = GetPreprocessTimeHoursVector(iStep, iPtor, 1);
                    preprocTimeHours.insert(preprocTimeHours.end(), preprocTimeHours2.begin(), preprocTimeHours2.end());
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

void asParametersCalibration::GetAllPreprocessTimesAndLevels(int iStep, int iPtor, vf &preprocLevels,
                                                             vd &preprocTimeHours) const
{
    for (int iPre = 0; iPre < GetPreprocessSize(iStep, iPtor); ++iPre) {
        vf preprocLevelsTmp = GetPreprocessLevelVector(iStep, iPtor, iPre);
        for (int i = 0; i < preprocLevelsTmp.size(); ++i) {
            bool sameFound = false;
            for (int k = 0; k < preprocLevels.size(); ++k) {
                if (preprocLevels[k] == preprocLevelsTmp[i]) {
                    sameFound = true;
                }
            }
            if (!sameFound) {
                preprocLevels.push_back(preprocLevelsTmp[i]);
            }
        }
        vd preprocTimeHoursTmp = GetPreprocessTimeHoursVector(iStep, iPtor, iPre);
        for (int i = 0; i < preprocTimeHoursTmp.size(); ++i) {
            bool sameFound = false;
            for (int k = 0; k < preprocTimeHours.size(); ++k) {
                if (preprocTimeHours[k] == preprocTimeHoursTmp[i]) {
                    sameFound = true;
                }
            }
            if (!sameFound) {
                preprocTimeHours.push_back(preprocTimeHoursTmp[i]);
            }
        }
    }
}

vd asParametersCalibration::GetVectorXmin(asFileParametersCalibration &fileParams, wxXmlNode *node, int iStep, int iPtor)
{
    vd vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty() || method.IsSameAs("fixed") || method.IsSameAs("array")) {
        vect = fileParams.GetVectorDouble(node);
    } else if (method.IsSameAs("minmax")) {
        if (GetPredictorGridType(iStep, iPtor).IsSameAs("Regular", false)) {
            vect = fileParams.GetVectorDouble(node);
        } else {
            double min, max;

            wxString valueMinStr = node->GetAttribute("min");
            if (!valueMinStr.ToDouble(&min)) {
                wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
            }

            wxString valueMaxStr = node->GetAttribute("max");
            if (!valueMaxStr.ToDouble(&max)) {
                wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
            }

            a1d xAxis = asGeoAreaCompositeGrid::GetXaxis(GetPredictorGridType(iStep, iPtor), min, max);
            for (int i = 0; i < xAxis.size(); ++i) {
                vect.push_back(xAxis[i]);
            }
        }

    } else {
        wxLogVerbose(_("The method is not correctly defined for Xmin in the calibration parameters file."));
        wxString valueStr = node->GetChildren()->GetContent();
        vect = asFileParameters::BuildVectorDouble(valueStr);
    }

    return vect;
}

vd asParametersCalibration::GetVectorYmin(asFileParametersCalibration &fileParams, wxXmlNode *node, int iStep, int iPtor)
{
    vd vect;
    wxString nodeName = node->GetName();
    wxString method = node->GetAttribute("method");
    if (method.IsEmpty() || method.IsSameAs("fixed") || method.IsSameAs("array")) {
        vect = fileParams.GetVectorDouble(node);
    } else if (method.IsSameAs("minmax")) {
        if (GetPredictorGridType(iStep, iPtor).IsSameAs("Regular", false)) {
            vect = fileParams.GetVectorDouble(node);
        } else {
            double min, max;

            wxString valueMinStr = node->GetAttribute("min");
            if (!valueMinStr.ToDouble(&min)) {
                wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
            }

            wxString valueMaxStr = node->GetAttribute("max");
            if (!valueMaxStr.ToDouble(&max)) {
                wxLogError(_("Failed at converting the value of the element %s (XML file)."), nodeName);
            }

            a1d yAxis = asGeoAreaCompositeGrid::GetYaxis(GetPredictorGridType(iStep, iPtor), min, max);
            for (int i = 0; i < yAxis.size(); ++i) {
                vect.push_back(yAxis[i]);
            }
        }

    } else {
        wxLogVerbose(_("The method is not correctly defined for Xmin in the calibration parameters file."));
        wxString valueStr = node->GetChildren()->GetContent();
        vect = asFileParameters::BuildVectorDouble(valueStr);
    }

    return vect;
}

bool asParametersCalibration::InputsOK() const
{
    // Time properties
    if (GetArchiveStart() <= 0) {
        wxLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetArchiveEnd() <= 0) {
        wxLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if (GetCalibrationStart() <= 0) {
        wxLogError(_("The beginning of the calibration period was not provided in the parameters file."));
        return false;
    }

    if (GetCalibrationEnd() <= 0) {
        wxLogError(_("The end of the calibration period was not provided in the parameters file."));
        return false;
    }

    if (GetValidationYearsVector().size() <= 0) {
        wxLogVerbose(_("The validation period was not provided in the parameters file (it can be on purpose)."));
        // allowed
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
        if (GetTimeArrayAnalogsIntervalDaysVector().size() == 0) {
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
        if (GetAnalogsNumberVector(i).size() == 0) {
            wxLogError(wxString::Format(_("The number of analogs (step %d) was not provided in the parameters file."),
                                        i));
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
                    if (GetPreprocessDataIdVector(i, j, k).size() == 0) {
                        wxLogError(_("The data for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessLevelVector(i, j, k).size() == 0) {
                        wxLogError(_("The level for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                    if (GetPreprocessTimeHoursVector(i, j, k).size() == 0) {
                        wxLogError(_("The time for preprocessing (step %d, predictor %d) was not provided in the parameters file."),
                                   i, j);
                        return false;
                    }
                }
            } else {
                if (GetPredictorDatasetId(i, j).IsEmpty()) {
                    wxLogError(_("The dataset (step %d, predictor %d) was not provided in the parameters file."), i, j);
                    return false;
                }
                if (GetPredictorDataIdVector(i, j).size() == 0) {
                    wxLogError(_("The data (step %d, predictor %d) was not provided in the parameters file."), i, j);
                    return false;
                }
                if (GetPredictorLevelVector(i, j).size() == 0) {
                    wxLogError(_("The level (step %d, predictor %d) was not provided in the parameters file."), i, j);
                    return false;
                }
                if (GetPredictorTimeHoursVector(i, j).size() == 0) {
                    wxLogError(_("The time (step %d, predictor %d) was not provided in the parameters file."), i, j);
                    return false;
                }
            }

            if (GetPredictorGridType(i, j).IsEmpty()) {
                wxLogError(_("The grid type (step %d, predictor %d) is empty in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorXminVector(i, j).size() == 0) {
                wxLogError(_("The X min value (step %d, predictor %d) was not provided in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorXptsnbVector(i, j).size() == 0) {
                wxLogError(_("The X points nb value (step %d, predictor %d) was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorYminVector(i, j).size() == 0) {
                wxLogError(_("The Y min value (step %d, predictor %d) was not provided in the parameters file."), i, j);
                return false;
            }
            if (GetPredictorYptsnbVector(i, j).size() == 0) {
                wxLogError(_("The Y points nb value (step %d, predictor %d) was not provided in the parameters file."),
                           i, j);
                return false;
            }
            if (GetPredictorCriteriaVector(i, j).size() == 0) {
                wxLogError(_("The criteria (step %d, predictor %d) was not provided in the parameters file."), i, j);
                return false;
            }
        }
    }

    // Analog values
    if (GetPredictandStationIdsVector().size() == 0) {
        wxLogWarning(_("The station ID was not provided in the parameters file (it can be on purpose)."));
        // allowed
    }

    // Forecast scores
    if (GetForecastScoreNameVector().size() == 0) {
        wxLogWarning(_("The forecast score was not provided in the parameters file."));
        return false;
    }

    // Forecast score final
    if (GetForecastScoreTimeArrayModeVector().size() == 0) {
        wxLogWarning(_("The final forecast score was not provided in the parameters file."));
        return false;
    }

    return true;
}

bool asParametersCalibration::FixTimeLimits()
{
    double minHour = 200.0, maxHour = -50.0;
    for (int i = 0; i < GetStepsNb(); i++) {
        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                for (int k = 0; k < GetPreprocessSize(i, j); k++) {
                    minHour = wxMin(GetPreprocessTimeHoursLowerLimit(i, j, k), minHour);
                    maxHour = wxMax(GetPreprocessTimeHoursUpperLimit(i, j, k), maxHour);
                }
            } else {
                minHour = wxMin(GetPredictorTimeHoursLowerLimit(i, j), minHour);
                maxHour = wxMax(GetPredictorTimeHoursUpperLimit(i, j), maxHour);
            }
        }
    }

    m_timeMinHours = minHour;
    m_timeMaxHours = maxHour;

    return true;
}

void asParametersCalibration::InitValues()
{
    wxASSERT(m_predictandStationIdsVect.size() > 0);
    wxASSERT(m_timeArrayAnalogsIntervalDaysVect.size() > 0);
    wxASSERT(m_forecastScoreVect.name.size() > 0);
    wxASSERT(m_forecastScoreVect.timeArrayMode.size() > 0);

    // Initialize the parameters values with the first values of the vectors
    m_predictandStationIds = m_predictandStationIdsVect[0];
    m_timeArrayAnalogsIntervalDays = m_timeArrayAnalogsIntervalDaysVect[0];
    SetForecastScoreName(m_forecastScoreVect.name[0]);
    SetForecastScoreTimeArrayMode(m_forecastScoreVect.timeArrayMode[0]);

    for (int i = 0; i < GetStepsNb(); i++) {
        SetAnalogsNumber(i, m_stepsVect[i].analogsNumber[0]);

        for (int j = 0; j < GetPredictorsNb(i); j++) {
            if (NeedsPreprocessing(i, j)) {
                unsigned long subDataNb = m_stepsVect[i].predictors[j].preprocessDataId.size();
                wxASSERT(subDataNb > 0);
                for (int k = 0; k < subDataNb; k++) {
                    wxASSERT(m_stepsVect[i].predictors[j].preprocessDataId.size() > 0);
                    wxASSERT(m_stepsVect[i].predictors[j].preprocessDataId[k].size() > 0);
                    wxASSERT(m_stepsVect[i].predictors[j].preprocessLevels.size() > 0);
                    wxASSERT(m_stepsVect[i].predictors[j].preprocessLevels[k].size() > 0);
                    wxASSERT(m_stepsVect[i].predictors[j].preprocessTimeHours.size() > 0);
                    wxASSERT(m_stepsVect[i].predictors[j].preprocessTimeHours[k].size() > 0);
                    SetPreprocessDataId(i, j, k, m_stepsVect[i].predictors[j].preprocessDataId[k][0]);
                    SetPreprocessLevel(i, j, k, m_stepsVect[i].predictors[j].preprocessLevels[k][0]);
                    SetPreprocessTimeHours(i, j, k, m_stepsVect[i].predictors[j].preprocessTimeHours[k][0]);
                }
            } else {
                SetPredictorDataId(i, j, m_stepsVect[i].predictors[j].dataId[0]);
                SetPredictorLevel(i, j, m_stepsVect[i].predictors[j].level[0]);
                SetPredictorTimeHours(i, j, m_stepsVect[i].predictors[j].timeHours[0]);
            }

            SetPredictorXmin(i, j, m_stepsVect[i].predictors[j].xMin[0]);
            SetPredictorXptsnb(i, j, m_stepsVect[i].predictors[j].xPtsNb[0]);
            SetPredictorYmin(i, j, m_stepsVect[i].predictors[j].yMin[0]);
            SetPredictorYptsnb(i, j, m_stepsVect[i].predictors[j].yPtsNb[0]);
            SetPredictorCriteria(i, j, m_stepsVect[i].predictors[j].criteria[0]);
            SetPredictorWeight(i, j, m_stepsVect[i].predictors[j].weight[0]);
        }

    }

    // Fixes and checks
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersCalibration::SetPredictandStationIdsVector(vvi val)
{
    if (val.size() < 1) {
        wxLogError(_("The provided predictand ID vector is empty."));
        return false;
    } else {
        if (val[0].size() < 1) {
            wxLogError(_("The provided predictand ID vector is empty."));
            return false;
        }

        for (int i = 0; i < (int) val.size(); i++) {
            for (int j = 0; j < (int) val[i].size(); j++) {
                if (asTools::IsNaN(val[i][j])) {
                    wxLogError(_("There are NaN values in the provided predictand ID vector."));
                    return false;
                }
            }
        }
    }

    m_predictandStationIdsVect = val;

    return true;
}

bool asParametersCalibration::SetTimeArrayAnalogsIntervalDaysVector(vi val)
{
    if (val.size() < 1) {
        wxLogError(_("The provided 'interval days' vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (asTools::IsNaN(val[i])) {
                wxLogError(_("There are NaN values in the provided 'interval days' vector."));
                return false;
            }
        }
    }
    m_timeArrayAnalogsIntervalDaysVect = val;
    return true;
}

bool asParametersCalibration::SetForecastScoreNameVector(vwxs val)
{
    if (val.size() < 1) {
        wxLogError(_("The provided forecast scores vector is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (val[i].IsEmpty()) {
                wxLogError(_("There are NaN values in the provided forecast scores vector."));
                return false;
            }

            if (val[i].IsSameAs("RankHistogram", false) || val[i].IsSameAs("RankHistogramReliability", false)) {
                wxLogError(_("The rank histogram can only be processed in the 'all scores' evalution method."));
                return false;
            }
        }
    }
    m_forecastScoreVect.name = val;
    return true;
}

bool asParametersCalibration::SetForecastScoreTimeArrayModeVector(vwxs val)
{
    if (val.size() < 1) {
        wxLogError(_("The provided time array mode vector for the forecast score is empty."));
        return false;
    } else {
        for (int i = 0; i < (int) val.size(); i++) {
            if (val[i].IsEmpty()) {
                wxLogError(_("There are NaN values in the provided time array mode vector for the forecast score."));
                return false;
            }
        }
    }
    m_forecastScoreVect.timeArrayMode = val;
    return true;
}

double asParametersCalibration::GetPreprocessTimeHoursLowerLimit(int iStep, int iPtor, int iPre) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    if (m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
        unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre].size() - 1;
        wxASSERT(lastrow >= 0);
        double val = asTools::MinArray(&m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre][0],
                                       &m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre][lastrow]);
        return val;
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessTimeHours (lower limit) in the parameters object."));
        return NaNd;
    }
}

double asParametersCalibration::GetPredictorXminLowerLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].xMin.size() - 1;
    wxASSERT(lastrow >= 0);
    double val = asTools::MinArray(&m_stepsVect[iStep].predictors[iPtor].xMin[0],
                                   &m_stepsVect[iStep].predictors[iPtor].xMin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorXptsnbLowerLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].xPtsNb.size() - 1;
    wxASSERT(lastrow >= 0);
    int val = asTools::MinArray(&m_stepsVect[iStep].predictors[iPtor].xPtsNb[0],
                                &m_stepsVect[iStep].predictors[iPtor].xPtsNb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorYminLowerLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].yMin.size() - 1;
    wxASSERT(lastrow >= 0);
    double val = asTools::MinArray(&m_stepsVect[iStep].predictors[iPtor].yMin[0],
                                   &m_stepsVect[iStep].predictors[iPtor].yMin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorYptsnbLowerLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].yPtsNb.size() - 1;
    wxASSERT(lastrow >= 0);
    int val = asTools::MinArray(&m_stepsVect[iStep].predictors[iPtor].yPtsNb[0],
                                &m_stepsVect[iStep].predictors[iPtor].yPtsNb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorTimeHoursLowerLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].timeHours.size() - 1;
    wxASSERT(lastrow >= 0);
    double val = asTools::MinArray(&m_stepsVect[iStep].predictors[iPtor].timeHours[0],
                                   &m_stepsVect[iStep].predictors[iPtor].timeHours[lastrow]);
    return val;
}

double asParametersCalibration::GetPreprocessTimeHoursUpperLimit(int iStep, int iPtor, int iPre) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    if (m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours.size() >= (unsigned) (iPre + 1)) {
        unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre].size() - 1;
        wxASSERT(lastrow >= 0);
        double val = asTools::MaxArray(&m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre][0],
                                       &m_stepsVect[iStep].predictors[iPtor].preprocessTimeHours[iPre][lastrow]);
        return val;
    } else {
        wxLogError(_("Trying to access to an element outside of preprocessTimeHours (upper limit) in the parameters object."));
        return NaNd;
    }
}

double asParametersCalibration::GetPredictorXminUpperLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].xMin.size() - 1;
    wxASSERT(lastrow >= 0);
    double val = asTools::MaxArray(&m_stepsVect[iStep].predictors[iPtor].xMin[0],
                                   &m_stepsVect[iStep].predictors[iPtor].xMin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorXptsnbUpperLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].xPtsNb.size() - 1;
    wxASSERT(lastrow >= 0);
    int val = asTools::MaxArray(&m_stepsVect[iStep].predictors[iPtor].xPtsNb[0],
                                &m_stepsVect[iStep].predictors[iPtor].xPtsNb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorYminUpperLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].yMin.size() - 1;
    wxASSERT(lastrow >= 0);
    double val = asTools::MaxArray(&m_stepsVect[iStep].predictors[iPtor].yMin[0],
                                   &m_stepsVect[iStep].predictors[iPtor].yMin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorYptsnbUpperLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].yPtsNb.size() - 1;
    wxASSERT(lastrow >= 0);
    int val = asTools::MaxArray(&m_stepsVect[iStep].predictors[iPtor].yPtsNb[0],
                                &m_stepsVect[iStep].predictors[iPtor].yPtsNb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorTimeHoursUpperLimit(int iStep, int iPtor) const
{
    wxASSERT((int) m_stepsVect[iStep].predictors.size() > iPtor);
    unsigned long lastrow = m_stepsVect[iStep].predictors[iPtor].timeHours.size() - 1;
    wxASSERT(lastrow >= 0);
    double val = asTools::MaxArray(&m_stepsVect[iStep].predictors[iPtor].timeHours[0],
                                   &m_stepsVect[iStep].predictors[iPtor].timeHours[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorXminIteration(int iStep, int iPtor) const
{
    if (m_stepsVect[iStep].predictors[iPtor].xMin.size() < 2)
        return 0;

    return m_stepsVect[iStep].predictors[iPtor].xMin[1] - m_stepsVect[iStep].predictors[iPtor].xMin[0];
}

int asParametersCalibration::GetPredictorXptsnbIteration(int iStep, int iPtor) const
{
    if (m_stepsVect[iStep].predictors[iPtor].xPtsNb.size() < 2)
        return 0;

    return m_stepsVect[iStep].predictors[iPtor].xPtsNb[1] - m_stepsVect[iStep].predictors[iPtor].xPtsNb[0];
}

double asParametersCalibration::GetPredictorYminIteration(int iStep, int iPtor) const
{
    if (m_stepsVect[iStep].predictors[iPtor].yMin.size() < 2)
        return 0;

    return m_stepsVect[iStep].predictors[iPtor].yMin[1] - m_stepsVect[iStep].predictors[iPtor].yMin[0];
}

int asParametersCalibration::GetPredictorYptsnbIteration(int iStep, int iPtor) const
{
    if (m_stepsVect[iStep].predictors[iPtor].yPtsNb.size() < 2)
        return 0;

    return m_stepsVect[iStep].predictors[iPtor].yPtsNb[1] - m_stepsVect[iStep].predictors[iPtor].yPtsNb[0];
}
