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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 * Portions Copyright 2013-2014 Pascal Horton, Terr@num.
 */

#include "asParametersCalibration.h"

#include <asFileParametersCalibration.h>

asParametersCalibration::asParametersCalibration()
:
asParametersScoring()
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
    stepVect.AnalogsNumber.push_back(0);
    m_stepsVect.push_back(stepVect);
}

bool asParametersCalibration::LoadFromFile(const wxString &filePath)
{
    asLogMessage(_("Loading parameters file."));

    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::ReadOnly);
    if(!fileParams.Open()) return false;

    if(!fileParams.CheckRootElement()) return false;

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
                            if(!SetArchiveYearStart(fileParams.GetInt(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "end_year") {
                            if(!SetArchiveYearEnd(fileParams.GetInt(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "start") {
                            if(!SetArchiveStart(fileParams.GetString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "end") {
                            if(!SetArchiveEnd(fileParams.GetString(nodeParam))) return false;
                        } else {
                            fileParams.UnknownNode(nodeParam);
                        }
                        nodeParam = nodeParam->GetNext();
                    }
                } else if (nodeParamBlock->GetName() == "calibration_period") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "start_year") {
                            if(!SetCalibrationYearStart(fileParams.GetInt(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "end_year") {
                            if(!SetCalibrationYearEnd(fileParams.GetInt(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "start") {
                            if(!SetCalibrationStart(fileParams.GetString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "end") {
                            if(!SetCalibrationEnd(fileParams.GetString(nodeParam))) return false;
                        } else {
                            fileParams.UnknownNode(nodeParam);
                        }
                        nodeParam = nodeParam->GetNext();
                    }
                } else if (nodeParamBlock->GetName() == "validation_period") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "years") {
                            if(!SetValidationYearsVector(fileParams.GetVectorInt(nodeParam))) return false;
                        } else {
                            fileParams.UnknownNode(nodeParam);
                        }
                        nodeParam = nodeParam->GetNext();
                    }
                } else if (nodeParamBlock->GetName() == "time_step") {
                    if(!SetTimeArrayTargetTimeStepHours(fileParams.GetDouble(nodeParamBlock))) return false;
                    if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetDouble(nodeParamBlock))) return false;
                } else if (nodeParamBlock->GetName() == "time_array_target") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "time_array") {
                            if(!SetTimeArrayTargetMode(fileParams.GetString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "predictand_serie_name") {
                            if(!SetTimeArrayTargetPredictandSerieName(fileParams.GetString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "predictand_min_threshold") {
                            if(!SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFloat(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "predictand_max_threshold") {
                            if(!SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFloat(nodeParam))) return false;
                        } else {
                            fileParams.UnknownNode(nodeParam);
                        }
                        nodeParam = nodeParam->GetNext();
                    }
                } else if (nodeParamBlock->GetName() == "time_array_analogs") {
                    wxXmlNode *nodeParam = nodeParamBlock->GetChildren();
                    while (nodeParam) {
                        if (nodeParam->GetName() == "time_array") {
                            if(!SetTimeArrayAnalogsMode(fileParams.GetString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "interval_days") {
                            if(!SetTimeArrayAnalogsIntervalDaysVector(fileParams.GetVectorInt(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "exclude_days") {
                            if(!SetTimeArrayAnalogsExcludeDays(fileParams.GetInt(nodeParam))) return false;
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
                    if(!SetAnalogsNumberVector(i_step, fileParams.GetVectorInt(nodeParamBlock))) return false;
                } else if (nodeParamBlock->GetName() == "predictor") {
                    AddPredictor(i_step);
                    AddPredictorVect(m_stepsVect[i_step]);
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
                                    if(!SetPreprocessMethod(i_step, i_ptor, fileParams.GetString(nodePreprocess))) return false;
                                } else if (nodePreprocess->GetName() == "preprocessing_data") {
                                    wxXmlNode *nodeParamPreprocess = nodePreprocess->GetChildren();
                                    while (nodeParamPreprocess) {
                                        if (nodeParamPreprocess->GetName() == "dataset_id") {
                                            if(!SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetString(nodeParamPreprocess))) return false;
                                        } else if (nodeParamPreprocess->GetName() == "data_id") {
                                            if(!SetPreprocessDataIdVector(i_step, i_ptor, i_dataset, fileParams.GetVectorString(nodeParamPreprocess))) return false;
                                            if(!SetPreprocessDataId(i_step, i_ptor, i_dataset, fileParams.GetVectorString(nodeParamPreprocess)[0])) return false;
                                        } else if (nodeParamPreprocess->GetName() == "level") {
                                            if(!SetPreprocessLevelVector(i_step, i_ptor, i_dataset, fileParams.GetVectorFloat(nodeParamPreprocess))) return false;
                                            if(!SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetVectorFloat(nodeParamPreprocess)[0])) return false;
                                        } else if (nodeParamPreprocess->GetName() == "time") {
                                            if(!SetPreprocessTimeHoursVector(i_step, i_ptor, i_dataset, fileParams.GetVectorDouble(nodeParamPreprocess))) return false;
                                            if(!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetVectorDouble(nodeParamPreprocess)[0])) return false;
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
                            if(!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "data_id") {
                            if(!SetPredictorDataIdVector(i_step, i_ptor, fileParams.GetVectorString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "level") {
                            if(!SetPredictorLevelVector(i_step, i_ptor, fileParams.GetVectorFloat(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "time") {
                            if(!SetPredictorTimeHoursVector(i_step, i_ptor, fileParams.GetVectorDouble(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "spatial_window") {
                            wxXmlNode *nodeWindow = nodeParam->GetChildren();
                            while (nodeWindow) {
                                if (nodeWindow->GetName() == "grid_type") {
                                    if(!SetPredictorGridType(i_step, i_ptor, fileParams.GetString(nodeWindow, "regular"))) return false;
                                } else if (nodeWindow->GetName() == "x_min") {
                                    if(!SetPredictorXminVector(i_step, i_ptor, fileParams.GetVectorDouble(nodeWindow))) return false;
                                } else if (nodeWindow->GetName() == "x_points_nb") {
                                    if(!SetPredictorXptsnbVector(i_step, i_ptor, fileParams.GetVectorInt(nodeWindow))) return false;
                                } else if (nodeWindow->GetName() == "x_step") {
                                    if(!SetPredictorXstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
                                } else if (nodeWindow->GetName() == "y_min") {
                                    if(!SetPredictorYminVector(i_step, i_ptor, fileParams.GetVectorDouble(nodeWindow))) return false;
                                } else if (nodeWindow->GetName() == "y_points_nb") {
                                    if(!SetPredictorYptsnbVector(i_step, i_ptor, fileParams.GetVectorInt(nodeWindow))) return false;
                                } else if (nodeWindow->GetName() == "y_step") {
                                    if(!SetPredictorYstep(i_step, i_ptor, fileParams.GetDouble(nodeWindow))) return false;
                                } else {
                                    fileParams.UnknownNode(nodeWindow);
                                }
                                nodeWindow = nodeWindow->GetNext();
                            }
                        } else if (nodeParam->GetName() == "criteria") {
                            if(!SetPredictorCriteriaVector(i_step, i_ptor, fileParams.GetVectorString(nodeParam))) return false;
                        } else if (nodeParam->GetName() == "weight") {
                            if(!SetPredictorWeightVector(i_step, i_ptor, fileParams.GetVectorFloat(nodeParam))) return false;
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
                            if(!SetPredictandStationIdsVector(fileParams.GetStationIdsVector(nodeParam))) return false;
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

        // Forecast scores
        } else if (nodeProcess->GetName() == "analog_forecast_score") {
            wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
            while (nodeParamBlock) {
                if (nodeParamBlock->GetName() == "score") {
                    if(!SetForecastScoreNameVector(fileParams.GetVectorString(nodeParamBlock))) return false;
                } else if (nodeParamBlock->GetName() == "threshold") {
                    SetForecastScoreThreshold(fileParams.GetFloat(nodeParamBlock));
                } else if (nodeParamBlock->GetName() == "quantile") {
                    SetForecastScoreQuantile(fileParams.GetFloat(nodeParamBlock));
                } else if (nodeParamBlock->GetName() == "postprocessing") {
                    asLogError(_("The postptocessing is not yet fully implemented."));
                } else {
                    fileParams.UnknownNode(nodeParamBlock);
                }
                nodeParamBlock = nodeParamBlock->GetNext();
            }

        // Forecast score final
        } else if (nodeProcess->GetName() == "analog_forecast_score_final") {
            wxXmlNode *nodeParamBlock = nodeProcess->GetChildren();
            while (nodeParamBlock) {
                if (nodeParamBlock->GetName() == "time_array") {
                    if(!SetForecastScoreTimeArrayModeVector(fileParams.GetVectorString(nodeParamBlock))) return false;
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
	if (!PreprocessingPropertiesOk()) return false;
    SetSpatialWindowProperties();
    SetPreloadingProperties();

    // Set sizes
    SetSizes();

    // Check inputs and init parameters
    if(!InputsOK()) return false;
    InitValues();

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    asLogMessage(_("Parameters file loaded."));

    return true;
}

bool asParametersCalibration::SetSpatialWindowProperties()
{
    for(int i_step=0; i_step<GetStepsNb();i_step++)
    {
        for(int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            double Xshift = std::fmod(GetPredictorXminVector(i_step, i_ptor)[0], GetPredictorXstep(i_step, i_ptor));
            if (Xshift<0) Xshift += GetPredictorXstep(i_step, i_ptor);
            if(!SetPredictorXshift(i_step, i_ptor, Xshift)) return false;

            double Yshift = std::fmod(GetPredictorYminVector(i_step, i_ptor)[0], GetPredictorYstep(i_step, i_ptor));
            if (Yshift<0) Yshift += GetPredictorYstep(i_step, i_ptor);
            if(!SetPredictorYshift(i_step, i_ptor, Yshift)) return false;

            VectorInt uptsnbs = GetPredictorXptsnbVector(i_step, i_ptor);
            VectorInt vptsnbs = GetPredictorYptsnbVector(i_step, i_ptor);
            if (asTools::MinArray(&uptsnbs[0], &uptsnbs[uptsnbs.size()-1])<=1 || asTools::MinArray(&vptsnbs[0], &vptsnbs[vptsnbs.size()-1])<=1)
            {
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            }
        }
    }

    return true;
}

bool asParametersCalibration::PreprocessingPropertiesOk()
{
	for (int i_step = 0; i_step<GetStepsNb(); i_step++)
	{
		for (int i_ptor = 0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
		{
			if (NeedsPreloading(i_step, i_ptor) && NeedsPreprocessing(i_step, i_ptor))
			{
				// Check the preprocessing method
				wxString method = GetPreprocessMethod(i_step, i_ptor);
				int preprocSize = GetPreprocessSize(i_step, i_ptor);

				// Check that the data ID is unique
				for (int i_preproc = 0; i_preproc<preprocSize; i_preproc++)
				{
					if (GetPreprocessDataIdVector(i_step, i_ptor, i_preproc).size() != 1)
					{
						asLogError(_("The preprocess DataId must be unique with the preload option."));
						return false;
					}
				}

				// Different actions depending on the preprocessing method.
				if (method.IsSameAs("Gradients"))
				{
					if (preprocSize != 1)
					{
						asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (1) in the preprocessing Gradients method."), preprocSize));
						return false;
					}
				}
				else if (method.IsSameAs("HumidityFlux"))
				{
					if (preprocSize != 4)
					{
						asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (4) in the preprocessing HumidityFlux method."), preprocSize));
						return false;
					}
				}
				else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply"))
				{
					if (preprocSize != 2)
					{
						asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (2) in the preprocessing Multiply method."), preprocSize));
						return false;
					}
				}
				else if (method.IsSameAs("FormerHumidityIndex"))
				{
					if (preprocSize != 4)
					{
						asLogError(wxString::Format(_("The size of the provided predictors (%d) does not match the requirements (4) in the preprocessing FormerHumidityIndex method."), preprocSize));
						return false;
					}
				}
				else
				{
					asLogWarning(wxString::Format(_("The %s preprocessing method is not yet handled with the preload option."), method));
				}
			}
		}
	}

	return true;
}

bool asParametersCalibration::SetPreloadingProperties()
{
    for(int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        for(int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            // Set maximum extent
            if (NeedsPreloading(i_step, i_ptor))
            {
                if(!SetPreloadXmin(i_step, i_ptor, GetPredictorXminVector(i_step, i_ptor)[0])) return false;
                if(!SetPreloadYmin(i_step, i_ptor, GetPredictorYminVector(i_step, i_ptor)[0])) return false;
                int Xbaseptsnb = std::abs(GetPredictorXminVector(i_step, i_ptor)[0]-GetPredictorXminVector(i_step, i_ptor)[GetPredictorXminVector(i_step, i_ptor).size()-1])/GetPredictorXstep(i_step, i_ptor);
                if(!SetPreloadXptsnb(i_step, i_ptor, Xbaseptsnb+GetPredictorXptsnbVector(i_step, i_ptor)[GetPredictorXptsnbVector(i_step, i_ptor).size()-1])) return false;
                int Ybaseptsnb = std::abs(GetPredictorYminVector(i_step, i_ptor)[0]-GetPredictorYminVector(i_step, i_ptor)[GetPredictorYminVector(i_step, i_ptor).size()-1])/GetPredictorYstep(i_step, i_ptor);
                if(!SetPreloadYptsnb(i_step, i_ptor, Ybaseptsnb+GetPredictorYptsnbVector(i_step, i_ptor)[GetPredictorYptsnbVector(i_step, i_ptor).size()-1])) return false;
            }

            // Change predictor properties when preprocessing 
            if (NeedsPreprocessing(i_step, i_ptor))
            {
                if(GetPreprocessSize(i_step, i_ptor)==1)
                {
                    SetPredictorDatasetId(i_step, i_ptor, GetPreprocessDatasetId(i_step, i_ptor, 0));
                    SetPredictorDataId(i_step, i_ptor, GetPreprocessDataId(i_step, i_ptor, 0));
                    SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
                    SetPredictorTimeHours(i_step, i_ptor, GetPreprocessTimeHours(i_step, i_ptor, 0));
                }
                else
                {
                    SetPredictorDatasetId(i_step, i_ptor, "mix");
                    SetPredictorDataId(i_step, i_ptor, "mix");
                    SetPredictorLevel(i_step, i_ptor, 0);
                    SetPredictorTimeHours(i_step, i_ptor, 0);
                }
            }

            // Set levels and time for preloading
            if (NeedsPreloading(i_step, i_ptor) && !NeedsPreprocessing(i_step, i_ptor))
            {
                if(!SetPreloadLevels(i_step, i_ptor, GetPredictorLevelVector(i_step, i_ptor))) return false;
                if(!SetPreloadTimeHours(i_step, i_ptor, GetPredictorTimeHoursVector(i_step, i_ptor))) return false;
            }
            else if (NeedsPreloading(i_step, i_ptor) && NeedsPreprocessing(i_step, i_ptor))
            {
                // Check the preprocessing method
                wxString method = GetPreprocessMethod(i_step, i_ptor);
                VectorFloat preprocLevels;
                VectorDouble preprocTimeHours;

                // Different actions depending on the preprocessing method.
                if (method.IsSameAs("Gradients"))
                {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(i_step, i_ptor, 0);
                }
                else if (method.IsSameAs("HumidityFlux"))
                {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(i_step, i_ptor, 0);
                }
                else if (method.IsSameAs("Multiplication") || method.IsSameAs("Multiply"))
                {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(i_step, i_ptor, 0);
                }
                else if (method.IsSameAs("FormerHumidityIndex"))
                {
                    preprocLevels = GetPreprocessLevelVector(i_step, i_ptor, 0);
                    preprocTimeHours = GetPreprocessTimeHoursVector(i_step, i_ptor, 0);
                    VectorDouble preprocTimeHours2 = GetPreprocessTimeHoursVector(i_step, i_ptor, 1);
                    preprocTimeHours.insert( preprocTimeHours.end(), preprocTimeHours2.begin(), preprocTimeHours2.end() );
                }
                else
                {
                    asLogWarning(wxString::Format(_("The %s preprocessing method is not yet handled with the preload option."), method));
                }

                if(!SetPreloadLevels(i_step, i_ptor, preprocLevels)) return false;
                if(!SetPreloadTimeHours(i_step, i_ptor, preprocTimeHours)) return false;
            }
        }
    }

    return true;
}

bool asParametersCalibration::InputsOK()
{
    // Time properties
    if(GetArchiveStart()<=0) {
        asLogError(_("The beginning of the archive period was not provided in the parameters file."));
        return false;
    }

    if(GetArchiveEnd()<=0) {
        asLogError(_("The end of the archive period was not provided in the parameters file."));
        return false;
    }

    if(GetCalibrationStart()<=0) {
        asLogError(_("The beginning of the calibration period was not provided in the parameters file."));
        return false;
    }

    if(GetCalibrationEnd()<=0) {
        asLogError(_("The end of the calibration period was not provided in the parameters file."));
        return false;
    }

    if(GetValidationYearsVector().size()<=0) {
        asLogMessage(_("The validation period was not provided in the parameters file (it can be on purpose)."));
        // allowed
    }

    if(GetTimeArrayTargetTimeStepHours()<=0) {
        asLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if(GetTimeArrayAnalogsTimeStepHours()<=0) {
        asLogError(_("The time step was not provided in the parameters file."));
        return false;
    }

    if(GetTimeArrayTargetMode().CmpNoCase("predictand_thresholds")==0 
        || GetTimeArrayTargetMode().CmpNoCase("PredictandThresholds")==0) {
        if(GetTimeArrayTargetPredictandSerieName().IsEmpty()) {
            asLogError(_("The predictand time series (for the threshold preselection) was not provided in the parameters file."));
            return false;
        }
        if(GetTimeArrayTargetPredictandMinThreshold()==GetTimeArrayTargetPredictandMaxThreshold()) {
            asLogError(_("The provided min/max predictand thresholds are equal in the parameters file."));
            return false;
        }
    }

    if(GetTimeArrayAnalogsMode().CmpNoCase("interval_days")==0 
        || GetTimeArrayAnalogsMode().CmpNoCase("IntervalDays")==0) {
        if(GetTimeArrayAnalogsIntervalDaysVector().size()==0) {
            asLogError(_("The interval days for the analogs preselection was not provided in the parameters file."));
            return false;
        }
        if(GetTimeArrayAnalogsExcludeDays()<=0) {
            asLogError(_("The number of days to exclude around the target date was not provided in the parameters file."));
            return false;
        }
    }

    // Analog dates
    for(int i=0;i<GetStepsNb();i++)
    {
        if(GetAnalogsNumberVector(i).size()==0) {
            asLogError(wxString::Format(_("The number of analogs (step %d) was not provided in the parameters file."), i));
            return false;
        }

        for(int j=0;j<GetPredictorsNb(i);j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                if(GetPreprocessMethod(i, j).IsEmpty()) {
                    asLogError(wxString::Format(_("The preprocessing method (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }

                for(int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    if(GetPreprocessDatasetId(i, j, k).IsEmpty()) {
                        asLogError(wxString::Format(_("The dataset for preprocessing (step %d, predictor %d) was not provided in the parameters file."), i, j));
                        return false;
                    }
                    if(GetPreprocessDataIdVector(i, j, k).size()==0) {
                        asLogError(wxString::Format(_("The data for preprocessing (step %d, predictor %d) was not provided in the parameters file."), i, j));
                        return false;
                    }
                    if(GetPreprocessLevelVector(i, j, k).size()==0) {
                        asLogError(wxString::Format(_("The level for preprocessing (step %d, predictor %d) was not provided in the parameters file."), i, j));
                        return false;
                    }
                    if(GetPreprocessTimeHoursVector(i, j, k).size()==0) {
                        asLogError(wxString::Format(_("The time for preprocessing (step %d, predictor %d) was not provided in the parameters file."), i, j));
                        return false;
                    }
                }
            }
            else
            {
                if(GetPredictorDatasetId(i, j).IsEmpty()) {
                    asLogError(wxString::Format(_("The dataset (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }
                if(GetPredictorDataIdVector(i, j).size()==0) {
                    asLogError(wxString::Format(_("The data (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }
                if(GetPredictorLevelVector(i, j).size()==0) {
                    asLogError(wxString::Format(_("The level (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }
                if(GetPredictorTimeHoursVector(i, j).size()==0) {
                    asLogError(wxString::Format(_("The time (step %d, predictor %d) was not provided in the parameters file."), i, j));
                    return false;
                }
            }

            if(GetPredictorGridType(i, j).IsEmpty()) {
                asLogError(wxString::Format(_("The grid type (step %d, predictor %d) is empty in the parameters file."), i, j));
                return false;
            }
            if(GetPredictorXminVector(i, j).size()==0) {
                asLogError(wxString::Format(_("The X min value (step %d, predictor %d) was not provided in the parameters file."), i, j));
                return false;
            }
            if(GetPredictorXptsnbVector(i, j).size()==0) {
                asLogError(wxString::Format(_("The X points nb value (step %d, predictor %d) was not provided in the parameters file."), i, j));
                return false;
            }
            if(GetPredictorYminVector(i, j).size()==0) {
                asLogError(wxString::Format(_("The Y min value (step %d, predictor %d) was not provided in the parameters file."), i, j));
                return false;
            }
            if(GetPredictorYptsnbVector(i, j).size()==0) {
                asLogError(wxString::Format(_("The Y points nb value (step %d, predictor %d) was not provided in the parameters file."), i, j));
                return false;
            }
            if(GetPredictorCriteriaVector(i, j).size()==0) {
                asLogError(wxString::Format(_("The criteria (step %d, predictor %d) was not provided in the parameters file."), i, j));
                return false;
            }
        }
    }

    // Analog values
    if(GetPredictandStationIdsVector().size()==0) {
        asLogWarning(_("The station ID was not provided in the parameters file (it can be on purpose)."));
        // allowed
    }

    // Forecast scores
    if(GetForecastScoreNameVector().size()==0) {
        asLogWarning(_("The forecast score was not provided in the parameters file."));
        return false;
    }

    // Forecast score final
    if(GetForecastScoreTimeArrayModeVector().size()==0) {
        asLogWarning(_("The final forecast score was not provided in the parameters file."));
        return false;
    }

    return true;
}

bool asParametersCalibration::FixTimeLimits()
{
    SetSizes();

    double minHour = 200.0, maxHour = -50.0;
    for(int i=0;i<GetStepsNb();i++)
    {
        for(int j=0;j<GetPredictorsNb(i);j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                for(int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    minHour = wxMin(GetPreprocessTimeHoursLowerLimit(i, j, k), minHour);
                    maxHour = wxMax(GetPreprocessTimeHoursUpperLimit(i, j, k), maxHour);
                }
            }
            else
            {
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
    wxASSERT(m_predictandStationIdsVect.size()>0);
    wxASSERT(m_timeArrayAnalogsIntervalDaysVect.size()>0);
    wxASSERT(m_forecastScoreVect.Name.size()>0);
    wxASSERT(m_forecastScoreVect.TimeArrayMode.size()>0);
    //wxASSERT(m_forecastScoreVect.TimeArrayDate.size()>0);
    //wxASSERT(m_forecastScoreVect.TimeArrayIntervalDays.size()>0);
    //wxASSERT(m_forecastScoreVect.PostprocessDupliExp.size()>0);

    // Initialize the parameters values with the first values of the vectors
    m_predictandStationIds = m_predictandStationIdsVect[0];
    m_timeArrayAnalogsIntervalDays = m_timeArrayAnalogsIntervalDaysVect[0];
    SetForecastScoreName(m_forecastScoreVect.Name[0]);
    SetForecastScoreTimeArrayMode(m_forecastScoreVect.TimeArrayMode[0]);
    //SetForecastScoreTimeArrayDate(m_forecastScoreVect.TimeArrayDate[0]);
    //SetForecastScoreTimeArrayIntervalDays(m_forecastScoreVect.TimeArrayIntervalDays[0]);
    //SetForecastScorePostprocessDupliExp(m_forecastScoreVect.PostprocessDupliExp[0]);

    for (int i=0; i<GetStepsNb(); i++)
    {
        SetAnalogsNumber(i, m_stepsVect[i].AnalogsNumber[0]);

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                int subDataNb = m_stepsVect[i].Predictors[j].PreprocessDataId.size();
                wxASSERT(subDataNb>0);
                for (int k=0; k<subDataNb; k++)
                {
                    wxASSERT(m_stepsVect[i].Predictors[j].PreprocessDataId.size()>0);
                    wxASSERT(m_stepsVect[i].Predictors[j].PreprocessDataId[k].size()>0);
                    wxASSERT(m_stepsVect[i].Predictors[j].PreprocessLevels.size()>0);
                    wxASSERT(m_stepsVect[i].Predictors[j].PreprocessLevels[k].size()>0);
                    wxASSERT(m_stepsVect[i].Predictors[j].PreprocessTimeHours.size()>0);
                    wxASSERT(m_stepsVect[i].Predictors[j].PreprocessTimeHours[k].size()>0);
                    SetPreprocessDataId(i,j,k, m_stepsVect[i].Predictors[j].PreprocessDataId[k][0]);
                    SetPreprocessLevel(i,j,k, m_stepsVect[i].Predictors[j].PreprocessLevels[k][0]);
                    SetPreprocessTimeHours(i,j,k, m_stepsVect[i].Predictors[j].PreprocessTimeHours[k][0]);
                }
            }
            else
            {
                SetPredictorDataId(i,j, m_stepsVect[i].Predictors[j].DataId[0]);
                SetPredictorLevel(i,j, m_stepsVect[i].Predictors[j].Level[0]);
                SetPredictorTimeHours(i,j, m_stepsVect[i].Predictors[j].TimeHours[0]);
            }

            SetPredictorXmin(i,j, m_stepsVect[i].Predictors[j].Xmin[0]);
            SetPredictorXptsnb(i,j, m_stepsVect[i].Predictors[j].Xptsnb[0]);
            SetPredictorYmin(i,j, m_stepsVect[i].Predictors[j].Ymin[0]);
            SetPredictorYptsnb(i,j, m_stepsVect[i].Predictors[j].Yptsnb[0]);
            SetPredictorCriteria(i,j, m_stepsVect[i].Predictors[j].Criteria[0]);
            SetPredictorWeight(i,j, m_stepsVect[i].Predictors[j].Weight[0]);
        }

    }

    // Fixes and checks
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}

bool asParametersCalibration::SetPredictandStationIdsVector(VVectorInt val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided predictand ID vector is empty."));
        return false;
    }
    else
    {
        if (val[0].size()<1)
        {
            asLogError(_("The provided predictand ID vector is empty."));
            return false;
        }

        for (int i=0; i<(int)val.size(); i++)
        {
            for (int j=0; j<(int)val[i].size(); j++)
            {
                if (asTools::IsNaN(val[i][j]))
                {
                    asLogError(_("There are NaN values in the provided predictand ID vector."));
                    return false;
                }
            }
        }
    }

    m_predictandStationIdsVect = val;

    return true;
}

bool asParametersCalibration::SetTimeArrayAnalogsIntervalDaysVector(VectorInt val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided 'interval days' vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided 'interval days' vector."));
                return false;
            }
        }
    }
    m_timeArrayAnalogsIntervalDaysVect = val;
    return true;
}

bool asParametersCalibration::SetAnalogsNumberVector(int i_step, VectorInt val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided analogs number vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided analogs number vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].AnalogsNumber = val;
    return true;
}

VectorString asParametersCalibration::GetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset)
{
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
    {
        return m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset];
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessDataId in the parameters object."));
        VectorString empty;
        return empty;
    }
}
    
bool asParametersCalibration::SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided preprocess data ID vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (val[i].IsEmpty())
            {
                asLogError(_("There are NaN values in the provided preprocess data ID vector."));
                return false;
            }
        }
    }

    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
    {
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset].clear();
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset] = val;
    }
    else
    {
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
    }

    return true;
}

VectorFloat asParametersCalibration::GetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset)
{
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
    {
        return m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
        VectorFloat empty;
        return empty;
    }
}

bool asParametersCalibration::SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided preprocess levels vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided preprocess levels vector."));
                return false;
            }
        }
    }

    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
    {
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].clear();
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
    }
    else
    {
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
    }

    return true;
}

VectorDouble asParametersCalibration::GetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset)
{
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
    {
        return m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessTimeHours (vect) in the parameters object."));
        VectorDouble empty;
        return empty;
    }
}

bool asParametersCalibration::SetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset, VectorDouble val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided preprocess time (hours) vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided preprocess time (hours) vector."));
                return false;
            }
        }
    }

    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
    {
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].clear();
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
    }
    else
    {
        m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
    }

    return true;
}

bool asParametersCalibration::SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided data ID vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (val[i].IsEmpty())
            {
                asLogError(_("There are NaN values in the provided data ID vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].DataId = val;
    return true;
}

bool asParametersCalibration::SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided predictor levels vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided predictor levels vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Level = val;
    return true;
}

bool asParametersCalibration::SetPredictorXminVector(int i_step, int i_predictor, VectorDouble val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided Xmin vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided Xmin vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Xmin = val;
    return true;
}

bool asParametersCalibration::SetPredictorXptsnbVector(int i_step, int i_predictor, VectorInt val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided Xptsnb vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided Xptsnb vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Xptsnb = val;
    return true;
}

bool asParametersCalibration::SetPredictorYminVector(int i_step, int i_predictor, VectorDouble val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided Ymin vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided Ymin vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Ymin = val;
    return true;
}

bool asParametersCalibration::SetPredictorYptsnbVector(int i_step, int i_predictor, VectorInt val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided Yptsnb vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided Yptsnb vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Yptsnb = val;
    return true;
}

bool asParametersCalibration::SetPredictorTimeHoursVector(int i_step, int i_predictor, VectorDouble val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided predictor time (hours) vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided predictor time (hours) vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].TimeHours = val;
    return true;
}

bool asParametersCalibration::SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided predictor criteria vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (val[i].IsEmpty())
            {
                asLogError(_("There are NaN values in the provided predictor criteria vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Criteria = val;
    return true;
}

bool asParametersCalibration::SetPredictorWeightVector(int i_step, int i_predictor, VectorFloat val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided predictor weights vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided predictor weights vector."));
                return false;
            }
        }
    }
    m_stepsVect[i_step].Predictors[i_predictor].Weight = val;
    return true;
}

bool asParametersCalibration::SetForecastScoreNameVector(VectorString val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided forecast scores vector is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (val[i].IsEmpty())
            {
                asLogError(_("There are NaN values in the provided forecast scores vector."));
                return false;
            }

            if (val[i].IsSameAs("RankHistogram", false) || val[i].IsSameAs("RankHistogramReliability", false))
            {
                asLogError(_("The rank histogram can only be processed in the 'all scores' evalution method."));
                return false;
            }
        }
    }
    m_forecastScoreVect.Name = val;
    return true;
}

bool asParametersCalibration::SetForecastScoreTimeArrayModeVector(VectorString val)
{
    if (val.size()<1)
    {
        asLogError(_("The provided time array mode vector for the forecast score is empty."));
        return false;
    }
    else
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (val[i].IsEmpty())
            {
                asLogError(_("There are NaN values in the provided time array mode vector for the forecast score."));
                return false;
            }
        }
    }
    m_forecastScoreVect.TimeArrayMode = val;
    return true;
}

bool asParametersCalibration::SetForecastScoreTimeArrayDateVector(VectorDouble val)
{
    m_forecastScoreVect.TimeArrayDate = val;
    return true;
}

bool asParametersCalibration::SetForecastScoreTimeArrayIntervalDaysVector(VectorInt val)
{
    m_forecastScoreVect.TimeArrayIntervalDays = val;
    return true;
}

bool asParametersCalibration::SetForecastScorePostprocessDupliExpVector(VectorFloat val)
{
    if (val.size()<1 && ForecastScoreNeedsPostprocessing())
    {
        asLogError(_("The provided 'PostprocessDupliExp' vector is empty."));
        return false;
    }
    else if (ForecastScoreNeedsPostprocessing())
    {
        for (int i=0; i<(int)val.size(); i++)
        {
            if (asTools::IsNaN(val[i]))
            {
                asLogError(_("There are NaN values in the provided 'PostprocessDupliExp' vector."));
                return false;
            }
        }
    }
    m_forecastScoreVect.PostprocessDupliExp = val;
    return true;
}

int asParametersCalibration::GetTimeArrayAnalogsIntervalDaysLowerLimit()
{
    int lastrow = m_timeArrayAnalogsIntervalDaysVect.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MinArray(&m_timeArrayAnalogsIntervalDaysVect[0],&m_timeArrayAnalogsIntervalDaysVect[lastrow]);
    return val;
}

int asParametersCalibration::GetAnalogsNumberLowerLimit(int i_step)
{
    int lastrow = m_stepsVect[i_step].AnalogsNumber.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MinArray(&m_stepsVect[i_step].AnalogsNumber[0],&m_stepsVect[i_step].AnalogsNumber[lastrow]);
    return val;
}

float asParametersCalibration::GetPreprocessLevelLowerLimit(int i_step, int i_predictor, int i_dataset)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
    {
        int lastrow = m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][0],&m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][lastrow]);
        return val;
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
        return NaNFloat;
    }
}

double asParametersCalibration::GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
    {
        int lastrow = m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][0],&m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][lastrow]);
        return val;
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessTimeHours (lower limit) in the parameters object."));
        return NaNDouble;
    }
}

float asParametersCalibration::GetPredictorLevelLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Level.size()-1;
    wxASSERT(lastrow>=0);
    float val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].Level[0],&m_stepsVect[i_step].Predictors[i_predictor].Level[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorXminLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Xmin.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].Xmin[0],&m_stepsVect[i_step].Predictors[i_predictor].Xmin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorXptsnbLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Xptsnb.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].Xptsnb[0],&m_stepsVect[i_step].Predictors[i_predictor].Xptsnb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorYminLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Ymin.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].Ymin[0],&m_stepsVect[i_step].Predictors[i_predictor].Ymin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorYptsnbLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Yptsnb.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].Yptsnb[0],&m_stepsVect[i_step].Predictors[i_predictor].Yptsnb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].TimeHours.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].TimeHours[0],&m_stepsVect[i_step].Predictors[i_predictor].TimeHours[lastrow]);
    return val;
}

float asParametersCalibration::GetPredictorWeightLowerLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Weight.size()-1;
    wxASSERT(lastrow>=0);
    float val = asTools::MinArray(&m_stepsVect[i_step].Predictors[i_predictor].Weight[0],&m_stepsVect[i_step].Predictors[i_predictor].Weight[lastrow]);
    return val;
}

double asParametersCalibration::GetForecastScoreTimeArrayDateLowerLimit()
{
    int lastrow = m_forecastScoreVect.TimeArrayDate.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MinArray(&m_forecastScoreVect.TimeArrayDate[0],&m_forecastScoreVect.TimeArrayDate[lastrow]);
    return val;
}

int asParametersCalibration::GetForecastScoreTimeArrayIntervalDaysLowerLimit()
{
    int lastrow = m_forecastScoreVect.TimeArrayIntervalDays.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MinArray(&m_forecastScoreVect.TimeArrayIntervalDays[0],&m_forecastScoreVect.TimeArrayIntervalDays[lastrow]);
    return val;
}

float asParametersCalibration::GetForecastScorePostprocessDupliExpLowerLimit()
{
    int lastrow = m_forecastScoreVect.PostprocessDupliExp.size()-1;
    wxASSERT(lastrow>=0);
    float val = asTools::MinArray(&m_forecastScoreVect.PostprocessDupliExp[0],&m_forecastScoreVect.PostprocessDupliExp[lastrow]);
    return val;
}

int asParametersCalibration::GetTimeArrayAnalogsIntervalDaysUpperLimit()
{
    int lastrow = m_timeArrayAnalogsIntervalDaysVect.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MaxArray(&m_timeArrayAnalogsIntervalDaysVect[0],&m_timeArrayAnalogsIntervalDaysVect[lastrow]);
    return val;
}

int asParametersCalibration::GetAnalogsNumberUpperLimit(int i_step)
{
    int lastrow = m_stepsVect[i_step].AnalogsNumber.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MaxArray(&m_stepsVect[i_step].AnalogsNumber[0],&m_stepsVect[i_step].AnalogsNumber[lastrow]);
    return val;
}

float asParametersCalibration::GetPreprocessLevelUpperLimit(int i_step, int i_predictor, int i_dataset)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
    {
        int lastrow = m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][0],&m_stepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][lastrow]);
        return val;
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
        return NaNFloat;
    }
}
    
double asParametersCalibration::GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
    {
        int lastrow = m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][0],&m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][lastrow]);
        return val;
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessTimeHours (upper limit) in the parameters object."));
        return NaNDouble;
    }
}

float asParametersCalibration::GetPredictorLevelUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Level.size()-1;
    wxASSERT(lastrow>=0);
    float val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].Level[0],&m_stepsVect[i_step].Predictors[i_predictor].Level[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorXminUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Xmin.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].Xmin[0],&m_stepsVect[i_step].Predictors[i_predictor].Xmin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorXptsnbUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Xptsnb.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].Xptsnb[0],&m_stepsVect[i_step].Predictors[i_predictor].Xptsnb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorYminUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Ymin.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].Ymin[0],&m_stepsVect[i_step].Predictors[i_predictor].Ymin[lastrow]);
    return val;
}

int asParametersCalibration::GetPredictorYptsnbUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Yptsnb.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].Yptsnb[0],&m_stepsVect[i_step].Predictors[i_predictor].Yptsnb[lastrow]);
    return val;
}

double asParametersCalibration::GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].TimeHours.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].TimeHours[0],&m_stepsVect[i_step].Predictors[i_predictor].TimeHours[lastrow]);
    return val;
}

float asParametersCalibration::GetPredictorWeightUpperLimit(int i_step, int i_predictor)
{
    wxASSERT((int)m_stepsVect[i_step].Predictors.size()>i_predictor);
    int lastrow = m_stepsVect[i_step].Predictors[i_predictor].Weight.size()-1;
    wxASSERT(lastrow>=0);
    float val = asTools::MaxArray(&m_stepsVect[i_step].Predictors[i_predictor].Weight[0],&m_stepsVect[i_step].Predictors[i_predictor].Weight[lastrow]);
    return val;
}

double asParametersCalibration::GetForecastScoreTimeArrayDateUpperLimit()
{
    int lastrow = m_forecastScoreVect.TimeArrayDate.size()-1;
    wxASSERT(lastrow>=0);
    double val = asTools::MaxArray(&m_forecastScoreVect.TimeArrayDate[0],&m_forecastScoreVect.TimeArrayDate[lastrow]);
    return val;
}
    
int asParametersCalibration::GetForecastScoreTimeArrayIntervalDaysUpperLimit()
{
    int lastrow = m_forecastScoreVect.TimeArrayIntervalDays.size()-1;
    wxASSERT(lastrow>=0);
    int val = asTools::MaxArray(&m_forecastScoreVect.TimeArrayIntervalDays[0],&m_forecastScoreVect.TimeArrayIntervalDays[lastrow]);
    return val;
}

float asParametersCalibration::GetForecastScorePostprocessDupliExpUpperLimit()
{
    int lastrow = m_forecastScoreVect.PostprocessDupliExp.size()-1;
    wxASSERT(lastrow>=0);
    float val = asTools::MaxArray(&m_forecastScoreVect.PostprocessDupliExp[0],&m_forecastScoreVect.PostprocessDupliExp[lastrow]);
    return val;
}

int asParametersCalibration::GetTimeArrayAnalogsIntervalDaysIteration()
{
    if (m_timeArrayAnalogsIntervalDaysVect.size()<2) return 0;
    int val = m_timeArrayAnalogsIntervalDaysVect[1] - m_timeArrayAnalogsIntervalDaysVect[0];
    return val;
}

int asParametersCalibration::GetAnalogsNumberIteration(int i_step)
{
    if (m_stepsVect[i_step].AnalogsNumber.size()<2) return 0;
    int val = m_stepsVect[i_step].AnalogsNumber[1] - m_stepsVect[i_step].AnalogsNumber[0];
    return val;
}

double asParametersCalibration::GetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset)
{
    if(m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
    {
        if (m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].size()<2) return 0;
        double val = m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][1] - m_stepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][0];
        return val;
    }
    else
    {
        asLogError(_("Trying to access to an element outside of PreprocessTimeHours (iteration) in the parameters object."));
        return NaNDouble;
    }
}

double asParametersCalibration::GetPredictorXminIteration(int i_step, int i_predictor)
{
    if (m_stepsVect[i_step].Predictors[i_predictor].Xmin.size()<2) return 0;
    int row = floor((float)m_stepsVect[i_step].Predictors[i_predictor].Xmin.size()/2.0);
    double val = m_stepsVect[i_step].Predictors[i_predictor].Xmin[row] - m_stepsVect[i_step].Predictors[i_predictor].Xmin[row-1];
    return val;
}

int asParametersCalibration::GetPredictorXptsnbIteration(int i_step, int i_predictor)
{
    if (m_stepsVect[i_step].Predictors[i_predictor].Xptsnb.size()<2) return 0;
    int val = m_stepsVect[i_step].Predictors[i_predictor].Xptsnb[1] - m_stepsVect[i_step].Predictors[i_predictor].Xptsnb[0];
    return val;
}

double asParametersCalibration::GetPredictorYminIteration(int i_step, int i_predictor)
{
    if (m_stepsVect[i_step].Predictors[i_predictor].Ymin.size()<2) return 0;
    int row = floor((float)m_stepsVect[i_step].Predictors[i_predictor].Ymin.size()/2.0);
    double val = m_stepsVect[i_step].Predictors[i_predictor].Ymin[row] - m_stepsVect[i_step].Predictors[i_predictor].Ymin[row-1];
    return val;
}

int asParametersCalibration::GetPredictorYptsnbIteration(int i_step, int i_predictor)
{
    if (m_stepsVect[i_step].Predictors[i_predictor].Yptsnb.size()<2) return 0;
    int val = m_stepsVect[i_step].Predictors[i_predictor].Yptsnb[1] - m_stepsVect[i_step].Predictors[i_predictor].Yptsnb[0];
    return val;
}

double asParametersCalibration::GetPredictorTimeHoursIteration(int i_step, int i_predictor)
{
    if (m_stepsVect[i_step].Predictors[i_predictor].TimeHours.size()<2) return 0;
    double val = m_stepsVect[i_step].Predictors[i_predictor].TimeHours[1] - m_stepsVect[i_step].Predictors[i_predictor].TimeHours[0];
    return val;
}

float asParametersCalibration::GetPredictorWeightIteration(int i_step, int i_predictor)
{
    if (m_stepsVect[i_step].Predictors[i_predictor].Weight.size()<2) return 0;
    float val = m_stepsVect[i_step].Predictors[i_predictor].Weight[1] - m_stepsVect[i_step].Predictors[i_predictor].Weight[0];
    return val;
}

double asParametersCalibration::GetForecastScoreTimeArrayDateIteration()
{
    if (m_forecastScoreVect.TimeArrayDate.size()<2) return 0;
    double val = m_forecastScoreVect.TimeArrayDate[1] - m_forecastScoreVect.TimeArrayDate[0];
    return val;
}

int asParametersCalibration::GetForecastScoreTimeArrayIntervalDaysIteration()
{
    if (m_forecastScoreVect.TimeArrayIntervalDays.size()<2) return 0;
    int val = m_forecastScoreVect.TimeArrayIntervalDays[1] - m_forecastScoreVect.TimeArrayIntervalDays[0];
    return val;
}

float asParametersCalibration::GetForecastScorePostprocessDupliExpIteration()
{
    if (m_forecastScoreVect.PostprocessDupliExp.size()<2) return 0;
    float val = m_forecastScoreVect.PostprocessDupliExp[1] - m_forecastScoreVect.PostprocessDupliExp[0];
    return val;
}
