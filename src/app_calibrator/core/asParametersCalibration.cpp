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
    AddPredictorVect(stepVect);
    m_StepsVect.push_back(stepVect);
}

bool asParametersCalibration::LoadFromFile(const wxString &filePath)
{
    asLogMessage(_("Loading parameters file."));

    VectorFloat emptyVectFloat(1);
    emptyVectFloat[0] = NaNFloat;

    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::ReadOnly);
    if(!fileParams.Open()) return false;

    if(!fileParams.GoToRootElement()) return false;

    // Get general parameters
    if(!fileParams.GoToFirstNodeWithPath("General")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.CheckDeprecatedChildNode("Period")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period")) return false;
    if(!SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
    if(!SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Calibration Period")) return false;
    if(!SetCalibrationYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
    if(!SetCalibrationYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Validation Period", asHIDE_WARNINGS))
    {
        if(!SetValidationYearsVector(GetFileParamIntVector(fileParams, "Years"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Time Properties", asHIDE_WARNINGS))
    {
        if(!SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
        if(!SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
        if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
    if(!SetTimeArrayTargetMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"))) return false;
    if(GetTimeArrayTargetMode().IsSameAs("PredictandThresholds"))
    {
        if(!SetTimeArrayTargetPredictandSerieName(fileParams.GetFirstElementAttributeValueText("PredictandSerieName", "value"))) return false;
        if(!SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFirstElementAttributeValueFloat("PredictandMinThreshold", "value"))) return false;
        if(!SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFirstElementAttributeValueFloat("PredictandMaxThreshold", "value"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
    if(!SetTimeArrayAnalogsMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"))) return false;
    if(!SetTimeArrayAnalogsExcludeDays(fileParams.GetFirstElementAttributeValueInt("ExcludeDays", "value"))) return false;
    if(!SetTimeArrayAnalogsIntervalDaysVector(GetFileParamIntVector(fileParams, "IntervalDays"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Dates processes
    int i_step = 0;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Dates")) return false;

    while(true)
    {
        AddStep();

        if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
        if(!fileParams.GoANodeBack()) return false;

        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method Name")) return false;
        if(!SetMethodName(i_step, fileParams.GetFirstElementAttributeValueText("MethodName", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;

        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
        if(!SetAnalogsNumberVector(i_step, GetFileParamIntVector(fileParams, "AnalogsNumber"))) return false;
        if(!fileParams.GoANodeBack()) return false;

        // Get data
        if(!fileParams.GoToFirstNodeWithPath("Data")) return false;
        bool dataOver = false;
        int i_ptor = 0;
        while(!dataOver)
        {
            wxString predictorNature = fileParams.GetThisElementAttributeValueText("name");

            if(predictorNature.IsSameAs("Predictor", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                SetPreprocess(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preprocess", "value"));
                if(NeedsPreprocessing(i_step, i_ptor))
                {
                    asLogError(_("Preprocessing option is not coherent."));
                    return false;
                }
                if(!fileParams.GoANodeBack()) return false;

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preload", asHIDE_WARNINGS))
                {
                    SetPreload(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preload", "value"));
                    if(!fileParams.GoANodeBack()) return false;
                }
                else
                {
                    SetPreload(i_step, i_ptor, false);
                }

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data")) return false;
                if(!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"))) return false;
                if(!SetPredictorDataIdVector(i_step, i_ptor, GetFileParamStringVector(fileParams, "DataId"))) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                if(!SetPredictorLevelVector(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Level"))) return false;
                if (NeedsPreloading(i_step, i_ptor))
                {
                    if(!SetPreloadLevels(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Level"))) return false;
                }
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Frame")) return false;
                if(!SetPredictorTimeHoursVector(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "TimeHours"))) return false;
                if(!SetPreloadTimeHours(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "TimeHours"))) return false;
                if(!fileParams.GoANodeBack()) return false;

            }
            else if(predictorNature.IsSameAs("Predictor Preprocessed", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preload", asHIDE_WARNINGS))
                {
                    SetPreload(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preload", "value"));
                    if(!fileParams.GoANodeBack()) return false;
                }
                else
                {
                    SetPreload(i_step, i_ptor, false);
                }

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                SetPreprocess(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preprocess", "value"));
                if(!SetPreprocessMethod(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("PreprocessMethod", "value"))) return false;
                if(!NeedsPreprocessing(i_step, i_ptor))
                {
                    asLogError(_("Preprocessing option is not coherent."));
                    return false;
                }

                if(!fileParams.GoToFirstNodeWithPath("SubData")) return false;
                int i_dataset = 0;
                bool preprocessDataOver = false;
                while(!preprocessDataOver)
                {
                    if(!SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessDatasetId", "value"))) return false;
                    if(!SetPreprocessDataIdVector(i_step, i_ptor, i_dataset, GetFileParamStringVector(fileParams, "PreprocessDataId"))) return false;
                    if(!SetPreprocessLevelVector(i_step, i_ptor, i_dataset, GetFileParamFloatVector(fileParams, "PreprocessLevel"))) return false;
                    if(!SetPreprocessTimeHoursVector(i_step, i_ptor, i_dataset, GetFileParamDoubleVector(fileParams, "PreprocessTimeHours"))) return false;

                    if(fileParams.GoToNextSameNode())
                    {
                        i_dataset++;
                    }
                    else
                    {
                        preprocessDataOver = true;
                    }
                }

                // Set data for predictor
                SetPredictorDatasetId(i_step, i_ptor, "mix");
                SetPredictorDataId(i_step, i_ptor, "mix");
                SetPredictorLevel(i_step, i_ptor, 0);
                SetPredictorTimeHours(i_step, i_ptor, 0);

                if(!fileParams.GoANodeBack()) return false;
                if(!fileParams.GoANodeBack()) return false;
            }
            else
            {
                asThrowException(_("Preprocessing option not correctly defined in the parameters file."));
            }

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area Moving")) return false;
            if(!SetPredictorGridType(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("GridType", "value", "Regular"))) return false;
            if(!SetPredictorUminVector(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "Umin"))) return false;
            if(!SetPredictorUptsnbVector(i_step, i_ptor, GetFileParamIntVector(fileParams, "Uptsnb"))) return false;
            if(!SetPredictorUstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ustep", "value"))) return false;
            double Ushift = fmod(GetPredictorUminVector(i_step, i_ptor)[0], GetPredictorUstep(i_step, i_ptor));
            if (Ushift<0) Ushift += GetPredictorUstep(i_step, i_ptor);
            if(!SetPredictorUshift(i_step, i_ptor, Ushift)) return false;
            if(!SetPredictorVminVector(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "Vmin"))) return false;
            if(!SetPredictorVptsnbVector(i_step, i_ptor, GetFileParamIntVector(fileParams, "Vptsnb"))) return false;
            if(!SetPredictorVstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vstep", "value"))) return false;
            double Vshift = fmod(GetPredictorVminVector(i_step, i_ptor)[0], GetPredictorVstep(i_step, i_ptor));
            if (Vshift<0) Vshift += GetPredictorVstep(i_step, i_ptor);
            if(!SetPredictorVshift(i_step, i_ptor, Vshift)) return false;
            VectorInt uptsnbs = GetPredictorUptsnbVector(i_step, i_ptor);
            VectorInt vptsnbs = GetPredictorVptsnbVector(i_step, i_ptor);
            if (asTools::MinArray(&uptsnbs[0], &uptsnbs[uptsnbs.size()-1])<=1 || asTools::MinArray(&vptsnbs[0], &vptsnbs[vptsnbs.size()-1])<=1)
            {
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            }
            if (NeedsPreloading(i_step, i_ptor))
            {
                // Set maximum extent
                if(!SetPreloadUmin(i_step, i_ptor, GetPredictorUminVector(i_step, i_ptor)[0])) return false;
                if(!SetPreloadVmin(i_step, i_ptor, GetPredictorVminVector(i_step, i_ptor)[0])) return false;
                int Ubaseptsnb = abs(GetPredictorUminVector(i_step, i_ptor)[0]-GetPredictorUminVector(i_step, i_ptor)[GetPredictorUminVector(i_step, i_ptor).size()-1])/GetPredictorUstep(i_step, i_ptor);
                if(!SetPreloadUptsnb(i_step, i_ptor, Ubaseptsnb+GetPredictorUptsnbVector(i_step, i_ptor)[GetPredictorUptsnbVector(i_step, i_ptor).size()-1])) return false;
                int Vbaseptsnb = abs(GetPredictorVminVector(i_step, i_ptor)[0]-GetPredictorVminVector(i_step, i_ptor)[GetPredictorVminVector(i_step, i_ptor).size()-1])/GetPredictorVstep(i_step, i_ptor);
                if(!SetPreloadVptsnb(i_step, i_ptor, Vbaseptsnb+GetPredictorVptsnbVector(i_step, i_ptor)[GetPredictorVptsnbVector(i_step, i_ptor).size()-1])) return false;
            }
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Criteria")) return false;
            if(!SetPredictorCriteriaVector(i_step, i_ptor, GetFileParamStringVector(fileParams, "Criteria"))) return false;
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            if(!SetPredictorWeightVector(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Weight"))) return false;
            if(!fileParams.GoANodeBack()) return false;

            if(fileParams.GoToNextSameNode())
            {
                i_ptor++;
                AddPredictor(i_step);
                AddPredictorVect(m_StepsVect[i_step]);
            }
            else
            {
                dataOver = true;
            }
        }

        if(!fileParams.GoANodeBack()) return false;

        // Find the next analogs date block
        if (!fileParams.GoToNextSameNodeWithAttributeValue("name", "Analogs Dates", asHIDE_WARNINGS)) break;

        i_step++;
    }
    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Values process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Values")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Predictand")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Database")) return false;
    if(!SetPredictandStationsIdVector(GetFileParamIntVector(fileParams, "PredictandStationId"))) return false;
    if(!SetPredictandTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandTimeHours", "value", 0))) return false;
    if(!fileParams.GoANodeBack()) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Forecast Scores process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Forecast Scores")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method")) return false;
    if(!SetForecastScoreNameVector(GetFileParamStringVector(fileParams, "Name"))) return false;
    SetForecastScoreThreshold(fileParams.GetFirstElementAttributeValueFloat("Threshold", "value")); // optional
    SetForecastScorePercentile(fileParams.GetFirstElementAttributeValueFloat("Percentile", "value")); // optional
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
    if(!SetForecastScoreAnalogsNumberVector(GetFileParamIntVector(fileParams, "AnalogsNumber"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Postprocessing", asHIDE_WARNINGS))
    {
        SetForecastScorePostprocess(1);
        if(!SetForecastScorePostprocessMethod(fileParams.GetFirstElementAttributeValueText("Method", "value"))) return false;
        if(!SetForecastScorePostprocessDupliExpVector(GetFileParamFloatVector(fileParams, "DuplicationExponent"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        SetForecastScorePostprocessMethod(wxEmptyString);
        SetForecastScorePostprocessDupliExpVector(emptyVectFloat);
    }

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Forecast Score Final process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Forecast Score Final")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array")) return false;
    if(!SetForecastScoreTimeArrayModeVector(GetFileParamStringVector(fileParams, "Mode"))) return false;
    if(!SetForecastScoreTimeArrayDateVector(GetFileParamDoubleVector(fileParams, "Date"))) return false;
    if(!SetForecastScoreTimeArrayIntervalDaysVector(GetFileParamIntVector(fileParams, "IntervalDays"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.CheckDeprecatedChildNode("Validation")) return false;

    // Set sizes
    SetSizes();

    InitValues();

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    asLogMessage(_("Parameters file loaded."));

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

    m_TimeMinHours = minHour;
    m_TimeMaxHours = maxHour;

    return true;
}

void asParametersCalibration::InitValues()
{
    wxASSERT(m_PredictandStationsIdVect.size()>0);
    wxASSERT(m_TimeArrayAnalogsIntervalDaysVect.size()>0);
    wxASSERT(m_ForecastScoreVect.Name.size()>0);
    wxASSERT(m_ForecastScoreVect.AnalogsNumber.size()>0);
    wxASSERT(m_ForecastScoreVect.TimeArrayMode.size()>0);
    wxASSERT(m_ForecastScoreVect.TimeArrayDate.size()>0);
    wxASSERT(m_ForecastScoreVect.TimeArrayIntervalDays.size()>0);
    wxASSERT(m_ForecastScoreVect.PostprocessDupliExp.size()>0);

    // Initialize the parameters values with the first values of the vectors
    m_PredictandStationId = m_PredictandStationsIdVect[0];
    m_TimeArrayAnalogsIntervalDays = m_TimeArrayAnalogsIntervalDaysVect[0];
    SetForecastScoreName(m_ForecastScoreVect.Name[0]);
    SetForecastScoreAnalogsNumber(m_ForecastScoreVect.AnalogsNumber[0]);
    SetForecastScoreTimeArrayMode(m_ForecastScoreVect.TimeArrayMode[0]);
    SetForecastScoreTimeArrayDate(m_ForecastScoreVect.TimeArrayDate[0]);
    SetForecastScoreTimeArrayIntervalDays(m_ForecastScoreVect.TimeArrayIntervalDays[0]);
    SetForecastScorePostprocessDupliExp(m_ForecastScoreVect.PostprocessDupliExp[0]);

    for (int i=0; i<GetStepsNb(); i++)
    {
        SetAnalogsNumber(i, m_StepsVect[i].AnalogsNumber[0]);

        for (int j=0; j<GetPredictorsNb(i); j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                int subDataNb = m_StepsVect[i].Predictors[j].PreprocessDataId.size();
                wxASSERT(subDataNb>0);
                for (int k=0; k<subDataNb; k++)
                {
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessDataId.size()>0);
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessDataId[k].size()>0);
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessLevels.size()>0);
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessLevels[k].size()>0);
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessTimeHours.size()>0);
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessTimeHours[k].size()>0);
                    SetPreprocessDataId(i,j,k, m_StepsVect[i].Predictors[j].PreprocessDataId[k][0]);
                    SetPreprocessLevel(i,j,k, m_StepsVect[i].Predictors[j].PreprocessLevels[k][0]);
                    SetPreprocessTimeHours(i,j,k, m_StepsVect[i].Predictors[j].PreprocessTimeHours[k][0]);
                }
            }
            else
            {
                SetPredictorDataId(i,j, m_StepsVect[i].Predictors[j].DataId[0]);
                SetPredictorLevel(i,j, m_StepsVect[i].Predictors[j].Level[0]);
                SetPredictorTimeHours(i,j, m_StepsVect[i].Predictors[j].TimeHours[0]);
            }

            SetPredictorUmin(i,j, m_StepsVect[i].Predictors[j].Umin[0]);
            SetPredictorUptsnb(i,j, m_StepsVect[i].Predictors[j].Uptsnb[0]);
            SetPredictorVmin(i,j, m_StepsVect[i].Predictors[j].Vmin[0]);
            SetPredictorVptsnb(i,j, m_StepsVect[i].Predictors[j].Vptsnb[0]);
            SetPredictorCriteria(i,j, m_StepsVect[i].Predictors[j].Criteria[0]);
            SetPredictorWeight(i,j, m_StepsVect[i].Predictors[j].Weight[0]);
        }

    }

    // Fixes and checks
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}
