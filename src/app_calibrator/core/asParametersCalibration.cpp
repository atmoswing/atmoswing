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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
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

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Period"))
    {
        asLogError(_("The Period tag is no more valid. Please update the parameters file."));
        return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period"))
    {
        asLogError(_("The 'Archive Period' tag is missing"));
        return false;
    }
    SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"));
    SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"));
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Calibration Period"))
    {
        asLogError(_("The 'Calibration Period' tag is missing"));
        return false;
    }
    SetCalibrationYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"));
    SetCalibrationYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"));
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Validation Period"))
    {
        SetValidationYearsVector(GetFileParamIntVector(fileParams, "Years"));
        if(!fileParams.GoANodeBack()) return false;
    }

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Time Properties"))
    {
        SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
        SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        if(!fileParams.GoANodeBack()) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
        SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        if(!fileParams.GoANodeBack()) return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
    SetTimeArrayTargetMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"));
    SetTimeArrayTargetPredictandSerieName(fileParams.GetFirstElementAttributeValueText("PredictandSerieName", "value"));
    SetTimeArrayTargetPredictandMinThreshold(fileParams.GetFirstElementAttributeValueFloat("PredictandMinThreshold", "value"));
    SetTimeArrayTargetPredictandMaxThreshold(fileParams.GetFirstElementAttributeValueFloat("PredictandMaxThreshold", "value"));
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
    SetTimeArrayAnalogsMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"));
    SetTimeArrayAnalogsExcludeDays(fileParams.GetFirstElementAttributeValueInt("ExcludeDays", "value"));
    SetTimeArrayAnalogsIntervalDaysVector(GetFileParamIntVector(fileParams, "IntervalDays"));
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
        SetMethodName(i_step, fileParams.GetFirstElementAttributeValueText("MethodName", "value"));
        if(!fileParams.GoANodeBack()) return false;

        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
        SetAnalogsNumberVector(i_step, GetFileParamIntVector(fileParams, "AnalogsNumber"));
        if(!fileParams.GoANodeBack()) return false;

        // Get data
        if(!fileParams.GoToFirstNodeWithPath("Data")) return false;
        bool dataOver = false;
        int i_ptor = 0;
        while(!dataOver)
        {
            wxString predictorNature = fileParams.GetThisElementAttributeValueText("name", "value");

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

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preload"))
                {
                    SetPreload(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preload", "value"));
                    if(!fileParams.GoANodeBack()) return false;
                }
                else
                {
                    SetPreload(i_step, i_ptor, false);
                }

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data")) return false;
                SetPredictorDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"));
                SetPredictorDataIdVector(i_step, i_ptor, GetFileParamStringVector(fileParams, "DataId"));
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                SetPredictorLevelVector(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Level"));
                if (NeedsPreloading(i_step, i_ptor))
                {
                    SetPreloadLevels(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Level"));
                }
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Frame")) return false;
                SetPredictorDTimeHoursVector(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "DTimeHours"));
                SetPreloadDTimeHours(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "DTimeHours"));
                if(!fileParams.GoANodeBack()) return false;

            }
            else if(predictorNature.IsSameAs("Predictor Preprocessed", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(fileParams.GoToChildNodeWithAttributeValue("name", "Preload"))
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
                SetPreprocessMethod(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("PreprocessMethod", "value"));
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
                    SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessDatasetId", "value"));
                    SetPreprocessDataIdVector(i_step, i_ptor, i_dataset, GetFileParamStringVector(fileParams, "PreprocessDataId"));
                    SetPreprocessLevelVector(i_step, i_ptor, i_dataset, GetFileParamFloatVector(fileParams, "PreprocessLevel"));
                    SetPreprocessDTimeHoursVector(i_step, i_ptor, i_dataset, GetFileParamDoubleVector(fileParams, "PreprocessDTimeHours"));

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
                SetPredictorLevel(i_step, i_ptor, NaNFloat);
                SetPredictorDTimeHours(i_step, i_ptor, NaNDouble);

                if(!fileParams.GoANodeBack()) return false;
                if(!fileParams.GoANodeBack()) return false;
            }
            else
            {
                asThrowException(_("Preprocessing option not correctly defined in the parameters file."));
            }

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area Moving")) return false;
            SetPredictorGridType(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("GridType", "value", "Regular"));
            SetPredictorUminVector(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "Umin"));
            SetPredictorUptsnbVector(i_step, i_ptor, GetFileParamIntVector(fileParams, "Uptsnb"));
            SetPredictorUstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ustep", "value"));
            double Ushift = fmod(GetPredictorUminVector(i_step, i_ptor)[0], GetPredictorUstep(i_step, i_ptor));
            if (Ushift<0) Ushift += GetPredictorUstep(i_step, i_ptor);
            SetPredictorUshift(i_step, i_ptor, Ushift);
            SetPredictorVminVector(i_step, i_ptor, GetFileParamDoubleVector(fileParams, "Vmin"));
            SetPredictorVptsnbVector(i_step, i_ptor, GetFileParamIntVector(fileParams, "Vptsnb"));
            SetPredictorVstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vstep", "value", 0));
            double Vshift = fmod(GetPredictorVminVector(i_step, i_ptor)[0], GetPredictorVstep(i_step, i_ptor));
            if (Vshift<0) Vshift += GetPredictorVstep(i_step, i_ptor);
            SetPredictorVshift(i_step, i_ptor, Vshift);
            VectorInt uptsnbs = GetPredictorUptsnbVector(i_step, i_ptor);
            VectorInt vptsnbs = GetPredictorVptsnbVector(i_step, i_ptor);
            if (asTools::MinArray(&uptsnbs[0], &uptsnbs[uptsnbs.size()-1])<=1 || asTools::MinArray(&vptsnbs[0], &vptsnbs[vptsnbs.size()-1])<=1)
            {
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            }
            if (NeedsPreloading(i_step, i_ptor))
            {
                // Set maximum extent
                SetPreloadUmin(i_step, i_ptor, GetPredictorUminVector(i_step, i_ptor)[0]);
                SetPreloadVmin(i_step, i_ptor, GetPredictorVminVector(i_step, i_ptor)[0]);
                int Ubaseptsnb = abs(GetPredictorUminVector(i_step, i_ptor)[0]-GetPredictorUminVector(i_step, i_ptor)[GetPredictorUminVector(i_step, i_ptor).size()-1])/GetPredictorUstep(i_step, i_ptor);
                SetPreloadUptsnb(i_step, i_ptor, Ubaseptsnb+GetPredictorUptsnbVector(i_step, i_ptor)[GetPredictorUptsnbVector(i_step, i_ptor).size()-1]);
                int Vbaseptsnb = abs(GetPredictorVminVector(i_step, i_ptor)[0]-GetPredictorVminVector(i_step, i_ptor)[GetPredictorVminVector(i_step, i_ptor).size()-1])/GetPredictorVstep(i_step, i_ptor);
                SetPreloadVptsnb(i_step, i_ptor, Vbaseptsnb+GetPredictorVptsnbVector(i_step, i_ptor)[GetPredictorVptsnbVector(i_step, i_ptor).size()-1]);
            }
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Criteria")) return false;
            SetPredictorCriteriaVector(i_step, i_ptor, GetFileParamStringVector(fileParams, "Criteria"));
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            SetPredictorWeightVector(i_step, i_ptor, GetFileParamFloatVector(fileParams, "Weight"));
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
        if (!fileParams.GoToNextSameNodeWithAttributeValue("name", "Analogs Dates")) break;

        i_step++;
    }
    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Values process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Values")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Predictand")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Database")) return false;
    SetPredictandStationsIdVector(GetFileParamIntVector(fileParams, "PredictandStationId"));
    SetPredictandDTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandDTimeHours", "value", 0.0));
    if(!fileParams.GoANodeBack()) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Forecast Scores process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Forecast Scores")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method")) return false;
    SetForecastScoreNameVector(GetFileParamStringVector(fileParams, "Name"));
    SetForecastScoreThreshold(fileParams.GetFirstElementAttributeValueFloat("Threshold", "value", NaNFloat));
    SetForecastScorePercentile(fileParams.GetFirstElementAttributeValueFloat("Percentile", "value", NaNFloat));
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
    SetForecastScoreAnalogsNumberVector(GetFileParamIntVector(fileParams, "AnalogsNumber"));
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Postprocessing"))
    {
        SetForecastScorePostprocess(1);
        SetForecastScorePostprocessMethod(fileParams.GetFirstElementAttributeValueText("Method", "value"));
        SetForecastScorePostprocessDupliExpVector(GetFileParamFloatVector(fileParams, "DuplicationExponent"));
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
    SetForecastScoreTimeArrayModeVector(GetFileParamStringVector(fileParams, "Mode"));
    SetForecastScoreTimeArrayDateVector(GetFileParamDoubleVector(fileParams, "Date"));
    SetForecastScoreTimeArrayIntervalDaysVector(GetFileParamIntVector(fileParams, "IntervalDays"));
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Validation"))
    {
        asLogError(_("The validation in the final score calculation is no more valid. Please update the parameters file."));
        return false;
    }

    // Set sizes
    SetSizes();

    InitValues();

    // Fixes
    FixTimeShift();
    FixWeights();
    FixCoordinates();

    asLogMessage(_("Parameters file loaded."));

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
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessDTimeHours.size()>0);
                    wxASSERT(m_StepsVect[i].Predictors[j].PreprocessDTimeHours[k].size()>0);
                    SetPreprocessDataId(i,j,k, m_StepsVect[i].Predictors[j].PreprocessDataId[k][0]);
                    SetPreprocessLevel(i,j,k, m_StepsVect[i].Predictors[j].PreprocessLevels[k][0]);
                    SetPreprocessDTimeHours(i,j,k, m_StepsVect[i].Predictors[j].PreprocessDTimeHours[k][0]);
                }
            }
            else
            {
                SetPredictorDataId(i,j, m_StepsVect[i].Predictors[j].DataId[0]);
                SetPredictorLevel(i,j, m_StepsVect[i].Predictors[j].Level[0]);
                SetPredictorDTimeHours(i,j, m_StepsVect[i].Predictors[j].DTimeHours[0]);
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
    FixTimeShift();
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}
