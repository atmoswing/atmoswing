/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asParametersForecast.h"

#include <asFileParametersForecast.h>

asParametersForecast::asParametersForecast()
:
asParameters()
{
    //ctor
}

asParametersForecast::~asParametersForecast()
{
    //dtor
}

void asParametersForecast::AddStep()
{
    asParameters::AddStep();
    ParamsStepForecast stepForecast;
    stepForecast.AnalogsNumberLeadTime.push_back(0);
    AddPredictorForecast(stepForecast);
    m_StepsForecast.push_back(stepForecast);
}

void asParametersForecast::AddPredictorForecast(ParamsStepForecast &step)
{
    ParamsPredictorForecast predictor;

    predictor.ArchiveDatasetId = wxEmptyString;
    predictor.ArchiveDataId = wxEmptyString;
    predictor.RealtimeDatasetId = wxEmptyString;
    predictor.RealtimeDataId = wxEmptyString;

    step.Predictors.push_back(predictor);
}

VectorInt asParametersForecast::GetFileParamInt(asFileParametersForecast &fileParams, const wxString &tag)
{
    VectorInt vect;
    wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
    vect = BuildVectorInt(txt);

    return vect;
}

VectorFloat asParametersForecast::GetFileParamFloat(asFileParametersForecast &fileParams, const wxString &tag)
{
    VectorFloat vect;
    wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
    vect = BuildVectorFloat(txt);

    return vect;
}

VectorDouble asParametersForecast::GetFileParamDouble(asFileParametersForecast &fileParams, const wxString &tag)
{
    VectorDouble vect;
    wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
    vect = BuildVectorDouble(txt);

    return vect;
}

VectorString asParametersForecast::GetFileParamString(asFileParametersForecast &fileParams, const wxString &tag)
{
    VectorString vect;
    wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
    vect = BuildVectorString(txt);

    return vect;
}

bool asParametersForecast::LoadFromFile(const wxString &filePath)
{
    asLogMessage(_("Loading parameters file."));

    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersForecast fileParams(filePath, asFile::ReadOnly);
    if(!fileParams.Open()) return false;

    if(!fileParams.GoToRootElement()) return false;

    // Get general parameters
    if(!fileParams.GoToFirstNodeWithPath("General")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "LeadTime"))
    {
        asLogError(_("The LeadTime tag is no more valid. Please update the parameters file."));
        return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Lead Time")) return false;
    SetLeadTimeDaysVector(GetFileParamInt(fileParams, "LeadTimeDays"));
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "ArchivePeriod"))
    {
        asLogError(_("The ArchivePeriod tag is no more valid. Please update the parameters file."));
        return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period")) return false;
    SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"));
    SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"));
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Time Properties"))
    {
        SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        SetPredictandDTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandDTimeHours", "value", 0.0));
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
        SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        SetPredictandDTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandDTimeHours", "value", 0.0));
        if(!fileParams.GoANodeBack()) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
        SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"));
        if(!fileParams.GoANodeBack()) return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
    SetTimeArrayAnalogsMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"));
    SetTimeArrayAnalogsExcludeDays(fileParams.GetFirstElementAttributeValueInt("ExcludeDays", "value"));
    SetTimeArrayAnalogsIntervalDays(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "value"));
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
        SetAnalogsNumberLeadTimeVector(i_step, GetFileParamInt(fileParams, "AnalogsNumber"));
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
                }
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data Realtime")) return false;
                SetPredictorRealtimeDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"));
                SetPredictorRealtimeDataId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DataId", "value"));
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data Archive")) return false;
                SetPredictorArchiveDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"));
                SetPredictorArchiveDataId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DataId", "value"));
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                SetPredictorLevel(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Level", "value"));
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Frame")) return false;
                SetPredictorDTimeHours(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("DTimeHours", "value"));
                if(!fileParams.GoANodeBack()) return false;

            }
            else if(predictorNature.IsSameAs("Predictor Preprocessed", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                SetPreprocess(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preprocess", "value"));
                SetPreprocessMethod(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("PreprocessMethod", "value"));
                if(!NeedsPreprocessing(i_step, i_ptor))
                {
                    asLogError(_("Preprocessing option is not coherent."));
                }

                if(!fileParams.GoToFirstNodeWithPath("SubData")) return false;
                int i_dataset = 0;
                bool preprocessDataOver = false;
                while(!preprocessDataOver)
                {
                    SetPreprocessRealtimeDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessRealtimeDatasetId", "value"));
                    SetPreprocessRealtimeDataId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessRealtimeDataId", "value"));
                    SetPreprocessArchiveDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessArchiveDatasetId", "value"));
                    SetPreprocessArchiveDataId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessArchiveDataId", "value"));
                    SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueFloat("PreprocessLevel", "value"));
                    SetPreprocessDTimeHours(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessDTimeHours", "value"));

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
                if(i_dataset>0)
                {
                    SetPredictorDatasetId(i_step, i_ptor, "mix");
                    SetPredictorDataId(i_step, i_ptor, "mix");
                    SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
                    SetPredictorDTimeHours(i_step, i_ptor, GetPreprocessDTimeHours(i_step, i_ptor, 0));
                }
                else
                {
                    SetPredictorDatasetId(i_step, i_ptor, GetPreprocessDatasetId(i_step, i_ptor, 0));
                    SetPredictorDataId(i_step, i_ptor, GetPreprocessDataId(i_step, i_ptor, 0));
                    SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
                    SetPredictorDTimeHours(i_step, i_ptor, GetPreprocessDTimeHours(i_step, i_ptor, 0));
                }
                if(!fileParams.GoANodeBack()) return false;
                if(!fileParams.GoANodeBack()) return false;
            }
            else
            {
                asThrowException(_("Preprocessing option not correctly defined in the parameters file."));
            }

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area")) return false;
            SetPredictorGridType(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("GridType", "value", "Regular"));
            SetPredictorUmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Umin", "value"));
            SetPredictorUptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Uptsnb", "value"));
            if (GetPredictorUptsnb(i_step, i_ptor)==0) SetPredictorUptsnb(i_step, i_ptor, 1);
            SetPredictorUstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ustep", "value"));
            SetPredictorVmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "value"));
            SetPredictorVptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Vptsnb", "value"));
            if (GetPredictorVptsnb(i_step, i_ptor)==0) SetPredictorVptsnb(i_step, i_ptor, 1);
            SetPredictorVstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vstep", "value"));

            if (GetPredictorUptsnb(i_step, i_ptor)==1 || GetPredictorVptsnb(i_step, i_ptor)==1)
            {
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            }
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Criteria")) return false;
            SetPredictorCriteria(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("Criteria", "value"));
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            SetPredictorWeight(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "value"));
            if(!fileParams.GoANodeBack()) return false;

            if(fileParams.GoToNextSameNode())
            {
                i_ptor++;
                AddPredictor(i_step);
                AddPredictorForecast(m_StepsForecast[i_step]);
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

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Predictand"))
	{
		if(!fileParams.GoToChildNodeWithAttributeValue("name", "Database"))
		{
			// May do something here

			if(!fileParams.GoANodeBack()) return false;
		}
		if(!fileParams.GoANodeBack()) return false;
	}

    if(!fileParams.GoANodeBack()) return false;

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

void asParametersForecast::InitValues()
{
    // Initialize the parameters values with the first values of the vectors
    for (int i=0; i<GetStepsNb(); i++)
    {
        SetAnalogsNumber(i, m_StepsForecast[i].AnalogsNumberLeadTime[0]);
    }

    // Fixes and checks
    FixTimeShift();
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}
