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

    if(!fileParams.CheckDeprecatedChildNode("LeadTime")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Lead Time")) return false;
    if(!SetLeadTimeDaysVector(GetFileParamInt(fileParams, "LeadTimeDays"))) return false;
    if(!fileParams.GoANodeBack()) return false;
    
    if(!fileParams.CheckDeprecatedChildNode("ArchivePeriod")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period")) return false;
    if(!SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
    if(!SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Time Properties", asHIDE_WARNINGS))
    {
        if(!SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!SetPredictandTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandTimeHours", "value", 0.0))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }
    else
    {
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
        if(!SetTimeArrayTargetTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!SetPredictandTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandTimeHours", "value", 0.0))) return false;
        if(!fileParams.GoANodeBack()) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
        if(!SetTimeArrayAnalogsTimeStepHours(fileParams.GetFirstElementAttributeValueDouble("TimeStepHours", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;
    }

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
    if(!SetTimeArrayAnalogsMode(fileParams.GetFirstElementAttributeValueText("TimeArrayMode", "value"))) return false;
    if(!SetTimeArrayAnalogsExcludeDays(fileParams.GetFirstElementAttributeValueInt("ExcludeDays", "value"))) return false;
    if(!SetTimeArrayAnalogsIntervalDays(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "value"))) return false;
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
        if(!SetAnalogsNumberLeadTimeVector(i_step, GetFileParamInt(fileParams, "AnalogsNumber"))) return false;
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
                if(!SetPredictorRealtimeDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"))) return false;
                if(!SetPredictorRealtimeDataId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DataId", "value"))) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data Archive")) return false;
                if(!SetPredictorArchiveDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"))) return false;
                if(!SetPredictorArchiveDataId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DataId", "value"))) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                if(!SetPredictorLevel(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Level", "value"))) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Frame")) return false;
                if(!SetPredictorTimeHours(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("TimeHours", "value"))) return false;
                if(!fileParams.GoANodeBack()) return false;

            }
            else if(predictorNature.IsSameAs("Predictor Preprocessed", false))
            {
                if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                SetPreprocess(i_step, i_ptor, fileParams.GetFirstElementAttributeValueBool("Preprocess", "value"));
                if(!SetPreprocessMethod(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("PreprocessMethod", "value"))) return false;
                if(!NeedsPreprocessing(i_step, i_ptor))
                {
                    asLogError(_("Preprocessing option is not coherent."));
                }

                if(!fileParams.GoToFirstNodeWithPath("SubData")) return false;
                int i_dataset = 0;
                bool preprocessDataOver = false;
                while(!preprocessDataOver)
                {
                    if(!SetPreprocessRealtimeDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessRealtimeDatasetId", "value"))) return false;
                    if(!SetPreprocessRealtimeDataId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessRealtimeDataId", "value"))) return false;
                    if(!SetPreprocessArchiveDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessArchiveDatasetId", "value"))) return false;
                    if(!SetPreprocessArchiveDataId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessArchiveDataId", "value"))) return false;
                    if(!SetPreprocessLevel(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueFloat("PreprocessLevel", "value"))) return false;
                    if(!SetPreprocessTimeHours(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueDouble("PreprocessTimeHours", "value"))) return false;

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
                    SetPredictorTimeHours(i_step, i_ptor, GetPreprocessTimeHours(i_step, i_ptor, 0));
                }
                else
                {
                    SetPredictorDatasetId(i_step, i_ptor, GetPreprocessDatasetId(i_step, i_ptor, 0));
                    SetPredictorDataId(i_step, i_ptor, GetPreprocessDataId(i_step, i_ptor, 0));
                    SetPredictorLevel(i_step, i_ptor, GetPreprocessLevel(i_step, i_ptor, 0));
                    SetPredictorTimeHours(i_step, i_ptor, GetPreprocessTimeHours(i_step, i_ptor, 0));
                }
                if(!fileParams.GoANodeBack()) return false;
                if(!fileParams.GoANodeBack()) return false;
            }
            else
            {
                asThrowException(_("Preprocessing option not correctly defined in the parameters file."));
            }

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area")) return false;
            if(!SetPredictorGridType(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("GridType", "value", "Regular"))) return false;
            if(!SetPredictorUmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Umin", "value"))) return false;
            if(!SetPredictorUptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Uptsnb", "value"))) return false;
            if (GetPredictorUptsnb(i_step, i_ptor)==0) SetPredictorUptsnb(i_step, i_ptor, 1);
            if(!SetPredictorUstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ustep", "value"))) return false;
            if(!SetPredictorVmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "value"))) return false;
            if(!SetPredictorVptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueInt("Vptsnb", "value"))) return false;
            if (GetPredictorVptsnb(i_step, i_ptor)==0) SetPredictorVptsnb(i_step, i_ptor, 1);
            if(!SetPredictorVstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vstep", "value"))) return false;

            if (GetPredictorUptsnb(i_step, i_ptor)==1 || GetPredictorVptsnb(i_step, i_ptor)==1)
            {
                SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
            }
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Criteria")) return false;
            if(!SetPredictorCriteria(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("Criteria", "value"))) return false;
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            if(!SetPredictorWeight(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Weight", "value"))) return false;
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

    if(fileParams.GoToChildNodeWithAttributeValue("name", "Predictand", asHIDE_WARNINGS))
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
    FixTimeLimits();
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
    FixWeights();
    FixCoordinates();
    FixAnalogsNb();
}
