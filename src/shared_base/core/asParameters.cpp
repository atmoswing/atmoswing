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

#include "asParameters.h"

#include "asFileParametersStandard.h"
#include "asFileAscii.h"


asParameters::asParameters()
{
    m_DateProcessed = asTime::GetStringTime(asTime::NowTimeStruct(asLOCAL));
    m_ArchiveStart = 0;
    m_ArchiveEnd = 0;
    m_TimeMinHours = 0;
    m_TimeMaxHours = 0;
    m_TimeArrayTargetMode = "Simple";
    m_TimeArrayTargetTimeStepHours = 0;
    m_TimeArrayAnalogsMode = "DaysInterval";
    m_TimeArrayAnalogsTimeStepHours = 0;
    m_TimeArrayAnalogsExcludeDays = 0;
    m_TimeArrayAnalogsIntervalDays = 0;
    m_PredictandTimeHours = 0;
    m_PredictandParameter = (DataParameter)0;
    m_PredictandTemporalResolution = (DataTemporalResolution)0;
    m_PredictandSpatialAggregation = (DataSpatialAggregation)0;
    m_PredictandDatasetId = wxEmptyString;
    m_StepsNb = 0;
    m_TimeArrayTargetPredictandMinThreshold = 0;
    m_TimeArrayTargetPredictandMaxThreshold = 0;
}

asParameters::~asParameters()
{
    //dtor
}

bool asParameters::IsOk()
{
    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if (GetPredictorUptsnb(i_step, i_ptor)<=0)
            {
                asLogError("The number of points on the U axis must be >= 1.");
                return false;
            }
            if (GetPredictorVptsnb(i_step, i_ptor)<=0)
            {
                asLogError("The number of points on the V axis must be >= 1.");
                return false;
            }
            if (GetPredictorGridType(i_step, i_ptor).IsSameAs("Regular") && GetPredictorUstep(i_step, i_ptor)<=0)
            {
                asLogError("The step length on the U axis cannot be null for regular grids.");
                return false;
            }
            if (GetPredictorGridType(i_step, i_ptor).IsSameAs("Regular") && GetPredictorVstep(i_step, i_ptor)<=0)
            {
                asLogError("The step length on the V axis cannot be null for regular grids.");
                return false;
            }

// TODO (phorton#1#): Check every parameter.
        }
    }

    return true;
}

void asParameters::AddStep()
{
    ParamsStep step;

    step.MethodName = wxEmptyString;
    step.AnalogsNumber = 0;

    AddPredictor(step);

    m_Steps.push_back(step);
    SetSizes();
}

void asParameters::AddPredictor()
{
    AddPredictor(m_Steps[m_Steps.size()-1]);
}

void asParameters::AddPredictor(ParamsStep &step)
{
    ParamsPredictor predictor;

    predictor.DatasetId = wxEmptyString;
    predictor.DataId = wxEmptyString;
    predictor.Preload = false;
    predictor.PreloadUmin = 0;
    predictor.PreloadUptsnb = 0;
    predictor.PreloadVmin = 0;
    predictor.PreloadVptsnb = 0;
    predictor.Preprocess = false;
    predictor.PreprocessMethod = wxEmptyString;
    predictor.Level = 0;
    predictor.Umin = 0;
    predictor.Uptsnb = 1;
    predictor.Ustep = 0;
    predictor.Ushift = 0;
    predictor.Vmin = 0;
    predictor.Vptsnb = 1;
    predictor.Vstep = 0;
    predictor.Vshift = 0;
    predictor.FlatAllowed = asFLAT_FORBIDDEN;
    predictor.TimeHours = 0;
    predictor.Criteria = wxEmptyString;
    predictor.Weight = 1;

    step.Predictors.push_back(predictor);
    SetSizes();
}

void asParameters::AddPredictor(int i_step)
{
    ParamsPredictor predictor;

    predictor.DatasetId = wxEmptyString;
    predictor.DataId = wxEmptyString;
    predictor.Preload = false;
    predictor.PreloadUmin = 0;
    predictor.PreloadUptsnb = 0;
    predictor.PreloadVmin = 0;
    predictor.PreloadVptsnb = 0;
    predictor.Preprocess = false;
    predictor.PreprocessMethod = wxEmptyString;
    predictor.Level = 0;
    predictor.GridType = wxEmptyString;
    predictor.Umin = 0;
    predictor.Uptsnb = 1;
    predictor.Ustep = 0;
    predictor.Ushift = 0;
    predictor.Vmin = 0;
    predictor.Vptsnb = 1;
    predictor.Vstep = 0;
    predictor.Vshift = 0;
    predictor.FlatAllowed = asFLAT_FORBIDDEN;
    predictor.TimeHours = 0;
    predictor.Criteria = wxEmptyString;
    predictor.Weight = 1;

    m_Steps[i_step].Predictors.push_back(predictor);
    SetSizes();
}

void asParameters::SetSizes()
{
    m_StepsNb = m_Steps.size();
    m_PredictorsNb.resize(m_StepsNb);
    for (int i=0; i<m_StepsNb; i++)
    {
        m_PredictorsNb[i] = m_Steps[i].Predictors.size();
    }
}

bool asParameters::FixAnalogsNb()
{
    // Check analogs number coherence
    int analogsNb = GetAnalogsNumber(0);
    for (unsigned int i_step=1; i_step<m_Steps.size(); i_step++)
    {
        if(GetAnalogsNumber(i_step)>analogsNb)
        {
            SetAnalogsNumber(i_step, analogsNb);
        }
        else
        {
            analogsNb = GetAnalogsNumber(i_step);
        }
    }

    return true;
}

VectorInt asParameters::BuildVectorInt(int min, int max, int step)
{
    int stepsnb = 1+(max-min)/step;
    VectorInt vect(stepsnb);
    for(int i=0; i<stepsnb; i++)
    {
        vect[i] = min+i*step;
    }

    return vect;
}

VectorInt asParameters::BuildVectorInt(wxString str)
{
    VectorInt vect;
    wxChar separator = ',';
    while(str.Find(separator)!=wxNOT_FOUND)
    {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        double val;
        strbefore.ToDouble(&val);
        int valint = (int) val;
        vect.push_back(valint);
    }
    if(!str.IsEmpty())
    {
        double val;
        str.ToDouble(&val);
        int valint = (int) val;
        vect.push_back(valint);
    }

    return vect;
}

VectorFloat asParameters::BuildVectorFloat(float min, float max, float step)
{
    int stepsnb = 1+(max-min)/step;
    VectorFloat vect(stepsnb);
    for(int i=0; i<stepsnb; i++)
    {
        vect[i] = min+(float)i*step;
    }

    return vect;
}

VectorFloat asParameters::BuildVectorFloat(wxString str)
{
    VectorFloat vect;
    wxChar separator = ',';
    while(str.Find(separator)!=wxNOT_FOUND)
    {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        double val;
        strbefore.ToDouble(&val);
        float valfloat = (float) val;
        vect.push_back(valfloat);
    }
    if(!str.IsEmpty())
    {
        double val;
        str.ToDouble(&val);
        float valfloat = (float) val;
        vect.push_back(valfloat);
    }

    return vect;
}

VectorDouble asParameters::BuildVectorDouble(double min, double max, double step)
{
    int stepsnb = 1+(max-min)/step;
    VectorDouble vect(stepsnb);
    for(int i=0; i<stepsnb; i++)
    {
        vect[i] = min+(double)i*step;
    }

    return vect;
}

VectorDouble asParameters::BuildVectorDouble(wxString str)
{
    VectorDouble vect;
    wxChar separator = ',';
    while(str.Find(separator)!=wxNOT_FOUND)
    {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        double val;
        strbefore.ToDouble(&val);
        vect.push_back(val);
    }
    if(!str.IsEmpty())
    {
        double val;
        str.ToDouble(&val);
        vect.push_back(val);
    }

    return vect;
}

VectorString asParameters::BuildVectorString(wxString str)
{
    VectorString vect;
    wxChar separator = ',';
    while(str.Find(separator)!=wxNOT_FOUND)
    {
        wxString strbefore = str.BeforeFirst(separator);
        str = str.AfterFirst(separator);
        vect.push_back(strbefore);
    }
    if(!str.IsEmpty())
    {
        vect.push_back(str);
    }

    return vect;
}

bool asParameters::LoadFromFile(const wxString &filePath)
{
    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersStandard fileParams(filePath, asFile::ReadOnly);
    if(!fileParams.Open()) return false;
    if(!fileParams.GoToRootElement()) return false;

    // Get general parameters
    if(!fileParams.GoToFirstNodeWithPath("General")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.CheckDeprecatedChildNode("Period")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period")) return false;
    wxString archiveStart = fileParams.GetFirstElementAttributeValueText("Start", "value");
    wxString archiveEnd = fileParams.GetFirstElementAttributeValueText("End", "value");
    if (!archiveStart.IsEmpty() && !archiveEnd.IsEmpty())
    {
        SetArchiveStart(archiveStart);
        SetArchiveEnd(archiveEnd);
    }
    else
    {
        if(!SetArchiveYearStart(fileParams.GetFirstElementAttributeValueInt("YearStart", "value"))) return false;
        if(!SetArchiveYearEnd(fileParams.GetFirstElementAttributeValueInt("YearEnd", "value"))) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

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
    if(!SetTimeArrayAnalogsIntervalDays(fileParams.GetFirstElementAttributeValueInt("IntervalDays", "value"))) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Get Analogs Dates processs
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
        if(!SetAnalogsNumber(i_step, fileParams.GetFirstElementAttributeValueInt("AnalogsNumber", "value"))) return false;
        if(!fileParams.GoANodeBack()) return false;

        // Get first data
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

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data")) return false;
                if(!SetPredictorDatasetId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DatasetId", "value"))) return false;
                if(!SetPredictorDataId(i_step, i_ptor, fileParams.GetFirstElementAttributeValueText("DataId", "value"))) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                if(!SetPredictorLevel(i_step, i_ptor, fileParams.GetFirstElementAttributeValueFloat("Level", "value"))) return false;
                if(!fileParams.GoANodeBack()) return false;

                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time frame")) return false;
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
                    if(!SetPreprocessDatasetId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessDatasetId", "value"))) return false;
                    if(!SetPreprocessDataId(i_step, i_ptor, i_dataset, fileParams.GetFirstElementAttributeValueText("PreprocessDataId", "value"))) return false;
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
                    SetPredictorLevel(i_step, i_ptor, -1);
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
            if(!SetPredictorUptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Uptsnb", "value"))) return false;
            if (GetPredictorUptsnb(i_step, i_ptor)==0) SetPredictorUptsnb(i_step, i_ptor, 1);
            if(!SetPredictorUstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Ustep", "value"))) return false;
            double Ushift = fmod(GetPredictorUmin(i_step, i_ptor), GetPredictorUstep(i_step, i_ptor));
            if (Ushift<0) Ushift += GetPredictorUstep(i_step, i_ptor);
            if(!SetPredictorUshift(i_step, i_ptor, Ushift)) return false;
            if(!SetPredictorVmin(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vmin", "value"))) return false;
            if(!SetPredictorVptsnb(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vptsnb", "value"))) return false;
            if (GetPredictorVptsnb(i_step, i_ptor)==0) SetPredictorVptsnb(i_step, i_ptor, 1);
            if(!SetPredictorVstep(i_step, i_ptor, fileParams.GetFirstElementAttributeValueDouble("Vstep", "value", 0))) return false;
            double Vshift = fmod(GetPredictorVmin(i_step, i_ptor), GetPredictorVstep(i_step, i_ptor));
            if (Vshift<0) Vshift += GetPredictorVstep(i_step, i_ptor);
            if(!SetPredictorVshift(i_step, i_ptor, Vshift)) return false;
            if (GetPredictorUptsnb(i_step, i_ptor)==1 || GetPredictorVptsnb(i_step, i_ptor)==1) SetPredictorFlatAllowed(i_step, i_ptor, asFLAT_ALLOWED);
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
                AddPredictor(m_Steps[i_step]);
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

    // Get first Analogs Dates process
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Values")) return false;
    if(!fileParams.GoToFirstNodeWithPath("Options")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Predictand")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Database")) return false;
    if(!SetPredictandStationIds(GetFileStationIds(fileParams.GetFirstElementAttributeValueText("PredictandStationId", "value")))) return false;
    if(!SetPredictandTimeHours(fileParams.GetFirstElementAttributeValueDouble("PredictandTimeHours", "value", 0.0))) return false;
    if(!fileParams.GoANodeBack()) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // Set sizes
    SetSizes();

    // Fixes
    FixTimeLimits();
    FixWeights();
    FixCoordinates();

    // Check
    if (!IsOk()) return false;

    return true;
}

VectorInt asParameters::GetFileStationIds(wxString stationIdsString)
{
    // Trim
    stationIdsString.Trim(true);
    stationIdsString.Trim(false);

    VectorInt ids;

    if (stationIdsString.IsEmpty())
    {
        asLogError(_("The station ID was not provided."));
        return ids;
    }

    // Multivariate
    if (stationIdsString.SubString(0, 0).IsSameAs("(") || stationIdsString.SubString(0, 1).IsSameAs("'("))
    {
        wxString subStr = wxEmptyString;
        if (stationIdsString.SubString(0, 0).IsSameAs("("))
        {
            subStr = stationIdsString.SubString(1, stationIdsString.Len()-1);
        }
        else
        {
            subStr = stationIdsString.SubString(2, stationIdsString.Len()-1);
        }

        // Check that it contains only 1 opening bracket
        if (subStr.Find("(") != wxNOT_FOUND)
        {
            asLogError(_("The format of the station ID is not correct (more than one opening bracket)."));
            return ids;
        }

        // Check that it contains 1 closing bracket at the end
        if (subStr.Find(")") != subStr.size()-1 && subStr.Find(")'") != subStr.size()-2)
        {
            asLogError(_("The format of the station ID is not correct (location of the closing bracket)."));
            return ids;
        }

        // Extract content
        wxChar separator = ',';
        while(subStr.Find(separator)!=wxNOT_FOUND)
        {
            wxString strBefore = subStr.BeforeFirst(separator);
            subStr = subStr.AfterFirst(separator);
            int id = wxAtoi(strBefore);
            ids.push_back(id);
        }
        if(!subStr.IsEmpty())
        {
            int id = wxAtoi(subStr);
            ids.push_back(id);
        }
    }
    else
    {
        // Check for single value
        if (stationIdsString.Find("(") != wxNOT_FOUND || stationIdsString.Find(")") != wxNOT_FOUND || stationIdsString.Find(",") != wxNOT_FOUND)
        {
            asLogError(_("The format of the station ID is not correct (should be only digits)."));
            return ids;
        }
        int id = wxAtoi(stationIdsString);
        ids.push_back(id);
    }

    return ids;
}

wxString asParameters::GetPredictandStationIdsString()
{
    wxString Ids;

    if (m_PredictandStationIds.size()==1)
    {
        Ids << m_PredictandStationIds[0];
    }
    else
    {
        Ids = "(";

        for (int i=0; i<m_PredictandStationIds.size(); i++)
        {
            Ids << m_PredictandStationIds[i];

            if (i<m_PredictandStationIds.size()-1)
            {
                Ids.Append(",");
            }
        }

        Ids.Append(")");
    }

    return Ids;
}

bool asParameters::FixTimeLimits()
{
    SetSizes();

    double minHour = 1000.0, maxHour = -1000.0;
    for(int i=0;i<m_StepsNb;i++)
    {
        for(int j=0;j<m_PredictorsNb[i];j++)
        {
            if (NeedsPreprocessing(i,j))
            {
                double minHourPredictor = 1000.0, maxHourPredictor = -1000.0;

                for(int k=0; k<GetPreprocessSize(i,j); k++)
                {
                    minHour = wxMin(m_Steps[i].Predictors[j].PreprocessTimeHours[k], minHour);
                    maxHour = wxMax(m_Steps[i].Predictors[j].PreprocessTimeHours[k], maxHour);
                    minHourPredictor = wxMin(m_Steps[i].Predictors[j].PreprocessTimeHours[k], minHourPredictor);
                    maxHourPredictor = wxMax(m_Steps[i].Predictors[j].PreprocessTimeHours[k], maxHourPredictor);
                    m_Steps[i].Predictors[j].TimeHours = minHourPredictor;
                }
            }
            else
            {
                minHour = wxMin(m_Steps[i].Predictors[j].TimeHours, minHour);
                maxHour = wxMax(m_Steps[i].Predictors[j].TimeHours, maxHour);
            }
        }
    }

    m_TimeMinHours = minHour;
    m_TimeMaxHours = maxHour;

    return true;
}

bool asParameters::FixWeights()
{
    for(int i=0;i<m_StepsNb;i++)
    {
        // Sum the weights
        float totWeight = 0;
        for(int j=0;j<m_PredictorsNb[i];j++)
        {
            totWeight += m_Steps[i].Predictors[j].Weight;
        }

        // Correct to set the total to 1
        for(int j=0;j<m_PredictorsNb[i];j++)
        {
            m_Steps[i].Predictors[j].Weight /= totWeight;
        }
    }

    return true;
}

bool asParameters::FixCoordinates()
{
    for(int i=0;i<m_StepsNb;i++)
    {
        for(int j=0;j<m_PredictorsNb[i];j++)
        {
            if (m_Steps[i].Predictors[j].GridType.IsSameAs("Regular", false))
            {

                // Check that the coordinates are a multiple of the steps
                if(abs(fmod(m_Steps[i].Predictors[j].Umin-m_Steps[i].Predictors[j].Ushift, m_Steps[i].Predictors[j].Ustep))>0)
                {
                    double factor = (m_Steps[i].Predictors[j].Umin-m_Steps[i].Predictors[j].Ushift)/m_Steps[i].Predictors[j].Ustep;
                    factor = asTools::Round(factor);
                    m_Steps[i].Predictors[j].Umin = factor*m_Steps[i].Predictors[j].Ustep+m_Steps[i].Predictors[j].Ushift;
                }

                if(abs(fmod(m_Steps[i].Predictors[j].Vmin-m_Steps[i].Predictors[j].Vshift, m_Steps[i].Predictors[j].Vstep))>0)
                {
                    double factor = (m_Steps[i].Predictors[j].Vmin-m_Steps[i].Predictors[j].Vshift)/m_Steps[i].Predictors[j].Vstep;
                    factor = asTools::Round(factor);
                    m_Steps[i].Predictors[j].Vmin = factor*m_Steps[i].Predictors[j].Vstep+m_Steps[i].Predictors[j].Vshift;
                }
            }

            if (m_Steps[i].Predictors[j].FlatAllowed==asFLAT_FORBIDDEN)
            {
                // Check that the size is larger than 1 point
                if(m_Steps[i].Predictors[j].Uptsnb<2)
                {
                    m_Steps[i].Predictors[j].Uptsnb = 2;
                }

                if(m_Steps[i].Predictors[j].Vptsnb<2)
                {
                    m_Steps[i].Predictors[j].Vptsnb = 2;
                }
            }
            else
            {
                // Check that the size is larger than 0
                if(m_Steps[i].Predictors[j].Uptsnb<1)
                {
                    m_Steps[i].Predictors[j].Uptsnb = 1;
                }

                if(m_Steps[i].Predictors[j].Vptsnb<1)
                {
                    m_Steps[i].Predictors[j].Vptsnb = 1;
                }
            }
        }
    }

    return true;
}

wxString asParameters::Print()
{
    // Create content string
    wxString content = wxEmptyString;

    content.Append(wxString::Format("Station\t%s\t", GetPredictandStationIdsString().c_str()));
    content.Append(wxString::Format("DaysInt\t%d\t", GetTimeArrayAnalogsIntervalDays()));
    content.Append(wxString::Format("ExcludeDays\t%d\t", GetTimeArrayAnalogsExcludeDays()));

    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        content.Append(wxString::Format("|||| Step(%d)\t", i_step));
        content.Append(wxString::Format("Anb\t%d\t", GetAnalogsNumber(i_step)));

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            content.Append(wxString::Format("|| Ptor(%d)\t", i_ptor));

            if (NeedsPreprocessing(i_step, i_ptor))
            {
                content.Append(wxString::Format("%s\t", GetPreprocessMethod(i_step, i_ptor).c_str()));

                for (int i_dataset=0; i_dataset<GetPreprocessSize(i_step,i_ptor); i_dataset++)
                {
                    content.Append(wxString::Format("| %s %s\t", GetPreprocessDatasetId(i_step, i_ptor, i_dataset).c_str(), GetPreprocessDataId(i_step, i_ptor, i_dataset)).c_str());
                    content.Append(wxString::Format("Level\t%g\t", GetPreprocessLevel(i_step, i_ptor, i_dataset)));
                    content.Append(wxString::Format("Time\t%g\t", GetPreprocessTimeHours(i_step, i_ptor, i_dataset)));
                }
            }
            else
            {
                content.Append(wxString::Format("%s %s\t", GetPredictorDatasetId(i_step, i_ptor).c_str(), GetPredictorDataId(i_step, i_ptor)).c_str());
                content.Append(wxString::Format("Level\t%g\t", GetPredictorLevel(i_step, i_ptor)));
                content.Append(wxString::Format("Time\t%g\t", GetPredictorTimeHours(i_step, i_ptor)));
            }

            content.Append(wxString::Format("GridType\t%s\t", GetPredictorGridType(i_step, i_ptor).c_str()));
            content.Append(wxString::Format("Umin\t%g\t", GetPredictorUmin(i_step, i_ptor)));
            content.Append(wxString::Format("Uptsnb\t%d\t", GetPredictorUptsnb(i_step, i_ptor)));
            content.Append(wxString::Format("Ustep\t%g\t", GetPredictorUstep(i_step, i_ptor)));
            content.Append(wxString::Format("Vmin\t%g\t", GetPredictorVmin(i_step, i_ptor)));
            content.Append(wxString::Format("Vptsnb\t%d\t", GetPredictorVptsnb(i_step, i_ptor)));
            content.Append(wxString::Format("Vstep\t%g\t", GetPredictorVstep(i_step, i_ptor)));
            content.Append(wxString::Format("Weight\t%e\t", GetPredictorWeight(i_step, i_ptor)));
            content.Append(wxString::Format("Criteria\t%s\t", GetPredictorCriteria(i_step, i_ptor).c_str()));
        }
    }

    return content;
}

bool asParameters::PrintAndSaveTemp(const wxString &filePath)
{
    wxString saveFilePath;

    if(filePath.IsEmpty())
    {
        saveFilePath = asConfig::GetTempDir() + "/AtmoSwingCurrentParameters.txt";
    }
    else
    {
        saveFilePath = filePath;
    }

    asFileAscii fileRes(saveFilePath, asFileAscii::Replace);
    if(!fileRes.Open()) return false;

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

    iLeft = stringVals.Find("DaysInt");
    iRight = stringVals.Find("ExcludeDays");
    strVal = stringVals.SubString(iLeft+8, iRight-2);
    strVal.ToLong(&lVal);
    SetTimeArrayAnalogsIntervalDays(int(lVal));
    stringVals = stringVals.SubString(iRight, stringVals.Length());

    iLeft = stringVals.Find("ExcludeDays");
    iRight = stringVals.Find("||||");
    strVal = stringVals.SubString(iLeft+12, iRight-2);
    strVal.ToLong(&lVal);
    SetTimeArrayAnalogsExcludeDays(int(lVal));
    stringVals = stringVals.SubString(iRight+5, stringVals.Length());

    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        iLeft = stringVals.Find("Anb");
        iRight = stringVals.Find("||");
        strVal = stringVals.SubString(iLeft+4, iRight-2);
        strVal.ToLong(&lVal);
        SetAnalogsNumber(i_step, int(lVal));
        stringVals = stringVals.SubString(iRight, stringVals.Length());

        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if (NeedsPreprocessing(i_step, i_ptor))
            {
                for (int i_dataset=0; i_dataset<GetPreprocessSize(i_step,i_ptor); i_dataset++)
                {
                    iLeft = stringVals.Find("Level");
                    iRight = stringVals.Find("Time");
                    strVal = stringVals.SubString(iLeft+6, iRight-2);
                    strVal.ToDouble(&dVal);
                    SetPreprocessLevel(i_step, i_ptor, i_dataset, float(dVal));
                    stringVals = stringVals.SubString(iRight+5, stringVals.Length());

                    iLeft = 0;
                    iRight = stringVals.Find("\t");
                    strVal = stringVals.SubString(iLeft, iRight-1);
                    strVal.ToDouble(&dVal);
                    SetPreprocessTimeHours(i_step, i_ptor, i_dataset, float(dVal));
                    stringVals = stringVals.SubString(iRight, stringVals.Length());
                }
            }
            else
            {
                iLeft = stringVals.Find("Level");
                iRight = stringVals.Find("Time");
                strVal = stringVals.SubString(iLeft+6, iRight-2);
                strVal.ToDouble(&dVal);
                SetPredictorLevel(i_step, i_ptor, float(dVal));
                stringVals = stringVals.SubString(iRight+5, stringVals.Length());

                iLeft = 0;
                iRight = stringVals.Find("\t");
                strVal = stringVals.SubString(iLeft, iRight-1);
                strVal.ToDouble(&dVal);
                SetPredictorTimeHours(i_step, i_ptor, float(dVal));
                stringVals = stringVals.SubString(iRight, stringVals.Length());
            }

            iLeft = stringVals.Find("Umin");
            iRight = stringVals.Find("Uptsnb");
            strVal = stringVals.SubString(iLeft+5, iRight-2);
            strVal.ToDouble(&dVal);
            SetPredictorUmin(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Uptsnb");
            iRight = stringVals.Find("Ustep");
            strVal = stringVals.SubString(iLeft+7, iRight-2);
            strVal.ToLong(&lVal);
            SetPredictorUptsnb(i_step, i_ptor, int(lVal));
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Ustep");
            iRight = stringVals.Find("Vmin");
            strVal = stringVals.SubString(iLeft+6, iRight-2);
            strVal.ToDouble(&dVal);
            SetPredictorUstep(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Vmin");
            iRight = stringVals.Find("Vptsnb");
            strVal = stringVals.SubString(iLeft+5, iRight-2);
            strVal.ToDouble(&dVal);
            SetPredictorVmin(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Vptsnb");
            iRight = stringVals.Find("Vstep");
            strVal = stringVals.SubString(iLeft+7, iRight-2);
            strVal.ToLong(&lVal);
            SetPredictorVptsnb(i_step, i_ptor, int(lVal));
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Vstep");
            iRight = stringVals.Find("Weight");
            strVal = stringVals.SubString(iLeft+6, iRight-2);
            strVal.ToDouble(&dVal);
            SetPredictorVstep(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());

            iLeft = stringVals.Find("Weight");
            iRight = stringVals.Find("Criteria");
            strVal = stringVals.SubString(iLeft+7, iRight-2);
            strVal.ToDouble(&dVal);
            SetPredictorWeight(i_step, i_ptor, dVal);
            stringVals = stringVals.SubString(iRight, stringVals.Length());
        }
    }

    return true;
}
