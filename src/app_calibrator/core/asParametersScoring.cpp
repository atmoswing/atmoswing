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

#include "asParametersScoring.h"

#include <asFileParametersCalibration.h>

asParametersScoring::asParametersScoring()
:
asParameters()
{
    m_CalibrationStart = 0;
    m_CalibrationEnd = 0;
    m_ForecastScore.Name = wxEmptyString;
    m_ForecastScore.TimeArrayMode = wxEmptyString;
    m_ForecastScore.TimeArrayDate = 0;
    m_ForecastScore.TimeArrayIntervalDays = 0;
    m_ForecastScore.Postprocess = false;
    m_ForecastScore.PostprocessDupliExp = 0;
    m_ForecastScore.PostprocessMethod = wxEmptyString;
}

asParametersScoring::~asParametersScoring()
{
    //dtor
}

void asParametersScoring::AddPredictorVect(ParamsStepVect &step)
{
    ParamsPredictorVect predictor;

    predictor.DataId.push_back(wxEmptyString);
    predictor.Level.push_back(0);
    predictor.Umin.push_back(0);
    predictor.Uptsnb.push_back(0);
    predictor.Vmin.push_back(0);
    predictor.Vptsnb.push_back(0);
    predictor.TimeHours.push_back(0);
    predictor.Criteria.push_back(wxEmptyString);
    predictor.Weight.push_back(0);

    step.Predictors.push_back(predictor);
}


bool asParametersScoring::GenerateSimpleParametersFile(const wxString &filePath)
{
    asLogMessage(_("Generating parameters file."));

    wxString val;

    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::Replace);
    if(!fileParams.Open()) return false;


    // Create root nodes
    if(!fileParams.InsertRootElement()) return false;


    // Set general parameters
    if(!fileParams.InsertElementAndAttribute("", "General", "name", "General")) return false;
    if(!fileParams.InsertElement("General", "Options")) return false;
    if(!fileParams.GoToFirstNodeWithPath("General")) return false;


    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Archive Period")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Archive Period")) return false;
    wxString archiveStart = asTime::GetStringTime(GetArchiveStart(), "DD.MM.YYYY");
    if(!fileParams.InsertElementAndAttribute("", "Start", "value", archiveStart)) return false;
    wxString archiveYearEnd = asTime::GetStringTime(GetArchiveEnd(), "DD.MM.YYYY");
    if(!fileParams.InsertElementAndAttribute("", "End", "value", archiveYearEnd)) return false;
    if(!fileParams.GoANodeBack()) return false;


    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Calibration Period")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Calibration Period")) return false;
    wxString calibStart = asTime::GetStringTime(GetCalibrationStart(), "DD.MM.YYYY");
    if(!fileParams.InsertElementAndAttribute("", "Start", "value", calibStart)) return false;
    wxString calibEnd = asTime::GetStringTime(GetCalibrationEnd(), "DD.MM.YYYY");
    if(!fileParams.InsertElementAndAttribute("", "End", "value", calibEnd)) return false;
    if(!fileParams.GoANodeBack()) return false;


    if(HasValidationPeriod())
    {
        if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Validation Period")) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Validation Period")) return false;
        wxString validationYears;
        VectorInt validationYearsVect = GetValidationYearsVector();
        for (int i=0; i<validationYearsVect.size(); i++)
        {
            validationYears << validationYearsVect[i];
            if (i!=validationYearsVect.size()-1)
            {
                validationYears << ", ";
            }
        }
        if(!fileParams.InsertElementAndAttribute("", "Years", "value", validationYears)) return false;
        if(!fileParams.SetElementAttribute("Years", "method", "array")) return false;
        if(!fileParams.GoANodeBack()) return false;
    }


    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Time Array Target")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Target")) return false;
    wxString timeArrayTargetTimeStepHours;
    timeArrayTargetTimeStepHours << GetTimeArrayTargetTimeStepHours();
    if(!fileParams.InsertElementAndAttribute("", "TimeStepHours", "value", timeArrayTargetTimeStepHours)) return false;
    if(!fileParams.InsertElementAndAttribute("", "TimeArrayMode", "value", GetTimeArrayTargetMode())) return false;
    if(GetTimeArrayTargetMode().IsSameAs("PredictandThresholds"))
    {
        if(!fileParams.InsertElementAndAttribute("", "PredictandSerieName", "value", GetTimeArrayTargetPredictandSerieName())) return false;
        wxString timeArrayTargetPredictandMinThreshold;
        timeArrayTargetPredictandMinThreshold << GetTimeArrayTargetPredictandMinThreshold();
        if(!fileParams.InsertElementAndAttribute("", "PredictandMinThreshold", "value", timeArrayTargetPredictandMinThreshold)) return false;
        wxString timeArrayTargetPredictandMaxThreshold;
        timeArrayTargetPredictandMaxThreshold << GetTimeArrayTargetPredictandMaxThreshold();
        if(!fileParams.InsertElementAndAttribute("", "PredictandMaxThreshold", "value", timeArrayTargetPredictandMaxThreshold)) return false;
    }
    if(!fileParams.GoANodeBack()) return false;


    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Time Array Analogs")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array Analogs")) return false;
    wxString timeArrayAnalogsTimeStepHours;
    timeArrayAnalogsTimeStepHours << GetTimeArrayAnalogsTimeStepHours();
    if(!fileParams.InsertElementAndAttribute("", "TimeStepHours", "value", timeArrayAnalogsTimeStepHours)) return false;
    if(!fileParams.InsertElementAndAttribute("", "TimeArrayMode", "value", GetTimeArrayAnalogsMode())) return false;
    wxString timeArrayAnalogsIntervalDays;
    timeArrayAnalogsIntervalDays << GetTimeArrayAnalogsIntervalDays();
    if(!fileParams.InsertElementAndAttribute("", "IntervalDays", "value", timeArrayAnalogsIntervalDays)) return false;
    wxString timeArrayAnalogsExcludeDays;
    timeArrayAnalogsExcludeDays << GetTimeArrayAnalogsExcludeDays();
    if(!fileParams.InsertElementAndAttribute("", "ExcludeDays", "value", timeArrayAnalogsExcludeDays)) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;


    // Set Analogs Dates processes
    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        if(!fileParams.InsertElementAndAttribute("", "Process", "name", "Analogs Dates")) return false;
        if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Analogs Dates")) return false;
        if(!fileParams.InsertElement("", "Options")) return false;


        if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Method Name")) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method Name")) return false;
        if(!fileParams.InsertElementAndAttribute("", "MethodName", "value", GetMethodName(i_step))) return false;
        if(!fileParams.GoANodeBack()) return false;


        if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Analogs Number")) return false;
        if(!fileParams.GoToChildNodeWithAttributeValue("name", "Analogs Number")) return false;
        wxString analogsNumber;
        analogsNumber << GetAnalogsNumber(i_step);
        if(!fileParams.InsertElementAndAttribute("", "AnalogsNumber", "value", analogsNumber)) return false;
        if(!fileParams.GoANodeBack()) return false;


        // Get data
        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            if (!NeedsPreprocessing(i_step, i_ptor))
            {
                if(!fileParams.InsertElementAndAttribute("", "Data", "name", "Predictor")) return false;
                if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Predictor")) return false;
                if(!fileParams.InsertElement("", "Options")) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Preload")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preload")) return false;
                if (NeedsPreloading(i_step, i_ptor))
                {
                    if(!fileParams.InsertElementAndAttribute("", "Preload", "value", "1")) return false;
                }
                else
                {
                    if(!fileParams.InsertElementAndAttribute("", "Preload", "value", "0")) return false;
                }
                if(!fileParams.GoANodeBack()) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Preprocessing")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                if(!fileParams.InsertElementAndAttribute("", "Preprocess", "value", "0")) return false;
                if(!fileParams.GoANodeBack()) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Data")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Data")) return false;
                if(!fileParams.InsertElementAndAttribute("", "DatasetId", "value", GetPredictorDatasetId(i_step, i_ptor))) return false;
                if(!fileParams.InsertElementAndAttribute("", "DataId", "value", GetPredictorDataId(i_step, i_ptor))) return false;
                if(!fileParams.GoANodeBack()) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Level")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Level")) return false;
                wxString level;
                level << GetPredictorLevel(i_step, i_ptor);
                if(!fileParams.InsertElementAndAttribute("", "Level", "value", level)) return false;
                if(!fileParams.GoANodeBack()) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Time frame")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time frame")) return false;
                wxString timeHours;
                timeHours << GetPredictorTimeHours(i_step, i_ptor);
                if(!fileParams.InsertElementAndAttribute("", "TimeHours", "value", timeHours)) return false;
                if(!fileParams.GoANodeBack()) return false;
            }
            else
            {
                if(!fileParams.InsertElementAndAttribute("", "Data", "name", "Predictor Preprocessed")) return false;
                if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Predictor Preprocessed")) return false;
                if(!fileParams.InsertElement("", "Options")) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Preload")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preload")) return false;
                if (NeedsPreloading(i_step, i_ptor))
                {
                    if(!fileParams.InsertElementAndAttribute("", "Preload", "value", "1")) return false;
                }
                else
                {
                    if(!fileParams.InsertElementAndAttribute("", "Preload", "value", "0")) return false;
                }
                if(!fileParams.GoANodeBack()) return false;


                if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Preprocessing")) return false;
                if(!fileParams.GoToChildNodeWithAttributeValue("name", "Preprocessing")) return false;
                if(!fileParams.InsertElementAndAttribute("", "Preprocess", "value", "1")) return false;
                if(!fileParams.InsertElementAndAttribute("", "PreprocessMethod", "value", GetPreprocessMethod(i_step, i_ptor))) return false;


                for (int i_preproc=0; i_preproc<GetPreprocessSize(i_step, i_ptor); i_preproc++)
                {
                    if(!fileParams.InsertElement("", "SubData")) return false;
                    if(!fileParams.GoToLastNodeWithPath("SubData")) return false;
                    if(!fileParams.InsertElementAndAttribute("", "PreprocessDatasetId", "value", GetPreprocessDatasetId(i_step, i_ptor, i_preproc))) return false;
                    if(!fileParams.InsertElementAndAttribute("", "PreprocessDataId", "value", GetPreprocessDataId(i_step, i_ptor, i_preproc))) return false;
                    wxString preprocessLevel = wxString::Format("%.0f", GetPreprocessLevel(i_step, i_ptor, i_preproc));
                    if(!fileParams.InsertElementAndAttribute("", "PreprocessLevel", "value", preprocessLevel)) return false;
                    wxString preprocessTimeHours;
                    preprocessTimeHours << GetPreprocessTimeHours(i_step, i_ptor, i_preproc);
                    if(!fileParams.InsertElementAndAttribute("", "PreprocessTimeHours", "value", preprocessTimeHours)) return false;

                    if(!fileParams.GoANodeBack()) return false;
                }

                if(!fileParams.GoANodeBack()) return false;
            }


            if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Area Moving")) return false;
            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Area Moving")) return false;
            if(!fileParams.InsertElementAndAttribute("", "GridType", "value", GetPredictorGridType(i_step, i_ptor))) return false;
            wxString umin;
            umin << GetPredictorUmin(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Umin", "value", umin)) return false;
            wxString uptsnb;
            uptsnb << GetPredictorUptsnb(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Uptsnb", "value", uptsnb)) return false;
            wxString ustep;
            ustep << GetPredictorUstep(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Ustep", "value", ustep)) return false;
            wxString vmin;
            vmin << GetPredictorVmin(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Vmin", "value", vmin)) return false;
            wxString vptsnb;
            vptsnb << GetPredictorVptsnb(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Vptsnb", "value", vptsnb)) return false;
            wxString vstep;
            vstep << GetPredictorVstep(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Vstep", "value", vstep)) return false;
            if(!fileParams.GoANodeBack()) return false;


            if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Criteria")) return false;
            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Criteria")) return false;
            if(!fileParams.InsertElementAndAttribute("", "Criteria", "value", GetPredictorCriteria(i_step, i_ptor))) return false;
            if(!fileParams.GoANodeBack()) return false;


            if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Weight")) return false;
            if(!fileParams.GoToChildNodeWithAttributeValue("name", "Weight")) return false;
            wxString weight;
            weight << GetPredictorWeight(i_step, i_ptor);
            if(!fileParams.InsertElementAndAttribute("", "Weight", "value", weight)) return false;
            if(!fileParams.GoANodeBack()) return false;

            if(!fileParams.GoANodeBack()) return false;
        }

        if(!fileParams.GoANodeBack()) return false;
    }

    // Set Analogs Values process
    if(!fileParams.InsertElementAndAttribute("", "Process", "name", "Analogs Values")) return false;
    if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Analogs Values")) return false;
    if(!fileParams.InsertElement("", "Options")) return false;


    if(!fileParams.InsertElementAndAttribute("", "Data", "name", "Predictand")) return false;
    if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Predictand")) return false;

    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Database")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Database")) return false;
    VVectorInt predictandStationIdsVect = GetPredictandStationsIdsVector();
    wxString predictandStationIds = GetPredictandStationIdsVectorString(predictandStationIdsVect);
    if(!fileParams.InsertElementAndAttribute("", "PredictandStationId", "value", predictandStationIds)) return false;
    if(!fileParams.SetElementAttribute("PredictandStationId", "method", "array")) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;


    // Set Analogs Forecast Scores process
    if(!fileParams.InsertElementAndAttribute("", "Process", "name", "Analogs Forecast Scores")) return false;
    if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Analogs Forecast Scores")) return false;
    if(!fileParams.InsertElement("", "Options")) return false;


    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Method")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Method")) return false;
    if(!fileParams.InsertElementAndAttribute("", "Name", "value", GetForecastScoreName())) return false;
    float fsThreshold = GetForecastScoreThreshold();
    if (!asTools::IsNaN(fsThreshold))
    {
        wxString fsThresholdStr;
        fsThresholdStr << fsThreshold;
        if(!fileParams.InsertElementAndAttribute("", "Threshold", "value", fsThresholdStr)) return false;
    }

    float fsPercentile = GetForecastScorePercentile();
    if (!asTools::IsNaN(fsPercentile))
    {
        wxString fsPercentileStr;
        fsPercentileStr << fsPercentile;
        if(!fileParams.InsertElementAndAttribute("", "Percentile", "value", fsPercentileStr)) return false;
    }
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;


    // Set Analogs Forecast Score Final process
    if(!fileParams.InsertElementAndAttribute("", "Process", "name", "Analogs Forecast Score Final")) return false;
    if(!fileParams.GoToLastChildNodeWithAttributeValue("name", "Analogs Forecast Score Final")) return false;
    if(!fileParams.InsertElement("", "Options")) return false;


    if(!fileParams.InsertElementAndAttribute("", "Params", "name", "Time Array")) return false;
    if(!fileParams.GoToChildNodeWithAttributeValue("name", "Time Array")) return false;
    if(!fileParams.InsertElementAndAttribute("", "Mode", "value", GetForecastScoreTimeArrayMode())) return false;
    if(!fileParams.GoANodeBack()) return false;

    if(!fileParams.GoANodeBack()) return false;

    // if(!SetForecastScoreTimeArrayDateVector(GetFileParamDoubleVector(fileParams, "Date"))) return false; ??
    // if(!SetForecastScoreTimeArrayIntervalDaysVector(GetFileParamIntVector(fileParams, "IntervalDays"))) return false; ??

    if(!fileParams.Save()) return false;
    if(!fileParams.Close()) return false;

    asLogMessage(_("Parameters file generated."));

    return true;
}

wxString asParametersScoring::GetPredictandStationIdsVectorString(VVectorInt &predictandStationIdsVect)
{
    wxString Ids;

    for (int i=0; i<predictandStationIdsVect.size(); i++)
    {
        VectorInt predictandStationIds = predictandStationIdsVect[i];

        if (predictandStationIds.size()==1)
        {
            Ids << predictandStationIds[0];
        }
        else
        {
            Ids.Append("(");

            for (int j=0; j<predictandStationIds.size(); j++)
            {
                Ids << predictandStationIds[j];

                if (j<predictandStationIds.size()-1)
                {
                    Ids.Append(",");
                }
            }

            Ids.Append(")");
        }

        if (i<predictandStationIdsVect.size()-1)
        {
            Ids.Append(",");
        }
    }

    return Ids;
}

VectorInt asParametersScoring::GetFileParamIntVector(asFileParameters &fileParams, const wxString &tag)
{
    VectorInt vect;
    wxString method = fileParams.GetFirstElementAttributeValueText(tag, "method");
    if(method.IsEmpty())
    {
        asLogMessage(wxString::Format(_("The method is not defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueInt(tag, "value"));
    }
    else if(method.IsSameAs("fixed"))
    {
        vect.push_back(fileParams.GetFirstElementAttributeValueInt(tag, "value"));

    }
    else if(method.IsSameAs("array"))
    {
        wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
        vect = BuildVectorInt(txt);
    }
    else if(method.IsSameAs("minmax"))
    {
        int min = fileParams.GetFirstElementAttributeValueInt(tag, "min");
        int max = fileParams.GetFirstElementAttributeValueInt(tag, "max");
        int step = fileParams.GetFirstElementAttributeValueInt(tag, "step");
        vect = BuildVectorInt(min, max, step);
    }
    else
    {
        asLogMessage(wxString::Format(_("The method is not correctly defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueInt(tag, "value"));
    }

    return vect;
}

VectorFloat asParametersScoring::GetFileParamFloatVector(asFileParameters &fileParams, const wxString &tag)
{
    VectorFloat vect;
    wxString method = fileParams.GetFirstElementAttributeValueText(tag, "method");
    if(method.IsEmpty())
    {
        asLogMessage(wxString::Format(_("The method is not defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueFloat(tag, "value"));
    }
    else if(method.IsSameAs("fixed"))
    {
        vect.push_back(fileParams.GetFirstElementAttributeValueFloat(tag, "value"));

    }
    else if(method.IsSameAs("array"))
    {
        wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
        vect = BuildVectorFloat(txt);
    }
    else if(method.IsSameAs("minmax"))
    {
        float min = fileParams.GetFirstElementAttributeValueFloat(tag, "min");
        float max = fileParams.GetFirstElementAttributeValueFloat(tag, "max");
        float step = fileParams.GetFirstElementAttributeValueFloat(tag, "step");
        vect = BuildVectorFloat(min, max, step);
    }
    else
    {
        asLogMessage(wxString::Format(_("The method is not correctly defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueFloat(tag, "value"));
    }

    return vect;
}

VectorDouble asParametersScoring::GetFileParamDoubleVector(asFileParameters &fileParams, const wxString &tag)
{
    VectorDouble vect;
    wxString method = fileParams.GetFirstElementAttributeValueText(tag, "method");
    if(method.IsEmpty())
    {
        asLogMessage(wxString::Format(_("The method is not defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueDouble(tag, "value"));
    }
    else if(method.IsSameAs("fixed"))
    {
        vect.push_back(fileParams.GetFirstElementAttributeValueDouble(tag, "value"));

    }
    else if(method.IsSameAs("array"))
    {
        wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
        vect = BuildVectorDouble(txt);
    }
    else if(method.IsSameAs("minmax"))
    {
        double min = fileParams.GetFirstElementAttributeValueDouble(tag, "min");
        double max = fileParams.GetFirstElementAttributeValueDouble(tag, "max");
        double step = fileParams.GetFirstElementAttributeValueDouble(tag, "step");
        vect = BuildVectorDouble(min, max, step);
    }
    else
    {
        asLogMessage(wxString::Format(_("The method is not correctly defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueDouble(tag, "value"));
    }

    return vect;
}

VectorString asParametersScoring::GetFileParamStringVector(asFileParameters &fileParams, const wxString &tag)
{
    VectorString vect;
    wxString method = fileParams.GetFirstElementAttributeValueText(tag, "method");
    if(method.IsEmpty())
    {
        asLogMessage(wxString::Format(_("The method is not defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueText(tag, "value"));
    }
    else if(method.IsSameAs("fixed"))
    {
        vect.push_back(fileParams.GetFirstElementAttributeValueText(tag, "value"));

    }
    else if(method.IsSameAs("array"))
    {
        wxString txt = fileParams.GetFirstElementAttributeValueText(tag, "value");
        vect = BuildVectorString(txt);
    }
    else
    {
        asLogMessage(wxString::Format(_("The method is not correctly defined for %s in the calibration parameters file."), tag));
        vect.push_back(fileParams.GetFirstElementAttributeValueText(tag, "value"));
    }

    return vect;
}

VVectorInt asParametersScoring::GetFileStationIdsVector(asFileParameters &fileParams)
{
    VVectorInt vect;
    wxString tag = "PredictandStationId";

    wxString method = fileParams.GetFirstElementAttributeValueText(tag, "method");
    if(method.IsSameAs("fixed"))
    {
        wxString value = fileParams.GetFirstElementAttributeValueText(tag, "value");
        VectorInt ids = GetFileStationIds(value);
        vect.push_back(ids);
    }
    else if(method.IsSameAs("array") || method.IsEmpty())
    {
        if(method.IsEmpty())
        {
            asLogMessage(wxString::Format(_("The method is not defined for %s in the calibration parameters file."), tag));
        }

        wxString value = fileParams.GetFirstElementAttributeValueText(tag, "value");

        // Explode the array
        wxChar separator = ',';
        while(value.Find(separator)!=wxNOT_FOUND)
        {
            // If presence of a group next
            int startBracket = value.Find('(');
            if (startBracket!=wxNOT_FOUND && startBracket<value.Find(separator))
            {
                int endBracket = value.Find(')');
                wxString bracketContent = value.SubString(startBracket, endBracket);
                VectorInt ids = GetFileStationIds(bracketContent);
                vect.push_back(ids);

                value = value.AfterFirst(')');
                if (value.Find(separator))
                {
                    value = value.AfterFirst(separator);
                }
            }
            else
            {
                wxString txtbefore = value.BeforeFirst(separator);
                value = value.AfterFirst(separator);
                if (!txtbefore.IsEmpty())
                {
                    VectorInt ids = GetFileStationIds(txtbefore);
                    vect.push_back(ids);
                }
            }
        }
        if(!value.IsEmpty())
        {
            VectorInt ids = GetFileStationIds(value);
            vect.push_back(ids);
        }
    }
    else if(method.IsSameAs("minmax"))
    {
        int min = fileParams.GetFirstElementAttributeValueInt(tag, "min");
        int max = fileParams.GetFirstElementAttributeValueInt(tag, "max");
        int step = fileParams.GetFirstElementAttributeValueInt(tag, "step");
        VectorInt ids = BuildVectorInt(min, max, step);
        for (int i=0; i<ids.size(); i++)
        {
            VectorInt id;
            id.push_back(ids[i]);
            vect.push_back(id);
        }
    }
    else
    {
        asLogMessage(wxString::Format(_("The method is not correctly defined for %s in the calibration parameters file."), tag));
        return VVectorInt(0);
    }

    return vect;
}

wxString asParametersScoring::Print()
{
    // Create content string
    wxString content = asParameters::Print();

    content.Append(wxString::Format("|||| Score \t%s\t", GetForecastScoreName().c_str()));
    if (!asTools::IsNaN(GetForecastScorePercentile()))
    {
        content.Append(wxString::Format("Percentile \t%f\t", GetForecastScorePercentile()));
    }
    if (!asTools::IsNaN(GetForecastScoreThreshold()))
    {
        content.Append(wxString::Format("Threshold \t%f\t", GetForecastScoreThreshold()));
    }
    content.Append(wxString::Format("TimeArray\t%s\t", GetForecastScoreTimeArrayMode().c_str()));

    return content;
}

bool asParametersScoring::GetValuesFromString(wxString stringVals)
{
    // Get the parameters values
    if(!asParameters::GetValuesFromString(stringVals))
    {
        return false;
    }

    int iLeft, iRight;
    wxString strVal;

    // Check that the score is similar
    iLeft = stringVals.Find("Score");
    stringVals = stringVals.SubString(iLeft+7, stringVals.Length());
    iLeft = 0;
    iRight = stringVals.Find("\t");
    strVal = stringVals.SubString(iLeft, iRight-1);
    if (!strVal.IsSameAs(GetForecastScoreName()))
    {
        asLogError(wxString::Format(_("The current score (%s) doesn't correspond to the previous one (%s)."), GetForecastScoreName().c_str(), strVal.c_str()));
        printf(wxString::Format(_("Error: The current score (%s) doesn't correspond to the previous one (%s).\n"), GetForecastScoreName().c_str(), strVal.c_str()));
        return false;
    }

    return true;
}
