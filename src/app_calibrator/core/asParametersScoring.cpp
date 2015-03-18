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
    m_calibrationStart = 0;
    m_calibrationEnd = 0;
    m_forecastScore.Name = wxEmptyString;
    m_forecastScore.TimeArrayMode = wxEmptyString;
    m_forecastScore.TimeArrayDate = 0;
    m_forecastScore.TimeArrayIntervalDays = 0;
    m_forecastScore.Postprocess = false;
    m_forecastScore.PostprocessDupliExp = 0;
    m_forecastScore.PostprocessMethod = wxEmptyString;
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
    predictor.Xmin.push_back(0);
    predictor.Xptsnb.push_back(0);
    predictor.Ymin.push_back(0);
    predictor.Yptsnb.push_back(0);
    predictor.TimeHours.push_back(0);
    predictor.Criteria.push_back(wxEmptyString);
    predictor.Weight.push_back(1);

    step.Predictors.push_back(predictor);
}


bool asParametersScoring::GenerateSimpleParametersFile(const wxString &filePath)
{
    asLogMessage(_("Generating parameters file."));

    if(filePath.IsEmpty())
    {
        asLogError(_("The given path to the parameters file is empty."));
        return false;
    }

    asFileParametersCalibration fileParams(filePath, asFile::Replace);
    if(!fileParams.Open()) return false;

    // Create root nodes
    if(!fileParams.EditRootElement()) return false;
    fileParams.GetRoot()->AddAttribute("target", "calibrator");

    
    // Description
    wxXmlNode * nodeDescr = new wxXmlNode(wxXML_ELEMENT_NODE ,"description" );
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("method_id", GetMethodId()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("method_id_display", GetMethodIdDisplay()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("specific_tag", GetSpecificTag()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("specific_tag_display", GetSpecificTagDisplay()));
    nodeDescr->AddChild(fileParams.CreateNodeWithValue("description", GetDescription()));
    
    // Time properties
    wxXmlNode * nodeTime = new wxXmlNode(wxXML_ELEMENT_NODE ,"time_properties" );

    wxXmlNode * nodeTimeArchivePeriod = new wxXmlNode(wxXML_ELEMENT_NODE ,"archive_period" );
    nodeTime->AddChild(nodeTimeArchivePeriod);
    wxString archiveStart = asTime::GetStringTime(GetArchiveStart(), "DD.MM.YYYY");
    nodeTimeArchivePeriod->AddChild(fileParams.CreateNodeWithValue("start", archiveStart));
    wxString archiveEnd = asTime::GetStringTime(GetArchiveEnd(), "DD.MM.YYYY");
    nodeTimeArchivePeriod->AddChild(fileParams.CreateNodeWithValue("end", archiveEnd));

    wxXmlNode * nodeTimeCalibrationPeriod = new wxXmlNode(wxXML_ELEMENT_NODE ,"calibration_period" );
    nodeTime->AddChild(nodeTimeCalibrationPeriod);
    wxString calibrationStart = asTime::GetStringTime(GetCalibrationStart(), "DD.MM.YYYY");
    nodeTimeCalibrationPeriod->AddChild(fileParams.CreateNodeWithValue("start", calibrationStart));
    wxString calibrationEnd = asTime::GetStringTime(GetCalibrationEnd(), "DD.MM.YYYY");
    nodeTimeCalibrationPeriod->AddChild(fileParams.CreateNodeWithValue("end", calibrationEnd));

    if(HasValidationPeriod())
    {
        wxXmlNode * nodeTimeValidationPeriod = new wxXmlNode(wxXML_ELEMENT_NODE ,"validation_period" );
        nodeTime->AddChild(nodeTimeValidationPeriod);
        wxString validationYears;
        VectorInt validationYearsVect = GetValidationYearsVector();
        for (int i=0; i<(int)validationYearsVect.size(); i++)
        {
            validationYears << validationYearsVect[i];
            if (i!=(int)validationYearsVect.size()-1)
            {
                validationYears << ", ";
            }
        }
        nodeTimeValidationPeriod->AddChild(fileParams.CreateNodeWithValue("years", validationYears));
    }

    wxString timeArrayTimeStep;
    timeArrayTimeStep << GetTimeArrayTargetTimeStepHours();
    nodeTime->AddChild(fileParams.CreateNodeWithValue("time_step", timeArrayTimeStep));

    wxXmlNode * nodeTimeArrayTarget = new wxXmlNode(wxXML_ELEMENT_NODE ,"time_array_target" );
    nodeTime->AddChild(nodeTimeArrayTarget);
    nodeTimeArrayTarget->AddChild(fileParams.CreateNodeWithValue("time_array", GetTimeArrayTargetMode()));
    if(GetTimeArrayTargetMode().IsSameAs("predictand_thresholds"))
    {
        nodeTimeArrayTarget->AddChild(fileParams.CreateNodeWithValue("predictand_serie_name", GetTimeArrayTargetPredictandSerieName()));
        nodeTimeArrayTarget->AddChild(fileParams.CreateNodeWithValue("predictand_min_threshold", GetTimeArrayTargetPredictandMinThreshold()));
        nodeTimeArrayTarget->AddChild(fileParams.CreateNodeWithValue("predictand_max_threshold", GetTimeArrayTargetPredictandMaxThreshold()));
    }

    wxXmlNode * nodeTimeArrayAnalogs = new wxXmlNode(wxXML_ELEMENT_NODE ,"time_array_analogs" );
    nodeTime->AddChild(nodeTimeArrayAnalogs);
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNodeWithValue("time_array", GetTimeArrayAnalogsMode()));
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNodeWithValue("interval_days", GetTimeArrayAnalogsIntervalDays()));
    nodeTimeArrayAnalogs->AddChild(fileParams.CreateNodeWithValue("exclude_days", GetTimeArrayAnalogsExcludeDays()));

    fileParams.AddChild(nodeTime);


    // Analog dates
    for (int i_step=0; i_step<GetStepsNb(); i_step++)
    {
        wxXmlNode * nodeAnalogDates = new wxXmlNode(wxXML_ELEMENT_NODE ,"analog_dates" );

        nodeAnalogDates->AddChild(fileParams.CreateNodeWithValue("analogs_number", GetAnalogsNumber(i_step)));

        // Predictors
        for (int i_ptor=0; i_ptor<GetPredictorsNb(i_step); i_ptor++)
        {
            wxXmlNode * nodePredictor = new wxXmlNode(wxXML_ELEMENT_NODE ,"predictor" );
            nodeAnalogDates->AddChild(nodePredictor);

            nodePredictor->AddChild(fileParams.CreateNodeWithValue("preload", NeedsPreloading(i_step, i_ptor)));

            if (NeedsPreprocessing(i_step, i_ptor))
            {
                wxXmlNode * nodePreprocessing = new wxXmlNode(wxXML_ELEMENT_NODE ,"preprocessing" );
                nodePredictor->AddChild(nodePreprocessing);

                nodePreprocessing->AddChild(fileParams.CreateNodeWithValue("preprocessing_method", GetPreprocessMethod(i_step, i_ptor)));

                for (int i_preproc=0; i_preproc<GetPreprocessSize(i_step, i_ptor); i_preproc++)
                {
                    wxXmlNode * nodePreprocessingData = new wxXmlNode(wxXML_ELEMENT_NODE ,"preprocessing_data" );
                    nodePreprocessing->AddChild(nodePreprocessingData);

                    nodePreprocessingData->AddChild(fileParams.CreateNodeWithValue("dataset_id", GetPreprocessDatasetId(i_step, i_ptor, i_preproc)));
                    nodePreprocessingData->AddChild(fileParams.CreateNodeWithValue("data_id", GetPreprocessDataId(i_step, i_ptor, i_preproc)));
                    nodePreprocessingData->AddChild(fileParams.CreateNodeWithValue("level", GetPreprocessLevel(i_step, i_ptor, i_preproc)));
                    nodePreprocessingData->AddChild(fileParams.CreateNodeWithValue("time", GetPreprocessTimeHours(i_step, i_ptor, i_preproc)));
                }
            }
            else
            {
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("dataset_id", GetPredictorDatasetId(i_step, i_ptor)));
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("data_id", GetPredictorDataId(i_step, i_ptor)));
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("level", GetPredictorLevel(i_step, i_ptor)));
                nodePredictor->AddChild(fileParams.CreateNodeWithValue("time", GetPredictorTimeHours(i_step, i_ptor)));
            }

            wxXmlNode * nodeWindow = new wxXmlNode(wxXML_ELEMENT_NODE ,"spatial_window" );
            nodePredictor->AddChild(nodeWindow);

            nodeWindow->AddChild(fileParams.CreateNodeWithValue("grid_type", GetPredictorGridType(i_step, i_ptor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("x_min", GetPredictorXmin(i_step, i_ptor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("x_points_nb", GetPredictorXptsnb(i_step, i_ptor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("x_step", GetPredictorXstep(i_step, i_ptor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("y_min", GetPredictorYmin(i_step, i_ptor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("y_points_nb", GetPredictorYptsnb(i_step, i_ptor)));
            nodeWindow->AddChild(fileParams.CreateNodeWithValue("y_step", GetPredictorYstep(i_step, i_ptor)));

            nodePredictor->AddChild(fileParams.CreateNodeWithValue("criteria", GetPredictorCriteria(i_step, i_ptor)));
            nodePredictor->AddChild(fileParams.CreateNodeWithValue("weight", GetPredictorWeight(i_step, i_ptor)));
        }

        fileParams.AddChild(nodeAnalogDates);
    }


    // Analogs values
    wxXmlNode * nodeAnalogValues = new wxXmlNode(wxXML_ELEMENT_NODE ,"analog_values" );

    wxXmlNode * nodePredictand = new wxXmlNode(wxXML_ELEMENT_NODE ,"predictand" );
    nodeAnalogValues->AddChild(nodePredictand);

    VVectorInt predictandStationIdsVect = GetPredictandStationIdsVector();
    wxString predictandStationIds = GetPredictandStationIdsVectorString(predictandStationIdsVect);
    nodePredictand->AddChild(fileParams.CreateNodeWithValue("station_id", predictandStationIds));

    fileParams.AddChild(nodeAnalogValues);


    // Forecast scores
    wxXmlNode * nodeAnalogScore = new wxXmlNode(wxXML_ELEMENT_NODE ,"analog_forecast_score" );

    nodeAnalogScore->AddChild(fileParams.CreateNodeWithValue("score", GetForecastScoreName()));

    float fsThreshold = GetForecastScoreThreshold();
    if (!asTools::IsNaN(fsThreshold))
    {
        nodeAnalogScore->AddChild(fileParams.CreateNodeWithValue("threshold", fsThreshold));
    }

    float fsQuantile = GetForecastScoreQuantile();
    if (!asTools::IsNaN(fsQuantile))
    {
        nodeAnalogScore->AddChild(fileParams.CreateNodeWithValue("quantile", fsQuantile));
    }

    fileParams.AddChild(nodeAnalogScore);


    // Forecast score final
    wxXmlNode * nodeAnalogScoreFinal = new wxXmlNode(wxXML_ELEMENT_NODE ,"analog_forecast_score_final" );

    nodeAnalogScoreFinal->AddChild(fileParams.CreateNodeWithValue("time_array", GetForecastScoreTimeArrayMode()));

    fileParams.AddChild(nodeAnalogScoreFinal);


    if(!fileParams.Save()) return false;
    if(!fileParams.Close()) return false;

    asLogMessage(_("Parameters file generated."));

    return true;
}

wxString asParametersScoring::GetPredictandStationIdsVectorString(VVectorInt &predictandStationIdsVect)
{
    wxString Ids;

    for (int i=0; i<(int)predictandStationIdsVect.size(); i++)
    {
        VectorInt predictandStationIds = predictandStationIdsVect[i];

        if (predictandStationIds.size()==1)
        {
            Ids << predictandStationIds[0];
        }
        else
        {
            Ids.Append("(");

            for (int j=0; j<(int)predictandStationIds.size(); j++)
            {
                Ids << predictandStationIds[j];

                if (j<(int)predictandStationIds.size()-1)
                {
                    Ids.Append(",");
                }
            }

            Ids.Append(")");
        }

        if (i<(int)predictandStationIdsVect.size()-1)
        {
            Ids.Append(",");
        }
    }

    return Ids;
}

wxString asParametersScoring::Print()
{
    // Create content string
    wxString content = asParameters::Print();

    content.Append(wxString::Format("|||| Score \t%s\t", GetForecastScoreName().c_str()));
    if (!asTools::IsNaN(GetForecastScoreQuantile()))
    {
        content.Append(wxString::Format("Quantile \t%f\t", GetForecastScoreQuantile()));
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
        asLogError(wxString::Format(_("The current score (%s) doesn't correspond to the previous one (%s)."), GetForecastScoreName(), strVal));
        printf(wxString::Format(_("Error: The current score (%s) doesn't correspond to the previous one (%s).\n"), GetForecastScoreName(), strVal));
        return false;
    }

    return true;
}
