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
 
#include "asParametersScoring.h"

#include <asFileParameters.h>

asParametersScoring::asParametersScoring()
:
asParameters()
{
    m_CalibrationYearStart = 0;
    m_CalibrationYearEnd = 0;
    m_ForecastScore.Name = wxEmptyString;
    m_ForecastScore.AnalogsNumber = 0;
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
    predictor.DTimeHours.push_back(0);
    predictor.Criteria.push_back(wxEmptyString);
    predictor.Weight.push_back(0);

    step.Predictors.push_back(predictor);
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

bool asParametersScoring::FixAnalogsNb()
{
    // Check analogs number coherence
    int analogsNb = GetAnalogsNumber(0);
    for (int i_step=1; i_step<GetStepsNb(); i_step++)
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

    // Check forecast score
    if(GetForecastScoreAnalogsNumber()>analogsNb)
    {
        SetForecastScoreAnalogsNumber(analogsNb);
    }

    return true;
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
    content.Append(wxString::Format("Anb\t%d\t", GetForecastScoreAnalogsNumber()));

    return content;
}
