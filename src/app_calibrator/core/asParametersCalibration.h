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
 
#ifndef ASPARAMETERSCALIBRATION_H
#define ASPARAMETERSCALIBRATION_H

#include "asIncludes.h"
#include <asParametersScoring.h>

class asFileParametersCalibration;


class asParametersCalibration : public asParametersScoring
{
public:

    asParametersCalibration();
    virtual ~asParametersCalibration();

    void AddStep();

    bool LoadFromFile(const wxString &filePath);

    bool FixTimeLimits();

    void InitValues();

    VVectorInt GetPredictandStationsIdsVector()
    {
        return m_PredictandStationsIdsVect;
    }

    bool SetPredictandStationsIdsVector(VVectorInt val)
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

            for (int i=0; i<val.size(); i++)
            {
                for (int j=0; j<val[i].size(); j++)
                {
                    if (asTools::IsNaN(val[i][j]))
                    {
                        asLogError(_("There are NaN values in the provided predictand ID vector."));
                        return false;
                    }
                }
            }
        }

        m_PredictandStationsIdsVect = val;

        return true;
    }

    VectorInt GetTimeArrayAnalogsIntervalDaysVector()
    {
        return m_TimeArrayAnalogsIntervalDaysVect;
    }

    bool SetTimeArrayAnalogsIntervalDaysVector(VectorInt val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided 'interval days' vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided 'interval days' vector."));
                    return false;
                }
            }
        }
        m_TimeArrayAnalogsIntervalDaysVect = val;
        return true;
    }

    VectorInt GetAnalogsNumberVector(int i_step)
    {
        return m_StepsVect[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumberVector(int i_step, VectorInt val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided analogs number vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided analogs number vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].AnalogsNumber = val;
        return true;
    }

    VectorString GetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDataId in the parameters object."));
            VectorString empty;
            return empty;
        }
    }

    bool SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided preprocess data ID vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided preprocess data ID vector."));
                    return false;
                }
            }
        }

        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset] = val;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }

        return true;
    }

    VectorFloat GetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            VectorFloat empty;
            return empty;
        }
    }

    bool SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided preprocess levels vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided preprocess levels vector."));
                    return false;
                }
            }
        }

        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }

        return true;
    }

    VectorDouble GetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessTimeHours (vect) in the parameters object."));
            VectorDouble empty;
            return empty;
        }
    }

    bool SetPreprocessTimeHoursVector(int i_step, int i_predictor, int i_dataset, VectorDouble val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided preprocess time (hours) vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided preprocess time (hours) vector."));
                    return false;
                }
            }
        }

        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset] = val;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.push_back(val);
        }

        return true;
    }

    VectorString GetPredictorDataIdVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided data ID vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided data ID vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].DataId = val;
        return true;
    }

    VectorFloat GetPredictorLevelVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided predictor levels vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided predictor levels vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Level = val;
        return true;
    }

    VectorDouble GetPredictorXminVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXminVector(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided Xmin vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided Xmin vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Xmin = val;
        return true;
    }

    VectorInt GetPredictorXptsnbVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnbVector(int i_step, int i_predictor, VectorInt val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided Xptsnb vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided Xptsnb vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Xptsnb = val;
        return true;
    }

    VectorDouble GetPredictorYminVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYminVector(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided Ymin vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided Ymin vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Ymin = val;
        return true;
    }

    VectorInt GetPredictorYptsnbVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnbVector(int i_step, int i_predictor, VectorInt val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided Yptsnb vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided Yptsnb vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Yptsnb = val;
        return true;
    }

    VectorDouble GetPredictorTimeHoursVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHoursVector(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided predictor time (hours) vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided predictor time (hours) vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].TimeHours = val;
        return true;
    }

    VectorString GetPredictorCriteriaVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided predictor criteria vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided predictor criteria vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Criteria = val;
        return true;
    }

    VectorFloat GetPredictorWeightVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeightVector(int i_step, int i_predictor, VectorFloat val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided predictor weights vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided predictor weights vector."));
                    return false;
                }
            }
        }
        m_StepsVect[i_step].Predictors[i_predictor].Weight = val;
        return true;
    }

    VectorString GetForecastScoreNameVector()
    {
        return m_ForecastScoreVect.Name;
    }

    bool SetForecastScoreNameVector(VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided forecast scores vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
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
        m_ForecastScoreVect.Name = val;
        return true;
    }

    VectorString GetForecastScoreTimeArrayModeVector()
    {
        return m_ForecastScoreVect.TimeArrayMode;
    }

    bool SetForecastScoreTimeArrayModeVector(VectorString val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided time array mode vector for the forecast score is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (val[i].IsEmpty())
                {
                    asLogError(_("There are NaN values in the provided time array mode vector for the forecast score."));
                    return false;
                }
            }
        }
        m_ForecastScoreVect.TimeArrayMode = val;
        return true;
    }

    VectorDouble GetForecastScoreTimeArrayDateVector()
    {
        return m_ForecastScoreVect.TimeArrayDate;
    }

    bool SetForecastScoreTimeArrayDateVector(VectorDouble val)
    {
        m_ForecastScoreVect.TimeArrayDate = val;
        return true;
    }

    VectorInt GetForecastScoreTimeArrayIntervalDaysVector()
    {
        return m_ForecastScoreVect.TimeArrayIntervalDays;
    }

    bool SetForecastScoreTimeArrayIntervalDaysVector(VectorInt val)
    {
        m_ForecastScoreVect.TimeArrayIntervalDays = val;
        return true;
    }

    VectorFloat GetForecastScorePostprocessDupliExpVector()
    {
        return m_ForecastScoreVect.PostprocessDupliExp;
    }

    bool SetForecastScorePostprocessDupliExpVector(VectorFloat val)
    {
        if (val.size()<1 && ForecastScoreNeedsPostprocessing())
        {
            asLogError(_("The provided 'PostprocessDupliExp' vector is empty."));
            return false;
        }
        else if (ForecastScoreNeedsPostprocessing())
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided 'PostprocessDupliExp' vector."));
                    return false;
                }
            }
        }
        m_ForecastScoreVect.PostprocessDupliExp = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit()
    {
        int lastrow = m_TimeArrayAnalogsIntervalDaysVect.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MinArray(&m_TimeArrayAnalogsIntervalDaysVect[0],&m_TimeArrayAnalogsIntervalDaysVect[lastrow]);
        return val;
    }

    int GetAnalogsNumberLowerLimit(int i_step)
    {
        int lastrow = m_StepsVect[i_step].AnalogsNumber.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MinArray(&m_StepsVect[i_step].AnalogsNumber[0],&m_StepsVect[i_step].AnalogsNumber[lastrow]);
        return val;
    }

    float GetPreprocessLevelLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].size()-1;
            wxASSERT(lastrow>=0);
            float val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            return NaNFloat;
        }
    }

    double GetPreprocessTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].size()-1;
            wxASSERT(lastrow>=0);
            double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessTimeHours (lower limit) in the parameters object."));
            return NaNDouble;
        }
    }

    float GetPredictorLevelLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Level.size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Level[0],&m_StepsVect[i_step].Predictors[i_predictor].Level[lastrow]);
        return val;
    }

    double GetPredictorXminLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Xmin.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Xmin[0],&m_StepsVect[i_step].Predictors[i_predictor].Xmin[lastrow]);
        return val;
    }

    int GetPredictorXptsnbLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Xptsnb.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Xptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Xptsnb[lastrow]);
        return val;
    }

    double GetPredictorYminLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Ymin.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Ymin[0],&m_StepsVect[i_step].Predictors[i_predictor].Ymin[lastrow]);
        return val;
    }

    int GetPredictorYptsnbLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Yptsnb.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Yptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Yptsnb[lastrow]);
        return val;
    }

    double GetPredictorTimeHoursLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].TimeHours.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].TimeHours[0],&m_StepsVect[i_step].Predictors[i_predictor].TimeHours[lastrow]);
        return val;
    }

    float GetPredictorWeightLowerLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Weight.size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Weight[0],&m_StepsVect[i_step].Predictors[i_predictor].Weight[lastrow]);
        return val;
    }

    double GetForecastScoreTimeArrayDateLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayDate.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MinArray(&m_ForecastScoreVect.TimeArrayDate[0],&m_ForecastScoreVect.TimeArrayDate[lastrow]);
        return val;
    }

    int GetForecastScoreTimeArrayIntervalDaysLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayIntervalDays.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MinArray(&m_ForecastScoreVect.TimeArrayIntervalDays[0],&m_ForecastScoreVect.TimeArrayIntervalDays[lastrow]);
        return val;
    }

    float GetForecastScorePostprocessDupliExpLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.PostprocessDupliExp.size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MinArray(&m_ForecastScoreVect.PostprocessDupliExp[0],&m_ForecastScoreVect.PostprocessDupliExp[lastrow]);
        return val;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit()
    {
        int lastrow = m_TimeArrayAnalogsIntervalDaysVect.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MaxArray(&m_TimeArrayAnalogsIntervalDaysVect[0],&m_TimeArrayAnalogsIntervalDaysVect[lastrow]);
        return val;
    }

    int GetAnalogsNumberUpperLimit(int i_step)
    {
        int lastrow = m_StepsVect[i_step].AnalogsNumber.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MaxArray(&m_StepsVect[i_step].AnalogsNumber[0],&m_StepsVect[i_step].AnalogsNumber[lastrow]);
        return val;
    }

    float GetPreprocessLevelUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].size()-1;
            wxASSERT(lastrow>=0);
            float val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            return NaNFloat;
        }
    }

    double GetPreprocessTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].size()-1;
            wxASSERT(lastrow>=0);
            double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessTimeHours (upper limit) in the parameters object."));
            return NaNDouble;
        }
    }

    float GetPredictorLevelUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Level.size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Level[0],&m_StepsVect[i_step].Predictors[i_predictor].Level[lastrow]);
        return val;
    }

    double GetPredictorXminUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Xmin.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Xmin[0],&m_StepsVect[i_step].Predictors[i_predictor].Xmin[lastrow]);
        return val;
    }

    int GetPredictorXptsnbUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Xptsnb.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Xptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Xptsnb[lastrow]);
        return val;
    }

    double GetPredictorYminUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Ymin.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Ymin[0],&m_StepsVect[i_step].Predictors[i_predictor].Ymin[lastrow]);
        return val;
    }

    int GetPredictorYptsnbUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Yptsnb.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Yptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Yptsnb[lastrow]);
        return val;
    }

    double GetPredictorTimeHoursUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].TimeHours.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].TimeHours[0],&m_StepsVect[i_step].Predictors[i_predictor].TimeHours[lastrow]);
        return val;
    }

    float GetPredictorWeightUpperLimit(int i_step, int i_predictor)
    {
        wxASSERT(m_StepsVect[i_step].Predictors.size()>i_predictor);
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Weight.size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Weight[0],&m_StepsVect[i_step].Predictors[i_predictor].Weight[lastrow]);
        return val;
    }

    double GetForecastScoreTimeArrayDateUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayDate.size()-1;
        wxASSERT(lastrow>=0);
        double val = asTools::MaxArray(&m_ForecastScoreVect.TimeArrayDate[0],&m_ForecastScoreVect.TimeArrayDate[lastrow]);
        return val;
    }

    int GetForecastScoreTimeArrayIntervalDaysUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayIntervalDays.size()-1;
        wxASSERT(lastrow>=0);
        int val = asTools::MaxArray(&m_ForecastScoreVect.TimeArrayIntervalDays[0],&m_ForecastScoreVect.TimeArrayIntervalDays[lastrow]);
        return val;
    }

    float GetForecastScorePostprocessDupliExpUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.PostprocessDupliExp.size()-1;
        wxASSERT(lastrow>=0);
        float val = asTools::MaxArray(&m_ForecastScoreVect.PostprocessDupliExp[0],&m_ForecastScoreVect.PostprocessDupliExp[lastrow]);
        return val;
    }

    int GetTimeArrayAnalogsIntervalDaysIteration()
    {
        if (m_TimeArrayAnalogsIntervalDaysVect.size()<2) return 0;
        int val = m_TimeArrayAnalogsIntervalDaysVect[1] - m_TimeArrayAnalogsIntervalDaysVect[0];
        return val;
    }

    int GetAnalogsNumberIteration(int i_step)
    {
        if (m_StepsVect[i_step].AnalogsNumber.size()<2) return 0;
        int val = m_StepsVect[i_step].AnalogsNumber[1] - m_StepsVect[i_step].AnalogsNumber[0];
        return val;
    }

    double GetPreprocessTimeHoursIteration(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            if (m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset].size()<2) return 0;
            double val = m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][1] - m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHours[i_dataset][0];
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessTimeHours (iteration) in the parameters object."));
            return NaNDouble;
        }
    }

    double GetPredictorXminIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Xmin.size()<2) return 0;
        int row = floor((float)m_StepsVect[i_step].Predictors[i_predictor].Xmin.size()/2.0);
        double val = m_StepsVect[i_step].Predictors[i_predictor].Xmin[row] - m_StepsVect[i_step].Predictors[i_predictor].Xmin[row-1];
        return val;
    }

    int GetPredictorXptsnbIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Xptsnb.size()<2) return 0;
        int val = m_StepsVect[i_step].Predictors[i_predictor].Xptsnb[1] - m_StepsVect[i_step].Predictors[i_predictor].Xptsnb[0];
        return val;
    }

    double GetPredictorYminIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Ymin.size()<2) return 0;
        int row = floor((float)m_StepsVect[i_step].Predictors[i_predictor].Ymin.size()/2.0);
        double val = m_StepsVect[i_step].Predictors[i_predictor].Ymin[row] - m_StepsVect[i_step].Predictors[i_predictor].Ymin[row-1];
        return val;
    }

    int GetPredictorYptsnbIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Yptsnb.size()<2) return 0;
        int val = m_StepsVect[i_step].Predictors[i_predictor].Yptsnb[1] - m_StepsVect[i_step].Predictors[i_predictor].Yptsnb[0];
        return val;
    }

    double GetPredictorTimeHoursIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].TimeHours.size()<2) return 0;
        double val = m_StepsVect[i_step].Predictors[i_predictor].TimeHours[1] - m_StepsVect[i_step].Predictors[i_predictor].TimeHours[0];
        return val;
    }

    float GetPredictorWeightIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Weight.size()<2) return 0;
        float val = m_StepsVect[i_step].Predictors[i_predictor].Weight[1] - m_StepsVect[i_step].Predictors[i_predictor].Weight[0];
        return val;
    }

    double GetForecastScoreTimeArrayDateIteration()
    {
        if (m_ForecastScoreVect.TimeArrayDate.size()<2) return 0;
        double val = m_ForecastScoreVect.TimeArrayDate[1] - m_ForecastScoreVect.TimeArrayDate[0];
        return val;
    }

    int GetForecastScoreTimeArrayIntervalDaysIteration()
    {
        if (m_ForecastScoreVect.TimeArrayIntervalDays.size()<2) return 0;
        int val = m_ForecastScoreVect.TimeArrayIntervalDays[1] - m_ForecastScoreVect.TimeArrayIntervalDays[0];
        return val;
    }

    float GetForecastScorePostprocessDupliExpIteration()
    {
        if (m_ForecastScoreVect.PostprocessDupliExp.size()<2) return 0;
        float val = m_ForecastScoreVect.PostprocessDupliExp[1] - m_ForecastScoreVect.PostprocessDupliExp[0];
        return val;
    }

protected:

private:
    VVectorInt m_PredictandStationsIdsVect;
    VectorInt m_TimeArrayAnalogsIntervalDaysVect;
    VectorParamsStepVect m_StepsVect;
    ParamsForecastScoreVect m_ForecastScoreVect;
};

#endif // ASPARAMETERSCALIBRATION_H
