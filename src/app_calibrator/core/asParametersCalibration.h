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

    void InitValues();


    VectorInt GetPredictandStationsIdVector()
    {
        return m_PredictandStationsIdVect;
    }

    void SetPredictandStationsIdVector(VectorInt val)
    {
        m_PredictandStationsIdVect = val;
    }

    VectorInt GetTimeArrayAnalogsIntervalDaysVector()
    {
        return m_TimeArrayAnalogsIntervalDaysVect;
    }

    void SetTimeArrayAnalogsIntervalDaysVector(VectorInt val)
    {
        m_TimeArrayAnalogsIntervalDaysVect = val;
    }

    VectorInt GetAnalogsNumberVector(int i_step)
    {
        return m_StepsVect[i_step].AnalogsNumber;
    }

    void SetAnalogsNumberVector(int i_step, VectorInt val)
    {
        m_StepsVect[i_step].AnalogsNumber = val;
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

    void SetPreprocessDataIdVector(int i_step, int i_predictor, int i_dataset, VectorString val)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId[i_dataset] = val;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDataId.push_back(val);
        }
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

    void SetPreprocessLevelVector(int i_step, int i_predictor, int i_dataset, VectorFloat val)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
    }

    VectorDouble GetPreprocessDTimeHoursVector(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            return m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDTimeHours (vect) in the parameters object."));
            VectorDouble empty;
            return empty;
        }
    }

    void SetPreprocessDTimeHoursVector(int i_step, int i_predictor, int i_dataset, VectorDouble val)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset] = val;
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeDays[i_dataset].clear();
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHour[i_dataset].clear();
            VectorDouble dTimeDays;
            VectorDouble timeHour;
            for(unsigned int i=0;i<val.size();i++)
            {
                dTimeDays.push_back((double)val[i]/24);
                timeHour.push_back(fmod(val[i], 24));
            }
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeDays[i_dataset] = dTimeDays;
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHour[i_dataset] = timeHour;
        }
        else
        {
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours.push_back(val);
            VectorDouble dTimeDays;
            VectorDouble timeHour;
            for(unsigned int i=0;i<val.size();i++)
            {
                dTimeDays.push_back((double)val[i]/24);
                timeHour.push_back(fmod(val[i], 24));
            }
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeDays.push_back(dTimeDays);
            m_StepsVect[i_step].Predictors[i_predictor].PreprocessTimeHour.push_back(timeHour);
        }
    }

    VectorString GetPredictorDataIdVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].DataId;
    }

    void SetPredictorDataIdVector(int i_step, int i_predictor, VectorString val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].DataId = val;
    }

    VectorFloat GetPredictorLevelVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Level;
    }

    void SetPredictorLevelVector(int i_step, int i_predictor, VectorFloat val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Level = val;
    }

    VectorDouble GetPredictorUminVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUminVector(int i_step, int i_predictor, VectorDouble val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Umin = val;
    }

    VectorInt GetPredictorUptsnbVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnbVector(int i_step, int i_predictor, VectorInt val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    VectorDouble GetPredictorVminVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVminVector(int i_step, int i_predictor, VectorDouble val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Vmin = val;
    }

    VectorInt GetPredictorVptsnbVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnbVector(int i_step, int i_predictor, VectorInt val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    VectorDouble GetPredictorDTimeHoursVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].DTimeHours;
    }

    void SetPredictorDTimeHoursVector(int i_step, int i_predictor, VectorDouble val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].DTimeHours = val;
    }

    VectorString GetPredictorCriteriaVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Criteria;
    }

    void SetPredictorCriteriaVector(int i_step, int i_predictor, VectorString val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Criteria = val;
    }

    VectorFloat GetPredictorWeightVector(int i_step, int i_predictor)
    {
        return m_StepsVect[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeightVector(int i_step, int i_predictor, VectorFloat val)
    {
        m_StepsVect[i_step].Predictors[i_predictor].Weight = val;
    }

    VectorString GetForecastScoreNameVector()
    {
        return m_ForecastScoreVect.Name;
    }

    void SetForecastScoreNameVector(VectorString val)
    {
        m_ForecastScoreVect.Name = val;
    }

    VectorInt GetForecastScoreAnalogsNumberVector()
    {
        return m_ForecastScoreVect.AnalogsNumber;
    }

    void SetForecastScoreAnalogsNumberVector(VectorInt val)
    {
        m_ForecastScoreVect.AnalogsNumber = val;
    }

    VectorString GetForecastScoreTimeArrayModeVector()
    {
        return m_ForecastScoreVect.TimeArrayMode;
    }

    void SetForecastScoreTimeArrayModeVector(VectorString val)
    {
        m_ForecastScoreVect.TimeArrayMode = val;
    }

    VectorDouble GetForecastScoreTimeArrayDateVector()
    {
        return m_ForecastScoreVect.TimeArrayDate;
    }

    void SetForecastScoreTimeArrayDateVector(VectorDouble val)
    {
        m_ForecastScoreVect.TimeArrayDate = val;
    }

    VectorInt GetForecastScoreTimeArrayIntervalDaysVector()
    {
        return m_ForecastScoreVect.TimeArrayIntervalDays;
    }

    void SetForecastScoreTimeArrayIntervalDaysVector(VectorInt val)
    {
        m_ForecastScoreVect.TimeArrayIntervalDays = val;
    }

    VectorFloat GetForecastScorePostprocessDupliExpVector()
    {
        return m_ForecastScoreVect.PostprocessDupliExp;
    }

    void SetForecastScorePostprocessDupliExpVector(VectorFloat val)
    {
        m_ForecastScoreVect.PostprocessDupliExp = val;
    }

    int GetPredictandStationIdLowerLimit()
    {
        int lastrow = m_PredictandStationsIdVect.size()-1;
        int val = asTools::MinArray(&m_PredictandStationsIdVect[0],&m_PredictandStationsIdVect[lastrow]);
        return val;
    }

    int GetTimeArrayAnalogsIntervalDaysLowerLimit()
    {
        int lastrow = m_TimeArrayAnalogsIntervalDaysVect.size()-1;
        int val = asTools::MinArray(&m_TimeArrayAnalogsIntervalDaysVect[0],&m_TimeArrayAnalogsIntervalDaysVect[lastrow]);
        return val;
    }

    int GetAnalogsNumberLowerLimit(int i_step)
    {
        int lastrow = m_StepsVect[i_step].AnalogsNumber.size()-1;
        int val = asTools::MinArray(&m_StepsVect[i_step].AnalogsNumber[0],&m_StepsVect[i_step].AnalogsNumber[lastrow]);
        return val;
    }

    float GetPreprocessLevelLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].size()-1;
            float val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            return NaNFloat;
        }
    }

    double GetPreprocessDTimeHoursLowerLimit(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset].size()-1;
            double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDTimeHours (lower limit) in the parameters object."));
            return NaNDouble;
        }
    }

    float GetPredictorLevelLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Level.size()-1;
        float val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Level[0],&m_StepsVect[i_step].Predictors[i_predictor].Level[lastrow]);
        return val;
    }

    double GetPredictorUminLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Umin.size()-1;
        double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Umin[0],&m_StepsVect[i_step].Predictors[i_predictor].Umin[lastrow]);
        return val;
    }

    int GetPredictorUptsnbLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Uptsnb.size()-1;
        int val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Uptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Uptsnb[lastrow]);
        return val;
    }

    double GetPredictorVminLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Vmin.size()-1;
        double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Vmin[0],&m_StepsVect[i_step].Predictors[i_predictor].Vmin[lastrow]);
        return val;
    }

    int GetPredictorVptsnbLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Vptsnb.size()-1;
        int val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Vptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Vptsnb[lastrow]);
        return val;
    }

    double GetPredictorDTimeHoursLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].DTimeHours.size()-1;
        double val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].DTimeHours[0],&m_StepsVect[i_step].Predictors[i_predictor].DTimeHours[lastrow]);
        return val;
    }

    float GetPredictorWeightLowerLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Weight.size()-1;
        float val = asTools::MinArray(&m_StepsVect[i_step].Predictors[i_predictor].Weight[0],&m_StepsVect[i_step].Predictors[i_predictor].Weight[lastrow]);
        return val;
    }

    int GetForecastScoreAnalogsNumberLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.AnalogsNumber.size()-1;
        int val = asTools::MinArray(&m_ForecastScoreVect.AnalogsNumber[0],&m_ForecastScoreVect.AnalogsNumber[lastrow]);
        return val;
    }

    double GetForecastScoreTimeArrayDateLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayDate.size()-1;
        double val = asTools::MinArray(&m_ForecastScoreVect.TimeArrayDate[0],&m_ForecastScoreVect.TimeArrayDate[lastrow]);
        return val;
    }

    int GetForecastScoreTimeArrayIntervalDaysLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayIntervalDays.size()-1;
        int val = asTools::MinArray(&m_ForecastScoreVect.TimeArrayIntervalDays[0],&m_ForecastScoreVect.TimeArrayIntervalDays[lastrow]);
        return val;
    }

    float GetForecastScorePostprocessDupliExpLowerLimit()
    {
        int lastrow = m_ForecastScoreVect.PostprocessDupliExp.size()-1;
        float val = asTools::MinArray(&m_ForecastScoreVect.PostprocessDupliExp[0],&m_ForecastScoreVect.PostprocessDupliExp[lastrow]);
        return val;
    }

    int GetPredictandStationsIdUpperLimit()
    {
        int lastrow = m_PredictandStationsIdVect.size()-1;
        int val = asTools::MaxArray(&m_PredictandStationsIdVect[0],&m_PredictandStationsIdVect[lastrow]);
        return val;
    }

    int GetTimeArrayAnalogsIntervalDaysUpperLimit()
    {
        int lastrow = m_TimeArrayAnalogsIntervalDaysVect.size()-1;
        int val = asTools::MaxArray(&m_TimeArrayAnalogsIntervalDaysVect[0],&m_TimeArrayAnalogsIntervalDaysVect[lastrow]);
        return val;
    }

    int GetAnalogsNumberUpperLimit(int i_step)
    {
        int lastrow = m_StepsVect[i_step].AnalogsNumber.size()-1;
        int val = asTools::MaxArray(&m_StepsVect[i_step].AnalogsNumber[0],&m_StepsVect[i_step].AnalogsNumber[lastrow]);
        return val;
    }

    float GetPreprocessLevelUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset].size()-1;
            float val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            return NaNFloat;
        }
    }

    double GetPreprocessDTimeHoursUpperLimit(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            int lastrow = m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset].size()-1;
            double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset][0],&m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset][lastrow]);
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDTimeHours (upper limit) in the parameters object."));
            return NaNDouble;
        }
    }

    float GetPredictorLevelUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Level.size()-1;
        float val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Level[0],&m_StepsVect[i_step].Predictors[i_predictor].Level[lastrow]);
        return val;
    }

    double GetPredictorUminUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Umin.size()-1;
        double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Umin[0],&m_StepsVect[i_step].Predictors[i_predictor].Umin[lastrow]);
        return val;
    }

    int GetPredictorUptsnbUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Uptsnb.size()-1;
        int val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Uptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Uptsnb[lastrow]);
        return val;
    }

    double GetPredictorVminUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Vmin.size()-1;
        double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Vmin[0],&m_StepsVect[i_step].Predictors[i_predictor].Vmin[lastrow]);
        return val;
    }

    int GetPredictorVptsnbUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Vptsnb.size()-1;
        int val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Vptsnb[0],&m_StepsVect[i_step].Predictors[i_predictor].Vptsnb[lastrow]);
        return val;
    }

    double GetPredictorDTimeHoursUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].DTimeHours.size()-1;
        double val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].DTimeHours[0],&m_StepsVect[i_step].Predictors[i_predictor].DTimeHours[lastrow]);
        return val;
    }

    float GetPredictorWeightUpperLimit(int i_step, int i_predictor)
    {
        int lastrow = m_StepsVect[i_step].Predictors[i_predictor].Weight.size()-1;
        float val = asTools::MaxArray(&m_StepsVect[i_step].Predictors[i_predictor].Weight[0],&m_StepsVect[i_step].Predictors[i_predictor].Weight[lastrow]);
        return val;
    }

    int GetForecastScoreAnalogsNumberUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.AnalogsNumber.size()-1;
        int val = asTools::MaxArray(&m_ForecastScoreVect.AnalogsNumber[0],&m_ForecastScoreVect.AnalogsNumber[lastrow]);
        return val;
    }

    double GetForecastScoreTimeArrayDateUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayDate.size()-1;
        double val = asTools::MaxArray(&m_ForecastScoreVect.TimeArrayDate[0],&m_ForecastScoreVect.TimeArrayDate[lastrow]);
        return val;
    }

    int GetForecastScoreTimeArrayIntervalDaysUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.TimeArrayIntervalDays.size()-1;
        int val = asTools::MaxArray(&m_ForecastScoreVect.TimeArrayIntervalDays[0],&m_ForecastScoreVect.TimeArrayIntervalDays[lastrow]);
        return val;
    }

    float GetForecastScorePostprocessDupliExpUpperLimit()
    {
        int lastrow = m_ForecastScoreVect.PostprocessDupliExp.size()-1;
        float val = asTools::MaxArray(&m_ForecastScoreVect.PostprocessDupliExp[0],&m_ForecastScoreVect.PostprocessDupliExp[lastrow]);
        return val;
    }

    int GetPredictandStationsIdIteration()
    {
        if (m_PredictandStationsIdVect.size()<2) return 0;
        int val = m_PredictandStationsIdVect[1] - m_PredictandStationsIdVect[0];
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

    double GetPreprocessDTimeHoursIteration(int i_step, int i_predictor, int i_dataset)
    {
        if(m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            if (m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset].size()<2) return 0;
            double val = m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset][1] - m_StepsVect[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset][0];
            return val;
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDTimeHours (iteration) in the parameters object."));
            return NaNDouble;
        }
    }

    double GetPredictorUminIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Umin.size()<2) return 0;
        int row = floor((float)m_StepsVect[i_step].Predictors[i_predictor].Umin.size()/2.0);
        double val = m_StepsVect[i_step].Predictors[i_predictor].Umin[row] - m_StepsVect[i_step].Predictors[i_predictor].Umin[row-1];
        return val;
    }

    int GetPredictorUptsnbIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Uptsnb.size()<2) return 0;
        int val = m_StepsVect[i_step].Predictors[i_predictor].Uptsnb[1] - m_StepsVect[i_step].Predictors[i_predictor].Uptsnb[0];
        return val;
    }

    double GetPredictorVminIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Vmin.size()<2) return 0;
        int row = floor((float)m_StepsVect[i_step].Predictors[i_predictor].Vmin.size()/2.0);
        double val = m_StepsVect[i_step].Predictors[i_predictor].Vmin[row] - m_StepsVect[i_step].Predictors[i_predictor].Vmin[row-1];
        return val;
    }

    int GetPredictorVptsnbIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Vptsnb.size()<2) return 0;
        int val = m_StepsVect[i_step].Predictors[i_predictor].Vptsnb[1] - m_StepsVect[i_step].Predictors[i_predictor].Vptsnb[0];
        return val;
    }

    double GetPredictorDTimeHoursIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].DTimeHours.size()<2) return 0;
        double val = m_StepsVect[i_step].Predictors[i_predictor].DTimeHours[1] - m_StepsVect[i_step].Predictors[i_predictor].DTimeHours[0];
        return val;
    }

    float GetPredictorWeightIteration(int i_step, int i_predictor)
    {
        if (m_StepsVect[i_step].Predictors[i_predictor].Weight.size()<2) return 0;
        float val = m_StepsVect[i_step].Predictors[i_predictor].Weight[1] - m_StepsVect[i_step].Predictors[i_predictor].Weight[0];
        return val;
    }

    int GetForecastScoreAnalogsNumberIteration()
    {
        if (m_ForecastScoreVect.AnalogsNumber.size()<2) return 0;
        int val = m_ForecastScoreVect.AnalogsNumber[1] - m_ForecastScoreVect.AnalogsNumber[0];
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
    VectorInt m_PredictandStationsIdVect;
    VectorInt m_TimeArrayAnalogsIntervalDaysVect;
    VectorParamsStepVect m_StepsVect;
    ParamsForecastScoreVect m_ForecastScoreVect;
};

#endif // ASPARAMETERSCALIBRATION_H
