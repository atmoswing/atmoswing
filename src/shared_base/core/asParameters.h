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
 
#ifndef ASPARAMETERS_H
#define ASPARAMETERS_H

#include "asIncludes.h"


class asParameters: public wxObject
{
public:

    // Structures
    typedef struct
    {
        wxString DatasetId;
        wxString DataId;
        bool Preload;
        VectorDouble PreloadDTimeHours;
        VectorFloat PreloadLevels;
        double PreloadUmin;
        int PreloadUptsnb;
        double PreloadVmin;
        int PreloadVptsnb;
        bool Preprocess;
        wxString PreprocessMethod;
        VectorString PreprocessDatasetIds;
        VectorString PreprocessDataIds;
        VectorFloat PreprocessLevels;
        VectorDouble PreprocessDTimeHours;
        VectorDouble PreprocessDTimeDays;
        VectorDouble PreprocessTimeHour;
        float Level;
        wxString GridType;
        double Umin;
        int Uptsnb;
        double Ustep;
        double Ushift;
        double Vmin;
        int Vptsnb;
        double Vstep;
        double Vshift;
        int FlatAllowed;
        double DTimeHours;
        double DTimeDays;
        double TimeHour;
        wxString Criteria;
        float Weight;
    } ParamsPredictor;

    typedef std::vector < ParamsPredictor > VectorParamsPredictors;

    typedef struct
    {
        wxString MethodName;
        int AnalogsNumber;
        VectorParamsPredictors Predictors;
    } ParamsStep;

    typedef std::vector < ParamsStep > VectorParamsStep;

    asParameters();
    virtual ~asParameters();

    bool IsOk();
    void AddStep();
    void AddPredictor(); // To the last step
    void AddPredictor(ParamsStep &step);
    void AddPredictor(int i_step);

    void SetSizes();

    bool FixAnalogsNb();

    VectorInt BuildVectorInt(int min, int max, int step);

    VectorInt BuildVectorInt(wxString txt);

    VectorFloat BuildVectorFloat(float min, float max, float step);

    VectorFloat BuildVectorFloat(wxString txt);

    VectorDouble BuildVectorDouble(double min, double max, double step);

    VectorDouble BuildVectorDouble(wxString txt);

    VectorString BuildVectorString(wxString txt);

    virtual bool LoadFromFile(const wxString &filePath = wxEmptyString);

    bool FixTimeShift();

    bool FixWeights();

    bool FixCoordinates();

    wxString Print();

    bool PrintAndSaveTemp(const wxString &filePath = wxEmptyString);


    wxString GetDateProcessed()
    {
        return m_DateProcessed;
    }

    void SetDateProcessed(const wxString& val)
    {
        m_DateProcessed = val;
    }

    int GetArchiveYearStart()
    {
        return m_ArchiveYearStart;
    }

    void SetArchiveYearStart(int val)
    {
        m_ArchiveYearStart = val;
    }

    int GetArchiveYearEnd()
    {
        return m_ArchiveYearEnd;
    }

    void SetArchiveYearEnd(int val)
    {
        m_ArchiveYearEnd = val;
    }

    int GetTimeShiftDays()
    {
        return m_TimeShiftDays;
    }

    int GetTimeSpanDays()
    {
        return m_TimeSpanDays;
    }

    double GetTimeArrayTargetTimeStepHours()
    {
        return m_TimeArrayTargetTimeStepHours;
    }

    void SetTimeArrayTargetTimeStepHours(double val)
    {
        m_TimeArrayTargetTimeStepHours = val;
    }

    double GetTimeArrayAnalogsTimeStepHours()
    {
        return m_TimeArrayAnalogsTimeStepHours;
    }

    void SetTimeArrayAnalogsTimeStepHours(double val)
    {
        m_TimeArrayAnalogsTimeStepHours = val;
    }

    wxString GetTimeArrayTargetMode()
    {
        return m_TimeArrayTargetMode;
    }

    void SetTimeArrayTargetMode(const wxString& val)
    {
        m_TimeArrayTargetMode = val;
    }

    wxString GetTimeArrayTargetPredictandSerieName()
    {
        return m_TimeArrayTargetPredictandSerieName;
    }

    void SetTimeArrayTargetPredictandSerieName(const wxString& val)
    {
        m_TimeArrayTargetPredictandSerieName = val;
    }

    float GetTimeArrayTargetPredictandMinThreshold()
    {
        return m_TimeArrayTargetPredictandMinThreshold;
    }

    void SetTimeArrayTargetPredictandMinThreshold(float val)
    {
        m_TimeArrayTargetPredictandMinThreshold = val;
    }

    float GetTimeArrayTargetPredictandMaxThreshold()
    {
        return m_TimeArrayTargetPredictandMaxThreshold;
    }

    void SetTimeArrayTargetPredictandMaxThreshold(float val)
    {
        m_TimeArrayTargetPredictandMaxThreshold = val;
    }

    wxString GetTimeArrayAnalogsMode()
    {
        return m_TimeArrayAnalogsMode;
    }

    void SetTimeArrayAnalogsMode(const wxString& val)
    {
        m_TimeArrayAnalogsMode = val;
    }

    int GetTimeArrayAnalogsExcludeDays()
    {
        return m_TimeArrayAnalogsExcludeDays;
    }

    void SetTimeArrayAnalogsExcludeDays(int val)
    {
        m_TimeArrayAnalogsExcludeDays = val;
    }

    int GetTimeArrayAnalogsIntervalDays()
    {
        return m_TimeArrayAnalogsIntervalDays;
    }

    void SetTimeArrayAnalogsIntervalDays(int val)
    {
        m_TimeArrayAnalogsIntervalDays = val;
    }

    int GetPredictandStationId()
    {
        return m_PredictandStationId;
    }

    void SetPredictandStationId(int val)
    {
        m_PredictandStationId = val;
    }
    
    wxString GePredictandtDatasetId()
    {
        return m_PredictandDatasetId;
    }

    void SetPredictandDatasetId(const wxString &val)
    {
        m_PredictandDatasetId = val;
    }

    DataParameter GetPredictandParameter()
    {
        return m_PredictandParameter;
    }

    void SetPredictandParameter(DataParameter val)
    {
        m_PredictandParameter = val;
    }

    DataTemporalResolution GetPredictandTemporalResolution()
    {
        return m_PredictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(DataTemporalResolution val)
    {
        m_PredictandTemporalResolution = val;
    }

    DataSpatialAggregation GetPredictandSpatialAggregation()
    {
        return m_PredictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(DataSpatialAggregation val)
    {
        m_PredictandSpatialAggregation = val;
    }

    double GetPredictandDTimeHours()
    {
        return m_PredictandDTimeHours;
    }

    double GetPredictandDTimeDays()
    {
        return m_PredictandDTimeDays;
    }

    void SetPredictandDTimeHours(double val)
    {
        m_PredictandDTimeHours = val;
        m_PredictandDTimeDays = val/24.0;
    }

    wxString GetMethodName(int i_step)
    {
        return m_Steps[i_step].MethodName;
    }

    void SetMethodName(int i_step, const wxString& val)
    {
        m_Steps[i_step].MethodName = val;
    }

    int GetAnalogsNumber(int i_step)
    {
        return m_Steps[i_step].AnalogsNumber;
    }

    void SetAnalogsNumber(int i_step, int val)
    {
        m_Steps[i_step].AnalogsNumber = val;
    }

    bool NeedsPreloading(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Preload;
    }

    void SetPreload(int i_step, int i_predictor, bool val)
    {
        m_Steps[i_step].Predictors[i_predictor].Preload = val;
    }

    VectorDouble GetPreloadDTimeHours(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadDTimeHours;
    }

    void SetPreloadDTimeHours(int i_step, int i_predictor, VectorDouble val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreloadDTimeHours = val;
    }

    VectorFloat GetPreloadLevels(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadLevels;
    }

    void SetPreloadLevels(int i_step, int i_predictor, VectorFloat val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreloadLevels = val;
    }

    double GetPreloadUmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadUmin;
    }

    void SetPreloadUmin(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreloadUmin = val;
    }

    int GetPreloadUptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadUptsnb;
    }

    void SetPreloadUptsnb(int i_step, int i_predictor, int val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreloadUptsnb = val;
    }

    double GetPreloadVmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadVmin;
    }

    void SetPreloadVmin(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreloadVmin = val;
    }

    int GetPreloadVptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadVptsnb;
    }

    void SetPreloadVptsnb(int i_step, int i_predictor, int val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreloadVptsnb = val;
    }

    bool NeedsPreprocessing(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Preprocess;
    }

    void SetPreprocess(int i_step, int i_predictor, bool val)
    {
        m_Steps[i_step].Predictors[i_predictor].Preprocess = val;
    }

    int GetPreprocessSize(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.size();
    }

    wxString GetPreprocessMethod(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreprocessMethod;
    }

    void SetPreprocessMethod(int i_step, int i_predictor, const wxString& val)
    {
        m_Steps[i_step].Predictors[i_predictor].PreprocessMethod = val;
    }

    wxString GetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            return m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDatasetIds in the parameters object."));
            return wxEmptyString;
        }
    }

    void SetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.push_back(val);
        }
    }

    wxString GetPreprocessDataId(int i_step, int i_predictor, int i_dataset)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.size()>=(unsigned)(i_dataset+1))
        {
            return m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDataIds in the parameters object."));
            return wxEmptyString;
        }
    }

    void SetPreprocessDataId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.push_back(val);
        }
    }

    float GetPreprocessLevel(int i_step, int i_predictor, int i_dataset)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            return m_Steps[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessLevels in the parameters object."));
            return NaNFloat;
        }
    }

    void SetPreprocessLevel(int i_step, int i_predictor, int i_dataset, float val)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
    }

    double GetPreprocessDTimeHours(int i_step, int i_predictor, int i_dataset)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            return m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDTimeHours (std) in the parameters object."));
            return NaNDouble;
        }
    }

    void SetPreprocessDTimeHours(int i_step, int i_predictor, int i_dataset, double val)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeHours[i_dataset] = val;
            m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeDays[i_dataset] = val/24.0;
            m_Steps[i_step].Predictors[i_predictor].PreprocessTimeHour[i_dataset] = fmod(val, 24.0);
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeHours.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeHours.push_back(val);
            m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeDays.push_back(val/24.0);
            m_Steps[i_step].Predictors[i_predictor].PreprocessTimeHour.push_back(fmod(val, 24.0));
        }
    }

    double GetPreprocessDTimeDays(int i_step, int i_predictor, int i_dataset)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeDays.size()>=(unsigned)(i_dataset+1))
        {
            return m_Steps[i_step].Predictors[i_predictor].PreprocessDTimeDays[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessDTimeDays in the parameters object."));
            return NaNDouble;
        }
    }

    double GetPreprocessTimeHour(int i_step, int i_predictor, int i_dataset)
    {
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessTimeHour.size()>=(unsigned)(i_dataset+1))
        {
            return m_Steps[i_step].Predictors[i_predictor].PreprocessTimeHour[i_dataset];
        }
        else
        {
            asLogError(_("Trying to access to an element outside of PreprocessTimeHour in the parameters object."));
            return NaNDouble;
        }
    }

    wxString GetPredictorDatasetId(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DatasetId;
    }

    void SetPredictorDatasetId(int i_step, int i_predictor, const wxString& val)
    {
        m_Steps[i_step].Predictors[i_predictor].DatasetId = val;
    }

    wxString GetPredictorDataId(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DataId;
    }

    void SetPredictorDataId(int i_step, int i_predictor, wxString val)
    {
        m_Steps[i_step].Predictors[i_predictor].DataId = val;
    }

    float GetPredictorLevel(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Level;
    }

    void SetPredictorLevel(int i_step, int i_predictor, float val)
    {
        m_Steps[i_step].Predictors[i_predictor].Level = val;
    }

    wxString GetPredictorGridType(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].GridType;
    }

    void SetPredictorGridType(int i_step, int i_predictor, wxString val)
    {
        m_Steps[i_step].Predictors[i_predictor].GridType = val;
    }

    double GetPredictorUmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Umin;
    }

    void SetPredictorUmin(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].Umin = val;
    }

    int GetPredictorUptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Uptsnb;
    }

    void SetPredictorUptsnb(int i_step, int i_predictor, int val)
    {
        m_Steps[i_step].Predictors[i_predictor].Uptsnb = val;
    }

    double GetPredictorUstep(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Ustep;
    }

    void SetPredictorUstep(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].Ustep = val;
    }

    double GetPredictorUshift(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Ushift;
    }

    void SetPredictorUshift(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].Ushift = val;
    }

    double GetPredictorVmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vmin;
    }

    void SetPredictorVmin(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].Vmin = val;
    }

    int GetPredictorVptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vptsnb;
    }

    void SetPredictorVptsnb(int i_step, int i_predictor, int val)
    {
        m_Steps[i_step].Predictors[i_predictor].Vptsnb = val;
    }

    double GetPredictorVstep(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vstep;
    }

    void SetPredictorVstep(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].Vstep = val;
    }

    double GetPredictorVshift(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vshift;
    }

    void SetPredictorVshift(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].Vshift = val;
    }

    int GetPredictorFlatAllowed(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].FlatAllowed;
    }

    void SetPredictorFlatAllowed(int i_step, int i_predictor, int val)
    {
        m_Steps[i_step].Predictors[i_predictor].FlatAllowed = val;
    }

    double GetPredictorDTimeHours(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DTimeHours;
    }

    void SetPredictorDTimeHours(int i_step, int i_predictor, double val)
    {
        m_Steps[i_step].Predictors[i_predictor].DTimeHours = val;
        m_Steps[i_step].Predictors[i_predictor].DTimeDays = val/24.0;
        m_Steps[i_step].Predictors[i_predictor].TimeHour = fmod(val, 24.0);

        if(m_Steps[i_step].Predictors[i_predictor].TimeHour<0)
        {
            m_Steps[i_step].Predictors[i_predictor].TimeHour += 24.0;
        }
    }

    double GetPredictorDTimeDays(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DTimeDays;
    }

    double GetPredictorTimeHour(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].TimeHour;
    }

    wxString GetPredictorCriteria(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Criteria;
    }

    void SetPredictorCriteria(int i_step, int i_predictor, const wxString& val)
    {
        m_Steps[i_step].Predictors[i_predictor].Criteria = val;
    }

    float GetPredictorWeight(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Weight;
    }

    void SetPredictorWeight(int i_step, int i_predictor, float val)
    {
        m_Steps[i_step].Predictors[i_predictor].Weight = val;
    }

    int GetStepsNb()
    {
        return m_StepsNb;
    }

    VectorInt GetVectorPredictorsNb()
    {
        return m_PredictorsNb;
    }

    int GetPredictorsNb(int i_step)
    {
        wxASSERT_MSG((unsigned)i_step<m_PredictorsNb.size(), wxString::Format(_("Trying to access element %d of m_PredictorsNb of size %d."), i_step+1, (int)m_PredictorsNb.size()));
        return m_PredictorsNb[i_step];
    }


protected:
    int m_ArchiveYearStart;
    int m_ArchiveYearEnd;
    int m_TimeArrayAnalogsIntervalDays;
    int m_PredictandStationId;

private:
    VectorParamsStep m_Steps; // Set as private to force use of setters.
    VectorInt m_PredictorsNb;
    int m_StepsNb;
    wxString m_DateProcessed;
    int m_TimeShiftDays;
    int m_TimeSpanDays;
    wxString m_TimeArrayTargetMode;
    double m_TimeArrayTargetTimeStepHours;
    wxString m_TimeArrayTargetPredictandSerieName;
    float m_TimeArrayTargetPredictandMinThreshold;
    float m_TimeArrayTargetPredictandMaxThreshold;
    wxString m_TimeArrayAnalogsMode;
    double m_TimeArrayAnalogsTimeStepHours;
    int m_TimeArrayAnalogsExcludeDays;
    DataParameter m_PredictandParameter;
    DataTemporalResolution m_PredictandTemporalResolution;
    DataSpatialAggregation m_PredictandSpatialAggregation;
    wxString m_PredictandDatasetId;
    double m_PredictandDTimeHours;
    double m_PredictandDTimeDays;

};

#endif // ASPARAMETERS_H
