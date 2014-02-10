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

    bool SetArchiveYearStart(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the archive start is null"));
            return false;
        }
        m_ArchiveYearStart = val;
        return true;
    }

    int GetArchiveYearEnd()
    {
        return m_ArchiveYearEnd;
    }

    bool SetArchiveYearEnd(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the archive end is null"));
            return false;
        }
        m_ArchiveYearEnd = val;
        return true;
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

    bool SetTimeArrayTargetTimeStepHours(double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the target time step is null"));
            return false;
        }
        m_TimeArrayTargetTimeStepHours = val;
        return true;
    }

    double GetTimeArrayAnalogsTimeStepHours()
    {
        return m_TimeArrayAnalogsTimeStepHours;
    }

    bool SetTimeArrayAnalogsTimeStepHours(double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the analogs time step is null"));
            return false;
        }
        m_TimeArrayAnalogsTimeStepHours = val;
        return true;
    }

    wxString GetTimeArrayTargetMode()
    {
        return m_TimeArrayTargetMode;
    }

    bool SetTimeArrayTargetMode(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the target time array mode is null"));
            return false;
        }
        m_TimeArrayTargetMode = val;
        return true;
    }

    wxString GetTimeArrayTargetPredictandSerieName()
    {
        return m_TimeArrayTargetPredictandSerieName;
    }

    bool SetTimeArrayTargetPredictandSerieName(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the predictand serie name is null"));
            return false;
        }
        m_TimeArrayTargetPredictandSerieName = val;
        return true;
    }

    float GetTimeArrayTargetPredictandMinThreshold()
    {
        return m_TimeArrayTargetPredictandMinThreshold;
    }

    bool SetTimeArrayTargetPredictandMinThreshold(float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictand min threshold is null"));
            return false;
        }
        m_TimeArrayTargetPredictandMinThreshold = val;
        return true;
    }

    float GetTimeArrayTargetPredictandMaxThreshold()
    {
        return m_TimeArrayTargetPredictandMaxThreshold;
    }

    bool SetTimeArrayTargetPredictandMaxThreshold(float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictand max threshold is null"));
            return false;
        }
        m_TimeArrayTargetPredictandMaxThreshold = val;
        return true;
    }

    wxString GetTimeArrayAnalogsMode()
    {
        return m_TimeArrayAnalogsMode;
    }

    bool SetTimeArrayAnalogsMode(const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the analogy time array mode is null"));
            return false;
        }
        m_TimeArrayAnalogsMode = val;
        return true;
    }

    int GetTimeArrayAnalogsExcludeDays()
    {
        return m_TimeArrayAnalogsExcludeDays;
    }

    bool SetTimeArrayAnalogsExcludeDays(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the 'exclude days' is null"));
            return false;
        }
        m_TimeArrayAnalogsExcludeDays = val;
        return true;
    }

    int GetTimeArrayAnalogsIntervalDays()
    {
        return m_TimeArrayAnalogsIntervalDays;
    }

    bool SetTimeArrayAnalogsIntervalDays(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the analogs interval days is null"));
            return false;
        }
        m_TimeArrayAnalogsIntervalDays = val;
        return true;
    }

    int GetPredictandStationId()
    {
        return m_PredictandStationId;
    }

    bool SetPredictandStationId(int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictand ID is null"));
            return false;
        }
        m_PredictandStationId = val;
        return true;
    }
    
    wxString GePredictandtDatasetId()
    {
        return m_PredictandDatasetId;
    }

    bool SetPredictandDatasetId(const wxString &val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the predictand dataset ID is null"));
            return false;
        }
        m_PredictandDatasetId = val;
        return true;
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

    bool SetPredictandDTimeHours(double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictand dTime (hours) is null"));
            return false;
        }
        m_PredictandDTimeHours = val;
        m_PredictandDTimeDays = val/24.0;
        return true;
    }

    wxString GetMethodName(int i_step)
    {
        return m_Steps[i_step].MethodName;
    }

    bool SetMethodName(int i_step, const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the method name is null"));
            return false;
        }
        m_Steps[i_step].MethodName = val;
        return true;
    }

    int GetAnalogsNumber(int i_step)
    {
        return m_Steps[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumber(int i_step, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the analogs number is null"));
            return false;
        }
        m_Steps[i_step].AnalogsNumber = val;
        return true;
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

    bool SetPreloadDTimeHours(int i_step, int i_predictor, VectorDouble val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided 'dTime (hours)' vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided 'dTime (hours)' vector."));
                    return false;
                }
            }
        }
        m_Steps[i_step].Predictors[i_predictor].PreloadDTimeHours = val;
        return true;
    }

    VectorFloat GetPreloadLevels(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadLevels;
    }

    bool SetPreloadLevels(int i_step, int i_predictor, VectorFloat val)
    {
        if (val.size()<1)
        {
            asLogError(_("The provided 'preload levels' vector is empty."));
            return false;
        }
        else
        {
            for (int i=0; i<val.size(); i++)
            {
                if (asTools::IsNaN(val[i]))
                {
                    asLogError(_("There are NaN values in the provided 'preload levels' vector."));
                    return false;
                }
            }
        }
        m_Steps[i_step].Predictors[i_predictor].PreloadLevels = val;
        return true;
    }

    double GetPreloadUmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadUmin;
    }

    bool SetPreloadUmin(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the preload Umin is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].PreloadUmin = val;
        return true;
    }

    int GetPreloadUptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadUptsnb;
    }

    bool SetPreloadUptsnb(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the preload points number on U is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].PreloadUptsnb = val;
        return true;
    }

    double GetPreloadVmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadVmin;
    }

    bool SetPreloadVmin(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the preload Vmin is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].PreloadVmin = val;
        return true;
    }

    int GetPreloadVptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadVptsnb;
    }

    bool SetPreloadVptsnb(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the preload points number on V is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].PreloadVptsnb = val;
        return true;
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

    bool SetPreprocessMethod(int i_step, int i_predictor, const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the preprocess method is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].PreprocessMethod = val;
        return true;
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

    bool SetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the preprocess dataset ID is null"));
            return false;
        }
        
        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessDatasetIds.push_back(val);
        }

        return true;
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

    bool SetPreprocessDataId(int i_step, int i_predictor, int i_dataset, const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the preprocess data ID is null"));
            return false;
        }

        if(m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessDataIds.push_back(val);
        }

        return true;
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

    bool SetPreprocessLevel(int i_step, int i_predictor, int i_dataset, float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the preprocess level is null"));
            return false;
        }

        if(m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.size()>=(unsigned)(i_dataset+1))
        {
            m_Steps[i_step].Predictors[i_predictor].PreprocessLevels[i_dataset] = val;
        }
        else
        {
            wxASSERT(m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.size()==i_dataset);
            m_Steps[i_step].Predictors[i_predictor].PreprocessLevels.push_back(val);
        }
        
        return true;
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

    bool SetPreprocessDTimeHours(int i_step, int i_predictor, int i_dataset, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the preprocess dTime (hours) is null"));
            return false;
        }

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

        return true;
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

    bool SetPredictorDatasetId(int i_step, int i_predictor, const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the predictor dataset is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].DatasetId = val;
        return true;
    }

    wxString GetPredictorDataId(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataId(int i_step, int i_predictor, wxString val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the predictor data is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].DataId = val;
        return true;
    }

    float GetPredictorLevel(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevel(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor level is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Level = val;
        return true;
    }

    wxString GetPredictorGridType(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].GridType;
    }

    bool SetPredictorGridType(int i_step, int i_predictor, wxString val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the predictor grid type is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].GridType = val;
        return true;
    }

    double GetPredictorUmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Umin;
    }

    bool SetPredictorUmin(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor Umin is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Umin = val;
        return true;
    }

    int GetPredictorUptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Uptsnb;
    }

    bool SetPredictorUptsnb(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor points number on U is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Uptsnb = val;
        return true;
    }

    double GetPredictorUstep(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Ustep;
    }

    bool SetPredictorUstep(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor U step is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Ustep = val;
        return true;
    }

    double GetPredictorUshift(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Ushift;
    }

    bool SetPredictorUshift(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor U shift is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Ushift = val;
        return true;
    }

    double GetPredictorVmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vmin;
    }

    bool SetPredictorVmin(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor Vmin is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Vmin = val;
        return true;
    }

    int GetPredictorVptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vptsnb;
    }

    bool SetPredictorVptsnb(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor points number on V is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Vptsnb = val;
        return true;
    }

    double GetPredictorVstep(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vstep;
    }

    bool SetPredictorVstep(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor V step is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Vstep = val;
        return true;
    }

    double GetPredictorVshift(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Vshift;
    }

    bool SetPredictorVshift(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor V shift is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Vshift = val;
        return true;
    }

    int GetPredictorFlatAllowed(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].FlatAllowed;
    }

    bool SetPredictorFlatAllowed(int i_step, int i_predictor, int val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the 'flat allowed' property is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].FlatAllowed = val;
        return true;
    }

    double GetPredictorDTimeHours(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DTimeHours;
    }

    bool SetPredictorDTimeHours(int i_step, int i_predictor, double val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor 'dTime (hours)' is null"));
            return false;
        }

        m_Steps[i_step].Predictors[i_predictor].DTimeHours = val;
        m_Steps[i_step].Predictors[i_predictor].DTimeDays = val/24.0;
        m_Steps[i_step].Predictors[i_predictor].TimeHour = fmod(val, 24.0);

        if(m_Steps[i_step].Predictors[i_predictor].TimeHour<0)
        {
            m_Steps[i_step].Predictors[i_predictor].TimeHour += 24.0;
        }
        
        return true;
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

    bool SetPredictorCriteria(int i_step, int i_predictor, const wxString& val)
    {
        if (val.IsEmpty())
        {
            asLogError(_("The provided value for the predictor criteria is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Criteria = val;
        return true;
    }

    float GetPredictorWeight(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeight(int i_step, int i_predictor, float val)
    {
        if (asTools::IsNaN(val))
        {
            asLogError(_("The provided value for the predictor weight is null"));
            return false;
        }
        m_Steps[i_step].Predictors[i_predictor].Weight = val;
        return true;
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
