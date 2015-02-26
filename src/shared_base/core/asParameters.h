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
        VectorDouble PreloadTimeHours;
        VectorFloat PreloadLevels;
        double PreloadXmin;
        int PreloadXptsnb;
        double PreloadYmin;
        int PreloadYptsnb;
        bool Preprocess;
        wxString PreprocessMethod;
        VectorString PreprocessDatasetIds;
        VectorString PreprocessDataIds;
        VectorFloat PreprocessLevels;
        VectorDouble PreprocessTimeHours;
        float Level;
        wxString GridType;
        double Xmin;
        int Xptsnb;
        double Xstep;
        double Xshift;
        double Ymin;
        int Yptsnb;
        double Ystep;
        double Yshift;
        int FlatAllowed;
        double TimeHours;
        wxString Criteria;
        float Weight;
    } ParamsPredictor;

    typedef std::vector < ParamsPredictor > VectorParamsPredictors;

    typedef struct
    {
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

    virtual bool LoadFromFile(const wxString &filePath = wxEmptyString);

    bool SetSpatialWindowProperties();

    bool SetPreloadingProperties();

    bool InputsOK();

    wxString GetPredictandStationIdsString();

    bool FixTimeLimits();

    bool FixWeights();

    bool FixCoordinates();

    wxString Print();

    bool PrintAndSaveTemp(const wxString &filePath = wxEmptyString);

    bool GetValuesFromString(wxString stringVals); // We copy the string as we'll modify it.

    bool SetPredictandStationIds(wxString val);
    
    wxString GetMethodId()
    {
        return m_MethodId;
    }

    void SetMethodId(const wxString& val)
    {
        m_MethodId = val;
    }
    
    wxString GetMethodIdDisplay()
    {
        return m_MethodIdDisplay;
    }

    void SetMethodIdDisplay(const wxString& val)
    {
        m_MethodIdDisplay = val;
    }
    
    wxString GetSpecificTag()
    {
        return m_SpecificTag;
    }

    void SetSpecificTag(const wxString& val)
    {
        m_SpecificTag = val;
    }
    
    wxString GetSpecificTagDisplay()
    {
        return m_SpecificTagDisplay;
    }

    void SetSpecificTagDisplay(const wxString& val)
    {
        m_SpecificTagDisplay = val;
    }
    
    wxString GetDescription()
    {
        return m_Description;
    }

    void SetDescription(const wxString& val)
    {
        m_Description = val;
    }

    wxString GetDateProcessed()
    {
        return m_DateProcessed;
    }

    void SetDateProcessed(const wxString& val)
    {
        m_DateProcessed = val;
    }

    bool SetArchiveYearStart(int val)
    {
        m_ArchiveStart = asTime::GetMJD(val, 1, 1);
        return true;
    }

    bool SetArchiveYearEnd(int val)
    {
        m_ArchiveEnd = asTime::GetMJD(val, 12, 31);
        return true;
    }

    double GetArchiveStart()
    {
        return m_ArchiveStart;
    }

    bool SetArchiveStart(double val)
    {
        m_ArchiveStart = val;
        return true;
    }

    bool SetArchiveStart(wxString val)
    {
        m_ArchiveStart = asTime::GetTimeFromString(val);
        return true;
    }
    
    double GetArchiveEnd()
    {
        return m_ArchiveEnd;
    }
    
    bool SetArchiveEnd(double val)
    {
        m_ArchiveEnd = val;
        return true;
    }
    
    bool SetArchiveEnd(wxString val)
    {
        m_ArchiveEnd = asTime::GetTimeFromString(val);
        return true;
    }

    double GetTimeMinHours()
    {
        return m_TimeMinHours;
    }

    double GetTimeMaxHours()
    {
        return m_TimeMaxHours;
    }

    int GetTimeShiftDays()
    {
        int shift = 0;
        if (m_TimeMinHours<0) {
            shift = floor(m_TimeMinHours/24.0);
        }
        return shift;
    }

    int GetTimeSpanDays()
    {
        return ceil(m_TimeMaxHours/24.0)+abs(GetTimeShiftDays());
    }

    double GetTimeArrayTargetTimeStepHours()
    {
        return m_TimeArrayTargetTimeStepHours;
    }

    bool SetTimeArrayTargetTimeStepHours(double val);

    double GetTimeArrayAnalogsTimeStepHours()
    {
        return m_TimeArrayAnalogsTimeStepHours;
    }

    bool SetTimeArrayAnalogsTimeStepHours(double val);

    wxString GetTimeArrayTargetMode()
    {
        return m_TimeArrayTargetMode;
    }

    bool SetTimeArrayTargetMode(const wxString& val);

    wxString GetTimeArrayTargetPredictandSerieName()
    {
        return m_TimeArrayTargetPredictandSerieName;
    }

    bool SetTimeArrayTargetPredictandSerieName(const wxString& val);

    float GetTimeArrayTargetPredictandMinThreshold()
    {
        return m_TimeArrayTargetPredictandMinThreshold;
    }

    bool SetTimeArrayTargetPredictandMinThreshold(float val);

    float GetTimeArrayTargetPredictandMaxThreshold()
    {
        return m_TimeArrayTargetPredictandMaxThreshold;
    }

    bool SetTimeArrayTargetPredictandMaxThreshold(float val);

    wxString GetTimeArrayAnalogsMode()
    {
        return m_TimeArrayAnalogsMode;
    }

    bool SetTimeArrayAnalogsMode(const wxString& val);

    int GetTimeArrayAnalogsExcludeDays()
    {
        return m_TimeArrayAnalogsExcludeDays;
    }

    bool SetTimeArrayAnalogsExcludeDays(int val);

    int GetTimeArrayAnalogsIntervalDays()
    {
        return m_TimeArrayAnalogsIntervalDays;
    }

    bool SetTimeArrayAnalogsIntervalDays(int val);

    VectorInt GetPredictandStationIds()
    {
        return m_PredictandStationIds;
    }

    VVectorInt GetPredictandStationIdsVector()
    {
        VVectorInt vec;
        vec.push_back(m_PredictandStationIds);
        return vec;
    }

    bool SetPredictandStationIds(VectorInt val);

    wxString GePredictandtDatasetId()
    {
        return m_PredictandDatasetId;
    }

    bool SetPredictandDatasetId(const wxString &val);

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

    double GetPredictandTimeHours()
    {
        return m_PredictandTimeHours;
    }

    bool SetPredictandTimeHours(double val);

    int GetAnalogsNumber(int i_step)
    {
        return m_Steps[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumber(int i_step, int val);

    bool NeedsPreloading(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Preload;
    }

    void SetPreload(int i_step, int i_predictor, bool val)
    {
        m_Steps[i_step].Predictors[i_predictor].Preload = val;
    }

    VectorDouble GetPreloadTimeHours(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadTimeHours;
    }

    bool SetPreloadTimeHours(int i_step, int i_predictor, VectorDouble val);

    bool SetPreloadTimeHours(int i_step, int i_predictor, double val);

    VectorFloat GetPreloadLevels(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadLevels;
    }

    bool SetPreloadLevels(int i_step, int i_predictor, VectorFloat val);

    bool SetPreloadLevels(int i_step, int i_predictor, float val);

    double GetPreloadXmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadXmin;
    }

    bool SetPreloadXmin(int i_step, int i_predictor, double val);

    int GetPreloadXptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadXptsnb;
    }

    bool SetPreloadXptsnb(int i_step, int i_predictor, int val);

    double GetPreloadYmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadYmin;
    }

    bool SetPreloadYmin(int i_step, int i_predictor, double val);

    int GetPreloadYptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].PreloadYptsnb;
    }

    bool SetPreloadYptsnb(int i_step, int i_predictor, int val);

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

    bool SetPreprocessMethod(int i_step, int i_predictor, const wxString& val);

    wxString GetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset, const wxString& val);

    wxString GetPreprocessDataId(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessDataId(int i_step, int i_predictor, int i_dataset, const wxString& val);

    float GetPreprocessLevel(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessLevel(int i_step, int i_predictor, int i_dataset, float val);

    double GetPreprocessTimeHours(int i_step, int i_predictor, int i_dataset);

    bool SetPreprocessTimeHours(int i_step, int i_predictor, int i_dataset, double val);

    wxString GetPredictorDatasetId(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DatasetId;
    }

    bool SetPredictorDatasetId(int i_step, int i_predictor, const wxString& val);

    wxString GetPredictorDataId(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataId(int i_step, int i_predictor, wxString val);

    float GetPredictorLevel(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevel(int i_step, int i_predictor, float val);

    wxString GetPredictorGridType(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].GridType;
    }

    bool SetPredictorGridType(int i_step, int i_predictor, wxString val);

    double GetPredictorXmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXmin(int i_step, int i_predictor, double val);

    int GetPredictorXptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnb(int i_step, int i_predictor, int val);

    double GetPredictorXstep(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Xstep;
    }

    bool SetPredictorXstep(int i_step, int i_predictor, double val);

    double GetPredictorXshift(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Xshift;
    }

    bool SetPredictorXshift(int i_step, int i_predictor, double val);

    double GetPredictorYmin(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYmin(int i_step, int i_predictor, double val);

    int GetPredictorYptsnb(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnb(int i_step, int i_predictor, int val);

    double GetPredictorYstep(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Ystep;
    }

    bool SetPredictorYstep(int i_step, int i_predictor, double val);

    double GetPredictorYshift(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Yshift;
    }

    bool SetPredictorYshift(int i_step, int i_predictor, double val);

    int GetPredictorFlatAllowed(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].FlatAllowed;
    }

    bool SetPredictorFlatAllowed(int i_step, int i_predictor, int val);

    double GetPredictorTimeHours(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHours(int i_step, int i_predictor, double val);

    wxString GetPredictorCriteria(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteria(int i_step, int i_predictor, const wxString& val);

    float GetPredictorWeight(int i_step, int i_predictor)
    {
        return m_Steps[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeight(int i_step, int i_predictor, float val);

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
    wxString m_MethodId;
    wxString m_MethodIdDisplay;
    wxString m_SpecificTag;
    wxString m_SpecificTagDisplay;
    wxString m_Description;
    double m_ArchiveStart;
    double m_ArchiveEnd;
    int m_TimeArrayAnalogsIntervalDays;
    VectorInt m_PredictandStationIds;
    double m_TimeMinHours;
    double m_TimeMaxHours;

private:
    VectorParamsStep m_Steps; // Set as private to force use of setters.
    VectorInt m_PredictorsNb;
    int m_StepsNb;
    wxString m_DateProcessed;
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
    double m_PredictandTimeHours;

};

#endif // ASPARAMETERS_H
