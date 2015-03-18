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
        return m_methodId;
    }

    void SetMethodId(const wxString& val)
    {
        m_methodId = val;
    }
    
    wxString GetMethodIdDisplay()
    {
        return m_methodIdDisplay;
    }

    void SetMethodIdDisplay(const wxString& val)
    {
        m_methodIdDisplay = val;
    }
    
    wxString GetSpecificTag()
    {
        return m_specificTag;
    }

    void SetSpecificTag(const wxString& val)
    {
        m_specificTag = val;
    }
    
    wxString GetSpecificTagDisplay()
    {
        return m_specificTagDisplay;
    }

    void SetSpecificTagDisplay(const wxString& val)
    {
        m_specificTagDisplay = val;
    }
    
    wxString GetDescription()
    {
        return m_description;
    }

    void SetDescription(const wxString& val)
    {
        m_description = val;
    }

    wxString GetDateProcessed()
    {
        return m_dateProcessed;
    }

    void SetDateProcessed(const wxString& val)
    {
        m_dateProcessed = val;
    }

    bool SetArchiveYearStart(int val)
    {
        m_archiveStart = asTime::GetMJD(val, 1, 1);
        return true;
    }

    bool SetArchiveYearEnd(int val)
    {
        m_archiveEnd = asTime::GetMJD(val, 12, 31);
        return true;
    }

    double GetArchiveStart()
    {
        return m_archiveStart;
    }

    bool SetArchiveStart(double val)
    {
        m_archiveStart = val;
        return true;
    }

    bool SetArchiveStart(wxString val)
    {
        m_archiveStart = asTime::GetTimeFromString(val);
        return true;
    }
    
    double GetArchiveEnd()
    {
        return m_archiveEnd;
    }
    
    bool SetArchiveEnd(double val)
    {
        m_archiveEnd = val;
        return true;
    }
    
    bool SetArchiveEnd(wxString val)
    {
        m_archiveEnd = asTime::GetTimeFromString(val);
        return true;
    }

    double GetTimeMinHours()
    {
        return m_timeMinHours;
    }

    double GetTimeMaxHours()
    {
        return m_timeMaxHours;
    }

    int GetTimeShiftDays()
    {
        int shift = 0;
        if (m_timeMinHours<0) {
            shift = floor(m_timeMinHours/24.0);
        }
        return shift;
    }

    int GetTimeSpanDays()
    {
        return ceil(m_timeMaxHours/24.0)+std::abs(GetTimeShiftDays());
    }

    double GetTimeArrayTargetTimeStepHours()
    {
        return m_timeArrayTargetTimeStepHours;
    }

    bool SetTimeArrayTargetTimeStepHours(double val);

    double GetTimeArrayAnalogsTimeStepHours()
    {
        return m_timeArrayAnalogsTimeStepHours;
    }

    bool SetTimeArrayAnalogsTimeStepHours(double val);

    wxString GetTimeArrayTargetMode()
    {
        return m_timeArrayTargetMode;
    }

    bool SetTimeArrayTargetMode(const wxString& val);

    wxString GetTimeArrayTargetPredictandSerieName()
    {
        return m_timeArrayTargetPredictandSerieName;
    }

    bool SetTimeArrayTargetPredictandSerieName(const wxString& val);

    float GetTimeArrayTargetPredictandMinThreshold()
    {
        return m_timeArrayTargetPredictandMinThreshold;
    }

    bool SetTimeArrayTargetPredictandMinThreshold(float val);

    float GetTimeArrayTargetPredictandMaxThreshold()
    {
        return m_timeArrayTargetPredictandMaxThreshold;
    }

    bool SetTimeArrayTargetPredictandMaxThreshold(float val);

    wxString GetTimeArrayAnalogsMode()
    {
        return m_timeArrayAnalogsMode;
    }

    bool SetTimeArrayAnalogsMode(const wxString& val);

    int GetTimeArrayAnalogsExcludeDays()
    {
        return m_timeArrayAnalogsExcludeDays;
    }

    bool SetTimeArrayAnalogsExcludeDays(int val);

    int GetTimeArrayAnalogsIntervalDays()
    {
        return m_timeArrayAnalogsIntervalDays;
    }

    bool SetTimeArrayAnalogsIntervalDays(int val);

    VectorInt GetPredictandStationIds()
    {
        return m_predictandStationIds;
    }

    VVectorInt GetPredictandStationIdsVector()
    {
        VVectorInt vec;
        vec.push_back(m_predictandStationIds);
        return vec;
    }

    bool SetPredictandStationIds(VectorInt val);

    wxString GePredictandtDatasetId()
    {
        return m_predictandDatasetId;
    }

    bool SetPredictandDatasetId(const wxString &val);

    DataParameter GetPredictandParameter()
    {
        return m_predictandParameter;
    }

    void SetPredictandParameter(DataParameter val)
    {
        m_predictandParameter = val;
    }

    DataTemporalResolution GetPredictandTemporalResolution()
    {
        return m_predictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(DataTemporalResolution val)
    {
        m_predictandTemporalResolution = val;
    }

    DataSpatialAggregation GetPredictandSpatialAggregation()
    {
        return m_predictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(DataSpatialAggregation val)
    {
        m_predictandSpatialAggregation = val;
    }

    double GetPredictandTimeHours()
    {
        return m_predictandTimeHours;
    }

    bool SetPredictandTimeHours(double val);

    int GetAnalogsNumber(int i_step)
    {
        return m_steps[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumber(int i_step, int val);

    bool NeedsPreloading(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Preload;
    }

    void SetPreload(int i_step, int i_predictor, bool val)
    {
        m_steps[i_step].Predictors[i_predictor].Preload = val;
    }

    VectorDouble GetPreloadTimeHours(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadTimeHours;
    }

    bool SetPreloadTimeHours(int i_step, int i_predictor, VectorDouble val);

    bool SetPreloadTimeHours(int i_step, int i_predictor, double val);

    VectorFloat GetPreloadLevels(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadLevels;
    }

    bool SetPreloadLevels(int i_step, int i_predictor, VectorFloat val);

    bool SetPreloadLevels(int i_step, int i_predictor, float val);

    double GetPreloadXmin(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadXmin;
    }

    bool SetPreloadXmin(int i_step, int i_predictor, double val);

    int GetPreloadXptsnb(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadXptsnb;
    }

    bool SetPreloadXptsnb(int i_step, int i_predictor, int val);

    double GetPreloadYmin(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadYmin;
    }

    bool SetPreloadYmin(int i_step, int i_predictor, double val);

    int GetPreloadYptsnb(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadYptsnb;
    }

    bool SetPreloadYptsnb(int i_step, int i_predictor, int val);

    bool NeedsPreprocessing(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Preprocess;
    }

    void SetPreprocess(int i_step, int i_predictor, bool val)
    {
        m_steps[i_step].Predictors[i_predictor].Preprocess = val;
    }

    int GetPreprocessSize(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreprocessDataIds.size();
    }

    wxString GetPreprocessMethod(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].PreprocessMethod;
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
        return m_steps[i_step].Predictors[i_predictor].DatasetId;
    }

    bool SetPredictorDatasetId(int i_step, int i_predictor, const wxString& val);

    wxString GetPredictorDataId(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataId(int i_step, int i_predictor, wxString val);

    float GetPredictorLevel(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevel(int i_step, int i_predictor, float val);

    wxString GetPredictorGridType(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].GridType;
    }

    bool SetPredictorGridType(int i_step, int i_predictor, wxString val);

    double GetPredictorXmin(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXmin(int i_step, int i_predictor, double val);

    int GetPredictorXptsnb(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnb(int i_step, int i_predictor, int val);

    double GetPredictorXstep(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Xstep;
    }

    bool SetPredictorXstep(int i_step, int i_predictor, double val);

    double GetPredictorXshift(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Xshift;
    }

    bool SetPredictorXshift(int i_step, int i_predictor, double val);

    double GetPredictorYmin(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYmin(int i_step, int i_predictor, double val);

    int GetPredictorYptsnb(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnb(int i_step, int i_predictor, int val);

    double GetPredictorYstep(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Ystep;
    }

    bool SetPredictorYstep(int i_step, int i_predictor, double val);

    double GetPredictorYshift(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Yshift;
    }

    bool SetPredictorYshift(int i_step, int i_predictor, double val);

    int GetPredictorFlatAllowed(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].FlatAllowed;
    }

    bool SetPredictorFlatAllowed(int i_step, int i_predictor, int val);

    double GetPredictorTimeHours(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHours(int i_step, int i_predictor, double val);

    wxString GetPredictorCriteria(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteria(int i_step, int i_predictor, const wxString& val);

    float GetPredictorWeight(int i_step, int i_predictor)
    {
        return m_steps[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeight(int i_step, int i_predictor, float val);

    int GetStepsNb()
    {
        return m_stepsNb;
    }

    VectorInt GetVectorPredictorsNb()
    {
        return m_predictorsNb;
    }

    int GetPredictorsNb(int i_step)
    {
        wxASSERT_MSG((unsigned)i_step<m_predictorsNb.size(), wxString::Format(_("Trying to access element %d of m_predictorsNb of size %d."), i_step+1, (int)m_predictorsNb.size()));
        return m_predictorsNb[i_step];
    }


protected:
    wxString m_methodId;
    wxString m_methodIdDisplay;
    wxString m_specificTag;
    wxString m_specificTagDisplay;
    wxString m_description;
    double m_archiveStart;
    double m_archiveEnd;
    int m_timeArrayAnalogsIntervalDays;
    VectorInt m_predictandStationIds;
    double m_timeMinHours;
    double m_timeMaxHours;

private:
    VectorParamsStep m_steps; // Set as private to force use of setters.
    VectorInt m_predictorsNb;
    int m_stepsNb;
    wxString m_dateProcessed;
    wxString m_timeArrayTargetMode;
    double m_timeArrayTargetTimeStepHours;
    wxString m_timeArrayTargetPredictandSerieName;
    float m_timeArrayTargetPredictandMinThreshold;
    float m_timeArrayTargetPredictandMaxThreshold;
    wxString m_timeArrayAnalogsMode;
    double m_timeArrayAnalogsTimeStepHours;
    int m_timeArrayAnalogsExcludeDays;
    DataParameter m_predictandParameter;
    DataTemporalResolution m_predictandTemporalResolution;
    DataSpatialAggregation m_predictandSpatialAggregation;
    wxString m_predictandDatasetId;
    double m_predictandTimeHours;

};

#endif // ASPARAMETERS_H
