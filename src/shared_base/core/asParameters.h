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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASPARAMETERS_H
#define ASPARAMETERS_H

#include "asIncludes.h"


class asParameters
        : public wxObject
{
public:
    typedef struct
    {
        wxString DatasetId;
        wxString DataId;
        bool Preload;
        VectorString PreloadDataIds;
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

    typedef std::vector<ParamsPredictor> VectorParamsPredictors;

    typedef struct
    {
        int AnalogsNumber;
        VectorParamsPredictors Predictors;
    } ParamsStep;

    typedef std::vector<ParamsStep> VectorParamsStep;

    asParameters();

    virtual ~asParameters();

    virtual void AddStep();

    void AddPredictor(); // To the last step

    void AddPredictor(ParamsStep &step);

    void AddPredictor(int i_step);

    virtual bool LoadFromFile(const wxString &filePath = wxEmptyString);

    bool FixAnalogsNb();

    void SortLevelsAndTime();

    virtual bool SetSpatialWindowProperties();

    virtual bool SetPreloadingProperties();

    virtual bool InputsOK() const;

    static VectorInt GetFileStationIds(wxString stationIdsString);

    wxString GetPredictandStationIdsString() const;

    virtual bool FixTimeLimits();

    bool FixWeights();

    bool FixCoordinates();

    virtual wxString Print() const;

    bool PrintAndSaveTemp(const wxString &filePath = wxEmptyString) const;

    virtual bool GetValuesFromString(wxString stringVals); // We copy the string as we'll modify it.

    bool SetPredictandStationIds(wxString val);

    VectorParamsPredictors GetVectorParamsPredictors(int i_step) const
    {
        wxASSERT(i_step < GetStepsNb());
        return m_steps[i_step].Predictors;
    }

    void SetVectorParamsPredictors(int i_step, VectorParamsPredictors ptors)
    {
        wxASSERT(i_step < GetStepsNb());
        m_steps[i_step].Predictors = ptors;
    }

    wxString GetMethodId() const
    {
        return m_methodId;
    }

    void SetMethodId(const wxString &val)
    {
        m_methodId = val;
    }

    wxString GetMethodIdDisplay() const
    {
        return m_methodIdDisplay;
    }

    void SetMethodIdDisplay(const wxString &val)
    {
        m_methodIdDisplay = val;
    }

    wxString GetSpecificTag() const
    {
        return m_specificTag;
    }

    void SetSpecificTag(const wxString &val)
    {
        m_specificTag = val;
    }

    wxString GetSpecificTagDisplay() const
    {
        return m_specificTagDisplay;
    }

    void SetSpecificTagDisplay(const wxString &val)
    {
        m_specificTagDisplay = val;
    }

    wxString GetDescription() const
    {
        return m_description;
    }

    void SetDescription(const wxString &val)
    {
        m_description = val;
    }

    wxString GetDateProcessed() const
    {
        return m_dateProcessed;
    }

    void SetDateProcessed(const wxString &val)
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

    double GetArchiveStart() const
    {
        return m_archiveStart;
    }

    bool SetArchiveStart(wxString val)
    {
        m_archiveStart = asTime::GetTimeFromString(val);
        return true;
    }

    double GetArchiveEnd() const
    {
        return m_archiveEnd;
    }

    bool SetArchiveEnd(wxString val)
    {
        m_archiveEnd = asTime::GetTimeFromString(val);
        return true;
    }

    double GetTimeMinHours() const
    {
        return m_timeMinHours;
    }

    double GetTimeMaxHours() const
    {
        return m_timeMaxHours;
    }

    int GetTimeShiftDays() const
    {
        int shift = 0;
        if (m_timeMinHours < 0) {
            shift = (int) floor(m_timeMinHours / 24.0);
        }
        return shift;
    }

    int GetTimeSpanDays() const
    {
        return ceil(m_timeMaxHours / 24.0) + std::abs(GetTimeShiftDays());
    }

    double GetTimeArrayTargetTimeStepHours() const
    {
        return m_timeArrayTargetTimeStepHours;
    }

    bool SetTimeArrayTargetTimeStepHours(double val);

    double GetTimeArrayAnalogsTimeStepHours() const
    {
        return m_timeArrayAnalogsTimeStepHours;
    }

    bool SetTimeArrayAnalogsTimeStepHours(double val);

    wxString GetTimeArrayTargetMode() const
    {
        return m_timeArrayTargetMode;
    }

    bool SetTimeArrayTargetMode(const wxString &val);

    wxString GetTimeArrayTargetPredictandSerieName() const
    {
        return m_timeArrayTargetPredictandSerieName;
    }

    bool SetTimeArrayTargetPredictandSerieName(const wxString &val);

    float GetTimeArrayTargetPredictandMinThreshold() const
    {
        return m_timeArrayTargetPredictandMinThreshold;
    }

    bool SetTimeArrayTargetPredictandMinThreshold(float val);

    float GetTimeArrayTargetPredictandMaxThreshold() const
    {
        return m_timeArrayTargetPredictandMaxThreshold;
    }

    bool SetTimeArrayTargetPredictandMaxThreshold(float val);

    wxString GetTimeArrayAnalogsMode() const
    {
        return m_timeArrayAnalogsMode;
    }

    bool SetTimeArrayAnalogsMode(const wxString &val);

    int GetTimeArrayAnalogsExcludeDays() const
    {
        return m_timeArrayAnalogsExcludeDays;
    }

    bool SetTimeArrayAnalogsExcludeDays(int val);

    int GetTimeArrayAnalogsIntervalDays() const
    {
        return m_timeArrayAnalogsIntervalDays;
    }

    bool SetTimeArrayAnalogsIntervalDays(int val);

    VectorInt GetPredictandStationIds() const
    {
        return m_predictandStationIds;
    }

    virtual VVectorInt GetPredictandStationIdsVector() const
    {
        VVectorInt vec;
        vec.push_back(m_predictandStationIds);
        return vec;
    }

    bool SetPredictandStationIds(VectorInt val);

    wxString GePredictandtDatasetId() const
    {
        return m_predictandDatasetId;
    }

    bool SetPredictandDatasetId(const wxString &val);

    DataParameter GetPredictandParameter() const
    {
        return m_predictandParameter;
    }

    void SetPredictandParameter(DataParameter val)
    {
        m_predictandParameter = val;
    }

    DataTemporalResolution GetPredictandTemporalResolution() const
    {
        return m_predictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(DataTemporalResolution val)
    {
        m_predictandTemporalResolution = val;
    }

    DataSpatialAggregation GetPredictandSpatialAggregation() const
    {
        return m_predictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(DataSpatialAggregation val)
    {
        m_predictandSpatialAggregation = val;
    }

    double GetPredictandTimeHours() const
    {
        return m_predictandTimeHours;
    }

    bool SetPredictandTimeHours(double val);

    int GetAnalogsNumber(int i_step) const
    {
        return m_steps[i_step].AnalogsNumber;
    }

    bool SetAnalogsNumber(int i_step, int val);

    bool NeedsPreloading(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Preload;
    }

    void SetPreload(int i_step, int i_predictor, bool val)
    {
        m_steps[i_step].Predictors[i_predictor].Preload = val;
    }

    VectorString GetPreloadDataIds(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadDataIds;
    }

    bool SetPreloadDataIds(int i_step, int i_predictor, VectorString val);

    bool SetPreloadDataIds(int i_step, int i_predictor, wxString val);

    VectorDouble GetPreloadTimeHours(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadTimeHours;
    }

    bool SetPreloadTimeHours(int i_step, int i_predictor, VectorDouble val);

    bool SetPreloadTimeHours(int i_step, int i_predictor, double val);

    VectorFloat GetPreloadLevels(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadLevels;
    }

    bool SetPreloadLevels(int i_step, int i_predictor, VectorFloat val);

    bool SetPreloadLevels(int i_step, int i_predictor, float val);

    double GetPreloadXmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadXmin;
    }

    bool SetPreloadXmin(int i_step, int i_predictor, double val);

    int GetPreloadXptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadXptsnb;
    }

    bool SetPreloadXptsnb(int i_step, int i_predictor, int val);

    double GetPreloadYmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadYmin;
    }

    bool SetPreloadYmin(int i_step, int i_predictor, double val);

    int GetPreloadYptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreloadYptsnb;
    }

    bool SetPreloadYptsnb(int i_step, int i_predictor, int val);

    bool NeedsPreprocessing(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Preprocess;
    }

    void SetPreprocess(int i_step, int i_predictor, bool val)
    {
        m_steps[i_step].Predictors[i_predictor].Preprocess = val;
    }

    virtual int GetPreprocessSize(int i_step, int i_predictor) const
    {
        return (int) m_steps[i_step].Predictors[i_predictor].PreprocessDataIds.size();
    }

    wxString GetPreprocessMethod(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].PreprocessMethod;
    }

    bool SetPreprocessMethod(int i_step, int i_predictor, const wxString &val);

    wxString GetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessDatasetId(int i_step, int i_predictor, int i_dataset, const wxString &val);

    wxString GetPreprocessDataId(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessDataId(int i_step, int i_predictor, int i_dataset, const wxString &val);

    float GetPreprocessLevel(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessLevel(int i_step, int i_predictor, int i_dataset, float val);

    double GetPreprocessTimeHours(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessTimeHours(int i_step, int i_predictor, int i_dataset, double val);

    wxString GetPredictorDatasetId(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].DatasetId;
    }

    bool SetPredictorDatasetId(int i_step, int i_predictor, const wxString &val);

    wxString GetPredictorDataId(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].DataId;
    }

    bool SetPredictorDataId(int i_step, int i_predictor, wxString val);

    float GetPredictorLevel(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Level;
    }

    bool SetPredictorLevel(int i_step, int i_predictor, float val);

    wxString GetPredictorGridType(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].GridType;
    }

    bool SetPredictorGridType(int i_step, int i_predictor, wxString val);

    double GetPredictorXmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Xmin;
    }

    bool SetPredictorXmin(int i_step, int i_predictor, double val);

    int GetPredictorXptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Xptsnb;
    }

    bool SetPredictorXptsnb(int i_step, int i_predictor, int val);

    double GetPredictorXstep(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Xstep;
    }

    bool SetPredictorXstep(int i_step, int i_predictor, double val);

    double GetPredictorXshift(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Xshift;
    }

    bool SetPredictorXshift(int i_step, int i_predictor, double val);

    double GetPredictorYmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Ymin;
    }

    bool SetPredictorYmin(int i_step, int i_predictor, double val);

    int GetPredictorYptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Yptsnb;
    }

    bool SetPredictorYptsnb(int i_step, int i_predictor, int val);

    double GetPredictorYstep(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Ystep;
    }

    bool SetPredictorYstep(int i_step, int i_predictor, double val);

    double GetPredictorYshift(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Yshift;
    }

    bool SetPredictorYshift(int i_step, int i_predictor, double val);

    int GetPredictorFlatAllowed(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].FlatAllowed;
    }

    bool SetPredictorFlatAllowed(int i_step, int i_predictor, int val);

    double GetPredictorTimeHours(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].TimeHours;
    }

    bool SetPredictorTimeHours(int i_step, int i_predictor, double val);

    wxString GetPredictorCriteria(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Criteria;
    }

    bool SetPredictorCriteria(int i_step, int i_predictor, const wxString &val);

    float GetPredictorWeight(int i_step, int i_predictor) const
    {
        return m_steps[i_step].Predictors[i_predictor].Weight;
    }

    bool SetPredictorWeight(int i_step, int i_predictor, float val);

    int GetStepsNb() const
    {
        return (int) m_steps.size();
    }

    int GetPredictorsNb(int i_step) const
    {
        wxASSERT((unsigned) i_step < m_steps.size());
        return (int) m_steps[i_step].Predictors.size();
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
