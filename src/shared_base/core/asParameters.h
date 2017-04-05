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

#include <wx/xml/xml.h>
#include "asIncludes.h"
#include "asDataPredictand.h"
#include "asFileParametersStandard.h"


class asParameters
        : public wxObject
{
public:
    typedef struct
    {
        wxString datasetId;
        wxString dataId;
        bool preload;
        VectorString preloadDataIds;
        VectorDouble preloadTimeHours;
        VectorFloat preloadLevels;
        double preloadXmin;
        int preloadXptsnb;
        double preloadYmin;
        int preloadYptsnb;
        bool preprocess;
        wxString preprocessMethod;
        VectorString preprocessDatasetIds;
        VectorString preprocessDataIds;
        VectorFloat preprocessLevels;
        VectorDouble preprocessTimeHours;
        VectorInt preprocessMembersNb;
        float level;
        wxString gridType;
        double xMin;
        int xPtsNb;
        double xStep;
        double xShift;
        double yMin;
        int yPtsNb;
        double yStep;
        double yShift;
        int flatAllowed;
        double timeHours;
        int membersNb;
        wxString criteria;
        float weight;
    } ParamsPredictor;

    typedef std::vector<ParamsPredictor> VectorParamsPredictors;

    typedef struct
    {
        int analogsNumber;
        VectorParamsPredictors predictors;
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
        return m_steps[i_step].predictors;
    }

    void SetVectorParamsPredictors(int i_step, VectorParamsPredictors ptors)
    {
        wxASSERT(i_step < GetStepsNb());
        m_steps[i_step].predictors = ptors;
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

    asDataPredictand::Parameter GetPredictandParameter() const
    {
        return m_predictandParameter;
    }

    void SetPredictandParameter(asDataPredictand::Parameter val)
    {
        m_predictandParameter = val;
    }

    asDataPredictand::TemporalResolution GetPredictandTemporalResolution() const
    {
        return m_predictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(asDataPredictand::TemporalResolution val)
    {
        m_predictandTemporalResolution = val;
    }

    asDataPredictand::SpatialAggregation GetPredictandSpatialAggregation() const
    {
        return m_predictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(asDataPredictand::SpatialAggregation val)
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
        return m_steps[i_step].analogsNumber;
    }

    bool SetAnalogsNumber(int i_step, int val);

    bool NeedsPreloading(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preload;
    }

    void SetPreload(int i_step, int i_predictor, bool val)
    {
        m_steps[i_step].predictors[i_predictor].preload = val;
    }

    VectorString GetPreloadDataIds(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadDataIds;
    }

    bool SetPreloadDataIds(int i_step, int i_predictor, VectorString val);

    bool SetPreloadDataIds(int i_step, int i_predictor, wxString val);

    VectorDouble GetPreloadTimeHours(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadTimeHours;
    }

    bool SetPreloadTimeHours(int i_step, int i_predictor, VectorDouble val);

    bool SetPreloadTimeHours(int i_step, int i_predictor, double val);

    VectorFloat GetPreloadLevels(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadLevels;
    }

    bool SetPreloadLevels(int i_step, int i_predictor, VectorFloat val);

    bool SetPreloadLevels(int i_step, int i_predictor, float val);

    double GetPreloadXmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadXmin;
    }

    bool SetPreloadXmin(int i_step, int i_predictor, double val);

    int GetPreloadXptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadXptsnb;
    }

    bool SetPreloadXptsnb(int i_step, int i_predictor, int val);

    double GetPreloadYmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadYmin;
    }

    bool SetPreloadYmin(int i_step, int i_predictor, double val);

    int GetPreloadYptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preloadYptsnb;
    }

    bool SetPreloadYptsnb(int i_step, int i_predictor, int val);

    bool NeedsPreprocessing(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preprocess;
    }

    void SetPreprocess(int i_step, int i_predictor, bool val)
    {
        m_steps[i_step].predictors[i_predictor].preprocess = val;
    }

    virtual int GetPreprocessSize(int i_step, int i_predictor) const
    {
        return (int) m_steps[i_step].predictors[i_predictor].preprocessDataIds.size();
    }

    wxString GetPreprocessMethod(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].preprocessMethod;
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

    int GetPreprocessMembersNb(int i_step, int i_predictor, int i_dataset) const;

    bool SetPreprocessMembersNb(int i_step, int i_predictor, int i_dataset, int val);

    wxString GetPredictorDatasetId(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].datasetId;
    }

    bool SetPredictorDatasetId(int i_step, int i_predictor, const wxString &val);

    wxString GetPredictorDataId(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].dataId;
    }

    bool SetPredictorDataId(int i_step, int i_predictor, wxString val);

    float GetPredictorLevel(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].level;
    }

    bool SetPredictorLevel(int i_step, int i_predictor, float val);

    wxString GetPredictorGridType(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].gridType;
    }

    bool SetPredictorGridType(int i_step, int i_predictor, wxString val);

    double GetPredictorXmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].xMin;
    }

    bool SetPredictorXmin(int i_step, int i_predictor, double val);

    int GetPredictorXptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].xPtsNb;
    }

    bool SetPredictorXptsnb(int i_step, int i_predictor, int val);

    double GetPredictorXstep(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].xStep;
    }

    bool SetPredictorXstep(int i_step, int i_predictor, double val);

    double GetPredictorXshift(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].xShift;
    }

    bool SetPredictorXshift(int i_step, int i_predictor, double val);

    double GetPredictorYmin(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].yMin;
    }

    bool SetPredictorYmin(int i_step, int i_predictor, double val);

    int GetPredictorYptsnb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].yPtsNb;
    }

    bool SetPredictorYptsnb(int i_step, int i_predictor, int val);

    double GetPredictorYstep(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].yStep;
    }

    bool SetPredictorYstep(int i_step, int i_predictor, double val);

    double GetPredictorYshift(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].yShift;
    }

    bool SetPredictorYshift(int i_step, int i_predictor, double val);

    int GetPredictorFlatAllowed(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].flatAllowed;
    }

    bool SetPredictorFlatAllowed(int i_step, int i_predictor, int val);

    double GetPredictorTimeHours(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].timeHours;
    }

    bool SetPredictorTimeHours(int i_step, int i_predictor, double val);

    int GetPredictorMembersNb(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].membersNb;
    }

    bool SetPredictorMembersNb(int i_step, int i_predictor, int val);

    wxString GetPredictorCriteria(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].criteria;
    }

    bool SetPredictorCriteria(int i_step, int i_predictor, const wxString &val);

    float GetPredictorWeight(int i_step, int i_predictor) const
    {
        return m_steps[i_step].predictors[i_predictor].weight;
    }

    bool SetPredictorWeight(int i_step, int i_predictor, float val);

    int GetStepsNb() const
    {
        return (int) m_steps.size();
    }

    int GetPredictorsNb(int i_step) const
    {
        wxASSERT((unsigned) i_step < m_steps.size());
        return (int) m_steps[i_step].predictors.size();
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
    asDataPredictand::Parameter m_predictandParameter;
    asDataPredictand::TemporalResolution m_predictandTemporalResolution;
    asDataPredictand::SpatialAggregation m_predictandSpatialAggregation;
    wxString m_predictandDatasetId;
    double m_predictandTimeHours;

    bool ParseDescription(asFileParametersStandard &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParametersStandard &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParametersStandard &fileParams, int i_step, const wxXmlNode *nodeProcess);

    bool ParsePredictors(asFileParametersStandard &fileParams, int i_step, int i_ptor, const wxXmlNode *nodeParamBlock);

    bool ParsePreprocessedPredictors(asFileParametersStandard &fileParams, int i_step, int i_ptor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParametersStandard &fileParams, const wxXmlNode *nodeProcess);

};

#endif // ASPARAMETERS_H
