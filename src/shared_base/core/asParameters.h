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
#include "asPredictand.h"


class asFileParameters;


class asParameters
        : public wxObject
{
public:
    typedef struct
    {
        wxString datasetId;
        wxString dataId;
        bool preload;
        vwxs preloadDataIds;
        vd preloadTimeHours;
        vf preloadLevels;
        double preloadXmin;
        int preloadXptsnb;
        double preloadYmin;
        int preloadYptsnb;
        bool preprocess;
        wxString preprocessMethod;
        vwxs preprocessDatasetIds;
        vwxs preprocessDataIds;
        vf preprocessLevels;
        vd preprocessTimeHours;
        vi preprocessMembersNb;
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

    void AddPredictor(int iStep);

    virtual bool LoadFromFile(const wxString &filePath = wxEmptyString);

    bool FixAnalogsNb();

    void SortLevelsAndTime();

    virtual bool SetSpatialWindowProperties();

    virtual bool SetPreloadingProperties();

    virtual bool InputsOK() const;

    bool PreprocessingPropertiesOk() const;

    static vi GetFileStationIds(wxString stationIdsString);

    wxString GetPredictandStationIdsString() const;

    virtual bool FixTimeLimits();

    bool FixWeights();

    bool FixCoordinates();

    virtual wxString Print() const;

    bool PrintAndSaveTemp(const wxString &filePath = wxEmptyString) const;

    virtual bool GetValuesFromString(wxString stringVals); // We copy the string as we'll modify it.

    bool SetPredictandStationIds(wxString val);

    VectorParamsPredictors GetVectorParamsPredictors(int iStep) const
    {
        wxASSERT(iStep < GetStepsNb());
        return m_steps[iStep].predictors;
    }

    void SetVectorParamsPredictors(int iStep, VectorParamsPredictors ptors)
    {
        wxASSERT(iStep < GetStepsNb());
        m_steps[iStep].predictors = ptors;
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

    vi GetPredictandStationIds() const
    {
        return m_predictandStationIds;
    }

    virtual vvi GetPredictandStationIdsVector() const
    {
        vvi vec;
        vec.push_back(m_predictandStationIds);
        return vec;
    }

    bool SetPredictandStationIds(vi val);

    wxString GePredictandtDatasetId() const
    {
        return m_predictandDatasetId;
    }

    bool SetPredictandDatasetId(const wxString &val);

    asPredictand::Parameter GetPredictandParameter() const
    {
        return m_predictandParameter;
    }

    void SetPredictandParameter(asPredictand::Parameter val)
    {
        m_predictandParameter = val;
    }

    asPredictand::TemporalResolution GetPredictandTemporalResolution() const
    {
        return m_predictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(asPredictand::TemporalResolution val)
    {
        m_predictandTemporalResolution = val;
    }

    asPredictand::SpatialAggregation GetPredictandSpatialAggregation() const
    {
        return m_predictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(asPredictand::SpatialAggregation val)
    {
        m_predictandSpatialAggregation = val;
    }

    double GetPredictandTimeHours() const
    {
        return m_predictandTimeHours;
    }

    bool SetPredictandTimeHours(double val);

    int GetAnalogsNumber(int iStep) const
    {
        return m_steps[iStep].analogsNumber;
    }

    bool SetAnalogsNumber(int iStep, int val);

    bool NeedsPreloading(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preload;
    }

    void SetPreload(int iStep, int iPtor, bool val)
    {
        m_steps[iStep].predictors[iPtor].preload = val;
    }

    vwxs GetPreloadDataIds(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadDataIds;
    }

    bool SetPreloadDataIds(int iStep, int iPtor, vwxs val);

    bool SetPreloadDataIds(int iStep, int iPtor, wxString val);

    vd GetPreloadTimeHours(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadTimeHours;
    }

    bool SetPreloadTimeHours(int iStep, int iPtor, vd val);

    bool SetPreloadTimeHours(int iStep, int iPtor, double val);

    vf GetPreloadLevels(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadLevels;
    }

    bool SetPreloadLevels(int iStep, int iPtor, vf val);

    bool SetPreloadLevels(int iStep, int iPtor, float val);

    double GetPreloadXmin(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadXmin;
    }

    bool SetPreloadXmin(int iStep, int iPtor, double val);

    int GetPreloadXptsnb(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadXptsnb;
    }

    bool SetPreloadXptsnb(int iStep, int iPtor, int val);

    double GetPreloadYmin(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadYmin;
    }

    bool SetPreloadYmin(int iStep, int iPtor, double val);

    int GetPreloadYptsnb(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preloadYptsnb;
    }

    bool SetPreloadYptsnb(int iStep, int iPtor, int val);

    bool NeedsPreprocessing(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preprocess;
    }

    void SetPreprocess(int iStep, int iPtor, bool val)
    {
        m_steps[iStep].predictors[iPtor].preprocess = val;
    }

    virtual int GetPreprocessSize(int iStep, int iPtor) const
    {
        return (int) m_steps[iStep].predictors[iPtor].preprocessDataIds.size();
    }

    wxString GetPreprocessMethod(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].preprocessMethod;
    }

    bool SetPreprocessMethod(int iStep, int iPtor, const wxString &val);

    wxString GetPreprocessDatasetId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessDatasetId(int iStep, int iPtor, int iPre, const wxString &val);

    wxString GetPreprocessDataId(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessDataId(int iStep, int iPtor, int iPre, const wxString &val);

    float GetPreprocessLevel(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessLevel(int iStep, int iPtor, int iPre, float val);

    double GetPreprocessTimeHours(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessTimeHours(int iStep, int iPtor, int iPre, double val);

    int GetPreprocessMembersNb(int iStep, int iPtor, int iPre) const;

    bool SetPreprocessMembersNb(int iStep, int iPtor, int iPre, int val);

    wxString GetPredictorDatasetId(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].datasetId;
    }

    bool SetPredictorDatasetId(int iStep, int iPtor, const wxString &val);

    wxString GetPredictorDataId(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].dataId;
    }

    bool SetPredictorDataId(int iStep, int iPtor, wxString val);

    float GetPredictorLevel(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].level;
    }

    bool SetPredictorLevel(int iStep, int iPtor, float val);

    wxString GetPredictorGridType(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].gridType;
    }

    bool SetPredictorGridType(int iStep, int iPtor, wxString val);

    double GetPredictorXmin(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].xMin;
    }

    bool SetPredictorXmin(int iStep, int iPtor, double val);

    int GetPredictorXptsnb(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].xPtsNb;
    }

    bool SetPredictorXptsnb(int iStep, int iPtor, int val);

    double GetPredictorXstep(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].xStep;
    }

    bool SetPredictorXstep(int iStep, int iPtor, double val);

    double GetPredictorXshift(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].xShift;
    }

    bool SetPredictorXshift(int iStep, int iPtor, double val);

    double GetPredictorYmin(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].yMin;
    }

    bool SetPredictorYmin(int iStep, int iPtor, double val);

    int GetPredictorYptsnb(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].yPtsNb;
    }

    bool SetPredictorYptsnb(int iStep, int iPtor, int val);

    double GetPredictorYstep(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].yStep;
    }

    bool SetPredictorYstep(int iStep, int iPtor, double val);

    double GetPredictorYshift(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].yShift;
    }

    bool SetPredictorYshift(int iStep, int iPtor, double val);

    int GetPredictorFlatAllowed(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].flatAllowed;
    }

    bool SetPredictorFlatAllowed(int iStep, int iPtor, int val);

    double GetPredictorTimeHours(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].timeHours;
    }

    bool SetPredictorTimeHours(int iStep, int iPtor, double val);

    int GetPredictorMembersNb(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].membersNb;
    }

    bool SetPredictorMembersNb(int iStep, int iPtor, int val);

    wxString GetPredictorCriteria(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].criteria;
    }

    bool SetPredictorCriteria(int iStep, int iPtor, const wxString &val);

    float GetPredictorWeight(int iStep, int iPtor) const
    {
        return m_steps[iStep].predictors[iPtor].weight;
    }

    bool SetPredictorWeight(int iStep, int iPtor, float val);

    int GetStepsNb() const
    {
        return (int) m_steps.size();
    }

    int GetPredictorsNb(int iStep) const
    {
        wxASSERT((unsigned) iStep < m_steps.size());
        return (int) m_steps[iStep].predictors.size();
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
    vi m_predictandStationIds;
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
    asPredictand::Parameter m_predictandParameter;
    asPredictand::TemporalResolution m_predictandTemporalResolution;
    asPredictand::SpatialAggregation m_predictandSpatialAggregation;
    wxString m_predictandDatasetId;
    double m_predictandTimeHours;

    bool ParseDescription(asFileParameters &fileParams, const wxXmlNode *nodeProcess);

    bool ParseTimeProperties(asFileParameters &fileParams, const wxXmlNode *nodeProcess);

    bool ParseAnalogDatesParams(asFileParameters &fileParams, int iStep, const wxXmlNode *nodeProcess);

    bool ParsePredictors(asFileParameters &fileParams, int iStep, int iPtor, const wxXmlNode *nodeParamBlock);

    bool ParsePreprocessedPredictors(asFileParameters &fileParams, int iStep, int iPtor,
                                     const wxXmlNode *nodeParam);

    bool ParseAnalogValuesParams(asFileParameters &fileParams, const wxXmlNode *nodeProcess);

};

#endif // ASPARAMETERS_H
