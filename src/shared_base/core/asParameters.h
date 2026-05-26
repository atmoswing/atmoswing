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

#ifndef AS_PARAMETERS_H
#define AS_PARAMETERS_H

#include <wx/xml/xml.h>

#include <utility>

#include "asHeadersBase.h"
#include "asPredictand.h"

class asFileParameters;

class asParameters : public wxObject {
  public:
    struct ParamsPredictor {
        bool preload = false;
        bool standardize = false;
        double standardizeMean = NAN;
        double standardizeSd = NAN;
        bool preprocess = false;
        std::string datasetId;
        std::string dataId;
        vstds preloadDataIds;
        vd preloadHours;
        vf preloadLevels;
        double preloadXmin = 0;
        double preloadYmin = 0;
        int preloadXptsnb = 0;
        int preloadYptsnb = 0;
        std::string preprocessMethod;
        vstds preprocessDatasetIds;
        vstds preprocessDataIds;
        vf preprocessLevels;
        vd preprocessHours;
        vi preprocessMembersNb;
        float level = 0;
        std::string gridType = "regular";
        double xMin = 0;
        double xStep = 0;
        double xShift = 0;
        double yMin = 0;
        double yStep = 0;
        double yShift = 0;
        int xPtsNb = 1;
        int yPtsNb = 1;
        int flatAllowed = asFLAT_FORBIDDEN;
        int membersNb = 0;
        double hour = 0;
        std::string criteria;
        float weight = 1;
    };

    using VectorParamsPredictors = vector<ParamsPredictor>;

    struct ParamsStep {
        int analogsNumber = 0;
        VectorParamsPredictors predictors;
    };

    using VectorParamsStep = vector<ParamsStep>;

    asParameters();

    ~asParameters() override = default;

    virtual void AddStep();

    void RemoveStep(int iStep);

    void AddPredictor();  // To the last step

    void AddPredictor(ParamsStep& step);

    void AddPredictor(int iStep);

    void RemovePredictor(int iStep, int iPtor);

    virtual bool LoadFromFile(const wxString& filePath = wxEmptyString);

    bool FixAnalogsNb();

    void SortLevelsAndTime();

    virtual bool SetSpatialWindowProperties();

    virtual bool SetPreloadingProperties();

    virtual bool InputsOK() const;

    bool PreprocessingPropertiesOk() const;

    static vi GetFileStationIds(wxString stationIdsString);

    wxString GetPredictandStationIdsString() const;

    static wxString PredictandStationIdsToString(const vi& predictandStationIds);

    virtual bool FixTimeLimits();

    virtual bool FixWeights();

    bool FixCoordinates();

    virtual wxString Print() const;

    bool IsSameAs(const asParameters& params) const;

    bool IsSameAs(const VectorParamsStep& params, const vi& predictandStationIds, int analogsIntervalDays) const;

    bool IsCloseTo(const asParameters& params) const;

    bool IsCloseTo(const VectorParamsStep& params, const vi& predictandStationIds, int analogsIntervalDays) const;

    bool PrintAndSaveTemp(const wxString& filePath = wxEmptyString) const;

    virtual bool GetValuesFromString(wxString stringVals);  // We copy the string as we'll modify it.

    void SetPredictandStationIds(wxString val);

    VectorParamsPredictors GetVectorParamsPredictors(int iStep) const {
        wxASSERT(iStep < GetStepsNb());
        return _steps[iStep].predictors;
    }

    void SetVectorParamsPredictors(int iStep, VectorParamsPredictors ptors) {
        wxASSERT(iStep < GetStepsNb());
        _steps[iStep].predictors = std::move(ptors);
    }

    wxString GetMethodId() const {
        return _methodId;
    }

    void SetMethodId(const wxString& val) {
        _methodId = val;
    }

    wxString GetMethodIdDisplay() const {
        return _methodIdDisplay;
    }

    void SetMethodIdDisplay(const wxString& val) {
        _methodIdDisplay = val;
    }

    wxString GetSpecificTag() const {
        return _specificTag;
    }

    void SetSpecificTag(const wxString& val) {
        _specificTag = val;
    }

    wxString GetSpecificTagDisplay() const {
        return _specificTagDisplay;
    }

    void SetSpecificTagDisplay(const wxString& val) {
        _specificTagDisplay = val;
    }

    wxString GetDescription() const {
        return _description;
    }

    void SetDescription(const wxString& val) {
        _description = val;
    }

    wxString GetDateProcessed() const {
        return _dateProcessed;
    }

    void SetDateProcessed(const wxString& val) {
        _dateProcessed = val;
    }

    void SetArchiveYearStart(int val);

    void SetArchiveYearEnd(int val);

    double GetArchiveStart() const {
        return _archiveStart;
    }

    void SetArchiveStart(const wxString& val);

    double GetArchiveEnd() const {
        return _archiveEnd;
    }

    void SetArchiveEnd(const wxString& val);

    double GetTimeShiftDays() const {
        if (_timeMinHours >= 0) {
            return 0;
        }
        if (_targetTimeStepHours < 24) {
            return 0;
        }

        return std::abs(floor(_timeMinHours / _targetTimeStepHours) * _targetTimeStepHours / 24.0);
    }

    double GetTimeSpanDays() const {
        double margin = 0;
        if (_timeMaxHours > 24 - _targetTimeStepHours) {
            margin = ceil(_timeMaxHours / _targetTimeStepHours) * _targetTimeStepHours / 24.0;
        }
        return std::abs(margin) + std::abs(GetTimeShiftDays());
    }

    double GetTargetTimeStepHours() const {
        return _targetTimeStepHours;
    }

    void SetTargetTimeStepHours(double val);

    double GetAnalogsTimeStepHours() const {
        return _analogsTimeStepHours;
    }

    void SetAnalogsTimeStepHours(double val);

    wxString GetTimeArrayTargetMode() const {
        return _timeArrayTargetMode;
    }

    void SetTimeArrayTargetMode(const wxString& val);

    wxString GetTimeArrayTargetPredictandSerieName() const {
        return _timeArrayTargetPredictandSerieName;
    }

    void SetTimeArrayTargetPredictandSerieName(const wxString& val);

    float GetTimeArrayTargetPredictandMinThreshold() const {
        return _timeArrayTargetPredictandMinThreshold;
    }

    void SetTimeArrayTargetPredictandMinThreshold(float val);

    float GetTimeArrayTargetPredictandMaxThreshold() const {
        return _timeArrayTargetPredictandMaxThreshold;
    }

    void SetTimeArrayTargetPredictandMaxThreshold(float val);

    wxString GetTimeArrayAnalogsMode() const {
        return _timeArrayAnalogsMode;
    }

    void SetTimeArrayAnalogsMode(const wxString& val);

    int GetAnalogsExcludeDays() const {
        return _analogsExcludeDays;
    }

    void SetAnalogsExcludeDays(int val);

    int GetAnalogsIntervalDays() const {
        return _analogsIntervalDays;
    }

    void SetAnalogsIntervalDays(int val);

    vi GetPredictandStationIds() const {
        return _predictandStationIds;
    }

    virtual vvi GetPredictandStationIdsVector() const {
        vvi vec;
        vec.push_back(_predictandStationIds);
        return vec;
    }

    void SetPredictandStationIds(const vi& val);

    double GetPredictandTimeHours() const {
        return _predictandTimeHours;
    }

    void SetPredictandTimeHours(double val);

    int GetAnalogsNumber(int iStep) const {
        return _steps[iStep].analogsNumber;
    }

    void SetAnalogsNumber(int iStep, int val);

    bool NeedsPreloading(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preload;
    }

    void SetPreload(int iStep, int iPtor, bool val) {
        _steps[iStep].predictors[iPtor].preload = val;
    }

    void SetStandardize(int iStep, int iPtor, bool val) {
        _steps[iStep].predictors[iPtor].standardize = val;
    }

    bool GetStandardize(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].standardize;
    }

    void SetStandardizeMean(int iStep, int iPtor, double val) {
        _steps[iStep].predictors[iPtor].standardizeMean = val;
    }

    double GetStandardizeMean(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].standardizeMean;
    }

    void SetStandardizeSd(int iStep, int iPtor, double val) {
        _steps[iStep].predictors[iPtor].standardizeSd = val;
    }

    double GetStandardizeSd(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].standardizeSd;
    }

    vwxs GetPreloadDataIds(int iStep, int iPtor) const {
        vwxs vals;
        for (const auto& preloadDataId : _steps[iStep].predictors[iPtor].preloadDataIds) {
            vals.push_back(preloadDataId);
        }
        return vals;
    }

    bool SetPreloadDataIds(int iStep, int iPtor, vwxs val);

    void SetPreloadDataIds(int iStep, int iPtor, wxString val);

    vd GetPreloadHours(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preloadHours;
    }

    bool SetPreloadHours(int iStep, int iPtor, vd val);

    void SetPreloadHours(int iStep, int iPtor, double val);

    vf GetPreloadLevels(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preloadLevels;
    }

    bool SetPreloadLevels(int iStep, int iPtor, vf val);

    void SetPreloadLevels(int iStep, int iPtor, float val);

    double GetPreloadXmin(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preloadXmin;
    }

    void SetPreloadXmin(int iStep, int iPtor, double val);

    int GetPreloadXptsnb(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preloadXptsnb;
    }

    void SetPreloadXptsnb(int iStep, int iPtor, int val);

    double GetPreloadYmin(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preloadYmin;
    }

    void SetPreloadYmin(int iStep, int iPtor, double val);

    int GetPreloadYptsnb(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preloadYptsnb;
    }

    void SetPreloadYptsnb(int iStep, int iPtor, int val);

    bool NeedsPreprocessing(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preprocess;
    }

    void SetPreprocess(int iStep, int iPtor, bool val) {
        _steps[iStep].predictors[iPtor].preprocess = val;
    }

    virtual int GetPreprocessSize(int iStep, int iPtor) const {
        return (int)_steps[iStep].predictors[iPtor].preprocessDataIds.size();
    }

    wxString GetPreprocessMethod(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].preprocessMethod;
    }

    void SetPreprocessMethod(int iStep, int iPtor, const wxString& val);

    bool NeedsGradientPreprocessing(int iStep, int iPtor) const;

    bool IsCriteriaUsingGradients(int iStep, int iPtor) const;

    void FixCriteriaIfGradientsPreprocessed(int iStep, int iPtor);

    void ForceUsingGradientsPreprocessing(int iStep, int iPtor);

    wxString GetPreprocessDatasetId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessDatasetId(int iStep, int iPtor, int iPre, const wxString& val);

    wxString GetPreprocessDataId(int iStep, int iPtor, int iPre) const;

    void SetPreprocessDataId(int iStep, int iPtor, int iPre, const wxString& val);

    float GetPreprocessLevel(int iStep, int iPtor, int iPre) const;

    void SetPreprocessLevel(int iStep, int iPtor, int iPre, float val);

    double GetPreprocessHour(int iStep, int iPtor, int iPre) const;

    double GetPreprocessTimeAsDays(int iStep, int iPtor, int iPre) const;

    void SetPreprocessHour(int iStep, int iPtor, int iPre, double val);

    int GetPreprocessMembersNb(int iStep, int iPtor, int iPre) const;

    void SetPreprocessMembersNb(int iStep, int iPtor, int iPre, int val);

    wxString GetPredictorDatasetId(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].datasetId;
    }

    void SetPredictorDatasetId(int iStep, int iPtor, const wxString& val);

    wxString GetPredictorDataId(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].dataId;
    }

    void SetPredictorDataId(int iStep, int iPtor, const wxString& val);

    float GetPredictorLevel(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].level;
    }

    void SetPredictorLevel(int iStep, int iPtor, float val);

    wxString GetPredictorGridType(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].gridType;
    }

    void SetPredictorGridType(int iStep, int iPtor, const wxString& val);

    double GetPredictorXmin(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].xMin;
    }

    void SetPredictorXmin(int iStep, int iPtor, double val);

    int GetPredictorXptsnb(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].xPtsNb;
    }

    void SetPredictorXptsnb(int iStep, int iPtor, int val);

    double GetPredictorXstep(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].xStep;
    }

    void SetPredictorXstep(int iStep, int iPtor, double val);

    double GetPredictorXshift(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].xShift;
    }

    void SetPredictorXshift(int iStep, int iPtor, double val);

    double GetPredictorYmin(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].yMin;
    }

    void SetPredictorYmin(int iStep, int iPtor, double val);

    int GetPredictorYptsnb(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].yPtsNb;
    }

    void SetPredictorYptsnb(int iStep, int iPtor, int val);

    double GetPredictorYstep(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].yStep;
    }

    void SetPredictorYstep(int iStep, int iPtor, double val);

    double GetPredictorYshift(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].yShift;
    }

    void SetPredictorYshift(int iStep, int iPtor, double val);

    int GetPredictorFlatAllowed(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].flatAllowed;
    }

    void SetPredictorFlatAllowed(int iStep, int iPtor, int val);

    double GetPredictorHour(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].hour;
    }

    double GetPredictorTimeAsDays(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].hour / 24.0;
    }

    void SetPredictorHour(int iStep, int iPtor, double val);

    int GetPredictorMembersNb(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].membersNb;
    }

    void SetPredictorMembersNb(int iStep, int iPtor, int val);

    wxString GetPredictorCriteria(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].criteria;
    }

    void SetPredictorCriteria(int iStep, int iPtor, const wxString& val);

    float GetPredictorWeight(int iStep, int iPtor) const {
        return _steps[iStep].predictors[iPtor].weight;
    }

    void SetPredictorWeight(int iStep, int iPtor, float val);

    int GetStepsNb() const {
        return (int)_steps.size();
    }

    int GetPredictorsNb(int iStep) const {
        wxASSERT(iStep < _steps.size());
        return (int)_steps[iStep].predictors.size();
    }

    virtual int GetPredictorDataIdNb(int iStep, int iPtor) const {
        (void)iStep;
        (void)iPtor;
        return 1;
    }

    VectorParamsStep GetParameters() const {
        return _steps;
    }

  protected:
    wxString _methodId;
    wxString _methodIdDisplay;
    wxString _specificTag;
    wxString _specificTagDisplay;
    wxString _description;
    double _archiveStart;
    double _archiveEnd;
    int _analogsIntervalDays;
    vi _predictandStationIds;
    double _timeMinHours;
    double _timeMaxHours;

  private:
    VectorParamsStep _steps;  // Set as private to force use of setters.
    wxString _dateProcessed;
    wxString _timeArrayTargetMode;
    double _targetTimeStepHours;
    wxString _timeArrayTargetPredictandSerieName;
    float _timeArrayTargetPredictandMinThreshold;
    float _timeArrayTargetPredictandMaxThreshold;
    wxString _timeArrayAnalogsMode;
    double _analogsTimeStepHours;
    int _analogsExcludeDays;
    asPredictand::Parameter _predictandParameter;
    asPredictand::TemporalResolution _predictandTemporalResolution;
    asPredictand::SpatialAggregation _predictandSpatialAggregation;
    wxString _predictandDatasetId;
    double _predictandTimeHours;

    bool ParseDescription(asFileParameters& fileParams, const wxXmlNode* nodeProcess);

    bool ParseTimeProperties(asFileParameters& fileParams, const wxXmlNode* nodeProcess);

    bool ParseAnalogDatesParams(asFileParameters& fileParams, int iStep, const wxXmlNode* nodeProcess);

    bool ParsePredictors(asFileParameters& fileParams, int iStep, int iPtor, const wxXmlNode* nodeParamBlock);

    bool ParsePreprocessedPredictors(asFileParameters& fileParams, int iStep, int iPtor, const wxXmlNode* nodeParam);

    bool ParseAnalogValuesParams(asFileParameters& fileParams, const wxXmlNode* nodeProcess);
};

#endif
