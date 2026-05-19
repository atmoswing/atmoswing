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

#ifndef AS_RESULTS_FORECAST_H
#define AS_RESULTS_FORECAST_H

#include "asIncludes.h"
#include "asParametersForecast.h"
#include "asResults.h"

class asResultsForecast : public asResults {
  public:
    asResultsForecast();

    ~asResultsForecast() override = default;

    void Init(asParametersForecast& params, double leadTimeOrigin);

    bool IsCompatibleWith(asResultsForecast* otherForecast) const;

    bool IsSameAs(asResultsForecast* otherForecast) const;

    bool IsSpecificForStationId(int stationId) const;

    int GetStationRowFromId(int stationId) const;

    void SetForecastsDirectory(const wxString& val) {
        _forecastsDir = val;
    }

    wxString GetPredictandDatasetId() const {
        return _predictandDatasetId;
    }

    void SetPredictandDatasetId(const wxString& val) {
        _predictandDatasetId = val;
    }

    wxString GetPredictandDatabase() const {
        return _predictandDatabase;
    }

    vi GetPredictandStationIds() const {
        return _predictandStationIds;
    }

    void SetPredictandStationIds(const vi& val) {
        _predictandStationIds = val;
    }

    void SetPredictandStationIds(const wxString& val);

    asPredictand::Parameter GetPredictandParameter() const {
        return _predictandParameter;
    }

    void SetPredictandParameter(const asPredictand::Parameter val) {
        _predictandParameter = val;
    }

    asPredictand::TemporalResolution GetPredictandTemporalResolution() const {
        return _predictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(const asPredictand::TemporalResolution val) {
        _predictandTemporalResolution = val;
    }

    asPredictand::SpatialAggregation GetPredictandSpatialAggregation() const {
        return _predictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(const asPredictand::SpatialAggregation val) {
        _predictandSpatialAggregation = val;
    }

    bool HasReferenceValues() const {
        return _hasReferenceValues;
    }

    wxString GetMethodId() const {
        return _methodId;
    }

    wxString GetMethodIdDisplay() const {
        return _methodIdDisplay;
    }

    wxString GetSpecificTag() const {
        return _specificTag;
    }

    wxString GetSpecificTagDisplay() const {
        return _specificTagDisplay;
    }

    wxString GetDescription() const {
        return _description;
    }

    double GetLeadTimeOrigin() const {
        return _leadTimeOrigin;
    }

    wxString GetLeadTimeOriginString() {
        wxString leadTimeStr = asTime::GetStringTime(_leadTimeOrigin, "DD.MM.YYYY hh:mm");
        return leadTimeStr;
    }

    int GetStationsNb() const {
        return (int)_stationIds.size();
    }

    a1i GetStationIds() const {
        return _stationIds;
    }

    wxString GetStationOfficialId(int i) const {
        wxASSERT(i >= 0);
        wxASSERT(i < _stationOfficialIds.size());
        return _stationOfficialIds[i];
    }

    wxString GetStationName(int i) const {
        wxASSERT(i >= 0);
        wxASSERT(i < _stationNames.size());
        return _stationNames[i];
    }

    wxArrayString GetStationNamesWxArray() const;

    wxArrayString GetStationNamesAndHeightsWxArray() const;

    wxString GetStationNameAndHeight(int iStat) const;

    void SetStationNames(const vwxs& stationsNames) {
        _stationNames = stationsNames;
    }

    int GetStationId(int i) const {
        wxASSERT(i >= 0);
        wxASSERT(i < _stationIds.size());
        return _stationIds[i];
    }

    void SetStationIds(const a1i& stationsIds) {
        _stationIds = stationsIds;
    }

    void SetStationOfficialIds(const vwxs& stationsOfficialIds) {
        _stationOfficialIds = stationsOfficialIds;
    }

    float GetStationHeight(int i) const {
        wxASSERT(i >= 0);
        wxASSERT(i < _stationHeights.size());
        return _stationHeights[i];
    }

    void SetStationHeights(const a1f& stationsHeights) {
        _stationHeights = stationsHeights;
    }

    double GetStationXCoord(int i) const {
        wxASSERT(i >= 0);
        wxASSERT(i < _stationXCoords.size());
        return _stationXCoords[i];
    }

    void SetStationXCoords(const a1d& stationsXCoords) {
        _stationXCoords = stationsXCoords;
    }

    double GetStationYCoord(int i) const {
        wxASSERT(i >= 0);
        wxASSERT(i < _stationYCoords.size());
        return _stationYCoords[i];
    }

    void SetStationYCoords(const a1d& stationsYCoords) {
        _stationYCoords = stationsYCoords;
    }

    wxString GetCoordinateSystem() const {
        return _coordinateSystem;
    }

    void SetCoordinateSystem(const wxString& val) {
        _coordinateSystem = val;
    }

    a1f GetReferenceAxis() const {
        return _referenceAxis;
    }

    void SetReferenceAxis(const a1f& referenceAxis) {
        _referenceAxis = referenceAxis;
        _hasReferenceValues = true;
    }

    float GetReferenceValue(int iStat, int iRef) const {
        if (!_hasReferenceValues) {
            wxLogWarning(_("The predictand has no reference values. GetReferenceValue() should not be called."));
            return NAN;
        }

        wxASSERT(iStat >= 0);
        wxASSERT(iRef >= 0);
        wxASSERT(iStat < _referenceValues.rows());
        wxASSERT(iRef < _referenceValues.cols());
        return _referenceValues(iStat, iRef);
    }

    a2f GetReferenceValues() const {
        if (!_hasReferenceValues) {
            wxLogWarning(_("The predictand has no reference values. GetReferenceValues() should not be called."));
            a2f nodata(0, 0);
            return nodata;
        }

        return _referenceValues;
    }

    void SetReferenceValues(const a2f& referenceValues) {
        _referenceValues = referenceValues;
    }

    void SetPredictorDatasetIdsOper(const vwxs& predictorDatasetIdsOper) {
        _predictorDatasetIdsOper = predictorDatasetIdsOper;
    }

    vwxs GetPredictorDatasetIdsOper() {
        return _predictorDatasetIdsOper;
    }

    void SetPredictorDatasetIdsArchive(const vwxs& predictorDatasetIdsArchive) {
        _predictorDatasetIdsArchive = predictorDatasetIdsArchive;
    }

    vwxs GetPredictorDatasetIdsArchive() {
        return _predictorDatasetIdsArchive;
    }

    void SetPredictorDataIdsOper(const vwxs& predictorDataIdsOper) {
        _predictorDataIdsOper = predictorDataIdsOper;
    }

    vwxs GetPredictorDataIdsOper() {
        return _predictorDataIdsOper;
    }

    void SetPredictorDataIdsArchive(const vwxs& predictorDataIdsArchive) {
        _predictorDataIdsArchive = predictorDataIdsArchive;
    }

    vwxs GetPredictorDataIdsArchive() {
        return _predictorDataIdsArchive;
    }

    void SetPredictorLevels(const vf& predictorLevels) {
        _predictorLevels = predictorLevels;
    }

    vf GetPredictorLevels() {
        return _predictorLevels;
    }

    void SetPredictorHours(const vf& predictorHours) {
        _predictorHours = predictorHours;
    }

    vf GetPredictorHours() {
        return _predictorHours;
    }

    void SetPredictorLonMin(const vf& predictorLonMin) {
        _predictorLonMin = predictorLonMin;
    }

    vf GetPredictorLonMin() {
        return _predictorLonMin;
    }

    void SetPredictorLonMax(const vf& predictorLonMax) {
        _predictorLonMax = predictorLonMax;
    }

    vf GetPredictorLonMax() {
        return _predictorLonMax;
    }

    void SetPredictorLatMin(const vf& predictorLatMin) {
        _predictorLatMin = predictorLatMin;
    }

    vf GetPredictorLatMin() {
        return _predictorLatMin;
    }

    void SetPredictorLatMax(const vf& predictorLatMax) {
        _predictorLatMax = predictorLatMax;
    }

    vf GetPredictorLatMax() {
        return _predictorLatMax;
    }

    int GetTargetDatesLength() const {
        return (int)_targetDates.size();
    }

    a1f& GetTargetDates() {
        return _targetDates;
    }

    void LimitDataToHours(int hours);

    void LimitDataToDays(int days);

    void LimitDataToNbTimeSteps(int length);

    wxString GetDateFormatting() const;

    double GetForecastTimeStepHours() const;

    bool IsSubDaily() const;

    wxArrayString GetTargetDatesWxArray() const;

    void SetTargetDates(const a1d& refDates) {
        _targetDates.resize(refDates.rows());
        for (int i = 0; i < refDates.size(); i++) {
            _targetDates[i] = (float)refDates[i];
            wxASSERT_MSG(_targetDates[i] > 1, _("The target time array has inconsistent values"));
        }
    }

    void SetTargetDates(const a1f& refDates) {
        _targetDates.resize(refDates.rows());
        _targetDates = refDates;
    }

    a1f& GetAnalogsCriteria(int i) {
        wxASSERT(_analogsCriteria.size() > i);
        return _analogsCriteria[i];
    }

    void SetAnalogsCriteria(int i, const a1f& analogsCriteria) {
        if (_analogsCriteria.size() >= i + 1) {
            _analogsCriteria[i] = analogsCriteria;
        } else if (_analogsCriteria.size() == i) {
            _analogsCriteria.push_back(analogsCriteria);
        } else {
            throw runtime_error(_("The size of the criteria array does not fit with the required index."));
        }
    }

    a2f& GetAnalogsValuesRaw(int iLead) {
        wxASSERT(_analogsValuesRaw.size() > iLead);
        return _analogsValuesRaw[iLead];
    }

    a1f GetAnalogsValuesRaw(int iLead, int iStat) const {
        wxASSERT(_analogsValuesRaw.size() > iLead);
        wxASSERT(_analogsValuesRaw[iLead].rows() > iStat);
        a1f vals = _analogsValuesRaw[iLead].row(iStat);
        return vals;
    }

    void SetAnalogsValuesRaw(int iLead, int iStat, const a1f& analogsValuesRaw) {
        if (_analogsValuesRaw.size() >= iLead + 1) {
            wxASSERT(_analogsValuesRaw[iLead].rows() > iStat);
            wxASSERT(_analogsValuesRaw[iLead].cols() == analogsValuesRaw.size());
            _analogsValuesRaw[iLead].row(iStat) = analogsValuesRaw;
        } else if (_analogsValuesRaw.size() == iLead) {
            a2f emptyBlock(_stationIds.size(), _analogsNb[iLead]);
            _analogsValuesRaw.push_back(emptyBlock);

            wxASSERT(_analogsValuesRaw[iLead].rows() > iStat);
            wxASSERT(_analogsValuesRaw[iLead].cols() == analogsValuesRaw.size());
            _analogsValuesRaw[iLead].row(iStat) = analogsValuesRaw;
        } else {
            throw runtime_error(_("The size of the values array does not fit with the required index."));
        }
    }

    a2f& GetAnalogsValuesNorm(int iLead) {
        wxASSERT(_analogsValuesNorm.size() > iLead);
        return _analogsValuesNorm[iLead];
    }

    a1f GetAnalogsValuesNorm(int iLead, int iStat) const {
        wxASSERT(_analogsValuesNorm.size() > iLead);
        wxASSERT(_analogsValuesNorm[iLead].rows() > iStat);
        a1f vals = _analogsValuesNorm[iLead].row(iStat);
        return vals;
    }

    void SetAnalogsValuesNorm(int iLead, int iStat, const a1f& analogsValuesNorm) {
        if (_analogsValuesNorm.size() >= iLead + 1) {
            wxASSERT(_analogsValuesNorm[iLead].rows() > iStat);
            wxASSERT(_analogsValuesNorm[iLead].cols() == analogsValuesNorm.size());
            _analogsValuesNorm[iLead].row(iStat) = analogsValuesNorm;
        } else if (_analogsValuesNorm.size() == iLead) {
            a2f emptyBlock(_stationIds.size(), _analogsNb[iLead]);
            _analogsValuesNorm.push_back(emptyBlock);

            wxASSERT(_analogsValuesNorm[iLead].rows() > iStat);
            wxASSERT(_analogsValuesNorm[iLead].cols() == analogsValuesNorm.size());
            _analogsValuesNorm[iLead].row(iStat) = analogsValuesNorm;
        } else {
            throw runtime_error(_("The size of the values array does not fit with the required index."));
        }
    }

    int GetAnalogsNumber(int i) const {
        wxASSERT(_analogsDates.size() > i);
        return (int)_analogsDates[i].size();
    }

    a1f& GetAnalogsDates(int i) {
        wxASSERT(_analogsDates.size() > i);
        return _analogsDates[i];
    }

    void SetAnalogsDates(int i, const a1f& analogsDates) {
        if (_analogsDates.size() >= i + 1) {
            _analogsDates[i] = analogsDates;
        } else if (_analogsDates.size() == i) {
            _analogsDates.push_back(analogsDates);
        } else {
            throw runtime_error(_("The size of the dates array does not fit with the required index."));
        }
    }

    bool Save() override;

    bool Load() override;

    wxString GetPredictandStationIdsString() const;

    Coo GetStationsMeanCoordinates();

  protected:
    void BuildFileName();

  private:
    wxString _methodId;
    wxString _methodIdDisplay;
    wxString _specificTag;
    wxString _specificTagDisplay;
    wxString _description;
    asPredictand::Parameter _predictandParameter;
    asPredictand::TemporalResolution _predictandTemporalResolution;
    asPredictand::SpatialAggregation _predictandSpatialAggregation;
    wxString _predictandDatasetId;
    wxString _predictandDatabase;
    wxString _coordinateSystem;
    vi _predictandStationIds;
    wxString _forecastsDir;
    bool _hasReferenceValues;
    double _leadTimeOrigin;
    a1f _targetDates;
    a1i _analogsNb;
    vwxs _stationNames;
    vwxs _stationOfficialIds;
    a1i _stationIds;
    a1f _stationHeights;
    a1d _stationXCoords;
    a1d _stationYCoords;
    a1f _referenceAxis;
    a2f _referenceValues;
    vwxs _predictorDatasetIdsOper;
    vwxs _predictorDatasetIdsArchive;
    vwxs _predictorDataIdsOper;
    vwxs _predictorDataIdsArchive;
    vf _predictorLevels;
    vf _predictorHours;
    vf _predictorLonMin;
    vf _predictorLonMax;
    vf _predictorLatMin;
    vf _predictorLatMax;
    va1f _analogsCriteria;
    va2f _analogsValuesRaw;
    va2f _analogsValuesNorm;
    va1f _analogsDates;
};

#endif
