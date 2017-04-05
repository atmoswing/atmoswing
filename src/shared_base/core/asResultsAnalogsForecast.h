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

#ifndef ASRESULTSANALOGSFORECAST_H
#define ASRESULTSANALOGSFORECAST_H

#include <asIncludes.h>
#include <asResults.h>
#include <asParametersForecast.h>

class asResultsAnalogsForecast
        : public asResults
{
public:
    asResultsAnalogsForecast();

    virtual ~asResultsAnalogsForecast();

    void Init(asParametersForecast &params, double leadTimeOrigin);

    bool IsCompatibleWith(asResultsAnalogsForecast *otherForecast) const;

    bool IsSameAs(asResultsAnalogsForecast *otherForecast) const;

    bool IsSpecificForStationId(int stationId) const;

    int GetStationRowFromId(int stationId) const;

    wxString GetForecastsDirectory() const
    {
        return m_forecastsDirectory;
    }

    void SetForecastsDirectory(const wxString &val)
    {
        m_forecastsDirectory = val;
    }

    wxString GetPredictandDatasetId() const
    {
        return m_predictandDatasetId;
    }

    void SetPredictandDatasetId(const wxString &val)
    {
        m_predictandDatasetId = val;
    }

    wxString GetPredictandDatabase() const
    {
        return m_predictandDatabase;
    }

    void SetPredictandDatabase(const wxString &val)
    {
        m_predictandDatabase = val;
    }

    VectorInt GetPredictandStationIds() const
    {
        return m_predictandStationIds;
    }

    void SetPredictandStationIds(const VectorInt val)
    {
        m_predictandStationIds = val;
    }

    void SetPredictandStationIds(wxString val);

    asDataPredictand::Parameter GetPredictandParameter() const
    {
        return m_predictandParameter;
    }

    void SetPredictandParameter(const asDataPredictand::Parameter val)
    {
        m_predictandParameter = val;
    }

    asDataPredictand::TemporalResolution GetPredictandTemporalResolution() const
    {
        return m_predictandTemporalResolution;
    }

    void SetPredictandTemporalResolution(const asDataPredictand::TemporalResolution val)
    {
        m_predictandTemporalResolution = val;
    }

    asDataPredictand::SpatialAggregation GetPredictandSpatialAggregation() const
    {
        return m_predictandSpatialAggregation;
    }

    void SetPredictandSpatialAggregation(const asDataPredictand::SpatialAggregation val)
    {
        m_predictandSpatialAggregation = val;
    }

    bool HasReferenceValues() const
    {
        return m_hasReferenceValues;
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

    double GetLeadTimeOrigin() const
    {
        return m_leadTimeOrigin;
    }

    wxString GetLeadTimeOriginString()
    {
        wxString leadTimeStr = asTime::GetStringTime(m_leadTimeOrigin, "DD.MM.YYYY hh:mm");
        return leadTimeStr;
    }

    int GetStationsNb() const
    {
        return (int) m_stationIds.size();
    }

    Array1DInt GetStationIds() const
    {
        return m_stationIds;
    }

    wxString GetStationOfficialId(int i) const
    {
        wxASSERT(i >= 0);
        wxASSERT((unsigned) i < m_stationOfficialIds.size());
        return m_stationOfficialIds[i];
    }

    wxString GetStationName(int i) const
    {
        wxASSERT(i >= 0);
        wxASSERT((unsigned) i < m_stationNames.size());
        return m_stationNames[i];
    }

    wxArrayString GetStationNamesWxArrayString() const;

    wxArrayString GetStationNamesAndHeightsWxArrayString() const;

    wxString GetStationNameAndHeight(int iStat) const;

    void SetStationNames(const VectorString &stationsNames)
    {
        m_stationNames = stationsNames;
    }

    int GetStationId(int i) const
    {
        wxASSERT(i >= 0);
        wxASSERT(i < m_stationIds.size());
        return m_stationIds[i];
    }

    void SetStationIds(const Array1DInt &stationsIds)
    {
        m_stationIds = stationsIds;
    }

    void SetStationOfficialIds(const VectorString &stationsOfficialIds)
    {
        m_stationOfficialIds = stationsOfficialIds;
    }

    float GetStationHeight(int i) const
    {
        wxASSERT(i >= 0);
        wxASSERT(i < m_stationHeights.size());
        return m_stationHeights[i];
    }

    void SetStationHeights(const Array1DFloat &stationsHeights)
    {
        m_stationHeights = stationsHeights;
    }

    Array1DDouble GetStationXCoords() const
    {
        return m_stationXCoords;
    }

    double GetStationXCoord(int i) const
    {
        wxASSERT(i >= 0);
        wxASSERT(i < m_stationXCoords.size());
        return m_stationXCoords[i];
    }

    void SetStationXCoords(const Array1DDouble &stationsXCoords)
    {
        m_stationXCoords = stationsXCoords;
    }

    Array1DDouble GetStationYCoords() const
    {
        return m_stationYCoords;
    }

    double GetStationYCoord(int i) const
    {
        wxASSERT(i >= 0);
        wxASSERT(i < m_stationYCoords.size());
        return m_stationYCoords[i];
    }

    void SetStationYCoords(const Array1DDouble &stationsYCoords)
    {
        m_stationYCoords = stationsYCoords;
    }

    Array1DFloat GetReferenceAxis() const
    {
        return m_referenceAxis;
    }

    void SetReferenceAxis(const Array1DFloat &referenceAxis)
    {
        m_referenceAxis = referenceAxis;
        m_hasReferenceValues = true;
    }

    float GetReferenceValue(int iStat, int iRef) const
    {
        if (!m_hasReferenceValues) {
            wxLogWarning(_("The predictand has no reference values. GetReferenceValue() should not be called."));
            return NaNFloat;
        }

        wxASSERT(iStat >= 0);
        wxASSERT(iRef >= 0);
        wxASSERT(iStat < m_referenceValues.rows());
        wxASSERT(iRef < m_referenceValues.cols());
        return m_referenceValues(iStat, iRef);
    }

    Array2DFloat GetReferenceValues() const
    {
        if (!m_hasReferenceValues) {
            wxLogWarning(_("The predictand has no reference values. GetReferenceValues() should not be called."));
            Array2DFloat nodata(0, 0);
            return nodata;
        }

        return m_referenceValues;
    }

    void SetReferenceValues(const Array2DFloat &referenceValues)
    {
        m_referenceValues = referenceValues;
    }

    int GetTargetDatesLength() const
    {
        return (int) m_targetDates.size();
    }

    Array1DFloat &GetTargetDates()
    {
        return m_targetDates;
    }

    void SetTargetDates(const Array1DDouble &refDates)
    {
        m_targetDates.resize(refDates.rows());
        for (int i = 0; i < refDates.size(); i++) {
            m_targetDates[i] = (float) refDates[i];
            wxASSERT_MSG(m_targetDates[i] > 1, _("The target time array has unconsistent values"));
        }
    }

    void SetTargetDates(const Array1DFloat &refDates)
    {
        m_targetDates.resize(refDates.rows());
        m_targetDates = refDates;
    }

    Array1DFloat &GetAnalogsCriteria(unsigned int i)
    {
        wxASSERT(m_analogsCriteria.size() > i);
        return m_analogsCriteria[i];
    }

    void SetAnalogsCriteria(unsigned int i, const Array1DFloat &analogsCriteria)
    {
        if (m_analogsCriteria.size() >= i + 1) {
            m_analogsCriteria[i] = analogsCriteria;
        } else if (m_analogsCriteria.size() == i) {
            m_analogsCriteria.push_back(analogsCriteria);
        } else {
            asThrowException(_("The size of the criteria array does not fit with the required index."));
        }
    }

    Array2DFloat &GetAnalogsValuesGross(unsigned int iLead)
    {
        wxASSERT(m_analogsValuesGross.size() > iLead);
        return m_analogsValuesGross[iLead];
    }

    Array1DFloat GetAnalogsValuesGross(unsigned int iLead, int iStat) const
    {
        wxASSERT(m_analogsValuesGross.size() > iLead);
        wxASSERT(m_analogsValuesGross[iLead].rows() > iStat);
        Array1DFloat vals = m_analogsValuesGross[iLead].row(iStat);
        return vals;
    }

    void SetAnalogsValuesGross(unsigned int iLead, int iStat, const Array1DFloat &analogsValuesGross)
    {
        if (m_analogsValuesGross.size() >= iLead + 1) {
            wxASSERT(m_analogsValuesGross[iLead].rows() > iStat);
            wxASSERT(m_analogsValuesGross[iLead].cols() == analogsValuesGross.size());
            m_analogsValuesGross[iLead].row(iStat) = analogsValuesGross;
        } else if (m_analogsValuesGross.size() == iLead) {
            Array2DFloat emptyBlock(m_stationIds.size(), m_analogsNb[iLead]);
            m_analogsValuesGross.push_back(emptyBlock);

            wxASSERT(m_analogsValuesGross[iLead].rows() > iStat);
            wxASSERT(m_analogsValuesGross[iLead].cols() == analogsValuesGross.size());
            m_analogsValuesGross[iLead].row(iStat) = analogsValuesGross;
        } else {
            asThrowException(_("The size of the values array does not fit with the required index."));
        }
    }

    int GetAnalogsDatesLength(int i) const
    {
        wxASSERT(m_analogsDates.size() > (unsigned) i);
        return (int) m_analogsDates[i].size();
    }

    int GetAnalogsNumber(int i) const
    {
        wxASSERT(m_analogsDates.size() > (unsigned) i);
        return (int) m_analogsDates[i].size();
    }

    Array1DFloat &GetAnalogsDates(int i)
    {
        wxASSERT(m_analogsDates.size() > (unsigned) i);
        return m_analogsDates[i];
    }

    void SetAnalogsDates(unsigned int i, const Array1DFloat &analogsDates)
    {
        if (m_analogsDates.size() >= i + 1) {
            m_analogsDates[i] = analogsDates;
        } else if (m_analogsDates.size() == i) {
            m_analogsDates.push_back(analogsDates);
        } else {
            asThrowException(_("The size of the dates array does not fit with the required index."));
        }
    }

    bool Save();

    bool Load();

    wxString GetPredictandStationIdsString() const;

protected:
    void BuildFileName();

private:
    wxString m_methodId;
    wxString m_methodIdDisplay;
    wxString m_specificTag;
    wxString m_specificTagDisplay;
    wxString m_description;
    asDataPredictand::Parameter m_predictandParameter;
    asDataPredictand::TemporalResolution m_predictandTemporalResolution;
    asDataPredictand::SpatialAggregation m_predictandSpatialAggregation;
    wxString m_predictandDatasetId;
    wxString m_predictandDatabase;
    VectorInt m_predictandStationIds;
    wxString m_forecastsDirectory;
    bool m_hasReferenceValues;
    double m_leadTimeOrigin;
    Array1DFloat m_targetDates;
    Array1DInt m_analogsNb;
    VectorString m_stationNames;
    VectorString m_stationOfficialIds;
    Array1DInt m_stationIds;
    Array1DFloat m_stationHeights;
    Array1DDouble m_stationXCoords;
    Array1DDouble m_stationYCoords;
    Array1DFloat m_referenceAxis;
    Array2DFloat m_referenceValues;
    VArray1DFloat m_analogsCriteria;
    VArray2DFloat m_analogsValuesGross;
    VArray1DFloat m_analogsDates;
};

#endif // ASRESULTSANALOGSFORECAST_H
