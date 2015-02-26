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

#ifndef ASRESULTSANALOGSFORECAST_H
#define ASRESULTSANALOGSFORECAST_H

#include <asIncludes.h>
#include <asResults.h>
#include <asParametersForecast.h>

class asResultsAnalogsForecast: public asResults
{
public:

    /** Default constructor */
    asResultsAnalogsForecast();

    /** Default destructor */
    virtual ~asResultsAnalogsForecast();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersForecast &params, double leadTimeOrigin);

    bool IsCompatibleWith(asResultsAnalogsForecast * otherForecast);

    bool IsSpecificForStationId(int stationId);

    int GetStationRowFromId(int stationId);

    wxString GetForecastsDirectory()
    {
        return m_ForecastsDirectory;
    }

    void SetForecastsDirectory(const wxString &val)
    {
        m_ForecastsDirectory = val;
    }

    wxString GetPredictandDatasetId()
    {
        return m_PredictandDatasetId;
    }

    void SetPredictandDatasetId(const wxString &val)
    {
        m_PredictandDatasetId = val;
    }

    wxString GetPredictandDatabase()
    {
        return m_PredictandDatabase;
    }

    void SetPredictandDatabase(const wxString &val)
    {
        m_PredictandDatabase = val;
    }

    VectorInt GetPredictandStationIds()
    {
        return m_PredictandStationIds;
    }

    void SetPredictandStationIds(VectorInt val)
    {
        m_PredictandStationIds = val;
    }
    
    void SetPredictandStationIds(wxString val);

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

    /** Access m_HasReferenceValues
    * \return The current value of m_HasReferenceValues
    */
    bool HasReferenceValues()
    {
        return m_HasReferenceValues;
    }

    wxString GetMethodId()
    {
        return m_MethodId;
    }

    void SetMethodId(const wxString &val)
    {
        m_MethodId = val;
    }

    wxString GetMethodIdDisplay()
    {
        return m_MethodIdDisplay;
    }

    void SetMethodIdDisplay(const wxString &val)
    {
        m_MethodIdDisplay = val;
    }

    wxString GetSpecificTag()
    {
        return m_SpecificTag;
    }

    void SetSpecificTag(const wxString &val)
    {
        m_SpecificTag = val;
    }

    wxString GetSpecificTagDisplay()
    {
        return m_SpecificTagDisplay;
    }

    void SetSpecificTagDisplay(const wxString &val)
    {
        m_SpecificTagDisplay = val;
    }

    wxString GetDescription()
    {
        return m_Description;
    }

    void SetDescription(const wxString &val)
    {
        m_Description = val;
    }

    /** Access m_LeadTimeOrigin
     * \return The origin of the lead time
     */
    double GetLeadTimeOrigin()
    {
        return m_LeadTimeOrigin;
    }

    /** Access m_LeadTimeOrigin
     * \return The origin of the lead time
     */
    wxString GetLeadTimeOriginString()
    {
        wxString leadTimeStr = asTime::GetStringTime(m_LeadTimeOrigin, "DD.MM.YYYY hh:mm");
        return leadTimeStr;
    }

    /** Get the number of stations
     * \return The number of stations
     */
    int GetStationsNb()
    {
        return (int)m_StationIds.size();
    }

    /** Access m_StationIds
     * \return The whole array m_StationIds
     */
    Array1DInt GetStationIds()
    {
        return m_StationIds;
    }

    /** Access an element of m_StationOfficialIds
     * \return An item of m_StationOfficialIds
     */
    wxString GetStationOfficialId(int i)
    {
        wxASSERT(i>=0);
        wxASSERT((unsigned)i<m_StationOfficialIds.size());
        return m_StationOfficialIds[i];
    }

    /** Access an element of m_StationNames
     * \return An item of m_StationNames
     */
    wxString GetStationName(int i)
    {
        wxASSERT(i>=0);
        wxASSERT((unsigned)i<m_StationNames.size());
        return m_StationNames[i];
    }


    wxArrayString GetStationNamesWxArrayString();


    wxArrayString GetStationNamesAndHeightsWxArrayString();

    wxString GetStationNameAndHeight(int i_stat);


    /** Set m_StationNames
     * \param stationsNames The new array to set
     */
    void SetStationNames(VectorString &stationsNames)
    {
        m_StationNames = stationsNames;
    }

    /** Access an element of m_StationIds
     * \return An item of m_StationIds
     */
    int GetStationId(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationIds.size());
        return m_StationIds[i];
    }

    /** Set m_StationIds
     * \param stationsIds The new array to set
     */
    void SetStationIds(Array1DInt &stationsIds)
    {
        m_StationIds = stationsIds;
    }

    void SetStationOfficialIds(VectorString &stationsOfficialIds)
    {
        m_StationOfficialIds = stationsOfficialIds;
    }

    /** Access an element of m_StationHeights
     * \return An item of m_StationHeights
     */
    int GetStationHeight(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationHeights.size());
        return m_StationHeights[i];
    }

    /** Set m_StationHeights
     * \param stationsHeights The new array to set
     */
    void SetStationHeights(Array1DFloat &stationsHeights)
    {
        m_StationHeights = stationsHeights;
    }

    /** Access m_StationXCoords
     * \return The whole array m_StationXCoords
     */
    Array1DDouble GetStationXCoords()
    {
        return m_StationXCoords;
    }

    /** Access an element of m_StationXCoords
     * \return An item of m_StationXCoords
     */
    double GetStationXCoord(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationXCoords.size());
        return m_StationXCoords[i];
    }

    /** Set m_StationXCoords
     * \param stationsXCoords The new array to set
     */
    void SetStationXCoords(Array1DDouble &stationsXCoords)
    {
        m_StationXCoords = stationsXCoords;
    }

    /** Access m_StationYCoords
     * \return The whole array m_StationYCoords
     */
    Array1DDouble GetStationYCoords()
    {
        return m_StationYCoords;
    }

    /** Access an element of m_StationYCoords
     * \return An item of m_StationYCoords
     */
    double GetStationYCoord(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationYCoords.size());
        return m_StationYCoords[i];
    }

    /** Set m_StationYCoords
     * \param stationsYCoords The new array to set
     */
    void SetStationYCoords(Array1DDouble &stationsYCoords)
    {
        m_StationYCoords = stationsYCoords;
    }

    /** Access m_ReferenceAxis
     * \return The whole array m_ReferenceAxis
     */
    Array1DFloat GetReferenceAxis()
    {
        return m_ReferenceAxis;
    }

    /** Set m_ReferenceAxis
     * \param referenceAxis The new array to set
     */
    void SetReferenceAxis(Array1DFloat &referenceAxis)
    {
        m_ReferenceAxis = referenceAxis;
        m_HasReferenceValues = true;
    }

    /** Access an element of m_ReferenceValues
     */
    float GetReferenceValue(int i_stat, int i_ref)
    {
        if (!m_HasReferenceValues)
        {
            asLogWarning(_("The predictand has no reference values. GetReferenceValue() should not be called."));
            return NaNFloat;
        }

        wxASSERT(i_stat>=0);
        wxASSERT(i_ref>=0);
        wxASSERT(i_stat<m_ReferenceValues.rows());
        wxASSERT(i_ref<m_ReferenceValues.cols());
        return m_ReferenceValues(i_stat, i_ref);
    }

    /** Access m_ReferenceValues
     * \return The whole array m_ReferenceValues
     */
    Array2DFloat GetReferenceValues()
    {
        if (!m_HasReferenceValues)
        {
            asLogWarning(_("The predictand has no reference values. GetReferenceValues() should not be called."));
            Array2DFloat nodata(0,0);
            return nodata;
        }

        return m_ReferenceValues;
    }

    /** Set m_ReferenceValues
     * \param referenceValues The new array to set
     */
    void SetReferenceValues(Array2DFloat &referenceValues)
    {
        m_ReferenceValues = referenceValues;
    }

    /** Get the size of m_TargetDates
     * \return The size of m_TargetDates
     */
    int GetTargetDatesLength()
    {
        return (int)m_TargetDates.size();
    }

    /** Access m_TargetDates
     * \return The whole array m_TargetDates
     */
    Array1DFloat &GetTargetDates()
    {
        return m_TargetDates;
    }

    /** Set m_TargetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DDouble &refDates)
    {
        m_TargetDates.resize(refDates.rows());
        for (int i=0; i<refDates.size(); i++)
        {
            m_TargetDates[i] = (float)refDates[i];
            wxASSERT_MSG(m_TargetDates[i]>1,_("The target time array has unconsistent values"));
        }
    }

    /** Set m_TargetDates
     * \param refDates The new array to set
     */
    void SetTargetDates(Array1DFloat &refDates)
    {
        m_TargetDates.resize(refDates.rows());
        m_TargetDates = refDates;
    }

    /** Access m_AnalogCriteria
     * \return The whole array m_AnalogCriteria
     */
    Array1DFloat &GetAnalogsCriteria(unsigned int i)
    {
        wxASSERT(m_AnalogsCriteria.size()>i);
        return m_AnalogsCriteria[i];
    }

    /** Set m_AnalogCriteria
     * \param analogCriteria The new array to set
     */
    void SetAnalogsCriteria(unsigned int i, Array1DFloat &analogsCriteria)
    {
        if (m_AnalogsCriteria.size()>=i+1)
        {
            m_AnalogsCriteria[i] = analogsCriteria;
        }
        else if (m_AnalogsCriteria.size()==i)
        {
            m_AnalogsCriteria.push_back(analogsCriteria);
        }
        else
        {
            asThrowException(_("The size of the criteria array does not fit with the required index."));
        }
    }

    /** Access m_AnalogValuesGross
     * \return The whole array m_AnalogValuesGross
     */
    Array2DFloat &GetAnalogsValuesGross(unsigned int i_leadtime)
    {
        wxASSERT(m_AnalogsValuesGross.size()>i_leadtime);
        return m_AnalogsValuesGross[i_leadtime];
    }

    /** Access m_AnalogValuesGross
     * \return The whole array m_AnalogValuesGross
     */
    Array1DFloat GetAnalogsValuesGross(unsigned int i_leadtime, int i_station)
    {
        wxASSERT(m_AnalogsValuesGross.size()>i_leadtime);
        wxASSERT(m_AnalogsValuesGross[i_leadtime].rows()>i_station);
        Array1DFloat vals = m_AnalogsValuesGross[i_leadtime].row(i_station);
        return vals;
    }

    /** Set m_AnalogValuesGross
     * \param analogValuesGross The new array to set
     */
    void SetAnalogsValuesGross(unsigned int i_leadtime, int i_station, Array1DFloat &analogsValuesGross)
    {
        if (m_AnalogsValuesGross.size()>=i_leadtime+1)
        {
            wxASSERT(m_AnalogsValuesGross[i_leadtime].rows()>i_station);
            wxASSERT(m_AnalogsValuesGross[i_leadtime].cols()==analogsValuesGross.size());
            m_AnalogsValuesGross[i_leadtime].row(i_station) = analogsValuesGross;
        }
        else if (m_AnalogsValuesGross.size()==i_leadtime)
        {
            Array2DFloat emptyBlock(m_StationIds.size(), m_AnalogsNb[i_leadtime]);
            m_AnalogsValuesGross.push_back(emptyBlock);

            wxASSERT(m_AnalogsValuesGross[i_leadtime].rows()>i_station);
            wxASSERT(m_AnalogsValuesGross[i_leadtime].cols()==analogsValuesGross.size());
            m_AnalogsValuesGross[i_leadtime].row(i_station) = analogsValuesGross;
        }
        else
        {
            asThrowException(_("The size of the values array does not fit with the required index."));
        }
    }

    /** Get the number of analogs
     * \return The number of analogs in m_AnalogsDates
     */
    int GetAnalogsDatesLength(int i)
    {
        wxASSERT(m_AnalogsDates.size()>(unsigned)i);
        return (int)m_AnalogsDates[i].size();
    }

    /** Get the length of the analogs dimension
     * \return The length of the analogs
     */
    int GetAnalogsNumber(int i)
    {
        wxASSERT(m_AnalogsDates.size()>(unsigned)i);
        return (int)m_AnalogsDates[i].size();
    }

    /** Access m_AnalogsDates
     * \return The whole array m_AnalogsDates
     */
    Array1DFloat &GetAnalogsDates(int i)
    {
        wxASSERT(m_AnalogsDates.size()>(unsigned)i);
        return m_AnalogsDates[i];
    }

    /** Set m_AnalogsDates
     * \param analogDates The new array to set
     */
    void SetAnalogsDates(unsigned int i, Array1DFloat &analogsDates)
    {
        if (m_AnalogsDates.size()>=i+1)
        {
            m_AnalogsDates[i] = analogsDates;
        }
        else if (m_AnalogsDates.size()==i)
        {
            m_AnalogsDates.push_back(analogsDates);
        }
        else
        {
            asThrowException(_("The size of the dates array does not fit with the required index."));
        }
    }

    /** Save the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Save(const wxString &AlternateFilePath = wxEmptyString);

    /** Load the result file
     * \param AlternateFilePath An optional file path
     * \return True on success
     */
    bool Load(const wxString &AlternateFilePath = wxEmptyString);

    wxString GetPredictandStationIdsString();


protected:

    /** Build the result file path
     */
    void BuildFileName();

private:
    wxString m_MethodId;
    wxString m_MethodIdDisplay;
    wxString m_SpecificTag;
    wxString m_SpecificTagDisplay;
    wxString m_Description;
    DataParameter m_PredictandParameter;
    DataTemporalResolution m_PredictandTemporalResolution;
    DataSpatialAggregation m_PredictandSpatialAggregation;
    wxString m_PredictandDatasetId;
    wxString m_PredictandDatabase;
    VectorInt m_PredictandStationIds;
    wxString m_ForecastsDirectory;
    bool m_HasReferenceValues;
    double m_LeadTimeOrigin;
    Array1DFloat m_TargetDates; //!< Member variable "m_TargetDates"
    Array1DInt m_AnalogsNb; //!< Member variable "m_AnalogsNb"
    VectorString m_StationNames; //!< Member variable "m_StationNames"
    VectorString m_StationOfficialIds;
    Array1DInt m_StationIds; //!< Member variable "m_StationIds"
    Array1DFloat m_StationHeights; //!< Member variable "m_StationHeights"
    Array1DDouble m_StationXCoords; //!< Member variable "m_StationXCoords"
    Array1DDouble m_StationYCoords; //!< Member variable "m_StationYCoords"
    Array1DFloat m_ReferenceAxis;
    Array2DFloat m_ReferenceValues;
    VArray1DFloat m_AnalogsCriteria; //!< Member variable "m_AnalogCriteria"
    VArray2DFloat m_AnalogsValuesGross; //!< Member variable "m_AnalogsValuesGross"
    VArray1DFloat m_AnalogsDates; //!< Member variable "m_AnalogDates"
};

#endif // ASRESULTSANALOGSFORECAST_H
