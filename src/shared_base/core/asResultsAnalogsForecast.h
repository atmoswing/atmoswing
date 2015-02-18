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
        return (int)m_StationsIds.size();
    }

    /** Access m_StationsIds
     * \return The whole array m_StationsIds
     */
    Array1DInt GetStationsIds()
    {
        return m_StationsIds;
    }

    /** Access an element of m_StationsNames
     * \return An item of m_StationsNames
     */
    wxString GetStationName(int i)
    {
        wxASSERT(i>=0);
        wxASSERT((unsigned)i<m_StationsNames.size());
        return m_StationsNames[i];
    }


    wxArrayString GetStationNamesWxArrayString();


    wxArrayString GetStationNamesAndHeightsWxArrayString();

    wxString GetStationNameAndHeight(int i_stat);


    /** Set m_StationsNames
     * \param stationsNames The new array to set
     */
    void SetStationsNames(VectorString &stationsNames)
    {
        m_StationsNames = stationsNames;
    }

    /** Access an element of m_StationsIds
     * \return An item of m_StationsIds
     */
    int GetStationId(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsIds.size());
        return m_StationsIds[i];
    }

    /** Set m_StationsIds
     * \param stationsIds The new array to set
     */
    void SetStationsIds(Array1DInt &stationsIds)
    {
        m_StationsIds = stationsIds;
    }

    /** Access an element of m_StationsHeights
     * \return An item of m_StationsHeights
     */
    int GetStationHeight(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsHeights.size());
        return m_StationsHeights[i];
    }

    /** Set m_StationsHeights
     * \param stationsHeights The new array to set
     */
    void SetStationsHeights(Array1DFloat &stationsHeights)
    {
        m_StationsHeights = stationsHeights;
    }

    /** Access m_StationsLat
     * \return The whole array m_StationsLat
     */
    Array1DDouble GetStationsLat()
    {
        return m_StationsLat;
    }

    /** Access an element of m_StationsLat
     * \return An item of m_StationsLat
     */
    double GetStationLat(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsLat.size());
        return m_StationsLat[i];
    }

    /** Set m_StationsLat
     * \param stationsLat The new array to set
     */
    void SetStationsLat(Array1DDouble &stationsLat)
    {
        m_StationsLat = stationsLat;
    }

    /** Access m_StationsLon
     * \return The whole array m_StationsLon
     */
    Array1DDouble GetStationsLon()
    {
        return m_StationsLon;
    }

    /** Access an element of m_StationsLon
     * \return An item of m_StationsLon
     */
    float GetStationLon(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsLon.size());
        return m_StationsLon[i];
    }

    /** Set m_StationsLon
     * \param stationsLon The new array to set
     */
    void SetStationsLon(Array1DDouble &stationsLon)
    {
        m_StationsLon = stationsLon;
    }

    /** Access m_StationsLocCoordX
     * \return The whole array m_StationsLocCoordX
     */
    Array1DDouble GetStationsLocCoordX()
    {
        return m_StationsLocCoordX;
    }

    /** Access an element of m_StationsLocCoordX
     * \return An item of m_StationsLocCoordX
     */
    float GetStationLocCoordX(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsLocCoordX.size());
        return m_StationsLocCoordX[i];
    }

    /** Set m_StationsLocCoordX
     * \param stationsLocCoordX The new array to set
     */
    void SetStationsLocCoordX(Array1DDouble &stationsLocCoordX)
    {
        m_StationsLocCoordX = stationsLocCoordX;
    }

    /** Access m_StationsLocCoordY
     * \return The whole array m_StationsLocCoordY
     */
    Array1DDouble GetStationsLocCoordY()
    {
        return m_StationsLocCoordY;
    }

    /** Access an element of m_StationsLocCoordY
     * \return An item of m_StationsLocCoordY
     */
    double GetStationLocCoordY(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsLocCoordY.size());
        return m_StationsLocCoordY[i];
    }

    /** Set m_StationsLocCoordY
     * \param stationsLocCoordY The new array to set
     */
    void SetStationsLocCoordY(Array1DDouble &stationsLocCoordY)
    {
        m_StationsLocCoordY = stationsLocCoordY;
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
            Array2DFloat emptyBlock(m_StationsIds.size(), m_AnalogsNb[i_leadtime]);
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
    VectorString m_StationsNames; //!< Member variable "m_StationsNames"
    Array1DInt m_StationsIds; //!< Member variable "m_StationsIds"
    Array1DFloat m_StationsHeights; //!< Member variable "m_StationsHeight"
    Array1DDouble m_StationsLat; //!< Member variable "m_StationsLat"
    Array1DDouble m_StationsLon; //!< Member variable "m_StationsLon"
    Array1DDouble m_StationsLocCoordX; //!< Member variable "m_StationsLocCoordX"
    Array1DDouble m_StationsLocCoordY; //!< Member variable "m_StationsLocCoordY"
    Array1DFloat m_ReferenceAxis;
    Array2DFloat m_ReferenceValues;
    VArray1DFloat m_AnalogsCriteria; //!< Member variable "m_AnalogCriteria"
    VArray2DFloat m_AnalogsValuesGross; //!< Member variable "m_AnalogsValuesGross"
    VArray1DFloat m_AnalogsDates; //!< Member variable "m_AnalogDates"
};

#endif // ASRESULTSANALOGSFORECAST_H
