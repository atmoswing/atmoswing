/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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
    asResultsAnalogsForecast(const wxString &modelName);

    /** Default destructor */
    virtual ~asResultsAnalogsForecast();

    /** Init
     * \param params The parameters structure
     */
    void Init(asParametersForecast &params, double leadTimeOrigin);


	wxString GetPredictandDatasetId()
    {
        return m_PredictandDatasetId;
    }

    void SetPredictandDatasetId(const wxString &val)
    {
        m_PredictandDatasetId = val;
    }

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

    /** Access m_ModelName
     * \return The model name
     */
    wxString GetModelName()
    {
        wxASSERT(!m_ModelName.IsEmpty());
        return m_ModelName;
    }

    /** Set m_ModelName
     * \param val The new model name to set
     */
    void SetModelName(const wxString &val)
    {
        m_ModelName = val;
        BuildFileName();
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

    /** Access m_StationsLocCoordU
     * \return The whole array m_StationsLocCoordU
     */
    Array1DDouble GetStationsLocCoordU()
    {
        return m_StationsLocCoordU;
    }

    /** Access an element of m_StationsLocCoordU
     * \return An item of m_StationsLocCoordU
     */
    float GetStationLocCoordU(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsLocCoordU.size());
        return m_StationsLocCoordU[i];
    }

    /** Set m_StationsLocCoordU
     * \param stationsLocCoordU The new array to set
     */
    void SetStationsLocCoordU(Array1DDouble &stationsLocCoordU)
    {
        m_StationsLocCoordU = stationsLocCoordU;
    }

    /** Access m_StationsLocCoordV
     * \return The whole array m_StationsLocCoordV
     */
    Array1DDouble GetStationsLocCoordV()
    {
        return m_StationsLocCoordV;
    }

    /** Access an element of m_StationsLocCoordV
     * \return An item of m_StationsLocCoordV
     */
    double GetStationLocCoordV(int i)
    {
        wxASSERT(i>=0);
        wxASSERT(i<m_StationsLocCoordV.size());
        return m_StationsLocCoordV[i];
    }

    /** Set m_StationsLocCoordV
     * \param stationsLocCoordV The new array to set
     */
    void SetStationsLocCoordV(Array1DDouble &stationsLocCoordV)
    {
        m_StationsLocCoordV = stationsLocCoordV;
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
    Array1DFloat &GetAnalogsCriteria(int i)
    {
        wxASSERT(m_AnalogsCriteria.size()>(unsigned)i);
        return m_AnalogsCriteria[i];
    }

    /** Set m_AnalogCriteria
     * \param analogCriteria The new array to set
     */
    void SetAnalogsCriteria(int i, Array1DFloat &analogsCriteria)
    {
        if (m_AnalogsCriteria.size()>=(unsigned)i+1)
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
    Array2DFloat &GetAnalogsValuesGross(int i_leadtime)
    {
        wxASSERT(m_AnalogsValuesGross.size()>(unsigned)i_leadtime);
        return m_AnalogsValuesGross[i_leadtime];
    }

    /** Access m_AnalogValuesGross
     * \return The whole array m_AnalogValuesGross
     */
    Array1DFloat GetAnalogsValuesGross(int i_leadtime, int i_station)
    {
        wxASSERT(m_AnalogsValuesGross.size()>(unsigned)i_leadtime);
        wxASSERT(m_AnalogsValuesGross[i_leadtime].rows()>i_station);
        Array1DFloat vals = m_AnalogsValuesGross[i_leadtime].row(i_station);
        return vals;
    }

    /** Set m_AnalogValuesGross
     * \param analogValuesGross The new array to set
     */
    void SetAnalogsValuesGross(int i_leadtime, int i_station, Array1DFloat &analogsValuesGross)
    {
        if (m_AnalogsValuesGross.size()>=(unsigned)i_leadtime+1)
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
    void SetAnalogsDates(int i, Array1DFloat &analogsDates)
    {
        if (m_AnalogsDates.size()>=(unsigned)i+1)
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


protected:

    /** Build the result file path
     */
    void BuildFileName();

private:
    DataParameter m_PredictandParameter;
	DataTemporalResolution m_PredictandTemporalResolution;
	DataSpatialAggregation m_PredictandSpatialAggregation;
	wxString m_PredictandDatasetId;


    wxString m_ModelName;
    wxString m_ModelLongName;


	bool m_HasReferenceValues;
    double m_LeadTimeOrigin;
    Array1DFloat m_TargetDates; //!< Member variable "m_TargetDates"
    Array1DInt m_AnalogsNb; //!< Member variable "m_AnalogsNb"
    VectorString m_StationsNames; //!< Member variable "m_StationsNames"
    Array1DInt m_StationsIds; //!< Member variable "m_StationsIds"
    Array1DFloat m_StationsHeights; //!< Member variable "m_StationsHeight"
    Array1DDouble m_StationsLat; //!< Member variable "m_StationsLat"
    Array1DDouble m_StationsLon; //!< Member variable "m_StationsLon"
    Array1DDouble m_StationsLocCoordU; //!< Member variable "m_StationsLocCoordU"
    Array1DDouble m_StationsLocCoordV; //!< Member variable "m_StationsLocCoordV"
    Array1DFloat m_ReferenceAxis;
    Array2DFloat m_ReferenceValues;
    VArray1DFloat m_AnalogsCriteria; //!< Member variable "m_AnalogCriteria"
    VArray2DFloat m_AnalogsValuesGross; //!< Member variable "m_AnalogsValuesGross"
    VArray1DFloat m_AnalogsDates; //!< Member variable "m_AnalogDates"
};

#endif // ASRESULTSANALOGSFORECAST_H
