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

#ifndef ASDATAPREDICTAND_H
#define ASDATAPREDICTAND_H

#include <asIncludes.h>

class asCatalogPredictands;

class asFileNetcdf;


class asDataPredictand
        : public wxObject
{
public:
    asDataPredictand(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution,
                     DataSpatialAggregation dataAggregation);

    virtual ~asDataPredictand();

    static asDataPredictand *GetInstance(const wxString &dataParameterStr, const wxString &dataTemporalResolutionStr,
                                         const wxString &dataSpatialAggregationStr);

    static asDataPredictand *GetInstance(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution,
                                         DataSpatialAggregation dataSpatialAggregation);

    static asDataPredictand *GetInstance(const wxString &filePath);

    virtual bool Load(const wxString &AlternateFilePath = wxEmptyString) = 0;

    virtual bool Save(const wxString &AlternateFilePath = wxEmptyString) const = 0;

    virtual bool BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString,
                                   const wxString &AlternatePatternDir = wxEmptyString,
                                   const wxString &AlternateDestinationDir = wxEmptyString) = 0;

    virtual Array1DFloat GetReferenceAxis() const
    {
        Array1DFloat nodata(1);
        nodata << NaNFloat;
        return nodata;
    }

    virtual float GetReferenceValue(int i_station, double duration, float reference) const
    {
        return NaNFloat;
    }

    virtual Array2DFloat GetReferenceValuesArray() const
    {
        Array1DFloat nodata(1);
        nodata << NaNFloat;
        return nodata;
    }

    wxString GetDatasetId() const
    {
        return m_datasetId;
    }

    DataParameter GetDataParameter() const
    {
        return m_dataParameter;
    }

    void SetDataParameter(DataParameter val)
    {
        m_dataParameter = val;
    }

    DataTemporalResolution GetDataTemporalResolution() const
    {
        return m_dataTemporalResolution;
    }

    void SetDataTemporalResolution(DataTemporalResolution val)
    {
        m_dataTemporalResolution = val;
    }

    DataSpatialAggregation GetDataSpatialAggregation() const
    {
        return m_dataSpatialAggregation;
    }

    void SetDataSpatialAggregation(DataSpatialAggregation val)
    {
        m_dataSpatialAggregation = val;
    }

    bool HasNormalizedData() const
    {
        return m_hasNormalizedData;
    }

    bool HasReferenceValues() const
    {
        return m_hasReferenceValues;
    }

    double GetTimeStepDays() const
    {
        return m_timeStepDays;
    }

    double GetTimeStepHours() const
    {
        return m_timeStepDays * 24;
    }

    int GetStationsNb() const
    {
        return m_stationsNb;
    }

    int GetTimeLength() const
    {
        return m_timeLength;
    }

    VectorString &GetStationNamesArray()
    {
        return m_stationNames;
    }

    VectorString &GetStationOfficialIdsArray()
    {
        return m_stationOfficialIds;
    }

    Array1DFloat &GetStationHeightsArray()
    {
        return m_stationHeights;
    }

    Array1DInt &GetStationsIdArray()
    {
        return m_stationIds;
    }

    Array1DDouble &GetStationXCoordsArray()
    {
        return m_stationXCoords;
    }

    Array1DDouble &GetStationYCoordsArray()
    {
        return m_stationYCoords;
    }

    Array2DFloat &GetDataGross()
    {
        return m_dataGross;
    }

    Array1DFloat GetDataGrossStation(int predictandStationId) const
    {
        int indexStation = GetStationIndex(predictandStationId);
        return m_dataGross.col(indexStation);
    }

    Array2DFloat &GetDataNormalized()
    {
        if (m_hasNormalizedData) {
            return m_dataNormalized;
        } else {
            return m_dataGross;
        }
    }

    Array1DFloat GetDataNormalizedStation(int predictandStationId) const
    {
        int indexStation = GetStationIndex(predictandStationId);
        if (m_hasNormalizedData) {
            return m_dataNormalized.col(indexStation);
        } else {
            return m_dataGross.col(indexStation);
        }
    }

    Array1DDouble &GetTime()
    {
        return m_time;
    }

    int GetStationIndex(int stationId) const;

protected:
    // Single value
    float m_fileVersion;
    DataParameter m_dataParameter;
    DataTemporalResolution m_dataTemporalResolution;
    DataSpatialAggregation m_dataSpatialAggregation;
    wxString m_datasetId;
    double m_timeStepDays;
    int m_timeLength;
    int m_stationsNb;
    double m_dateProcessed;
    double m_dateStart;
    double m_dateEnd;
    bool m_hasNormalizedData;
    bool m_hasReferenceValues;
    // Matrix data
    Array2DFloat m_dataGross;
    Array2DFloat m_dataNormalized;
    // Vector (dim = time)
    Array1DDouble m_time;
    // Vector (dim = stations)
    VectorString m_stationNames;
    VectorString m_stationOfficialIds;
    Array1DInt m_stationIds;
    Array1DFloat m_stationHeights;
    Array1DDouble m_stationXCoords;
    Array1DDouble m_stationYCoords;
    Array1DDouble m_stationStarts;
    Array1DDouble m_stationEnds;

    wxString GetDBFilePathSaving(const wxString &destinationDir) const;

    bool InitMembers(const wxString &catalogFilePath);

    bool InitBaseContainers();

    bool LoadCommonData(asFileNetcdf &ncFile);

    void SetCommonDefinitions(asFileNetcdf &ncFile) const;

    bool SaveCommonData(asFileNetcdf &ncFile) const;

    bool ParseData(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString,
                   const wxString &AlternatePatternDir = wxEmptyString);

    Array2DFloat GetAnnualMax(double timeStepDays = 1, int nansNbMax = 10) const;

    bool SetStationProperties(asCatalogPredictands &currentData, size_t stationIndex);

    bool GetFileContent(asCatalogPredictands &currentData, size_t stationIndex,
                        const wxString &AlternateDataDir = wxEmptyString,
                        const wxString &AlternatePatternDir = wxEmptyString);

private:

    float ParseAndCheckDataValue(asCatalogPredictands &currentData, wxString &dataStr) const;
};

#endif // ASDATAPREDICTAND_H
