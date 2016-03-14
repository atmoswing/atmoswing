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


class asDataPredictand: public wxObject
{
    public:
        asDataPredictand(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataAggregation);
        virtual ~asDataPredictand();

        static asDataPredictand* GetInstance(const wxString& dataParameterStr, const wxString& dataTemporalResolutionStr, const wxString& dataSpatialAggregationStr);
        
        static asDataPredictand* GetInstance(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataSpatialAggregation);
        
        static asDataPredictand* GetInstance(const wxString& filePath);

        /** Load the database from a local file
         * \param AlternateFilePath An alternate file path
         * \return True on success
         */
        virtual bool Load(const wxString &AlternateFilePath = wxEmptyString) = 0;

        virtual bool Save(const wxString &AlternateFilePath = wxEmptyString) = 0;

        virtual bool BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString, const wxString &AlternateDestinationDir = wxEmptyString) = 0;
        
        virtual Array1DFloat GetReferenceAxis()
        {
            Array1DFloat nodata(1);
            nodata << NaNFloat;
            return nodata;
        }

        virtual float GetReferenceValue(int i_station, double duration, float reference)
        {
            return NaNFloat;
        }

        virtual Array2DFloat GetReferenceValuesArray()
        {
            Array1DFloat nodata(1);
            nodata << NaNFloat;
            return nodata;
        }

        /** Access m_datasetId
         * \return The current value of m_datasetId
         */
        wxString GetDatasetId()
        {
            return m_datasetId;
        }
        
        DataParameter GetDataParameter()
        {
            return m_dataParameter;
        }

        void SetDataParameter(DataParameter val)
        {
            m_dataParameter = val;
        }

        DataTemporalResolution GetDataTemporalResolution()
        {
            return m_dataTemporalResolution;
        }

        void SetDataTemporalResolution(DataTemporalResolution val)
        {
            m_dataTemporalResolution = val;
        }

        DataSpatialAggregation GetDataSpatialAggregation()
        {
            return m_dataSpatialAggregation;
        }

        void SetDataSpatialAggregation(DataSpatialAggregation val)
        {
            m_dataSpatialAggregation = val;
        }

        /** Access m_hasNormalizedData
         * \return The current value of m_hasNormalizedData
         */
        bool HasNormalizedData()
        {
            return m_hasNormalizedData;
        }
        
        /** Access m_hasReferenceValues
         * \return The current value of m_hasReferenceValues
         */
        bool HasReferenceValues()
        {
            return m_hasReferenceValues;
        }

        /** Access m_timeStepDays
         * \return The current value of m_timeStepDays
         */
        double GetTimeStepDays()
        {
            return m_timeStepDays;
        }

        /** Access m_timeStepDays in hours
         * \return The current value of m_timeStepDays in hours
         */
        double GetTimeStepHours()
        {
            return m_timeStepDays*24;
        }

        /** Access m_stationsNb
         * \return The current value of m_stationsNb
         */
        int GetStationsNb()
        {
            return m_stationsNb;
        }

        /** Access m_stationsNb
         * \return The current value of m_stationsNb
         */
        int GetTimeLength()
        {
            return m_timeLength;
        }

        /** Access m_stationNames
         * \return The current value of m_stationNames
         */
        VectorString& GetStationNamesArray()
        {
            return m_stationNames;
        }

        /** Access m_stationOfficialIds
         * \return The current value of m_stationOfficialIds
         */
        VectorString& GetStationOfficialIdsArray()
        {
            return m_stationOfficialIds;
        }

        /** Access m_stationHeights
         * \return The current value of m_stationHeights
         */
        Array1DFloat& GetStationHeightsArray()
        {
            return m_stationHeights;
        }

        /** Access m_stationIds
         * \return The current value of m_stationIds
         */
        Array1DInt& GetStationsIdArray()
        {
            return m_stationIds;
        }

        /** Access m_stationXCoords
         * \return The current value of m_stationXCoords
         */
        Array1DDouble& GetStationXCoordsArray()
        {
            return m_stationXCoords;
        }

        /** Access m_stationYCoords
         * \return The current value of m_stationYCoords
         */
        Array1DDouble& GetStationYCoordsArray()
        {
            return m_stationYCoords;
        }

        /** Access m_dataGross: data(station,time)
         * \return The current value of m_dataGross
         */
        Array2DFloat& GetDataGross()
        {
            return m_dataGross;
        }

        /** Access m_dataGross: data(station,time)
         * \return The current value of m_dataGross
         */
        Array1DFloat GetDataGrossStation(int predictandStationId)
        {
            int indexStation = GetStationIndex(predictandStationId);
            return m_dataGross.col(indexStation);
        }

        /** Access m_dataNormalized: data(station,time)
         * \return The current value of m_dataNormalized
         */
        Array2DFloat& GetDataNormalized()
        {
            if(m_hasNormalizedData)
            {
                return m_dataNormalized;
            }
            else
            {
                return m_dataGross;
            }
        }

        /** Access m_dataNormalized for 1 station: data(station,time)
         * \return The current value of m_dataNormalized
         */
        Array1DFloat GetDataNormalizedStation(int predictandStationId)
        {
            int indexStation = GetStationIndex(predictandStationId);
            if(m_hasNormalizedData)
            {
                return m_dataNormalized.col(indexStation);
            }
            else
            {
                return m_dataGross.col(indexStation);
            }
        }

        /** Access m_time
         * \return The current value of m_time
         */
        Array1DDouble& GetTime()
        {
            return m_time;
        }

        /** Get the index of a station by its ID
         * \return The station index
         */
        int GetStationIndex(int stationId);


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


        wxString GetDBFilePathSaving(const wxString &destinationDir);


        /** Initialize the members
         * \param predictandDB The DB to build
         * \return True on success
         */
        bool InitMembers(const wxString &catalogFilePath);

        /** Initialize the containers size
         * \param predictandDB The DB to build
         * \return True on success
         */
        bool InitBaseContainers();


        bool LoadCommonData(asFileNetcdf &ncFile);

        void SetCommonDefinitions(asFileNetcdf &ncFile);

        bool SaveCommonData(asFileNetcdf &ncFile);

        bool ParseData(const wxString &catalogFilePath, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString);
        
        Array2DFloat GetAnnualMax(double timeStepDays = 1, int nansNbMax = 10);


        /** Set the stations properties
         * \param currentData The dataset description
         * \return True on success
         */
        bool SetStationProperties(asCatalogPredictands &currentData, size_t stationIndex);

        /** Get the file content
         * \param currentData The dataset description
         * \return True on success
         */
        bool GetFileContent(asCatalogPredictands &currentData, size_t stationIndex, const wxString &AlternateDataDir = wxEmptyString, const wxString &AlternatePatternDir = wxEmptyString);

    private:

        float ParseAndCheckDataValue(asCatalogPredictands &currentData, wxString &dataStr) const;
};

#endif // ASDATAPREDICTAND_H
