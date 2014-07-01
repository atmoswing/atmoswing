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

        /** Access m_DatasetId
         * \return The current value of m_DatasetId
         */
        wxString GetDatasetId()
        {
            return m_DatasetId;
        }
        
        DataParameter GetDataParameter()
        {
            return m_DataParameter;
        }

        void SetDataParameter(DataParameter val)
        {
            m_DataParameter = val;
        }

        DataTemporalResolution GetDataTemporalResolution()
        {
            return m_DataTemporalResolution;
        }

        void SetDataTemporalResolution(DataTemporalResolution val)
        {
            m_DataTemporalResolution = val;
        }

        DataSpatialAggregation GetDataSpatialAggregation()
        {
            return m_DataSpatialAggregation;
        }

        void SetDataSpatialAggregation(DataSpatialAggregation val)
        {
            m_DataSpatialAggregation = val;
        }

        /** Access m_HasNormalizedData
         * \return The current value of m_HasNormalizedData
         */
        bool HasNormalizedData()
        {
            return m_HasNormalizedData;
        }
        
        /** Access m_HasReferenceValues
         * \return The current value of m_HasReferenceValues
         */
        bool HasReferenceValues()
        {
            return m_HasReferenceValues;
        }

        /** Access m_TimeStepDays
         * \return The current value of m_TimeStepDays
         */
        double GetTimeStepDays()
        {
            return m_TimeStepDays;
        }

        /** Access m_TimeStepDays in hours
         * \return The current value of m_TimeStepDays in hours
         */
        double GetTimeStepHours()
        {
            return m_TimeStepDays*24;
        }

        /** Access m_StationsNb
         * \return The current value of m_StationsNb
         */
        int GetStationsNb()
        {
            return m_StationsNb;
        }

        /** Access m_StationsNb
         * \return The current value of m_StationsNb
         */
        int GetTimeLength()
        {
            return m_TimeLength;
        }

        /** Access m_StationsName
         * \return The current value of m_StationsName
         */
        VectorString& GetStationsNameArray()
        {
            return m_StationsName;
        }

        /** Access m_StationsHeight
         * \return The current value of m_StationsHeight
         */
        Array1DFloat& GetStationsHeightArray()
        {
            return m_StationsHeight;
        }

        /** Access m_StationsIds
         * \return The current value of m_StationsIds
         */
        Array1DInt& GetStationsIdArray()
        {
            return m_StationsIds;
        }

        /** Access m_StationsLocCoordU
         * \return The current value of m_StationsLocCoordU
         */
        Array1DDouble& GetStationsLocCoordUArray()
        {
            return m_StationsLocCoordU;
        }

        /** Access m_StationsLocCoordV
         * \return The current value of m_StationsLocCoordV
         */
        Array1DDouble& GetStationsLocCoordVArray()
        {
            return m_StationsLocCoordV;
        }

        /** Access m_StationsLon
         * \return The current value of m_StationsLon
         */
        Array1DDouble& GetStationsLonArray()
        {
            return m_StationsLon;
        }

        /** Access m_StationsLat
         * \return The current value of m_StationsLat
         */
        Array1DDouble& GetStationsLatArray()
        {
            return m_StationsLat;
        }

        /** Access m_DataGross: data(station,time)
         * \return The current value of m_DataGross
         */
        Array2DFloat& GetDataGross()
        {
            return m_DataGross;
        }

        /** Access m_DataGross: data(station,time)
         * \return The current value of m_DataGross
         */
        Array1DFloat GetDataGrossStation(int predictandStationId)
        {
            int indexStation = GetStationIndex(predictandStationId);
            return m_DataGross.col(indexStation);
        }

        /** Access m_DataNormalized: data(station,time)
         * \return The current value of m_DataNormalized
         */
        Array2DFloat& GetDataNormalized()
        {
            if(m_HasNormalizedData)
            {
                return m_DataNormalized;
            }
            else
            {
                return m_DataGross;
            }
        }

        /** Access m_DataNormalized for 1 station: data(station,time)
         * \return The current value of m_DataNormalized
         */
        Array1DFloat GetDataNormalizedStation(int predictandStationId)
        {
            int indexStation = GetStationIndex(predictandStationId);
            if(m_HasNormalizedData)
            {
                return m_DataNormalized.col(indexStation);
            }
            else
            {
                return m_DataGross.col(indexStation);
            }
        }

        /** Access m_Time
         * \return The current value of m_Time
         */
        Array1DDouble& GetTime()
        {
            return m_Time;
        }

        /** Get the index of a station by its ID
         * \return The station index
         */
        int GetStationIndex(int stationId);


    protected:
        // Single value
        float m_FileVersion;
        DataParameter m_DataParameter;
        DataTemporalResolution m_DataTemporalResolution;
        DataSpatialAggregation m_DataSpatialAggregation;
        wxString m_DatasetId;
        double m_TimeStepDays;
        int m_TimeLength;
        int m_StationsNb;
        double m_DateProcessed;
        double m_DateStart;
        double m_DateEnd;
        bool m_HasNormalizedData;
        bool m_HasReferenceValues;
        // Matrix data
        Array2DFloat m_DataGross;
        Array2DFloat m_DataNormalized;
        // Vector (dim = time)
        Array1DDouble m_Time;
        // Vector (dim = stations)
        VectorString m_StationsName;
        Array1DInt m_StationsIds;
        Array1DFloat m_StationsHeight;
        Array1DDouble m_StationsLocCoordU;
        Array1DDouble m_StationsLocCoordV;
        Array1DDouble m_StationsLon;
        Array1DDouble m_StationsLat;
        Array1DDouble m_StationsStart;
        Array1DDouble m_StationsEnd;



        wxString GetDBFilePathSaving(const wxString &AlternateDestinationDir);


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

};

#endif // ASDATAPREDICTAND_H
