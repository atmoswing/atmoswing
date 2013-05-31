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
 
#ifndef ASDATAPREDICTAND_H
#define ASDATAPREDICTAND_H

#include <asIncludes.h>

class asCatalogPredictands;


class asDataPredictand: public wxObject
{
    public:
        asDataPredictand(PredictandDB predictandDB);
        virtual ~asDataPredictand();

        static asDataPredictand* GetInstance(const wxString& PredictandDBStr);

        /** Load the database from a local file
         * \param AlternateFilePath An alternate file path
         * \return True on success
         */
        virtual bool Load(const wxString &AlternateFilePath = wxEmptyString) = 0;

        virtual bool Save(const wxString &AlternateFilePath = wxEmptyString) = 0;

        virtual bool BuildPredictandDB(const wxString &AlternateDestinationDir = wxEmptyString) = 0;

        virtual Array1DFloat &GetReferenceAxis() = 0;

        virtual float GetReferenceValue(int i_station, double duration, float reference) = 0;

        virtual Array2DFloat &GetReferenceValuesArray() = 0;


        Array2DFloat GetAnnualMax(double timeStepDays = 1, int nansNbMax = 10);

        /** Access m_PredictandDB
         * \return The current value of m_PredictandDB
         */
        PredictandDB GetPredictandDB()
        {
            return m_PredictandDB;
        }

        /** Access m_PredictandDB as a string
         * \return The current value of m_PredictandDB as a string
         */
        wxString GetPredictandDBString()
        {
            return asGlobEnums::PredictandDBEnumToString(m_PredictandDB);
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
            return m_DataNormalized;
        }

        /** Access m_DataNormalized for 1 station: data(station,time)
         * \return The current value of m_DataNormalized
         */
        Array1DFloat GetDataNormalizedStation(int predictandStationId)
        {
            int indexStation = GetStationIndex(predictandStationId);
            return m_DataNormalized.col(indexStation);
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
        PredictandDB m_PredictandDB;
        double m_TimeStepDays;
        int m_TimeLength;
        int m_StationsNb;
        double m_DateProcessed;
        double m_DateStart;
        double m_DateEnd;
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

        /** Initialize the members
         * \param predictandDB The DB to build
         * \return True on success
         */
        bool InitMembers(const wxString &filePath = wxEmptyString);

        /** Initialize the containers size
         * \param predictandDB The DB to build
         * \return True on success
         */
        bool InitBaseContainers();

        /** Check if the dataset should be included in the DB
         * \param datasetID The dataset ID
         * \return True on success
         */
        bool IncludeInDB(const wxString &datasetID, const wxString &AlternateFilePath = wxEmptyString);

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
