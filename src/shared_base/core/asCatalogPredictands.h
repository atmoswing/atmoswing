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
 
#ifndef ASCATALOGPREDICTAND_H
#define ASCATALOGPREDICTAND_H

#include <asIncludes.h>


class asCatalogPredictands: public wxObject
{
public:
    
    //!< Structure for datasets IDs
    struct DatasetIdList
    {
        VectorString Id;
        VectorString Name;
        VectorString Description;
    };

    //!< Structure for data IDs
    struct DataIdListStr
    {
        VectorString Id;
        VectorString Name;
    };

    //!< Structure for data IDs
    struct DataIdListInt
    {
        VectorInt Id;
        VectorString Name;
    };

    //!< Structure for data information
    struct DataStruct
    {
        int Id;
        wxString OfficialId;
        wxString Name;
        wxString Filename;
        wxString Filepattern;
        double Start;
        double End;
        Coo Coord;
        float Height;
    };

    /** Default constructor
     * \param DataSetId The dataset ID
     * \param StationId The station ID. If not set, load the whole database information
     */
    asCatalogPredictands(const wxString &filePath);

    /** Default destructor */
    virtual ~asCatalogPredictands();

    bool Load();

    /** Access m_catalogFilePath
     * \return The current value of m_catalogFilePath
     */
    wxString GetCatalogFilePath()
    {
        return m_catalogFilePath;
    }

    /** Set m_catalogFilePath
     * \param val New value to set
     */
    void SetCatalogFilePath(const wxString &val)
    {
        m_catalogFilePath = val;
    }

    /** Access m_setId
     * \return The current value of m_setId
     */
    wxString GetSetId()
    {
        return m_setId;
    }

    /** Set m_setId
     * \param val New value to set
     */
    void SetSetId(const wxString &val)
    {
        m_setId = val;
    }

    /** Access m_name
     * \return The current value of m_name
     */
    wxString GetName()
    {
        return m_name;
    }

    /** Set m_name
     * \param val New value to set
     */
    void SetName(const wxString &val)
    {
        m_name = val;
    }

    /** Access m_description
     * \return The current value of m_description
     */
    wxString GetDescription()
    {
        return m_description;
    }

    /** Set m_description
     * \param val New value to set
     */
    void SetDescription(const wxString &val)
    {
        m_description = val;
    }

    /** Access m_start
     * \return The current value of m_start
     */
    double GetStart()
    {
        return m_start;
    }

    /** Set m_start
     * \param val New value to set
     */
    void SetStart(double val)
    {
        m_start = val;
    }

    /** Access m_end
     * \return The current value of m_end
     */
    double GetEnd()
    {
        return m_end;
    }

    /** Set m_end
     * \param val New value to set
     */
    void SetEnd(double val)
    {
        m_end = val;
    }

    /** Access m_timeZoneHours
     * \return The current value of m_timeZoneHours
     */
    float GetTimeZoneHours()
    {
        return m_timeZoneHours;
    }

    /** Set m_timeZoneHours
     * \param val New value to set
     */
    void SetTimeZone(float val)
    {
        m_timeZoneHours = val;
    }

    /** Access m_timeStep in minutes
     * \return The current value of m_timeStep in minutes
     */
    double GetTimeStepMinutes()
    {
        return m_timeStepHours*60;
    }

    /** Set m_timeStep in minutes
     * \param val New value to set in minutes
     */
    void SetTimeStepMinutes(double val)
    {
        m_timeStepHours = val/60;
    }

    /** Access m_timeStep in hours
     * \return The current value of m_timeStep in hours
     */
    double GetTimeStepHours()
    {
        return m_timeStepHours;
    }

    /** Set m_timeStep in hours
     * \param val New value to set in hours
     */
    void SetTimeStepHours(double val)
    {
        m_timeStepHours = val;
    }

    /** Access m_timeStep in days
     * \return The current value of m_timeStep in days
     */
    double GetTimeStepDays()
    {
        return m_timeStepHours/24;
    }

    /** Set m_timeStep in days
     * \param val New value to set in days
     */
    void SetTimeStepDays(double val)
    {
        m_timeStepHours = val*24;
    }

    /** Access m_firstTimeStepHour in minutes
     * \return The current value of m_firstTimeStep in minutes
     */
    double GetFirstTimeStepMinutes()
    {
        return m_firstTimeStepHour*60;
    }

    /** Set m_firstTimeStepHour in minutes
     * \param val New value to set in minutes
     */
    void SetFirstTimeStepMinutes(double val)
    {
        m_firstTimeStepHour = val/60;
    }

    /** Access m_firstTimeStepHour in hours
     * \return The current value of m_firstTimeStep in hours
     */
    double GetFirstTimeStepHours()
    {
        return m_firstTimeStepHour;
    }

    /** Set m_firstTimeStepHour in hours
     * \param val New value to set in hours
     */
    void SetFirstTimeStepHours(double val)
    {
        m_firstTimeStepHour = val;
    }

    /** Access m_dataPath
     * \return The current value of m_dataPath
     */
    wxString GetDataPath()
    {
        return m_dataPath;
    }

    /** Set m_dataPath
     * \param val New value to set
     */
    void SetDataPath(const wxString &val)
    {
        m_dataPath = val;
    }

    /** Access m_nan
     * \return The current value of m_nan
     */
    VectorString GetNan()
    {
        return m_nan;
    }

    /** Set m_nan
     * \param val New value to set
     */
    void SetNan(const VectorString &val)
    {
        m_nan = val;
    }

    /** Access m_coordSys
     * \return The current value of m_coordSys
     */
    wxString GetCoordSys()
    {
        return m_coordSys;
    }

    /** Set m_coordSys
     * \param val New value to set
     */
    void SetCoordSys(wxString val)
    {
        m_coordSys = val;
    }

    /** Access m_stations
     * \return The current value of m_stations
     */
    DataStruct GetStationInfo(int index)
    {
        return m_stations[index];
    }

    /** Access m_parameter
     * \return The current value of m_parameter
     */
    DataParameter GetParameter()
    {
        return m_parameter;
    }

    /** Set m_parameter
     * \param val New value to set
     */
    void SetParameter(DataParameter val)
    {
        m_parameter = val;
    }

    /** Access m_unit
     * \return The current value of m_unit
     */
    DataUnit GetUnit()
    {
        return m_unit;
    }

    /** Set m_unit
     * \param val New value to set
     */
    void SetUnit(DataUnit val)
    {
        m_unit = val;
    }

    /** Access m_stations[index].Id
     * \return The current value of m_stations[index].Id
     */
    int GetStationId(int index)
    {
        return m_stations[index].Id;
    }

    /** Set m_stations[index].Id
     * \param val New value to set
     */
    void SetStationId(int index, int val)
    {
        m_stations[index].Id = val;
    }

    /** Access m_stations[index].OfficialId
     * \return The current value of m_stations[index].OfficialId
     */
    wxString GetStationOfficialId(int index)
    {
        return m_stations[index].OfficialId;
    }

    /** Set m_stations[index].OfficialId
     * \param val New value to set
     */
    void SetStationOfficialId(int index, const wxString &val)
    {
        m_stations[index].OfficialId = val;
    }

    /** Access m_stations[index].Name
     * \return The current value of m_stations[index].Name
     */
    wxString GetStationName(int index)
    {
        return m_stations[index].Name;
    }

    /** Set m_stations[index].Name
     * \param val New value to set
     */
    void SetStationName(int index, const wxString &val)
    {
        m_stations[index].Name = val;
    }

    /** Access m_stations[index].Filename
     * \return The current value of m_stations[index].Filename
     */
    wxString GetStationFilename(int index)
    {
        return m_stations[index].Filename;
    }

    /** Set m_stations[index].Filename
     * \param val New value to set
     */
    void SetStationFilename(int index, const wxString &val)
    {
        m_stations[index].Filename = val;
    }

    /** Access m_stations[index].Filepattern
     * \return The current value of m_stations[index].Filepattern
     */
    wxString GetStationFilepattern(int index)
    {
        return m_stations[index].Filepattern;
    }

    /** Set m_stations[index].Filepattern
     * \param val New value to set
     */
    void SetStationFilepattern(int index, const wxString &val)
    {
        m_stations[index].Filepattern = val;
    }

    /** Access m_stations[index].Start
     * \return The current value of m_stations[index].Start
     */
    double GetStationStart(int index)
    {
        return m_stations[index].Start;
    }

    /** Set m_stations[index].Start
     * \param val New value to set
     */
    void SetStationStart(int index, double val)
    {
        m_stations[index].Start = val;
    }

    /** Access m_stations[index].End
     * \return The current value of m_stations[index].End
     */
    double GetStationEnd(int index)
    {
        return m_stations[index].End;
    }

    /** Set m_stations[index].End
     * \param val New value to set
     */
    void SetStationEnd(int index, double val)
    {
        m_stations[index].End = val;
    }

    /** Access m_stations[index].Coord
     * \return The current value of m_stations[index].Coord
     */
    Coo GetStationCoord(int index)
    {
        return m_stations[index].Coord;
    }

    /** Set m_stations[index].Coord
     * \param val New value to set
     */
    void SetStationCoord(int index, const Coo &val)
    {
        m_stations[index].Coord = val;
    }

    /** Access m_stations[index].Height
     * \return The current value of m_stations[index].Height
     */
    float GetStationHeight(int index)
    {
        return m_stations[index].Height;
    }

    /** Set m_stations[index].Height
     * \param val New value to set
     */
    void SetStationHeight(int index, float val)
    {
        m_stations[index].Height = val;
    }
    
    int GetStationsNb()
    {
        return int(m_stations.size());
    }

protected:

private:
    wxString m_catalogFilePath; //!< Member variable "m_catalogFilePath"
    wxString m_setId; //!< Member variable "m_setId"
    wxString m_name; //!< Member variable "m_name"
    wxString m_description; //!< Member variable "m_description"
    double m_start; //!< Member variable "m_start"
    double m_end; //!< Member variable "m_end"
    float m_timeZoneHours; //!< Member variable "m_timezone" in hours
    double m_timeStepHours; //!< Member variable "m_timeStep" in hours
    double m_firstTimeStepHour; //!< Member variable "m_firstTimeStep" in hours
    wxString m_dataPath; //!< Member variable "m_filePath"
    VectorString m_nan; //!< Member variable "m_nan"
    wxString m_coordSys; //!< Member variable "m_coordSys"
    DataParameter m_parameter; //!< Member variable "m_parameter"
    DataUnit m_unit; //!< Member variable "m_unit"
    std::vector < DataStruct > m_stations; //!< Member variable "m_stations"
    DataTemporalResolution m_temporalResolution;
    DataSpatialAggregation m_spatialAggregation;

};

#endif // ASCATALOGPREDICTAND_H
