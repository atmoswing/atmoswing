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

    /** Access m_CatalogFilePath
     * \return The current value of m_CatalogFilePath
     */
    wxString GetCatalogFilePath()
    {
        return m_CatalogFilePath;
    }

    /** Set m_CatalogFilePath
     * \param val New value to set
     */
    void SetCatalogFilePath(const wxString &val)
    {
        m_CatalogFilePath = val;
    }

    /** Access m_SetId
     * \return The current value of m_SetId
     */
    wxString GetSetId()
    {
        return m_SetId;
    }

    /** Set m_SetId
     * \param val New value to set
     */
    void SetSetId(const wxString &val)
    {
        m_SetId = val;
    }

    /** Access m_Name
     * \return The current value of m_Name
     */
    wxString GetName()
    {
        return m_Name;
    }

    /** Set m_Name
     * \param val New value to set
     */
    void SetName(const wxString &val)
    {
        m_Name = val;
    }

    /** Access m_Description
     * \return The current value of m_Description
     */
    wxString GetDescription()
    {
        return m_Description;
    }

    /** Set m_Description
     * \param val New value to set
     */
    void SetDescription(const wxString &val)
    {
        m_Description = val;
    }

    /** Access m_Start
     * \return The current value of m_Start
     */
    double GetStart()
    {
        return m_Start;
    }

    /** Set m_Start
     * \param val New value to set
     */
    void SetStart(double val)
    {
        m_Start = val;
    }

    /** Access m_End
     * \return The current value of m_End
     */
    double GetEnd()
    {
        return m_End;
    }

    /** Set m_End
     * \param val New value to set
     */
    void SetEnd(double val)
    {
        m_End = val;
    }

    /** Access m_TimeZoneHours
     * \return The current value of m_TimeZoneHours
     */
    float GetTimeZoneHours()
    {
        return m_TimeZoneHours;
    }

    /** Set m_TimeZoneHours
     * \param val New value to set
     */
    void SetTimeZone(float val)
    {
        m_TimeZoneHours = val;
    }

    /** Access m_TimeStep in minutes
     * \return The current value of m_TimeStep in minutes
     */
    double GetTimeStepMinutes()
    {
        return m_TimeStepHours*60;
    }

    /** Set m_TimeStep in minutes
     * \param val New value to set in minutes
     */
    void SetTimeStepMinutes(double val)
    {
        m_TimeStepHours = val/60;
    }

    /** Access m_TimeStep in hours
     * \return The current value of m_TimeStep in hours
     */
    double GetTimeStepHours()
    {
        return m_TimeStepHours;
    }

    /** Set m_TimeStep in hours
     * \param val New value to set in hours
     */
    void SetTimeStepHours(double val)
    {
        m_TimeStepHours = val;
    }

    /** Access m_TimeStep in days
     * \return The current value of m_TimeStep in days
     */
    double GetTimeStepDays()
    {
        return m_TimeStepHours/24;
    }

    /** Set m_TimeStep in days
     * \param val New value to set in days
     */
    void SetTimeStepDays(double val)
    {
        m_TimeStepHours = val*24;
    }

    /** Access m_FirstTimeStepHour in minutes
     * \return The current value of m_FirstTimeStep in minutes
     */
    double GetFirstTimeStepMinutes()
    {
        return m_FirstTimeStepHour*60;
    }

    /** Set m_FirstTimeStepHour in minutes
     * \param val New value to set in minutes
     */
    void SetFirstTimeStepMinutes(double val)
    {
        m_FirstTimeStepHour = val/60;
    }

    /** Access m_FirstTimeStepHour in hours
     * \return The current value of m_FirstTimeStep in hours
     */
    double GetFirstTimeStepHours()
    {
        return m_FirstTimeStepHour;
    }

    /** Set m_FirstTimeStepHour in hours
     * \param val New value to set in hours
     */
    void SetFirstTimeStepHours(double val)
    {
        m_FirstTimeStepHour = val;
    }

    /** Access m_DataPath
     * \return The current value of m_DataPath
     */
    wxString GetDataPath()
    {
        return m_DataPath;
    }

    /** Set m_DataPath
     * \param val New value to set
     */
    void SetDataPath(const wxString &val)
    {
        m_DataPath = val;
    }

    /** Access m_Nan
     * \return The current value of m_Nan
     */
    VectorDouble GetNan()
    {
        return m_Nan;
    }

    /** Set m_Nan
     * \param val New value to set
     */
    void SetNan(const VectorDouble &val)
    {
        m_Nan = val;
    }

    /** Access m_CoordSys
     * \return The current value of m_CoordSys
     */
    wxString GetCoordSys()
    {
        return m_CoordSys;
    }

    /** Set m_CoordSys
     * \param val New value to set
     */
    void SetCoordSys(wxString val)
    {
        m_CoordSys = val;
    }

    /** Access m_Stations
     * \return The current value of m_Stations
     */
    DataStruct GetStationInfo(int index)
    {
        return m_Stations[index];
    }

    /** Access m_Parameter
     * \return The current value of m_Parameter
     */
    DataParameter GetParameter()
    {
        return m_Parameter;
    }

    /** Set m_Parameter
     * \param val New value to set
     */
    void SetParameter(DataParameter val)
    {
        m_Parameter = val;
    }

    /** Access m_Unit
     * \return The current value of m_Unit
     */
    DataUnit GetUnit()
    {
        return m_Unit;
    }

    /** Set m_Unit
     * \param val New value to set
     */
    void SetUnit(DataUnit val)
    {
        m_Unit = val;
    }

    /** Access m_Stations[index].Id
     * \return The current value of m_Stations[index].Id
     */
    int GetStationId(int index)
    {
        return m_Stations[index].Id;
    }

    /** Set m_Stations[index].Id
     * \param val New value to set
     */
    void SetStationId(int index, int val)
    {
        m_Stations[index].Id = val;
    }

    /** Access m_Stations[index].OfficialId
     * \return The current value of m_Stations[index].OfficialId
     */
    wxString GetStationOfficialId(int index)
    {
        return m_Stations[index].OfficialId;
    }

    /** Set m_Stations[index].OfficialId
     * \param val New value to set
     */
    void SetStationOfficialId(int index, const wxString &val)
    {
        m_Stations[index].OfficialId = val;
    }

    /** Access m_Stations[index].Name
     * \return The current value of m_Stations[index].Name
     */
    wxString GetStationName(int index)
    {
        return m_Stations[index].Name;
    }

    /** Set m_Stations[index].Name
     * \param val New value to set
     */
    void SetStationName(int index, const wxString &val)
    {
        m_Stations[index].Name = val;
    }

    /** Access m_Stations[index].Filename
     * \return The current value of m_Stations[index].Filename
     */
    wxString GetStationFilename(int index)
    {
        return m_Stations[index].Filename;
    }

    /** Set m_Stations[index].Filename
     * \param val New value to set
     */
    void SetStationFilename(int index, const wxString &val)
    {
        m_Stations[index].Filename = val;
    }

    /** Access m_Stations[index].Filepattern
     * \return The current value of m_Stations[index].Filepattern
     */
    wxString GetStationFilepattern(int index)
    {
        return m_Stations[index].Filepattern;
    }

    /** Set m_Stations[index].Filepattern
     * \param val New value to set
     */
    void SetStationFilepattern(int index, const wxString &val)
    {
        m_Stations[index].Filepattern = val;
    }

    /** Access m_Stations[index].Start
     * \return The current value of m_Stations[index].Start
     */
    double GetStationStart(int index)
    {
        return m_Stations[index].Start;
    }

    /** Set m_Stations[index].Start
     * \param val New value to set
     */
    void SetStationStart(int index, double val)
    {
        m_Stations[index].Start = val;
    }

    /** Access m_Stations[index].End
     * \return The current value of m_Stations[index].End
     */
    double GetStationEnd(int index)
    {
        return m_Stations[index].End;
    }

    /** Set m_Stations[index].End
     * \param val New value to set
     */
    void SetStationEnd(int index, double val)
    {
        m_Stations[index].End = val;
    }

    /** Access m_Stations[index].Coord
     * \return The current value of m_Stations[index].Coord
     */
    Coo GetStationCoord(int index)
    {
        return m_Stations[index].Coord;
    }

    /** Set m_Stations[index].Coord
     * \param val New value to set
     */
    void SetStationCoord(int index, const Coo &val)
    {
        m_Stations[index].Coord = val;
    }

    /** Access m_Stations[index].Height
     * \return The current value of m_Stations[index].Height
     */
    float GetStationHeight(int index)
    {
        return m_Stations[index].Height;
    }

    /** Set m_Stations[index].Height
     * \param val New value to set
     */
    void SetStationHeight(int index, float val)
    {
        m_Stations[index].Height = val;
    }
    
    int GetStationsNb()
    {
        return int(m_Stations.size());
    }

protected:

private:
    wxString m_CatalogFilePath; //!< Member variable "m_CatalogFilePath"
    wxString m_SetId; //!< Member variable "m_SetId"
    wxString m_Name; //!< Member variable "m_Name"
    wxString m_Description; //!< Member variable "m_Description"
    double m_Start; //!< Member variable "m_Start"
    double m_End; //!< Member variable "m_End"
    float m_TimeZoneHours; //!< Member variable "m_Timezone" in hours
    double m_TimeStepHours; //!< Member variable "m_TimeStep" in hours
    double m_FirstTimeStepHour; //!< Member variable "m_FirstTimeStep" in hours
    wxString m_DataPath; //!< Member variable "m_FilePath"
    VectorDouble m_Nan; //!< Member variable "m_Nan"
    wxString m_CoordSys; //!< Member variable "m_CoordSys"
    DataParameter m_Parameter; //!< Member variable "m_Parameter"
    DataUnit m_Unit; //!< Member variable "m_Unit"
    std::vector < DataStruct > m_Stations; //!< Member variable "m_Stations"
    DataTemporalResolution m_TemporalResolution;
    DataSpatialAggregation m_SpatialAggregation;

};

#endif // ASCATALOGPREDICTAND_H
