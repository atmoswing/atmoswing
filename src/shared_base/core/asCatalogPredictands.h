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
 
#ifndef ASCATALOGPREDICTAND_H
#define ASCATALOGPREDICTAND_H

#include <asIncludes.h>
#include <asCatalog.h>


class asCatalogPredictands: public asCatalog
{
public:

    //!< Structure for data information
    struct DataStruct
    {
        int Id;
        wxString LocalId;
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

    bool Load(int StationId = 0);

    /** Method that get data information from file
    * \param DataSetId The data Set ID
    */
    bool LoadDatasetProp();

    /** Method that get data information from file
    * \param DataSetId The data Set ID
    * \param DataId The station ID
    */
    bool LoadDataProp(int StationId);

    /** Access m_Station
     * \return The current value of m_Station
     */
    DataStruct GetStationInfo()
    {
        return m_Station;
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

    /** Access m_Station.Id
     * \return The current value of m_Station.Id
     */
    int GetStationId()
    {
        return m_Station.Id;
    }

    /** Set m_Station.Id
     * \param val New value to set
     */
    void SetStationId(int val)
    {
        m_Station.Id = val;
    }

    /** Access m_Station.LocalId
     * \return The current value of m_Station.LocalId
     */
    wxString GetStationLocalId()
    {
        return m_Station.LocalId;
    }

    /** Set m_Station.LocalId
     * \param val New value to set
     */
    void SetStationLocalId(const wxString &val)
    {
        m_Station.LocalId = val;
    }

    /** Access m_Station.Name
     * \return The current value of m_Station.Name
     */
    wxString GetStationName()
    {
        return m_Station.Name;
    }

    /** Set m_Station.Name
     * \param val New value to set
     */
    void SetStationName(const wxString &val)
    {
        m_Station.Name = val;
    }

    /** Access m_Station.Filename
     * \return The current value of m_Station.Filename
     */
    wxString GetStationFilename()
    {
        return m_Station.Filename;
    }

    /** Set m_Station.Filename
     * \param val New value to set
     */
    void SetStationFilename(const wxString &val)
    {
        m_Station.Filename = val;
    }

    /** Access m_Station.Filepattern
     * \return The current value of m_Station.Filepattern
     */
    wxString GetStationFilepattern()
    {
        return m_Station.Filepattern;
    }

    /** Set m_Station.Filepattern
     * \param val New value to set
     */
    void SetStationFilepattern(const wxString &val)
    {
        m_Station.Filepattern = val;
    }

    /** Access m_Station.Start
     * \return The current value of m_Station.Start
     */
    double GetStationStart()
    {
        return m_Station.Start;
    }

    /** Set m_Station.Start
     * \param val New value to set
     */
    void SetStationStart(double val)
    {
        m_Station.Start = val;
    }

    /** Access m_Station.End
     * \return The current value of m_Station.End
     */
    double GetStationEnd()
    {
        return m_Station.End;
    }

    /** Set m_Station.End
     * \param val New value to set
     */
    void SetStationEnd(double val)
    {
        m_Station.End = val;
    }

    /** Access m_Station.Coord
     * \return The current value of m_Station.Coord
     */
    Coo GetStationCoord()
    {
        return m_Station.Coord;
    }

    /** Set m_Station.Coord
     * \param val New value to set
     */
    void SetStationCoord(const Coo &val)
    {
        m_Station.Coord = val;
    }

    /** Access m_Station.Height
     * \return The current value of m_Station.Height
     */
    float GetStationHeight()
    {
        return m_Station.Height;
    }

    /** Set m_Station.Height
     * \param val New value to set
     */
    void SetStationHeight(float val)
    {
        m_Station.Height = val;
    }


protected:
    DataParameter m_Parameter; //!< Member variable "m_Parameter"
    DataUnit m_Unit; //!< Member variable "m_Unit"
    DataStruct m_Station; //!< Member variable "m_Station"
	DataTemporalResolution m_TemporalResolution;
	DataSpatialAggregation m_SpatialAggregation;


private:


};

#endif // ASCATALOGPREDICTAND_H
