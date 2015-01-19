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

    bool Load();

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

    /** Access m_Stations[index].LocalId
     * \return The current value of m_Stations[index].LocalId
     */
    wxString GetStationLocalId(int index)
    {
        return m_Stations[index].LocalId;
    }

    /** Set m_Stations[index].LocalId
     * \param val New value to set
     */
    void SetStationLocalId(int index, const wxString &val)
    {
        m_Stations[index].LocalId = val;
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
    DataParameter m_Parameter; //!< Member variable "m_Parameter"
    DataUnit m_Unit; //!< Member variable "m_Unit"
    std::vector < DataStruct > m_Stations; //!< Member variable "m_Stations"
    DataTemporalResolution m_TemporalResolution;
    DataSpatialAggregation m_SpatialAggregation;


private:


};

#endif // ASCATALOGPREDICTAND_H
