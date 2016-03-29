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


class asCatalogPredictands
        : public wxObject
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

    asCatalogPredictands(const wxString &filePath);

    virtual ~asCatalogPredictands();

    bool Load();

    wxString GetCatalogFilePath()
    {
        return m_catalogFilePath;
    }

    void SetCatalogFilePath(const wxString &val)
    {
        m_catalogFilePath = val;
    }

    wxString GetSetId()
    {
        return m_setId;
    }

    void SetSetId(const wxString &val)
    {
        m_setId = val;
    }

    wxString GetName()
    {
        return m_name;
    }

    void SetName(const wxString &val)
    {
        m_name = val;
    }

    wxString GetDescription()
    {
        return m_description;
    }

    void SetDescription(const wxString &val)
    {
        m_description = val;
    }

    double GetStart()
    {
        return m_start;
    }

    void SetStart(double val)
    {
        m_start = val;
    }

    double GetEnd()
    {
        return m_end;
    }

    void SetEnd(double val)
    {
        m_end = val;
    }

    float GetTimeZoneHours()
    {
        return m_timeZoneHours;
    }

    void SetTimeZone(float val)
    {
        m_timeZoneHours = val;
    }

    double GetTimeStepMinutes()
    {
        return m_timeStepHours * 60;
    }

    void SetTimeStepMinutes(double val)
    {
        m_timeStepHours = val / 60;
    }

    double GetTimeStepHours()
    {
        return m_timeStepHours;
    }

    void SetTimeStepHours(double val)
    {
        m_timeStepHours = val;
    }

    double GetTimeStepDays()
    {
        return m_timeStepHours / 24;
    }

    void SetTimeStepDays(double val)
    {
        m_timeStepHours = val * 24;
    }

    double GetFirstTimeStepMinutes()
    {
        return m_firstTimeStepHour * 60;
    }

    void SetFirstTimeStepMinutes(double val)
    {
        m_firstTimeStepHour = val / 60;
    }

    double GetFirstTimeStepHours()
    {
        return m_firstTimeStepHour;
    }

    void SetFirstTimeStepHours(double val)
    {
        m_firstTimeStepHour = val;
    }

    wxString GetDataPath()
    {
        return m_dataPath;
    }

    void SetDataPath(const wxString &val)
    {
        m_dataPath = val;
    }

    VectorString GetNan()
    {
        return m_nan;
    }

    void SetNan(const VectorString &val)
    {
        m_nan = val;
    }

    wxString GetCoordSys()
    {
        return m_coordSys;
    }

    void SetCoordSys(wxString val)
    {
        m_coordSys = val;
    }

    DataStruct GetStationInfo(int index)
    {
        return m_stations[index];
    }

    DataParameter GetParameter()
    {
        return m_parameter;
    }

    void SetParameter(DataParameter val)
    {
        m_parameter = val;
    }

    DataUnit GetUnit()
    {
        return m_unit;
    }

    void SetUnit(DataUnit val)
    {
        m_unit = val;
    }

    int GetStationId(int index)
    {
        return m_stations[index].Id;
    }

    void SetStationId(int index, int val)
    {
        m_stations[index].Id = val;
    }

    wxString GetStationOfficialId(int index)
    {
        return m_stations[index].OfficialId;
    }

    void SetStationOfficialId(int index, const wxString &val)
    {
        m_stations[index].OfficialId = val;
    }

    wxString GetStationName(int index)
    {
        return m_stations[index].Name;
    }

    void SetStationName(int index, const wxString &val)
    {
        m_stations[index].Name = val;
    }

    wxString GetStationFilename(int index)
    {
        return m_stations[index].Filename;
    }

    void SetStationFilename(int index, const wxString &val)
    {
        m_stations[index].Filename = val;
    }

    wxString GetStationFilepattern(int index)
    {
        return m_stations[index].Filepattern;
    }

    void SetStationFilepattern(int index, const wxString &val)
    {
        m_stations[index].Filepattern = val;
    }

    double GetStationStart(int index)
    {
        return m_stations[index].Start;
    }

    void SetStationStart(int index, double val)
    {
        m_stations[index].Start = val;
    }

    double GetStationEnd(int index)
    {
        return m_stations[index].End;
    }

    void SetStationEnd(int index, double val)
    {
        m_stations[index].End = val;
    }

    Coo GetStationCoord(int index)
    {
        return m_stations[index].Coord;
    }

    void SetStationCoord(int index, const Coo &val)
    {
        m_stations[index].Coord = val;
    }

    float GetStationHeight(int index)
    {
        return m_stations[index].Height;
    }

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
    wxString m_catalogFilePath;
    wxString m_setId;
    wxString m_name;
    wxString m_description;
    double m_start;
    double m_end;
    float m_timeZoneHours;
    double m_timeStepHours;
    double m_firstTimeStepHour;
    wxString m_dataPath;
    VectorString m_nan;
    wxString m_coordSys;
    DataParameter m_parameter;
    DataUnit m_unit;
    std::vector<DataStruct> m_stations;
    DataTemporalResolution m_temporalResolution;
    DataSpatialAggregation m_spatialAggregation;

};

#endif // ASCATALOGPREDICTAND_H
