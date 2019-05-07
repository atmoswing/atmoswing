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

#ifndef AS_CATALOG_PREDICTAND_H
#define AS_CATALOG_PREDICTAND_H

#include <asIncludes.h>
#include "asPredictand.h"


class asCatalogPredictands
        : public wxObject
{
public:
    //!< Structure for data information
    struct DataStruct
    {
        int id;
        wxString officialId;
        wxString name;
        wxString fileName;
        wxString filePattern;
        double startDate;
        double endDate;
        Coo coord;
        float height;
    };

    explicit asCatalogPredictands(const wxString &filePath);

    ~asCatalogPredictands() override = default;

    bool Load();

    wxString GetSetId() const
    {
        return m_setId;
    }

    wxString GetName() const
    {
        return m_name;
    }

    wxString GetDescription() const
    {
        return m_description;
    }

    double GetStart() const
    {
        return m_start;
    }

    double GetEnd() const
    {
        return m_end;
    }

    float GetTimeZoneHours() const
    {
        return m_timeZoneHours;
    }

    double GetTimeStepHours() const
    {
        return m_timeStepHours;
    }

    double GetTimeStepDays() const
    {
        return m_timeStepHours / 24;
    }

    double GetFirstTimeStepHours() const
    {
        return m_firstTimeStepHour;
    }

    wxString GetDataPath() const
    {
        return m_dataPath;
    }

    vwxs GetNan() const
    {
        return m_nan;
    }

    wxString GetCoordSys() const
    {
        return m_coordSys;
    }

    asPredictand::Parameter GetParameter() const
    {
        return m_parameter;
    }

    asPredictand::Unit GetUnit() const
    {
        return m_unit;
    }

    int GetStationId(int index) const
    {
        return m_stations[index].id;
    }

    wxString GetStationOfficialId(int index) const
    {
        return m_stations[index].officialId;
    }

    wxString GetStationName(int index) const
    {
        return m_stations[index].name;
    }

    wxString GetStationFilename(int index) const
    {
        return m_stations[index].fileName;
    }

    wxString GetStationFilepattern(int index) const
    {
        return m_stations[index].filePattern;
    }

    double GetStationStart(int index) const
    {
        return m_stations[index].startDate;
    }

    double GetStationEnd(int index) const
    {
        return m_stations[index].endDate;
    }

    Coo GetStationCoord(int index) const
    {
        return m_stations[index].coord;
    }

    float GetStationHeight(int index) const
    {
        return m_stations[index].height;
    }

    int GetStationsNb() const
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
    vwxs m_nan;
    wxString m_coordSys;
    asPredictand::Parameter m_parameter;
    asPredictand::Unit m_unit;
    std::vector<DataStruct> m_stations;
    asPredictand::TemporalResolution m_temporalResolution;
    asPredictand::SpatialAggregation m_spatialAggregation;

};

#endif
