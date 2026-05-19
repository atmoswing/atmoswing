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

#include "asIncludes.h"
#include "asPredictand.h"

class asCatalogPredictands : public wxObject {
  public:
    //!< Structure for data information
    struct DataStruct {
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

    explicit asCatalogPredictands(const wxString& filePath);

    ~asCatalogPredictands() override = default;

    bool Load();

    wxString GetSetId() const {
        return _setId;
    }

    wxString GetName() const {
        return _name;
    }

    wxString GetDescription() const {
        return _description;
    }

    double GetStart() const {
        return _start;
    }

    double GetEnd() const {
        return _end;
    }

    float GetTimeZoneHours() const {
        return _timeZoneHours;
    }

    double GetTimeStepHours() const {
        return _timeStepHours;
    }

    double GetTimeStepDays() const {
        return _timeStepHours / 24;
    }

    double GetFirstTimeStepHours() const {
        return _firstTimeStepHour;
    }

    wxString GetDataPath() const {
        return _dataPath;
    }

    vwxs GetNan() const {
        return _nan;
    }

    wxString GetCoordSys() const {
        return _coordSys;
    }

    asPredictand::Parameter GetParameter() const {
        return _parameter;
    }

    asPredictand::Unit GetUnit() const {
        return _unit;
    }

    int GetStationId(int index) const {
        return _stations[index].id;
    }

    wxString GetStationOfficialId(int index) const {
        return _stations[index].officialId;
    }

    wxString GetStationName(int index) const {
        return _stations[index].name;
    }

    wxString GetStationFilename(int index) const {
        return _stations[index].fileName;
    }

    wxString GetStationFilepattern(int index) const {
        return _stations[index].filePattern;
    }

    double GetStationStart(int index) const {
        return _stations[index].startDate;
    }

    double GetStationEnd(int index) const {
        return _stations[index].endDate;
    }

    Coo GetStationCoord(int index) const {
        return _stations[index].coord;
    }

    float GetStationHeight(int index) const {
        return _stations[index].height;
    }

    int GetStationsNb() const {
        return int(_stations.size());
    }

  protected:
  private:
    wxString _catalogFilePath;
    wxString _setId;
    wxString _name;
    wxString _description;
    double _start;
    double _end;
    float _timeZoneHours;
    double _timeStepHours;
    double _firstTimeStepHour;
    wxString _dataPath;
    vwxs _nan;
    wxString _coordSys;
    asPredictand::Parameter _parameter;
    asPredictand::Unit _unit;
    vector<DataStruct> _stations;
    asPredictand::TemporalResolution _temporalResolution;
    asPredictand::SpatialAggregation _spatialAggregation;
};

#endif
