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

#ifndef AS_PREDICTAND_H
#define AS_PREDICTAND_H

#include "asFileDat.h"
#include "asHeadersBase.h"

class asCatalogPredictands;

class asFileNetcdf;

class asPredictand : public wxObject {
  public:
    enum Parameter {
        Precipitation,
        AirTemperature,
        Lightning,
        Wind
    };

    enum Unit {
        nb,
        mm,
        m,
        in,
        percent,
        degC,
        degK
    };

    enum TemporalResolution {
        Daily,
        SixHourly,
        Hourly,
        OneHourlyMTW,
        ThreeHourlyMTW,
        SixHourlyMTW,
        TwelveHourlyMTW
    };

    enum SpatialAggregation {
        Station,
        Groupment,
        Catchment,
        Region
    };

    asPredictand(Parameter parameter, TemporalResolution temporalResolution, SpatialAggregation spatialAggregation);

    ~asPredictand() override = default;

    static Parameter StringToParameterEnum(const wxString& parameterStr);

    static wxString ParameterEnumToString(Parameter parameter);

    static Unit StringToUnitEnum(const wxString& unitStr);

    static TemporalResolution StringToTemporalResolutionEnum(const wxString& temporalResolution);

    static wxString TemporalResolutionEnumToString(TemporalResolution temporalResolution);

    static SpatialAggregation StringToSpatialAggregationEnum(const wxString& spatialAggregation);

    static wxString SpatialAggregationEnumToString(SpatialAggregation spatialAggregation);

    static asPredictand* GetInstance(const wxString& parameterStr, const wxString& temporalResolutionStr,
                                     const wxString& spatialAggregationStr);

    static asPredictand* GetInstance(Parameter parameter, TemporalResolution temporalResolution,
                                     SpatialAggregation spatialAggregation);

    static asPredictand* GetInstance(const wxString& filePath);

    virtual bool Load(const wxString& filePath) = 0;

    virtual bool Save(const wxString& filePath) const = 0;

    virtual bool BuildPredictandDB(const wxString& catalogFilePath, const wxString& dataDir, const wxString& patternDir,
                                   const wxString& destinationDir) = 0;

    virtual bool HasReferenceAxis() const {
        return false;
    }

    virtual a1f GetReferenceAxis() const {
        a1f nodata(1);
        nodata[0] = NAN;
        return nodata;
    }

    virtual float GetReferenceValue(int /*iStat*/, double /*duration*/, float /*reference*/) const {
        return NAN;
    }

    virtual a2f GetReferenceValuesArray() const {
        a1f nodata(1);
        nodata[0] = NAN;
        return nodata;
    }

    wxString GetDatasetId() const {
        return _datasetId;
    }

    wxString GetCoordSys() const {
        return _coordSys;
    }

    Parameter GetDataParameter() const {
        return _parameter;
    }

    TemporalResolution GetDataTemporalResolution() const {
        return _temporalResolution;
    }

    SpatialAggregation GetDataSpatialAggregation() const {
        return _spatialAggregation;
    }

    void SetHasReferenceValues(bool val) {
        _hasReferenceValues = val;
        _hasNormalizedData = val;
    }

    int GetStationsNb() const {
        return _stationsNb;
    }

    int GetTimeLength() const {
        return _timeLength;
    }

    vwxs& GetStationNamesArray() {
        return _stationNames;
    }

    vwxs& GetStationOfficialIdsArray() {
        return _stationOfficialIds;
    }

    a1f& GetStationHeightsArray() {
        return _stationHeights;
    }

    a1i& GetStationsIdArray() {
        return _stationIds;
    }

    a1d& GetStationXCoordsArray() {
        return _stationXCoords;
    }

    a1d& GetStationYCoordsArray() {
        return _stationYCoords;
    }

    a1f GetDataRawStation(int predictandStationId) const {
        int indexStation = GetStationIndex(predictandStationId);
        return _dataRaw.col(indexStation);
    }

    a1f GetDataNormalizedStation(int predictandStationId) const {
        int indexStation = GetStationIndex(predictandStationId);
        if (_hasNormalizedData) {
            return _dataNormalized.col(indexStation);
        } else {
            return _dataRaw.col(indexStation);
        }
    }

    a1d& GetTime() {
        return _time;
    }

    int GetStationIndex(int stationId) const;

  protected:
    // Single value
    float _fileVersion;
    Parameter _parameter;
    TemporalResolution _temporalResolution;
    SpatialAggregation _spatialAggregation;
    wxString _datasetId;
    wxString _coordSys;
    double _timeStepDays;
    int _timeLength;
    int _stationsNb;
    double _dateProcessed;
    double _dateStart;
    double _dateEnd;
    bool _hasNormalizedData;
    bool _hasReferenceValues;
    // Matrix data
    a2f _dataRaw;
    a2f _dataNormalized;
    // Vector (dim = time)
    a1d _time;
    // Vector (dim = stations)
    vwxs _stationNames;
    vwxs _stationOfficialIds;
    a1i _stationIds;
    a1f _stationHeights;
    a1d _stationXCoords;
    a1d _stationYCoords;
    a1d _stationStarts;
    a1d _stationEnds;

    wxString GetDBFilePathSaving(const wxString& destinationDir) const;

    bool InitMembers(const wxString& catalogFilePath);

    bool InitBaseContainers();

    bool LoadCommonData(asFileNetcdf& ncFile);

    void SetCommonDefinitions(asFileNetcdf& ncFile) const;

    bool SaveCommonData(asFileNetcdf& ncFile) const;

    bool ParseData(const wxString& catalogFile, const wxString& directory = wxEmptyString,
                   const wxString& patternDir = wxEmptyString);

    a2f GetAnnualMax(double timeStepDays = 1, int nansNbMax = 10) const;

    bool SetStationProperties(asCatalogPredictands& currentData, size_t stationIndex);

    bool GetFileContent(asCatalogPredictands& currentData, int stationIndex, const wxString& directory = wxEmptyString,
                        const wxString& patternDir = wxEmptyString);

  private:
    float ParseAndCheckDataValue(asCatalogPredictands& currentData, wxString& dataStr) const;

    bool ParseConstantWidthContent(int stationIndex, const asFileDat::Pattern& pattern, const wxString& lineContent,
                                   asCatalogPredictands& currentData, int& timeIndex);

    bool ParseTabsDelimitedContent(int stationIndex, const asFileDat::Pattern& pattern, const wxString& lineContent,
                                   asCatalogPredictands& currentData, int& timeIndex);
};

#endif
