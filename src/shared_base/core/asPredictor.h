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

#ifndef AS_PREDICTOR_H
#define AS_PREDICTOR_H

#include <wx/filename.h>  // wxFileName::GetPathSeparator()

#include "asFileGrib.h"
#include "asFileNetcdf.h"
#include "asHeadersBase.h"

class asTimeArray;

class asGeo;

class asAreaGrid;

class asPredictor : public wxObject {
  public:
    enum Parameter {
        ParameterUndefined,
        AirTemperature,
        PotentialTemperature,
        Geopotential,
        GeopotentialHeight,
        GeopotentialHeightAnomaly,
        PrecipitableWater,
        TotalColumnWater,
        Precipitation,
        PrecipitationRate,
        RelativeHumidity,
        SpecificHumidity,
        VerticalVelocity,
        Wind,
        Uwind,
        Vwind,
        PotentialVorticity,
        PotentialEvaporation,
        SurfaceLiftedIndex,
        Pressure,
        SoilMoisture,
        SoilTemperature,
        SnowWaterEquivalent,
        CloudCover,
        CloudWater,
        Radiation,
        MomentumFlux,
        GravityWaveStress,
        SeaSurfaceTemperature,
        SeaSurfaceTemperatureAnomaly,
        CAPE,
        CIN,
        LapseRate,
        StreamFunction,
        AbsoluteVorticity,
        VelocityPotential,
        WindShear,
        Divergence,
        MontgomeryPotential,
        Vorticity,
        DewpointTemperature,
        MoistureFlux,
        MaximumBuoyancy,
        Other
    };

    enum Unit {
        UnitUndefined,
        unitary,
        nb,
        mm,
        m,
        gpm,
        km,
        percent,
        fraction,
        degC,
        degK,
        Pa,
        Pa_s,
        g_kg,
        kg_kg,
        _s,
        W_m2,
        kg_m2,
        kg_m2_s,
        N_m2,
        m2_s2,
        m2_s,
        degKm2_kg_s,
        mm_d,
        J_kg,
        degK_m,
        per_s,
        J_m2
    };

    explicit asPredictor(const wxString& dataId);

    ~asPredictor() override = default;

    static asPredictor* GetInstance(const wxString& datasetId, const wxString& dataId,
                                    const wxString& directory = wxEmptyString);

    virtual bool Init();

    void CheckLevelTypeIsDefined();

    bool CheckFilesPresence();

    bool Load(asAreaGrid* desiredArea, asTimeArray& timeArray, float level);

    bool Load(asAreaGrid& desiredArea, asTimeArray& timeArray, float level);

    bool Load(asAreaGrid& desiredArea, double date, float level);

    bool Load(asAreaGrid* desiredArea, double date, float level);

    bool ClipToArea(asAreaGrid* desiredArea);

    bool StandardizeData(double mean = NAN, double sd = NAN);

    bool Inline();

    void DumpData();

    bool SaveDumpFile();

    bool LoadDumpedData();

    bool DumpFileExists() const;

    bool SetData(vva2f& val);

    float GetMinValue() const;

    float GetMaxValue() const;

    bool HasNaN() const;

    vva2f& GetData() {
        wxASSERT((int)_data.size() >= (int)_time.size());
        wxASSERT(_data.size() >= 1);
        wxASSERT(_data[0].size() >= 1);
        wxASSERT(_data[0][0].cols() > 0);
        wxASSERT(_data[0][0].rows() > 0);

        return _data;
    }

    a2f* GetData(int iTime, int iMem) {
        wxASSERT((int)_data.size() >= (int)_time.size());
        wxASSERT(_data.size() >= iTime);
        wxASSERT(_data[iTime].size() >= iMem);
        wxASSERT(_data[iTime][iMem].cols() > 0);
        wxASSERT(_data[iTime][iMem].rows() > 0);

        return &_data[iTime][iMem];
    }

    bool HasSingleArray() {
        return (_data.size() == 1) && (_data[0].size() == 1);
    }

    wxString GetDataId() const {
        return _dataId;
    }

    void SetDatasetId(const wxString& datasetId) {
        _datasetId = datasetId;
    }

    wxString GetProduct() const {
        return _product;
    }

    Parameter GetParameter() const {
        return _parameter;
    }

    void SetDirectoryPath(wxString directoryPath) {
        if (directoryPath.Last() != '/' && directoryPath.Last() != '\\') {
            directoryPath.Append(wxFileName::GetPathSeparator());
        }

        _directoryPath = directoryPath;
    }

    wxString GetDirectoryPath() const {
        wxString directoryPath = _directoryPath;
        if (!directoryPath.IsEmpty()) {
            if (directoryPath.Last() != '/' && directoryPath.Last() != '\\') {
                directoryPath.Append(wxFileName::GetPathSeparator());
            }
        }

        return directoryPath;
    }

    wxString GetFullDirectoryPath() const {
        wxString directoryPath = _directoryPath;
        if (!directoryPath.IsEmpty()) {
            if (directoryPath.Last() != '/' && directoryPath.Last() != '\\') {
                directoryPath.Append(wxFileName::GetPathSeparator());
            }

            directoryPath += _product;
            if (directoryPath.Last() != '/' && directoryPath.Last() != '\\') {
                directoryPath.Append(wxFileName::GetPathSeparator());
            }
        }

        return directoryPath;
    }

    int GetTimeSize() const {
        return (int)_time.size();
    }

    int GetMembersNb() const {
        wxASSERT(!_data.empty());

        return (int)_data[0].size();
    }

    int GetLatPtsnb() const {
        return _latPtsnb;
    }

    int GetLonPtsnb() const {
        return _lonPtsnb;
    }

    void SetStandardized(bool val = true) {
        _standardized = val;
    }

    static bool IsLatLon(const wxString& datasetId);

    bool IsLatLon() const {
        return _isLatLon;
    }

    bool IsPreprocessed() const {
        return _isPreprocessed;
    }

    void SetIsPreprocessed(bool val) {
        _isPreprocessed = val;
    }

    bool IsEnsemble() const {
        return _isEnsemble;
    }

    bool CanBeClipped() const {
        return _canBeClipped;
    }

    void SetCanBeClipped(bool val) {
        _canBeClipped = val;
    }

    wxString GetPreprocessMethod() const {
        return _preprocessMethod;
    }

    void SetPreprocessMethod(const wxString& val) {
        _preprocessMethod = val;
    }

    void SelectFirstMember() {
        if (!_isEnsemble) {
            throw runtime_error(_("Dataset is not an ensemble, you cannot select a member."));
        }

        _fInd.memberStart = 0;
        _fInd.memberCount = 1;
        _membersNb = 1;
    }

    void SelectMember(int memberNum) {
        if (!_isEnsemble) {
            throw runtime_error(_("Dataset is not an ensemble, you cannot select a member."));
        }

        // memberNum is 1-based, netcdf index is 0-based
        _fInd.memberStart = memberNum - 1;
        _fInd.memberCount = 1;
        _membersNb = 1;
    }

    void SelectMembers(int memberNb) {
        if (!_isEnsemble) {
            throw runtime_error(_("Dataset is not an ensemble, you cannot select a member."));
        }

        // memberNum is 1-based, netcdf index is 0-based
        _fInd.memberStart = 0;
        _fInd.memberCount = memberNb;
        _membersNb = memberNb;
    }

    int GetMembersNb() {
        return std::max(_membersNb, 1);
    }

    a1d GetLatAxis() const {
        return _axisLat;
    }

    a1d GetLonAxis() const {
        return _axisLon;
    }

    a1d* GetLatAxisPt() {
        return &_axisLat;
    }

    a1d* GetLonAxisPt() {
        return &_axisLon;
    }

    double GetXmin() const {
        wxASSERT(_axisLon.size() > 0);

        return _axisLon[0];
    }

    double GetYmin() const {
        wxASSERT(_axisLat.size() > 0);

        return std::min(_axisLat[_axisLat.size() - 1], _axisLat[0]);
    }

    double GetXmax() const {
        wxASSERT(_axisLon.size() > 0);

        return _axisLon[_axisLon.size() - 1];
    }

    double GetYmax() const {
        wxASSERT(_axisLat.size() > 0);

        return std::max(_axisLat[_axisLat.size() - 1], _axisLat[0]);
    }

    void SetWarnMissingLevels(bool val) {
        _warnMissingLevels = val;
    }

    void SetLevel(float val) {
        _level = val;
    }

    void SetTimeArray(const a1d& time) {
        _time = time;
    }

    void SetWasDumped(bool val) {
        _wasDumped = val;
    }

    bool WasDumped() const {
        return _wasDumped;
    }

    int GetPercentMissingAllowed() const {
        return _percentMissingAllowed;
    }

  protected:
    struct FileStructure {
        wxString dimLatName;
        wxString dimLonName;
        wxString dimTimeName;
        wxString dimLevelName;
        wxString dimMemberName;
        bool hasLevelDim;
        bool singleLevel;
        bool singleTimeStep;
        a1d lons;
        a1d lats;
        a1d levels;
        a1d time;
        a1i members;
        double timeStep;
        double firstHour;
    };
    struct FileIndexesArea {
        int lonStart;
        int lonCount;
        int latStart;
        int latCount;
    };
    struct FileIndexes {
        FileIndexesArea area;
        int lonStep;
        int latStep;
        int timeStartFile;
        int timeStartStorage;
        int timeCountFile;
        int timeCountStorage;
        bool timeConsistent;
        int timeStep;
        int level;
        int memberStart;
        int memberCount;
    };
    FileStructure _fStr;
    FileIndexes _fInd;
    asFile::FileType _fileType;
    bool _initialized;
    bool _standardized;
    bool _axesChecked;
    bool _wasDumped;
    wxString _dataId;
    wxString _datasetId;
    wxString _datasetName;
    wxString _provider;
    wxString _transformedBy;
    vd _nanValues;
    Parameter _parameter;
    wxString _parameterName;
    vi _gribCode;
    wxString _product;
    wxString _fileVarName;
    wxString _fileNamePattern;
    Unit _unit;
    bool _strideAllowed;
    float _level;
    a1d _time;
    vva2f _data;
    int _membersNb;
    int _latPtsnb;
    int _lonPtsnb;
    a1d _axisLat;
    a1d _axisLon;
    bool _isLatLon;
    bool _isPreprocessed;
    bool _isEnsemble;
    bool _canBeClipped;
    bool _parseTimeReference;
    bool _warnMissingFiles;
    bool _warnMissingLevels;
    wxString _fileExtension;
    wxString _preprocessMethod;
    vwxs _files;
    int _percentMissingAllowed;

    virtual void ListFiles(asTimeArray& timeArray);

    bool EnquireFileStructure(asTimeArray& timeArray);

    virtual bool ExtractFromFiles(asAreaGrid*& dataArea, asTimeArray& timeArray);

    virtual void ConvertToMjd(a1d& time, double refValue = NAN) const;

    virtual double FixTimeValue(double time) const;

    virtual bool CheckTimeArray(asTimeArray& timeArray);

    virtual bool GetAxesIndexes(asAreaGrid*& dataArea, asTimeArray& timeArray);

    size_t* GetIndexesStartNcdf() const;

    size_t* GetIndexesCountNcdf() const;

    ptrdiff_t* GetIndexesStrideNcdf() const;

    int* GetIndexesStartGrib() const;

    int* GetIndexesCountGrib() const;

    bool GetDataFromFile(asFileNetcdf& ncFile);

    bool GetDataFromFile(asFileGrib& gbFile);

    bool EnquireNetcdfFileStructure();

    bool ExtractFromNetcdfFile(const wxString& fileName, asAreaGrid*& dataArea, asTimeArray& timeArray);

    bool EnquireGribFileStructure(asTimeArray& timeArray);

    bool ExtractFromGribFile(const wxString& fileName, asAreaGrid*& dataArea, asTimeArray& timeArray);

    bool ParseFileStructure(asFileNetcdf& ncFile);

    bool ParseFileStructure(asFileGrib* gbFile0);

    bool ParseFileStructure(asFileGrib* gbFile0, asFileGrib* gbFile1);

    bool CheckFileStructure();

    bool HasDesiredLevel(bool useWarnings = true);

    bool InterpolateOnGrid(asAreaGrid* dataArea, asAreaGrid* desiredArea);

    bool TransformData();

    asAreaGrid* CreateMatchingArea(asAreaGrid* desiredArea);

    bool IsPressureLevel() const;

    bool IsIsentropicLevel() const;

    bool IsSurfaceLevel() const;

    bool IsSurfaceFluxesLevel() const;

    bool IsTotalColumnLevel() const;

    bool IsPVLevel() const;

    bool IsGeopotential() const;

    bool IsGeopotentialHeight() const;

    bool IsAirTemperature() const;

    bool IsRelativeHumidity() const;

    bool IsSpecificHumidity() const;

    bool IsVerticalVelocity() const;

    bool IsTotalColumnWater() const;

    bool IsTotalColumnWaterVapour() const;

    bool IsPrecipitableWater() const;

    bool IsPressure() const;

    bool IsSeaLevelPressure() const;

    bool IsUwindComponent() const;

    bool IsVwindComponent() const;

    bool IsPotentialVorticity() const;

    bool IsTotalPrecipitation() const;

    bool IsPrecipitationRate() const;

  private:
    wxString _directoryPath;

    bool ExtractSpatialAxes(asFileNetcdf& ncFile);

    bool ExtractLevelAxis(asFileNetcdf& ncFile);

    bool ExtractTimeAxis(asFileNetcdf& ncFile);

    bool FillWithNaNs();

    size_t CreateHash() const;

    wxString GetDumpFileName() const;
};

#endif
