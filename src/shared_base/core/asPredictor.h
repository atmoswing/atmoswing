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

#ifndef ASPREDICTOR_H
#define ASPREDICTOR_H

#include <asIncludes.h>
#include <asFileGrib.h>
#include <asFileNetcdf.h>

class asTimeArray;

class asGeo;

class asAreaCompGrid;


class asPredictor
        : public wxObject
{
public:
    enum Parameter
    {
        ParameterUndefined,
        AirTemperature,
        PotentialTemperature,
        Geopotential,
        GeopotentialHeight,
        GeopotentialHeightAnomaly,
        PrecipitableWater,
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
        WaterVapour,
        MoistureFlux,
        Other
    };

    enum Unit
    {
        UnitUndefined, unitary, nb, mm, m, gpm, km, percent, fraction, degC, degK, Pa, Pa_s, g_kg, kg_kg, m_s, W_m2,
        kg_m2, kg_m2_s, N_m2, m2_s2, m2_s, degKm2_kg_s, mm_d, J_kg, degK_m, per_s, J_m2
    };

    explicit asPredictor(const wxString &dataId);

    ~asPredictor() override = default;

    static asPredictor *GetInstance(const wxString &datasetId, const wxString &dataId,
                                    const wxString &directory = wxEmptyString);

    virtual bool Init();

    void CheckLevelTypeIsDefined();

    bool CheckFilesPresence();

    bool Load(asAreaCompGrid *desiredArea, asTimeArray &timeArray, float level);

    bool Load(asAreaCompGrid &desiredArea, asTimeArray &timeArray, float level);

    bool Load(asAreaCompGrid &desiredArea, double date, float level);

    bool Load(asAreaCompGrid *desiredArea, double date, float level);

    bool ClipToArea(asAreaCompGrid *desiredArea);

    bool StandardizeData();

    bool Inline();

    bool SetData(vva2f &val);

    float GetMinValue() const;

    float GetMaxValue() const;

    bool HasNaN() const;

    vva2f &GetData()
    {
        wxASSERT((int) m_data.size() >= (int) m_time.size());
        wxASSERT(m_data.size() >= 1);
        wxASSERT(m_data[0].size() >= 1);
        wxASSERT(m_data[0][0].cols() > 0);
        wxASSERT(m_data[0][0].rows() > 0);

        return m_data;
    }

    wxString GetDataId() const
    {
        return m_dataId;
    }

    wxString GetProduct() const
    {
        return m_product;
    }

    Parameter GetParameter() const
    {
        return m_parameter;
    }

    void SetDirectoryPath(wxString directoryPath)
    {
        if ((directoryPath.Last() != wxFileName::GetPathSeparators())) {
            directoryPath.Append(wxFileName::GetPathSeparators());
        }

        m_directoryPath = directoryPath;
    }

    wxString GetDirectoryPath() const
    {
        wxString directoryPath = m_directoryPath;
        if ((directoryPath.Last() != wxFileName::GetPathSeparators())) {
            directoryPath.Append(wxFileName::GetPathSeparators());
        }

        return directoryPath;
    }

    wxString GetFullDirectoryPath() const
    {
        wxString directoryPath = m_directoryPath;
        if ((directoryPath.Last() != wxFileName::GetPathSeparators())) {
            directoryPath.Append(wxFileName::GetPathSeparators());
        }

        directoryPath += m_product;
        if ((directoryPath.Last() != wxFileName::GetPathSeparators())) {
            directoryPath.Append(wxFileName::GetPathSeparators());
        }

        return directoryPath;
    }

    int GetTimeSize() const
    {
        return (int) m_time.size();
    }

    int GetMembersNb() const
    {
        wxASSERT(!m_data.empty());

        return (int) m_data[0].size();
    }

    int GetLatPtsnb() const
    {
        return m_latPtsnb;
    }

    int GetLonPtsnb() const
    {
        return m_lonPtsnb;
    }

    void SetStandardize(bool val = true)
    {
        m_standardize = val;
    }

    static bool IsLatLon(const wxString &datasetId);

    bool IsLatLon() const
    {
        return m_isLatLon;
    }

    bool IsPreprocessed() const
    {
        return m_isPreprocessed;
    }

    void SetIsPreprocessed(bool val)
    {
        m_isPreprocessed = val;
    }

    bool IsEnsemble() const
    {
        return m_isEnsemble;
    }

    bool CanBeClipped() const
    {
        return m_canBeClipped;
    }

    void SetCanBeClipped(bool val)
    {
        m_canBeClipped = val;
    }

    wxString GetPreprocessMethod() const
    {
        return m_preprocessMethod;
    }

    void SetPreprocessMethod(const wxString &val)
    {
        m_preprocessMethod = val;
    }

    void SelectFirstMember()
    {
        if (!m_isEnsemble) {
            asThrowException(_("Dataset is not an ensemble, you cannot select a member."));
        }

        m_fInd.memberStart = 0;
        m_fInd.memberCount = 1;
    }

    void SelectMember(int memberNum)
    {
        if (!m_isEnsemble) {
            asThrowException(_("Dataset is not an ensemble, you cannot select a member."));
        }

        // memberNum is 1-based, netcdf index is 0-based
        m_fInd.memberStart = memberNum - 1;
        m_fInd.memberCount = 1;
    }

    void SelectMembers(int memberNb)
    {
        if (!m_isEnsemble) {
            asThrowException(_("Dataset is not an ensemble, you cannot select a member."));
        }

        // memberNum is 1-based, netcdf index is 0-based
        m_fInd.memberStart = 0;
        m_fInd.memberCount = memberNb;
    }

    int GetMembersNb()
    {
        return wxMax(m_fInd.memberCount, 1);
    }

    a1d GetLatAxis() const
    {
        return m_axisLat;
    }

    a1d GetLonAxis() const
    {
        return m_axisLon;
    }

    double GetXmin() const
    {
        wxASSERT(m_axisLon.size() > 0);

        return m_axisLon[0];
    }

    double GetYmin() const
    {
        wxASSERT(m_axisLat.size() > 0);

        return wxMin(m_axisLat[m_axisLat.size() - 1], m_axisLat[0]);
    }

    double GetXmax() const
    {
        wxASSERT(m_axisLon.size() > 0);

        return m_axisLon[m_axisLon.size() - 1];
    }

    double GetYmax() const
    {
        wxASSERT(m_axisLat.size() > 0);

        return wxMax(m_axisLat[m_axisLat.size() - 1], m_axisLat[0]);
    }


protected:
    struct FileStructure
    {
        wxString dimLatName;
        wxString dimLonName;
        wxString dimTimeName;
        wxString dimLevelName;
        wxString dimMemberName;
        bool hasLevelDim;
        bool singleLevel;
        a1d lons;
        a1d lats;
        a1d levels;
        a1i members;
        double timeStart;
        double timeEnd;
        double timeStep;
        double firstHour;
        size_t timeLength;
    };
    struct FileIndexesArea
    {
        int lonStart;
        int lonCount;
        int latStart;
        int latCount;
    };
    struct FileIndexes
    {
        std::vector<FileIndexesArea> areas;
        int lonStep;
        int latStep;
        int timeStart;
        int timeCount;
        int timeArrayCount;
        int timeStep;
        int level;
        int cutStart;
        int cutEnd;
        int memberStart;
        int memberCount;
    };
    FileStructure m_fStr;
    FileIndexes m_fInd;
    asFile::FileType m_fileType;
    bool m_initialized;
    bool m_standardize;
    bool m_axesChecked;
    wxString m_dataId;
    wxString m_datasetId;
    wxString m_datasetName;
    wxString m_provider;
    wxString m_transformedBy;
    vd m_nanValues;
    Parameter m_parameter;
    wxString m_parameterName;
    vi m_gribCode;
    wxString m_product;
    wxString m_fileVarName;
    wxString m_fileNamePattern;
    Unit m_unit;
    bool m_strideAllowed;
    float m_level;
    a1d m_time;
    vva2f m_data;
    int m_latPtsnb;
    int m_lonPtsnb;
    a1d m_axisLat;
    a1d m_axisLon;
    bool m_isLatLon;
    bool m_isPreprocessed;
    bool m_isEnsemble;
    bool m_canBeClipped;
    bool m_parseTimeReference;
    wxString m_fileExtension;
    wxString m_preprocessMethod;
    vwxs m_files;

    virtual void ListFiles(asTimeArray &timeArray);

    bool EnquireFileStructure(asTimeArray &timeArray);

    bool ExtractFromFiles(asAreaCompGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData);

    virtual double ConvertToMjd(double timeValue, double refValue = NaNd) const;

    virtual bool CheckTimeArray(asTimeArray &timeArray);

    virtual bool GetAxesIndexes(asAreaCompGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData);

    size_t *GetIndexesStartNcdf(int iArea) const;

    size_t *GetIndexesCountNcdf(int iArea) const;

    ptrdiff_t *GetIndexesStrideNcdf() const;

    int *GetIndexesStartGrib(int iArea) const;

    int *GetIndexesCountGrib(int iArea) const;

    bool GetDataFromFile(asFileNetcdf &ncFile, vvva2f &compositeData);

    bool GetDataFromFile(asFileGrib &gbFile, vvva2f &compositeData);

    bool EnquireNetcdfFileStructure();

    bool ExtractFromNetcdfFile(const wxString &fileName, asAreaCompGrid *&dataArea, asTimeArray &timeArray,
                               vvva2f &compositeData);

    bool EnquireGribFileStructure(asTimeArray &timeArray);

    bool ExtractFromGribFile(const wxString &fileName, asAreaCompGrid *&dataArea, asTimeArray &timeArray,
                             vvva2f &compositeData);

    bool ParseFileStructure(asFileNetcdf &ncFile);

    bool ParseFileStructure(asFileGrib *gbFile0, asFileGrib *gbFile1 = nullptr);

    bool CheckFileStructure();

    bool HasDesiredLevel();

    bool MergeComposites(vvva2f &compositeData, asAreaCompGrid *area);

    bool InterpolateOnGrid(asAreaCompGrid *dataArea, asAreaCompGrid *desiredArea);

    bool TransformData(vvva2f &compositeData);

    asAreaCompGrid *CreateMatchingArea(asAreaCompGrid *desiredArea);

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

    bool IsPrecipitableWater() const;

    bool IsPressure() const;

    bool IsSeaLevelPressure() const;

    bool IsUwindComponent() const;

    bool IsVwindComponent() const;

    bool IsPotentialVorticity() const;

    bool IsTotalPrecipitation() const;

    bool IsPrecipitationRate() const;

private:
    wxString m_directoryPath;

    bool ExtractSpatialAxes(asFileNetcdf &ncFile);

    bool ExtractLevelAxis(asFileNetcdf &ncFile);

    bool ExtractTimeAxis(asFileNetcdf &ncFile);

    bool FillWithNaNs(vvva2f &compositeData) const;
};

#endif // ASPREDICTOR_H
