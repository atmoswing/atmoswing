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
#include <asFileGrib2.h>
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
        GeopotentialHeight,
        Geopotential,
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
        Radiation,
        MomentumFlux,
        GravityWaveStress,
        SeaSurfaceTemperature,
        SeaSurfaceTemperatureAnomaly
    };

    enum Unit
    {
        UnitUndefined, unitary, nb, mm, m, gpm, km, percent, fraction, degC, degK, Pa, Pa_s, g_kg, kg_kg, m_s, W_m2,
        kg_m2, kg_m2_s, N_m2, m2_s2, degKm2_kg_s, mm_d
    };

    explicit asPredictor(const wxString &dataId);

    ~asPredictor() override = default;

    virtual bool Init() = 0;

    void CheckLevelTypeIsDefined();

    bool CheckFilesPresence();

    bool Load(asAreaCompGrid *desiredArea, asTimeArray &timeArray);

    bool Load(asAreaCompGrid &desiredArea, asTimeArray &timeArray);

    bool Load(asAreaCompGrid &desiredArea, double date);

    bool Load(asAreaCompGrid *desiredArea, double date);

    bool TransformData(vvva2f &compositeData);

    bool Inline();

    bool SetData(vva2f &val);

    float GetMinValue() const;

    float GetMaxValue() const;

    vva2f &GetData()
    {
        wxASSERT((int) m_data.size() >= (int) m_time.size());
        wxASSERT(m_data.size() >= 1);
        wxASSERT(m_data[0].size() >= 1);
        wxASSERT(m_data[0][0].cols() > 0);
        wxASSERT(m_data[0][0].rows() > 0);
        wxASSERT(m_data[1][0].cols() > 0);
        wxASSERT(m_data[1][0].rows() > 0);

        return m_data;
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

        directoryPath += m_subFolder;
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
        a1f lons;
        a1f lats;
        a1f levels;
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
    bool m_axesChecked;
    wxString m_subFolder;
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
    Unit m_unit;
    bool m_strideAllowed;
    float m_level;
    a1d m_time;
    vva2f m_data;
    int m_latPtsnb;
    int m_lonPtsnb;
    a1f m_axisLat;
    a1f m_axisLon;
    bool m_isPreprocessed;
    bool m_isEnsemble;
    bool m_canBeClipped;
    bool m_parseTimeReference;
    wxString m_fileExtension;
    wxString m_preprocessMethod;
    vwxs m_files;

    virtual void ListFiles(asTimeArray &timeArray) = 0;

    bool EnquireFileStructure();

    bool ExtractFromFiles(asAreaCompGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData);

    virtual double ConvertToMjd(double timeValue, double refValue = NaNd) const = 0;

    virtual bool CheckTimeArray(asTimeArray &timeArray) const = 0;

    virtual bool GetAxesIndexes(asAreaCompGrid *&dataArea, asTimeArray &timeArray, vvva2f &compositeData) = 0;

    size_t *GetIndexesStartNcdf(int iArea) const;

    size_t *GetIndexesCountNcdf(int iArea) const;

    ptrdiff_t *GetIndexesStrideNcdf() const;

    int *GetIndexesStartGrib(int iArea) const;

    int *GetIndexesCountGrib(int iArea) const;

    bool GetDataFromFile(asFileNetcdf &ncFile, vvva2f &compositeData);

    bool GetDataFromFile(asFileGrib2 &gbFile, vvva2f &compositeData);

    bool EnquireNetcdfFileStructure();

    bool ExtractFromNetcdfFile(const wxString &fileName, asAreaCompGrid *&dataArea, asTimeArray &timeArray,
                               vvva2f &compositeData);

    bool EnquireGribFileStructure();

    bool ExtractFromGribFile(const wxString &fileName, asAreaCompGrid *&dataArea, asTimeArray &timeArray,
                             vvva2f &compositeData);

    bool ParseFileStructure(asFileNetcdf &ncFile);

    bool ParseFileStructure(asFileGrib2 *gbFile0, asFileGrib2 *gbFile1 = nullptr);

    bool CheckFileStructure();

    bool MergeComposites(vvva2f &compositeData, asAreaCompGrid *area);

    bool InterpolateOnGrid(asAreaCompGrid *dataArea, asAreaCompGrid *desiredArea);

    asAreaCompGrid *CreateMatchingArea(asAreaCompGrid *desiredArea);

    asAreaCompGrid *AdjustAxes(asAreaCompGrid *dataArea, vvva2f &compositeData);

    void AssignGribCode(const int arr[]);

private:
    wxString m_directoryPath;

    bool ExtractSpatialAxes(asFileNetcdf &ncFile);

    bool ExtractLevelAxis(asFileNetcdf &ncFile);

    bool ExtractTimeAxis(asFileNetcdf &ncFile);
};

#endif // ASPREDICTOR_H
