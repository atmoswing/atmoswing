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

#ifndef ASDATAPREDICTOR_H
#define ASDATAPREDICTOR_H

#include <asIncludes.h>
#include <asFileGrib2.h>
#include <asFileNetcdf.h>

class asTimeArray;

class asGeo;

class asGeoAreaCompositeGrid;


class asDataPredictor
        : public wxObject
{
public:
    enum Parameter
    {
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
        unitary, nb, mm, m, gpm, km, percent, fraction, degC, degK, Pa, Pa_s, kg_kg, m_s, W_m2, kg_m2, kg_m2_s, N_m2,
        m2_s2, degKm2_kg_s, mm_d
    };

    asDataPredictor(const wxString &dataId);

    virtual ~asDataPredictor();

    static Parameter StringToParameterEnum(const wxString &parameterStr);

    static wxString ParameterEnumToString(Parameter parameter);

    static Unit StringToUnitEnum(const wxString &unitStr);

    virtual bool Init() = 0;

    void CheckLevelTypeIsDefined();

    bool CheckFilesPresence(vwxs &filesList);

    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray);

    bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray);

    bool Load(asGeoAreaCompositeGrid &desiredArea, double date);

    bool Load(asGeoAreaCompositeGrid *desiredArea, double date);

    bool LoadFullArea(double date, float level);

    bool TransformData(vvva2f &compositeData);

    bool Inline();

    bool SetData(vva2f &val);

    float GetMinValue() const;

    float GetMaxValue() const;

    vva2f &GetData()
    {
        wxASSERT((int) m_data.size() == (int) m_time.size());
        wxASSERT(m_data.size() >= 1);
        wxASSERT(m_data[0].size() >= 1);
        wxASSERT(m_data[0][0].cols() > 0);
        wxASSERT(m_data[0][0].rows() > 0);
        wxASSERT(m_data[1][0].cols() > 0);
        wxASSERT(m_data[1][0].rows() > 0);

        return m_data;
    }

    a1d &GetTime()
    {
        return m_time;
    }

    a1f GetAxisLon() const
    {
        return m_axisLon;
    }

    a1f GetAxisLat() const
    {
        return m_axisLat;
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
        wxASSERT(m_data.size() > 0);

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

    double GetTimeStart() const
    {
        return m_time[0];
    }

    double GetTimeEnd() const
    {
        return m_time[m_time.size() - 1];
    }

    bool IsPreprocessed() const
    {
        return m_isPreprocessed;
    }

    bool GetIsPreprocessed() const
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

    void SetPreprocessMethod(wxString val)
    {
        m_preprocessMethod = val;
    }

    wxString GetDataId() const
    {
        return m_dataId;
    }

    wxString GetDatasetName() const
    {
        return m_datasetName;
    }

    double GetXaxisStep() const
    {
        return m_xAxisStep;
    }

    void SetXaxisStep(const double val)
    {
        m_xAxisStep = (float) val;
    }

    double GetXaxisShift() const
    {
        return m_xAxisShift;
    }

    double GetYaxisStep() const
    {
        return m_yAxisStep;
    }

    void SetYaxisStep(const double val)
    {
        m_yAxisStep = (float) val;
    }

    double GetYaxisShift() const
    {
        return m_yAxisShift;
    }

    void SetTimeStepHours(double val)
    {
        m_timeStepHours = val;
    }

    void SelectFirstMember()
    {
        if (!m_isEnsemble) {
            asThrowException(_("Dataset is not an ensemble, you cannot select a member."));
        }

        m_fileIndexes.memberStart = 0;
        m_fileIndexes.memberCount = 1;
    }

    void SelectMember(int memberNum)
    {
        if (!m_isEnsemble) {
            asThrowException(_("Dataset is not an ensemble, you cannot select a member."));
        }

        // memberNum is 1-based, netcdf index is 0-based
        m_fileIndexes.memberStart = memberNum-1;
        m_fileIndexes.memberCount = 1;
    }

    void SelectMembers(int memberNb)
    {
        if (!m_isEnsemble) {
            asThrowException(_("Dataset is not an ensemble, you cannot select a member."));
        }

        // memberNum is 1-based, netcdf index is 0-based
        m_fileIndexes.memberStart = 0;
        m_fileIndexes.memberCount = memberNb;
    }

    int GetMembersNb()
    {
        return wxMax(m_fileIndexes.memberCount, 1);
    }

protected:
    struct FileStructure
    {
        wxString dimLatName;
        wxString dimLonName;
        wxString dimTimeName;
        wxString dimLevelName;
        wxString dimMemberName;
        bool hasLevelDimension;
        bool singleLevel;
        a1f axisLon;
        a1f axisLat;
        a1f axisLevel;
        a1i axisMember;
        double axisTimeFirstValue;
        double axisTimeLastValue;
        size_t axisTimeLength;
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
    FileStructure m_fileStructure;
    FileIndexes m_fileIndexes;
    bool m_initialized;
    bool m_axesChecked;
    wxString m_subFolder;
    wxString m_dataId;
    wxString m_datasetId;
    wxString m_originalProvider;
    wxString m_transformedBy;
    wxString m_datasetName;
    double m_timeZoneHours;
    double m_timeStepHours;
    double m_firstTimeStepHours;
    vd m_nanValues;
    Parameter m_parameter;
    wxString m_parameterName;
    vi m_gribCode;
    wxString m_product;
    wxString m_fileVariableName;
    Unit m_unit;
    float m_xAxisStep;
    float m_yAxisStep;
    bool m_strideAllowed;
    float m_xAxisShift;
    float m_yAxisShift;
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
    wxString m_fileExtension;
    wxString m_preprocessMethod;

    virtual vwxs GetListOfFiles(asTimeArray &timeArray) const = 0;

    virtual bool ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                 vvva2f &compositeData) = 0;

    virtual double ConvertToMjd(double timeValue, double refValue = NaNd) const = 0;

    virtual bool CheckTimeArray(asTimeArray &timeArray) const = 0;

    virtual bool ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                  vvva2f &compositeData) = 0;

    virtual bool GetAxesIndexes(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                vvva2f &compositeData) = 0;

    size_t *GetIndexesStartNcdf(int iArea) const;

    size_t *GetIndexesCountNcdf(int iArea) const;

    ptrdiff_t *GetIndexesStrideNcdf() const;

    int *GetIndexesStartGrib(int iArea) const;

    int *GetIndexesCountGrib(int iArea) const;

    bool GetDataFromFile(asFileNetcdf &ncFile, vvva2f &compositeData);

    bool GetDataFromFile(asFileGrib2 &gbFile, vvva2f &compositeData);

    bool ExtractFromNetcdfFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                               vvva2f &compositeData);

    bool ExtractFromGribFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                             vvva2f &compositeData);

    bool ParseFileStructure(asFileNetcdf &ncFile);

    bool ParseFileStructure(asFileGrib2 &gbFile);

    bool CheckFileStructure();

    bool MergeComposites(vvva2f &compositeData, asGeoAreaCompositeGrid *area);

    bool InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid *CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid *AdjustAxes(asGeoAreaCompositeGrid *dataArea, vvva2f &compositeData);

    void AssignGribCode(const int arr[]);

private:
    wxString m_directoryPath;
};

#endif // ASDATAPREDICTOR_H
