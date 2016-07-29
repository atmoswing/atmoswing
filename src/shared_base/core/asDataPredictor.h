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
        PrecipitationRate,
        RelativeHumidity,
        SpecificHumidity,
        Omega,
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
        nb, mm, m, gpm, km, percent, fraction, degC, degK, Pa, Pa_s, kg_kg, m_s, W_m2, kg_m2, kg_m2_s, N_m2, m2_s2
    };

    enum Product
    {
        Any,
        PressureLevel,
        PressureLevelAssimilation,
        Surface,
        SurfaceFlux,
        OtherFlux,
        Tropopause,
        PotentialTemperatureLevel,
        PotentialVorticityLevel,
        ModelLevel,
        IsentropicLevel,
        VerticallyIntegrated
    };

    asDataPredictor(const wxString &dataId);

    virtual ~asDataPredictor();

    static Parameter StringToParameterEnum(const wxString &parameterStr);

    static wxString ParameterEnumToString(Parameter parameter);

    static Unit StringToUnitEnum(const wxString &unitStr);

    static Product StringToProductEnum(const wxString &productStr);

    static wxString LevelEnumToString(Product product);

    virtual bool Init() = 0;

    void CheckLevelTypeIsDefined();

    bool CheckFilesPresence(const VectorString &filesList);

    bool Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray);

    bool Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray);

    bool Load(asGeoAreaCompositeGrid &desiredArea, double date);

    bool Load(asGeoAreaCompositeGrid *desiredArea, double date);

    bool LoadFullArea(double date, float level);

    bool Inline();

    bool SetData(VArray2DFloat &val);

    float GetMinValue() const;

    float GetMaxValue() const;

    VArray2DFloat &GetData()
    {
        wxASSERT((int) m_data.size() == (int) m_time.size());
        wxASSERT(m_data.size() > 1);
        wxASSERT(m_data[0].cols() > 0);
        wxASSERT(m_data[0].rows() > 0);
        wxASSERT(m_data[1].cols() > 0);
        wxASSERT(m_data[1].rows() > 0);

        return m_data;
    }

    Array1DDouble &GetTime()
    {
        return m_time;
    }

    Array1DFloat GetAxisLon() const
    {
        return m_axisLon;
    }

    Array1DFloat GetAxisLat() const
    {
        return m_axisLat;
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

protected:
    struct FileStructure
    {
        wxString dimLatName;
        wxString dimLonName;
        wxString dimTimeName;
        wxString dimLevelName;
        bool hasLevelDimension;
        bool singleLevel;
        Array1DFloat axisLon;
        Array1DFloat axisLat;
        Array1DFloat axisLevel;
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
    VectorDouble m_nanValues;
    Parameter m_parameter;
    wxString m_parameterName;
    VectorInt m_gribCode;
    Product m_product;
    wxString m_fileVariableName;
    Unit m_unit;
    float m_xAxisStep;
    float m_yAxisStep;
    float m_xAxisShift;
    float m_yAxisShift;
    float m_level;
    Array1DDouble m_time;
    VArray2DFloat m_data;
    int m_latPtsnb;
    int m_lonPtsnb;
    Array1DFloat m_axisLat;
    Array1DFloat m_axisLon;
    bool m_isPreprocessed;
    bool m_canBeClipped;
    wxString m_fileExtension;
    wxString m_preprocessMethod;

    virtual VectorString GetListOfFiles(asTimeArray &timeArray) const = 0;

    virtual bool ExtractFromFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                 VVArray2DFloat &compositeData) = 0;

    virtual double ConvertToMjd(double timeValue, double refValue = NaNDouble) const = 0;

    virtual bool CheckTimeArray(asTimeArray &timeArray) const = 0;

    virtual bool ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                  VVArray2DFloat &compositeData) = 0;

    virtual bool GetAxesIndexes(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                VVArray2DFloat &compositeData) = 0;

    size_t *GetIndexesStartNcdf(int i_area) const;

    size_t *GetIndexesCountNcdf(int i_area) const;

    ptrdiff_t *GetIndexesStrideNcdf(int i_area) const;

    int *GetIndexesStartGrib(int i_area) const;

    int *GetIndexesCountGrib(int i_area) const;

    bool GetDataFromFile(asFileNetcdf &ncFile, VVArray2DFloat &compositeData);

    bool GetDataFromFile(asFileGrib2 &gbFile, VVArray2DFloat &compositeData);

    bool ExtractFromNetcdfFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                               VVArray2DFloat &compositeData);

    bool ExtractFromGribFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                             VVArray2DFloat &compositeData);

    bool ParseFileStructure(asFileNetcdf &ncFile, asGeoAreaCompositeGrid *&dataArea,
                                             asTimeArray &timeArray, VVArray2DFloat &compositeData);

    bool ParseFileStructure(asFileGrib2 &gbFile, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                             VVArray2DFloat &compositeData);

    bool CheckFileStructure();

    bool MergeComposites(VVArray2DFloat &compositeData, asGeoAreaCompositeGrid *area);

    bool InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid *CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid *AdjustAxes(asGeoAreaCompositeGrid *dataArea, VVArray2DFloat &compositeData);

private:
    wxString m_directoryPath;
};

#endif // ASDATAPREDICTOR_H
