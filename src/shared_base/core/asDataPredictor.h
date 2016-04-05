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

class asTimeArray;

class asGeo;

class asGeoAreaCompositeGrid;


class asDataPredictor
        : public wxObject
{
public:
    asDataPredictor(const wxString &dataId);

    virtual ~asDataPredictor();

    virtual bool Init() = 0;

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

    Array1DFloat GetAxisLon()
    {
        return m_axisLon;
    }

    Array1DFloat GetAxisLat()
    {
        return m_axisLat;
    }

    void SetDirectoryPath(wxString directoryPath)
    {
        if ((directoryPath.Last() != '/') && (directoryPath.Last() != '\\')) {
            directoryPath.Append('/');
        }

        m_directoryPath = directoryPath;
    }

    wxString GetDirectoryPath()
    {
        return m_directoryPath;
    }

    int GetTimeSize()
    {
        return (int) m_time.size();
    }

    int GetLatPtsnb()
    {
        return m_latPtsnb;
    }

    int GetLonPtsnb()
    {
        return m_lonPtsnb;
    }

    double GetTimeStart()
    {
        return m_time[0];
    }

    double GetTimeEnd()
    {
        return m_time[m_time.size() - 1];
    }

    bool IsPreprocessed()
    {
        return m_isPreprocessed;
    }

    bool GetIsPreprocessed()
    {
        return m_isPreprocessed;
    }

    void SetIsPreprocessed(bool val)
    {
        m_isPreprocessed = val;
    }

    bool CanBeClipped()
    {
        return m_canBeClipped;
    }

    void SetCanBeClipped(bool val)
    {
        m_canBeClipped = val;
    }

    wxString GetPreprocessMethod()
    {
        return m_preprocessMethod;
    }

    void SetPreprocessMethod(wxString val)
    {
        m_preprocessMethod = val;
    }

    wxString GetFinalProviderWebsite()
    {
        return m_finalProviderWebsite;
    }

    wxString GetFinalProviderFTP()
    {
        return m_finalProviderFTP;
    }

    wxString GetDataId()
    {
        return m_dataId;
    }

    wxString GetDatasetName()
    {
        return m_datasetName;
    }

    double GetXaxisStep()
    {
        return m_xAxisStep;
    }

    void SetXaxisStep(const double val)
    {
        m_xAxisStep = (float) val;
    }

    double GetXaxisShift()
    {
        return m_xAxisShift;
    }

    double GetYaxisStep()
    {
        return m_yAxisStep;
    }

    void SetYaxisStep(const double val)
    {
        m_yAxisStep = (float) val;
    }

    double GetYaxisShift()
    {
        return m_yAxisShift;
    }

protected:
    wxString m_directoryPath;
    bool m_initialized;
    bool m_axesChecked;
    wxString m_dataId;
    wxString m_datasetId;
    wxString m_originalProvider;
    wxString m_finalProvider;
    wxString m_finalProviderWebsite;
    wxString m_finalProviderFTP;
    wxString m_datasetName;
    double m_timeZoneHours;
    double m_timeStepHours;
    double m_firstTimeStepHours;
    VectorDouble m_nanValues;
    DataParameter m_dataParameter;
    wxString m_fileVariableName;
    DataUnit m_unit;
    float m_xAxisStep;
    float m_yAxisStep;
    float m_xAxisShift;
    float m_yAxisShift;
    float m_level;
    Array1DDouble m_time;
    VArray2DFloat m_data;
    int m_latPtsnb;
    int m_lonPtsnb;
    size_t m_latIndexStep;
    size_t m_lonIndexStep;
    size_t m_timeIndexStep;
    Array1DFloat m_axisLat;
    Array1DFloat m_axisLon;
    bool m_isPreprocessed;
    bool m_canBeClipped;
    wxString m_preprocessMethod;
    wxString m_fileAxisLatName;
    wxString m_fileAxisLonName;
    wxString m_fileAxisTimeName;
    wxString m_fileAxisLevelName;
    wxString m_fileExtension;

    virtual bool CheckTimeArray(asTimeArray &timeArray)
    {
        return false;
    }

    virtual bool ExtractFromFiles(asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                  VVArray2DFloat &compositeData)
    {
        return false;
    }

    bool MergeComposites(VVArray2DFloat &compositeData, asGeoAreaCompositeGrid *area);

    bool InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid *CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea);

    asGeoAreaCompositeGrid *AdjustAxes(asGeoAreaCompositeGrid *dataArea, Array1DFloat &axisDataLon,
                                       Array1DFloat &axisDataLat, VVArray2DFloat &compositeData);

};

#endif // ASDATAPREDICTOR_H
