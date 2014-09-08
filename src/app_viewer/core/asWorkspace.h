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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#ifndef ASWORKSPACE_H
#define ASWORKSPACE_H

#include <asIncludes.h>
#include <asFileWorkspace.h>

class asWorkspace : public wxObject
{
public:
    /** Default constructor */
    asWorkspace();

    /** Default destructor */
    virtual ~asWorkspace();

    bool Load(const wxString &filePath);
    bool Save();
    int GetLayersNb();

    wxString GetFilePath()
    {
        return m_FilePath;
    }

    wxString GetCoordinateSys()
    {
        return m_CoordinateSys;
    }

    wxString GetForecastsDirectory()
    {
        return m_ForecastsDirectory;
    }
    
    void SetForecastsDirectory(wxString val)
    {
        m_ForecastsDirectory = val;
    }

    wxString GetLayerPath(int i)
    {
        wxASSERT(m_LayerPaths.size()>i);
        return m_LayerPaths[i];
    }

    wxString GetLayerType(int i)
    {
        wxASSERT(m_LayerTypes.size()>i);
        return m_LayerTypes[i];
    }

    int GetLayerTransparency(int i)
    {
        wxASSERT(m_LayerTransparencies.size()>i);
        return m_LayerTransparencies[i];
    }

    bool GetLayerVisibility(int i)
    {
        wxASSERT(m_LayerVisibilities.size()>i);
        return m_LayerVisibilities[i];
    }

    int GetLayerLineWidth(int i)
    {
        wxASSERT(m_LayerLineWidths.size()>i);
        return m_LayerLineWidths[i];
    }

    wxColour GetLayerLineColor(int i)
    {
        wxASSERT(m_LayerLineColors.size()>i);
        return m_LayerLineColors[i];
    }

    wxColour GetLayerFillColor(int i)
    {
        wxASSERT(m_LayerFillColors.size()>i);
        return m_LayerFillColors[i];
    }
    
    double GetColorbarMaxValue()
    {
        return m_ColorbarMaxValue;
    }
    
    void SetColorbarMaxValue(double val)
    {
        m_ColorbarMaxValue = val;
    }

    int GetTimeSeriesPlotPastDaysNb()
    {
        return m_TimeSeriesPlotPastDaysNb;
    }

    void SetTimeSeriesPlotPastDaysNb(int val)
    {
        m_TimeSeriesPlotPastDaysNb = val;
    }

    int GetAlarmsPanelReturnPeriod()
    {
        return m_AlarmsPanelReturnPeriod;
    }
    
    void SetAlarmsPanelReturnPeriod(int val)
    {
        m_AlarmsPanelReturnPeriod = val;
    }

    float GetAlarmsPanelPercentile()
    {
        return m_AlarmsPanelPercentile;
    }
    
    void SetAlarmsPanelPercentile(float val)
    {
        m_AlarmsPanelPercentile = val;
    }


protected:
private:
    wxString m_FilePath;
    wxString m_CoordinateSys;
    wxString m_ForecastsDirectory;
    VectorString m_LayerPaths;
    VectorString m_LayerTypes;
    VectorInt m_LayerTransparencies;
    VectorBool m_LayerVisibilities;
    VectorInt m_LayerLineWidths;
    vector < wxColour > m_LayerLineColors;
    vector < wxColour > m_LayerFillColors;
    double m_ColorbarMaxValue;
    int m_TimeSeriesPlotPastDaysNb;
    int m_AlarmsPanelReturnPeriod;
    float m_AlarmsPanelPercentile;

};

#endif // ASWORKSPACE_H
