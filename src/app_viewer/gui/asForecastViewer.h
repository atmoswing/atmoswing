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
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef ASFORECASTVIEWER_H
#define ASFORECASTVIEWER_H

#include "asIncludes.h"
#include "vroomgis.h"


class asForecastManager;
class asFrameForecast;

class asForecastViewer
{
public:
    /** Default constructor */
    asForecastViewer(asFrameForecast* parent, asForecastManager *forecastManager, vrLayerManager *layerManager, vrViewerLayerManager *viewerLayerManager);
    /** Default destructor */
    virtual ~asForecastViewer();

    wxArrayString GetForecastDisplayStringArray();
    wxArrayString GetPercentilesStringArray();
    void SetModel(int i);
    void SetLastModel();
    wxString GetStationName(int i_stat);
    float GetSelectedTargetDate();
    void SetForecastDisplay(int i);
    void SetPercentile(int i);
    void LoadPastForecast();
    void Redraw();
    void ChangeLeadTime( int val );

    int GetModelSelection()
    {
        return m_ModelSelection;
    }

    int GetForecastDisplaySelection()
    {
        return m_ForecastDisplaySelection;
    }

    int GetPercentileSelection()
    {
        return m_PercentileSelection;
    }

    /** Access the maximum value of the current layer
     * \return The current maximum value
     */
    float GetLayerMaxValue()
    {
        return m_LayerMaxValue;
    }

    /** Access the value of m_LeadTimeIndex
     * \return The current value of m_LeadTimeIndex
     */
    int GetLeadTimeIndex()
    {
        return m_LeadTimeIndex;
    }

protected:

private:
    int m_LeadTimeIndex;
    float m_LayerMaxValue;
    bool m_Opened;
    asFrameForecast* m_Parent;
    asForecastManager* m_ForecastManager;
    vrLayerManager *m_LayerManager;
    vrViewerLayerManager *m_ViewerLayerManager;
    wxArrayString m_DisplayForecast;
    wxArrayString m_DisplayPercentiles;
    VectorFloat m_ReturnPeriods;
    VectorFloat m_Percentiles;
    int m_ForecastDisplaySelection;
    int m_PercentileSelection;
    int m_ModelSelection;

};

#endif // ASFORECASTVIEWER_H
