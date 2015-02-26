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
    wxArrayString GetQuantilesStringArray();
    void FixForecastSelection();
    void SetForecast(int methodRow, int forecastRow);
    wxString GetStationName(int i_stat);
    float GetSelectedTargetDate();
    void SetForecastDisplay(int i);
    void SetQuantile(int i);
    void LoadPastForecast();
    void Redraw();
    void ChangeLeadTime( int val );
    void SetLeadTimeDate(float date);

    int GetMethodSelection()
    {
        return m_MethodSelection;
    }

    int GetForecastSelection()
    {
        return m_ForecastSelection;
    }

    int GetForecastDisplaySelection()
    {
        return m_ForecastDisplaySelection;
    }

    int GetQuantileSelection()
    {
        return m_QuantileSelection;
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

    /** Access the value of m_LeadTimeDate
     * \return The current value of m_LeadTimeDate
     */
    float GetLeadTimeDate()
    {
        return m_LeadTimeDate;
    }

protected:

private:
    int m_LeadTimeIndex;
    float m_LeadTimeDate;
    float m_LayerMaxValue;
    bool m_Opened;
    asFrameForecast* m_Parent;
    asForecastManager* m_ForecastManager;
    vrLayerManager *m_LayerManager;
    vrViewerLayerManager *m_ViewerLayerManager;
    wxArrayString m_DisplayForecast;
    wxArrayString m_DisplayQuantiles;
    VectorFloat m_ReturnPeriods;
    VectorFloat m_Quantiles;
    int m_ForecastDisplaySelection;
    int m_QuantileSelection;
    int m_MethodSelection;
    int m_ForecastSelection;

};

#endif // ASFORECASTVIEWER_H
