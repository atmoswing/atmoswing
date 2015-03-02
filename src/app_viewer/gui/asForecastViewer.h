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
        return m_methodSelection;
    }

    int GetForecastSelection()
    {
        return m_forecastSelection;
    }

    int GetForecastDisplaySelection()
    {
        return m_forecastDisplaySelection;
    }

    int GetQuantileSelection()
    {
        return m_quantileSelection;
    }

    /** Access the maximum value of the current layer
     * \return The current maximum value
     */
    float GetLayerMaxValue()
    {
        return m_layerMaxValue;
    }

    /** Access the value of m_leadTimeIndex
     * \return The current value of m_leadTimeIndex
     */
    int GetLeadTimeIndex()
    {
        return m_leadTimeIndex;
    }

    /** Access the value of m_leadTimeDate
     * \return The current value of m_leadTimeDate
     */
    float GetLeadTimeDate()
    {
        return m_leadTimeDate;
    }

protected:

private:
    int m_leadTimeIndex;
    float m_leadTimeDate;
    float m_layerMaxValue;
    bool m_opened;
    asFrameForecast* m_parent;
    asForecastManager* m_forecastManager;
    vrLayerManager *m_layerManager;
    vrViewerLayerManager *m_viewerLayerManager;
    wxArrayString m_displayForecast;
    wxArrayString m_displayQuantiles;
    VectorFloat m_returnPeriods;
    VectorFloat m_quantiles;
    int m_forecastDisplaySelection;
    int m_quantileSelection;
    int m_methodSelection;
    int m_forecastSelection;

};

#endif // ASFORECASTVIEWER_H
