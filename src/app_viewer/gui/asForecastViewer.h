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
 * Portions Copyright 2014-2015 Pascal Horton, Terranum.
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
    asForecastViewer(asFrameForecast *parent, asForecastManager *forecastManager, vrLayerManager *layerManager,
                     vrViewerLayerManager *viewerLayerManager);

    virtual ~asForecastViewer();

    wxArrayString GetForecastDisplayStringArray();

    wxArrayString GetQuantilesStringArray();

    void FixForecastSelection();

    void ResetForecastSelection();

    void SetForecast(int methodRow, int forecastRow);

    wxString GetStationName(int i_stat);

    float GetSelectedTargetDate();

    void SetForecastDisplay(int i);

    void SetQuantile(int i);

    void LoadPastForecast();

    void Redraw();

    void ChangeLeadTime(int val);

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

    float GetLayerMaxValue()
    {
        return m_layerMaxValue;
    }

    int GetLeadTimeIndex()
    {
        return m_leadTimeIndex;
    }

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
    asFrameForecast *m_parent;
    asForecastManager *m_forecastManager;
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
