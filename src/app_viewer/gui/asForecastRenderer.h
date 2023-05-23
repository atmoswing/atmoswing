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

#ifndef AS_FORECAST_RENDERER_H
#define AS_FORECAST_RENDERER_H

#include "asIncludes.h"
#include "vroomgis.h"

class asForecastManager;

class asFrameViewer;

class asForecastRenderer {
  public:
    asForecastRenderer(asFrameViewer* parent, asForecastManager* forecastManager, vrLayerManager* layerManager,
                       vrViewerLayerManager* viewerLayerManager);

    virtual ~asForecastRenderer();

    void FixForecastSelection();

    void ResetForecastSelection();

    void SetForecast(int methodRow, int forecastRow);

    float GetSelectedTargetDate();

    void SetForecastDisplay(int i);

    void SetQuantile(int i);

    void LoadPastForecast();

    void Redraw();

    void ChangeLeadTime(int val);

    void SetLeadTimeDate(float date);

    wxArrayString GetForecastDisplayStringArray() const {
        return m_displayForecast;
    }

    wxArrayString GetQuantilesStringArray() const {
        return m_displayQuantiles;
    }

    int GetMethodSelection() const {
        return m_methodSelection;
    }

    int GetForecastSelection() const {
        return m_forecastSelection;
    }

    int GetForecastDisplaySelection() const {
        return m_forecastDisplaySelection;
    }

    int GetQuantileSelection() const {
        return m_quantileSelection;
    }

    float GetLayerMaxValue() const {
        return m_layerMaxValue;
    }

    int GetLeadTimeIndex() const {
        return m_leadTimeIndex;
    }

    float GetLeadTimeDate() const {
        return m_leadTimeDate;
    }

  protected:
  private:
    asFrameViewer* m_parent;
    asForecastManager* m_forecastManager;
    vrLayerManager* m_layerManager;
    vrViewerLayerManager* m_viewerLayerManager;
    wxArrayString m_displayForecast;
    wxArrayString m_displayQuantiles;
    vf m_returnPeriods;
    vf m_quantiles;
    int m_leadTimeIndex;
    float m_leadTimeDate;
    float m_leadTimeStep;
    float m_layerMaxValue;
    int m_forecastDisplaySelection;
    int m_quantileSelection;
    int m_methodSelection;
    int m_forecastSelection;
    bool m_opened;

    void AdaptLeadTimeIndex();
};

#endif
