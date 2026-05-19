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

    void FixMethodSelection();

    wxArrayString GetForecastDisplayStringArray() const {
        return _displayForecast;
    }

    wxArrayString GetQuantilesStringArray() const {
        return _displayQuantiles;
    }

    int GetMethodSelection() const {
        return _methodSelection;
    }

    int GetForecastSelection() const {
        return _forecastSelection;
    }

    int GetForecastDisplaySelection() const {
        return _forecastDisplaySelection;
    }

    int GetQuantileSelection() const {
        return _quantileSelection;
    }

    float GetLayerMaxValue() const {
        return _layerMaxValue;
    }

    int GetLeadTimeIndex() const {
        return _leadTimeIndex;
    }

    float GetLeadTimeDate() const {
        return _leadTimeDate;
    }

  protected:
  private:
    asFrameViewer* _parent;
    asForecastManager* _forecastManager;
    vrLayerManager* _layerManager;
    vrViewerLayerManager* _viewerLayerManager;
    wxArrayString _displayForecast;
    wxArrayString _displayQuantiles;
    vf _returnPeriods;
    vf _quantiles;
    int _leadTimeIndex;
    float _leadTimeDate;
    float _leadTimeStep;
    float _layerMaxValue;
    int _forecastDisplaySelection;
    int _quantileSelection;
    int _methodSelection;
    int _forecastSelection;
    bool _opened;

    void AdaptLeadTimeIndex();
};

#endif
