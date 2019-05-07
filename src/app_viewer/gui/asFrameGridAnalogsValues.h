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

#ifndef AS_FRAME_GRID_ANALOGS_VALUES_H
#define AS_FRAME_GRID_ANALOGS_VALUES_H

#include "AtmoswingViewerGui.h"
#include "asIncludes.h"

class asForecastManager;

class asFrameGridAnalogsValues
        : public asFrameGridAnalogsValuesVirtual
{
public:
    asFrameGridAnalogsValues(wxWindow *parent, int methodRow, int forecastRow, asForecastManager *forecastManager,
                             wxWindowID id = asWINDOW_GRID_ANALOGS);

    void Init();

protected:
    void OnChoiceForecastChange(wxCommandEvent &event) override;

    void OnChoiceStationChange(wxCommandEvent &event) override;

    void OnChoiceDateChange(wxCommandEvent &event) override;

    void SortGrid(wxGridEvent &event) override;

private:
    asForecastManager *m_forecastManager;
    int m_selectedMethod;
    int m_selectedForecast;
    int m_selectedStation;
    int m_selectedDate;
    int m_sortAfterCol;
    Order m_sortOrder;

    void RebuildChoiceForecast();

    bool UpdateGrid();
};

#endif
