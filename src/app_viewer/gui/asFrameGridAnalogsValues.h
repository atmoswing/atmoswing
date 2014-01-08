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
 
#ifndef __asFrameGridAnalogsValues__
#define __asFrameGridAnalogsValues__

/**
@file
Subclass of asFrameGridAnalogsValues, which is generated by wxFormBuilder.
*/

#include "AtmoswingViewerGui.h"
#include "asIncludes.h"

class asForecastManager;

/** Implementing asFrameGridAnalogsValues */
class asFrameGridAnalogsValues : public asFrameGridAnalogsValuesVirtual
{
public:
    /** Constructor */
    asFrameGridAnalogsValues( wxWindow* parent, int selectedForecast, asForecastManager *forecastManager, wxWindowID id=asWINDOW_GRID_ANALOGS );
    void Init();

protected:
    void OnChoiceForecastChange( wxCommandEvent& event );
    void OnChoiceStationChange( wxCommandEvent& event );
    void OnChoiceDateChange( wxCommandEvent& event );
    virtual void SortGrid( wxGridEvent& event );

private:
    asForecastManager *m_ForecastManager;
    int m_SelectedForecast;
    int m_SelectedStation;
    int m_SelectedDate;
    int m_SortAfterCol;
    Order m_SortOrder;

    bool UpdateGrid();
};

#endif // __asFrameGridAnalogsValues__
