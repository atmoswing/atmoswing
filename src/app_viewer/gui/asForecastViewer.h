/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASFORECASTVIEWER_H
#define ASFORECASTVIEWER_H

#include "vroomgis.h"
#include "asIncludes.h"


class asForecastManager;

class asForecastViewer
{
public:
    //!< The file structure type
    enum DisplayType
    {
        ForecastRings,
        ForecastDots
    };

    /** Default constructor */
    asForecastViewer(wxWindow* parent, asForecastManager *forecastManager, vrLayerManager *layerManager, vrViewerLayerManager *viewerLayerManager);
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

    /** Access m_DisplayType
     * \return The current value of m_DisplayType
     */
    DisplayType GetDisplayType()
    {
        return m_DisplayType;
    }

    /** Set m_DisplayType
     * \param val New value to set
     */
    void SetDisplayType(DisplayType val)
    {
        m_DisplayType = val;
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
    DisplayType m_DisplayType;
    bool m_Opened;
    wxWindow* m_Parent;
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
