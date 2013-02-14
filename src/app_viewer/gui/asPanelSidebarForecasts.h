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
 
#ifndef __asPanelSidebarForecasts__
#define __asPanelSidebarForecasts__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include "asListBoxModels.h"
#include "asListBoxForecastDisplay.h"
#include "asListBoxPercentiles.h"

/** Implementing asPanelSidebarForecasts */
class asPanelSidebarForecasts : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarForecasts( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarForecasts();

    void ClearForecasts();
    void AddForecast(const wxString &modelName, const wxString &leadTimeOriginStr);

    asListBoxModels *GetModelsCtrl()
    {
        return m_ModelsCtrl;
    }

    asListBoxPercentiles *GetPercentilesCtrl()
    {
        return m_PercentilesCtrl;
    }

    asListBoxForecastDisplay *GetForecastDisplayCtrl()
    {
        return m_ForecastDisplayCtrl;
    }

private:
    asListBoxModels *m_ModelsCtrl;
    asListBoxPercentiles *m_PercentilesCtrl;
    asListBoxForecastDisplay *m_ForecastDisplayCtrl;
};

#endif // __asPanelSidebarForecasts__
