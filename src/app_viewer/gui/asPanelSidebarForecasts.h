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

#ifndef __asPanelSidebarForecasts__
#define __asPanelSidebarForecasts__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include "asListBoxForecasts.h"
#include "asListBoxForecastDisplay.h"
#include "asListBoxQuantiles.h"
#include "asForecastManager.h"

class asPanelSidebarForecasts
        : public asPanelSidebar
{
public:
    asPanelSidebarForecasts(wxWindow *parent, asForecastManager *forecastManager, wxWindowID id = wxID_ANY,
                            const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize,
                            long style = wxTAB_TRAVERSAL);

    ~asPanelSidebarForecasts();

    void ClearForecasts();

    void Update();

    asListBoxForecasts *GetForecastsCtrl() const
    {
        return m_forecastsCtrl;
    }

    asListBoxQuantiles *GetQuantilesCtrl() const
    {
        return m_quantilesCtrl;
    }

    asListBoxForecastDisplay *GetForecastDisplayCtrl() const
    {
        return m_forecastDisplayCtrl;
    }

private:
    asListBoxForecasts *m_forecastsCtrl;
    asListBoxQuantiles *m_quantilesCtrl;
    asListBoxForecastDisplay *m_forecastDisplayCtrl;
};

#endif // __asPanelSidebarForecasts__
