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

#include "asPanelsManagerForecasts.h"

#include <asPanelForecast.h>

asPanelsManagerForecasts::asPanelsManagerForecasts()
        : asPanelsManager()
{
    //ctor
}

asPanelsManagerForecasts::~asPanelsManagerForecasts()
{
    // Destroy panels
    for (auto &panel : m_arrayPanels) {
        panel->Destroy();
    }
}

void asPanelsManagerForecasts::AddPanel(asPanelForecast *panel)
{
    // Set a pointer to the PanelsManager
    panel->SetPanelsManager(this);

    // Add to the array
    long arraylength = m_arrayPanels.size();
    panel->SetId((wxWindowID) arraylength);
    m_arrayPanels.push_back(panel);
}

void asPanelsManagerForecasts::RemovePanel(asPanelForecast *panel)
{
    wxWindow *parent = panel->GetParent();

    int id = panel->GetId();

    std::vector<asPanelForecast *> tmpArrayPanels;
    tmpArrayPanels = m_arrayPanels;
    m_arrayPanels.clear();

    for (auto &tmpPanel : tmpArrayPanels) {
        if (tmpPanel->GetId() != id) {
            tmpPanel->SetId(m_arrayPanels.size());
            m_arrayPanels.push_back(tmpPanel);
        }
    }

    // Delete it at least (not before to keep the reference to the Id)
    panel->Destroy();

    LayoutFrame(parent);
}

void asPanelsManagerForecasts::Clear()
{
    // Destroy panels
    for (auto &arrayPanel : m_arrayPanels) {
        wxASSERT(arrayPanel);
        arrayPanel->Destroy();
    }
    m_arrayPanels.clear();
}

asPanelForecast *asPanelsManagerForecasts::GetPanel(int i) const
{
    wxASSERT(i < (int) m_arrayPanels.size());
    return m_arrayPanels[i];
}

int asPanelsManagerForecasts::GetPanelsNb() const
{
    auto nb = (int) m_arrayPanels.size();
    return nb;
}

void asPanelsManagerForecasts::SetForecastLedRunning(int num)
{
    if (num < m_arrayPanels.size()) {
        awxLed *led = m_arrayPanels[num]->GetLed();
        if (!led)
            return;

        led->SetColour(awxLED_YELLOW);
        led->SetState(awxLED_ON);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecasts::SetForecastLedError(int num)
{
    if (num < m_arrayPanels.size()) {
        awxLed *led = m_arrayPanels[num]->GetLed();
        if (!led)
            return;

        led->SetColour(awxLED_RED);
        led->SetState(awxLED_ON);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecasts::SetForecastLedDone(int num)
{
    if (num < m_arrayPanels.size()) {
        awxLed *led = m_arrayPanels[num]->GetLed();
        if (!led)
            return;

        led->SetColour(awxLED_GREEN);
        led->SetState(awxLED_ON);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecasts::SetForecastLedOff(int num)
{
    if (num < m_arrayPanels.size()) {
        awxLed *led = m_arrayPanels[num]->GetLed();
        if (!led)
            return;

        led->SetState(awxLED_OFF);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecasts::SetForecastsAllLedsOff()
{
    for (auto &panel : m_arrayPanels) {
        awxLed *led = panel->GetLed();
        if (!led)
            return;

        led->SetState(awxLED_OFF);
        led->Update();
        led->Refresh();
    }
}
