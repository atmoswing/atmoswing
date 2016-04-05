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

#ifndef ASPANELSMANAGERFORECASTS_H
#define ASPANELSMANAGERFORECASTS_H

#include "asIncludes.h"

#include <asPanelsManager.h>

class asPanelForecast;

class asPanelsManagerForecasts
        : public asPanelsManager
{
public:
    asPanelsManagerForecasts();

    virtual ~asPanelsManagerForecasts();

    void AddPanel(asPanelForecast *panel);

    void RemovePanel(asPanelForecast *panel);

    void Clear();

    asPanelForecast *GetPanel(int i) const;

    int GetPanelsNb() const;

    void SetForecastLedRunning(int num);

    void SetForecastLedError(int num);

    void SetForecastLedDone(int num);

    void SetForecastLedOff(int num);

    void SetForecastsAllLedsOff();

protected:
    std::vector<asPanelForecast *> m_arrayPanels;

private:

};

#endif // ASPANELSMANAGERFORECASTS_H
