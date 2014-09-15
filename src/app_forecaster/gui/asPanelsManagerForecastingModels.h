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
 
#ifndef ASPANELSMANAGERFORECASTINGMODELS_H
#define ASPANELSMANAGERFORECASTINGMODELS_H

#include "asIncludes.h"

#include <asPanelsManager.h>

class asPanelForecastingModel;

class asPanelsManagerForecastingModels : public asPanelsManager
{
public:
    asPanelsManagerForecastingModels();
    virtual ~asPanelsManagerForecastingModels();

    void AddPanel(asPanelForecastingModel* panel);
    void RemovePanel(asPanelForecastingModel* panel);
    void Clear();

    asPanelForecastingModel* GetPanel( int i );
    int GetPanelsNb();

    void SetForecastingModelLedRunning( int num );
    void SetForecastingModelLedError( int num );
    void SetForecastingModelLedDone( int num );
    void SetForecastingModelLedOff( int num );
    void SetForecastingModelsAllLedsOff();

protected:
    std::vector <asPanelForecastingModel*> m_ArrayPanels;

private:

};

#endif // ASPANELSMANAGERFORECASTINGMODELS_H
