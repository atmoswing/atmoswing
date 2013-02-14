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
 
#ifndef ASPANELSMANAGERFORECASTINGMODELS_H
#define ASPANELSMANAGERFORECASTINGMODELS_H

#include "asIncludes.h"

#include <asPanelsManager.h>

class asPanelForecastingModel;
class asFileForecastingModels;

class asPanelsManagerForecastingModels : public asPanelsManager
{
public:
    asPanelsManagerForecastingModels();
    virtual ~asPanelsManagerForecastingModels();

    void AddPanel(asPanelForecastingModel* panel);

    void RemovePanel(asPanelForecastingModel* panel);
    void Clear();

    void SetForecastingModelLedRunning( int num );
    void SetForecastingModelLedError( int num );
    void SetForecastingModelLedDone( int num );
    void SetForecastingModelLedOff( int num );
    void SetForecastingModelsAllLedsOff();

    bool GenerateXML(asFileForecastingModels &file);

protected:
    std::vector <asPanelForecastingModel*> m_ArrayPanels;

private:

};

#endif // ASPANELSMANAGERFORECASTINGMODELS_H
