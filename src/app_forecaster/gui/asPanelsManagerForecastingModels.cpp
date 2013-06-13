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
 
#include "asPanelsManagerForecastingModels.h"

#include <asPanelForecastingModel.h>
#include <asFileForecastingModels.h>

asPanelsManagerForecastingModels::asPanelsManagerForecastingModels()
:
asPanelsManager()
{
    //ctor
}

asPanelsManagerForecastingModels::~asPanelsManagerForecastingModels()
{
    // Destroy panels
    for (unsigned int i=0; i<m_ArrayPanels.size(); i++)
    {
        m_ArrayPanels[i]->Destroy();
    }
}

void asPanelsManagerForecastingModels::AddPanel(asPanelForecastingModel* panel)
{
    // Set a pointer to the PanelsManager
    panel->SetPanelsManager(this);

    // Add to the array
    int arraylength = m_ArrayPanels.size();
    panel->SetId(arraylength);
    m_ArrayPanels.push_back(panel);
}

void asPanelsManagerForecastingModels::RemovePanel(asPanelForecastingModel* panel)
{
    wxWindow* parent = panel->GetParent();

    int id = panel->GetId();

    std::vector <asPanelForecastingModel*> tmpArrayPanels;
    tmpArrayPanels = m_ArrayPanels;
    m_ArrayPanels.clear();

    for (unsigned int i=0; i<tmpArrayPanels.size(); i++)
    {
        if (tmpArrayPanels[i]->GetId()!=id)
        {
            tmpArrayPanels[i]->SetId(m_ArrayPanels.size());
            m_ArrayPanels.push_back(tmpArrayPanels[i]);
        }
    }

    // Delete it at least (not before to keep the reference to the Id)
    panel->Destroy();

    LayoutFrame(parent);
}

void asPanelsManagerForecastingModels::Clear()
{
    // Destroy panels
    for (unsigned int i=0; i<m_ArrayPanels.size(); i++)
    {
        wxASSERT(m_ArrayPanels[i]);
        m_ArrayPanels[i]->Destroy();
    }
    m_ArrayPanels.clear();
}

void asPanelsManagerForecastingModels::SetForecastingModelLedRunning( int num )
{
    if ((unsigned)num<m_ArrayPanels.size())
    {
        awxLed *led = m_ArrayPanels[num]->GetLed();
        if (!led) return;

        led->SetColour(awxLED_YELLOW);
        led->SetState(awxLED_ON);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecastingModels::SetForecastingModelLedError( int num )
{
    if ((unsigned)num<m_ArrayPanels.size())
    {
        awxLed *led = m_ArrayPanels[num]->GetLed();
        if (!led) return;

        led->SetColour(awxLED_RED);
        led->SetState(awxLED_ON);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecastingModels::SetForecastingModelLedDone( int num )
{
    if ((unsigned)num<m_ArrayPanels.size())
    {
        awxLed *led = m_ArrayPanels[num]->GetLed();
        if (!led) return;

        led->SetColour(awxLED_GREEN);
        led->SetState(awxLED_ON);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecastingModels::SetForecastingModelLedOff( int num )
{
    if ((unsigned)num<m_ArrayPanels.size())
    {
        awxLed *led = m_ArrayPanels[num]->GetLed();
        if (!led) return;

        led->SetState(awxLED_OFF);
        led->Update();
        led->Refresh();
    }
}

void asPanelsManagerForecastingModels::SetForecastingModelsAllLedsOff( )
{
    for (unsigned int i=0; i<m_ArrayPanels.size(); i++)
    {
        awxLed *led = m_ArrayPanels[i]->GetLed();
        if (!led) return;

        led->SetState(awxLED_OFF);
        led->Update();
        led->Refresh();
    }
}

bool asPanelsManagerForecastingModels::GenerateXML(asFileForecastingModels &file)
{
    asLogMessage(_("Setting the root tag in the xml file."));

    // Create the node in the XML doc
    if(!file.InsertElement(wxEmptyString, "ModelsList")) return false;
    if(!file.GoToLastNodeWithPath("ModelsList")) return false;

    asLogMessage(_("Generating xml file content."));

    for (unsigned int i=0; i<m_ArrayPanels.size(); i++)
    {
        if (!m_ArrayPanels[i]->GenerateXML(file)) return false;
    }

    return true;
}
