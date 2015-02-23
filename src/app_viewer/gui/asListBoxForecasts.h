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
 
#ifndef ASLISTBOXFORECASTS_H
#define ASLISTBOXFORECASTS_H

#include <wx/treectrl.h>

#include "asIncludes.h"
#include "asForecastManager.h"

class asForecastViewer;


class asForecastTreeItemData : public wxTreeItemData
{
public:
    asForecastTreeItemData(int methodRow, int forecastRow);

    void ShowInfo(wxTreeCtrl *tree);

    bool IsAggregator()
    {
        return (m_ForecastRow<0);
    }

    int GetMethodRow() 
    { 
        return m_MethodRow; 
    }

    void SetMethodRow(int methodRow)
    {
        m_MethodRow = methodRow;
    }

    int GetForecastRow() 
    { 
        return m_ForecastRow; 
    }

    void SetForecastRow(int forecastRow)
    {
        m_ForecastRow = forecastRow;
    }

private:
    int m_MethodRow;
    int m_ForecastRow;
};


class asMessageForecastChoice : public wxObject
{
public:
    asMessageForecastChoice(int methodRow, int forecastRow);

    bool IsAggregator()
    {
        return (m_ForecastRow<0);
    }

    int GetMethodRow() 
    { 
        return m_MethodRow; 
    }

    void SetMethodRow(int methodRow)
    {
        m_MethodRow = methodRow;
    }

    int GetForecastRow() 
    { 
        return m_ForecastRow; 
    }

    void SetForecastRow(int forecastRow)
    {
        m_ForecastRow = forecastRow;
    }

private:
    int m_MethodRow;
    int m_ForecastRow;
};


class asListBoxForecasts : public wxTreeCtrl
{
public:
    enum
    {
        TreeCtrlIcon_Precipitation,
        TreeCtrlIcon_Temperature,
        TreeCtrlIcon_Lightnings,
        TreeCtrlIcon_Wind,
        TreeCtrlIcon_Other
    };

    asListBoxForecasts(wxWindow *parent, asForecastManager* forecastManager, wxWindowID id=wxID_ANY, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize);
    virtual ~asListBoxForecasts();
    void CreateImageList();
    void Update();
    void Clear();
    void SetSelection(int methodId, int forecastId);
    void SelectFirst();

protected:

private:
    bool m_SkipSlctChangeEvent;
    asForecastManager* m_ForecastManager;

    void OnForecastSlctChange( wxTreeEvent& event );

    DECLARE_EVENT_TABLE();
};


#endif // ASLISTBOXFORECASTS_H
