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

#ifndef ASLISTBOXFORECASTS_H
#define ASLISTBOXFORECASTS_H

#include <wx/treectrl.h>

#include "asIncludes.h"
#include "asForecastManager.h"

class asForecastViewer;


class asForecastTreeItemData
        : public wxTreeItemData
{
public:
    asForecastTreeItemData(int methodRow, int forecastRow);

    bool IsAggregator()
    {
        return (m_forecastRow < 0);
    }

    int GetMethodRow()
    {
        return m_methodRow;
    }

    void SetMethodRow(int methodRow)
    {
        m_methodRow = methodRow;
    }

    int GetForecastRow()
    {
        return m_forecastRow;
    }

    void SetForecastRow(int forecastRow)
    {
        m_forecastRow = forecastRow;
    }

private:
    int m_methodRow;
    int m_forecastRow;
};


class asMessageForecastChoice
        : public wxObject
{
public:
    asMessageForecastChoice(int methodRow, int forecastRow);

    bool IsAggregator()
    {
        return (m_forecastRow < 0);
    }

    int GetMethodRow()
    {
        return m_methodRow;
    }

    void SetMethodRow(int methodRow)
    {
        m_methodRow = methodRow;
    }

    int GetForecastRow()
    {
        return m_forecastRow;
    }

    void SetForecastRow(int forecastRow)
    {
        m_forecastRow = forecastRow;
    }

private:
    int m_methodRow;
    int m_forecastRow;
};


class asListBoxForecasts
        : public wxTreeCtrl
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

    asListBoxForecasts(wxWindow *parent, asForecastManager *forecastManager, wxWindowID id = wxID_ANY,
                       const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize);

    virtual ~asListBoxForecasts();

    void CreateImageList();

    void Update();

    void Clear();

    void SetSelection(int methodRow, int forecastRow);

    void SelectFirst();

protected:

private:
    bool m_skipSlctChangeEvent;
    asForecastManager *m_forecastManager;

    void OnForecastSlctChange(wxTreeEvent &event);

DECLARE_EVENT_TABLE();
};


#endif // ASLISTBOXFORECASTS_H
