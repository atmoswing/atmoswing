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

#ifndef AS_LIST_BOX_FORECASTS_H
#define AS_LIST_BOX_FORECASTS_H

#include <wx/treectrl.h>

#include "asIncludes.h"
#include "asForecastManager.h"

class asForecastViewer;


class asForecastTreeItemData
        : public wxTreeItemData
{
public:
    asForecastTreeItemData(int methodRow, int forecastRow);

    int GetMethodRow() const
    {
        return m_methodRow;
    }

    int GetForecastRow() const
    {
        return m_forecastRow;
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

    int GetMethodRow() const
    {
        return m_methodRow;
    }

    int GetForecastRow() const
    {
        return m_forecastRow;
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

    ~asListBoxForecasts() override = default;

    void CreateImageList();

    void Update() override;

    void Clear();

    void SetSelection(int methodRow, int forecastRow);

    void SelectFirst();

protected:

private:
    asForecastManager *m_forecastManager;
    bool m_skipSlctChangeEvent;

    void OnForecastSlctChange(wxTreeEvent &event);

DECLARE_EVENT_TABLE()
};


#endif
