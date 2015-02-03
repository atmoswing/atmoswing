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
 
#ifndef ASLISTBOXMODELS_H
#define ASLISTBOXMODELS_H

#include <wx/treectrl.h>

#include "asIncludes.h"

class asForecastViewer;


class asModelTreeItemData : public wxTreeItemData
{
public:
    asModelTreeItemData(const wxString& methodId, DataParameter dataParameter, bool isAggregator = false);

    void ShowInfo(wxTreeCtrl *tree);

    bool IsAggregator()
    {
        return m_IsAggregator;
    }

    wxString const& GetMethodId() const 
    { 
        return m_MethodId; 
    }

    void SetMethodId(wxString const& methodId)
    {
        m_MethodId = methodId;
    }

    wxString const& GetMethodIdDisplay() const 
    { 
        return m_MethodIdDisplay; 
    }

    void SetMethodIdDisplay(wxString const& methodIdDisplay)
    {
        m_MethodIdDisplay = methodIdDisplay;
    }

    wxString const& GetSpecificTag() const 
    { 
        return m_SpecificTag; 
    }

    void SetSpecificTag(wxString const& specificTag)
    {
        m_SpecificTag = specificTag;
    }

    wxString const& GetSpecificTagDisplay() const 
    { 
        return m_SpecificTagDisplay; 
    }

    void SetSpecificTagDisplay(wxString const& specificTagDisplay)
    {
        m_SpecificTagDisplay = specificTagDisplay;
    }

    DataParameter const& GetDataParameter() const 
    { 
        return m_DataParameter; 
    }

    void SetDataParameter(DataParameter dataParameter)
    {
        m_DataParameter = dataParameter;
    }

    DataTemporalResolution const& GetDataTemporalResolution() const 
    { 
        return m_DataTemporalResolution; 
    }

    void SetDataTemporalResolution(DataTemporalResolution dataTemporalResolution)
    {
        m_DataTemporalResolution = dataTemporalResolution;
    }

private:
    bool m_IsAggregator;
    wxString m_MethodId;
    wxString m_MethodIdDisplay;
    wxString m_SpecificTag;
    wxString m_SpecificTagDisplay;
    DataParameter m_DataParameter;
    DataTemporalResolution m_DataTemporalResolution;
};


class asListBoxModels : public wxTreeCtrl
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

    asListBoxModels(wxWindow *parent, wxWindowID id=wxID_ANY, const wxPoint &pos=wxDefaultPosition, const wxSize &size=wxDefaultSize);
    virtual ~asListBoxModels();
    void CreateImageList();
    bool Add(const wxString &methodId, const wxString &methodIdDisplay, const wxString &specificTag, const wxString &specificTagDisplay, DataParameter dataParameter, DataTemporalResolution dataTemporalResolution);

protected:

private:
    void OnModelSlctChange( wxCommandEvent & event );

    DECLARE_EVENT_TABLE();
};


#endif // ASLISTBOXMODELS_H
