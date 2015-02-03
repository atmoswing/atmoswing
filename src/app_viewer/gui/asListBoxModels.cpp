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
 
#include "asListBoxModels.h"

#include <asForecastViewer.h>

#include "img_treectrl.h"


BEGIN_EVENT_TABLE(asListBoxModels, wxTreeCtrl)
    EVT_LISTBOX(wxID_ANY, asListBoxModels::OnModelSlctChange)
END_EVENT_TABLE()

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED, wxCommandEvent);


asModelTreeItemData::asModelTreeItemData(const wxString& methodId, DataParameter dataParameter, bool isAggregator)
:
wxTreeItemData()
{
    m_MethodId = methodId;
    m_DataParameter = dataParameter;
    m_IsAggregator = isAggregator;
}


asListBoxModels::asListBoxModels(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size)
:
wxTreeCtrl(parent, id, pos, size, wxTR_DEFAULT_STYLE|wxTR_HIDE_ROOT|wxTR_TWIST_BUTTONS|wxTR_FULL_ROW_HIGHLIGHT|wxTR_NO_LINES|wxNO_BORDER, wxDefaultValidator)
{
    CreateImageList();
    unsigned int indent = GetIndent();
    if (indent > 16) SetIndent( indent-5 );
}

asListBoxModels::~asListBoxModels()
{
    //dtor
}

void asListBoxModels::CreateImageList()
{
    int size = 16;

    // Make an image list containing small icons
    wxImageList *images = new wxImageList(size, size, true);

    // Images must match the enum
    images->Add(img_precipitation_s);
    images->Add(img_temperature_s);
    images->Add(img_lightning_s);
    images->Add(img_wind_s);
    images->Add(img_other_s);

    AssignImageList(images);
}

bool asListBoxModels::Add(const wxString &methodId, const wxString &methodIdDisplay, const wxString &specificTag, const wxString &specificTagDisplay, DataParameter dataParameter, DataTemporalResolution dataTemporalResolution)
{
    if(!GetRootItem().IsOk())
    {
        AddRoot(_("Root"), -1, -1, new wxTreeItemData());
    }

    wxTreeItemId parentItemId;

    // Check if the method ID already exists
    wxTreeItemId itemId = GetFirstVisibleItem();
    while (itemId.IsOk())
    {
        bool isSameCategory = true;
        asModelTreeItemData *item = (asModelTreeItemData *)GetItemData(itemId);

        if (!item->GetMethodId().IsSameAs(methodId)) isSameCategory = false;
        if (item->GetDataParameter() != dataParameter) isSameCategory = false;
        if (item->GetDataTemporalResolution() != dataTemporalResolution) isSameCategory = false;

        if (isSameCategory)
        {
            parentItemId = itemId;
            break;
        }

        itemId = GetNextSibling(itemId);
    }

    // If parent was not found
    if (!parentItemId.IsOk())
    {
        int image;
        switch (dataParameter)
        {
            case Precipitation:
                image = asListBoxModels::TreeCtrlIcon_Precipitation;
                break;
            case AirTemperature:
                image = asListBoxModels::TreeCtrlIcon_Temperature;
                break;
            case Lightnings:
                image = asListBoxModels::TreeCtrlIcon_Lightnings;
                break;
            case Wind:
                image = asListBoxModels::TreeCtrlIcon_Wind;
                break;
            default:
                image = asListBoxModels::TreeCtrlIcon_Other;
        }

        asModelTreeItemData *newItemAggregator = new asModelTreeItemData(methodId, dataParameter, true);
        newItemAggregator->SetMethodIdDisplay(methodIdDisplay);
        newItemAggregator->SetDataTemporalResolution(dataTemporalResolution);

        parentItemId = AppendItem( GetRootItem(), methodIdDisplay, image, image, newItemAggregator);

    }

    // Create the new item
    asModelTreeItemData *newItem = new asModelTreeItemData(methodId, dataParameter);
    newItem->SetMethodIdDisplay(methodIdDisplay);
    newItem->SetSpecificTag(specificTag);
    newItem->SetSpecificTagDisplay(specificTagDisplay);
    newItem->SetDataTemporalResolution(dataTemporalResolution);

    wxString name = specificTagDisplay;
    if (name.IsEmpty()) name = methodIdDisplay;
    wxTreeItemId newItemId = AppendItem( parentItemId, name, -1, -1, newItem);

    return true;
}

void asListBoxModels::OnModelSlctChange( wxCommandEvent & event )
{
    wxCommandEvent eventSlct (asEVT_ACTION_FORECAST_MODEL_SELECTION_CHANGED);
    eventSlct.SetInt(event.GetInt());
    GetParent()->ProcessWindowEvent(eventSlct);
}
