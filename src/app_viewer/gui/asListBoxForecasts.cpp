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

#include "asListBoxForecasts.h"

#include <asForecastViewer.h>

#include "images.h"


BEGIN_EVENT_TABLE(asListBoxForecasts, wxTreeCtrl)
    EVT_TREE_SEL_CHANGED(wxID_ANY, asListBoxForecasts::OnForecastSlctChange)
END_EVENT_TABLE()

wxDEFINE_EVENT(asEVT_ACTION_FORECAST_SELECTION_CHANGED, wxCommandEvent);


asForecastTreeItemData::asForecastTreeItemData(int methodRow, int forecastRow)
        : wxTreeItemData(),
          m_methodRow(methodRow),
          m_forecastRow(forecastRow)
{

}


asMessageForecastChoice::asMessageForecastChoice(int methodRow, int forecastRow)
        : wxObject(),
          m_methodRow(methodRow),
          m_forecastRow(forecastRow)
{
}


asListBoxForecasts::asListBoxForecasts(wxWindow *parent, asForecastManager *forecastManager, wxWindowID id,
                                       const wxPoint &pos, const wxSize &size)
        : wxTreeCtrl(parent, id, pos, size,
                     wxTR_DEFAULT_STYLE | wxTR_HIDE_ROOT | wxTR_TWIST_BUTTONS | wxTR_FULL_ROW_HIGHLIGHT |
                     wxTR_NO_LINES | wxNO_BORDER, wxDefaultValidator),
          m_forecastManager(forecastManager),
          m_skipSlctChangeEvent(false)
{
    CreateImageList();
    unsigned int indent = GetIndent();
    if (indent > 16)
        SetIndent(indent - 5);
}

void asListBoxForecasts::CreateImageList()
{
    int size = 16 * g_ppiScaleDc;

    // Make an image list containing small icons
    auto *images = new wxImageList(size, size, true);

    // Images must match the enum
    images->Add(*_img_icon_precip);
    images->Add(*_img_icon_temp);
    images->Add(*_img_icon_lightning);
    images->Add(*_img_icon_wind);
    images->Add(*_img_icon_other);

    AssignImageList(images);
}

void asListBoxForecasts::Update()
{
    Clear();

    if (!GetRootItem().IsOk()) {
        AddRoot(_("Root"), -1, -1, new wxTreeItemData());
    }

    for (int methodRow = 0; methodRow < m_forecastManager->GetMethodsNb(); methodRow++) {
        asResultsForecast *forecastFirst = m_forecastManager->GetForecast(methodRow, 0);

        int image;
        switch (forecastFirst->GetPredictandParameter()) {
            case asPredictand::Precipitation:
                image = asListBoxForecasts::TreeCtrlIcon_Precipitation;
                break;
            case asPredictand::AirTemperature:
                image = asListBoxForecasts::TreeCtrlIcon_Temperature;
                break;
            case asPredictand::Lightnings:
                image = asListBoxForecasts::TreeCtrlIcon_Lightnings;
                break;
            case asPredictand::Wind:
                image = asListBoxForecasts::TreeCtrlIcon_Wind;
                break;
            default:
                image = asListBoxForecasts::TreeCtrlIcon_Other;
        }

        auto *itemMethod = new asForecastTreeItemData(methodRow, -1);

        wxString label = wxString::Format("%d. %s (%s)", methodRow + 1, forecastFirst->GetMethodIdDisplay(),
                                          forecastFirst->GetMethodId());
        wxTreeItemId parentItemId = AppendItem(GetRootItem(), label, image, image, itemMethod);

        if (parentItemId.IsOk()) {
            for (int forecastRow = 0; forecastRow < m_forecastManager->GetForecastsNb(methodRow); forecastRow++) {
                asResultsForecast *forecast = m_forecastManager->GetForecast(methodRow, forecastRow);

                // Create the new forecast item
                auto *itemForecast = new asForecastTreeItemData(methodRow, forecastRow);

                wxString name = forecast->GetSpecificTagDisplay();
                if (name.IsEmpty())
                    name = forecast->GetMethodIdDisplay();
                AppendItem(parentItemId, name, -1, -1, itemForecast);
            }
        }
    }
}

void asListBoxForecasts::OnForecastSlctChange(wxTreeEvent &event)
{
    wxBusyCursor wait;

    wxTreeItemId itemId = event.GetItem();

    if (!m_skipSlctChangeEvent && itemId.IsOk()) {
        auto *item = (asForecastTreeItemData *) GetItemData(itemId);

        int methodRow = item->GetMethodRow();
        int forecastRow = item->GetForecastRow();

        if (methodRow >= 0) {
            wxCommandEvent eventSlct(asEVT_ACTION_FORECAST_SELECTION_CHANGED);

            auto *message = new asMessageForecastChoice(methodRow, forecastRow);

            eventSlct.SetClientData(message);
            GetParent()->ProcessWindowEvent(eventSlct);
        }
    }
}

void asListBoxForecasts::Clear()
{
    m_skipSlctChangeEvent = true;
    DeleteAllItems();
    m_skipSlctChangeEvent = false;
}

void asListBoxForecasts::SetSelection(int methodRow, int forecastRow)
{
    m_skipSlctChangeEvent = true;

    // Look for the correct entry in the treectrl
    wxTreeItemId methodItemId = GetFirstVisibleItem();
    for (int i = 0; i < methodRow; i++) {
        if (methodItemId.IsOk()) {
            methodItemId = GetNextSibling(methodItemId);
        }
    }

    if (forecastRow < 0) {
        if (methodItemId.IsOk()) {
            SelectItem(methodItemId);
        }
    } else {
        if (methodItemId.IsOk()) {
            wxTreeItemIdValue cookie;
            wxTreeItemId forecastItemId = GetFirstChild(methodItemId, cookie);
            for (int i = 0; i < forecastRow; i++) {
                if (forecastItemId.IsOk()) {
                    forecastItemId = GetNextSibling(forecastItemId);
                }
            }
            if (forecastItemId.IsOk()) {
                SelectItem(methodItemId);
            }
        }
    }

    m_skipSlctChangeEvent = false;
}

void asListBoxForecasts::SelectFirst()
{
    wxTreeItemId itemId = GetFirstVisibleItem();
    if (itemId.IsOk()) {
        SelectItem(itemId);
    }
}
