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

#include "asFrameGridAnalogsValues.h"

#include "asForecastManager.h"

asFrameGridAnalogsValues::asFrameGridAnalogsValues(wxWindow *parent, int methodRow, int forecastRow,
                                                   asForecastManager *forecastManager, wxWindowID id)
    : asFrameGridAnalogsValuesVirtual(parent),
      m_forecastManager(forecastManager),
      m_selectedMethod(methodRow),
      m_selectedForecast(wxMax(forecastRow, 0)),
      m_selectedStation(0),
      m_selectedDate(0),
      m_sortAfterCol(0),
      m_sortOrder(Asc) {
    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFrameGridAnalogsValues::Init() {
    // Forecast list
    RebuildChoiceForecast();

    // Dates list
    wxArrayString arrayDates = m_forecastManager->GetLeadTimes(m_selectedMethod, m_selectedForecast);
    m_choiceDate->Set(arrayDates);
    m_choiceDate->Select(m_selectedDate);

    // Stations list
    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(m_selectedMethod, m_selectedForecast);
    m_choiceStation->Set(arrayStation);
    m_choiceStation->Select(m_selectedStation);

    // Set grid
    m_grid->SetColFormatNumber(0);
    m_grid->SetColFormatFloat(2, -1, 1);
    m_grid->SetColFormatFloat(3, -1, 3);
    UpdateGrid();
}

void asFrameGridAnalogsValues::RebuildChoiceForecast() {
    // Reset forecast list
    wxArrayString arrayForecasts = m_forecastManager->GetAllForecastNamesWxArray();
    m_choiceForecast->Set(arrayForecasts);
    int linearIndex = m_forecastManager->GetLinearIndex(m_selectedMethod, m_selectedForecast);
    m_choiceForecast->Select(linearIndex);

    // Highlight the specific forecasts
    for (int methodRow = 0; methodRow < m_forecastManager->GetMethodsNb(); methodRow++) {
        int stationId =
            m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast)->GetStationId(m_selectedStation);
        int forecastRow = m_forecastManager->GetForecastRowSpecificForStationId(methodRow, stationId);
        int index = m_forecastManager->GetLinearIndex(methodRow, forecastRow);
        wxString val = " --> " + m_choiceForecast->GetString(index) + " <-- ";
        m_choiceForecast->SetString(index, val);
    }
}

void asFrameGridAnalogsValues::OnChoiceForecastChange(wxCommandEvent &event) {
    int linearIndex = event.GetInt();
    m_selectedMethod = m_forecastManager->GetMethodRowFromLinearIndex(linearIndex);
    m_selectedForecast = m_forecastManager->GetForecastRowFromLinearIndex(linearIndex);

    // Dates list
    wxArrayString arrayDates = m_forecastManager->GetLeadTimes(m_selectedMethod, m_selectedForecast);
    m_choiceDate->Set(arrayDates);
    if (arrayDates.size() <= m_selectedDate) {
        m_selectedDate = 0;
    }
    m_choiceDate->Select(m_selectedDate);

    // Stations list
    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(m_selectedMethod, m_selectedForecast);
    m_choiceStation->Set(arrayStation);
    if (arrayStation.size() <= m_selectedStation) {
        m_selectedStation = 0;
    }
    m_choiceStation->Select(m_selectedStation);

    UpdateGrid();
}

void asFrameGridAnalogsValues::OnChoiceStationChange(wxCommandEvent &event) {
    m_selectedStation = event.GetInt();

    RebuildChoiceForecast();

    UpdateGrid();  // Doesn't change for criteria
}

void asFrameGridAnalogsValues::OnChoiceDateChange(wxCommandEvent &event) {
    m_selectedDate = event.GetInt();

    UpdateGrid();
}

void asFrameGridAnalogsValues::SortGrid(wxGridEvent &event) {
    // On a row label
    if (event.GetCol() == -1) {
        event.Skip();
        return;
    }

    // Check if twice on the same col
    if (m_sortAfterCol == event.GetCol()) {
        if (m_sortOrder == Asc) {
            m_sortOrder = Desc;
        } else {
            m_sortOrder = Asc;
        }
    } else {
        m_sortOrder = Asc;
    }
    m_sortAfterCol = event.GetCol();

    UpdateGrid();
}

bool asFrameGridAnalogsValues::UpdateGrid() {
    wxBusyCursor wait;

    if (m_forecastManager->GetMethodsNb() < 1) return false;

    asResultsForecast *forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    a1f dates = forecast->GetAnalogsDates(m_selectedDate);
    a1f values = forecast->GetAnalogsValuesRaw(m_selectedDate, m_selectedStation);
    a1f criteria = forecast->GetAnalogsCriteria(m_selectedDate);
    a1f analogNb = a1f::LinSpaced(dates.size(), 1, dates.size());

    m_grid->Hide();

    // m_grid->ClearGrid();
    m_grid->DeleteRows(0, m_grid->GetNumberRows());
    m_grid->InsertRows(0, dates.size());

    if (m_sortAfterCol > 0 || m_sortOrder == Desc) {
        if (m_sortAfterCol == 0)  // Analog nb
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&analogNb[0], &analogNb[analogNb.size() - 1], &vIndices[0], &vIndices[analogNb.size() - 1],
                         m_sortOrder);

            a1f copyDates = dates;
            a1f copyValues = values;
            a1f copyCriteria = criteria;

            for (int i = 0; i < dates.size(); i++) {
                int index = vIndices(i);
                dates[i] = copyDates[index];
                values[i] = copyValues[index];
                criteria[i] = copyCriteria[index];
            }
        } else if (m_sortAfterCol == 1)  // date
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&dates[0], &dates[dates.size() - 1], &vIndices[0], &vIndices[dates.size() - 1], m_sortOrder);

            a1f copyAnalogNb = analogNb;
            a1f copyValues = values;
            a1f copyCriteria = criteria;

            for (int i = 0; i < dates.size(); i++) {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                values[i] = copyValues[index];
                criteria[i] = copyCriteria[index];
            }
        } else if (m_sortAfterCol == 2)  // value
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&values[0], &values[values.size() - 1], &vIndices[0], &vIndices[values.size() - 1],
                         m_sortOrder);

            a1f copyAnalogNb = analogNb;
            a1f copyDates = dates;
            a1f copyCriteria = criteria;

            for (int i = 0; i < values.size(); i++) {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                dates[i] = copyDates[index];
                criteria[i] = copyCriteria[index];
            }

        } else if (m_sortAfterCol == 3)  // criteria
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&criteria[0], &criteria[criteria.size() - 1], &vIndices[0], &vIndices[criteria.size() - 1],
                         m_sortOrder);

            a1f copyAnalogNb = analogNb;
            a1f copyValues = values;
            a1f copyDates = dates;

            for (int i = 0; i < dates.size(); i++) {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                values[i] = copyValues[index];
                dates[i] = copyDates[index];
            }
        }
    }

    for (int i = 0; i < dates.size(); i++) {
        wxString buf;
        buf.Printf("%d", (int)analogNb[i]);
        m_grid->SetCellValue(i, 0, buf);

        buf.Printf("%s", asTime::GetStringTime(dates[i], "DD.MM.YYYY"));
        m_grid->SetCellValue(i, 1, buf);

        buf.Printf("%g", values[i]);
        m_grid->SetCellValue(i, 2, buf);

        buf.Printf("%g", criteria[i]);
        m_grid->SetCellValue(i, 3, buf);
    }

    m_grid->Show();

    return true;
}
