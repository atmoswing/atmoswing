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

asFrameGridAnalogsValues::asFrameGridAnalogsValues(wxWindow* parent, int methodRow, int forecastRow,
                                                   asForecastManager* forecastManager, wxWindowID id)
    : asFrameGridAnalogsValuesVirtual(parent),
      _forecastManager(forecastManager),
      _selectedMethod(methodRow),
      _selectedForecast(wxMax(forecastRow, 0)),
      _selectedStation(0),
      _selectedDate(0),
      _sortAfterCol(0),
      _sortOrder(Asc) {
    SetLabel(_("Analogs details"));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFrameGridAnalogsValues::Init() {
    // Forecast list
    RebuildChoiceForecast();

    // Dates list
    wxArrayString arrayDates = _forecastManager->GetTargetDatesWxArray(_selectedMethod, _selectedForecast);
    _choiceDate->Set(arrayDates);
    _choiceDate->Select(_selectedDate);

    // Stations list
    wxArrayString arrayStation = _forecastManager->GetStationNamesWithHeights(_selectedMethod, _selectedForecast);
    _choiceStation->Set(arrayStation);
    _choiceStation->Select(_selectedStation);

    // Set grid
    _grid->SetColFormatNumber(0);
    _grid->SetColFormatFloat(2, -1, 1);
    _grid->SetColFormatFloat(3, -1, 3);
    UpdateGrid();
}

void asFrameGridAnalogsValues::RebuildChoiceForecast() {
    // Reset forecast list
    wxArrayString arrayForecasts = _forecastManager->GetCombinedForecastNamesWxArray();
    _choiceForecast->Set(arrayForecasts);
    int linearIndex = _forecastManager->GetLinearIndex(_selectedMethod, _selectedForecast);
    _choiceForecast->Select(linearIndex);

    // Highlight the specific forecasts
    for (int methodRow = 0; methodRow < _forecastManager->GetMethodsNb(); methodRow++) {
        int stationId = _forecastManager->GetForecast(_selectedMethod, _selectedForecast)
                            ->GetStationId(_selectedStation);
        int forecastRow = _forecastManager->GetForecastRowSpecificForStationId(methodRow, stationId);
        int index = _forecastManager->GetLinearIndex(methodRow, forecastRow);
        wxString val = "* " + _choiceForecast->GetString(index) + " *";
        _choiceForecast->SetString(index, val);
    }
}

void asFrameGridAnalogsValues::OnChoiceForecastChange(wxCommandEvent& event) {
    int linearIndex = event.GetInt();
    _selectedMethod = _forecastManager->GetMethodRowFromLinearIndex(linearIndex);
    _selectedForecast = _forecastManager->GetForecastRowFromLinearIndex(linearIndex);

    // Dates list
    wxArrayString arrayDates = _forecastManager->GetTargetDatesWxArray(_selectedMethod, _selectedForecast);
    _choiceDate->Set(arrayDates);
    if (arrayDates.size() <= _selectedDate) {
        _selectedDate = 0;
    }
    _choiceDate->Select(_selectedDate);

    // Stations list
    wxArrayString arrayStation = _forecastManager->GetStationNamesWithHeights(_selectedMethod, _selectedForecast);
    _choiceStation->Set(arrayStation);
    if (arrayStation.size() <= _selectedStation) {
        _selectedStation = 0;
    }
    _choiceStation->Select(_selectedStation);

    UpdateGrid();
}

void asFrameGridAnalogsValues::OnChoiceStationChange(wxCommandEvent& event) {
    _selectedStation = event.GetInt();

    RebuildChoiceForecast();

    UpdateGrid();  // Doesn't change for criteria
}

void asFrameGridAnalogsValues::OnChoiceDateChange(wxCommandEvent& event) {
    _selectedDate = event.GetInt();

    UpdateGrid();
}

void asFrameGridAnalogsValues::SortGrid(wxGridEvent& event) {
    // On a row label
    if (event.GetCol() == -1) {
        event.Skip();
        return;
    }

    // Check if twice on the same col
    if (_sortAfterCol == event.GetCol()) {
        if (_sortOrder == Asc) {
            _sortOrder = Desc;
        } else {
            _sortOrder = Asc;
        }
    } else {
        _sortOrder = Asc;
    }
    _sortAfterCol = event.GetCol();

    UpdateGrid();
}

bool asFrameGridAnalogsValues::UpdateGrid() {
    wxBusyCursor wait;

    if (_forecastManager->GetMethodsNb() < 1) return false;

    asResultsForecast* forecast = _forecastManager->GetForecast(_selectedMethod, _selectedForecast);
    a1f dates = forecast->GetAnalogsDates(_selectedDate);
    a1f values = forecast->GetAnalogsValuesRaw(_selectedDate, _selectedStation);
    a1f criteria = forecast->GetAnalogsCriteria(_selectedDate);
    a1f analogNb = a1f::LinSpaced(dates.size(), 1, dates.size());

    wxString dateFormat = forecast->GetDateFormatting();

    _grid->Hide();

    // _grid->ClearGrid();
    _grid->DeleteRows(0, _grid->GetNumberRows());
    _grid->InsertRows(0, dates.size());

    if (_sortAfterCol > 0 || _sortOrder == Desc) {
        if (_sortAfterCol == 0)  // Analog nb
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&analogNb[0], &analogNb[analogNb.size() - 1], &vIndices[0], &vIndices[analogNb.size() - 1],
                         _sortOrder);

            a1f copyDates = dates;
            a1f copyValues = values;
            a1f copyCriteria = criteria;

            for (int i = 0; i < dates.size(); i++) {
                int index = vIndices(i);
                dates[i] = copyDates[index];
                values[i] = copyValues[index];
                criteria[i] = copyCriteria[index];
            }
        } else if (_sortAfterCol == 1)  // date
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&dates[0], &dates[dates.size() - 1], &vIndices[0], &vIndices[dates.size() - 1], _sortOrder);

            a1f copyAnalogNb = analogNb;
            a1f copyValues = values;
            a1f copyCriteria = criteria;

            for (int i = 0; i < dates.size(); i++) {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                values[i] = copyValues[index];
                criteria[i] = copyCriteria[index];
            }
        } else if (_sortAfterCol == 2)  // value
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&values[0], &values[values.size() - 1], &vIndices[0], &vIndices[values.size() - 1],
                         _sortOrder);

            a1f copyAnalogNb = analogNb;
            a1f copyDates = dates;
            a1f copyCriteria = criteria;

            for (int i = 0; i < values.size(); i++) {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                dates[i] = copyDates[index];
                criteria[i] = copyCriteria[index];
            }

        } else if (_sortAfterCol == 3)  // criteria
        {
            a1f vIndices = a1f::LinSpaced(dates.size(), 0, dates.size() - 1);

            asSortArrays(&criteria[0], &criteria[criteria.size() - 1], &vIndices[0], &vIndices[criteria.size() - 1],
                         _sortOrder);

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
        _grid->SetCellValue(i, 0, buf);

        buf.Printf("%s", asTime::GetStringTime(dates[i], dateFormat));
        _grid->SetCellValue(i, 1, buf);

        buf.Printf("%g", values[i]);
        _grid->SetCellValue(i, 2, buf);

        buf.Printf("%g", criteria[i]);
        _grid->SetCellValue(i, 3, buf);
    }

    _grid->Show(true);

    return true;
}
