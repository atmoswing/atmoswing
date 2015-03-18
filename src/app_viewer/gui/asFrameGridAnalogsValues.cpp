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
 
#include "asFrameGridAnalogsValues.h"

#include "asForecastManager.h"
#include "asResultsAnalogsForecast.h"


asFrameGridAnalogsValues::asFrameGridAnalogsValues( wxWindow* parent, int methodRow, int forecastRow, asForecastManager *forecastManager, wxWindowID id )
:
asFrameGridAnalogsValuesVirtual( parent )
{
    forecastRow = wxMax(forecastRow, 0);

    m_forecastManager = forecastManager;
    m_selectedMethod = methodRow;
    m_selectedForecast = forecastRow;
    m_selectedStation = 0;
    m_selectedDate = 0;
    m_sortAfterCol = 0;
    m_sortOrder = Asc;

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFrameGridAnalogsValues::Init()
{
    // Forecast list
    wxArrayString arrayForecasts = m_forecastManager->GetAllForecastNamesWxArray();
    m_choiceForecast->Set(arrayForecasts);
    int linearIndex = m_forecastManager->GetLinearIndex(m_selectedMethod, m_selectedForecast);
    m_choiceForecast->Select(linearIndex);

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
    m_grid->SetColFormatFloat(2,-1,1);
    m_grid->SetColFormatFloat(3,-1,3);
    UpdateGrid();
}

void asFrameGridAnalogsValues::OnChoiceForecastChange( wxCommandEvent& event )
{
    int linearIndex = event.GetInt();
    m_selectedMethod = m_forecastManager->GetMethodRowFromLinearIndex(linearIndex);
    m_selectedForecast = m_forecastManager->GetForecastRowFromLinearIndex(linearIndex);

    // Dates list
    wxArrayString arrayDates = m_forecastManager->GetLeadTimes(m_selectedMethod, m_selectedForecast);
    m_choiceDate->Set(arrayDates);
    if (arrayDates.size()<=(unsigned)m_selectedDate)
    {
        m_selectedDate = 0;
    }
    m_choiceDate->Select(m_selectedDate);

    // Stations list
    wxArrayString arrayStation = m_forecastManager->GetStationNamesWithHeights(m_selectedMethod, m_selectedForecast);
    m_choiceStation->Set(arrayStation);
    if (arrayStation.size()<=(unsigned)m_selectedStation)
    {
        m_selectedStation = 0;
    }
    m_choiceStation->Select(m_selectedStation);

    UpdateGrid();
}

void asFrameGridAnalogsValues::OnChoiceStationChange( wxCommandEvent& event )
{
    m_selectedStation = event.GetInt();

    UpdateGrid(); // Doesn't change for criteria
}

void asFrameGridAnalogsValues::OnChoiceDateChange( wxCommandEvent& event )
{
    m_selectedDate = event.GetInt();

    UpdateGrid();
}

void asFrameGridAnalogsValues::SortGrid( wxGridEvent& event )
{
    // On a row label
    if (event.GetCol() == -1)
    {
        event.Skip();
        return;
    }

    // Check if twice on the same col
    if (m_sortAfterCol == event.GetCol())
    {
        if (m_sortOrder==Asc)
        {
            m_sortOrder = Desc;
        }
        else
        {
            m_sortOrder = Asc;
        }
    }
    else
    {
        m_sortOrder = Asc;
    }
    m_sortAfterCol = event.GetCol();

    UpdateGrid();
}

bool asFrameGridAnalogsValues::UpdateGrid()
{
    if (m_forecastManager->GetMethodsNb()<1) return false;

    asResultsAnalogsForecast* forecast = m_forecastManager->GetForecast(m_selectedMethod, m_selectedForecast);
    Array1DFloat dates = forecast->GetAnalogsDates(m_selectedDate);
    Array1DFloat values = forecast->GetAnalogsValuesGross(m_selectedDate, m_selectedStation);
    Array1DFloat criteria = forecast->GetAnalogsCriteria(m_selectedDate);
    Array1DFloat analogNb = Array1DFloat::LinSpaced(dates.size(),1,dates.size());

    m_grid->Hide();

    //m_grid->ClearGrid();
    m_grid->DeleteRows(0, m_grid->GetRows());
    m_grid->InsertRows(0, dates.size());

    if (m_sortAfterCol>0 || m_sortOrder==Desc)
    {
        if (m_sortAfterCol==0) // Analog nb
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&analogNb[0], &analogNb[analogNb.size()-1],
                        &vIndices[0], &vIndices[analogNb.size()-1],
                        m_sortOrder);

            Array1DFloat copyDates = dates;
            Array1DFloat copyValues = values;
            Array1DFloat copyCriteria = criteria;

            for (int i=0; i<dates.size(); i++)
            {
                int index = vIndices(i);
                dates[i] = copyDates[index];
                values[i] = copyValues[index];
                criteria[i] = copyCriteria[index];
            }
        }
        else if (m_sortAfterCol==1) // date
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&dates[0], &dates[dates.size()-1],
                        &vIndices[0], &vIndices[dates.size()-1],
                        m_sortOrder);

            Array1DFloat copyAnalogNb = analogNb;
            Array1DFloat copyValues = values;
            Array1DFloat copyCriteria = criteria;

            for (int i=0; i<dates.size(); i++)
            {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                values[i] = copyValues[index];
                criteria[i] = copyCriteria[index];
            }
        }
        else if (m_sortAfterCol==2) // value
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&values[0], &values[values.size()-1],
                        &vIndices[0], &vIndices[values.size()-1],
                        m_sortOrder);

            Array1DFloat copyAnalogNb = analogNb;
            Array1DFloat copyDates = dates;
            Array1DFloat copyCriteria = criteria;

            for (int i=0; i<values.size(); i++)
            {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                dates[i] = copyDates[index];
                criteria[i] = copyCriteria[index];
            }

        }
        else if (m_sortAfterCol==3) // criteria
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&criteria[0], &criteria[criteria.size()-1],
                        &vIndices[0], &vIndices[criteria.size()-1],
                        m_sortOrder);

            Array1DFloat copyAnalogNb = analogNb;
            Array1DFloat copyValues = values;
            Array1DFloat copyDates = dates;

            for (int i=0; i<dates.size(); i++)
            {
                int index = vIndices(i);
                analogNb[i] = copyAnalogNb[index];
                values[i] = copyValues[index];
                dates[i] = copyDates[index];
            }
        }
    }

    for (int i=0; i<dates.size(); i++)
    {
        wxString buf;
        buf.Printf("%d", (int)analogNb[i]);
        m_grid->SetCellValue(i,0,buf);

        buf.Printf("%s", asTime::GetStringTime(dates[i], "DD.MM.YYYY"));
        m_grid->SetCellValue(i,1,buf);

        buf.Printf("%g", values[i]);
        m_grid->SetCellValue(i,2,buf);

        buf.Printf("%g", criteria[i]);
        m_grid->SetCellValue(i,3,buf);
    }

    m_grid->Show();

    return true;
}
