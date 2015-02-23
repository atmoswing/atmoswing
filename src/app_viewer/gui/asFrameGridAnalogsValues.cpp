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

    m_ForecastManager = forecastManager;
    m_SelectedMethod = methodRow;
    m_SelectedForecast = forecastRow;
    m_SelectedStation = 0;
    m_SelectedDate = 0;
    m_SortAfterCol = 0;
    m_SortOrder = Asc;

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFrameGridAnalogsValues::Init()
{
    // Forecast list
    wxArrayString arrayForecasts = m_ForecastManager->GetAllForecastNamesWxArray();
    m_ChoiceForecast->Set(arrayForecasts);
    m_ChoiceForecast->Select(m_SelectedForecast);

    // Dates list
    wxArrayString arrayDates = m_ForecastManager->GetLeadTimes(m_SelectedMethod, m_SelectedForecast);
    m_ChoiceDate->Set(arrayDates);
    m_ChoiceDate->Select(m_SelectedDate);

    // Stations list
    wxArrayString arrayStation = m_ForecastManager->GetStationNamesWithHeights(m_SelectedMethod, m_SelectedForecast);
    m_ChoiceStation->Set(arrayStation);
    m_ChoiceStation->Select(m_SelectedStation);

    // Set grid
    m_Grid->SetColFormatNumber(0);
    m_Grid->SetColFormatFloat(2,-1,1);
    m_Grid->SetColFormatFloat(3,-1,3);
    UpdateGrid();
}

void asFrameGridAnalogsValues::OnChoiceForecastChange( wxCommandEvent& event )
{
    m_SelectedForecast = event.GetInt();

    // Dates list
    wxArrayString arrayDates = m_ForecastManager->GetLeadTimes(m_SelectedMethod, m_SelectedForecast);
    m_ChoiceDate->Set(arrayDates);
    if (arrayDates.size()<=(unsigned)m_SelectedDate)
    {
        m_SelectedDate = 0;
    }
    m_ChoiceDate->Select(m_SelectedDate);

    // Stations list
    wxArrayString arrayStation = m_ForecastManager->GetStationNamesWithHeights(m_SelectedMethod, m_SelectedForecast);
    m_ChoiceStation->Set(arrayStation);
    if (arrayStation.size()<=(unsigned)m_SelectedStation)
    {
        m_SelectedStation = 0;
    }
    m_ChoiceStation->Select(m_SelectedStation);

    UpdateGrid();
}

void asFrameGridAnalogsValues::OnChoiceStationChange( wxCommandEvent& event )
{
    m_SelectedStation = event.GetInt();

    UpdateGrid(); // Doesn't change for criteria
}

void asFrameGridAnalogsValues::OnChoiceDateChange( wxCommandEvent& event )
{
    m_SelectedDate = event.GetInt();

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
    if (m_SortAfterCol == event.GetCol())
    {
        if (m_SortOrder==Asc)
        {
            m_SortOrder = Desc;
        }
        else
        {
            m_SortOrder = Asc;
        }
    }
    else
    {
        m_SortOrder = Asc;
    }
    m_SortAfterCol = event.GetCol();

    UpdateGrid();
}

bool asFrameGridAnalogsValues::UpdateGrid()
{
    if (m_ForecastManager->GetMethodsNb()<1) return false;

    asResultsAnalogsForecast* forecast = m_ForecastManager->GetForecast(m_SelectedMethod, m_SelectedForecast);
    Array1DFloat dates = forecast->GetAnalogsDates(m_SelectedDate);
    Array1DFloat values = forecast->GetAnalogsValuesGross(m_SelectedDate, m_SelectedStation);
    Array1DFloat criteria = forecast->GetAnalogsCriteria(m_SelectedDate);
    Array1DFloat analogNb = Array1DFloat::LinSpaced(dates.size(),1,dates.size());

    m_Grid->Hide();

    //m_Grid->ClearGrid();
    m_Grid->DeleteRows(0, m_Grid->GetRows());
    m_Grid->InsertRows(0, dates.size());

    if (m_SortAfterCol>0 || m_SortOrder==Desc)
    {
        if (m_SortAfterCol==0) // Analog nb
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&analogNb[0], &analogNb[analogNb.size()-1],
                        &vIndices[0], &vIndices[analogNb.size()-1],
                        m_SortOrder);

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
        else if (m_SortAfterCol==1) // date
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&dates[0], &dates[dates.size()-1],
                        &vIndices[0], &vIndices[dates.size()-1],
                        m_SortOrder);

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
        else if (m_SortAfterCol==2) // value
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&values[0], &values[values.size()-1],
                        &vIndices[0], &vIndices[values.size()-1],
                        m_SortOrder);

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
        else if (m_SortAfterCol==3) // criteria
        {
            Array1DFloat vIndices = Array1DFloat::LinSpaced(Eigen::Sequential,dates.size(),
                            0,dates.size()-1);

            asTools::SortArrays(&criteria[0], &criteria[criteria.size()-1],
                        &vIndices[0], &vIndices[criteria.size()-1],
                        m_SortOrder);

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
        m_Grid->SetCellValue(i,0,buf);

        buf.Printf("%s", asTime::GetStringTime(dates[i], "DD.MM.YYYY").c_str());
        m_Grid->SetCellValue(i,1,buf);

        buf.Printf("%g", values[i]);
        m_Grid->SetCellValue(i,2,buf);

        buf.Printf("%g", criteria[i]);
        m_Grid->SetCellValue(i,3,buf);
    }

    m_Grid->Show();

    return true;
}
