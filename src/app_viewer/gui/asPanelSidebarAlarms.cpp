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
 
#include "asPanelSidebarAlarms.h"

#include "img_bullets.h"


asPanelSidebarAlarms::asPanelSidebarAlarms( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Alarms"));

    m_BmpAlarms = NULL;
    m_Gdc = NULL;
    m_PanelDrawing = NULL;
    m_Mode = 1; // 1: values
                // 2: thresholds -> not available yet

    Connect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarAlarms::OnPaint ), NULL, this );

    Array1DFloat emptyArray;
    Array2DFloat empty2DArray;
    VectorString emptyStringArr;

    Layout();
}

asPanelSidebarAlarms::~asPanelSidebarAlarms()
{
    Disconnect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarAlarms::OnPaint ), NULL, this );
    wxDELETE(m_BmpAlarms);
}

void asPanelSidebarAlarms::SetData(Array1DFloat &dates, const VectorString &models, Array2DFloat &values)
{
    DrawAlarms(dates, models, values);
}

void asPanelSidebarAlarms::DrawAlarms(Array1DFloat &dates, const VectorString &models, Array2DFloat &values)
{
    // Get sizes
	int cols = dates.size();
	int rows = models.size();
	wxASSERT_MSG( (values.cols()==cols) , wxString::Format("values.cols()=%d, cols=%d", (int)values.cols(), cols));
	wxASSERT_MSG( (values.rows()==rows) , wxString::Format("values.rows()=%d, rows=%d", (int)values.rows(), rows));

	// Height of a grid row
	int cellHeight = 20;

    // Create bitmap
    int totHeight = cellHeight*rows+20;
    wxBitmap * bmp = new wxBitmap(240,totHeight);
    wxASSERT(bmp);

    // Create device context
	wxMemoryDC dc (*bmp);
	dc.SetBackground(wxBrush(GetBackgroundColour()));
	#if defined(__UNIX__)
        dc.SetBackground(wxBrush(g_LinuxBgColour));
    #endif
	dc.Clear();

    // Create graphics context
	wxGraphicsContext * gc = wxGraphicsContext::Create(dc);

    if ( gc && cols>0 && rows>0 )
    {
        gc->SetPen( *wxBLACK );
        wxFont datesFont(6, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
        wxFont numFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
        gc->SetFont( datesFont, *wxBLACK );

        wxPoint startText(17, 0);
        wxPoint startNb(0, 14);
        wxPoint startGrid(12, 10);
        int cellWitdh = 226/dates.size();

        for (int i_leadtime=0; i_leadtime<dates.size(); i_leadtime++)
        {
            wxString dateStr = asTime::GetStringTime(dates[i_leadtime], "DD.MM");
            gc->SetFont( datesFont, *wxBLACK );
            CreateDatesText(gc, startText, cellWitdh, i_leadtime, dateStr);

            for (int i_model=0; (unsigned)i_model<models.size(); i_model++)
            {
                if (i_leadtime==0)
                {
                    wxString modelStr = wxString::Format("%d", i_model+1);
                    gc->SetFont( numFont, *wxBLACK );
                    CreateNbText(gc, startNb, cellHeight, i_model, modelStr);
                }

                wxGraphicsPath path = gc->CreatePath();
                CreatePath(path, startGrid, cellWitdh, cellHeight, i_leadtime, i_model, cols, rows);
                float value = values(i_model,i_leadtime);
                FillPath(gc, path, value);
            }
        }

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmapAlarms(bmp);
	wxDELETE(bmp);
	wxASSERT(!bmp);

	Refresh();
}

void asPanelSidebarAlarms::SetBitmapAlarms(wxBitmap * bmp)
{
    wxWindow* topFrame = GetTopFrame(this);
    topFrame->Freeze();

	wxDELETE(m_BmpAlarms);
	wxASSERT(!m_BmpAlarms);

	if (bmp != NULL)
    {
        wxASSERT(bmp);
		m_BmpAlarms = new wxBitmap(*bmp);
		wxASSERT(m_BmpAlarms);
	}

	if (m_PanelDrawing != NULL)
    {
        m_PanelDrawing->Destroy();
    }

    wxSize panelSize = m_BmpAlarms->GetSize();

    m_PanelDrawing = new wxPanel( this, wxID_ANY, wxDefaultPosition, panelSize, wxTAB_TRAVERSAL );
    wxASSERT(m_PanelDrawing);
    m_SizerContent->Add( m_PanelDrawing, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5 );
    m_PanelDrawing->Layout();

    // Refresh elements
    m_SizerMain->Layout();
    Layout();
    GetSizer()->Fit(GetParent());
    topFrame->Layout();
    topFrame->Refresh();

    topFrame->Thaw();
}

void asPanelSidebarAlarms::OnPaint(wxPaintEvent & event)
{
	if (m_BmpAlarms != NULL)
    {
        wxASSERT(m_PanelDrawing);
        wxPaintDC dc(m_PanelDrawing);
		dc.DrawBitmap(*m_BmpAlarms, 0,0, true);
	}

    Layout();

	event.Skip();
}

void asPanelSidebarAlarms::CreatePath(wxGraphicsPath & path, const wxPoint & start, int cellWitdh, int cellHeight, int i_col, int i_row, int cols, int rows)
{
    double startPointX = (double)start.x+i_col*cellWitdh;

    double startPointY = (double)start.y+i_row*cellHeight;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint( startPointX+cellWitdh, startPointY );
    path.AddLineToPoint( startPointX+cellWitdh, startPointY+cellHeight );
    path.AddLineToPoint( startPointX, startPointY+cellHeight );
    path.AddLineToPoint( startPointX, startPointY );

    path.CloseSubpath();
}

void asPanelSidebarAlarms::FillPath( wxGraphicsContext *gc, wxGraphicsPath & path, float value )
{
    wxColour colour;

    switch (m_Mode)
    {
        case (1):
        {
            if (asTools::IsNaN(value)) // NaN -> gray
            {
                colour.Set(150,150,150);
            }
            else if (value==0) // No rain -> white
            {
                colour.Set(255,255,255);
            }
            else if ( value<=0.5 ) // light green to yellow
            {
                int baseVal = 200;
                int valColour = ((value/(0.5)))*baseVal;
                int valColourCompl = ((value/(0.5)))*(255-baseVal);
                if (valColour>baseVal) valColour = baseVal;
                if (valColourCompl+baseVal>255) valColourCompl = 255-baseVal;
                colour.Set((baseVal+valColourCompl),255,(baseVal-valColour));
            }
            else // Yellow to red
            {
                int valColour = ((value-0.5)/(0.5))*255;
                if (valColour>255) valColour = 255;
                colour.Set(255,(255-valColour),0);
            }
            break;
        }

        case (2):
        {
            if ( value==1 ) // Green
            {
                colour.Set(200,255,200);
            }
            else if ( value==2 ) // Yellow
            {
                colour.Set(255,255,118);
            }
            else if ( value==3 ) // Red
            {
                colour.Set(255,80,80);
            }
            else
            {
                colour.Set(150,150,150);
            }
            break;
        }
    }

    wxBrush brush(colour, wxSOLID);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asPanelSidebarAlarms::CreateDatesText( wxGraphicsContext *gc, const wxPoint& start, int cellWitdh, int i_col, const wxString &label)
{
    double pointX = (double)start.x+i_col*cellWitdh;
    double pointY = (double)start.y;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}

void asPanelSidebarAlarms::CreateNbText( wxGraphicsContext *gc, const wxPoint& start, int cellHeight, int i_row, const wxString &label)
{
    double pointX = (double)start.x;
    double pointY = (double)start.y+i_row*cellHeight;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}

void asPanelSidebarAlarms::UpdateAlarms(Array1DFloat &dates, VectorString &models, std::vector <asResultsAnalogsForecast*> forecasts)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    int returnPeriodRef;
    pConfig->Read("/SidebarAlarms/ReturnPeriod", &returnPeriodRef, 10);
    float percentileThreshold;
    pConfig->Read("/SidebarAlarms/Percentile", &percentileThreshold, (float)0.9);

    wxString returnPeriodStr = pConfig->Read("/SidebarAlarms/ReturnPeriod", "??");
    wxString percentileThresholdStr = pConfig->Read("/SidebarAlarms/Percentile", "??");
    wxString spec = wxString::Format("T=%s, q=%s", returnPeriodStr.c_str(), percentileThresholdStr.c_str());
    m_Header->SetLabelText(wxString::Format(_("Alarms (%s)"), spec.c_str()));

    switch (m_Mode)
    {
        case (1):
        {
            wxASSERT(returnPeriodRef>=2);
            wxASSERT(percentileThreshold>0);
            wxASSERT(percentileThreshold<1);
            if (returnPeriodRef<2) returnPeriodRef = 2;
            if (percentileThreshold<=0) percentileThreshold = (float)0.9;
            if (percentileThreshold>1) percentileThreshold = (float)0.9;

            Array2DFloat values = Array2DFloat::Ones(models.size(), dates.size());
            values *= NaNFloat;

            for (unsigned int i_model=0; i_model<models.size(); i_model++)
            {
                asResultsAnalogsForecast* forecast = forecasts[i_model];
                int stationsNb = forecast->GetStationsNb();

                // Get return period index
                int indexReferenceAxis;
				if(forecast->HasReferenceValues())
				{
					Array1DFloat forecastReferenceAxis = forecast->GetReferenceAxis();

					indexReferenceAxis = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriodRef);
					if ( (indexReferenceAxis==asNOT_FOUND) || (indexReferenceAxis==asOUT_OF_RANGE) )
					{
						asLogError(_("The desired return period is not available in the forecast file."));
						return;
					}
				}

                // Check lead times effectively available for the current model
                int leadtimeMin = 0;
                int leadtimeMax = dates.size()-1;

                Array1DFloat availableDates = forecast->GetTargetDates();

                while (dates[leadtimeMin]<availableDates[0])
                {
                    leadtimeMin++;
                }
                while (dates[leadtimeMax]>availableDates[availableDates.size()-1])
                {
                    leadtimeMax--;
                }
                wxASSERT(leadtimeMin<leadtimeMax);

                for (int i_leadtime=leadtimeMin; i_leadtime<=leadtimeMax; i_leadtime++)
                {
                    float maxVal = 0;

                    // Get the maximum value of every station
                    for (int i_station=0; i_station<stationsNb; i_station++)
                    {
                        float thisVal = 0;

                        // Get values for return period
                        float factor = 1;
						if(forecast->HasReferenceValues())
						{
							float precip = forecast->GetReferenceValue(i_station, indexReferenceAxis);
							wxASSERT(precip>0);
							wxASSERT(precip<500);
							factor = 1.0/precip;
							wxASSERT(factor>0);
						}

                        // Get values
                        Array1DFloat theseVals = forecast->GetAnalogsValuesGross(i_leadtime, i_station);

                        // Process percentiles
                        if(asTools::HasNaN(&theseVals[0], &theseVals[theseVals.size()-1]))
                        {
                            thisVal = NaNFloat;
                        }
                        else
                        {
                            float forecastVal = asTools::Percentile(&theseVals[0], &theseVals[theseVals.size()-1], percentileThreshold);
                            wxASSERT_MSG(forecastVal>=0, wxString::Format("Forecast value = %g", forecastVal));
                            forecastVal *= factor;
                            thisVal = forecastVal;
                        }

                        // Keep it if higher
                        if (thisVal>maxVal)
                        {
                            maxVal = thisVal;
                        }
                    }

                    values(i_model,i_leadtime) = maxVal;
                }
            }

            SetData(dates, models, values);
            break;
        }

        case (2):
        {
            // Get display option
            float percentileThreshold1 = (float) 0.9;
            float returnPeriodThreshold1 = 2;
            float percentileThreshold2 = (float) 0.9;
            float returnPeriodThreshold2 = 5;

            // Checks of the ranges
            if ( (percentileThreshold1>1) || (percentileThreshold1<0) || (percentileThreshold2>1) || (percentileThreshold2<0) )
            {
                asLogError(_("The given percentile thresholds for the alarms are outside the allowed range."));
                return;
            }
            if ( (returnPeriodThreshold1>500) || (returnPeriodThreshold1<0) || (returnPeriodThreshold2>500) || (returnPeriodThreshold2<0) )
            {
                asLogError(_("The given return periods thresholds for the alarms are outside the allowed range."));
                return;
            }

            Array2DFloat values = Array2DFloat::Ones(models.size(), dates.size());
            values *= NaNFloat;

            for (unsigned int i_model=0; i_model<models.size(); i_model++)
            {
                asResultsAnalogsForecast* forecast = forecasts[i_model];
                int stationsNb = forecast->GetStationsNb();

                // Get return period index
                int indexReferenceAxis1, indexReferenceAxis2;
				if(forecast->HasReferenceValues())
				{
					if ( (returnPeriodThreshold1>0) && (returnPeriodThreshold2>0) )
					{
						Array1DFloat forecastReferenceAxis = forecast->GetReferenceAxis();

						indexReferenceAxis1 = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriodThreshold1);
						indexReferenceAxis2 = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriodThreshold2);
						if ( (indexReferenceAxis1==asNOT_FOUND) || (indexReferenceAxis1==asOUT_OF_RANGE) || (indexReferenceAxis2==asNOT_FOUND) || (indexReferenceAxis2==asOUT_OF_RANGE) )
						{
							asLogError(_("The desired return period is not available in the forecast file."));
							return;
						}
					}
				}

                // Check lead times effectively available for the current model
                int leadtimeMin = 0;
                int leadtimeMax = dates.size()-1;

                Array1DFloat availableDates = forecast->GetTargetDates();

                while (dates[leadtimeMin]<availableDates[0])
                {
                    leadtimeMin++;
                }
                while (dates[leadtimeMax]>availableDates[availableDates.size()-1])
                {
                    leadtimeMax--;
                }
                wxASSERT(leadtimeMin<leadtimeMax);

                for (int i_leadtime=leadtimeMin; i_leadtime<=leadtimeMax; i_leadtime++)
                {
                    float maxVal1 = 0;
                    float maxVal2 = 0;

                    // Get the maximum value of every station
                    for (int i_station=0; i_station<stationsNb; i_station++)
                    {
                        float thisVal1 = 0;
                        float thisVal2 = 0;

                        // Get values for return period
                        float factor1 = 1;
                        float factor2 = 1;
						if(forecast->HasReferenceValues())
						{
							if ( (returnPeriodThreshold1>0) && (returnPeriodThreshold2>0) )
							{
								float precip1 = forecast->GetReferenceValue(i_station, indexReferenceAxis1);
								float precip2 = forecast->GetReferenceValue(i_station, indexReferenceAxis2);
								wxASSERT(precip1>0);
								wxASSERT(precip2>0);
								wxASSERT(precip1<500);
								wxASSERT(precip2<500);
								factor1 = 1.0/precip1;
								factor2 = 1.0/precip2;
								wxASSERT(factor1>0);
								wxASSERT(factor2>0);
							}
						}

                        // Get values
                        Array1DFloat theseVals = forecast->GetAnalogsValuesGross(i_leadtime, i_station);

                        // Process percentiles
                        if(asTools::HasNaN(&theseVals[0], &theseVals[theseVals.size()-1]))
                        {
                            thisVal1 = NaNFloat;
                            thisVal2 = NaNFloat;
                        }
                        else
                        {
                            if ( (percentileThreshold1>=0) && (percentileThreshold2>=0) )
                            {
                                float forecastVal1 = asTools::Percentile(&theseVals[0], &theseVals[theseVals.size()-1], percentileThreshold1);
                                float forecastVal2 = asTools::Percentile(&theseVals[0], &theseVals[theseVals.size()-1], percentileThreshold2);
                                wxASSERT_MSG(forecastVal1>=0, wxString::Format("Forecast value = %g", forecastVal1));
                                wxASSERT_MSG(forecastVal2>=0, wxString::Format("Forecast value = %g", forecastVal2));
                                forecastVal1 *= factor1;
                                forecastVal2 *= factor2;
                                thisVal1 = forecastVal1;
                                thisVal2 = forecastVal2;
                            }
                            else
                            {
                                thisVal1 = NaNFloat;
                                thisVal2 = NaNFloat;
                            }
                        }

                        // Keep it if higher
                        if (thisVal1>maxVal1)
                        {
                            maxVal1 = thisVal1;
                        }
                        if (thisVal2>maxVal2)
                        {
                            maxVal2 = thisVal2;
                        }
                    }

                    // Apply the rules
                    if (maxVal2>1) // Most critical
                    {
                        values(i_model,i_leadtime) = 3;
                    }
                    else if (maxVal1>1)
                    {
                        values(i_model,i_leadtime) = 2;
                    }
                    else
                    {
                        values(i_model,i_leadtime) = 1;
                    }
                }
            }

            SetData(dates, models, values);
        }
    }
}
