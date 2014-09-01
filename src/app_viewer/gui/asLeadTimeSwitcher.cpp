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
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */
 
#include "asLeadTimeSwitcher.h"

wxDEFINE_EVENT(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, wxCommandEvent);

asLeadTimeSwitcher::asLeadTimeSwitcher( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
wxPanel( parent, id, pos, size, style )
{
    m_Bmp = NULL;
    m_Gdc = NULL;
    m_CellWidth = 10;

    Connect( wxEVT_PAINT, wxPaintEventHandler( asLeadTimeSwitcher::OnPaint ), NULL, this );
    Connect( wxEVT_LEFT_UP, wxMouseEventHandler( asLeadTimeSwitcher::OnLeadTimeSlctChange ), NULL, this);
    Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( asLeadTimeSwitcher::OnLeadTimeSlctChange ), NULL, this);
    Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( asLeadTimeSwitcher::OnLeadTimeSlctChange ), NULL, this);

    Layout();
}

asLeadTimeSwitcher::~asLeadTimeSwitcher()
{
    Disconnect( wxEVT_PAINT, wxPaintEventHandler( asLeadTimeSwitcher::OnPaint ), NULL, this );
    Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( asLeadTimeSwitcher::OnLeadTimeSlctChange ), NULL, this);
    Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( asLeadTimeSwitcher::OnLeadTimeSlctChange ), NULL, this);
    Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( asLeadTimeSwitcher::OnLeadTimeSlctChange ), NULL, this);

    wxDELETE(m_Bmp);
}

void asLeadTimeSwitcher::SetParent(wxWindow* parent)
{
    m_Parent = parent;
}

void asLeadTimeSwitcher::OnLeadTimeSlctChange( wxMouseEvent& event )
{
    wxPoint position = event.GetPosition();
    int val = floor(position.x/m_CellWidth);
    wxCommandEvent eventSlct (asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED);
    eventSlct.SetInt(val);
    GetParent()->ProcessWindowEvent(eventSlct);
}

void asLeadTimeSwitcher::Draw(Array1DFloat &dates, const VectorString &models, std::vector <asResultsAnalogsForecast*> forecasts)
{
    // Get sizes
    int cols = dates.size();
    int rows = models.size();
    
    // Required size
    m_CellWidth = 40;
    int width = (cols+1)*m_CellWidth;
    int height = m_CellWidth+5;

    // Get color values
    Array1DFloat values = GetValues(dates, models, forecasts);
    wxASSERT(values.size()==dates.size());

    // Create bitmap
    wxBitmap * bmp = new wxBitmap(width,height);
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
        wxFont datesFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
        gc->SetFont( datesFont, *wxBLACK );

        wxPoint startText(5, m_CellWidth/2 - 10);

        // For every lead time
        for (int i_leadtime=0; i_leadtime<dates.size(); i_leadtime++)
        {
            gc->SetPen( wxPen(GetBackgroundColour(), 3, wxSOLID ) );

            wxGraphicsPath path = gc->CreatePath();
            CreatePath(path, i_leadtime);
            FillPath(gc, path, values[i_leadtime]);

            wxString dateStr = asTime::GetStringTime(dates[i_leadtime], "DD.MM");
            gc->SetFont( datesFont, *wxBLACK );
            CreateDatesText(gc, startText, i_leadtime, dateStr);
        }

        // For the global view option
        gc->SetPen( wxPen(*wxWHITE, 1, wxSOLID ) );
        wxGraphicsPath path = gc->CreatePath();

        int segmentsTot = 7;
        const double scale = 0.16;
        wxPoint center(width-m_CellWidth/2, m_CellWidth/2);

        for (int i=0; i<segmentsTot; i++)
        {
            CreatePathRing( path, center, scale, segmentsTot, i );
        }

        gc->StrokePath(path);

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmap(bmp);
    wxDELETE(bmp);
    wxASSERT(!bmp);

    Refresh();

}

void asLeadTimeSwitcher::SetLeadTime(int leadTime)
{
    m_LeadTime = leadTime;
}

void asLeadTimeSwitcher::SetLeadTimeMarker(int leadTime)
{
    // Clear overlay
    {
		wxClientDC dc (this);
		wxDCOverlay overlaydc (m_Overlay, &dc);
		overlaydc.Clear();
	}
	m_Overlay.Reset();
    
    // Get overlay
	wxClientDC dc (this);
	wxDCOverlay overlaydc (m_Overlay, &dc);
	overlaydc.Clear();

    // Create graphics context
    wxGraphicsContext * gc = wxGraphicsContext::Create(dc);

    if ( gc && leadTime>=0 )
    {
        // Set Lead time marker
        wxGraphicsPath markerPath = gc->CreatePath();
        CreatePathMarker(markerPath, leadTime);
        gc->SetBrush(wxBrush(*wxWHITE, wxSOLID));
        gc->SetPen(wxPen(*wxBLACK, 1, wxSOLID ));
        gc->DrawPath(markerPath);

        wxDELETE(gc);
    }
}

Array1DFloat asLeadTimeSwitcher::GetValues(Array1DFloat &dates, const VectorString &models, std::vector <asResultsAnalogsForecast*> forecasts)
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    int returnPeriodRef;
    pConfig->Read("/SidebarAlarms/ReturnPeriod", &returnPeriodRef, 10);
    float percentileThreshold;
    pConfig->Read("/SidebarAlarms/Percentile", &percentileThreshold, (float)0.9);

    wxASSERT(returnPeriodRef>=2);
    wxASSERT(percentileThreshold>0);
    wxASSERT(percentileThreshold<1);
    if (returnPeriodRef<2) returnPeriodRef = 2;
    if (percentileThreshold<=0) percentileThreshold = (float)0.9;
    if (percentileThreshold>1) percentileThreshold = (float)0.9;

    Array2DFloat allValues = Array2DFloat::Ones(models.size(), dates.size());
    allValues *= NaNFloat;

    for (unsigned int i_model=0; i_model<models.size(); i_model++)
    {
        asResultsAnalogsForecast* forecast = forecasts[i_model];
        int stationsNb = forecast->GetStationsNb();

        // Get return period index
        int indexReferenceAxis=asNOT_FOUND;
        if(forecast->HasReferenceValues())
        {
            Array1DFloat forecastReferenceAxis = forecast->GetReferenceAxis();

            indexReferenceAxis = asTools::SortedArraySearch(&forecastReferenceAxis[0], &forecastReferenceAxis[forecastReferenceAxis.size()-1], returnPeriodRef);
            if ( (indexReferenceAxis==asNOT_FOUND) || (indexReferenceAxis==asOUT_OF_RANGE) )
            {
                asLogError(_("The desired return period is not available in the forecast file."));
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

            allValues(i_model,i_leadtime) = maxVal;
        }
    }

    // Extract the highest values
    Array1DFloat values = allValues.colwise().maxCoeff();

    return values;
}

void asLeadTimeSwitcher::SetBitmap(wxBitmap * bmp)
{
    wxDELETE(m_Bmp);
    wxASSERT(!m_Bmp);

    if (bmp != NULL)
    {
        wxASSERT(bmp);
        m_Bmp = new wxBitmap(*bmp);
        wxASSERT(m_Bmp);
    }
}

wxBitmap* asLeadTimeSwitcher::GetBitmap()
{
    wxASSERT(m_Bmp);

    if (m_Bmp != NULL)
    {
        return m_Bmp;
    }

    return NULL;
}

void asLeadTimeSwitcher::OnPaint(wxPaintEvent & event)
{
    if (m_Bmp != NULL)
    {
        wxPaintDC dc(this);
        dc.DrawBitmap(*m_Bmp, 0,0, true);
    }

    Layout();

    SetLeadTimeMarker(m_LeadTime);

    event.Skip();
}

void asLeadTimeSwitcher::CreatePath(wxGraphicsPath & path, int i_col)
{
    wxPoint start(0, 0);

    double startPointX = (double)start.x+i_col*m_CellWidth;

    double startPointY = (double)start.y;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint( startPointX+m_CellWidth, startPointY );
    path.AddLineToPoint( startPointX+m_CellWidth, startPointY+m_CellWidth-1 );
    path.AddLineToPoint( startPointX, startPointY+m_CellWidth-1 );
    path.AddLineToPoint( startPointX, startPointY );

    path.CloseSubpath();
}

void asLeadTimeSwitcher::CreatePathRing(wxGraphicsPath & path, const wxPoint & center, double scale, int segmentsTotNb, int segmentNb)
{
    const wxDouble radiusOut = 100*scale;
    const wxDouble radiusIn = 40*scale;

    wxDouble segmentStart = -0.5*M_PI + ((double)segmentNb/(double)segmentsTotNb)*(1.5*M_PI);
    wxDouble segmentEnd = -0.5*M_PI + ((double)(segmentNb+1)/(double)segmentsTotNb)*(1.5*M_PI);
    wxDouble centerX = (wxDouble)center.x;
    wxDouble centerY = (wxDouble)center.y;

    // Get starting point
    double dX = cos(segmentStart)*radiusOut;
    double dY = sin(segmentStart)*radiusOut;
    wxDouble startPointX = centerX+dX;
    wxDouble startPointY = centerY+dY;

    path.MoveToPoint(startPointX, startPointY);

    path.AddArc( centerX, centerY, radiusOut, segmentStart, segmentEnd, true );

    const wxDouble radiusRatio = ((radiusOut-radiusIn)/radiusOut);
    wxPoint2DDouble currentPoint = path.GetCurrentPoint();
    wxDouble newPointX = currentPoint.m_x-(currentPoint.m_x-centerX)*radiusRatio;
    wxDouble newPointY = currentPoint.m_y-(currentPoint.m_y-centerY)*radiusRatio;

    path.AddLineToPoint( newPointX, newPointY );

    path.AddArc( centerX, centerY, radiusIn, segmentEnd, segmentStart, false );

    path.CloseSubpath();
}

void asLeadTimeSwitcher::FillPath( wxGraphicsContext *gc, wxGraphicsPath & path, float value )
{
    wxColour colour;

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

    wxBrush brush(colour, wxSOLID);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asLeadTimeSwitcher::CreateDatesText( wxGraphicsContext *gc, const wxPoint& start, int i_col, const wxString &label)
{
    double pointX = (double)start.x+i_col*m_CellWidth;
    double pointY = (double)start.y;

    // Draw text
    gc->DrawText(label, pointX, pointY);
}

void asLeadTimeSwitcher::CreatePathMarker(wxGraphicsPath & path, int i_col)
{
    int outlier = 5;
    int markerHeight = 13;
    wxPoint start(m_CellWidth/2, m_CellWidth-markerHeight);

    double startPointX = (double)start.x+i_col*m_CellWidth;
    double startPointY = (double)start.y;

    path.MoveToPoint(startPointX, startPointY);

    path.AddLineToPoint( startPointX-m_CellWidth/5, startPointY+markerHeight+outlier-1 );
    path.AddLineToPoint( startPointX+m_CellWidth/5, startPointY+markerHeight+outlier-1 );
    path.AddLineToPoint( startPointX, startPointY );

    path.CloseSubpath();
}