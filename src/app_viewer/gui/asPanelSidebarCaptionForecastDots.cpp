/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asPanelSidebarCaptionForecastDots.h"

#include "img_bullets.h"

asPanelSidebarCaptionForecastDots::asPanelSidebarCaptionForecastDots( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Forecast caption"));

    m_BmpColorbar = NULL;
    m_Gdc = NULL;

    m_PanelDrawing = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxSize(240,50), wxTAB_TRAVERSAL );
	m_SizerContent->Add( m_PanelDrawing, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5 );
	m_SizerContent->Fit(this);

    Connect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarCaptionForecastDots::OnPaint ), NULL, this );

    DrawColorbar(1);

    m_PanelDrawing->Layout();
    Layout();
    m_SizerMain->Fit( this );
}

asPanelSidebarCaptionForecastDots::~asPanelSidebarCaptionForecastDots()
{
    Disconnect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarCaptionForecastDots::OnPaint ), NULL, this );
    wxDELETE(m_BmpColorbar);
}

void asPanelSidebarCaptionForecastDots::SetColorbarMax(double valmax)
{
    this->DrawColorbar(valmax);
}

void asPanelSidebarCaptionForecastDots::DrawColorbar(double valmax)
{
    wxBitmap * bmp = new wxBitmap(240,50);
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

    if (gc)
    {
        gc->SetPen( *wxBLACK );
        wxFont defFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
        gc->SetFont( defFont, *wxBLACK );
        wxGraphicsPath path = gc->CreatePath();

        CreateColorbarPath( path );
        FillColorbar( gc, path );
        CreateColorbarText( gc, path, valmax );
        CreateColorbarOtherClasses(gc, path );

        wxDELETE(gc);
    }

    dc.SelectObject(wxNullBitmap);

    this->SetBitmapColorbar(bmp);
	wxDELETE(bmp);
	wxASSERT(!bmp);

	Refresh();
}

void asPanelSidebarCaptionForecastDots::SetBitmapColorbar(wxBitmap * bmp)
{
	wxDELETE(m_BmpColorbar);
	wxASSERT(!m_BmpColorbar);

	if (bmp != NULL)
    {
        wxASSERT(bmp);
		m_BmpColorbar = new wxBitmap(*bmp);
		wxASSERT(m_BmpColorbar);
	}
}

void asPanelSidebarCaptionForecastDots::OnPaint(wxPaintEvent & event)
{
    wxASSERT(m_PanelDrawing);

	wxPaintDC dc(m_PanelDrawing);

    if (m_BmpColorbar != NULL)
    {
		dc.DrawBitmap(*m_BmpColorbar, 0,0, true);
	}

	Layout();

	event.Skip();
}

void asPanelSidebarCaptionForecastDots::CreateColorbarPath(wxGraphicsPath & path )
{
    path.MoveToPoint(30,1);
    path.AddLineToPoint(210, 1);
    path.AddLineToPoint(210, 11);
    path.AddLineToPoint(30, 11);
    path.CloseSubpath();
}

void asPanelSidebarCaptionForecastDots::FillColorbar(wxGraphicsContext * gc, wxGraphicsPath & path )
{
    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    wxGraphicsGradientStops stops( wxColour(200,255,200), wxColour(255,0,0) );
    stops.Add(wxColour(255,255,0), 0.5);
    wxGraphicsBrush brush = gc->CreateLinearGradientBrush(x, y, x+w, y, stops);
    gc->SetBrush(brush);
    gc->DrawPath(path);
}

void asPanelSidebarCaptionForecastDots::CreateColorbarText( wxGraphicsContext * gc, wxGraphicsPath & path, double valmax)
{
    gc->SetPen( *wxBLACK );

    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    // Correction of the coordinates needed on Linux
    #if defined(__WXMSW__)
        int corr = 0;
    #elif defined(__WXMAC__)
        int corr = 0;
    #elif defined(__UNIX__)
        int corr = 1;
    #endif

    // Set ticks
    wxGraphicsPath pathTickStart = gc->CreatePath();
    pathTickStart.MoveToPoint(x+corr,y+corr);
    pathTickStart.AddLineToPoint(x+corr,y+corr+h+5);
    gc->StrokePath(pathTickStart);
    wxGraphicsPath pathTickMid = gc->CreatePath();
    pathTickMid.MoveToPoint(x+w/2,y+corr);
    pathTickMid.AddLineToPoint(x+w/2,y+corr+h+5);
    gc->StrokePath(pathTickMid);
    wxGraphicsPath pathTickEnd = gc->CreatePath();
    pathTickEnd.MoveToPoint(x-corr+w,y+corr);
    pathTickEnd.AddLineToPoint(x-corr+w,y+corr+h+5);
    gc->StrokePath(pathTickEnd);

    // Set labels
    wxString labelStart = "0";
    wxString labelMid = wxString::Format("%g", valmax/2.0);
    wxString labelEnd = wxString::Format("%g", valmax);

    // Draw text
    gc->DrawText(labelStart, x+4, y+12);
    gc->DrawText(labelMid, x+w/2+4, y+12);
    gc->DrawText(labelEnd, x+w+4, y+12);

}

void asPanelSidebarCaptionForecastDots::CreateColorbarOtherClasses(wxGraphicsContext * gc, wxGraphicsPath & path )
{
    gc->SetPen( *wxBLACK );

    // Get the path box
    wxDouble x, y, w, h;
    path.GetBox(&x, &y, &w, &h);

    // Create first box
    wxGraphicsPath pathBox1 = gc->CreatePath();
    pathBox1.MoveToPoint(x,y+h+20);
    pathBox1.AddLineToPoint(x,y+h+20+10);
    pathBox1.AddLineToPoint(x+10,y+h+20+10);
    pathBox1.AddLineToPoint(x+10,y+h+20);
    pathBox1.CloseSubpath();

    wxColour colour = wxColour();
    colour.Set(255,255,255);
    wxBrush brush1(colour, wxSOLID);
    gc->SetBrush(brush1);
    gc->DrawPath(pathBox1);

    // Create second box
    wxGraphicsPath pathBox2 = gc->CreatePath();
    pathBox2.MoveToPoint(x+w/2,y+h+20);
    pathBox2.AddLineToPoint(x+w/2,y+h+20+10);
    pathBox2.AddLineToPoint(x+w/2+10,y+h+20+10);
    pathBox2.AddLineToPoint(x+w/2+10,y+h+20);
    pathBox2.CloseSubpath();

    colour.Set(150,150,150);
    wxBrush brush2(colour, wxSOLID);
    gc->SetBrush(brush2);
    gc->DrawPath(pathBox2);

    // Set labels
    wxString label1 = _("No rainfall");
    wxString label2 = _("Missing data");

    // Draw text
    gc->DrawText(label1, x+14,y+h+19);
    gc->DrawText(label2, x+w/2+14,y+h+19);
}
