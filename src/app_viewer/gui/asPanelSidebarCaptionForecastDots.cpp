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
 
#include "asPanelSidebarCaptionForecastDots.h"

#include "img_bullets.h"

/*
 * asPanelSidebarCaptionForecastDots
 */

asPanelSidebarCaptionForecastDots::asPanelSidebarCaptionForecastDots( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
asPanelSidebar( parent, id, pos, size, style )
{
    m_Header->SetLabelText(_("Forecast caption"));

    m_PanelDrawing = new asPanelSidebarCaptionForecastDotsDrawing( this, wxID_ANY, wxDefaultPosition, wxSize(240,50), wxTAB_TRAVERSAL );
    m_SizerContent->Add( m_PanelDrawing, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5 );

    Connect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarCaptionForecastDots::OnPaint ), NULL, this );

    Layout();
    m_SizerMain->Fit( this );
    FitInside();
}

asPanelSidebarCaptionForecastDots::~asPanelSidebarCaptionForecastDots()
{
    Disconnect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarCaptionForecastDots::OnPaint ), NULL, this );
}

void asPanelSidebarCaptionForecastDots::OnPaint(wxPaintEvent & event)
{
    event.Skip();
}

void asPanelSidebarCaptionForecastDots::SetColorbarMax(double valmax)
{
    m_PanelDrawing->DrawColorbar(valmax);
}


/*
 * asPanelSidebarCaptionForecastDotsDrawing
 */

asPanelSidebarCaptionForecastDotsDrawing::asPanelSidebarCaptionForecastDotsDrawing( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style )
:
wxPanel( parent, id, pos, size, style )
{
    m_BmpColorbar = NULL;
    m_Gdc = NULL;

    Connect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarCaptionForecastDotsDrawing::OnPaint ), NULL, this );

    DrawColorbar(1);

    Layout();
}

asPanelSidebarCaptionForecastDotsDrawing::~asPanelSidebarCaptionForecastDotsDrawing()
{
    Disconnect( wxEVT_PAINT, wxPaintEventHandler( asPanelSidebarCaptionForecastDotsDrawing::OnPaint ), NULL, this );
    wxDELETE(m_BmpColorbar);
}

void asPanelSidebarCaptionForecastDotsDrawing::DrawColorbar(double valmax)
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

void asPanelSidebarCaptionForecastDotsDrawing::SetBitmapColorbar(wxBitmap * bmp)
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

void asPanelSidebarCaptionForecastDotsDrawing::OnPaint(wxPaintEvent & event)
{
    wxPaintDC dc(this);

    if (m_BmpColorbar != NULL)
    {
        dc.DrawBitmap(*m_BmpColorbar, 0,0, true);
    }

    Layout();

    event.Skip();
}

void asPanelSidebarCaptionForecastDotsDrawing::CreateColorbarPath(wxGraphicsPath & path )
{
    path.MoveToPoint(30,1);
    path.AddLineToPoint(210, 1);
    path.AddLineToPoint(210, 11);
    path.AddLineToPoint(30, 11);
    path.CloseSubpath();
}

void asPanelSidebarCaptionForecastDotsDrawing::FillColorbar(wxGraphicsContext * gc, wxGraphicsPath & path )
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

void asPanelSidebarCaptionForecastDotsDrawing::CreateColorbarText( wxGraphicsContext * gc, wxGraphicsPath & path, double valmax)
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

void asPanelSidebarCaptionForecastDotsDrawing::CreateColorbarOtherClasses(wxGraphicsContext * gc, wxGraphicsPath & path )
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
