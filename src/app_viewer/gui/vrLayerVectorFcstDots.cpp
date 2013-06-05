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
 

#include "vrLayerVectorFcstDots.h"
#include "vrlabel.h"
#include "vrrender.h"
#include "asTools.h"


vrLayerVectorFcstDots::vrLayerVectorFcstDots()
{
	wxASSERT(m_Dataset==NULL);
	wxASSERT(m_Layer==NULL);
	m_DriverType = vrDRIVER_VECTOR_MEMORY;
	m_ValueMax = 1;
}

vrLayerVectorFcstDots::~vrLayerVectorFcstDots()
{
}

long vrLayerVectorFcstDots::AddFeature(OGRGeometry * geometry, void * data)
{
	wxASSERT(m_Layer);
	OGRFeature * feature = OGRFeature::CreateFeature(m_Layer->GetLayerDefn());
	wxASSERT(m_Layer);
	feature->SetGeometry(geometry);

	if (data != NULL)
    {
        wxArrayDouble * dataArray = (wxArrayDouble*) data;
		wxASSERT(dataArray->GetCount() == 2);

        for (unsigned int i_dat=0; i_dat<dataArray->size(); i_dat++)
        {
            feature->SetField(i_dat, dataArray->Item(i_dat));
        }
	}

	if(m_Layer->CreateFeature(feature) != OGRERR_NONE)
    {
		asLogError(_("Error creating feature"));
		OGRFeature::DestroyFeature(feature);
		return wxNOT_FOUND;
	}

	long featureID = feature->GetFID();
	wxASSERT(featureID != OGRNullFID);
	OGRFeature::DestroyFeature(feature);
	return featureID;
}

void vrLayerVectorFcstDots::_DrawPoint(wxDC * dc, OGRFeature * feature, OGRGeometry * geometry, const wxRect2DDouble & coord, const vrRender * render,  vrLabel * label, double pxsize)
{
    // Set the defaut pen
	wxASSERT(render->GetType() == vrRENDER_VECTOR);
	wxPen defaultPen (*wxBLACK, 1);
	wxPen selPen (*wxGREEN, 3);
	
	// Get graphics context 
	wxGraphicsContext *gc = dc->GetGraphicsContext();
	wxASSERT(gc);

	if (gc)
	{
		// Get extent
		double extWidth = 0, extHeight = 0;
		gc->GetSize(&extWidth, &extHeight);
		wxRect2DDouble extWndRect (0,0,extWidth, extHeight);
		
		// Set font
		wxFont defFont(7, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
		gc->SetFont( defFont, *wxBLACK );

		// Get geometries
		OGRPoint * geom = (OGRPoint*) geometry;

		wxPoint point = _GetPointFromReal(wxPoint2DDouble(geom->getX(),geom->getY()),
										 coord.GetLeftTop(),
										 pxsize);

		// Create graphics path
		wxGraphicsPath path = gc->CreatePath();

		// Create first segment
		_CreatePath(path, point);
		
		// Ensure intersecting display
		wxRect2DDouble pathRect = path.GetBox();
		if (pathRect.Intersects(extWndRect) ==false) 
		{
			return;
		}
		if (pathRect.GetSize().x < 1 && pathRect.GetSize().y < 1)
		{
			return;
		}
		
		// Set the defaut pen
		gc->SetPen(defaultPen);
		if (IsFeatureSelected(feature->GetFID())==true) {
			gc->SetPen(selPen);
		}

		// Get value to set color
		double realValue = feature->GetFieldAsDouble(0);
		double normValue = feature->GetFieldAsDouble(1);
		_Paint(gc, path, normValue);
		_AddLabel(gc, point, realValue);
	}
	else
	{
		asLogError(_("Drawing of the symbol failed."));
	}
	
	return;
}

void vrLayerVectorFcstDots::_CreatePath(wxGraphicsPath & path, const wxPoint & center)
{
    const wxDouble radius = 15;

    path.AddCircle(center.x, center.y, radius);
}

void vrLayerVectorFcstDots::_Paint(wxGraphicsContext * gdc, wxGraphicsPath & path, double value)
{
    // wxColour colour(255,0,0); -> red
    // wxColour colour(0,0,255); -> blue
    // wxColour colour(0,255,0); -> green

    wxColour colour;

    if (asTools::IsNaN(value)) // NaN -> gray
    {
        colour.Set(150,150,150);
    }
    else if (value==0) // No rain -> white
    {
        colour.Set(255,255,255);
    }
    else if ( value/m_ValueMax<=0.5 ) // Light green to yellow
    {
        int baseVal = 200;
        int valColour = ((value/(0.5*m_ValueMax)))*baseVal;
        int valColourCompl = ((value/(0.5*m_ValueMax)))*(255-baseVal);
        if (valColour>baseVal) valColour = baseVal;
        if (valColourCompl+baseVal>255) valColourCompl = 255-baseVal;
        colour.Set((baseVal+valColourCompl),255,(baseVal-valColour));
    }
    else // Yellow to red
    {
        int valColour = ((value-0.5*m_ValueMax)/(0.5*m_ValueMax))*255;
        if (valColour>255) valColour = 255;
        colour.Set(255,(255-valColour),0);
    }

    wxBrush brush(colour, wxSOLID);
    gdc->SetBrush(brush);
    gdc->DrawPath(path);
}

void vrLayerVectorFcstDots::_AddLabel(wxGraphicsContext * gdc, const wxPoint & center, double value)
{
    wxString label = wxString::Format("%1.1f", value);
    wxDouble w, h;
    gdc->GetTextExtent(label, &w, &h);
    gdc->DrawText(label, center.x-w/2.0, center.y-h/2.0);
}
