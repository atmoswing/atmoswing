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

bool vrLayerVectorFcstDots::_DrawPoints(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
								   const vrRender * render, vrLabel * label, double pxsize)
{
    m_ObjectDrawn = 0;
	wxASSERT(gdc);
	wxStopWatch sw;

    // Set the defaut pen
	wxASSERT(render->GetType() == vrRENDER_VECTOR);
	wxPen defaultPen (*wxBLACK, 1);
	wxPen selPen (*wxGREEN, 3);

	// Set font
	wxFont defFont(7, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
    gdc->SetFont( defFont, *wxBLACK );

	// Iterating and drawing geometries
	OGRPoint * geom = NULL;
	long iCount = 0;
	double width = 0, height = 0;
	gdc->GetSize(&width, &height);
	wxRect2DDouble windowRect (0,0,width, height);
	while (1)
    {
		OGRFeature * feat = GetNextFeature(false);
		if (feat == NULL) break;
		geom = NULL;
		geom = (OGRPoint*) feat->GetGeometryRef();
		wxASSERT(geom);

		wxPoint point = _GetPointFromReal(wxPoint2DDouble(geom->getX(),geom->getY()),
										 coord.GetLeftTop(),
										 pxsize);

        // Create graphics path
        wxGraphicsPath path = gdc->CreatePath();

        // Create first segment
        _CreatePath(path, point);

        // Ensure intersecting display
        wxRect2DDouble pathRect = path.GetBox();
        if (pathRect.Intersects(windowRect) ==false)
        {
            OGRFeature::DestroyFeature(feat);
            feat = NULL;
            continue;
        }

        if (pathRect.GetSize().x < 1 && pathRect.GetSize().y < 1)
        {
            OGRFeature::DestroyFeature(feat);
            feat = NULL;
            continue;
        }
        iCount++;

        // Pen selection
        gdc->SetPen(defaultPen);
        if (IsFeatureSelected(feat->GetFID())==true)
        {
            gdc->SetPen(selPen);
        }

        // Get value to set color
        double realValue = feat->GetFieldAsDouble(0);
        double normValue = feat->GetFieldAsDouble(1);
        _Paint(gdc, path, normValue);
        _AddLabel(gdc, point, realValue);

		OGRFeature::DestroyFeature(feat);
		feat = NULL;
	}

	m_ObjectDrawn = iCount;

	if (iCount == 0) return false;

	return true;
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

bool vrLayerVectorFcstDots::_DrawLines(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
								  const vrRender * render, const vrLabel * label, double pxsize)
{
	m_ObjectDrawn = 0;
	return false;
}

bool vrLayerVectorFcstDots::_DrawPolygons(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
									 const vrRender * render, const vrLabel * label, double pxsize)
{
	m_ObjectDrawn = 0;
	return false;
}

bool vrLayerVectorFcstDots::_DrawMultiPolygons(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
									 const vrRender * render, const vrLabel * label, double pxsize)
{
	m_ObjectDrawn = 0;
	return false;
}
