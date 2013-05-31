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
 

#include "vrLayerVectorFcstRing.h"
#include "vrlabel.h"
#include "vrrender.h"
#include "asTools.h"


vrLayerVectorFcstRing::vrLayerVectorFcstRing()
{
	wxASSERT(m_Dataset==NULL);
	wxASSERT(m_Layer==NULL);
	m_DriverType = vrDRIVER_VECTOR_MEMORY;
	m_ValueMax = 1;
}

vrLayerVectorFcstRing::~vrLayerVectorFcstRing()
{
}

long vrLayerVectorFcstRing::AddFeature(OGRGeometry * geometry, void * data)
{
	wxASSERT(m_Layer);
	OGRFeature * feature = OGRFeature::CreateFeature(m_Layer->GetLayerDefn());
	wxASSERT(m_Layer);
	feature->SetGeometry(geometry);

	if (data != NULL)
    {
		wxArrayDouble * dataArray = (wxArrayDouble*) data;
		wxASSERT(dataArray->GetCount() >= 1);

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

bool vrLayerVectorFcstRing::_DrawPoints(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
								   const vrRender * render, vrLabel * label, double pxsize)
{
    m_ObjectDrawn = 0;
	wxASSERT(gdc);
	wxStopWatch sw;

    // Set the defaut pen
	wxASSERT(render->GetType() == vrRENDER_VECTOR);
	wxPen defaultPen (*wxBLACK, 1);
	wxPen selPen (*wxGREEN, 3);

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

		// Get lead time size
		int leadTimeSize = (int)feat->GetFieldAsDouble(0);
		wxASSERT(leadTimeSize>0);

		wxPoint point = _GetPointFromReal(wxPoint2DDouble(geom->getX(),geom->getY()),
										 coord.GetLeftTop(),
										 pxsize);

        // Create graphics path
        wxGraphicsPath path = gdc->CreatePath();

        // Create first segment
        _CreatePath(path, point, leadTimeSize, 0);

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
        double value = feat->GetFieldAsDouble(1);
        _Paint(gdc, path, value);

        // Draw next segments
        for (int i_leadtime=1; i_leadtime<leadTimeSize; i_leadtime++)
        {
            // Create shape
            path = gdc->CreatePath();
            _CreatePath(path, point, leadTimeSize, i_leadtime);

            // Get value to set color
            double value = feat->GetFieldAsDouble(i_leadtime+1);
            _Paint(gdc, path, value);

        }

        // Create a mark at the center
        path.AddCircle(point.x, point.y, 2);

/*      // Cross
        path.MoveToPoint(point.x+2, point.y);
        path.AddLineToPoint(point.x-2, point.y);
        path.MoveToPoint(point.x, point.y+2);
        path.AddLineToPoint(point.x, point.y-2);
*/
        gdc->StrokePath(path);

		OGRFeature::DestroyFeature(feat);
		feat = NULL;
	}

	m_ObjectDrawn = iCount;

	if (iCount == 0) return false;

	return true;
}

void vrLayerVectorFcstRing::_CreatePath(wxGraphicsPath & path, const wxPoint & center, int segmentsTotNb, int segmentNb)
{
    const wxDouble radiusOut = 25;
    const wxDouble radiusIn = 10;

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

void vrLayerVectorFcstRing::_Paint(wxGraphicsContext * gdc, wxGraphicsPath & path, double value)
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
    else if ( value/m_ValueMax<=0.5 ) // light green to yellow
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

bool vrLayerVectorFcstRing::_DrawLines(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
								  const vrRender * render, const vrLabel * label, double pxsize)
{
	m_ObjectDrawn = 0;
	return false;
}

bool vrLayerVectorFcstRing::_DrawPolygons(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
									 const vrRender * render, const vrLabel * label, double pxsize)
{
	m_ObjectDrawn = 0;
	return false;
}

bool vrLayerVectorFcstRing::_DrawMultiPolygons(wxGraphicsContext * gdc, const wxRect2DDouble & coord,
									 const vrRender * render, const vrLabel * label, double pxsize)
{
	m_ObjectDrawn = 0;
	return false;
}
