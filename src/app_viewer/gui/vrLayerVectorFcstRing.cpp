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
		wxASSERT(dataArray->GetCount() >= 3);

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

void vrLayerVectorFcstRing::_DrawPoint(wxDC * dc, OGRFeature * feature, OGRGeometry * geometry, const wxRect2DDouble & coord, const vrRender * render,  vrLabel * label, double pxsize)
{
    // Set the defaut pen
	wxASSERT(render->GetType() == vrRENDER_VECTOR);
    vrRenderVector * renderVector = (vrRenderVector *) render;
    wxPen defaultPen (renderVector->GetColorPen(), renderVector->GetSize());
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
	
		// Get geometries
		OGRPoint * geom = (OGRPoint*) geometry;

		wxPoint point = _GetPointFromReal(wxPoint2DDouble(geom->getX(),geom->getY()),
										 coord.GetLeftTop(),
										 pxsize);

		// Get lead time size
		int leadTimeSize = (int)feature->GetFieldAsDouble(2);
		wxASSERT(leadTimeSize>0);

		// Create graphics path
		wxGraphicsPath path = gc->CreatePath();

		// Create first segment
		_CreatePath(path, point, leadTimeSize, 0);
		
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
		double value = feature->GetFieldAsDouble(3);
		_Paint(gc, path, value);

		// Draw next segments
		for (int i_leadtime=1; i_leadtime<leadTimeSize; i_leadtime++)
		{
			// Create shape
			path = gc->CreatePath();
			_CreatePath(path, point, leadTimeSize, i_leadtime);

			// Get value to set color
			double value = feature->GetFieldAsDouble(i_leadtime+3);
			_Paint(gc, path, value);

		}

		// Create a mark at the center
		path.AddCircle(point.x, point.y, 2);

/*      // Cross
		path.MoveToPoint(point.x+2, point.y);
		path.AddLineToPoint(point.x-2, point.y);
		path.MoveToPoint(point.x, point.y+2);
		path.AddLineToPoint(point.x, point.y-2);
*/
		gc->StrokePath(path);
	}
	else
	{
		asLogError(_("Drawing of the symbol failed."));
	}
	
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
