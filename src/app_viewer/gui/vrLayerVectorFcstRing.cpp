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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */


#include "vrLayerVectorFcstRing.h"
#include "vrlabel.h"
#include "vrrender.h"


vrLayerVectorFcstRing::vrLayerVectorFcstRing()
{
    wxASSERT(m_dataset == NULL);
    wxASSERT(m_layer == NULL);
    m_driverType = vrDRIVER_VECTOR_MEMORY;
    m_valueMax = 1;
}

vrLayerVectorFcstRing::~vrLayerVectorFcstRing()
{
}

long vrLayerVectorFcstRing::AddFeature(OGRGeometry *geometry, void *data)
{
    wxASSERT(m_layer);
    OGRFeature *feature = OGRFeature::CreateFeature(m_layer->GetLayerDefn());
    wxASSERT(m_layer);
    feature->SetGeometry(geometry);

    if (data != NULL) {
        wxArrayDouble *dataArray = (wxArrayDouble *) data;
        wxASSERT(dataArray->GetCount() >= 3);

        for (unsigned int iDat = 0; iDat < dataArray->size(); iDat++) {
            feature->SetField(iDat, dataArray->Item(iDat));
        }
    }

    if (m_layer->CreateFeature(feature) != OGRERR_NONE) {
        wxLogError(_("Error creating feature"));
        OGRFeature::DestroyFeature(feature);
        return wxNOT_FOUND;
    }

    long featureID = feature->GetFID();
    wxASSERT(featureID != OGRNullFID);
    OGRFeature::DestroyFeature(feature);
    return featureID;
}

void vrLayerVectorFcstRing::_DrawPoint(wxDC *dc, OGRFeature *feature, OGRGeometry *geometry,
                                       const wxRect2DDouble &coord, const vrRender *render, vrLabel *label,
                                       double pxsize)
{
    // Set the defaut pen
    wxASSERT(render->GetType() == vrRENDER_VECTOR);
    vrRenderVector *renderVector = (vrRenderVector *) render;
    wxPen defaultPen(renderVector->GetColorPen(), renderVector->GetSize());
    wxPen selPen(*wxGREEN, 3);

    // Get graphics context
    wxGraphicsContext *gc = dc->GetGraphicsContext();
    wxASSERT(gc);

    if (gc) {
        // Get extent
        double extWidth = 0, extHeight = 0;
        gc->GetSize(&extWidth, &extHeight);
        wxRect2DDouble extWndRect(0, 0, extWidth, extHeight);

        // Get geometries
        OGRPoint *geom = (OGRPoint *) geometry;

        wxPoint point = _GetPointFromReal(wxPoint2DDouble(geom->getX(), geom->getY()), coord.GetLeftTop(), pxsize);

        // Get lead time size
        int leadTimeSize = (int) feature->GetFieldAsDouble(2);
        wxASSERT(leadTimeSize > 0);

        // Create graphics path
        wxGraphicsPath path = gc->CreatePath();

        // Create first segment
        _CreatePath(path, point, leadTimeSize, 0);

        // Ensure intersecting display
        wxRect2DDouble pathRect = path.GetBox();
        if (!pathRect.Intersects(extWndRect)) {
            return;
        }
        if (pathRect.GetSize().x < 1 && pathRect.GetSize().y < 1) {
            return;
        }

        // Set the defaut pen
        gc->SetPen(defaultPen);
        if (IsFeatureSelected(feature->GetFID())) {
            gc->SetPen(selPen);
        }

        // Get value to set color
        double value = feature->GetFieldAsDouble(3);
        _Paint(gc, path, value);

        // Draw next segments
        for (int iLead = 1; iLead < leadTimeSize; iLead++) {
            // Create shape
            path = gc->CreatePath();
            _CreatePath(path, point, leadTimeSize, iLead);

            // Get value to set color
            value = feature->GetFieldAsDouble(iLead + 3);
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
    } else {
        wxLogError(_("Drawing of the symbol failed."));
    }

}

void vrLayerVectorFcstRing::_CreatePath(wxGraphicsPath &path, const wxPoint &center, int segmentsTotNb, int segmentNb)
{
    const wxDouble radiusOut = 25 * g_ppiScaleDc;
    const wxDouble radiusIn = 10 * g_ppiScaleDc;

    wxDouble segmentStart = -0.5 * M_PI + ((double) segmentNb / (double) segmentsTotNb) * (1.5 * M_PI);
    wxDouble segmentEnd = -0.5 * M_PI + ((double) (segmentNb + 1) / (double) segmentsTotNb) * (1.5 * M_PI);
    wxDouble centerX = (wxDouble) center.x;
    wxDouble centerY = (wxDouble) center.y;

    // Get starting point
    double dX = cos(segmentStart) * radiusOut;
    double dY = sin(segmentStart) * radiusOut;
    wxDouble startPointX = centerX + dX;
    wxDouble startPointY = centerY + dY;

    path.MoveToPoint(startPointX, startPointY);

    path.AddArc(centerX, centerY, radiusOut, segmentStart, segmentEnd, true);

    const wxDouble radiusRatio = ((radiusOut - radiusIn) / radiusOut);
    wxPoint2DDouble currentPoint = path.GetCurrentPoint();
    wxDouble newPointX = currentPoint.m_x - (currentPoint.m_x - centerX) * radiusRatio;
    wxDouble newPointY = currentPoint.m_y - (currentPoint.m_y - centerY) * radiusRatio;

    path.AddLineToPoint(newPointX, newPointY);

    path.AddArc(centerX, centerY, radiusIn, segmentEnd, segmentStart, false);

    path.CloseSubpath();
}

void vrLayerVectorFcstRing::_Paint(wxGraphicsContext *gdc, wxGraphicsPath &path, double value)
{
    // wxColour colour(255,0,0); -> red
    // wxColour colour(0,0,255); -> blue
    // wxColour colour(0,255,0); -> green

    wxColour colour;

    if (asTools::IsNaN(value)) // NaN -> gray
    {
        colour.Set(150, 150, 150);
    } else if (value == 0) // No rain -> white
    {
        colour.Set(255, 255, 255);
    } else if (value / m_valueMax <= 0.5) // light green to yellow
    {
        int baseVal = 200;
        int valColour = ((value / (0.5 * m_valueMax))) * baseVal;
        int valColourCompl = ((value / (0.5 * m_valueMax))) * (255 - baseVal);
        if (valColour > baseVal)
            valColour = baseVal;
        if (valColourCompl + baseVal > 255)
            valColourCompl = 255 - baseVal;
        colour.Set((baseVal + valColourCompl), 255, (baseVal - valColour));
    } else // Yellow to red
    {
        int valColour = ((value - 0.5 * m_valueMax) / (0.5 * m_valueMax)) * 255;
        if (valColour > 255)
            valColour = 255;
        colour.Set(255, (255 - valColour), 0);
    }

    wxBrush brush(colour, wxBRUSHSTYLE_SOLID);
    gdc->SetBrush(brush);
    gdc->DrawPath(path);
}
