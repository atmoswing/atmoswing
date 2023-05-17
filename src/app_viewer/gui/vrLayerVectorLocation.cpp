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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include "vrLayerVectorLocation.h"

#include "vrlabel.h"
#include "vrrender.h"

vrLayerVectorLocation::vrLayerVectorLocation() {
    wxASSERT(!m_dataset);
    wxASSERT(!m_layer);
    m_driverType = vrDRIVER_VECTOR_MEMORY;
}

vrLayerVectorLocation::~vrLayerVectorLocation() = default;

long vrLayerVectorLocation::AddFeature(OGRGeometry* geometry, void* data) {
    wxASSERT(m_layer);
    OGRFeature* feature = OGRFeature::CreateFeature(m_layer->GetLayerDefn());
    wxASSERT(m_layer);
    feature->SetGeometry(geometry);

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

void vrLayerVectorLocation::_DrawPoint(wxDC* dc, OGRFeature* feature, OGRGeometry* geometry,
                                       const wxRect2DDouble& coord, const vrRender* render, vrLabel* label,
                                       double pxsize) {
    // Set the default pen
    wxASSERT(render->GetType() == vrRENDER_VECTOR);
    auto renderVector = const_cast<vrRenderVector*>(dynamic_cast<const vrRenderVector*>(render));
    wxPen defaultPen(*wxBLACK_PEN);
    defaultPen.SetWidth(renderVector->GetSize());

    // Get graphics context
    wxGraphicsContext* gc = dc->GetGraphicsContext();
    wxASSERT(gc);

    if (gc) {
        // Get extent
        double extWidth = 0, extHeight = 0;
        gc->GetSize(&extWidth, &extHeight);
        wxRect2DDouble extWndRect(0, 0, extWidth, extHeight);

        // Get geometries
        auto geom = dynamic_cast<OGRPoint*>(geometry);

        wxPoint point = _GetPointFromReal(wxPoint2DDouble(geom->getX(), geom->getY()), coord.GetLeftTop(), pxsize);

        // Create graphics path
        wxGraphicsPath path = gc->CreatePath();

        path.AddCircle(point.x, point.y, 3 * g_ppiScaleDc);

        // Ensure intersecting display
        wxRect2DDouble pathRect = path.GetBox();
        if (!pathRect.Intersects(extWndRect)) {
            return;
        }

        // Set the default pen
        gc->SetPen(defaultPen);

        // Set color
        wxColour colour;
        colour.Set(255, 255, 255);

        wxBrush brush(colour, wxBRUSHSTYLE_SOLID);
        gc->SetBrush(brush);
        gc->DrawPath(path);
    } else {
        wxLogError(_("Drawing of the symbol failed."));
    }
}
