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

#ifndef VR_LAYER_VECTORS_FCST_DOTS_H
#define VR_LAYER_VECTORS_FCST_DOTS_H

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
// Include wxWidgets' headers
#ifndef WX_PRECOMP

#include <wx/wx.h>

#endif

#include "asIncludes.h"
#include "vrlayervector.h"

class vrRender;

class vrLabel;

class vrLayerVectorFcstDots : public vrLayerVectorOGR {
  public:
    /**
     * The constructor for the vroomgis layer class showing the forecast values as dots.
     */
    vrLayerVectorFcstDots();

    /**
     * The destructor for the vroomgis layer class showing the forecast values as dots.
     */
    ~vrLayerVectorFcstDots() override;

    /**
     * Add a feature to the layer.
     * 
     * @param geometry The geometry of the feature.
     * @param data The data of the feature.
     * @return The ID of the feature.
     */
    long AddFeature(OGRGeometry* geometry, void* data) override;

    /**
     * Set the maximum value of the forecast values.
     * 
     * @param val The maximum value.
     */
    void SetMaxValue(double val) {
        if (val < 0.1) {
            wxLogWarning(
                _("The given maximum value for the vrLayerVectorFcstDots class was too small, so it has been "
                  "increased."));
            val = 0.1;
        }
        m_valueMax = val;
    }

  protected:
    double m_valueMax; /**< The maximum value of the forecast values. */

    /**
     * Draw a point (vroomgis function).
     * 
     * @param dc The device context.
     * @param feature The feature.
     * @param geometry The geometry.
     * @param coord The coordinates.
     * @param render The render.
     * @param label The label.
     * @param pxsize The pixel size.
     */
    void _DrawPoint(wxDC* dc, OGRFeature* feature, OGRGeometry* geometry, const wxRect2DDouble& coord,
                    const vrRender* render, vrLabel* label, double pxsize) override;

    /**
     * Create the path for the dot.
     * 
     * @param path The path.
     * @param center The center of the dot.
     */
    static void CreatePath(wxGraphicsPath& path, const wxPoint& center);

    /**
     * Paint the dots.
     * 
     * @param gdc The graphics context.
     * @param path The path.
     * @param value The predictand value.
     */
    void Paint(wxGraphicsContext* gdc, wxGraphicsPath& path, double value) const;

    /**
     * Add a label to the dot.
     * 
     * @param gdc The graphics context.
     * @param center The center of the dot.
     * @param value The predictand value.
     */
    static void AddLabel(wxGraphicsContext* gdc, const wxPoint& center, double value);
};

#endif
