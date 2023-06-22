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
 * Portions Copyright 2022-2023 Pascal Horton, Terranum.
 */

#ifndef AS_PREDICTORS_VIEWER_H
#define AS_PREDICTORS_VIEWER_H

#include "asIncludes.h"
#include "asPanelPredictorsColorbar.h"
#include "vroomgis.h"

class asPredictorsManager;

class vrLayerRasterPredictor;

/**
 * The class handling the predictors rendering for the predictors mapping.
 */
class asPredictorsRenderer {
  public:
    /**
     * The constructor of the renderer for the predictors (asPredictorsRenderer).
     *
     * @param parent The parent window.
     * @param layerManager The layer manager.
     * @param predictorsManagerTarget The predictors manager for the target data.
     * @param predictorsManagerAnalog The predictors manager for the analog data.
     * @param viewerLayerManagerTarget The viewer layer manager for the target data.
     * @param viewerLayerManagerAnalog The viewer layer manager for the analog data.
     */
    asPredictorsRenderer(wxWindow* parent, vrLayerManager* layerManager, asPredictorsManager* predictorsManagerTarget,
                         asPredictorsManager* predictorsManagerAnalog, vrViewerLayerManager* viewerLayerManagerTarget,
                         vrViewerLayerManager* viewerLayerManagerAnalog);

    /**
     * The destructor of the renderer for the predictors (asPredictorsRenderer).
     */
    virtual ~asPredictorsRenderer();

    /**
     * Associate the colorbars to the renderer.
     *
     * @param colorbarTarget The colorbar for the target data.
     * @param colorbarAnalog The colorbar for the analog data.
     */
    void LinkToColorbars(asPanelPredictorsColorbar* colorbarTarget, asPanelPredictorsColorbar* colorbarAnalog);

    /**
     * Redraw the predictor maps.
     *
     * @param domain The spatial window.
     * @param location The target location.
     * @param predictorSelection The predictor selection.
     */
    void Redraw(vf& domain, Coo& location, int predictorSelection);

    /**
     * Redraw the predictor raster.
     *
     * @param name The name of the layer.
     * @param viewerLayerManager The layer manager.
     * @param predictorsManager The predictors manager.
     * @param minVal The minimum value of the data.
     * @param maxVal The maximum value of the data.
     */
    vrLayerRasterPredictor* RedrawRasterPredictor(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                                  asPredictorsManager* predictorsManager, double minVal, double maxVal);

    /**
     * Redraw the contour lines.
     *
     * @param name The name of the layer.
     * @param viewerLayerManager The layer manager.
     * @param layerRaster The raster layer.
     * @param step The step of the contour lines.
     */
    void RedrawContourLines(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                            vrLayerRasterPredictor* layerRaster, double step);

    /**
     * Redraw the spatial windows.
     *
     * @param name The name of the layer.
     * @param viewerLayerManager The layer manager.
     * @param domain The spatial window extent.
     */
    void RedrawSpatialWindow(const wxString& name, vrViewerLayerManager* viewerLayerManager, vf& domain);

    /**
     * Redraw the target location.
     *
     * @param name The name of the layer.
     * @param viewerLayerManager The layer manager.
     * @param location The target location.
     */
    void RedrawLocation(const wxString& name, vrViewerLayerManager* viewerLayerManager, Coo& location);

  protected:
  private:
    wxWindow* m_parent; /**< The parent window. */
    vrLayerManager* m_layerManager; /**< The layer manager. */
    asPredictorsManager* m_predictorsManagerTarget; /**< The predictors manager for the target data. */
    asPredictorsManager* m_predictorsManagerAnalog; /**< The predictors manager for the analog data. */
    vrViewerLayerManager* m_viewerLayerManagerTarget; /**< The viewer layer manager for the target data. */
    vrViewerLayerManager* m_viewerLayerManagerAnalog; /**< The viewer layer manager for the analog data. */
    asPanelPredictorsColorbar* m_colorbarTarget; /**< The colorbar for the target data. */
    asPanelPredictorsColorbar* m_colorbarAnalog; /**< The colorbar for the analog data. */

    /**
     * Close the layer if present.
     *
     * @param viewerLayerManager The layer manager.
     * @param memoryVector The memory vector name.
     */
    void CloseLayerIfPresent(vrViewerLayerManager* viewerLayerManager, const wxFileName& memoryVector);

    /**
     * Compute the step for the contour lines.
     *
     * @param minVal The minimum value.
     * @param maxVal The maximum value.
     * @return The step for the contour lines.
     */
    double ComputeStep(double minVal, double maxVal) const;
};

#endif
