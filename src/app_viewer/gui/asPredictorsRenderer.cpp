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

#include "asPredictorsRenderer.h"

#include <gdal_alg.h>

#include "asPredictorsManager.h"
#include "vrLayerRasterPredictor.h"
#include "vrRenderRasterPredictor.h"
#include "vrLayerVectorContours.h"
#include "vrLayerVectorDomain.h"
#include "vrLayerVectorLocation.h"
#include "vrlayerraster.h"
#include "vrrender.h"

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
asPredictorsRenderer::asPredictorsRenderer(wxWindow* parent, vrLayerManager* layerManager,
                                           asPredictorsManager* predictorsManagerTarget,
                                           asPredictorsManager* predictorsManagerAnalog,
                                           vrViewerLayerManager* viewerLayerManagerTarget,
                                           vrViewerLayerManager* viewerLayerManagerAnalog)
    : m_parent(parent),
      m_layerManager(layerManager),
      m_predictorsManagerTarget(predictorsManagerTarget),
      m_predictorsManagerAnalog(predictorsManagerAnalog),
      m_viewerLayerManagerTarget(viewerLayerManagerTarget),
      m_viewerLayerManagerAnalog(viewerLayerManagerAnalog) {}

/**
 * The destructor of the renderer for the predictors (asPredictorsRenderer).
*/
asPredictorsRenderer::~asPredictorsRenderer() = default;

/**
 * Associate the colorbars to the renderer.
 * 
 * @param colorbarTarget The colorbar for the target data.
 * @param colorbarAnalog The colorbar for the analog data.
*/
void asPredictorsRenderer::LinkToColorbars(asPanelPredictorsColorbar* colorbarTarget,
                                           asPanelPredictorsColorbar* colorbarAnalog) {
    m_colorbarTarget = colorbarTarget;
    m_colorbarAnalog = colorbarAnalog;
}

/**
 * Redraw the predictor maps.
 * 
 * @param domain The spatial window.
 * @param location The target location.
*/
void asPredictorsRenderer::Redraw(vf& domain, Coo& location) {
    bool targetDataLoaded = false;
    bool analogDataLoaded = false;
    try {
        targetDataLoaded = m_predictorsManagerTarget->LoadData();
        analogDataLoaded = m_predictorsManagerAnalog->LoadData();
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught during data loading: %s"), msg);
    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught during data loading: %s"), msg);
    }

    double minVal = 99999999999;
    double maxVal = -99999999999;

    if (targetDataLoaded) {
        minVal = wxMin(m_predictorsManagerTarget->GetDataMin(), minVal);
        maxVal = wxMax(m_predictorsManagerTarget->GetDataMax(), maxVal);
    }
    if (analogDataLoaded) {
        minVal = wxMin(m_predictorsManagerAnalog->GetDataMin(), minVal);
        maxVal = wxMax(m_predictorsManagerAnalog->GetDataMax(), maxVal);
    }

    double step = ComputeStep(minVal, maxVal);

    // Set range and step to colorbar
    wxASSERT(m_colorbarTarget);
    wxASSERT(m_colorbarAnalog);
    m_colorbarTarget->SetRange(minVal, maxVal);
    m_colorbarAnalog->SetRange(minVal, maxVal);
    m_colorbarTarget->SetStep(step);
    m_colorbarAnalog->SetStep(step);

    if (targetDataLoaded) {
        m_viewerLayerManagerTarget->FreezeBegin();
        wxString rasterPredictorName = _("Predictor - target");
        wxString contoursName = _("Contours - target");
        wxString spatialWindowName = _("Spatial window (left)");
        wxString locationName = _("Location (left)");
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", rasterPredictorName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", contoursName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", spatialWindowName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", locationName, "memory"));
        vrLayerRasterPredictor* layerTarget = RedrawRasterPredictor(rasterPredictorName, m_viewerLayerManagerTarget,
                                                                    m_predictorsManagerTarget, minVal, maxVal);
        RedrawContourLines(contoursName, m_viewerLayerManagerTarget, layerTarget, step);
        RedrawSpatialWindow(spatialWindowName, m_viewerLayerManagerTarget, domain);
        RedrawLocation(locationName, m_viewerLayerManagerTarget, location);
        m_viewerLayerManagerTarget->FreezeEnd();
        m_colorbarTarget->Refresh();
    }

    if (analogDataLoaded) {
        m_viewerLayerManagerAnalog->FreezeBegin();
        wxString rasterPredictorName = _("Predictor - analog");
        wxString contoursName = _("Contours - analog");
        wxString spatialWindowName = _("Spatial window (right)");
        wxString locationName = _("Location (right)");
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", rasterPredictorName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", contoursName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", spatialWindowName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", locationName, "memory"));
        vrLayerRasterPredictor* layerAnalog = RedrawRasterPredictor(rasterPredictorName, m_viewerLayerManagerAnalog,
                                                                    m_predictorsManagerAnalog, minVal, maxVal);
        RedrawContourLines(contoursName, m_viewerLayerManagerAnalog, layerAnalog, step);
        RedrawSpatialWindow(spatialWindowName, m_viewerLayerManagerAnalog, domain);
        RedrawLocation(locationName, m_viewerLayerManagerAnalog, location);
        m_viewerLayerManagerAnalog->FreezeEnd();
        m_colorbarAnalog->Refresh();
    }
}

/**
 * Redraw the predictor raster.
 * 
 * @param name The name of the layer.
 * @param viewerLayerManager The layer manager.
 * @param predictorsManager The predictors manager.
 * @param minVal The minimum value of the data.
 * @param maxVal The maximum value of the data.
*/
vrLayerRasterPredictor* asPredictorsRenderer::RedrawRasterPredictor(const wxString& name,
                                                                    vrViewerLayerManager* viewerLayerManager,
                                                                    asPredictorsManager* predictorsManager,
                                                                    double minVal, double maxVal) {
    // Create a memory layer
    wxFileName memoryRaster("", name, "memory");

    // Create the layers
    auto layerRaster = new vrLayerRasterPredictor(predictorsManager, minVal, maxVal);

    if (!layerRaster->CreateInMemory(memoryRaster)) {
        wxFAIL;
        wxDELETE(layerRaster);
        return nullptr;
    }

    // Add layers to the layer manager
    m_layerManager->Add(layerRaster);

    // Create render and add to the layer managers
    auto render = new vrRenderRasterPredictor();
    render->SetTransparency(20);
    viewerLayerManager->Add(1, layerRaster, render, nullptr, true);

    m_colorbarTarget->SetRender(render);
    m_colorbarAnalog->SetRender(render);

    return layerRaster;
}

/**
 * Redraw the contour lines.
 * 
 * @param name The name of the layer.
 * @param viewerLayerManager The layer manager.
 * @param layerRaster The raster layer.
 * @param step The step of the contour lines.
*/
void asPredictorsRenderer::RedrawContourLines(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                              vrLayerRasterPredictor* layerRaster, double step) {
    // Create a memory layer
    wxFileName memoryVector("", name, "memory");

    if (!layerRaster) return;

    // Create the layers
    auto layerVector = new vrLayerVectorContours();

    if (!layerVector->Create(memoryVector, wkbLineString)) {
        wxFAIL;
        wxDELETE(layerVector);
        return;
    }

    // Specify the contour intervals
    char **options = NULL;
    options = CSLSetNameValue(options, "LEVEL_INTERVAL", asStrF("%g", step));

    // Generate the contours
    GDALContourGenerateEx(layerRaster->GetDatasetRef()->GetRasterBand(1), layerVector->GetLayerRef(), options, nullptr,
                          nullptr);
    CSLDestroy(options);

    // Add layers to the layer manager
    m_layerManager->Add(layerVector);

    // Create render and add to the layer managers
    auto render = new vrRenderVector();
    render->SetTransparency(0);
    viewerLayerManager->Add(-1, layerVector, render, nullptr, true);
}

/**
 * Redraw the spatial windows.
 * 
 * @param name The name of the layer.
 * @param viewerLayerManager The layer manager.
 * @param domain The spatial window extent.
*/
void asPredictorsRenderer::RedrawSpatialWindow(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                               vf &domain) {
    // Create a memory layer
    wxFileName memoryVector("", name, "memory");

    // Create the layers
    auto layerVector = new vrLayerVectorDomain();

    if (!layerVector->Create(memoryVector, wkbPolygon)) {
        wxFAIL;
        wxDELETE(layerVector);
        return;
    }

    // Plot the domains
    OGRLinearRing* ring = new OGRLinearRing();
    ring->addPoint(domain[0], domain[3]);
    ring->addPoint(domain[1], domain[3]);
    ring->addPoint(domain[1], domain[2]);
    ring->addPoint(domain[0], domain[2]);
    ring->addPoint(domain[0], domain[3]);
    ring->closeRings();

    OGRPolygon* domainPoly = new OGRPolygon();
    domainPoly->addRingDirectly(ring);
    domainPoly->closeRings();

    layerVector->AddFeature(domainPoly);

    // Add layers to the layer manager
    m_layerManager->Add(layerVector);

    // Create render and add to the layer managers
    auto render = new vrRenderVector();
    render->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
    render->SetTransparency(0);
    render->SetSize(2);
    viewerLayerManager->Add(-1, layerVector, render, nullptr, true);
}

/**
 * Redraw the target location.
 * 
 * @param name The name of the layer.
 * @param viewerLayerManager The layer manager.
 * @param location The target location.
*/
void asPredictorsRenderer::RedrawLocation(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                          Coo& location) {
    if (location.x == 0 && location.y == 0) {
        return;
    }

    // Create a memory layer
    wxFileName memoryVector("", name, "memory");

    // Create the layers
    auto layerVector = new vrLayerVectorLocation();

    if (!layerVector->Create(memoryVector, wkbPoint)) {
        wxFAIL;
        wxDELETE(layerVector);
        return;
    }

    // Plot the domains
    OGRPoint* point = new OGRPoint(location.x, location.y);
    layerVector->AddFeature(point, nullptr);

    // Add layers to the layer manager
    m_layerManager->Add(layerVector);

    // Create render and add to the layer managers
    auto render = new vrRenderVector();
    render->SetTransparency(0);
    render->SetSize(2);
    viewerLayerManager->Add(-1, layerVector, render, nullptr, true);
}

/**
 * Close the layer if present.
 * 
 * @param viewerLayerManager The layer manager.
 * @param memoryVector The memory vector name.
*/
void asPredictorsRenderer::CloseLayerIfPresent(vrViewerLayerManager* viewerLayerManager,
                                               const wxFileName& memoryVector) {
    for (int i = 0; i < viewerLayerManager->GetCount(); i++) {
        if (viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryVector) {
            vrRenderer* renderer = viewerLayerManager->GetRenderer(i);
            vrLayer* layer = renderer->GetLayer();
            wxASSERT(renderer);
            viewerLayerManager->Remove(renderer);
            // Close layer
            m_layerManager->Close(layer);
        }
    }
}

/**
 * Compute the step for the contour lines.
 * 
 * @param minVal The minimum value.
 * @param maxVal The maximum value.
 * @return The step for the contour lines.
*/
double asPredictorsRenderer::ComputeStep(double minVal, double maxVal) const {
    if (maxVal == minVal) {
        return 100;
    }

    double range = maxVal - minVal;
    double stepApprox = range / 10;
    double magnitudeFull = log10(stepApprox);
    double magnitudeHalf = log10(2 * stepApprox);
    double stepFull = pow(10, ceil(magnitudeFull));
    double stepHalf = pow(10, ceil(magnitudeHalf)) / 2;
    double step = stepFull;
    if (abs(stepHalf - stepApprox) < abs(stepFull - stepApprox)) {
        step = stepHalf;
    }

    return step;
}