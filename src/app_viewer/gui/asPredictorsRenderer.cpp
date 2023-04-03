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
#include "vrlayerraster.h"
#include "vrrender.h"

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

asPredictorsRenderer::~asPredictorsRenderer() = default;

void asPredictorsRenderer::Redraw(vf &domain) {
    if (m_predictorsManagerTarget->LoadData()) {
        m_viewerLayerManagerTarget->FreezeBegin();
        wxString rasterPredictorName = _("Predictor - target");
        wxString contoursName = _("Contours - target");
        wxString spatialWindowName = _("Spatial window target");
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", rasterPredictorName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", contoursName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerTarget, wxFileName("", spatialWindowName, "memory"));
        vrLayerRasterPredictor* layerTarget = RedrawRasterPredictor(rasterPredictorName, m_viewerLayerManagerTarget, m_predictorsManagerTarget);
        RedrawContourLines(contoursName, m_viewerLayerManagerTarget, layerTarget);
        RedrawSpatialWindow(spatialWindowName, m_viewerLayerManagerTarget, domain);
        m_viewerLayerManagerTarget->FreezeEnd();
    }

    if (m_predictorsManagerAnalog->LoadData()) {
        m_viewerLayerManagerAnalog->FreezeBegin();
        wxString rasterPredictorName = _("Predictor - analog");
        wxString contoursName = _("Contours - analog");
        wxString spatialWindowName = _("Spatial window analog");
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", rasterPredictorName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", contoursName, "memory"));
        CloseLayerIfPresent(m_viewerLayerManagerAnalog, wxFileName("", spatialWindowName, "memory"));
        vrLayerRasterPredictor* layerAnalog = RedrawRasterPredictor(rasterPredictorName, m_viewerLayerManagerAnalog, m_predictorsManagerAnalog);
        RedrawContourLines(contoursName, m_viewerLayerManagerAnalog, layerAnalog);
        RedrawSpatialWindow(spatialWindowName, m_viewerLayerManagerAnalog, domain);
        m_viewerLayerManagerAnalog->FreezeEnd();
    }
}

vrLayerRasterPredictor* asPredictorsRenderer::RedrawRasterPredictor(const wxString& name,
                                                                    vrViewerLayerManager* viewerLayerManager,
                                                                    asPredictorsManager* predictorsManager) {
    // Create a memory layer
    wxFileName memoryRaster("", name, "memory");

    // Create the layers
    auto* layerRaster = new vrLayerRasterPredictor(predictorsManager);

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

    return layerRaster;
}

void asPredictorsRenderer::RedrawContourLines(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                              vrLayerRasterPredictor* layerRaster) {
    // Create a memory layer
    wxFileName memoryVector("", name, "memory");

    if (!layerRaster) return;

    // Create the layers
    auto* layerVector = new vrLayerVectorContours();

    if (!layerVector->Create(memoryVector, wkbLineString)) {
        wxFAIL;
        wxDELETE(layerVector);
        return;
    }

    // Specify the contour intervals
    char **options = NULL;
    switch (layerRaster->GetParameter()) {
        case asPredictor::GeopotentialHeight:
            options = CSLSetNameValue(options, "LEVEL_INTERVAL", "100");
            break;
        case asPredictor::RelativeHumidity:
            options = CSLSetNameValue(options, "LEVEL_INTERVAL", "20");
            break;
        case asPredictor::PrecipitableWater:
        case asPredictor::TotalColumnWater:
            options = CSLSetNameValue(options, "LEVEL_INTERVAL", "10");
            break;
        default:
            options = CSLSetNameValue(options, "LEVEL_INTERVAL", "10");
    }

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

void asPredictorsRenderer::RedrawSpatialWindow(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                               vf &domain) {
    // Create a memory layer
    wxFileName memoryVector("", name, "memory");

    // Create the layers
    auto* layerVector = new vrLayerVectorDomain();

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
