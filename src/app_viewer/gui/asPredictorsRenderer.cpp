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

#include "asPredictorsManager.h"
#include "vrlayervector.h"
#include "vrrender.h"

asPredictorsRenderer::asPredictorsRenderer(wxWindow* parent, vrLayerManager* layerManager,
                                           asPredictorsManager* predictorsManager,
                                           vrViewerLayerManager* viewerLayerManagerTarget,
                                           vrViewerLayerManager* viewerLayerManagerAnalog,
                                           wxCheckListBox* checkListPredictors) {
    m_parent = parent;
    m_layerManager = layerManager;
    m_predictorsManager = predictorsManager;
    m_viewerLayerManagerTarget = viewerLayerManagerTarget;
    m_viewerLayerManagerAnalog = viewerLayerManagerAnalog;
}

asPredictorsRenderer::~asPredictorsRenderer() = default;

void asPredictorsRenderer::Redraw(double targetDate, double analogDate) {
    m_viewerLayerManagerTarget->FreezeBegin();
    RedrawRasterPredictor(_("Predictor - target"), m_viewerLayerManagerTarget);
    m_viewerLayerManagerTarget->FreezeEnd();

    m_viewerLayerManagerAnalog->FreezeBegin();
    RedrawRasterPredictor(_("Predictor - analog"), m_viewerLayerManagerAnalog);
    m_viewerLayerManagerAnalog->FreezeEnd();
}

void asPredictorsRenderer::RedrawRasterPredictor(const wxString &name, vrViewerLayerManager* viewerLayerManager) {
    // Create a memory layer
    wxFileName memoryRaster("", name, "");

    // Check if memory layer already added
    for (int i = 0; i < viewerLayerManager->GetCount(); i++) {
        if (viewerLayerManager->GetRenderer(i)->GetLayer()->GetFileName() == memoryRaster) {
            vrRenderer* renderer = viewerLayerManager->GetRenderer(i);
            vrLayer* layer = renderer->GetLayer();
            wxASSERT(renderer);
            viewerLayerManager->Remove(renderer);
            // Close layer
            m_layerManager->Close(layer);
        }
    }

    // Create the layers
    auto* layerRaster = new vrLayerRasterPredictor();

    if (!layerRaster->CreateInMemory(memoryRaster)) {
        wxFAIL;
        wxDELETE(layerRaster);
        return;
    }

    // Add layers to the layer manager
    m_layerManager->Add(layerRaster);

    // Create render and add to the layer managers
    auto render = new vrRenderRasterPredictor();
    render->SetTransparency(20);
    viewerLayerManager->Add(-1, layerRaster, render, nullptr, true);
}