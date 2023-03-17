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
    // Create a memory layer
    wxFileName memoryRasterTarget("", _("Predictor - target"), "memory");
    wxFileName memoryRasterAnalog("", _("Predictor - analog"), "memory");

    // Check if memory layer already added
    m_viewerLayerManagerTarget->FreezeBegin();
    m_viewerLayerManagerAnalog->FreezeBegin();
    for (int i = 0; i < m_viewerLayerManagerTarget->GetCount(); i++) {
        if (m_viewerLayerManagerTarget->GetRenderer(i)->GetLayer()->GetFileName() == memoryRasterTarget) {
            vrRenderer* renderer = m_viewerLayerManagerTarget->GetRenderer(i);
            vrLayer* layer = renderer->GetLayer();
            wxASSERT(renderer);
            m_viewerLayerManagerTarget->Remove(renderer);
            // Close layer
            m_layerManager->Close(layer);
        }
    }
    for (int i = 0; i < m_viewerLayerManagerAnalog->GetCount(); i++) {
        if (m_viewerLayerManagerAnalog->GetRenderer(i)->GetLayer()->GetFileName() == memoryRasterAnalog) {
            vrRenderer* renderer = m_viewerLayerManagerAnalog->GetRenderer(i);
            vrLayer* layer = renderer->GetLayer();
            wxASSERT(renderer);
            m_viewerLayerManagerAnalog->Remove(renderer);
            // Close layer
            m_layerManager->Close(layer);
        }
    }

    // Create the layers
    auto* layerRasterTarget = new vrLayerRasterPredictor();
    auto* layerRasterAnalog = new vrLayerRasterPredictor();

    if (!layerRasterTarget->CreateInMemory(memoryRasterTarget)) {
        wxFAIL;
        m_viewerLayerManagerTarget->FreezeEnd();
        m_viewerLayerManagerAnalog->FreezeEnd();
        wxDELETE(layerRasterTarget);
        wxDELETE(layerRasterAnalog);
        return;
    }
    if (!layerRasterAnalog->CreateInMemory(memoryRasterAnalog)) {
        wxFAIL;
        m_viewerLayerManagerTarget->FreezeEnd();
        m_viewerLayerManagerAnalog->FreezeEnd();
        wxDELETE(layerRasterTarget);
        wxDELETE(layerRasterAnalog);
        return;
    }
}
