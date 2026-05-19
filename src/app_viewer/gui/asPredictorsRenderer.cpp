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
#include "vrLayerVectorContours.h"
#include "vrLayerVectorDomain.h"
#include "vrLayerVectorLocation.h"
#include "vrRenderRasterPredictor.h"
#include "vrlayerraster.h"
#include "vrrender.h"

asPredictorsRenderer::asPredictorsRenderer(wxWindow* parent, vrLayerManager* layerManager,
                                           asPredictorsManager* predictorsManagerTarget,
                                           asPredictorsManager* predictorsManagerAnalog,
                                           vrViewerLayerManager* viewerLayerManagerTarget,
                                           vrViewerLayerManager* viewerLayerManagerAnalog)
    : _parent(parent),
      _layerManager(layerManager),
      _predictorsManagerTarget(predictorsManagerTarget),
      _predictorsManagerAnalog(predictorsManagerAnalog),
      _viewerLayerManagerTarget(viewerLayerManagerTarget),
      _viewerLayerManagerAnalog(viewerLayerManagerAnalog) {}

asPredictorsRenderer::~asPredictorsRenderer() = default;

void asPredictorsRenderer::LinkToColorbars(asPanelPredictorsColorbar* colorbarTarget,
                                           asPanelPredictorsColorbar* colorbarAnalog) {
    _colorbarTarget = colorbarTarget;
    _colorbarAnalog = colorbarAnalog;
}

void asPredictorsRenderer::Redraw(vf& domain, Coo& location, int predictorSelection) {
    bool targetDataLoaded = false;
    bool analogDataLoaded = false;
    try {
        targetDataLoaded = _predictorsManagerTarget->LoadData(predictorSelection);
        analogDataLoaded = _predictorsManagerAnalog->LoadData(predictorSelection);
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
        minVal = wxMin(_predictorsManagerTarget->GetDataMin(), minVal);
        maxVal = wxMax(_predictorsManagerTarget->GetDataMax(), maxVal);
    }
    if (analogDataLoaded) {
        minVal = wxMin(_predictorsManagerAnalog->GetDataMin(), minVal);
        maxVal = wxMax(_predictorsManagerAnalog->GetDataMax(), maxVal);
    }

    double step = ComputeStep(minVal, maxVal);

    // Set range and step to colorbar
    wxASSERT(_colorbarTarget);
    wxASSERT(_colorbarAnalog);
    _colorbarTarget->SetRange(minVal, maxVal);
    _colorbarAnalog->SetRange(minVal, maxVal);
    _colorbarTarget->SetStep(step);
    _colorbarAnalog->SetStep(step);

    if (targetDataLoaded) {
        _viewerLayerManagerTarget->FreezeBegin();
        wxString rasterPredictorName = _("Predictor - target");
        wxString contoursName = _("Contours - target");
        wxString spatialWindowName = _("Spatial window (left)");
        wxString locationName = _("Location (left)");
        CloseLayerIfPresent(_viewerLayerManagerTarget, wxFileName("", rasterPredictorName, "memory"));
        CloseLayerIfPresent(_viewerLayerManagerTarget, wxFileName("", contoursName, "memory"));
        CloseLayerIfPresent(_viewerLayerManagerTarget, wxFileName("", spatialWindowName, "memory"));
        CloseLayerIfPresent(_viewerLayerManagerTarget, wxFileName("", locationName, "memory"));
        vrLayerRasterPredictor* layerTarget = RedrawRasterPredictor(rasterPredictorName, _viewerLayerManagerTarget,
                                                                    _predictorsManagerTarget, minVal, maxVal);
        RedrawContourLines(contoursName, _viewerLayerManagerTarget, layerTarget, step);
        RedrawSpatialWindow(spatialWindowName, _viewerLayerManagerTarget, domain);
        RedrawLocation(locationName, _viewerLayerManagerTarget, location);
        _viewerLayerManagerTarget->FreezeEnd();
        _colorbarTarget->Refresh();
    }

    if (analogDataLoaded) {
        _viewerLayerManagerAnalog->FreezeBegin();
        wxString rasterPredictorName = _("Predictor - analog");
        wxString contoursName = _("Contours - analog");
        wxString spatialWindowName = _("Spatial window (right)");
        wxString locationName = _("Location (right)");
        CloseLayerIfPresent(_viewerLayerManagerAnalog, wxFileName("", rasterPredictorName, "memory"));
        CloseLayerIfPresent(_viewerLayerManagerAnalog, wxFileName("", contoursName, "memory"));
        CloseLayerIfPresent(_viewerLayerManagerAnalog, wxFileName("", spatialWindowName, "memory"));
        CloseLayerIfPresent(_viewerLayerManagerAnalog, wxFileName("", locationName, "memory"));
        vrLayerRasterPredictor* layerAnalog = RedrawRasterPredictor(rasterPredictorName, _viewerLayerManagerAnalog,
                                                                    _predictorsManagerAnalog, minVal, maxVal);
        RedrawContourLines(contoursName, _viewerLayerManagerAnalog, layerAnalog, step);
        RedrawSpatialWindow(spatialWindowName, _viewerLayerManagerAnalog, domain);
        RedrawLocation(locationName, _viewerLayerManagerAnalog, location);
        _viewerLayerManagerAnalog->FreezeEnd();
        _colorbarAnalog->Refresh();
    }
}

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
    _layerManager->Add(layerRaster);

    // Create render and add to the layer managers
    auto render = new vrRenderRasterPredictor();
    render->SetTransparency(20);
    viewerLayerManager->Add(1, layerRaster, render, nullptr, true);

    _colorbarTarget->SetRender(render);
    _colorbarAnalog->SetRender(render);

    return layerRaster;
}

void asPredictorsRenderer::RedrawContourLines(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                              vrLayerRasterPredictor* layerRaster, double step) {
    if (!layerRaster) return;

    if (layerRaster->GetParameter() == asPredictor::RelativeHumidity ||
        layerRaster->GetParameter() == asPredictor::SpecificHumidity ||
        layerRaster->GetParameter() == asPredictor::PrecipitableWater ||
        layerRaster->GetParameter() == asPredictor::TotalColumnWater) {
        return;
    }

    // Create a memory layer
    wxFileName memoryVector("", name, "memory");

    // Create the layers
    auto layerVector = new vrLayerVectorContours();

    if (!layerVector->Create(memoryVector, wkbLineString)) {
        wxFAIL;
        wxDELETE(layerVector);
        return;
    }

    // Specify the contour intervals
    char** options = NULL;
    options = CSLSetNameValue(options, "LEVEL_INTERVAL", asStrF("%g", step));

    // Generate the contours
    GDALContourGenerateEx(layerRaster->GetDatasetRef()->GetRasterBand(1), layerVector->GetLayerRef(), options, nullptr,
                          nullptr);
    CSLDestroy(options);

    // Add layers to the layer manager
    _layerManager->Add(layerVector);

    // Create render and add to the layer managers
    auto render = new vrRenderVector();
    render->SetTransparency(0);
    viewerLayerManager->Add(-1, layerVector, render, nullptr, true);
}

void asPredictorsRenderer::RedrawSpatialWindow(const wxString& name, vrViewerLayerManager* viewerLayerManager,
                                               vf& domain) {
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
    _layerManager->Add(layerVector);

    // Create render and add to the layer managers
    auto render = new vrRenderVector();
    render->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
    render->SetTransparency(0);
    render->SetSize(2);
    viewerLayerManager->Add(-1, layerVector, render, nullptr, true);
}

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
    _layerManager->Add(layerVector);

    // Create render and add to the layer managers
    auto render = new vrRenderVector();
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
            _layerManager->Close(layer);
        }
    }
}

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