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

#include "vrLayerRasterPredictor.h"

#include "vrlabel.h"
#include "vrrealrect.h"

vrLayerRasterPredictor::vrLayerRasterPredictor()
    : vrLayerRasterGDAL(){};

vrLayerRasterPredictor::~vrLayerRasterPredictor() = default;

bool vrLayerRasterPredictor::Close() {
    if (m_dataset == nullptr) {
        return false;
    }

    GDALClose(m_dataset);
    m_dataset = nullptr;
    return true;
}

bool vrLayerRasterPredictor::CreateInMemory() {
    // Try to close
    Close();
    wxASSERT(m_dataset == nullptr);

    // Init filename and type
    m_fileName = _("Predictor");
    m_driverType = vrDRIVER_USER_DEFINED;

    // Get driver
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("MEM");
    if (poDriver == nullptr) {
        wxLogError("Cannot get the memory driver.");
        return false;
    }

    // Create dataset
    m_dataset = poDriver->Create(_("Predictor"), int(m_longitudes.size()), int(m_latitudes.size()), 1, GDT_Float32, nullptr);
    if (m_dataset == nullptr) {
        wxLogError(_("Creation of memory dataset failed."));
        return false;
    }

    // Set projection
    if (m_dataset->SetProjection("EPSG:4326") != CE_None) {
        wxLogError(_("Setting projection to predictor layer failed."));
        return false;
    }

    return true;
}
