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

#ifndef VR_LAYER_RASTER_PREDICTOR_H
#define VR_LAYER_RASTER_PREDICTOR_H

#include <asIncludes.h>
#include <wx/image.h>

#include "asPredictor.h"
#include "asPredictorsManager.h"
#include "vrlayerraster.h"

class vrLayerRasterPredictor : public vrLayerRasterGDAL {
  public:
    vrLayerRasterPredictor(asPredictorsManager* predictorsManager, double minVal, double maxVal);

    ~vrLayerRasterPredictor() override;

    bool CreateInMemory(const wxFileName& name);

    wxFileName GetDisplayName() override;

    asPredictor::Parameter GetParameter() {
        return m_parameter;
    }

  protected:
    virtual bool _GetRasterData(unsigned char** imgData, const wxSize& outImgPxSize, const wxRect& readImgPxInfo,
                                const vrRender* render);

  private:
    asPredictorsManager* m_predictorsManager;
    asPredictor::Parameter m_parameter;
    double m_minVal;
    double m_maxVal;

    bool Close();
};

#endif
