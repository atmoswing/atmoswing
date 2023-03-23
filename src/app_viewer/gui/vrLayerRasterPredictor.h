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

#include "vrlayerraster.h"


class vrLayerRasterPredictor : public vrLayerRasterGDAL {
  public:
    vrLayerRasterPredictor();

    ~vrLayerRasterPredictor() override;

    bool CreateInMemory(const wxFileName &name);

    void SetParameter(asPredictor::Parameter parameter) {
        m_parameter = parameter;
    }

    void SetData(const a2f& data) {
        m_data = data;
    }

    void SetLongitudes(const a1f& lon) {
        m_longitudes = lon;
    }

    void SetLatitudes(const a1f& lat) {
        m_latitudes = lat;
    }

  protected:
    virtual bool _GetRasterData(unsigned char** imgData, const wxSize& outImgPxSize, const wxRect& readImgPxInfo,
                                const vrRender* render);

  private:
    asPredictor::Parameter m_parameter;
    a2f m_data;
    a1f m_longitudes;
    a1f m_latitudes;

    bool Close();
};

#endif
