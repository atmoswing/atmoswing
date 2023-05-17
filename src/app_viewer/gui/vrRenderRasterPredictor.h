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

#ifndef VR_RENDER_RASTER_PREDICTOR_H
#define VR_RENDER_RASTER_PREDICTOR_H

#include <wx/image.h>

#include "asIncludes.h"
#include "asPredictor.h"
#include "vrrender.h"

class vrRenderRasterPredictor : public vrRenderRaster {
  public:
    vrRenderRasterPredictor();

    ~vrRenderRasterPredictor() override;

    wxImage::RGBValue GetColorFromTable(double pxVal, double minVal, double range);

    void Init(asPredictor::Parameter parameter);

  protected:
  private:
    wxFileName m_colorTableFile;
    a2f m_colorTable;
    asPredictor::Parameter m_parameter;

    bool ParseColorTable();

    bool ParseACTfile();

    bool ParseRGBfile();

    void ResizeColorTable(int size);

    void ScaleColors();

    void SelectColorTable();
};

#endif
