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

/**
 * @brief Vroomgis render class for the predictor raster.
 *
 * Vroomgis render class for the predictor raster.
 */
class vrRenderRasterPredictor : public vrRenderRaster {
  public:
    /**
     * The constructor for the vroomgis layer class containing the raster.
     */
    vrRenderRasterPredictor();

    /**
     * The destructor for the vroomgis layer class containing the raster.
     */
    ~vrRenderRasterPredictor() override;

    /**
     * Get the color for the given pixel value from the color table.
     *
     * @param pxVal The pixel value.
     * @param minVal The minimum value of the color table.
     * @param range The range of the color table.
     * @return The color.
     */
    wxImage::RGBValue GetColorFromTable(double pxVal, double minVal, double range);

    /**
     * Initialize the layer.
     *
     * @param parameter The meteorological parameter.
     */
    void Init(asPredictor::Parameter parameter);

    /**
     * Check that the renderer is OK.
     *
     * @return True if the renderer is OK.
     */
    bool IsOk() const;

  protected:
  private:
    wxFileName m_colorTableFile; /**< The color table file. */
    a2f m_colorTable; /**< The color table. */
    asPredictor::Parameter m_parameter; /**< The meteorological parameter. */

    /**
     * Parse the color table file.
     *
     * @return True if the color table was parsed successfully.
     */
    bool ParseColorTable();

    /**
     * Parse the color table file in ACT format.
     *
     * @return True if the color table was parsed successfully.
     */
    bool ParseACTfile();

    /**
     * Parse the color table file in RGB format.
     *
     * @return True if the color table was parsed successfully.
     */
    bool ParseRGBfile();

    /**
     * Resize the color table class attribute.
     *
     * @param size The new size of the color table.
     */
    void ResizeColorTable(int size);

    /**
     * Scale the color table values to the range 0-255.
     */
    void ScaleColors();

    /**
     * Select the color table according to the meteorological parameter.
     */
    void SelectColorTable();
};

#endif
