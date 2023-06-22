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
    /**
     * The constructor for the vroomgis layer class showing raster predictor data.
     *
     * @param predictorsManager The predictors manager.
     * @param minVal The minimum value.
     * @param maxVal The maximum value.
     */
    vrLayerRasterPredictor(asPredictorsManager* predictorsManager, double minVal, double maxVal);

    /**
     * The destructor for the vroomgis layer class showing raster predictor data.
     */
    ~vrLayerRasterPredictor() override;

    /**
     * Create the layer in memory.
     *
     * @param name The filename.
     * @return True if successful.
     */
    bool CreateInMemory(const wxFileName& name);

    /**
     * Get the layer name to display.
     *
     * @return The layer name.
     */
    wxFileName GetDisplayName() override;

    asPredictor::Parameter GetParameter() {
        return m_parameter;
    }

  protected:
    /**
     * Transform the raster data into a bitmap.
     *
     * @param imgData The image data pointer.
     * @param outImgPxSize The output image size.
     * @param readImgPxInfo The bounding box of the data to read.
     * @param render The render.
     * @return True if successful.
     */
    virtual bool _GetRasterData(unsigned char** imgData, const wxSize& outImgPxSize, const wxRect& readImgPxInfo,
                                const vrRender* render);

  private:
    asPredictorsManager* m_predictorsManager; /**< The predictors manager. */
    asPredictor::Parameter m_parameter; /**< The meteorological parameter. */
    double m_minVal; /**< The minimum value. */
    double m_maxVal; /**< The maximum value. */

    /**
     * Close the layer and the dataset.
     *
     * @return True if successful.
     */
    bool Close();
};

#endif
