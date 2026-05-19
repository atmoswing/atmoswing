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

#include "vrRenderRasterPredictor.h"
#include "vrlabel.h"
#include "vrrealrect.h"

#define UseRasterIO 0

vrLayerRasterPredictor::vrLayerRasterPredictor(asPredictorsManager* predictorsManager, double minVal, double maxVal)
    : vrLayerRasterGDAL(),
      _predictorsManager(predictorsManager),
      _minVal(minVal),
      _maxVal(maxVal) {
    _driverType = vrDRIVER_RASTER_MEMORY;
}

vrLayerRasterPredictor::~vrLayerRasterPredictor() = default;

bool vrLayerRasterPredictor::Close() {
    if (_dataset == nullptr) {
        return false;
    }

    GDALClose(_dataset);
    _dataset = nullptr;
    return true;
}

bool vrLayerRasterPredictor::CreateInMemory(const wxFileName& name) {
    // Try to close
    Close();
    wxASSERT(_dataset == nullptr);

    // Init filename
    _fileName = name;

    // Get driver
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("MEM");
    if (poDriver == nullptr) {
        wxLogError("Cannot get the memory driver.");
        return false;
    }

    // Create dataset
    _dataset = poDriver->Create((const char*)_fileName.GetFullPath().mb_str(wxConvUTF8),
                                _predictorsManager->GetLongitudesNb(), _predictorsManager->GetLatitudesNb(), 1,
                                GDT_Float32, nullptr);
    if (_dataset == nullptr) {
        wxLogError(_("Creation of memory dataset failed."));
        return false;
    }

    // Set projection
    if (_dataset->SetProjection("EPSG:4326") != CE_None) {
        wxLogError(_("Setting projection to predictor layer failed."));
        return false;
    }

    // Set geotransform
    double adfGeoTransform[6];
    adfGeoTransform[0] = _predictorsManager->GetLongitudeMin();    // top left x
    adfGeoTransform[1] = _predictorsManager->GetLongitudeResol();  // w-e pixel resolution
    adfGeoTransform[2] = 0;                                        // rotation, 0 if image is "north up"
    adfGeoTransform[3] = _predictorsManager->GetLatitudeMax();     // top left y
    adfGeoTransform[4] = 0;                                        // rotation, 0 if image is "north up"
    adfGeoTransform[5] = _predictorsManager->GetLatitudeResol();   // n-s pixel resolution
    if (_dataset->SetGeoTransform(adfGeoTransform) != CE_None) {
        wxLogError(_("Setting geotransform to predictor layer failed."));
        return false;
    }

    // Set data
    GDALRasterBand* band = _dataset->GetRasterBand(1);

#if UseRasterIO
    if (band->RasterIO(GF_Write, 0, 0, _predictorsManager->GetLongitudesNb(), _predictorsManager->GetLatitudesNb(),
                       _predictorsManager->GetData(), _predictorsManager->GetLongitudesNb(),
                       _predictorsManager->GetLatitudesNb(), GDT_Float32, 0, 0, NULL) != CE_None) {
        wxLogError(_("Setting data to predictor layer failed."));
        return false;
    }
#else
    int xBlockSize, yBlockSize;
    band->GetBlockSize(&xBlockSize, &yBlockSize);

    if (_predictorsManager->GetLongitudesNb() != xBlockSize) {
        wxLogError(_("The x block size does not match the data."));
        return false;
    }
    if (yBlockSize != 1) {
        wxLogError(_("The y block size should be 1."));
        return false;
    }

    for (int y = 0; y < _predictorsManager->GetLatitudesNb(); y++) {
        if (band->WriteBlock(0, y, _predictorsManager->GetDataRow(y)) != CE_None) {
            wxLogError(_("Setting data to predictor layer failed."));
            return false;
        }
    }
#endif

    _parameter = _predictorsManager->GetParameter();

    return true;
}

wxFileName vrLayerRasterPredictor::GetDisplayName() {
    wxFileName myName(_fileName);
    myName.SetExt(wxEmptyString);
    return myName;
}

bool vrLayerRasterPredictor::_GetRasterData(unsigned char** imgData, const wxSize& outImgPxSize,
                                            const wxRect& readImgPxInfo, const vrRender* render) {
    wxASSERT(_dataset);
    _dataset->FlushCache();

    // Create array for image data
    unsigned int imgRGBLen = outImgPxSize.GetWidth() * outImgPxSize.GetHeight() * 3;
    *imgData = (unsigned char*)malloc(imgRGBLen);
    if (*imgData == nullptr) {
        wxLogError(_("Image creation failed, out of memory"));
        return false;
    }

    // Read band
    GDALRasterBand* band = _dataset->GetRasterBand(1);
    int dataSize = GDALGetDataTypeSize(GDT_Float32) / 8;
    void* rasterData = CPLMalloc(dataSize * outImgPxSize.GetWidth() * outImgPxSize.GetHeight());
    if (band->RasterIO(GF_Read, readImgPxInfo.GetX(), readImgPxInfo.GetY(), readImgPxInfo.GetWidth(),
                       readImgPxInfo.GetHeight(), rasterData, outImgPxSize.GetWidth(), outImgPxSize.GetHeight(),
                       GDT_Float32, 0, 0) != CE_None) {
        wxLogError(_("Error getting raster predictor data."));
        if (rasterData != nullptr) {
            CPLFree(rasterData);
        }
        return false;
    }

    double range = _maxVal - _minVal;
    if (range <= 0) {
        range = 1;
    }

    auto predictorRender = dynamic_cast<vrRenderRasterPredictor*>(const_cast<vrRender*>(render));
    wxASSERT(predictorRender);
    predictorRender->Init(_parameter);

    // Transform to RGB
    for (unsigned int i = 0; i < imgRGBLen; i += 3) {
        double pxVal = _ReadGDALValueToDouble(rasterData, GDT_Float32, i / 3);

        // Hande nodata
        if (isnan(pxVal)) {
            *(*imgData + i) = 255;
            *(*imgData + i + 1) = 255;
            *(*imgData + i + 2) = 255;

            continue;
        }

        wxImage::RGBValue valRGB = predictorRender->GetColorFromTable(pxVal, _minVal, range);

        *(*imgData + i) = valRGB.red;
        *(*imgData + i + 1) = valRGB.green;
        *(*imgData + i + 2) = valRGB.blue;
    }
    wxASSERT(rasterData != nullptr);
    CPLFree(rasterData);
    rasterData = nullptr;

    CPLFree(rasterData);
    rasterData = nullptr;

    return true;
}