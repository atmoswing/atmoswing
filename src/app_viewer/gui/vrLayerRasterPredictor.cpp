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

vrLayerRasterPredictor::vrLayerRasterPredictor()
    : vrLayerRasterGDAL() {
    m_driverType = vrDRIVER_RASTER_MEMORY;
}

vrLayerRasterPredictor::~vrLayerRasterPredictor() = default;

bool vrLayerRasterPredictor::Close() {
    if (m_dataset == nullptr) {
        return false;
    }

    GDALClose(m_dataset);
    m_dataset = nullptr;
    return true;
}

bool vrLayerRasterPredictor::CreateInMemory(const wxFileName &name) {
    // Try to close
    Close();
    wxASSERT(m_dataset == nullptr);

    // Init filename
    m_fileName = name;

    // Get driver
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("MEM");
    if (poDriver == nullptr) {
        wxLogError("Cannot get the memory driver.");
        return false;
    }

    // Create dataset
    m_dataset = poDriver->Create((const char*)m_fileName.GetFullPath().mb_str(wxConvUTF8), int(m_longitudes.size()),
                                 int(m_latitudes.size()), 1, GDT_Float32, nullptr);
    if (m_dataset == nullptr) {
        wxLogError(_("Creation of memory dataset failed."));
        return false;
    }

    // Set projection
    if (m_dataset->SetProjection("EPSG:4326") != CE_None) {
        wxLogError(_("Setting projection to predictor layer failed."));
        return false;
    }

    // Set geotransform
    double adfGeoTransform[6];
    adfGeoTransform[0] = m_longitudes.minCoeff(); // top left x
    adfGeoTransform[1] = m_longitudes[1] - m_longitudes[0]; // w-e pixel resolution
    adfGeoTransform[2] = 0; // rotation, 0 if image is "north up"
    adfGeoTransform[3] = m_latitudes.maxCoeff(); // top left y
    adfGeoTransform[4] = 0; // rotation, 0 if image is "north up"
    adfGeoTransform[5] = m_latitudes[1] - m_latitudes[0];  // n-s pixel resolution (negative value)
    if (m_dataset->SetGeoTransform(adfGeoTransform) != CE_None) {
        wxLogError(_("Setting geotransform to predictor layer failed."));
        return false;
    }

    // Set data
    GDALRasterBand* band = m_dataset->GetRasterBand(1);

#if UseRasterIO
    if (band->RasterIO(GF_Write, 0, 0, int(m_longitudes.size()), int(m_latitudes.size()), &m_data(0, 0),
                       int(m_longitudes.size()), int(m_latitudes.size()), GDT_Float32, 0, 0, NULL) != CE_None) {
        wxLogError(_("Setting data to predictor layer failed."));
        return false;
    }
#else
    int xBlockSize, yBlockSize;
    band->GetBlockSize(&xBlockSize, &yBlockSize);

    if (m_longitudes.size() != xBlockSize) {
        wxLogError(_("The x block size does not match the data."));
        return false;
    }
    if (yBlockSize != 1) {
        wxLogError(_("The y block size should be 1."));
        return false;
    }

    for (int y = 0; y < m_latitudes.size(); y++) {
        if (band->WriteBlock(0, y, &m_data(y, 0)) != CE_None) {
            wxLogError(_("Setting data to predictor layer failed."));
            return false;
        }
    }
#endif

    return true;
}

wxFileName vrLayerRasterPredictor::GetDisplayName() {
    wxFileName myName(m_fileName);
    myName.SetExt(wxEmptyString);
    return myName;
}

bool vrLayerRasterPredictor::_GetRasterData(unsigned char** imgData, const wxSize& outImgPxSize,
                                            const wxRect& readImgPxInfo, const vrRender* render) {
    wxASSERT(m_dataset);
    m_dataset->FlushCache();

    // Create array for image data
    unsigned int imgRGBLen = outImgPxSize.GetWidth() * outImgPxSize.GetHeight() * 3;
    *imgData = (unsigned char*)malloc(imgRGBLen);
    if (*imgData == nullptr) {
        wxLogError(_("Image creation failed, out of memory"));
        return false;
    }

    // Read band
    GDALRasterBand* band = m_dataset->GetRasterBand(1);
    int dataSize = GDALGetDataTypeSize(GDT_Float32) / 8;
    void* rasterData = CPLMalloc(dataSize * outImgPxSize.GetWidth() * outImgPxSize.GetHeight());
    if (band->RasterIO(GF_Read, readImgPxInfo.GetX(), readImgPxInfo.GetY(), readImgPxInfo.GetWidth(),
                       readImgPxInfo.GetHeight(), rasterData, outImgPxSize.GetWidth(), outImgPxSize.GetHeight(),
                       GDT_Float32, 0, 0) != CE_None) {
        wxLogError(_("Error getting raster predictor data."));
        if (rasterData != nullptr) {
            CPLFree(rasterData);
            rasterData = nullptr;
        }
        if (*imgData != nullptr) {
            CPLFree(*imgData);
            *imgData = nullptr;
        }
        return false;
    }

    // Computing statistics if not existing
    if (!_HasStat()) {
        if (!_ComputeStat()) {
            if (rasterData != nullptr) {
                CPLFree(rasterData);
                rasterData = nullptr;
            }
            if (*imgData != nullptr) {
                CPLFree(*imgData);
                *imgData = nullptr;
            }
            return false;
        }
    }

    double range = m_oneBandMax - m_oneBandMin;
    if (range <= 0) {
        range = 1;
    }

    auto predictorRender = dynamic_cast<vrRenderRasterPredictor*>(const_cast<vrRender*>(render));
    wxASSERT(predictorRender);
    predictorRender->Init(m_parameter);

    // Transform to RGB
    for (unsigned int i = 0; i < imgRGBLen; i += 3) {
        double pxVal = _ReadGDALValueToDouble(rasterData, GDT_Float32, i / 3);

        // Hande nodata
        if (wxIsSameDouble(pxVal, m_oneBandNoData)) {
            *(*imgData + i) = 255;
            *(*imgData + i + 1) = 255;
            *(*imgData + i + 2) = 255;

            continue;
        }

        wxImage::RGBValue valRGB = predictorRender->GetColorFromTable(pxVal, m_oneBandMin, range);

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