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

#include "vrRenderRasterPredictor.h"

#include <wx/fileconf.h>
#include <wx/tokenzr.h>

#include <fstream>

#include "asFileText.h"
#include "vrlabel.h"

vrRenderRasterPredictor::vrRenderRasterPredictor()
    : vrRenderRaster(),
      _parameter(asPredictor::GeopotentialHeight) {}

vrRenderRasterPredictor::~vrRenderRasterPredictor() = default;

void vrRenderRasterPredictor::Init(asPredictor::Parameter parameter) {
    _parameter = parameter;

    SelectColorTable();
    ParseColorTable();
    ScaleColors();
}

bool vrRenderRasterPredictor::IsOk() const {
    if (_colorTable.rows() < 255) return false;
    if (_colorTable.rows() > 256) return false;
    if (_colorTable.cols() != 3) return false;

    return true;
}

void vrRenderRasterPredictor::SelectColorTable() {
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString dirData = asConfig::GetShareDir();
    wxString colorFilesDir = dirData + DS + "atmoswing" + DS + "color_tables";
    wxString filePath;

    switch (_parameter) {
        case asPredictor::GeopotentialHeight:
            filePath = pConfig->Read("/ColorTable/GeopotentialHeight", colorFilesDir + DS + "NEO_grav_anom.act");
            break;
        case asPredictor::PrecipitableWater:
        case asPredictor::TotalColumnWater:
            filePath = pConfig->Read("/ColorTable/PrecipitableWater", colorFilesDir + DS + "NEO_soil_moisture.act");
            break;
        case asPredictor::RelativeHumidity:
            filePath = pConfig->Read("/ColorTable/RelativeHumidity", colorFilesDir + DS + "NEO_soil_moisture.act");
            break;
        case asPredictor::SpecificHumidity:
            filePath = pConfig->Read("/ColorTable/SpecificHumidity", colorFilesDir + DS + "NEO_soil_moisture.act");
            break;
        default:
            filePath = colorFilesDir + DS + "MPL_viridis.rgb";
    }

    _colorTableFile = wxFileName(filePath);
}

wxImage::RGBValue vrRenderRasterPredictor::GetColorFromTable(double pxVal, double minVal, double range) {
    int nColors = int(_colorTable.rows());

    if (nColors == 0) {
        return {255, 255, 255};
    }

    int index = static_cast<int>((pxVal - minVal) * (nColors / range));
    if (index < 0) index = 0;
    if (index >= nColors) index = nColors - 1;

    wxImage::RGBValue valRGB(int(_colorTable(index, 0)), int(_colorTable(index, 1)), int(_colorTable(index, 2)));

    return valRGB;
}

bool vrRenderRasterPredictor::ParseColorTable() {
    wxString extension = _colorTableFile.GetExt();
    if (extension == "act") {
        return ParseACTfile();
    } else if (extension == "rgb") {
        return ParseRGBfile();
    }

    wxLogError(_("The color table format %s is not supported."), extension);
    return false;
}

bool vrRenderRasterPredictor::ParseACTfile() {
    ResizeColorTable(255);

    FILE* f;
    uint8_t palette[255][3];

    // Open the ACT file
    f = fopen(_colorTableFile.GetFullPath(), "rb");
    if (!f) {
        wxLogError(_("Color table file %s could not be opened..."), _colorTableFile.GetFullPath());
        return false;
    }

    // Read the entire contents into the palette array
    size_t nRows = fread(palette, 3, 255, f);
    fclose(f);
    if (nRows == 0) {
        wxLogError(_("Color table file %s is empty..."), _colorTableFile.GetFullPath());
        return false;
    }

    // Copy the palette values to the color table array
    for (int i = 0; i < 255; i++) {
        _colorTable(i, 0) = palette[i][0];
        _colorTable(i, 1) = palette[i][1];
        _colorTable(i, 2) = palette[i][2];
    }

    return true;
}

bool vrRenderRasterPredictor::ParseRGBfile() {
    asFileText file(_colorTableFile.GetFullPath(), asFile::ReadOnly);
    if (!file.Open()) {
        wxLogError(_("Color table file %s could not be opened..."), _colorTableFile.GetFullPath());
        return false;
    }

    // First line should be the number of colors
    wxString fileLine = file.GetNextLine();

    int indexNColors = fileLine.Find("ncolors=");
    if (indexNColors == wxNOT_FOUND) {
        wxLogError(_("Color table format not supported (file %s)..."), _colorTableFile.GetFullPath());
        return false;
    }
    wxString strNColors = fileLine.SubString(indexNColors + 8, fileLine.Len() - 1);
    long nColorsVal;
    if (!strNColors.ToLong(&nColorsVal)) {
        wxLogError(_("Color table format not supported (file %s)..."), _colorTableFile.GetFullPath());
        return false;
    }
    wxASSERT(nColorsVal > 0);

    ResizeColorTable(nColorsVal);

    // Skipping header
    if (!file.SkipLines(1)) {
        wxLogError(_("Color table format not supported (file %s)..."), _colorTableFile.GetFullPath());
        return false;
    }

    for (int i = 0; i < nColorsVal; ++i) {
        // Get next line
        fileLine = file.GetNextLine();
        if (fileLine.IsEmpty()) break;

        int n = 0;
        wxStringTokenizer tokenizer(fileLine, " ");
        while (tokenizer.HasMoreTokens()) {
            wxString token = tokenizer.GetNextToken();
            double val;
            if (!token.ToDouble(&val)) {
                wxLogError(_("Color table format not supported (file %s)..."), _colorTableFile.GetFullPath());
                return false;
            }

            _colorTable(i, n) = float(val);

            if (n == 2) break;
            n++;
        }
        if (file.EndOfFile()) break;
    }

    if (!file.Close()) {
        wxLogError(_("Color table file %s could not be closed..."), _colorTableFile.GetFullPath());
        return false;
    }

    return false;
}

void vrRenderRasterPredictor::ResizeColorTable(int size) {
    _colorTable.resize(size, 3);
    _colorTable.fill(0);
}

void vrRenderRasterPredictor::ScaleColors() {
    if (_colorTable.maxCoeff() <= 1) {
        _colorTable *= 255;
    }
}