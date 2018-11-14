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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include <wx/filename.h>
#include "asCatalogPredictands.h"
#include "gtest/gtest.h"


TEST(CatalogPredictand, LoadCatalogProp)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/catalog_precipitation_MCH.xml");

    asCatalogPredictands catalog(filepath);
    catalog.Load();

    int samestr = catalog.GetSetId().CompareTo(_T("MeteoSwiss-Rhone"));
    EXPECT_EQ(0, samestr);
    samestr = catalog.GetName().CompareTo(_T("MeteoSwiss daily rainfall measurements for the Rhone catchment"));
    EXPECT_EQ(0, samestr);
    samestr = catalog.GetDescription().CompareTo(
            _T("Precipitation measurements made by MeteoSwiss at a daily timestep for the Rhone catchment"));
    EXPECT_EQ(0, samestr);
    asPredictand::Parameter paramval = catalog.GetParameter();
    asPredictand::Parameter paramref = asPredictand::Precipitation;
    EXPECT_EQ(paramref, paramval);
    int parameter = catalog.GetParameter();
    int parameterreal = asPredictand::Precipitation;
    EXPECT_EQ(parameterreal, parameter);
    int unit = catalog.GetUnit();
    int unitreal = asPredictand::mm;
    EXPECT_EQ(unitreal, unit);
    EXPECT_EQ(0, catalog.GetTimeZoneHours());
    double startreal = asTime::GetMJD(1940, 01, 01, 00, 00);
    EXPECT_EQ(startreal, catalog.GetStart());
    double endreal = asTime::GetMJD(2009, 12, 31);
    EXPECT_EQ(endreal, catalog.GetEnd());
    EXPECT_EQ(24, catalog.GetTimeStepHours());
    EXPECT_EQ(0, catalog.GetFirstTimeStepHours());
    vwxs nans = catalog.GetNan();
    EXPECT_EQ(true, nans[0].IsSameAs("32767"));
    samestr = catalog.GetCoordSys().CompareTo(_T("EPSG:3857"));
    EXPECT_EQ(0, samestr);
}

TEST(CatalogPredictand, LoadDataProp)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/catalog_precipitation_MCH.xml");

    asCatalogPredictands catalog(filepath);
    catalog.Load();

    EXPECT_EQ(1, catalog.GetStationId(0));
    wxString stationName = wxString(L"G\u00FCtsch ob Andermatt", wxConvUTF8);
    wxString stationNameFile = catalog.GetStationName(0);
    int samestr = stationNameFile.Cmp(stationName);
    EXPECT_EQ(0, samestr);
    samestr = catalog.GetStationOfficialId(0).CompareTo(_T("4020"));
    EXPECT_EQ(0, samestr);
    EXPECT_EQ(690140, catalog.GetStationCoord(0).x);
    EXPECT_EQ(167590, catalog.GetStationCoord(0).y);
    EXPECT_EQ(2287, catalog.GetStationHeight(0));
    samestr = catalog.GetStationFilename(0).CompareTo(_T("CH4020.dat"));
    EXPECT_EQ(0, samestr);
    samestr = catalog.GetStationFilepattern(0).CompareTo(_T("MeteoSwiss_Climap"));
    EXPECT_EQ(0, samestr);
    double startreal = asTime::GetMJD(1940, 01, 01, 00, 00);
    EXPECT_EQ(startreal, catalog.GetStationStart(0));
    double endreal = asTime::GetMJD(2009, 12, 31, 00, 00);
    EXPECT_EQ(endreal, catalog.GetStationEnd(0));
}
