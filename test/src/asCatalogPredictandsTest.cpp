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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include <wx/filename.h>

#include "include_tests.h"
#include "asCatalogPredictands.h"

#include "UnitTest++.h"

namespace
{

TEST(LoadCatalogProp)
{
	wxPrintf("Testing predictand catalogs...\n");

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/catalog_precipitation_MCH.xml");

    asCatalogPredictands catalog(filepath);
    catalog.Load();

    int samestr = catalog.GetSetId().CompareTo(_T("MeteoSwiss-Rhone"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetName().CompareTo(_T("MeteoSwiss daily rainfall measurements for the Rhone catchment"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetDescription().CompareTo(_T("Precipitation measurements made by MeteoSwiss at a daily timestep for the Rhone catchment"));
    CHECK_EQUAL(0,samestr);
    DataParameter paramval = catalog.GetParameter();
    DataParameter paramref = Precipitation;
    CHECK_EQUAL(paramref,paramval);
    int parameter = catalog.GetParameter();
    int parameterreal = Precipitation;
    CHECK_EQUAL(parameterreal,parameter);
    int unit = catalog.GetUnit();
    int unitreal = mm;
    CHECK_EQUAL(unitreal,unit);
    CHECK_EQUAL(0,catalog.GetTimeZoneHours());
    double startreal = asTime::GetMJD(1940,01,01,00,00);
    CHECK_EQUAL(startreal,catalog.GetStart());
    double endreal = asTime::GetMJD(2009,12,31);
    CHECK_EQUAL(endreal,catalog.GetEnd());
    CHECK_EQUAL(24,catalog.GetTimeStepHours());
    CHECK_EQUAL(0,catalog.GetFirstTimeStepHours());
    VectorDouble nans = catalog.GetNan();
    CHECK_EQUAL(32767,nans[0]);
    samestr = catalog.GetCoordSys().CompareTo(_T("EPSG:3857"));
    CHECK_EQUAL(0,samestr);
}

TEST(LoadDataProp)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/catalog_precipitation_MCH.xml");

    asCatalogPredictands catalog(filepath);
    catalog.Load();

    CHECK_EQUAL(1,catalog.GetStationId(0));
    wxString stationName = wxString("G\u00FCtsch ob Andermatt", wxConvUTF8);
    wxString stationNameFile = catalog.GetStationName(0);
    int samestr = stationNameFile.Cmp(stationName);
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetStationOfficialId(0).CompareTo(_T("4020"));
    CHECK_EQUAL(0,samestr);
    CHECK_EQUAL(690140,catalog.GetStationCoord(0).x);
    CHECK_EQUAL(167590,catalog.GetStationCoord(0).y);
    CHECK_EQUAL(2287,catalog.GetStationHeight(0));
    samestr = catalog.GetStationFilename(0).CompareTo(_T("CH4020.dat"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetStationFilepattern(0).CompareTo(_T("MeteoSwiss_Climap"));
    CHECK_EQUAL(0,samestr);
    double startreal = asTime::GetMJD(1940,01,01,00,00);
    CHECK_EQUAL(startreal,catalog.GetStationStart(0));
    double endreal = asTime::GetMJD(2009,12,31,00,00);
    CHECK_EQUAL(endreal,catalog.GetStationEnd(0));
}

}
