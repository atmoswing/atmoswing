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
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/catalog_precipitation_MCH.xml");

    asCatalogPredictands catalog(filepath);
    catalog.Load(0);

    int samestr = catalog.GetSetId().CompareTo(_T("MCHDR"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetName().CompareTo(_T("MeteoSwiss daily rainfall measurements"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetDescription().CompareTo(_T("Precipitation measurements made by MeteoSwiss at a daily timestep"));
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
    double endreal = asTime::GetMJD(2007,12,31);
    CHECK_EQUAL(endreal,catalog.GetEnd());
    CHECK_EQUAL(24,catalog.GetTimeStepHours());
    CHECK_EQUAL(0,catalog.GetFirstTimeStepHours());
    int formatraw = catalog.GetFormatRaw();
    int formatrawreal = dat;
    CHECK_EQUAL(formatrawreal,formatraw);
    VectorDouble nans = catalog.GetNan();
    CHECK_EQUAL(32767,nans[0]);
    int coordinatesys = catalog.GetCoordSys();
    int coordinatesysreal = CH1903;
    CHECK_EQUAL(coordinatesysreal,coordinatesys);
}

TEST(LoadDataProp)
{
    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/catalog_precipitation_MCH.xml");

    asCatalogPredictands catalog(filepath);
    catalog.Load(1);

    CHECK_EQUAL(1,catalog.GetStationId());
    int samestr = catalog.GetStationName().CompareTo(_T("Disentis / Sedrun"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetStationLocalId().CompareTo(_T("0060"));
    CHECK_EQUAL(0,samestr);
    CHECK_EQUAL(708200,catalog.GetStationCoord().u);
    CHECK_EQUAL(173800,catalog.GetStationCoord().v);
    CHECK_EQUAL(1190,catalog.GetStationHeight());
    samestr = catalog.GetStationFilename().CompareTo(_T("0060_1948-2007.dat"));
    CHECK_EQUAL(0,samestr);
    samestr = catalog.GetStationFilepattern().CompareTo(_T("MCH_Climap_standard"));
    CHECK_EQUAL(0,samestr);
    double startreal = asTime::GetMJD(1948,01,01,00,00);
    CHECK_EQUAL(startreal,catalog.GetStationStart());
    double endreal = asTime::GetMJD(2007,12,31,00,00);
    CHECK_EQUAL(endreal,catalog.GetStationEnd());
}

}
